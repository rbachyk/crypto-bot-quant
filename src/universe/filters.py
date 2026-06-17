"""Per-symbol universe filters (AGENTS.md Section 9 "Default Universe Filters").

Each candidate symbol is scored against every configured filter using the data
the platform actually owns (volume / history / spread / coverage from the
Parquet store) and its ``[VERIFIED]`` exchange metadata. A symbol is promoted to
``active`` only when it passes EVERY filter; otherwise it is recorded with a
per-filter reason and a status reflecting the severity of the failure:

* a **hard** failure (missing data series, unverified/unstable metadata) means
  the symbol is data/metadata-unsafe → ``quarantined``;
* a **soft** failure (too little volume/history, wide spread) is recoverable →
  ``research_only``.

No symbol is ever silently dropped (Section 9: prevent trading on symbols that
do not pass gates, but keep their membership history).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from src.data.config import DataConfig
from src.data.gaps import find_gaps
from src.data.schema import OHLCV, SPREAD, SeriesKey
from src.data.store import SeriesStore
from src.db.models import SymbolStatus
from src.universe.config import UniverseConfig

_DAY_MS = 86_400_000

# Severity of each filter. Hard failures mean the symbol is unsafe to trade
# (data/metadata), soft failures mean it is merely below a quality bar.
HARD = "hard"
SOFT = "soft"


@dataclass(slots=True)
class FilterOutcome:
    name: str
    passed: bool
    severity: str
    detail: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "severity": self.severity,
            "detail": self.detail,
        }


@dataclass(slots=True)
class SymbolEvaluation:
    symbol: str
    outcomes: list[FilterOutcome] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    @property
    def passed_all(self) -> bool:
        return all(o.passed for o in self.outcomes)

    @property
    def status(self) -> SymbolStatus:
        if self.passed_all:
            return SymbolStatus.ACTIVE
        if any((not o.passed) and o.severity == HARD for o in self.outcomes):
            return SymbolStatus.QUARANTINED
        return SymbolStatus.RESEARCH_ONLY

    def reason(self) -> str:
        failed = [o.name for o in self.outcomes if not o.passed]
        return "passes all filters" if not failed else "failed: " + ", ".join(failed)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "status": self.status.value,
            "passed_all": self.passed_all,
            "reason": self.reason(),
            "metrics": self.metrics,
            "filters": [o.to_dict() for o in self.outcomes],
        }


@dataclass(slots=True)
class SymbolMetaView:
    """The metadata facts the universe filters need (Section 6/9)."""

    verified: bool = False
    status: str = ""
    contract_type: str = ""
    quote_currency: str = ""
    has_funding: bool = False
    has_open_interest: bool = False


class UniverseFilterEvaluator:
    """Evaluates the Section 9 filters for one exchange/window."""

    def __init__(self, store: SeriesStore, data_cfg: DataConfig, uni_cfg: UniverseConfig) -> None:
        self.store = store
        self.data_cfg = data_cfg
        self.uni_cfg = uni_cfg

    # -- metric helpers -------------------------------------------------- #
    def _ohlcv_key(self, symbol: str) -> SeriesKey:
        return SeriesKey(self.uni_cfg.exchange_id, OHLCV, symbol, self.uni_cfg.eval_timeframe)

    def _daily_notional(self, symbol: str) -> float:
        key = self._ohlcv_key(symbol)
        rows = self.store.read(key, self.data_cfg.window_start_ms, self.data_cfg.window_end_ms)
        if not rows:
            return 0.0
        total = sum(r["volume"] * r["close"] for r in rows)
        window_ms = self.data_cfg.window_end_ms - self.data_cfg.window_start_ms
        if window_ms <= 0:
            return 0.0
        return total * _DAY_MS / window_ms

    def _history_bars(self, symbol: str) -> int:
        return self.store.count(
            self._ohlcv_key(symbol), self.data_cfg.window_start_ms, self.data_cfg.window_end_ms
        )

    def _listing_age_days(self, symbol: str) -> float:
        key = self._ohlcv_key(symbol)
        bars = self.store.count(key, self.data_cfg.window_start_ms, self.data_cfg.window_end_ms)
        # Age = covered span = bars * bar interval (number of complete intervals).
        return bars * key.interval_ms / _DAY_MS

    def _missing_pct(self, symbol: str) -> float:
        expected = 0
        missing = 0
        for key in self.data_cfg.required_keys(symbol):
            gap = find_gaps(
                self.store, key, self.data_cfg.window_start_ms, self.data_cfg.window_end_ms
            )
            expected += gap.expected
            missing += len(gap.missing_ts)
        return 0.0 if expected == 0 else 100.0 * missing / expected

    def _median_spread_bps(self, symbol: str) -> float | None:
        key = SeriesKey(self.uni_cfg.exchange_id, SPREAD, symbol, self.data_cfg.base_timeframe)
        rows = self.store.read(key, self.data_cfg.window_start_ms, self.data_cfg.window_end_ms)
        if not rows:
            return None
        return statistics.median(r["spread_bps"] for r in rows)

    def _all_required_series_present(self, symbol: str) -> bool:
        for key in self.data_cfg.required_keys(symbol):
            gap = find_gaps(
                self.store, key, self.data_cfg.window_start_ms, self.data_cfg.window_end_ms
            )
            if not gap.covered:
                return False
        return True

    # -- evaluation ------------------------------------------------------ #
    def evaluate(self, symbol: str, meta: SymbolMetaView) -> SymbolEvaluation:
        f = self.uni_cfg.filters
        notional = self._daily_notional(symbol)
        history = self._history_bars(symbol)
        age = self._listing_age_days(symbol)
        missing = self._missing_pct(symbol)
        spread = self._median_spread_bps(symbol)
        series_present = self._all_required_series_present(symbol)

        ev = SymbolEvaluation(
            symbol=symbol,
            metrics={
                "daily_notional_usd": round(notional, 2),
                "history_bars": history,
                "listing_age_days": round(age, 4),
                "missing_data_pct": round(missing, 4),
                "median_spread_bps": None if spread is None else round(spread, 4),
                "verified_metadata": meta.verified,
                "contract_status": meta.status,
            },
        )

        def add(name: str, passed: bool, severity: str, detail: str) -> None:
            ev.outcomes.append(FilterOutcome(name, passed, severity, detail))

        # Hard (data/metadata safety) filters.
        add(
            "data_availability",
            series_present,
            HARD,
            "all required series present" if series_present else "missing required series",
        )
        if f.require_metadata_verified:
            add(
                "metadata_verified",
                meta.verified,
                HARD,
                "[VERIFIED]" if meta.verified else "metadata not [VERIFIED]",
            )
        if f.require_stable_status:
            ok = meta.status == "trading"
            add("stable_status", ok, HARD, f"status={meta.status or 'unknown'}")
        add(
            "quote_currency",
            meta.quote_currency == f.quote_currency,
            HARD,
            f"{meta.quote_currency or 'unknown'} (want {f.quote_currency})",
        )
        add(
            "contract_type",
            meta.contract_type == f.contract_type,
            HARD,
            f"{meta.contract_type or 'unknown'} (want {f.contract_type})",
        )
        if f.require_funding_history:
            add("funding_available", meta.has_funding, HARD, "funding series present")
        if f.require_open_interest:
            add("open_interest_available", meta.has_open_interest, HARD, "open-interest present")

        # Soft (quality) filters.
        add(
            "min_daily_notional",
            notional >= f.min_daily_notional_usd,
            SOFT,
            f"{notional:.0f} >= {f.min_daily_notional_usd:.0f}",
        )
        add(
            "min_history_bars",
            history >= f.min_history_bars,
            SOFT,
            f"{history} >= {f.min_history_bars}",
        )
        add(
            "min_listing_age",
            age >= f.min_listing_age_days,
            SOFT,
            f"{age:.3f}d >= {f.min_listing_age_days}d",
        )
        add(
            "max_missing_data_pct",
            missing <= f.max_missing_data_pct,
            SOFT,
            f"{missing:.3f}% <= {f.max_missing_data_pct}%",
        )
        add(
            "max_median_spread",
            spread is not None and spread <= f.max_median_spread_bps,
            SOFT,
            "no spread data" if spread is None else f"{spread:.2f}bps <= {f.max_median_spread_bps}",
        )
        return ev
