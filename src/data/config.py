"""Loader for ``configs/data.yaml`` — the data-platform contract.

Turns the YAML into a typed :class:`DataConfig` and enumerates the concrete
set of :class:`SeriesKey` the platform must own for the coverage window. The
Gate Runner, ingestion jobs, validation and ``scripts/backfill`` all read this
one object so they never drift (Section 4 config-driven behaviour).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

import yaml

from src.config.settings import REPO_ROOT
from src.data.schema import (
    FUNDING,
    OHLCV,
    OPEN_INTEREST,
    SeriesKey,
    parse_utc_ms,
)

DATA_YAML = REPO_ROOT / "configs" / "data.yaml"
_HOUR_MS = 3_600_000


def _resolve_window_end_ms(window: dict, as_of_ms: int | None = None) -> int:
    """Resolve the window end. ``as_of: now`` (or a missing/empty as_of) anchors the END to the
    CURRENT time — floored to the hour and backed off one hour — so the window always reaches
    fresh data and the still-forming / lagging recent candles are NOT required (which would make
    validation spuriously fail as 'missing'). An explicit ISO ``as_of`` is used verbatim.

    ``as_of_ms`` pins the window end explicitly (overrides the yaml) — used to FREEZE a ``now``
    window for a reproducible snapshot / test: the same as_of_ms always yields the same window,
    so the snapshot id is stable across re-runs."""
    if as_of_ms is not None:
        return (int(as_of_ms) // _HOUR_MS) * _HOUR_MS  # snap to the hour grid for determinism
    as_of = window.get("as_of")
    if as_of is None or str(as_of).strip().lower() in ("", "now"):
        now_ms = int(datetime.now(UTC).timestamp() * 1000)
        return (now_ms // _HOUR_MS) * _HOUR_MS - _HOUR_MS  # last fully-closed hour boundary
    return parse_utc_ms(str(as_of))


@dataclass(frozen=True, slots=True)
class ValidationThresholds:
    max_unfilled_gap_bars: int = 0
    max_spread_bps: float = 250.0
    min_price: float = 0.0
    max_price: float = 1.0e12
    max_funding_misalignment_s: int = 1
    max_markindex_misalignment_s: int = 1
    clock_drift_tolerance_s: float = 2.0


@dataclass(frozen=True, slots=True)
class DataConfig:
    exchange_id: str
    data_version: str
    symbols: list[str]
    timeframes: list[str]
    base_timeframe: str
    funding_interval_hours: int
    required_series: list[str]
    window_start_ms: int
    window_end_ms: int
    insufficient_history: list[str] = field(default_factory=list)
    thresholds: ValidationThresholds = field(default_factory=ValidationThresholds)
    # Optional coarser grid for open interest. Some venues (e.g. Bybit) only serve
    # recent OI and retain longer history at coarser intervals, so OI may be sampled
    # on its own grid while mark/index/spread stay on ``base_timeframe``. ``None`` ⇒
    # OI shares the base grid (the offline skeleton default, unchanged).
    oi_timeframe: str | None = None
    # Decision timeframes whose engine inputs are PRE-BUILT (and cached) at download time, so
    # validation/backtests load them instantly instead of rebuilding (~hours on 4h, days on 5m).
    # ``None`` ⇒ pre-build every decision timeframe; set a subset to skip the slow ones till needed.
    prebuild_input_timeframes: list[str] | None = None

    @property
    def prebuild_timeframes(self) -> list[str]:
        """Decision timeframes to pre-build inputs for at download time (defaults to every one)."""
        return list(self.prebuild_input_timeframes or self.timeframes)

    @property
    def funding_timeframe(self) -> str:
        return f"{self.funding_interval_hours}h"

    @property
    def oi_grid(self) -> str:
        """The timeframe label open-interest is sampled on (defaults to base)."""
        return self.oi_timeframe or self.base_timeframe

    def active_symbols(self) -> list[str]:
        """Symbols that must be fully covered (insufficient-history excluded)."""
        excluded = set(self.insufficient_history)
        return [s for s in self.symbols if s not in excluded]

    def required_keys(self, symbol: str) -> list[SeriesKey]:
        """Every series this symbol must have over the coverage window."""
        keys: list[SeriesKey] = []
        for series in self.required_series:
            if series == OHLCV:
                for tf in self.timeframes:
                    keys.append(SeriesKey(self.exchange_id, OHLCV, symbol, tf))
            elif series == FUNDING:
                keys.append(SeriesKey(self.exchange_id, FUNDING, symbol, self.funding_timeframe))
            elif series == OPEN_INTEREST:
                keys.append(SeriesKey(self.exchange_id, OPEN_INTEREST, symbol, self.oi_grid))
            else:
                keys.append(SeriesKey(self.exchange_id, series, symbol, self.base_timeframe))
        return keys

    def all_required_keys(self) -> list[SeriesKey]:
        keys: list[SeriesKey] = []
        for symbol in self.active_symbols():
            keys.extend(self.required_keys(symbol))
        return keys


@lru_cache
def _read_data_yaml(path: str | None) -> dict:
    """Parse the YAML once (cheap, cached). The window is resolved per-call in load_data_config so
    a dynamic ``as_of: now`` advances over time (long-lived workers must not freeze it)."""
    yaml_path = Path(path) if path else DATA_YAML
    return yaml.safe_load(yaml_path.read_text(encoding="utf-8"))


def load_data_config(path: str | None = None, *, as_of_ms: int | None = None) -> DataConfig:
    """Load the data config. Pass ``as_of_ms`` to PIN the window end (freeze a ``now`` window) so
    the resolved window — and therefore the snapshot id — is reproducible across runs/tests."""
    raw = _read_data_yaml(path)
    data = raw["data"]
    window = data["window"]
    end_ms = _resolve_window_end_ms(window, as_of_ms)  # dynamic when as_of is 'now'/absent
    start_ms = end_ms - int(window["duration_hours"]) * 3_600_000
    v = data.get("validation", {})
    thresholds = ValidationThresholds(
        max_unfilled_gap_bars=int(v.get("max_unfilled_gap_bars", 0)),
        max_spread_bps=float(v.get("max_spread_bps", 250.0)),
        min_price=float(v.get("min_price", 0.0)),
        max_price=float(v.get("max_price", 1.0e12)),
        max_funding_misalignment_s=int(v.get("max_funding_misalignment_s", 1)),
        max_markindex_misalignment_s=int(v.get("max_markindex_misalignment_s", 1)),
        clock_drift_tolerance_s=float(v.get("clock_drift_tolerance_s", 2.0)),
    )
    return DataConfig(
        exchange_id=str(data["exchange_id"]),
        data_version=str(data.get("data_version", "data_0001")),
        symbols=list(data["symbols"]),
        timeframes=list(data["timeframes"]),
        base_timeframe=str(data["base_timeframe"]),
        funding_interval_hours=int(data["funding_interval_hours"]),
        required_series=list(data["required_series"]),
        window_start_ms=start_ms,
        window_end_ms=end_ms,
        insufficient_history=list(data.get("insufficient_history", [])),
        thresholds=thresholds,
        oi_timeframe=(str(data["oi_timeframe"]) if data.get("oi_timeframe") else None),
        prebuild_input_timeframes=(
            list(data["prebuild_input_timeframes"])
            if data.get("prebuild_input_timeframes")
            else None
        ),
    )
