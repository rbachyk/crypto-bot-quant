"""Data-quality validation (AGENTS.md Section 23 + Section 8 Data Quality Gate).

Runs the safety-critical data checks over the stored series for the coverage
window and produces a :class:`DataQualityReport`. Data quality is a trading
safety issue (Section 23): a *critical* violation blocks research/paper/live.

Checks implemented (per Section 8/23):
* no critical missing candles (gaps);
* no duplicate records;
* no out-of-order timestamps;
* no future timestamps;
* no impossible prices (non-positive, OHLC inconsistency, above ceiling);
* no extreme unexplained price gaps;
* funding timestamps aligned to the funding grid;
* mark / index / perp timestamps aligned;
* spreads within the abnormal-spread threshold;
* clock within NTP tolerance.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from src.data.config import DataConfig
from src.data.gaps import find_gaps
from src.data.schema import (
    FUNDING,
    INDEX,
    MARK,
    OHLCV,
    SPREAD,
    SeriesKey,
    ms_to_iso,
)
from src.data.store import SeriesStore

CRITICAL = "critical"
WARNING = "warning"

# A close-to-close move larger than this fraction is an "extreme unexplained gap".
_EXTREME_MOVE_FRAC = 0.5


@dataclass(slots=True)
class Violation:
    check: str
    severity: str
    detail: str
    series: str = ""

    def to_dict(self) -> dict:
        return {
            "check": self.check,
            "severity": self.severity,
            "series": self.series,
            "detail": self.detail,
        }


@dataclass(slots=True)
class DataQualityReport:
    generated_at: str
    window: dict
    checks_run: list[str] = field(default_factory=list)
    violations: list[Violation] = field(default_factory=list)
    series_validated: int = 0

    @property
    def critical(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == CRITICAL]

    @property
    def passed(self) -> bool:
        return not self.critical

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "window": self.window,
            "series_validated": self.series_validated,
            "checks_run": self.checks_run,
            "passed": self.passed,
            "critical_count": len(self.critical),
            "violation_count": len(self.violations),
            "violations": [v.to_dict() for v in self.violations],
        }


class DataValidator:
    def __init__(self, store: SeriesStore, cfg: DataConfig) -> None:
        self.store = store
        self.cfg = cfg

    def validate(self) -> DataQualityReport:
        start, end = self.cfg.window_start_ms, self.cfg.window_end_ms
        report = DataQualityReport(
            generated_at=_now_iso(),
            window={"from": ms_to_iso(start), "to": ms_to_iso(end)},
            checks_run=[
                "missing_candles",
                "duplicates",
                "ordering",
                "future_timestamps",
                "impossible_prices",
                "extreme_gaps",
                "funding_alignment",
                "markindex_alignment",
                "abnormal_spread",
                "clock_drift",
            ],
        )
        now_ms = int(time.time() * 1000)
        validated = 0
        for symbol in self.cfg.active_symbols():
            for key in self.cfg.required_keys(symbol):
                validated += 1
                self._validate_series(key, start, end, now_ms, report)

        self._check_markindex_alignment(start, end, report)
        self._check_clock_drift(report)
        report.series_validated = validated
        return report

    # -- per-series checks ---------------------------------------------- #
    def _validate_series(
        self, key: SeriesKey, start: int, end: int, now_ms: int, report: DataQualityReport
    ) -> None:
        rows = self.store.read(key, start, end)
        label = key.label()

        gaps = find_gaps(self.store, key, start, end)
        if gaps.missing_ts:
            # A few scattered missing candles over a multi-year window (exchange maintenance) are
            # expected and should NOT fail the snapshot. ``max_unfilled_gap_bars`` is the tolerance:
            # > tolerance missing ⇒ CRITICAL (data genuinely incomplete); ≤ tolerance ⇒ WARNING
            # (recorded, but the snapshot stays valid). The reference config keeps tolerance 0.
            tol = self.cfg.thresholds.max_unfilled_gap_bars
            severity = CRITICAL if len(gaps.missing_ts) > tol else WARNING
            report.violations.append(
                Violation(
                    "missing_candles",
                    severity,
                    f"{len(gaps.missing_ts)} missing of {gaps.expected} expected "
                    f"(tolerance {tol})",
                    label,
                )
            )

        ts_list = [r["ts"] for r in rows]
        if len(set(ts_list)) != len(ts_list):
            report.violations.append(
                Violation("duplicates", CRITICAL, "duplicate timestamps present", label)
            )
        if ts_list != sorted(ts_list):
            report.violations.append(
                Violation("ordering", CRITICAL, "timestamps out of order", label)
            )
        if not all(ts % key.interval_ms == 0 for ts in ts_list):
            report.violations.append(
                Violation("ordering", CRITICAL, "timestamp off the expected grid", label)
            )

        future = [ts for ts in ts_list if ts > now_ms]
        if future:
            report.violations.append(
                Violation(
                    "future_timestamps",
                    CRITICAL,
                    f"{len(future)} timestamps in the future (e.g. {ms_to_iso(future[0])})",
                    label,
                )
            )

        if key.data_type == OHLCV:
            self._check_ohlcv_values(key, rows, report)
        self._check_funding_alignment(key, ts_list, report)
        self._check_spread(key, rows, report)

    def _check_ohlcv_values(
        self, key: SeriesKey, rows: list[dict], report: DataQualityReport
    ) -> None:
        label = key.label()
        th = self.cfg.thresholds
        prev_close: float | None = None
        for r in rows:
            o, h, low, c = r["open"], r["high"], r["low"], r["close"]
            if min(o, h, low, c) <= th.min_price or max(o, h, low, c) > th.max_price:
                report.violations.append(
                    Violation(
                        "impossible_prices", CRITICAL, f"price out of bounds at {r['ts']}", label
                    )
                )
                break
            if h < low or h < max(o, c) or low > min(o, c):
                report.violations.append(
                    Violation(
                        "impossible_prices", CRITICAL, f"OHLC inconsistent at {r['ts']}", label
                    )
                )
                break
        for r in rows:
            c = r["close"]
            if prev_close is not None and prev_close > 0:
                move = abs(c - prev_close) / prev_close
                if move > _EXTREME_MOVE_FRAC:
                    report.violations.append(
                        Violation(
                            "extreme_gaps",
                            CRITICAL,
                            f"{move:.1%} close-to-close move at {r['ts']}",
                            label,
                        )
                    )
                    break
            prev_close = c

    def _check_funding_alignment(
        self, key: SeriesKey, ts_list: list[int], report: DataQualityReport
    ) -> None:
        if key.data_type != FUNDING:
            return
        funding_ms = self.cfg.funding_interval_hours * 3_600_000
        misaligned = [ts for ts in ts_list if ts % funding_ms != 0]
        if misaligned:
            report.violations.append(
                Violation(
                    "funding_alignment",
                    CRITICAL,
                    f"{len(misaligned)} funding timestamps off the {key.timeframe} grid",
                    key.label(),
                )
            )

    def _check_spread(self, key: SeriesKey, rows: list[dict], report: DataQualityReport) -> None:
        if key.data_type != SPREAD:
            return
        th = self.cfg.thresholds
        abnormal = [r for r in rows if r["spread_bps"] > th.max_spread_bps]
        if abnormal:
            report.violations.append(
                Violation(
                    "abnormal_spread",
                    CRITICAL,
                    f"{len(abnormal)} samples above {th.max_spread_bps} bps (toxic execution)",
                    key.label(),
                )
            )

    # -- cross-series checks -------------------------------------------- #
    def _check_markindex_alignment(self, start: int, end: int, report: DataQualityReport) -> None:
        """Mark/index/perp must share the base-timeframe grid (Section 8/23)."""
        ex = self.cfg.exchange_id
        base = self.cfg.base_timeframe
        for symbol in self.cfg.active_symbols():
            perp = self.store.timestamps(SeriesKey(ex, OHLCV, symbol, base), start, end)
            for dt in (MARK, INDEX):
                if dt not in self.cfg.required_series:
                    continue
                other = self.store.timestamps(SeriesKey(ex, dt, symbol, base), start, end)
                if other != perp:
                    report.violations.append(
                        Violation(
                            "markindex_alignment",
                            CRITICAL,
                            f"{dt} timestamps not aligned with perp {base} grid",
                            f"{symbol}:{dt}",
                        )
                    )

    def _check_clock_drift(self, report: DataQualityReport) -> None:
        """Verify the system clock advances consistently with the monotonic clock.

        Offline we cannot query an NTP server; we verify the wall clock is not
        broken (jumping/frozen) relative to the monotonic clock. Absolute NTP
        synchronisation is enforced at deploy time by the host (chrony) and is
        re-verified by the Phase 13 MON gate.
        """
        th = self.cfg.thresholds
        w0, m0 = time.time(), time.monotonic()
        w1, m1 = time.time(), time.monotonic()
        skew = abs((w1 - w0) - (m1 - m0))
        if skew > th.clock_drift_tolerance_s:
            report.violations.append(
                Violation(
                    "clock_drift",
                    CRITICAL,
                    f"wall/monotonic skew {skew:.3f}s exceeds {th.clock_drift_tolerance_s}s",
                )
            )


def _now_iso() -> str:
    return ms_to_iso(int(time.time() * 1000))
