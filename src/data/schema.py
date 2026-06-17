"""Canonical data-platform schema: series identity, row shapes, grid math.

A single source of truth for *what* the data platform stores and *where*
(AGENTS.md Section 8, Appendix B.5). Every series lives on a regular time grid
of fixed ``interval_ms`` so coverage, gap detection and validation share one
notion of an "expected timestamp" (no ambiguity between backtest and live —
the Parity Rule, Section 10).

Timestamps are UTC epoch milliseconds (int). For OHLCV the timestamp is the
candle **open** time; for point-in-time series (mark/index/OI/spread/funding)
it is the sample time on the grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

# Supported timeframes -> milliseconds. Custom single code path (Appendix C):
# we never depend on a library's timeframe parsing.
TIMEFRAME_MS: dict[str, int] = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}

# data_type values the platform knows about (Section 8 required data types).
OHLCV = "ohlcv"
MARK = "mark"
INDEX = "index"
FUNDING = "funding"
OPEN_INTEREST = "open_interest"
SPREAD = "spread"

# Column order per data_type (the parquet/row schema). The first column is the
# grid timestamp ("ts") and is the dedup/primary key for every series.
COLUMNS: dict[str, list[str]] = {
    OHLCV: ["ts", "open", "high", "low", "close", "volume"],
    MARK: ["ts", "mark_price"],
    INDEX: ["ts", "index_price"],
    FUNDING: ["ts", "funding_rate", "funding_interval_hours"],
    OPEN_INTEREST: ["ts", "open_interest"],
    SPREAD: ["ts", "bid", "ask", "spread", "spread_bps"],
}


def timeframe_ms(timeframe: str) -> int:
    try:
        return TIMEFRAME_MS[timeframe]
    except KeyError as exc:  # pragma: no cover - guarded by config validation
        raise ValueError(f"unknown timeframe: {timeframe!r}") from exc


def parse_utc_ms(value: str) -> int:
    """Parse an ISO-8601 UTC timestamp (``...Z``) to epoch milliseconds."""
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp() * 1000)


def ms_to_iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=UTC).isoformat().replace("+00:00", "Z")


def expected_grid(start_ms: int, end_ms: int, interval_ms: int) -> list[int]:
    """Inclusive-start, exclusive-end regular grid: [start, start+iv, ... < end]."""
    if interval_ms <= 0:
        raise ValueError("interval_ms must be positive")
    if start_ms % interval_ms != 0:
        # Snap start up to the next grid boundary so timestamps are canonical.
        start_ms = ((start_ms // interval_ms) + 1) * interval_ms
    return list(range(start_ms, end_ms, interval_ms))


@dataclass(frozen=True, slots=True)
class SeriesKey:
    """Identity of one stored series (Appendix B.5 partition keys).

    ``timeframe`` is the grid label: the OHLCV candle size for OHLCV, the base
    sampling timeframe for mark/index/OI/spread, and the funding-interval label
    (e.g. ``8h``) for funding.
    """

    exchange_id: str
    data_type: str
    symbol: str
    timeframe: str

    @property
    def interval_ms(self) -> int:
        return timeframe_ms(self.timeframe)

    @property
    def columns(self) -> list[str]:
        return COLUMNS[self.data_type]

    def symbol_path(self) -> str:
        """Filesystem-safe symbol token (``BTC/USDT:USDT`` -> ``BTC_USDT_USDT``)."""
        return self.symbol.replace("/", "_").replace(":", "_")

    def label(self) -> str:
        return f"{self.symbol}:{self.data_type}:{self.timeframe}"
