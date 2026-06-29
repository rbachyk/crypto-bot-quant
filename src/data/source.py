"""Data sources behind a single :class:`DataSource` interface (Section 5/6).

Phase 2 ships :class:`DeterministicSource`, an OFFLINE source that fabricates
reproducible, schema-valid market data as a **pure function of timestamp** —
``value(ts)`` never depends on the requested range, so a backfill of any
sub-range yields byte-identical rows to a full download (idempotent dedup, and
reproducible features for the Phase 3 FEAT gate). No network is touched.

Real venues are wired later via ccxt + a native SDK fallback behind this same
interface; no strategy/feature/gate ever calls the venue directly (Section 6).
"""

from __future__ import annotations

import abc
import hashlib
import math

from src.data.schema import (
    FUNDING,
    INDEX,
    MARK,
    OHLCV,
    OPEN_INTEREST,
    SPREAD,
    SeriesKey,
    expected_grid,
)

_DAY_MS = 86_400_000


class DataSource(abc.ABC):
    """The only path to raw market data (Section 6)."""

    @abc.abstractmethod
    def fetch(self, key: SeriesKey, start_ms: int, end_ms: int) -> list[dict]:
        """Return rows for ``key`` on its grid within ``[start_ms, end_ms)``."""

    @abc.abstractmethod
    def has_symbol(self, symbol: str) -> bool:
        """Whether the source has history for ``symbol`` at all."""


def _unit(*parts: object) -> float:
    """Deterministic float in [0, 1) from the hash of ``parts``."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


class DeterministicSource(DataSource):
    """Offline, reproducible market-data generator (no network).

    ``missing_symbols`` simulates symbols the exchange genuinely lacks history
    for (used to exercise the DATA-COV insufficient-history / quarantine path).
    """

    def __init__(self, exchange_id: str = "skeleton", missing_symbols: set[str] | None = None):
        self.exchange_id = exchange_id
        self._missing = set(missing_symbols or set())

    # -- public API ------------------------------------------------------ #
    def has_symbol(self, symbol: str) -> bool:
        return symbol not in self._missing

    def fetch(self, key: SeriesKey, start_ms: int, end_ms: int) -> list[dict]:
        if not self.has_symbol(key.symbol):
            return []
        grid = expected_grid(start_ms, end_ms, key.interval_ms)
        builder = _BUILDERS[key.data_type]
        return [builder(self, key, ts) for ts in grid]

    # -- deterministic primitives --------------------------------------- #
    def _base_price(self, symbol: str) -> float:
        # Stable per-symbol base in a realistic range (e.g. 12 .. 60012).
        return 12.0 + _unit("base", symbol) * 60_000.0

    def _close(self, symbol: str, ts: int) -> float:
        """Close price as a pure function of (symbol, ts)."""
        base = self._base_price(symbol)
        cycle = 1.0 + 0.05 * math.sin(2.0 * math.pi * (ts % _DAY_MS) / _DAY_MS)
        noise = 1.0 + (_unit("noise", symbol, ts) - 0.5) * 0.004  # +-0.2%
        return base * cycle * noise


# --------------------------------------------------------------------------- #
# Per-data_type row builders (module-level so they are easy to read/test).     #
# --------------------------------------------------------------------------- #
def _ohlcv(src: DeterministicSource, key: SeriesKey, ts: int) -> dict:
    sym = key.symbol
    open_ = src._close(sym, ts - key.interval_ms)  # continuity: prev close
    close = src._close(sym, ts)
    hi_pad = 1.0 + _unit("hi", sym, ts) * 0.003
    lo_pad = 1.0 - _unit("lo", sym, ts) * 0.003
    high = max(open_, close) * hi_pad
    low = min(open_, close) * lo_pad
    volume = 10.0 + _unit("vol", sym, ts) * 1_000.0
    return {
        "ts": ts,
        "open": round(open_, 6),
        "high": round(high, 6),
        "low": round(low, 6),
        "close": round(close, 6),
        "volume": round(volume, 6),
    }


def _mark(src: DeterministicSource, key: SeriesKey, ts: int) -> dict:
    premium = 1.0 + (_unit("mark", key.symbol, ts) - 0.5) * 0.0006
    return {"ts": ts, "mark_price": round(src._close(key.symbol, ts) * premium, 6)}


def _index(src: DeterministicSource, key: SeriesKey, ts: int) -> dict:
    premium = 1.0 + (_unit("index", key.symbol, ts) - 0.5) * 0.0004
    return {"ts": ts, "index_price": round(src._close(key.symbol, ts) * premium, 6)}


def _funding(src: DeterministicSource, key: SeriesKey, ts: int) -> dict:
    rate = (_unit("funding", key.symbol, ts) - 0.5) * 0.0010  # +-0.05%
    return {
        "ts": ts,
        "funding_rate": round(rate, 8),
        "funding_interval_hours": key.interval_ms // 3_600_000,
    }


def _open_interest(src: DeterministicSource, key: SeriesKey, ts: int) -> dict:
    oi = 1_000_000.0 * (1.0 + _unit("oi", key.symbol, ts))
    return {"ts": ts, "open_interest": round(oi, 4)}


def _spread(src: DeterministicSource, key: SeriesKey, ts: int) -> dict:
    mid = src._close(key.symbol, ts)
    frac = 0.0002 + _unit("spread", key.symbol, ts) * 0.0008  # 2 .. 10 bps
    bid = mid * (1.0 - frac / 2.0)
    ask = mid * (1.0 + frac / 2.0)
    return {
        "ts": ts,
        "bid": round(bid, 6),
        "ask": round(ask, 6),
        "spread": round(ask - bid, 6),
        "spread_bps": round(frac * 10_000.0, 4),
    }


_BUILDERS = {
    OHLCV: _ohlcv,
    MARK: _mark,
    INDEX: _index,
    FUNDING: _funding,
    OPEN_INTEREST: _open_interest,
    SPREAD: _spread,
}


def get_data_source(exchange_id: str | None = None, *, exchange_env: str = "live") -> DataSource:
    """Return the active data source for ``exchange_id``.

    ``skeleton`` (the default, used by tests + the offline data gates) returns the deterministic
    offline source; any real exchange id (e.g. ``bybit``) returns the live ccxt-backed source.
    Real downloads are opt-in via config/CLI (``configs/data.yaml`` ships ``skeleton``).

    ``exchange_env`` routes the ccxt client: testnet reads testnet klines (matching its venue),
    demo/live read mainnet (Bybit's demo endpoint serves no public klines). Passed so a live
    session's SEED/backfill REST reads the same environment as its streaming feed.
    """
    eid = exchange_id or "skeleton"
    if eid == "skeleton":
        return DeterministicSource(eid)
    from src.data.ccxt_source import CcxtDataSource

    return CcxtDataSource(eid, exchange_env=exchange_env)
