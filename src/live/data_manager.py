"""Live Data Manager (AGENTS.md Section 8).

Guards live decisions against bad real-time data. It tracks per-symbol freshness from a
streaming/polling :class:`FeedSource`, detects **stale streams** and **disconnects**,
**backfills via REST after a reconnect**, **compares websocket vs REST** where both are
available, and **prevents feature calculation from stale data** — halting the affected
symbol when its critical live data is stale, and halting *all* trading when exchange-wide
data integrity fails.

The manager is transport-agnostic behind :class:`FeedSource`: the production source polls
ccxt REST (and a ccxt.pro websocket can implement the same Protocol); tests inject a fake.
``now_ms`` is supplied by the caller (the live loop's clock) so the manager stays pure and
deterministic.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


class FeedSource(Protocol):
    """A real-time market-data transport (websocket or REST polling)."""

    def connected(self) -> bool: ...

    def latest_bar(self, symbol: str) -> tuple[int, dict] | None:
        """Most recent CLOSED bar as ``(close_ts_ms, ohlcv_row)`` or ``None``."""

    def backfill(self, symbol: str, since_ms: int, end_ms: int) -> list[dict]:
        """REST gap-fill of closed bars in ``[since_ms, end_ms)`` (after a reconnect)."""


@dataclass(slots=True)
class SymbolFreshness:
    last_bar_ts: int = -1  # close ts of the freshest bar seen
    last_update_ms: int = -1  # wall clock when that bar arrived
    stale: bool = False
    integrity_fault: bool = False  # ws-vs-REST divergence beyond tolerance


@dataclass(slots=True)
class DataHealth:
    ts: int
    connected: bool
    fresh: list[str] = field(default_factory=list)
    stale: list[str] = field(default_factory=list)
    exchange_halt: bool = False  # halt ALL trading
    reason: str = ""


class LiveDataManager:
    """Per-symbol freshness + disconnect/integrity halts for the live feed (Section 8)."""

    def __init__(
        self,
        source: FeedSource,
        symbols: Sequence[str],
        *,
        interval_ms: int,
        stale_after_intervals: int = 2,
        ws_rest_tol_bps: float = 10.0,
    ) -> None:
        self.source = source
        self.symbols = list(symbols)
        self.interval_ms = int(interval_ms)
        self.stale_after_ms = int(stale_after_intervals) * int(interval_ms)
        self.ws_rest_tol_bps = ws_rest_tol_bps
        self._state: dict[str, SymbolFreshness] = {s: SymbolFreshness() for s in self.symbols}

    # -- polling --------------------------------------------------------- #
    def poll(self, now_ms: int) -> DataHealth:
        """Pull the latest bar per symbol, update freshness, and compute halts."""
        connected = bool(self.source.connected())
        for sym in self.symbols:
            st = self._state[sym]
            if connected:
                got = self.source.latest_bar(sym)
                if got is not None:
                    bar_ts, _row = got
                    if bar_ts > st.last_bar_ts:
                        st.last_bar_ts = bar_ts
                        st.last_update_ms = now_ms
            # Stale = disconnected, never seen a bar, or no fresh bar within the window.
            st.stale = (
                (not connected)
                or st.last_update_ms < 0
                or (now_ms - st.last_update_ms) > self.stale_after_ms
            )

        fresh = [s for s in self.symbols if not self._state[s].stale]
        stale = [s for s in self.symbols if self._state[s].stale]
        # Exchange-wide halt: disconnected, every symbol stale, or any integrity fault.
        integrity = any(self._state[s].integrity_fault for s in self.symbols)
        halt = (not connected) or (len(fresh) == 0 and bool(self.symbols)) or integrity
        reason = ""
        if not connected:
            reason = "exchange disconnected"
        elif integrity:
            reason = "ws/REST integrity fault"
        elif halt:
            reason = "all symbols stale"
        return DataHealth(
            ts=now_ms,
            connected=connected,
            fresh=fresh,
            stale=stale,
            exchange_halt=halt,
            reason=reason,
        )

    # -- queries (Section 8: prevent feature calc from stale data) ------- #
    def is_fresh(self, symbol: str) -> bool:
        st = self._state.get(symbol)
        return st is not None and not st.stale and not st.integrity_fault

    def stale_symbols(self) -> set[str]:
        return {s for s, st in self._state.items() if st.stale}

    # -- reconnect backfill --------------------------------------------- #
    def backfill_after_reconnect(self, symbol: str, now_ms: int) -> int:
        """REST-backfill the gap since the last seen bar; returns bars recovered."""
        st = self._state.get(symbol)
        if st is None:
            return 0
        since = (
            (st.last_bar_ts + self.interval_ms)
            if st.last_bar_ts >= 0
            else now_ms - self.stale_after_ms
        )
        rows = self.source.backfill(symbol, since, now_ms) or []
        if rows:
            st.last_bar_ts = max(st.last_bar_ts, max(int(r["ts"]) for r in rows))
            st.last_update_ms = now_ms
            st.stale = False
        return len(rows)

    # -- ws vs REST cross-check ----------------------------------------- #
    def compare_ws_rest(self, symbol: str, ws_close: float, rest_close: float) -> bool:
        """Flag (and record) a ws-vs-REST divergence beyond tolerance. Returns True if OK."""
        if rest_close <= 0:
            return True
        bps = abs(ws_close - rest_close) / rest_close * 10_000.0
        ok = bps <= self.ws_rest_tol_bps
        st = self._state.get(symbol)
        if st is not None:
            st.integrity_fault = not ok
        return ok


class CcxtPollingSource:
    """Production :class:`FeedSource` over ccxt REST (a ccxt.pro ws can swap in later)."""

    def __init__(
        self, exchange_id: str = "bybit", timeframe: str = "1m", *, client: Any | None = None
    ):
        from src.data.ccxt_source import CcxtDataSource

        self._src = CcxtDataSource(exchange_id, client=client)
        self._ex = exchange_id
        self.timeframe = timeframe

    def connected(self) -> bool:
        return self._src.ping()

    def latest_bar(self, symbol: str) -> tuple[int, dict] | None:
        from src.data.schema import OHLCV, SeriesKey, timeframe_ms

        iv = timeframe_ms(self.timeframe)
        # Poll a short recent window; the last CLOSED bar is the freshest.
        import time

        now = int(time.time() * 1000)
        rows = self._src.fetch(
            SeriesKey(self._ex, OHLCV, symbol, self.timeframe), now - 5 * iv, now
        )
        # Drop the still-FORMING candle: its open-ts (floor(now/iv)*iv) is < now so it passes the
        # fetch's ts<end filter, and returning it would feed a partial bar into the feature pipeline
        # (and a decision_ts in the future). Keep only bars whose close time has passed.
        rows = [r for r in rows if int(r["ts"]) + iv <= now]
        if not rows:
            return None
        last = rows[-1]
        return int(last["ts"]), last

    def backfill(self, symbol: str, since_ms: int, end_ms: int) -> list[dict]:
        from src.data.schema import OHLCV, SeriesKey

        return self._src.fetch(SeriesKey(self._ex, OHLCV, symbol, self.timeframe), since_ms, end_ms)
