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

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import structlog

_log = structlog.get_logger("live.data_manager")


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
        self._last_err_log_ms = -1  # throttle per-symbol fetch-error logging

    # -- polling --------------------------------------------------------- #
    def poll(self, now_ms: int, *, should_stop: Callable[[], bool] | None = None) -> DataHealth:
        """Pull the latest bar per symbol, update freshness, and compute halts.

        ``should_stop`` (the live loop's cancel/kill check) is polled BETWEEN per-symbol fetches so
        a Stop is honored mid-cycle even when the exchange is slow — otherwise a many-symbol poll
        could sit through several fetch timeouts before the loop's top-level cancel check is reached
        (the freeze that made stuck basket sessions un-cancellable). A per-symbol fetch that raises
        is treated as "no fresh bar" (the symbol goes stale) and logged, not propagated — a
        transient blip degrades to a visible halt-and-sleep instead of killing the session."""
        connected = bool(self.source.connected())
        for sym in self.symbols:
            if should_stop is not None and should_stop():
                break  # bail mid-cycle; the loop's top-level check will exit next iteration
            st = self._state[sym]
            if connected:
                try:
                    got = self.source.latest_bar(sym)
                except Exception as exc:  # noqa: BLE001 - transient fetch failure ⇒ symbol stale
                    got = None
                    self._log_fetch_error(sym, exc, now_ms)
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

    def _log_fetch_error(self, symbol: str, exc: Exception, now_ms: int) -> None:
        """Surface a live-feed fetch failure at most once per 5 min so a persistent outage is
        visible (the stall used to be totally silent) without flooding the log every poll."""
        if self._last_err_log_ms >= 0 and now_ms - self._last_err_log_ms < 300_000:
            return
        self._last_err_log_ms = now_ms
        _log.warning(
            "live_feed_fetch_error",
            symbol=symbol,
            error=f"{type(exc).__name__}: {exc}",
        )

    # -- queries (Section 8: prevent feature calc from stale data) ------- #
    def is_fresh(self, symbol: str) -> bool:
        st = self._state.get(symbol)
        return st is not None and not st.stale and not st.integrity_fault

    def stale_symbols(self) -> set[str]:
        return {s for s, st in self._state.items() if st.stale}

    # -- reconnect backfill --------------------------------------------- #
    def backfill_after_reconnect(
        self, symbol: str, now_ms: int, *, since_ms: int | None = None
    ) -> list[dict]:
        """REST-backfill the closed bars missed since the last seen bar and return them (sorted by
        ts), advancing this symbol's freshness watermark. The live feed appends the returned rows to
        its rolling reader so a reconnect gap doesn't leave feature lookback silently holed (M13).

        ``since_ms`` overrides the start anchor — the feed passes its ROLLING READER's watermark,
        which is the true source of the gap (the manager's own freshness watermark may already have
        advanced past it via ``poll``)."""
        st = self._state.get(symbol)
        if st is None:
            return []
        since = (
            since_ms
            if since_ms is not None
            else (st.last_bar_ts + self.interval_ms)
            if st.last_bar_ts >= 0
            else now_ms - self.stale_after_ms
        )
        rows = sorted(self.source.backfill(symbol, since, now_ms) or [], key=lambda r: int(r["ts"]))
        if rows:
            st.last_bar_ts = max(st.last_bar_ts, max(int(r["ts"]) for r in rows))
            st.last_update_ms = now_ms
            st.stale = False
        return rows

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
        self,
        exchange_id: str = "bybit",
        timeframe: str = "1m",
        *,
        client: Any | None = None,
        exchange_env: str = "live",
    ):
        from src.data.ccxt_source import CcxtDataSource

        # Live polling, NOT a download: fail fast. The download source retries 8× with backoff to
        # 60s (up to ~3min of uninterruptible sleep per call) — appropriate for a multi-year fetch,
        # fatal for a live poll loop that must stay cancellable. Here a call that fails is retried
        # by the NEXT poll cycle a few seconds later, so cap retries low, keep the backoff short,
        # and set a hard socket timeout so one hung fetch can't freeze the session (the 24h freeze).
        self._src = CcxtDataSource(
            exchange_id,
            client=client,
            exchange_env=exchange_env,
            max_retries=2,
            retry_base_sec=0.5,
            retry_max_sec=2.0,
            timeout_ms=15_000,
        )
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
