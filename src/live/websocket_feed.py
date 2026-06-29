"""Real-time websocket market feed (AGENTS.md Section 8).

A :class:`~src.live.data_manager.FeedSource` backed by **ccxt.pro websocket streams**. It runs
an asyncio event loop on a daemon thread, subscribing to ``watch_ohlcv`` per symbol and caching
the latest closed bar in a thread-safe dict; the synchronous ``latest_bar`` / ``connected`` /
``backfill`` surface the :class:`~src.live.data_manager.LiveDataManager` already consumes reads
that cache without blocking. ccxt.pro auto-reconnects its watch loops; on error we back off and
retry, and ``connected()`` reflects stream health so the data manager's staleness/disconnect
halts work against a true live feed.

The watch coroutine is injectable, so tests drive the threading + cache deterministically with a
fake stream — no network or keys. Production uses ``ccxt.pro`` (no API key needed for public OHLCV).
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

# watcher(symbol) -> awaitable yielding a batch of ccxt OHLCV candles ([ts,o,h,l,c,v], ...)
Watcher = Callable[[str], Awaitable[list]]


def _now_ms() -> int:
    return int(time.time() * 1000)


class WebsocketFeedSource:
    """ccxt.pro websocket FeedSource (default Bybit testnet), behind the sync FeedSource API."""

    def __init__(
        self,
        symbols: Sequence[str],
        *,
        exchange_id: str = "bybit",
        timeframe: str = "1m",
        exchange_env: str = "testnet",
        watcher: Watcher | None = None,
        rest_source: Any | None = None,
        reconnect_sec: float = 1.0,
    ) -> None:
        self.symbols = list(symbols)
        self.exchange_id = exchange_id
        self.timeframe = timeframe
        self.exchange_env = exchange_env
        self._watcher = watcher
        self._rest_source = rest_source
        self._reconnect_sec = reconnect_sec

        self._cache: dict[str, tuple[int, dict]] = {}
        self._lock = threading.Lock()
        self._last_msg_ms = -1
        self._error: str | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._ex: Any = None

    # -- lifecycle ------------------------------------------------------- #
    def start(self) -> WebsocketFeedSource:
        if self._running:
            return self
        self._running = True
        self._thread = threading.Thread(target=self._run, name="ws-feed", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._running = False
        thread = self._thread
        if thread is not None:
            thread.join(timeout=3.0)
        self._thread = None

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._main())
        except Exception as exc:  # noqa: BLE001 - surface as a disconnect, never crash the thread
            self._error = str(exc)
        finally:
            loop.close()

    async def _main(self) -> None:
        if self._watcher is None:
            import ccxt.pro as ccxtpro  # noqa: PLC0415 - optional ws dependency, loaded lazily

            from src.execution.live_venue import apply_exchange_env

            klass = getattr(ccxtpro, self.exchange_id)
            self._ex = klass({"enableRateLimit": True, "options": {"defaultType": "swap"}})
            apply_exchange_env(self._ex, self.exchange_env)  # live | testnet | demo
            self._watcher = lambda sym: self._ex.watch_ohlcv(sym, self.timeframe)
        try:
            await asyncio.gather(*(self._watch(s) for s in self.symbols), return_exceptions=True)
        finally:
            if self._ex is not None and hasattr(self._ex, "close"):
                with contextlib.suppress(Exception):
                    await self._ex.close()

    async def _watch(self, symbol: str) -> None:
        assert self._watcher is not None
        while self._running:
            try:
                candles = await self._watcher(symbol)
            except Exception as exc:  # noqa: BLE001 - disconnect: back off and retry (reconnect)
                self._error = str(exc)
                await asyncio.sleep(self._reconnect_sec)
                continue
            self._error = None
            if candles:
                # ccxt.pro emits the currently-FORMING candle as the last element; advancing on it
                # feeds a partial bar into the feature pipeline. Take the last candle whose close
                # time has passed (ts + interval <= now); skip the cycle if only a forming bar is in.
                from src.data.schema import timeframe_ms

                iv = timeframe_ms(self.timeframe)
                now = _now_ms()
                closed = [c for c in candles if int(c[0]) + iv <= now]
                with self._lock:
                    self._last_msg_ms = now  # a message arrived → stream alive, forming or not
                    if closed:
                        c = closed[-1]
                        self._cache[symbol] = (int(c[0]), {
                            "ts": int(c[0]),
                            "open": float(c[1]),
                            "high": float(c[2]),
                            "low": float(c[3]),
                            "close": float(c[4]),
                            "volume": float(c[5]) if c[5] is not None else 0.0,
                        })

    # -- FeedSource API -------------------------------------------------- #
    def connected(self) -> bool:
        alive = self._thread is not None and self._thread.is_alive()
        return bool(self._running and alive and self._error is None and self._last_msg_ms >= 0)

    def latest_bar(self, symbol: str) -> tuple[int, dict] | None:
        with self._lock:
            return self._cache.get(symbol)

    def backfill(self, symbol: str, since_ms: int, end_ms: int) -> list[dict]:
        """REST gap-fill after a reconnect (websockets do not replay missed history)."""
        from src.data.schema import OHLCV, SeriesKey

        src = self._rest_source
        if src is None:
            from src.data.ccxt_source import CcxtDataSource

            # Reconnect gap-fill must read the SAME environment as the websocket (testnet→testnet),
            # not mainnet, or the backfilled bars come from a different venue than the live stream.
            src = CcxtDataSource(self.exchange_id, exchange_env=self.exchange_env)
            self._rest_source = src
        return src.fetch(
            SeriesKey(self.exchange_id, OHLCV, symbol, self.timeframe), since_ms, end_ms
        )


def live_feed_source(
    symbols: Sequence[str],
    *,
    transport: str = "rest",
    exchange_id: str = "bybit",
    timeframe: str = "1m",
    exchange_env: str = "testnet",
) -> Any:
    """Build the live market-data transport: ``rest`` polling (default) or ``ws`` websocket.

    The websocket source is started before return. Both satisfy the FeedSource Protocol the
    LiveDataManager consumes, so staleness/disconnect/backfill/halt logic is transport-agnostic.
    """
    if transport == "ws":
        return WebsocketFeedSource(
            symbols, exchange_id=exchange_id, timeframe=timeframe, exchange_env=exchange_env
        ).start()
    if transport == "rest":
        from src.live.data_manager import CcxtPollingSource

        return CcxtPollingSource(exchange_id, timeframe, exchange_env=exchange_env)
    raise ValueError(f"transport must be 'ws' or 'rest', got {transport!r}")
