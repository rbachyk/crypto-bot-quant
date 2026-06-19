"""Section 8: the real-time websocket FeedSource (driven by a fake async stream — no network)."""

from __future__ import annotations

import asyncio
import time

import pytest
from src.live.data_manager import CcxtPollingSource, LiveDataManager
from src.live.websocket_feed import WebsocketFeedSource, live_feed_source

SYM = "BTC/USDT:USDT"
_CANDLE = [1_700_000_000_000, 100.0, 101.0, 99.0, 100.5, 10.0]


class FakeWatcher:
    """An injectable async watch coroutine standing in for ccxt.pro watch_ohlcv."""

    def __init__(self, *, candle=None, raises=False, delay=0.02) -> None:
        self.candle = candle
        self.raises = raises
        self.delay = delay
        self.calls = 0

    async def __call__(self, symbol: str) -> list:
        self.calls += 1
        await asyncio.sleep(self.delay)
        if self.raises:
            raise RuntimeError("ws disconnected")
        return [self.candle] if self.candle else []


def _wait_for(predicate, timeout: float = 3.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_ws_source_caches_latest_bar_and_is_connected() -> None:
    src = WebsocketFeedSource([SYM], watcher=FakeWatcher(candle=_CANDLE))
    src.start()
    try:
        assert _wait_for(lambda: src.latest_bar(SYM) is not None), "ws never delivered a bar"
        ts, bar = src.latest_bar(SYM)
        assert ts == _CANDLE[0] and bar["close"] == 100.5
        assert src.connected() is True
    finally:
        src.stop()
    assert not src.connected()  # stopped → no longer connected


def test_ws_source_disconnect_is_visible() -> None:
    src = WebsocketFeedSource([SYM], watcher=FakeWatcher(raises=True), reconnect_sec=0.02)
    src.start()
    try:
        # The stream only ever errors → never connected, never caches a bar.
        assert _wait_for(lambda: not src.connected() and src._error is not None)  # noqa: SLF001
        assert src.latest_bar(SYM) is None
    finally:
        src.stop()


def test_ws_source_feeds_the_data_manager() -> None:
    src = WebsocketFeedSource([SYM], watcher=FakeWatcher(candle=_CANDLE))
    src.start()
    try:
        assert _wait_for(lambda: src.latest_bar(SYM) is not None)
        mgr = LiveDataManager(src, [SYM], interval_ms=60_000)
        health = mgr.poll(int(time.time() * 1000))
        assert SYM in health.fresh and not health.exchange_halt
        assert mgr.is_fresh(SYM)
    finally:
        src.stop()


def test_live_feed_source_factory() -> None:
    assert isinstance(live_feed_source([SYM], transport="rest"), CcxtPollingSource)
    with pytest.raises(ValueError, match="transport must be"):
        live_feed_source([SYM], transport="bogus")
