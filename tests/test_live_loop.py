"""M7: the live trading loop (replay-driven; paper venue + injected testnet venue).

Hermetic — a seeded lake feeds the SAME decision pipeline tick by tick. Covers paper
execution, the kill-switch halt, mode validation, and driving a real (mocked) ccxt
testnet venue so the loop's venue-agnostic path is exercised end to end.
"""

from __future__ import annotations

import pytest
from src.config import Settings
from src.data.config import DataConfig, ValidationThresholds
from src.data.schema import (
    FUNDING,
    INDEX,
    MARK,
    OHLCV,
    OPEN_INTEREST,
    SPREAD,
    SeriesKey,
    timeframe_ms,
)
from src.data.source import DeterministicSource
from src.data.store import SeriesStore
from src.exchange.metadata import load_metadata_config
from src.execution.live_venue import CcxtLiveVenue
from src.killswitch import KillSwitch
from src.live.loop import LiveLoop, ReplayFeed
from src.paper.lake import build_lake_paper_inputs

EX = "bybit"
SYM = "BTC/USDT:USDT"
TF = "5m"
OI_TF = "1h"
FUND = "8h"


def _seed(store: SeriesStore, start: int, end: int) -> None:
    src = DeterministicSource(EX)
    for dt, tf in (
        (OHLCV, TF),
        (MARK, TF),
        (INDEX, TF),
        (SPREAD, TF),
        (OPEN_INTEREST, OI_TF),
        (FUNDING, FUND),
    ):
        key = SeriesKey(EX, dt, SYM, tf)
        store.write(key, src.fetch(key, start, end))


def _cfg(start: int, end: int) -> DataConfig:
    return DataConfig(
        exchange_id=EX,
        data_version="t",
        symbols=[SYM],
        timeframes=[TF],
        base_timeframe=TF,
        funding_interval_hours=8,
        required_series=[OHLCV, MARK, INDEX, FUNDING, OPEN_INTEREST, SPREAD],
        window_start_ms=start,
        window_end_ms=end,
        thresholds=ValidationThresholds(),
        oi_timeframe=OI_TF,
    )


def _feed(tmp_path) -> ReplayFeed:
    store = SeriesStore(tmp_path)
    start, end = 0, 400 * timeframe_ms(TF)
    _seed(store, start, end)
    inputs, _, _ = build_lake_paper_inputs(
        _cfg(start, end), timeframe=TF, symbols=[SYM], store=store
    )
    return ReplayFeed(inputs)


def test_replay_feed_groups_in_time_order(tmp_path) -> None:
    feed = _feed(tmp_path)
    tss = [ts for ts, _ in feed.groups()]
    assert tss == sorted(tss) and len(tss) == len(set(tss))


def test_live_loop_paper_executes(tmp_path) -> None:
    feed = _feed(tmp_path)
    result = LiveLoop(mode="paper").run(feed, session_name="t")
    assert not result.halted
    assert result.ticks  # processed decision times
    assert result.executed > 0
    assert result.executed + result.rejected == sum(t.candidates for t in result.ticks)
    assert result.session.session_id.startswith("paper:")


def test_live_loop_halts_on_kill_switch(tmp_path) -> None:
    feed = _feed(tmp_path)
    ks = KillSwitch()
    ks.engage(reason="test")
    try:
        result = LiveLoop(mode="paper", kill_switch=ks).run(feed, session_name="t")
    finally:
        ks.disengage()
    assert result.halted
    assert result.executed == 0  # nothing trades once halted


def test_live_loop_rejects_bad_mode() -> None:
    with pytest.raises(ValueError, match="mode must be one of"):
        LiveLoop(mode="bogus")


class FakeCcxt:
    def __init__(self) -> None:
        self.orders: list[dict] = []

    def create_order(self, symbol, type, side, qty, price, params=None):  # noqa: A002
        self.orders.append({"symbol": symbol, "side": side, "params": params or {}})
        return {"average": price or 100.0, "filled": qty, "fee": {"cost": 0.0}}

    def cancel_order(self, *a, **k):
        return {}

    def fetch_positions(self):
        return []


def test_live_loop_drives_testnet_venue(tmp_path) -> None:
    feed = _feed(tmp_path)
    fake = FakeCcxt()
    settings = Settings(
        _env_file=None, exchange_env="testnet", exchange_api_key="k", exchange_api_secret="s"
    )
    venue = CcxtLiveVenue(load_metadata_config(), settings, client=fake)
    result = LiveLoop(mode="testnet", venue=venue, settings=settings).run(feed, session_name="t")
    assert result.session.session_id.startswith("testnet:")
    assert result.executed > 0
    assert fake.orders  # real (testnet) orders were placed through the loop
    # every order carried the ownership prefix as clientOrderId
    assert all(o["params"].get("clientOrderId") for o in fake.orders)
