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
from src.execution.venue import SimulatedVenue, VenuePosition
from src.killswitch import KillSwitch
from src.live.loop import LiveLoop, ReplayFeed
from src.paper.lake import build_lake_paper_inputs

EX = "bybit"
SYM = "BTC/USDT:USDT"
TF = "5m"
OI_TF = "1h"
FUND = "8h"


@pytest.fixture(autouse=True)
def _clear_kill_switch():
    """The kill switch is global file/redis state; ensure it is clear so these loop tests
    are not affected by a lingering engagement from another test file (full-suite ordering)."""
    from src.killswitch import KillSwitch

    KillSwitch().disengage()
    yield
    KillSwitch().disengage()


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


def test_time_stop_flattens_only_aged_positions() -> None:
    """The bot-side time-stop flattens an owned position once it reaches its hold_bars horizon and
    leaves younger ones alone (the exchange can't express 'close after N bars')."""
    settings = _testnet_settings()
    venue = SimulatedVenue(load_metadata_config())
    loop = LiveLoop(mode="testnet", venue=venue, settings=settings)
    session = loop.engine.new_session("t")
    iv = timeframe_ms(TF)
    for sym in ("A/USDT:USDT", "B/USDT:USDT"):
        loop.venue.positions[sym] = VenuePosition(symbol=sym, side=1, qty=1.0, entry_price=100.0)
    loop._bar_iv = iv
    loop._open_age["A/USDT:USDT"] = (0, 2)  # entered at ts 0, hold 2 bars → aged at ts 2·iv
    loop._open_age["B/USDT:USDT"] = (0, 10)  # hold 10 bars → still young

    loop._apply_time_stops(2 * iv, session)
    assert "A/USDT:USDT" not in loop.venue.positions  # time-stopped
    assert "A/USDT:USDT" not in loop._open_age  # and stopped tracking it
    assert "B/USDT:USDT" in loop.venue.positions  # younger position untouched
    assert any(e.get("phase") == "time_stop" for e in session.reconciliation_events)


def test_time_stop_is_a_noop_until_bar_interval_known() -> None:
    """Before the bar interval is inferred (tick 0) the time-stop does nothing — it can't compute
    an age yet, so it never closes a fresh position prematurely."""
    settings = _testnet_settings()
    venue = SimulatedVenue(load_metadata_config())
    loop = LiveLoop(mode="testnet", venue=venue, settings=settings)
    session = loop.engine.new_session("t")
    loop.venue.positions["A/USDT:USDT"] = VenuePosition(
        symbol="A/USDT:USDT", side=1, qty=1.0, entry_price=100.0
    )
    loop._open_age["A/USDT:USDT"] = (0, 1)
    loop._apply_time_stops(10 * timeframe_ms(TF), session)  # _bar_iv still 0
    assert "A/USDT:USDT" in loop.venue.positions  # not closed (interval unknown)


class FakeCcxt:
    def __init__(self, positions=None, open_orders=None) -> None:
        self.orders: list[dict] = []
        self._positions = positions or []
        self._open_orders = open_orders or []

    def create_order(self, symbol, type, side, qty, price, params=None):  # noqa: A002
        self.orders.append({"symbol": symbol, "side": side, "params": params or {}})
        return {"average": price or 100.0, "filled": qty, "fee": {"cost": 0.0}}

    def cancel_order(self, *a, **k):
        return {}

    def fetch_positions(self):
        return self._positions

    def fetch_open_orders(self):
        return self._open_orders


def test_live_loop_drives_testnet_venue(tmp_path) -> None:
    feed = _feed(tmp_path)
    fake = FakeCcxt()
    settings = Settings(
        _env_file=None,
        exchange_env="testnet",
        exchange_id="skeleton",  # matches the injected skeleton metadata (venue guard, Section 6)
        exchange_api_key="k",
        exchange_api_secret="s",
    )
    venue = CcxtLiveVenue(load_metadata_config(), settings, client=fake)
    result = LiveLoop(mode="testnet", venue=venue, settings=settings).run(feed, session_name="t")
    assert result.session.session_id.startswith("testnet:")
    assert result.executed > 0
    assert fake.orders  # real (testnet) orders were placed through the loop
    # every order carried the ownership prefix as clientOrderId
    assert all(o["params"].get("clientOrderId") for o in fake.orders)


_PREFIX = "QBOT_LOCAL_v1_"


def _testnet_settings(**over) -> Settings:
    base = {
        "_env_file": None,
        "exchange_env": "testnet",
        "exchange_id": "skeleton",  # offline test venue matches the skeleton metadata
        "exchange_api_key": "k",
        "exchange_api_secret": "s",
        "order_client_id_prefix": _PREFIX,
    }
    base.update(over)
    return Settings(**base)


def test_startup_reconciliation_halts_on_foreign_position(tmp_path) -> None:
    """A pre-existing FOREIGN (manual) position on the exchange halts the loop before any
    tick — we never trade on top of an un-attributable book (Section 7)."""
    feed = _feed(tmp_path)
    fake = FakeCcxt(
        positions=[
            {
                "symbol": "XRP/USDT:USDT",
                "side": "long",
                "contracts": 10.0,
                "entryPrice": 0.5,
                "info": {"clientOrderId": "MANUAL_human_1"},
            }
        ]
    )
    settings = _testnet_settings()
    venue = CcxtLiveVenue(load_metadata_config(), settings, client=fake)
    result = LiveLoop(mode="testnet", venue=venue, settings=settings).run(feed, session_name="t")
    assert result.halted
    assert result.executed == 0  # never traded
    assert result.startup_recon is not None and result.startup_recon.halt_required
    assert "XRP/USDT:USDT" in result.startup_recon.foreign_positions
    assert not fake.orders


def test_startup_reconciliation_adopts_owned_and_runs(tmp_path) -> None:
    """An OWNED position already on the exchange (carries our prefix) is adopted into the
    mirror and does not halt; the loop runs normally."""
    feed = _feed(tmp_path)
    fake = FakeCcxt(
        positions=[
            {
                "symbol": "ETH/USDT:USDT",
                "side": "long",
                "contracts": 0.1,
                "entryPrice": 3_000.0,
                "info": {"clientOrderId": f"{_PREFIX}entry_1"},
            }
        ]
    )
    settings = _testnet_settings()
    venue = CcxtLiveVenue(load_metadata_config(), settings, client=fake)
    result = LiveLoop(mode="testnet", venue=venue, settings=settings).run(feed, session_name="t")
    assert not result.halted
    assert result.startup_recon is not None and not result.startup_recon.halt_required
    assert "ETH/USDT:USDT" in result.startup_recon.owned_positions
    assert "ETH/USDT:USDT" in venue.positions  # adopted into the mirror


def test_per_tick_reconciliation_halts_on_foreign_position(tmp_path) -> None:
    """Mid-session, a foreign/manual position appearing on the real exchange book halts the loop
    (the per-tick Section-7 control now re-pulls actual exchange state, not the venue mirror)."""
    settings = _testnet_settings()
    fake = FakeCcxt(positions=[
        {"symbol": "XRP/USDT:USDT", "side": "long", "contracts": 5.0, "entryPrice": 0.5,
         "info": {"clientOrderId": "MANUAL_human_1"}},
    ])
    venue = CcxtLiveVenue(load_metadata_config(), settings, client=fake)
    loop = LiveLoop(mode="testnet", venue=venue, settings=settings)
    session = loop.engine.new_session("t")
    assert loop._reconcile_live(session) is True  # foreign → halt
    assert session.foreign_order_halt_triggered


def test_per_tick_reconciliation_refreshes_owned_protection(tmp_path) -> None:
    """Per tick, an owned exchange position is refreshed into the mirror with its REAL stop
    state (so an owned position lacking an exchange-side stop is visible), and a clean book
    does not halt."""
    settings = _testnet_settings()
    fake = FakeCcxt(positions=[
        {"symbol": "ETH/USDT:USDT", "side": "long", "contracts": 0.1, "entryPrice": 3_000.0,
         "stopLossPrice": 2_950.0,
         "info": {"clientOrderId": f"{_PREFIX}e1", "stopLoss": "2950"}},
    ])
    venue = CcxtLiveVenue(load_metadata_config(), settings, client=fake)
    loop = LiveLoop(mode="testnet", venue=venue, settings=settings)
    session = loop.engine.new_session("t")
    assert loop._reconcile_live(session) is False  # clean book, no halt
    assert "ETH/USDT:USDT" in venue.positions
    assert venue.positions["ETH/USDT:USDT"].has_exchange_side_stop() is True  # real stop read


def test_per_tick_reconciliation_debounce_drops_closed_position(tmp_path) -> None:
    """An owned mirror position the exchange stops listing (closed via its SL/TP) is retired after
    a debounce, freeing its concurrency slot — not leaked forever (and not false-dropped on the
    first absent tick, which could be fill latency)."""
    settings = _testnet_settings()
    fake = FakeCcxt()  # exchange reports NO open positions
    venue = CcxtLiveVenue(load_metadata_config(), settings, client=fake)
    from src.execution.venue import VenuePosition

    venue.positions["BTC/USDT:USDT"] = VenuePosition(
        symbol="BTC/USDT:USDT", side=1, qty=0.01, entry_price=50_000.0, owned=True
    )
    loop = LiveLoop(mode="testnet", venue=venue, settings=settings)
    # Seed the ENGINE risk mirror too — it must be pruned in lock-step so the Section-17 caps
    # release the slot (not just the bounded-live guard).
    from src.risk.portfolio import Position

    loop.engine._open_positions["BTC/USDT:USDT"] = Position(
        symbol="BTC/USDT:USDT", side=1, qty=0.01, entry_price=50_000.0,
        risk_amount=10.0, beta_to_btc=1.0, regime="low_vol_up",
    )
    session = loop.engine.new_session("t")
    loop._reconcile_live(session)
    assert "BTC/USDT:USDT" in venue.positions  # 1st absent tick → kept (debounce)
    loop._reconcile_live(session)
    assert "BTC/USDT:USDT" not in venue.positions  # 2nd absent tick → dropped (slot freed)
    assert "BTC/USDT:USDT" not in loop.engine._open_positions  # engine risk mirror pruned too


def test_startup_reconciliation_clean_paper_is_noop(tmp_path) -> None:
    """Offline paper has no real exchange book — startup reconciliation is a clean no-op."""
    feed = _feed(tmp_path)
    result = LiveLoop(mode="paper").run(feed, session_name="t")
    assert result.startup_recon is not None and not result.startup_recon.halt_required
    assert not result.halted


def test_live_loop_halts_on_data_integrity_failure(tmp_path) -> None:
    """Section 8: an exchange-wide data-integrity failure halts the loop like a kill switch."""
    from src.live.data_manager import DataHealth

    feed = _feed(tmp_path)

    class _HaltingDataManager:
        def poll(self, now_ms):
            return DataHealth(ts=now_ms, connected=False, exchange_halt=True, reason="disconnected")

    result = LiveLoop(mode="paper", data_manager=_HaltingDataManager()).run(feed, session_name="t")
    assert result.halted
    assert result.executed == 0  # nothing trades while live data integrity is down
