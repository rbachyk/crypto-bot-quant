"""Execution engine unit tests (AGENTS.md Section 18 / Section 7).

Offline tests of the capital-critical execution module: tick/lot/min-notional-
respecting order building, atomic exchange-resident SL/TP, exchange-native
trailing, cancel/replace, partial fills, slippage measurement, reconciliation,
order ownership, and the revalidate-before-execute / emergency-close guards.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from src.config import Settings
from src.exchange.metadata import load_metadata_config
from src.execution import (
    ExecutionEngine,
    OrderBuilder,
    OwnershipPolicy,
    Reconciler,
    SimulatedVenue,
    load_execution_config,
)
from src.execution.order import BUY, SELL, Order, OrderType
from src.killswitch import KillSwitch
from src.ranking import Candidate
from src.risk import (
    AccountState,
    BreakerInputs,
    PortfolioState,
    RiskManager,
    load_risk_config,
)

BTC = "BTC/USDT:USDT"
EQUITY = 100_000.0


def _settings(tmp: str | None = None) -> Settings:
    kw: dict = {"_env_file": None}
    if tmp:
        kw["data_lake_path"] = Path(tmp) / "dl"
        kw["redis_url"] = "redis://127.0.0.1:1/0"
    return Settings(**kw)


def _meta():
    return load_metadata_config()


def _cand(symbol=BTC, *, stop_frac=0.01, tp_frac=0.02, spread_bps=3.0, **over) -> Candidate:
    base = {
        "symbol": symbol,
        "strategy": "t",
        "strategy_version": "t",
        "side": 1,
        "entry_price": 50_000.0,
        "stop_frac": stop_frac,
        "tp_frac": tp_frac,
        "regime": "low_vol_up",
        "session": 2,
        "spread_bps": spread_bps,
        "slippage_est": 0.0005,
        "latency_ms": 40.0,
    }
    base.update(over)
    return Candidate(**base)  # type: ignore[arg-type]


def _flat():
    return AccountState(
        portfolio=PortfolioState(equity=EQUITY),
        breakers=BreakerInputs(equity=EQUITY, peak_equity=EQUITY, daily_pnl=0.0),
    )


def _rm():
    return RiskManager(load_risk_config(), _meta())


def _engine(settings: Settings, venue: SimulatedVenue):
    return ExecutionEngine(load_execution_config(), _meta(), OwnershipPolicy(settings), venue)


# --------------------------------------------------------------------------- #
# Ownership                                                                    #
# --------------------------------------------------------------------------- #
def test_ownership_prefix_and_is_own() -> None:
    own = OwnershipPolicy(_settings())
    assert own.configured()
    cid = own.new_client_id("entry")
    assert own.is_own(cid)
    assert not own.is_own("MANUAL_foreign_1")
    assert not own.is_own(None)
    assert own.new_client_id("entry") != cid  # unique


def test_all_order_legs_prefixed_and_tagged() -> None:
    s = _settings()
    own = OwnershipPolicy(s)
    builder = OrderBuilder(load_execution_config(), own)
    cand = _cand()
    res = builder.build(cand, _rm().evaluate(cand, _flat()), _meta().spec(BTC))
    assert res.ok and res.plan is not None
    for leg in res.plan.legs():
        assert own.is_own(leg.client_id)
        assert leg.tags["bot_instance_id"] and leg.tags["config_version"]


# --------------------------------------------------------------------------- #
# Order builder constraints                                                    #
# --------------------------------------------------------------------------- #
def test_builder_respects_tick_lot_min_notional() -> None:
    builder = OrderBuilder(load_execution_config(), OwnershipPolicy(_settings()))
    for sym in _meta().symbols():
        spec = _meta().spec(sym)
        assert spec is not None
        ref = {"BTC/USDT:USDT": 50000.0, "ETH/USDT:USDT": 3000.0, "SOL/USDT:USDT": 150.0}[sym]
        cand = _cand(sym, entry_price=ref)
        res = builder.build(cand, _rm().evaluate(cand, _flat()), spec)
        assert res.ok and res.plan is not None
        tick = float(spec.fields["tick_size"])
        step = float(spec.fields["qty_step"])
        price = res.plan.entry.price if res.plan.entry.price is not None else ref
        assert abs((price / tick) - round(price / tick)) < 1e-6
        assert abs((res.plan.qty / step) - round(res.plan.qty / step)) < 1e-6
        assert res.plan.qty * price >= float(spec.fields["min_notional"])


def test_builder_rejects_unapproved_decision() -> None:
    builder = OrderBuilder(load_execution_config(), OwnershipPolicy(_settings()))
    small = AccountState(
        portfolio=PortfolioState(equity=100.0),
        breakers=BreakerInputs(equity=100.0, peak_equity=100.0, daily_pnl=0.0),
    )
    cand = _cand("SOL/USDT:USDT", stop_frac=0.05, entry_price=150.0)
    res = builder.build(cand, _rm().evaluate(cand, small), _meta().spec("SOL/USDT:USDT"))
    assert not res.ok


# --------------------------------------------------------------------------- #
# Atomic bracket / trailing                                                    #
# --------------------------------------------------------------------------- #
def test_atomic_exchange_side_sl_tp() -> None:
    s = _settings()
    venue = SimulatedVenue(_meta())
    cand = _cand(tp_frac=0.02)  # finite TP
    res = _engine(s, venue).execute(
        cand, _rm().evaluate(cand, _flat()), realized_slippage_frac=0.0005
    )
    assert res.placed and res.position is not None
    assert res.position.stop_order_id and res.position.tp_order_id
    assert venue.order_status(res.position.stop_order_id) == "open"
    assert venue.order_status(res.position.tp_order_id) == "open"
    assert res.position.has_exchange_side_stop()


def test_momentum_uses_native_trailing_not_fixed_tp() -> None:
    s = _settings()
    venue = SimulatedVenue(_meta())
    cand = _cand(tp_frac=1.0)  # >= NO_FIXED_TP_FRAC ⇒ no fixed TP
    res = _engine(s, venue).execute(
        cand, _rm().evaluate(cand, _flat()), realized_slippage_frac=0.0005
    )
    assert res.placed and res.position is not None
    assert res.position.trail_order_id is not None
    assert res.position.tp_order_id is None
    trail = venue.open_orders[res.position.trail_order_id]
    assert trail.order_type is OrderType.TRAILING_STOP and trail.reduce_only


# --------------------------------------------------------------------------- #
# Cancel / replace / partial fill                                             #
# --------------------------------------------------------------------------- #
def test_cancel_and_cancel_replace() -> None:
    s = _settings()
    own = OwnershipPolicy(s)
    venue = SimulatedVenue(_meta())
    cand = _cand(tp_frac=0.02)
    res = ExecutionEngine(load_execution_config(), _meta(), own, venue).execute(
        cand, _rm().evaluate(cand, _flat()), realized_slippage_frac=0.0005
    )
    stop_id = res.position.stop_order_id
    new_stop = Order(
        client_id=own.new_client_id("stop"),
        symbol=BTC,
        side=SELL,
        qty=res.plan.qty,
        order_type=OrderType.STOP_MARKET,
        role="stop",
        stop_price=49000.0,
        reduce_only=True,
        tags=own.tags(),
    )
    assert venue.cancel_replace(stop_id, new_stop) == new_stop.client_id
    assert venue.order_status(stop_id) == "cancelled"
    assert venue.order_status(new_stop.client_id) == "open"
    assert venue.cancel(new_stop.client_id) is True


def test_partial_fill_tracks_remaining() -> None:
    s = _settings()
    venue = SimulatedVenue(_meta())
    cand = _cand(tp_frac=0.02)
    dec = _rm().evaluate(cand, _flat())
    res = _engine(s, venue).execute(cand, dec, realized_slippage_frac=0.0005, fill_ratio=0.5)
    assert res.placed and not res.fully_filled
    assert abs(res.position.qty - dec.qty * 0.5) < 1e-9
    assert res.remaining_qty > 0


# --------------------------------------------------------------------------- #
# Slippage measurement                                                         #
# --------------------------------------------------------------------------- #
def test_momentum_carries_both_reachable_tp_and_trailing() -> None:
    """Parity: a momentum candidate with a REACHABLE R-target TP plus its own trail_frac arms BOTH
    a take-profit AND a trailing stop on the bracket (Bybit holds SL+TP+trail at once), so live
    reproduces the backtest's stop/TP/trail OR-of-exits — not the legacy trail-only path."""
    own = OwnershipPolicy(_settings())
    builder = OrderBuilder(load_execution_config(), own)
    cand = _cand(tp_frac=0.02, trail_frac=0.03)  # reachable TP + a 3% trailing offset
    res = builder.build(cand, _rm().evaluate(cand, _flat()), _meta().spec(BTC))
    assert res.ok and res.plan is not None
    assert res.plan.take_profit is not None  # reachable TP attached
    assert res.plan.trailing is not None  # trailing armed from candidate.trail_frac
    assert res.plan.trailing.trail_offset == pytest.approx(0.03)  # strategy offset, ≥ the stop


def test_maker_candidate_posts_passive_limit_entry() -> None:
    """Parity: a maker candidate posts a POST_ONLY entry limit_offset_frac INSIDE the reference
    (buy below), not a market order — matching the backtest's passive maker fill."""
    own = OwnershipPolicy(_settings())
    builder = OrderBuilder(load_execution_config(), own)
    cand = _cand(maker=True, limit_offset_frac=0.001)  # long: post 0.1% below the reference
    res = builder.build(cand, _rm().evaluate(cand, _flat()), _meta().spec(BTC))
    assert res.ok and res.plan is not None
    assert res.plan.entry.order_type is OrderType.POST_ONLY
    assert res.plan.entry.price is not None and res.plan.entry.price < 50_000.0  # inside the ref


def test_taker_slippage_measured_maker_zero() -> None:
    s = _settings()
    venue = SimulatedVenue(_meta())
    cand = _cand(tp_frac=0.02)
    taker = _engine(s, venue).execute(
        cand, _rm().evaluate(cand, _flat()), realized_slippage_frac=0.001, entry_style="taker"
    )
    assert taker.fill.slippage_frac > 0
    assert taker.fill.actual_price != taker.fill.expected_price
    assert not taker.fill.maker

    venue2 = SimulatedVenue(_meta())
    maker = _engine(s, venue2).execute(
        cand, _rm().evaluate(cand, _flat()), realized_slippage_frac=0.001, entry_style="maker_first"
    )
    assert maker.fill.maker and maker.fill.slippage_frac == 0.0


# --------------------------------------------------------------------------- #
# Revalidate-before-execute                                                    #
# --------------------------------------------------------------------------- #
def test_revalidate_blocks_toxic_spread() -> None:
    s = _settings()
    venue = SimulatedVenue(_meta())
    cand = _cand(spread_bps=80.0)
    res = _engine(s, venue).execute(
        cand, _rm().evaluate(cand, _flat()), realized_slippage_frac=0.0005
    )
    assert not res.placed and "toxic_spread" in res.reason
    assert not venue.positions  # nothing opened


def test_execute_aborts_when_kill_switch_engaged() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        s = _settings(tmp)
        ks = KillSwitch(s)
        venue = SimulatedVenue(_meta())
        engine = ExecutionEngine(
            load_execution_config(), _meta(), OwnershipPolicy(s), venue, kill_switch=ks
        )
        ks.engage(reason="test", actor="test")
        cand = _cand()
        res = engine.execute(cand, _rm().evaluate(cand, _flat()), realized_slippage_frac=0.0005)
        assert not res.placed and res.reason == "kill_switch_engaged"
        ks.disengage()


# --------------------------------------------------------------------------- #
# Reconciliation + ownership                                                   #
# --------------------------------------------------------------------------- #
def test_reconciliation_clean_book_ok() -> None:
    s = _settings()
    venue = SimulatedVenue(_meta())
    cand = _cand(tp_frac=0.02)
    _engine(s, venue).execute(cand, _rm().evaluate(cand, _flat()), realized_slippage_frac=0.0005)
    rr = Reconciler(OwnershipPolicy(s)).reconcile(
        venue.open_orders,
        venue.positions,
        known_order_ids=set(venue.open_orders),
        known_position_symbols=set(venue.positions),
    )
    assert rr.ok and not rr.halt_required


def test_reconciliation_detects_foreign_order_and_position() -> None:
    s = _settings()
    own = OwnershipPolicy(s)
    venue = SimulatedVenue(_meta())
    venue.inject_foreign_order(
        Order(
            client_id="MANUAL_x_1",
            symbol=BTC,
            side=BUY,
            qty=1.0,
            order_type=OrderType.LIMIT,
            price=49000.0,
            tags={},
        )
    )
    venue.inject_foreign_position("XRP/USDT:USDT", side=1, qty=5.0, price=0.5)
    rr = Reconciler(own).reconcile(
        venue.open_orders, venue.positions, known_order_ids=set(), known_position_symbols=set()
    )
    assert rr.halt_required
    assert "MANUAL_x_1" in rr.unknown_orders
    assert "XRP/USDT:USDT" in rr.unknown_positions


def test_cancel_owned_only_refuses_foreign() -> None:
    venue = SimulatedVenue(_meta())
    venue.inject_foreign_order(
        Order(
            client_id="MANUAL_x_1",
            symbol=BTC,
            side=BUY,
            qty=1.0,
            order_type=OrderType.LIMIT,
            price=49000.0,
            tags={},
        )
    )
    assert venue.cancel("MANUAL_x_1", owned_only=True) is False
    assert venue.order_status("MANUAL_x_1") == "open"


def test_emergency_close_requires_confirmation() -> None:
    venue = SimulatedVenue(_meta())
    venue.inject_foreign_position("DOGE/USDT:USDT", side=1, qty=1.0, price=0.1)
    try:
        venue.emergency_close_all(confirm=False)
        raised = False
    except PermissionError:
        raised = True
    assert raised
    assert venue.emergency_close_all(confirm=True) >= 1
    assert not venue.positions
