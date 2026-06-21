"""M8: the live-activation guard — the chokepoint before any real-money order.

Unit-tests the four-gate live-safety check + bounded caps with injected gate/approval
checks (no live deployment needed), then proves the real guard plugs into the live venue.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from src.config import Settings
from src.exchange.metadata import load_metadata_config
from src.execution.live_venue import CcxtLiveVenue
from src.execution.order import Order, OrderPlan, OrderType
from src.live.guard import LiveActivationGuard, LiveLimits

_PREFIX = "QBOT_TEST_v1_"
_TAGS = {"bot_instance_id": "bot1"}


def _plan(qty: float = 0.01, entry_price: float = 50_000.0) -> OrderPlan:
    entry = Order(
        client_id=f"{_PREFIX}entry_1",
        symbol="BTC/USDT:USDT",
        side="buy",
        qty=qty,
        order_type=OrderType.LIMIT,
        role="entry",
        price=entry_price,
        tags=_TAGS,
    )
    stop = Order(
        client_id=f"{_PREFIX}stop_1",
        symbol="BTC/USDT:USDT",
        side="sell",
        qty=qty,
        order_type=OrderType.STOP_MARKET,
        role="stop",
        stop_price=entry_price * 0.98,
        reduce_only=True,
        tags=_TAGS,
    )
    return OrderPlan(symbol="BTC/USDT:USDT", side=1, qty=qty, entry=entry, stop=stop)


_LIMITS = LiveLimits(
    max_orders_per_session=2,
    max_open_positions=2,
    max_order_notional_pct=0.05,
    account_equity=10_000.0,  # cap = 500 notional/order
)


def _guard(*, allowed=True, gates=True, approved=True, limits=_LIMITS) -> LiveActivationGuard:
    return LiveActivationGuard(
        SimpleNamespace(live_trading_allowed=allowed),
        limits=limits,
        gates_pass=lambda: gates,
        approved=lambda: approved,
    )


def test_allows_when_all_preconditions_hold_and_within_caps() -> None:
    ok, reason = _guard().allow_live_order(_plan(qty=0.01))  # notional 500 == cap
    assert ok and reason == "ok"


def test_denies_when_live_not_enabled() -> None:
    ok, reason = _guard(allowed=False).allow_live_order(_plan())
    assert not ok and "live trading not enabled" in reason


def test_denies_when_gates_not_green() -> None:
    ok, reason = _guard(gates=False).allow_live_order(_plan())
    assert not ok and "blocks_live gates" in reason


def test_denies_without_sign_off() -> None:
    ok, reason = _guard(approved=False).allow_live_order(_plan())
    assert not ok and "sign-off" in reason


def test_denies_when_order_notional_exceeds_cap() -> None:
    ok, reason = _guard().allow_live_order(_plan(qty=0.02))  # 0.02*50_000 = 1000 > 500
    assert not ok and "notional" in reason


def test_enforces_max_orders_per_session() -> None:
    g = _guard()
    assert g.allow_live_order(_plan())[0]
    assert g.allow_live_order(_plan())[0]
    ok, reason = g.allow_live_order(_plan())  # third exceeds max_orders_per_session=2
    assert not ok and "max_orders_per_session" in reason


def test_enforces_max_open_positions_and_register_close() -> None:
    limits = LiveLimits(max_orders_per_session=5, max_open_positions=1, account_equity=10_000.0)
    g = _guard(limits=limits)
    assert g.allow_live_order(_plan())[0]
    ok, reason = g.allow_live_order(_plan())  # second concurrent position blocked
    assert not ok and "max_open_positions" in reason
    g.register_close()  # a position closed frees a slot
    assert g.allow_live_order(_plan())[0]


def test_paper_settings_short_circuit_without_db() -> None:
    # A real paper Settings has live_trading_allowed False → denied before any DB call.
    guard = LiveActivationGuard(
        Settings(_env_file=None),
        limits=_LIMITS,
        gates_pass=lambda: (_ for _ in ()).throw(AssertionError("must not be consulted")),
        approved=lambda: (_ for _ in ()).throw(AssertionError("must not be consulted")),
    )
    ok, reason = guard.allow_live_order(_plan())
    assert not ok and "live trading not enabled" in reason


def test_real_guard_plugs_into_live_venue() -> None:
    fake = SimpleNamespace(
        orders=[],
        create_order=lambda *a, **k: ({"average": 50_000.0, "filled": a[3], "fee": {"cost": 0.0}}),
    )
    # Record orders by wrapping create_order.
    calls: list = []

    def _create(symbol, type, side, qty, price, params=None):  # noqa: A002
        calls.append(params or {})
        return {"average": price or 50_000.0, "filled": qty, "fee": {"cost": 0.0}}

    fake.create_order = _create
    settings = Settings(
        _env_file=None,
        exchange_env="live",
        exchange_id="skeleton",  # matches the skeleton metadata so the venue guard passes
        exchange_api_key="k",
        exchange_api_secret="s",
        order_client_id_prefix=_PREFIX,
    )
    meta = load_metadata_config()

    venue_ok = CcxtLiveVenue(meta, settings, client=fake, guard=_guard())
    venue_ok.place_bracket(_plan(), ref_price=50_000.0, realized_slippage_frac=0.0, latency_ms=5.0)
    assert calls and calls[0]["clientOrderId"].startswith(_PREFIX)

    venue_blocked = CcxtLiveVenue(meta, settings, client=fake, guard=_guard(gates=False))
    with pytest.raises(PermissionError, match="blocks_live gates"):
        venue_blocked.place_bracket(
            _plan(), ref_price=50_000.0, realized_slippage_frac=0.0, latency_ms=5.0
        )
