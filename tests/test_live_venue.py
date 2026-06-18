"""M6: ccxt-backed live/testnet execution venue (mocked client — no network/keys).

Proves the live venue honours the Section 2.2 atomic-bracket invariant (entry carries
an exchange-resident stop), the Section 7 ownership prefix, and the safety gating:
no anonymous trading, and no real-money (mainnet) order without an activation guard.
"""

from __future__ import annotations

import pytest
from src.config import Settings
from src.exchange.metadata import load_metadata_config
from src.execution.live_venue import CcxtLiveVenue, get_venue
from src.execution.order import Order, OrderPlan, OrderType
from src.execution.venue import SimulatedVenue, Venue

_PREFIX = "QBOT_TEST_v1_"
_TAGS = {"bot_instance_id": "bot1"}


def _testnet_settings(**over) -> Settings:
    base = {
        "_env_file": None,
        "exchange_env": "testnet",
        "exchange_api_key": "k",
        "exchange_api_secret": "s",
        "order_client_id_prefix": _PREFIX,
    }
    base.update(over)
    return Settings(**base)


def _plan() -> OrderPlan:
    entry = Order(
        client_id=f"{_PREFIX}entry_1",
        symbol="BTC/USDT:USDT",
        side="buy",
        qty=0.01,
        order_type=OrderType.MARKET,
        role="entry",
        tags=_TAGS,
    )
    stop = Order(
        client_id=f"{_PREFIX}stop_1",
        symbol="BTC/USDT:USDT",
        side="sell",
        qty=0.01,
        order_type=OrderType.STOP_MARKET,
        role="stop",
        stop_price=49_000.0,
        reduce_only=True,
        tags=_TAGS,
    )
    tp = Order(
        client_id=f"{_PREFIX}tp_1",
        symbol="BTC/USDT:USDT",
        side="sell",
        qty=0.01,
        order_type=OrderType.TAKE_PROFIT_MARKET,
        role="take_profit",
        stop_price=52_000.0,
        reduce_only=True,
        tags=_TAGS,
    )
    return OrderPlan(
        symbol="BTC/USDT:USDT", side=1, qty=0.01, entry=entry, stop=stop, take_profit=tp
    )


class FakeCcxt:
    def __init__(self) -> None:
        self.orders: list[dict] = []
        self.cancelled: list[str] = []
        self._positions: list[dict] = []

    def create_order(self, symbol, type, side, qty, price, params=None):  # noqa: A002
        self.orders.append(
            {
                "symbol": symbol,
                "type": type,
                "side": side,
                "qty": qty,
                "price": price,
                "params": params or {},
            }
        )
        return {"average": 50_000.0, "filled": qty, "fee": {"cost": 0.5}}

    def cancel_order(self, oid, symbol, params=None):
        self.cancelled.append(oid)
        return {}

    def fetch_positions(self):
        return self._positions


class AllowGuard:
    def allow_live_order(self, plan):
        return True, "ok"


class DenyGuard:
    def allow_live_order(self, plan):
        return False, "gates not green"


def _venue(client, **over) -> CcxtLiveVenue:
    return CcxtLiveVenue(load_metadata_config(), _testnet_settings(**over), client=client)


def test_place_bracket_attaches_atomic_sl_tp_and_ownership() -> None:
    fake = FakeCcxt()
    res = _venue(fake).place_bracket(
        _plan(), ref_price=50_000.0, realized_slippage_frac=0.001, latency_ms=5.0
    )
    params = fake.orders[0]["params"]
    assert "stopLoss" in params and params["stopLoss"]["triggerPrice"] == 49_000.0
    assert "takeProfit" in params and params["takeProfit"]["triggerPrice"] == 52_000.0
    assert params["clientOrderId"].startswith(_PREFIX)  # Section 7 ownership prefix
    assert res.position.has_exchange_side_stop()  # Section 2.2 invariant
    assert res.fully_filled and res.position.qty == 0.01


def test_requires_credentials_without_injected_client() -> None:
    settings = _testnet_settings(exchange_api_key="", exchange_api_secret="")
    with pytest.raises(ValueError, match="requires EXCHANGE_API"):
        CcxtLiveVenue(load_metadata_config(), settings)


def test_live_mainnet_refuses_without_guard() -> None:
    fake = FakeCcxt()
    venue = CcxtLiveVenue(
        load_metadata_config(), _testnet_settings(exchange_env="live"), client=fake
    )
    assert venue.is_live
    with pytest.raises(PermissionError, match="activation guard"):
        venue.place_bracket(_plan(), ref_price=50_000.0, realized_slippage_frac=0.0, latency_ms=5.0)
    assert not fake.orders  # nothing was sent to the exchange


def test_live_mainnet_guard_allows_and_denies() -> None:
    meta = load_metadata_config()
    s = _testnet_settings(exchange_env="live")
    allowed = CcxtLiveVenue(meta, s, client=FakeCcxt(), guard=AllowGuard())
    res = allowed.place_bracket(
        _plan(), ref_price=50_000.0, realized_slippage_frac=0.0, latency_ms=5.0
    )
    assert res.fully_filled

    fake = FakeCcxt()
    denied = CcxtLiveVenue(meta, s, client=fake, guard=DenyGuard())
    with pytest.raises(PermissionError, match="gates not green"):
        denied.place_bracket(
            _plan(), ref_price=50_000.0, realized_slippage_frac=0.0, latency_ms=5.0
        )
    assert not fake.orders


def test_get_venue_defaults_to_simulated_and_opts_into_live() -> None:
    meta = load_metadata_config()
    assert isinstance(get_venue(meta, _testnet_settings(), live=False), SimulatedVenue)
    live = get_venue(meta, _testnet_settings(), live=True, client=FakeCcxt())
    assert isinstance(live, CcxtLiveVenue)
    assert isinstance(live, Venue)  # satisfies the runtime-checkable Protocol


def test_fetch_exchange_positions_marks_ownership() -> None:
    fake = FakeCcxt()
    fake._positions = [
        {
            "symbol": "BTC/USDT:USDT",
            "side": "long",
            "contracts": 0.01,
            "entryPrice": 50_000.0,
            "info": {"clientOrderId": f"{_PREFIX}entry_1"},
        },
        {
            "symbol": "ETH/USDT:USDT",
            "side": "short",
            "contracts": 0.1,
            "entryPrice": 3_000.0,
            "info": {"clientOrderId": "MANUAL_999"},
        },
        {
            "symbol": "SOL/USDT:USDT",
            "side": "long",
            "contracts": 0.0,
            "entryPrice": 0.0,
            "info": {},
        },
    ]
    pos = _venue(fake).fetch_exchange_positions()
    assert pos["BTC/USDT:USDT"].owned is True
    assert pos["ETH/USDT:USDT"].owned is False  # foreign / manual order → not owned
    assert "SOL/USDT:USDT" not in pos  # zero-qty positions ignored


def test_emergency_close_requires_confirmation() -> None:
    venue = _venue(FakeCcxt())
    with pytest.raises(PermissionError, match="confirmation"):
        venue.emergency_close_all(confirm=False)
