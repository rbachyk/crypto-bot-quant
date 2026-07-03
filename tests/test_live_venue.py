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
        # The offline test venue is the 'skeleton' exchange, matching the skeleton metadata
        # the tests load — so the venue's metadata guard (Section 6) sees a verified, matched
        # spec. A real Bybit run would use exchange_id='bybit' + verified Bybit metadata.
        "exchange_id": "skeleton",
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
        self._open_orders: list[dict] = []

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

    def fetch_open_orders(self):
        return self._open_orders


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


def _builder_plan(*, tp_frac: float, settings: Settings):
    """Build a real bracket plan through OrderBuilder (not hand-rolled) so the venue
    integration is exercised end-to-end — this is what would have caught the TP-drop bug."""
    from src.execution import OrderBuilder, OwnershipPolicy, load_execution_config
    from src.ranking import Candidate
    from src.risk import (
        AccountState,
        BreakerInputs,
        PortfolioState,
        RiskManager,
        load_risk_config,
    )

    meta = load_metadata_config()
    cand = Candidate(
        symbol="BTC/USDT:USDT",
        strategy="t",
        strategy_version="t",
        side=1,
        entry_price=50_000.0,
        stop_frac=0.01,
        tp_frac=tp_frac,
        regime="low_vol_up",
        session=2,
        spread_bps=3.0,
        slippage_est=0.0005,
        latency_ms=40.0,
    )
    acct = AccountState(
        portfolio=PortfolioState(equity=100_000.0),
        breakers=BreakerInputs(equity=100_000.0, peak_equity=100_000.0, daily_pnl=0.0),
    )
    decision = RiskManager(load_risk_config(), meta).evaluate(cand, acct)
    res = OrderBuilder(load_execution_config(), OwnershipPolicy(settings)).build(
        cand, decision, meta.spec("BTC/USDT:USDT")
    )
    assert res.ok and res.plan is not None
    return res.plan


def test_orderbuilder_bracket_attaches_both_sl_and_tp() -> None:
    """Integration: OrderBuilder → CcxtLiveVenue.place_bracket attaches BOTH stopLoss and
    takeProfit. Regression for the bug where the TP leg carried only ``price`` (not the
    trigger the venue reads), so ``takeProfit`` was silently dropped from every real order."""
    settings = _testnet_settings()
    plan = _builder_plan(tp_frac=0.02, settings=settings)  # finite TP ⇒ TAKE_PROFIT leg
    assert plan.take_profit is not None
    fake = FakeCcxt()
    CcxtLiveVenue(load_metadata_config(), settings, client=fake).place_bracket(
        plan, ref_price=50_000.0, realized_slippage_frac=0.0005, latency_ms=5.0
    )
    params = fake.orders[0]["params"]
    assert "stopLoss" in params and params["stopLoss"]["triggerPrice"] > 0
    assert "takeProfit" in params and params["takeProfit"]["triggerPrice"] > 0
    # TP target is above entry for a long (sanity on the trigger we forwarded).
    assert params["takeProfit"]["triggerPrice"] > params["stopLoss"]["triggerPrice"]


def test_orderbuilder_momentum_attaches_sl_and_trailing() -> None:
    """A no-fixed-TP (momentum) plan still attaches the initial stopLoss plus a trailing
    stop — the position is never opened without exchange-side protection (Section 2.2)."""
    settings = _testnet_settings()
    plan = _builder_plan(tp_frac=1.0, settings=settings)  # >= NO_FIXED_TP_FRAC ⇒ trailing
    assert plan.take_profit is None and plan.trailing is not None
    fake = FakeCcxt()
    CcxtLiveVenue(load_metadata_config(), settings, client=fake).place_bracket(
        plan, ref_price=50_000.0, realized_slippage_frac=0.0005, latency_ms=5.0
    )
    params = fake.orders[0]["params"]
    assert "stopLoss" in params and params["stopLoss"]["triggerPrice"] > 0
    assert params.get("trailingPercent", 0) > 0


def test_order_blocked_when_metadata_is_for_wrong_exchange() -> None:
    """A venue trading 'bybit' with metadata verified only for the offline 'skeleton' venue
    must refuse to place orders — never size/route on a placeholder spec (Section 6)."""
    fake = FakeCcxt()
    # Skeleton metadata (exchange_id='skeleton') but the venue trades exchange_id='bybit'.
    venue = CcxtLiveVenue(
        load_metadata_config(), _testnet_settings(exchange_id="bybit"), client=fake
    )
    with pytest.raises(PermissionError, match="unverified exchange metadata"):
        venue.place_bracket(
            _plan(), ref_price=50_000.0, realized_slippage_frac=0.0, latency_ms=5.0
        )
    assert not fake.orders  # nothing sent to the exchange


def test_order_blocked_when_metadata_unverified() -> None:
    """Metadata marked ``verified: false`` (operator review pending) blocks order placement
    even when the exchange matches — the Bybit demo/testnet metadata ships unverified."""
    from src.exchange.metadata import load_metadata_for

    bybit_meta = load_metadata_for("bybit")
    assert bybit_meta.exchange_id == "bybit" and not bybit_meta.verified
    fake = FakeCcxt()
    venue = CcxtLiveVenue(bybit_meta, _testnet_settings(exchange_id="bybit"), client=fake)
    with pytest.raises(PermissionError, match="UNVERIFIED|unverified"):
        venue.place_bracket(
            _plan(), ref_price=50_000.0, realized_slippage_frac=0.0, latency_ms=5.0
        )
    assert not fake.orders


def test_place_bracket_keeps_marker_when_stop_not_yet_visible() -> None:
    """A market fill's attached stop commonly hasn't propagated to the position read yet, so the
    place-time check is ADVISORY: it keeps the optimistic stop marker (no false downgrade / false
    CRITICAL) and the per-tick reconciliation makes the authoritative unprotected call later."""
    fake = FakeCcxt()
    # The immediate post-placement read shows the position but no stop yet (propagation lag).
    fake._positions = [
        {
            "symbol": "BTC/USDT:USDT", "side": "long", "contracts": 0.01, "entryPrice": 50_000.0,
            "info": {"clientOrderId": f"{_PREFIX}entry_1", "stopLoss": "0"},
        }
    ]
    res = _venue(fake).place_bracket(
        _plan(), ref_price=50_000.0, realized_slippage_frac=0.001, latency_ms=5.0
    )
    assert res.position.has_exchange_side_stop() is True  # optimistic marker kept (not downgraded)


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


def test_fetch_open_orders_marks_ownership() -> None:
    fake = FakeCcxt()
    fake._open_orders = [
        {
            "symbol": "BTC/USDT:USDT",
            "side": "buy",
            "amount": 0.01,
            "price": 49_000.0,
            "clientOrderId": f"{_PREFIX}entry_1",
            "info": {},
        },
        {
            "symbol": "ETH/USDT:USDT",
            "side": "sell",
            "amount": 0.1,
            "price": 3_100.0,
            "clientOrderId": "MANUAL_human_42",
            "info": {},
        },
    ]
    orders = _venue(fake).fetch_open_orders()
    assert orders[f"{_PREFIX}entry_1"].tags.get("bot_instance_id")  # owned → tagged
    assert not orders["MANUAL_human_42"].tags  # foreign → no ownership tag


def test_startup_reconciliation_detects_foreign_and_adopts_owned() -> None:
    from src.execution.ownership import OwnershipPolicy
    from src.execution.reconciliation import reconcile_startup

    fake = FakeCcxt()
    fake._open_orders = [
        {
            "symbol": "BTC/USDT:USDT",
            "side": "buy",
            "amount": 0.01,
            "price": 49_000.0,
            "clientOrderId": "MANUAL_human_42",
            "info": {},
        }
    ]
    fake._positions = [
        {
            "symbol": "ETH/USDT:USDT",
            "side": "long",
            "contracts": 0.1,
            "entryPrice": 3_000.0,
            "info": {"clientOrderId": f"{_PREFIX}entry_1"},
        }
    ]
    settings = _testnet_settings()
    venue = CcxtLiveVenue(load_metadata_config(), settings, client=fake)
    res = reconcile_startup(venue, OwnershipPolicy(settings), environment="testnet")
    assert res.halt_required  # foreign order present
    assert "MANUAL_human_42" in res.foreign_orders
    assert "ETH/USDT:USDT" in res.owned_positions  # our position adopted
    assert "ETH/USDT:USDT" in venue.positions
    assert "HALT" in res.report()


def test_fetch_positions_reads_real_exchange_side_stop() -> None:
    """A reconciled position reflects the REAL exchange-side stop/TP, so an owned position with
    no stop is correctly reported unprotected (not trusting a self-minted marker)."""
    fake = FakeCcxt()
    fake._positions = [
        {  # protected: carries a stopLoss in info
            "symbol": "BTC/USDT:USDT", "side": "long", "contracts": 0.01, "entryPrice": 50_000.0,
            "stopLossPrice": 49_000.0,
            "info": {"clientOrderId": f"{_PREFIX}e1", "stopLoss": "49000"},
        },
        {  # UNPROTECTED: no stop on the exchange
            "symbol": "ETH/USDT:USDT", "side": "long", "contracts": 0.1, "entryPrice": 3_000.0,
            "info": {"clientOrderId": f"{_PREFIX}e2", "stopLoss": "0"},
        },
    ]
    pos = _venue(fake).fetch_exchange_positions()
    assert pos["BTC/USDT:USDT"].has_exchange_side_stop() is True
    assert pos["ETH/USDT:USDT"].has_exchange_side_stop() is False


def test_emergency_close_requires_confirmation() -> None:
    venue = _venue(FakeCcxt())
    with pytest.raises(PermissionError, match="confirmation"):
        venue.emergency_close_all(confirm=False)


class _EnvClient:
    """Records which Bybit-environment switch a ccxt client received."""

    def __init__(self) -> None:
        self.sandbox = None
        self.demo = None

    def set_sandbox_mode(self, on):
        self.sandbox = on

    def enable_demo_trading(self, on):
        self.demo = on


def test_place_order_refuses_a_bare_entry() -> None:
    """An entry must go through place_bracket (stop attached atomically); place_order — used for
    cancel/replace of protective legs — refuses an entry so it can't fill unprotected."""
    from src.execution.order import Order, OrderType

    venue = _venue(FakeCcxt())
    entry = Order(
        client_id=f"{_PREFIX}entry_9", symbol="BTC/USDT:USDT", side="buy", qty=0.01,
        order_type=OrderType.MARKET, role="entry", tags=_TAGS,
    )
    with pytest.raises(ValueError, match="cannot place an entry"):
        venue.place_order(entry)


def test_apply_exchange_env_raises_when_testnet_sandbox_unavailable() -> None:
    """A ccxt build without sandbox support must RAISE for testnet, never silently fall through
    to the live mainnet endpoints with testnet keys."""
    from types import SimpleNamespace

    from src.execution.live_venue import apply_exchange_env

    no_sandbox = SimpleNamespace()  # no set_sandbox_mode / enable_demo_trading
    with pytest.raises(ValueError, match="no sandbox support"):
        apply_exchange_env(no_sandbox, "testnet")


def test_apply_exchange_env_routes_to_the_right_environment() -> None:
    from src.execution.live_venue import apply_exchange_env

    testnet = _EnvClient()
    apply_exchange_env(testnet, "testnet")
    assert testnet.sandbox is True and testnet.demo is None  # testnet.bybit.com

    demo = _EnvClient()
    apply_exchange_env(demo, "demo")
    assert demo.demo is True and demo.sandbox is None  # api-demo.bybit.com (NOT testnet)

    live = _EnvClient()
    apply_exchange_env(live, "live")
    assert live.sandbox is None and live.demo is None  # mainnet, no switch


def test_demo_env_is_not_treated_as_live() -> None:
    venue = CcxtLiveVenue(
        load_metadata_config(), _testnet_settings(exchange_env="demo"), client=FakeCcxt()
    )
    assert venue.is_live is False  # demo uses virtual funds → no activation guard required


def test_invalid_exchange_env_is_rejected() -> None:
    with pytest.raises(ValueError, match="EXCHANGE_ENV"):
        Settings(_env_file=None, exchange_env="sandbox")


# --------------------------------------------------------------------------- #
# C5: entry fills are OBSERVED via order-status polling, never assumed         #
# --------------------------------------------------------------------------- #
class PollingFakeCcxt(FakeCcxt):
    """Fake whose create response carries NO fill info (like Bybit's create response for
    a resting order) — the fill is only observable via fetch_order. cancel_order moves
    the order to 'canceled' unless ``cancel_raises`` is set."""

    def __init__(
        self,
        *,
        status: str = "open",
        filled: float = 0.0,
        average: float | None = None,
        cancel_raises: bool = False,
    ) -> None:
        super().__init__()
        self.status = status
        self.filled = filled
        self.average = average
        self.cancel_raises = cancel_raises
        self.fetch_calls: list = []

    def create_order(self, symbol, type, side, qty, price, params=None):  # noqa: A002
        super().create_order(symbol, type, side, qty, price, params)
        return {"id": "oid-1"}  # Bybit create response: no filled/average/status

    def fetch_order(self, oid, symbol, params=None):
        self.fetch_calls.append((oid, symbol, dict(params or {})))
        return {
            "id": oid,
            "status": self.status,
            "filled": self.filled,
            "average": self.average,
            "fee": {"cost": 0.1},
        }

    def cancel_order(self, oid, symbol, params=None):
        if self.cancel_raises:
            raise RuntimeError("cancel rejected")
        self.cancelled.append(oid)
        self.status = "canceled"
        return {}


def _polling_venue(fake) -> CcxtLiveVenue:
    return CcxtLiveVenue(
        load_metadata_config(),
        _testnet_settings(),
        client=fake,
        fill_timeout_s=0.05,
        fill_poll_interval_s=0.01,
    )


def test_fill_observed_via_fetch_order_when_create_response_omits_filled() -> None:
    """The create response omits ``filled`` (Bybit does); the venue must POLL the order
    and book the OBSERVED fill qty + average price — not assume a full fill at limit."""
    fake = PollingFakeCcxt(status="closed", filled=0.01, average=50_100.0)
    res = _polling_venue(fake).place_bracket(
        _plan(), ref_price=50_000.0, realized_slippage_frac=0.001, latency_ms=5.0
    )
    assert fake.fetch_calls  # the fill was observed, not assumed
    assert res.fully_filled and res.fill.qty == 0.01
    assert res.position.qty == 0.01 and res.position.entry_price == 50_100.0
    assert res.position.has_exchange_side_stop()
    assert not fake.cancelled  # nothing to cancel — it filled


def test_never_fills_cancels_remainder_and_books_nothing() -> None:
    """Unfilled at the window end: the remainder is cancelled (and VERIFIED) and the
    result is a clean no-position non-fill — no phantom VenuePosition, no trade."""
    fake = PollingFakeCcxt(status="open", filled=0.0)
    venue = _polling_venue(fake)
    res = venue.place_bracket(
        _plan(), ref_price=50_000.0, realized_slippage_frac=0.001, latency_ms=5.0
    )
    assert fake.cancelled == ["oid-1"]  # remainder cancelled on the exchange
    assert res.fill.qty == 0.0 and res.position.qty == 0.0
    assert not res.position.has_exchange_side_stop()  # no protection markers on a non-fill
    assert not res.fully_filled and res.remaining_qty == pytest.approx(0.01)
    assert not venue.positions  # no phantom position in the mirror
    assert not venue.open_orders  # cancel confirmed → nothing left tracked as resting


def test_partial_fill_books_only_the_filled_part() -> None:
    """Partially filled at the window end: cancel the remainder, book the filled qty at
    the observed average, and size the position (whose attached SL/TP cover it) to it."""
    fake = PollingFakeCcxt(status="open", filled=0.004, average=50_050.0)
    venue = _polling_venue(fake)
    res = venue.place_bracket(
        _plan(), ref_price=50_000.0, realized_slippage_frac=0.001, latency_ms=5.0
    )
    assert fake.cancelled == ["oid-1"]
    assert res.fill.qty == pytest.approx(0.004)
    assert res.position.qty == pytest.approx(0.004)
    assert res.position.entry_price == 50_050.0
    assert res.position.has_exchange_side_stop()  # the filled part IS protected
    assert not res.fully_filled and res.remaining_qty == pytest.approx(0.006)
    assert venue.positions["BTC/USDT:USDT"].qty == pytest.approx(0.004)
    assert not venue.open_orders  # remainder cancelled → not tracked as resting


def test_unconfirmed_cancel_keeps_the_entry_tracked() -> None:
    """If the remainder cancel FAILS and the order is still open, the venue must not
    assume it died: the entry stays tracked as a resting order for reconciliation."""
    fake = PollingFakeCcxt(status="open", filled=0.0, cancel_raises=True)
    venue = _polling_venue(fake)
    res = venue.place_bracket(
        _plan(), ref_price=50_000.0, realized_slippage_frac=0.001, latency_ms=5.0
    )
    assert res.fill.qty == 0.0 and not venue.positions
    assert f"{_PREFIX}entry_1" in venue.open_orders  # still resting on the exchange → tracked
    assert f"{_PREFIX}entry_1" in res.resting_order_ids


def test_create_response_with_positive_full_fill_is_trusted_without_polling() -> None:
    """A create response that positively reports the full fill (the pre-existing fake
    contract) books directly — no polling, no behaviour change for such venues."""
    fake = FakeCcxt()  # returns filled == qty in the create response
    res = _venue(fake).place_bracket(
        _plan(), ref_price=50_000.0, realized_slippage_frac=0.001, latency_ms=5.0
    )
    assert res.fully_filled and res.position.qty == 0.01


# --------------------------------------------------------------------------- #
# C6: place_order sends protective legs as conditional reduce-only orders      #
# --------------------------------------------------------------------------- #
def _protective(order_type: OrderType, *, role: str, stop_price=None, price=None) -> Order:
    return Order(
        client_id=f"{_PREFIX}{role}_9",
        symbol="BTC/USDT:USDT",
        side="sell",
        qty=0.01,
        order_type=order_type,
        role=role,
        price=price,
        stop_price=stop_price,
        reduce_only=True,
        tags=_TAGS,
    )


def test_place_order_stop_market_is_conditional_reduce_only() -> None:
    fake = FakeCcxt()
    _venue(fake).place_order(
        _protective(OrderType.STOP_MARKET, role="stop", stop_price=49_000.0)
    )
    sent = fake.orders[-1]
    assert sent["type"] == "market" and sent["price"] is None
    assert sent["params"]["stopLossPrice"] == 49_000.0  # trigger — NOT an immediate market
    assert sent["params"]["reduceOnly"] is True
    assert sent["params"]["clientOrderId"].startswith(_PREFIX)


def test_place_order_take_profit_market_is_conditional_reduce_only() -> None:
    fake = FakeCcxt()
    _venue(fake).place_order(
        _protective(OrderType.TAKE_PROFIT_MARKET, role="take_profit", stop_price=52_000.0)
    )
    sent = fake.orders[-1]
    assert sent["type"] == "market" and sent["price"] is None
    assert sent["params"]["takeProfitPrice"] == 52_000.0
    assert sent["params"]["reduceOnly"] is True


def test_place_order_stop_limit_carries_trigger_and_limit_price() -> None:
    fake = FakeCcxt()
    _venue(fake).place_order(
        _protective(OrderType.STOP_LIMIT, role="stop", stop_price=49_000.0, price=48_950.0)
    )
    sent = fake.orders[-1]
    assert sent["type"] == "limit" and sent["price"] == 48_950.0
    assert sent["params"]["stopLossPrice"] == 49_000.0
    assert sent["params"]["reduceOnly"] is True


def test_place_order_trailing_stop_raises_instead_of_degrading_to_market() -> None:
    fake = FakeCcxt()
    trail = _protective(OrderType.TRAILING_STOP, role="trailing")
    trail.trail_offset = 0.02
    with pytest.raises(NotImplementedError, match="trailing"):
        _venue(fake).place_order(trail)
    assert not fake.orders  # nothing was sent — no silent market-order degradation


def test_place_order_trigger_leg_without_trigger_price_is_refused() -> None:
    fake = FakeCcxt()
    with pytest.raises(ValueError, match="no.*trigger price"):
        _venue(fake).place_order(_protective(OrderType.STOP_MARKET, role="stop"))
    assert not fake.orders


def test_cancel_replace_of_a_stop_leg_stays_conditional() -> None:
    """cancel_replace (its documented purpose: protective-leg replacement) must place the
    replacement as a conditional reduce-only order, not an instant market order."""
    fake = FakeCcxt()
    venue = _venue(fake)
    old = _protective(OrderType.STOP_MARKET, role="stop", stop_price=49_000.0)
    venue.open_orders[old.client_id] = old
    new = _protective(OrderType.STOP_MARKET, role="stop", stop_price=49_500.0)
    new.client_id = f"{_PREFIX}stop_10"
    assert venue.cancel_replace(old.client_id, new) == new.client_id
    assert old.client_id in fake.cancelled
    sent = fake.orders[-1]
    assert sent["params"]["stopLossPrice"] == 49_500.0
    assert sent["params"]["reduceOnly"] is True


# --------------------------------------------------------------------------- #
# H8: post-only is exchange-enforced; maker/slippage observed; escalation      #
# --------------------------------------------------------------------------- #
def _post_only_plan(style: str = "maker_first") -> OrderPlan:
    plan = _plan()
    plan.entry.order_type = OrderType.POST_ONLY
    plan.entry.price = 49_900.0
    plan.entry_style = style
    return plan


class PostOnlyRejectFake(FakeCcxt):
    """Rejects a postOnly create for crossing (like Bybit); market creates fill fully."""

    def create_order(self, symbol, type, side, qty, price, params=None):  # noqa: A002
        if (params or {}).get("postOnly"):
            raise RuntimeError("bybit: post only order would be filled immediately")
        super().create_order(symbol, type, side, qty, price, params)
        return {"average": 50_100.0, "filled": qty, "status": "closed", "fee": {"cost": 0.5}}


def test_post_only_entry_sends_the_real_post_only_flag() -> None:
    """A maker-intent (POST_ONLY) entry must carry ccxt's unified postOnly flag — without it
    the limit can cross the book and pay taker while being recorded maker."""
    fake = FakeCcxt()
    res = _venue(fake).place_bracket(
        _post_only_plan(), ref_price=50_000.0, realized_slippage_frac=0.001, latency_ms=5.0
    )
    sent = fake.orders[0]
    assert sent["type"] == "limit"
    assert sent["params"]["postOnly"] is True
    assert res.fill.maker is True  # post-only enforced ⇒ a resting fill IS maker


def test_maker_first_crossing_rejection_is_a_clean_nonfill() -> None:
    """maker_first semantics: a post-only entry the exchange rejects for crossing is a clean
    no-fill (no escalation, no phantom position, no crash)."""
    fake = PostOnlyRejectFake()
    venue = _venue(fake)
    res = venue.place_bracket(
        _post_only_plan("maker_first"),
        ref_price=50_000.0, realized_slippage_frac=0.001, latency_ms=5.0,
    )
    assert res.fill.qty == 0.0 and not res.fully_filled
    assert not fake.orders  # nothing escalated — maker_first never sends a taker order
    assert not venue.positions and not venue.open_orders


def test_passive_then_taker_escalates_on_crossing_rejection() -> None:
    """passive_then_taker: the crossing-rejected post-only entry escalates ONCE to a taker
    market order carrying the same attached protection, under a prefix-owned client id."""
    fake = PostOnlyRejectFake()
    venue = _venue(fake)
    plan = _post_only_plan("passive_then_taker")
    res = venue.place_bracket(
        plan, ref_price=50_000.0, realized_slippage_frac=0.001, latency_ms=5.0
    )
    assert len(fake.orders) == 1 and fake.orders[0]["type"] == "market"
    params = fake.orders[0]["params"]
    assert "postOnly" not in params
    assert params["clientOrderId"].startswith(_PREFIX)
    assert params["clientOrderId"] != plan.entry.client_id  # fresh id (no duplicate orderLinkId)
    assert "stopLoss" in params  # protection rides the escalated entry too (Section 2.2)
    assert res.fully_filled and res.fill.qty == pytest.approx(0.01)
    assert res.fill.maker is False  # HONEST: this fill paid taker
    assert res.fill.actual_price == 50_100.0
    assert res.fill.slippage_frac == pytest.approx(abs(50_100.0 - 49_900.0) / 49_900.0)
    assert venue.positions["BTC/USDT:USDT"].qty == pytest.approx(0.01)


class PartialThenEscalateFake(FakeCcxt):
    """Post-only create rests (no fill info); fetch shows a partial fill; the window-end
    cancel succeeds; the market escalation fills the remainder."""

    def __init__(self) -> None:
        super().__init__()
        self.status = "open"

    def create_order(self, symbol, type, side, qty, price, params=None):  # noqa: A002
        super().create_order(symbol, type, side, qty, price, params)
        if type == "market":
            return {
                "id": "m1", "filled": qty, "average": 50_150.0,
                "status": "closed", "fee": {"cost": 0.3},
            }
        return {"id": "l1"}  # resting: Bybit's create response has no fill fields

    def fetch_order(self, oid, symbol, params=None):
        return {
            "id": oid, "status": self.status, "filled": 0.004,
            "average": 49_900.0, "fee": {"cost": 0.1},
        }

    def cancel_order(self, oid, symbol, params=None):
        self.cancelled.append(oid)
        self.status = "canceled"
        return {}


def test_passive_then_taker_escalates_unfilled_remainder_at_window_end() -> None:
    """Partially filled at the observation-window end: the remainder is cancelled (verified),
    then escalated once as a market order; the fill books the volume-weighted average and is
    honestly non-maker."""
    fake = PartialThenEscalateFake()
    venue = _polling_venue(fake)
    res = venue.place_bracket(
        _post_only_plan("passive_then_taker"),
        ref_price=50_000.0, realized_slippage_frac=0.001, latency_ms=5.0,
    )
    assert fake.cancelled == ["l1"]  # passive remainder cancelled before escalating
    assert [o["type"] for o in fake.orders] == ["limit", "market"]
    assert fake.orders[1]["qty"] == pytest.approx(0.006)  # only the unfilled remainder
    expected_avg = (49_900.0 * 0.004 + 50_150.0 * 0.006) / 0.01
    assert res.fill.qty == pytest.approx(0.01) and res.fully_filled
    assert res.fill.actual_price == pytest.approx(expected_avg)
    assert res.fill.maker is False
    assert res.fill.fee == pytest.approx(0.1 + 0.3)
    assert venue.positions["BTC/USDT:USDT"].entry_price == pytest.approx(expected_avg)


def test_maker_first_never_escalates_at_window_end() -> None:
    """maker_first: an unfilled post-only entry at the window end stays a clean non-fill —
    exactly one (limit) order ever reaches the exchange."""
    fake = PollingFakeCcxt(status="open", filled=0.0)
    venue = _polling_venue(fake)
    res = venue.place_bracket(
        _post_only_plan("maker_first"),
        ref_price=50_000.0, realized_slippage_frac=0.001, latency_ms=5.0,
    )
    assert res.fill.qty == 0.0
    assert len(fake.orders) == 1 and fake.orders[0]["type"] == "limit"


def test_taker_slippage_recorded_from_observed_fill_not_model() -> None:
    """H8: a market entry's slippage_frac comes from the OBSERVED average vs the reference,
    not the hardcoded model value passed in."""

    class SlippedFake(FakeCcxt):
        def create_order(self, symbol, type, side, qty, price, params=None):  # noqa: A002
            super().create_order(symbol, type, side, qty, price, params)
            return {"average": 50_100.0, "filled": qty, "status": "closed", "fee": {"cost": 0.5}}

    res = _venue(SlippedFake()).place_bracket(
        _plan(), ref_price=50_000.0, realized_slippage_frac=0.009, latency_ms=5.0
    )
    assert res.fill.maker is False
    assert res.fill.slippage_frac == pytest.approx(100.0 / 50_000.0)  # observed, not 0.009


# --------------------------------------------------------------------------- #
# H7: exit fills for exchange-closed positions                                 #
# --------------------------------------------------------------------------- #
def test_fetch_exit_fill_vwaps_close_side_executions() -> None:
    class F(FakeCcxt):
        def fetch_my_trades(self, symbol, since=None, limit=None):
            return [
                # the entry execution — wrong side, must be ignored
                {"side": "buy", "amount": 0.01, "price": 50_000.0, "fee": {"cost": 0.5}},
                {"side": "sell", "amount": 0.006, "price": 51_000.0, "fee": {"cost": 0.2}},
                {"side": "sell", "amount": 0.004, "price": 51_500.0, "fee": {"cost": 0.1}},
            ]

    got = _venue(F()).fetch_exit_fill("BTC/USDT:USDT", 1, since_ts=123)
    assert got is not None
    price, fee = got
    assert price == pytest.approx((51_000.0 * 0.006 + 51_500.0 * 0.004) / 0.01)
    assert fee == pytest.approx(0.3)


def test_fetch_exit_fill_returns_none_without_history() -> None:
    # FakeCcxt has no fetch_my_trades → the venue degrades to None (caller falls back to mark).
    assert _venue(FakeCcxt()).fetch_exit_fill("BTC/USDT:USDT", 1) is None


# --------------------------------------------------------------------------- #
# M14: emergency close acts on the REAL exchange book                          #
# --------------------------------------------------------------------------- #
def test_emergency_close_all_flattens_exchange_book_after_restart() -> None:
    """After a restart the in-memory mirror is empty — emergency close must fetch the real
    exchange positions/orders: reduce-only close EVERY position and cancel OUR resting orders."""
    fake = FakeCcxt()
    fake._positions = [
        {"symbol": "BTC/USDT:USDT", "side": "long", "contracts": 0.01, "entryPrice": 50_000.0,
         "info": {"clientOrderId": f"{_PREFIX}entry_1"}},
        {"symbol": "ETH/USDT:USDT", "side": "short", "contracts": 0.1, "entryPrice": 3_000.0,
         "info": {"clientOrderId": "MANUAL_999"}},
    ]
    fake._open_orders = [
        {"symbol": "BTC/USDT:USDT", "side": "sell", "amount": 0.01, "price": 49_000.0,
         "clientOrderId": f"{_PREFIX}stop_1", "info": {}},
        {"symbol": "ETH/USDT:USDT", "side": "sell", "amount": 0.1, "price": 3_100.0,
         "clientOrderId": "MANUAL_h", "info": {}},
    ]
    venue = _venue(fake)  # fresh mirror: venue.positions/open_orders empty (the M14 bug)
    n = venue.emergency_close_all(confirm=True)
    closes = [o for o in fake.orders if o["params"].get("reduceOnly")]
    assert {(o["symbol"], o["side"]) for o in closes} == {
        ("BTC/USDT:USDT", "sell"),  # long → sell to flatten
        ("ETH/USDT:USDT", "buy"),  # short → buy to flatten
    }
    assert fake.cancelled == [f"{_PREFIX}stop_1"]  # only OUR order cancelled (Section 7)
    assert n == 3
    assert not venue.positions and not venue.open_orders


# --------------------------------------------------------------------------- #
# M15: cancel / close_position never pretend success                           #
# --------------------------------------------------------------------------- #
def _resting_stop() -> Order:
    return Order(
        client_id=f"{_PREFIX}stop_77", symbol="BTC/USDT:USDT", side="sell", qty=0.01,
        order_type=OrderType.STOP_MARKET, role="stop", stop_price=49_000.0,
        reduce_only=True, tags=_TAGS,
    )


def test_cancel_failure_with_live_order_keeps_it_tracked() -> None:
    class F(FakeCcxt):
        def cancel_order(self, oid, symbol, params=None):
            raise RuntimeError("network down")

        def fetch_order(self, oid, symbol, params=None):
            return {"id": oid, "status": "open"}  # verification: still live on the exchange

    venue = _venue(F())
    order = _resting_stop()
    venue.open_orders[order.client_id] = order
    assert venue.cancel(order.client_id) is False
    assert order.client_id in venue.open_orders  # never dropped on an unconfirmed cancel


def test_cancel_error_with_order_already_terminal_counts_as_cancelled() -> None:
    class F(FakeCcxt):
        def cancel_order(self, oid, symbol, params=None):
            raise RuntimeError("order not found")  # already gone / race

        def fetch_order(self, oid, symbol, params=None):
            return {"id": oid, "status": "canceled"}

    venue = _venue(F())
    order = _resting_stop()
    venue.open_orders[order.client_id] = order
    assert venue.cancel(order.client_id) is True
    assert order.client_id not in venue.open_orders


def test_close_position_failure_keeps_the_position_tracked() -> None:
    from src.execution.venue import VenuePosition

    class F(FakeCcxt):
        def create_order(self, *a, **k):  # the reduce-only close fails
            raise RuntimeError("exchange 5xx")

    venue = _venue(F())
    venue.positions["BTC/USDT:USDT"] = VenuePosition(
        symbol="BTC/USDT:USDT", side=1, qty=0.01, entry_price=50_000.0
    )
    stop = _resting_stop()
    venue.open_orders[stop.client_id] = stop
    assert venue.close_position("BTC/USDT:USDT") is False
    assert "BTC/USDT:USDT" in venue.positions  # still open on the exchange → still tracked
    assert stop.client_id in venue.open_orders  # legs untouched (position still needs them)
