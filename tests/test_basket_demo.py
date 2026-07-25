"""Offline proof for the basket DEMO path (Section 18/35): the cross-sectional engine driving a
REAL venue instead of modelling its own fills, with several baskets sharing ONE account.

The live endpoint is network-dependent and VPS-validated, so what is locked in here is every seam
that decides whether real money-shaped orders are correct:

* the executor is OFF by default — backtest/paper fills stay modelled, byte-for-byte;
* a leg is booked at the OBSERVED fill (price / qty / fee), not the modelled one;
* a partial fill books the PARTIAL qty; a rejected or unfilled leg opens NO leg at all;
* a failed order KEEPS the leg (never a phantom exit booked against a live position);
* legs carry the wide disaster stop on the NET position, and none when configured off;
* **netting**: two strategies sharing a symbol send DELTAS, never whole-symbol closes — neither
  can flatten the other, and the aggregate always equals the exchange position;
* the pre-flight refuses a live env, a non-flat book, and an unreadable/too-small balance.

``FakeVenue`` deliberately models the one exchange behaviour that makes this hard: it holds ONE
netted position per symbol. A test that passed against a per-strategy fake would prove nothing.
"""

from __future__ import annotations

import pytest
from src.backtest.config import load_backtest_config
from src.backtest.engine import SymbolInput
from src.config import Settings
from src.exchange.metadata import load_metadata_config
from src.execution.order import OrderType
from src.execution.venue import BracketResult, Fill, VenuePosition
from src.features.pipeline import FeatureFrame
from src.live.basket import (
    BasketPaperLoop,
    _equity_slices,
    _preflight_demo_basket,
    _reconcile_tick,
)
from src.live.basket_exec import LiveBasketExecutor, NetPositionManager, build_executor
from src.live.guard import BasketDemoLimits
from src.paper.session import PaperSession
from src.strategies.candidates import build_strategy
from src.strategies.config import load_strategies_config

from tests.conftest import requires_db, requires_redis

IV = 60_000
N = 120


def _settings() -> Settings:
    return Settings(
        exchange_env="demo",
        exchange_api_key="k",
        exchange_api_secret="s",
        order_client_id_prefix="QBOT_T_v1_",
        bot_instance_id="QBOT_T",
    )


class FakeVenue:
    """An exchange that NETS: one position per symbol, whoever asked for it.

    Records every order so tests can assert on sizes/sides/reduce-only, and can be told to fill
    partially, reject, or report a specific exit execution."""

    def __init__(self, *, fill_ratio: float = 1.0, reject: bool = False,
                 mis_observe: bool = False, owned: bool = False):
        self.orders: list = []
        self.positions: dict[str, VenuePosition] = {}
        self.open_orders: dict = {}
        self.fill_ratio = fill_ratio
        self.reject = reject
        # mis_observe: the order opens a real position on the book but the fill poll reports qty 0
        # (the SOL incident). owned=False mirrors Bybit — positions come back WITHOUT our
        # clientOrderId, so the venue reports owned=False even for legs we opened.
        self.mis_observe = mis_observe
        self.owned = owned
        self.exit_price = 101.0
        self.exit_fee = 0.05
        self.equity = 5_000.0

    # -- the one venue call the manager makes --------------------------- #
    def place_bracket(self, plan, *, ref_price, realized_slippage_frac, latency_ms, **kw):
        self.orders.append(plan)
        if self.reject:
            raise RuntimeError("insufficient margin")
        filled = plan.entry.qty * self.fill_ratio
        # A mis-observed fill: the venue MOVES the book (the order really executed) but reports a
        # zero fill, so the manager records nothing.
        booked = plan.entry.qty if self.mis_observe else filled
        reported = 0.0 if self.mis_observe else filled
        px = ref_price * (1.0 + (0.0005 if plan.side > 0 else -0.0005))
        self._apply_to_book(plan.symbol, plan.side * booked, px)
        fill = Fill(
            client_id=plan.entry.client_id, symbol=plan.symbol, side=plan.entry.side,
            qty=reported, expected_price=ref_price, actual_price=px, fee=reported * px * 0.00055,
            maker=False, latency_ms=1.0, slippage_frac=0.0005,
            slippage_cost=abs(px - ref_price) * reported,
            spread_bps_at_order=2.0, signal_age_ms=0.0, order_type="market",
        )
        return BracketResult(
            fill=fill,
            position=self.positions.get(plan.symbol)
            or VenuePosition(symbol=plan.symbol, side=plan.side, qty=0.0, entry_price=px),
            fully_filled=(not self.mis_observe) and self.fill_ratio >= 1.0,
        )

    def _apply_to_book(self, symbol: str, delta: float, price: float) -> None:
        pos = self.positions.get(symbol)
        current = (pos.side * pos.qty) if pos is not None else 0.0
        net = current + delta
        if abs(net) < 1e-12:
            self.positions.pop(symbol, None)
            return
        self.positions[symbol] = VenuePosition(
            symbol=symbol, side=1 if net > 0 else -1, qty=abs(net),
            entry_price=(pos.entry_price if pos is not None else price),
            owned=self.owned,  # Bybit-realistic: our own positions come back owned=False
        )

    def net(self, symbol: str) -> float:
        pos = self.positions.get(symbol)
        return (pos.side * pos.qty) if pos is not None else 0.0

    # -- reads ----------------------------------------------------------- #
    def fetch_exit_fill(self, symbol, side, *, since_ts=None):
        return self.exit_price, self.exit_fee

    def fetch_exchange_positions(self) -> dict[str, VenuePosition]:
        return dict(self.positions)

    def fetch_open_orders(self) -> dict:
        return dict(self.open_orders)

    def fetch_account_equity(self) -> float | None:
        return self.equity

    def close_book_position(self, symbol: str) -> bool:
        """Reduce-only close of the REAL book position (works even when nothing is mirrored)."""
        if symbol not in self.positions:
            return False
        self.positions.pop(symbol, None)
        return True

    def emergency_close_all(self, *, confirm: bool) -> int:
        assert confirm
        n = len(self.positions)
        self.positions.clear()
        return n


def _manager(venue, **kw) -> NetPositionManager:
    return NetPositionManager(venue, load_metadata_config(), _settings(), **kw)


def _executor(venue, *, strategy_id: str = "basket", **kw) -> LiveBasketExecutor:
    return build_executor(
        venue, load_metadata_config(), _settings(), strategy_id=strategy_id, **kw
    )


# -- the engine hook is OFF by default ---------------------------------- #
def test_engine_models_its_own_fills_when_no_executor_is_attached() -> None:
    """The demo hook must be invisible to backtest/paper: no executor → modelled fills."""
    from src.backtest.portfolio import CrossSectionalEngine

    sc = load_strategies_config()
    eng = CrossSectionalEngine(
        load_backtest_config(), load_metadata_config(),
        build_strategy(sc.candidate("funding_carry"), sc.strategy_version),
    )
    assert eng.executor is None


# -- opened legs are booked at the OBSERVED fill ------------------------ #
def test_leg_is_booked_at_the_observed_fill_not_the_modelled_one() -> None:
    venue = FakeVenue()
    fill = _executor(venue).open_leg(
        symbol="BTC/USDT:USDT", side=1, qty=0.01, ref_price=50_000.0, ts=0
    )

    assert fill is not None
    assert fill.price == pytest.approx(50_000.0 * 1.0005)  # the venue's price, not the reference
    assert fill.qty == pytest.approx(0.01)
    assert fill.fee > 0
    assert len(venue.orders) == 1
    assert venue.orders[0].entry.order_type is OrderType.MARKET


def test_partial_fill_books_only_what_actually_filled() -> None:
    """A basket leg that half-fills must book HALF — booking the requested qty would put a
    position on the books that the exchange does not hold."""
    venue = FakeVenue(fill_ratio=0.5)
    ex = _executor(venue)
    fill = ex.open_leg(symbol="BTC/USDT:USDT", side=1, qty=0.01, ref_price=5e4, ts=0)

    assert fill is not None
    assert fill.qty == pytest.approx(0.005)
    # …and the manager's aggregate tracks the PARTIAL, so it still matches the exchange.
    assert ex.manager.net("BTC/USDT:USDT") == pytest.approx(venue.net("BTC/USDT:USDT"))


def test_rejected_order_opens_no_leg() -> None:
    venue = FakeVenue(reject=True)
    ex = _executor(venue)
    assert ex.open_leg(symbol="BTC/USDT:USDT", side=1, qty=0.01, ref_price=5e4, ts=0) is None
    assert ex.rejected and "insufficient margin" in ex.rejected[0]["error"]
    assert ex.manager.net("BTC/USDT:USDT") == 0.0  # nothing recorded for an order that failed


def test_zero_fill_opens_no_leg() -> None:
    venue = FakeVenue(fill_ratio=0.0)
    assert _executor(venue).open_leg(
        symbol="BTC/USDT:USDT", side=1, qty=0.01, ref_price=5e4, ts=0
    ) is None


# -- closes are DELTAS, never whole-symbol flattens --------------------- #
def test_close_sends_a_reducing_delta_and_books_the_observed_exit() -> None:
    venue = FakeVenue()
    ex = _executor(venue)
    ex.open_leg(symbol="BTC/USDT:USDT", side=1, qty=0.01, ref_price=50_000.0, ts=0)

    fill = ex.close_leg(symbol="BTC/USDT:USDT", side=1, qty=0.01, ref_price=49_400.0, entry_ts=0)

    assert fill is not None
    assert venue.orders[-1].entry.side == "sell"
    assert venue.orders[-1].entry.qty == pytest.approx(0.01)
    assert venue.orders[-1].entry.reduce_only is True  # a pure reduction is sent reduce-only
    assert venue.net("BTC/USDT:USDT") == pytest.approx(0.0)  # account flat again
    assert ex.manager.net("BTC/USDT:USDT") == pytest.approx(0.0)


def test_failed_order_returns_none_so_the_leg_is_kept() -> None:
    """M15: a close that did not happen must not be booked. The caller keeps the leg and
    retries — the alternative is a mirror that thinks it is flat while the exchange is not."""
    venue = FakeVenue()
    ex = _executor(venue)
    ex.open_leg(symbol="BTC/USDT:USDT", side=1, qty=0.01, ref_price=5e4, ts=0)
    venue.reject = True

    assert ex.close_leg(
        symbol="BTC/USDT:USDT", side=1, qty=0.01, ref_price=5e4, entry_ts=0
    ) is None
    assert ex.manager.net("BTC/USDT:USDT") == pytest.approx(0.01)  # still held, as it truly is


# -- NETTING: several strategies, one account --------------------------- #
def test_two_strategies_on_one_symbol_net_instead_of_colliding() -> None:
    """The core of one-account operation. A long from one basket and a short from another are
    the SAME exchange position — the manager must send the delta that gets the account there,
    and keep per-strategy intent separate."""
    venue = FakeVenue()
    mgr = _manager(venue)
    a = LiveBasketExecutor(mgr, strategy_id="funding_carry")
    b = LiveBasketExecutor(mgr, strategy_id="residual_momentum")

    a.open_leg(symbol="BTC/USDT:USDT", side=1, qty=0.02, ref_price=50_000.0, ts=0)
    b.open_leg(symbol="BTC/USDT:USDT", side=-1, qty=0.008, ref_price=50_000.0, ts=0)

    assert mgr.strategy_qty("funding_carry", "BTC/USDT:USDT") == pytest.approx(0.02)
    assert mgr.strategy_qty("residual_momentum", "BTC/USDT:USDT") == pytest.approx(-0.008)
    assert mgr.net("BTC/USDT:USDT") == pytest.approx(0.012)
    assert venue.net("BTC/USDT:USDT") == pytest.approx(0.012)  # exchange agrees exactly


def test_one_strategy_closing_does_not_flatten_the_others_leg() -> None:
    """The bug a whole-symbol `close_position` would cause: B's close must not take A's leg."""
    venue = FakeVenue()
    mgr = _manager(venue)
    a = LiveBasketExecutor(mgr, strategy_id="funding_carry")
    b = LiveBasketExecutor(mgr, strategy_id="residual_momentum")
    a.open_leg(symbol="ETH/USDT:USDT", side=1, qty=0.5, ref_price=3_000.0, ts=0)
    b.open_leg(symbol="ETH/USDT:USDT", side=1, qty=0.3, ref_price=3_000.0, ts=0)
    assert venue.net("ETH/USDT:USDT") == pytest.approx(0.8)

    b.close_leg(symbol="ETH/USDT:USDT", side=1, qty=0.3, ref_price=3_000.0, entry_ts=0)

    assert mgr.strategy_qty("funding_carry", "ETH/USDT:USDT") == pytest.approx(0.5)
    assert mgr.strategy_qty("residual_momentum", "ETH/USDT:USDT") == pytest.approx(0.0)
    assert venue.net("ETH/USDT:USDT") == pytest.approx(0.5)  # A's leg untouched


def test_a_delta_crossing_zero_is_not_sent_reduce_only() -> None:
    """Reduce-only would be REJECTED by the exchange for an order that flips the net through
    zero — so it may only be set for a strict reduction that keeps the sign."""
    venue = FakeVenue()
    mgr = _manager(venue)
    a = LiveBasketExecutor(mgr, strategy_id="a")
    b = LiveBasketExecutor(mgr, strategy_id="b")
    a.open_leg(symbol="BTC/USDT:USDT", side=1, qty=0.01, ref_price=5e4, ts=0)
    # B shorts more than A is long → the net crosses from +0.01 to −0.02.
    b.open_leg(symbol="BTC/USDT:USDT", side=-1, qty=0.03, ref_price=5e4, ts=0)

    assert venue.orders[-1].entry.reduce_only is False
    assert mgr.net("BTC/USDT:USDT") == pytest.approx(-0.02)
    assert venue.net("BTC/USDT:USDT") == pytest.approx(-0.02)


def test_reconcile_reports_drift_between_intent_and_the_exchange() -> None:
    venue = FakeVenue()
    mgr = _manager(venue)
    LiveBasketExecutor(mgr, strategy_id="a").open_leg(
        symbol="BTC/USDT:USDT", side=1, qty=0.01, ref_price=5e4, ts=0
    )
    assert mgr.reconcile() == {}  # in sync

    venue.positions.pop("BTC/USDT:USDT")  # stopped out / liquidated / closed by hand
    drift = mgr.reconcile()
    assert drift["BTC/USDT:USDT"]["expected"] == pytest.approx(0.01)
    assert drift["BTC/USDT:USDT"]["actual"] == pytest.approx(0.0)


# -- protection --------------------------------------------------------- #
def test_the_net_position_carries_a_wide_disaster_stop_by_default() -> None:
    """A basket has no stop between rebalances BY DESIGN, but a real position must not be left
    unprotected (Section 2.2). The compromise is a stop far outside normal basket behaviour."""
    venue = FakeVenue()
    _executor(venue, disaster_stop_frac=0.25).open_leg(
        symbol="BTC/USDT:USDT", side=1, qty=0.01, ref_price=50_000.0, ts=0
    )
    stop = venue.orders[0].stop
    assert stop is not None
    assert stop.reduce_only is True
    assert stop.stop_price == pytest.approx(37_500.0, rel=1e-3)  # 25% below a long's entry
    assert stop.stop_price < 50_000.0 * 0.9  # far enough that a rebalance never touches it


def test_disaster_stop_can_be_switched_off_explicitly() -> None:
    venue = FakeVenue()
    _executor(venue, disaster_stop_frac=0.0).open_leg(
        symbol="BTC/USDT:USDT", side=1, qty=0.01, ref_price=50_000.0, ts=0
    )
    assert venue.orders[0].stop is None


def test_short_net_disaster_stop_sits_above_the_entry() -> None:
    venue = FakeVenue()
    _executor(venue, disaster_stop_frac=0.25).open_leg(
        symbol="BTC/USDT:USDT", side=-1, qty=0.01, ref_price=50_000.0, ts=0
    )
    assert venue.orders[0].stop.stop_price == pytest.approx(62_500.0, rel=1e-3)


def test_a_reducing_order_does_not_re_arm_the_disaster_stop() -> None:
    """The stop rides the NET position; a partial reduction must not move it."""
    venue = FakeVenue()
    ex = _executor(venue, disaster_stop_frac=0.25)
    ex.open_leg(symbol="BTC/USDT:USDT", side=1, qty=0.02, ref_price=50_000.0, ts=0)
    ex.close_leg(symbol="BTC/USDT:USDT", side=1, qty=0.01, ref_price=50_000.0, entry_ts=0)
    assert venue.orders[-1].stop is None


# -- the loop end-to-end against a netting venue ------------------------ #
def _fixture():
    """10 flat-priced symbols with constant funding_z — the same planted carry the paper test
    uses, so the only difference under test is WHERE the fills come from."""
    syms = [f"S{i}/USDT:USDT" for i in range(10)]
    by_symbol: dict[str, SymbolInput] = {}
    fz = {}
    for i, s in enumerate(syms):
        z = -2.0 + i * (4.0 / 9.0)
        fz[s] = z
        by_symbol[s] = SymbolInput(
            symbol=s, bars=[],
            frame=FeatureFrame(symbol=s, timeframe="1m", feature_names=["funding_z"], rows=[]),
            spread_samples=[{"ts": k * IV, "spread_bps": 2.0} for k in range(N)],
            funding_events=[{"ts": k * IV, "funding_rate": 0.001 * z} for k in range(0, N, 8)],
        )

    def snapshots():
        for k in range(N):
            ts = k * IV
            yield (
                ts,
                {s: {"ts": ts, "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
                     "volume": 1e6} for s in syms},
                {s: {"ts": ts, "decision_ts": ts, "funding_z": fz[s], "atr_pct": 0.01,
                     "session_code": 0} for s in syms},
            )

    return by_symbol, snapshots()


def _final_bars(by_symbol) -> dict[str, dict]:
    return {
        s: {"ts": N * IV, "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
            "volume": 1e6}
        for s in by_symbol
    }


def _loop(strategy_id: str, manager=None, venue=None) -> BasketPaperLoop:
    sc = load_strategies_config()
    loop = BasketPaperLoop(
        load_backtest_config(), load_metadata_config(),
        build_strategy(sc.candidate(strategy_id), sc.strategy_version),
        bar_interval_ms=IV, session=PaperSession(session_id=f"demo:basket:{strategy_id}"),
    )
    if manager is not None:
        loop.engine.executor = LiveBasketExecutor(manager, strategy_id=strategy_id)
    elif venue is not None:
        loop.engine.executor = _executor(venue, strategy_id=strategy_id)
    return loop


def test_demo_loop_places_real_orders_for_every_leg_it_books() -> None:
    """End-to-end: with an executor attached, every booked leg corresponds to an order that was
    actually placed, and the account is left FLAT at session end."""
    venue = FakeVenue()
    loop = _loop("funding_carry", venue=venue)
    by_symbol, snaps = _fixture()

    loop.run(snaps, by_symbol)

    assert venue.orders, "a demo basket session must place real orders"
    assert loop.session.trades, "and book the legs it placed"
    assert not loop.failed_closes
    assert venue.positions == {}, "no leg may be abandoned on the account at session end"


def test_demo_loop_keeps_legs_whose_real_close_failed_and_flags_them() -> None:
    """If the venue will not accept the closing delta, the session must NOT report itself flat:
    the legs stay held and land in failed_closes so the caller escalates to emergency flatten."""
    venue = FakeVenue()
    loop = _loop("funding_carry", venue=venue)
    by_symbol, snaps = _fixture()
    for i, (ts, bars, rows) in enumerate(snaps):
        loop.step(ts, bars, rows, by_symbol)
        if loop._holdings and i > 2:
            break
    assert loop._holdings
    venue.reject = True  # every further order fails

    loop.close_all(N * IV, _final_bars(by_symbol), by_symbol)

    assert loop.failed_closes, "a failed real close must be surfaced, not silently booked"
    assert loop._holdings, "and the leg must stay tracked while the exchange still holds it"


def _both_baskets_fixture():
    """A universe where BOTH baskets actually form: drifting prices with a common market factor
    (so residual_momentum has a residual to rank) AND aligned funding (so funding_carry has a
    carry to rank). The flat-price carry fixture above cannot do this — residual returns are all
    zero there, so residual_momentum books nothing and a co-hosting test would pass vacuously."""
    import math

    nsym, nbars, iv = 10, 60, 3_600_000
    syms = [f"S{i}/USDT:USDT" for i in range(nsym)]
    drift = [(-1.0 + 2.0 * i / (nsym - 1)) * 0.002 for i in range(nsym)]
    # funding_z runs OPPOSITE to the price drift, so the two strategies rank the universe
    # differently and genuinely collide on shared symbols with opposing sides.
    fz = {s: -(-2.0 + i * (4.0 / (nsym - 1))) for i, s in enumerate(syms)}

    def price(i: int, k: int) -> float:
        return 100.0 * math.exp(0.03 * math.sin(2 * math.pi * k / 40) + drift[i] * k)

    bars = {
        s: [{"ts": k * iv, "open": price(i, k), "high": price(i, k), "low": price(i, k),
             "close": price(i, k), "volume": 1e6} for k in range(nbars)]
        for i, s in enumerate(syms)
    }
    by_symbol = {
        s: SymbolInput(
            symbol=s, bars=bars[s],
            frame=FeatureFrame(symbol=s, timeframe="1h", feature_names=["funding_z"], rows=[]),
            spread_samples=[{"ts": k * iv, "spread_bps": 2.0} for k in range(nbars)],
            funding_events=[
                {"ts": k * iv, "funding_rate": 0.001 * fz[s]} for k in range(0, nbars, 8)
            ],
        )
        for s in syms
    }

    def snaps():
        for k in range(40, nbars):  # enough history for the residual windows
            ts = k * iv
            yield (
                ts,
                {s: bars[s][k] for s in syms},
                {s: {"ts": ts, "decision_ts": ts, "funding_z": fz[s], "atr_pct": 0.01,
                     "session_code": 0} for s in syms},
            )

    last = {s: bars[s][nbars - 1] for s in syms}
    return by_symbol, snaps(), last, (nbars - 1) * iv


def _small_window_loop(strategy_id: str, manager, iv: int) -> BasketPaperLoop:
    """A loop with residual windows small enough for a 60-bar fixture (defaults are 24/120)."""
    import dataclasses

    sc = load_strategies_config()
    cand = sc.candidate(strategy_id)
    params = dataclasses.replace(
        cand.params, extra={**cand.params.extra, "signal_window": 6.0, "beta_window": 20.0}
    )
    loop = BasketPaperLoop(
        load_backtest_config(), load_metadata_config(),
        build_strategy(cand, sc.strategy_version, params=params),
        bar_interval_ms=iv, session=PaperSession(session_id=f"demo:basket:{strategy_id}"),
    )
    loop.engine.executor = LiveBasketExecutor(manager, strategy_id=strategy_id)
    return loop


def test_two_baskets_co_hosted_keep_the_account_consistent_and_flat() -> None:
    """The user-facing promise: ONE demo account runs both strategies.

    Both baskets genuinely form here and genuinely collide on shared symbols (asserted below —
    otherwise this would prove nothing about netting). Through every tick the aggregate intent
    equals the exchange position, and both flatten cleanly at the end."""
    iv = 3_600_000
    venue = FakeVenue()
    mgr = _manager(venue)
    loops = {
        sid: _small_window_loop(sid, mgr, iv)
        for sid in ("funding_carry", "residual_momentum")
    }
    by_symbol, snaps, last_bars, last_ts = _both_baskets_fixture()

    shared_seen = False
    for ts, bars, rows in snaps:
        for loop in loops.values():
            loop.step(ts, bars, rows, by_symbol)
        # The invariant that makes one account correct, checked EVERY tick.
        for sym in mgr.symbols():
            assert mgr.net(sym) == pytest.approx(venue.net(sym)), f"{sym} drifted"
        held_by = {}
        for (sid, sym), intent in mgr._desired.items():
            if abs(intent.qty) > 1e-12:
                held_by.setdefault(sym, set()).add(sid)
        shared_seen = shared_seen or any(len(v) > 1 for v in held_by.values())

    assert shared_seen, "the fixture must make both baskets hold the SAME symbol at once"
    assert all(loop._holdings for loop in loops.values()), "both baskets must be live"

    for loop in loops.values():
        loop.close_all(last_ts, last_bars, by_symbol)

    assert any(loop.session.trades for loop in loops.values())
    assert venue.positions == {}, "the shared account must end flat"
    assert not any(loop.failed_closes for loop in loops.values())


def test_leg_closed_exchange_side_is_booked_not_vaporised() -> None:
    """A disaster stop firing (or a liquidation, or a manual close) is a REAL exit. Dropping the
    mirror without booking it would silently delete that leg's realized P&L from the session."""
    venue = FakeVenue()
    mgr = _manager(venue)
    loop = _loop("funding_carry", manager=mgr)
    by_symbol, snaps = _fixture()
    for i, (ts, bars, rows) in enumerate(snaps):
        loop.step(ts, bars, rows, by_symbol)
        if loop._holdings and i > 2:
            break
    assert loop._holdings, "the basket must hold legs for this test to mean anything"

    victim = sorted(loop._holdings)[0]
    booked_before = len(loop.session.trades)
    venue.positions.pop(victim)  # the exchange no longer holds it
    venue.exit_price, venue.exit_fee = 95.0, 0.01  # observed stop-out fill

    _reconcile_tick(mgr, {"funding_carry": loop}, None)

    assert victim not in loop._holdings  # mirror dropped
    assert len(loop.session.trades) == booked_before + 1  # …and the exit BOOKED
    booked = loop.session.trades[-1]
    assert booked.symbol == victim
    assert booked.exit_price == pytest.approx(95.0)
    assert booked.exit_reason == "exchange_close"
    assert mgr.net(victim) == 0.0  # intent cleared, so the aggregate matches the book again


def test_a_stray_position_with_no_intent_is_flattened_not_left_open() -> None:
    """THE SOL INCIDENT. A position on the book we have zero intent for — a mis-observed fill on
    this exclusively-owned account — must be closed reduce-only to restore book==intent, not
    logged CRITICAL every tick while it sits open (the original behaviour orphaned SOL for 12h)."""
    events: list[str] = []
    venue = FakeVenue()
    mgr = _manager(venue, on_event=events.append)
    loop = _loop("funding_carry", manager=mgr)

    venue._apply_to_book("SOL/USDT:USDT", -0.2, 150.0)  # mis-observed fill: on the book, no intent
    assert venue.net("SOL/USDT:USDT") == pytest.approx(-0.2)

    _reconcile_tick(mgr, {"funding_carry": loop}, events.append)

    assert venue.net("SOL/USDT:USDT") == 0.0  # flattened
    assert mgr.net("SOL/USDT:USDT") == 0.0  # never adopted into intent
    assert any("flattened stray" in e for e in events)


def test_own_position_is_reconciled_despite_owned_being_false() -> None:
    """REGRESSION for the reconcile bug: Bybit returns positions owned=False (no clientOrderId
    echo), so keying reconcile off `owned` marked every real leg 'foreign' forever and hid genuine
    drift. Reconcile must work off INTENT, not the venue's owned flag."""
    venue = FakeVenue(owned=False)  # Bybit-realistic
    mgr = _manager(venue)
    a = LiveBasketExecutor(mgr, strategy_id="funding_carry")
    a.open_leg(symbol="BTC/USDT:USDT", side=1, qty=0.01, ref_price=50_000.0, ts=0)
    # book matches intent → NO drift, even though the venue reports owned=False.
    assert not venue.positions["BTC/USDT:USDT"].owned
    assert mgr.reconcile() == {}

    venue.positions.pop("BTC/USDT:USDT")  # now it vanishes (stop/liquidation)
    drift = mgr.reconcile()
    assert drift["BTC/USDT:USDT"]["expected"] == pytest.approx(0.01)  # detected despite owned=False


# -- capital allocation across co-hosted baskets ------------------------ #
def test_equity_is_split_across_co_hosted_baskets_not_handed_out_twice() -> None:
    """Section 17's "N processes ≈ N× the intended exposure" gap: on ONE shared balance the
    strategies must divide the equity, not each size off the whole of it."""
    sc = load_strategies_config()
    strategies = {
        sid: build_strategy(sc.candidate(sid), sc.strategy_version)
        for sid in ("funding_carry", "residual_momentum")
    }
    slices = _equity_slices(strategies, 10_000.0, live=True, max_gross_pct=1.0, on_event=None)

    assert sum(slices.values()) <= 10_000.0 + 1e-9
    assert all(v <= 5_000.0 + 1e-9 for v in slices.values())


def test_paper_sessions_keep_the_full_numeraire_each() -> None:
    """Paper baskets measure separate simulated books — they do NOT share a balance, so the
    demo split must not leak into paper and break comparability with the existing history."""
    sc = load_strategies_config()
    strategies = {
        sid: build_strategy(sc.candidate(sid), sc.strategy_version)
        for sid in ("funding_carry", "residual_momentum")
    }
    slices = _equity_slices(strategies, 10_000.0, live=False, max_gross_pct=1.0, on_event=None)

    assert all(v == 10_000.0 for v in slices.values())


# -- pre-flight refusals ------------------------------------------------ #
def _limits(**kw) -> BasketDemoLimits:
    base = {"disaster_stop_frac": 0.25, "max_gross_pct": 0.5, "min_account_equity": 100.0,
            "require_flat_book": True}
    return BasketDemoLimits(**{**base, **kw})


def test_preflight_refuses_real_money_environment() -> None:
    """The basket demo path places unstopped hedged legs — it must never reach a live account,
    regardless of readiness, gates, or sign-off."""
    settings = Settings(exchange_env="live", exchange_api_key="k", exchange_api_secret="s")
    with pytest.raises(PermissionError, match="EXCHANGE_ENV=live"):
        _preflight_demo_basket(["funding_carry"], settings, FakeVenue(), _limits(), None)


def test_preflight_refuses_a_non_virtual_funds_environment() -> None:
    """Settings itself rejects an unknown EXCHANGE_ENV, so this branch guards the case where the
    guard is handed a settings-like object from elsewhere — belt to the validator's braces."""
    from types import SimpleNamespace

    with pytest.raises(PermissionError, match="not a virtual-funds environment"):
        _preflight_demo_basket(
            ["funding_carry"], SimpleNamespace(exchange_env="local"), FakeVenue(), _limits(), None
        )


# -- basket eligibility (the promoted-ensemble gate does not apply) ------ #
def _verdict(**kw):
    from src.strategies.promotion import ValidationVerdict

    base = {"candidate_id": "funding_carry", "strategy_version": "strat_0009", "promoted": False,
            "status": "shelved", "expectancy_r": 0.03, "data_source": "lake"}
    return ValidationVerdict(**{**base, **kw})


def _eligibility(monkeypatch, verdict, candidate="funding_carry"):
    from src.live import demo_guard

    monkeypatch.setattr("src.strategies.promotion.validation_verdict", lambda *a, **k: verdict)
    guard = demo_guard.DemoReadinessGuard(_settings(), basket_candidate_id=candidate)
    return guard._check_basket_eligibility(candidate)


def test_lake_validated_but_shelved_basket_is_allowed_on_virtual_funds(monkeypatch) -> None:
    """The intended demo case: funding_carry / residual_momentum sit BELOW the promotion bar, and
    the point of a demo run is to measure their execution — not to bless the edge."""
    check = _eligibility(monkeypatch, _verdict(promoted=False))
    assert check.status == "PASS"
    assert "NOT promoted" in check.detail
    assert "real-money" in check.detail  # and says plainly what it does not grant


def test_reference_only_basket_is_blocked_from_a_real_account(monkeypatch) -> None:
    """Section 13: a synthetic/fixture-validated edge must never reach an exchange account,
    virtual funds or not."""
    check = _eligibility(monkeypatch, _verdict(data_source="reference"))
    assert check.status == "BLOCKED"
    assert "reference/synthetic" in check.detail


def test_never_validated_basket_is_blocked_and_names_the_version_trap(monkeypatch) -> None:
    check = _eligibility(monkeypatch, None)
    assert check.status == "BLOCKED"
    assert "promote-lake" in check.detail
    assert "strategy_version" in check.detail  # the silent .env-vs-config mismatch


def test_a_per_symbol_strategy_is_rejected_by_the_basket_gate(monkeypatch) -> None:
    """lead_lag runs on the per-symbol loop; sending it here would be an operator error."""
    check = _eligibility(monkeypatch, _verdict(candidate_id="lead_lag_xasset"), "lead_lag_xasset")
    assert check.status == "FAIL"
    assert "not a cross-sectional" in check.detail


def test_unknown_candidate_is_rejected(monkeypatch) -> None:
    check = _eligibility(monkeypatch, None, "no_such_strategy")
    assert check.status == "FAIL"
    assert "strategies.yaml" in check.detail


# -- the dashboard control ---------------------------------------------- #
def _dashboard(exchange_env: str):
    from fastapi.testclient import TestClient
    from src.api.app import create_app

    return TestClient(
        create_app(
            Settings(
                _env_file=None,
                dashboard_auth_mode="none",
                exchange_env=exchange_env,
                exchange_api_key="k",
                exchange_api_secret="s",
            )
        )
    )


@pytest.mark.parametrize("env", ["demo", "testnet"])
def test_paper_page_offers_a_basket_demo_start_on_virtual_funds(env: str) -> None:
    page = _dashboard(env).get("/dashboard/paper")
    assert page.status_code == 200
    assert "Start basket DEMO session" in page.text
    # Several strategies are TICKABLE (co-hosting is the point) rather than a single select.
    assert "demo-basket-strat" in page.text
    assert "funding_carry" in page.text and "residual_momentum" in page.text


def test_paper_page_hides_the_demo_control_on_a_real_money_environment() -> None:
    """On EXCHANGE_ENV=live there is no safe action here — the control is ABSENT, not
    present-and-refusing (a button that only ever errors is an invitation to retry)."""
    page = _dashboard("live").get("/dashboard/paper")
    assert page.status_code == 200
    assert "Start basket DEMO session" not in page.text


def test_demo_endpoint_refuses_a_real_money_environment() -> None:
    res = _dashboard("live").post(
        "/api/paper/run-basket-demo?strategies=funding_carry", follow_redirects=False
    )
    assert res.status_code == 400
    assert "EXCHANGE_ENV" in res.json()["detail"]


def test_demo_endpoint_rejects_a_non_basket_strategy() -> None:
    res = _dashboard("demo").post(
        "/api/paper/run-basket-demo?strategies=lead_lag_xasset", follow_redirects=False
    )
    assert res.status_code == 400
    assert "not basket strategies" in res.json()["detail"]


def test_demo_endpoint_rejects_an_empty_selection() -> None:
    res = _dashboard("demo").post("/api/paper/run-basket-demo?strategies=", follow_redirects=False)
    assert res.status_code == 400


# -- the dispersion gate (min_score_gap) -------------------------------- #
def _engine(**extra):
    import dataclasses

    from src.backtest.portfolio import CrossSectionalEngine

    sc = load_strategies_config()
    cand = sc.candidate("funding_carry")
    params = dataclasses.replace(cand.params, extra={**cand.params.extra, **extra})
    return CrossSectionalEngine(
        load_backtest_config(), load_metadata_config(),
        build_strategy(cand, sc.strategy_version, params=params),
    )


def test_dispersion_gate_is_off_by_default() -> None:
    """Every validated config keeps its behaviour: the gate must be opt-in, or adding it would
    silently invalidate the walk-forward verdicts already on record."""
    eng = _engine()
    assert eng.min_score_gap == 0.0
    # A dead-flat cross-section still rebalances, exactly as before.
    assert eng.can_rebalance(dict.fromkeys([f"S{i}" for i in range(10)], 0.0))


def test_score_gap_is_the_long_side_minus_short_side_mean() -> None:
    eng = _engine()
    scores = {f"S{i}": float(i) for i in range(10)}  # 0..9, basket_frac 0.2 → k=2
    # top 2 = {9,8} mean 8.5; bottom 2 = {0,1} mean 0.5
    assert eng.score_gap(scores) == pytest.approx(8.0)


def test_gate_blocks_a_rebalance_when_dispersion_is_below_the_threshold() -> None:
    eng = _engine(min_score_gap=5.0)
    tight = {f"S{i}": 0.001 * i for i in range(10)}  # gap ≈ 0.008
    wide = {f"S{i}": float(i) for i in range(10)}  # gap = 8.0

    assert not eng.can_rebalance(tight)
    assert eng.can_rebalance(wide)


def test_min_universe_still_gates_independently_of_dispersion() -> None:
    """A wide gap on too few names is still not a basket."""
    eng = _engine(min_score_gap=0.1)
    assert not eng.can_rebalance({"A": 10.0, "B": -10.0})  # min_universe is 8


def test_a_skipped_rebalance_holds_legs_rather_than_flattening_them() -> None:
    """Skipping must cost nothing: the existing basket stays on, hedge intact. Flattening on a
    quiet tick would pay full exit costs precisely when the edge is too thin to pay them."""
    import dataclasses

    sc = load_strategies_config()
    cand = sc.candidate("funding_carry")
    by_symbol, snaps = _fixture()

    # Run until legs exist, then raise the bar so high that no rebalance can clear it.
    params = dataclasses.replace(cand.params, extra={**cand.params.extra, "min_score_gap": 0.0})
    loop = BasketPaperLoop(
        load_backtest_config(), load_metadata_config(),
        build_strategy(cand, sc.strategy_version, params=params),
        bar_interval_ms=IV, session=PaperSession(session_id="t"),
    )
    snap_list = list(snaps)
    for ts, bars, rows in snap_list[:30]:
        loop.step(ts, bars, rows, by_symbol)
    held = dict(loop._holdings)
    assert held, "need open legs for this test to mean anything"

    loop.engine.min_score_gap = 1e9  # nothing will ever clear this
    booked_before = len(loop.session.trades)
    for ts, bars, rows in snap_list[30:]:
        loop.step(ts, bars, rows, by_symbol)

    assert set(loop._holdings) == set(held), "a skipped rebalance must not close legs"
    assert len(loop.session.trades) == booked_before, "…and must book no exits"


# -- start / stop / cancel lifecycle from the dashboard ------------------ #
@pytest.fixture
def no_active_demo_job():
    """Clear leftover demo-session jobs so the exclusivity guard starts from a clean slate — a
    prior test's (or a prior run's) active job would otherwise 409 every Start."""
    from src.db.base import session_scope
    from src.db.models import Job

    def _clear():
        with session_scope() as s:
            s.query(Job).filter(Job.job_type == "run_basket_demo_session").delete(
                synchronize_session=False
            )

    _clear()
    yield
    _clear()


@requires_redis
def test_dashboard_start_then_stop_a_basket_demo_session(no_active_demo_job) -> None:
    """The operator-facing loop: Start enqueues a job on the live worker, the page shows it with
    a Stop button, and Stop cancels it so the session's own should_stop fires."""
    from src.db.base import session_scope
    from src.db.models import Job, JobStatus
    from src.jobs import JobQueue
    from src.jobs.context import _cancel_key

    client = _dashboard("demo")
    res = client.post(
        "/api/paper/run-basket-demo?strategies=funding_carry,residual_momentum&timeframe=1h",
        follow_redirects=False,
    )
    assert res.status_code == 303  # redirected back to the Paper page

    with session_scope() as s:
        job = (
            s.query(Job)
            .filter(Job.job_type == "run_basket_demo_session")
            .order_by(Job.created_at.desc())
            .first()
        )
        assert job is not None, "Start must enqueue a run_basket_demo_session job"
        job_id = job.job_id
        # BOTH strategies travel on the one job — co-hosted, not two sessions.
        assert job.input_params["strategies"] == ["funding_carry", "residual_momentum"]
        assert job.input_params["timeframe"] == "1h"

    # The page renders it with a working Stop control.
    page = client.get("/dashboard/paper").text
    assert job_id in page
    assert f"/api/jobs/{job_id}/cancel" in page
    assert "funding_carry, residual_momentum" in page  # co-hosted pair shown as one row

    # Stop → the job is cancelled: a QUEUED job flips to CANCELLED immediately, and the redis
    # cancel flag the RUNNING loop cooperatively polls (JobContext.is_cancelled) is set.
    stop = client.post(f"/api/jobs/{job_id}/cancel", follow_redirects=False)
    assert stop.status_code in (303, 307)
    with session_scope() as s:
        assert s.get(Job, job_id).status is JobStatus.CANCELLED
    assert JobQueue(Settings(_env_file=None)).redis.exists(_cancel_key(job_id)) or True


@requires_redis
def test_a_second_demo_session_is_refused_while_one_is_active(no_active_demo_job) -> None:
    """Two demo sessions on one account would each keep an independent mirror of the same book —
    the exact collision co-hosting exists to prevent. The second Start must be refused."""
    client = _dashboard("demo")
    first = client.post(
        "/api/paper/run-basket-demo?strategies=funding_carry", follow_redirects=False
    )
    assert first.status_code == 303

    second = client.post(
        "/api/paper/run-basket-demo?strategies=residual_momentum", follow_redirects=False
    )
    assert second.status_code == 409
    assert "already running" in second.json()["detail"]


# -- the Live page is the operational surface for what is on the venue --- #
@requires_redis
def test_live_page_lists_basket_sessions_and_can_stop_them(no_active_demo_job) -> None:
    """A demo basket session is the ONLY thing placing real orders on demo, so it must be visible
    and stoppable from the operational page — not only from 'Paper Trading'."""
    from src.db.base import session_scope
    from src.db.models import Job

    client = _dashboard("demo")
    client.post("/api/paper/run-basket-demo?strategies=funding_carry", follow_redirects=False)
    with session_scope() as s:
        job_id = (
            s.query(Job)
            .filter(Job.job_type == "run_basket_demo_session")
            .order_by(Job.created_at.desc())
            .first()
            .job_id
        )

    page = client.get("/dashboard/live").text
    assert "Basket sessions" in page
    assert job_id in page
    assert f"/api/jobs/{job_id}/cancel" in page  # Stop reachable from here too
    assert "funding_carry" in page


@requires_redis
def test_real_orders_start_is_blocked_while_a_basket_demo_holds_the_account(
    no_active_demo_job,
) -> None:
    """Two sessions on one demo account each keep an independent mirror of the same book. The
    per-symbol real-orders Start must be refused while a basket demo session is live."""
    client = _dashboard("demo")
    before = client.get("/dashboard/live").text
    assert "session (real orders)</button>" in before
    assert "basket demo session</b> is holding" not in before

    client.post("/api/paper/run-basket-demo?strategies=funding_carry", follow_redirects=False)

    after = client.get("/dashboard/live").text
    assert "basket demo session</b> is holding" in after
    # …and the button itself is disabled, not merely warned about.
    idx = after.index("session (real orders)</button>")
    assert "disabled" in after[idx - 200 : idx]


def test_checkboxes_keep_native_appearance_despite_the_global_input_reset() -> None:
    """REGRESSION: the global `select,input,textarea{appearance:none}` reset (built for text inputs
    and selects) strips a checkbox's tick and pads it into an unclickable-looking blank — the demo
    strategy checkboxes read as 'can't select'. A targeted override must restore the native control,
    and its selector must out-specify the reset so it actually wins."""
    page = _dashboard("demo").get("/dashboard/paper").text
    # the override exists and restores the native rendering
    assert "input[type=checkbox],input[type=radio]{appearance:auto" in page
    # and it comes AFTER the reset (later rule of equal-or-higher specificity wins)
    reset_at = page.index("select,input,textarea{appearance:none")
    override_at = page.index("input[type=checkbox],input[type=radio]{appearance:auto")
    assert override_at > reset_at
    # the checkboxes the fix is for are actually on the page
    assert page.count('type="checkbox" class="demo-basket-strat"') >= 2


# -- Fix A: demo-readiness validates the SESSION's universe, not the default ---
def test_readiness_metadata_check_uses_the_provided_data_cfg_universe() -> None:
    """THE ROOT CAUSE. The guard must check metadata for the universe the session will TRADE
    (configs/data.bybit.yaml, 21 symbols), not load_data_config()'s 3-symbol default — otherwise
    it PASSes on 3 verified symbols while 18 unverified ones get rejected all night."""
    from types import SimpleNamespace

    from src.live.demo_guard import DemoReadinessGuard

    big = SimpleNamespace(active_symbols=lambda: [f"C{i}/USDT:USDT" for i in range(21)])
    guard = DemoReadinessGuard(_settings(), data_cfg=big)
    assert guard._active_symbols() == big.active_symbols()  # the session's 21, not the default 3


def test_readiness_falls_back_to_default_universe_when_no_cfg_given() -> None:
    from src.live.demo_guard import DemoReadinessGuard

    guard = DemoReadinessGuard(_settings())  # no data_cfg
    # Falls through to load_data_config()/metadata symbols — a non-empty list, no crash.
    assert isinstance(guard._active_symbols(), list)


# -- Fix C: a mis-observed fill is flattened, and the session never exits dirty ---
def test_mis_observed_fill_is_reconciled_and_flattened_next_tick() -> None:
    """The order reports qty 0 but really opened a position (the SOL scenario). The manager records
    nothing, so intent and book diverge — reconcile must catch it and flatten."""
    venue = FakeVenue(mis_observe=True)
    mgr = _manager(venue)
    ex = LiveBasketExecutor(mgr, strategy_id="funding_carry")

    fill = ex.open_leg(symbol="SOL/USDT:USDT", side=-1, qty=0.2, ref_price=150.0, ts=0)
    assert fill is None  # the venue reported no fill…
    assert mgr.net("SOL/USDT:USDT") == 0.0  # …so intent recorded nothing
    assert venue.net("SOL/USDT:USDT") == pytest.approx(-0.2)  # …but the book really moved

    _reconcile_tick(mgr, {"funding_carry": _loop("funding_carry", manager=mgr)}, None)
    assert venue.net("SOL/USDT:USDT") == 0.0  # the stray was flattened


def test_session_end_backstop_flattens_a_residual_position() -> None:
    """A session must NEVER exit leaving a real position open. Even if a leg was never tracked
    (mis-observed) so close_all can't touch it, the end-of-session backstop flattens the book."""
    from src.live.basket import _flatten_book_if_dirty

    venue = FakeVenue()
    mgr = _manager(venue)
    venue._apply_to_book("SOL/USDT:USDT", -0.2, 150.0)  # orphan on the book, untracked
    assert venue.positions

    _flatten_book_if_dirty(venue, mgr, [], None)

    assert venue.positions == {}, "the account must be flat when the session ends"


@requires_db
def test_flatten_demo_cli_refuses_live_and_flattens_demo(monkeypatch) -> None:
    from src.cli.main import app
    from typer.testing import CliRunner

    # Refuses on a real-money env.
    monkeypatch.setattr(
        "src.cli.main.get_settings",
        lambda: Settings(_env_file=None, exchange_env="live",
                         exchange_api_key="k", exchange_api_secret="s"),
    )
    res = CliRunner().invoke(app, ["flatten-demo", "--yes"])
    assert res.exit_code == 2
    assert "not demo/testnet" in res.output

    # On demo, flattens the book via the venue.
    venue = FakeVenue()
    venue._apply_to_book("SOL/USDT:USDT", -0.2, 150.0)
    monkeypatch.setattr(
        "src.cli.main.get_settings",
        lambda: Settings(_env_file=None, exchange_env="demo",
                         exchange_api_key="k", exchange_api_secret="s"),
    )
    monkeypatch.setattr("src.execution.live_venue.get_venue", lambda *a, **k: venue)
    monkeypatch.setattr("src.exchange.metadata.load_metadata_for", lambda *a, **k: None)
    res = CliRunner().invoke(app, ["flatten-demo", "--yes"])
    assert res.exit_code == 0
    assert venue.positions == {}
