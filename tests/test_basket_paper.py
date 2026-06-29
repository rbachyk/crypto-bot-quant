"""Offline proof for the live/paper basket loop: a planted funding-carry edge, fed one snapshot
at a time, books profitable PaperTrades — confirming the live path reuses the engine math and the
Trade→PaperTrade conversion is sound (before it ever touches a live feed)."""

from __future__ import annotations

from src.backtest.config import load_backtest_config
from src.backtest.engine import SymbolInput
from src.config import Settings
from src.exchange.metadata import load_metadata_config
from src.features.pipeline import FeatureFrame
from src.killswitch import KillSwitch
from src.live.basket import BasketPaperLoop, _halt_check
from src.paper.session import PaperSession
from src.strategies.candidates import build_strategy
from src.strategies.config import load_strategies_config

IV = 60_000
N = 240


def _fixture():
    """10 symbols, flat price (carry is the only P&L), constant funding_z, funding rate ALIGNED
    with funding_z (high funding ⇒ shorts collect, low ⇒ longs collect). Returns the per-symbol
    SymbolInput (spread + funding) and the snapshot stream."""
    syms = [f"S{i}/USDT:USDT" for i in range(10)]
    by_symbol: dict[str, SymbolInput] = {}
    fz = {}
    for i, s in enumerate(syms):
        z = -2.0 + i * (4.0 / 9.0)
        fz[s] = z
        funding = [{"ts": k * IV, "funding_rate": 0.001 * z} for k in range(0, N, 8)]
        spread = [{"ts": k * IV, "spread_bps": 2.0} for k in range(N)]
        frame = FeatureFrame(symbol=s, timeframe="1m", feature_names=["funding_z"], rows=[])
        by_symbol[s] = SymbolInput(
            symbol=s, bars=[], frame=frame, spread_samples=spread, funding_events=funding
        )

    def snapshots():
        for k in range(N):
            ts = k * IV
            bars_at = {
                s: {"ts": ts, "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
                    "volume": 1e6}
                for s in syms
            }
            rows_at = {
                s: {"ts": ts, "decision_ts": ts, "funding_z": fz[s], "atr_pct": 0.01,
                    "session_code": 0}
                for s in syms
            }
            yield (ts, bars_at, rows_at)

    return by_symbol, snapshots()


def test_basket_paper_loop_books_profitable_carry_trades():
    cfg = load_backtest_config()
    meta = load_metadata_config()
    sc = load_strategies_config()
    strat = build_strategy(sc.candidate("funding_carry"), sc.strategy_version)
    session = PaperSession(session_id="t")
    by_symbol, snaps = _fixture()

    BasketPaperLoop(cfg, meta, strat, bar_interval_ms=IV, session=session).run(snaps, by_symbol)

    assert session.trades, "the basket loop should book paper trades"
    longs = [t for t in session.trades if t.side > 0]
    shorts = [t for t in session.trades if t.side < 0]
    assert longs and shorts  # dollar-neutral basket trades both sides
    assert sum(t.pnl for t in session.trades) > 0.0  # planted carry is harvested
    assert all(t.execution_route == "maker" for t in session.trades)  # rebalanced as maker


def test_basket_loop_residual_momentum_forms_a_basket() -> None:
    """REGRESSION: the live loop must score residual_momentum via the engine's _residual_score
    (from the rolling bars), NOT strategy.score() — which returns None, so the basket never formed
    and booked 0 legs (the live VPS symptom). With the parity fix it forms a basket and books legs.
    Planted: a common market factor + per-symbol idiosyncratic drift the residual must isolate."""
    import dataclasses
    import math

    cfg = load_backtest_config()
    meta = load_metadata_config()
    sc = load_strategies_config()
    cand = sc.candidate("residual_momentum")
    # small windows so the test needs few bars (defaults are 24/120)
    extra = {**cand.params.extra, "signal_window": 6.0, "beta_window": 20.0}
    params = dataclasses.replace(cand.params, extra=extra)
    strat = build_strategy(cand, sc.strategy_version, params=params)

    nsym, nbars, iv = 10, 60, 3_600_000
    syms = [f"S{i}/USDT:USDT" for i in range(nsym)]
    drift = [(-1.0 + 2.0 * i / (nsym - 1)) * 0.002 for i in range(nsym)]  # ±, symmetric

    def price(i: int, k: int) -> float:
        return 100.0 * math.exp(0.03 * math.sin(2 * math.pi * k / 40) + drift[i] * k)

    bars = {s: [{"ts": k * iv, "open": price(i, k), "high": price(i, k), "low": price(i, k),
                 "close": price(i, k), "volume": 1e6} for k in range(nbars)]
            for i, s in enumerate(syms)}
    by_symbol = {s: SymbolInput(
        symbol=s, bars=bars[s],
        frame=FeatureFrame(symbol=s, timeframe="1h", feature_names=[], rows=[]),
        spread_samples=[{"ts": k * iv, "spread_bps": 2.0} for k in range(nbars)], funding_events=[],
    ) for s in syms}

    def snaps():
        for k in range(40, nbars):  # enough history before each ts for the 6+20 windows
            ts = k * iv
            yield (ts, {s: bars[s][k] for s in syms},
                   {s: {"ts": ts, "decision_ts": ts, "atr_pct": 0.01, "session_code": 0}
                    for s in syms})

    session = PaperSession(session_id="t")
    BasketPaperLoop(cfg, meta, strat, bar_interval_ms=iv, session=session).run(snaps(), by_symbol)

    assert session.trades, "residual_momentum must FORM a basket in the live loop (was 0 legs)"
    assert any(t.side > 0 for t in session.trades) and any(t.side < 0 for t in session.trades)


def test_basket_loop_reports_open_positions() -> None:
    """The loop must emit its held legs (marked to market, with unrealized P&L) via on_positions
    while they're open, and clear them (empty list) when the session goes flat — what powers the
    dashboard's live Open-positions panel."""
    cfg = load_backtest_config()
    meta = load_metadata_config()
    sc = load_strategies_config()
    strat = build_strategy(sc.candidate("funding_carry"), sc.strategy_version)
    by_symbol, snaps = _fixture()
    captured: list[list[dict]] = []
    BasketPaperLoop(cfg, meta, strat, bar_interval_ms=IV, session=PaperSession("t"),
                    on_positions=lambda p: captured.append(list(p))).run(snaps, by_symbol)

    held = [snap for snap in captured if snap]
    assert held, "loop should report open positions while legs are held"
    pos = held[0][0]
    assert {"symbol", "side", "qty", "entry_price", "mark_price", "unrealized_pnl", "strategy"} \
        <= set(pos)
    assert captured[-1] == []  # close_all flattened → panel cleared


def test_engine_open_positions_snapshot_marks_to_market() -> None:
    """The per-symbol engine snapshots its open positions marked to market (with strategy + entry
    ts) — wiring lead_lag into the same live open-positions panel the baskets feed."""
    from src.paper.engine import PaperTradingEngine
    from src.risk.portfolio import Position

    eng = PaperTradingEngine()
    eng._open_positions["ETH/USDT:USDT"] = Position(
        symbol="ETH/USDT:USDT", side=1, qty=2.0, entry_price=100.0, risk_amount=10.0,
        beta_to_btc=1.0, regime="R1",
    )
    eng._position_meta["ETH/USDT:USDT"] = ("lead_lag_xasset", 123)

    p = eng.open_positions(lambda _s: 110.0)[0]
    assert p["strategy"] == "lead_lag_xasset" and p["entry_ts"] == 123
    assert p["mark_price"] == 110.0 and p["unrealized_pnl"] == 20.0  # +1 × (110-100) × 2
    # no price available → mark at entry → 0 unrealized (never a bogus number)
    assert eng.open_positions(lambda _s: None)[0]["unrealized_pnl"] == 0.0


def test_engine_simulates_paper_bracket_and_time_stop_exits() -> None:
    """A held PAPER position (exit_move_frac=0, so the engine never closes it inline) must be
    flattened by simulate_paper_exits when a new bar breaches its stop/TP/time-stop — the offline
    SimulatedVenue never fills the resting bracket, so without this paper positions never close and
    the session books no realized P&L. The existing open trade record is closed IN PLACE (not
    duplicated) and realized P&L is accumulated."""
    from src.paper.engine import PaperTradingEngine
    from src.paper.session import PaperSession, PaperTrade
    from src.risk.portfolio import Position

    def _seed(eng: PaperTradingEngine, sess: PaperSession, sym: str, side: int,
              stop: float, tp: float, hold: int) -> None:
        eng._open_positions[sym] = Position(
            symbol=sym, side=side, qty=2.0, entry_price=100.0, risk_amount=10.0,
            beta_to_btc=1.0, regime="R1",
        )
        eng._position_meta[sym] = ("lead_lag_xasset", 1_000)
        eng._exit_levels[sym] = (stop, tp, hold, 1_000)
        sess.trades.append(PaperTrade(
            trade_id=sym[:3], symbol=sym, strategy="lead_lag_xasset", side=side, qty=2.0,
            entry_price=100.0, stop_price=stop, tp_price=tp, regime="R1", session=0,
            decision_ts=1_000, entry_ts=1_000, exit_ts=1_000, exit_price=100.0, exit_reason="open",
            fee=0.2, slippage_cost=0.0, pnl=-0.2, pnl_r=0.0, has_exchange_side_stop=True,
            execution_route="maker", spread_bps_at_entry=0.0, slippage_frac=0.0,
        ))

    # --- long: take-profit breach closes in place with a realized profit ---
    eng = PaperTradingEngine()
    sess = PaperSession(session_id="t")
    _seed(eng, sess, "ETH/USDT:USDT", 1, 95.0, 110.0, 0)
    assert eng.simulate_paper_exits(lambda _s: 102.0, 2_000, sess) == 0  # inside the bracket
    assert "ETH/USDT:USDT" in eng._open_positions
    assert eng.simulate_paper_exits(lambda _s: 111.0, 3_000, sess) == 1  # TP breached
    assert "ETH/USDT:USDT" not in eng._open_positions  # risk slot released
    assert len(sess.trades) == 1  # closed in place, NOT duplicated
    t = sess.trades[0]
    assert t.exit_reason == "take_profit" and t.exit_price == 111.0 and t.exit_ts == 3_000
    assert t.pnl > 0 and eng._realized_pnl == t.pnl  # +1×(111−100)×2 − fees, realized

    # --- short: a price RISE to the stop (above entry) closes for a loss ---
    eng = PaperTradingEngine()
    sess = PaperSession(session_id="t")
    _seed(eng, sess, "SOL/USDT:USDT", -1, 105.0, 90.0, 0)
    assert eng.simulate_paper_exits(lambda _s: 104.0, 2_000, sess) == 0
    assert eng.simulate_paper_exits(lambda _s: 106.0, 3_000, sess) == 1  # short stop breached
    assert sess.trades[0].exit_reason == "stop" and sess.trades[0].pnl < 0

    # --- time-stop: no bracket breach, but the hold horizon elapses ---
    eng = PaperTradingEngine()
    sess = PaperSession(session_id="t")
    _seed(eng, sess, "BTC/USDT:USDT", 1, 1.0, 1e9, hold=3)  # bracket never breached
    iv = 60_000
    assert eng.simulate_paper_exits(lambda _s: 100.0, 1_000 + 2 * iv, sess, bar_iv=iv) == 0
    assert eng.simulate_paper_exits(lambda _s: 100.0, 1_000 + 3 * iv, sess, bar_iv=iv) == 1
    assert sess.trades[0].exit_reason == "time_stop"


def test_rebalance_closes_ghost_leg_with_no_current_bar() -> None:
    """REGRESSION (C2): a held leg whose symbol has NO bar at a rebalance (halted/delisted/feed gap
    over a multi-day run) must still be flattened on its last known close — not left as a ghost leg
    that never closes and keeps accruing funding against a frozen mark."""
    from src.backtest.engine import BacktestResult
    from src.backtest.portfolio import CrossSectionalEngine, _Leg

    cfg = load_backtest_config()
    meta = load_metadata_config()
    sc = load_strategies_config()
    strat = build_strategy(sc.candidate("funding_carry"), sc.strategy_version)
    eng = CrossSectionalEngine(cfg, meta, strat)
    eng._grid_iv = 3_600_000  # used by the close's bars_held math

    sym = "BTC/USDT:USDT"
    leg = _Leg(symbol=sym, side=1, qty=1.0, entry_ts=0, entry_price=100.0, notional=100.0,
               risk_amount=1.0, entry_fee=0.0, funding=0.0, slippage_cost=0.0, regime="R1",
               session=0, last_funding_ts=0)
    holdings = {sym: leg}
    by_symbol = {sym: SymbolInput(
        symbol=sym,
        bars=[{"ts": 0, "open": 100.0, "high": 100.0, "low": 100.0, "close": 105.0, "volume": 1.0}],
        frame=FeatureFrame(symbol=sym, timeframe="1h", feature_names=[], rows=[]),
        spread_samples=[], funding_events=[],
    )}
    result = BacktestResult(initial_equity=10_000.0, symbols=[sym])
    # empty scores → the symbol is no longer a target → must close; bars_by_ts has NO bar at ts,
    # exercising the last-known-close fallback.
    eng._rebalance(holdings, {}, {sym: {}}, {sym: {}}, by_symbol, 10 * 3_600_000, 10_000.0, result)

    assert sym not in holdings  # ghost leg flattened, not left open
    assert any(t.symbol == sym for t in result.trades)  # closed on the last known bar


def test_funding_settled_by_timestamp_survives_event_list_rebuild() -> None:
    """REGRESSION (C1): funding is settled by TIMESTAMP so the live loop rebuilding/sliding the
    funding_events list between ticks never skips or double-charges carry — a positional index into
    that list mis-pointed once the window slid (funding_carry's entire edge). Each funding event is
    charged exactly once even as the list is replaced and old events drop off the front."""
    from src.backtest.portfolio import CrossSectionalEngine, _Leg

    cfg = load_backtest_config()
    meta = load_metadata_config()
    sc = load_strategies_config()
    strat = build_strategy(sc.candidate("funding_carry"), sc.strategy_version)
    eng = CrossSectionalEngine(cfg, meta, strat)

    sym = "BTC/USDT:USDT"
    H = 3_600_000
    leg = _Leg(symbol=sym, side=1, qty=1.0, entry_ts=0, entry_price=100.0, notional=100.0,
               risk_amount=1.0, entry_fee=0.0, funding=0.0, slippage_cost=0.0, regime="R1",
               session=0, last_funding_ts=0)

    def _sin(events):
        return SymbolInput(symbol=sym, bars=[],
                           frame=FeatureFrame(symbol=sym, timeframe="1h", feature_names=[], rows=[]),
                           spread_samples=[], funding_events=events)

    one = eng.funding.payment(side=1, notional=100.0, funding_rate=0.001)

    # tick 1: events at 8h and 16h are in the window; both settle (ts <= 16h, after entry).
    leg.last_funding_ts = eng._charge_funding(
        leg, _sin([{"ts": 8 * H, "funding_rate": 0.001}, {"ts": 16 * H, "funding_rate": 0.001}]),
        16 * H,
    )
    assert leg.funding == 2 * one  # both charged once

    # tick 2: list REBUILT + SLID — the 8h event dropped off the front, a new 24h event appeared.
    # By timestamp only the 24h event is new: 16h is NOT double-charged, the dropped 8h is NOT lost.
    leg.last_funding_ts = eng._charge_funding(
        leg, _sin([{"ts": 16 * H, "funding_rate": 0.001}, {"ts": 24 * H, "funding_rate": 0.001}]),
        24 * H,
    )
    assert leg.funding == 3 * one  # exactly one more (24h), not two


def test_loss_on_first_bar_of_new_day_counts_toward_daily_window() -> None:
    """REGRESSION (R3): a position closed at a LOSS on the first bar of a new UTC day must count
    toward that day's daily-loss breaker. The window roll runs BEFORE simulate_paper_exits books the
    exit, so the loss isn't snapshotted out of the new day's window (it previously escaped both
    yesterday's closed window and today's freshly-snapshotted one)."""
    from src.paper.engine import PaperTradingEngine
    from src.paper.session import PaperSession, PaperTrade
    from src.risk.portfolio import Position

    DAY = 86_400_000
    eng = PaperTradingEngine()
    sess = PaperSession(session_id="t")
    eng._roll_loss_windows(5 * DAY + 1000)  # establish the prior day's window at realized=0
    assert eng._day_start_realized == 0.0

    sym = "ETH/USDT:USDT"
    eng._open_positions[sym] = Position(symbol=sym, side=1, qty=2.0, entry_price=100.0,
                                        risk_amount=10.0, beta_to_btc=1.0, regime="R1")
    eng._position_meta[sym] = ("lead_lag_xasset", 5 * DAY)
    eng._exit_levels[sym] = (95.0, 110.0, 0, 5 * DAY)  # stop 95, tp 110
    sess.trades.append(PaperTrade(
        trade_id="e", symbol=sym, strategy="lead_lag_xasset", side=1, qty=2.0, entry_price=100.0,
        stop_price=95.0, tp_price=110.0, regime="R1", session=0, decision_ts=5 * DAY,
        entry_ts=5 * DAY, exit_ts=5 * DAY, exit_price=100.0, exit_reason="open", fee=0.2,
        slippage_cost=0.0, pnl=-0.2, pnl_r=0.0, has_exchange_side_stop=True,
        execution_route="maker", spread_bps_at_entry=0.0, slippage_frac=0.0,
    ))

    new_day_ts = 6 * DAY + 1000  # first bar of the NEXT UTC day; price gaps below the stop
    assert eng.simulate_paper_exits(lambda _s: 90.0, new_day_ts, sess) == 1
    assert eng._day_key == new_day_ts // DAY  # rolled to the new day
    assert eng._day_start_realized == 0.0  # snapshot taken BEFORE booking the loss
    # today's daily-window P&L = realized - day_start = the loss → it counts (didn't escape)
    assert eng._realized_pnl < 0 and (eng._realized_pnl - eng._day_start_realized) < 0


def test_basket_loop_seeds_paper_base_equity() -> None:
    """The PAPER basket loop must seed at the shared paper base (so its $ P&L / equity curve line up
    with the per-symbol engine + dashboard), while the default keeps the config numeraire."""
    from src.paper.session import PAPER_BASE_EQUITY

    cfg = load_backtest_config()
    meta = load_metadata_config()
    strat = build_strategy(load_strategies_config().candidate("funding_carry"),
                           load_strategies_config().strategy_version)
    sess = PaperSession(session_id="t")
    paper = BasketPaperLoop(cfg, meta, strat, bar_interval_ms=IV, session=sess,
                            initial_equity=PAPER_BASE_EQUITY)
    assert paper._equity == PAPER_BASE_EQUITY == 10_000.0
    default = BasketPaperLoop(cfg, meta, strat, bar_interval_ms=IV, session=PaperSession("t2"))
    assert default._equity == cfg.account.initial_equity  # backtest numeraire unchanged


def test_basket_halt_check_honours_global_kill_switch(tmp_path) -> None:
    """The basket loop's stop condition must trip on the GLOBAL kill switch (emergency halt), not
    just the caller's Stop/job-cancel — or a kill switch would stop directional trading but leave
    the carry/factor baskets running. Isolated file-backed switch (unreachable redis)."""
    iso = Settings(
        _env_file=None, app_env="paper",
        data_lake_path=tmp_path / "dl", redis_url="redis://127.0.0.1:1/0",
    )
    kill = KillSwitch(iso)
    kill.disengage()  # clean slate

    # caller never asks to stop; only the kill switch governs.
    halt = _halt_check(lambda: False, kill)
    assert halt() is False  # not engaged → keep running
    kill.engage(reason="test")
    assert halt() is True  # engaged → halt the basket loop
    kill.disengage()
    assert halt() is False

    # caller's own stop still works independently of the kill switch.
    assert _halt_check(lambda: True, kill)() is True
    assert _halt_check(None, kill)() is False
