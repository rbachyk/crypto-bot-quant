"""Unit tests for the SHARED bracket-exit geometry (src/brackets.py).

This is the single source of truth the backtest engine, the lake-replay walk, and the paper/realtime
exit simulator all call, so it is tested directly and exhaustively — a drift here would silently
change every execution mode at once. Also proves the paper simulator's intrabar-vs-close-only fill
selection, which is what closes the realtime paper↔backtest parity gap (audit item a).
"""

from __future__ import annotations

from src.brackets import effective_stop, resolve_bracket_exit


class TestEffectiveStop:
    def test_no_trail_returns_fixed_stop(self) -> None:
        assert effective_stop(1, stop=90.0, peak=120.0, trail_dist=0.0) == 90.0
        assert effective_stop(-1, stop=110.0, peak=80.0, trail_dist=0.0) == 110.0

    def test_long_trail_ratchets_up_never_below_fixed_stop(self) -> None:
        # peak 120, trail 8 -> 112 (above the fixed 90) => trailing stop tightened up.
        assert effective_stop(1, stop=90.0, peak=120.0, trail_dist=8.0) == 112.0
        # peak only 95, trail 8 -> 87 < 90, so the fixed stop still governs.
        assert effective_stop(1, stop=90.0, peak=95.0, trail_dist=8.0) == 90.0

    def test_short_trail_ratchets_down_never_above_fixed_stop(self) -> None:
        assert effective_stop(-1, stop=110.0, peak=80.0, trail_dist=8.0) == 88.0
        assert effective_stop(-1, stop=110.0, peak=105.0, trail_dist=8.0) == 110.0

    def test_zero_stop_disables_trail(self) -> None:
        # No fixed stop set -> trail is not armed (matches the paper guard).
        assert effective_stop(1, stop=0.0, peak=120.0, trail_dist=8.0) == 0.0


class TestResolveBracketExit:
    def test_long_hard_stop(self) -> None:
        reason, level = resolve_bracket_exit(
            1, high=101.0, low=89.0, stop=90.0, tp=115.0, peak=100.0, trail_dist=0.0
        )
        assert reason == "stop" and level == 90.0

    def test_long_take_profit(self) -> None:
        reason, level = resolve_bracket_exit(
            1, high=116.0, low=99.0, stop=90.0, tp=115.0, peak=100.0, trail_dist=0.0
        )
        assert reason == "take_profit" and level == 115.0

    def test_stop_checked_before_take_profit_when_bar_spans_both(self) -> None:
        # A bar whose range touches BOTH stop and TP must resolve to the stop (conservative).
        reason, level = resolve_bracket_exit(
            1, high=116.0, low=89.0, stop=90.0, tp=115.0, peak=100.0, trail_dist=0.0
        )
        assert reason == "stop" and level == 90.0

    def test_long_trailing_stop_labelled_distinctly(self) -> None:
        # peak 120, trail 8 -> eff 112; a low of 111 trips the trailed stop (not the fixed 90).
        reason, level = resolve_bracket_exit(
            1, high=113.0, low=111.0, stop=90.0, tp=200.0, peak=120.0, trail_dist=8.0
        )
        assert reason == "trailing_stop" and level == 112.0

    def test_short_hard_stop_and_tp(self) -> None:
        reason, level = resolve_bracket_exit(
            -1, high=111.0, low=100.0, stop=110.0, tp=85.0, peak=100.0, trail_dist=0.0
        )
        assert reason == "stop" and level == 110.0
        reason, level = resolve_bracket_exit(
            -1, high=101.0, low=84.0, stop=110.0, tp=85.0, peak=100.0, trail_dist=0.0
        )
        assert reason == "take_profit" and level == 85.0

    def test_no_fixed_tp_sentinel_never_take_profits(self) -> None:
        # tp <= 0 = "no fixed TP" (momentum). Even a huge favorable bar does not take profit.
        reason, _ = resolve_bracket_exit(
            1, high=500.0, low=99.0, stop=90.0, tp=0.0, peak=100.0, trail_dist=0.0
        )
        assert reason is None

    def test_survives_when_nothing_breached(self) -> None:
        reason, level = resolve_bracket_exit(
            1, high=105.0, low=96.0, stop=90.0, tp=115.0, peak=100.0, trail_dist=0.0
        )
        assert reason is None and level == 0.0


class TestPaperExitIntrabarVsCloseOnly:
    """simulate_paper_exits fills at the exact bracket level when given the bar's wick (hl_of), and
    is close-only otherwise — the realtime parity fix (a)."""

    def _engine_with_open_long(self, entry: float, stop: float):
        from src.paper.engine import PaperTradingEngine
        from src.paper.session import PaperTrade
        from src.risk.portfolio import Position

        eng = PaperTradingEngine()
        session = eng.new_session("t")
        sym = "BTC/USDT:USDT"
        eng._open_positions[sym] = Position(
            symbol=sym, side=1, qty=1.0, entry_price=entry, risk_amount=entry - stop,
            beta_to_btc=1.0, regime="trend",
        )
        eng._exit_levels[sym] = (stop, 0.0, 0, 0)  # fixed stop, no TP, no time-stop
        eng._trail_dist[sym] = 0.0
        eng._peak[sym] = entry
        eng._position_funding[sym] = 0.0
        session.trades.append(PaperTrade(
            trade_id="t1", symbol=sym, strategy="s", side=1, qty=1.0, entry_price=entry,
            stop_price=stop, tp_price=0.0, regime="trend", session=0, decision_ts=0, entry_ts=0,
            exit_ts=0, exit_price=entry, exit_reason="open", fee=0.0, slippage_cost=0.0, pnl=0.0,
            pnl_r=0.0, has_exchange_side_stop=True, execution_route="taker",
            spread_bps_at_entry=2.0, slippage_frac=0.0,
        ))
        return eng, session, sym

    def test_intrabar_wick_exits_at_stop(self) -> None:
        # Bar wicks to 89 (below the 90 stop) but CLOSES at 101 (above). With hl_of the stop fills.
        eng, session, sym = self._engine_with_open_long(entry=100.0, stop=90.0)
        closed = eng.simulate_paper_exits(
            lambda _s: 101.0, now_ts=1000, session=session, hl_of=lambda _s: (101.0, 89.0)
        )
        assert closed == 1
        assert sym not in eng._open_positions
        trade = session.trades[-1]
        assert trade.exit_reason == "stop"
        assert trade.exit_price == 90.0  # filled at the bracket level, not the close

    def test_close_only_holds_through_the_same_wick(self) -> None:
        # Same bar, but WITHOUT hl_of: the close (101, above the stop) is used -> position survives.
        eng, session, sym = self._engine_with_open_long(entry=100.0, stop=90.0)
        closed = eng.simulate_paper_exits(lambda _s: 101.0, now_ts=1000, session=session)
        assert closed == 0
        assert sym in eng._open_positions
        assert session.trades[-1].exit_reason == "open"
