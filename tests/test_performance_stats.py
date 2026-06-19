"""M5: realized trading-performance stats from paper_trades (the TradeZella backbone)."""

from __future__ import annotations

import pytest
from src.api.stats import compute_trading_stats, resolve_window
from src.db.base import session_scope
from src.db.models import PaperTradeRecord

_SID = "perf_test_session"
_STRAT = "perf_test_strategy"


@pytest.fixture
def seeded_trades():
    pnls = [100.0, 50.0, -30.0, -20.0]
    pnl_rs = [1.0, 0.5, -0.3, -0.2]
    regimes = ["trend_up", "range", "trend_up", "range"]
    symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT", "BTC/USDT:USDT", "ETH/USDT:USDT"]
    with session_scope() as s:
        s.query(PaperTradeRecord).filter_by(session_id=_SID).delete()
        for i, (pnl, r, reg, sym) in enumerate(zip(pnls, pnl_rs, regimes, symbols, strict=True)):
            s.add(
                PaperTradeRecord(
                    session_id=_SID,
                    trade_id=f"pt_{i}",
                    symbol=sym,
                    strategy=_STRAT,
                    side=1,
                    pnl=pnl,
                    pnl_r=r,
                    fee=1.0,
                    slippage_cost=0.5,
                    regime=reg,
                )
            )
    yield
    with session_scope() as s:
        s.query(PaperTradeRecord).filter_by(session_id=_SID).delete()


def test_trading_stats_core_metrics(seeded_trades) -> None:
    t = compute_trading_stats(resolve_window("all", None, None), strategy=_STRAT)
    assert t.total_trades == 4
    assert t.winning_trades == 2 and t.losing_trades == 2
    assert t.win_rate == 0.5
    assert t.realized_pnl == 100.0
    assert t.gross_win == 150.0 and t.gross_loss == -50.0
    assert t.profit_factor == 3.0
    assert t.expectancy_r == pytest.approx(0.25)
    assert t.total_fees_paid == 4.0
    assert t.avg_win == 75.0 and t.avg_loss == -25.0
    assert t.largest_win == 100.0 and t.largest_loss == -30.0


def test_trading_stats_equity_curve_and_drawdown(seeded_trades) -> None:
    t = compute_trading_stats(resolve_window("all", None, None), strategy=_STRAT)
    # base 10_000 → +100 → +50 → -30 → -20 (5 points incl. the base).
    assert t.equity_curve[0] == 10_000.0
    assert t.equity_curve[-1] == pytest.approx(10_100.0)
    assert len(t.equity_curve) == 5
    # peak 10_150 then down to 10_100 → drawdown 50/10_150.
    assert t.max_drawdown_pct == pytest.approx(50.0 / 10_150.0, abs=1e-4)


def test_trading_stats_breakdowns(seeded_trades) -> None:
    t = compute_trading_stats(resolve_window("all", None, None), strategy=_STRAT)
    by_sym = {b["group"]: b for b in t.by_symbol}
    assert by_sym["BTC/USDT:USDT"]["pnl"] == 70.0  # +100 - 30
    assert by_sym["ETH/USDT:USDT"]["pnl"] == 30.0  # +50 - 20
    by_reg = {b["group"]: b for b in t.by_regime}
    assert set(by_reg) == {"trend_up", "range"}
    # sorted by pnl desc
    assert [b["group"] for b in t.by_symbol] == ["BTC/USDT:USDT", "ETH/USDT:USDT"]


def test_trading_stats_empty_window_is_zero_safe(seeded_trades) -> None:
    t = compute_trading_stats(resolve_window("all", None, None), strategy="no_such_strategy")
    assert t.total_trades == 0
    assert t.win_rate == 0.0 and t.profit_factor == 0.0
    assert t.equity_curve == []


def test_trade_series_pairs_each_trade(seeded_trades) -> None:
    t = compute_trading_stats(resolve_window("all", None, None), strategy=_STRAT)
    assert len(t.trade_series) == 4
    assert all(len(p) == 2 for p in t.trade_series)  # (epoch_ms, pnl)
    assert [round(p[1], 2) for p in t.trade_series] == [100.0, 50.0, -30.0, -20.0]


@pytest.fixture
def seeded_envs():
    """One paper-tagged and one demo-tagged session, to prove environment separation."""
    rows = [("paper_env_sess", "PX"), ("demo:env_sess", "DM")]
    with session_scope() as s:
        for sid, _ in rows:
            s.query(PaperTradeRecord).filter_by(session_id=sid).delete()
        for sid, strat in rows:
            for i in range(3):
                s.add(
                    PaperTradeRecord(
                        session_id=sid, trade_id=f"{sid}_{i}", symbol="BTC/USDT:USDT",
                        strategy=strat, side=1, pnl=10.0, pnl_r=0.1, fee=0.0, slippage_cost=0.0,
                        regime="range",
                    )
                )
    yield
    with session_scope() as s:
        for sid, _ in rows:
            s.query(PaperTradeRecord).filter_by(session_id=sid).delete()


def test_env_scoping_separates_demo_from_paper(seeded_envs) -> None:
    win = resolve_window("all", None, None)
    demo = compute_trading_stats(win, env="demo", strategy="DM")
    paper = compute_trading_stats(win, env="paper", strategy="DM")
    assert demo.total_trades == 3  # demo: session counted under demo
    assert paper.total_trades == 0  # ...and NOT under paper
    # paper env excludes demo-tagged sessions
    assert compute_trading_stats(win, env="paper", strategy="PX").total_trades == 3
    assert compute_trading_stats(win, env="demo", strategy="PX").total_trades == 0


def test_get_environment_summary_and_traded_symbols(seeded_envs) -> None:
    from src.api.stats import get_environment_summary, get_traded_symbols

    summary = {e["env"]: e for e in get_environment_summary()}
    assert set(summary) == {"paper", "demo", "testnet", "live"}
    assert summary["demo"]["trades"] >= 3
    assert "BTC/USDT:USDT" in get_traded_symbols("demo")


@pytest.fixture
def seeded_junk():
    """A real-strategy trade and a stale/test-strategy trade, to prove the selector filters."""
    rows = [("scopes_real", "basis_reversion"), ("scopes_junk", "paper_strat_zzz999")]
    with session_scope() as s:
        for sid, _ in rows:
            s.query(PaperTradeRecord).filter_by(session_id=sid).delete()
        for sid, strat in rows:
            s.add(
                PaperTradeRecord(
                    session_id=sid, trade_id=f"{sid}_0", symbol="BTC/USDT:USDT", strategy=strat,
                    side=1, pnl=1.0, pnl_r=0.1, fee=0.0, slippage_cost=0.0, regime="range",
                )
            )
    yield
    with session_scope() as s:
        for sid, _ in rows:
            s.query(PaperTradeRecord).filter_by(session_id=sid).delete()


def test_get_trade_scopes_excludes_stale_strategies(seeded_junk) -> None:
    from src.api.stats import get_trade_scopes

    scopes = get_trade_scopes()
    assert "basis_reversion" in scopes["strategies"]  # real config candidate shown
    assert "paper_strat_zzz999" not in scopes["strategies"]  # stale/test name filtered out
    assert "scopes_junk" not in scopes["sessions"]  # session of only-junk strategies filtered
