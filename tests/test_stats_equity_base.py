"""The dashboard's equity curve + drawdown must be measured against the capital that traded.

Every environment used to be reconstructed from a hardcoded 10 000 notional base. A demo basket
sizes off the REAL account (a ~1 000 account, sliced per strategy), so a 10% real drawdown rendered
as ~1% and "current equity" described no account that exists — and drawdown is precisely the number
an operator reads to decide whether to let a session keep running.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select
from src.api.stats import compute_trading_stats, resolve_window
from src.db.base import session_scope
from src.db.models import PaperRun, PaperTradeRecord
from src.paper.run import persist_paper_session
from src.paper.session import PAPER_BASE_EQUITY, PaperSession, PaperTrade

from tests.conftest import requires_db


def _trade(pnl: float, trade_id: str) -> PaperTrade:
    return PaperTrade(
        trade_id=trade_id, symbol="BTC/USDT:USDT", strategy="funding_carry", side=1, qty=1.0,
        entry_price=100.0, stop_price=95.0, tp_price=110.0, regime="R1", session=0,
        decision_ts=1, entry_ts=1, exit_ts=2, exit_price=100.0 + pnl, exit_reason="rebalance",
        fee=0.0, slippage_cost=0.0, pnl=pnl, pnl_r=pnl / 5.0, has_exchange_side_stop=False,
        execution_route="taker", spread_bps_at_entry=0.0, slippage_frac=0.0,
    )


def _cleanup(sid: str) -> None:
    with session_scope() as s:
        s.query(PaperTradeRecord).filter_by(session_id=sid).delete()
        run = s.execute(select(PaperRun).where(PaperRun.session_id == sid)).scalars().first()
        if run is not None:
            s.delete(run)


@requires_db
def test_drawdown_is_measured_against_the_real_account_not_the_paper_base() -> None:
    """A 100 loss on a 1 000 demo account is a 10% drawdown — not the ~1% a 10k base reports."""
    sid = f"demo:basket:funding_carry:d:1h:{uuid.uuid4().hex[:8]}"
    sess = PaperSession(session_id=sid, initial_equity=1_000.0)
    sess.trades.append(_trade(-100.0, "a1"))
    try:
        persist_paper_session(sess, write_report=False, write_logs=False)

        with session_scope() as s:
            run = s.execute(select(PaperRun).where(PaperRun.session_id == sid)).scalars().first()
            assert run.initial_equity == pytest.approx(1_000.0)  # stamped at persist

        st = compute_trading_stats(resolve_window("all", None, None), session_id=sid)
        assert st.max_drawdown_pct == pytest.approx(0.10, abs=1e-6)
        assert st.current_equity == pytest.approx(900.0)
    finally:
        _cleanup(sid)


@requires_db
def test_a_session_with_no_recorded_base_still_renders_on_the_paper_numeraire() -> None:
    """Rows written before paper_runs.initial_equity existed have no base to read; those curves
    must render exactly as they always did rather than collapsing to zero."""
    sid = f"paper:legacy:{uuid.uuid4().hex[:8]}"
    sess = PaperSession(session_id=sid)
    sess.trades.append(_trade(-100.0, "b1"))
    try:
        persist_paper_session(sess, write_report=False, write_logs=False)
        with session_scope() as s:  # simulate a pre-migration row
            run = s.execute(select(PaperRun).where(PaperRun.session_id == sid)).scalars().first()
            run.initial_equity = None

        st = compute_trading_stats(resolve_window("all", None, None), session_id=sid)
        assert st.current_equity == pytest.approx(PAPER_BASE_EQUITY - 100.0)
        assert st.max_drawdown_pct == pytest.approx(100.0 / PAPER_BASE_EQUITY, abs=1e-6)
    finally:
        _cleanup(sid)
