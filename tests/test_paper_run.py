"""Paper-run consumer tests (Section 26): a paper session sources candidates ONLY from PROMOTED
strategies (closing the research→paper link) and persists its trades for the dashboard."""

from __future__ import annotations

import uuid

from sqlalchemy import select
from src.config import get_settings
from src.db.base import session_scope
from src.db.models import PaperRun, PaperTradeRecord
from src.paper.run import build_promoted_inputs, run_paper_session
from src.strategies.promotion import persist_validations
from src.strategies.research import CandidateValidation, SideDecision

from tests.conftest import requires_db


def _promote(candidate_id: str, version: str) -> None:
    sd = SideDecision(
        allow_long=True,
        allow_short=True,
        long_expectancy_r=0.2,
        short_expectancy_r=0.1,
        long_trades=30,
        short_trades=20,
        disabled=[],
    )
    persist_validations(
        [
            CandidateValidation(
                candidate_id=candidate_id,
                family="B",
                strategy_version=version,
                promoted=True,
                status="promoted",
                shelved_reasons=[],
                side_decision=sd,
                hypothesis={},
                report={"expectancy_r": 0.2},
                walk_forward={},
                fee_stress={},
                slippage_stress={},
                noise_control={},
            )
        ]
    )


@requires_db
def test_incremental_persist_writes_trades_without_report_file() -> None:
    """Incremental persistence (write_report/write_logs=False) upserts the run + trade rows so a
    long-running live session is visible mid-run and survives a worker restart, but does NOT dump a
    fresh timestamped report file each call (which would flood the lake over a multi-day run). The
    final persist writes the report once."""
    from src.paper.report import build_paper_report
    from src.paper.run import persist_paper_session
    from src.paper.session import PaperSession, PaperTrade

    sid = f"paper:incr:{uuid.uuid4().hex[:8]}"
    sess = PaperSession(session_id=sid)
    sess.trades.append(PaperTrade(
        trade_id="a1", symbol="ETH/USDT:USDT", strategy="lead_lag_xasset", side=1, qty=1.0,
        entry_price=100.0, stop_price=95.0, tp_price=110.0, regime="R1", session=0,
        decision_ts=1, entry_ts=1, exit_ts=2, exit_price=110.0, exit_reason="take_profit",
        fee=0.1, slippage_cost=0.0, pnl=9.9, pnl_r=1.0, has_exchange_side_stop=True,
        execution_route="taker", spread_bps_at_entry=0.0, slippage_frac=0.0,
    ))
    try:
        persist_paper_session(sess, write_report=False, write_logs=False)  # incremental flush
        with session_scope() as s:
            run = s.execute(select(PaperRun).where(PaperRun.session_id == sid)).scalars().first()
            assert run is not None and not run.report_path  # no report file dumped yet
            rows = s.execute(
                select(PaperTradeRecord).where(PaperTradeRecord.session_id == sid)
            ).scalars().all()
            assert len(rows) == 1 and rows[0].exit_reason == "take_profit"  # trade visible mid-run
        persist_paper_session(sess, build_paper_report(sess))  # final persist
        with session_scope() as s:
            run = s.execute(select(PaperRun).where(PaperRun.session_id == sid)).scalars().first()
            assert run.report_path  # report written once at the end
    finally:
        with session_scope() as s:
            for old in s.execute(
                select(PaperTradeRecord).where(PaperTradeRecord.session_id == sid)
            ).scalars().all():
                s.delete(old)
            run = s.execute(select(PaperRun).where(PaperRun.session_id == sid)).scalars().first()
            if run is not None:
                s.delete(run)


@requires_db
def test_build_promoted_inputs_only_promoted_and_approved() -> None:
    ver = get_settings().strategy_version
    strat = f"paper_strat_{uuid.uuid4().hex[:6]}"
    _promote(strat, ver)

    inputs = build_promoted_inputs(ver)
    strategies = {inp.candidate.strategy for inp in inputs}
    assert strat in strategies  # promoted strategy is sourced
    # Every sourced candidate is config-live-approved BECAUSE its strategy is promoted
    # (the flag comes from the registry, not a hardcoded True).
    assert all(inp.candidate.config_live_approved for inp in inputs)


@requires_db
def test_run_paper_session_persists_trades() -> None:
    ver = get_settings().strategy_version
    _promote(f"paper_strat_{uuid.uuid4().hex[:6]}", ver)

    session, _report, sid = run_paper_session(session_name=f"test_{uuid.uuid4().hex[:6]}")
    with session_scope() as db:
        run = db.execute(select(PaperRun).where(PaperRun.session_id == sid)).scalars().first()
        trades = (
            db.execute(select(PaperTradeRecord).where(PaperTradeRecord.session_id == sid))
            .scalars()
            .all()
        )
    assert run is not None
    assert run.executed_count == len(trades)
    assert session.executed_count > 0  # promoted strategies produced executed paper trades
