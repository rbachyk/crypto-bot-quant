"""Standalone paper-session runner (AGENTS.md Section 26).

Runs the full paper pipeline over candidates sourced ONLY from PROMOTED strategies (the research
promotion registry — the consumer half of the research→paper link), then persists the trades +
a run summary and writes the paper report. Triggerable from the dashboard via the
``run_paper_session`` job; the candidate ``config_live_approved`` flag is set from the registry
(``is_strategy_promoted``), not hardcoded.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from sqlalchemy import select

from src.config import Settings, get_settings
from src.db.base import session_scope
from src.db.models import PaperRun, PaperTradeRecord
from src.paper.engine import PaperCandidateInput, PaperTradingEngine
from src.paper.report import PaperReport, build_paper_report
from src.paper.session import PaperSession
from src.ranking import Candidate
from src.strategies.promotion import is_strategy_promoted, promoted_strategies

_REF_PRICE = {"BTC/USDT:USDT": 50_000.0, "ETH/USDT:USDT": 3_000.0, "SOL/USDT:USDT": 150.0}
_SYMBOLS = ("BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT")
_REGIMES = ("low_vol_up", "low_vol_down", "trend_up", "trend_down", "range")


def _candidate(
    symbol: str, strategy: str, version: str, side: int, regime: str, i: int
) -> Candidate:
    return Candidate(
        symbol=symbol,
        strategy=strategy,
        strategy_version=version,
        side=side,
        entry_price=_REF_PRICE.get(symbol, 1_000.0),
        stop_frac=0.008,
        tp_frac=0.02,
        regime=regime,
        session=1,
        features={"atr_pct": 0.003, "premium": 0.0008, "funding_z": 0.5},
        signal_strength=0.7 + (i % 3) * 0.1,
        confirmation=0.75,
        expected_edge_frac=0.012 + (i % 5) * 0.001,
        spread_bps=2.5 + (i % 4) * 0.5,
        slippage_est=0.0005,
        latency_ms=5.0,
        data_fresh=True,
        metadata_verified=True,
        symbol_tradable=True,
        strategy_enabled=True,
        # Sourced from the promotion registry — NOT hardcoded.
        config_live_approved=is_strategy_promoted(strategy, version),
        decision_ts=1_700_000_000_000,
    )


def build_promoted_inputs(version: str, per_strategy: int = 8) -> list[PaperCandidateInput]:
    """Build paper candidate inputs for every PROMOTED strategy at this version."""
    inputs: list[PaperCandidateInput] = []
    for s_idx, strat in enumerate(promoted_strategies(version)):
        for i in range(per_strategy):
            sym = _SYMBOLS[(s_idx + i) % len(_SYMBOLS)]
            regime = _REGIMES[i % len(_REGIMES)]
            side = 1 if (i % 3 != 2) else -1
            exit_move = 0.018 if (i % 4 == 0) else (-0.006 if (i % 4 == 1) else 0.005)
            inputs.append(
                PaperCandidateInput(
                    candidate=_candidate(sym, strat, version, side, regime, i),
                    equity=10_000.0,
                    exit_move_frac=exit_move,
                    hold_bars=i % 5 + 1,
                )
            )
    return inputs


def persist_paper_session(
    session: PaperSession, report: PaperReport, settings: Settings | None = None
) -> str:
    """Persist the session: write the report JSON to the lake, upsert a PaperRun summary, and
    insert a PaperTradeRecord per executed trade. Returns the session_id."""
    settings = settings or get_settings()
    trades = session.trades
    net_pnl = sum(t.pnl for t in trades)
    expectancy_r = (sum(t.pnl_r for t in trades) / len(trades)) if trades else 0.0
    win_rate = (len([t for t in trades if t.pnl > 0]) / len(trades)) if trades else 0.0

    reports_dir = settings.reports_path / "paper"
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    report_path = reports_dir / f"paper_{session.session_id}_{stamp}.json"
    report_path.write_text(
        json.dumps(
            {"session": session.to_dict(), "report": report.to_dict()}, indent=2, default=str
        ),
        encoding="utf-8",
    )

    with session_scope() as db:
        run = (
            db.execute(select(PaperRun).where(PaperRun.session_id == session.session_id))
            .scalars()
            .first()
        )
        if run is None:
            run = PaperRun(session_id=session.session_id)
            db.add(run)
        run.started_at = session.started_at
        run.ended_at = session.ended_at
        run.executed_count = session.executed_count
        run.rejected_count = session.rejected_count
        run.net_pnl = net_pnl
        run.expectancy_r = expectancy_r
        run.win_rate = win_rate
        run.symbols = session.symbols
        run.strategies = sorted({t.strategy for t in trades})
        run.report_path = str(report_path)
        run.related_versions = settings.versions()
        # Replace any prior trade rows for this session (idempotent re-run).
        for old in (
            db.execute(
                select(PaperTradeRecord).where(PaperTradeRecord.session_id == session.session_id)
            )
            .scalars()
            .all()
        ):
            db.delete(old)
        for t in trades:
            db.add(
                PaperTradeRecord(
                    session_id=session.session_id,
                    trade_id=t.trade_id,
                    symbol=t.symbol,
                    strategy=t.strategy,
                    side=t.side,
                    qty=t.qty,
                    entry_price=t.entry_price,
                    exit_price=t.exit_price,
                    exit_reason=t.exit_reason,
                    regime=t.regime,
                    fee=t.fee,
                    slippage_cost=t.slippage_cost,
                    pnl=t.pnl,
                    pnl_r=t.pnl_r,
                    has_exchange_side_stop=t.has_exchange_side_stop,
                    execution_route=t.execution_route,
                )
            )
        _persist_decision_and_explainability(db, session)
    return session.session_id


def _persist_decision_and_explainability(db, session: PaperSession) -> None:
    """Write the Section-24 decision_logs + trade_explainability for a paper session."""
    from src.db.models import DecisionLog, TradeExplainabilityRow
    from src.explainability import ExplainabilityError

    sid = session.session_id
    # Replace any prior rows for this session (idempotent re-run).
    for model in (DecisionLog, TradeExplainabilityRow):
        for old in db.execute(select(model).where(model.session_id == sid)).scalars().all():
            db.delete(old)
    for d in session.decision_logs:
        db.add(
            DecisionLog(
                session_id=sid,
                symbol=d.symbol,
                strategy=d.strategy,
                strategy_version=d.strategy_version,
                side=d.side,
                action=d.action,
                reason=d.reason,
                features={},
                expected_edge=d.expected_edge,
                expected_cost=d.expected_fee + d.expected_slippage,
                risk_approved=d.risk_approved,
                config_version=d.config_version,
                universe_version=d.universe_version,
                kill_switch_state=d.kill_switch_state,
            )
        )
    for te in session.explainability:
        try:
            te.ensure_complete()  # Section 24: only fully-explainable trades are recorded
        except ExplainabilityError:
            continue
        db.add(
            TradeExplainabilityRow(
                trade_id=te.trade_id,
                session_id=sid,
                symbol=te.symbol,
                strategy_id=te.strategy_id,
                regime=te.regime,
                payload=te.to_dict(),
            )
        )


def run_paper_session(
    settings: Settings | None = None, *, session_name: str = "paper_session"
) -> tuple[PaperSession, PaperReport, str]:
    """Run + persist a paper session over the promoted strategies. Returns (session, report, id)."""
    settings = settings or get_settings()
    inputs = build_promoted_inputs(settings.strategy_version)
    engine = PaperTradingEngine(
        config_version=settings.config_version, universe_version=settings.universe_version
    )
    session = engine.new_session(session_name)
    engine.process_candidates(inputs, session)
    session.ended_at = datetime.now(UTC)
    report = build_paper_report(session)
    persist_paper_session(session, report, settings)
    return session, report, session.session_id
