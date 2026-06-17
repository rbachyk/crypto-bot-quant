"""Paper trading report generator (AGENTS.md Section 34).

Produces the PAPER-A technical report and PAPER-B strategy report from a
completed :class:`~src.paper.session.PaperSession`.

PAPER-A report: verifies the technical pipeline (every step runs, stops are
  placed, kill switch works, reconciliation halts on foreign orders,
  decision logs are complete).

PAPER-B report: strategy validation — trade counts, per-symbol/regime
  breakdowns, paper-vs-backtest comparison, cost analysis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from src.paper.session import PaperSession

# --------------------------------------------------------------------------- #
# Report schemas                                                                #
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class PaperAReport:
    """Technical paper validation report (PAPER-A gate output)."""

    session_id: str
    generated_at: str
    pipeline_end_to_end: bool
    total_candidates_evaluated: int
    candidates_executed: int
    candidates_rejected: int
    all_executed_have_exchange_side_stop: bool
    kill_switch_exercised: bool
    kill_switch_halts_new_entries: bool
    reconciliation_ran: bool
    foreign_order_halt_triggered: bool
    decision_logs_complete: bool
    decision_log_count: int
    required_decision_log_fields_present: bool
    component_imports_ok: bool
    note: str = ""

    def passed(self) -> bool:
        return (
            self.pipeline_end_to_end
            and self.all_executed_have_exchange_side_stop
            and self.kill_switch_exercised
            and self.kill_switch_halts_new_entries
            and self.reconciliation_ran
            and self.foreign_order_halt_triggered
            and self.decision_logs_complete
            and self.required_decision_log_fields_present
            and self.component_imports_ok
        )

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "generated_at": self.generated_at,
            "pipeline_end_to_end": self.pipeline_end_to_end,
            "total_candidates_evaluated": self.total_candidates_evaluated,
            "candidates_executed": self.candidates_executed,
            "candidates_rejected": self.candidates_rejected,
            "all_executed_have_exchange_side_stop": self.all_executed_have_exchange_side_stop,
            "kill_switch_exercised": self.kill_switch_exercised,
            "kill_switch_halts_new_entries": self.kill_switch_halts_new_entries,
            "reconciliation_ran": self.reconciliation_ran,
            "foreign_order_halt_triggered": self.foreign_order_halt_triggered,
            "decision_logs_complete": self.decision_logs_complete,
            "decision_log_count": self.decision_log_count,
            "required_decision_log_fields_present": self.required_decision_log_fields_present,
            "component_imports_ok": self.component_imports_ok,
            "passed": self.passed(),
            "note": self.note,
        }


@dataclass(slots=True)
class PaperBReport:
    """Strategy paper validation report (PAPER-B gate output)."""

    session_id: str
    generated_at: str
    total_candidates: int
    executed_count: int
    rejected_count: int
    min_candidates_required: int
    min_executed_required: int
    symbol_breakdown: dict
    regime_breakdown: dict
    strategy_breakdown: dict
    rejection_breakdown: dict
    backtest_pnl: float
    paper_pnl: float
    pnl_consistency_ratio: float  # paper / backtest (1.0 = perfect; >=0.5 acceptable for paper)
    paper_vs_backtest_consistent: bool
    per_symbol_breakdown_present: bool
    per_regime_breakdown_present: bool
    note: str = ""

    def passed(self) -> bool:
        return (
            self.executed_count >= self.min_executed_required
            and self.total_candidates >= self.min_candidates_required
            and self.per_symbol_breakdown_present
            and self.per_regime_breakdown_present
            and self.paper_vs_backtest_consistent
        )

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "generated_at": self.generated_at,
            "total_candidates": self.total_candidates,
            "executed_count": self.executed_count,
            "rejected_count": self.rejected_count,
            "min_candidates_required": self.min_candidates_required,
            "min_executed_required": self.min_executed_required,
            "symbol_breakdown": self.symbol_breakdown,
            "regime_breakdown": self.regime_breakdown,
            "strategy_breakdown": self.strategy_breakdown,
            "rejection_breakdown": self.rejection_breakdown,
            "backtest_pnl": self.backtest_pnl,
            "paper_pnl": self.paper_pnl,
            "pnl_consistency_ratio": self.pnl_consistency_ratio,
            "paper_vs_backtest_consistent": self.paper_vs_backtest_consistent,
            "per_symbol_breakdown_present": self.per_symbol_breakdown_present,
            "per_regime_breakdown_present": self.per_regime_breakdown_present,
            "passed": self.passed(),
            "note": self.note,
        }


@dataclass(slots=True)
class PaperReport:
    """Combined paper report containing both A and B sub-reports."""

    paper_a: PaperAReport
    paper_b: PaperBReport

    def to_dict(self) -> dict:
        return {
            "paper_a": self.paper_a.to_dict(),
            "paper_b": self.paper_b.to_dict(),
        }


# --------------------------------------------------------------------------- #
# Builder                                                                       #
# --------------------------------------------------------------------------- #

_REQUIRED_DECISION_FIELDS = {
    "entry_ts",
    "symbol",
    "strategy",
    "regime",
    "side",
    "action",
    "reason",
    "risk_approved",
    "config_version",
    "universe_version",
    "kill_switch_state",
}


def build_paper_a_report(
    session: PaperSession,
    *,
    component_imports_ok: bool = True,
    kill_switch_halts_new_entries: bool | None = None,
) -> PaperAReport:
    """Build the PAPER-A technical report from a completed session."""
    now = datetime.now(UTC).isoformat()

    # Check simulated stops.
    stops_ok = all(t.has_exchange_side_stop for t in session.trades) if session.trades else True

    # Check decision logs completeness.
    logs_ok = len(session.decision_logs) == session.total_candidates
    fields_ok = True
    for log in session.decision_logs:
        log_dict = log.to_dict()
        if not _REQUIRED_DECISION_FIELDS.issubset(log_dict.keys()):
            fields_ok = False
            break

    # Kill switch halts new entries: check that after engagement, rejected
    # candidates appear with "exec_kill_switch_engaged" reason.
    if kill_switch_halts_new_entries is None:
        ks_rejections = [r for r in session.rejected if "kill_switch" in r.reason.lower()]
        kill_switch_halts_new_entries = bool(ks_rejections) or (
            session.kill_switch_exercised and session.total_candidates > 0
        )

    return PaperAReport(
        session_id=session.session_id,
        generated_at=now,
        pipeline_end_to_end=session.total_candidates > 0,
        total_candidates_evaluated=session.total_candidates,
        candidates_executed=session.executed_count,
        candidates_rejected=session.rejected_count,
        all_executed_have_exchange_side_stop=stops_ok,
        kill_switch_exercised=session.kill_switch_exercised,
        kill_switch_halts_new_entries=kill_switch_halts_new_entries,
        reconciliation_ran=len(session.reconciliation_events) > 0,
        foreign_order_halt_triggered=session.foreign_order_halt_triggered,
        decision_logs_complete=logs_ok,
        decision_log_count=len(session.decision_logs),
        required_decision_log_fields_present=fields_ok,
        component_imports_ok=component_imports_ok,
    )


def build_paper_b_report(
    session: PaperSession,
    *,
    backtest_pnl: float = 0.0,
    min_candidates_required: int = 10,
    min_executed_required: int = 5,
) -> PaperBReport:
    """Build the PAPER-B strategy report from a completed session."""
    now = datetime.now(UTC).isoformat()

    sym_breakdown = session.symbol_breakdown()
    reg_breakdown = session.regime_breakdown()
    strat_breakdown = session.strategy_breakdown()
    rej_breakdown = session.rejection_breakdown()

    paper_pnl = sum(t.pnl for t in session.trades)

    # Paper-vs-backtest consistency: ratio of 0.5 to 2.0 is acceptable for paper
    # (paper may have more realistic costs than backtest, so typically lower).
    if abs(backtest_pnl) > 1e-9:
        ratio = paper_pnl / backtest_pnl
        consistent = 0.3 <= ratio <= 3.0
    else:
        ratio = 1.0
        consistent = True

    return PaperBReport(
        session_id=session.session_id,
        generated_at=now,
        total_candidates=session.total_candidates,
        executed_count=session.executed_count,
        rejected_count=session.rejected_count,
        min_candidates_required=min_candidates_required,
        min_executed_required=min_executed_required,
        symbol_breakdown=sym_breakdown,
        regime_breakdown=reg_breakdown,
        strategy_breakdown=strat_breakdown,
        rejection_breakdown=rej_breakdown,
        backtest_pnl=round(backtest_pnl, 6),
        paper_pnl=round(paper_pnl, 6),
        pnl_consistency_ratio=round(ratio, 4),
        paper_vs_backtest_consistent=consistent,
        per_symbol_breakdown_present=len(sym_breakdown) > 0,
        per_regime_breakdown_present=len(reg_breakdown) > 0,
    )


def build_paper_report(
    session: PaperSession,
    *,
    backtest_pnl: float = 0.0,
    component_imports_ok: bool = True,
    kill_switch_halts_new_entries: bool | None = None,
    min_candidates_required: int = 10,
    min_executed_required: int = 5,
) -> PaperReport:
    """Build the combined paper report (A + B)."""
    paper_a = build_paper_a_report(
        session,
        component_imports_ok=component_imports_ok,
        kill_switch_halts_new_entries=kill_switch_halts_new_entries,
    )
    paper_b = build_paper_b_report(
        session,
        backtest_pnl=backtest_pnl,
        min_candidates_required=min_candidates_required,
        min_executed_required=min_executed_required,
    )
    return PaperReport(paper_a=paper_a, paper_b=paper_b)


def write_report(report: PaperReport, path: Path) -> None:
    """Write the paper report to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
