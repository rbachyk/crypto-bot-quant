"""Gate runner tests (AGENTS.md Section 25, Appendix A/B.10/B.11).

Verifies the Phase 1 gates PASS, that dependency resolution BLOCKs a gate whose
upstream is not PASS, and that non-PASS verdicts produce remediation actions.
"""

from __future__ import annotations

from sqlalchemy import desc, select
from src.db.base import session_scope
from src.db.models import GateResult, GateStatus, RemediationAction
from src.gates import GateRunner

from tests.conftest import requires_redis


@requires_redis
def test_phase1_gates_pass() -> None:
    runner = GateRunner()
    for gate_id in ("INFRA", "DB", "QUEUE", "STORAGE", "MON", "BACKUP"):
        result = runner.run(gate_id)
        assert result.overall == "PASS", f"{gate_id} -> {result.overall}: {result.note}"
        assert result.criteria
        assert result.report_path


@requires_redis
def test_result_persisted_to_db() -> None:
    GateRunner().run("STORAGE")
    with session_scope() as session:
        latest = (
            session.execute(
                select(GateResult)
                .where(GateResult.gate_id == "STORAGE")
                .order_by(desc(GateResult.id))
            )
            .scalars()
            .first()
        )
        assert latest is not None
        assert latest.status is GateStatus.PASSED


@requires_redis
def test_dependency_blocks_downstream() -> None:
    # RISK depends on SETUP, which has no check yet (NOT_RUN, introduced in Phase 5)
    # -> RISK is BLOCKED on the unmet dependency (Appendix A dependency rules).
    # (BT/WF/FEE/SLIP now have checks and PASS as of Phase 4, so they no longer
    # block their downstreams — the first unimplemented upstream is SETUP.)
    runner = GateRunner()
    risk = runner.run("RISK")
    assert risk.overall == "BLOCKED"
    assert "SETUP" in risk.note


@requires_redis
def test_blocked_gate_creates_remediation_actions() -> None:
    GateRunner().run("RISK")
    with session_scope() as session:
        actions = (
            session.execute(select(RemediationAction).where(RemediationAction.gate_id == "RISK"))
            .scalars()
            .all()
        )
        assert actions, "blocked gate must produce remediation actions (never a dead end)"
        # First action points at resolving the upstream dependency.
        assert any("upstream" in a.description.lower() for a in actions)
