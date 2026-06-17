"""Phase 6 gate tests: SETUP, RISK, EXEC, KILL, ORDER-OWN (Appendix D).

Integration-level checks that exercise the Gate Runner end-to-end (it resolves the
full upstream dependency chain and persists a GateResult), so they require the DB.
The pure-component behaviour is covered offline in test_risk/test_ranking/
test_execution.
"""

from __future__ import annotations

import pytest
from src.gates import GateRunner
from src.gates.result import GateVerdict

from tests.conftest import requires_db


@requires_db
@pytest.mark.parametrize("gate_id", ["SETUP", "RISK", "EXEC", "KILL", "ORDER-OWN"])
def test_phase6_gate_passes(gate_id: str) -> None:
    result = GateRunner().run(gate_id)
    assert result.overall == GateVerdict.PASS.value, (gate_id, result.note, result.criteria)
    assert result.criteria
    assert all(c["status"] == "PASS" for c in result.criteria), [
        c for c in result.criteria if c["status"] != "PASS"
    ]
    assert result.report_path


@requires_db
def test_phase6_dependency_chain() -> None:
    runner = GateRunner()
    assert runner.catalog["SETUP"].depends_on == ["FEAT"]
    assert runner.catalog["RISK"].depends_on == ["SETUP"]
    assert set(runner.catalog["EXEC"].depends_on) >= {"META", "RISK"}
    assert runner.catalog["KILL"].depends_on == ["EXEC"]
    assert runner.catalog["ORDER-OWN"].depends_on == ["EXEC"]


@requires_db
def test_phase6_gates_persist_results() -> None:
    from sqlalchemy import desc, select
    from src.db.base import session_scope
    from src.db.models import GateResult, GateStatus

    GateRunner().run("RISK")
    with session_scope() as session:
        latest = (
            session.execute(
                select(GateResult).where(GateResult.gate_id == "RISK").order_by(desc(GateResult.id))
            )
            .scalars()
            .first()
        )
        assert latest is not None
        assert latest.status is GateStatus.PASSED
        assert latest.criteria
