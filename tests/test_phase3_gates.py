"""Phase 3 gate tests: META, UNIV, FEAT pass on a Reviewer re-run (Appendix D)."""

from __future__ import annotations

import pytest
from src.gates import GateRunner
from src.gates.result import GateVerdict

from tests.conftest import requires_db


@requires_db
@pytest.mark.parametrize("gate_id", ["META", "UNIV", "FEAT"])
def test_phase3_gate_passes(gate_id: str) -> None:
    result = GateRunner().run(gate_id)
    assert result.overall == GateVerdict.PASS.value, (gate_id, result.note, result.criteria)
    assert result.criteria
    assert all(c["status"] == "PASS" for c in result.criteria), [
        c for c in result.criteria if c["status"] != "PASS"
    ]


@requires_db
def test_feat_depends_on_univ_and_datacov() -> None:
    # FEAT must list DATA-COV and UNIV upstream (Appendix A dependency rules).
    runner = GateRunner()
    assert set(runner.catalog["FEAT"].depends_on) >= {"DATA-COV", "UNIV"}
    assert runner.catalog["UNIV"].depends_on == ["DATA-COV"]


@requires_db
def test_univ_blocks_when_datacov_missing(monkeypatch) -> None:
    # If DATA-COV does not PASS, UNIV is reported BLOCKED, never silently run.
    runner = GateRunner()
    original = runner._evaluate

    from src.gates.result import GateRunResult

    def fake_eval(gate_id, cache):  # type: ignore[no-untyped-def]
        if gate_id == "DATA-COV":
            return GateRunResult("DATA-COV", GateVerdict.FAIL.value, note="forced")
        return original(gate_id, cache)

    monkeypatch.setattr(runner, "_evaluate", fake_eval)
    result = runner.run("UNIV")
    assert result.overall == GateVerdict.BLOCKED.value
