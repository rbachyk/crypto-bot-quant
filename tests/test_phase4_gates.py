"""Phase 4 gate tests: BT, WF, FEE, SLIP pass on a Reviewer re-run (Appendix D).

Integration-level checks that exercise the Gate Runner end-to-end (persisting a
``BacktestRun`` index row + report), so they require the DB. The pure-engine
behaviour is covered offline in ``test_backtest.py``.
"""

from __future__ import annotations

import pytest
from src.gates import GateRunner
from src.gates.result import GateVerdict

from tests.conftest import requires_db


@requires_db
@pytest.mark.parametrize("gate_id", ["BT", "WF", "FEE", "SLIP"])
def test_phase4_gate_passes(gate_id: str) -> None:
    result = GateRunner().run(gate_id)
    assert result.overall == GateVerdict.PASS.value, (gate_id, result.note, result.criteria)
    assert result.criteria
    assert all(c["status"] == "PASS" for c in result.criteria), [
        c for c in result.criteria if c["status"] != "PASS"
    ]
    assert result.report_path


@requires_db
def test_phase4_dependency_chain() -> None:
    # Appendix A dependency graph for the Phase 4 gates.
    runner = GateRunner()
    assert set(runner.catalog["BT"].depends_on) >= {"DATA-COV", "DQ", "UNIV", "META", "FEAT"}
    assert runner.catalog["WF"].depends_on == ["BT"]
    assert set(runner.catalog["FEE"].depends_on) >= {"BT", "WF"}
    assert set(runner.catalog["SLIP"].depends_on) >= {"BT", "WF"}


@requires_db
def test_bt_persists_backtest_run_row() -> None:
    from sqlalchemy import desc, select
    from src.db.base import session_scope
    from src.db.models import BacktestRun

    GateRunner().run("BT")
    with session_scope() as session:
        # The BT gate persists a REFERENCE backtest (no dataset_version); filter to those
        # so real-data lake runs / leaderboard rows in the shared DB don't shadow it.
        latest = (
            session.execute(
                select(BacktestRun)
                .where(BacktestRun.kind == "backtest")
                .where(BacktestRun.dataset_version.is_(None))
                .order_by(desc(BacktestRun.id))
            )
            .scalars()
            .first()
        )
        assert latest is not None
        assert latest.trade_count > 0
        assert latest.report_path
