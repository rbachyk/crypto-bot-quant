"""DATA-COV and DQ gates end to end (Appendix A, Appendix D Phase 2)."""

from __future__ import annotations

from sqlalchemy import desc, select
from src.db.base import session_scope
from src.db.models import DatasetVersion, GateResult, GateStatus
from src.gates import GateRunner

from tests.conftest import requires_redis


@requires_redis
def test_data_cov_passes_and_records_dataset_version() -> None:
    result = GateRunner().run("DATA-COV")
    assert result.overall == "PASS", f"DATA-COV -> {result.overall}: {result.note}"
    assert {c["id"] for c in result.criteria} >= {
        "required_series_present",
        "zero_unfilled_gaps",
        "immutable_snapshot_produced",
        "manifest_complete",
        "dataset_version_recorded",
    }
    assert all(c["status"] == "PASS" for c in result.criteria)
    with session_scope() as session:
        rows = session.query(DatasetVersion).all()
        assert rows and any(r.validation_status == "valid" for r in rows)


@requires_redis
def test_dq_passes_with_no_critical_violations() -> None:
    result = GateRunner().run("DQ")
    assert result.overall == "PASS", f"DQ -> {result.overall}: {result.note}"
    # Every Section 23 check is represented and clean.
    ids = {c["id"] for c in result.criteria}
    assert {"missing_candles", "markindex_alignment", "clock_drift"} <= ids
    assert all(c["status"] == "PASS" for c in result.criteria)


@requires_redis
def test_data_cov_persists_passed_result() -> None:
    GateRunner().run("DATA-COV")
    with session_scope() as session:
        latest = (
            session.execute(
                select(GateResult)
                .where(GateResult.gate_id == "DATA-COV")
                .order_by(desc(GateResult.id))
            )
            .scalars()
            .first()
        )
        assert latest is not None
        assert latest.status is GateStatus.PASSED
        assert latest.related_versions.get("DATA_VERSION")
