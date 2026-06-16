"""Health-check and DB-schema tests (Appendix B.11; INFRA/DB gates)."""

from __future__ import annotations

from sqlalchemy import inspect
from src.db.base import get_engine
from src.monitoring import check_health

from tests.conftest import requires_db, requires_redis

REQUIRED_TABLES = {
    "jobs",
    "job_logs",
    "gates",
    "gate_results",
    "remediation_actions",
    "approvals",
    "audit_logs",
    "exchange_metadata",
    "universe_versions",
    "universe_members",
}


@requires_db
def test_required_tables_exist() -> None:
    tables = set(inspect(get_engine()).get_table_names())
    missing = REQUIRED_TABLES - tables
    assert not missing, f"missing tables: {missing}"


@requires_db
def test_jobs_index_exists() -> None:
    indexes = {ix["name"] for ix in inspect(get_engine()).get_indexes("jobs")}
    assert "ix_jobs_type_status" in indexes


@requires_redis
def test_health_report_healthy() -> None:
    report = check_health()
    names = {c.name for c in report.components}
    assert {"database", "redis", "storage"} <= names
    assert report.healthy is True, report.to_dict()
