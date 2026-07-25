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


def test_redis_probe_has_bounded_timeouts(monkeypatch) -> None:
    """L35: the Redis probe must set BOTH connect and socket timeouts — a hung-but-accepting
    Redis would otherwise block PING forever and hang every health endpoint."""
    import redis as redis_mod
    from src.config import get_settings
    from src.monitoring.health import _check_redis

    captured: dict = {}
    real_from_url = redis_mod.Redis.from_url

    def _spy(url, **kwargs):
        captured.update(kwargs)
        return real_from_url(url, **kwargs)

    monkeypatch.setattr(redis_mod.Redis, "from_url", _spy)
    _check_redis(get_settings())
    assert captured.get("socket_connect_timeout") == 1
    assert captured.get("socket_timeout") == 1


@requires_db
@requires_redis
def test_readiness_probe_covers_db_and_redis() -> None:
    from src.monitoring import check_readiness

    report = check_readiness()
    assert {c.name for c in report.components} == {"database", "redis"}
    assert report.healthy is True, report.to_dict()


@requires_db
def test_infra_gate_notices_a_database_behind_the_migration_head() -> None:
    """A deploy that ships new code without running migrations leaves the DB behind the ORM, and
    the only symptom is dashboard pages rendering the generic error shell while "column X does not
    exist" sits in the logs. The INFRA gate only checked that the alembic_version TABLE existed,
    which says nothing about the revision — so drift passed the gate."""
    from src.config import Settings
    from src.gates.checks import _check_schema_at_head, check_db

    criteria = {c.id: c for c in check_db(Settings(_env_file=None))}
    assert "schema_at_head" in criteria
    assert criteria["schema_at_head"].status == "PASS"  # the test DB is migrated to head

    class _StaleDB:
        """An engine stamped at an ancient revision."""

        def connect(self):
            class _Conn:
                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    return False

                def execute(self, *_a, **_k):
                    return [("0001",)]

            return _Conn()

    stale = _check_schema_at_head(_StaleDB())
    assert stale.status == "FAIL"
    assert "0001" in stale.detail and "make migrate" in stale.detail
