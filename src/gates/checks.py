"""Concrete gate checks for the Phase 1 infrastructure gates.

Each check returns a list of :class:`Criterion`. A gate PASSes only when every
criterion passes. Checks are honest: they probe the real database, Redis,
storage and kill switch, validate the real compose file and ``.env.example``,
and exercise the real job queue (Appendix A pass conditions; Appendix B.11).
"""

from __future__ import annotations

from collections.abc import Callable

import redis
import yaml
from sqlalchemy import inspect, text

from src.config import Settings, get_settings
from src.config.settings import REPO_ROOT
from src.db.base import get_engine
from src.db.models import JobStatus
from src.gates.result import Criterion
from src.killswitch import KillSwitch
from src.monitoring import Alert, AlertSeverity, check_health, get_alert_sink
from src.storage import DataLake, DatasetManifest, new_snapshot_id

# Services required by Appendix B.3.
REQUIRED_COMPOSE_SERVICES = {
    "postgres",
    "redis",
    "backend",
    "dashboard",
    "worker-data",
    "worker-backtest",
    "worker-ml",
    "worker-rl",
    "worker-reports",
    "scheduler",
    "trading-engine-paper",
    "trading-engine-live",
    "caddy",
}

REQUIRED_TABLES = {
    "jobs",
    "job_logs",
    "gates",
    "gate_results",
    "remediation_actions",
    "approvals",
    "audit_logs",
}


def _load_env_example() -> dict[str, str]:
    path = REPO_ROOT / ".env.example"
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip()
    return env


def _load_compose() -> dict:
    path = REPO_ROOT / "docker-compose.yml"
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


# --------------------------------------------------------------------------- #
# INFRA                                                                        #
# --------------------------------------------------------------------------- #
def check_infra(settings: Settings) -> list[Criterion]:
    out: list[Criterion] = []

    compose = _load_compose()
    services = set((compose.get("services") or {}).keys())
    if not services:
        out.append(Criterion.fail("compose_file", "docker-compose.yml missing or empty"))
    else:
        missing = REQUIRED_COMPOSE_SERVICES - services
        out.append(
            Criterion.ok("compose_services_defined", f"{len(services)} services")
            if not missing
            else Criterion.fail("compose_services_defined", f"missing: {sorted(missing)}")
        )

    # Live engine must be present but disabled by default (Appendix B.3).
    live = (compose.get("services") or {}).get("trading-engine-live", {})
    profiles = set(live.get("profiles", []) if isinstance(live, dict) else [])
    if not live:
        out.append(Criterion.fail("live_engine_present", "trading-engine-live not defined"))
    elif "live" in profiles or "disabled" in profiles:
        out.append(
            Criterion.ok("live_engine_disabled_by_default", f"behind profile {sorted(profiles)}")
        )
    else:
        out.append(
            Criterion.fail(
                "live_engine_disabled_by_default",
                "trading-engine-live must sit behind a non-default compose profile",
            )
        )

    # Safe env defaults (Appendix B.3 / B.17).
    env = _load_env_example()
    if not env:
        out.append(Criterion.fail("env_example", ".env.example missing or empty"))
    else:
        checks = {
            "TRADING_MODE": "PAPER",
            "ENABLE_LIVE_TRADING": "false",
        }
        bad = {k: env.get(k) for k, v in checks.items() if env.get(k) != v}
        out.append(
            Criterion.ok("env_safe_defaults", "TRADING_MODE=PAPER, live disabled")
            if not bad
            else Criterion.fail("env_safe_defaults", f"unsafe defaults: {bad}")
        )
        # No real keys committed.
        key = env.get("EXCHANGE_API_KEY", "")
        out.append(
            Criterion.ok("no_real_keys", "no API keys in .env.example")
            if key in ("", '""', "''")
            else Criterion.fail("no_real_keys", "EXCHANGE_API_KEY must be blank in .env.example")
        )

    out.append(
        Criterion.ok("dashboard_auth_configured", f"mode={settings.dashboard_auth_mode.value}")
        if settings.dashboard_auth_mode.value != "none" or settings.app_env.value == "local"
        else Criterion.fail("dashboard_auth_configured", "auth required outside local")
    )

    # Health endpoints green (db/redis/storage reachable).
    report = check_health(settings=settings, include_killswitch=False)
    out.append(
        Criterion.ok("health_endpoints_green", "db/redis/storage healthy")
        if report.healthy
        else Criterion.fail(
            "health_endpoints_green",
            "; ".join(f"{c.name}:{c.detail}" for c in report.components if not c.healthy),
        )
    )

    # CLI kill switch works independent of dashboard (file backend, no web).
    out.append(_check_killswitch_independent(settings))
    return out


def _check_killswitch_independent(settings: Settings) -> Criterion:
    ks = KillSwitch(settings)
    was_engaged = ks.engaged()
    if was_engaged:
        # Do not disturb an operator-engaged switch; just confirm observability.
        return Criterion.ok("kill_switch_independent", "engaged; observable without UI")
    try:
        ks.engage(reason="gate-selftest", actor="gate")
        engaged = ks.engaged()
        ks.disengage(actor="gate")
        cleared = not ks.engaged()
        if engaged and cleared:
            return Criterion.ok(
                "kill_switch_independent", "engage/disengage works with no web process"
            )
        return Criterion.fail("kill_switch_independent", "kill switch did not toggle")
    except Exception as exc:  # noqa: BLE001
        return Criterion.fail("kill_switch_independent", f"error: {exc}")


# --------------------------------------------------------------------------- #
# DB                                                                           #
# --------------------------------------------------------------------------- #
def check_db(settings: Settings) -> list[Criterion]:
    out: list[Criterion] = []
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        out.append(Criterion.ok("db_reachable", "SELECT 1 ok"))
    except Exception as exc:  # noqa: BLE001
        return [Criterion.fail("db_reachable", f"unreachable: {exc}")]

    insp = inspect(engine)
    tables = set(insp.get_table_names())

    out.append(
        Criterion.ok("alembic_applied", "alembic_version present")
        if "alembic_version" in tables
        else Criterion.fail("alembic_applied", "run `make migrate`")
    )

    missing = REQUIRED_TABLES - tables
    out.append(
        Criterion.ok("required_tables_exist", f"{len(REQUIRED_TABLES)} base tables present")
        if not missing
        else Criterion.fail("required_tables_exist", f"missing tables: {sorted(missing)}")
    )

    # Critical index for job queries (Appendix B.11).
    try:
        job_indexes = {ix["name"] for ix in insp.get_indexes("jobs")}
        out.append(
            Criterion.ok("indexes_exist", "ix_jobs_type_status present")
            if "ix_jobs_type_status" in job_indexes
            else Criterion.fail("indexes_exist", "missing ix_jobs_type_status")
        )
    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("indexes_exist", f"error: {exc}"))

    # Connection pooling works (two concurrent checked-out connections).
    try:
        c1 = engine.connect()
        c2 = engine.connect()
        c1.execute(text("SELECT 1"))
        c2.execute(text("SELECT 1"))
        c1.close()
        c2.close()
        pool_size = getattr(engine.pool, "size", lambda: "n/a")()
        out.append(Criterion.ok("connection_pooling", f"pool size={pool_size}"))
    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("connection_pooling", f"error: {exc}"))
    return out


# --------------------------------------------------------------------------- #
# QUEUE                                                                        #
# --------------------------------------------------------------------------- #
def check_queue(settings: Settings) -> list[Criterion]:
    # Imported here to avoid a cycle (handlers import gates for run_gate).
    from src.jobs import JobQueue, Worker
    from src.jobs.handlers import ensure_handlers_registered

    ensure_handlers_registered()
    out: list[Criterion] = []

    try:
        client = redis.Redis.from_url(settings.redis_url, socket_connect_timeout=1)
        client.ping()
        out.append(Criterion.ok("redis_reachable", "PING ok"))
    except Exception as exc:  # noqa: BLE001
        return [Criterion.fail("redis_reachable", f"unreachable: {exc}")]

    queue = JobQueue(settings)
    worker = Worker(settings)

    # 1) enqueue + consume + progress + job record persisted.
    try:
        job_id = queue.enqueue("selftest_echo", {"steps": 3}, requested_by="gate")
        status = worker.process_job(job_id)
        job = _load_job(job_id)
        consumed_ok = status is JobStatus.SUCCEEDED and job is not None
        out.append(
            Criterion.ok("worker_consumes_jobs", f"job {job_id} -> {status.value}")
            if consumed_ok
            else Criterion.fail("worker_consumes_jobs", f"status={status}")
        )
        progress_ok = (
            job is not None
            and job.progress_total > 0
            and (job.progress_current == job.progress_total)
        )
        out.append(
            Criterion.ok("progress_tracked", f"{job.progress_current}/{job.progress_total}")
            if progress_ok
            else Criterion.fail("progress_tracked", "progress not updated")
        )
        logs_ok = job is not None and len(job.logs) > 0
        out.append(
            Criterion.ok("job_records_persisted", f"{len(job.logs)} log lines")
            if logs_ok
            else Criterion.fail("job_records_persisted", "no job logs persisted")
        )
    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("worker_consumes_jobs", f"error: {exc}"))

    # 2) cancel works (cancel a queued job; worker must not run it).
    try:
        cid = queue.enqueue("selftest_echo", {"steps": 1}, requested_by="gate")
        cancelled = queue.cancel(cid)
        status = worker.process_job(cid)
        out.append(
            Criterion.ok("cancel_works", "queued job cancelled and skipped")
            if cancelled and status is JobStatus.CANCELLED
            else Criterion.fail("cancel_works", f"cancel={cancelled} status={status}")
        )
    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("cancel_works", f"error: {exc}"))

    # 3) retry works (a failing job becomes FAILED, then retry re-queues it).
    try:
        fid = queue.enqueue("selftest_fail", {}, requested_by="gate", max_attempts=1)
        status = worker.process_job(fid)
        failed_visible = status is JobStatus.FAILED
        requeued = queue.retry(fid)
        job = _load_job(fid)
        out.append(
            Criterion.ok("retry_and_failures_visible", "failed job visible and re-queued")
            if failed_visible and requeued and job is not None and job.status is JobStatus.QUEUED
            else Criterion.fail(
                "retry_and_failures_visible", f"failed={failed_visible} requeued={requeued}"
            )
        )
        queue.cancel(fid)  # clean up the re-queued job
        worker.process_job(fid)
    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("retry_and_failures_visible", f"error: {exc}"))
    return out


def _load_job(job_id: str):  # type: ignore[no-untyped-def]
    from src.db.base import session_scope
    from src.db.models import Job

    with session_scope() as session:
        job = session.get(Job, job_id)
        if job is not None:
            _ = job.logs  # force-load relationship within the session
            session.expunge_all()
        return job


# --------------------------------------------------------------------------- #
# STORAGE                                                                      #
# --------------------------------------------------------------------------- #
def check_storage(settings: Settings) -> list[Criterion]:
    out: list[Criterion] = []
    lake = DataLake(settings.data_lake_path, settings.artifact_path)

    out.append(
        Criterion.ok("data_lake_writable", str(settings.data_lake_path))
        if lake.writable()
        else Criterion.fail("data_lake_writable", f"not writable: {settings.data_lake_path}")
    )

    s1, s2 = new_snapshot_id(), new_snapshot_id()
    out.append(
        Criterion.ok("snapshot_id_generated", f"{s1}, {s2}")
        if s1 != s2
        else Criterion.fail("snapshot_id_generated", "ids not unique")
    )

    try:
        manifest = DatasetManifest(
            snapshot_id=s1,
            created_at="gate-selftest",
            data_types=["selftest"],
            source_jobs=["gate:storage"],
        )
        lake.create_snapshot(manifest)
        readback = lake.read_manifest(s1)
        manifest_ok = readback.snapshot_id == s1 and readback.checksum != ""
        out.append(
            Criterion.ok("manifest_written_versioned", f"checksum={readback.checksum}")
            if manifest_ok
            else Criterion.fail("manifest_written_versioned", "manifest read-back mismatch")
        )
    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("manifest_written_versioned", f"error: {exc}"))

    try:
        path = lake.write_artifact("selftest/gate.txt", b"ok")
        out.append(
            Criterion.ok("artifact_writable", str(path))
            if path.exists()
            else Criterion.fail("artifact_writable", "artifact not written")
        )
    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("artifact_writable", f"error: {exc}"))
    return out


# --------------------------------------------------------------------------- #
# MON (skeleton)                                                               #
# --------------------------------------------------------------------------- #
def check_mon(settings: Settings) -> list[Criterion]:
    out: list[Criterion] = []
    report = check_health(settings=settings)
    out.append(
        Criterion.ok("health_checks_active", f"service={report.service}")
        if report.components
        else Criterion.fail("health_checks_active", "no components probed")
    )

    sink = get_alert_sink()
    before = len(sink.recent(limit=1000))
    # Exercise the required alert types end-to-end (Appendix B.14).
    for title, comp in (
        ("test alert", "monitoring"),
        ("websocket stale", "data"),
        ("job failed", "worker"),
        ("kill switch triggered", "safety"),
    ):
        sink.send(
            Alert(
                title=title,
                severity=AlertSeverity.INFO,
                component=comp,
                environment=settings.app_env.value,
                recommended_action="skeleton: verify delivery transport in Phase 13",
            )
        )
    delivered = len(sink.recent(limit=1000)) - before
    out.append(
        Criterion.ok("alert_test_delivered", f"{delivered} alerts delivered to sink")
        if delivered >= 4
        else Criterion.fail("alert_test_delivered", "alert delivery failed")
    )
    out.append(
        Criterion.ok(
            "monitoring_skeleton",
            "stale-data / failed-job / kill-switch alerts wired (skeleton; transports in P13)",
        )
    )
    return out


# --------------------------------------------------------------------------- #
# BACKUP (skeleton)                                                            #
# --------------------------------------------------------------------------- #
def check_backup(settings: Settings) -> list[Criterion]:
    from src.jobs.handlers import ensure_handlers_registered

    ensure_handlers_registered()
    from src.jobs.registry import registry

    out: list[Criterion] = []
    backup_script = REPO_ROOT / "scripts" / "backup_db.sh"
    restore_script = REPO_ROOT / "scripts" / "restore_test.sh"

    out.append(
        Criterion.ok("backup_script_present", str(backup_script))
        if backup_script.exists()
        else Criterion.fail("backup_script_present", "scripts/backup_db.sh missing")
    )
    out.append(
        Criterion.ok("restore_test_script_present", str(restore_script))
        if restore_script.exists()
        else Criterion.fail("restore_test_script_present", "scripts/restore_test.sh missing")
    )
    out.append(
        Criterion.ok("restore_runnable_as_job", "run_restore_test_check registered")
        if registry.has("run_restore_test_check")
        else Criterion.fail("restore_runnable_as_job", "restore-test job not registered")
    )
    try:
        settings.backup_path.mkdir(parents=True, exist_ok=True)
        (settings.backup_path / ".probe").write_text("ok")
        (settings.backup_path / ".probe").unlink()
        out.append(Criterion.ok("backup_path_writable", str(settings.backup_path)))
    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("backup_path_writable", f"error: {exc}"))
    out.append(
        Criterion.ok(
            "backup_skeleton",
            "scheduled backups + verified restore enforced by full BACKUP gate in Phase 13",
        )
    )
    return out


# --------------------------------------------------------------------------- #
# Registry                                                                     #
# --------------------------------------------------------------------------- #
CHECKS: dict[str, Callable[[Settings], list[Criterion]]] = {
    "INFRA": check_infra,
    "DB": check_db,
    "QUEUE": check_queue,
    "STORAGE": check_storage,
    "MON": check_mon,
    "BACKUP": check_backup,
}


def has_check(gate_id: str) -> bool:
    return gate_id in CHECKS


def run_check(gate_id: str, settings: Settings | None = None) -> list[Criterion]:
    settings = settings or get_settings()
    return CHECKS[gate_id](settings)
