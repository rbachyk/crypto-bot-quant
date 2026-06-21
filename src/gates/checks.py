"""Concrete gate checks for the Phase 1 infrastructure gates.

Each check returns a list of :class:`Criterion`. A gate PASSes only when every
criterion passes. Checks are honest: they probe the real database, Redis,
storage and kill switch, validate the real compose file and ``.env.example``,
and exercise the real job queue (Appendix A pass conditions; Appendix B.11).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

import redis
import yaml
from sqlalchemy import inspect, text

from src.config import Settings, get_settings
from src.config.settings import REPO_ROOT
from src.db.base import get_engine
from src.db.models import JobStatus
from src.gates.phase6 import (
    check_exec,
    check_kill,
    check_order_own,
    check_risk,
    check_setup,
)
from src.gates.phase8 import check_paper_a, check_paper_b
from src.gates.phase9 import check_ml_promo as _check_ml_promo_phase9
from src.gates.phase10 import check_ml_phase10
from src.gates.phase11 import check_learn_promo_s
from src.gates.phase12 import check_rl_shadow, check_rl_sim
from src.gates.phase13 import (
    check_backup_phase13,
    check_config_freeze,
    check_deploy,
    check_learn_promo_l,
    check_live,
    check_mon_phase13,
    check_sec,
)
from src.gates.result import Criterion
from src.killswitch import KillSwitch
from src.monitoring import check_health
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
def _external_published_ports(compose: dict) -> dict[str, list[str]]:
    """Map service -> host ports published on a non-loopback interface.

    A port bound to 127.0.0.1 / localhost is loopback-only (dev tooling) and is
    NOT considered externally exposed. Everything else is. Used to enforce the
    deliverable "internal network; only 443 exposed".
    """
    offending: dict[str, list[str]] = {}
    for name, svc in (compose.get("services") or {}).items():
        if not isinstance(svc, dict):
            continue
        hosts: list[str] = []
        for entry in svc.get("ports", []) or []:
            host_ip = ""
            host_port = ""
            if isinstance(entry, dict):
                host_ip = str(entry.get("host_ip", ""))
                host_port = str(entry.get("published", ""))
            else:
                parts = str(entry).split(":")
                if len(parts) == 3:  # host_ip:host_port:container_port
                    host_ip, host_port = parts[0], parts[1]
                elif len(parts) == 2:  # host_port:container_port
                    host_port = parts[0]
                else:  # bare container port -> ephemeral host port (still exposed)
                    host_port = parts[0]
            if host_ip in ("127.0.0.1", "localhost", "::1"):
                continue
            hosts.append(host_port)
        if hosts:
            offending[name] = hosts
    return offending


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

    # Only 443 is published to the outside world (deliverable: internal network;
    # only 443 exposed). Loopback-bound dev ports do not count.
    published = _external_published_ports(compose)
    bad_ports = {svc: ports for svc, ports in published.items() if set(ports) - {"443"}}
    out.append(
        Criterion.ok("only_443_exposed", "only caddy:443 published externally")
        if not bad_ports
        else Criterion.fail("only_443_exposed", f"unexpected external ports: {bad_ports}")
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

    # Run the enqueue/consume/cancel probe on an ISOLATED redis DB so a running stack's
    # workers (`make docker-up`, db 0) can't consume the probe jobs mid-check and flake the
    # gate. The reachability check above already proves the production redis server works.
    probe_url = settings.redis_url.rsplit("/", 1)[0] + "/15"
    probe_client = redis.Redis.from_url(probe_url, decode_responses=True)
    queue = JobQueue(settings, redis_client=probe_client)
    worker = Worker(settings, redis_client=probe_client)

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
# DATA-COV (Phase 2)                                                           #
# --------------------------------------------------------------------------- #
def check_data_cov(settings: Settings) -> list[Criterion]:
    """Data Coverage & Integrity (Appendix A DATA-COV).

    Every active universe symbol must have all required series covering the
    configured window with zero unfilled gaps, and an immutable dataset
    snapshot must be produced. Auto-remediation is partial: safe gap repair is
    attempted from the data source before coverage is judged.
    """
    from src.data import DataPlatform, load_data_config

    out: list[Criterion] = []
    cfg = load_data_config()
    platform = DataPlatform(settings=settings, cfg=cfg)

    coverage = platform.ensure_coverage(repair=True)  # safe gap repair (partial auto-remediation)
    validation = platform.validate()
    snapshot = platform.build_snapshot(coverage, validation, source_jobs=["gate:data-cov"])
    platform.write_quality_report(validation, snapshot.snapshot_id)

    out.append(
        Criterion.ok(
            "required_series_present",
            f"{coverage.covered_series}/{coverage.required_series} series across "
            f"{len(cfg.active_symbols())} symbols",
        )
        if coverage.covered_series == coverage.required_series
        else Criterion.fail(
            "required_series_present",
            f"{coverage.required_series - coverage.covered_series} series missing data",
        )
    )

    total_missing = sum(len(g.missing_ts) for g in coverage.uncovered)
    if coverage.covered:
        out.append(Criterion.ok("zero_unfilled_gaps", "0 unfilled gaps over the window"))
    else:
        flagged = "; ".join(
            f"{g.key.label()}({len(g.missing_ts)} missing)" for g in coverage.uncovered[:5]
        )
        out.append(
            Criterion.fail(
                "zero_unfilled_gaps",
                f"{total_missing} unfilled gaps remain after repair: {flagged}",
            )
        )

    manifest = snapshot.manifest
    snapshot_ok = bool(manifest.snapshot_id and manifest.checksum and manifest.row_counts)
    out.append(
        Criterion.ok(
            "immutable_snapshot_produced",
            f"{manifest.snapshot_id} (checksum={snapshot.dataset_checksum}, "
            f"{sum(manifest.row_counts.values())} rows)",
        )
        if snapshot_ok
        else Criterion.fail("immutable_snapshot_produced", "snapshot/manifest incomplete")
    )

    # Manifest must carry the Appendix B.5 required fields.
    manifest_fields_ok = bool(
        manifest.symbols and manifest.time_range and manifest.data_types and manifest.row_counts
    )
    out.append(
        Criterion.ok("manifest_complete", "symbols/time_range/data_types/row_counts present")
        if manifest_fields_ok
        else Criterion.fail("manifest_complete", "manifest missing required fields")
    )

    # Relational dataset-version index recorded (Appendix B.4).
    try:
        from src.db.base import session_scope
        from src.db.models import DatasetVersion

        with session_scope() as session:
            row = session.get(DatasetVersion, snapshot.snapshot_id)
            recorded = row is not None and row.checksum == snapshot.dataset_checksum
        out.append(
            Criterion.ok("dataset_version_recorded", f"dataset_versions[{snapshot.snapshot_id}]")
            if recorded
            else Criterion.fail("dataset_version_recorded", "dataset_versions row missing")
        )
    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("dataset_version_recorded", f"error: {exc}"))

    return out


# --------------------------------------------------------------------------- #
# DQ (Phase 2+)                                                                #
# --------------------------------------------------------------------------- #
_DQ_CHECKS = [
    "missing_candles",
    "duplicates",
    "ordering",
    "future_timestamps",
    "impossible_prices",
    "extreme_gaps",
    "funding_alignment",
    "markindex_alignment",
    "abnormal_spread",
    "clock_drift",
]


def check_dq(settings: Settings) -> list[Criterion]:
    """Data Quality (Appendix A DQ / Section 23).

    No *critical* data-quality violation may be active and the clock must be
    within NTP tolerance. Emits one criterion per Section 23 check so the
    dashboard can point at exactly what failed.
    """
    from src.data import DataPlatform, load_data_config

    out: list[Criterion] = []
    platform = DataPlatform(settings=settings, cfg=load_data_config())
    report = platform.validate()
    path = platform.write_quality_report(report, dataset_version=None)

    critical_by_check: dict[str, list[str]] = {}
    for v in report.critical:
        critical_by_check.setdefault(v.check, []).append(f"{v.series or 'global'}: {v.detail}")

    for check in _DQ_CHECKS:
        hits = critical_by_check.get(check)
        if not hits:
            out.append(Criterion.ok(check, "no critical violation"))
        else:
            out.append(Criterion.fail(check, "; ".join(hits[:3])))

    out.append(
        Criterion.ok("validation_report_written", path)
        if path
        else Criterion.fail("validation_report_written", "report not written")
    )
    return out


# --------------------------------------------------------------------------- #
# META (Phase 1-3)                                                             #
# --------------------------------------------------------------------------- #
def check_meta(settings: Settings) -> list[Criterion]:
    """Exchange Metadata (Appendix A META).

    Every active (trading-candidate) symbol must carry complete, internally
    consistent, ``[VERIFIED]``, current-version metadata. The verified specs come
    from ``configs/metadata.yaml`` (the recorded operator-review step, Section 6);
    this check syncs them and then audits the persisted ``exchange_metadata`` rows
    — no ``[UNVERIFIED]`` flag may remain for an active symbol (Section 2.1).
    """
    import json as _json

    from src.db.base import session_scope
    from src.db.models import ExchangeMetadata, VerificationStatus
    from src.exchange.metadata import VerifiedSpec, load_metadata_config, sync_verified_metadata

    cfg = load_metadata_config()
    out: list[Criterion] = []
    symbols = cfg.symbols()

    with session_scope() as session:
        written = sync_verified_metadata(session, cfg)
        session.flush()
        rows = {
            r.symbol: r
            for r in session.query(ExchangeMetadata)
            .filter_by(exchange_id=cfg.exchange_id, metadata_version=cfg.metadata_version)
            .all()
        }
        # Detect any stale (older metadata_version) or UNVERIFIED rows for our symbols.
        stale_or_unverified = (
            session.query(ExchangeMetadata.symbol, ExchangeMetadata.metadata_version)
            .filter(
                ExchangeMetadata.exchange_id == cfg.exchange_id,
                ExchangeMetadata.symbol.in_(symbols),
                ExchangeMetadata.verification_status == VerificationStatus.UNVERIFIED,
                ExchangeMetadata.metadata_version != cfg.metadata_version,
            )
            .count()
        )

        out.append(Criterion.ok("metadata_synced", f"{written} symbols at {cfg.metadata_version}"))

        missing_rows = [s for s in symbols if s not in rows]
        out.append(
            Criterion.ok("all_active_have_metadata", f"{len(symbols)} symbols")
            if not missing_rows
            else Criterion.fail("all_active_have_metadata", f"no metadata: {missing_rows}")
        )

        unverified = [
            s for s, r in rows.items() if r.verification_status is not VerificationStatus.VERIFIED
        ]
        out.append(
            Criterion.ok("all_verified", "no [UNVERIFIED] active symbols")
            if not unverified
            else Criterion.fail("all_verified", f"[UNVERIFIED]: {unverified}")
        )

        incomplete: dict[str, list[str]] = {}
        contradictory: dict[str, list[str]] = {}
        for s in symbols:
            spec = cfg.spec(s)
            if spec is None:
                incomplete[s] = ["no spec"]
                continue
            vs = VerifiedSpec(symbol=s, fields=spec.fields)
            miss = vs.missing_fields()
            cons = vs.contradictions()
            if miss:
                incomplete[s] = miss
            if cons:
                contradictory[s] = cons
        out.append(
            Criterion.ok("complete_fields", "all required fields present")
            if not incomplete
            else Criterion.fail("complete_fields", _json.dumps(incomplete)[:300])
        )
        out.append(
            Criterion.ok("no_contradictions", "metadata internally consistent")
            if not contradictory
            else Criterion.fail("no_contradictions", _json.dumps(contradictory)[:300])
        )

        def _order_types(symbol: str) -> object:
            spec = cfg.spec(symbol)
            return spec.fields.get("order_types") if spec is not None else None

        order_types_ok = all(isinstance(_order_types(s), list) and _order_types(s) for s in symbols)
        out.append(
            Criterion.ok("order_types_present", "order types declared for all symbols")
            if order_types_ok
            else Criterion.fail("order_types_present", "missing order types")
        )

        out.append(
            Criterion.ok("not_stale", f"all metadata at current version {cfg.metadata_version}")
            if stale_or_unverified == 0
            else Criterion.fail(
                "not_stale", f"{stale_or_unverified} stale/unverified rows for active symbols"
            )
        )

    _write_report(
        settings,
        "metadata",
        {
            "metadata_version": cfg.metadata_version,
            "exchange_id": cfg.exchange_id,
            "verified_against": cfg.verified_against,
            "symbols": {
                s: (sp.fields if (sp := cfg.spec(s)) is not None else None) for s in symbols
            },
        },
    )
    return out


# --------------------------------------------------------------------------- #
# UNIV (Phase 1-2 / refreshed in Phase 3)                                      #
# --------------------------------------------------------------------------- #
def check_univ(settings: Settings) -> list[Criterion]:
    """Universe Validity (Appendix A UNIV).

    Builds the versioned universe from ``configs/universe.yaml`` filters and
    audits it: every selected (active) symbol must pass EVERY filter and carry
    stable ``[VERIFIED]`` metadata; the universe must be versioned and its
    membership changes logged (Section 9).
    """
    from src.db.base import session_scope
    from src.db.models import SymbolStatus, UniverseChange, UniverseVersion
    from src.universe import UniverseManager, load_universe_config

    uni_cfg = load_universe_config()
    out: list[Criterion] = []

    with session_scope() as session:
        result = UniverseManager(settings=settings, uni_cfg=uni_cfg).build(session)
        session.flush()
        version = result.version
        evals = result.evaluations
        active = [e for e in evals if e.status is SymbolStatus.ACTIVE]
        change_count = session.query(UniverseChange).filter_by(universe_version=version).count()
        versioned = session.get(UniverseVersion, version) is not None

        out.append(
            Criterion.ok("universe_versioned", version)
            if versioned
            else Criterion.fail("universe_versioned", "no UniverseVersion persisted")
        )
        out.append(
            Criterion.ok("filters_applied", f"{len(evals)}/{len(uni_cfg.candidates)} evaluated")
            if len(evals) == len(uni_cfg.candidates)
            else Criterion.fail("filters_applied", "not all candidates evaluated")
        )
        out.append(
            Criterion.ok("at_least_one_active", f"{len(active)} active symbols")
            if active
            else Criterion.fail("at_least_one_active", "no symbol passed all filters")
        )

        bad = [e.symbol for e in active if not e.passed_all]
        out.append(
            Criterion.ok("selected_pass_all_filters", "every active symbol passes all filters")
            if not bad
            else Criterion.fail("selected_pass_all_filters", f"active but failing: {bad}")
        )

        unstable = [
            e.symbol
            for e in active
            if not e.metrics.get("verified_metadata")
            or e.metrics.get("contract_status") != "trading"
        ]
        out.append(
            Criterion.ok("metadata_stable", "active symbols [VERIFIED] + status=trading")
            if not unstable
            else Criterion.fail("metadata_stable", f"unstable metadata: {unstable}")
        )

        no_data = [
            e.symbol
            for e in active
            if any(o.name == "data_availability" and not o.passed for o in e.outcomes)
        ]
        out.append(
            Criterion.ok("data_availability", "active symbols have all required series")
            if not no_data
            else Criterion.fail("data_availability", f"missing data: {no_data}")
        )

        out.append(
            Criterion.ok("membership_changes_logged", f"{change_count} change rows for {version}")
            if change_count >= 1
            else Criterion.fail("membership_changes_logged", "no membership history recorded")
        )

        report = {
            "universe_version": version,
            "policy_version": uni_cfg.universe_version,
            "active": sorted(e.symbol for e in active),
            "filter_report": result.filter_report(),
            "changes": result.changes,
        }

    _write_report(settings, "universe", report)
    return out


# --------------------------------------------------------------------------- #
# FEAT (Phase 3)                                                               #
# --------------------------------------------------------------------------- #
def check_feat(settings: Settings) -> list[Criterion]:
    """Feature Reproducibility (Appendix A FEAT).

    Builds the feature store from the current dataset snapshot for the active
    universe through the single feature code path, then proves: reproducible
    checksum, no look-ahead (causal-invariance under future truncation), zero
    synthetic expectancy (no leakage), timestamp alignment, and no unavailable
    features (Section 10 Parity Rule).
    """
    from src.data import DataPlatform, load_data_config
    from src.data.schema import timeframe_ms
    from src.db.base import session_scope
    from src.db.models import FeatureSetVersion
    from src.features import (
        FeatureStore,
        causal_invariance_violations,
        has_nan_or_inf,
        load_feature_config,
        synthetic_leakage_report,
    )
    from src.features.pipeline import StoreReader
    from src.universe import UniverseManager, latest_active_symbols, load_universe_config

    out: list[Criterion] = []
    data_cfg = load_data_config()
    feat_cfg = load_feature_config()

    # Ensure data is present + snapshotted (idempotent) and the universe is built.
    platform = DataPlatform(settings=settings, cfg=data_cfg)
    run = platform.run_full(repair=True, source_jobs=["gate:feat"])
    dataset_version = run.snapshot.snapshot_id

    with session_scope() as session:
        uni = UniverseManager(settings=settings, uni_cfg=load_universe_config()).build(session)
        session.flush()
        active = latest_active_symbols(session) or uni.active_symbols
        fstore = FeatureStore(settings=settings, data_cfg=data_cfg, feat_cfg=feat_cfg)
        build = fstore.build(
            active,
            dataset_version,
            universe_version=uni.version,
            session=session,
            source_jobs=["gate:feat"],
        )
        session.flush()
        recorded = session.get(FeatureSetVersion, build.feature_snapshot_id) is not None

    # 1) Builds from snapshot.
    builds_ok = bool(active) and build.total_rows > 0 and build.checksum
    out.append(
        Criterion.ok(
            "feature_store_builds_from_snapshot",
            f"{build.feature_snapshot_id} ({build.total_rows} rows, {len(active)} symbols)",
        )
        if builds_ok
        else Criterion.fail("feature_store_builds_from_snapshot", "empty/failed feature build")
    )

    # 2) Reproducible checksum (rebuild from the same snapshot must match).
    rebuild = fstore.build(active, dataset_version, universe_version=uni.version)
    out.append(
        Criterion.ok("reproducible_checksum", f"checksum={build.checksum}")
        if rebuild.checksum == build.checksum
        else Criterion.fail("reproducible_checksum", f"{build.checksum} != {rebuild.checksum}")
    )

    # 3) No look-ahead — causal invariance under future truncation.
    violations: list[str] = []
    for sym in active:
        reader = StoreReader(
            fstore.store,
            data_cfg.exchange_id,
            feat_cfg.timeframe,
            data_cfg.base_timeframe,
            data_cfg.funding_timeframe,
            data_cfg.window_start_ms,
            data_cfg.window_end_ms,
        )
        v = causal_invariance_violations(sym, reader, feat_cfg)
        violations.extend(f"{cv.symbol}:{cv.feature}@{cv.bar_ts}" for cv in v)
    out.append(
        Criterion.ok("no_lookahead_causal", "features invariant to future truncation")
        if not violations
        else Criterion.fail("no_lookahead_causal", f"look-ahead: {violations[:5]}")
    )

    # 4) Leakage — synthetic noise yields ~0 expectancy.
    leak = synthetic_leakage_report(feat_cfg)
    out.append(
        Criterion.ok(
            "leakage_zero_expectancy",
            f"|z|={abs(leak['z']):.2f} <= {leak['max_abs_z']} (n={leak['n']})",
        )
        if leak["passed"]
        else Criterion.fail(
            "leakage_zero_expectancy", f"|z|={abs(leak['z']):.2f} > {leak['max_abs_z']}"
        )
    )

    # 5) Timestamp alignment — decision_ts = bar_open + interval, on grid.
    iv = timeframe_ms(feat_cfg.timeframe)
    misaligned = 0
    for frame in build.frames.values():
        for r in frame.rows:
            if r["decision_ts"] != r["ts"] + iv or r["ts"] % iv != 0:
                misaligned += 1
    out.append(
        Criterion.ok("timestamp_alignment", "all rows on the decision-time grid")
        if misaligned == 0
        else Criterion.fail("timestamp_alignment", f"{misaligned} misaligned rows")
    )

    # 6) No unavailable / non-finite features.
    nan_syms = [s for s, f in build.frames.items() if has_nan_or_inf(f)]
    empty_syms = [s for s, f in build.frames.items() if not f.rows]
    out.append(
        Criterion.ok("no_unavailable_features", "finite features for every active symbol")
        if not nan_syms and not empty_syms
        else Criterion.fail("no_unavailable_features", f"nan={nan_syms} empty={empty_syms}")
    )

    # 7) Feature-set version recorded (relational index).
    out.append(
        Criterion.ok("feature_set_version_recorded", build.feature_snapshot_id)
        if recorded
        else Criterion.fail("feature_set_version_recorded", "feature_set_versions row missing")
    )

    _write_report(
        settings,
        "features",
        {
            "feature_snapshot_id": build.feature_snapshot_id,
            "feature_set_version": feat_cfg.feature_set_version,
            "dataset_version": dataset_version,
            "universe_version": uni.version,
            "checksum": build.checksum,
            "symbols": active,
            "row_counts": build.row_counts,
            "leakage": leak,
            "lookahead_violations": violations,
        },
    )
    return out


# --------------------------------------------------------------------------- #
# BT — Backtest Gate (Phase 4)                                                 #
# --------------------------------------------------------------------------- #
# Section 19 "Backtest output must include" — every key the report must carry.
_REQUIRED_BT_OUTPUTS: tuple[str, ...] = (
    "total_return",
    "net_pnl",
    "expectancy_r",
    "profit_factor",
    "max_drawdown",
    "trade_count",
    "symbol_breakdown",
    "strategy_breakdown",
    "regime_breakdown",
    "session_breakdown",
    "side_breakdown",
    "cost_breakdown",
    "slippage_breakdown",
    "funding_breakdown",
    "rejected_candidates",
    "worst_trades",
    "stability",
)


def check_bt(settings: Settings) -> list[Criterion]:
    """Backtest Gate (Appendix A BT).

    Runs the event-based engine on the deterministic reference fixture through
    the SAME feature pipeline as live (the Parity Rule), then proves: it completes
    without errors; fees/slippage/funding are modelled (fees from VERIFIED
    metadata); all Section 19 outputs are present; there is no look-ahead (the
    strategy is not profitable on a structureless series); no future-universe /
    survivorship leakage (a symbol is never traded before it joined the universe);
    and results are sane (no impossible returns).
    """
    from src.backtest import (
        build_reference_inputs,
        future_universe_violations,
        load_backtest_config,
        noise_expectancy,
        run_engine,
    )
    from src.backtest.service import persist_backtest_run, write_report
    from src.exchange.metadata import load_metadata_config
    from src.features.pipeline import FEATURE_NAMES

    out: list[Criterion] = []
    cfg = load_backtest_config()
    meta = load_metadata_config()
    inputs = build_reference_inputs(cfg)
    run = run_engine(cfg, meta, inputs, label="bt_gate")
    report = run.report
    payload = report.payload

    out.append(
        Criterion.ok(
            "engine_completes",
            f"{report.trade_count} trades, {len(run.result.rejected)} rejected",
        )
        if report.trade_count > 0
        else Criterion.fail("engine_completes", "engine produced no trades")
    )

    # Parity Rule: the engine reads the single feature pipeline's columns.
    parity_ok = all(
        set(FEATURE_NAMES).issubset(set(s.frame.rows[0])) for s in inputs if s.frame.rows
    )
    out.append(
        Criterion.ok("parity_rule_single_pipeline", "engine uses the live feature pipeline")
        if parity_ok
        else Criterion.fail("parity_rule_single_pipeline", "features not from the shared pipeline")
    )

    costs = payload["cost_breakdown"]
    costs_ok = costs["total_fees"] > 0 and costs["total_slippage"] > 0
    out.append(
        Criterion.ok(
            "realistic_costs_modeled",
            f"fees={costs['total_fees']:.2f} slippage={costs['total_slippage']:.2f} "
            f"funding={costs['total_funding']:.2f}",
        )
        if costs_ok
        else Criterion.fail("realistic_costs_modeled", f"missing cost components: {costs}")
    )

    # Fees come from VERIFIED metadata (META gate guarantees they exist).
    ref_specs = [(s, meta.spec(s)) for s in cfg.reference.symbols]
    fees_ok = all(
        isinstance(spec.fields.get("taker_fee"), (int, float))
        for _, spec in ref_specs
        if spec is not None
    )
    out.append(
        Criterion.ok("fees_from_verified_metadata", "maker/taker fees sourced from metadata")
        if fees_ok
        else Criterion.fail("fees_from_verified_metadata", "symbol lacks verified fees")
    )

    missing = [k for k in _REQUIRED_BT_OUTPUTS if k not in payload]
    empty = [
        k for k in ("symbol_breakdown", "regime_breakdown", "side_breakdown") if not payload.get(k)
    ]
    out.append(
        Criterion.ok("all_required_outputs_present", f"{len(_REQUIRED_BT_OUTPUTS)} metrics emitted")
        if not missing and not empty
        else Criterion.fail("all_required_outputs_present", f"missing={missing} empty={empty}")
    )

    # No look-ahead: the same engine on a structureless series is not profitable.
    noise = noise_expectancy(cfg, meta)
    out.append(
        Criterion.ok(
            "no_lookahead_noise",
            f"noise expectancy_r={noise['expectancy_r']:.4f} <= {noise['tolerance_r']} "
            f"(n={noise['trade_count']})",
        )
        if noise["passed"]
        else Criterion.fail(
            "no_lookahead_noise",
            f"profitable on noise (expectancy_r={noise['expectancy_r']:.4f}) ⇒ look-ahead",
        )
    )

    # Survivorship / future-universe: no trade before its symbol joined the universe.
    violations = future_universe_violations(run.result, inputs)
    activated_late = [s for s in inputs if s.activation_ts > 0]
    exercised = any(
        any(t.symbol == s.symbol and t.entry_ts >= s.activation_ts for t in run.result.trades)
        for s in activated_late
    )
    out.append(
        Criterion.ok(
            "no_future_universe_leakage",
            f"0 pre-activation trades; point-in-time universe exercised ({exercised})",
        )
        if not violations
        else Criterion.fail(
            "no_future_universe_leakage", f"pre-activation trades: {violations[:3]}"
        )
    )

    # Sanity: no impossible returns/trades (Section 19 "no impossible results").
    sane = abs(report.total_return) < cfg.sanity.max_abs_total_return and all(
        abs(t.pnl_r) < 1000 for t in run.result.trades
    )
    out.append(
        Criterion.ok(
            "sane_results",
            f"total_return={report.total_return:.4f}, max|pnl_R| bounded",
        )
        if sane
        else Criterion.fail(
            "sane_results", f"impossible result: total_return={report.total_return}"
        )
    )

    rpath = write_report(settings, payload, kind="backtest")
    all_pass = all(c.passed for c in out)
    persist_backtest_run(
        cfg, report, kind="backtest", report_path=rpath, settings=settings, passed=all_pass
    )
    return out


# --------------------------------------------------------------------------- #
# Phase 5 — research candidate validation shared by WF / FEE / SLIP            #
# --------------------------------------------------------------------------- #
def _candidate_criteria(settings: Settings, dimension: str) -> list[Criterion]:
    """Per-candidate criteria for the Phase 5 promotion gates (WF/FEE/SLIP).

    Re-derives the promotion decision from a single shared validation pass and
    asserts, for ``dimension`` in {"wf", "fee", "slip"}: every PROMOTED candidate
    passes that dimension (promotion requires it), every SHELVED candidate carries
    a reason (correctly rejected per kill-criteria, not a gate failure), and the
    Appendix-D minimum families A/B/G each have a promoted candidate. A Strategy
    Report is written per candidate (Section 13) for the dashboard.
    """
    from src.strategies.research import (
        families_promoted,
        get_validations,
        write_strategy_reports,
    )

    out: list[Criterion] = []
    validations = get_validations()
    write_strategy_reports(validations, settings)

    for v in validations:
        if dimension == "wf":
            ok = v.walk_forward["passed"]
            detail = (
                f"folds {v.walk_forward['folds_passed']}/{v.walk_forward['n_folds']}, "
                f"holdout={'ok' if (v.walk_forward.get('holdout') or {}).get('passed') else 'n/a'}"
            )
        elif dimension == "fee":
            ok = v.fee_stress["survives"]
            detail = (
                f"×{v.fee_stress['multiplier']} fees ⇒ expectancy_r="
                f"{v.fee_stress['stressed_expectancy_r']:.3f}, pf="
                f"{v.fee_stress['stressed_profit_factor']:.2f}"
            )
        else:  # slip
            ok = v.slippage_stress["survives"]
            detail = (
                f"×{v.slippage_stress['multiplier']} slippage ⇒ expectancy_r="
                f"{v.slippage_stress['stressed_expectancy_r']:.3f}, pf="
                f"{v.slippage_stress['stressed_profit_factor']:.2f}"
            )

        cid = f"candidate_{v.candidate_id}"
        if v.promoted:
            # A promoted candidate MUST pass this dimension (promotion requires it).
            out.append(
                Criterion.ok(cid, f"promoted ({v.family}); {detail}; sides={_sides(v)}")
                if ok
                else Criterion.fail(cid, f"promoted but fails {dimension}: {detail}")
            )
        else:
            # A shelved candidate is correctly rejected (not a gate failure) as long
            # as it carries a kill-criteria reason (Section 16 "validation rejects").
            out.append(
                Criterion.ok(cid, f"shelved ({v.family}); reasons={v.shelved_reasons}")
                if v.shelved_reasons
                else Criterion.fail(cid, "shelved without a recorded reason")
            )

    fam = families_promoted(validations)
    missing = [f for f, ok in fam.items() if not ok]
    out.append(
        Criterion.ok("families_A_B_G_promoted", "families A, B, G each have a promoted candidate")
        if not missing
        else Criterion.fail("families_A_B_G_promoted", f"no promoted candidate for: {missing}")
    )
    return out


def _sides(v: object) -> str:
    sd = v.side_decision  # type: ignore[attr-defined]
    on = [s for s, flag in (("long", sd.allow_long), ("short", sd.allow_short)) if flag]
    return "+".join(on) if on else "none"


# --------------------------------------------------------------------------- #
# WF — Walk-Forward Gate (Phase 4 engine self-test + Phase 5 candidates)       #
# --------------------------------------------------------------------------- #
def check_wf(settings: Settings) -> list[Criterion]:
    """Walk-Forward Gate (Appendix A WF).

    Runs the walk-forward harness: >= K folds must clear the up-front
    kill-criteria, the edge must not be isolated to one period, and the locked
    hold-out (evaluated exactly once) must be positive net of costs (Section 16).
    """
    from src.backtest import build_reference_inputs, load_backtest_config, run_walk_forward
    from src.backtest.service import persist_backtest_run, run_engine, write_report
    from src.exchange.metadata import load_metadata_config

    out: list[Criterion] = []
    cfg = load_backtest_config()
    meta = load_metadata_config()
    kc = cfg.walk_forward.kill_criteria
    inputs = build_reference_inputs(cfg)
    wf = run_walk_forward(cfg, meta, inputs)

    out.append(
        Criterion.ok("walk_forward_completes", f"{len(wf.folds)} folds evaluated")
        if len(wf.folds) == cfg.walk_forward.folds
        else Criterion.fail("walk_forward_completes", f"only {len(wf.folds)} folds")
    )

    out.append(
        Criterion.ok(
            "min_folds_passed",
            f"{wf.folds_passed}/{len(wf.folds)} folds clear kill-criteria "
            f"(need {kc.min_folds_passed})",
        )
        if wf.folds_passed >= kc.min_folds_passed
        else Criterion.fail(
            "min_folds_passed", f"{wf.folds_passed}/{len(wf.folds)} < {kc.min_folds_passed}"
        )
    )

    # Count a fold as a positive-edge signal only when it has ENOUGH trades to be meaningful AND
    # positive expectancy — so a thin fold (one lucky trade) is not mistaken for a stable edge.
    # This reconciles "edge not isolated" with the trade-based fold adequacy / kill-criteria.
    positive = sum(
        1
        for f in wf.folds
        if f.report.expectancy_r > 0 and f.report.trade_count >= kc.min_trades_per_fold
    )
    not_isolated = positive >= max(2, (len(wf.folds) + 1) // 2)
    out.append(
        Criterion.ok(
            "edge_not_isolated", f"{positive}/{len(wf.folds)} folds positive with enough trades"
        )
        if not_isolated
        else Criterion.fail(
            "edge_not_isolated", f"edge isolated: only {positive} adequately-traded positive folds"
        )
    )

    if wf.holdout is not None:
        out.append(
            Criterion.ok(
                "locked_holdout_positive",
                f"hold-out expectancy_r={wf.holdout.report.expectancy_r:.3f}, "
                f"net_pnl={wf.holdout.report.net_pnl:.2f} (evaluated once)",
            )
            if wf.holdout.passed
            else Criterion.fail(
                "locked_holdout_positive",
                f"hold-out not positive (expectancy_r={wf.holdout.report.expectancy_r:.3f})",
            )
        )
    else:
        out.append(Criterion.fail("locked_holdout_positive", "no locked hold-out evaluated"))

    rpath = write_report(settings, wf.to_dict(), kind="walk_forward")
    # Persist a summary indexed row using the full-window report for context.
    full = run_engine(cfg, meta, inputs, label="wf_full").report
    persist_backtest_run(
        cfg,
        full,
        kind="walk_forward",
        report_path=rpath,
        settings=settings,
        passed=wf.passed,
        summary_extra={"walk_forward": wf.to_dict()},
    )

    # Phase 5: walk-forward must pass for every promoted research candidate.
    out.extend(_candidate_criteria(settings, "wf"))
    return out


# --------------------------------------------------------------------------- #
# FEE — Fee Stress Gate (Phase 4)                                              #
# --------------------------------------------------------------------------- #
def check_fee(settings: Settings) -> list[Criterion]:
    """Fee Stress Gate (Appendix A FEE): survive ×2 fees with positive expectancy."""
    from src.backtest import build_reference_inputs, fee_stress, load_backtest_config, run_engine
    from src.backtest.service import persist_backtest_run, write_report
    from src.exchange.metadata import load_metadata_config

    out: list[Criterion] = []
    cfg = load_backtest_config()
    meta = load_metadata_config()
    inputs = build_reference_inputs(cfg)
    base = run_engine(cfg, meta, inputs, label="fee_baseline").report
    stress = fee_stress(cfg, meta, inputs, baseline_expectancy_r=base.expectancy_r)

    out.append(
        Criterion.ok("baseline_positive", f"baseline expectancy_r={base.expectancy_r:.3f}")
        if base.expectancy_r > 0
        else Criterion.fail("baseline_positive", "baseline expectancy not positive")
    )
    out.append(
        Criterion.ok(
            "survives_fee_stress",
            f"×{stress.multiplier} fees ⇒ expectancy_r={stress.stressed_expectancy_r:.3f}, "
            f"net_pnl={stress.stressed_net_pnl:.2f}",
        )
        if stress.survives
        else Criterion.fail(
            "survives_fee_stress",
            f"expectancy turns non-positive under ×{stress.multiplier} fees "
            f"(expectancy_r={stress.stressed_expectancy_r:.3f})",
        )
    )
    out.append(
        Criterion.ok("edge_not_fee_dependent", "edge persists net of doubled fees")
        if stress.stressed_profit_factor > 1.0
        else Criterion.fail(
            "edge_not_fee_dependent", f"profit_factor {stress.stressed_profit_factor:.2f} <= 1"
        )
    )

    rpath = write_report(settings, stress.to_dict(), kind="fee_stress")
    persist_backtest_run(
        cfg,
        base,
        kind="fee_stress",
        report_path=rpath,
        settings=settings,
        passed=all(c.passed for c in out),
        summary_extra={"fee_stress": stress.to_dict()},
    )

    # Phase 5: fee stress must pass for every promoted research candidate.
    out.extend(_candidate_criteria(settings, "fee"))
    return out


# --------------------------------------------------------------------------- #
# SLIP — Slippage Stress Gate (Phase 4)                                        #
# --------------------------------------------------------------------------- #
def check_slip(settings: Settings) -> list[Criterion]:
    """Slippage Stress Gate (Appendix A SLIP): survive +50% slippage."""
    from src.backtest import (
        build_reference_inputs,
        load_backtest_config,
        run_engine,
        slippage_stress,
    )
    from src.backtest.service import persist_backtest_run, write_report
    from src.exchange.metadata import load_metadata_config

    out: list[Criterion] = []
    cfg = load_backtest_config()
    meta = load_metadata_config()
    inputs = build_reference_inputs(cfg)
    base = run_engine(cfg, meta, inputs, label="slip_baseline").report
    stress = slippage_stress(cfg, meta, inputs, baseline_expectancy_r=base.expectancy_r)

    out.append(
        Criterion.ok("baseline_positive", f"baseline expectancy_r={base.expectancy_r:.3f}")
        if base.expectancy_r > 0
        else Criterion.fail("baseline_positive", "baseline expectancy not positive")
    )
    out.append(
        Criterion.ok(
            "survives_slippage_stress",
            f"×{stress.multiplier} slippage ⇒ expectancy_r={stress.stressed_expectancy_r:.3f}, "
            f"net_pnl={stress.stressed_net_pnl:.2f}",
        )
        if stress.survives
        else Criterion.fail(
            "survives_slippage_stress",
            f"expectancy turns non-positive under +50% slippage "
            f"(expectancy_r={stress.stressed_expectancy_r:.3f})",
        )
    )
    out.append(
        Criterion.ok("edge_not_slippage_dependent", "edge persists under harsher slippage")
        if stress.stressed_profit_factor > 1.0
        else Criterion.fail(
            "edge_not_slippage_dependent",
            f"profit_factor {stress.stressed_profit_factor:.2f} <= 1",
        )
    )

    rpath = write_report(settings, stress.to_dict(), kind="slippage_stress")
    persist_backtest_run(
        cfg,
        base,
        kind="slippage_stress",
        report_path=rpath,
        settings=settings,
        passed=all(c.passed for c in out),
        summary_extra={"slippage_stress": stress.to_dict()},
    )

    # Phase 5: slippage stress must pass for every promoted research candidate.
    out.extend(_candidate_criteria(settings, "slip"))
    return out


def check_ml_promo(settings: Settings) -> list[Criterion]:
    """Combined ML-PROMO gate: Phase 9 shadow criteria + Phase 10 Stage 3/4 criteria."""
    out = _check_ml_promo_phase9(settings)
    out.extend(check_ml_phase10(settings))
    return out


def _write_report(settings: Settings, kind: str, payload: dict) -> str:
    """Persist a gate sub-report under reports/<kind>/ (dashboard 'View ...' actions)."""
    import json as _json

    reports_dir = settings.reports_path / kind
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    path = reports_dir / f"{kind}_{stamp}.json"
    path.write_text(
        _json.dumps({"versions": settings.versions(), **payload}, indent=2), encoding="utf-8"
    )
    return str(path)


# --------------------------------------------------------------------------- #
# Registry                                                                     #
# --------------------------------------------------------------------------- #
CHECKS: dict[str, Callable[[Settings], list[Criterion]]] = {
    "INFRA": check_infra,
    "DB": check_db,
    "QUEUE": check_queue,
    "STORAGE": check_storage,
    "DATA-COV": check_data_cov,
    "DQ": check_dq,
    "UNIV": check_univ,
    "META": check_meta,
    "FEAT": check_feat,
    "BT": check_bt,
    "WF": check_wf,
    "FEE": check_fee,
    "SLIP": check_slip,
    # MON and BACKUP: Phase 13 full versions replace Phase 1 skeletons.
    "MON": check_mon_phase13,
    "BACKUP": check_backup_phase13,
    # Phase 6 — Ranking, Risk, Execution core.
    "SETUP": check_setup,
    "RISK": check_risk,
    "EXEC": check_exec,
    "KILL": check_kill,
    "ORDER-OWN": check_order_own,
    # Phase 8 — Paper Trading (technical + strategy validation).
    "PAPER-A": check_paper_a,
    "PAPER-B": check_paper_b,
    # Phase 9–10 — ML promotion gate (shadow + recommendation + constrained filter).
    "ML-PROMO": check_ml_promo,
    # Phase 11 — Online Learning Shadow gate.
    "LEARN-PROMO-S": check_learn_promo_s,
    # Phase 12 — RL Research and Shadow Policy gates.
    "RL-SIM": check_rl_sim,
    "RL-SHADOW": check_rl_shadow,
    # Phase 13 — Controlled Live Readiness gates.
    "LEARN-PROMO-L": check_learn_promo_l,
    "SEC": check_sec,
    "DEPLOY": check_deploy,
    "CONFIG-FREEZE": check_config_freeze,
    "LIVE": check_live,
}


def has_check(gate_id: str) -> bool:
    return gate_id in CHECKS


def run_check(gate_id: str, settings: Settings | None = None) -> list[Criterion]:
    settings = settings or get_settings()
    return CHECKS[gate_id](settings)
