"""Built-in job handlers (AGENTS.md Appendix B.7).

Phase 1 ships the gate-runner jobs, the data/universe skeleton jobs
(``sync_exchange_metadata``, ``build_symbol_universe``), backup/restore jobs,
and a few ``selftest_*`` handlers the QUEUE gate uses to prove the queue works.
Heavy research/ML/RL jobs are added in their phases.
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime

from src.config.settings import REPO_ROOT
from src.db.base import session_scope
from src.db.models import ExchangeMetadata, VerificationStatus
from src.exchange import get_adapter
from src.jobs.context import JobContext
from src.jobs.registry import job_handler

_registered = False


def ensure_handlers_registered() -> None:
    """Importing this module registers all handlers; call to be explicit."""
    global _registered
    _registered = True


# --------------------------------------------------------------------------- #
# Self-test handlers (used by the QUEUE gate)                                  #
# --------------------------------------------------------------------------- #
@job_handler("selftest_echo")
def _selftest_echo(ctx: JobContext, params: dict) -> dict:
    steps = int(params.get("steps", 1))
    ctx.log("selftest_echo starting")
    for i in range(steps):
        ctx.check_cancelled()
        ctx.progress(i + 1, steps, f"step {i + 1}/{steps}")
    ctx.log("selftest_echo done")
    return {"message": f"echoed {steps} steps", "steps": steps}


@job_handler("selftest_fail")
def _selftest_fail(ctx: JobContext, params: dict) -> dict:
    ctx.log("selftest_fail will raise", level="WARNING")
    raise RuntimeError("intentional failure for retry/failure-visibility test")


# --------------------------------------------------------------------------- #
# Data / universe skeleton jobs                                               #
# --------------------------------------------------------------------------- #
@job_handler("sync_exchange_metadata")
def _sync_exchange_metadata(ctx: JobContext, params: dict) -> dict:
    """Skeleton metadata sync: fetch symbols from the adapter and persist
    placeholder, ``[UNVERIFIED]`` metadata snapshots (Section 6)."""
    adapter = get_adapter(params.get("exchange_id"))
    symbols = adapter.fetch_symbols()
    ctx.log(f"syncing metadata for {len(symbols)} symbols from {adapter.exchange_id}")
    version = params.get("metadata_version", "meta_0001")
    with session_scope() as session:
        for i, symbol in enumerate(symbols):
            ctx.check_cancelled()
            meta = adapter.fetch_metadata(symbol)
            session.add(
                ExchangeMetadata(
                    exchange_id=adapter.exchange_id,
                    symbol=symbol,
                    metadata_version=version,
                    verification_status=VerificationStatus.UNVERIFIED,
                    source="skeleton",
                    fetched_at=datetime.now(UTC),
                    raw=meta.raw,
                )
            )
            ctx.progress(i + 1, len(symbols), f"synced {symbol}")
    return {"message": f"synced {len(symbols)} symbols (UNVERIFIED)", "symbols": symbols}


@job_handler("verify_exchange_metadata")
def _verify_exchange_metadata(ctx: JobContext, params: dict) -> dict:
    """Apply operator-verified metadata from ``configs/metadata.yaml`` (Section 6
    verification workflow step 3): persist ``[VERIFIED]`` metadata rows the META
    gate reads. Idempotent."""
    from src.exchange import load_metadata_config, sync_verified_metadata

    cfg = load_metadata_config()
    ctx.log(
        f"applying [VERIFIED] metadata for {len(cfg.symbols())} symbols ({cfg.metadata_version})"
    )
    with session_scope() as session:
        written = sync_verified_metadata(session, cfg)
    ctx.progress(1, 1, f"{written} verified")
    return {
        "message": f"verified {written} symbols ({cfg.metadata_version})",
        "symbols": cfg.symbols(),
    }


@job_handler("build_symbol_universe")
def _build_symbol_universe(ctx: JobContext, params: dict) -> dict:
    """Build a versioned universe: filter every candidate (Section 9) and promote
    passing symbols to ``active``, logging entering/leaving symbols."""
    from src.universe import UniverseManager

    ctx.log("building dynamic universe (Phase 3 filters)")
    with session_scope() as session:
        result = UniverseManager().build(session)
        version = result.version
        active = result.active_symbols
        changes = len(result.changes)
    ctx.progress(1, 1, f"universe {version}: {len(active)} active, {changes} changes")
    return {
        "message": f"built universe {version} ({len(active)} active)",
        "version": version,
        "active": active,
        "changes": changes,
    }


@job_handler("build_feature_store")
def _build_feature_store(ctx: JobContext, params: dict) -> dict:
    """Build the immutable feature store from the current dataset snapshot for the
    active universe through the single feature code path (Section 10)."""
    from src.data import DataPlatform, load_data_config
    from src.features import FeatureStore
    from src.universe import UniverseManager, latest_active_symbols

    ctx.log("ensuring data snapshot + building feature store")
    platform = DataPlatform(cfg=load_data_config())
    run = platform.run_full(repair=True, source_jobs=["job:build_feature_store"])
    with session_scope() as session:
        uni = UniverseManager().build(session)
        active = latest_active_symbols(session) or uni.active_symbols
        build = FeatureStore().build(
            active,
            run.snapshot.snapshot_id,
            universe_version=uni.version,
            session=session,
            source_jobs=["job:build_feature_store"],
        )
        snapshot_id = build.feature_snapshot_id
        rows = build.total_rows
    ctx.progress(1, 1, f"features {snapshot_id} ({rows} rows)")
    return {
        "message": f"built features {snapshot_id} ({rows} rows)",
        "feature_snapshot_id": snapshot_id,
        "checksum": build.checksum,
        "rows": rows,
    }


@job_handler("run_feature_leakage_test")
def _run_feature_leakage_test(ctx: JobContext, params: dict) -> dict:
    """Run the synthetic-data leakage test (Section 16 / FEAT gate)."""
    from src.features import load_feature_config, synthetic_leakage_report

    ctx.log("running synthetic-data leakage test")
    report = synthetic_leakage_report(load_feature_config())
    ctx.progress(
        1, 1, f"leakage {'PASS' if report['passed'] else 'FAIL'} (|z|={abs(report['z']):.2f})"
    )
    return {"message": f"leakage {'PASS' if report['passed'] else 'FAIL'}", **report}


# --------------------------------------------------------------------------- #
# Data platform jobs (Appendix B.7 data jobs; Phase 2)                        #
# --------------------------------------------------------------------------- #
def _download_series(ctx: JobContext, data_types: list[str]) -> dict:
    """Shared body for the per-series historical download jobs."""
    from src.data import DataPlatform, load_data_config

    cfg = load_data_config()
    platform = DataPlatform(cfg=cfg)
    keys = [k for k in cfg.all_required_keys() if k.data_type in data_types]
    written = 0
    for i, key in enumerate(keys):
        ctx.check_cancelled()
        written += platform.download(key)
        ctx.progress(i + 1, len(keys), f"downloaded {key.label()}")
    ctx.log(f"downloaded {written} rows across {len(keys)} series ({', '.join(data_types)})")
    return {"message": f"downloaded {written} rows", "series": len(keys), "rows": written}


@job_handler("download_ohlcv_history")
def _download_ohlcv_history(ctx: JobContext, params: dict) -> dict:
    return _download_series(ctx, ["ohlcv"])


@job_handler("download_mark_index_history")
def _download_mark_index_history(ctx: JobContext, params: dict) -> dict:
    return _download_series(ctx, ["mark", "index"])


@job_handler("download_funding_history")
def _download_funding_history(ctx: JobContext, params: dict) -> dict:
    return _download_series(ctx, ["funding"])


@job_handler("download_open_interest_history")
def _download_open_interest_history(ctx: JobContext, params: dict) -> dict:
    return _download_series(ctx, ["open_interest"])


@job_handler("download_spread_snapshots")
def _download_spread_snapshots(ctx: JobContext, params: dict) -> dict:
    return _download_series(ctx, ["spread"])


@job_handler("download_liquidation_history")
def _download_liquidation_history(ctx: JobContext, params: dict) -> dict:
    """Liquidation data is "if available" (Section 8). The offline source does
    not provide it; the job records that it is unavailable rather than failing,
    so liquidation is simply not a required series this phase."""
    ctx.log("liquidation history unavailable for the skeleton source; skipping")
    ctx.progress(1, 1, "liquidation unavailable")
    return {"message": "liquidation history unavailable (not required this phase)", "rows": 0}


@job_handler("repair_missing_data")
def _repair_missing_data(ctx: JobContext, params: dict) -> dict:
    """Detect gaps and backfill only the missing ranges (safe gap repair)."""
    from src.data import DataPlatform, load_data_config

    cfg = load_data_config()
    platform = DataPlatform(cfg=cfg)
    keys = cfg.all_required_keys()
    written = 0
    remaining = 0
    for i, key in enumerate(keys):
        ctx.check_cancelled()
        result = platform.ingestor.repair(key, cfg.window_start_ms, cfg.window_end_ms)
        written += result.rows_written
        remaining += result.gaps_after
        ctx.progress(i + 1, len(keys), f"repaired {key.label()}")
    ctx.log(f"repaired {written} rows; {remaining} timestamps still missing")
    return {"message": f"repaired {written} rows", "rows": written, "still_missing": remaining}


@job_handler("validate_data_quality")
def _validate_data_quality(ctx: JobContext, params: dict) -> dict:
    """Run the Section 23 data-quality checks and persist the report."""
    from src.data import DataPlatform, load_data_config

    platform = DataPlatform(cfg=load_data_config())
    ctx.log("validating data quality")
    report = platform.validate()
    path = platform.write_quality_report(report, params.get("dataset_version"))
    ctx.progress(1, 1, f"data quality: {'PASS' if report.passed else 'FAIL'}")
    return {
        "message": f"data quality {'PASS' if report.passed else 'FAIL'}",
        "passed": report.passed,
        "critical": len(report.critical),
        "artifact_uri": path,
    }


@job_handler("build_dataset_version")
def _build_dataset_version(ctx: JobContext, params: dict) -> dict:
    """Ensure coverage, validate, and produce an immutable dataset snapshot."""
    from src.data import DataPlatform, load_data_config

    platform = DataPlatform(cfg=load_data_config())
    ctx.log("ensuring coverage + building dataset snapshot")
    run = platform.run_full(repair=True, source_jobs=["job:build_dataset_version"])
    ctx.progress(1, 1, f"snapshot {run.snapshot.snapshot_id}")
    return {
        "message": f"dataset {run.snapshot.snapshot_id} (covered={run.coverage.covered})",
        "dataset_version": run.snapshot.snapshot_id,
        "covered": run.coverage.covered,
        "validation_passed": run.validation.passed,
        "artifact_uri": run.report_path,
    }


# --------------------------------------------------------------------------- #
# Gate runner jobs                                                            #
# --------------------------------------------------------------------------- #
@job_handler("run_gate")
def _run_gate(ctx: JobContext, params: dict) -> dict:
    from src.gates import GateRunner

    gate_id = params["gate_id"]
    ctx.log(f"running gate {gate_id}")
    result = GateRunner().run(gate_id)
    ctx.progress(1, 1, f"{gate_id}: {result.overall}")
    return {"message": f"{gate_id}: {result.overall}", "artifact_uri": result.report_path}


@job_handler("run_all_gates")
def _run_all_gates(ctx: JobContext, params: dict) -> dict:
    from src.gates import GateRunner

    ctx.log("running all gates in dependency order")
    results = GateRunner().run_all()
    summary = {r.gate_id: r.overall for r in results}
    ctx.progress(len(results), len(results), "all gates evaluated")
    return {"message": "ran all gates", "summary": summary}


# --------------------------------------------------------------------------- #
# Backup / restore jobs (skeleton; full BACKUP gate in Phase 13)              #
# --------------------------------------------------------------------------- #
@job_handler("run_backup_check")
def _run_backup_check(ctx: JobContext, params: dict) -> dict:
    script = REPO_ROOT / "scripts" / "backup_db.sh"
    ctx.log(f"running backup script {script}")
    proc = subprocess.run(  # noqa: S603
        ["bash", str(script)], capture_output=True, text=True, cwd=str(REPO_ROOT)
    )
    ctx.log(proc.stdout[-2000:] or "(no stdout)")
    if proc.returncode != 0:
        raise RuntimeError(f"backup failed: {proc.stderr[-500:]}")
    return {"message": "backup completed"}


@job_handler("run_restore_test_check")
def _run_restore_test_check(ctx: JobContext, params: dict) -> dict:
    script = REPO_ROOT / "scripts" / "restore_test.sh"
    ctx.log(f"running restore-test script {script}")
    proc = subprocess.run(  # noqa: S603
        ["bash", str(script)], capture_output=True, text=True, cwd=str(REPO_ROOT)
    )
    ctx.log(proc.stdout[-2000:] or "(no stdout)")
    if proc.returncode != 0:
        raise RuntimeError(f"restore test failed: {proc.stderr[-500:]}")
    return {"message": "restore test passed"}


ensure_handlers_registered()
