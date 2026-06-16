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
from src.universe import UniverseBuilder

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


@job_handler("build_symbol_universe")
def _build_symbol_universe(ctx: JobContext, params: dict) -> dict:
    """Skeleton universe build: persist a versioned universe snapshot whose
    members are all ``research_only`` (Section 9)."""
    builder = UniverseBuilder(get_adapter(params.get("exchange_id")))
    ctx.log("building universe skeleton")
    with session_scope() as session:
        uv = builder.build(session, version=params.get("version"))
        version = uv.version
    ctx.progress(1, 1, f"universe {version} built")
    return {"message": f"built universe {version}", "version": version}


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
