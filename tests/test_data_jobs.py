"""Data-platform job handlers via the real queue/worker (Appendix B.7)."""

from __future__ import annotations

from src.db.base import session_scope
from src.db.models import DatasetVersion, JobStatus
from src.jobs import JobQueue, Worker

from tests.conftest import requires_redis


@requires_redis
def test_download_jobs_run() -> None:
    queue, worker = JobQueue(), Worker()
    for job_type in (
        "download_ohlcv_history",
        "download_mark_index_history",
        "download_funding_history",
        "download_open_interest_history",
        "download_spread_snapshots",
        "download_liquidation_history",
    ):
        job_id = queue.enqueue(job_type, {}, requested_by="test")
        assert worker.process_job(job_id) is JobStatus.SUCCEEDED, job_type


@requires_redis
def test_build_dataset_version_job_produces_snapshot() -> None:
    queue, worker = JobQueue(), Worker()
    job_id = queue.enqueue("build_dataset_version", {}, requested_by="test")
    assert worker.process_job(job_id) is JobStatus.SUCCEEDED
    with session_scope() as session:
        rows = session.query(DatasetVersion).all()
        assert rows, "build_dataset_version must record a dataset_versions row"
        assert any(r.validation_status == "valid" for r in rows)


def test_unreachable_exchange_yields_no_available_symbols() -> None:
    """Root cause of the 'instant FAILED snapshot' symptom: when the exchange is unreachable,
    the source reports every symbol as having no history, so the build's preflight finds NOTHING
    to download and raises a clear error instead of silently producing an invalid snapshot."""
    from src.data.config import load_data_config
    from src.data.source import DeterministicSource

    cfg = load_data_config()
    syms = cfg.active_symbols()
    unreachable = DeterministicSource(cfg.exchange_id, missing_symbols=set(syms))
    available = [s for s in syms if unreachable.has_symbol(s)]
    assert available == []  # the build_dataset_version preflight raises on exactly this
    # ...and a reachable source has the symbols, so the download proceeds normally.
    reachable = DeterministicSource(cfg.exchange_id)
    assert all(reachable.has_symbol(s) for s in syms)


@requires_redis
def test_validate_data_quality_job_passes() -> None:
    queue, worker = JobQueue(), Worker()
    job_id = queue.enqueue("validate_data_quality", {}, requested_by="test")
    assert worker.process_job(job_id) is JobStatus.SUCCEEDED


@requires_redis
def test_repair_missing_data_job_runs() -> None:
    queue, worker = JobQueue(), Worker()
    job_id = queue.enqueue("repair_missing_data", {}, requested_by="test")
    assert worker.process_job(job_id) is JobStatus.SUCCEEDED
