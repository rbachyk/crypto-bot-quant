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
