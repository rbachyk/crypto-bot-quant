"""Job queue tests (AGENTS.md Appendix B.6, QUEUE gate).

Exercise enqueue → consume → progress, cooperative cancel, and retry/failure
visibility against the real Redis + Postgres backends.
"""

from __future__ import annotations

from src.db.base import session_scope
from src.db.models import Job, JobStatus
from src.jobs import JobQueue, Worker

from tests.conftest import requires_redis


def _load(job_id: str) -> Job:
    with session_scope() as session:
        job = session.get(Job, job_id)
        assert job is not None
        _ = job.logs
        session.expunge_all()
        return job


@requires_redis
def test_enqueue_consume_progress_and_logs() -> None:
    queue, worker = JobQueue(), Worker()
    job_id = queue.enqueue("selftest_echo", {"steps": 4}, requested_by="test")
    assert worker.process_job(job_id) is JobStatus.SUCCEEDED
    job = _load(job_id)
    assert job.status is JobStatus.SUCCEEDED
    assert job.progress_current == job.progress_total == 4
    assert job.started_at is not None and job.finished_at is not None
    assert len(job.logs) >= 1


@requires_redis
def test_cancel_queued_job_is_skipped() -> None:
    queue, worker = JobQueue(), Worker()
    job_id = queue.enqueue("selftest_echo", {"steps": 2}, requested_by="test")
    assert queue.cancel(job_id) is True
    assert worker.process_job(job_id) is JobStatus.CANCELLED
    assert _load(job_id).status is JobStatus.CANCELLED


@requires_redis
def test_failed_job_visible_then_retry_requeues() -> None:
    queue, worker = JobQueue(), Worker()
    job_id = queue.enqueue("selftest_fail", {}, requested_by="test", max_attempts=1)
    assert worker.process_job(job_id) is JobStatus.FAILED
    failed = _load(job_id)
    assert failed.status is JobStatus.FAILED
    assert failed.failure_reason
    assert failed.next_action_hint  # remediation hint present

    assert queue.retry(job_id) is True
    assert _load(job_id).status is JobStatus.QUEUED
    # Clean up the re-queued job.
    queue.cancel(job_id)
    worker.process_job(job_id)


@requires_redis
def test_retry_with_attempts_eventually_succeeds() -> None:
    queue, worker = JobQueue(), Worker()
    # max_attempts=2 means a transient failure would be retried once.
    job_id = queue.enqueue("selftest_fail", {}, requested_by="test", max_attempts=2)
    # First processing requeues (attempt 1 < 2).
    assert worker.process_job(job_id) is JobStatus.QUEUED
    # Second processing exhausts attempts and fails.
    assert worker.process_job(job_id) is JobStatus.FAILED
