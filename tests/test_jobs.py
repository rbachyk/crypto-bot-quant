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
def test_atomic_claim_skips_already_claimed_job() -> None:
    """A job not in QUEUED state (already claimed by another worker / already terminal) is NOT
    run again — the atomic QUEUED→RUNNING claim returns the existing status without executing,
    preventing double execution under reaper/heartbeat races."""
    queue, worker = JobQueue(), Worker()
    job_id = queue.enqueue("selftest_echo", {"steps": 1}, requested_by="test")
    # Simulate another worker having already claimed it.
    with session_scope() as session:
        session.get(Job, job_id).status = JobStatus.RUNNING
    assert worker.process_job(job_id) is JobStatus.RUNNING  # skipped, not re-run
    # A terminal job is likewise never re-run.
    with session_scope() as session:
        session.get(Job, job_id).status = JobStatus.SUCCEEDED
    assert worker.process_job(job_id) is JobStatus.SUCCEEDED


@requires_redis
def test_fencing_token_blocks_superseded_terminal_write() -> None:
    """A run that was falsely reaped (its fencing token cleared/replaced) must NOT write a
    terminal state over the re-run — prevents double-persist on heartbeat-starvation false reaps."""
    queue, worker = JobQueue(), Worker()
    job_id = queue.enqueue("selftest_echo", {}, requested_by="test")
    with session_scope() as session:
        j = session.get(Job, job_id)
        j.status = JobStatus.RUNNING
        j.run_token = "real_token"
    # A stale run (different token) tries to finish → superseded, no-op.
    worker._finish_succeeded(job_id, {}, run_token="stale_token")
    assert _load(job_id).status is JobStatus.RUNNING  # NOT overwritten to SUCCEEDED
    # The owning run finishes correctly.
    worker._finish_succeeded(job_id, {}, run_token="real_token")
    assert _load(job_id).status is JobStatus.SUCCEEDED


@requires_redis
def test_still_owns_reflects_run_token() -> None:
    """ctx.still_owns() is True only while THIS run's fencing token matches the DB — a long
    handler polls it to stop when reaped/superseded (no double live execution)."""
    from src.jobs.context import JobContext

    queue = JobQueue()
    job_id = queue.enqueue("selftest_echo", {}, requested_by="test")
    with session_scope() as session:
        session.get(Job, job_id).run_token = "tok"
    assert JobContext(job_id, {}, queue.redis, run_token="tok").still_owns() is True
    assert JobContext(job_id, {}, queue.redis, run_token="other").still_owns() is False


@requires_redis
def test_retry_with_attempts_eventually_succeeds() -> None:
    queue, worker = JobQueue(), Worker()
    # max_attempts=2 means a transient failure would be retried once.
    job_id = queue.enqueue("selftest_fail", {}, requested_by="test", max_attempts=2)
    # First processing requeues (attempt 1 < 2).
    assert worker.process_job(job_id) is JobStatus.QUEUED
    # Second processing exhausts attempts and fails.
    assert worker.process_job(job_id) is JobStatus.FAILED


@requires_redis
def test_run_backtest_job_persists_result() -> None:
    # The dashboard-triggered backtest runs as a background job and writes a BacktestRun
    # index row the dashboard reads.
    from sqlalchemy import select
    from src.db.models import BacktestRun
    from src.jobs.routing import queue_class

    assert queue_class("run_backtest") == "backtest"  # routed to the dedicated worker
    queue, worker = JobQueue(), Worker()
    job_id = queue.enqueue("run_backtest", {"label": "unit_test_bt"}, requested_by="test")
    assert worker.process_job(job_id) is JobStatus.SUCCEEDED
    with session_scope() as session:
        rows = session.execute(select(BacktestRun)).scalars().all()
        mine = [r for r in rows if (r.summary or {}).get("label") == "unit_test_bt"]
        assert mine, "run_backtest must persist a BacktestRun row"
        assert mine[0].kind == "backtest"
        assert mine[0].trade_count > 0


def test_basket_paper_session_routes_to_live_worker_and_is_registered() -> None:
    """The continuous cross-sectional paper loop must ride the dedicated `live` queue (like
    run_live_session, so it never blocks/gets blocked) and have a registered handler."""
    import src.jobs.handlers  # noqa: F401 - import registers all handlers
    from src.jobs.registry import registry
    from src.jobs.routing import queue_class

    assert queue_class("run_basket_paper_session") == "live"
    assert registry.get("run_basket_paper_session") is not None  # raises KeyError if missing


@requires_redis
def test_enqueue_routes_to_class_queue() -> None:
    # Heavy ML / data / gate jobs are routed to dedicated class queues (B.13 isolation),
    # so a heavy job never lands on a light worker's queue.
    from src.jobs.routing import queue_class, queue_key

    assert queue_class("train_ml_models") == "ml"
    assert queue_class("download_ohlcv_history") == "data"
    assert queue_class("run_gate") == "gates"
    assert queue_class("selftest_echo") == "default"

    queue = JobQueue()
    r = queue.redis
    job_id = queue.enqueue("train_ml_models", {}, requested_by="test")
    try:
        # The id is on the ml queue, and NOT on the default queue.
        assert r.lrem(queue_key("ml"), 0, job_id) == 1
        assert r.lrem(queue_key("default"), 0, job_id) == 0
    finally:
        queue.cancel(job_id)


@requires_redis
def test_reaper_requeues_orphan_of_dead_worker() -> None:
    # A job claimed by a worker that then dies (no liveness beacon) must be recovered from
    # its processing list and put back on the queue — never silently lost.
    from datetime import UTC, datetime

    from src.jobs.routing import processing_key, queue_class, queue_key
    from src.jobs.worker import reap_orphaned_jobs

    queue = JobQueue()
    r = queue.redis
    cls = queue_class("selftest_echo")
    job_id = queue.enqueue("selftest_echo", {"steps": 1}, requested_by="test")

    # Simulate a dead worker mid-job: id sits in its processing list, DB row is RUNNING,
    # and there is NO beacon for that worker.
    dead = "deadworker:host:999:abcdef01"
    pkey = processing_key(dead)
    r.lrem(queue_key(cls), 0, job_id)  # it was claimed (removed from the queue)
    r.lpush(pkey, job_id)
    with session_scope() as s:
        j = s.get(Job, job_id)
        assert j is not None
        j.status = JobStatus.RUNNING
        j.started_at = datetime.now(UTC)
        j.attempts = 1

    assert reap_orphaned_jobs(r) >= 1
    assert r.llen(pkey) == 0  # processing list drained
    assert _load(job_id).status is JobStatus.QUEUED  # DB row recovered
    assert r.lrem(queue_key(cls), 0, job_id) == 1  # back on its class queue
    queue.cancel(job_id)


@requires_redis
def test_reaper_leaves_live_workers_alone() -> None:
    # A worker with a current beacon is alive; its in-flight job must NOT be reaped.
    from src.jobs.routing import processing_key, worker_key
    from src.jobs.worker import reap_orphaned_jobs

    queue = JobQueue()
    r = queue.redis
    job_id = queue.enqueue("selftest_echo", {}, requested_by="test")
    alive = "aliveworker:host:1:beefbeef"
    r.set(worker_key(alive), alive, ex=60)  # present beacon → alive
    r.lpush(processing_key(alive), job_id)
    try:
        reap_orphaned_jobs(r)
        assert r.lrem(processing_key(alive), 0, job_id) == 1  # still held, not reaped
    finally:
        r.delete(worker_key(alive))
        queue.cancel(job_id)
