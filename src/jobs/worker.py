"""Job worker (AGENTS.md Appendix B.6/B.13).

Consumes job ids from per-class Redis queues, runs the registered handler with a
:class:`JobContext`, and maintains the durable job record: status transitions,
timestamps, attempts/retry, failure reason + next-action hint, and a final progress
update. Runs as a dedicated process (``make run-worker-*``), never inside the API (B.17).

Reliability (B.13): consumption is ``RPOPLPUSH job-queue -> per-worker processing list``
so a popped id is never lost — it lives in the processing list until the job reaches a
terminal/requeue state, then it's ``LREM``'d. A liveness beacon (TTL key, refreshed by a
small heartbeat thread independent of job duration) lets the reaper detect a dead worker
and re-queue its in-flight jobs. The heartbeat thread carries no job work (B.17), only
liveness. Isolation (B.13): a worker serves only the queue classes it is configured for, so
heavy ML/RL/backtest jobs run on dedicated workers and never starve light data/gate jobs.
"""

from __future__ import annotations

import os
import socket
import threading
import time
import uuid
from contextlib import suppress
from datetime import UTC, datetime
from typing import cast

import redis
import structlog
from sqlalchemy import update

from src.config import Settings, get_settings
from src.db.base import session_scope
from src.db.models import Job, JobStatus
from src.jobs.context import JobCancelled, JobContext, _cancel_key
from src.jobs.events import format_progress, publish_job_event
from src.jobs.registry import registry
from src.jobs.routing import (
    PROCESSING_PREFIX,
    REAPER_LOCK_KEY,
    parse_queue_classes,
    processing_key,
    queue_class,
    queue_key,
    worker_key,
)

_log = structlog.get_logger("worker")


def _recover_orphan(job_id: str, redis_client: redis.Redis) -> bool:
    """Re-queue a single job orphaned by a dead worker. Terminal jobs (the worker died
    *after* finishing the DB transition but before removing the id) are dropped, never
    re-run. A non-terminal job is reset to QUEUED with a fresh attempt granted so even a
    ``max_attempts=1`` job can run again, then routed back to its class queue."""
    with session_scope() as session:
        job = session.get(Job, job_id)
        if job is None:
            return False
        if job.status in (
            JobStatus.SUCCEEDED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
            JobStatus.EXPIRED,
        ):
            return False
        job.status = JobStatus.QUEUED
        job.started_at = None
        if job.max_attempts <= job.attempts:
            job.max_attempts = job.attempts + 1
        job_type = job.job_type
    redis_client.delete(_cancel_key(job_id))
    redis_client.lpush(queue_key(queue_class(job_type)), job_id)
    return True


def reap_orphaned_jobs(redis_client: redis.Redis, settings: Settings | None = None) -> int:
    """Re-queue jobs left in the processing lists of workers whose liveness beacon has
    expired (crashed / SIGKILL'd mid-job). Guarded by a short Redis lock so concurrent
    workers don't double-recover. Returns the number of jobs re-queued."""
    _ = settings  # reserved; behaviour is driven by the beacon TTL set on each worker
    if not redis_client.set(REAPER_LOCK_KEY, "1", nx=True, ex=15):
        return 0
    requeued = 0
    try:
        for pkey in redis_client.scan_iter(match=f"{PROCESSING_PREFIX}:*", count=100):
            key = cast(str, pkey)  # clients here use decode_responses=True
            worker_id = key.split(":", 2)[2]
            if redis_client.exists(worker_key(worker_id)):
                continue  # owner still alive — leave its in-flight job alone
            while True:
                job_id = cast("str | None", redis_client.lpop(key))  # drain processing list
                if not job_id:
                    break
                if _recover_orphan(job_id, redis_client):
                    requeued += 1
                    _log.warning("job_reaped", job_id=job_id, dead_worker=worker_id)
    finally:
        redis_client.delete(REAPER_LOCK_KEY)
    return requeued


class Worker:
    """Single-process job worker. Concurrency is one job at a time per process;
    scale by running multiple worker processes (B.13)."""

    def __init__(
        self,
        settings: Settings | None = None,
        redis_client: redis.Redis | None = None,
        *,
        queues: str | None = None,
    ):
        self.settings = settings or get_settings()
        self._redis = redis_client or redis.Redis.from_url(
            self.settings.redis_url, decode_responses=True
        )
        self._queues = parse_queue_classes(
            queues if queues is not None else self.settings.worker_queues
        )
        service = os.environ.get("SERVICE_NAME", "worker")
        self._worker_id = f"{service}:{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
        self._processing_key = processing_key(self._worker_id)
        self._stop = False
        self._hb_stop = threading.Event()
        self._hb_thread: threading.Thread | None = None

    @property
    def queues(self) -> list[str]:
        """The queue classes this worker serves, in priority order."""
        return list(self._queues)

    # -- liveness beacon ------------------------------------------------- #
    def _beat(self) -> None:
        """Refresh the TTL beacon so a busy worker (even mid-long-job) is seen as alive."""
        with suppress(redis.RedisError):
            self._redis.set(
                worker_key(self._worker_id),
                self._worker_id,
                ex=self.settings.worker_heartbeat_ttl_sec,
            )

    def _heartbeat_loop(self) -> None:
        while not self._hb_stop.is_set():
            self._beat()
            self._hb_stop.wait(self.settings.worker_heartbeat_sec)

    # -- main loop ------------------------------------------------------- #
    def run(self, *, max_jobs: int | None = None) -> int:
        """Run the consume loop. ``max_jobs`` bounds it (used by tests)."""
        self._beat()  # publish liveness before reaping so our own list is never reaped
        self._hb_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._hb_thread.start()
        reap_orphaned_jobs(self._redis, self.settings)  # recover peers that died while we were off
        last_reap = time.monotonic()
        processed = 0
        try:
            while not self._stop:
                if max_jobs is not None and processed >= max_jobs:
                    break
                if time.monotonic() - last_reap >= self.settings.worker_reaper_interval_sec:
                    reap_orphaned_jobs(self._redis, self.settings)
                    last_reap = time.monotonic()
                job_id = self._pop(timeout=2)
                if job_id is None:
                    if max_jobs is not None:
                        break
                    continue
                self.process_job(job_id)
                processed += 1
        finally:
            self._hb_stop.set()
            if self._hb_thread is not None:
                self._hb_thread.join(timeout=2)
            with suppress(redis.RedisError):
                self._redis.delete(worker_key(self._worker_id))
        return processed

    def stop(self) -> None:
        self._stop = True

    def _pop(self, timeout: int = 2) -> str | None:
        """Reliably claim the next job: atomically move its id from a served class queue to
        this worker's processing list (so a crash can't lose it). Sweep all served queues in
        priority order (non-blocking), then block briefly on the first to avoid busy-spin."""
        for cls in self._queues:
            jid = cast("str | None", self._redis.rpoplpush(queue_key(cls), self._processing_key))
            if jid:
                return jid
        return cast(
            "str | None",
            self._redis.brpoplpush(queue_key(self._queues[0]), self._processing_key, timeout),
        )

    def _unwatch(self, job_id: str) -> None:
        """Remove a finished/requeued job id from this worker's processing list. A safe
        no-op when the job wasn't claimed via _pop (e.g. direct process_job() in tests)."""
        with suppress(redis.RedisError):
            self._redis.lrem(self._processing_key, 0, job_id)

    # -- single job ------------------------------------------------------ #
    def process_job(self, job_id: str) -> JobStatus:
        # ATOMIC claim: flip QUEUED→RUNNING in a single UPDATE guarded by status='queued'. Only
        # the worker whose UPDATE affects a row proceeds, so a job that is on the queue twice
        # (reaper/requeue race) or false-reaped while still running can never be double-executed —
        # the second claimant sees rowcount==0 and skips. (Replaces a check-then-act with no lock.)
        with session_scope() as session:
            job = session.get(Job, job_id)
            if job is None:
                self._unwatch(job_id)
                return JobStatus.EXPIRED
            if job.status is JobStatus.CANCELLED:
                self._redis.delete(_cancel_key(job_id))
                self._unwatch(job_id)
                return JobStatus.CANCELLED
            # Capture fields while the ORM object is fresh (the Core UPDATE below expires it).
            job_type = job.job_type
            params = dict(job.input_params or {})
            attempts = job.attempts + 1  # what the UPDATE sets attempts to
            max_attempts = job.max_attempts
            claimed = session.execute(
                update(Job)
                .where(Job.job_id == job_id, Job.status == JobStatus.QUEUED)
                .values(
                    status=JobStatus.RUNNING,
                    started_at=datetime.now(UTC),
                    attempts=Job.attempts + 1,
                )
            ).rowcount  # type: ignore[attr-defined]  # CursorResult.rowcount
            if not claimed:
                # Another worker already claimed it (or it isn't queued) — do not run it again.
                self._unwatch(job_id)
                session.refresh(job)
                _log.info("job_claim_skipped", job_id=job_id, status=job.status.value)
                return job.status
        publish_job_event(self._redis, job_id, status="running")  # async dashboard push

        ctx = JobContext(job_id, params, self._redis)

        # Cancellation may have been requested before we started.
        if ctx.is_cancelled():
            return self._finish_cancelled(job_id)

        try:
            handler = registry.get(job_type)
        except KeyError as exc:
            return self._finish_failed(
                job_id, str(exc), "Register a handler for this job_type (Appendix B.7)."
            )

        try:
            result = handler(ctx, params) or {}
        except JobCancelled:
            return self._finish_cancelled(job_id)
        except Exception as exc:  # noqa: BLE001
            _log.warning("job_failed", job_id=job_id, job_type=job_type, error=str(exc))
            if attempts < max_attempts:
                return self._requeue(job_id)
            return self._finish_failed(
                job_id,
                f"{type(exc).__name__}: {exc}",
                "Inspect job logs; fix the handler or inputs, then retry the job.",
            )

        return self._finish_succeeded(job_id, result)

    # -- terminal transitions ------------------------------------------- #
    def _finish_succeeded(self, job_id: str, result: dict) -> JobStatus:
        # Honor a cancel requested WHILE the handler ran: a handler that doesn't checkpoint via
        # ctx.check_cancelled() would otherwise report SUCCEEDED and the cancel would be silently
        # lost (the dashboard Cancel button would look like a no-op). Reflect it as CANCELLED.
        try:
            if self._redis.exists(_cancel_key(job_id)):
                return self._finish_cancelled(job_id)
        except Exception:  # noqa: BLE001 - a redis hiccup must not block finishing the job
            pass
        with session_scope() as session:
            job = session.get(Job, job_id)
            if job is None:
                self._unwatch(job_id)
                return JobStatus.EXPIRED
            job.status = JobStatus.SUCCEEDED
            job.finished_at = datetime.now(UTC)
            if job.progress_total and job.progress_current < job.progress_total:
                job.progress_current = job.progress_total
            job.artifact_uri = result.get("artifact_uri") if isinstance(result, dict) else None
            if isinstance(result, dict) and result.get("message"):
                job.progress_message = str(result["message"])
            final_prog = format_progress(job.progress_current, job.progress_total)
            final_msg = job.progress_message
        self._redis.delete(_cancel_key(job_id))
        self._unwatch(job_id)
        publish_job_event(
            self._redis, job_id, status="succeeded", progress=final_prog, message=final_msg
        )
        return JobStatus.SUCCEEDED

    def _finish_failed(self, job_id: str, reason: str, hint: str) -> JobStatus:
        with session_scope() as session:
            job = session.get(Job, job_id)
            if job is None:
                self._unwatch(job_id)
                return JobStatus.EXPIRED
            job.status = JobStatus.FAILED
            job.finished_at = datetime.now(UTC)
            job.failure_reason = reason
            job.next_action_hint = hint
        self._redis.delete(_cancel_key(job_id))
        self._unwatch(job_id)
        publish_job_event(self._redis, job_id, status="failed", message=reason)
        return JobStatus.FAILED

    def _finish_cancelled(self, job_id: str) -> JobStatus:
        with session_scope() as session:
            job = session.get(Job, job_id)
            if job is None:
                self._unwatch(job_id)
                return JobStatus.EXPIRED
            job.status = JobStatus.CANCELLED
            job.finished_at = datetime.now(UTC)
            job.failure_reason = "cancelled"
        self._redis.delete(_cancel_key(job_id))
        self._unwatch(job_id)
        publish_job_event(self._redis, job_id, status="cancelled", message="cancelled")
        return JobStatus.CANCELLED

    def _requeue(self, job_id: str) -> JobStatus:
        with session_scope() as session:
            job = session.get(Job, job_id)
            if job is None:
                self._unwatch(job_id)
                return JobStatus.EXPIRED
            job.status = JobStatus.QUEUED
            job_type = job.job_type
        # Route back to the class queue, then drop from the processing list (pipelined so a
        # crash between the two can at worst leave a duplicate the reaper ignores, never a loss).
        pipe = self._redis.pipeline()
        pipe.lpush(queue_key(queue_class(job_type)), job_id)
        pipe.lrem(self._processing_key, 0, job_id)
        pipe.execute()
        return JobStatus.QUEUED
