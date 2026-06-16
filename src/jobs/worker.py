"""Job worker (AGENTS.md Appendix B.6/B.13).

Consumes job ids from the Redis queue, runs the registered handler with a
:class:`JobContext`, and maintains the durable job record: status transitions,
timestamps, attempts/retry, failure reason + next-action hint, and a final
progress update. Runs as a dedicated process (``make run-worker-*``), never
inside the API process (B.17).
"""

from __future__ import annotations

from datetime import UTC, datetime

import redis
import structlog

from src.config import Settings, get_settings
from src.db.base import session_scope
from src.db.models import Job, JobStatus
from src.jobs.context import JobCancelled, JobContext, _cancel_key
from src.jobs.queue import QUEUE_KEY
from src.jobs.registry import registry

_log = structlog.get_logger("worker")


class Worker:
    """Single-process job worker. Concurrency is one job at a time per process;
    scale by running multiple worker processes (B.13)."""

    def __init__(self, settings: Settings | None = None, redis_client: redis.Redis | None = None):
        self.settings = settings or get_settings()
        self._redis = redis_client or redis.Redis.from_url(
            self.settings.redis_url, decode_responses=True
        )
        self._stop = False

    # -- main loop ------------------------------------------------------- #
    def run(self, *, max_jobs: int | None = None) -> int:
        """Run the consume loop. ``max_jobs`` bounds it (used by tests)."""
        processed = 0
        while not self._stop:
            if max_jobs is not None and processed >= max_jobs:
                break
            job_id = self._pop(timeout=2)
            if job_id is None:
                if max_jobs is not None:
                    break
                continue
            self.process_job(job_id)
            processed += 1
        return processed

    def stop(self) -> None:
        self._stop = True

    def _pop(self, timeout: int = 2) -> str | None:
        result = self._redis.brpop([QUEUE_KEY], timeout=timeout)
        if result is None:
            return None
        # decode_responses=True => (key, value) as str
        _, value = result
        return value if isinstance(value, str) else value.decode()

    # -- single job ------------------------------------------------------ #
    def process_job(self, job_id: str) -> JobStatus:
        # Load + guard.
        with session_scope() as session:
            job = session.get(Job, job_id)
            if job is None:
                return JobStatus.EXPIRED
            if job.status is JobStatus.CANCELLED:
                self._redis.delete(_cancel_key(job_id))
                return JobStatus.CANCELLED
            job_type = job.job_type
            params = dict(job.input_params or {})
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now(UTC)
            job.attempts += 1
            attempts = job.attempts
            max_attempts = job.max_attempts

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
        with session_scope() as session:
            job = session.get(Job, job_id)
            if job is None:
                return JobStatus.EXPIRED
            job.status = JobStatus.SUCCEEDED
            job.finished_at = datetime.now(UTC)
            if job.progress_total and job.progress_current < job.progress_total:
                job.progress_current = job.progress_total
            job.artifact_uri = result.get("artifact_uri") if isinstance(result, dict) else None
            if isinstance(result, dict) and result.get("message"):
                job.progress_message = str(result["message"])
        self._redis.delete(_cancel_key(job_id))
        return JobStatus.SUCCEEDED

    def _finish_failed(self, job_id: str, reason: str, hint: str) -> JobStatus:
        with session_scope() as session:
            job = session.get(Job, job_id)
            if job is None:
                return JobStatus.EXPIRED
            job.status = JobStatus.FAILED
            job.finished_at = datetime.now(UTC)
            job.failure_reason = reason
            job.next_action_hint = hint
        self._redis.delete(_cancel_key(job_id))
        return JobStatus.FAILED

    def _finish_cancelled(self, job_id: str) -> JobStatus:
        with session_scope() as session:
            job = session.get(Job, job_id)
            if job is None:
                return JobStatus.EXPIRED
            job.status = JobStatus.CANCELLED
            job.finished_at = datetime.now(UTC)
            job.failure_reason = "cancelled"
        self._redis.delete(_cancel_key(job_id))
        return JobStatus.CANCELLED

    def _requeue(self, job_id: str) -> JobStatus:
        with session_scope() as session:
            job = session.get(Job, job_id)
            if job is None:
                return JobStatus.EXPIRED
            job.status = JobStatus.QUEUED
        self._redis.lpush(QUEUE_KEY, job_id)
        return JobStatus.QUEUED
