"""Redis-backed job queue with PostgreSQL job records (Appendix B.6).

Responsibilities:
* create a durable :class:`~src.db.models.Job` record on enqueue;
* push the job id onto a Redis list the worker consumes;
* support cancel (cooperative for running jobs) and retry of failed jobs.

The queue persists every required job field (B.6) and never hides work in a
background thread (B.17).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import redis

from src.config import Settings, get_settings
from src.db.base import session_scope
from src.db.models import Job, JobStatus
from src.jobs.context import _cancel_key

QUEUE_KEY = "qbot:queue"


class JobQueue:
    """Enqueue / cancel / retry jobs backed by Redis + Postgres."""

    def __init__(self, settings: Settings | None = None, redis_client: redis.Redis | None = None):
        self.settings = settings or get_settings()
        self._redis = redis_client or redis.Redis.from_url(
            self.settings.redis_url, decode_responses=True
        )

    @property
    def redis(self) -> redis.Redis:
        return self._redis

    # -- enqueue --------------------------------------------------------- #
    def enqueue(
        self,
        job_type: str,
        params: dict | None = None,
        *,
        requested_by: str = "system",
        related_gate_id: str | None = None,
        max_attempts: int = 1,
    ) -> str:
        job_id = f"job_{uuid.uuid4().hex[:16]}"
        with session_scope() as session:
            session.add(
                Job(
                    job_id=job_id,
                    job_type=job_type,
                    status=JobStatus.QUEUED,
                    requested_by=requested_by,
                    environment=self.settings.app_env.value,
                    input_params=params or {},
                    related_gate_id=related_gate_id,
                    max_attempts=max_attempts,
                    created_at=datetime.now(UTC),
                )
            )
        self._redis.lpush(QUEUE_KEY, job_id)
        return job_id

    # -- cancel ---------------------------------------------------------- #
    def cancel(self, job_id: str) -> bool:
        """Cancel a job.

        Queued jobs are marked cancelled immediately; running jobs get a Redis
        cancel flag that the handler observes cooperatively (Appendix B.6).
        Returns False if the job is already terminal.
        """
        with session_scope() as session:
            job = session.get(Job, job_id)
            if job is None:
                return False
            if job.status in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED):
                return False
            if job.status is JobStatus.QUEUED:
                job.status = JobStatus.CANCELLED
                job.finished_at = datetime.now(UTC)
                job.failure_reason = "cancelled before start"
                self._redis.set(_cancel_key(job_id), "1")
                return True
            # RUNNING: signal cooperative cancellation.
            self._redis.set(_cancel_key(job_id), "1")
            return True

    # -- retry ----------------------------------------------------------- #
    def retry(self, job_id: str) -> bool:
        """Re-queue a failed/cancelled job (Appendix B.6)."""
        with session_scope() as session:
            job = session.get(Job, job_id)
            if job is None:
                return False
            if job.status not in (JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.EXPIRED):
                return False
            job.status = JobStatus.QUEUED
            job.failure_reason = None
            job.finished_at = None
            job.started_at = None
            job.progress_current = 0
            if job.max_attempts <= job.attempts:
                job.max_attempts = job.attempts + 1
        self._redis.delete(_cancel_key(job_id))
        self._redis.lpush(QUEUE_KEY, job_id)
        return True

    # -- introspection --------------------------------------------------- #
    def depth(self) -> int:
        return int(self._redis.llen(QUEUE_KEY))
