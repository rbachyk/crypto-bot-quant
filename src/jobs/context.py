"""Execution context handed to job handlers.

Provides progress reporting, log streaming (persisted to ``job_logs``), and
cooperative cancellation. Handlers must call :meth:`JobContext.check_cancelled`
at safe points so cancellation works for long-running jobs (Appendix B.6).
"""

from __future__ import annotations

from datetime import UTC, datetime

import redis

from src.db.base import session_scope
from src.db.models import Job, JobLog


class JobCancelled(Exception):
    """Raised inside a handler when the job has been cancelled."""


def _cancel_key(job_id: str) -> str:
    return f"qbot:cancel:{job_id}"


class JobContext:
    """Per-execution context object passed to a job handler."""

    def __init__(
        self,
        job_id: str,
        params: dict,
        redis_client: redis.Redis | None,
        *,
        run_token: str | None = None,
    ) -> None:
        self.job_id = job_id
        self.params = params
        self._redis = redis_client
        self.run_token = run_token  # fencing token for THIS run (None outside the worker)

    # -- logging --------------------------------------------------------- #
    def log(self, message: str, level: str = "INFO") -> None:
        with session_scope() as session:
            session.add(
                JobLog(
                    job_id=self.job_id,
                    level=level,
                    message=message,
                    ts=datetime.now(UTC),
                )
            )

    # -- progress -------------------------------------------------------- #
    def progress(self, current: int, total: int, message: str = "") -> None:
        with session_scope() as session:
            job = session.get(Job, self.job_id)
            if job is None:
                return
            job.progress_current = current
            job.progress_total = total
            if message:
                job.progress_message = message
        # Push the update to the dashboard SSE stream (async; no client polling).
        from src.jobs.events import format_progress, publish_job_event

        publish_job_event(
            self._redis,
            self.job_id,
            status="running",
            progress=format_progress(current, total),
            message=message or None,
        )

    # -- cancellation ---------------------------------------------------- #
    def is_cancelled(self) -> bool:
        if self._redis is None:
            return False
        try:
            return bool(self._redis.exists(_cancel_key(self.job_id)))
        except Exception:
            return False

    def check_cancelled(self) -> None:
        """Raise :class:`JobCancelled` if a cancel request is pending."""
        if self.is_cancelled():
            raise JobCancelled(self.job_id)

    # -- ownership (fencing) --------------------------------------------- #
    def still_owns(self) -> bool:
        """Whether THIS run still owns the job (its fencing token still matches the DB row).

        A long handler should poll this and stop if it returns False — that means the job was
        reaped/requeued (e.g. heartbeat starvation) and another worker is now running it, so this
        run must not keep executing (no double live session / double side effects). Best-effort:
        a DB hiccup returns True (fail-open to keep running) since the terminal write is fenced
        anyway."""
        if self.run_token is None:
            return True
        try:
            with session_scope() as session:
                job = session.get(Job, self.job_id)
                return job is None or job.run_token == self.run_token
        except Exception:  # noqa: BLE001 - don't kill a healthy run on a transient DB blip
            return True
