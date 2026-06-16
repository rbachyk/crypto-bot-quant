"""Job-type registry (AGENTS.md Appendix B.7).

Handlers are first-class, registered by ``job_type``. A handler is a callable
``(JobContext, dict) -> dict | None`` where the return value (JSON-serialisable)
is stored as the job artifact summary.
"""

from __future__ import annotations

from collections.abc import Callable

from src.jobs.context import JobContext

JobHandler = Callable[[JobContext, dict], "dict | None"]


class _Registry:
    def __init__(self) -> None:
        self._handlers: dict[str, JobHandler] = {}

    def register(self, job_type: str, handler: JobHandler) -> None:
        self._handlers[job_type] = handler

    def get(self, job_type: str) -> JobHandler:
        if job_type not in self._handlers:
            raise KeyError(f"no handler registered for job_type={job_type!r}")
        return self._handlers[job_type]

    def known(self) -> list[str]:
        return sorted(self._handlers)

    def has(self, job_type: str) -> bool:
        return job_type in self._handlers


registry = _Registry()


def job_handler(job_type: str) -> Callable[[JobHandler], JobHandler]:
    """Decorator registering a function as the handler for ``job_type``."""

    def _decorator(fn: JobHandler) -> JobHandler:
        registry.register(job_type, fn)
        return fn

    return _decorator
