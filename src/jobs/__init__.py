"""Background job system (AGENTS.md Appendix B.6).

A Redis-backed queue with PostgreSQL job records + logs, cooperative
cancellation, retry, and progress tracking. The live/paper engines and heavy
research run as their own processes/workers, never inside the API (B.2/B.17).
"""

from src.jobs.context import JobCancelled, JobContext
from src.jobs.queue import JobQueue
from src.jobs.registry import job_handler, registry
from src.jobs.worker import Worker

__all__ = [
    "JobCancelled",
    "JobContext",
    "JobQueue",
    "Worker",
    "job_handler",
    "registry",
]
