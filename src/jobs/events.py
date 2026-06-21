"""Job lifecycle events on a redis pub/sub channel (async dashboard push).

The dashboard subscribes to this channel over Server-Sent Events (``/api/jobs/stream``) so job
status + progress update in the browser the instant a worker writes them — there is NO
client-side polling. Publishing is best-effort: a redis hiccup must never fail or slow a job, so
every publish is wrapped and swallowed.
"""

from __future__ import annotations

import contextlib
import json
from typing import Any

JOB_EVENTS_CHANNEL = "qbot:jobs:events"


def format_progress(current: int | None, total: int | None) -> str:
    """Render progress exactly as the dashboard expects (matches the /api/jobs/status format)."""
    cur = current or 0
    if total:
        return f"{cur}/{total}"
    return str(cur) if cur else "-"


def publish_job_event(
    redis_client: Any | None,
    job_id: str,
    *,
    status: str | None = None,
    progress: str | None = None,
    message: str | None = None,
) -> None:
    """Publish a job update to the SSE channel. Best-effort; never raises."""
    if redis_client is None:
        return
    payload: dict[str, Any] = {"job_id": job_id}
    if status is not None:
        payload["status"] = status
    if progress is not None:
        payload["progress"] = progress
    if message is not None:
        payload["message"] = message
    # A pub/sub hiccup must not affect the job.
    with contextlib.suppress(Exception):
        redis_client.publish(JOB_EVENTS_CHANNEL, json.dumps(payload))


__all__ = ["JOB_EVENTS_CHANNEL", "format_progress", "publish_job_event"]
