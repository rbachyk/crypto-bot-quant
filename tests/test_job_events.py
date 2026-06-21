"""Async job events (SSE push): job lifecycle updates are published to a redis pub/sub channel
so the dashboard streams status/progress without polling. Proves the publish helper is correct
and that running a job through the worker emits the expected events end-to-end."""

from __future__ import annotations

import json

from src.jobs.events import JOB_EVENTS_CHANNEL, format_progress, publish_job_event

from tests.conftest import requires_redis


class _RecordingRedis:
    def __init__(self) -> None:
        self.published: list[tuple[str, str]] = []

    def publish(self, channel: str, message: str) -> None:
        self.published.append((channel, message))


def test_format_progress() -> None:
    assert format_progress(2, 5) == "2/5"
    assert format_progress(3, 0) == "3"
    assert format_progress(0, 0) == "-"
    assert format_progress(None, None) == "-"


def test_publish_job_event_payload() -> None:
    r = _RecordingRedis()
    publish_job_event(r, "job_1", status="running", progress="2/5", message="working")
    assert len(r.published) == 1
    channel, raw = r.published[0]
    assert channel == JOB_EVENTS_CHANNEL
    assert json.loads(raw) == {
        "job_id": "job_1",
        "status": "running",
        "progress": "2/5",
        "message": "working",
    }


def test_publish_job_event_is_best_effort() -> None:
    """A redis hiccup (or no client) must never raise — publishing can't break a job."""
    publish_job_event(None, "job_x", status="running")  # no client → no-op

    class _Boom:
        def publish(self, *a, **k):
            raise RuntimeError("redis down")

    publish_job_event(_Boom(), "job_x", status="failed")  # swallowed, no raise


@requires_redis
def test_worker_publishes_lifecycle_events_to_channel() -> None:
    """Running a job emits running → progress → succeeded on the SSE channel."""
    import redis

    from src.config import get_settings
    from src.jobs import JobQueue, Worker

    client = redis.Redis.from_url(get_settings().redis_url, decode_responses=True)
    pubsub = client.pubsub(ignore_subscribe_messages=True)
    pubsub.subscribe(JOB_EVENTS_CHANNEL)
    # Force the SUBSCRIBE round-trip to complete so the server has registered us BEFORE the
    # worker publishes (pub/sub does not buffer for not-yet-registered subscribers).
    pubsub.get_message(timeout=2.0)
    try:
        queue, worker = JobQueue(), Worker()
        job_id = queue.enqueue("selftest_echo", {"steps": 3}, requested_by="test")
        worker.process_job(job_id)

        statuses: list[str] = []
        # Drain everything currently queued on the subscription.
        for _ in range(200):
            msg = pubsub.get_message(timeout=1.0)
            if msg is None:
                break
            if msg.get("type") != "message":
                continue
            payload = json.loads(msg["data"])
            if payload.get("job_id") == job_id and payload.get("status"):
                statuses.append(payload["status"])
        assert "running" in statuses
        assert "succeeded" in statuses
    finally:
        pubsub.close()
        client.close()
