"""Periodic job scheduler (AGENTS.md Appendix B.13 — recurring maintenance).

Closes the "nothing fires on a timer" gap: this enqueues recurring **shadow-only** jobs (research
re-validation, paper sessions, ML shadow passes) on a cadence, gated by the ``enable_*`` toggles
so flipping a toggle actually changes behaviour. Last-run state lives in redis (survives restarts;
a redis lock means duplicate schedulers never double-fire). It only ENQUEUES — the dedicated
workers consume — and never touches live trading.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import redis
import structlog

from src.config import Settings, get_settings

_log = structlog.get_logger("scheduler")

_LAST_KEY = "qbot:sched:last:{job}"
_LOCK_KEY = "qbot:sched:lock"
_PAUSE_KEY = "qbot:sched:paused"  # runtime kill-switch for the recurring jobs (dashboard toggle)

# Cadences are deliberately conservative defaults (seconds). All jobs are shadow-only.
_HOUR = 3600


def is_scheduler_paused(redis_client: redis.Redis) -> bool:
    """Whether the recurring background jobs are paused at runtime (dashboard toggle)."""
    try:
        return bool(redis_client.get(_PAUSE_KEY))
    except Exception:  # noqa: BLE001 - a redis hiccup must not crash the dashboard
        return False


def set_scheduler_paused(redis_client: redis.Redis, paused: bool) -> None:
    """Pause/resume the recurring background jobs without restarting the scheduler service."""
    if paused:
        redis_client.set(_PAUSE_KEY, "1")
    else:
        redis_client.delete(_PAUSE_KEY)


@dataclass(frozen=True)
class ScheduledJob:
    job_type: str
    interval_sec: int
    is_enabled: Callable[[Settings], bool]
    params: dict = field(default_factory=dict)


def default_schedule() -> list[ScheduledJob]:
    return [
        ScheduledJob(
            "run_strategy_validation", 6 * _HOUR, lambda s: s.enable_background_research_jobs
        ),
        ScheduledJob("run_paper_session", _HOUR, lambda s: s.enable_background_research_jobs),
        ScheduledJob("run_ml_shadow_pass", _HOUR, lambda s: s.enable_ml_shadow),
    ]


class Scheduler:
    """Enqueues due recurring jobs. Drive with :meth:`run` (loop) or :meth:`tick` (one pass)."""

    def __init__(
        self,
        settings: Settings | None = None,
        redis_client: redis.Redis | None = None,
        schedule: list[ScheduledJob] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._redis = redis_client or redis.Redis.from_url(
            self.settings.redis_url, decode_responses=True
        )
        self.schedule = schedule if schedule is not None else default_schedule()
        self._stop = False

    def due(self, now: float) -> list[ScheduledJob]:
        """Jobs whose toggle is on and whose interval has elapsed since their last run."""
        out: list[ScheduledJob] = []
        for job in self.schedule:
            if not job.is_enabled(self.settings):
                continue
            last = self._redis.get(_LAST_KEY.format(job=job.job_type))
            if last is None or (now - float(last)) >= job.interval_sec:
                out.append(job)
        return out

    def tick(self, now: float) -> list[str]:
        """Enqueue every due job once (lock-guarded). Returns the enqueued job types."""
        from src.jobs import JobQueue

        # Runtime pause (dashboard toggle) — enqueue nothing while paused.
        if is_scheduler_paused(self._redis):
            return []
        # Only one scheduler enqueues per tick window (others no-op).
        if not self._redis.set(
            _LOCK_KEY, "1", nx=True, ex=max(5, self.settings.scheduler_tick_sec)
        ):
            return []
        enqueued: list[str] = []
        try:
            queue = JobQueue(self.settings, redis_client=self._redis)
            for job in self.due(now):
                queue.enqueue(job.job_type, dict(job.params), requested_by="scheduler")
                self._redis.set(_LAST_KEY.format(job=job.job_type), str(now))
                enqueued.append(job.job_type)
                _log.info("scheduled_job_enqueued", job_type=job.job_type)
        finally:
            self._redis.delete(_LOCK_KEY)
        return enqueued

    def stop(self) -> None:
        self._stop = True

    def run(self, *, max_ticks: int | None = None) -> int:
        """Loop: every ``scheduler_tick_sec`` enqueue due jobs (when scheduler_enabled)."""
        ticks = 0
        while not self._stop:
            if max_ticks is not None and ticks >= max_ticks:
                break
            if self.settings.scheduler_enabled:
                self.tick(time.time())
            ticks += 1
            time.sleep(self.settings.scheduler_tick_sec)
        return ticks
