"""Scheduler tests (Appendix B.13): recurring jobs fire on a cadence and are gated by the
enable_* toggles (closing the 'nothing fires on a timer' + 'inert toggles' gaps)."""

from __future__ import annotations

import time

from src.config import Settings
from src.scheduler import Scheduler

from tests.conftest import requires_redis


def _clear(sched: Scheduler) -> None:
    for k in sched._redis.scan_iter("qbot:sched:*"):
        sched._redis.delete(k)


@requires_redis
def test_due_respects_toggles() -> None:
    # Background research is OFF by default (safe default) — nothing is due.
    sched_off = Scheduler(settings=Settings(_env_file=None))
    _clear(sched_off)
    assert sched_off.due(time.time()) == []

    # Turn research ON → strategy validation + paper session become schedulable; ml stays off.
    sched = Scheduler(settings=Settings(_env_file=None, enable_background_research_jobs=True))
    _clear(sched)
    due = {j.job_type for j in sched.due(time.time())}
    assert "run_strategy_validation" in due
    assert "run_paper_session" in due
    assert "run_ml_shadow_pass" not in due  # ml toggle is off

    # Flip the ML toggle → the ML shadow pass becomes schedulable.
    sched_ml = Scheduler(
        settings=Settings(
            _env_file=None, enable_background_research_jobs=True, enable_ml_shadow=True
        )
    )
    assert "run_ml_shadow_pass" in {j.job_type for j in sched_ml.due(time.time())}


@requires_redis
def test_tick_enqueues_then_not_due_until_interval() -> None:
    sched = Scheduler(settings=Settings(_env_file=None, enable_background_research_jobs=True))
    _clear(sched)
    now = time.time()
    enqueued = set(sched.tick(now))
    assert "run_strategy_validation" in enqueued and "run_paper_session" in enqueued
    # Immediately after, nothing is due again (interval has not elapsed).
    assert sched.tick(now + 1) == []
    # Far in the future, the hourly paper job is due again.
    assert "run_paper_session" in set(sched.tick(now + 7200))


@requires_redis
def test_runtime_pause_stops_enqueuing() -> None:
    """The dashboard pause toggle stops the scheduler from enqueuing without a restart."""
    from src.scheduler import is_scheduler_paused, set_scheduler_paused

    sched = Scheduler(settings=Settings(_env_file=None, enable_background_research_jobs=True))
    _clear(sched)
    sched._redis.delete("qbot:sched:paused")
    assert not is_scheduler_paused(sched._redis)
    set_scheduler_paused(sched._redis, True)
    assert is_scheduler_paused(sched._redis)
    assert sched.tick(time.time()) == []  # paused → nothing enqueued (even though jobs are due)
    set_scheduler_paused(sched._redis, False)
    assert "run_paper_session" in set(sched.tick(time.time()))  # resumes
