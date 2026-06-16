"""Kill-switch tests (AGENTS.md Section 2.2 / KILL gate).

The kill switch must work independently of the dashboard and even when Redis is
unreachable (file backend). These tests point Redis at an unreachable URL to
prove the file backend alone halts.
"""

from __future__ import annotations

from src.config import Settings
from src.killswitch import KillSwitch


def _settings(tmp_path, redis_url="redis://127.0.0.1:6399/0") -> Settings:
    return Settings(
        _env_file=None,
        data_lake_path=tmp_path / "datalake",
        redis_url=redis_url,
    )


def test_engage_and_disengage_file_backend(tmp_path) -> None:
    ks = KillSwitch(_settings(tmp_path, redis_url="redis://127.0.0.1:1/0"))  # unreachable redis
    assert ks.engaged() is False
    ks.engage(reason="unit-test", actor="test")
    assert ks.engaged() is True  # file backend alone is sufficient
    status = ks.status()
    assert status["file_backend"] is True
    assert status["redis_reachable"] is False
    ks.disengage(actor="test")
    assert ks.engaged() is False


def test_engaged_is_failsafe_with_dead_redis(tmp_path) -> None:
    ks = KillSwitch(_settings(tmp_path, redis_url="redis://127.0.0.1:1/0"))
    ks.engage(reason="failsafe", actor="test")
    # Even with Redis down, the switch reads engaged (any signal halts).
    assert ks.engaged() is True
    ks.disengage()
