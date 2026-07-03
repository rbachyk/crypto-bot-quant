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


# --------------------------------------------------------------------------- #
# M17: engaged() must be cheap under frequent polling (H6 wires it into the    #
# live loops' stop conditions at ~1s cadence)                                  #
# --------------------------------------------------------------------------- #
def test_redis_outage_does_not_reconnect_per_call(tmp_path, monkeypatch) -> None:
    """During a Redis outage frequent polling must NOT pay a connect timeout per call:
    one failed connect starts a back-off window (and the result cache absorbs the rest);
    the file backend keeps answering."""
    import types

    calls = {"n": 0}

    def _from_url(url, **kw):
        calls["n"] += 1
        raise ConnectionError("redis down")

    fake_mod = types.SimpleNamespace(Redis=types.SimpleNamespace(from_url=_from_url))
    monkeypatch.setattr("src.killswitch.redis", fake_mod)
    ks = KillSwitch(_settings(tmp_path))
    for _ in range(20):
        assert ks.engaged() is False
    assert calls["n"] == 1  # one connect attempt, then back-off + result cache


def test_cached_redis_client_is_reused_across_polls(tmp_path, monkeypatch) -> None:
    """A healthy Redis connection is created ONCE and reused — not a fresh connect+ping
    per engaged() call (the M17 hot-path cost)."""
    import types

    class _FakeRedis:
        def __init__(self) -> None:
            self.exists_calls = 0

        def ping(self):
            return True

        def exists(self, key):
            self.exists_calls += 1
            return 0

    fake = _FakeRedis()
    connects = {"n": 0}

    def _from_url(url, **kw):
        connects["n"] += 1
        return fake

    fake_mod = types.SimpleNamespace(Redis=types.SimpleNamespace(from_url=_from_url))
    monkeypatch.setattr("src.killswitch.redis", fake_mod)
    monkeypatch.setattr(KillSwitch, "_CACHE_TTL_S", 0.0)  # bypass the result cache
    ks = KillSwitch(_settings(tmp_path))
    for _ in range(5):
        assert ks.engaged() is False
    assert connects["n"] == 1  # single connection, reused
    assert fake.exists_calls == 5  # every (uncached) poll asked redis, cheaply


def test_engage_and_disengage_bust_the_result_cache(tmp_path) -> None:
    """The ~1.5s result cache must never delay THIS instance's own engage/disengage —
    both invalidate the cache so the next engaged() reflects reality immediately."""
    ks = KillSwitch(_settings(tmp_path, redis_url="redis://127.0.0.1:1/0"))
    assert ks.engaged() is False  # primes the cache with False
    ks.engage(reason="cache-bust", actor="test")
    assert ks.engaged() is True  # visible immediately despite the cache
    ks.disengage(actor="test")
    assert ks.engaged() is False
