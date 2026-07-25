"""One trading session per exchange account (audit 2026-07-25e O-1).

The API refuses to ENQUEUE a conflicting session, but `qbot demo-basket` calls the session function
directly. This lock lives in the session's own pre-flight, which every entry point passes through.
"""

from __future__ import annotations

import pytest
from src.config import Settings
from src.live.account_lock import AccountLock


class _FakeRedis:
    """Just enough Redis for the lock: SET NX EX, GET, DELETE."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    def ping(self) -> bool:
        return True

    def set(self, key, value, nx=False, ex=None):  # noqa: ARG002 - ex is a TTL we don't simulate
        if nx and key in self.store:
            return None
        self.store[key] = value
        return True

    def get(self, key):
        v = self.store.get(key)
        return v.encode() if isinstance(v, str) else v

    def delete(self, key) -> int:
        return 1 if self.store.pop(key, None) is not None else 0


def _lock(settings: Settings, shared: _FakeRedis, owner: str) -> AccountLock:
    lock = AccountLock(settings, owner=owner)
    lock._client = shared  # the connection the session would have opened
    return lock


def _settings() -> Settings:
    return Settings(_env_file=None, exchange_env="demo")


def test_a_second_session_is_refused_and_told_who_holds_the_account() -> None:
    """An exchange holds ONE net position per symbol: a basket rebalance would resize the
    per-symbol strategy's stop-managed position, and each session's reconciler would read the
    other's activity as drift."""
    settings, redis = _settings(), _FakeRedis()
    first = _lock(settings, redis, "basket:funding_carry")
    first.acquire()

    second = _lock(settings, redis, "live:testnet")
    with pytest.raises(PermissionError) as exc:
        second.acquire()
    assert "basket:funding_carry" in str(exc.value)  # names the holder
    assert "live_symbols" in str(exc.value)  # …and how to make them coexist


def test_releasing_frees_the_account_for_the_next_session() -> None:
    settings, redis = _settings(), _FakeRedis()
    first = _lock(settings, redis, "basket:funding_carry")
    first.acquire()
    first.release()

    _lock(settings, redis, "live:testnet").acquire()  # must not raise


def test_a_session_never_releases_a_claim_it_does_not_hold() -> None:
    """A late release from a session whose claim already expired (TTL) must not free the account
    out from under whoever took it next."""
    settings, redis = _settings(), _FakeRedis()
    first = _lock(settings, redis, "basket:funding_carry")
    first.acquire()
    redis.store["qbot:account-session:demo"] = "someone-else"  # the TTL lapsed; another took it

    first.release()

    assert redis.store["qbot:account-session:demo"] == "someone-else"


def test_an_unreachable_lock_store_does_not_refuse_the_start() -> None:
    """Redis being down is not evidence that another session exists, and refusing to trade because
    the cache is unavailable would turn an infrastructure blip into an outage — strictly worse than
    the hazard, and a regression against the previous behaviour (no lock at all). It starts, loudly.
    """
    settings = Settings(_env_file=None, exchange_env="demo", redis_url="redis://127.0.0.1:1/0")
    AccountLock(settings, owner="basket:x").acquire()  # must not raise
