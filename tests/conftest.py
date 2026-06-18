"""Shared test fixtures.

Integration tests use the configured PostgreSQL + Redis (from ``.env``); they
are skipped automatically if those services are unreachable, so the unit suite
always runs even on a bare checkout.
"""

from __future__ import annotations

import os

import pytest
import redis
from sqlalchemy import text
from src.config import get_settings
from src.jobs.handlers import ensure_handlers_registered

# Run the suite against a dedicated redis DB (15) so it never collides with a running stack:
# `make docker-up` workers consume the shared db-0 class queues, which would otherwise race
# with tests that enqueue then expect to consume their own jobs. Postgres stays shared (tests
# use unique ids). Must run before any get_settings() call so the override takes effect.
os.environ["REDIS_URL"] = (os.environ.get("REDIS_URL") or "redis://localhost:6379/0").rsplit(
    "/", 1
)[0] + "/15"
get_settings.cache_clear()

ensure_handlers_registered()


def _db_reachable() -> bool:
    try:
        from src.db.base import get_engine

        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def _redis_reachable() -> bool:
    try:
        client = redis.Redis.from_url(get_settings().redis_url, socket_connect_timeout=1)
        client.ping()
        return True
    except Exception:
        return False


DB_OK = _db_reachable()
REDIS_OK = _redis_reachable()

requires_db = pytest.mark.skipif(not DB_OK, reason="database not reachable")
requires_redis = pytest.mark.skipif(not (DB_OK and REDIS_OK), reason="redis (and db) not reachable")


@pytest.fixture
def settings():
    return get_settings()
