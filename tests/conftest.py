"""Shared test fixtures.

Integration tests use the configured PostgreSQL + Redis (from ``.env``); they
are skipped automatically if those services are unreachable, so the unit suite
always runs even on a bare checkout.
"""

from __future__ import annotations

import pytest
import redis
from sqlalchemy import text
from src.config import get_settings
from src.jobs.handlers import ensure_handlers_registered

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
