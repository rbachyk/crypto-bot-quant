"""Shared test fixtures.

Integration tests use the configured PostgreSQL + Redis (from ``.env``); they
are skipped automatically if those services are unreachable, so the unit suite
always runs even on a bare checkout.
"""

from __future__ import annotations

import os

import pytest
import redis
from sqlalchemy import create_engine, text
from src.config import get_settings
from src.jobs.handlers import ensure_handlers_registered

# Run the suite against a dedicated redis DB (15) so it never collides with a running stack:
# `make docker-up` workers consume the shared db-0 class queues, which would otherwise race
# with tests that enqueue then expect to consume their own jobs. Must run before any
# get_settings() call so the override takes effect.
os.environ["REDIS_URL"] = (os.environ.get("REDIS_URL") or "redis://localhost:6379/0").rsplit(
    "/", 1
)[0] + "/15"

# Isolate the test DATABASE so the suite never writes into the dashboard's data (the host
# postgres is shared with `make docker-up`). Mirror the redis-db-15 isolation: swap the database
# name to `<name>_test` and create it on first use. Best-effort — if the server is unreachable or
# the db can't be created, DB_OK below stays False and the db-backed tests skip as before.
_DEFAULT_DB = "postgresql+psycopg://postgres:postgres@localhost:5432/trading_bot"
_db_url = os.environ.get("DATABASE_URL") or _DEFAULT_DB
_base, _name = _db_url.rsplit("/", 1)
_name = _name.split("?")[0]
if not _name.endswith("_test"):
    _test_name = f"{_name}_test"
    os.environ["DATABASE_URL"] = f"{_base}/{_test_name}"
    try:  # CREATE DATABASE must run outside a transaction (AUTOCOMMIT) on the maintenance db.
        _admin = create_engine(f"{_base}/postgres", isolation_level="AUTOCOMMIT")
        with _admin.connect() as _conn:
            if not _conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": _test_name}
            ).scalar():
                _conn.execute(text(f'CREATE DATABASE "{_test_name}"'))
        _admin.dispose()
    except Exception:  # noqa: BLE001 - no server / no perms → tests skip via DB_OK
        pass

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

# Build the isolated test database via the REAL migrations (alembic upgrade head), so the test
# schema is exactly a migrated production schema — including ALTERs (new columns) that a plain
# create_all would skip on an already-existing table (the test DB persists across runs). Falls
# back to create_all + a head stamp if alembic can't run, so a bare environment still works.
if DB_OK:
    try:
        import src.db.models  # noqa: F401  (registers every table on Base.metadata)
        from alembic import command
        from alembic.config import Config as _AlembicConfig

        command.upgrade(_AlembicConfig("alembic.ini"), "head")
    except Exception:  # noqa: BLE001 - fall back to create_all so a bare env still runs
        try:
            from alembic.config import Config as _AlembicConfig2
            from alembic.script import ScriptDirectory
            from src.db.base import Base, get_engine

            Base.metadata.create_all(get_engine())
            _head = ScriptDirectory.from_config(_AlembicConfig2("alembic.ini")).get_current_head()
            with get_engine().begin() as _conn:
                _conn.execute(
                    text(
                        "CREATE TABLE IF NOT EXISTS alembic_version "
                        "(version_num VARCHAR(32) NOT NULL)"
                    )
                )
                if not _conn.execute(text("SELECT 1 FROM alembic_version")).scalar() and _head:
                    _conn.execute(
                        text("INSERT INTO alembic_version (version_num) VALUES (:v)"), {"v": _head}
                    )
        except Exception:  # noqa: BLE001 - leave DB_OK; a schema error surfaces in the test
            pass

requires_db = pytest.mark.skipif(not DB_OK, reason="database not reachable")
requires_redis = pytest.mark.skipif(not (DB_OK and REDIS_OK), reason="redis (and db) not reachable")


@pytest.fixture
def settings():
    return get_settings()
