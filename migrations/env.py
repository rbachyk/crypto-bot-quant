"""Alembic environment.

The database URL is taken from the validated application settings (env-driven),
never hardcoded, so migrations always target the configured environment.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool
from src.config import get_settings
from src.db import models  # noqa: F401  (import registers all tables on Base.metadata)
from src.db.base import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# The URL is used DIRECTLY (create_engine / context.configure), never round-tripped through
# alembic's ConfigParser (L19): ConfigParser treats ``%`` as interpolation syntax, so a DB password
# containing ``%`` would raise InterpolationSyntaxError or silently corrupt the URL. We still record
# an ESCAPED copy in the config so tools reading ``sqlalchemy.url`` (``alembic current``) still run.
_DB_URL = get_settings().sync_database_url
config.set_main_option("sqlalchemy.url", _DB_URL.replace("%", "%%"))

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=_DB_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        # Detect column TYPE drift so autogenerate emits real ALTERs against an already-deployed
        # schema, not just a create_all snapshot (L19). compare_server_default is intentionally OFF:
        # no model declares a server_default (all defaults are Python-side), so it can only produce
        # spurious "drift" on a future `alembic check` (L-O).
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = create_engine(_DB_URL, poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,  # compare_server_default off (see offline mode, L-O)
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
