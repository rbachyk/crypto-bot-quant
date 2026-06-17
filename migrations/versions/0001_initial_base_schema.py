"""Initial Phase 1 base schema.

Creates the operational/auditable tables (jobs, job_logs, gates, gate_results,
remediation_actions, approvals, audit_logs) plus exchange-metadata and universe
skeleton tables. Schema is derived from the ORM metadata so it can never drift
from ``src.db.models``. Idempotent: ``create_all(checkfirst=True)`` only creates
missing objects, and Alembic's version table makes repeated upgrades a no-op.

TimescaleDB is enabled opportunistically; its absence is tolerated (see
docs/decisions/0002-timescaledb-optional-in-phase-1.md).

Revision ID: 0001
Revises:
Create Date: 2026-06-16
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from src.db import models  # noqa: F401  (registers tables on Base.metadata)
from src.db.base import Base

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()

    # Best-effort TimescaleDB extension; tolerated if the build lacks it.
    # Run inside a SAVEPOINT so a failure rolls back only this step and leaves
    # the surrounding migration transaction usable (decision 0002).
    try:
        with bind.begin_nested():
            bind.execute(sa.text("CREATE EXTENSION IF NOT EXISTS timescaledb"))
    except Exception:  # pragma: no cover - depends on server build
        pass

    Base.metadata.create_all(bind=bind, checkfirst=True)


def downgrade() -> None:
    bind = op.get_bind()
    Base.metadata.drop_all(bind=bind)
