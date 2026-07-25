"""Record each paper/demo run's STARTING equity (audit 2026-07-25b O-1).

The dashboard reconstructed every equity curve — and every max-drawdown % — from a hardcoded
10 000 notional base, in EVERY environment. A demo session sizes off the real account (the running
session's log shows ``equity=1000.83``, sliced 250.21 per basket), so a 10% real drawdown rendered
as ~1% and "current equity" described no account that exists. Drawdown is exactly the number an
operator reads to decide whether to let a session keep running.

The base cannot be recovered after the fact — nothing persisted it — so it is stamped at session
start from here on. NULL means "written before this column existed"; readers fall back to the paper
base for those rows, so historical curves keep rendering exactly as they do today.

Revision ID: 0015
Revises: 0014
Create Date: 2026-07-25
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0015"
down_revision: str | None = "0014"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Postgres-native IF NOT EXISTS keeps this idempotent against a create_all'd schema.
    op.execute("ALTER TABLE paper_runs ADD COLUMN IF NOT EXISTS initial_equity DOUBLE PRECISION")


def downgrade() -> None:
    op.execute("ALTER TABLE paper_runs DROP COLUMN IF EXISTS initial_equity")
