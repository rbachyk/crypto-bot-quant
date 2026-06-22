"""Add jobs.run_token — fencing token against false-reap double execution (Appendix B.6).

Stamped on each QUEUED→RUNNING claim; terminal transitions only commit while it matches and the
orphan-reaper clears it on requeue, so a falsely-reaped-but-still-running worker can't
double-persist or double-run a job.

Revision ID: 0011
Revises: 0010
Create Date: 2026-06-22
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0011"
down_revision: str | None = "0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Postgres-native IF NOT EXISTS keeps this idempotent against a create_all'd schema.
    op.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS run_token VARCHAR(40)")


def downgrade() -> None:
    op.execute("ALTER TABLE jobs DROP COLUMN IF EXISTS run_token")
