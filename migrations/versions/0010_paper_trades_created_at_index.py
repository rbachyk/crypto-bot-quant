"""Index paper_trades.created_at — the hot stats filter/sort key.

Every dashboard stats query filters by a time window and orders by ``created_at``; without an
index this full-scans + sorts ``paper_trades`` on each render, degrading as the table grows.
Idempotent: created with ``IF NOT EXISTS`` semantics via create_index(if_not_exists) fallback.

Revision ID: 0010
Revises: 0009
Create Date: 2026-06-21
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0010"
down_revision: str | None = "0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_INDEX = "ix_paper_trades_created_at"


def upgrade() -> None:
    # Postgres-native IF NOT EXISTS keeps this idempotent (the index may already exist from a
    # create_all on a fresh DB) without poisoning the migration transaction.
    op.execute(f"CREATE INDEX IF NOT EXISTS {_INDEX} ON paper_trades (created_at)")


def downgrade() -> None:
    op.execute(f"DROP INDEX IF EXISTS {_INDEX}")
