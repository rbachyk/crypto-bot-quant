"""Add funding columns to open_positions and paper_trades.

Surfaces accrued funding (COST convention: >0 paid, <0 carry received) separately from price P&L,
so the dashboard / closed-trade analysis can show how much of a position's P&L is carry vs price.
The value is already netted into unrealized_pnl (open) / pnl (closed); these columns just break it
out. 0.0 for per-trade-engine rows (lead_lag etc.); meaningful for basket legs.

Revision ID: 0013
Revises: 0012
Create Date: 2026-06-30
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0013"
down_revision: str | None = "0012"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Postgres-native IF NOT EXISTS keeps this idempotent against a create_all'd schema.
    op.execute(
        "ALTER TABLE open_positions ADD COLUMN IF NOT EXISTS "
        "funding DOUBLE PRECISION NOT NULL DEFAULT 0.0"
    )
    op.execute(
        "ALTER TABLE paper_trades ADD COLUMN IF NOT EXISTS "
        "funding DOUBLE PRECISION NOT NULL DEFAULT 0.0"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE open_positions DROP COLUMN IF EXISTS funding")
    op.execute("ALTER TABLE paper_trades DROP COLUMN IF EXISTS funding")
