"""Add engine_version to strategy_promotions (audit H-D).

The live active-set readers filter promotions to the CURRENT engine cost/geometry model version,
so a verdict validated under a superseded engine (NULL on pre-existing rows) is excluded until it is
re-validated — an engine cost/geometry change can no longer leave a stale, more-favorable promotion
silently active. Indexed for the equality filter.

Revision ID: 0014
Revises: 0013
Create Date: 2026-07-04
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0014"
down_revision: str | None = "0013"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Postgres-native IF NOT EXISTS keeps this idempotent against a create_all'd schema.
    op.execute(
        "ALTER TABLE strategy_promotions ADD COLUMN IF NOT EXISTS engine_version VARCHAR(16)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_strategy_promotions_engine_version "
        "ON strategy_promotions (engine_version)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_strategy_promotions_engine_version")
    op.execute("ALTER TABLE strategy_promotions DROP COLUMN IF EXISTS engine_version")
