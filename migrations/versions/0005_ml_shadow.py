"""Phase 9 ML Shadow schema.

Adds ``shadow_logs`` (AGENTS.md Section 24 shadow_log) and
``ml_model_registry`` (Section 20 model registry). Schema is derived from
the ORM metadata via ``create_all`` so it can never drift from
``src.db.models``. Idempotent: ``create_all(checkfirst=True)`` only creates
missing objects, and Alembic's version table makes repeated upgrades a no-op.

Revision ID: 0005
Revises: 0004
Create Date: 2026-06-17
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
from src.db import models  # noqa: F401  (registers tables on Base.metadata)
from src.db.base import Base

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_NEW_TABLES = ("shadow_logs", "ml_model_registry")


def upgrade() -> None:
    bind = op.get_bind()
    tables = [Base.metadata.tables[name] for name in _NEW_TABLES]
    Base.metadata.create_all(bind=bind, tables=tables, checkfirst=True)


def downgrade() -> None:
    bind = op.get_bind()
    tables = [Base.metadata.tables[name] for name in _NEW_TABLES]
    Base.metadata.drop_all(bind=bind, tables=tables)
