"""Strategy promotion schema.

Adds ``strategy_promotions`` (research promote/shelve verdicts, Section 12/13). Schema is
derived from the ORM metadata via ``create_all`` so it can never drift from ``src.db.models``.
Idempotent: ``create_all(checkfirst=True)`` only creates missing objects.

Revision ID: 0007
Revises: 0006
Create Date: 2026-06-18
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
from src.db import models  # noqa: F401  (registers tables on Base.metadata)
from src.db.base import Base

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_NEW_TABLES = ("strategy_promotions",)


def upgrade() -> None:
    bind = op.get_bind()
    tables = [Base.metadata.tables[name] for name in _NEW_TABLES]
    Base.metadata.create_all(bind=bind, tables=tables, checkfirst=True)


def downgrade() -> None:
    bind = op.get_bind()
    tables = [Base.metadata.tables[name] for name in _NEW_TABLES]
    Base.metadata.drop_all(bind=bind, tables=tables)
