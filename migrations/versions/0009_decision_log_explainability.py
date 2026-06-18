"""Decision log + trade explainability schema (Section 24).

Adds ``decision_logs`` (per-signal chosen action + rejected alternatives + decision-time
features + version stamps) and ``trade_explainability`` (the full per-trade explainability
schema). Schema is derived from the ORM metadata via ``create_all`` so it can never drift
from ``src.db.models``. Idempotent: ``create_all(checkfirst=True)`` only creates missing objects.

Revision ID: 0009
Revises: 0008
Create Date: 2026-06-19
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
from src.db import models  # noqa: F401  (registers tables on Base.metadata)
from src.db.base import Base

revision: str = "0009"
down_revision: str | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_NEW_TABLES = ("decision_logs", "trade_explainability")


def upgrade() -> None:
    bind = op.get_bind()
    tables = [Base.metadata.tables[name] for name in _NEW_TABLES]
    Base.metadata.create_all(bind=bind, tables=tables, checkfirst=True)


def downgrade() -> None:
    bind = op.get_bind()
    tables = [Base.metadata.tables[name] for name in _NEW_TABLES]
    Base.metadata.drop_all(bind=bind, tables=tables)
