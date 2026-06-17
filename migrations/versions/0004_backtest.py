"""Phase 4 backtest schema.

Adds ``backtest_runs`` (the relational index/summary for event-based backtest,
walk-forward and fee/slippage-stress runs; the full Section 19 reports live in
the reports lake as JSON). Schema is derived from the ORM metadata via
``create_all`` so it can never drift from ``src.db.models``. Idempotent:
``create_all(checkfirst=True)`` only creates missing objects, and Alembic's
version table makes repeated upgrades a no-op.

Revision ID: 0004
Revises: 0003
Create Date: 2026-06-17
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
from src.db import models  # noqa: F401  (registers tables on Base.metadata)
from src.db.base import Base

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_NEW_TABLES = ("backtest_runs",)


def upgrade() -> None:
    bind = op.get_bind()
    tables = [Base.metadata.tables[name] for name in _NEW_TABLES]
    Base.metadata.create_all(bind=bind, tables=tables, checkfirst=True)


def downgrade() -> None:
    bind = op.get_bind()
    tables = [Base.metadata.tables[name] for name in _NEW_TABLES]
    Base.metadata.drop_all(bind=bind, tables=tables)
