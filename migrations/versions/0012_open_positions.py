"""Open positions (live unrealized-P&L tracking).

Adds ``open_positions`` — currently-held basket legs / live entries, marked to market by the
running session so the dashboard can show unrealized P&L until they close. Live state, not history
(closed positions become ``paper_trades``). Schema derived from the ORM metadata via ``create_all``
so it can never drift from ``src.db.models``; idempotent (``checkfirst=True``).

Revision ID: 0012
Revises: 0011
Create Date: 2026-06-28
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
from src.db import models  # noqa: F401  (registers tables on Base.metadata)
from src.db.base import Base

revision: str = "0012"
down_revision: str | None = "0011"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_NEW_TABLES = ("open_positions",)


def upgrade() -> None:
    bind = op.get_bind()
    tables = [Base.metadata.tables[name] for name in _NEW_TABLES]
    Base.metadata.create_all(bind=bind, tables=tables, checkfirst=True)


def downgrade() -> None:
    bind = op.get_bind()
    tables = [Base.metadata.tables[name] for name in _NEW_TABLES]
    Base.metadata.drop_all(bind=bind, tables=tables)
