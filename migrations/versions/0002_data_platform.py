"""Phase 2 data-platform schema.

Adds ``dataset_versions`` (immutable dataset snapshots / manifests, Appendix
B.4/B.5) and ``data_quality_reports`` (persisted Section 8/23 validation
reports). Schema is derived from the ORM metadata via ``create_all`` so it can
never drift from ``src.db.models``. Idempotent: ``create_all(checkfirst=True)``
only creates missing objects, and Alembic's version table makes repeated
upgrades a no-op.

Revision ID: 0002
Revises: 0001
Create Date: 2026-06-17
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

from src.db import models  # noqa: F401  (registers tables on Base.metadata)
from src.db.base import Base

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_NEW_TABLES = ("dataset_versions", "data_quality_reports")


def upgrade() -> None:
    bind = op.get_bind()
    tables = [Base.metadata.tables[name] for name in _NEW_TABLES]
    Base.metadata.create_all(bind=bind, tables=tables, checkfirst=True)


def downgrade() -> None:
    bind = op.get_bind()
    tables = [Base.metadata.tables[name] for name in _NEW_TABLES]
    Base.metadata.drop_all(bind=bind, tables=tables)
