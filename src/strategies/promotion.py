"""Strategy promotion registry (AGENTS.md Section 12/13).

The research harness (``src.strategies.research.validate_all``) decides which candidates are
promoted vs shelved. This module persists that verdict and lets the paper/live pipeline source
candidates ONLY from promoted strategies — closing the gap where a research verdict was written
to a report and read by nothing.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime

from sqlalchemy import select

from src.config import get_settings
from src.db.base import session_scope
from src.db.models import StrategyPromotion
from src.strategies.research import CandidateValidation


def persist_validations(validations: Iterable[CandidateValidation]) -> int:
    """Upsert a StrategyPromotion row per validation (keyed by candidate + version)."""
    versions = get_settings().versions()
    written = 0
    with session_scope() as session:
        for v in validations:
            row = (
                session.execute(
                    select(StrategyPromotion).where(
                        StrategyPromotion.candidate_id == v.candidate_id,
                        StrategyPromotion.strategy_version == v.strategy_version,
                    )
                )
                .scalars()
                .first()
            )
            if row is None:
                row = StrategyPromotion(
                    candidate_id=v.candidate_id, strategy_version=v.strategy_version
                )
                session.add(row)
            row.family = v.family
            row.promoted = v.promoted
            row.status = v.status
            row.expectancy_r = (
                float(v.report.get("expectancy_r", 0.0)) if isinstance(v.report, dict) else 0.0
            )
            row.allow_long = bool(v.side_decision.allow_long)
            row.allow_short = bool(v.side_decision.allow_short)
            row.shelved_reasons = list(v.shelved_reasons)
            row.summary = {"side_decision": v.side_decision.to_dict()}
            row.validated_at = datetime.now(UTC)
            row.related_versions = versions
            written += 1
    return written


def promoted_strategies(strategy_version: str | None = None) -> list[str]:
    """candidate_ids of currently-promoted strategies (optionally for one version)."""
    with session_scope() as session:
        q = select(StrategyPromotion).where(StrategyPromotion.promoted.is_(True))
        if strategy_version:
            q = q.where(StrategyPromotion.strategy_version == strategy_version)
        return [r.candidate_id for r in session.execute(q).scalars().all()]


def is_strategy_promoted(candidate_id: str, strategy_version: str | None = None) -> bool:
    """Whether a candidate has a promoted verdict (optionally pinned to a version)."""
    with session_scope() as session:
        q = select(StrategyPromotion).where(
            StrategyPromotion.candidate_id == candidate_id,
            StrategyPromotion.promoted.is_(True),
        )
        if strategy_version:
            q = q.where(StrategyPromotion.strategy_version == strategy_version)
        return session.execute(q).scalars().first() is not None
