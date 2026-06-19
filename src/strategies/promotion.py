"""Strategy promotion registry (AGENTS.md Section 12/13).

The research harness (``src.strategies.research.validate_all``) decides which candidates are
promoted vs shelved. This module persists that verdict and lets the paper/live pipeline source
candidates ONLY from promoted strategies — closing the gap where a research verdict was written
to a report and read by nothing.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import desc, select

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


@dataclass(frozen=True, slots=True)
class PromotedStrategy:
    """A promoted candidate with the validated score the live engine ranks the top-N by."""

    candidate_id: str
    family: str
    strategy_version: str
    expectancy_r: float
    allow_long: bool
    allow_short: bool
    active: bool = False  # within the top-N the live/demo engine actually runs


def promoted_strategy_details(strategy_version: str | None = None) -> list[PromotedStrategy]:
    """All promoted strategies, ranked by validated expectancy_r (desc).

    The first ``max_active_strategies`` are flagged ``active=True`` — the set the live/demo
    engine runs concurrently (Section 13). The rest are promoted-but-benched."""
    from src.strategies.config import load_strategies_config

    cap = load_strategies_config().max_active_strategies
    with session_scope() as session:
        q = select(StrategyPromotion).where(StrategyPromotion.promoted.is_(True))
        if strategy_version:
            q = q.where(StrategyPromotion.strategy_version == strategy_version)
        rows = list(
            session.execute(q.order_by(desc(StrategyPromotion.expectancy_r))).scalars().all()
        )
    out: list[PromotedStrategy] = []
    for i, r in enumerate(rows):
        out.append(
            PromotedStrategy(
                candidate_id=r.candidate_id,
                family=r.family,
                strategy_version=r.strategy_version,
                expectancy_r=float(r.expectancy_r),
                allow_long=bool(r.allow_long),
                allow_short=bool(r.allow_short),
                active=(cap <= 0 or i < cap),
            )
        )
    return out


def active_strategy_ids(
    strategy_version: str | None = None, *, limit: int | None = None
) -> list[str]:
    """candidate_ids of the strategies the live/demo engine runs — the top-N promoted by
    expectancy_r (``limit`` overrides ``max_active_strategies``; ``None``/0 = the config cap)."""
    details = [d for d in promoted_strategy_details(strategy_version) if d.active]
    if limit is not None and limit > 0:
        details = details[:limit]
    return [d.candidate_id for d in details]


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
