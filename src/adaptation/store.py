"""Learner log persistence (AGENTS.md Section 21.8, Section 24).

Writes :class:`LearnerLogEntry` rows to the ``learner_logs`` DB table and
:class:`Recommendation` records to ``learner_recommendations``. Also provides
a lightweight in-memory sink for unit tests (``write_to_db=False``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from src.adaptation.action_space import BoundedAction

if TYPE_CHECKING:
    pass


@dataclass
class LearnerLogEntry:
    """One learner decision log row (Section 21.8 LearnerLogEntry schema)."""

    ts: datetime
    learner_id: str
    learner_version: str
    mode: str  # SHADOW | RECOMMEND | LIVE_BOUNDED
    symbol: str | None
    context_features: dict
    proposed_action: dict  # serialised BoundedAction
    projected_outcome: float
    realized_outcome: float | None
    applied: bool
    clamped_fields: list[str]
    rollback_event: str | None
    config_version: str


@dataclass
class InMemoryLearnerStore:
    """In-memory sink for tests and shadow evaluation (write_to_db=False path)."""

    entries: list[LearnerLogEntry] = field(default_factory=list)

    def write(self, entry: LearnerLogEntry) -> None:
        self.entries.append(entry)

    def recent(self, limit: int = 100) -> list[LearnerLogEntry]:
        return self.entries[-limit:]


# Module-level in-memory sink (used in tests and gate self-checks).
_memory_sink: InMemoryLearnerStore = InMemoryLearnerStore()


def get_memory_sink() -> InMemoryLearnerStore:
    return _memory_sink


def reset_memory_sink() -> None:
    global _memory_sink  # noqa: PLW0603
    _memory_sink = InMemoryLearnerStore()


def write_learner_log(
    *,
    learner_id: str,
    learner_version: str,
    mode: str,
    symbol: str | None,
    context_features: dict,
    proposed_action: BoundedAction,
    projected_outcome: float,
    realized_outcome: float | None,
    applied: bool,
    clamped_fields: list[str],
    rollback_event: str | None = None,
    config_version: str = "cfg_0001",
    write_to_db: bool = True,
) -> LearnerLogEntry:
    """Write one learner decision to the learner log.

    If ``write_to_db=True`` the row is also persisted to ``learner_logs``.
    The in-memory sink always receives the entry (for tests and gate checks).
    """
    action_dict = {
        "strategy_weights": proposed_action.strategy_weights,
        "size_bucket": proposed_action.size_bucket,
        "take": proposed_action.take,
        "exec_style": proposed_action.exec_style,
        "param_nudges": proposed_action.param_nudges,
        "learner_id": proposed_action.learner_id,
        "learner_version": proposed_action.learner_version,
        "mode": proposed_action.mode,
        "rationale": proposed_action.rationale,
    }
    entry = LearnerLogEntry(
        ts=datetime.now(UTC),
        learner_id=learner_id,
        learner_version=learner_version,
        mode=mode,
        symbol=symbol,
        context_features=context_features,
        proposed_action=action_dict,
        projected_outcome=projected_outcome,
        realized_outcome=realized_outcome,
        applied=applied,
        clamped_fields=clamped_fields,
        rollback_event=rollback_event,
        config_version=config_version,
    )
    _memory_sink.write(entry)

    if write_to_db:
        _write_to_db(entry)

    return entry


def _write_to_db(entry: LearnerLogEntry) -> None:
    try:
        from src.db.base import session_scope
        from src.db.models import LearnerLog

        with session_scope() as session:
            row = LearnerLog(
                ts=entry.ts,
                learner_id=entry.learner_id,
                learner_version=entry.learner_version,
                mode=entry.mode,
                symbol=entry.symbol,
                context_features=entry.context_features,
                proposed_action=entry.proposed_action,
                projected_outcome=entry.projected_outcome,
                realized_outcome=entry.realized_outcome,
                applied=entry.applied,
                clamped_fields=entry.clamped_fields,
                rollback_event=entry.rollback_event,
                config_version=entry.config_version,
            )
            session.add(row)
    except Exception:  # noqa: BLE001
        pass  # never block on log failure
