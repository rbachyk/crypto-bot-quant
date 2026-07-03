"""Learner log persistence (AGENTS.md Section 21.8, Section 24).

Writes :class:`LearnerLogEntry` rows to the ``learner_logs`` DB table and
:class:`Recommendation` records to ``learner_recommendations``. Also provides
a lightweight in-memory sink for unit tests (``write_to_db=False``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from src.adaptation.action_space import BoundedAction

if TYPE_CHECKING:
    from src.monitoring.health import ComponentHealth

logger = logging.getLogger(__name__)

# Count of failed learner_log DB writes since process start (audit L30): the
# Section 21.8 audit trail silently stopping is a degraded-health condition
# that must be visible, not swallowed.
_db_write_failures: int = 0


def get_db_write_failure_count() -> int:
    """Number of learner_log DB writes that failed since process start."""
    return _db_write_failures


def reset_db_write_failure_count() -> None:
    global _db_write_failures  # noqa: PLW0603
    _db_write_failures = 0


def learner_store_health() -> ComponentHealth:
    """Health probe in the :mod:`src.monitoring.health` component style.

    Unhealthy as soon as any learner_log DB write has failed — the Section 21.8
    audit log is then incomplete and needs operator attention.
    """
    from src.monitoring.health import ComponentHealth

    if _db_write_failures == 0:
        return ComponentHealth("learner_store", True, "all learner_log DB writes persisted")
    return ComponentHealth(
        "learner_store",
        False,
        f"{_db_write_failures} learner_log DB write(s) failed — audit log incomplete",
    )


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
        # M31 contract fields (see policy_base): R-scale projection and the
        # policy's genuine probability (None when the policy has none).
        "projected_outcome_r": proposed_action.projected_outcome_r,
        "win_probability": proposed_action.win_probability,
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
        # Never block the decision path on a log failure — but never hide it
        # either (audit L30): count it and log loudly so repeated failures are
        # visible via learner_store_health().
        global _db_write_failures  # noqa: PLW0603
        _db_write_failures += 1
        logger.exception(
            "learner_log DB write FAILED (failure #%d, learner_id=%s mode=%s) — "
            "the Section 21.8 audit log is not persisting; check learner_store_health()",
            _db_write_failures,
            entry.learner_id,
            entry.mode,
        )
