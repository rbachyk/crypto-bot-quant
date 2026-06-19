"""Environment-scoped statistics admin for live/demo/testnet runs (AGENTS.md Section 26/34).

Live, demo, and testnet sessions all persist through the same paper tables, separated only by
the ``env:`` prefix on ``session_id`` (see :meth:`src.live.loop.LiveLoop.env_label`). That
prefix lets an operator **zero one environment's statistics without touching the others** — the
explicit requirement before starting a fresh demo-testing run: clear all prior ``demo:`` history
while leaving paper/testnet/live data intact.

Everything here is a synchronous DB operation (no job worker needed) so the dashboard Reset
button gives immediate, visible feedback.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import delete, func, select

from src.db.base import session_scope
from src.db.models import (
    DecisionLog,
    PaperRun,
    PaperTradeRecord,
    TradeExplainabilityRow,
)

# The four tables a session writes; all keyed by the env-prefixed session_id.
_SESSION_TABLES = (PaperRun, PaperTradeRecord, DecisionLog, TradeExplainabilityRow)


def _prefix(env: str) -> str:
    """SQL LIKE pattern matching every session_id for one environment (``demo:%``)."""
    return f"{env}:%"


@dataclass(slots=True)
class EnvStatsSummary:
    """Counts for one environment's persisted statistics (for confirm-before-reset display)."""

    env: str
    runs: int
    trades: int
    decision_logs: int
    explainability: int

    @property
    def total(self) -> int:
        return self.runs + self.trades + self.decision_logs + self.explainability

    def to_dict(self) -> dict[str, int | str]:
        return {
            "env": self.env,
            "runs": self.runs,
            "trades": self.trades,
            "decision_logs": self.decision_logs,
            "explainability": self.explainability,
            "total": self.total,
        }


def summarize_env_stats(env: str = "demo") -> EnvStatsSummary:
    """Count the persisted rows for one environment (``demo`` by default)."""
    like = _prefix(env)
    with session_scope() as db:
        counts = [
            db.execute(
                select(func.count()).select_from(model).where(model.session_id.like(like))
            ).scalar_one()
            for model in _SESSION_TABLES
        ]
    return EnvStatsSummary(env, *counts)


def reset_env_stats(env: str = "demo") -> EnvStatsSummary:
    """Delete every persisted row for one environment and return what was removed.

    Only rows whose ``session_id`` starts with ``"{env}:"`` are touched — paper/testnet/live
    history is left untouched. Safe to call when there is nothing to delete (returns zeros)."""
    removed = summarize_env_stats(env)
    like = _prefix(env)
    with session_scope() as db:
        for model in _SESSION_TABLES:
            db.execute(delete(model).where(model.session_id.like(like)))
    return removed
