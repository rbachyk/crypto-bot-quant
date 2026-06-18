"""Trade explainability + decision logging (AGENTS.md Section 24).

Every live trade must be explainable via :class:`TradeExplainability`. If the schema cannot
be fully populated, the trade is **not taken** — :meth:`TradeExplainability.ensure_complete`
raises :class:`ExplainabilityError`, which the execution path treats as a hard block.

Decision logging (chosen action + rejected alternatives + decision-time features + version
stamps) and explainability persistence are written **off the execution hot path** (after the
decision/fill is committed), so they never block trading (Section 24).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


class ExplainabilityError(RuntimeError):
    """Raised when a trade cannot be fully explained — the trade must not be taken."""


@dataclass(slots=True)
class TradeExplainability:
    """The Section-24 explainability schema for one executed trade."""

    trade_id: str
    symbol: str
    strategy_id: str
    setup_type: str
    regime: str
    signal_features: dict[str, float]
    expected_edge_after_costs: float
    expected_fees: float
    expected_slippage: float
    stop_price: float
    execution_route: str
    risk_approved: bool
    risk_reason: str
    config_version: str
    universe_version: str
    why_selected: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    expected_funding_impact: float | None = None
    invalidation_conditions: list[str] = field(default_factory=list)
    model_version: str | None = None
    learner_version: str | None = None
    why_rejected_others: list[dict] = field(default_factory=list)

    # Fields that MUST be present/non-trivial for a trade to be allowed.
    _REQUIRED_STR = (
        "trade_id",
        "symbol",
        "strategy_id",
        "regime",
        "execution_route",
        "config_version",
        "universe_version",
        "why_selected",
    )

    def ensure_complete(self) -> TradeExplainability:
        """Raise :class:`ExplainabilityError` if the schema is not fully populated."""
        for name in self._REQUIRED_STR:
            if not str(getattr(self, name) or "").strip():
                raise ExplainabilityError(f"trade not taken: explainability.{name} is empty")
        if self.stop_price <= 0:
            raise ExplainabilityError("trade not taken: explainability.stop_price must be > 0")
        if not self.signal_features:
            raise ExplainabilityError("trade not taken: explainability.signal_features empty")
        if not self.invalidation_conditions:
            raise ExplainabilityError(
                "trade not taken: explainability.invalidation_conditions empty"
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "strategy_id": self.strategy_id,
            "setup_type": self.setup_type,
            "regime": self.regime,
            "signal_features": self.signal_features,
            "expected_edge_after_costs": self.expected_edge_after_costs,
            "expected_fees": self.expected_fees,
            "expected_slippage": self.expected_slippage,
            "expected_funding_impact": self.expected_funding_impact,
            "stop_price": self.stop_price,
            "invalidation_conditions": self.invalidation_conditions,
            "execution_route": self.execution_route,
            "risk_approved": self.risk_approved,
            "risk_reason": self.risk_reason,
            "model_version": self.model_version,
            "learner_version": self.learner_version,
            "config_version": self.config_version,
            "universe_version": self.universe_version,
            "why_selected": self.why_selected,
            "why_rejected_others": self.why_rejected_others,
        }


def write_trade_explainability(te: TradeExplainability, *, session_id: str | None = None) -> None:
    """Persist a (validated) explainability record to ``trade_explainability``."""
    te.ensure_complete()
    from src.db.base import session_scope
    from src.db.models import TradeExplainabilityRow

    with session_scope() as session:
        existing = (
            session.query(TradeExplainabilityRow).filter_by(trade_id=te.trade_id).one_or_none()
        )
        row = existing or TradeExplainabilityRow(trade_id=te.trade_id)
        if existing is None:
            session.add(row)
        row.session_id = session_id
        row.symbol = te.symbol
        row.strategy_id = te.strategy_id
        row.regime = te.regime
        row.payload = te.to_dict()


def write_decision_log(
    *,
    symbol: str,
    strategy: str,
    action: str,
    reason: str = "",
    side: int = 0,
    strategy_version: str = "",
    rejected_alternatives: list[dict] | None = None,
    features: dict | None = None,
    expected_edge: float = 0.0,
    expected_cost: float = 0.0,
    risk_approved: bool = False,
    config_version: str = "",
    model_version: str | None = None,
    universe_version: str | None = None,
    kill_switch_state: str = "clear",
    session_id: str | None = None,
) -> None:
    """Append a per-signal decision row to ``decision_logs`` (off the hot path)."""
    from src.db.base import session_scope
    from src.db.models import DecisionLog

    with session_scope() as session:
        session.add(
            DecisionLog(
                session_id=session_id,
                symbol=symbol,
                strategy=strategy,
                strategy_version=strategy_version,
                side=side,
                action=action,
                reason=reason,
                rejected_alternatives=rejected_alternatives or [],
                features=features or {},
                expected_edge=expected_edge,
                expected_cost=expected_cost,
                risk_approved=risk_approved,
                config_version=config_version,
                model_version=model_version,
                universe_version=universe_version,
                kill_switch_state=kill_switch_state,
            )
        )
