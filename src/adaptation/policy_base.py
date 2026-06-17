"""Policy Protocol and Context/Outcome types (AGENTS.md Section 21.10).

A :class:`Policy` is the minimal interface every learner must implement.
It is a subordinate advisor: it emits a :class:`~.action_space.BoundedAction`,
which then passes through :func:`~.action_space.validate` →
:func:`~.envelope_guard.enforce` before reaching the Risk Layer.

The learner NEVER calls exchange APIs or the execution engine directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from src.adaptation.action_space import BoundedAction


@dataclass
class Context:
    """Inputs available to the learner at decision time (reproducible features).

    Only data available at decision time is included here (Parity Rule,
    Section 10). The full dict is stored in ``learner_log.context_features``
    for offline replay and scoring.
    """

    ts: datetime = field(default_factory=lambda: datetime.now(UTC))
    symbol: str | None = None
    regime: str | None = None
    signal_strength: float = 0.0
    expected_edge_frac: float = 0.0
    spread_bps: float = 0.0
    slippage_est: float = 0.0
    atr_pct: float = 0.0
    funding_z: float = 0.0
    strategy_id: str | None = None
    config_version: str = "cfg_0001"
    # Additional free-form features from the feature pipeline.
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "ts": self.ts.isoformat(),
            "symbol": self.symbol,
            "regime": self.regime,
            "signal_strength": self.signal_strength,
            "expected_edge_frac": self.expected_edge_frac,
            "spread_bps": self.spread_bps,
            "slippage_est": self.slippage_est,
            "atr_pct": self.atr_pct,
            "funding_z": self.funding_z,
            "strategy_id": self.strategy_id,
            "config_version": self.config_version,
            **self.extra,
        }


@dataclass
class Outcome:
    """Realized outcome after a decision (filled in post-trade / post-bar)."""

    realized_pnl_r: float | None = None  # realised P&L in R-units
    trade_taken: bool = False
    fill_ts: datetime | None = None


@runtime_checkable
class Policy(Protocol):
    """Minimal interface every learner must implement (Section 21.10)."""

    def decide(self, ctx: Context) -> BoundedAction:
        """Produce a bounded action for the given context.

        In SHADOW and RECOMMEND modes the action is logged but not applied to
        real orders. In LIVE_BOUNDED it may influence the order after Risk
        approves.
        """
        ...

    def update(self, ctx: Context, action: BoundedAction, outcome: Outcome) -> None:
        """Update internal state from the realised outcome.

        In SHADOW and RECOMMEND modes this is a no-op (the controller enforces
        this — implementations should still handle the call gracefully).
        """
        ...

    def snapshot(self) -> bytes:
        """Serialise current policy state to bytes (for versioning and rollback)."""
        ...

    def load(self, blob: bytes) -> None:
        """Restore policy state from a :meth:`snapshot` blob."""
        ...
