"""Learner state machine — SHADOW → RECOMMEND → LIVE_BOUNDED (AGENTS.md Section 21.7).

The :class:`LearnerController` orchestrates the full decision path:

    deterministic candidate
        → policy.decide(ctx)
        → action_space.validate(action, bounds)
        → envelope_guard.enforce(action)
        → (LIVE_BOUNDED only) risk_manager.approve(order) [external]
        → store.write_learner_log(...)

In SHADOW and RECOMMEND modes ``update()`` is still called so the model trains
online; but the resulting action is logged with ``applied=False``.

Promotion (SHADOW → RECOMMEND, RECOMMEND → LIVE_BOUNDED) is MANUAL only:
a human operator must approve after the LEARN-PROMO-S / LEARN-PROMO-L gates
pass (Section 21.3, 27). This class never auto-promotes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum

from src.adaptation.action_space import ActionBounds, BoundedAction, validate
from src.adaptation.envelope_guard import enforce
from src.adaptation.policy_base import Context, Outcome, Policy


class LearnerMode(str, Enum):
    SHADOW = "SHADOW"
    RECOMMEND = "RECOMMEND"
    LIVE_BOUNDED = "LIVE_BOUNDED"
    FROZEN = "FROZEN"


@dataclass
class ControllerDecision:
    """Result of one controller.run() call."""

    action: BoundedAction | None
    applied: bool
    clamped_fields: list[str]
    rejected: bool
    rejection_reason: str | None
    mode: str
    ts: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class LearnerController:
    """Orchestrates the learner decision path (Section 21.7).

    Parameters
    ----------
    policy:
        The active learner policy (OnlineLogRegPolicy, GaussianTSBandit, etc.).
    bounds:
        The declared action bounds from adaptation.yaml.
    mode:
        Current operational mode (default SHADOW — never auto-promoted).
    frozen_policy:
        A fallback policy activated on rollback (Section 21.7). Loaded from the
        frozen-fallback path in adaptation.yaml.
    """

    policy: Policy
    bounds: ActionBounds
    mode: LearnerMode = LearnerMode.SHADOW
    frozen_policy: Policy | None = None
    _decision_count: int = field(default=0, init=False, repr=False)
    _frozen: bool = field(default=False, init=False, repr=False)

    def run(
        self,
        ctx: Context,
        *,
        active_strategies: set[str] | None = None,
    ) -> ControllerDecision:
        """Run one decision cycle.

        The action produced is validated and guard-enforced in every mode.
        ``applied=True`` only in LIVE_BOUNDED; the Risk Layer independently
        approves before any order is placed (not modelled here).
        """
        self._decision_count += 1
        effective_policy = self._effective_policy()
        raw_action = effective_policy.decide(ctx)
        # Force mode field to match controller mode.
        raw_action.mode = self.mode.value

        # Validate then guard.
        val = validate(raw_action, self.bounds)
        if val.rejected:
            return ControllerDecision(
                action=None,
                applied=False,
                clamped_fields=val.clamped_fields,
                rejected=True,
                rejection_reason=val.rejection_reason,
                mode=self.mode.value,
            )

        guard = enforce(
            val.action,
            active_strategies=active_strategies,
        )
        if guard.rejected:
            return ControllerDecision(
                action=None,
                applied=False,
                clamped_fields=guard.clamped_fields,
                rejected=True,
                rejection_reason=guard.rejection_reason,
                mode=self.mode.value,
            )

        applied = self.mode is LearnerMode.LIVE_BOUNDED and not self._frozen
        return ControllerDecision(
            action=guard.action,
            applied=applied,
            clamped_fields=val.clamped_fields + guard.clamped_fields,
            rejected=False,
            rejection_reason=None,
            mode=self.mode.value,
        )

    def record_outcome(
        self,
        ctx: Context,
        decision: ControllerDecision,
        outcome: Outcome,
    ) -> None:
        """Feed realized outcome back to the policy for online updates.

        In SHADOW and RECOMMEND modes the call is forwarded so the model trains;
        the controller guarantees the resulting action was never applied.
        """
        if decision.action is None:
            return
        self._effective_policy().update(ctx, decision.action, outcome)

    def freeze(self, reason: str = "") -> None:
        """Freeze the controller (rollback trigger — Section 21.7)."""
        self._frozen = True
        self.mode = LearnerMode.FROZEN

    def is_frozen(self) -> bool:
        return self._frozen

    def _effective_policy(self) -> Policy:
        if self._frozen and self.frozen_policy is not None:
            return self.frozen_policy
        return self.policy
