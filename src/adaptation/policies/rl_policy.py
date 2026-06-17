"""RL policy stub — research/shadow only (AGENTS.md Section 21.4, 21.5).

Phase 11 delivers this as a stub that can be imported and logged in shadow mode.
A full RL policy (gymnasium + Stable-Baselines3) is delivered in Phase 12.

The stub always emits a neutral :class:`~src.adaptation.action_space.BoundedAction`
(take=True, size_bucket=1.0, no weights/nudges) so that Phase 11 gate checks can
verify it can be imported and produces valid bounded actions.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field

from src.adaptation.action_space import BoundedAction
from src.adaptation.policy_base import Context, Outcome


@dataclass
class RLPolicyStub:
    """Phase 11 stub. Phase 12 replaces this with the trained RL policy.

    Always operates in SHADOW mode; ``update()`` is a no-op in this stub.
    """

    learner_id: str = "rl_policy_stub_v1"
    learner_version: str = "learner_0001"
    _n_decisions: int = field(default=0, init=False, repr=False)

    def decide(self, ctx: Context) -> BoundedAction:
        self._n_decisions += 1
        return BoundedAction(
            strategy_weights={},
            size_bucket=1.0,
            take=True,
            exec_style="maker",
            param_nudges={},
            learner_id=self.learner_id,
            learner_version=self.learner_version,
            mode="SHADOW",
            rationale=f"rl_stub decision #{self._n_decisions}",
        )

    def update(self, ctx: Context, action: BoundedAction, outcome: Outcome) -> None:
        pass  # no-op in stub; full RL in Phase 12

    def snapshot(self) -> bytes:
        return pickle.dumps(
            {
                "learner_id": self.learner_id,
                "learner_version": self.learner_version,
                "n_decisions": self._n_decisions,
            }
        )

    def load(self, blob: bytes) -> None:
        state = pickle.loads(blob)  # noqa: S301
        self.learner_id = state["learner_id"]
        self.learner_version = state["learner_version"]
        self._n_decisions = state.get("n_decisions", 0)
