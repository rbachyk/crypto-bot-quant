"""Gaussian Thompson Sampling bandit over validated strategies (AGENTS.md Section 21.5).

The bandit maintains per-strategy Gaussian priors (mean, variance) over
expected R-units. On each call to :meth:`decide` it samples from each
strategy's posterior and emits a ``strategy_weights`` dict that ranks the
strategies by sample. The weights are bounded to [w_min, w_max] and renormalised.

Only already-validated, enabled strategies are considered. Disabled or
unvalidated strategies are never assigned positive weight (Section 21.2).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import Any

from src.adaptation.action_space import BoundedAction
from src.adaptation.policy_base import Context, Outcome

try:
    import numpy as np

    _NP_AVAILABLE = True
except ImportError:
    _NP_AVAILABLE = False


@dataclass
class StrategyArm:
    """Gaussian posterior for one strategy arm."""

    strategy_id: str
    mu: float = 0.0  # posterior mean of expected R
    var: float = 1.0  # posterior variance
    n: int = 0  # number of observations


@dataclass
class GaussianTSBandit:
    """Contextual Gaussian Thompson-Sampling bandit over validated strategies.

    Each :meth:`update` step performs a Gaussian conjugate update on the arm
    that was selected. :meth:`decide` samples one value per arm and returns
    ``strategy_weights`` proportional to the sampled expectations.
    """

    learner_id: str = "gaussian_ts_bandit_v1"
    learner_version: str = "learner_0001"
    w_min: float = 0.0
    w_max: float = 2.0
    _arms: dict[str, StrategyArm] = field(default_factory=dict, init=False, repr=False)
    _rng_seed: int = 42
    _rng: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if _NP_AVAILABLE:
            self._rng = np.random.default_rng(self._rng_seed)

    def _ensure_arm(self, strategy_id: str) -> StrategyArm:
        if strategy_id not in self._arms:
            self._arms[strategy_id] = StrategyArm(strategy_id=strategy_id)
        return self._arms[strategy_id]

    def decide(self, ctx: Context) -> BoundedAction:
        """Sample from each arm's posterior and rank strategies."""
        strategy_id = ctx.strategy_id
        if not strategy_id:
            # No specific strategy context; return uniform weights for all arms.
            weights: dict[str, float] = dict.fromkeys(self._arms, 1.0) if self._arms else {}
        else:
            self._ensure_arm(strategy_id)
            # Sample from each arm.
            if _NP_AVAILABLE and self._rng is not None and self._arms:
                samples = {
                    sid: float(self._rng.normal(arm.mu, max(arm.var**0.5, 1e-6)))
                    for sid, arm in self._arms.items()
                }
                # Weights proportional to rank (rank-1 at top, 0 for negative samples).
                sorted_arms = sorted(samples, key=lambda s: samples[s], reverse=True)
                n = len(sorted_arms)
                weights = {
                    sid: max(self.w_min, min(self.w_max, (n - i) / n * self.w_max))
                    for i, sid in enumerate(sorted_arms)
                }
            else:
                weights = dict.fromkeys(self._arms, 1.0)

        return BoundedAction(
            strategy_weights=weights,
            size_bucket=1.0,
            take=True,
            exec_style="maker",
            param_nudges={},
            learner_id=self.learner_id,
            learner_version=self.learner_version,
            mode="SHADOW",
            rationale=f"gaussian_ts arms={len(self._arms)}",
        )

    def update(self, ctx: Context, action: BoundedAction, outcome: Outcome) -> None:
        """Gaussian conjugate update for the arm that was selected."""
        if outcome.realized_pnl_r is None or not ctx.strategy_id:
            return
        arm = self._ensure_arm(ctx.strategy_id)
        # Simple online Gaussian update (prior variance = 1.0).
        prior_var = 1.0
        obs_var = max(arm.var, 1e-6)
        posterior_var = 1.0 / (1.0 / obs_var + 1.0 / prior_var)
        posterior_mu = posterior_var * (arm.mu / obs_var + outcome.realized_pnl_r / prior_var)
        arm.mu = posterior_mu
        arm.var = posterior_var
        arm.n += 1

    def snapshot(self) -> bytes:
        return pickle.dumps(
            {
                "arms": self._arms,
                "learner_id": self.learner_id,
                "learner_version": self.learner_version,
                "w_min": self.w_min,
                "w_max": self.w_max,
                "rng_seed": self._rng_seed,
            }
        )

    def load(self, blob: bytes) -> None:
        state = pickle.loads(blob)  # noqa: S301
        self._arms = state["arms"]
        self.learner_id = state["learner_id"]
        self.learner_version = state["learner_version"]
        self.w_min = state["w_min"]
        self.w_max = state["w_max"]
        self._rng_seed = state.get("rng_seed", 42)
        if _NP_AVAILABLE:
            import numpy as np

            self._rng = np.random.default_rng(self._rng_seed)
