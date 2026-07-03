"""Gaussian Thompson Sampling bandit over validated strategies (AGENTS.md Section 21.5).

The bandit maintains per-strategy Gaussian posteriors (mean, variance) over
expected R-units. On each call to :meth:`decide` it samples one value from each
strategy's posterior and emits a ``strategy_weights`` dict of RANK-based
weights: arms are sorted by sampled value and the i-th ranked of n arms gets
weight ``(n - i) / n * w_max``, clamped to ``[w_min, w_max]``. The weights are
NOT renormalised to sum to 1, and arms with negative samples still receive
their (low) rank-based weight — an arm's weight only reaches ``w_min`` at the
bottom rank. Each :meth:`update` performs a standard Gaussian conjugate update
(known observation noise = 1 R²) on the arm of the decision's strategy.

Restriction to already-validated, enabled strategies is enforced downstream:
:func:`~src.adaptation.action_space.validate` drops weights for strategies
outside ``bounds.allowed_strategies`` and
:func:`~src.adaptation.envelope_guard.enforce` rejects unknown strategies
(Section 21.2); this class itself only creates arms it is asked about.

Outcome-projection contract (audit M31, see policy_base): the bandit's
per-decision expectation is R-scale — ``BoundedAction.projected_outcome_r`` is
the context arm's posterior mean. It emits no ``win_probability``.
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
    rank-based ``strategy_weights`` in ``[w_min, w_max]`` (top-ranked arm gets
    ``w_max``; weights are not renormalised — see module docstring).
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

        # M31 contract: projected outcome is R-scale — the context arm's
        # posterior mean expected R (None when no strategy context).
        projected_r = self._arms[strategy_id].mu if strategy_id in self._arms else None
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
            projected_outcome_r=projected_r,
            win_probability=None,  # the bandit has no calibrated probability
        )

    def update(self, ctx: Context, action: BoundedAction, outcome: Outcome) -> None:
        """Gaussian conjugate update for the arm that was selected."""
        if outcome.realized_pnl_r is None or not ctx.strategy_id:
            return
        arm = self._ensure_arm(ctx.strategy_id)
        # Gaussian conjugate update with known observation noise (1 R²); the
        # arm's current posterior acts as the prior for the next observation.
        # (Audit L31: names previously swapped — the math was already correct.)
        obs_noise_var = 1.0
        prior_var = max(arm.var, 1e-6)
        posterior_var = 1.0 / (1.0 / prior_var + 1.0 / obs_noise_var)
        posterior_mu = posterior_var * (arm.mu / prior_var + outcome.realized_pnl_r / obs_noise_var)
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
