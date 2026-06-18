"""RL shadow policy — Phase 12 (AGENTS.md Section 21.4, 21.5, 32).

Replaces the Phase 11 stub with a policy backed by a trained simulation model.
The model is a lightweight linear policy trained via cross-entropy method (CEM)
on the :class:`~src.rl.environment.TradingEnv` (no PyTorch/GPU required).

Key invariants (hard; enforced here + by envelope_guard):
  - Always operates in SHADOW mode until RL-SIM + RL-SHADOW gates pass AND
    the operator manually promotes (Section 21.3, AGENTS.md 27).
  - ``update()`` is a no-op in SHADOW mode; the policy only trains offline.
  - Emits only valid :class:`~src.adaptation.action_space.BoundedAction` instances.
  - No live trading influence whatsoever.

The class exposes both a pre-built default (trained at import time with a
short simulation run) and a ``from_trained`` constructor for loading
externally trained weights.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field

import numpy as np

from src.adaptation.action_space import BoundedAction
from src.adaptation.policy_base import Context, Outcome

# Learner identity constants.
LEARNER_ID = "rl_policy_v1"
LEARNER_VERSION = "learner_0002"  # incremented from phase-11 stub's learner_0001


@dataclass
class RLPolicy:
    """RL shadow policy backed by a simulation-trained linear model.

    Attributes
    ----------
    learner_id:
        Stable identifier for this policy class.
    learner_version:
        Incremented on each training run.
    weights:
        Trained weight matrix W (obs_dim × n_actions). None → uses heuristic fallback.
    """

    learner_id: str = LEARNER_ID
    learner_version: str = LEARNER_VERSION
    weights: np.ndarray | None = field(default=None, repr=False)
    _n_decisions: int = field(default=0, init=False, repr=False)

    def decide(self, ctx: Context) -> BoundedAction:
        """Produce a SHADOW-mode :class:`BoundedAction` from the context.

        If trained weights are available, uses the linear policy. Otherwise
        falls back to a simple heuristic (take=True when signal_strength>0.5,
        size_bucket=0.5, exec_style="maker").
        """
        self._n_decisions += 1
        obs = self._ctx_to_obs(ctx)

        if self.weights is not None:
            action_arr = self._predict(obs)
            from src.rl.environment import EXEC_MAP, SIZE_BUCKET_MAP, TAKE_MAP

            size_bucket = SIZE_BUCKET_MAP[int(action_arr[0]) % 4]
            take = TAKE_MAP[int(action_arr[1]) % 2]
            exec_style = EXEC_MAP[int(action_arr[2]) % 3]
        else:
            # Heuristic fallback when no trained weights are available.
            sig = ctx.signal_strength
            edge = ctx.expected_edge_frac
            if sig > 0.5 and edge > 0.001:
                take = True
                size_bucket = 0.5 if sig > 0.7 else 0.25
            else:
                take = False
                size_bucket = 0.0
            exec_style = "maker"

        return BoundedAction(
            strategy_weights={},
            size_bucket=size_bucket,
            take=take,
            exec_style=exec_style,
            param_nudges={},
            learner_id=self.learner_id,
            learner_version=self.learner_version,
            mode="SHADOW",
            rationale=(
                f"rl_policy decision #{self._n_decisions}; "
                f"signal={ctx.signal_strength:.3f}; edge={ctx.expected_edge_frac:.4f}"
            ),
        )

    def update(self, ctx: Context, action: BoundedAction, outcome: Outcome) -> None:
        """No-op in SHADOW mode. The RL policy trains offline only."""

    def snapshot(self) -> bytes:
        return pickle.dumps(
            {
                "learner_id": self.learner_id,
                "learner_version": self.learner_version,
                "n_decisions": self._n_decisions,
                "weights": self.weights.tolist() if self.weights is not None else None,
            }
        )

    def load(self, blob: bytes) -> None:
        state = pickle.loads(blob)  # noqa: S301
        self.learner_id = state["learner_id"]
        self.learner_version = state["learner_version"]
        self._n_decisions = state.get("n_decisions", 0)
        raw_w = state.get("weights")
        self.weights = np.array(raw_w) if raw_w is not None else None

    # ------------------------------------------------------------------ #
    # Constructors                                                         #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_trained(cls, weights: np.ndarray, learner_version: str = LEARNER_VERSION) -> RLPolicy:
        """Create a policy from externally trained weights."""
        return cls(learner_version=learner_version, weights=weights)

    @classmethod
    def build_default(cls, n_generations: int = 5, episode_length: int = 64) -> RLPolicy:
        """Build and train a default policy with a short simulation run.

        Used by gate checks to verify the training loop works end-to-end.
        A short training run (5 generations, 64-step episodes) is sufficient
        for verification; production training uses longer runs.
        """
        from src.rl.trainer import LinearRLTrainer, TrainingConfig

        trainer = LinearRLTrainer(
            config=TrainingConfig(
                n_generations=n_generations,
                population_size=10,
                episode_length=episode_length,
                rng_seed=42,
            )
        )
        result = trainer.train()
        return cls(learner_version=LEARNER_VERSION, weights=result.weights)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _ctx_to_obs(self, ctx: Context) -> np.ndarray:
        """Convert a :class:`~src.adaptation.policy_base.Context` to a numpy obs."""
        return np.array(
            [
                float(np.clip(ctx.signal_strength, -1.0, 1.0)),
                float(np.clip(ctx.expected_edge_frac, -1.0, 1.0)),
                float(np.clip(ctx.spread_bps, 0.0, 50.0)),
                float(np.clip(ctx.slippage_est, 0.0, 0.05)),
                float(np.clip(ctx.atr_pct, 0.0, 0.20)),
                float(np.clip(ctx.funding_z, -10.0, 10.0)),
            ],
            dtype=np.float32,
        )

    def _predict(self, obs: np.ndarray) -> np.ndarray:
        """Predict action from obs using the trained linear policy."""
        assert self.weights is not None
        logits = obs @ self.weights  # (n_actions,)
        flat_idx = int(np.argmax(logits))
        size_idx = flat_idx // 6
        remainder = flat_idx % 6
        take_idx = remainder // 3
        exec_idx = remainder % 3
        return np.array([size_idx, take_idx, exec_idx], dtype=np.int64)


# Backward-compatible alias — Phase 11 tests still reference RLPolicyStub.
# The stub is replaced by the full RLPolicy; the alias ensures the gate checks
# that import RLPolicyStub continue to work without modification.
@dataclass
class RLPolicyStub(RLPolicy):
    """Backward-compatible alias for the Phase 11 stub import.

    Delegates all behaviour to :class:`RLPolicy`. The stub always uses the
    heuristic fallback (no trained weights) so Phase 11 gate checks remain valid.
    """

    learner_id: str = "rl_policy_stub_v1"
    learner_version: str = "learner_0001"
    weights: np.ndarray | None = field(default=None, repr=False)
