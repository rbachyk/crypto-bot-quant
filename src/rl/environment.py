"""TradingEnv — gymnasium-compatible RL environment for shadow policy research.

(AGENTS.md Section 21.4, 32 Phase 12; Appendix C)

The environment simulates a synthetic signal stream and evaluates the RL policy's
action choices against a risk-adjusted, cost-net reward function. It is used for:
  - Simulation training (never connected to live data or exchange)
  - Stress tests (high volatility, no-edge, toxic execution scenarios)
  - Shadow policy evaluation

Observation space (Box, float32):
  [signal_strength, expected_edge_frac, spread_bps, slippage_est, atr_pct, funding_z]

Action space (MultiDiscrete):
  [size_bucket_idx, take_idx, exec_style_idx]
  - size_bucket_idx ∈ {0,1,2,3} → {0.0, 0.25, 0.5, 1.0}
  - take_idx ∈ {0, 1} → {False, True}
  - exec_style_idx ∈ {0,1,2} → {"maker","taker","passive_then_taker"}

The action space is bounded by construction — no invalid BoundedAction can be
emitted. All actions are validated by action_space.validate() before use.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, MultiDiscrete

from src.adaptation.action_space import (
    VALID_SIZE_BUCKETS,
    ActionBounds,
    BoundedAction,
    validate,
)
from src.rl.reward import RewardConfig, RewardState, RiskAdjustedReward

# Observation feature indices.
OBS_SIGNAL_STRENGTH = 0
OBS_EXPECTED_EDGE = 1
OBS_SPREAD_BPS = 2
OBS_SLIPPAGE_EST = 3
OBS_ATR_PCT = 4
OBS_FUNDING_Z = 5
OBS_DIM = 6

# Action index maps.
SIZE_BUCKET_MAP: list[float] = list(VALID_SIZE_BUCKETS)  # [0.0, 0.25, 0.5, 1.0]
TAKE_MAP: list[bool] = [False, True]
EXEC_MAP: list[str] = ["maker", "taker", "passive_then_taker"]


@dataclass
class EnvConfig:
    """Environment configuration (all tunable; envelope constants are never here)."""

    episode_length: int = 252  # synthetic trading days per episode
    rng_seed: int | None = 42  # reproducibility; None = random
    # Signal generation parameters.
    base_edge: float = 0.002  # 0.2% expected edge
    edge_noise_scale: float = 0.004  # noise on realized edge
    signal_strength_scale: float = 0.7
    spread_bps_mean: float = 3.0
    spread_bps_std: float = 1.5
    slippage_mean: float = 0.0003
    slippage_std: float = 0.0001
    atr_pct_mean: float = 0.02
    atr_pct_std: float = 0.008
    funding_z_scale: float = 1.0
    # Stress test mode: no-edge, high-vol, toxic.
    stress_mode: str = "normal"  # "normal" | "no_edge" | "high_vol" | "toxic"


class TradingEnv(Env):  # type: ignore[misc]
    """Gymnasium-compatible RL trading environment (research/shadow only).

    Every episode generates a synthetic sequence of trading signals. The agent
    decides whether to take each signal and at what size; the risk-adjusted,
    cost-net reward is computed by :class:`~src.rl.reward.RiskAdjustedReward`.

    This environment is never connected to a live exchange or real data pipeline.
    It is used solely for simulation training and stress tests.

    Usage::

        env = TradingEnv()
        obs, info = env.reset()
        for _ in range(env.cfg.episode_length):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: EnvConfig | None = None,
        reward_config: RewardConfig | None = None,
        action_bounds: ActionBounds | None = None,
    ) -> None:
        super().__init__()
        self.cfg = config or EnvConfig()
        self._reward_fn = RiskAdjustedReward(reward_config)
        self._bounds = action_bounds or ActionBounds()

        # Observation space: 6 continuous features.
        low = np.array([-1.0, -1.0, 0.0, 0.0, 0.0, -10.0], dtype=np.float32)
        high = np.array([1.0, 1.0, 50.0, 0.05, 0.20, 10.0], dtype=np.float32)
        self.observation_space = Box(low=low, high=high, dtype=np.float32)

        # Action space: [size_bucket_idx, take_idx, exec_style_idx]
        self.action_space = MultiDiscrete([4, 2, 3], seed=self.cfg.rng_seed)

        # Episode state.
        self._rng: random.Random | None = None
        self._np_rng: np.random.Generator | None = None
        self._step: int = 0
        self._reward_state: RewardState = RewardState()
        self._current_obs: np.ndarray = np.zeros(OBS_DIM, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # gymnasium interface                                                 #
    # ------------------------------------------------------------------ #

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        effective_seed = seed if seed is not None else self.cfg.rng_seed
        self._rng = random.Random(effective_seed)
        self._np_rng = np.random.default_rng(effective_seed)
        self._step = 0
        self._reward_state = RewardState()
        self._current_obs = self._generate_obs()
        return self._current_obs.copy(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one environment step.

        Parameters
        ----------
        action:
            Array of shape (3,) with values [size_bucket_idx, take_idx, exec_style_idx].

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        size_bucket_idx = int(action[0])
        take_idx = int(action[1])
        exec_style_idx = int(action[2])

        size_bucket = SIZE_BUCKET_MAP[size_bucket_idx]
        take = TAKE_MAP[take_idx]
        exec_style = EXEC_MAP[exec_style_idx]

        obs = self._current_obs
        expected_edge = float(obs[OBS_EXPECTED_EDGE])
        spread_bps = float(obs[OBS_SPREAD_BPS])
        slippage_est = float(obs[OBS_SLIPPAGE_EST])
        funding_z = float(obs[OBS_FUNDING_Z])

        # Stochastic noise for the realized PnL (drawn from fixed RNG).
        noise_scale = float(obs[OBS_ATR_PCT]) * 0.2
        rng = self._np_rng
        stochastic_noise = float(rng.normal(0.0, noise_scale)) if rng is not None else 0.0

        reward = self._reward_fn.compute(
            expected_edge_frac=expected_edge,
            size_bucket=size_bucket,
            take=take,
            exec_style=exec_style,
            spread_bps=spread_bps,
            slippage_est=slippage_est,
            funding_z=funding_z,
            state=self._reward_state,
            stochastic_noise=stochastic_noise,
        )

        self._step += 1
        truncated = self._step >= self.cfg.episode_length
        # Terminate early if drawdown exceeds 10% (envelope circuit breaker).
        terminated = self._reward_state.current_drawdown >= 0.10

        self._current_obs = self._generate_obs()
        info = {
            "step": self._step,
            "cumulative_pnl": self._reward_state.cumulative_pnl,
            "drawdown": self._reward_state.current_drawdown,
            "heat": self._reward_state.current_heat,
            "take": take,
            "size_bucket": size_bucket,
        }
        return self._current_obs.copy(), float(reward), terminated, truncated, info

    def bounded_action_from(
        self, action: np.ndarray, learner_id: str = "rl_policy"
    ) -> BoundedAction:
        """Convert a numpy action array to a validated :class:`BoundedAction`.

        Always validates through :func:`~src.adaptation.action_space.validate`
        before returning. The resulting action must pass ``envelope_guard.enforce``
        before any downstream use.
        """
        size_bucket = SIZE_BUCKET_MAP[int(action[0]) % 4]
        take = TAKE_MAP[int(action[1]) % 2]
        exec_style = EXEC_MAP[int(action[2]) % 3]

        raw = BoundedAction(
            strategy_weights={},
            size_bucket=size_bucket,
            take=take,
            exec_style=exec_style,
            param_nudges={},
            learner_id=learner_id,
            learner_version="rl_v1",
            mode="SHADOW",
            rationale=f"rl env action {action.tolist()}",
        )
        result = validate(raw, self._bounds)
        if result.rejected:
            # Safety fallback: emit a neutral skip action.
            return BoundedAction(
                size_bucket=0.0,
                take=False,
                exec_style="maker",
                learner_id=learner_id,
                learner_version="rl_v1",
                mode="SHADOW",
                rationale="action rejected by validator; fallback to skip",
            )
        return result.action

    # ------------------------------------------------------------------ #
    # Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _generate_obs(self) -> np.ndarray:
        """Generate a synthetic observation from the configured distribution."""
        rng = self._np_rng
        if rng is None:
            rng = np.random.default_rng()

        cfg = self.cfg

        if cfg.stress_mode == "no_edge":
            expected_edge = float(rng.normal(0.0, 0.001))
            spread_bps = float(rng.normal(cfg.spread_bps_mean * 2, cfg.spread_bps_std))
        elif cfg.stress_mode == "high_vol":
            expected_edge = float(rng.normal(cfg.base_edge, cfg.edge_noise_scale * 3))
            spread_bps = float(rng.normal(cfg.spread_bps_mean * 3, cfg.spread_bps_std * 3))
        elif cfg.stress_mode == "toxic":
            expected_edge = float(rng.normal(-cfg.base_edge, cfg.edge_noise_scale))
            spread_bps = float(rng.normal(20.0, 5.0))
        else:  # normal
            expected_edge = float(rng.normal(cfg.base_edge, cfg.edge_noise_scale))
            spread_bps = float(rng.normal(cfg.spread_bps_mean, cfg.spread_bps_std))

        signal_strength = float(np.clip(rng.normal(cfg.signal_strength_scale, 0.2), 0.0, 1.0))
        slippage_est = float(np.clip(rng.normal(cfg.slippage_mean, cfg.slippage_std), 0.0, 0.05))
        atr_pct = float(np.clip(rng.normal(cfg.atr_pct_mean, cfg.atr_pct_std), 0.0, 0.20))
        funding_z = float(rng.normal(0.0, cfg.funding_z_scale))

        obs = np.array(
            [
                np.clip(signal_strength, -1.0, 1.0),
                np.clip(expected_edge, -1.0, 1.0),
                np.clip(spread_bps, 0.0, 50.0),
                np.clip(slippage_est, 0.0, 0.05),
                np.clip(atr_pct, 0.0, 0.20),
                np.clip(funding_z, -10.0, 10.0),
            ],
            dtype=np.float32,
        )
        return obs
