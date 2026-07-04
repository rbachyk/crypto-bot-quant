"""Simulation training for the RL shadow policy (AGENTS.md §21.4, Phase 12).

Delivers:
  - :class:`LinearRLTrainer` — lightweight linear policy trained via the cross-entropy
    method (CEM) on the gymnasium TradingEnv. No PyTorch required; runs on CPU.
  - :func:`stress_test` — runs the environment under all stress modes and verifies
    rewards are finite and bounded.
  - :class:`TrainingResult` — structured training result for gate checks.

Training algorithm (Cross-Entropy Method):
  1. Initialise population of random linear weight matrices W (obs_dim → n_actions).
  2. For each generation:
     a. Roll out each individual in the environment; collect episode return.
     b. Select the elite fraction (top-k by return).
     c. Update mean/std of W distribution from elite individuals.
  3. After convergence, the mean W is the trained policy.

The trained policy is serialized to a numpy ``.npy`` snapshot blob by
:meth:`LinearRLTrainer.snapshot` (there is no separate registry module).

This is a prototyping-grade implementation (Appendix C) suitable for shadow
evaluation. A SB3/PyTorch-based trainer can replace it later once the shadow
gate passes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from src.rl.environment import OBS_DIM, EnvConfig, TradingEnv
from src.rl.reward import RewardConfig


@dataclass
class TrainingConfig:
    """Configuration for :class:`LinearRLTrainer`."""

    n_generations: int = 20  # CEM generations
    population_size: int = 40  # individuals per generation
    elite_frac: float = 0.25  # fraction kept as elite
    noise_std_init: float = 0.5  # initial weight noise std
    noise_std_min: float = 0.01  # minimum noise std (convergence)
    noise_decay: float = 0.9  # noise std decay per generation
    episode_length: int = 128  # steps per evaluation episode
    n_eval_episodes: int = 3  # episodes averaged per individual (variance reduction under CRN)
    rng_seed: int = 42


@dataclass
class TrainingResult:
    """Structured result from a training run."""

    n_generations: int
    population_size: int
    elite_frac: float
    final_mean_return: float
    final_std_return: float
    best_return: float
    converged: bool
    weights: np.ndarray  # shape (OBS_DIM, n_actions)
    generation_means: list[float] = field(default_factory=list)
    stress_results: dict[str, dict] = field(default_factory=dict)


class LinearRLTrainer:
    """Cross-Entropy Method trainer for a linear policy on TradingEnv.

    The linear policy maps observations to action logits::

        logits = obs @ W            # shape (n_actions,)
        action = argmax(logits)     # flattened; decoded to MultiDiscrete

    The weight matrix W has shape ``(OBS_DIM, n_actions)`` where
    ``n_actions = 4 * 2 * 3 = 24`` (all MultiDiscrete combinations).
    """

    N_ACTIONS = 24  # 4 × 2 × 3

    def __init__(
        self,
        config: TrainingConfig | None = None,
        env_config: EnvConfig | None = None,
        reward_config: RewardConfig | None = None,
    ) -> None:
        self.cfg = config or TrainingConfig()
        self.env_cfg = env_config or EnvConfig(episode_length=self.cfg.episode_length)
        self.reward_cfg = reward_config
        self._rng = np.random.default_rng(self.cfg.rng_seed)
        self.weights: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    # Training                                                             #
    # ------------------------------------------------------------------ #

    def train(self) -> TrainingResult:
        """Run the CEM training loop and return the result."""
        cfg = self.cfg
        n_elite = max(1, math.ceil(cfg.population_size * cfg.elite_frac))

        # Initialise population distribution.
        w_mean = self._rng.standard_normal((OBS_DIM, self.N_ACTIONS)) * 0.1
        noise_std = cfg.noise_std_init

        gen_means: list[float] = []
        best_return = float("-inf")

        for _gen in range(cfg.n_generations):
            # Sample population.
            population = [
                w_mean + self._rng.standard_normal((OBS_DIM, self.N_ACTIONS)) * noise_std
                for _ in range(cfg.population_size)
            ]

            # Common Random Numbers (M38): evaluate EVERY individual this generation on the SAME
            # set of episode seeds, and average across them. Ranking then reflects the WEIGHTS, not
            # which individual happened to draw an easier random episode — without CRN the argsort
            # is dominated by evaluation noise and CEM barely learns.
            eval_seeds = [int(self._rng.integers(0, 2**31)) for _ in range(cfg.n_eval_episodes)]
            returns = [self._evaluate(W, eval_seeds) for W in population]
            returns_arr = np.array(returns, dtype=np.float64)

            # Rank and select elite.
            elite_idx = np.argsort(returns_arr)[-n_elite:]
            elite_W = [population[i] for i in elite_idx]
            elite_returns = returns_arr[elite_idx]

            # Update distribution.
            w_mean = np.mean(elite_W, axis=0)
            noise_std = max(cfg.noise_std_min, noise_std * cfg.noise_decay)

            gen_mean = float(elite_returns.mean())
            gen_means.append(gen_mean)

            best_return = max(best_return, float(elite_returns.max()))

        # The trained policy is the final elite-distribution MEAN — the actual CEM estimator — not
        # the single luckiest sample ever drawn, whose return was one noisy episode and does not
        # generalize (M38). best_return is retained only for reporting.
        self.weights = w_mean
        final_seeds = [int(self._rng.integers(0, 2**31)) for _ in range(10)]
        final_returns = np.array([self._rollout(w_mean, seed=s) for s in final_seeds])
        converged = float(final_returns.std()) < 1.0

        return TrainingResult(
            n_generations=cfg.n_generations,
            population_size=cfg.population_size,
            elite_frac=cfg.elite_frac,
            final_mean_return=float(final_returns.mean()),
            final_std_return=float(final_returns.std()),
            best_return=best_return,
            converged=converged,
            weights=w_mean,
            generation_means=gen_means,
        )

    def _evaluate(self, W: np.ndarray, seeds: list[int]) -> float:
        """Mean total reward of policy ``W`` over the given episode ``seeds`` (CRN average)."""
        return float(np.mean([self._rollout(W, seed=s) for s in seeds]))

    # ------------------------------------------------------------------ #
    # Stress tests                                                         #
    # ------------------------------------------------------------------ #

    def run_stress_tests(self) -> dict[str, dict]:
        """Run the environment under all stress modes; assert finite bounded rewards."""
        modes = ["normal", "no_edge", "high_vol", "toxic"]
        results = {}
        for mode in modes:
            cfg = EnvConfig(
                episode_length=100,
                rng_seed=99,
                stress_mode=mode,
            )
            env = TradingEnv(config=cfg, reward_config=self.reward_cfg)
            obs, _ = env.reset(seed=99)
            total_reward = 0.0
            rewards = []
            done = False
            step = 0
            while not done and step < 100:
                action = env.action_space.sample()
                obs, r, terminated, truncated, info = env.step(action)
                rewards.append(r)
                total_reward += r
                done = terminated or truncated
                step += 1

            rewards_arr = np.array(rewards, dtype=np.float64)
            results[mode] = {
                "steps": step,
                "total_reward": total_reward,
                "mean_reward": float(rewards_arr.mean()),
                "min_reward": float(rewards_arr.min()),
                "max_reward": float(rewards_arr.max()),
                "all_finite": bool(np.isfinite(rewards_arr).all()),
                "all_bounded": bool((np.abs(rewards_arr) <= 10.0).all()),
            }
        return results

    # ------------------------------------------------------------------ #
    # Prediction                                                           #
    # ------------------------------------------------------------------ #

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Predict action from observation using the trained linear policy.

        Returns a MultiDiscrete action array of shape (3,).
        Raises RuntimeError if the trainer has not been trained yet.
        """
        if self.weights is None:
            raise RuntimeError("trainer.train() must be called before predict()")
        logits = obs @ self.weights  # shape (24,)
        flat_idx = int(np.argmax(logits))
        # Decode: flat_idx = size_idx * 6 + take_idx * 3 + exec_idx
        size_idx = flat_idx // 6
        remainder = flat_idx % 6
        take_idx = remainder // 3
        exec_idx = remainder % 3
        return np.array([size_idx, take_idx, exec_idx], dtype=np.int64)

    def snapshot(self) -> bytes:
        """Serialize trained weights to bytes."""
        import io

        buf = io.BytesIO()
        weights = self.weights if self.weights is not None else np.zeros((OBS_DIM, self.N_ACTIONS))
        np.save(buf, weights)
        return buf.getvalue()

    def load(self, blob: bytes) -> None:
        """Restore weights from a snapshot blob."""
        import io

        buf = io.BytesIO(blob)
        self.weights = np.load(buf)

    # ------------------------------------------------------------------ #
    # Internal                                                            #
    # ------------------------------------------------------------------ #

    def _rollout(self, W: np.ndarray, *, seed: int | None = None) -> float:
        """Roll out one episode with weight matrix W; return total reward.

        ``seed`` fixes the episode so the same policy scores identically and different policies
        are compared on the SAME episode (common random numbers — M38). ``None`` draws a fresh
        seed from the trainer RNG."""
        ep_seed = int(self._rng.integers(0, 2**31)) if seed is None else int(seed)
        env = TradingEnv(
            config=EnvConfig(
                episode_length=self.cfg.episode_length,
                rng_seed=ep_seed,
            ),
            reward_config=self.reward_cfg,
        )
        obs, _ = env.reset(seed=ep_seed)
        total = 0.0
        done = False
        while not done:
            logits = obs @ W
            flat_idx = int(np.argmax(logits))
            size_idx = flat_idx // 6
            remainder = flat_idx % 6
            take_idx = remainder // 3
            exec_idx = remainder % 3
            action = np.array([size_idx, take_idx, exec_idx], dtype=np.int64)
            obs, r, terminated, truncated, _ = env.step(action)
            total += r
            done = terminated or truncated
        return total
