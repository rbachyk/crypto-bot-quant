"""RL Research and Shadow Policy module — Phase 12 (AGENTS.md Section 21.4, 32).

Phase 12 delivers the RL research scaffold (research/shadow only):

  * ``src/rl/environment.py``  — TradingEnv (gymnasium.Env); risk-adj cost-net reward
  * ``src/rl/reward.py``       — RiskAdjustedReward calculation
  * ``src/rl/trainer.py``      — simulation training + stress tests; LinearRLTrainer
  * ``src/rl/registry.py``     — versioned RL policy artifact store

The RL policy is ALWAYS shadow-only until the RL-SIM + RL-SHADOW gates pass AND
manual promotion is approved (Section 21.3). See src/adaptation/policies/rl_policy.py
for the shadow-mode integration with the learner controller.
"""

from src.rl.environment import TradingEnv
from src.rl.reward import RewardConfig, RiskAdjustedReward
from src.rl.trainer import LinearRLTrainer, TrainingResult

__all__ = [
    "TradingEnv",
    "RiskAdjustedReward",
    "RewardConfig",
    "LinearRLTrainer",
    "TrainingResult",
]
