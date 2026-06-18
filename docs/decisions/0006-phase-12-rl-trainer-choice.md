# Decision: Phase 12 — RL Trainer: Cross-Entropy Method instead of SB3

**Date:** 2026-06-18  
**Phase:** 12 — RL Research and Shadow Policy  
**Status:** Accepted

## Context

AGENTS.md Appendix C recommends "gymnasium + Stable-Baselines3 for prototyping" for the RL layer.
SB3 requires PyTorch (~1 GB installed) as a hard dependency.

## Decision

Use a custom **cross-entropy method (CEM)** trainer implemented with numpy instead of SB3.

## Rationale

1. **Dependency weight:** SB3 requires PyTorch (~1 GB). Adding it as a runtime dependency
   makes the base image significantly heavier with no benefit to the production paper/live
   trading path (RL is research/shadow only).

2. **Functional equivalence for gates:** The gate criteria check that:
   - The TradingEnv runs end-to-end (met by gymnasium alone)
   - The reward function is finite/bounded (met by reward.py)
   - Simulation training completes (met by LinearRLTrainer/CEM)
   - The RL policy produces valid BoundedActions in SHADOW mode (met by RLPolicy)
   
   CEM achieves all of these without PyTorch.

3. **"For prototyping" in spec:** AGENTS.md says SB3 is "for prototyping" — it is a
   recommendation, not a hard requirement. CEM is a valid RL training algorithm.

4. **Forward path:** If production research requires SB3 (e.g., PPO for continuous
   state spaces), it can be added as a dev/research dependency and plugged into
   `LinearRLTrainer._rollout()` without changing the interface.

## Consequences

- gymnasium added as main dependency (lightweight, ~4 MB).
- SB3/PyTorch NOT added — avoids 1+ GB of GPU-oriented dependencies.
- LinearRLTrainer (CEM) is the production RL trainer for Phase 12 gates.
- Future phases can introduce SB3 as a dev-only dep if deeper research warrants it.
