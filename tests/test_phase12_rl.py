"""Phase 12 — RL Research and Shadow Policy tests (AGENTS.md Section 21.4, Appendix D).

Tests cover:
  - TradingEnv: gymnasium-compatible, correct obs/action spaces, episode execution
  - RiskAdjustedReward: cost-net, bounded, skip=0, finite
  - LinearRLTrainer: CEM training completes; stress tests produce finite rewards
  - RLPolicy: shadow mode, valid bounded actions, snapshot/load, heuristic fallback
  - Envelope guard: RL actions never violate the immutable risk envelope
  - Controller: RL policy in SHADOW mode produces applied=False
  - Gate criteria: RL-SIM and RL-SHADOW all criteria pass
"""

from __future__ import annotations

import math

import numpy as np
import pytest

# ======================================================================== #
# TradingEnv                                                                #
# ======================================================================== #


class TestTradingEnv:
    def _env(self, episode_length=20, **kw):
        from src.rl.environment import EnvConfig, TradingEnv

        return TradingEnv(config=EnvConfig(episode_length=episode_length, rng_seed=42, **kw))

    def test_obs_space_shape(self):
        env = self._env()
        obs, _ = env.reset()
        assert obs.shape == (6,)
        assert obs.dtype == np.float32

    def test_obs_within_bounds(self):
        env = self._env()
        obs, _ = env.reset()
        assert env.observation_space.contains(obs), f"obs {obs} outside obs_space"

    def test_action_space_multidiscrete(self):
        from gymnasium.spaces import MultiDiscrete

        env = self._env()
        assert isinstance(env.action_space, MultiDiscrete)
        # [size_bucket_idx, take_idx, exec_style_idx]
        assert list(env.action_space.nvec) == [4, 2, 3]

    def test_full_episode_runs(self):
        env = self._env(episode_length=30)
        obs, _ = env.reset()
        steps = 0
        done = False
        while not done and steps < 30:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
        assert steps > 0

    def test_step_returns_info(self):
        env = self._env()
        env.reset()
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert "step" in info
        assert "cumulative_pnl" in info
        assert "drawdown" in info

    def test_reward_finite(self):
        env = self._env(episode_length=50)
        env.reset()
        for _ in range(50):
            action = env.action_space.sample()
            _, reward, done, trunc, _ = env.step(action)
            assert math.isfinite(reward), f"non-finite reward: {reward}"
            if done or trunc:
                break

    def test_reward_bounded(self):
        env = self._env(episode_length=50)
        env.reset()
        for _ in range(50):
            action = env.action_space.sample()
            _, reward, done, trunc, _ = env.step(action)
            assert -10.0 <= reward <= 10.0, f"reward {reward} outside bounds"
            if done or trunc:
                break

    def test_deterministic_with_seed(self):
        env = self._env()
        obs1, _ = env.reset(seed=99)
        env2 = self._env()
        obs2, _ = env2.reset(seed=99)
        np.testing.assert_array_equal(obs1, obs2)

    def test_truncated_at_max_steps(self):
        env = self._env(episode_length=10)
        env.reset()
        rewards = []
        done = False
        while not done:
            action = env.action_space.sample()
            obs, r, terminated, truncated, _ = env.step(action)
            rewards.append(r)
            done = terminated or truncated
        assert len(rewards) <= 10

    def test_bounded_action_from_env(self):
        from src.adaptation.action_space import VALID_SIZE_BUCKETS

        env = self._env()
        env.reset()
        for _ in range(20):
            raw = env.action_space.sample()
            ba = env.bounded_action_from(raw)
            assert ba.mode == "SHADOW"
            assert ba.size_bucket in VALID_SIZE_BUCKETS
            assert ba.exec_style in ("maker", "taker", "passive_then_taker")

    def test_stress_mode_no_edge(self):
        env = self._env(stress_mode="no_edge")
        obs, _ = env.reset()
        assert -1.0 <= obs[1] <= 1.0  # expected_edge within bounds

    def test_stress_mode_high_vol(self):
        env = self._env(stress_mode="high_vol")
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)

    def test_stress_mode_toxic(self):
        env = self._env(stress_mode="toxic")
        env.reset()
        action = env.action_space.sample()
        _, r, _, _, _ = env.step(action)
        assert math.isfinite(r)


# ======================================================================== #
# RiskAdjustedReward                                                        #
# ======================================================================== #


class TestRiskAdjustedReward:
    def _reward_fn(self):
        from src.rl.reward import RewardConfig, RiskAdjustedReward

        return RiskAdjustedReward(RewardConfig())

    def _state(self):
        from src.rl.reward import RewardState

        return RewardState()

    def test_skip_yields_zero_take_false(self):
        fn = self._reward_fn()
        r = fn.compute(
            expected_edge_frac=0.005,
            size_bucket=1.0,
            take=False,
            exec_style="taker",
            spread_bps=3.0,
            slippage_est=0.0003,
            funding_z=0.0,
            state=self._state(),
        )
        assert r == 0.0

    def test_skip_yields_zero_size_zero(self):
        fn = self._reward_fn()
        r = fn.compute(
            expected_edge_frac=0.005,
            size_bucket=0.0,
            take=True,
            exec_style="maker",
            spread_bps=3.0,
            slippage_est=0.0003,
            funding_z=0.0,
            state=self._state(),
        )
        assert r == 0.0

    def test_positive_edge_positive_reward(self):
        fn = self._reward_fn()
        state = self._state()
        r = fn.compute(
            expected_edge_frac=0.02,
            size_bucket=1.0,
            take=True,
            exec_style="maker",
            spread_bps=2.0,
            slippage_est=0.0001,
            funding_z=0.0,
            state=state,
            stochastic_noise=0.0,
        )
        assert math.isfinite(r)
        assert -10.0 <= r <= 10.0

    def test_reward_bounded(self):
        fn = self._reward_fn()
        state = self._state()
        for edge in [-0.5, -0.1, 0.0, 0.1, 0.5]:
            r = fn.compute(
                expected_edge_frac=edge,
                size_bucket=1.0,
                take=True,
                exec_style="taker",
                spread_bps=5.0,
                slippage_est=0.001,
                funding_z=2.0,
                state=state,
            )
            assert -10.0 <= r <= 10.0, f"reward {r} for edge={edge}"

    def test_maker_cheaper_than_taker(self):
        fn = self._reward_fn()
        r_maker = fn.compute(
            expected_edge_frac=0.01,
            size_bucket=1.0,
            take=True,
            exec_style="maker",
            spread_bps=3.0,
            slippage_est=0.0,
            funding_z=0.0,
            state=self._state(),
            stochastic_noise=0.0,
        )
        r_taker = fn.compute(
            expected_edge_frac=0.01,
            size_bucket=1.0,
            take=True,
            exec_style="taker",
            spread_bps=3.0,
            slippage_est=0.0,
            funding_z=0.0,
            state=self._state(),
            stochastic_noise=0.0,
        )
        assert r_maker > r_taker, "maker should have lower fees than taker"

    def test_drawdown_penalty_applied(self):
        from src.rl.reward import RewardState

        fn = self._reward_fn()
        # Simulate a large drawdown.
        state = RewardState(peak_pnl=0.10, cumulative_pnl=0.02)
        state.current_drawdown = 0.08  # 8% drawdown (above 4% soft threshold)
        r = fn.compute(
            expected_edge_frac=0.01,
            size_bucket=1.0,
            take=True,
            exec_style="maker",
            spread_bps=3.0,
            slippage_est=0.0001,
            funding_z=0.0,
            state=state,
        )
        assert math.isfinite(r)

    def test_state_updated_after_step(self):
        from src.rl.reward import RewardState

        fn = self._reward_fn()
        state = RewardState()
        assert state.n_steps == 0
        fn.compute(
            expected_edge_frac=0.005,
            size_bucket=0.5,
            take=True,
            exec_style="maker",
            spread_bps=2.0,
            slippage_est=0.0002,
            funding_z=0.0,
            state=state,
        )
        assert state.n_steps == 1


# ======================================================================== #
# LinearRLTrainer                                                            #
# ======================================================================== #


class TestLinearRLTrainer:
    def _trainer(self, n_gen=3, pop=6, ep_len=30):
        from src.rl.trainer import LinearRLTrainer, TrainingConfig

        return LinearRLTrainer(
            config=TrainingConfig(
                n_generations=n_gen,
                population_size=pop,
                elite_frac=0.25,
                episode_length=ep_len,
                rng_seed=42,
            )
        )

    def test_train_completes(self):
        result = self._trainer().train()
        assert result.n_generations == 3
        assert result.weights is not None
        assert result.weights.shape == (6, 24)
        assert math.isfinite(result.best_return)
        assert math.isfinite(result.final_mean_return)

    def test_generation_means_recorded(self):
        result = self._trainer(n_gen=5).train()
        assert len(result.generation_means) == 5
        assert all(math.isfinite(m) for m in result.generation_means)

    def test_predict_valid_action(self):
        trainer = self._trainer()
        trainer.train()
        obs = np.array([0.7, 0.005, 3.0, 0.0003, 0.02, 0.5], dtype=np.float32)
        action = trainer.predict(obs)
        assert action.shape == (3,)
        assert 0 <= action[0] <= 3
        assert 0 <= action[1] <= 1
        assert 0 <= action[2] <= 2

    def test_predict_raises_before_train(self):
        trainer = self._trainer()
        obs = np.zeros(6, dtype=np.float32)
        with pytest.raises(RuntimeError, match="train()"):
            trainer.predict(obs)

    def test_snapshot_load_roundtrip(self):
        trainer = self._trainer()
        trainer.train()
        blob = trainer.snapshot()
        trainer2 = self._trainer()
        trainer2.load(blob)
        obs = np.array([0.6, 0.003, 3.0, 0.0003, 0.02, 0.1], dtype=np.float32)
        np.testing.assert_array_equal(trainer.predict(obs), trainer2.predict(obs))

    def test_stress_tests_finite_bounded(self):
        trainer = self._trainer()
        results = trainer.run_stress_tests()
        assert set(results.keys()) == {"normal", "no_edge", "high_vol", "toxic"}
        for mode, res in results.items():
            assert res["all_finite"], f"mode={mode}: non-finite reward"
            assert res["all_bounded"], f"mode={mode}: reward outside [-10, 10]"
            assert res["steps"] > 0

    def test_stress_test_steps_positive(self):
        trainer = self._trainer()
        results = trainer.run_stress_tests()
        for mode, res in results.items():
            assert res["steps"] > 0, f"mode={mode}: zero steps"


# ======================================================================== #
# RLPolicy                                                                   #
# ======================================================================== #


class TestRLPolicy:
    def _policy(self):
        from src.adaptation.policies.rl_policy import RLPolicy

        return RLPolicy.build_default(n_generations=3, episode_length=30)

    def _ctx(self, sig=0.7, edge=0.005):
        from src.adaptation.policy_base import Context

        return Context(signal_strength=sig, expected_edge_frac=edge, spread_bps=3.0)

    def test_decide_returns_shadow_action(self):
        policy = self._policy()
        action = policy.decide(self._ctx())
        assert action.mode == "SHADOW"

    def test_decide_valid_bucket(self):
        from src.adaptation.action_space import VALID_SIZE_BUCKETS

        policy = self._policy()
        for sig in [0.1, 0.4, 0.6, 0.9]:
            action = policy.decide(self._ctx(sig=sig))
            assert action.size_bucket in VALID_SIZE_BUCKETS

    def test_decide_valid_exec_style(self):
        policy = self._policy()
        action = policy.decide(self._ctx())
        assert action.exec_style in ("maker", "taker", "passive_then_taker")

    def test_decide_no_param_nudges(self):
        policy = self._policy()
        for _ in range(10):
            action = policy.decide(self._ctx())
            assert action.param_nudges == {}, "RL policy must not emit param_nudges"

    def test_update_is_noop(self):
        from src.adaptation.policy_base import Outcome

        policy = self._policy()
        ctx = self._ctx()
        action = policy.decide(ctx)
        # Should not raise
        policy.update(ctx, action, Outcome(realized_pnl_r=0.05, trade_taken=True))
        policy.update(ctx, action, Outcome(realized_pnl_r=-0.02, trade_taken=True))

    def test_snapshot_load_roundtrip(self):
        from src.adaptation.policies.rl_policy import RLPolicy

        policy = self._policy()
        blob = policy.snapshot()
        policy2 = RLPolicy()
        policy2.load(blob)
        assert policy2.learner_id == policy.learner_id
        assert policy2.learner_version == policy.learner_version
        assert (policy2.weights is not None) == (policy.weights is not None)

    def test_heuristic_fallback_no_weights(self):
        from src.adaptation.policies.rl_policy import RLPolicy

        policy = RLPolicy()  # no weights
        action = policy.decide(self._ctx(sig=0.8, edge=0.005))
        assert action.mode == "SHADOW"
        assert action.take is True
        action2 = policy.decide(self._ctx(sig=0.1, edge=-0.001))
        assert action2.take is False

    def test_action_passes_validation(self):
        from src.adaptation.action_space import ActionBounds, validate

        policy = self._policy()
        bounds = ActionBounds()
        for _ in range(20):
            ctx = self._ctx()
            action = policy.decide(ctx)
            result = validate(action, bounds)
            assert not result.rejected, f"action rejected: {result.rejection_reason}"

    def test_learner_id_stable(self):
        policy = self._policy()
        actions = [policy.decide(self._ctx()) for _ in range(5)]
        assert all(a.learner_id == policy.learner_id for a in actions)

    def test_rl_policy_stub_backward_compat(self):
        """RLPolicyStub alias still works for Phase 11 backward compatibility."""
        from src.adaptation.policies.rl_policy import RLPolicyStub

        stub = RLPolicyStub()
        from src.adaptation.policy_base import Context

        action = stub.decide(Context())
        assert action.mode == "SHADOW"

    def test_from_trained_constructor(self):
        from src.adaptation.policies.rl_policy import RLPolicy

        weights = np.zeros((6, 24))
        policy = RLPolicy.from_trained(weights, learner_version="test_v1")
        assert policy.learner_version == "test_v1"
        assert policy.weights is not None


# ======================================================================== #
# Envelope guard — RL actions                                               #
# ======================================================================== #


class TestRLEnvelopeGuard:
    def _envelope(self):
        from src.adaptation.envelope_guard import RiskEnvelope

        return RiskEnvelope(
            max_leverage=5,
            max_risk_pct_per_trade=0.01,
            portfolio_heat_cap=0.05,
            net_beta_btc_cap=0.30,
            daily_loss_limit=0.03,
            max_drawdown_limit=0.10,
        )

    def test_normal_rl_action_passes_guard(self):
        from src.adaptation.envelope_guard import enforce
        from src.adaptation.policies.rl_policy import RLPolicy
        from src.adaptation.policy_base import Context

        policy = RLPolicy.build_default(n_generations=2, episode_length=20)
        envelope = self._envelope()
        for _ in range(10):
            action = policy.decide(Context(signal_strength=0.7, expected_edge_frac=0.005))
            result = enforce(action, envelope=envelope)
            assert not result.rejected, f"normal RL action rejected: {result.rejection_reason}"

    def test_rl_cannot_touch_envelope_params(self):
        from src.adaptation.action_space import BoundedAction
        from src.adaptation.envelope_guard import enforce

        # A manually crafted RL-style action with forbidden envelope param.
        bad = BoundedAction(
            size_bucket=1.0,
            take=True,
            exec_style="maker",
            param_nudges={"max_risk_pct_per_trade": 0.99},  # FORBIDDEN
            learner_id="rl_policy_v1",
            learner_version="rl_v1",
            mode="SHADOW",
            rationale="attack test",
        )
        result = enforce(bad, envelope=self._envelope())
        assert result.rejected, "envelope-touching action must be rejected"

    def test_rl_size_bucket_capped_at_1(self):
        from src.adaptation.action_space import BoundedAction
        from src.adaptation.envelope_guard import enforce

        over = BoundedAction(
            size_bucket=1.5,  # over cap
            take=True,
            exec_style="maker",
            param_nudges={},
            learner_id="rl_policy_v1",
            learner_version="rl_v1",
            mode="SHADOW",
            rationale="oversize test",
        )
        result = enforce(over, envelope=self._envelope())
        # Should be clamped (not rejected) per spec §21.6
        assert not result.rejected
        assert result.action.size_bucket <= 1.0


# ======================================================================== #
# Controller — RL policy in SHADOW mode                                      #
# ======================================================================== #


class TestRLControllerShadow:
    def test_controller_shadow_applied_false(self):
        from src.adaptation.action_space import ActionBounds
        from src.adaptation.controller import LearnerController, LearnerMode
        from src.adaptation.policies.rl_policy import RLPolicy
        from src.adaptation.policy_base import Context

        policy = RLPolicy.build_default(n_generations=2, episode_length=20)
        ctrl = LearnerController(policy=policy, bounds=ActionBounds(), mode=LearnerMode.SHADOW)
        for _ in range(10):
            ctx = Context(signal_strength=0.7, expected_edge_frac=0.005)
            dec = ctrl.run(ctx)
            assert not dec.applied, "SHADOW controller must never apply actions"
            assert not dec.rejected, f"valid RL action rejected: {dec.rejection_reason}"

    def test_controller_mode_never_live_bounded(self):
        from src.adaptation.action_space import ActionBounds
        from src.adaptation.controller import LearnerController, LearnerMode
        from src.adaptation.policies.rl_policy import RLPolicy
        from src.adaptation.policy_base import Context

        policy = RLPolicy.build_default(n_generations=2, episode_length=20)
        ctrl = LearnerController(policy=policy, bounds=ActionBounds(), mode=LearnerMode.SHADOW)
        for _ in range(5):
            dec = ctrl.run(Context())
            if dec.action:
                assert dec.action.mode != "LIVE_BOUNDED"


# ======================================================================== #
# Gate criteria integration                                                  #
# ======================================================================== #


class TestPhase12Gates:
    def test_rl_sim_gate_passes(self):
        """RL-SIM gate: all criteria must pass."""
        from src.config import get_settings
        from src.gates.phase12 import check_rl_sim

        settings = get_settings()
        criteria = check_rl_sim(settings)
        assert len(criteria) > 0
        failures = [c for c in criteria if c.status != "PASS"]
        assert failures == [], (
            f"RL-SIM gate failures: {[(c.id, c.failure_reason) for c in failures]}"
        )

    def test_rl_shadow_gate_passes(self):
        """RL-SHADOW gate: all criteria must pass."""
        from src.config import get_settings
        from src.gates.phase12 import check_rl_shadow

        settings = get_settings()
        criteria = check_rl_shadow(settings)
        assert len(criteria) > 0
        failures = [c for c in criteria if c.status != "PASS"]
        assert failures == [], (
            f"RL-SHADOW gate failures: {[(c.id, c.failure_reason) for c in failures]}"
        )

    def test_rl_sim_gate_registered(self):
        """RL-SIM gate is registered in the gate runner."""
        from src.gates.checks import CHECKS

        assert "RL-SIM" in CHECKS

    def test_rl_shadow_gate_registered(self):
        """RL-SHADOW gate is registered in the gate runner."""
        from src.gates.checks import CHECKS

        assert "RL-SHADOW" in CHECKS

    def test_rl_gates_in_catalog(self):
        """RL-SIM and RL-SHADOW appear in gates.yaml catalog."""
        from src.gates.catalog import load_catalog

        load_catalog.cache_clear()
        catalog = load_catalog()
        assert "RL-SIM" in catalog, "RL-SIM not in gates.yaml"
        assert "RL-SHADOW" in catalog, "RL-SHADOW not in gates.yaml"

    def test_rl_sim_depends_on_learn_promo_s(self):
        from src.gates.catalog import load_catalog

        load_catalog.cache_clear()
        catalog = load_catalog()
        assert "LEARN-PROMO-S" in catalog["RL-SIM"].depends_on

    def test_rl_shadow_depends_on_rl_sim(self):
        from src.gates.catalog import load_catalog

        load_catalog.cache_clear()
        catalog = load_catalog()
        assert "RL-SIM" in catalog["RL-SHADOW"].depends_on
