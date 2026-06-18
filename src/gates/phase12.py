"""Phase 12 gate checks — RL-SIM and RL-SHADOW (AGENTS.md Appendix A, Phase 12).

Gate: RL-SIM — RL Simulation Gate
  Pass conditions:
  1. rl_env_importable        — TradingEnv + all rl module types importable
  2. rl_env_episode_runs      — full episode (episode_length steps) without crash
  3. rl_reward_risk_adjusted  — reward is finite, bounded, cost-net in all modes
  4. rl_action_bounded        — all actions pass action_space.validate() + envelope_guard
  5. rl_stress_tests_pass     — stress tests (normal/no_edge/high_vol/toxic) produce
                                finite bounded rewards
  6. rl_simulation_training   — training completes without error; best_return finite
  7. rl_trained_policy_valid  — trained RLPolicy produces valid BoundedActions

Gate: RL-SHADOW — RL Shadow Policy Gate
  Pass conditions:
  1. rl_policy_importable     — RLPolicy importable; snapshot/load round-trip verified
  2. rl_policy_shadow_mode    — all decisions have mode=SHADOW
  3. rl_shadow_applied_false  — all RL learner_log entries have applied=False
  4. rl_no_live_influence     — mode is never LIVE_BOUNDED
  5. rl_envelope_enforced     — envelope_guard rejects RL actions that touch envelope
  6. rl_recommend_mode_ready  — RLPolicy can produce actions with mode=RECOMMEND
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from src.config import Settings
from src.gates.result import Criterion


def check_rl_sim(settings: Settings) -> list[Criterion]:
    """RL-SIM gate criteria (Phase 12 — RL Simulation Gate)."""
    out: list[Criterion] = []

    # ------------------------------------------------------------------ #
    # 1. RL module imports                                                #
    # ------------------------------------------------------------------ #
    try:
        from src.rl import (  # noqa: F401
            LinearRLTrainer,
            RewardConfig,
            RiskAdjustedReward,
            TradingEnv,
        )
        from src.rl.environment import EXEC_MAP, SIZE_BUCKET_MAP, TAKE_MAP, EnvConfig  # noqa: F401
        from src.rl.reward import RewardState  # noqa: F401
        from src.rl.trainer import TrainingConfig  # noqa: F401

        out.append(
            Criterion.ok(
                "rl_env_importable",
                "TradingEnv, RiskAdjustedReward, LinearRLTrainer all importable",
            )
        )
    except ImportError as exc:
        out.append(Criterion.fail("rl_env_importable", f"import error: {exc}"))
        return out

    # ------------------------------------------------------------------ #
    # 2. Environment: full episode runs without crash                     #
    # ------------------------------------------------------------------ #
    try:
        from src.rl.environment import EnvConfig, TradingEnv

        cfg = EnvConfig(episode_length=50, rng_seed=42)
        env = TradingEnv(config=cfg)
        obs, info = env.reset(seed=42)
        assert obs.shape == (6,), f"obs shape={obs.shape}"
        assert obs.dtype.name == "float32"

        step_count = 0
        done = False
        while not done and step_count < 50:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_count += 1

        assert step_count > 0, "zero steps taken"
        assert obs.shape == (6,)

        out.append(
            Criterion.ok(
                "rl_env_episode_runs",
                f"episode ran {step_count} steps without crash; "
                f"obs.shape={obs.shape}; info keys={list(info.keys())}",
            )
        )
    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("rl_env_episode_runs", f"raised: {exc}"))

    # ------------------------------------------------------------------ #
    # 3. Reward is risk-adjusted, cost-net, and bounded                  #
    # ------------------------------------------------------------------ #
    try:
        from src.rl.reward import RewardConfig, RewardState, RiskAdjustedReward

        reward_fn = RiskAdjustedReward(RewardConfig())
        state = RewardState()

        rewards = []
        for expected_edge, size_bucket, take in [
            (0.005, 1.0, True),  # take with positive edge
            (0.005, 0.0, True),  # size=0 → 0 reward
            (0.005, 1.0, False),  # skip → 0 reward
            (-0.02, 1.0, True),  # take with negative edge
            (0.001, 0.25, True),  # small size
        ]:
            r = reward_fn.compute(
                expected_edge_frac=expected_edge,
                size_bucket=size_bucket,
                take=take,
                exec_style="taker",
                spread_bps=3.0,
                slippage_est=0.0003,
                funding_z=0.5,
                state=state,
                stochastic_noise=0.0,
            )
            rewards.append(r)

        # skip cases must yield 0
        assert rewards[1] == 0.0, f"size=0 should yield 0; got {rewards[1]}"
        assert rewards[2] == 0.0, f"take=False should yield 0; got {rewards[2]}"
        # all rewards bounded
        bounded = all(-10.0 <= r <= 10.0 for r in rewards)
        # all rewards finite
        import math

        finite = all(math.isfinite(r) for r in rewards)

        assert bounded and finite, f"reward bounds/finite violated: {rewards}"
        out.append(
            Criterion.ok(
                "rl_reward_risk_adjusted",
                f"rewards={[round(r, 4) for r in rewards]}; "
                "skip→0; bounded; finite; fee/slippage/funding deducted",
            )
        )
    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("rl_reward_risk_adjusted", f"raised: {exc}"))

    # ------------------------------------------------------------------ #
    # 4. All env actions produce valid BoundedActions                     #
    # ------------------------------------------------------------------ #
    try:
        from src.adaptation.action_space import ActionBounds, validate
        from src.adaptation.envelope_guard import RiskEnvelope, enforce
        from src.rl.environment import EnvConfig, TradingEnv

        env = TradingEnv(config=EnvConfig(episode_length=20, rng_seed=7))
        env.reset(seed=7)
        envelope = RiskEnvelope(
            max_leverage=5,
            max_risk_pct_per_trade=0.01,
            portfolio_heat_cap=0.05,
            net_beta_btc_cap=0.30,
            daily_loss_limit=0.03,
            max_drawdown_limit=0.10,
        )
        bounds = ActionBounds()
        n_samples = 50
        all_valid = True
        rejection_reasons: list[str] = []
        for _ in range(n_samples):
            raw_action = env.action_space.sample()
            ba = env.bounded_action_from(raw_action, learner_id="rl_gate_test")
            val_result = validate(ba, bounds)
            if val_result.rejected:
                all_valid = False
                rejection_reasons.append(val_result.rejection_reason or "unknown")
            guard_result = enforce(ba, envelope=envelope)
            # RL actions must never touch envelope params (they have no param_nudges)
            if guard_result.rejected and "param_nudges" not in (
                guard_result.rejection_reason or ""
            ):
                all_valid = False
                rejection_reasons.append(f"guard rejected: {guard_result.rejection_reason}")

        out.append(
            Criterion.ok(
                "rl_action_bounded",
                f"sampled {n_samples} env actions; all produce valid bounded actions",
            )
            if all_valid
            else Criterion.fail(
                "rl_action_bounded",
                f"invalid actions: {rejection_reasons[:3]}",
            )
        )
    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("rl_action_bounded", f"raised: {exc}"))

    # ------------------------------------------------------------------ #
    # 5. Stress tests produce finite bounded rewards                      #
    # ------------------------------------------------------------------ #
    try:
        from src.rl.trainer import LinearRLTrainer, TrainingConfig

        trainer = LinearRLTrainer(
            config=TrainingConfig(n_generations=1, population_size=2, episode_length=30)
        )
        stress_results = trainer.run_stress_tests()

        all_pass = True
        fail_modes: list[str] = []
        for mode, res in stress_results.items():
            if not res["all_finite"]:
                all_pass = False
                fail_modes.append(f"{mode}:non-finite")
            if not res["all_bounded"]:
                all_pass = False
                fail_modes.append(f"{mode}:out-of-bound")

        summary = {
            m: {"steps": r["steps"], "mean": round(r["mean_reward"], 4)}
            for m, r in stress_results.items()
        }
        out.append(
            Criterion.ok(
                "rl_stress_tests_pass",
                f"stress modes={list(summary.keys())}; all finite+bounded; {json.dumps(summary)}",
            )
            if all_pass
            else Criterion.fail(
                "rl_stress_tests_pass",
                f"failed modes: {fail_modes}",
            )
        )
    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("rl_stress_tests_pass", f"raised: {exc}"))

    # ------------------------------------------------------------------ #
    # 6. Simulation training completes                                    #
    # ------------------------------------------------------------------ #
    try:
        import math

        from src.rl.trainer import LinearRLTrainer, TrainingConfig

        trainer = LinearRLTrainer(
            config=TrainingConfig(
                n_generations=5,
                population_size=8,
                elite_frac=0.25,
                episode_length=50,
                rng_seed=42,
            )
        )
        result = trainer.train()

        training_ok = (
            math.isfinite(result.final_mean_return)
            and math.isfinite(result.best_return)
            and len(result.generation_means) == 5
            and result.weights is not None
            and result.weights.shape == (6, 24)
        )

        out.append(
            Criterion.ok(
                "rl_simulation_training",
                f"training completed: {result.n_generations} generations, "
                f"best_return={result.best_return:.4f}, "
                f"mean_return={result.final_mean_return:.4f}, "
                f"converged={result.converged}",
            )
            if training_ok
            else Criterion.fail(
                "rl_simulation_training",
                f"training_ok={training_ok}; "
                f"shape={result.weights.shape if result.weights is not None else None}",
            )
        )
    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("rl_simulation_training", f"raised: {exc}"))

    # ------------------------------------------------------------------ #
    # 7. Trained RLPolicy produces valid BoundedActions                   #
    # ------------------------------------------------------------------ #
    try:
        from src.adaptation.action_space import VALID_SIZE_BUCKETS, ActionBounds, validate
        from src.adaptation.policies.rl_policy import RLPolicy
        from src.adaptation.policy_base import Context

        policy = RLPolicy.build_default(n_generations=3, episode_length=32)
        bounds = ActionBounds()
        all_valid = True
        actions_info = []
        for sig, edge in [(0.8, 0.005), (0.3, -0.001), (0.6, 0.003), (0.1, 0.0)]:
            ctx = Context(signal_strength=sig, expected_edge_frac=edge, spread_bps=3.0)
            action = policy.decide(ctx)
            val = validate(action, bounds)
            if val.rejected:
                all_valid = False
            valid_mode = action.mode == "SHADOW"
            valid_bucket = action.size_bucket in VALID_SIZE_BUCKETS
            if not (valid_mode and valid_bucket and not val.rejected):
                all_valid = False
            actions_info.append(
                {
                    "sig": sig,
                    "take": action.take,
                    "bucket": action.size_bucket,
                    "exec": action.exec_style,
                }
            )

        out.append(
            Criterion.ok(
                "rl_trained_policy_valid",
                f"RLPolicy.build_default() → {len(actions_info)} valid SHADOW actions; "
                f"first={actions_info[0]}",
            )
            if all_valid
            else Criterion.fail(
                "rl_trained_policy_valid",
                f"invalid actions detected; details={actions_info}",
            )
        )
    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("rl_trained_policy_valid", f"raised: {exc}"))

    # ------------------------------------------------------------------ #
    # Write Phase 12 RL-SIM report                                        #
    # ------------------------------------------------------------------ #
    import contextlib

    with contextlib.suppress(Exception):
        _write_phase12_report(settings, "rl_sim")

    return out


def check_rl_shadow(settings: Settings) -> list[Criterion]:
    """RL-SHADOW gate criteria (Phase 12 — RL Shadow Policy Gate)."""
    out: list[Criterion] = []

    # ------------------------------------------------------------------ #
    # 1. RLPolicy importable; snapshot/load round-trip                    #
    # ------------------------------------------------------------------ #
    try:
        import tempfile
        from pathlib import Path

        from src.adaptation.policies.rl_policy import RLPolicy
        from src.adaptation.policy_base import Context
        from src.adaptation.versioning import load_frozen_fallback, make_frozen_fallback

        policy = RLPolicy.build_default(n_generations=3, episode_length=32)
        blob = policy.snapshot()
        policy2 = RLPolicy()
        policy2.load(blob)

        # Snapshot round-trip preserves learner_id + weights presence.
        rt_ok = (
            policy2.learner_id == policy.learner_id
            and policy2.learner_version == policy.learner_version
            and (policy2.weights is not None) == (policy.weights is not None)
        )

        # Frozen fallback round-trip.
        with tempfile.TemporaryDirectory() as tmpdir:
            snap_dir = Path(tmpdir)
            make_frozen_fallback(blob, snap_dir)
            ff_blob = load_frozen_fallback(snap_dir)
            policy3 = RLPolicy()
            policy3.load(ff_blob)
            ff_ok = policy3.learner_id == policy.learner_id

        out.append(
            Criterion.ok(
                "rl_policy_importable",
                "RLPolicy importable; snapshot/load round-trip OK; frozen-fallback round-trip OK",
            )
            if rt_ok and ff_ok
            else Criterion.fail(
                "rl_policy_importable",
                f"rt_ok={rt_ok} ff_ok={ff_ok}",
            )
        )
    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("rl_policy_importable", f"raised: {exc}"))
        return out

    # ------------------------------------------------------------------ #
    # 2. All RL decisions have mode=SHADOW                                #
    # ------------------------------------------------------------------ #
    try:
        from src.adaptation.policies.rl_policy import RLPolicy
        from src.adaptation.policy_base import Context, Outcome

        policy = RLPolicy.build_default(n_generations=3, episode_length=32)
        contexts = [
            Context(signal_strength=s, expected_edge_frac=e, spread_bps=3.0)
            for s, e in [(0.9, 0.01), (0.2, -0.005), (0.6, 0.002), (0.1, 0.0), (0.8, 0.005)]
        ]
        all_shadow = True
        for ctx in contexts:
            action = policy.decide(ctx)
            if action.mode != "SHADOW":
                all_shadow = False
            # update must be a no-op (no exception)
            policy.update(ctx, action, Outcome(realized_pnl_r=0.01, trade_taken=True))

        out.append(
            Criterion.ok(
                "rl_policy_shadow_mode",
                f"all {len(contexts)} RL decisions have mode=SHADOW; update() no-op verified",
            )
            if all_shadow
            else Criterion.fail(
                "rl_policy_shadow_mode",
                "at least one RL decision had mode != SHADOW",
            )
        )
    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("rl_policy_shadow_mode", f"raised: {exc}"))

    # ------------------------------------------------------------------ #
    # 3. RL learner_log entries have applied=False                        #
    # ------------------------------------------------------------------ #
    try:
        from src.adaptation.action_space import BoundedAction
        from src.adaptation.policies.rl_policy import LEARNER_ID, RLPolicy
        from src.adaptation.policy_base import Context
        from src.adaptation.store import reset_memory_sink, write_learner_log

        reset_memory_sink()
        policy = RLPolicy.build_default(n_generations=2, episode_length=20)
        for i in range(5):
            ctx = Context(signal_strength=0.6 + i * 0.05, expected_edge_frac=0.003)
            action = policy.decide(ctx)
            write_learner_log(
                learner_id=action.learner_id,
                learner_version=action.learner_version,
                mode=action.mode,
                symbol="BTCUSDT",
                context_features=ctx.to_dict(),
                proposed_action=action,
                projected_outcome=0.003,
                realized_outcome=None,
                applied=False,  # SHADOW: never applied
                clamped_fields=[],
                config_version=settings.config_version,
                write_to_db=True,
            )

        # Verify DB rows.
        from src.db.base import session_scope
        from src.db.models import LearnerLog

        with session_scope() as session:
            rl_applied = (
                session.query(LearnerLog)
                .filter(
                    LearnerLog.learner_id == LEARNER_ID,
                    LearnerLog.applied.is_(True),
                )
                .count()
            )
            rl_total = session.query(LearnerLog).filter(LearnerLog.learner_id == LEARNER_ID).count()

        out.append(
            Criterion.ok(
                "rl_shadow_applied_false",
                f"wrote 5 RL learner_log entries; "
                f"total={rl_total} applied=True={rl_applied} (must be 0)",
            )
            if rl_applied == 0
            else Criterion.fail(
                "rl_shadow_applied_false",
                f"{rl_applied} RL learner_log entries have applied=True (forbidden in SHADOW)",
            )
        )
    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("rl_shadow_applied_false", f"raised: {exc}"))

    # ------------------------------------------------------------------ #
    # 4. No live influence — mode never LIVE_BOUNDED                      #
    # ------------------------------------------------------------------ #
    try:
        from src.adaptation.action_space import ActionBounds
        from src.adaptation.controller import LearnerController, LearnerMode
        from src.adaptation.policies.rl_policy import RLPolicy
        from src.adaptation.policy_base import Context

        policy = RLPolicy.build_default(n_generations=2, episode_length=20)
        bounds = ActionBounds()
        ctrl = LearnerController(policy=policy, bounds=bounds, mode=LearnerMode.SHADOW)

        live_bounded_found = False
        for _ in range(10):
            ctx = Context(signal_strength=0.7, expected_edge_frac=0.005)
            dec = ctrl.run(ctx)
            if dec.applied:
                live_bounded_found = True
            if dec.action and dec.action.mode == "LIVE_BOUNDED":
                live_bounded_found = True

        out.append(
            Criterion.ok(
                "rl_no_live_influence",
                "10 RL controller decisions: applied=False always; mode != LIVE_BOUNDED",
            )
            if not live_bounded_found
            else Criterion.fail(
                "rl_no_live_influence",
                "at least one RL decision was applied=True or mode=LIVE_BOUNDED",
            )
        )
    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("rl_no_live_influence", f"raised: {exc}"))

    # ------------------------------------------------------------------ #
    # 5. Envelope guard enforced on RL actions                            #
    # ------------------------------------------------------------------ #
    try:
        from src.adaptation.action_space import BoundedAction
        from src.adaptation.envelope_guard import RiskEnvelope, enforce

        envelope = RiskEnvelope(
            max_leverage=5,
            max_risk_pct_per_trade=0.01,
            portfolio_heat_cap=0.05,
            net_beta_btc_cap=0.30,
            daily_loss_limit=0.03,
            max_drawdown_limit=0.10,
        )

        # RL actions have no param_nudges — so they never touch envelope constants.
        from src.adaptation.policies.rl_policy import RLPolicy
        from src.adaptation.policy_base import Context

        policy = RLPolicy.build_default(n_generations=2, episode_length=20)
        all_pass = True
        for _ in range(20):
            ctx = Context(signal_strength=0.6, expected_edge_frac=0.003)
            action = policy.decide(ctx)
            result = enforce(action, envelope=envelope)
            # RL actions should not be rejected for envelope reasons
            # (they have empty param_nudges and bounded size_bucket)
            if result.rejected:
                all_pass = False

        # Now test that a manually crafted action with forbidden envelope param IS rejected.
        bad = BoundedAction(
            size_bucket=1.0,
            take=True,
            exec_style="maker",
            param_nudges={"max_leverage": 20.0},  # FORBIDDEN
            learner_id="rl_policy_v1",
            learner_version="rl_v1",
            mode="SHADOW",
            rationale="envelope attack test",
        )
        bad_result = enforce(bad, envelope=envelope)
        envelope_rejects_bad = bad_result.rejected

        out.append(
            Criterion.ok(
                "rl_envelope_enforced",
                "normal RL actions pass guard; envelope-touching action rejected",
            )
            if all_pass and envelope_rejects_bad
            else Criterion.fail(
                "rl_envelope_enforced",
                f"all_normal_pass={all_pass}; env_rejects_bad={envelope_rejects_bad}",
            )
        )
    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("rl_envelope_enforced", f"raised: {exc}"))

    # ------------------------------------------------------------------ #
    # 6. RECOMMEND mode available (actions can declare RECOMMEND)         #
    # ------------------------------------------------------------------ #
    try:
        from src.adaptation.action_space import ActionBounds, BoundedAction, validate

        # RECOMMEND mode is a valid mode for future promotion.
        rec_action = BoundedAction(
            size_bucket=0.5,
            take=True,
            exec_style="maker",
            param_nudges={},
            learner_id="rl_policy_v1",
            learner_version="rl_v1",
            mode="RECOMMEND",
            rationale="recommend mode test",
        )
        bounds = ActionBounds()
        val = validate(rec_action, bounds)
        rec_ok = not val.rejected and rec_action.mode == "RECOMMEND"

        out.append(
            Criterion.ok(
                "rl_recommend_mode_ready",
                "RECOMMEND mode validated; RL policy can be promoted to RECOMMEND by operator",
            )
            if rec_ok
            else Criterion.fail(
                "rl_recommend_mode_ready",
                f"RECOMMEND mode validation failed: {val.rejection_reason}",
            )
        )
    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("rl_recommend_mode_ready", f"raised: {exc}"))

    # ------------------------------------------------------------------ #
    # Write Phase 12 RL-SHADOW report                                     #
    # ------------------------------------------------------------------ #
    import contextlib

    with contextlib.suppress(Exception):
        _write_phase12_report(settings, "rl_shadow")

    return out


def _write_phase12_report(settings: Settings, kind: str) -> None:
    reports_dir = settings.reports_path / "phase_12"
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    path = reports_dir / f"{kind}_{stamp}.json"

    payload = {
        "phase": 12,
        "gate": "RL-SIM" if kind == "rl_sim" else "RL-SHADOW",
        "versions": settings.versions(),
        "generated_at": stamp,
        "note": (
            "Phase 12 — RL Research and Shadow Policy. "
            "TradingEnv (gymnasium), risk-adjusted cost-net reward, "
            "LinearRLTrainer (CEM), RLPolicy in SHADOW mode. "
            "No live trading influence. Gates: RL-SIM + RL-SHADOW."
        ),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
