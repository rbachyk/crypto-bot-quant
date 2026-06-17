"""Phase 11 — Online Learning Shadow tests (AGENTS.md Section 21, Appendix D).

Tests cover:
  - BoundedAction schema and validation (action_space.validate)
  - Envelope guard (envelope_guard.enforce)
  - Policy implementations (OnlineLogRegPolicy, GaussianTSBandit, RLPolicyStub)
  - Controller state machine (SHADOW mode applied=False)
  - Rollback guard circuit breakers
  - Scorer (walk-forward + hold-out + calibration + drift)
  - Versioning (snapshot / frozen-fallback round-trip)
  - Store (LearnerLog DB + in-memory)
  - Config loader (adaptation.yaml)
  - Integration: full shadow decision path
"""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

# ======================================================================== #
# action_space                                                              #
# ======================================================================== #


class TestBoundedActionValidation:
    def _bounds(self, **kw):
        from src.adaptation.action_space import ActionBounds

        defaults = {
            "w_min": 0.0,
            "w_max": 2.0,
            "size_buckets": (0.0, 0.25, 0.5, 1.0),
            "registered_tunables": {},
            "allowed_strategies": set(),
        }
        defaults.update(kw)
        return ActionBounds(**defaults)

    def _action(self, **kw):
        from src.adaptation.action_space import BoundedAction

        defaults = {
            "strategy_weights": {},
            "size_bucket": 1.0,
            "take": True,
            "exec_style": "maker",
            "param_nudges": {},
            "learner_id": "test",
            "learner_version": "v0",
            "mode": "SHADOW",
            "rationale": "test",
        }
        defaults.update(kw)
        return BoundedAction(**defaults)

    def test_valid_action_passes(self):
        from src.adaptation.action_space import validate

        action = self._action()
        result = validate(action, self._bounds())
        assert not result.rejected
        assert result.clamped_fields == []

    def test_invalid_bucket_rejected_by_default(self):
        from src.adaptation.action_space import validate

        action = self._action(size_bucket=0.7)
        result = validate(action, self._bounds(), reject_on_bad_bucket=True)
        assert result.rejected
        assert "size_bucket" in (result.rejection_reason or "")

    def test_invalid_bucket_clamped_when_configured(self):
        from src.adaptation.action_space import validate

        action = self._action(size_bucket=0.7)
        result = validate(action, self._bounds(), reject_on_bad_bucket=False)
        assert not result.rejected
        assert result.action.size_bucket in (0.0, 0.25, 0.5, 1.0)
        assert "size_bucket" in result.clamped_fields

    def test_size_bucket_never_exceeds_1(self):
        from src.adaptation.action_space import validate

        action = self._action(size_bucket=1.5)
        result = validate(action, self._bounds(), reject_on_bad_bucket=False)
        assert not result.rejected
        assert result.action.size_bucket <= 1.0

    def test_unregistered_param_nudge_rejected(self):
        from src.adaptation.action_space import validate

        action = self._action(param_nudges={"unknown_param": 0.5})
        result = validate(action, self._bounds())
        assert result.rejected
        assert "unknown_param" in (result.rejection_reason or "")

    def test_invalid_mode_rejected(self):
        from src.adaptation.action_space import validate

        action = self._action(mode="LIVE")  # not a valid mode literal
        result = validate(action, self._bounds())
        assert result.rejected

    def test_strategy_weight_clamped(self):
        from src.adaptation.action_space import validate

        action = self._action(strategy_weights={"strat_A": 5.0})  # > w_max=2.0
        bounds = self._bounds(allowed_strategies={"strat_A"})
        result = validate(action, bounds)
        assert not result.rejected
        assert result.action.strategy_weights["strat_A"] == 2.0
        assert "strategy_weights" in result.clamped_fields

    def test_unknown_strategy_removed_from_weights(self):
        from src.adaptation.action_space import validate

        action = self._action(strategy_weights={"strat_A": 1.0, "unknown_strat": 1.0})
        bounds = self._bounds(allowed_strategies={"strat_A"})
        result = validate(action, bounds)
        assert not result.rejected
        assert "unknown_strat" not in result.action.strategy_weights

    def test_registered_tunable_clamped_in_range(self):
        from src.adaptation.action_space import validate

        action = self._action(param_nudges={"entry_offset": 0.9})
        bounds = self._bounds(registered_tunables={"entry_offset": {"lo": 0.0, "hi": 0.5}})
        result = validate(action, bounds)
        assert not result.rejected
        assert result.action.param_nudges["entry_offset"] == 0.5
        assert "param_nudges.entry_offset" in result.clamped_fields


# ======================================================================== #
# envelope_guard                                                            #
# ======================================================================== #


class TestEnvelopeGuard:
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

    def _action(self, **kw):
        from src.adaptation.action_space import BoundedAction

        defaults = {
            "strategy_weights": {},
            "size_bucket": 1.0,
            "take": True,
            "exec_style": "maker",
            "param_nudges": {},
            "learner_id": "test",
            "learner_version": "v0",
            "mode": "SHADOW",
            "rationale": "test",
        }
        defaults.update(kw)
        return BoundedAction(**defaults)

    def test_valid_action_passes(self):
        from src.adaptation.envelope_guard import enforce

        result = enforce(self._action(), envelope=self._envelope())
        assert not result.rejected

    def test_forbidden_envelope_param_rejected(self):
        from src.adaptation.envelope_guard import enforce

        action = self._action(param_nudges={"max_leverage": 10.0})
        result = enforce(action, envelope=self._envelope())
        assert result.rejected
        assert "max_leverage" in (result.rejection_reason or "")

    def test_forbidden_stop_frac_rejected(self):
        from src.adaptation.envelope_guard import enforce

        action = self._action(param_nudges={"stop_frac": 0.0001})
        result = enforce(action, envelope=self._envelope())
        assert result.rejected

    def test_unknown_strategy_rejected(self):
        from src.adaptation.envelope_guard import enforce

        action = self._action(strategy_weights={"rogue_strat": 1.0})
        result = enforce(
            action,
            active_strategies={"valid_strat"},
            envelope=self._envelope(),
        )
        assert result.rejected
        assert "rogue_strat" in (result.rejection_reason or "")

    def test_size_bucket_clamped_to_1(self):
        from src.adaptation.envelope_guard import enforce

        action = self._action(size_bucket=1.5)
        result = enforce(action, envelope=self._envelope())
        assert not result.rejected
        assert result.action.size_bucket == 1.0
        assert "size_bucket" in result.clamped_fields

    def test_normal_strategy_passes(self):
        from src.adaptation.envelope_guard import enforce

        action = self._action(strategy_weights={"valid_strat": 1.0})
        result = enforce(
            action,
            active_strategies={"valid_strat"},
            envelope=self._envelope(),
        )
        assert not result.rejected

    def test_forbidden_drawdown_param_rejected(self):
        from src.adaptation.envelope_guard import enforce

        action = self._action(param_nudges={"max_drawdown_limit": 0.20})
        result = enforce(action, envelope=self._envelope())
        assert result.rejected


# ======================================================================== #
# OnlineLogRegPolicy                                                        #
# ======================================================================== #


class TestOnlineLogRegPolicy:
    def test_decide_produces_bounded_action(self):
        from src.adaptation.policies.online_logreg import OnlineLogRegPolicy
        from src.adaptation.policy_base import Context

        policy = OnlineLogRegPolicy()
        ctx = Context(signal_strength=0.7, expected_edge_frac=0.02)
        action = policy.decide(ctx)
        assert action.mode == "SHADOW"
        assert action.size_bucket in (0.0, 0.25, 0.5, 1.0)
        assert not action.param_nudges

    def test_update_does_not_raise(self):
        from src.adaptation.policies.online_logreg import OnlineLogRegPolicy
        from src.adaptation.policy_base import Context, Outcome

        policy = OnlineLogRegPolicy()
        ctx = Context(signal_strength=0.6)
        action = policy.decide(ctx)
        for pnl in [0.1, -0.05, 0.2]:
            policy.update(ctx, action, Outcome(realized_pnl_r=pnl, trade_taken=True))

    def test_snapshot_roundtrip(self):
        from src.adaptation.policies.online_logreg import OnlineLogRegPolicy
        from src.adaptation.policy_base import Context, Outcome

        policy = OnlineLogRegPolicy(learner_id="test_lr", learner_version="v1")
        ctx = Context(signal_strength=0.5)
        action = policy.decide(ctx)
        for pnl in [0.1, 0.2, -0.05]:
            policy.update(ctx, action, Outcome(realized_pnl_r=pnl, trade_taken=True))
        blob = policy.snapshot()
        policy2 = OnlineLogRegPolicy()
        policy2.load(blob)
        assert policy2.learner_id == "test_lr"
        assert policy2._n_updates == policy._n_updates

    def test_no_forbidden_nudges_ever(self):
        from src.adaptation.policies.online_logreg import OnlineLogRegPolicy
        from src.adaptation.policy_base import Context

        policy = OnlineLogRegPolicy()
        for _ in range(10):
            ctx = Context(signal_strength=0.5 + _ * 0.05)
            action = policy.decide(ctx)
            assert not action.param_nudges
            assert action.strategy_weights == {}


# ======================================================================== #
# GaussianTSBandit                                                          #
# ======================================================================== #


class TestGaussianTSBandit:
    def test_decide_produces_bounded_action(self):
        from src.adaptation.policies.bandit import GaussianTSBandit
        from src.adaptation.policy_base import Context

        bandit = GaussianTSBandit()
        action = bandit.decide(Context(strategy_id="strat_A"))
        assert action.mode == "SHADOW"
        assert not action.param_nudges

    def test_update_learns_positive_arm(self):
        from src.adaptation.policies.bandit import GaussianTSBandit
        from src.adaptation.policy_base import Context, Outcome

        bandit = GaussianTSBandit(w_min=0.0, w_max=2.0)
        ctx_a = Context(strategy_id="strat_A")
        ctx_b = Context(strategy_id="strat_B")
        action_a = bandit.decide(ctx_a)
        action_b = bandit.decide(ctx_b)
        # strat_A gets positive outcomes, strat_B negative.
        for _ in range(5):
            bandit.update(ctx_a, action_a, Outcome(realized_pnl_r=0.1, trade_taken=True))
            bandit.update(ctx_b, action_b, Outcome(realized_pnl_r=-0.1, trade_taken=True))
        assert "strat_A" in bandit._arms
        assert bandit._arms["strat_A"].mu > bandit._arms["strat_B"].mu

    def test_snapshot_roundtrip(self):
        from src.adaptation.policies.bandit import GaussianTSBandit
        from src.adaptation.policy_base import Context, Outcome

        bandit = GaussianTSBandit()
        ctx = Context(strategy_id="strat_A")
        action = bandit.decide(ctx)
        bandit.update(ctx, action, Outcome(realized_pnl_r=0.1, trade_taken=True))
        blob = bandit.snapshot()
        bandit2 = GaussianTSBandit()
        bandit2.load(blob)
        assert "strat_A" in bandit2._arms


# ======================================================================== #
# RLPolicyStub                                                              #
# ======================================================================== #


class TestRLPolicyStub:
    def test_decide_produces_valid_shadow_action(self):
        from src.adaptation.policies.rl_policy import RLPolicyStub
        from src.adaptation.policy_base import Context

        stub = RLPolicyStub()
        action = stub.decide(Context())
        assert action.mode == "SHADOW"
        assert action.size_bucket in (0.0, 0.25, 0.5, 1.0)
        assert not action.param_nudges

    def test_update_is_noop(self):
        from src.adaptation.policies.rl_policy import RLPolicyStub
        from src.adaptation.policy_base import Context, Outcome

        stub = RLPolicyStub()
        ctx = Context()
        action = stub.decide(ctx)
        stub.update(ctx, action, Outcome(realized_pnl_r=0.1))  # must not raise

    def test_snapshot_roundtrip(self):
        from src.adaptation.policies.rl_policy import RLPolicyStub
        from src.adaptation.policy_base import Context

        stub = RLPolicyStub(learner_id="rl_test")
        stub.decide(Context())
        blob = stub.snapshot()
        stub2 = RLPolicyStub()
        stub2.load(blob)
        assert stub2.learner_id == "rl_test"
        assert stub2._n_decisions == 1


# ======================================================================== #
# LearnerController                                                         #
# ======================================================================== #


class TestLearnerController:
    def _make_ctrl(self, mode="SHADOW"):
        from src.adaptation.action_space import ActionBounds
        from src.adaptation.controller import LearnerController, LearnerMode
        from src.adaptation.policies.online_logreg import OnlineLogRegPolicy

        bounds = ActionBounds()
        policy = OnlineLogRegPolicy()
        return LearnerController(
            policy=policy,
            bounds=bounds,
            mode=LearnerMode(mode),
        )

    def test_shadow_mode_applied_always_false(self):
        ctrl = self._make_ctrl("SHADOW")
        from src.adaptation.policy_base import Context

        for _ in range(5):
            dec = ctrl.run(Context(signal_strength=0.7))
            assert not dec.applied
            assert not dec.rejected

    def test_shadow_mode_action_is_valid(self):
        ctrl = self._make_ctrl("SHADOW")
        from src.adaptation.policy_base import Context

        dec = ctrl.run(Context(signal_strength=0.8))
        assert dec.action is not None
        assert dec.action.mode == "SHADOW"
        assert dec.action.size_bucket in (0.0, 0.25, 0.5, 1.0)

    def test_freeze_transitions_to_frozen(self):
        ctrl = self._make_ctrl("SHADOW")
        assert not ctrl.is_frozen()
        ctrl.freeze(reason="test")
        assert ctrl.is_frozen()
        from src.adaptation.controller import LearnerMode

        assert ctrl.mode is LearnerMode.FROZEN

    def test_live_bounded_applied_true(self):
        from src.adaptation.action_space import ActionBounds
        from src.adaptation.controller import LearnerController, LearnerMode
        from src.adaptation.policies.online_logreg import OnlineLogRegPolicy
        from src.adaptation.policy_base import Context

        bounds = ActionBounds()
        policy = OnlineLogRegPolicy()
        ctrl = LearnerController(
            policy=policy,
            bounds=bounds,
            mode=LearnerMode.LIVE_BOUNDED,
        )
        dec = ctrl.run(Context(signal_strength=0.7))
        # In LIVE_BOUNDED mode, applied should be True (assuming not frozen).
        assert not ctrl.is_frozen()
        assert dec.applied

    def test_record_outcome_does_not_raise(self):
        ctrl = self._make_ctrl("SHADOW")
        from src.adaptation.policy_base import Context, Outcome

        ctx = Context(signal_strength=0.6)
        dec = ctrl.run(ctx)
        ctrl.record_outcome(ctx, dec, Outcome(realized_pnl_r=0.1, trade_taken=True))


# ======================================================================== #
# RollbackGuard                                                             #
# ======================================================================== #


class TestRollbackGuard:
    def _make_ctrl_and_guard(self, **guard_kw):
        from src.adaptation.action_space import ActionBounds
        from src.adaptation.controller import LearnerController, LearnerMode
        from src.adaptation.policies.online_logreg import OnlineLogRegPolicy
        from src.adaptation.rollback import RollbackGuard

        bounds = ActionBounds()
        policy = OnlineLogRegPolicy()
        ctrl = LearnerController(policy=policy, bounds=bounds, mode=LearnerMode.LIVE_BOUNDED)
        defaults = {"rollback_window": 5, "rollback_margin": 0.02}
        defaults.update(guard_kw)
        guard = RollbackGuard(**defaults)
        return ctrl, guard

    def test_no_rollback_without_trigger(self):
        ctrl, guard = self._make_ctrl_and_guard()
        event = guard.check(ctrl)
        assert event is None
        assert not ctrl.is_frozen()

    def test_envelope_breaker_triggers_rollback(self):
        ctrl, guard = self._make_ctrl_and_guard()
        guard.set_envelope_breaker(True)
        event = guard.check(ctrl)
        assert event is not None
        assert event.trigger == "envelope_breaker"
        assert ctrl.is_frozen()

    def test_regime_unsafe_triggers_rollback(self):
        ctrl, guard = self._make_ctrl_and_guard()
        guard.set_regime_unsafe("R8_DATA_UNSAFE")
        event = guard.check(ctrl)
        assert event is not None
        assert ctrl.is_frozen()

    def test_underperformance_triggers_rollback(self):
        ctrl, guard = self._make_ctrl_and_guard(rollback_window=4, rollback_margin=0.01)
        for _i in range(4):
            guard.add_decision(
                projected_outcome=0.10,
                realized_outcome=0.05,  # shortfall = 0.05 > 0.01
            )
        event = guard.check(ctrl)
        assert event is not None
        assert event.trigger == "underperformance"
        assert ctrl.is_frozen()

    def test_no_rollback_if_performance_ok(self):
        ctrl, guard = self._make_ctrl_and_guard(rollback_window=4, rollback_margin=0.05)
        for _ in range(4):
            guard.add_decision(projected_outcome=0.10, realized_outcome=0.09)  # small gap
        event = guard.check(ctrl)
        assert event is None

    def test_no_second_rollback_after_freeze(self):
        ctrl, guard = self._make_ctrl_and_guard()
        guard.set_envelope_breaker(True)
        event1 = guard.check(ctrl)
        assert event1 is not None
        event2 = guard.check(ctrl)  # already frozen; check returns None
        assert event2 is None

    def test_events_accumulated(self):
        ctrl, guard = self._make_ctrl_and_guard()
        guard.set_envelope_breaker(True)
        guard.check(ctrl)
        assert len(guard.events()) == 1


# ======================================================================== #
# Scorer                                                                    #
# ======================================================================== #


class TestScorer:
    def _decisions(self, n: int, pnl: float = 0.05, projected: float = 0.04):
        from src.adaptation.scorer import ShadowDecision

        return [
            ShadowDecision(
                ts=datetime.now(UTC),
                symbol="BTCUSDT",
                projected_outcome=projected,
                realized_outcome=pnl,
                take=True,
                mode="SHADOW",
            )
            for _ in range(n)
        ]

    def test_insufficient_outcomes_returns_ineligible(self):
        from src.adaptation.scorer import score_shadow_decisions

        result = score_shadow_decisions([])
        assert not result.promotion_eligible
        assert "insufficient" in result.note

    def test_positive_edge_detected(self):
        from src.adaptation.scorer import score_shadow_decisions

        decisions = self._decisions(40, pnl=0.05, projected=0.04)
        result = score_shadow_decisions(
            decisions, baseline_mean=-0.01, n_folds=4, min_holdout_edge=0.0
        )
        assert result.holdout_edge is not None and result.holdout_edge > 0.0

    def test_drift_scores_computed(self):
        from src.adaptation.scorer import score_shadow_decisions

        decisions = self._decisions(30, pnl=0.03, projected=0.05)  # drift=0.02
        result = score_shadow_decisions(decisions, drift_window=10, max_drift_per_window=0.20)
        assert len(result.drift_scores) >= 1
        assert all(s >= 0 for s in result.drift_scores)

    def test_brier_score_computed(self):
        from src.adaptation.scorer import score_shadow_decisions

        decisions = self._decisions(30, pnl=0.05, projected=0.04)
        result = score_shadow_decisions(decisions)
        assert result.brier_score is not None

    def test_large_drift_fails_calibration(self):
        from src.adaptation.scorer import score_shadow_decisions

        # very large drift (projected=2.0, realized=0.01)
        decisions = self._decisions(30, pnl=0.01, projected=2.0)
        result = score_shadow_decisions(decisions, max_drift_per_window=0.10, drift_window=10)
        assert not result.drift_passed


# ======================================================================== #
# Versioning                                                                #
# ======================================================================== #


class TestVersioning:
    def test_snapshot_save_and_load(self):
        from src.adaptation.policies.online_logreg import OnlineLogRegPolicy
        from src.adaptation.versioning import load_snapshot, save_snapshot

        with tempfile.TemporaryDirectory() as tmpdir:
            policy = OnlineLogRegPolicy(learner_id="test_snap")
            blob = policy.snapshot()
            meta = save_snapshot(blob, "test_snap", "v1", "SHADOW", Path(tmpdir))
            assert meta.checksum
            loaded = load_snapshot(meta.snapshot_id, Path(tmpdir))
            assert loaded == blob

    def test_frozen_fallback_roundtrip(self):
        from src.adaptation.policies.online_logreg import OnlineLogRegPolicy
        from src.adaptation.versioning import load_frozen_fallback, make_frozen_fallback

        with tempfile.TemporaryDirectory() as tmpdir:
            policy = OnlineLogRegPolicy(learner_id="ff_test")
            blob = policy.snapshot()
            make_frozen_fallback(blob, Path(tmpdir))
            loaded = load_frozen_fallback(Path(tmpdir))
            assert loaded == blob
            policy2 = OnlineLogRegPolicy()
            policy2.load(loaded)
            assert policy2.learner_id == "ff_test"

    def test_snapshot_missing_raises(self):
        from src.adaptation.versioning import load_snapshot

        with tempfile.TemporaryDirectory() as tmpdir, pytest.raises(FileNotFoundError):
            load_snapshot("nonexistent.pkl", Path(tmpdir))

    def test_frozen_fallback_missing_raises(self):
        from src.adaptation.versioning import load_frozen_fallback

        with tempfile.TemporaryDirectory() as tmpdir, pytest.raises(FileNotFoundError):
            load_frozen_fallback(Path(tmpdir))


# ======================================================================== #
# Store (in-memory)                                                         #
# ======================================================================== #


class TestInMemoryStore:
    def test_write_and_recall(self):
        from src.adaptation.action_space import BoundedAction
        from src.adaptation.store import InMemoryLearnerStore, LearnerLogEntry

        store = InMemoryLearnerStore()
        BoundedAction(
            size_bucket=0.5,
            learner_id="test",
            learner_version="v0",
            mode="SHADOW",
            rationale="test",
        )
        entry = LearnerLogEntry(
            ts=datetime.now(UTC),
            learner_id="test",
            learner_version="v0",
            mode="SHADOW",
            symbol="BTCUSDT",
            context_features={"x": 1},
            proposed_action={"size_bucket": 0.5},
            projected_outcome=0.05,
            realized_outcome=None,
            applied=False,
            clamped_fields=[],
            rollback_event=None,
            config_version="cfg_0001",
        )
        store.write(entry)
        recent = store.recent()
        assert len(recent) == 1
        assert recent[0].mode == "SHADOW"
        assert not recent[0].applied

    def test_write_learner_log_in_memory(self):
        from src.adaptation.action_space import BoundedAction
        from src.adaptation.store import get_memory_sink, reset_memory_sink, write_learner_log

        reset_memory_sink()
        action = BoundedAction(
            size_bucket=1.0,
            learner_id="test",
            learner_version="v0",
            mode="SHADOW",
            rationale="test",
        )
        entry = write_learner_log(
            learner_id="test",
            learner_version="v0",
            mode="SHADOW",
            symbol=None,
            context_features={},
            proposed_action=action,
            projected_outcome=0.0,
            realized_outcome=None,
            applied=False,
            clamped_fields=[],
            write_to_db=False,
        )
        assert not entry.applied
        sink = get_memory_sink()
        assert len(sink.entries) == 1


# ======================================================================== #
# Config                                                                    #
# ======================================================================== #


class TestAdaptationConfig:
    def test_load_config(self):
        from src.adaptation.config import load_adaptation_config

        # Clear LRU cache to avoid cross-test contamination.
        load_adaptation_config.cache_clear()
        cfg = load_adaptation_config()
        assert cfg.mode == "SHADOW"
        assert cfg.enabled
        assert cfg.learner_id
        assert cfg.rollback.auto_freeze_on_breaker  # immutable: always true
        assert 0.0 in cfg.bounds.size_buckets
        assert 1.0 in cfg.bounds.size_buckets

    def test_rollback_auto_freeze_always_true(self):
        """auto_freeze_on_breaker is hard-coded True; config cannot disable it."""
        from src.adaptation.config import load_adaptation_config

        load_adaptation_config.cache_clear()
        cfg = load_adaptation_config()
        assert cfg.rollback.auto_freeze_on_breaker is True


# ======================================================================== #
# Integration — full shadow decision path                                   #
# ======================================================================== #


class TestShadowIntegration:
    def test_full_shadow_path_no_side_effects(self):
        """End-to-end: policy → validate → guard → log → all applied=False."""
        from src.adaptation.action_space import ActionBounds, validate
        from src.adaptation.controller import LearnerController, LearnerMode
        from src.adaptation.envelope_guard import RiskEnvelope, enforce
        from src.adaptation.policies.online_logreg import OnlineLogRegPolicy
        from src.adaptation.policy_base import Context, Outcome
        from src.adaptation.store import reset_memory_sink, write_learner_log

        reset_memory_sink()
        bounds = ActionBounds()
        policy = OnlineLogRegPolicy()
        ctrl = LearnerController(policy=policy, bounds=bounds, mode=LearnerMode.SHADOW)
        envelope = RiskEnvelope(
            max_leverage=5,
            max_risk_pct_per_trade=0.01,
            portfolio_heat_cap=0.05,
            net_beta_btc_cap=0.30,
            daily_loss_limit=0.03,
            max_drawdown_limit=0.10,
        )
        for i in range(10):
            ctx = Context(signal_strength=0.5 + i * 0.04, expected_edge_frac=0.02)
            dec = ctrl.run(ctx)
            assert not dec.applied
            assert dec.action is not None
            assert not dec.action.param_nudges

            # Verify action passes both validate and guard independently.
            val = validate(dec.action, bounds)
            assert not val.rejected
            guard = enforce(val.action, envelope=envelope)
            assert not guard.rejected

            # Log to in-memory store.
            write_learner_log(
                learner_id=policy.learner_id,
                learner_version=policy.learner_version,
                mode="SHADOW",
                symbol="BTCUSDT",
                context_features=ctx.to_dict(),
                proposed_action=dec.action,
                projected_outcome=0.04,
                realized_outcome=0.05 if i % 2 == 0 else -0.01,
                applied=False,
                clamped_fields=dec.clamped_fields,
                write_to_db=False,
            )
            # Feed outcome back for online learning.
            ctrl.record_outcome(ctx, dec, Outcome(realized_pnl_r=0.05))

        from src.adaptation.store import get_memory_sink

        entries = get_memory_sink().recent()
        assert len(entries) == 10
        assert all(not e.applied for e in entries)
        assert all(e.mode == "SHADOW" for e in entries)

    def test_envelope_guard_blocks_forbidden_action_in_pipeline(self):
        """Forbidden envelope action is blocked before it reaches the controller."""
        from src.adaptation.action_space import ActionBounds, BoundedAction, validate
        from src.adaptation.envelope_guard import RiskEnvelope, enforce

        bounds = ActionBounds()
        envelope = RiskEnvelope(
            max_leverage=5,
            max_risk_pct_per_trade=0.01,
            portfolio_heat_cap=0.05,
            net_beta_btc_cap=0.30,
            daily_loss_limit=0.03,
            max_drawdown_limit=0.10,
        )
        bad_action = BoundedAction(
            param_nudges={"max_leverage": 10.0},  # FORBIDDEN
            learner_id="attacker",
            learner_version="v0",
            mode="SHADOW",
            rationale="trying to break envelope",
        )
        # validate allows it (no registered_tunables in bounds so this would be rejected
        # by validate first since max_leverage is not registered).
        val = validate(bad_action, bounds)
        assert val.rejected  # validate blocks unregistered tunables

        # Even if validate were bypassed, guard blocks it.
        bad_action2 = BoundedAction(
            param_nudges={},  # cleared so validate passes
            learner_id="attacker",
            learner_version="v0",
            mode="SHADOW",
            rationale="direct guard test",
        )
        # Manually set param_nudges after validate to simulate bypass attempt.
        bad_action2.param_nudges = {"max_leverage": 10.0}
        guard_result = enforce(bad_action2, envelope=envelope)
        assert guard_result.rejected
        assert "max_leverage" in (guard_result.rejection_reason or "")
