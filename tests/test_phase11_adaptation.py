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

    def test_brier_score_computed_from_win_probability(self):
        """M31: Brier is computed from the policy's genuine probability field."""
        from src.adaptation.scorer import score_shadow_decisions

        decisions = self._decisions(30, pnl=0.05, projected=0.04)
        for d in decisions:
            d.win_probability = 0.7
        result = score_shadow_decisions(decisions)
        # All realized positive, p=0.7 → Brier = (0.7-1)^2 = 0.09.
        assert result.brier_score == pytest.approx(0.09)
        assert result.calibration_passed is True

    def test_brier_not_evaluated_without_win_probability(self):
        """M31: no win_probability → calibration not evaluated, never fake-passed;
        an R-scale projected_outcome must NOT be sigmoided into a pseudo-probability."""
        from src.adaptation.scorer import score_shadow_decisions

        decisions = self._decisions(30, pnl=0.05, projected=0.04)
        result = score_shadow_decisions(decisions)
        assert result.brier_score is None
        assert result.calibration_passed is None
        assert "calibration not evaluated" in result.note

    def test_badly_calibrated_probability_blocks_promotion(self):
        """M31: a real but bad probability now actually fails the calibration gate."""
        from src.adaptation.scorer import score_shadow_decisions

        decisions = self._decisions(40, pnl=0.05, projected=0.04)  # all positive outcomes
        for d in decisions:
            d.win_probability = 0.05  # confidently wrong → Brier ≈ 0.90
        result = score_shadow_decisions(decisions, baseline_mean=-0.01)
        assert result.calibration_passed is False
        assert not result.promotion_eligible

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

    def test_corrupt_index_is_preserved_not_overwritten(self):
        """A torn/corrupt index.json must be set aside (not silently overwritten) when the next
        save_snapshot starts a fresh index, so prior metadata is recoverable (L-versioning)."""
        from src.adaptation.versioning import INDEX_FILENAME, list_snapshots, save_snapshot

        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            (d / INDEX_FILENAME).write_text("{ this is not valid json", encoding="utf-8")
            save_snapshot(b"blob", "snap", "v1", "SHADOW", d)
            # The fresh index has exactly the new entry, and the corrupt bytes were backed up.
            assert len(list_snapshots(d)) == 1
            backups = list(d.glob(f"{INDEX_FILENAME}.corrupt.*"))
            assert backups, "corrupt index must be preserved for recovery"
            assert backups[0].read_text(encoding="utf-8") == "{ this is not valid json"


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


# ======================================================================== #
# LEARN-PROMO-S — real-data promotion criterion (audit H16)                 #
# ======================================================================== #


class TestRealShadowPromotionCriterion:
    """shadow_policy_beats_baseline: scored from REAL learner_logs, fail-closed."""

    def _cfg(self):
        from src.adaptation.config import load_adaptation_config

        return load_adaptation_config()

    def _decisions(self, n: int, realized_pattern, projected: float = 0.06):
        from src.adaptation.scorer import ShadowDecision

        return [
            ShadowDecision(
                ts=datetime.now(UTC),
                symbol="BTCUSDT",
                projected_outcome=projected,
                realized_outcome=realized_pattern(i),
                take=True,
                mode="SHADOW",
            )
            for i in range(n)
        ]

    def test_fails_closed_with_no_decisions(self):
        from src.gates.phase11 import _score_real_shadow_decisions

        crit = _score_real_shadow_decisions([], self._cfg())
        assert crit.id == "shadow_policy_beats_baseline"
        assert not crit.passed
        assert "insufficient REAL SHADOW decisions" in crit.detail
        assert "does not fabricate" in crit.detail

    def test_fails_closed_below_min_samples(self):
        from src.gates.phase11 import _score_real_shadow_decisions

        cfg = self._cfg()
        decisions = self._decisions(10, lambda i: 0.08)
        crit = _score_real_shadow_decisions(decisions, cfg)
        assert not crit.passed
        assert "insufficient" in crit.detail

    def test_positive_edge_policy_passes(self):
        from src.gates.phase11 import _score_real_shadow_decisions

        cfg = self._cfg()
        n = max(cfg.min_samples_to_start, cfg.scoring.min_shadow_decisions) + 10
        decisions = self._decisions(n, lambda i: 0.08 if i % 3 != 0 else -0.02)
        crit = _score_real_shadow_decisions(decisions, cfg)
        assert crit.passed, crit.detail
        assert "promotion_eligible=True" in crit.detail

    def test_negative_edge_policy_fails(self):
        """The audit H16 regression: a negative-edge shadow learner must FAIL."""
        from src.gates.phase11 import _score_real_shadow_decisions

        cfg = self._cfg()
        n = max(cfg.min_samples_to_start, cfg.scoring.min_shadow_decisions) + 10
        decisions = self._decisions(n, lambda i: -0.08 if i % 3 != 0 else 0.02)
        crit = _score_real_shadow_decisions(decisions, cfg)
        assert not crit.passed
        assert "does not beat baseline" in crit.detail

    def test_loader_excludes_self_test_and_synthetic_rows(self):
        """_load_real_shadow_decisions: only the configured learner's SHADOW rows
        with realized outcomes count; gate self-test and pre-fix synthetic-marker
        rows are excluded."""
        from tests.conftest import DB_OK

        if not DB_OK:
            pytest.skip("database not reachable")

        from src.db.base import session_scope
        from src.db.models import LearnerLog
        from src.gates.phase11 import (
            _SELF_TEST_LEARNER_ID,
            _SYNTHETIC_RATIONALE_PREFIX,
            _load_real_shadow_decisions,
        )

        cfg = self._cfg()
        marker_symbol = "GATE11TST"
        ids: list[int] = []
        try:
            with session_scope() as session:
                genuine = LearnerLog(
                    learner_id=cfg.learner_id,
                    learner_version=cfg.learner_version,
                    mode="SHADOW",
                    symbol=marker_symbol,
                    proposed_action={"take": True, "rationale": "shadow decision"},
                    projected_outcome=0.05,
                    realized_outcome=0.11,
                    applied=False,
                )
                self_test = LearnerLog(
                    learner_id=_SELF_TEST_LEARNER_ID,
                    learner_version=cfg.learner_version,
                    mode="SHADOW",
                    symbol=marker_symbol,
                    proposed_action={"take": True, "rationale": "gate self-test"},
                    projected_outcome=0.05,
                    realized_outcome=0.99,
                    applied=False,
                )
                synthetic = LearnerLog(
                    learner_id=cfg.learner_id,
                    learner_version=cfg.learner_version,
                    mode="SHADOW",
                    symbol=marker_symbol,
                    proposed_action={
                        "take": True,
                        "rationale": f"{_SYNTHETIC_RATIONALE_PREFIX}7",
                    },
                    projected_outcome=0.05,
                    realized_outcome=0.99,
                    applied=False,
                )
                no_outcome = LearnerLog(
                    learner_id=cfg.learner_id,
                    learner_version=cfg.learner_version,
                    mode="SHADOW",
                    symbol=marker_symbol,
                    proposed_action={"take": True, "rationale": "shadow decision"},
                    projected_outcome=0.05,
                    realized_outcome=None,
                    applied=False,
                )
                session.add_all([genuine, self_test, synthetic, no_outcome])
                session.flush()
                ids = [genuine.id, self_test.id, synthetic.id, no_outcome.id]

            decisions = _load_real_shadow_decisions(cfg)
            ours = [d for d in decisions if d.symbol == marker_symbol]
            assert len(ours) == 1, f"expected exactly the genuine row, got {len(ours)}"
            assert ours[0].realized_outcome == pytest.approx(0.11)
        finally:
            with session_scope() as session:
                if ids:
                    session.query(LearnerLog).filter(LearnerLog.id.in_(ids)).delete(
                        synchronize_session=False
                    )

    def test_gate_exposes_plumbing_and_real_criteria(self):
        """Full gate run: scorer_plumbing verifies mechanism; the promotion verdict
        criterion is present and (on a system without real learner data) FAILS."""
        from tests.conftest import DB_OK

        if not DB_OK:
            pytest.skip("database not reachable")

        from src.config import get_settings
        from src.gates.phase11 import check_learn_promo_s

        criteria = check_learn_promo_s(get_settings())
        by_name = {c.id: c for c in criteria}
        assert "scorer_plumbing" in by_name
        assert "shadow_policy_beats_baseline" in by_name
        assert by_name["scorer_plumbing"].passed, by_name["scorer_plumbing"].detail
        assert "scorer_runs" not in by_name  # old always-green criterion removed

    def test_gate_removes_its_learner_log_self_test_rows(self):
        from tests.conftest import DB_OK

        if not DB_OK:
            pytest.skip("database not reachable")

        from src.config import get_settings
        from src.db.base import session_scope
        from src.db.models import LearnerLog
        from src.gates.phase11 import _SELF_TEST_LEARNER_ID, check_learn_promo_s

        check_learn_promo_s(get_settings())
        with session_scope() as session:
            leftover = (
                session.query(LearnerLog)
                .filter(LearnerLog.learner_id == _SELF_TEST_LEARNER_ID)
                .count()
            )
        assert leftover == 0, "gate left self-test learner_log rows behind"


# ======================================================================== #
# H17 — freeze/rollback restores the frozen fallback (or reports disabled)  #
# ======================================================================== #


class TestFrozenFallbackSemantics:
    """Audit H17: freeze() activates the approved fallback snapshot in SHADOW,
    or the controller honestly reports learner-disabled when none exists."""

    def _trained_policy(self):
        from src.adaptation.policies.online_logreg import OnlineLogRegPolicy
        from src.adaptation.policy_base import Context, Outcome

        policy = OnlineLogRegPolicy(learner_id="ff_approved", learner_version="learner_0009")
        ctx = Context(signal_strength=0.7, expected_edge_frac=0.02)
        action = policy.decide(ctx)
        for pnl in [0.1, -0.05, 0.2, 0.15, -0.03]:
            policy.update(ctx, action, Outcome(realized_pnl_r=pnl, trade_taken=True))
        return policy

    def _ctrl(self, fallback_path):
        from src.adaptation.action_space import ActionBounds
        from src.adaptation.controller import LearnerController, LearnerMode
        from src.adaptation.policies.online_logreg import OnlineLogRegPolicy

        return LearnerController(
            policy=OnlineLogRegPolicy(),
            bounds=ActionBounds(),
            mode=LearnerMode.LIVE_BOUNDED,
            fallback_snapshot_path=fallback_path,
        )

    def test_freeze_with_snapshot_runs_fallback_in_shadow(self):
        from src.adaptation.policy_base import Context
        from src.adaptation.versioning import make_frozen_fallback

        with tempfile.TemporaryDirectory() as tmpdir:
            snap_dir = Path(tmpdir)
            approved = self._trained_policy()
            path = make_frozen_fallback(approved.snapshot(), snap_dir)
            ctrl = self._ctrl(path)

            ctrl.freeze(reason="test rollback")
            assert ctrl.is_frozen()
            assert ctrl.fallback_active
            assert not ctrl.learner_disabled
            # The fallback slot holds the APPROVED snapshot's state.
            assert ctrl.frozen_policy.learner_id == "ff_approved"

            # Decisions still flow: valid (not rejected), SHADOW, never applied.
            for _ in range(3):
                dec = ctrl.run(Context(signal_strength=0.8, expected_edge_frac=0.03))
                assert not dec.rejected, dec.rejection_reason
                assert dec.action is not None
                assert dec.action.mode == "SHADOW"  # never LIVE while frozen
                assert not dec.applied

    def test_freeze_without_snapshot_reports_disabled(self):
        from src.adaptation.policy_base import Context

        with tempfile.TemporaryDirectory() as tmpdir:
            ctrl = self._ctrl(Path(tmpdir) / "missing_fallback.pkl")
            ctrl.freeze(reason="breaker fired")
            assert ctrl.is_frozen()
            assert not ctrl.fallback_active  # never claims a fallback is active
            assert ctrl.learner_disabled

            dec = ctrl.run(Context(signal_strength=0.8))
            assert dec.rejected
            assert dec.action is None
            assert not dec.applied
            assert "DISABLED" in (dec.rejection_reason or "")
            assert "breaker fired" in (dec.rejection_reason or "")

    def test_freeze_records_reason(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ctrl = self._ctrl(Path(tmpdir) / "missing.pkl")
            ctrl.freeze(reason="divergence > 0.20")
            assert ctrl.frozen_reason == "divergence > 0.20"

    def test_frozen_fallback_not_trained_by_outcomes(self):
        """The fallback is the approved snapshot; record_outcome must not mutate it."""
        from src.adaptation.policy_base import Context, Outcome
        from src.adaptation.versioning import make_frozen_fallback

        with tempfile.TemporaryDirectory() as tmpdir:
            approved = self._trained_policy()
            path = make_frozen_fallback(approved.snapshot(), Path(tmpdir))
            ctrl = self._ctrl(path)
            ctrl.freeze(reason="test")
            ctx = Context(signal_strength=0.7)
            dec = ctrl.run(ctx)
            n_before = ctrl.frozen_policy._n_updates
            ctrl.record_outcome(ctx, dec, Outcome(realized_pnl_r=0.5, trade_taken=True))
            assert ctrl.frozen_policy._n_updates == n_before

    def test_revert_event_and_alert_are_honest(self):
        """RollbackGuard.revert: fallback_active truthfully reflects reality and
        the alert text says fallback-active vs learner-disabled accordingly."""
        from src.adaptation.rollback import RollbackGuard
        from src.adaptation.versioning import make_frozen_fallback
        from src.monitoring.alerts import AlertSink

        class _CaptureSink(AlertSink):
            def __init__(self):
                self.alerts = []

            def send(self, alert):
                self.alerts.append(alert)
                return True

        # Without a snapshot → disabled, honestly reported.
        with tempfile.TemporaryDirectory() as tmpdir:
            sink = _CaptureSink()
            ctrl = self._ctrl(Path(tmpdir) / "missing.pkl")
            guard = RollbackGuard(alert_sink=sink)
            event = guard.revert(ctrl, trigger="envelope_breaker", detail="daily loss")
            assert event.controller_frozen
            assert event.fallback_active is False
            assert "DISABLED" in sink.alerts[0].recommended_action
            assert "no frozen-fallback snapshot" in sink.alerts[0].recommended_action

        # With a snapshot → fallback restored into the slot and reported active.
        with tempfile.TemporaryDirectory() as tmpdir:
            sink = _CaptureSink()
            approved = self._trained_policy()
            path = make_frozen_fallback(approved.snapshot(), Path(tmpdir))
            ctrl = self._ctrl(path)
            guard = RollbackGuard(alert_sink=sink)
            event = guard.revert(ctrl, trigger="divergence", detail="live-vs-shadow")
            assert event.fallback_active is True
            assert ctrl.frozen_policy is not None
            assert ctrl.frozen_policy.learner_id == "ff_approved"
            assert "frozen-fallback policy active in SHADOW" in sink.alerts[0].recommended_action


# ======================================================================== #
# M32 — underperformance window over last N REALIZED decisions              #
# ======================================================================== #


class TestRollbackRealizedWindow:
    def _make(self, **kw):
        from src.adaptation.action_space import ActionBounds
        from src.adaptation.controller import LearnerController, LearnerMode
        from src.adaptation.policies.online_logreg import OnlineLogRegPolicy
        from src.adaptation.rollback import RollbackGuard

        ctrl = LearnerController(
            policy=OnlineLogRegPolicy(), bounds=ActionBounds(), mode=LearnerMode.LIVE_BOUNDED
        )
        return ctrl, RollbackGuard(**kw)

    def test_pending_outcomes_do_not_disable_trigger(self):
        """M32 regression: interleaved pending (None) outcomes must not stop the
        underperformance trigger from evaluating the last N realized decisions."""
        ctrl, guard = self._make(rollback_window=4, rollback_margin=0.01)
        for _ in range(4):
            guard.add_decision(projected_outcome=0.10, realized_outcome=0.05)
            guard.add_decision(projected_outcome=0.10, realized_outcome=None)  # pending
        event = guard.check(ctrl)
        assert event is not None
        assert event.trigger == "underperformance"
        assert ctrl.is_frozen()

    def test_not_triggered_until_enough_realized(self):
        ctrl, guard = self._make(rollback_window=4, rollback_margin=0.01)
        for _ in range(3):  # only 3 realized < window of 4
            guard.add_decision(projected_outcome=0.10, realized_outcome=0.05)
        for _ in range(10):
            guard.add_decision(projected_outcome=0.10, realized_outcome=None)
        assert guard.check(ctrl) is None
        assert not ctrl.is_frozen()


# ======================================================================== #
# M34 — scoring config wired into promotion eligibility                     #
# ======================================================================== #


class TestScorerConfigWiring:
    def _decisions(self, n, pnl=0.08, projected=0.06):
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

    def test_min_shadow_decisions_gates_eligibility(self):
        """M34 regression: a handful of outcomes must not reach eligibility when
        config demands more."""
        from src.adaptation.scorer import score_shadow_decisions

        decisions = self._decisions(8)
        result = score_shadow_decisions(
            decisions, baseline_mean=-0.01, min_shadow_decisions=50
        )
        assert not result.promotion_eligible
        assert "min_shadow_decisions" in result.note

        eligible = score_shadow_decisions(
            self._decisions(60), baseline_mean=-0.01, min_shadow_decisions=50
        )
        assert eligible.promotion_eligible

    def test_min_wf_folds_positive_enforced(self):
        from src.adaptation.scorer import score_shadow_decisions

        decisions = self._decisions(60)
        # n_folds=4 → 3 walk-forward folds, all beating baseline -0.01.
        result = score_shadow_decisions(
            decisions, baseline_mean=-0.01, n_folds=4, min_wf_folds_positive=3
        )
        assert result.folds_passed == 3
        assert result.promotion_eligible

        # Demand more positive folds than exist → not eligible.
        strict = score_shadow_decisions(
            decisions, baseline_mean=-0.01, n_folds=4, min_wf_folds_positive=4
        )
        assert not strict.promotion_eligible
        assert "min_wf_folds_positive" in strict.note

    def test_n_folds_1_does_not_crash(self):
        """M34: n_folds=1 must not raise ZeroDivisionError."""
        from src.adaptation.scorer import score_shadow_decisions

        result = score_shadow_decisions(self._decisions(20), n_folds=1, baseline_mean=-0.01)
        assert result.folds == []
        assert result.holdout_edge is not None


# ======================================================================== #
# M37 — online logreg feature normalization (Welford running stats)         #
# ======================================================================== #


class TestOnlineLogRegNormalization:
    def test_running_stats_updated_and_applied(self):
        from src.adaptation.policies.online_logreg import OnlineLogRegPolicy
        from src.adaptation.policy_base import Context, Outcome

        policy = OnlineLogRegPolicy()
        # Features spanning orders of magnitude (spread_bps ~1e1, edge ~1e-3).
        for i in range(20):
            ctx = Context(
                signal_strength=0.5 + 0.02 * i,
                expected_edge_frac=0.001 + 0.0001 * i,
                spread_bps=30.0 + i,
                slippage_est=0.0005,
                atr_pct=0.01,
                funding_z=0.5,
            )
            action = policy.decide(ctx)
            policy.update(ctx, action, Outcome(realized_pnl_r=0.1 if i % 2 else -0.1))
        assert policy._scaler.n == 20
        # Running mean tracks the big-magnitude feature (spread_bps index 2).
        assert policy._scaler.mean[2] == pytest.approx(30.0 + 19 / 2, rel=0.01)
        # Transform standardises: a sample at the mean maps to ~0 for that feature.
        mean_sample = list(policy._scaler.mean)
        transformed = policy._scaler.transform(mean_sample)
        assert all(abs(v) < 1e-9 for v in transformed)
        # decide() still yields valid actions after normalized training.
        dec = policy.decide(Context(signal_strength=0.9, spread_bps=31.0))
        assert dec.size_bucket in (0.0, 0.25, 0.5, 1.0)

    def test_snapshot_persists_running_stats(self):
        from src.adaptation.policies.online_logreg import OnlineLogRegPolicy
        from src.adaptation.policy_base import Context, Outcome

        policy = OnlineLogRegPolicy()
        ctx = Context(signal_strength=0.6, spread_bps=12.0)
        action = policy.decide(ctx)
        for pnl in [0.1, -0.2, 0.3]:
            policy.update(ctx, action, Outcome(realized_pnl_r=pnl))
        blob = policy.snapshot()

        restored = OnlineLogRegPolicy()
        restored.load(blob)
        assert restored._scaler.n == policy._scaler.n
        assert restored._scaler.mean == policy._scaler.mean
        assert restored._scaler.m2 == policy._scaler.m2

    def test_legacy_snapshot_without_stats_loads(self):
        """Snapshot compat: pre-M37 blobs (no norm_stats) load with fresh stats."""
        import pickle

        from src.adaptation.policies.online_logreg import OnlineLogRegPolicy

        donor = OnlineLogRegPolicy(learner_id="legacy", learner_version="learner_0001")
        legacy_blob = pickle.dumps(
            {
                "model": donor._model,
                "scaler": None,  # legacy unused StandardScaler slot
                "n_updates": 0,
                "learner_id": "legacy",
                "learner_version": "learner_0001",
                "feature_names": donor.feature_names,
            }
        )
        policy = OnlineLogRegPolicy()
        policy.load(legacy_blob)
        assert policy.learner_id == "legacy"
        assert policy._scaler.n == 0  # fresh stats, accumulate from zero

    def test_win_probability_only_when_trained(self):
        """M31: the 0.5 untrained placeholder is never emitted as a probability."""
        from src.adaptation.policies.online_logreg import OnlineLogRegPolicy
        from src.adaptation.policy_base import Context, Outcome

        policy = OnlineLogRegPolicy()
        ctx = Context(signal_strength=0.7)
        assert policy.decide(ctx).win_probability is None  # untrained
        action = policy.decide(ctx)
        for pnl in [0.1, -0.1, 0.2]:
            policy.update(ctx, action, Outcome(realized_pnl_r=pnl))
        trained_action = policy.decide(ctx)
        assert trained_action.win_probability is not None
        assert 0.0 <= trained_action.win_probability <= 1.0
        assert trained_action.projected_outcome_r is None  # no R-scale estimate


# ======================================================================== #
# M31 — bandit emits R-scale projection; store persists contract fields     #
# ======================================================================== #


class TestOutcomeProjectionContract:
    def test_bandit_projected_outcome_is_r_scale(self):
        from src.adaptation.policies.bandit import GaussianTSBandit
        from src.adaptation.policy_base import Context, Outcome

        bandit = GaussianTSBandit()
        ctx = Context(strategy_id="strat_A")
        first = bandit.decide(ctx)
        assert first.projected_outcome_r == pytest.approx(0.0)  # prior mean
        assert first.win_probability is None
        for _ in range(5):
            bandit.update(ctx, first, Outcome(realized_pnl_r=0.4, trade_taken=True))
        action = bandit.decide(ctx)
        assert action.projected_outcome_r == pytest.approx(bandit._arms["strat_A"].mu)
        assert action.projected_outcome_r > 0.0

    def test_store_serializes_contract_fields(self):
        from src.adaptation.action_space import BoundedAction
        from src.adaptation.store import get_memory_sink, reset_memory_sink, write_learner_log

        reset_memory_sink()
        action = BoundedAction(
            size_bucket=0.5,
            learner_id="m31",
            learner_version="v0",
            mode="SHADOW",
            rationale="contract",
            projected_outcome_r=0.12,
            win_probability=0.66,
        )
        write_learner_log(
            learner_id="m31",
            learner_version="v0",
            mode="SHADOW",
            symbol=None,
            context_features={},
            proposed_action=action,
            projected_outcome=0.12,
            realized_outcome=None,
            applied=False,
            clamped_fields=[],
            write_to_db=False,
        )
        entry = get_memory_sink().recent()[0]
        assert entry.proposed_action["projected_outcome_r"] == pytest.approx(0.12)
        assert entry.proposed_action["win_probability"] == pytest.approx(0.66)


# ======================================================================== #
# L30 — learner_log DB write failures are visible                           #
# ======================================================================== #


class TestStoreFailureVisibility:
    def test_db_write_failure_counted_and_health_degraded(self, monkeypatch, caplog):
        import logging

        import src.adaptation.store as store_mod
        from src.adaptation.action_space import BoundedAction

        store_mod.reset_memory_sink()
        store_mod.reset_db_write_failure_count()

        def _boom(*a, **kw):
            raise RuntimeError("db down")

        # Force the DB path to fail inside _write_to_db.
        import src.db.base as db_base

        monkeypatch.setattr(db_base, "session_scope", _boom)

        action = BoundedAction(
            size_bucket=0.5,
            learner_id="l30",
            learner_version="v0",
            mode="SHADOW",
            rationale="failure test",
        )
        with caplog.at_level(logging.ERROR, logger="src.adaptation.store"):
            entry = store_mod.write_learner_log(
                learner_id="l30",
                learner_version="v0",
                mode="SHADOW",
                symbol=None,
                context_features={},
                proposed_action=action,
                projected_outcome=0.0,
                realized_outcome=None,
                applied=False,
                clamped_fields=[],
                write_to_db=True,
            )
        assert entry is not None  # never blocks the decision path
        assert store_mod.get_db_write_failure_count() == 1
        assert any("learner_log DB write FAILED" in r.message for r in caplog.records)

        health = store_mod.learner_store_health()
        assert health.name == "learner_store"
        assert not health.healthy
        assert "1 learner_log DB write(s) failed" in health.detail

        store_mod.reset_db_write_failure_count()
        assert store_mod.learner_store_health().healthy


# ======================================================================== #
# L31 — bandit conjugate update behaves as (now) documented                 #
# ======================================================================== #


class TestBanditDocumentedBehaviour:
    def test_weights_rank_based_not_renormalized(self):
        from src.adaptation.policies.bandit import GaussianTSBandit
        from src.adaptation.policy_base import Context, Outcome

        bandit = GaussianTSBandit(w_min=0.0, w_max=2.0)
        for sid, r in (("a", 0.5), ("b", -0.5), ("c", 0.1)):
            ctx = Context(strategy_id=sid)
            act = bandit.decide(ctx)
            bandit.update(ctx, act, Outcome(realized_pnl_r=r, trade_taken=True))
        action = bandit.decide(Context(strategy_id="a"))
        weights = action.strategy_weights
        assert set(weights) == {"a", "b", "c"}
        # Rank-based: top arm gets w_max; weights don't sum to 1 (no renormalization).
        assert max(weights.values()) == pytest.approx(2.0)
        assert sum(weights.values()) != pytest.approx(1.0)
        # Even the negative-outcome arm keeps its (bounded) rank weight ≥ w_min.
        assert all(0.0 <= w <= 2.0 for w in weights.values())

    def test_conjugate_update_shrinks_variance_and_moves_mean(self):
        from src.adaptation.policies.bandit import GaussianTSBandit
        from src.adaptation.policy_base import Context, Outcome

        bandit = GaussianTSBandit()
        ctx = Context(strategy_id="s")
        act = bandit.decide(ctx)
        arm0_var = bandit._arms["s"].var
        bandit.update(ctx, act, Outcome(realized_pnl_r=1.0, trade_taken=True))
        arm = bandit._arms["s"]
        assert arm.var < arm0_var  # posterior variance shrinks
        assert 0.0 < arm.mu < 1.0  # mean moves toward the observation
        # Known-noise conjugate update with prior var 1, obs noise 1 → posterior 0.5.
        assert arm.var == pytest.approx(0.5)
        assert arm.mu == pytest.approx(0.5)
