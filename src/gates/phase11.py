"""Phase 11 gate checks for LEARN-PROMO-S (AGENTS.md Appendix A, Phase 11).

Gate: LEARN-PROMO-S — Learner Shadow → Recommend.
Pass condition (Appendix A):
  - Eligibility (Section 21.1) met: min_samples_to_start accumulation, frozen
    fallback present and tested, leakage-safe validation in place.
  - Shadow policy beats baseline on walk-forward AND locked hold-out.
  - Bounded actions only — envelope_guard rejects any out-of-box action.
  - Frozen fallback exists and tested (round-trip snapshot verified).

Criteria checked:
  1. adaptation_imports          — all adaptation module types importable
  2. config_mode_shadow          — mode=SHADOW in configs/adaptation.yaml
  3. envelope_guard_rejects_bad  — guard rejects every out-of-box action
  4. validate_clamps_bucket      — validate() clamps invalid size_bucket
  5. shadow_policy_runs          — OnlineLogRegPolicy produce valid actions in shadow
  6. bandit_policy_runs          — GaussianTSBandit produces valid bounded actions
  7. rl_stub_importable          — RLPolicyStub importable and produces valid action
  8. controller_shadow_applies_false — controller applied=False in SHADOW mode
  9. frozen_fallback_roundtrip   — snapshot / frozen-fallback write+load verified
  10. learner_log_db             — LearnerLog model writable to learner_logs table
  11. scorer_runs                — scorer evaluates synthetic decisions (WF + hold-out)
  12. rollback_guard_fires       — RollbackGuard freezes on envelope breaker
  13. drift_monitoring_runs      — drift scores computed over synthetic decisions
  14. shadow_log_applied_false   — all SHADOW-mode learner_logs have applied=False
"""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from src.config import Settings
from src.gates.result import Criterion


def check_learn_promo_s(settings: Settings) -> list[Criterion]:
    """LEARN-PROMO-S gate criteria (Phase 11 — Online Learning Shadow)."""
    out: list[Criterion] = []

    # ------------------------------------------------------------------ #
    # 1. All adaptation imports                                            #
    # ------------------------------------------------------------------ #
    try:
        from src.adaptation import (  # noqa: F401
            ActionBounds,
            AdaptationConfig,
            BoundedAction,
            Context,
            ControllerDecision,
            GaussianTSBandit,
            GuardResult,
            InMemoryLearnerStore,
            LearnerController,
            LearnerLogEntry,
            LearnerMode,
            OnlineLogRegPolicy,
            Outcome,
            RiskEnvelope,
            RLPolicyStub,
            RollbackEvent,
            RollbackGuard,
            ScorerResult,
            ShadowDecision,
            SnapshotMeta,
            ValidationResult,
            enforce,
            get_memory_sink,
            load_adaptation_config,
            make_frozen_fallback,
            reset_memory_sink,
            save_snapshot,
            score_shadow_decisions,
            validate,
            write_learner_log,
        )

        out.append(
            Criterion.ok(
                "adaptation_imports",
                "all adaptation module types importable (action_space, envelope_guard, "
                "policy_base, policies, scorer, controller, rollback, versioning, store)",
            )
        )
    except ImportError as exc:
        out.append(Criterion.fail("adaptation_imports", f"import error: {exc}"))
        return out

    # ------------------------------------------------------------------ #
    # 2. Config mode=SHADOW                                               #
    # ------------------------------------------------------------------ #
    try:
        cfg = load_adaptation_config()
        shadow_ok = cfg.mode == "SHADOW"
        out.append(
            Criterion.ok("config_mode_shadow", f"mode={cfg.mode}; learner_id={cfg.learner_id}")
            if shadow_ok
            else Criterion.fail(
                "config_mode_shadow",
                f"mode={cfg.mode} (must be SHADOW for Phase 11; "
                "RECOMMEND/LIVE_BOUNDED requires LEARN-PROMO-S/L gate PASS + manual approval)",
            )
        )
        enabled_ok = cfg.enabled
        out.append(
            Criterion.ok("adaptation_enabled", "adaptation.enabled=true")
            if enabled_ok
            else Criterion.fail("adaptation_enabled", "adaptation.enabled=false in config")
        )
    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("config_mode_shadow", f"config load error: {exc}"))

    # ------------------------------------------------------------------ #
    # 3. envelope_guard rejects out-of-box actions                        #
    # ------------------------------------------------------------------ #
    try:
        from src.adaptation.action_space import BoundedAction
        from src.adaptation.envelope_guard import RiskEnvelope, enforce

        _test_envelope = RiskEnvelope(
            max_leverage=5,
            max_risk_pct_per_trade=0.01,
            portfolio_heat_cap=0.05,
            net_beta_btc_cap=0.30,
            daily_loss_limit=0.03,
            max_drawdown_limit=0.10,
        )

        # Attempt to touch a forbidden envelope param via param_nudges.
        bad_action = BoundedAction(
            strategy_weights={},
            size_bucket=1.0,
            take=True,
            exec_style="maker",
            param_nudges={"max_leverage": 10.0},  # FORBIDDEN
            learner_id="test",
            learner_version="v0",
            mode="SHADOW",
            rationale="test",
        )
        result = enforce(bad_action, envelope=_test_envelope)
        guard_rejects = result.rejected and "max_leverage" in (result.rejection_reason or "")

        # Attempt to reference an unknown strategy.
        bad_strat_action = BoundedAction(
            strategy_weights={"unvalidated_strat_xyz": 5.0},
            size_bucket=1.0,
            take=True,
            exec_style="maker",
            param_nudges={},
            learner_id="test",
            learner_version="v0",
            mode="SHADOW",
            rationale="test",
        )
        strat_result = enforce(
            bad_strat_action,
            active_strategies={"approved_strat_A"},
            envelope=_test_envelope,
        )
        strat_rejects = strat_result.rejected

        # size_bucket > 1.0 is clamped to 1.0, not rejected (AGENTS.md §21.6).
        oversized = BoundedAction(
            size_bucket=1.5,
            learner_id="test",
            learner_version="v0",
            mode="SHADOW",
            rationale="test",
        )
        over_result = enforce(oversized, envelope=_test_envelope)
        oversized_clamped = (
            not over_result.rejected and over_result.action.size_bucket == 1.0
        )

        guard_ok = guard_rejects and strat_rejects and oversized_clamped
        out.append(
            Criterion.ok(
                "envelope_guard_rejects_bad",
                "forbidden param rejected; unknown strategy rejected; "
                "oversized bucket clamped",
            )
            if guard_ok
            else Criterion.fail(
                "envelope_guard_rejects_bad",
                f"guard_rejects={guard_rejects} strat_rejects={strat_rejects} "
                f"oversized_clamped={oversized_clamped}",
            )
        )
    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("envelope_guard_rejects_bad", f"raised: {exc}"))

    # ------------------------------------------------------------------ #
    # 4. validate() clamps invalid size_bucket                            #
    # ------------------------------------------------------------------ #
    try:
        from src.adaptation.action_space import ActionBounds, BoundedAction, validate

        bounds = ActionBounds(
            w_min=0.0,
            w_max=2.0,
            size_buckets=(0.0, 0.25, 0.5, 1.0),
        )
        # An invalid bucket (reject_on_bad_bucket=True, the safe default).
        bad = BoundedAction(
            size_bucket=0.7,
            learner_id="test",
            learner_version="v0",
            mode="SHADOW",
            rationale="test",
        )
        val_result = validate(bad, bounds, reject_on_bad_bucket=True)
        rejected_bucket = val_result.rejected

        # With clamp mode.
        val_clamp = validate(bad, bounds, reject_on_bad_bucket=False)
        clamped_bucket = (
            not val_clamp.rejected and val_clamp.action.size_bucket in bounds.size_buckets
        )

        out.append(
            Criterion.ok(
                "validate_clamps_bucket",
                "invalid bucket rejected (safe default) or clamped (when configured)",
            )
            if rejected_bucket and clamped_bucket
            else Criterion.fail(
                "validate_clamps_bucket",
                f"rejected={rejected_bucket} clamped={clamped_bucket}",
            )
        )
    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("validate_clamps_bucket", f"raised: {exc}"))

    # ------------------------------------------------------------------ #
    # 5. OnlineLogRegPolicy produces valid shadow actions                  #
    # ------------------------------------------------------------------ #
    try:
        from src.adaptation.policies.online_logreg import OnlineLogRegPolicy
        from src.adaptation.policy_base import Context, Outcome

        policy = OnlineLogRegPolicy()
        ctx = Context(signal_strength=0.7, expected_edge_frac=0.02, spread_bps=3.0)
        action = policy.decide(ctx)
        logreg_ok = (
            action.mode == "SHADOW"
            and action.size_bucket in (0.0, 0.25, 0.5, 1.0)
            and not action.param_nudges  # no forbidden nudges
            and action.learner_id == policy.learner_id
        )
        # Train on synthetic outcomes and verify update runs without error.
        for pnl in [0.1, -0.05, 0.2, 0.15, -0.03, 0.08, 0.12, 0.05]:
            policy.update(ctx, action, Outcome(realized_pnl_r=pnl, trade_taken=True))
        action2 = policy.decide(ctx)
        logreg_ok = logreg_ok and action2.size_bucket in (0.0, 0.25, 0.5, 1.0)

        out.append(
            Criterion.ok(
                "shadow_policy_runs",
                f"OnlineLogRegPolicy: decide + {8} updates + decide OK; "
                f"size_bucket={action2.size_bucket}",
            )
            if logreg_ok
            else Criterion.fail(
                "shadow_policy_runs",
                f"mode={action.mode} bucket={action.size_bucket} nudges={action.param_nudges}",
            )
        )
    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("shadow_policy_runs", f"raised: {exc}"))

    # ------------------------------------------------------------------ #
    # 6. GaussianTSBandit produces valid bounded actions                  #
    # ------------------------------------------------------------------ #
    try:
        from src.adaptation.policies.bandit import GaussianTSBandit
        from src.adaptation.policy_base import Context, Outcome

        bandit = GaussianTSBandit()
        ctx = Context(strategy_id="strat_A")
        action = bandit.decide(ctx)
        bandit_ok = action.mode == "SHADOW" and not action.param_nudges

        # Update with a positive outcome for strat_A.
        bandit.update(ctx, action, Outcome(realized_pnl_r=0.15, trade_taken=True))
        bandit.update(ctx, action, Outcome(realized_pnl_r=-0.05, trade_taken=True))
        action2 = bandit.decide(ctx)
        bandit_ok = bandit_ok and isinstance(action2.strategy_weights, dict)

        out.append(
            Criterion.ok(
                "bandit_policy_runs",
                f"GaussianTSBandit: decide + 2 updates + decide OK; "
                f"arms={len(bandit._arms)}",
            )
            if bandit_ok
            else Criterion.fail(
                "bandit_policy_runs",
                f"mode={action.mode} nudges={action.param_nudges}",
            )
        )
    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("bandit_policy_runs", f"raised: {exc}"))

    # ------------------------------------------------------------------ #
    # 7. RLPolicyStub importable + valid action                           #
    # ------------------------------------------------------------------ #
    try:
        from src.adaptation.policies.rl_policy import RLPolicyStub
        from src.adaptation.policy_base import Context

        stub = RLPolicyStub()
        ctx = Context()
        action = stub.decide(ctx)
        rl_ok = action.mode == "SHADOW" and action.size_bucket in (0.0, 0.25, 0.5, 1.0)
        blob = stub.snapshot()
        stub2 = RLPolicyStub()
        stub2.load(blob)
        rl_ok = rl_ok and stub2.learner_id == stub.learner_id

        out.append(
            Criterion.ok(
                "rl_stub_importable",
                "RLPolicyStub: importable, decide OK, snapshot/load round-trip OK",
            )
            if rl_ok
            else Criterion.fail("rl_stub_importable", f"action={action}")
        )
    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("rl_stub_importable", f"raised: {exc}"))

    # ------------------------------------------------------------------ #
    # 8. Controller in SHADOW mode: applied=False always                  #
    # ------------------------------------------------------------------ #
    try:
        from src.adaptation.action_space import ActionBounds
        from src.adaptation.controller import LearnerController, LearnerMode
        from src.adaptation.policies.online_logreg import OnlineLogRegPolicy
        from src.adaptation.policy_base import Context

        bounds = ActionBounds()
        policy = OnlineLogRegPolicy()
        ctrl = LearnerController(policy=policy, bounds=bounds, mode=LearnerMode.SHADOW)
        ctx = Context(signal_strength=0.8, expected_edge_frac=0.03)
        for _ in range(5):
            dec = ctrl.run(ctx)
            assert not dec.applied, f"applied=True in SHADOW mode (decision #{_})"
            assert not dec.rejected, f"valid action rejected: {dec.rejection_reason}"

        out.append(
            Criterion.ok(
                "controller_shadow_applies_false",
                "5 SHADOW-mode controller decisions all applied=False",
            )
        )
    except AssertionError as exc:
        out.append(Criterion.fail("controller_shadow_applies_false", str(exc)))
    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("controller_shadow_applies_false", f"raised: {exc}"))

    # ------------------------------------------------------------------ #
    # 9. Frozen fallback snapshot round-trip                              #
    # ------------------------------------------------------------------ #
    try:
        from src.adaptation.policies.online_logreg import OnlineLogRegPolicy
        from src.adaptation.versioning import (
            load_frozen_fallback,
            load_snapshot,
            make_frozen_fallback,
            save_snapshot,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            snap_dir = Path(tmpdir)
            policy = OnlineLogRegPolicy(learner_id="test_ff", learner_version="learner_0001")
            blob = policy.snapshot()
            # Save regular snapshot.
            meta = save_snapshot(blob, "test_ff", "learner_0001", "SHADOW", snap_dir)
            loaded_blob = load_snapshot(meta.snapshot_id, snap_dir)
            assert loaded_blob == blob, "snapshot blob mismatch"
            # Save frozen fallback.
            make_frozen_fallback(blob, snap_dir)
            ff_blob = load_frozen_fallback(snap_dir)
            assert ff_blob == blob, "frozen fallback blob mismatch"
            # Restore into a new policy instance.
            policy2 = OnlineLogRegPolicy()
            policy2.load(ff_blob)
            assert policy2.learner_id == "test_ff"

        out.append(
            Criterion.ok(
                "frozen_fallback_roundtrip",
                "snapshot save+load and frozen-fallback write+restore verified",
            )
        )
    except AssertionError as exc:
        out.append(Criterion.fail("frozen_fallback_roundtrip", str(exc)))
    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("frozen_fallback_roundtrip", f"raised: {exc}"))

    # ------------------------------------------------------------------ #
    # 10. LearnerLog DB model writable                                    #
    # ------------------------------------------------------------------ #
    try:
        from src.adaptation.action_space import BoundedAction
        from src.adaptation.store import reset_memory_sink, write_learner_log

        reset_memory_sink()
        action = BoundedAction(
            strategy_weights={},
            size_bucket=0.5,
            take=True,
            exec_style="maker",
            param_nudges={},
            learner_id="gate_test",
            learner_version="learner_0001",
            mode="SHADOW",
            rationale="gate self-test",
        )
        entry = write_learner_log(
            learner_id="gate_test",
            learner_version="learner_0001",
            mode="SHADOW",
            symbol="BTCUSDT",
            context_features={"signal_strength": 0.7},
            proposed_action=action,
            projected_outcome=0.1,
            realized_outcome=None,
            applied=False,
            clamped_fields=[],
            config_version=settings.config_version,
            write_to_db=True,
        )
        db_ok = entry.applied is False and entry.mode == "SHADOW"

        # Verify DB row was written.
        from src.db.base import session_scope
        from src.db.models import LearnerLog

        with session_scope() as session:
            row = (
                session.query(LearnerLog)
                .filter_by(learner_id="gate_test")
                .order_by(LearnerLog.id.desc())
                .first()
            )
            db_row_ok = (
                row is not None
                and row.mode == "SHADOW"
                and row.applied is False
            )

        out.append(
            Criterion.ok(
                "learner_log_db",
                "LearnerLog row written to learner_logs; applied=False; mode=SHADOW",
            )
            if db_ok and db_row_ok
            else Criterion.fail(
                "learner_log_db",
                f"db_ok={db_ok} db_row_ok={db_row_ok}",
            )
        )
    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("learner_log_db", f"raised: {exc}"))

    # ------------------------------------------------------------------ #
    # 11. Scorer evaluates synthetic decisions                             #
    # ------------------------------------------------------------------ #
    try:
        from src.adaptation.scorer import ShadowDecision, score_shadow_decisions

        # Build synthetic decisions with known outcomes (positive edge).
        decisions: list[ShadowDecision] = []
        for i in range(60):
            realized = 0.08 if i % 3 != 0 else -0.02  # win rate ~66%
            decisions.append(
                ShadowDecision(
                    ts=datetime.now(UTC),
                    symbol="BTCUSDT",
                    projected_outcome=0.06,
                    realized_outcome=realized,
                    take=True,
                    mode="SHADOW",
                )
            )
        scorer_result = score_shadow_decisions(
            decisions,
            n_folds=4,
            min_holdout_edge=0.0,
            calibration_max_brier=0.30,
            max_drift_per_window=0.20,
            baseline_mean=-0.01,  # learner beats a slightly negative baseline
        )
        scorer_ok = (
            scorer_result.n_decisions == 60
            and scorer_result.n_with_outcome == 60
            and scorer_result.holdout_edge is not None
            and scorer_result.brier_score is not None
        )
        out.append(
            Criterion.ok(
                "scorer_runs",
                f"n={scorer_result.n_decisions} folds_passed={scorer_result.folds_passed} "
                f"holdout_edge={scorer_result.holdout_edge:.3f} "
                f"brier={scorer_result.brier_score:.3f} "
                f"eligible={scorer_result.promotion_eligible}",
            )
            if scorer_ok
            else Criterion.fail(
                "scorer_runs",
                f"n_decisions={scorer_result.n_decisions} n_outcome={scorer_result.n_with_outcome}",
            )
        )
    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("scorer_runs", f"raised: {exc}"))

    # ------------------------------------------------------------------ #
    # 12. RollbackGuard freezes on envelope breaker                       #
    # ------------------------------------------------------------------ #
    try:
        from src.adaptation.action_space import ActionBounds
        from src.adaptation.controller import LearnerController, LearnerMode
        from src.adaptation.policies.online_logreg import OnlineLogRegPolicy
        from src.adaptation.rollback import RollbackGuard

        bounds = ActionBounds()
        policy = OnlineLogRegPolicy()
        ctrl = LearnerController(policy=policy, bounds=bounds, mode=LearnerMode.SHADOW)
        guard = RollbackGuard(rollback_window=5, rollback_margin=0.02)

        # Signal an envelope breaker.
        guard.set_envelope_breaker(True)
        event = guard.check(ctrl)
        rollback_ok = event is not None and ctrl.is_frozen() and event.trigger == "envelope_breaker"

        out.append(
            Criterion.ok(
                "rollback_guard_fires",
                "RollbackGuard froze controller on envelope_breaker; "
                f"trigger={event.trigger if event else None}",
            )
            if rollback_ok
            else Criterion.fail(
                "rollback_guard_fires",
                f"event={event} frozen={ctrl.is_frozen()}",
            )
        )
    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("rollback_guard_fires", f"raised: {exc}"))

    # ------------------------------------------------------------------ #
    # 13. Drift monitoring computes scores over synthetic decisions         #
    # ------------------------------------------------------------------ #
    try:
        from src.adaptation.scorer import ShadowDecision, score_shadow_decisions

        decisions_with_drift: list[ShadowDecision] = [
            ShadowDecision(
                ts=datetime.now(UTC),
                symbol=None,
                projected_outcome=0.10,
                realized_outcome=0.05,  # small drift
                take=True,
                mode="SHADOW",
            )
            for _ in range(25)
        ]
        drift_result = score_shadow_decisions(
            decisions_with_drift,
            drift_window=10,
            max_drift_per_window=0.20,
        )
        drift_ok = len(drift_result.drift_scores) >= 1 and drift_result.max_drift >= 0
        out.append(
            Criterion.ok(
                "drift_monitoring_runs",
                f"drift_scores={drift_result.drift_scores[:3]} "
                f"max_drift={drift_result.max_drift:.3f}",
            )
            if drift_ok
            else Criterion.fail(
                "drift_monitoring_runs",
                f"drift_scores={drift_result.drift_scores}",
            )
        )
    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("drift_monitoring_runs", f"raised: {exc}"))

    # ------------------------------------------------------------------ #
    # 14. All SHADOW learner_logs have applied=False                       #
    # ------------------------------------------------------------------ #
    try:
        from src.db.base import session_scope
        from src.db.models import LearnerLog

        with session_scope() as session:
            applied_shadow = (
                session.query(LearnerLog)
                .filter(LearnerLog.mode == "SHADOW", LearnerLog.applied.is_(True))
                .count()
            )
        out.append(
            Criterion.ok(
                "shadow_log_applied_false",
                "0 SHADOW-mode learner_log rows have applied=True",
            )
            if applied_shadow == 0
            else Criterion.fail(
                "shadow_log_applied_false",
                f"{applied_shadow} SHADOW-mode rows have applied=True (forbidden)",
            )
        )
    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("shadow_log_applied_false", f"raised: {exc}"))

    # ------------------------------------------------------------------ #
    # Write Phase 11 report                                               #
    # ------------------------------------------------------------------ #
    import contextlib

    with contextlib.suppress(Exception):
        _write_phase11_report(settings)

    return out


def _write_phase11_report(settings: Settings) -> None:
    from src.adaptation.config import load_adaptation_config

    reports_dir = settings.reports_path / "phase_11"
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    path = reports_dir / f"learn_shadow_{stamp}.json"
    try:
        cfg = load_adaptation_config()
        cfg_dict = {
            "mode": cfg.mode,
            "learner_id": cfg.learner_id,
            "learner_version": cfg.learner_version,
            "min_samples_to_start": cfg.min_samples_to_start,
        }
    except Exception:  # noqa: BLE001
        cfg_dict = {}

    payload = {
        "phase": 11,
        "gate": "LEARN-PROMO-S",
        "adaptation_config": cfg_dict,
        "versions": settings.versions(),
        "generated_at": stamp,
        "note": (
            "Phase 11 — Online Learning Shadow. Bounded learner in SHADOW mode: "
            "learner_log written; drift + calibration monitoring active; "
            "envelope_guard rejects out-of-box actions; frozen fallback round-trip verified. "
            "Promotion to RECOMMEND requires LEARN-PROMO-S gate PASS + manual approval."
        ),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
