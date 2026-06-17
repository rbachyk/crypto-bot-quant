"""Phase 9 gate check: ML-PROMO (AGENTS.md Appendix A, Phase 9).

ML-PROMO — ML Promotion (Phase 9–10):
  Pass: in shadow the model improves expectancy net of costs; preserves/raises
  profit factor; preserves/reduces max DD; does not remove most best trades;
  reduces tail risk; stable across folds/symbols/regimes/OOS; calibrated;
  explainable; no leakage; manually reviewed.

The check uses the deterministic reference dataset from :mod:`src.ml.labels`
(:func:`~src.ml.labels.build_reference_dataset`) — this is the Phase 9
"synthetic supervised test" that proves the full pipeline (feature construction
→ model training → shadow logging → offline scoring) works end-to-end before
real paper-trade outcomes accumulate.  The synthetic dataset is designed so a
correctly-trained meta-labeler beats the always-take baseline.

Infrastructure criteria (must PASS):
  * ml_shadow_imports — all components importable
  * ml_feature_matrix — feature matrix builds from reference candidates
  * ml_models_train — all 5 models train without error
  * ml_shadow_log_writes — shadow predictions logged with applied=False
  * ml_no_live_influence — mode=SHADOW in every log entry
  * ml_registry_records — model artifacts registered
  * ml_leakage_check — noise dataset yields ~0 expectancy improvement

Performance criteria (evaluated per kill-criteria; gate PASS requires all):
  * ml_expectancy_improves — meta-labeler beats always-take baseline
  * ml_profit_factor_preserved — PF ratio ≥ 1.0 (or infinite)
  * ml_tail_risk_reduced — worst-trade ratio ≤ 1.0
  * ml_best_trades_preserved — ≤ 20% of top-10 trades removed

Manual review is required for promotion; this gate verifies the shadow
infrastructure is in place and initial metrics are acceptable.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from src.config import Settings
from src.gates.result import Criterion

if TYPE_CHECKING:
    from src.ml.config import MLConfig
    from src.ml.scorer import ShadowScorerResult


def check_ml_promo(settings: Settings) -> list[Criterion]:  # noqa: ARG001
    """ML-PROMO gate check (Appendix A, Phase 9)."""
    out: list[Criterion] = []

    # ------------------------------------------------------------------ #
    # 1. Component imports                                                  #
    # ------------------------------------------------------------------ #
    try:
        from src.ml import MLRegistry, ModelArtifact, ShadowPredictor, ShadowScorer
        from src.ml.config import load_ml_config
        from src.ml.features import FEATURE_NAMES, build_feature_matrix
        from src.ml.labels import (
            LabeledSample,
            baseline_expectancy,
            build_reference_dataset,
            filtered_expectancy,
            train_test_split,
        )
        from src.ml.models import (  # noqa: F401
            ExecQualityModel,
            MetaLabeler,
            RegimeClassifier,
            StrategySelector,
            SymbolRanker,
        )

        out.append(Criterion.ok("ml_shadow_imports", "all ML shadow components importable"))
    except ImportError as exc:
        out.append(Criterion.fail("ml_shadow_imports", f"import error: {exc}"))
        return out

    ml_cfg = load_ml_config()

    # ------------------------------------------------------------------ #
    # 2. Build reference dataset + feature matrix                          #
    # ------------------------------------------------------------------ #
    try:
        all_samples = build_reference_dataset(n_good=40, n_bad=30, n_neutral=30, seed=42)
        candidates = [s.candidate for s in all_samples]
        X_all = build_feature_matrix(candidates)
        out.append(
            Criterion.ok(
                "ml_feature_matrix",
                f"{len(X_all)} rows × {len(FEATURE_NAMES)} features from reference dataset",
            )
            if X_all and len(X_all[0]) == len(FEATURE_NAMES)
            else Criterion.fail("ml_feature_matrix", "feature matrix empty or wrong width")
        )
    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("ml_feature_matrix", f"feature build raised: {exc}"))
        return out

    # ------------------------------------------------------------------ #
    # 3. Train all five shadow models                                       #
    # ------------------------------------------------------------------ #
    train_samples, test_samples = train_test_split(all_samples, test_fraction=0.25, seed=42)
    predictor = ShadowPredictor.from_config(ml_cfg)

    try:
        all_metrics = predictor.train(train_samples)
        expected_types = {
            "meta_labeler",
            "regime_classifier",
            "exec_quality",
            "strategy_selector",
            "symbol_ranker",
        }
        missing = expected_types - set(all_metrics.keys())
        out.append(
            Criterion.ok(
                "ml_models_train",
                f"all 5 models trained; "
                f"meta_labeler accuracy={all_metrics.get('meta_labeler', {}).get('accuracy', '?')}",
            )
            if not missing
            else Criterion.fail("ml_models_train", f"models not trained: {sorted(missing)}")
        )
    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("ml_models_train", f"training raised: {exc}"))
        return out

    # ------------------------------------------------------------------ #
    # 4. Shadow logging (applied=False; mode=SHADOW)                       #
    # ------------------------------------------------------------------ #
    try:
        result = predictor.run(
            candidates[:20],  # use a small subset to keep gate fast
            settings=settings,
            write_to_db=True,
        )
        n_logged = len(result.shadow_log_ids)
        out.append(
            Criterion.ok(
                "ml_shadow_log_writes",
                f"{n_logged} shadow log entries written, applied=False on all",
            )
            if n_logged > 0 and not result.applied
            else Criterion.fail(
                "ml_shadow_log_writes",
                f"n_logged={n_logged} applied={result.applied} (expected n>0 and applied=False)",
            )
        )
    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("ml_shadow_log_writes", f"shadow run raised: {exc}"))

    # Verify mode=SHADOW in the DB entries written above.
    try:
        from src.db.base import session_scope
        from src.db.models import ShadowLog

        with session_scope() as session:
            live_entries = session.query(ShadowLog).filter(ShadowLog.applied.is_(True)).count()
            shadow_entries = session.query(ShadowLog).filter(ShadowLog.mode == "SHADOW").count()
        out.append(
            Criterion.ok(
                "ml_no_live_influence",
                f"{shadow_entries} SHADOW entries; 0 applied entries",
            )
            if live_entries == 0
            else Criterion.fail(
                "ml_no_live_influence",
                f"{live_entries} shadow_log entries have applied=True (no live influence allowed)",
            )
        )
    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("ml_no_live_influence", f"DB check raised: {exc}"))

    # ------------------------------------------------------------------ #
    # 5. Model registry records versioned artifacts                        #
    # ------------------------------------------------------------------ #
    try:
        registry = MLRegistry(settings.artifact_path)
        registered: list[str] = []
        for model in predictor.all_models:
            artifact = ModelArtifact(
                model_id=model.model_id,
                model_version=ml_cfg.model_version,
                model_type=model.model_type,
                ml_stage=ml_cfg.ml_stage,
                promotion_status="shadow",
                label_definition={"type": "binary", "source": "paper_trade_outcome"},
                performance_metrics=model.performance_report(),
                known_failure_modes=[
                    "insufficient_history: requires ≥30 real paper trades",
                    "covariate_shift: live features may differ from synthetic training data",
                    "concept_drift: market regime changes may degrade model",
                ],
                notes=(
                    "Phase 9 — shadow only; promotion requires ML-PROMO gate PASS + manual review"
                ),
            )
            registry.register(model, artifact, write_db=True)
            registered.append(model.model_id)

        out.append(
            Criterion.ok(
                "ml_registry_records",
                f"{len(registered)} model artifacts registered: {registered}",
            )
        )
    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("ml_registry_records", f"registry raised: {exc}"))

    # ------------------------------------------------------------------ #
    # 6. Leakage check — noise dataset → ≈0 expectancy improvement         #
    # ------------------------------------------------------------------ #
    try:
        from src.ml.labels import build_reference_dataset, filtered_expectancy, train_test_split

        # Build a noise dataset by shuffling labels randomly.
        noise_samples = build_reference_dataset(seed=42)
        rng = random.Random(99)
        shuffled_labels = [s.label for s in noise_samples]
        rng.shuffle(shuffled_labels)
        from src.ml.labels import LabeledSample

        noise_samples_shuffled = [
            LabeledSample(
                candidate=s.candidate,
                label=lbl,
                realized_pnl=s.realized_pnl,
                hold_bars=s.hold_bars,
            )
            for s, lbl in zip(noise_samples, shuffled_labels, strict=False)
        ]
        noise_train, noise_test = train_test_split(noise_samples_shuffled, seed=99)

        noise_predictor = ShadowPredictor.from_config(ml_cfg)
        noise_predictor.train(noise_train)
        noise_result = noise_predictor.run(
            [s.candidate for s in noise_test],
            settings=settings,
            write_to_db=False,  # noise run: no DB writes
        )
        noise_preds = [b.meta_label.label if b.meta_label else 1 for b in noise_result.bundles]
        noise_baseline = baseline_expectancy(noise_test)
        noise_filtered = filtered_expectancy(noise_test, noise_preds)
        noise_delta = noise_filtered - noise_baseline
        # Leakage only manifests as positive improvement on noise data.
        # A negative delta means the model avoided trades on shuffled labels — expected
        # noise behaviour, not leakage. Only positive inflation > 0.3R is suspicious.
        leakage_ok = noise_delta <= 0.3
        out.append(
            Criterion.ok(
                "ml_leakage_check",
                f"noise improvement={noise_delta:.4f}R (≤ 0.30 → no systematic leakage)",
            )
            if leakage_ok
            else Criterion.fail(
                "ml_leakage_check",
                f"noise improvement={noise_delta:.4f}R exceeds 0.30 → possible leakage",
            )
        )
    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("ml_leakage_check", f"leakage check raised: {exc}"))

    # ------------------------------------------------------------------ #
    # 7. Performance criteria (test split)                                 #
    # ------------------------------------------------------------------ #
    try:
        test_result = predictor.run(
            [s.candidate for s in test_samples],
            settings=settings,
            write_to_db=False,
        )
        test_preds = [b.meta_label.label if b.meta_label else 1 for b in test_result.bundles]
        kc = ml_cfg.kill_criteria
        scorer = ShadowScorer(
            min_improvement=kc.min_improvement_over_baseline,
            min_pf_ratio=kc.min_profit_factor_ratio,
            max_tail_loss_ratio=kc.max_tail_loss_ratio,
            max_best_removed_pct=kc.max_best_trades_removed_pct,
        )
        score = scorer.score(test_samples, test_preds)

        # Expectancy improvement.
        out.append(
            Criterion.ok(
                "ml_expectancy_improves",
                f"model expectancy={score.model_expectancy:.4f}R "
                f"vs baseline={score.baseline_expectancy:.4f}R "
                f"(improvement={score.expectancy_improvement:+.4f}R)",
            )
            if score.expectancy_improvement >= kc.min_improvement_over_baseline
            else Criterion.fail(
                "ml_expectancy_improves",
                f"no improvement: model={score.model_expectancy:.4f}R "
                f"baseline={score.baseline_expectancy:.4f}R "
                f"delta={score.expectancy_improvement:+.4f}R",
            )
        )

        # Profit factor preserved.
        out.append(
            Criterion.ok(
                "ml_profit_factor_preserved",
                f"model PF={score.model_profit_factor:.3f} "
                f"baseline PF={score.baseline_profit_factor:.3f} "
                f"ratio={score.profit_factor_ratio:.3f}",
            )
            if score.profit_factor_ratio >= kc.min_profit_factor_ratio
            else Criterion.fail(
                "ml_profit_factor_preserved",
                f"PF degraded: model={score.model_profit_factor:.3f} "
                f"baseline={score.baseline_profit_factor:.3f} "
                f"ratio={score.profit_factor_ratio:.3f}",
            )
        )

        # Tail risk not worsened.
        out.append(
            Criterion.ok(
                "ml_tail_risk_reduced",
                f"model worst={score.model_worst_trade:.4f}R "
                f"baseline worst={score.baseline_worst_trade:.4f}R "
                f"ratio={score.tail_loss_ratio:.3f}",
            )
            if score.tail_loss_ratio <= kc.max_tail_loss_ratio
            else Criterion.fail(
                "ml_tail_risk_reduced",
                f"tail risk worsened: model worst={score.model_worst_trade:.4f}R "
                f"ratio={score.tail_loss_ratio:.3f} > {kc.max_tail_loss_ratio}",
            )
        )

        # Best trades preserved.
        out.append(
            Criterion.ok(
                "ml_best_trades_preserved",
                f"{score.best_trades_removed_pct:.1%} of top trades removed "
                f"(≤ {kc.max_best_trades_removed_pct:.0%} allowed)",
            )
            if score.best_trades_removed_pct <= kc.max_best_trades_removed_pct
            else Criterion.fail(
                "ml_best_trades_preserved",
                f"too many top trades removed: {score.best_trades_removed_pct:.1%} "
                f"> {kc.max_best_trades_removed_pct:.0%}",
            )
        )

        # Write a human-readable scoring report.
        _write_ml_report(settings, ml_cfg, all_metrics, score)

    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("ml_expectancy_improves", f"scoring raised: {exc}"))

    return out


def _write_ml_report(
    settings: Settings,
    ml_cfg: MLConfig,
    train_metrics: dict,
    score: ShadowScorerResult,
) -> None:
    import json
    from datetime import UTC, datetime

    reports_dir = settings.reports_path / "ml_shadow"
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    path = reports_dir / f"ml_shadow_{stamp}.json"
    payload = {
        "model_version": ml_cfg.model_version,
        "ml_stage": ml_cfg.ml_stage,
        "train_metrics": train_metrics,
        "scoring": score.to_dict(),
        "versions": settings.versions(),
        "generated_at": stamp,
        "note": (
            "Phase 9 shadow ML gate check using synthetic reference dataset. "
            "Performance criteria validated on 25%% held-out test split. "
            "Manual review required before promotion to Stage 3+."
        ),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
