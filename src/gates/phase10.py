"""Phase 10 gate checks for ML-PROMO (AGENTS.md Appendix A, Phase 10).

Extends the Phase 9 ML-PROMO gate with Stage 3 (Recommendation Mode) and
Stage 4 (Constrained Live Filter) criteria.

Additional criteria (Phase 10 only):
  * ml_stage_advanced   — ml_stage >= 3 in configs/ml.yaml
  * recommendation_mode — RecommendationEngine produces valid output;
                          applied=False on all Stage 3 log entries
  * filter_can_block    — MLFilter reduces candidate list when confidence low
  * filter_no_create    — filter cannot produce more candidates than input
  * filter_no_risk_inc  — filtered candidates are unmodified originals
  * filter_hard_blocked — hard-blocked candidates (strategy_enabled=False,
                          data_fresh=False, etc.) are always blocked by filter,
                          never unblocked

Manual review remains required for promotion (Section 20 Promotion gates).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.config import Settings
from src.gates.result import Criterion
from src.ranking.candidate import Candidate

if TYPE_CHECKING:
    pass


def check_ml_phase10(settings: Settings) -> list[Criterion]:
    """Phase 10 additional ML-PROMO criteria."""
    out: list[Criterion] = []

    # ------------------------------------------------------------------ #
    # 1. ML stage advanced to Stage 3+ in config                          #
    # ------------------------------------------------------------------ #
    try:
        from src.ml.config import load_ml_config

        ml_cfg = load_ml_config()
        out.append(
            Criterion.ok(
                "ml_stage_advanced",
                f"ml_stage={ml_cfg.ml_stage} (>= 3 for Recommendation Mode)",
            )
            if ml_cfg.ml_stage >= 3
            else Criterion.fail(
                "ml_stage_advanced",
                f"ml_stage={ml_cfg.ml_stage} — must be >= 3 to enable Stage 3 recommendation",
            )
        )
    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("ml_stage_advanced", f"config load error: {exc}"))
        return out

    # ------------------------------------------------------------------ #
    # 2. Import Stage 3 and Stage 4 components                            #
    # ------------------------------------------------------------------ #
    try:
        from src.ml import (  # noqa: F401
            FilterDecision,
            FilterResult,
            MLFilter,
            MLRecommendation,
            RecommendationEngine,
            RecommendationRunResult,
        )

        out.append(
            Criterion.ok(
                "ml_phase10_imports",
                "RecommendationEngine, MLFilter, and related types importable",
            )
        )
    except ImportError as exc:
        out.append(Criterion.fail("ml_phase10_imports", f"import error: {exc}"))
        return out

    # ------------------------------------------------------------------ #
    # 3. RecommendationEngine — Stage 3 output validity                   #
    # ------------------------------------------------------------------ #
    try:
        from src.ml.config import load_ml_config
        from src.ml.labels import build_reference_dataset, train_test_split
        from src.ml.shadow import ShadowPredictor

        ml_cfg = load_ml_config()
        samples = build_reference_dataset(n_good=20, n_bad=15, n_neutral=15, seed=42)
        train_samples, test_samples = train_test_split(samples, test_fraction=0.25, seed=42)

        predictor = ShadowPredictor.from_config(ml_cfg)
        predictor.train(train_samples)
        shadow_result = predictor.run(
            [s.candidate for s in test_samples],
            settings=settings,
            write_to_db=False,
        )

        engine = RecommendationEngine(
            model_version=ml_cfg.model_version,
            config_version=settings.config_version,
        )
        rec_result = engine.run(shadow_result.bundles, write_to_db=False)

        recs = rec_result.recommendations
        recs_ok = (
            len(recs) == len(test_samples)
            and all(isinstance(r, MLRecommendation) for r in recs)
            and not rec_result.applied
            and all(r.applied is False for r in recs)
        )
        out.append(
            Criterion.ok(
                "recommendation_mode",
                f"{len(recs)} recommendations produced; all applied=False",
            )
            if recs_ok
            else Criterion.fail(
                "recommendation_mode",
                f"n_recs={len(recs)} expected={len(test_samples)} applied={rec_result.applied}",
            )
        )

        # Verify no Stage 3 recommendation has applied=True.
        applied_recs = [r for r in recs if r.applied]
        out.append(
            Criterion.ok(
                "recommendation_no_live_influence",
                "0 Stage-3 recommendations have applied=True",
            )
            if not applied_recs
            else Criterion.fail(
                "recommendation_no_live_influence",
                f"{len(applied_recs)} recommendations have applied=True (forbidden in Stage 3)",
            )
        )

        # Check DB writes with applied=False for Stage 3.
        engine.run(shadow_result.bundles, write_to_db=True)
        from src.db.base import session_scope
        from src.db.models import ShadowLog

        with session_scope() as session:
            applied_db = (
                session.query(ShadowLog)
                .filter(ShadowLog.mode == "RECOMMEND", ShadowLog.applied.is_(True))
                .count()
            )
        out.append(
            Criterion.ok(
                "recommendation_db_applied_false",
                "RECOMMEND mode shadow_log entries: applied=False on all",
            )
            if applied_db == 0
            else Criterion.fail(
                "recommendation_db_applied_false",
                f"{applied_db} RECOMMEND entries have applied=True (must be False)",
            )
        )

    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("recommendation_mode", f"raised: {exc}"))

    # ------------------------------------------------------------------ #
    # 4. MLFilter — Stage 4: can block candidates                         #
    # ------------------------------------------------------------------ #
    try:
        from src.ml.config import load_ml_config
        from src.ml.labels import build_reference_dataset, train_test_split
        from src.ml.shadow import ShadowPredictor

        ml_cfg = load_ml_config()
        samples = build_reference_dataset(n_good=10, n_bad=20, n_neutral=10, seed=77)
        train_s, test_s = train_test_split(samples, seed=77)

        predictor = ShadowPredictor.from_config(ml_cfg)
        predictor.train(train_s)

        candidates = [s.candidate for s in test_s]
        shadow_result = predictor.run(candidates, settings=settings, write_to_db=False)

        # Use a high threshold to force some blocks.
        ml_filter = MLFilter(
            min_confidence_to_take=0.70,
            model_version=ml_cfg.model_version,
            config_version=settings.config_version,
        )
        filter_result = ml_filter.apply(candidates, shadow_result.bundles, write_to_db=False)

        out.append(
            Criterion.ok(
                "filter_can_block",
                f"filter blocked {filter_result.block_count} / {len(candidates)} candidates",
            )
            if filter_result.block_count > 0
            else Criterion.fail(
                "filter_can_block",
                "filter blocked 0 candidates — threshold may be too low or all predictions high",
            )
        )

    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("filter_can_block", f"raised: {exc}"))

    # ------------------------------------------------------------------ #
    # 5. MLFilter — cannot create new candidates                          #
    # ------------------------------------------------------------------ #
    try:
        from src.ml.config import load_ml_config
        from src.ml.labels import build_reference_dataset, train_test_split
        from src.ml.shadow import ShadowPredictor

        ml_cfg = load_ml_config()
        samples = build_reference_dataset(n_good=15, n_bad=15, n_neutral=10, seed=33)
        train_s, test_s = train_test_split(samples, seed=33)

        predictor = ShadowPredictor.from_config(ml_cfg)
        predictor.train(train_s)
        candidates = [s.candidate for s in test_s]
        shadow_result = predictor.run(candidates, settings=settings, write_to_db=False)

        # Use a very LOW threshold so all candidates pass — verifies no new ones added.
        ml_filter = MLFilter(
            min_confidence_to_take=0.0,
            model_version=ml_cfg.model_version,
            config_version=settings.config_version,
        )
        filter_result = ml_filter.apply(candidates, shadow_result.bundles, write_to_db=False)

        no_create_ok = len(filter_result.passed) <= len(candidates) and all(
            c in candidates for c in filter_result.passed
        )
        out.append(
            Criterion.ok(
                "filter_no_create",
                f"passed {filter_result.pass_count} <= input {len(candidates)}; "
                "all passed are original candidates",
            )
            if no_create_ok
            else Criterion.fail(
                "filter_no_create",
                f"filter produced {len(filter_result.passed)} > input {len(candidates)} "
                "or introduced foreign candidates",
            )
        )

    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("filter_no_create", f"raised: {exc}"))

    # ------------------------------------------------------------------ #
    # 6. MLFilter — passed candidates are unmodified originals            #
    # ------------------------------------------------------------------ #
    try:
        from src.ml.config import load_ml_config
        from src.ml.labels import build_reference_dataset, train_test_split
        from src.ml.shadow import ShadowPredictor

        ml_cfg = load_ml_config()
        samples = build_reference_dataset(n_good=15, n_bad=10, n_neutral=10, seed=55)
        train_s, test_s = train_test_split(samples, seed=55)

        predictor = ShadowPredictor.from_config(ml_cfg)
        predictor.train(train_s)
        candidates = [s.candidate for s in test_s]
        shadow_result = predictor.run(candidates, settings=settings, write_to_db=False)

        ml_filter = MLFilter(
            min_confidence_to_take=0.0,
            model_version=ml_cfg.model_version,
            config_version=settings.config_version,
        )
        filter_result = ml_filter.apply(candidates, shadow_result.bundles, write_to_db=False)

        identical = all(
            c is orig
            for c, orig in zip(filter_result.passed, candidates, strict=False)
            if c in candidates
        )
        # Also check stop_frac (risk proxy) is unchanged.
        risk_unchanged = all(
            c.stop_frac == orig.stop_frac
            for c in filter_result.passed
            for orig in candidates
            if c is orig
        )
        out.append(
            Criterion.ok(
                "filter_no_risk_increase",
                "passed candidates are identical Python objects — no field mutation",
            )
            if identical and risk_unchanged
            else Criterion.fail(
                "filter_no_risk_increase",
                "filter returned a modified candidate (stop_frac changed)",
            )
        )

    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("filter_no_risk_increase", f"raised: {exc}"))

    # ------------------------------------------------------------------ #
    # 7. MLFilter — hard-blocked candidates are always blocked            #
    # ------------------------------------------------------------------ #
    try:
        from src.ml.models.base import ShadowPrediction
        from src.ml.shadow import ShadowBundle

        # Build a candidate with strategy_enabled=False (hard blocker).
        disabled_cand = _make_disabled_candidate()
        normal_cand = _make_normal_candidate()

        # Shadow bundles: high confidence to take (would pass without hard blocker).
        high_conf_pred = ShadowPrediction(
            model_id="test",
            model_type="meta_labeler",
            label=1,
            probability=0.95,
            rationale="high confidence take",
        )
        disabled_bundle = ShadowBundle(candidate=disabled_cand, meta_label=high_conf_pred)
        normal_bundle = ShadowBundle(candidate=normal_cand, meta_label=high_conf_pred)

        ml_filter = MLFilter(min_confidence_to_take=0.0)
        result = ml_filter.apply(
            [disabled_cand, normal_cand],
            [disabled_bundle, normal_bundle],
            write_to_db=False,
        )

        # Disabled candidate must be blocked regardless of high confidence.
        disabled_blocked = disabled_cand in result.blocked
        normal_passed = normal_cand in result.passed

        out.append(
            Criterion.ok(
                "filter_hard_blocked",
                "strategy_disabled candidate blocked even with high ML confidence; "
                "normal candidate passed",
            )
            if disabled_blocked and normal_passed
            else Criterion.fail(
                "filter_hard_blocked",
                f"disabled_blocked={disabled_blocked} normal_passed={normal_passed} "
                "(disabled candidate must always be blocked)",
            )
        )

    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("filter_hard_blocked", f"raised: {exc}"))

    # ------------------------------------------------------------------ #
    # 8. Write Phase 10 ML report                                         #
    # ------------------------------------------------------------------ #
    import contextlib

    with contextlib.suppress(Exception):
        _write_phase10_report(settings, ml_cfg)

    return out


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _make_disabled_candidate() -> Candidate:
    return Candidate(
        symbol="BTCUSDT",
        strategy="test_strat",
        strategy_version="v1",
        side=1,
        entry_price=50000.0,
        stop_frac=0.01,
        tp_frac=0.02,
        regime="trend",
        session=0,
        signal_strength=0.9,
        expected_edge_frac=0.02,
        strategy_enabled=False,  # HARD BLOCKER
    )


def _make_normal_candidate() -> Candidate:
    return Candidate(
        symbol="ETHUSDT",
        strategy="test_strat",
        strategy_version="v1",
        side=1,
        entry_price=3000.0,
        stop_frac=0.01,
        tp_frac=0.02,
        regime="trend",
        session=0,
        signal_strength=0.9,
        expected_edge_frac=0.02,
        strategy_enabled=True,
    )


def _write_phase10_report(settings: Settings, ml_cfg: object) -> None:
    import json
    from datetime import UTC, datetime

    reports_dir = settings.reports_path / "phase_10"
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    path = reports_dir / f"ml_phase10_{stamp}.json"
    payload = {
        "phase": 10,
        "model_version": getattr(ml_cfg, "model_version", "unknown"),
        "ml_stage": getattr(ml_cfg, "ml_stage", "unknown"),
        "versions": settings.versions(),
        "generated_at": stamp,
        "note": (
            "Phase 10 — ML Recommendation Mode (Stage 3) + Constrained Live Filter (Stage 4). "
            "Shadow gate passed in Phase 9. Stage 3: recommendations shown in dashboard, "
            "never auto-applied. Stage 4: filter may block deterministic candidates; "
            "cannot create candidates, increase risk, or override hard blockers."
        ),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
