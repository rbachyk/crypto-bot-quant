"""Phase 9 Shadow ML tests (AGENTS.md Appendix D Phase 9).

Tests cover:
  - Label generation and reference dataset construction.
  - Feature extraction from Candidate objects.
  - Model training and prediction (all five shadow models).
  - ShadowPredictor: train + run + applied=False invariant.
  - ShadowScorer: baseline vs model expectancy comparison.
  - MLRegistry: artifact save/load round-trip.
  - Gate check: ML-PROMO (offline; no live DB required for unit tests).
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

from src.ml.features import FEATURE_NAMES, build_feature_matrix, candidate_to_row
from src.ml.labels import (
    LabeledSample,
    baseline_expectancy,
    best_n_trades,
    build_reference_dataset,
    filtered_expectancy,
    label_from_outcome,
    profit_factor,
    synthetic_labels,
    train_test_split,
    worst_n_trades,
    worst_trade,
)
from src.ml.models import (
    ExecQualityModel,
    MetaLabeler,
    RegimeClassifier,
    StrategySelector,
    SymbolRanker,
)
from src.ml.registry import MLRegistry, ModelArtifact
from src.ml.scorer import ShadowScorer
from src.ml.shadow import ShadowPredictor

from tests.conftest import requires_db

# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #


def _make_reference(n_good: int = 20, n_bad: int = 15, n_neutral: int = 15) -> list[LabeledSample]:
    return build_reference_dataset(n_good=n_good, n_bad=n_bad, n_neutral=n_neutral, seed=42)


def _make_predictor() -> ShadowPredictor:
    from src.ml.config import load_ml_config

    return ShadowPredictor.from_config(load_ml_config())


# --------------------------------------------------------------------------- #
# label generation                                                              #
# --------------------------------------------------------------------------- #


def test_label_from_outcome_positive():
    assert label_from_outcome(0.5) == 1


def test_label_from_outcome_negative():
    assert label_from_outcome(-0.3) == 0


def test_label_from_outcome_zero():
    assert label_from_outcome(0.0) == 0


def test_synthetic_labels_reproducible():
    samples = _make_reference(n_good=10, n_bad=10, n_neutral=10)
    candidates = [s.candidate for s in samples]
    a = synthetic_labels(candidates, seed=42)
    b = synthetic_labels(candidates, seed=42)
    assert [s.label for s in a] == [s.label for s in b]


def test_reference_dataset_size():
    samples = build_reference_dataset(n_good=40, n_bad=30, n_neutral=30, seed=42)
    assert len(samples) == 100


def test_reference_dataset_has_positive_label_class():
    samples = _make_reference()
    positives = [s for s in samples if s.label == 1]
    assert len(positives) > 0, "good candidates must produce label=1 trades"


def test_train_test_split_sizes():
    samples = _make_reference()
    train, test = train_test_split(samples, test_fraction=0.25, seed=42)
    n = len(samples)
    expected_train = int(n * 0.75)
    assert len(train) == expected_train or len(train) == expected_train + 1
    assert len(train) + len(test) == n


def test_baseline_expectancy_all_negative():
    samples = [
        LabeledSample(candidate=s.candidate, label=0, realized_pnl=-1.0, hold_bars=1)
        for s in _make_reference(n_good=0, n_bad=5, n_neutral=0)
    ]
    assert baseline_expectancy(samples) < 0


def test_filtered_expectancy_empty_predictions():
    samples = _make_reference()
    preds = [0] * len(samples)
    # No trades taken → expectancy is 0
    assert filtered_expectancy(samples, preds) == 0.0


def test_profit_factor_no_losses():
    samples = [
        LabeledSample(candidate=s.candidate, label=1, realized_pnl=1.0, hold_bars=1)
        for s in _make_reference(n_good=5)
    ]
    pf = profit_factor(samples, [1] * len(samples))
    assert math.isinf(pf) or pf > 1.0


def test_worst_trade_value():
    samples = _make_reference()
    wt = worst_trade(samples)
    assert wt <= 0 or wt < max(s.realized_pnl for s in samples)


def test_best_n_trades_order():
    samples = _make_reference()
    top5 = best_n_trades(samples, 5)
    assert top5 == sorted(samples, key=lambda s: s.realized_pnl, reverse=True)[:5]


def test_worst_n_trades_order():
    samples = _make_reference()
    bot5 = worst_n_trades(samples, 5)
    assert bot5 == sorted(samples, key=lambda s: s.realized_pnl)[:5]


# --------------------------------------------------------------------------- #
# feature extraction                                                            #
# --------------------------------------------------------------------------- #


def test_candidate_to_row_keys():
    samples = _make_reference(n_good=2)
    row = candidate_to_row(samples[0].candidate)
    assert "signal_strength" in row
    assert "expected_edge_frac" in row
    assert "spread_bps" in row


def test_feature_matrix_shape():
    samples = _make_reference(n_good=5, n_bad=5, n_neutral=5)
    candidates = [s.candidate for s in samples]
    X = build_feature_matrix(candidates)
    assert len(X) == len(candidates)
    assert len(X[0]) == len(FEATURE_NAMES)


def test_feature_matrix_subset():
    samples = _make_reference(n_good=3)
    candidates = [s.candidate for s in samples]
    subset = ["signal_strength", "spread_bps"]
    X = build_feature_matrix(candidates, subset)
    assert len(X[0]) == 2


def test_feature_names_count():
    assert len(FEATURE_NAMES) >= 7, "at least 7 features required by meta_labeler config"


# --------------------------------------------------------------------------- #
# individual model train/predict                                                #
# --------------------------------------------------------------------------- #


def _train_binary(model_cls, samples, feature_list=None):
    from src.ml.features import feature_names_for

    feat_names = feature_names_for(feature_list) if feature_list else list(FEATURE_NAMES)
    candidates = [s.candidate for s in samples]
    X = build_feature_matrix(candidates, feat_names)
    y = [s.label for s in samples]
    model = model_cls(f"test_{model_cls.__name__.lower()}", "v0")
    metrics = model.train(X, y, feat_names)
    return model, metrics


def test_meta_labeler_trains_and_predicts():
    samples = _make_reference(n_good=20, n_bad=20, n_neutral=20)
    model, metrics = _train_binary(MetaLabeler, samples)
    assert "accuracy" in metrics
    assert 0 <= metrics["accuracy"] <= 1

    preds = model.predict(build_feature_matrix([s.candidate for s in samples[:5]]))
    assert len(preds) == 5
    assert all(p.label in (0, 1) for p in preds)
    assert model.model_type == "meta_labeler"


def test_exec_quality_trains_and_predicts():
    samples = _make_reference(n_good=20, n_bad=20)
    model, metrics = _train_binary(ExecQualityModel, samples)
    preds = model.predict(build_feature_matrix([s.candidate for s in samples[:3]]))
    assert len(preds) == 3
    assert model.model_type == "exec_quality"


def test_strategy_selector_trains_and_predicts():
    samples = _make_reference(n_good=20, n_bad=20)
    model, _ = _train_binary(StrategySelector, samples)
    preds = model.predict(build_feature_matrix([s.candidate for s in samples[:3]]))
    assert len(preds) == 3
    assert model.model_type == "strategy_selector"


def test_symbol_ranker_trains_and_predicts():
    samples = _make_reference(n_good=20, n_bad=20)
    model, _ = _train_binary(SymbolRanker, samples)
    preds = model.predict(build_feature_matrix([s.candidate for s in samples[:3]]))
    assert len(preds) == 3
    assert model.model_type == "symbol_ranker"


def test_regime_classifier_trains_and_predicts():
    samples = _make_reference(n_good=20, n_bad=20, n_neutral=20)
    _REGIME_MAP = {
        "low_vol_up": 0,
        "low_vol_down": 0,
        "trend_up": 1,
        "low_vol_range": 0,
        "trend": 1,
    }
    candidates = [s.candidate for s in samples]
    X = build_feature_matrix(candidates)
    y = [_REGIME_MAP.get(s.candidate.regime, 0) for s in samples]
    model = RegimeClassifier("regime_classifier_v0", "v0")
    metrics = model.train(X, y, list(FEATURE_NAMES))
    assert "accuracy" in metrics
    preds = model.predict(X[:3])
    assert len(preds) == 3
    assert model.model_type == "regime_classifier"


def test_model_snapshot_round_trip():
    samples = _make_reference(n_good=20, n_bad=20)
    model, _ = _train_binary(MetaLabeler, samples)
    blob = model.snapshot()
    assert isinstance(blob, bytes) and len(blob) > 0

    model2 = MetaLabeler("test_round_trip", "v0")
    model2.load(blob)
    X = build_feature_matrix([s.candidate for s in samples[:5]])
    preds_original = model.predict(X)
    preds_restored = model2.predict(X)
    assert [p.label for p in preds_original] == [p.label for p in preds_restored]


# --------------------------------------------------------------------------- #
# ShadowPredictor                                                               #
# --------------------------------------------------------------------------- #


def test_shadow_predictor_from_config():
    predictor = _make_predictor()
    assert len(predictor.all_models) == 5


def test_shadow_predictor_train_all():
    samples = _make_reference(n_good=30, n_bad=20, n_neutral=10)
    train_samples, _ = train_test_split(samples, test_fraction=0.25, seed=42)
    predictor = _make_predictor()
    metrics = predictor.train(train_samples)
    assert "meta_labeler" in metrics
    assert "regime_classifier" in metrics
    assert "exec_quality" in metrics
    assert "strategy_selector" in metrics
    assert "symbol_ranker" in metrics


def test_shadow_predictor_run_no_db():
    """run() with write_to_db=False must not require a database."""
    samples = _make_reference(n_good=30, n_bad=20, n_neutral=10)
    train_samples, test_samples = train_test_split(samples, seed=42)
    predictor = _make_predictor()
    predictor.train(train_samples)

    from src.config import get_settings

    settings = get_settings()
    result = predictor.run(
        [s.candidate for s in test_samples[:10]],
        settings=settings,
        write_to_db=False,
    )
    assert not result.applied, "applied must always be False"
    assert len(result.bundles) == 10
    assert result.shadow_log_ids == []


def test_shadow_predictor_applied_invariant():
    """Regardless of DB writes, applied must always be False."""
    samples = _make_reference(n_good=30, n_bad=20)
    predictor = _make_predictor()
    predictor.train(samples)

    from src.config import get_settings

    result = predictor.run(
        [s.candidate for s in samples[:5]], settings=get_settings(), write_to_db=False
    )
    assert result.applied is False


def test_shadow_bundle_has_meta_label():
    samples = _make_reference(n_good=30, n_bad=20)
    train_samples, test_samples = train_test_split(samples, seed=42)
    predictor = _make_predictor()
    predictor.train(train_samples)

    from src.config import get_settings

    result = predictor.run(
        [s.candidate for s in test_samples[:5]], settings=get_settings(), write_to_db=False
    )
    for bundle in result.bundles:
        assert bundle.meta_label is not None
        assert bundle.meta_label.label in (0, 1)


# --------------------------------------------------------------------------- #
# ShadowScorer                                                                  #
# --------------------------------------------------------------------------- #


def test_scorer_improvement_on_reference_data():
    """Meta-labeler trained on reference data must improve expectancy over always-take."""
    samples = _make_reference(n_good=40, n_bad=30, n_neutral=30)
    train_samples, test_samples = train_test_split(samples, test_fraction=0.25, seed=42)
    predictor = _make_predictor()
    predictor.train(train_samples)

    from src.config import get_settings

    result = predictor.run(
        [s.candidate for s in test_samples], settings=get_settings(), write_to_db=False
    )
    test_preds = [b.meta_label.label if b.meta_label else 1 for b in result.bundles]

    scorer = ShadowScorer(
        min_improvement=0.0,
        min_pf_ratio=1.0,
        max_tail_loss_ratio=1.0,
        max_best_removed_pct=0.2,
    )
    score = scorer.score(test_samples, test_preds)
    assert score.expectancy_improvement > 0, (
        f"meta-labeler must improve over always-take baseline; "
        f"model={score.model_expectancy:.4f} baseline={score.baseline_expectancy:.4f}"
    )


def test_scorer_always_take_parity():
    """If predictions=all-1, model expectancy == baseline expectancy."""
    samples = _make_reference()
    preds = [1] * len(samples)
    scorer = ShadowScorer()
    score = scorer.score(samples, preds)
    assert abs(score.expectancy_improvement) < 1e-9


def test_scorer_to_dict_has_required_keys():
    samples = _make_reference()
    scorer = ShadowScorer()
    score = scorer.score(samples, [1] * len(samples))
    d = score.to_dict()
    for key in ("n_test", "baseline_expectancy", "model_expectancy", "passed", "fail_reasons"):
        assert key in d, f"missing key: {key}"


# --------------------------------------------------------------------------- #
# MLRegistry                                                                    #
# --------------------------------------------------------------------------- #


def test_registry_save_load_round_trip():
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = MLRegistry(Path(tmpdir))
        samples = _make_reference(n_good=20, n_bad=20)
        model, _ = _train_binary(MetaLabeler, samples)
        blob = model.snapshot()

        path = registry.save_artifact("meta_labeler_test", blob)
        assert Path(path).exists()

        loaded = registry.load_artifact("meta_labeler_test")
        assert loaded == blob


def test_registry_register_no_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = MLRegistry(Path(tmpdir))
        samples = _make_reference(n_good=20, n_bad=20)
        model, metrics = _train_binary(MetaLabeler, samples)
        artifact = ModelArtifact(
            model_id="meta_labeler_test",
            model_version="v0",
            model_type="meta_labeler",
            performance_metrics=metrics,
        )
        result = registry.register(model, artifact, write_db=False)
        assert result.artifact_path is not None
        assert Path(result.artifact_path).exists()


# --------------------------------------------------------------------------- #
# ML config                                                                     #
# --------------------------------------------------------------------------- #


def test_ml_config_loads():
    from src.ml.config import load_ml_config

    cfg = load_ml_config()
    assert cfg.ml_stage >= 2  # Phase 10 advances to Stage 4; >= 2 is valid
    assert cfg.shadow.mode == "SHADOW"
    assert not cfg.shadow.applied_to_live


def test_ml_config_shadow_mode_is_shadow():
    from src.ml.config import load_ml_config

    cfg = load_ml_config()
    assert cfg.shadow.applied_to_live is False, "Shadow mode must never influence live trading"


def test_ml_config_meta_labeler_features():
    from src.ml.config import load_ml_config

    cfg = load_ml_config()
    assert len(cfg.meta_labeler.features) >= 5


# --------------------------------------------------------------------------- #
# Gate check — real-data performance criteria (audit H15)                       #
# --------------------------------------------------------------------------- #

_PERF_NAMES = (
    "ml_expectancy_improves",
    "ml_profit_factor_preserved",
    "ml_tail_risk_reduced",
    "ml_best_trades_preserved",
)


def test_real_performance_criteria_fail_closed_without_data():
    """With no real linked shadow outcomes, all four performance criteria FAIL
    with explicit remediation — the gate must never pass on synthetic data."""
    from src.gates.phase9 import _real_performance_criteria
    from src.ml.config import load_ml_config

    criteria, score = _real_performance_criteria(0, [], load_ml_config())
    assert score is None
    assert [c.id for c in criteria] == list(_PERF_NAMES)
    for c in criteria:
        assert not c.passed
        assert "REAL shadow decisions" in c.detail
        assert "H18" in c.detail


def test_real_performance_criteria_fail_closed_below_minimum():
    """A handful of linked outcomes is still insufficient (min_train_samples)."""
    from src.gates.phase9 import _real_performance_criteria
    from src.ml.config import load_ml_config

    linked = [(1, 0.5)] * 5
    criteria, score = _real_performance_criteria(5, linked, load_ml_config())
    assert score is None
    assert all(not c.passed for c in criteria)


def test_real_performance_criteria_pass_on_good_real_outcomes():
    """When enough REAL outcomes exist and the model filtered the losers, the
    four kill-criteria pass."""
    from src.gates.phase9 import _real_performance_criteria
    from src.ml.config import load_ml_config

    # 20 winners the model took, 20 losers the model skipped.
    linked = [(1, 0.5)] * 20 + [(0, -0.5)] * 20
    criteria, score = _real_performance_criteria(40, linked, load_ml_config())
    assert score is not None
    assert all(c.passed for c in criteria), [(c.id, c.detail) for c in criteria]
    assert score.expectancy_improvement > 0


def test_real_performance_criteria_fail_on_negative_edge_model():
    """A model that keeps losers and skips winners must FAIL on real outcomes."""
    from src.gates.phase9 import _real_performance_criteria
    from src.ml.config import load_ml_config

    linked = [(0, 0.5)] * 20 + [(1, -0.5)] * 20  # takes only the losers
    criteria, _score = _real_performance_criteria(40, linked, load_ml_config())
    by_name = {c.id: c for c in criteria}
    assert not by_name["ml_expectancy_improves"].passed


@requires_db
def test_load_real_shadow_outcomes_links_and_excludes_synthetic():
    """The real-outcome loader excludes synthetic-tagged rows and joins the rest
    to paper trades on the same symbol within the horizon."""
    from datetime import UTC, datetime, timedelta

    from src.db.base import session_scope
    from src.db.models import PaperTradeRecord, ShadowLog
    from src.gates.phase9 import _load_real_shadow_outcomes
    from src.ml.shadow import SYNTHETIC_CONTEXT_KEY

    symbol = "GATE9TST/USDT:USDT"  # unique symbol so shared-DB rows can't interfere
    now = datetime.now(UTC)
    row_ids: list[int] = []
    trade_ids: list[int] = []
    try:
        with session_scope() as session:
            real_row = ShadowLog(
                ts=now - timedelta(hours=2),
                model_id="meta_labeler_x",
                model_type="meta_labeler",
                mode="SHADOW",
                symbol=symbol,
                context_features={"signal_strength": 0.7},
                prediction={"label": 1, "probability": 0.8},
                applied=False,
            )
            synthetic_row = ShadowLog(
                ts=now - timedelta(hours=2),
                model_id="meta_labeler_x",
                model_type="meta_labeler",
                mode="SHADOW",
                symbol=symbol,
                context_features={SYNTHETIC_CONTEXT_KEY: 1.0},
                prediction={"label": 1, "probability": 0.9},
                applied=False,
            )
            session.add_all([real_row, synthetic_row])
            session.flush()
            row_ids = [real_row.id, synthetic_row.id]
            trade = PaperTradeRecord(
                session_id="gate9-test",
                trade_id="gate9-t1",
                created_at=now - timedelta(hours=1),
                symbol=symbol,
                strategy="test",
                pnl_r=0.42,
            )
            session.add(trade)
            session.flush()
            trade_ids = [trade.id]

        n_real, linked = _load_real_shadow_outcomes()
        assert n_real >= 1
        assert (1, 0.42) in linked
        # The synthetic-tagged row must not have produced a second identical link
        # for our symbol (only one real row was seeded).
        assert sum(1 for label, pnl in linked if pnl == 0.42 and label == 1) == 1
    finally:
        with session_scope() as session:
            if row_ids:
                session.query(ShadowLog).filter(ShadowLog.id.in_(row_ids)).delete(
                    synchronize_session=False
                )
            if trade_ids:
                session.query(PaperTradeRecord).filter(
                    PaperTradeRecord.id.in_(trade_ids)
                ).delete(synchronize_session=False)


@requires_db
def test_ml_promo_gate_runs(tmp_path):
    """ML-PROMO gate must run without exceptions and return Criterion objects."""
    from src.config import get_settings
    from src.gates.phase9 import check_ml_promo
    from src.gates.result import Criterion

    settings = get_settings()
    criteria = check_ml_promo(settings)
    assert isinstance(criteria, list)
    assert all(isinstance(c, Criterion) for c in criteria)
    ids = [c.id for c in criteria]
    for name in _PERF_NAMES:
        assert name in ids, f"performance criterion {name} missing"


@requires_db
def test_ml_promo_gate_no_live_influence(tmp_path):
    """ML-PROMO gate must verify applied=False in every shadow log entry."""
    from src.config import get_settings
    from src.gates.phase9 import check_ml_promo

    settings = get_settings()
    criteria = check_ml_promo(settings)
    by_name = {c.id: c for c in criteria}
    assert "ml_no_live_influence" in by_name
    assert by_name["ml_no_live_influence"].passed, (
        f"no live influence check failed: {by_name['ml_no_live_influence'].detail}"
    )


@requires_db
def test_ml_promo_gate_shadow_imports(tmp_path):
    """ml_shadow_imports must pass — all components must be importable."""
    from src.config import get_settings
    from src.gates.phase9 import check_ml_promo

    settings = get_settings()
    criteria = check_ml_promo(settings)
    by_name = {c.id: c for c in criteria}
    assert by_name["ml_shadow_imports"].passed


@requires_db
def test_ml_promo_gate_trains_all_models(tmp_path):
    """ml_models_train_plumbing must pass — all 5 models must train successfully."""
    from src.config import get_settings
    from src.gates.phase9 import check_ml_promo

    settings = get_settings()
    criteria = check_ml_promo(settings)
    by_name = {c.id: c for c in criteria}
    assert "ml_models_train_plumbing" in by_name
    assert by_name["ml_models_train_plumbing"].passed, (
        f"model training failed: {by_name['ml_models_train_plumbing'].detail}"
    )


@requires_db
def test_ml_promo_gate_cleans_up_its_shadow_log_writes(tmp_path):
    """The gate's plumbing shadow-log rows are tagged synthetic AND deleted at the
    end of the run — no residue accumulates in production tables (audit C4-adj)."""
    from src.config import get_settings
    from src.db.base import session_scope
    from src.db.models import ShadowLog
    from src.gates.phase9 import check_ml_promo
    from src.ml.shadow import SYNTHETIC_CONTEXT_KEY

    def _tagged_count() -> int:
        with session_scope() as session:
            rows = session.query(ShadowLog.context_features).all()
        return sum(1 for (ctx,) in rows if (ctx or {}).get(SYNTHETIC_CONTEXT_KEY))

    before = _tagged_count()
    criteria = check_ml_promo(get_settings())
    by_name = {c.id: c for c in criteria}
    assert by_name["ml_shadow_log_writes_plumbing"].passed, (
        by_name["ml_shadow_log_writes_plumbing"].detail
    )
    assert _tagged_count() == before, "gate left synthetic shadow_log rows behind"


# --------------------------------------------------------------------------- #
# M35 — ModelCfg.enabled / min_train_samples / test_fraction enforced          #
# --------------------------------------------------------------------------- #


def _cfg_with(**meta_overrides):
    """Fresh MLConfig with the meta_labeler cfg overridden."""
    import dataclasses

    from src.ml.config import load_ml_config

    cfg = load_ml_config()
    meta = dataclasses.replace(cfg.meta_labeler, **meta_overrides)
    return dataclasses.replace(cfg, meta_labeler=meta)


def test_disabled_model_skipped_in_train_and_run():
    """M35: enabled=false → no training, no predictions, honest skip status."""
    cfg = _cfg_with(enabled=False)
    predictor = ShadowPredictor.from_config(cfg)
    samples = _make_reference(n_good=30, n_bad=20, n_neutral=10)
    metrics = predictor.train(samples)

    assert metrics["meta_labeler"]["status"] == "skipped"
    assert "disabled" in metrics["meta_labeler"]["reason"]
    assert predictor._meta.is_trained is False
    # Other (enabled) models still trained.
    assert metrics["exec_quality"].get("status") == "trained"

    from src.config import get_settings

    result = predictor.run(
        [s.candidate for s in samples[:5]], settings=get_settings(), write_to_db=False
    )
    assert all(b.meta_label is None for b in result.bundles)
    assert all(b.exec_quality is not None for b in result.bundles)


def test_min_train_samples_prevents_sklearn_crash():
    """M35: 1-2 samples must be a clean skip, not an sklearn raise."""
    predictor = _make_predictor()
    samples = _make_reference(n_good=1, n_bad=1, n_neutral=0)  # 2 samples < 30
    metrics = predictor.train(samples)  # must not raise
    for mtype in (
        "meta_labeler",
        "regime_classifier",
        "exec_quality",
        "strategy_selector",
        "symbol_ranker",
    ):
        assert metrics[mtype]["status"] == "skipped"
        assert "insufficient samples" in metrics[mtype]["reason"]
    assert predictor._meta.is_trained is False


def test_configured_test_fraction_used():
    """M35: the model's test_fraction carves a hold-out and reports OOS metrics."""
    cfg = _cfg_with(test_fraction=0.5)
    predictor = ShadowPredictor.from_config(cfg)
    samples = _make_reference(n_good=30, n_bad=20, n_neutral=10)  # 60 samples
    metrics = predictor.train(samples)

    meta = metrics["meta_labeler"]
    assert meta["test_fraction"] == 0.5
    assert meta["test_samples"] == 30  # half held out
    assert meta["train_samples"] == 30  # trained on the other half
    assert 0.0 <= meta["test_accuracy"] <= 1.0
    # Default 0.25 fraction on another model in the same run.
    assert metrics["exec_quality"]["test_samples"] == 15


def test_single_class_training_data_skipped():
    """M35: a train split with one label class skips cleanly instead of raising."""
    predictor = _make_predictor()
    samples = _make_reference(n_good=0, n_bad=40, n_neutral=0)  # all label=0
    metrics = predictor.train(samples)
    assert metrics["meta_labeler"]["status"] == "skipped"
    assert "single-class" in metrics["meta_labeler"]["reason"]


# --------------------------------------------------------------------------- #
# L33 — train_test_split sorts by decision_ts (temporal leakage guard)          #
# --------------------------------------------------------------------------- #


def test_train_test_split_sorts_by_decision_ts():
    """L33 regression: an unsorted batch must be re-ordered chronologically so
    the test split is strictly the most-recent samples."""
    samples = _make_reference(n_good=20, n_bad=15, n_neutral=15)
    shuffled = list(reversed(samples))  # deliberately break time order

    train, test = train_test_split(shuffled, test_fraction=0.25, seed=42)
    max_train_ts = max(s.candidate.decision_ts for s in train)
    min_test_ts = min(s.candidate.decision_ts for s in test)
    assert max_train_ts < min_test_ts, "test split must be strictly later than train"

    # Same result regardless of input order (deterministic chronological split).
    train2, test2 = train_test_split(samples, test_fraction=0.25, seed=42)
    assert [s.candidate.decision_ts for s in train] == [
        s.candidate.decision_ts for s in train2
    ]


def test_reference_dataset_timestamps_follow_shuffled_order():
    """The synthetic reference dataset must stay class-balanced under the
    chronological sort: timestamps are assigned AFTER the shuffle."""
    samples = _make_reference(n_good=20, n_bad=15, n_neutral=15)
    ts = [s.candidate.decision_ts for s in samples]
    assert ts == sorted(ts)  # list order == chronological order
    # Both halves contain positive and negative labels (balance preserved).
    half = len(samples) // 2
    assert 0 < sum(s.label for s in samples[:half]) < half
    assert 0 < sum(s.label for s in samples[half:]) <= half
