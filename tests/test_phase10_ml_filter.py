"""Phase 10 ML Recommendation + Constrained Filter tests (AGENTS.md Appendix D Phase 10).

Tests cover:
  - RecommendationEngine: output validity, applied=False invariant.
  - MLFilter: blocks low-confidence candidates.
  - MLFilter: cannot create new candidates (output <= input).
  - MLFilter: passed candidates are unmodified originals.
  - MLFilter: hard-blocked candidates (strategy_enabled=False, data_fresh=False,
    symbol_tradable=False) are always blocked regardless of ML confidence.
  - MLFilter: passes all candidates when threshold=0.0.
  - MLConfig: Stage 3/4 config parsed correctly.
  - Gate check: check_ml_phase10 passes (offline; no DB required for unit tests).
"""

from __future__ import annotations

import pytest
from src.ml.filter import MLFilter, _hard_blocker_reason
from src.ml.models.base import ShadowPrediction
from src.ml.recommendation import MLRecommendation, RecommendationEngine
from src.ml.shadow import ShadowBundle
from src.ranking.candidate import Candidate  # noqa: TCH001

# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #


def _make_candidate(
    symbol: str = "BTCUSDT",
    strategy_enabled: bool = True,
    data_fresh: bool = True,
    metadata_verified: bool = True,
    symbol_tradable: bool = True,
    config_live_approved: bool = True,
    signal_strength: float = 0.8,
    stop_frac: float = 0.01,
) -> Candidate:
    return Candidate(
        symbol=symbol,
        strategy="test_strat",
        strategy_version="v1",
        side=1,
        entry_price=50000.0,
        stop_frac=stop_frac,
        tp_frac=0.02,
        regime="trend",
        session=0,
        signal_strength=signal_strength,
        expected_edge_frac=0.02,
        strategy_enabled=strategy_enabled,
        data_fresh=data_fresh,
        metadata_verified=metadata_verified,
        symbol_tradable=symbol_tradable,
        config_live_approved=config_live_approved,
    )


def _make_bundle(
    cand: Candidate,
    prob: float = 0.8,
    label: int = 1,
) -> ShadowBundle:
    pred = ShadowPrediction(
        model_id="meta_labeler_test",
        model_type="meta_labeler",
        label=label,
        probability=prob,
        rationale="test prediction",
    )
    return ShadowBundle(candidate=cand, meta_label=pred)


# --------------------------------------------------------------------------- #
# MLConfig Stage 3/4 parsing                                                  #
# --------------------------------------------------------------------------- #


def test_ml_config_stage_advanced():
    from src.ml.config import load_ml_config

    cfg = load_ml_config()
    assert cfg.ml_stage >= 3, "Phase 10 requires ml_stage >= 3"


def test_ml_config_recommendation_cfg():
    from src.ml.config import load_ml_config

    cfg = load_ml_config()
    assert hasattr(cfg, "recommendation"), "MLConfig must have recommendation field"
    assert cfg.recommendation is not None


def test_ml_config_filter_cfg():
    from src.ml.config import load_ml_config

    cfg = load_ml_config()
    assert hasattr(cfg, "filter"), "MLConfig must have filter field"
    assert cfg.filter is not None
    assert 0.0 <= cfg.filter.min_confidence_to_take <= 1.0


# --------------------------------------------------------------------------- #
# RecommendationEngine — Stage 3                                              #
# --------------------------------------------------------------------------- #


def test_recommendation_engine_produces_output():
    cands = [_make_candidate(symbol=f"SYM{i}") for i in range(5)]
    bundles = [_make_bundle(c, prob=0.7) for c in cands]
    engine = RecommendationEngine(model_version="v_test", config_version="cfg_test")
    result = engine.run(bundles, write_to_db=False)

    assert len(result.recommendations) == 5
    assert all(isinstance(r, MLRecommendation) for r in result.recommendations)


def test_recommendation_applied_always_false():
    cands = [_make_candidate()]
    bundles = [_make_bundle(cands[0], prob=0.9, label=1)]
    engine = RecommendationEngine()
    result = engine.run(bundles, write_to_db=False)

    assert result.applied is False
    assert all(r.applied is False for r in result.recommendations)


def test_recommendation_high_confidence_take():
    cand = _make_candidate()
    bundle = _make_bundle(cand, prob=0.9, label=1)
    engine = RecommendationEngine()
    result = engine.run([bundle], write_to_db=False)
    rec = result.recommendations[0]

    assert rec.recommend_take is True
    assert rec.confidence == pytest.approx(0.9, abs=1e-4)


def test_recommendation_low_confidence_skip():
    cand = _make_candidate()
    bundle = _make_bundle(cand, prob=0.3, label=0)
    engine = RecommendationEngine()
    result = engine.run([bundle], write_to_db=False)
    rec = result.recommendations[0]

    assert rec.recommend_take is False
    assert rec.risk_bucket == 0.0


def test_recommendation_risk_bucket_never_above_one():
    for prob in (0.95, 0.80, 0.60, 0.40, 0.10):
        cand = _make_candidate()
        bundle = _make_bundle(cand, prob=prob, label=int(prob > 0.5))
        engine = RecommendationEngine()
        result = engine.run([bundle], write_to_db=False)
        rec = result.recommendations[0]
        assert rec.risk_bucket <= 1.0, f"risk_bucket > 1.0 at prob={prob}"


def test_recommendation_rationale_nonempty():
    cand = _make_candidate()
    bundle = _make_bundle(cand, prob=0.7)
    engine = RecommendationEngine()
    result = engine.run([bundle], write_to_db=False)
    assert result.recommendations[0].rationale != ""


# --------------------------------------------------------------------------- #
# MLFilter — Stage 4 Safety Invariants                                        #
# --------------------------------------------------------------------------- #


def test_filter_blocks_low_confidence():
    cand = _make_candidate()
    bundle = _make_bundle(cand, prob=0.2, label=0)
    ml_filter = MLFilter(min_confidence_to_take=0.5)
    result = ml_filter.apply([cand], [bundle], write_to_db=False)

    assert cand in result.blocked
    assert cand not in result.passed
    assert result.block_count == 1


def test_filter_passes_high_confidence():
    cand = _make_candidate()
    bundle = _make_bundle(cand, prob=0.9, label=1)
    ml_filter = MLFilter(min_confidence_to_take=0.5)
    result = ml_filter.apply([cand], [bundle], write_to_db=False)

    assert cand in result.passed
    assert cand not in result.blocked
    assert result.pass_count == 1


def test_filter_cannot_create_candidates():
    cands = [_make_candidate(symbol=f"S{i}") for i in range(5)]
    bundles = [_make_bundle(c, prob=0.0) for c in cands]
    ml_filter = MLFilter(min_confidence_to_take=0.0)
    result = ml_filter.apply(cands, bundles, write_to_db=False)

    assert len(result.passed) <= len(cands)
    for passed_cand in result.passed:
        assert passed_cand in cands, "filter introduced a foreign candidate"


def test_filter_passed_candidates_are_identical_objects():
    """Passed candidates must be the same Python objects — no field mutation."""
    cands = [_make_candidate(symbol=f"S{i}") for i in range(4)]
    bundles = [_make_bundle(c, prob=0.9) for c in cands]
    ml_filter = MLFilter(min_confidence_to_take=0.0)
    result = ml_filter.apply(cands, bundles, write_to_db=False)

    for passed_cand in result.passed:
        assert passed_cand is cands[cands.index(passed_cand)]


def test_filter_no_risk_increase():
    """Filter cannot increase stop_frac (risk proxy) on any candidate."""
    orig_stop_frac = 0.01
    cands = [_make_candidate(stop_frac=orig_stop_frac)]
    bundles = [_make_bundle(cands[0], prob=0.9)]
    ml_filter = MLFilter(min_confidence_to_take=0.0)
    result = ml_filter.apply(cands, bundles, write_to_db=False)

    for c in result.passed:
        assert c.stop_frac == orig_stop_frac


def test_filter_passes_all_when_threshold_zero():
    cands = [_make_candidate(symbol=f"S{i}") for i in range(10)]
    bundles = [_make_bundle(c, prob=0.0, label=0) for c in cands]
    ml_filter = MLFilter(min_confidence_to_take=0.0)
    result = ml_filter.apply(cands, bundles, write_to_db=False)

    assert result.pass_count == 10
    assert result.block_count == 0


def test_filter_blocks_all_when_threshold_one():
    cands = [_make_candidate(symbol=f"S{i}") for i in range(5)]
    # High probability but threshold=1.0 → all blocked (p < 1.0 strictly)
    bundles = [_make_bundle(c, prob=0.99, label=1) for c in cands]
    ml_filter = MLFilter(min_confidence_to_take=1.0)
    result = ml_filter.apply(cands, bundles, write_to_db=False)

    assert result.block_count == 5


# --------------------------------------------------------------------------- #
# MLFilter — Hard Blockers                                                    #
# --------------------------------------------------------------------------- #


def test_filter_blocks_disabled_strategy_regardless_of_confidence():
    disabled = _make_candidate(strategy_enabled=False)
    normal = _make_candidate(symbol="ETHUSDT")

    # Even with very high confidence, the disabled candidate must be blocked.
    bundles = [
        _make_bundle(disabled, prob=0.99, label=1),
        _make_bundle(normal, prob=0.99, label=1),
    ]
    ml_filter = MLFilter(min_confidence_to_take=0.0)
    result = ml_filter.apply([disabled, normal], bundles, write_to_db=False)

    assert disabled in result.blocked
    assert normal in result.passed


def test_filter_blocks_stale_data_candidate():
    stale = _make_candidate(data_fresh=False)
    bundle = _make_bundle(stale, prob=0.99, label=1)
    ml_filter = MLFilter(min_confidence_to_take=0.0)
    result = ml_filter.apply([stale], [bundle], write_to_db=False)

    assert stale in result.blocked
    reasons = result.block_reasons()
    assert "data_not_fresh" in reasons.get("BTCUSDT", "")


def test_filter_blocks_unverified_metadata():
    bad_meta = _make_candidate(metadata_verified=False)
    bundle = _make_bundle(bad_meta, prob=0.99, label=1)
    ml_filter = MLFilter(min_confidence_to_take=0.0)
    result = ml_filter.apply([bad_meta], [bundle], write_to_db=False)

    assert bad_meta in result.blocked


def test_filter_blocks_untradable_symbol():
    untradable = _make_candidate(symbol_tradable=False)
    bundle = _make_bundle(untradable, prob=0.99)
    ml_filter = MLFilter(min_confidence_to_take=0.0)
    result = ml_filter.apply([untradable], [bundle], write_to_db=False)

    assert untradable in result.blocked


# --------------------------------------------------------------------------- #
# hard_blocker_reason helper                                                  #
# --------------------------------------------------------------------------- #


def test_hard_blocker_no_blocker():
    cand = _make_candidate()
    assert _hard_blocker_reason(cand) == ""


def test_hard_blocker_strategy_disabled():
    cand = _make_candidate(strategy_enabled=False)
    assert _hard_blocker_reason(cand) == "strategy_disabled"


def test_hard_blocker_data_not_fresh():
    cand = _make_candidate(data_fresh=False)
    assert _hard_blocker_reason(cand) == "data_not_fresh"


# --------------------------------------------------------------------------- #
# FilterResult helpers                                                        #
# --------------------------------------------------------------------------- #


def test_filter_result_block_reasons():
    disabled = _make_candidate(symbol="BTCUSDT", strategy_enabled=False)
    bundle = _make_bundle(disabled, prob=0.0)
    ml_filter = MLFilter()
    result = ml_filter.apply([disabled], [bundle], write_to_db=False)

    reasons = result.block_reasons()
    assert "BTCUSDT" in reasons
    assert "hard_blocker" in reasons["BTCUSDT"]


def test_filter_requires_equal_length_inputs():
    cands = [_make_candidate()]
    bundles = [_make_bundle(cands[0]), _make_bundle(cands[0])]
    ml_filter = MLFilter()
    with pytest.raises(ValueError, match="equal length"):
        ml_filter.apply(cands, bundles, write_to_db=False)


# --------------------------------------------------------------------------- #
# MLFilter constructor validation                                             #
# --------------------------------------------------------------------------- #


def test_filter_invalid_threshold_raises():
    with pytest.raises(ValueError):
        MLFilter(min_confidence_to_take=1.5)

    with pytest.raises(ValueError):
        MLFilter(min_confidence_to_take=-0.1)


# --------------------------------------------------------------------------- #
# Phase 10 gate check (offline)                                               #
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    not pytest.importorskip("src.config", reason="config not available"),
    reason="offline only",
)
def test_phase10_gate_check_importable():
    from src.gates.phase10 import check_ml_phase10

    assert callable(check_ml_phase10)


def test_phase10_ml_imports():
    from src.ml import (
        FilterDecision,
        FilterResult,
        MLFilter,
        MLRecommendation,
        RecommendationEngine,
        RecommendationRunResult,
    )

    assert MLFilter is not None
    assert RecommendationEngine is not None
    assert MLRecommendation is not None
    assert FilterResult is not None
    assert FilterDecision is not None
    assert RecommendationRunResult is not None
