"""Shadow predictor — orchestrates all ML shadow models (AGENTS.md Section 20).

The :class:`ShadowPredictor` runs each model in turn, logs every prediction to
``shadow_logs``, and returns the predictions for offline scoring.  It **never**
affects live trading decisions: all log entries carry ``applied=False`` and
``mode="SHADOW"``.

ML Stage 2 contract (Section 20):
  * shadow predictions logged to ``shadow_log``
  * shadow trade approval/rejection recommendations logged
  * shadow regime labels logged
  * shadow strategy ranking logged
  * shadow symbol ranking logged
  * shadow execution recommendation logged
  * NOT allowed: changing actual bot behavior
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from src.config import Settings, get_settings
from src.ranking.candidate import Candidate

from .config import MLConfig, load_ml_config
from .features import build_feature_matrix, feature_names_for
from .models.base import ShadowModel, ShadowPrediction
from .models.exec_quality import ExecQualityModel
from .models.meta_labeler import MetaLabeler
from .models.regime_classifier import RegimeClassifier
from .models.strategy_selector import StrategySelector
from .models.symbol_ranker import SymbolRanker

_MODE = "SHADOW"

# Context-features marker stamped on shadow_log rows produced from the SYNTHETIC
# reference dataset (gate plumbing self-checks, demo jobs). Readers that evaluate
# REAL shadow performance (the ML-PROMO gate) must exclude rows carrying this key —
# synthetic rows are pipeline exercises, never evidence the model works (audit H15).
SYNTHETIC_CONTEXT_KEY = "synthetic_reference"


@dataclass
class ShadowBundle:
    """All shadow predictions for one candidate."""

    candidate: Candidate
    meta_label: ShadowPrediction | None = None
    regime: ShadowPrediction | None = None
    exec_quality: ShadowPrediction | None = None
    strategy: ShadowPrediction | None = None
    symbol: ShadowPrediction | None = None


@dataclass
class ShadowRunResult:
    """Result of one shadow predictor run over a list of candidates."""

    bundles: list[ShadowBundle]
    shadow_log_ids: list[int] = field(default_factory=list)
    model_version: str = "ml_shadow_0001"
    applied: bool = False  # always False


class ShadowPredictor:
    """Orchestrates shadow ML models for a set of candidates.

    Usage::

        predictor = ShadowPredictor.from_config()
        predictor.train(train_samples)  # one-off on reference data
        result = predictor.run(candidates, settings=settings)
        # result.bundles → offline scoring; result.applied is always False
    """

    def __init__(
        self,
        meta_labeler: MetaLabeler,
        regime_clf: RegimeClassifier,
        exec_quality: ExecQualityModel,
        strategy_sel: StrategySelector,
        symbol_rnk: SymbolRanker,
        cfg: MLConfig,
    ) -> None:
        self._meta = meta_labeler
        self._regime = regime_clf
        self._exec = exec_quality
        self._strat = strategy_sel
        self._sym = symbol_rnk
        self.cfg = cfg

    @classmethod
    def from_config(cls, ml_cfg: MLConfig | None = None) -> ShadowPredictor:
        cfg = ml_cfg or load_ml_config()
        ver = cfg.model_version
        return cls(
            meta_labeler=MetaLabeler(f"meta_labeler_{ver}", ver),
            regime_clf=RegimeClassifier(f"regime_classifier_{ver}", ver),
            exec_quality=ExecQualityModel(f"exec_quality_{ver}", ver),
            strategy_sel=StrategySelector(f"strategy_selector_{ver}", ver),
            symbol_rnk=SymbolRanker(f"symbol_ranker_{ver}", ver),
            cfg=cfg,
        )

    @property
    def all_models(self) -> list[ShadowModel]:
        return [self._meta, self._regime, self._exec, self._strat, self._sym]

    def train(self, samples: list) -> dict[str, dict]:
        """Train all models on labeled samples.  Returns per-model metrics.

        Per-model config is ENFORCED here (audit M35):
          * ``enabled=false`` → the model is skipped (``status="skipped"``) and
            stays untrained;
          * fewer samples than ``min_train_samples`` → clean skip, no sklearn
            crash on 1-2 samples;
          * ``test_fraction`` → a chronological hold-out is carved off before
            fitting and out-of-sample metrics are reported alongside the
            training metrics.
        """
        from .labels import LabeledSample

        assert all(isinstance(s, LabeledSample) for s in samples), "samples must be LabeledSample"

        labels = [s.label for s in samples]

        metrics: dict[str, dict] = {}
        for model, model_cfg in (
            (self._meta, self.cfg.meta_labeler),
            (self._exec, self.cfg.exec_quality),
            (self._strat, self.cfg.strategy_selector),
            (self._sym, self.cfg.symbol_ranker),
        ):
            metrics[model.model_type] = self._train_one(model, model_cfg, samples, labels)

        # Regime classifier uses encoded regime index as labels for multi-class.
        _REGIME_MAP = {
            "low_vol_range": 0,
            "low_vol_up": 0,
            "low_vol_down": 0,
            "trend": 1,
            "trend_up": 1,
            "trend_down": 1,
            "high_vol_expansion": 2,
            "high_vol_chop": 3,
            "market_wide_impulse": 4,
            "range": 0,
        }
        regime_labels = [_REGIME_MAP.get(s.candidate.regime, 0) for s in samples]
        metrics["regime_classifier"] = self._train_one(
            self._regime, self.cfg.regime_classifier, samples, regime_labels
        )

        return metrics

    def _train_one(
        self,
        model: ShadowModel,
        model_cfg,
        samples: list,
        labels: list[int],
    ) -> dict:
        """Train one model honoring its :class:`~src.ml.config.ModelCfg` (audit M35)."""
        if not model_cfg.enabled:
            return {"status": "skipped", "reason": "model disabled in ML config"}
        if len(samples) < model_cfg.min_train_samples:
            return {
                "status": "skipped",
                "reason": (
                    f"insufficient samples: n={len(samples)} < "
                    f"min_train_samples={model_cfg.min_train_samples}"
                ),
            }

        # Chronological hold-out per the configured test_fraction.
        n = len(samples)
        test_fraction = float(model_cfg.test_fraction)
        split = max(1, int(n * (1.0 - test_fraction))) if 0.0 < test_fraction < 1.0 else n
        order = sorted(range(n), key=lambda i: samples[i].candidate.decision_ts)
        train_idx, test_idx = order[:split], order[split:]

        train_labels = [labels[i] for i in train_idx]
        if len(set(train_labels)) < 2:
            return {
                "status": "skipped",
                "reason": "single-class training data — nothing to learn",
            }

        feat_names = feature_names_for(model_cfg.features) if model_cfg.features else []
        candidates = [s.candidate for s in samples]
        X = build_feature_matrix(candidates, feat_names or None)
        X_train = [X[i] for i in train_idx]
        m = dict(model.train(X_train, train_labels, feat_names))
        m.setdefault("status", "trained")
        m["test_fraction"] = test_fraction

        if test_idx:
            X_test = [X[i] for i in test_idx]
            y_test = [labels[i] for i in test_idx]
            preds = model.predict(X_test)
            correct = sum(1 for p, y in zip(preds, y_test, strict=False) if p.label == y)
            m["test_samples"] = len(test_idx)
            m["test_accuracy"] = round(correct / len(test_idx), 4)
        else:
            m["test_samples"] = 0
        return m

    def run(
        self,
        candidates: list[Candidate],
        *,
        settings: Settings | None = None,
        write_to_db: bool = True,
        synthetic_source: bool = False,
    ) -> ShadowRunResult:
        """Run all shadow models; log predictions.  Returns :class:`ShadowRunResult`.

        ``synthetic_source=True`` tags every written shadow_log row with
        :data:`SYNTHETIC_CONTEXT_KEY` so real-performance readers can exclude it.
        Callers running on the reference dataset MUST pass it.
        """
        settings = settings or get_settings()
        config_version = settings.config_version
        model_version = self.cfg.model_version
        bundles: list[ShadowBundle] = []
        log_ids: list[int] = []

        # Compute feature matrices for each model. Disabled models (audit M35)
        # produce no predictions at all — their bundle slot stays None.
        def _predict(model: ShadowModel, model_cfg) -> list[ShadowPrediction | None]:
            if not model_cfg.enabled:
                return [None] * len(candidates)
            names = feature_names_for(model_cfg.features) if model_cfg.features else []
            X = build_feature_matrix(candidates, names or None)
            return list(model.predict(X))

        meta_preds = _predict(self._meta, self.cfg.meta_labeler)
        regime_preds = _predict(self._regime, self.cfg.regime_classifier)
        exec_preds = _predict(self._exec, self.cfg.exec_quality)
        strat_preds = _predict(self._strat, self.cfg.strategy_selector)
        sym_preds = _predict(self._sym, self.cfg.symbol_ranker)

        for i, cand in enumerate(candidates):
            bundle = ShadowBundle(
                candidate=cand,
                meta_label=meta_preds[i],
                regime=regime_preds[i],
                exec_quality=exec_preds[i],
                strategy=strat_preds[i],
                symbol=sym_preds[i],
            )
            bundles.append(bundle)

        if write_to_db:
            log_ids = _write_shadow_logs(
                bundles,
                model_version=model_version,
                config_version=config_version,
                synthetic_source=synthetic_source,
            )

        return ShadowRunResult(
            bundles=bundles,
            shadow_log_ids=log_ids,
            model_version=model_version,
            applied=False,  # NEVER True in Phase 9
        )


def _write_shadow_logs(
    bundles: list[ShadowBundle],
    *,
    model_version: str,
    config_version: str,
    synthetic_source: bool = False,
) -> list[int]:
    """Persist shadow log entries; return inserted IDs."""
    from src.db.base import session_scope
    from src.db.models import ShadowLog

    from .features import candidate_to_row

    ids: list[int] = []
    ts_now = datetime.now(UTC)

    _PRED_ATTRS: list[tuple[str, str]] = [
        ("meta_label", "meta_labeler"),
        ("regime", "regime_classifier"),
        ("exec_quality", "exec_quality"),
        ("strategy", "strategy_selector"),
        ("symbol", "symbol_ranker"),
    ]

    with session_scope() as session:
        for bundle in bundles:
            ctx = candidate_to_row(bundle.candidate)
            if synthetic_source:
                ctx[SYNTHETIC_CONTEXT_KEY] = 1.0
            for attr, mtype in _PRED_ATTRS:
                pred: ShadowPrediction | None = getattr(bundle, attr)
                if pred is None:
                    continue
                row = ShadowLog(
                    ts=ts_now,
                    model_id=pred.model_id,
                    model_version=model_version,
                    model_type=mtype,
                    mode=_MODE,
                    symbol=bundle.candidate.symbol,
                    context_features=ctx,
                    prediction=pred.to_dict(),
                    confidence=pred.probability,
                    deterministic_baseline=None,
                    applied=False,  # NEVER True in Phase 9
                    config_version=config_version,
                    created_at=ts_now,
                )
                session.add(row)
                session.flush()
                ids.append(row.id)

    return ids
