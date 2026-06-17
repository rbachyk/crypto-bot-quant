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
        """Train all models on labeled samples.  Returns per-model metrics."""
        from .features import feature_names_for
        from .labels import LabeledSample

        assert all(isinstance(s, LabeledSample) for s in samples), "samples must be LabeledSample"

        candidates = [s.candidate for s in samples]
        labels = [s.label for s in samples]

        metrics: dict[str, dict] = {}
        for model, cfg_key in (
            (self._meta, self.cfg.meta_labeler),
            (self._exec, self.cfg.exec_quality),
            (self._strat, self.cfg.strategy_selector),
            (self._sym, self.cfg.symbol_ranker),
        ):
            feat_names = feature_names_for(cfg_key.features) if cfg_key.features else []
            X = build_feature_matrix(candidates, feat_names or None)
            m = model.train(X, labels, feat_names)  # type: ignore[union-attr]
            metrics[model.model_type] = m  # type: ignore[union-attr]

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
        rc_feat = (
            feature_names_for(self.cfg.regime_classifier.features)
            if self.cfg.regime_classifier.features
            else []
        )
        X_rc = build_feature_matrix(candidates, rc_feat or None)
        m_rc = self._regime.train(X_rc, regime_labels, rc_feat)
        metrics["regime_classifier"] = m_rc

        return metrics

    def run(
        self,
        candidates: list[Candidate],
        *,
        settings: Settings | None = None,
        write_to_db: bool = True,
    ) -> ShadowRunResult:
        """Run all shadow models; log predictions.  Returns :class:`ShadowRunResult`."""
        settings = settings or get_settings()
        config_version = settings.config_version
        model_version = self.cfg.model_version
        bundles: list[ShadowBundle] = []
        log_ids: list[int] = []

        # Compute feature matrices for each model.
        def _X(feature_list: list[str]) -> list[list[float]]:
            names = feature_names_for(feature_list) if feature_list else []
            return build_feature_matrix(candidates, names or None)

        meta_preds = self._meta.predict(_X(self.cfg.meta_labeler.features))
        regime_preds = self._regime.predict(_X(self.cfg.regime_classifier.features))
        exec_preds = self._exec.predict(_X(self.cfg.exec_quality.features))
        strat_preds = self._strat.predict(_X(self.cfg.strategy_selector.features))
        sym_preds = self._sym.predict(_X(self.cfg.symbol_ranker.features))

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
                bundles, model_version=model_version, config_version=config_version
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
