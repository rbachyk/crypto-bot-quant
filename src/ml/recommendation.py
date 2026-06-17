"""ML Stage 3: Recommendation Mode (AGENTS.md Section 20, ML Stage 3).

Converts shadow model predictions into structured dashboard recommendations.
Recommendations are NEVER automatically applied — they surface in the
dashboard and reports for the operator to review. No live trading behavior
is changed.

ML Stage 3 contract (Section 20):
  * recommend skip/take — shown on dashboard; operator decides
  * recommend preferred strategy — informational only
  * recommend preferred symbol — informational only
  * recommend execution route — informational only
  * recommend risk bucket reduction — informational only
  * NOT allowed: automatic behavior changes, risk increases, order placement
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

from src.ranking.candidate import Candidate

from .shadow import ShadowBundle

_MODE = "RECOMMEND"

RiskBucket = float  # one of 0.0 / 0.25 / 0.5 / 1.0; never > 1.0 (Section 20)


@dataclass(slots=True)
class MLRecommendation:
    """One structured ML recommendation for a deterministic candidate.

    Produced by :class:`RecommendationEngine` from a :class:`ShadowBundle`.
    Always carries ``applied=False`` — Stage 3 recommendations are never
    auto-applied (Section 20 ML Stage 3).
    """

    id: str
    ts: datetime
    model_version: str
    config_version: str
    symbol: str
    strategy: str
    recommend_take: bool  # meta-labeler says take (True) or skip (False)
    confidence: float  # meta-labeler probability of take
    risk_bucket: RiskBucket  # recommended size multiplier (never > 1.0)
    exec_style: Literal["maker", "taker", "passive_then_taker"]
    rationale: str  # human-readable explanation for the dashboard
    mode: str = _MODE
    applied: bool = False  # ALWAYS False for Stage 3
    source_features: dict = field(default_factory=dict)


@dataclass
class RecommendationRunResult:
    """Result of one recommendation pass over a batch of candidates."""

    recommendations: list[MLRecommendation]
    model_version: str
    log_ids: list[int] = field(default_factory=list)
    applied: bool = False  # always False


class RecommendationEngine:
    """Converts shadow bundles into structured ML recommendations.

    Usage::

        engine = RecommendationEngine(model_version="ml_shadow_0001",
                                      config_version="cfg_0001")
        result = engine.run(shadow_bundles)
        # result.applied is always False
        # result.recommendations → dashboard / reports
    """

    _RISK_BUCKET_THRESHOLDS = [
        (0.85, 1.0),
        (0.70, 0.5),
        (0.50, 0.25),
    ]

    def __init__(
        self,
        model_version: str = "ml_shadow_0001",
        config_version: str = "cfg_0001",
    ) -> None:
        self.model_version = model_version
        self.config_version = config_version

    def run(
        self,
        bundles: list[ShadowBundle],
        *,
        write_to_db: bool = True,
    ) -> RecommendationRunResult:
        """Convert shadow bundles into structured recommendations.

        Parameters
        ----------
        bundles:
            Shadow bundles from :class:`~src.ml.shadow.ShadowPredictor`.
        write_to_db:
            Persist recommendation entries to ``shadow_logs`` with
            ``mode="RECOMMEND"`` and ``applied=False``.
        """
        ts_now = datetime.now(UTC)
        recs: list[MLRecommendation] = []

        for bundle in bundles:
            rec = self._bundle_to_rec(bundle, ts_now)
            recs.append(rec)

        log_ids: list[int] = []
        if write_to_db:
            log_ids = _write_recommendation_logs(
                recs,
                bundles,
                model_version=self.model_version,
                config_version=self.config_version,
            )

        return RecommendationRunResult(
            recommendations=recs,
            model_version=self.model_version,
            log_ids=log_ids,
            applied=False,  # NEVER True for Stage 3
        )

    def _bundle_to_rec(self, bundle: ShadowBundle, ts: datetime) -> MLRecommendation:
        cand: Candidate = bundle.candidate
        meta = bundle.meta_label

        # Default: take=True (pass-through) when no meta-labeler prediction.
        prob_take = meta.probability if meta is not None else 1.0
        label_take = (meta.label == 1) if meta is not None else True

        risk_bucket = self._risk_bucket(prob_take, label_take)
        exec_style = self._exec_style(bundle)
        rationale = self._build_rationale(bundle, prob_take, label_take)

        from src.ml.features import candidate_to_row

        return MLRecommendation(
            id=f"rec_{cand.symbol}_{ts.strftime('%Y%m%dT%H%M%S%f')}",
            ts=ts,
            model_version=self.model_version,
            config_version=self.config_version,
            symbol=cand.symbol,
            strategy=cand.strategy,
            recommend_take=label_take,
            confidence=round(prob_take, 4),
            risk_bucket=risk_bucket,
            exec_style=exec_style,
            rationale=rationale,
            mode=_MODE,
            applied=False,
            source_features=candidate_to_row(cand),
        )

    def _risk_bucket(self, prob_take: float, label_take: bool) -> RiskBucket:
        """Recommend a risk bucket based on meta-labeler confidence.

        Never returns > 1.0 (Section 20: ML may not increase risk).
        Returns 0.0 when the model recommends skip.
        """
        if not label_take:
            return 0.0
        for threshold, bucket in self._RISK_BUCKET_THRESHOLDS:
            if prob_take >= threshold:
                return bucket  # type: ignore[return-value]
        return 0.0

    def _exec_style(self, bundle: ShadowBundle) -> Literal["maker", "taker", "passive_then_taker"]:
        """Recommend execution style based on exec_quality model."""
        if bundle.exec_quality is None:
            return "passive_then_taker"
        if bundle.exec_quality.label == 1 and bundle.exec_quality.probability >= 0.70:
            return "maker"
        if bundle.exec_quality.probability < 0.40:
            return "taker"
        return "passive_then_taker"

    def _build_rationale(self, bundle: ShadowBundle, prob_take: float, label_take: bool) -> str:
        parts: list[str] = []
        if not label_take:
            parts.append(f"meta_labeler: skip (p_take={prob_take:.2f})")
        else:
            parts.append(f"meta_labeler: take (p_take={prob_take:.2f})")
        if bundle.regime is not None:
            parts.append(f"regime_clf: class={bundle.regime.label}")
        if bundle.exec_quality is not None:
            parts.append(f"exec_quality: p={bundle.exec_quality.probability:.2f}")
        if bundle.strategy is not None:
            parts.append(f"strategy_sel: label={bundle.strategy.label}")
        if bundle.symbol is not None:
            parts.append(f"symbol_rnk: label={bundle.symbol.label}")
        return "; ".join(parts)


# --------------------------------------------------------------------------- #
# DB persistence                                                               #
# --------------------------------------------------------------------------- #


def _write_recommendation_logs(
    recs: list[MLRecommendation],
    bundles: list[ShadowBundle],
    *,
    model_version: str,
    config_version: str,
) -> list[int]:
    """Persist recommendations as shadow_log entries with mode=RECOMMEND."""
    from src.db.base import session_scope
    from src.db.models import ShadowLog

    ids: list[int] = []
    ts_now = datetime.now(UTC)

    with session_scope() as session:
        for rec, _bundle in zip(recs, bundles, strict=False):
            row = ShadowLog(
                ts=ts_now,
                model_id=f"recommendation_engine_{model_version}",
                model_version=model_version,
                model_type="recommendation",
                mode=_MODE,
                symbol=rec.symbol,
                context_features=rec.source_features,
                prediction={
                    "recommend_take": rec.recommend_take,
                    "confidence": rec.confidence,
                    "risk_bucket": rec.risk_bucket,
                    "exec_style": rec.exec_style,
                    "rationale": rec.rationale,
                },
                confidence=rec.confidence,
                deterministic_baseline=None,
                applied=False,  # NEVER True for Stage 3
                config_version=config_version,
                created_at=ts_now,
            )
            session.add(row)
            session.flush()
            ids.append(row.id)

    return ids
