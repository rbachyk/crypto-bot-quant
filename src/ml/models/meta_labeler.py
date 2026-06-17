"""Meta-labeling shadow model (AGENTS.md Section 20 "best first use").

The meta-labeler receives a deterministic candidate and estimates whether to
take (1) or skip (0) it.  It answers "should we?", never "which direction?".

Model class: LogisticRegression (preferred simple model per AGENTS.md Appendix C).
ML Stage 2: Shadow mode — predictions are logged to shadow_log; never applied.
"""

from __future__ import annotations

from typing import Any

from .base import ShadowPrediction, _SklearnModelMixin


class MetaLabeler(_SklearnModelMixin):
    """Binary classifier: take (1) or skip (0) a deterministic candidate."""

    model_type: str = "meta_labeler"

    def __init__(self, model_id: str, model_version: str) -> None:
        self.model_id = model_id
        self.model_version = model_version
        self._clf: Any = None
        self._metrics: dict = {}
        self._feature_names: list[str] = []
        self.is_trained = False

    def train(
        self,
        X: list[list[float]],
        y: list[int],
        feature_names: list[str],
    ) -> dict:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score,
            brier_score_loss,
            precision_score,
            recall_score,
        )

        self._feature_names = list(feature_names)
        clf = LogisticRegression(max_iter=500, random_state=42, C=1.0)
        clf.fit(X, y)
        preds = clf.predict(X)
        probas = clf.predict_proba(X)[:, 1]
        self._clf = clf
        self.is_trained = True
        self._metrics = {
            "accuracy": round(float(accuracy_score(y, preds)), 4),
            "precision": round(float(precision_score(y, preds, zero_division=0.0)), 4),
            "recall": round(float(recall_score(y, preds, zero_division=0.0)), 4),
            "brier_score": round(float(brier_score_loss(y, probas)), 4),
            "train_samples": len(y),
            "positive_rate": round(sum(y) / max(len(y), 1), 4),
        }
        return self._metrics

    def predict(self, X: list[list[float]]) -> list[ShadowPrediction]:
        if self._clf is None:
            return [
                ShadowPrediction(
                    model_id=self.model_id,
                    model_type=self.model_type,
                    label=1,
                    probability=0.5,
                    rationale="untrained — defaulting to take",
                )
                for _ in X
            ]
        labels = self._clf.predict(X)
        probas = self._clf.predict_proba(X)[:, 1]
        return [
            ShadowPrediction(
                model_id=self.model_id,
                model_type=self.model_type,
                label=int(lbl),
                probability=round(float(prob), 4),
                rationale="take" if lbl == 1 else "skip",
            )
            for lbl, prob in zip(labels, probas, strict=False)
        ]

    def predict_proba(self, X: list[list[float]]) -> list[float]:
        if self._clf is None:
            return [0.5] * len(X)
        return [round(float(p), 4) for p in self._clf.predict_proba(X)[:, 1]]
