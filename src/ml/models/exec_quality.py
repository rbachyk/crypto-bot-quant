"""Execution quality shadow model (AGENTS.md Section 18, Phase 9).

Predicts whether execution conditions are favorable (1=good) or unfavorable
(0=poor) based on spread, slippage estimate, and volatility.  Shadow-only.
"""

from __future__ import annotations

from typing import Any

from .base import ShadowPrediction, _SklearnModelMixin


class ExecQualityModel(_SklearnModelMixin):
    """Binary classifier: good (1) or poor (0) execution conditions."""

    model_type: str = "exec_quality"

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
        from sklearn.metrics import accuracy_score

        self._feature_names = list(feature_names)
        clf = LogisticRegression(max_iter=500, random_state=42)
        clf.fit(X, y)
        preds = clf.predict(X)
        self._clf = clf
        self.is_trained = True
        self._metrics = {
            "accuracy": round(float(accuracy_score(y, preds)), 4),
            "train_samples": len(y),
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
                    rationale="untrained — defaulting to good",
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
                rationale="good_execution" if lbl == 1 else "poor_execution",
            )
            for lbl, prob in zip(labels, probas, strict=False)
        ]

    def predict_proba(self, X: list[list[float]]) -> list[float]:
        if self._clf is None:
            return [0.5] * len(X)
        return [round(float(p), 4) for p in self._clf.predict_proba(X)[:, 1]]
