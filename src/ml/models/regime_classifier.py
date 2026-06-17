"""Regime classifier shadow model (AGENTS.md Section 11, Phase 9).

Shadow-only: classifies market regimes to compare against the deterministic
regime engine.  Predictions are logged but do NOT affect trading decisions.

Model class: RandomForestClassifier (good at non-linear regime boundaries).
"""

from __future__ import annotations

from typing import Any

from .base import ShadowPrediction, _SklearnModelMixin

_REGIME_LABELS: list[str] = [
    "low_vol_range",
    "trend",
    "high_vol_expansion",
    "high_vol_chop",
    "market_wide_impulse",
]


class RegimeClassifier(_SklearnModelMixin):
    """Multi-class regime classifier (shadow mode only)."""

    model_type: str = "regime_classifier"

    def __init__(self, model_id: str, model_version: str) -> None:
        self.model_id = model_id
        self.model_version = model_version
        self._clf: Any = None
        self._metrics: dict = {}
        self._feature_names: list[str] = []
        self._classes: list[str] = []
        self.is_trained = False

    def train(
        self,
        X: list[list[float]],
        y: list[int],
        feature_names: list[str],
    ) -> dict:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score

        self._feature_names = list(feature_names)
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X, y)
        preds = clf.predict(X)
        self._clf = clf
        self._classes = [str(c) for c in clf.classes_]
        self.is_trained = True
        self._metrics = {
            "accuracy": round(float(accuracy_score(y, preds)), 4),
            "train_samples": len(y),
            "n_classes": len(self._classes),
        }
        return self._metrics

    def predict(self, X: list[list[float]]) -> list[ShadowPrediction]:
        if self._clf is None:
            return [
                ShadowPrediction(
                    model_id=self.model_id,
                    model_type=self.model_type,
                    label=0,
                    probability=0.5,
                    rationale="untrained",
                )
                for _ in X
            ]
        labels = self._clf.predict(X)
        probas = self._clf.predict_proba(X)
        return [
            ShadowPrediction(
                model_id=self.model_id,
                model_type=self.model_type,
                label=int(lbl),
                probability=round(float(max(proba)), 4),
                rationale=(
                    f"regime={self._classes[int(lbl)]}"
                    if int(lbl) < len(self._classes)
                    else f"regime={lbl}"
                ),
                extra={
                    "regime_probas": {
                        cls: round(float(p), 4)
                        for cls, p in zip(self._classes, proba, strict=False)
                    }
                },
            )
            for lbl, proba in zip(labels, probas, strict=False)
        ]

    def predict_proba(self, X: list[list[float]]) -> list[float]:
        if self._clf is None:
            return [0.5] * len(X)
        return [round(float(max(row)), 4) for row in self._clf.predict_proba(X)]
