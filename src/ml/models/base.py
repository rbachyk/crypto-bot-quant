"""Base types for shadow ML models (AGENTS.md Section 20).

All models implement the :class:`ShadowModel` protocol: train on labeled
samples, predict on candidates, serialize/deserialize for the artifact
registry.  Models are always shadow-only in Phase 9.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(slots=True)
class ShadowPrediction:
    """One model prediction in SHADOW mode."""

    model_id: str
    model_type: str
    label: int  # 0 or 1 (take/skip, good/poor regime, etc.)
    probability: float  # confidence in the positive class
    rationale: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "label": self.label,
            "probability": round(self.probability, 4),
            "rationale": self.rationale,
            **self.extra,
        }


@runtime_checkable
class ShadowModel(Protocol):
    """Protocol every shadow ML model must satisfy (Section 20)."""

    model_id: str
    model_version: str
    model_type: str
    is_trained: bool

    def train(
        self,
        X: list[list[float]],
        y: list[int],
        feature_names: list[str],
    ) -> dict:
        """Train the model; return performance metrics dict."""
        ...

    def predict(
        self,
        X: list[list[float]],
    ) -> list[ShadowPrediction]:
        """Return one :class:`ShadowPrediction` per row in *X*."""
        ...

    def predict_proba(self, X: list[list[float]]) -> list[float]:
        """Return probability of positive class for each row."""
        ...

    def snapshot(self) -> bytes:
        """Serialize model state to bytes for the artifact registry."""
        ...

    def load(self, blob: bytes) -> None:
        """Restore model state from bytes."""
        ...

    def performance_report(self) -> dict:
        """Return the last training performance metrics."""
        ...


def _pickle_snapshot(obj: object) -> bytes:
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


def _pickle_load(blob: bytes) -> object:
    return pickle.loads(blob)  # noqa: S301 — internal use only, no untrusted input


class _SklearnModelMixin:
    """Mixin providing default snapshot/load/performance_report for sklearn models."""

    _clf: object = None
    _metrics: dict = {}
    _feature_names: list[str] = []
    is_trained: bool = False

    def snapshot(self) -> bytes:
        return _pickle_snapshot({"clf": self._clf, "features": self._feature_names})

    def load(self, blob: bytes) -> None:
        state = _pickle_load(blob)
        self._clf = state["clf"]  # type: ignore[index]
        self._feature_names = state["features"]  # type: ignore[index]
        self.is_trained = self._clf is not None

    def performance_report(self) -> dict:
        return dict(self._metrics)
