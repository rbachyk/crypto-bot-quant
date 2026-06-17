"""Incremental logistic-regression policy for meta-filter weighting (AGENTS.md Section 21.5).

Uses scikit-learn's :class:`~sklearn.linear_model.SGDClassifier` (``loss='log_loss'``)
for online updates.  The policy outputs a ``take`` decision and a ``size_bucket``
based on the predicted probability of a good outcome.

Shadow mode: ``update()`` applies gradient steps as normal (the learner trains
in shadow), but the resulting action is logged and not applied to real orders.
The controller enforces the no-apply rule; this class does not need to check it.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import Any

from src.adaptation.action_space import BoundedAction
from src.adaptation.policy_base import Context, Outcome

try:
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler

    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


_DEFAULT_FEATURES = [
    "signal_strength",
    "expected_edge_frac",
    "spread_bps",
    "slippage_est",
    "atr_pct",
    "funding_z",
]


@dataclass
class OnlineLogRegPolicy:
    """SGD-based incremental logistic regression (meta-filter weighting).

    Trained online: every logged outcome with ``realized_pnl_r`` known updates
    the classifier.  In SHADOW mode :meth:`update` trains the model; :meth:`decide`
    produces a ``take`` decision and size bucket.
    """

    learner_id: str = "online_logreg_v1"
    learner_version: str = "learner_0001"
    feature_names: list[str] = field(default_factory=lambda: list(_DEFAULT_FEATURES))
    take_threshold: float = 0.55  # probability above which we take
    _model: Any = field(default=None, init=False, repr=False)
    _scaler: Any = field(default=None, init=False, repr=False)
    _n_updates: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if _SKLEARN_AVAILABLE:
            self._model = SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=0.01,
                random_state=42,
                warm_start=True,
                n_iter_no_change=10,
                max_iter=1,
            )
            self._scaler = StandardScaler()
        else:
            self._model = None
            self._scaler = None

    # ---------------------------------------------------------------------- #
    def _extract_features(self, ctx: Context) -> list[float]:
        mapping = {
            "signal_strength": ctx.signal_strength,
            "expected_edge_frac": ctx.expected_edge_frac,
            "spread_bps": ctx.spread_bps,
            "slippage_est": ctx.slippage_est,
            "atr_pct": ctx.atr_pct,
            "funding_z": ctx.funding_z,
        }
        return [mapping.get(f, ctx.extra.get(f, 0.0)) for f in self.feature_names]

    def decide(self, ctx: Context) -> BoundedAction:
        """Produce a shadow action. In SHADOW mode this is logged but never applied."""
        prob = 0.5  # default when model not yet trained
        if _SKLEARN_AVAILABLE and self._model is not None and self._n_updates >= 2:
            import numpy as np

            x = np.array([self._extract_features(ctx)], dtype=float)
            try:
                prob = float(self._model.predict_proba(x)[0, 1])
            except Exception:  # noqa: BLE001
                prob = 0.5

        take = prob >= self.take_threshold
        if prob >= 0.75:
            bucket = 1.0
        elif prob >= 0.60:
            bucket = 0.5
        elif prob >= self.take_threshold:
            bucket = 0.25
        else:
            bucket = 0.0

        return BoundedAction(
            strategy_weights={},
            size_bucket=bucket,
            take=take,
            exec_style="maker",
            param_nudges={},
            learner_id=self.learner_id,
            learner_version=self.learner_version,
            mode="SHADOW",
            rationale=f"logreg p(good)={prob:.3f}",
        )

    def update(self, ctx: Context, action: BoundedAction, outcome: Outcome) -> None:
        """Incremental update on a realised outcome."""
        if not _SKLEARN_AVAILABLE or outcome.realized_pnl_r is None:
            return
        import numpy as np

        x = np.array([self._extract_features(ctx)], dtype=float)
        y = np.array([1 if outcome.realized_pnl_r > 0 else 0])
        try:
            # partial_fit requires the full classes list on first call.
            if self._n_updates == 0:
                self._model.partial_fit(x, y, classes=np.array([0, 1]))
            else:
                self._model.partial_fit(x, y)
            self._n_updates += 1
        except Exception:  # noqa: BLE001
            pass

    def snapshot(self) -> bytes:
        return pickle.dumps(
            {
                "model": self._model,
                "scaler": self._scaler,
                "n_updates": self._n_updates,
                "learner_id": self.learner_id,
                "learner_version": self.learner_version,
                "feature_names": self.feature_names,
            }
        )

    def load(self, blob: bytes) -> None:
        state = pickle.loads(blob)  # noqa: S301
        self._model = state["model"]
        self._scaler = state["scaler"]
        self._n_updates = state["n_updates"]
        self.learner_id = state["learner_id"]
        self.learner_version = state["learner_version"]
        self.feature_names = state["feature_names"]
