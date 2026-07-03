"""Incremental logistic-regression policy for meta-filter weighting (AGENTS.md Section 21.5).

Uses scikit-learn's :class:`~sklearn.linear_model.SGDClassifier` (``loss='log_loss'``)
for online updates.  The policy outputs a ``take`` decision and a ``size_bucket``
based on the predicted probability of a good outcome.

Feature normalization (audit M37): raw context features span ~4 orders of
magnitude (edge fractions ~1e-3 vs spread_bps ~1e1), which destabilises
per-sample SGD. A fit-once ``StandardScaler`` does not suit ``partial_fit``
streams, so the policy keeps ONLINE running mean/variance per feature
(Welford's algorithm), updated on every training sample and applied before
both ``partial_fit`` and ``predict``. The running stats are persisted in the
snapshot (``norm_stats``); legacy snapshots without them load cleanly and
start accumulating stats from zero.

Shadow mode: ``update()`` applies gradient steps as normal (the learner trains
in shadow), but the resulting action is logged and not applied to real orders.
The controller enforces the no-apply rule; this class does not need to check it.

Outcome-projection contract (audit M31, see policy_base): this policy emits a
genuine probability — ``BoundedAction.win_probability`` — once trained. It has
no R-scale outcome estimate, so ``projected_outcome_r`` stays None.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import Any

from src.adaptation.action_space import BoundedAction
from src.adaptation.policy_base import Context, Outcome

try:
    from sklearn.linear_model import SGDClassifier

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

# Snapshot format version. v2 adds Welford running stats ("norm_stats");
# v1 snapshots (no version key) carried an unused StandardScaler.
_SNAPSHOT_VERSION = 2


class _WelfordScaler:
    """Online per-feature standardisation via Welford running mean/variance.

    ``observe`` folds one sample into the running stats; ``transform`` applies
    ``(x - mean) / std`` with the CURRENT stats. Until two samples are seen the
    transform is the identity (no meaningful variance estimate yet).
    """

    def __init__(self, n: int = 0, mean: list[float] | None = None, m2: list[float] | None = None):
        self.n = n
        self.mean = list(mean) if mean is not None else []
        self.m2 = list(m2) if m2 is not None else []

    def observe(self, x: list[float]) -> None:
        if not self.mean:
            self.mean = [0.0] * len(x)
            self.m2 = [0.0] * len(x)
        self.n += 1
        for i, v in enumerate(x):
            delta = v - self.mean[i]
            self.mean[i] += delta / self.n
            self.m2[i] += delta * (v - self.mean[i])

    def transform(self, x: list[float]) -> list[float]:
        if self.n < 2:
            return list(x)
        out = []
        for i, v in enumerate(x):
            var = self.m2[i] / (self.n - 1)
            std = var**0.5
            out.append((v - self.mean[i]) / std if std > 1e-12 else v - self.mean[i])
        return out

    def state(self) -> dict:
        return {"n": self.n, "mean": list(self.mean), "m2": list(self.m2)}

    @classmethod
    def from_state(cls, state: dict | None) -> _WelfordScaler:
        if not state:
            return cls()
        return cls(n=int(state["n"]), mean=state["mean"], m2=state["m2"])


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
    _scaler: _WelfordScaler = field(default_factory=_WelfordScaler, init=False, repr=False)
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
        else:
            self._model = None

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
        trained_prob = False
        if _SKLEARN_AVAILABLE and self._model is not None and self._n_updates >= 2:
            import numpy as np

            feats = self._scaler.transform(self._extract_features(ctx))
            x = np.array([feats], dtype=float)
            try:
                prob = float(self._model.predict_proba(x)[0, 1])
                trained_prob = True
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
            # M31 contract: a REAL probability only once the model is trained;
            # never emit the 0.5 placeholder as if it were calibrated.
            win_probability=prob if trained_prob else None,
            projected_outcome_r=None,  # this policy has no R-scale estimate
        )

    def update(self, ctx: Context, action: BoundedAction, outcome: Outcome) -> None:
        """Incremental update on a realised outcome."""
        if not _SKLEARN_AVAILABLE or outcome.realized_pnl_r is None:
            return
        import numpy as np

        raw = self._extract_features(ctx)
        # Fold the sample into the running stats FIRST, then standardise with
        # the updated stats (audit M37: normalization actually applied).
        self._scaler.observe(raw)
        x = np.array([self._scaler.transform(raw)], dtype=float)
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
                "snapshot_version": _SNAPSHOT_VERSION,
                "model": self._model,
                "norm_stats": self._scaler.state(),
                "n_updates": self._n_updates,
                "learner_id": self.learner_id,
                "learner_version": self.learner_version,
                "feature_names": self.feature_names,
            }
        )

    def load(self, blob: bytes) -> None:
        state = pickle.loads(blob)  # noqa: S301
        self._model = state["model"]
        # v2 snapshots persist Welford stats; v1 (legacy) snapshots carried an
        # unused sklearn StandardScaler under "scaler" — ignore it and start
        # running stats fresh (the SGD model adapts online).
        self._scaler = _WelfordScaler.from_state(state.get("norm_stats"))
        self._n_updates = state["n_updates"]
        self.learner_id = state["learner_id"]
        self.learner_version = state["learner_version"]
        self.feature_names = state["feature_names"]
