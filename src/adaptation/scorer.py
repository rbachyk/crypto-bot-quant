"""Shadow scorer — realized vs projected outcome scoring (AGENTS.md Section 21.3, 21.5).

Computes promotion metrics for the LEARN-PROMO-S gate:
  - Walk-forward scoring: does the shadow policy beat the baseline across folds?
  - Locked hold-out edge: is the hold-out expectancy non-negative?
  - Calibration: is the Brier score below the configured threshold?
  - Drift: is the per-window mean drift within the configured limit?

Unit contract (audit M31, documented in :mod:`src.adaptation.policy_base`):
``projected_outcome`` and ``realized_outcome`` are both R-values — drift and
underperformance are R-to-R comparisons. ``win_probability`` is the policy's
genuine P(positive R) when it has one; the Brier score is computed ONLY from
it. Decisions without a probability are excluded from calibration; when no
decision carries one, calibration is *not evaluated* (``calibration_passed``
is None) rather than fake-passed.

The scorer operates on :class:`LearnerLogEntry` rows persisted by
:mod:`src.adaptation.store`. It never touches live trading.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class ShadowDecision:
    """One logged learner decision with its projected and realized outcomes."""

    ts: datetime
    symbol: str | None
    projected_outcome: float  # policy's own expectation, R-units (M31 contract)
    realized_outcome: float | None  # realized R; filled post-trade; None if not yet known
    take: bool
    mode: str
    win_probability: float | None = None  # policy's P(positive R), when it emits one


@dataclass
class FoldScore:
    """Scoring result for one walk-forward fold."""

    fold_index: int
    n_decisions: int
    learner_mean_realized: float
    baseline_mean_realized: float
    beats_baseline: bool
    brier_score: float | None = None


@dataclass
class ScorerResult:
    """Full scoring result for the LEARN-PROMO-S gate."""

    n_decisions: int
    n_with_outcome: int
    folds: list[FoldScore]
    folds_passed: int
    holdout_edge: float | None
    holdout_passed: bool
    brier_score: float | None
    calibration_passed: bool | None  # None = not evaluated (no win_probability data)
    drift_scores: list[float]
    max_drift: float
    drift_passed: bool
    promotion_eligible: bool
    note: str


def score_shadow_decisions(
    decisions: list[ShadowDecision],
    *,
    n_folds: int = 4,
    min_holdout_edge: float = 0.0,
    calibration_max_brier: float = 0.30,
    max_drift_per_window: float = 0.15,
    drift_window: int = 20,
    baseline_mean: float = 0.0,
    min_shadow_decisions: int | None = None,
    min_wf_folds_positive: int | None = None,
) -> ScorerResult:
    """Score shadow decisions against eligibility criteria (Section 21.3).

    ``baseline_mean``: the deterministic system's mean realized R (from the
    backtest / walk-forward); used as the comparison baseline.

    ``min_shadow_decisions``: minimum realized outcomes required for
    ``promotion_eligible`` (adaptation.yaml ``scoring.min_shadow_decisions``;
    audit M34). Defaults to 2 when not provided.

    ``min_wf_folds_positive``: walk-forward folds that must beat the baseline
    for eligibility (``scoring.min_wf_folds_positive``; audit M34). Defaults
    to the legacy ``max(1, n_folds - 2)`` when not provided.

    Walk-forward logic: the decisions are split into ``n_folds`` chronological
    segments; the last segment is the locked hold-out (evaluated once).
    """
    decided = [d for d in decisions if d.realized_outcome is not None]
    n_total = len(decisions)
    n_with_outcome = len(decided)
    min_outcomes = min_shadow_decisions if min_shadow_decisions is not None else 2
    required_folds = (
        min_wf_folds_positive if min_wf_folds_positive is not None else max(1, n_folds - 2)
    )

    if n_with_outcome < 2:
        return ScorerResult(
            n_decisions=n_total,
            n_with_outcome=n_with_outcome,
            folds=[],
            folds_passed=0,
            holdout_edge=None,
            holdout_passed=False,
            brier_score=None,
            calibration_passed=None,
            drift_scores=[],
            max_drift=0.0,
            drift_passed=False,
            promotion_eligible=False,
            note="insufficient realized outcomes for scoring",
        )

    # -- walk-forward folds -------------------------------------------------- #
    folds_size = max(1, n_with_outcome // max(1, n_folds))
    fold_results: list[FoldScore] = []
    # Reserve the last fold as the locked hold-out.
    train_decisions = decided[:-folds_size] if len(decided) > folds_size else []
    holdout_decisions = decided[-folds_size:]

    # n_folds <= 1 → everything is hold-out, no walk-forward folds (guards the
    # n_folds - 1 divisor; audit M34).
    if train_decisions and n_folds > 1:
        fold_chunk = max(1, len(train_decisions) // (n_folds - 1))
        for i in range(n_folds - 1):
            chunk = train_decisions[i * fold_chunk : (i + 1) * fold_chunk]
            if not chunk:
                continue
            realized = [float(d.realized_outcome) for d in chunk if d.realized_outcome is not None]
            mean_r = sum(realized) / len(realized) if realized else 0.0
            beats = mean_r > baseline_mean
            brier = _brier(chunk)
            fold_results.append(
                FoldScore(
                    fold_index=i,
                    n_decisions=len(chunk),
                    learner_mean_realized=mean_r,
                    baseline_mean_realized=baseline_mean,
                    beats_baseline=beats,
                    brier_score=brier,
                )
            )

    folds_passed = sum(1 for f in fold_results if f.beats_baseline)

    # -- hold-out ------------------------------------------------------------ #
    holdout_realized = [
        float(d.realized_outcome) for d in holdout_decisions if d.realized_outcome is not None
    ]
    holdout_edge = sum(holdout_realized) / len(holdout_realized) if holdout_realized else None
    holdout_passed = holdout_edge is not None and holdout_edge >= min_holdout_edge

    # -- calibration (Brier on decisions carrying a REAL probability) -------- #
    brier_all = _brier(decided)
    # None = not evaluated — no win_probability data (M31); never a fake pass.
    calibration_passed: bool | None = (
        None if brier_all is None else brier_all <= calibration_max_brier
    )

    # -- drift monitoring (R-to-R; M31 contract) ------------------------------ #
    drift_scores: list[float] = []
    for i in range(0, len(decided), drift_window):
        window = decided[i : i + drift_window]
        if len(window) < 2:
            continue
        diffs = [
            abs(d.projected_outcome - d.realized_outcome)  # type: ignore[arg-type]
            for d in window
            if d.realized_outcome is not None
        ]
        if diffs:
            drift_scores.append(sum(diffs) / len(diffs))

    max_drift = max(drift_scores, default=0.0)
    drift_passed = max_drift <= max_drift_per_window

    promotion_eligible = (
        folds_passed >= required_folds
        and holdout_passed
        and calibration_passed is not False  # None (not evaluated) does not block
        and drift_passed
        and n_with_outcome >= min_outcomes
    )

    notes = []
    if n_with_outcome < min_outcomes:
        notes.append(f"n_with_outcome {n_with_outcome} < min_shadow_decisions {min_outcomes}")
    if folds_passed < required_folds:
        notes.append(f"folds_passed {folds_passed} < min_wf_folds_positive {required_folds}")
    if not holdout_passed:
        notes.append(f"hold-out edge {holdout_edge} < {min_holdout_edge}")
    if calibration_passed is None:
        notes.append("calibration not evaluated (no win_probability emitted)")
    elif not calibration_passed:
        notes.append(f"brier {brier_all:.3f} > {calibration_max_brier}")
    if not drift_passed:
        notes.append(f"max_drift {max_drift:.3f} > {max_drift_per_window}")

    return ScorerResult(
        n_decisions=n_total,
        n_with_outcome=n_with_outcome,
        folds=fold_results,
        folds_passed=folds_passed,
        holdout_edge=holdout_edge,
        holdout_passed=holdout_passed,
        brier_score=brier_all,
        calibration_passed=calibration_passed,
        drift_scores=drift_scores,
        max_drift=max_drift,
        drift_passed=drift_passed,
        promotion_eligible=promotion_eligible,
        note="; ".join(notes) if notes else "OK",
    )


def _brier(decisions: list[ShadowDecision]) -> float | None:
    """Mean Brier score over decisions carrying a genuine ``win_probability``.

    Audit M31: previously ``projected_outcome`` (an R-value) was squashed
    through a sigmoid and treated as P(positive) — sigmoid of a small R is
    ≈ 0.5, making the calibration gate vacuous. Now only decisions where the
    policy emitted a real probability are scored; returns None (calibration
    not evaluated) when there are none.
    """
    valid = [
        d for d in decisions if d.realized_outcome is not None and d.win_probability is not None
    ]
    if not valid:
        return None
    total = 0.0
    for d in valid:
        p = float(d.win_probability)  # type: ignore[arg-type]
        y = 1.0 if d.realized_outcome > 0 else 0.0  # type: ignore[operator]
        total += (p - y) ** 2
    return total / len(valid)
