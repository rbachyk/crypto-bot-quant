"""Shadow scorer — offline comparison vs deterministic baseline (AGENTS.md Section 20).

The scorer measures whether the meta-labeler improves over the deterministic
always-take baseline.  ML-PROMO gate criteria (Appendix A):
  * improves expectancy net of costs
  * preserves/raises profit factor
  * preserves/reduces max drawdown
  * does not remove most best trades
  * reduces worst-trade frequency / tail risk

All metrics are computed on the test split (never the training set).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .labels import LabeledSample, best_n_trades


@dataclass
class ShadowScorerResult:
    """Comparison of meta-labeler vs always-take baseline on the test split."""

    n_test: int
    n_baseline_taken: int
    n_model_taken: int

    baseline_expectancy: float
    model_expectancy: float
    expectancy_improvement: float  # model − baseline; positive = better

    baseline_profit_factor: float
    model_profit_factor: float
    profit_factor_ratio: float  # model / baseline; ≥ 1.0 = preserved

    baseline_worst_trade: float
    model_worst_trade: float
    tail_loss_ratio: float  # |model_worst| / |baseline_worst|; ≤ 1.0 = reduced

    baseline_best_n_pnl: list[float] = field(default_factory=list)
    model_best_n_pnl: list[float] = field(default_factory=list)
    best_trades_removed_pct: float = 0.0  # fraction of top-N removed

    passed: bool = False
    fail_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_test": self.n_test,
            "n_baseline_taken": self.n_baseline_taken,
            "n_model_taken": self.n_model_taken,
            "baseline_expectancy": round(self.baseline_expectancy, 4),
            "model_expectancy": round(self.model_expectancy, 4),
            "expectancy_improvement": round(self.expectancy_improvement, 4),
            "baseline_profit_factor": round(self.baseline_profit_factor, 4),
            "model_profit_factor": round(self.model_profit_factor, 4),
            "profit_factor_ratio": round(self.profit_factor_ratio, 4),
            "baseline_worst_trade": round(self.baseline_worst_trade, 4),
            "model_worst_trade": round(self.model_worst_trade, 4),
            "tail_loss_ratio": round(self.tail_loss_ratio, 4),
            "best_trades_removed_pct": round(self.best_trades_removed_pct, 4),
            "passed": self.passed,
            "fail_reasons": self.fail_reasons,
        }


class ShadowScorer:
    """Scores meta-labeler predictions against the always-take baseline.

    Uses only the test split of labeled samples (never the training set) to
    produce an honest out-of-sample performance comparison.
    """

    def __init__(
        self,
        min_improvement: float = 0.0,
        min_pf_ratio: float = 1.0,
        max_tail_loss_ratio: float = 1.0,
        max_best_removed_pct: float = 0.2,
        top_n: int = 10,
    ) -> None:
        self.min_improvement = min_improvement
        self.min_pf_ratio = min_pf_ratio
        self.max_tail_loss_ratio = max_tail_loss_ratio
        self.max_best_removed_pct = max_best_removed_pct
        self.top_n = top_n

    def score(
        self,
        test_samples: list[LabeledSample],
        model_predictions: list[int],
    ) -> ShadowScorerResult:
        """Compare model predictions vs always-take baseline on test samples."""
        n = len(test_samples)
        n_taken = sum(model_predictions)
        baseline_exp = _expectancy([s.realized_pnl for s in test_samples])
        _zipped = list(zip(test_samples, model_predictions, strict=False))
        taken_pnls = [s.realized_pnl for s, p in _zipped if p == 1]
        model_exp = _expectancy(taken_pnls)
        exp_improvement = model_exp - baseline_exp

        baseline_pf = _profit_factor([s.realized_pnl for s in test_samples])
        model_pf = _profit_factor(taken_pnls)
        pf_ratio = model_pf / max(baseline_pf, 1e-9)

        baseline_worst = min((s.realized_pnl for s in test_samples), default=0.0)
        model_worst = min(taken_pnls, default=0.0)
        tail_ratio = abs(model_worst) / max(abs(baseline_worst), 1e-9)

        # Best-trade preservation.
        top_n = min(self.top_n, n)
        baseline_best = best_n_trades(test_samples, top_n)
        baseline_best_ids = {id(s) for s in baseline_best}
        model_taken_samples = [s for s, p in _zipped if p == 1]
        model_best = best_n_trades(model_taken_samples, min(top_n, len(model_taken_samples)))
        model_best_ids = {id(s) for s in model_best}
        removed = len(baseline_best_ids - model_best_ids)
        removed_pct = removed / max(top_n, 1)

        fail_reasons: list[str] = []
        if exp_improvement < self.min_improvement:
            fail_reasons.append(
                f"expectancy did not improve: "
                f"model={model_exp:.4f} baseline={baseline_exp:.4f} "
                f"delta={exp_improvement:.4f}"
            )
        if pf_ratio < self.min_pf_ratio:
            fail_reasons.append(
                f"profit factor not preserved: "
                f"model={model_pf:.3f} baseline={baseline_pf:.3f} ratio={pf_ratio:.3f}"
            )
        if tail_ratio > self.max_tail_loss_ratio:
            fail_reasons.append(
                f"tail loss worsened: "
                f"model_worst={model_worst:.4f} baseline_worst={baseline_worst:.4f} "
                f"ratio={tail_ratio:.3f}"
            )
        if removed_pct > self.max_best_removed_pct:
            fail_reasons.append(
                f"removed too many best trades: {removed}/{top_n} "
                f"({removed_pct:.1%} > {self.max_best_removed_pct:.0%})"
            )

        return ShadowScorerResult(
            n_test=n,
            n_baseline_taken=n,
            n_model_taken=n_taken,
            baseline_expectancy=baseline_exp,
            model_expectancy=model_exp,
            expectancy_improvement=exp_improvement,
            baseline_profit_factor=baseline_pf,
            model_profit_factor=model_pf,
            profit_factor_ratio=pf_ratio,
            baseline_worst_trade=baseline_worst,
            model_worst_trade=model_worst,
            tail_loss_ratio=tail_ratio,
            baseline_best_n_pnl=[s.realized_pnl for s in baseline_best],
            model_best_n_pnl=[s.realized_pnl for s in model_best],
            best_trades_removed_pct=removed_pct,
            passed=len(fail_reasons) == 0,
            fail_reasons=fail_reasons,
        )


def _expectancy(pnls: list[float]) -> float:
    return sum(pnls) / len(pnls) if pnls else 0.0


def _profit_factor(pnls: list[float]) -> float:
    gains = sum(p for p in pnls if p > 0)
    losses = abs(sum(p for p in pnls if p < 0))
    if losses == 0:
        return math.inf if gains > 0 else 1.0
    return gains / losses
