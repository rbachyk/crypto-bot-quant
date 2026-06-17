"""Walk-forward validation harness (AGENTS.md Section 16, WF gate).

Splits the test window into ``folds`` disjoint, time-ordered out-of-sample
segments plus a **locked hold-out** (the most-recent ``holdout_frac``) that is
untouched during all folds and evaluated **exactly once** at the end. Each fold
is judged against the kill-criteria declared up front in ``configs/backtest.yaml``
(Section 16: "kill-criteria declared up front ... validation exists to reject").

WF passes only when ``>= min_folds_passed`` folds clear every kill-criterion AND
the locked hold-out is positive net of costs — i.e. the edge is stable across
periods, not isolated to one (Section 14/16). In Phase 4 the strategy parameters
are fixed (real optimization is Phase 5), so the "train" portion is notional and
each fold is a pure OOS evaluation on a distinct time segment; this proves the
*harness*. The same machinery validates real candidates in later phases.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.backtest.config import BacktestConfig, KillCriteria
from src.backtest.engine import SymbolInput
from src.backtest.metrics import BacktestReport
from src.backtest.service import rebase_window, run_engine
from src.exchange.metadata import MetadataConfig


@dataclass(slots=True)
class FoldResult:
    index: int
    lo_ts: int
    hi_ts: int
    passed: bool
    failures: list[str]
    report: BacktestReport


@dataclass(slots=True)
class WalkForwardResult:
    folds: list[FoldResult] = field(default_factory=list)
    holdout: FoldResult | None = None
    folds_passed: int = 0
    passed: bool = False
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "folds_passed": self.folds_passed,
            "n_folds": len(self.folds),
            "reasons": self.reasons,
            "folds": [
                {
                    "index": f.index,
                    "lo_ts": f.lo_ts,
                    "hi_ts": f.hi_ts,
                    "passed": f.passed,
                    "failures": f.failures,
                    "trade_count": f.report.trade_count,
                    "expectancy_r": f.report.expectancy_r,
                    "profit_factor": f.report.profit_factor,
                    "max_drawdown": f.report.max_drawdown,
                    "total_return": f.report.total_return,
                }
                for f in self.folds
            ],
            "holdout": None
            if self.holdout is None
            else {
                "lo_ts": self.holdout.lo_ts,
                "hi_ts": self.holdout.hi_ts,
                "passed": self.holdout.passed,
                "trade_count": self.holdout.report.trade_count,
                "expectancy_r": self.holdout.report.expectancy_r,
                "profit_factor": self.holdout.report.profit_factor,
                "net_pnl": self.holdout.report.net_pnl,
                "max_drawdown": self.holdout.report.max_drawdown,
            },
        }


def _evaluate_fold(report: BacktestReport, kc: KillCriteria) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if report.trade_count < kc.min_trades_per_fold:
        failures.append(f"trades {report.trade_count} < {kc.min_trades_per_fold}")
    if report.expectancy_r < kc.min_oos_expectancy_r:
        failures.append(f"expectancy_r {report.expectancy_r:.3f} < {kc.min_oos_expectancy_r}")
    if report.profit_factor < kc.min_oos_profit_factor:
        failures.append(f"profit_factor {report.profit_factor:.3f} < {kc.min_oos_profit_factor}")
    if report.max_drawdown > kc.max_oos_drawdown:
        failures.append(f"max_drawdown {report.max_drawdown:.3f} > {kc.max_oos_drawdown}")
    return (not failures), failures


def run_walk_forward(
    cfg: BacktestConfig,
    meta: MetadataConfig,
    inputs: list[SymbolInput],
) -> WalkForwardResult:
    wf = cfg.walk_forward
    kc = wf.kill_criteria
    iv = _iv(inputs)
    n_bars = max((len(s.bars) for s in inputs), default=0)
    out = WalkForwardResult()
    if n_bars < (wf.folds + 1) * max(1, kc.min_trades_per_fold):
        out.reasons.append("insufficient bars for the requested fold layout")

    span_ts = n_bars * iv
    holdout_bars = int(n_bars * wf.holdout_frac)
    test_end_ts = (n_bars - holdout_bars) * iv
    fold_span = test_end_ts // wf.folds if wf.folds > 0 else test_end_ts

    # 1) Out-of-sample folds across the pre-holdout window.
    for i in range(wf.folds):
        lo = i * fold_span
        hi = (i + 1) * fold_span if i < wf.folds - 1 else test_end_ts
        windowed = rebase_window(inputs, lo, hi)
        report = run_engine(cfg, meta, windowed, label=f"wf_fold_{i}").report
        passed, failures = _evaluate_fold(report, kc)
        out.folds.append(FoldResult(i, lo, hi, passed, failures, report))

    out.folds_passed = sum(1 for f in out.folds if f.passed)

    # 2) Locked hold-out — evaluated exactly once, here at the end (Section 16).
    holdout_report: BacktestReport | None = None
    if holdout_bars > 0:
        windowed = rebase_window(inputs, test_end_ts, span_ts)
        holdout_report = run_engine(cfg, meta, windowed, label="wf_holdout").report
        holdout_passed = holdout_report.expectancy_r > 0 and holdout_report.net_pnl > 0
        out.holdout = FoldResult(-1, test_end_ts, span_ts, holdout_passed, [], holdout_report)

    # 3) Verdict.
    if out.folds_passed < kc.min_folds_passed:
        out.reasons.append(
            f"only {out.folds_passed}/{len(out.folds)} folds passed (need {kc.min_folds_passed})"
        )
    if out.holdout is not None and not out.holdout.passed:
        out.reasons.append("locked hold-out not positive net of costs")
    if out.holdout is None:
        out.reasons.append("no locked hold-out evaluated (holdout_frac=0)")

    out.passed = not out.reasons
    return out


def _iv(inputs: list[SymbolInput]) -> int:
    for s in inputs:
        if len(s.bars) >= 2:
            return int(s.bars[1]["ts"] - s.bars[0]["ts"])
    return 1
