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
from src.backtest.strategy import PortfolioStrategy, Strategy
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

    def overfitting(self) -> dict:
        """Section-16 anti-overfitting controls over the folds (multiple-testing aware)."""
        from src.backtest.overfitting import overfitting_summary

        # Each fold is a trial; its out-of-sample expectancy is a per-trial 'Sharpe' proxy.
        trial_sharpes = [f.report.expectancy_r for f in self.folds] or [0.0]
        n_trades = sum(f.report.trade_count for f in self.folds)
        return overfitting_summary(trial_sharpes, trial_sharpes, n_trades).to_dict()

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "folds_passed": self.folds_passed,
            "n_folds": len(self.folds),
            "reasons": self.reasons,
            "overfitting": self.overfitting(),
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
    strategy: Strategy | PortfolioStrategy | None = None,
) -> WalkForwardResult:
    wf = cfg.walk_forward
    kc = wf.kill_criteria
    iv = _iv(inputs)
    out = WalkForwardResult()

    # Anchor folds to the ACTUAL data timestamp range, not a bar COUNT from ts=0. Real lake data
    # is rebased to the window start, so a contract listed mid-window has its first bar at a large
    # ts offset (not 0); laying folds over [0, n_bars*iv) would shift every fold off the real data
    # and evaluate the edge on empty pre-listing time. With dense data starting at ts=0 this
    # reduces exactly to the old bar-count arithmetic (data_lo=0, n_span=n_bars).
    with_bars = [s for s in inputs if s.bars]
    data_lo = min((s.bars[0]["ts"] for s in with_bars), default=0)
    data_hi = max((s.bars[-1]["ts"] for s in with_bars), default=-iv) + iv
    n_span = max(0, (data_hi - data_lo) // iv)  # grid slots the data actually spans
    holdout_slots = int(n_span * wf.holdout_frac)
    test_end_ts = data_lo + (n_span - holdout_slots) * iv
    fold_region = test_end_ts - data_lo
    fold_span = fold_region // wf.folds if wf.folds > 0 else fold_region

    # 1) Out-of-sample folds across the pre-holdout window.
    for i in range(wf.folds):
        lo = data_lo + i * fold_span
        hi = data_lo + (i + 1) * fold_span if i < wf.folds - 1 else test_end_ts
        windowed = rebase_window(inputs, lo, hi)
        report = run_engine(cfg, meta, windowed, strategy=strategy, label=f"wf_fold_{i}").report
        passed, failures = _evaluate_fold(report, kc)
        out.folds.append(FoldResult(i, lo, hi, passed, failures, report))

    out.folds_passed = sum(1 for f in out.folds if f.passed)

    # Trade-based adequacy (not a bars heuristic): a fold with too few REALIZED trades cannot
    # evaluate the edge. If too few folds clear the min-trades bar, the layout is too thin — FAIL
    # clearly (extend the window or reduce the fold count) rather than judging the edge on noise.
    folds_with_trades = sum(
        1 for f in out.folds if f.report.trade_count >= kc.min_trades_per_fold
    )
    if folds_with_trades < kc.min_folds_passed:
        out.reasons.append(
            f"insufficient trades: only {folds_with_trades}/{len(out.folds)} folds have "
            f">= {kc.min_trades_per_fold} trades (need {kc.min_folds_passed}) — too thin to "
            "evaluate; extend the window or reduce folds"
        )

    # 2) Locked hold-out — evaluated exactly once, here at the end (Section 16).
    holdout_report: BacktestReport | None = None
    if holdout_slots > 0:
        windowed = rebase_window(inputs, test_end_ts, data_hi)
        holdout_report = run_engine(
            cfg, meta, windowed, strategy=strategy, label="wf_holdout"
        ).report
        # The locked hold-out is the strongest, evaluated-once OOS check — hold it to the
        # SAME kill-criteria as every fold (min trades, expectancy, PF, drawdown), not a bare
        # "expectancy>0 and net>0" that a single lucky trade could clear.
        holdout_passed, holdout_failures = _evaluate_fold(holdout_report, kc)
        out.holdout = FoldResult(
            -1, test_end_ts, data_hi, holdout_passed, holdout_failures, holdout_report
        )

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
    """Bar interval (ms): the SMALLEST positive gap between consecutive bars across all symbols.
    Taking the minimum recovers the true timeframe even if a series happens to start with an
    interior hole (a naive ``bars[1]-bars[0]`` would then mis-size every fold)."""
    best: int | None = None
    for s in inputs:
        for a, b in zip(s.bars, s.bars[1:], strict=False):
            d = int(b["ts"] - a["ts"])
            if d > 0 and (best is None or d < best):
                best = d
    return best or 1
