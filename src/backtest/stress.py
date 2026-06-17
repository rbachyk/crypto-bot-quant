"""Fee & slippage stress tests (AGENTS.md Section 16, FEE + SLIP gates).

Re-runs the SAME backtest with multiplied costs and checks the edge survives
(Section 16 "does the edge survive ×2 fees, +50% slippage?"). A strategy whose
expectancy turns negative under stress is fee/slippage-dependent and must be
disabled or de-risked (FEE/SLIP remediation) rather than promoted. The baseline
expectancy is reported alongside so the dashboard can show how much margin the
edge has over its cost assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.backtest.config import BacktestConfig
from src.backtest.engine import SymbolInput
from src.backtest.metrics import BacktestReport
from src.backtest.service import run_engine
from src.backtest.strategy import PortfolioStrategy, Strategy
from src.exchange.metadata import MetadataConfig


@dataclass(slots=True)
class StressResult:
    kind: str  # "fee" | "slippage"
    multiplier: float
    baseline_expectancy_r: float
    stressed_expectancy_r: float
    stressed_profit_factor: float
    stressed_net_pnl: float
    trade_count: int
    survives: bool

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "multiplier": self.multiplier,
            "baseline_expectancy_r": self.baseline_expectancy_r,
            "stressed_expectancy_r": self.stressed_expectancy_r,
            "stressed_profit_factor": self.stressed_profit_factor,
            "stressed_net_pnl": self.stressed_net_pnl,
            "trade_count": self.trade_count,
            "survives": self.survives,
        }


def _baseline(
    cfg: BacktestConfig,
    meta: MetadataConfig,
    inputs: list[SymbolInput],
    strategy: Strategy | PortfolioStrategy | None,
) -> float:
    return run_engine(cfg, meta, inputs, strategy=strategy, label="baseline").report.expectancy_r


def fee_stress(
    cfg: BacktestConfig,
    meta: MetadataConfig,
    inputs: list[SymbolInput],
    *,
    multiplier: float | None = None,
    baseline_expectancy_r: float | None = None,
    strategy: Strategy | PortfolioStrategy | None = None,
) -> StressResult:
    mult = multiplier if multiplier is not None else cfg.stress.fee_multiplier
    base = (
        baseline_expectancy_r
        if baseline_expectancy_r is not None
        else _baseline(cfg, meta, inputs, strategy)
    )
    stressed = run_engine(
        cfg.with_cost_overrides(fee_multiplier=mult),
        meta,
        inputs,
        strategy=strategy,
        label=f"fee_x{mult}",
    ).report
    return _result("fee", mult, base, stressed)


def slippage_stress(
    cfg: BacktestConfig,
    meta: MetadataConfig,
    inputs: list[SymbolInput],
    *,
    multiplier: float | None = None,
    baseline_expectancy_r: float | None = None,
    strategy: Strategy | PortfolioStrategy | None = None,
) -> StressResult:
    mult = multiplier if multiplier is not None else cfg.stress.slippage_multiplier
    base = (
        baseline_expectancy_r
        if baseline_expectancy_r is not None
        else _baseline(cfg, meta, inputs, strategy)
    )
    stressed = run_engine(
        cfg.with_cost_overrides(slippage_multiplier=mult),
        meta,
        inputs,
        strategy=strategy,
        label=f"slip_x{mult}",
    ).report
    return _result("slippage", mult, base, stressed)


def _result(kind: str, mult: float, base: float, stressed: BacktestReport) -> StressResult:
    return StressResult(
        kind=kind,
        multiplier=mult,
        baseline_expectancy_r=round(base, 6),
        stressed_expectancy_r=round(stressed.expectancy_r, 6),
        stressed_profit_factor=round(stressed.profit_factor, 6),
        stressed_net_pnl=round(stressed.net_pnl, 6),
        trade_count=stressed.trade_count,
        survives=stressed.expectancy_r > 0 and stressed.net_pnl > 0,
    )
