"""Event-based Backtest Engine (AGENTS.md Section 19, Phase 4).

A strictly event-based engine (vectorized backtests are exploration-only) that
shares the ONE feature pipeline with paper/live (the Parity Rule, Section 10) and
models realistic fees, slippage and funding tied to verified exchange metadata.
It simulates risk sizing and execution, logs rejected candidates, and generates
the full required report. Look-ahead is prevented structurally (signals fill at
the next bar open); survivorship / future-universe leakage is prevented by a
point-in-time universe. Backs the BT / WF / FEE / SLIP gates.

Phase 4 ships the engine + harnesses; the validated trading strategies that run
through it arrive in Phase 5. A deterministic reference strategy on a
deterministic reference series (with a known causal edge) exercises and proves
the engine, walk-forward and stress machinery.
"""

from src.backtest.config import BacktestConfig, load_backtest_config
from src.backtest.costs import FeeModel, FundingModel, SlippageModel
from src.backtest.engine import (
    BacktestEngine,
    BacktestResult,
    RejectedCandidate,
    SymbolInput,
    Trade,
)
from src.backtest.guards import future_universe_violations, noise_expectancy
from src.backtest.metrics import BacktestReport, build_report, max_drawdown
from src.backtest.reference import ReferenceReader
from src.backtest.risk import RiskSimulator, SizingResult
from src.backtest.service import (
    BacktestRunResult,
    build_reference_inputs,
    make_strategy,
    rebase_window,
    run_engine,
    run_reference_backtest,
)
from src.backtest.strategy import ReferenceMomentumStrategy, Signal, Strategy
from src.backtest.stress import StressResult, fee_stress, slippage_stress
from src.backtest.walkforward import WalkForwardResult, run_walk_forward

__all__ = [
    "BacktestConfig",
    "BacktestEngine",
    "BacktestReport",
    "BacktestResult",
    "BacktestRunResult",
    "FeeModel",
    "FundingModel",
    "RejectedCandidate",
    "ReferenceMomentumStrategy",
    "ReferenceReader",
    "RiskSimulator",
    "Signal",
    "SizingResult",
    "SlippageModel",
    "Strategy",
    "StressResult",
    "SymbolInput",
    "Trade",
    "WalkForwardResult",
    "build_reference_inputs",
    "build_report",
    "fee_stress",
    "future_universe_violations",
    "load_backtest_config",
    "make_strategy",
    "max_drawdown",
    "noise_expectancy",
    "rebase_window",
    "run_engine",
    "run_reference_backtest",
    "run_walk_forward",
    "slippage_stress",
]
