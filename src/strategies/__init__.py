"""Deterministic quant research candidates (AGENTS.md Phase 5, Section 12/13).

Real deterministic strategy logic for families A (cross-asset lead-lag), B
(perpetual premium / basis mean reversion) and G (cross-sectional relative
strength / dispersion), each declared as a research candidate with a full
hypothesis (Section 13) and validated through the SAME event-based engine,
walk-forward and fee/slippage stress as Phase 4 (the Parity Rule, Section 10).

Strategies generate candidates only — they never place orders (Section 5). The
data they are validated on is a deterministic synthetic fixture (no live data
exists offline), so they are research candidates, not proven live edges.
"""

from src.strategies.base import StrategyHypothesis
from src.strategies.candidates import (
    BasisReversionStrategy,
    CrossSectionalRSStrategy,
    LeadLagStrategy,
    build_strategy,
    is_portfolio_family,
)
from src.strategies.config import (
    CandidateConfig,
    StrategiesConfig,
    StrategyParams,
    load_strategies_config,
)
from src.strategies.fixtures import build_candidate_inputs
from src.strategies.research import (
    CandidateValidation,
    SideDecision,
    strategy_report_payload,
    validate_all,
    validate_candidate,
    write_strategy_reports,
)

__all__ = [
    "BasisReversionStrategy",
    "CandidateConfig",
    "CandidateValidation",
    "CrossSectionalRSStrategy",
    "LeadLagStrategy",
    "SideDecision",
    "StrategiesConfig",
    "StrategyHypothesis",
    "StrategyParams",
    "build_candidate_inputs",
    "build_strategy",
    "is_portfolio_family",
    "load_strategies_config",
    "strategy_report_payload",
    "validate_all",
    "validate_candidate",
    "write_strategy_reports",
]
