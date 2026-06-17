"""Risk Manager package (AGENTS.md Section 17 / Section 2.2).

The risk manager has absolute authority: it approves every order, sizing it
deterministically inside the immutable risk envelope. Nothing — strategy, ML,
learner, RL — may bypass it.
"""

from __future__ import annotations

from src.risk.breakers import BreakerInputs, BreakerVerdict, CircuitBreakers
from src.risk.config import BreakerConfig, RiskConfig, load_risk_config
from src.risk.envelope import HARD_CEILINGS, RiskEnvelope
from src.risk.manager import AccountState, RiskDecision, RiskManager
from src.risk.portfolio import PortfolioState, Position

__all__ = [
    "HARD_CEILINGS",
    "RiskEnvelope",
    "RiskConfig",
    "BreakerConfig",
    "load_risk_config",
    "BreakerInputs",
    "BreakerVerdict",
    "CircuitBreakers",
    "PortfolioState",
    "Position",
    "AccountState",
    "RiskDecision",
    "RiskManager",
]
