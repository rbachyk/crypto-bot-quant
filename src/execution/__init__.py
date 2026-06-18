"""Execution Engine package (AGENTS.md Section 18 / Section 7).

Builds and places orders only after risk approval, with atomic exchange-resident
stops, native trailing, reconciliation and strict order ownership. The bot only
ever manages orders it created.
"""

from __future__ import annotations

from src.execution.config import ExecutionPolicyConfig, load_execution_config
from src.execution.engine import ExecutionEngine, ExecutionResult
from src.execution.live_venue import CcxtLiveVenue, get_venue
from src.execution.order import (
    NO_FIXED_TP_FRAC,
    BuildResult,
    Order,
    OrderBuilder,
    OrderPlan,
    OrderType,
)
from src.execution.ownership import OwnershipPolicy
from src.execution.reconciliation import Reconciler, ReconResult
from src.execution.venue import BracketResult, Fill, SimulatedVenue, Venue, VenuePosition

__all__ = [
    "ExecutionPolicyConfig",
    "load_execution_config",
    "ExecutionEngine",
    "ExecutionResult",
    "Order",
    "OrderType",
    "OrderPlan",
    "OrderBuilder",
    "BuildResult",
    "NO_FIXED_TP_FRAC",
    "OwnershipPolicy",
    "Reconciler",
    "ReconResult",
    "SimulatedVenue",
    "Venue",
    "CcxtLiveVenue",
    "get_venue",
    "Fill",
    "VenuePosition",
    "BracketResult",
]
