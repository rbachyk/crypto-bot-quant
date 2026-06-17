"""Paper Trading Engine (AGENTS.md Section 26 / Phase 8).

Phase A — technical validation: full pipeline in paper mode.
Phase B — strategy validation: sufficient trades, breakdowns, paper-vs-backtest.
"""

from src.paper.engine import PaperTradingEngine
from src.paper.report import PaperReport, build_paper_report
from src.paper.session import (
    PaperDecisionLog,
    PaperSession,
    PaperTrade,
    RejectedPaperCandidate,
)

__all__ = [
    "PaperTradingEngine",
    "PaperSession",
    "PaperTrade",
    "PaperDecisionLog",
    "RejectedPaperCandidate",
    "PaperReport",
    "build_paper_report",
]
