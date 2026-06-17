"""Candidate Ranking + Setup Quality package (AGENTS.md Section 7 / Section 15).

Strategies generate candidates; this package scores their setup quality
deterministically, drops any with an active hard blocker, and ranks the survivors
across the universe before the risk manager approves the winner.
"""

from __future__ import annotations

from src.ranking.candidate import Candidate
from src.ranking.config import RankingConfig, load_ranking_config
from src.ranking.engine import (
    CandidateRankingEngine,
    RankedCandidate,
    RankingResult,
    RejectedAlternative,
)
from src.ranking.setup_quality import (
    NO_TRADE_REGIMES,
    SetupContext,
    SetupQualityScorer,
    SetupScore,
)

__all__ = [
    "Candidate",
    "RankingConfig",
    "load_ranking_config",
    "CandidateRankingEngine",
    "RankedCandidate",
    "RankingResult",
    "RejectedAlternative",
    "NO_TRADE_REGIMES",
    "SetupContext",
    "SetupQualityScorer",
    "SetupScore",
]
