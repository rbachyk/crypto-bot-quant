"""Candidate Ranking Engine (AGENTS.md Section 7 + Section 15 attribution).

Signals are evaluated across the whole active universe each cycle (Section 9/15).
This engine scores every candidate with the deterministic setup-quality scorer,
drops anything tripping a hard blocker or below the threshold, and ranks the
survivors so the risk manager sees the best candidate first (Section 7: rank
candidates *before* risk approval).

It records the full Section 15 multi-symbol attribution — which symbol/setup won,
and the rejected alternatives and why they lost — so the system never acts on a
trade it cannot attribute. The ranking is deterministic (stable sort on
``(setup_quality_score, expected_value_after_costs, symbol, strategy)``).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.exchange.metadata import MetadataConfig
from src.ranking.candidate import Candidate
from src.ranking.config import RankingConfig
from src.ranking.setup_quality import SetupContext, SetupQualityScorer, SetupScore


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    candidate: Candidate
    score: SetupScore
    rank: int

    def to_dict(self) -> dict:
        return {
            "symbol": self.candidate.symbol,
            "strategy": self.candidate.strategy,
            "side": "long" if self.candidate.side > 0 else "short",
            "rank": self.rank,
            "setup_quality_score": round(self.score.total, 4),
            "expected_value_after_costs": round(self.score.expected_value_after_costs, 8),
            "components": {k: round(v, 4) for k, v in self.score.components.items()},
        }


@dataclass(frozen=True, slots=True)
class RejectedAlternative:
    symbol: str
    strategy: str
    side: int
    reason: str
    setup_quality_score: float

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "strategy": self.strategy,
            "side": "long" if self.side > 0 else "short",
            "reason": self.reason,
            "setup_quality_score": round(self.setup_quality_score, 4),
        }


@dataclass(frozen=True, slots=True)
class RankingResult:
    selected: tuple[RankedCandidate, ...]
    rejected: tuple[RejectedAlternative, ...]
    ranking_version: str

    @property
    def winner(self) -> RankedCandidate | None:
        return self.selected[0] if self.selected else None

    def attribution(self) -> dict:
        """The Section 15 decision-log attribution for this scan cycle."""
        return {
            "ranking_version": self.ranking_version,
            "selected": [r.to_dict() for r in self.selected],
            "winner": self.winner.to_dict() if self.winner else None,
            "rejected_alternatives": [r.to_dict() for r in self.rejected],
            "evaluated": len(self.selected) + len(self.rejected),
        }


class CandidateRankingEngine:
    """Score → filter (blockers + threshold) → rank (Section 7/15)."""

    def __init__(self, cfg: RankingConfig, meta: MetadataConfig) -> None:
        self.cfg = cfg
        self.scorer = SetupQualityScorer(cfg, meta)

    def rank(
        self,
        candidates: list[Candidate],
        ctx: SetupContext | None = None,
        contexts: dict[str, SetupContext] | None = None,
    ) -> RankingResult:
        """Rank candidates. ``contexts`` may override the state per symbol."""
        ctx = ctx or SetupContext()
        scored: list[tuple[Candidate, SetupScore]] = []
        rejected: list[RejectedAlternative] = []

        for cand in candidates:
            cctx = (contexts or {}).get(cand.symbol, ctx)
            score = self.scorer.score(cand, cctx)
            if score.blockers:
                rejected.append(
                    RejectedAlternative(
                        cand.symbol,
                        cand.strategy,
                        cand.side,
                        f"hard_blocker:{','.join(score.blockers)}",
                        score.total,
                    )
                )
                continue
            if not score.passed_threshold:
                rejected.append(
                    RejectedAlternative(
                        cand.symbol,
                        cand.strategy,
                        cand.side,
                        f"below_threshold({score.total:.1f}<{self.cfg.threshold})",
                        score.total,
                    )
                )
                continue
            scored.append((cand, score))

        # Deterministic ranking driven by configs/ranking.yaml `rank_by` (each key sorted
        # best-first), with symbol/strategy as stable tiebreakers for a reproducible order.
        # Unknown keys are ignored; an empty list falls back to score then EV-after-costs.
        _RANK_KEYS = {
            "setup_quality_score": lambda cs: -cs[1].total,
            "expected_value_after_costs": lambda cs: -cs[1].expected_value_after_costs,
        }
        extractors = [_RANK_KEYS[k] for k in self.cfg.rank_by if k in _RANK_KEYS] or [
            _RANK_KEYS["setup_quality_score"],
            _RANK_KEYS["expected_value_after_costs"],
        ]
        scored.sort(key=lambda cs: (*(f(cs) for f in extractors), cs[0].symbol, cs[0].strategy))
        selected = tuple(
            RankedCandidate(cand, score, rank=i + 1) for i, (cand, score) in enumerate(scored)
        )
        # Losers below the winner are rejected alternatives too (rank > 1).
        for ranked in selected[1:]:
            rejected.append(
                RejectedAlternative(
                    ranked.candidate.symbol,
                    ranked.candidate.strategy,
                    ranked.candidate.side,
                    f"outranked(rank={ranked.rank})",
                    ranked.score.total,
                )
            )
        return RankingResult(
            selected=selected, rejected=tuple(rejected), ranking_version=self.cfg.ranking_version
        )
