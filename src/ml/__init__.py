"""ML layer — Phase 9–10 (AGENTS.md Section 20).

Phase 9: ML Stage 2 (Shadow Mode) — predictions logged, never applied.
Phase 10: ML Stage 3 (Recommendation Mode) + ML Stage 4 (Constrained Filter).

Stage 3: recommendations shown in dashboard / reports; never auto-applied.
Stage 4: filter may BLOCK deterministic candidates; cannot create, increase
         risk, or override hard blockers.  Promotion required ML-PROMO gate
         PASS + manual review (Section 20 Promotion gates).
"""

from .filter import FilterDecision, FilterResult, MLFilter
from .recommendation import MLRecommendation, RecommendationEngine, RecommendationRunResult
from .registry import MLRegistry, ModelArtifact
from .scorer import ShadowScorer, ShadowScorerResult
from .shadow import ShadowPredictor

__all__ = [
    "FilterDecision",
    "FilterResult",
    "MLFilter",
    "MLRecommendation",
    "MLRegistry",
    "ModelArtifact",
    "RecommendationEngine",
    "RecommendationRunResult",
    "ShadowPredictor",
    "ShadowScorer",
    "ShadowScorerResult",
]
