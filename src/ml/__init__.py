"""ML shadow layer — Phase 9 (AGENTS.md Section 20).

ML Stage 2: Shadow Mode only.  No live influence; all predictions logged to
``shadow_logs`` with ``applied=False``.  Promotion to Stage 3+ requires the
ML-PROMO gate to PASS and manual review (Section 20 Promotion gates).
"""

from .registry import MLRegistry, ModelArtifact
from .scorer import ShadowScorer, ShadowScorerResult
from .shadow import ShadowPredictor

__all__ = [
    "MLRegistry",
    "ModelArtifact",
    "ShadowPredictor",
    "ShadowScorer",
    "ShadowScorerResult",
]
