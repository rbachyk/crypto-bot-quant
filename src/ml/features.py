"""ML feature matrix construction (AGENTS.md Section 10 Parity Rule).

Builds a feature matrix from :class:`~src.ranking.candidate.Candidate` objects
and their decision-time feature dicts.  The same feature code path is used for
training and inference so there is no backtest/live divergence.

Features are assembled from:
1. Candidate-level fields (signal_strength, expected_edge_frac, spread_bps, …)
2. Pipeline features stored in ``candidate.features`` (atr_pct, premium, funding_z, …)

All values are floats (no forward-looking data, no future labels).  Missing
pipeline features default to 0.0 so the matrix is always dense.
"""

from __future__ import annotations

from src.ranking.candidate import Candidate

# Candidate-level fields extracted as ML features.
_CANDIDATE_FIELDS: tuple[str, ...] = (
    "signal_strength",
    "expected_edge_frac",
    "spread_bps",
    "slippage_est",
)

# Pipeline features copied from candidate.features dict (Section 10).
_PIPELINE_FIELDS: tuple[str, ...] = (
    "atr_pct",
    "premium",
    "funding_z",
    "rv_short",
    "ret_1",
)

# Full canonical feature name list (order matters — must be stable across versions).
FEATURE_NAMES: tuple[str, ...] = _CANDIDATE_FIELDS + _PIPELINE_FIELDS


def candidate_to_row(candidate: Candidate) -> dict[str, float]:
    """Extract a flat float dict from a candidate (decision-time features only)."""
    row: dict[str, float] = {}
    for fname in _CANDIDATE_FIELDS:
        row[fname] = float(getattr(candidate, fname, 0.0))
    for fname in _PIPELINE_FIELDS:
        row[fname] = float(candidate.features.get(fname, 0.0))
    return row


def build_feature_matrix(
    candidates: list[Candidate],
    feature_names: list[str] | None = None,
) -> list[list[float]]:
    """Build an N×F matrix (list of rows) from a list of candidates.

    *feature_names* defaults to :data:`FEATURE_NAMES`.  Pass a subset to
    restrict the features used by a specific model.
    """
    cols = list(feature_names) if feature_names else list(FEATURE_NAMES)
    matrix: list[list[float]] = []
    for cand in candidates:
        row_dict = candidate_to_row(cand)
        matrix.append([row_dict.get(col, 0.0) for col in cols])
    return matrix


def feature_names_for(model_feature_list: list[str]) -> list[str]:
    """Return the intersection of requested features with the canonical list.

    Preserves canonical order so feature indices are stable.
    """
    canonical = set(FEATURE_NAMES)
    return [f for f in FEATURE_NAMES if f in set(model_feature_list) and f in canonical]
