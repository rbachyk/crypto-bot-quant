"""Feature pipeline (AGENTS.md Section 10 Parity Rule, Phase 3).

A single feature-computation code path for backtest and live (the only
difference is the data-reading adapter), with decision-time-only inputs,
reproducible builds from a dataset snapshot, and a leakage/look-ahead harness
backing the FEAT gate.
"""

from src.features.config import FeatureConfig, load_feature_config
from src.features.leakage import (
    CausalViolation,
    SyntheticReader,
    causal_invariance_violations,
    expectancy_z,
    forward_labels,
    momentum_signals,
    synthetic_leakage_report,
)
from src.features.pipeline import (
    FEATURE_NAMES,
    FeatureDataReader,
    FeatureFrame,
    StoreReader,
    TruncatedReader,
    compute_features,
    has_nan_or_inf,
)
from src.features.store import FeatureBuildResult, FeatureStore

__all__ = [
    "FEATURE_NAMES",
    "CausalViolation",
    "FeatureBuildResult",
    "FeatureConfig",
    "FeatureDataReader",
    "FeatureFrame",
    "FeatureStore",
    "StoreReader",
    "SyntheticReader",
    "TruncatedReader",
    "causal_invariance_violations",
    "compute_features",
    "expectancy_z",
    "forward_labels",
    "has_nan_or_inf",
    "load_feature_config",
    "momentum_signals",
    "synthetic_leakage_report",
]
