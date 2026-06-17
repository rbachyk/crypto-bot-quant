"""Adaptation configuration loader (AGENTS.md Section 21.9).

Loads ``configs/adaptation.yaml`` and exposes typed dataclasses for use
throughout the adaptation module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from src.config.settings import REPO_ROOT

ADAPTATION_YAML = REPO_ROOT / "configs" / "adaptation.yaml"


@dataclass
class BoundsConfig:
    w_min: float = 0.0
    w_max: float = 2.0
    size_buckets: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)
    max_change_per_update: float = 0.10
    max_change_rate: float = 0.25


@dataclass
class RollbackConfig:
    rollback_window: int = 20
    rollback_margin: float = 0.05
    max_divergence: float = 0.20
    auto_freeze_on_breaker: bool = True  # immutable: always true


@dataclass
class ScoringConfig:
    min_shadow_decisions: int = 50
    min_wf_folds_positive: int = 2
    min_holdout_edge: float = 0.0
    calibration_max_brier: float = 0.30
    max_drift_per_window: float = 0.15


@dataclass
class MonitoringConfig:
    enabled: bool = True
    drift_window: int = 20
    calibration_window: int = 50


@dataclass
class AdaptationConfig:
    """Full adaptation configuration."""

    version: int = 1
    learner_version: str = "learner_0001"
    enabled: bool = True
    mode: str = "SHADOW"
    min_samples_to_start: int = 50
    learner_id: str = "online_shadow_v1"
    registered_tunables: dict = field(default_factory=dict)
    bounds: BoundsConfig = field(default_factory=BoundsConfig)
    rollback: RollbackConfig = field(default_factory=RollbackConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    frozen_fallback_policy: str = "var/adaptation/frozen_fallback.pkl"


@lru_cache(maxsize=1)
def load_adaptation_config(path: Path | None = None) -> AdaptationConfig:
    p = path or ADAPTATION_YAML
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    a = raw.get("adaptation", {})
    b = a.get("bounds", {})
    r = a.get("rollback", {})
    sc = a.get("scoring", {})
    m = a.get("monitoring", {})
    return AdaptationConfig(
        version=raw.get("version", 1),
        learner_version=raw.get("learner_version", "learner_0001"),
        enabled=a.get("enabled", True),
        mode=a.get("mode", "SHADOW"),
        min_samples_to_start=a.get("min_samples_to_start", 50),
        learner_id=a.get("learner_id", "online_shadow_v1"),
        registered_tunables=a.get("registered_tunables") or {},
        bounds=BoundsConfig(
            w_min=b.get("strategy_weight", {}).get("w_min", 0.0),
            w_max=b.get("strategy_weight", {}).get("w_max", 2.0),
            size_buckets=tuple(b.get("size_buckets", [0.0, 0.25, 0.5, 1.0])),
            max_change_per_update=b.get("max_change_per_update", 0.10),
            max_change_rate=b.get("max_change_rate", 0.25),
        ),
        rollback=RollbackConfig(
            rollback_window=r.get("rollback_window", 20),
            rollback_margin=r.get("rollback_margin", 0.05),
            max_divergence=r.get("max_divergence", 0.20),
            auto_freeze_on_breaker=True,  # always true; immutable
        ),
        scoring=ScoringConfig(
            min_shadow_decisions=sc.get("min_shadow_decisions", 50),
            min_wf_folds_positive=sc.get("min_wf_folds_positive", 2),
            min_holdout_edge=sc.get("min_holdout_edge", 0.0),
            calibration_max_brier=sc.get("calibration_max_brier", 0.30),
            max_drift_per_window=sc.get("max_drift_per_window", 0.15),
        ),
        monitoring=MonitoringConfig(
            enabled=m.get("enabled", True),
            drift_window=m.get("drift_window", 20),
            calibration_window=m.get("calibration_window", 50),
        ),
        frozen_fallback_policy=a.get(
            "frozen_fallback_policy", "var/adaptation/frozen_fallback.pkl"
        ),
    )
