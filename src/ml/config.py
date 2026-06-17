"""ML shadow configuration (AGENTS.md Section 20, Phase 9).

Loaded from ``configs/ml.yaml`` — the single source for ML stage, model
classes, feature lists, kill-criteria, and shadow-mode settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from src.config.settings import REPO_ROOT

ML_YAML = REPO_ROOT / "configs" / "ml.yaml"


@dataclass
class ModelCfg:
    """Per-model configuration."""

    enabled: bool = True
    model_class: str = "logistic_regression"
    min_train_samples: int = 30
    test_fraction: float = 0.25
    features: list[str] = field(default_factory=list)
    n_estimators: int = 50  # for random_forest only
    extra: dict = field(default_factory=dict)


@dataclass
class ShadowCfg:
    mode: str = "SHADOW"
    applied_to_live: bool = False
    log_context_features: bool = True


@dataclass
class KillCriteria:
    min_improvement_over_baseline: float = 0.0
    min_profit_factor_ratio: float = 1.0
    max_tail_loss_ratio: float = 1.0
    max_best_trades_removed_pct: float = 0.2


@dataclass
class RecommendationCfg:
    """ML Stage 3 — Recommendation Mode config (Section 20 ML Stage 3)."""

    enabled: bool = False
    log_to_db: bool = True


@dataclass
class FilterCfg:
    """ML Stage 4 — Constrained Live Filter config (Section 20 ML Stage 4).

    ``min_confidence_to_take``: meta-labeler probability below which a
    deterministic candidate is blocked. Candidates above the threshold pass
    through unchanged; the filter NEVER creates or modifies candidates.
    """

    enabled: bool = False
    min_confidence_to_take: float = 0.4  # block if p_take < this
    log_to_db: bool = True


@dataclass
class MLConfig:
    """Root ML configuration object."""

    model_version: str = "ml_shadow_0001"
    ml_stage: int = 2
    meta_labeler: ModelCfg = field(default_factory=ModelCfg)
    regime_classifier: ModelCfg = field(default_factory=ModelCfg)
    exec_quality: ModelCfg = field(default_factory=ModelCfg)
    strategy_selector: ModelCfg = field(default_factory=ModelCfg)
    symbol_ranker: ModelCfg = field(default_factory=ModelCfg)
    shadow: ShadowCfg = field(default_factory=ShadowCfg)
    kill_criteria: KillCriteria = field(default_factory=KillCriteria)
    recommendation: RecommendationCfg = field(default_factory=RecommendationCfg)
    filter: FilterCfg = field(default_factory=FilterCfg)


def _model_cfg(raw: dict) -> ModelCfg:
    return ModelCfg(
        enabled=raw.get("enabled", True),
        model_class=raw.get("model_class", "logistic_regression"),
        min_train_samples=int(raw.get("min_train_samples", 30)),
        test_fraction=float(raw.get("test_fraction", 0.25)),
        features=list(raw.get("features", [])),
        n_estimators=int(raw.get("n_estimators", 50)),
        extra={
            k: v
            for k, v in raw.items()
            if k
            not in {
                "enabled",
                "model_class",
                "min_train_samples",
                "test_fraction",
                "features",
                "n_estimators",
            }
        },
    )


@lru_cache
def load_ml_config(path: str | None = None) -> MLConfig:
    """Parse ``configs/ml.yaml`` into an :class:`MLConfig`."""
    yaml_path = Path(path) if path else ML_YAML
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    sh = raw.get("shadow", {})
    kc = raw.get("kill_criteria", {})
    rc = raw.get("recommendation", {})
    fc = raw.get("filter", {})
    return MLConfig(
        model_version=raw.get("model_version", "ml_shadow_0001"),
        ml_stage=int(raw.get("ml_stage", 2)),
        meta_labeler=_model_cfg(raw.get("meta_labeler", {})),
        regime_classifier=_model_cfg(raw.get("regime_classifier", {})),
        exec_quality=_model_cfg(raw.get("exec_quality", {})),
        strategy_selector=_model_cfg(raw.get("strategy_selector", {})),
        symbol_ranker=_model_cfg(raw.get("symbol_ranker", {})),
        shadow=ShadowCfg(
            mode=sh.get("mode", "SHADOW"),
            applied_to_live=bool(sh.get("applied_to_live", False)),
            log_context_features=bool(sh.get("log_context_features", True)),
        ),
        kill_criteria=KillCriteria(
            min_improvement_over_baseline=float(kc.get("min_improvement_over_baseline", 0.0)),
            min_profit_factor_ratio=float(kc.get("min_profit_factor_ratio", 1.0)),
            max_tail_loss_ratio=float(kc.get("max_tail_loss_ratio", 1.0)),
            max_best_trades_removed_pct=float(kc.get("max_best_trades_removed_pct", 0.2)),
        ),
        recommendation=RecommendationCfg(
            enabled=bool(rc.get("enabled", False)),
            log_to_db=bool(rc.get("log_to_db", True)),
        ),
        filter=FilterCfg(
            enabled=bool(fc.get("enabled", False)),
            min_confidence_to_take=float(fc.get("min_confidence_to_take", 0.4)),
            log_to_db=bool(fc.get("log_to_db", True)),
        ),
    )
