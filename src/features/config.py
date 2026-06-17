"""Loader for ``configs/features.yaml`` — the Feature Pipeline contract.

Read by the feature pipeline, the feature store, the leakage harness and the
FEAT gate so they share one definition of the feature set, windows and label
horizon (Section 4 config-driven; Section 10 Parity Rule).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from src.config.settings import REPO_ROOT

FEATURES_YAML = REPO_ROOT / "configs" / "features.yaml"


@dataclass(frozen=True, slots=True)
class FeatureWindows:
    short: int = 12
    long: int = 48
    rank: int = 96


@dataclass(frozen=True, slots=True)
class LeakageConfig:
    synthetic_bars: int = 4000
    max_synthetic_expectancy_z: float = 4.0


@dataclass(frozen=True, slots=True)
class FeatureConfig:
    feature_set_version: str
    timeframe: str
    windows: FeatureWindows
    label_horizon: int
    leakage: LeakageConfig

    @property
    def warmup(self) -> int:
        """Bars of closed history required before the first feature row."""
        return max(self.windows.short, self.windows.long) + 1


@lru_cache
def load_feature_config(path: str | None = None) -> FeatureConfig:
    yaml_path = Path(path) if path else FEATURES_YAML
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    data = raw["features"]
    w = data.get("windows", {})
    lk = data.get("leakage", {})
    return FeatureConfig(
        feature_set_version=str(data["feature_set_version"]),
        timeframe=str(data["timeframe"]),
        windows=FeatureWindows(
            short=int(w.get("short", 12)),
            long=int(w.get("long", 48)),
            rank=int(w.get("rank", 96)),
        ),
        label_horizon=int(data.get("label_horizon", 12)),
        leakage=LeakageConfig(
            synthetic_bars=int(lk.get("synthetic_bars", 4000)),
            max_synthetic_expectancy_z=float(lk.get("max_synthetic_expectancy_z", 4.0)),
        ),
    )
