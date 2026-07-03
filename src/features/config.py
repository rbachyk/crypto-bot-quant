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
    # funding_z is a FIXED-LENGTH rolling z-score over the trailing ``funding_z_lookback_days``
    # of settlements — never an expanding window anchored at the run's window start, so the same
    # symbol/ts scores identically regardless of how far back the run window reaches (M3:
    # reproducibility + backtest↔live parity; funding_carry ranks by this feature). With fewer
    # than ``funding_z_min_samples`` settlements in the window the feature is UNAVAILABLE and
    # reported as the neutral 0.0 sentinel (same convention as premium with no mark/index).
    funding_z_lookback_days: int = 30
    funding_z_min_samples: int = 30

    @property
    def warmup(self) -> int:
        """Bars of closed history required before the first feature row."""
        return max(self.windows.short, self.windows.long) + 1

    @property
    def funding_z_lookback_ms(self) -> int:
        return self.funding_z_lookback_days * 86_400_000


@lru_cache
def load_feature_config(path: str | None = None) -> FeatureConfig:
    yaml_path = Path(path) if path else FEATURES_YAML
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    data = raw["features"]
    w = data.get("windows", {})
    lk = data.get("leakage", {})
    fz = data.get("funding_z", {})
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
        funding_z_lookback_days=int(fz.get("lookback_days", 30)),
        funding_z_min_samples=int(fz.get("min_samples", 30)),
    )
