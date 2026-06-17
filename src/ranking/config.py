"""Loader for ``configs/ranking.yaml`` — Setup Quality + Ranking (Section 7/15).

The component maxima are the Section 15 fixed points (sum = 100); the pass
``threshold`` is a walk-forward-validated tunable (SETUP gate). All consumers —
the setup-quality scorer, the ranking engine and the SETUP gate — read this one
config so scoring is single-sourced and versioned (Section 4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from src.config.settings import REPO_ROOT

RANKING_YAML = REPO_ROOT / "configs" / "ranking.yaml"

# The Section 15 component maxima (authoritative; sum must be 100).
_SPEC_COMPONENTS: dict[str, float] = {
    "regime_alignment": 20.0,
    "signal_strength": 20.0,
    "cross_signal_confirmation": 15.0,
    "expected_move_after_costs": 15.0,
    "execution_quality": 15.0,
    "risk_reward_quality": 10.0,
    "session_context": 5.0,
}


@dataclass(frozen=True, slots=True)
class RankingConfig:
    ranking_version: str
    components: dict[str, float]
    threshold: float
    rr_target: float
    max_spread_bps: float
    max_slippage_frac: float
    rank_by: tuple[str, ...] = field(default_factory=tuple)

    @property
    def max_score(self) -> float:
        return sum(self.components.values())


@lru_cache
def load_ranking_config(path: str | None = None) -> RankingConfig:
    yaml_path = Path(path) if path else RANKING_YAML
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    data = raw["ranking"]

    # Component maxima come from the spec; config may not change them silently.
    components = dict(_SPEC_COMPONENTS)
    for key, val in (data.get("components") or {}).items():
        if key in components:
            components[key] = float(val)

    return RankingConfig(
        ranking_version=str(data.get("ranking_version", "rank_0001")),
        components=components,
        threshold=float(data.get("threshold", 55.0)),
        rr_target=float(data.get("rr_target", 2.0)),
        max_spread_bps=float(data.get("max_spread_bps", 25.0)),
        max_slippage_frac=float(data.get("max_slippage_frac", 0.01)),
        rank_by=tuple(data.get("rank_by", ["setup_quality_score", "expected_value_after_costs"])),
    )
