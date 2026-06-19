"""Loader for ``configs/strategies.yaml`` — the Phase 5 research-candidate contract.

Turns the YAML into typed, frozen dataclasses read by the strategy implementations
(:mod:`src.strategies.candidates`), the deterministic fixtures
(:mod:`src.strategies.fixtures`), and the research/validation harness
(:mod:`src.strategies.research`). All runtime behaviour is config-driven and
versioned (Section 4): thresholds, exit geometry, side permissions and fixture
parameters all live here, so a change is a new ``STRATEGY_VERSION``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from src.config.settings import REPO_ROOT

STRATEGIES_YAML = REPO_ROOT / "configs" / "strategies.yaml"


@dataclass(frozen=True, slots=True)
class StrategyParams:
    """Entry/exit knobs shared by every candidate (family-specific keys in extra)."""

    stop_frac: float
    tp_frac: float
    hold_bars: int
    allow_long: bool = True
    allow_short: bool = True
    extra: dict[str, float] = field(default_factory=dict)

    def with_sides(self, *, allow_long: bool, allow_short: bool) -> StrategyParams:
        return replace(self, allow_long=allow_long, allow_short=allow_short)


@dataclass(frozen=True, slots=True)
class FixtureConfig:
    """Deterministic-fixture parameters for one candidate (planted causal edge)."""

    seed: str
    bars: int
    timeframe: str
    # Heterogeneous per-fixture params straight from YAML (floats, ints, strings, string
    # lists). Consumers convert explicitly at each use (float(...)/int(...)/str(...)).
    values: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CandidateConfig:
    id: str
    family: str
    enabled: bool
    exit_profile: str
    params: StrategyParams
    fixture: FixtureConfig


@dataclass(frozen=True, slots=True)
class StrategiesConfig:
    strategy_version: str
    min_side_expectancy_r: float
    candidates: tuple[CandidateConfig, ...]
    # Cap on how many promoted strategies the live/demo engine runs concurrently — the top-N by
    # validated expectancy (Section 13). Keeps the live ensemble small/diversified rather than
    # firing every promoted candidate at once. 0 = no cap.
    max_active_strategies: int = 5

    def enabled_candidates(self) -> list[CandidateConfig]:
        return [c for c in self.candidates if c.enabled]

    def candidate(self, candidate_id: str) -> CandidateConfig | None:
        for c in self.candidates:
            if c.id == candidate_id:
                return c
        return None


# Reserved param keys that map to StrategyParams fields; everything else is "extra".
_RESERVED = {"stop_frac", "tp_frac", "hold_bars", "allow_long", "allow_short"}


def _parse_params(raw: dict) -> StrategyParams:
    extra = {k: float(v) for k, v in raw.items() if k not in _RESERVED}
    return StrategyParams(
        stop_frac=float(raw["stop_frac"]),
        tp_frac=float(raw["tp_frac"]),
        hold_bars=int(raw["hold_bars"]),
        allow_long=bool(raw.get("allow_long", True)),
        allow_short=bool(raw.get("allow_short", True)),
        extra=extra,
    )


def _parse_fixture(raw: dict) -> FixtureConfig:
    reserved = {"seed", "bars", "timeframe"}
    values = {k: v for k, v in raw.items() if k not in reserved}
    return FixtureConfig(
        seed=str(raw["seed"]),
        bars=int(raw["bars"]),
        timeframe=str(raw["timeframe"]),
        values=values,
    )


@lru_cache
def load_strategies_config(path: str | None = None) -> StrategiesConfig:
    yaml_path = Path(path) if path else STRATEGIES_YAML
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    data = raw["strategies"]
    candidates = tuple(
        CandidateConfig(
            id=str(c["id"]),
            family=str(c["family"]),
            enabled=bool(c.get("enabled", True)),
            exit_profile=str(c["exit_profile"]),
            params=_parse_params(c["params"]),
            fixture=_parse_fixture(c["fixture"]),
        )
        for c in data["candidates"]
    )
    return StrategiesConfig(
        strategy_version=str(data.get("strategy_version", "strat_0001")),
        min_side_expectancy_r=float(data.get("min_side_expectancy_r", 0.0)),
        candidates=candidates,
        max_active_strategies=int(data.get("max_active_strategies", 5)),
    )
