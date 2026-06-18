"""Loader for ``configs/execution.yaml`` — the Execution Engine contract (Section 18).

Single-sources every execution policy knob (entry style, the toxic-spread /
slippage / latency hard blockers, exchange-resident-stop requirement, native
trailing offset, the simulated-venue fill model, and the emergency-close
confirmation requirement) so the order builder, the venue and the EXEC gate all
agree. Versioned via ``execution_policy_version`` (Section 4).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from src.config.settings import REPO_ROOT

EXECUTION_YAML = REPO_ROOT / "configs" / "execution.yaml"


@dataclass(frozen=True, slots=True)
class ExecutionPolicyConfig:
    execution_policy_version: str
    default_entry_style: str
    max_spread_bps: float
    max_slippage_frac: float
    max_latency_ms: float
    attach_take_profit: bool
    trailing_offset_frac: float
    simulated_latency_ms: float
    simulated_partial_fill_ratio: float
    emergency_close_requires_confirmation: bool


@lru_cache
def load_execution_config(path: str | None = None) -> ExecutionPolicyConfig:
    yaml_path = Path(path) if path else EXECUTION_YAML
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    data = raw["execution"]
    return ExecutionPolicyConfig(
        execution_policy_version=str(data.get("execution_policy_version", "exec_0001")),
        default_entry_style=str(data.get("default_entry_style", "maker_first")),
        max_spread_bps=float(data.get("max_spread_bps", 25.0)),
        max_slippage_frac=float(data.get("max_slippage_frac", 0.01)),
        max_latency_ms=float(data.get("max_latency_ms", 1500.0)),
        attach_take_profit=bool(data.get("attach_take_profit", True)),
        trailing_offset_frac=float(data.get("trailing_offset_frac", 0.0)),
        simulated_latency_ms=float(data.get("simulated_latency_ms", 40.0)),
        simulated_partial_fill_ratio=float(data.get("simulated_partial_fill_ratio", 0.5)),
        emergency_close_requires_confirmation=bool(
            data.get("emergency_close_requires_confirmation", True)
        ),
    )
