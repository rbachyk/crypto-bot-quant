"""Loader for ``configs/gates.yaml`` — the single source of truth for gates.

The Gate Runner, dashboard and Reviewer all read the same catalog (Appendix A).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from src.config.settings import REPO_ROOT

GATES_YAML = REPO_ROOT / "configs" / "gates.yaml"


@dataclass(slots=True)
class GateSpec:
    gate_id: str
    name: str
    phase: str
    depends_on: list[str] = field(default_factory=list)
    blocks_live: str = "true"
    pass_condition: str = ""
    remediation_steps: list[str] = field(default_factory=list)
    rerun_job: str = ""


@lru_cache
def load_catalog(path: str | None = None) -> dict[str, GateSpec]:
    """Parse the gate catalog into ``{gate_id: GateSpec}``."""
    yaml_path = Path(path) if path else GATES_YAML
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    specs: dict[str, GateSpec] = {}
    for raw in data.get("gates", []):
        specs[raw["id"]] = GateSpec(
            gate_id=raw["id"],
            name=raw.get("name", raw["id"]),
            phase=str(raw.get("phase", "")),
            depends_on=list(raw.get("depends_on", [])),
            blocks_live=str(raw.get("blocks_live", "true")),
            pass_condition=raw.get("pass_condition", ""),
            remediation_steps=list(raw.get("remediation_steps", [])),
            rerun_job=raw.get("rerun_job", ""),
        )
    return specs
