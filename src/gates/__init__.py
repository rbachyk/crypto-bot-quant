"""Gate catalog, checks and runner (AGENTS.md Section 25, Appendix A/B.11)."""

from src.gates.catalog import GateSpec, load_catalog
from src.gates.result import Criterion, GateRunResult, GateVerdict
from src.gates.runner import GateRunner

__all__ = [
    "Criterion",
    "GateRunResult",
    "GateRunner",
    "GateSpec",
    "GateVerdict",
    "load_catalog",
]
