"""Loader for ``configs/gates.yaml`` — the single source of truth for gates.

The Gate Runner, dashboard and Reviewer all read the same catalog (Appendix A).

``blocks_live`` semantics (audit H13/M25):
  * YAML booleans and strings are normalised case-insensitively ("true"/"false").
  * Conditional values ("if_ml_enabled", "if_learner_enabled", "if_rl_enabled")
    are resolved against the owning subsystem's config by
    :meth:`GateSpec.blocks_live_resolved` — consumers must call that helper
    instead of comparing the raw string, otherwise conditional gates are
    silently treated as non-critical.
  * The ``defaults:`` block in gates.yaml supplies the fallback for gates that
    omit ``blocks_live``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from src.config.settings import REPO_ROOT

GATES_YAML = REPO_ROOT / "configs" / "gates.yaml"

# Canonical blocks_live values after normalisation. Conditionals resolve at read
# time via GateSpec.blocks_live_resolved().
_BLOCKS_LIVE_VALUES = {"true", "false", "if_ml_enabled", "if_learner_enabled", "if_rl_enabled"}


def _normalize_blocks_live(raw: object) -> str:
    """Normalise a raw YAML ``blocks_live`` value to its canonical string form.

    YAML parses a bare ``true``/``false`` as a Python bool — ``str(True)``
    produced ``"True"`` which failed every ``== "true"`` comparison and silently
    demoted the gate to non-critical (audit M25). Accept booleans and strings
    case-insensitively; reject unknown values loudly so a typo in gates.yaml can
    never silently demote a live-blocking gate.
    """
    if isinstance(raw, bool):
        return "true" if raw else "false"
    value = str(raw).strip().lower()
    if value not in _BLOCKS_LIVE_VALUES:
        raise ValueError(
            f"invalid blocks_live value {raw!r} in gates.yaml; "
            f"expected one of {sorted(_BLOCKS_LIVE_VALUES)}"
        )
    return value


def _ml_influences_decisions() -> bool:
    """True when the ML subsystem is configured past pure shadow (configs/ml.yaml).

    Stage >= 3 surfaces recommendations to the operator and Stage 4 can BLOCK
    live candidates; either makes the ML promotion gate live-critical. The
    stage-3/4 feature switches are checked independently of the declared stage
    so an inconsistent config still fails closed.
    """
    from src.ml.config import load_ml_config

    cfg = load_ml_config()
    return cfg.ml_stage >= 3 or cfg.recommendation.enabled or cfg.filter.enabled


def _learner_enabled() -> bool:
    """The bounded online learner's master switch (configs/adaptation.yaml)."""
    from src.adaptation.config import load_adaptation_config

    return load_adaptation_config().enabled


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

    def blocks_live_resolved(self) -> bool:
        """Resolve ``blocks_live`` to a concrete bool (audit H13).

        Deliberate conditional → config-flag mapping:
          * ``"true"`` / ``"false"`` — literal.
          * ``"if_ml_enabled"`` — ``configs/ml.yaml``: ``ml_stage >= 3`` or the
            Stage-3 recommendation / Stage-4 constrained-live-filter switches
            are on. Below that the ML models are shadow-only and cannot touch a
            live decision, so ML-PROMO need not block live.
          * ``"if_learner_enabled"`` — ``configs/adaptation.yaml``
            ``adaptation.enabled`` (the bounded learner's master switch).
          * ``"if_rl_enabled"`` — the same ``adaptation.enabled`` switch: RL has
            no independent enable flag; RL policies
            (src/adaptation/policies/rl_policy.py) only ever execute inside the
            adaptation controller, so the RL gates block live exactly when the
            adaptation subsystem does.

        Any error reading a subsystem config resolves to ``True`` — fail
        closed: an unreadable config must never demote a live-blocking gate.
        """
        if self.blocks_live == "true":
            return True
        if self.blocks_live == "false":
            return False
        try:
            if self.blocks_live == "if_ml_enabled":
                return _ml_influences_decisions()
            if self.blocks_live in ("if_learner_enabled", "if_rl_enabled"):
                return _learner_enabled()
        except Exception:  # noqa: BLE001 - unreadable subsystem config → fail closed
            return True
        # Unknown value (unreachable after load-time normalisation): fail closed.
        return True


@lru_cache
def load_catalog(path: str | None = None) -> dict[str, GateSpec]:
    """Parse the gate catalog into ``{gate_id: GateSpec}``."""
    yaml_path = Path(path) if path else GATES_YAML
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    # The ``defaults:`` block supplies the fallback for per-gate keys that GateSpec
    # models — today that is only ``blocks_live`` (``manual_review`` and
    # ``auto_remediation`` are operator-facing documentation not consumed by code).
    # Honoured rather than removed so gates.yaml stays self-describing (audit M25:
    # this block was previously dead and the fallback was hardcoded here).
    defaults = data.get("defaults") or {}
    default_blocks_live = _normalize_blocks_live(defaults.get("blocks_live", True))
    specs: dict[str, GateSpec] = {}
    for raw in data.get("gates", []):
        specs[raw["id"]] = GateSpec(
            gate_id=raw["id"],
            name=raw.get("name", raw["id"]),
            phase=str(raw.get("phase", "")),
            depends_on=list(raw.get("depends_on", [])),
            blocks_live=_normalize_blocks_live(raw.get("blocks_live", default_blocks_live)),
            pass_condition=raw.get("pass_condition", ""),
            remediation_steps=list(raw.get("remediation_steps", [])),
            rerun_job=raw.get("rerun_job", ""),
        )
    return specs
