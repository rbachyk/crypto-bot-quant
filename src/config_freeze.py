"""Config-freeze manifest (CONFIG-FREEZE gate; AGENTS.md Section 4).

Freezing records the full version set (config/strategy/data/risk/exec/model/RL/…) + the git
commit into a manifest. The CONFIG-FREEZE gate then proves the *running* config matches that
deliberately-frozen manifest — i.e. there is no drift — instead of merely checking that version
strings are non-empty. Re-freeze (``make config-freeze``) intentionally whenever versions change.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from src.approvals import current_git_commit
from src.config import Settings, get_settings


def manifest_path(settings: Settings) -> Path:
    return settings.reports_path / "phase_13" / "config_freeze.json"


def freeze_config(settings: Settings | None = None) -> str:
    """Write the freeze manifest from the current settings; returns the path."""
    settings = settings or get_settings()
    path = manifest_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "versions": settings.versions(),
        "git_commit": current_git_commit(),
        "frozen_at": datetime.now(UTC).isoformat(),
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return str(path)


def load_manifest(settings: Settings | None = None) -> dict | None:
    settings = settings or get_settings()
    path = manifest_path(settings)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (ValueError, OSError):
        return None
