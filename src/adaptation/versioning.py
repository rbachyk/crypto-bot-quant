"""Learner versioning and frozen snapshots (AGENTS.md Section 21.7, Section 4).

Every accepted update in LIVE_BOUNDED mode writes a new ``learner_version``
snapshot so any version is restorable (Section 21.7). Phase 11 delivers the
snapshot/restore infrastructure; it is exercised in shadow mode to verify the
round-trip.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass
class SnapshotMeta:
    """Metadata record for a saved policy snapshot."""

    snapshot_id: str
    learner_id: str
    learner_version: str
    mode: str
    created_at: str
    size_bytes: int
    checksum: str


def _checksum(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()[:16]


def save_snapshot(
    policy_blob: bytes,
    learner_id: str,
    learner_version: str,
    mode: str,
    snapshot_dir: Path,
) -> SnapshotMeta:
    """Persist a policy snapshot and return its metadata.

    The snapshot file is named ``<learner_id>_<learner_version>_<checksum>.pkl``
    so it is immutable (any change → different name) and idempotent.
    """
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    cksum = _checksum(policy_blob)
    filename = f"{learner_id}_{learner_version}_{cksum}.pkl"
    path = snapshot_dir / filename
    if not path.exists():
        path.write_bytes(policy_blob)

    stamp = datetime.now(UTC).isoformat()
    return SnapshotMeta(
        snapshot_id=filename,
        learner_id=learner_id,
        learner_version=learner_version,
        mode=mode,
        created_at=stamp,
        size_bytes=len(policy_blob),
        checksum=cksum,
    )


def load_snapshot(snapshot_id: str, snapshot_dir: Path) -> bytes:
    """Load a saved policy snapshot blob.

    Raises :exc:`FileNotFoundError` if the snapshot does not exist — the caller
    should fall back to the frozen fallback policy (Section 21.7).
    """
    path = snapshot_dir / snapshot_id
    if not path.exists():
        raise FileNotFoundError(f"snapshot not found: {path}")
    blob = path.read_bytes()
    # Verify integrity.
    cksum = _checksum(blob)
    if snapshot_id.endswith(f"_{cksum}.pkl"):
        return blob
    # checksum embedded in name does not match — refuse to load corrupted snapshot.
    raise ValueError(f"snapshot checksum mismatch: {snapshot_id}")


def make_frozen_fallback(
    policy_blob: bytes,
    snapshot_dir: Path,
    filename: str = "frozen_fallback.pkl",
) -> Path:
    """Write the frozen fallback policy blob to ``snapshot_dir/filename``.

    This is the last-manually-approved policy the system reverts to on any
    rollback trigger (Section 2.2, 21.7).
    """
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_dir / filename
    path.write_bytes(policy_blob)
    return path


def load_frozen_fallback(snapshot_dir: Path, filename: str = "frozen_fallback.pkl") -> bytes:
    """Load the frozen fallback blob. Raises :exc:`FileNotFoundError` if absent."""
    path = snapshot_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"frozen fallback not found: {path}")
    return path.read_bytes()
