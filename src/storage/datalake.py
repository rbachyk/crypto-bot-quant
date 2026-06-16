"""Local data lake & artifact store with versioned snapshots and manifests.

Implements the Phase 1 slice of AGENTS.md Appendix B.5: a writable, versioned
store where every dataset snapshot has an id and a manifest (symbols, time
range, data types, row counts, missing ranges, validation status, source jobs).
The MVP backend is the local filesystem; MinIO/S3 are drop-in later (B.5).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

# Monotonic-ish counter so two snapshots created in the same second differ.
_counter = 0


def new_snapshot_id(prefix: str = "ds") -> str:
    """Generate a unique, sortable dataset-snapshot id.

    Time is read at call sites (not via a frozen clock) so ids stay unique;
    a process-local counter disambiguates same-instant calls.
    """
    global _counter
    _counter += 1
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    return f"{prefix}_{ts}_{_counter:04d}"


@dataclass(slots=True)
class DatasetManifest:
    """Manifest for a dataset version (Appendix B.5)."""

    snapshot_id: str
    created_at: str
    symbols: list[str] = field(default_factory=list)
    time_range: dict[str, str] = field(default_factory=dict)
    data_types: list[str] = field(default_factory=list)
    row_counts: dict[str, int] = field(default_factory=dict)
    missing_ranges: list[dict] = field(default_factory=list)
    validation_status: str = "unvalidated"
    source_jobs: list[str] = field(default_factory=list)
    checksum: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class DataLake:
    """Filesystem-backed data lake / artifact store.

    Layout (Appendix B.5 partitions are applied by ingestion in later phases):
        <root>/datasets/<snapshot_id>/manifest.json
        <root>/datasets/<snapshot_id>/<files...>
    """

    def __init__(self, root: Path, artifact_root: Path | None = None) -> None:
        self.root = Path(root)
        self.artifact_root = Path(artifact_root) if artifact_root else self.root / "artifacts"

    # -- lifecycle ------------------------------------------------------- #
    def ensure_ready(self) -> None:
        """Create the lake directories. Raises on permission errors so the
        Storage gate can surface them (Appendix B.11: 'permission errors are
        surfaced')."""
        for path in (self.root, self.root / "datasets", self.artifact_root, self.root / "reports"):
            path.mkdir(parents=True, exist_ok=True)

    def writable(self) -> bool:
        """True if the lake root accepts writes (used by the Storage gate)."""
        try:
            self.ensure_ready()
            probe = self.root / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            return True
        except OSError:
            return False

    # -- snapshots ------------------------------------------------------- #
    def dataset_dir(self, snapshot_id: str) -> Path:
        return self.root / "datasets" / snapshot_id

    def create_snapshot(self, manifest: DatasetManifest) -> Path:
        """Create a versioned dataset directory and persist its manifest.

        Snapshots are immutable: re-creating an existing snapshot id is refused
        (datasets are never silently overwritten; Appendix B.17)."""
        ddir = self.dataset_dir(manifest.snapshot_id)
        if ddir.exists():
            raise FileExistsError(f"dataset snapshot already exists: {manifest.snapshot_id}")
        ddir.mkdir(parents=True)
        self.write_manifest(manifest)
        return ddir

    def write_manifest(self, manifest: DatasetManifest) -> Path:
        ddir = self.dataset_dir(manifest.snapshot_id)
        ddir.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(manifest.to_dict(), indent=2, sort_keys=True)
        manifest.checksum = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        path = ddir / "manifest.json"
        path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return path

    def read_manifest(self, snapshot_id: str) -> DatasetManifest:
        path = self.dataset_dir(snapshot_id) / "manifest.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        return DatasetManifest(**data)

    def list_snapshots(self) -> list[str]:
        ddir = self.root / "datasets"
        if not ddir.exists():
            return []
        return sorted(p.name for p in ddir.iterdir() if p.is_dir())

    # -- artifacts ------------------------------------------------------- #
    def write_artifact(self, relative_path: str, content: bytes) -> Path:
        """Write a versioned artifact (model/report/etc.) under the artifact root."""
        path = self.artifact_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    @staticmethod
    def free_bytes(path: Path) -> int:
        try:
            return os.statvfs(path).f_bavail * os.statvfs(path).f_frsize
        except (OSError, AttributeError):  # pragma: no cover - non-posix
            return -1
