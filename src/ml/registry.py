"""ML model artifact registry (AGENTS.md Section 20 model requirements).

Every model used by the shadow predictor is versioned and recorded in the
``ml_model_registry`` table.  Artifacts (pickled model states) are stored in
the data lake under ``ml/models/<model_id>.pkl``.

Required registry fields per AGENTS.md:
  * model_id, data/feature versions, label & target definitions
  * train/validation/OOS periods, performance + calibration + explainability
  * known failure modes, promotion status
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path


@dataclass
class ModelArtifact:
    """In-memory representation of a model artifact record."""

    model_id: str
    model_version: str
    model_type: str
    ml_stage: int = 2  # Shadow
    promotion_status: str = "shadow"
    dataset_version: str | None = None
    feature_set_version: str | None = None
    train_period: dict = field(default_factory=dict)
    oos_period: dict = field(default_factory=dict)
    label_definition: dict = field(default_factory=dict)
    performance_metrics: dict = field(default_factory=dict)
    known_failure_modes: list[str] = field(default_factory=list)
    artifact_path: str | None = None
    manually_reviewed: bool = False
    notes: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class MLRegistry:
    """Versioned model artifact registry backed by the database and data lake.

    Usage::

        registry = MLRegistry(artifact_path)
        registry.register(model, artifact)  # persist model + record
        registry.load(model_id)             # restore model state
    """

    def __init__(self, artifact_root: Path) -> None:
        self._root = artifact_root / "ml" / "models"
        self._root.mkdir(parents=True, exist_ok=True)

    def save_artifact(self, model_id: str, blob: bytes) -> str:
        """Write model pickle to the artifact store; return the path."""
        path = self._root / f"{model_id}.pkl"
        path.write_bytes(blob)
        return str(path)

    def load_artifact(self, model_id: str) -> bytes | None:
        """Read model pickle from the artifact store; None if not found."""
        path = self._root / f"{model_id}.pkl"
        if path.exists():
            return path.read_bytes()
        return None

    def register(
        self,
        model: object,
        artifact: ModelArtifact,
        *,
        write_db: bool = True,
    ) -> ModelArtifact:
        """Serialize model state to disk and record metadata in the DB.

        Returns the artifact with ``artifact_path`` populated.
        """
        blob = model.snapshot()  # type: ignore[attr-defined]
        path = self.save_artifact(artifact.model_id, blob)
        artifact.artifact_path = path

        if write_db:
            _upsert_db(artifact)
        return artifact

    def load(self, model_id: str) -> bytes | None:
        """Return the raw bytes for a model artifact (caller unpickles)."""
        return self.load_artifact(model_id)


def _upsert_db(artifact: ModelArtifact) -> None:
    from src.db.base import session_scope
    from src.db.models import MLModelRegistry

    with session_scope() as session:
        # Look up by model_id (unique).
        existing = (
            session.query(MLModelRegistry).filter_by(model_id=artifact.model_id).one_or_none()
        )
        if existing is None:
            existing = MLModelRegistry(model_id=artifact.model_id)
            session.add(existing)

        existing.model_version = artifact.model_version
        existing.model_type = artifact.model_type
        existing.ml_stage = artifact.ml_stage
        existing.promotion_status = artifact.promotion_status
        existing.dataset_version = artifact.dataset_version
        existing.feature_set_version = artifact.feature_set_version
        existing.train_period = artifact.train_period
        existing.oos_period = artifact.oos_period
        existing.label_definition = artifact.label_definition
        existing.performance_metrics = artifact.performance_metrics
        existing.known_failure_modes = artifact.known_failure_modes
        existing.artifact_path = artifact.artifact_path
        existing.manually_reviewed = artifact.manually_reviewed
        existing.notes = artifact.notes
