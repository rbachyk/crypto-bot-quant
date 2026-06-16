"""Data lake & artifact storage (AGENTS.md Appendix B.5)."""

from src.storage.datalake import DataLake, DatasetManifest, new_snapshot_id

__all__ = ["DataLake", "DatasetManifest", "new_snapshot_id"]
