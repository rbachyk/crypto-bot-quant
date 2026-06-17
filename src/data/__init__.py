"""Data Platform (AGENTS.md Section 5 layer 2, Section 8, Appendix B.5).

Owns acquisition, validation, storage and versioning of all market data the
bot needs. Phase 2 scope: OHLCV / mark / index / funding / open-interest /
spread ingestion from an offline deterministic source, a Parquet series store
(append-only, deduplicated, checksummed), gap detection + backfill, data-quality
validation, immutable dataset snapshots, and the DATA-COV / DQ gates.
"""

from __future__ import annotations

from src.data.config import DataConfig, load_data_config
from src.data.coverage import CoverageReport, compute_coverage
from src.data.gaps import GapReport, find_gaps
from src.data.ingest import Ingestor
from src.data.platform import DataPlatform, PlatformRun
from src.data.schema import SeriesKey
from src.data.snapshot import SnapshotResult, build_dataset_version
from src.data.source import DataSource, DeterministicSource, get_data_source
from src.data.store import SeriesStore
from src.data.validation import DataQualityReport, DataValidator

__all__ = [
    "CoverageReport",
    "DataConfig",
    "DataPlatform",
    "DataQualityReport",
    "DataSource",
    "DataValidator",
    "DeterministicSource",
    "GapReport",
    "Ingestor",
    "PlatformRun",
    "SeriesKey",
    "SeriesStore",
    "SnapshotResult",
    "build_dataset_version",
    "compute_coverage",
    "find_gaps",
    "get_data_source",
    "load_data_config",
]
