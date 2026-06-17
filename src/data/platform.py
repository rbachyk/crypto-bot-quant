"""Data Platform orchestrator (AGENTS.md Section 5 layer 2, Section 8).

Wires the data source, the Parquet series store, the data lake, validation and
the relational index together. It is the single entry point the data jobs, the
``DATA-COV`` / ``DQ`` gates and ``scripts/backfill`` call, so coverage, repair,
validation, snapshotting and reporting all behave identically wherever they are
triggered.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime

from src.config import Settings, get_settings
from src.data.config import DataConfig, load_data_config
from src.data.coverage import CoverageReport, compute_coverage
from src.data.ingest import Ingestor
from src.data.schema import SeriesKey
from src.data.snapshot import SnapshotResult, build_dataset_version
from src.data.source import DataSource, get_data_source
from src.data.store import SeriesStore
from src.data.validation import DataQualityReport, DataValidator
from src.db.base import session_scope
from src.db.models import DataQualityReportRow, DatasetVersion
from src.storage import DataLake


@dataclass(slots=True)
class PlatformRun:
    coverage: CoverageReport
    validation: DataQualityReport
    snapshot: SnapshotResult
    report_path: str


class DataPlatform:
    def __init__(
        self,
        settings: Settings | None = None,
        cfg: DataConfig | None = None,
        source: DataSource | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.cfg = cfg or load_data_config()
        self.source = source or get_data_source(self.cfg.exchange_id)
        self.store = SeriesStore(self.settings.data_lake_path)
        self.lake = DataLake(self.settings.data_lake_path, self.settings.artifact_path)
        self.lake.ensure_ready()
        self.ingestor = Ingestor(self.source, self.store)

    # -- ingestion ------------------------------------------------------- #
    def download(self, key: SeriesKey) -> int:
        return self.ingestor.download(key, self.cfg.window_start_ms, self.cfg.window_end_ms)

    def download_all(self) -> int:
        """Idempotent full download of every required series over the window."""
        written = 0
        for key in self.cfg.all_required_keys():
            written += self.download(key)
        return written

    def ensure_coverage(self, repair: bool = True) -> CoverageReport:
        """Repair safe gaps (auto-remediation: partial) then report coverage."""
        if repair:
            for symbol in self.cfg.active_symbols():
                if not self.source.has_symbol(symbol):
                    continue
                for key in self.cfg.required_keys(symbol):
                    self.ingestor.repair(key, self.cfg.window_start_ms, self.cfg.window_end_ms)
        return compute_coverage(self.store, self.cfg)

    # -- validation ------------------------------------------------------ #
    def validate(self) -> DataQualityReport:
        return DataValidator(self.store, self.cfg).validate()

    # -- snapshot -------------------------------------------------------- #
    def build_snapshot(
        self, coverage: CoverageReport, validation: DataQualityReport, source_jobs: list[str]
    ) -> SnapshotResult:
        status = "valid" if (coverage.covered and validation.passed) else "invalid"
        result = build_dataset_version(
            self.lake, self.store, self.cfg, coverage, status, source_jobs
        )
        self._persist_dataset_version(result, status)
        return result

    def _persist_dataset_version(self, result: SnapshotResult, status: str) -> None:
        m = result.manifest
        with session_scope() as session:
            row = session.get(DatasetVersion, result.snapshot_id)
            if row is None:
                row = DatasetVersion(version=result.snapshot_id)
                session.add(row)
            row.data_version = self.cfg.data_version
            row.exchange_id = self.cfg.exchange_id
            row.symbols = m.symbols
            row.data_types = m.data_types
            row.timeframes = list(self.cfg.timeframes)
            row.time_range = m.time_range
            row.row_counts = m.row_counts
            row.missing_ranges = m.missing_ranges
            row.checksum = result.dataset_checksum
            row.validation_status = status
            row.manifest_path = str(self.lake.dataset_dir(result.snapshot_id) / "manifest.json")
            row.source_jobs = m.source_jobs

    # -- reporting ------------------------------------------------------- #
    def write_quality_report(
        self, validation: DataQualityReport, dataset_version: str | None
    ) -> str:
        """Persist the data-validation report to disk + DB (Section 34)."""
        reports_dir = self.settings.reports_path / "data"
        reports_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        path = reports_dir / f"quality_{stamp}.json"
        payload = {
            "data_version": self.cfg.data_version,
            "dataset_version": dataset_version,
            "versions": self.settings.versions(),
            **validation.to_dict(),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        with session_scope() as session:
            session.add(
                DataQualityReportRow(
                    dataset_version=dataset_version,
                    passed=validation.passed,
                    critical_count=len(validation.critical),
                    violation_count=len(validation.violations),
                    series_validated=validation.series_validated,
                    window=validation.window,
                    report=validation.to_dict(),
                    report_path=str(path),
                )
            )
        return str(path)

    # -- end to end ------------------------------------------------------ #
    def run_full(self, repair: bool = True, source_jobs: list[str] | None = None) -> PlatformRun:
        """Coverage (+repair) -> validate -> snapshot -> persist report.

        This is what ``build_dataset_version`` and the DATA-COV/DQ gates call so
        the platform always reaches a consistent, recorded state."""
        coverage = self.ensure_coverage(repair=repair)
        validation = self.validate()
        snapshot = self.build_snapshot(
            coverage, validation, source_jobs or ["data_platform.run_full"]
        )
        report_path = self.write_quality_report(validation, snapshot.snapshot_id)
        return PlatformRun(coverage, validation, snapshot, report_path)
