"""Data Platform orchestrator (AGENTS.md Section 5 layer 2, Section 8).

Wires the data source, the Parquet series store, the data lake, validation and
the relational index together. It is the single entry point the data jobs, the
``DATA-COV`` / ``DQ`` gates and ``scripts/backfill`` call, so coverage, repair,
validation, snapshotting and reporting all behave identically wherever they are
triggered.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime

import structlog

from src.config import Settings, get_settings
from src.data.config import DataConfig, load_data_config
from src.data.coverage import CoverageReport, compute_coverage
from src.data.ingest import Ingestor
from src.data.schema import SeriesKey
from src.data.snapshot import (
    SnapshotResult,
    SnapshotVerification,
    build_dataset_version,
    verify_snapshot,
)
from src.data.source import DataSource, get_data_source
from src.data.store import SeriesStore
from src.data.validation import DataQualityReport, DataValidator
from src.db.base import session_scope
from src.db.models import DataQualityReportRow, DatasetVersion
from src.storage import DataLake

_log = structlog.get_logger("data.platform")


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
    # Ingestion + coverage use the PER-SERIES end (cfg.series_end_ms): a timeframe coarser
    # than the hour-grid window end (e.g. 4h) may have a still-forming bar before the raw
    # window end — it is neither required nor fetched (the source refuses partial klines).
    def download(self, key: SeriesKey) -> int:
        return self.ingestor.download(key, self.cfg.window_start_ms, self.cfg.series_end_ms(key))

    def update_incremental(self, key: SeriesKey) -> int:
        """Download only the candles that appeared since the last download (the tail up to the
        window end) — the efficient refresh that avoids re-fetching years of existing data."""
        return self.ingestor.update_incremental(
            key, self.cfg.window_start_ms, self.cfg.series_end_ms(key)
        )

    def download_all(self, *, full: bool = False) -> int:
        """Download every required series over the window, returning the count of NEW rows written.

        INCREMENTAL by default: each series resumes from its newest stored ts, so a re-run only
        fetches the candles that appeared since the last download (an empty/never-downloaded series
        still fetches the whole window and records its listing watermark). ``full=True`` forces a
        re-fetch of the entire window per series — use it only to rebuild a suspected-corrupt store;
        writes are idempotent either way, so the full path just re-pulls data already on disk."""
        written = 0
        for key in self.cfg.all_required_keys():
            written += self.download(key) if full else self.update_incremental(key)
        return written

    def ensure_coverage(self, repair: bool = True) -> CoverageReport:
        """Repair safe gaps (auto-remediation: partial) then report coverage."""
        if repair:
            for symbol in self.cfg.active_symbols():
                if not self.source.has_symbol(symbol):
                    continue
                for key in self.cfg.required_keys(symbol):
                    self.ingestor.repair(key, self.cfg.window_start_ms, self.cfg.series_end_ms(key))
        return compute_coverage(self.store, self.cfg)

    # -- validation ------------------------------------------------------ #
    def validate(self) -> DataQualityReport:
        # The source is passed so clock drift can be measured against the venue's server
        # time when a live source is available (offline sources skip the check cleanly).
        return DataValidator(self.store, self.cfg, source=self.source).validate()

    # -- snapshot -------------------------------------------------------- #
    def build_snapshot(
        self, coverage: CoverageReport, validation: DataQualityReport, source_jobs: list[str]
    ) -> SnapshotResult:
        status = "valid" if (coverage.covered and validation.passed) else "invalid"
        result = build_dataset_version(
            self.lake, self.store, self.cfg, coverage, status, source_jobs
        )
        self._persist_dataset_version(result)
        return result

    def verify_snapshot(self, snapshot_id: str) -> SnapshotVerification:
        """Re-verify a snapshot's recorded per-series checksums against the live store
        (verification-on-use — see ``src.data.snapshot`` module docstring)."""
        return verify_snapshot(self.lake, self.store, snapshot_id, self.cfg.exchange_id)

    def _persist_dataset_version(self, result: SnapshotResult) -> None:
        # The MANIFEST is the source of truth for validation_status (content-addressed lake):
        # on an idempotent re-snapshot (created=False) the manifest keeps the ORIGINAL status
        # and the DB row is aligned to it, never overwritten with the current run's status —
        # identical content cannot legitimately change validity, so a differing current status
        # indicates an environment problem, not new information about the snapshot.
        m = result.manifest
        status = m.validation_status
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

    # -- retention (L17) -------------------------------------------------- #
    def prune_snapshots(self, keep_last: int | None = None) -> dict:
        """Retention pass for dataset snapshots (bounds ``as_of: now`` growth).

        With a rolling window every hourly ``run_full`` mints a NEW snapshot id (the id is
        content-addressed over the window), so dataset_versions rows/dirs/quality-reports grow
        without bound. Policy: keep the newest ``keep_last`` snapshots of THIS config's
        fingerprint (``data_version`` + ``exchange_id`` — the platform never touches another
        config's snapshots), plus ANY snapshot still referenced by ``backtest_runs`` (the
        leaderboard reads these rows), ``feature_set_versions``, ``ml_model_registry`` or a
        job's ``related_dataset_version``. Pruning removes the lake directory, the
        ``dataset_versions`` row and its ``data_quality_reports`` rows together (DB rows are
        deleted first and committed; directories are removed after, so a crash can only leave
        an orphaned directory, never a DB row pointing at deleted files)."""
        from sqlalchemy import select

        from src.db.models import BacktestRun, FeatureSetVersion, Job, MLModelRegistry

        keep = self.cfg.snapshot_keep_last if keep_last is None else keep_last
        if keep <= 0:  # retention disabled by config
            return {"keep_last": keep, "pruned": [], "kept_referenced": [], "disabled": True}

        pruned: list[str] = []
        kept_referenced: list[str] = []
        with session_scope() as session:
            rows = (
                session.execute(
                    select(DatasetVersion)
                    .where(
                        DatasetVersion.data_version == self.cfg.data_version,
                        DatasetVersion.exchange_id == self.cfg.exchange_id,
                    )
                    .order_by(DatasetVersion.created_at.desc(), DatasetVersion.version.desc())
                )
                .scalars()
                .all()
            )
            candidates = rows[keep:]
            if candidates:
                referenced: set[str] = set()
                for col in (
                    BacktestRun.dataset_version,
                    FeatureSetVersion.dataset_version,
                    MLModelRegistry.dataset_version,
                    Job.related_dataset_version,
                ):
                    referenced.update(v for (v,) in session.execute(select(col).distinct()) if v)
                for row in candidates:
                    if row.version in referenced:
                        kept_referenced.append(row.version)
                        continue
                    session.query(DataQualityReportRow).filter_by(
                        dataset_version=row.version
                    ).delete()
                    session.delete(row)
                    pruned.append(row.version)
        # Files go after the DB commit (see docstring); a failure here leaves only orphan dirs.
        for version in pruned:
            ddir = self.lake.dataset_dir(version)
            if ddir.exists():
                shutil.rmtree(ddir, ignore_errors=True)
        if pruned or kept_referenced:
            _log.info(
                "snapshots_pruned",
                data_version=self.cfg.data_version,
                keep_last=keep,
                pruned=len(pruned),
                kept_referenced=kept_referenced,
            )
        return {"keep_last": keep, "pruned": pruned, "kept_referenced": kept_referenced}

    # -- reporting ------------------------------------------------------- #
    def write_quality_report(
        self, validation: DataQualityReport, dataset_version: str | None
    ) -> str:
        """Persist the data-validation report to disk + DB (Section 34)."""
        reports_dir = self.settings.reports_path / "data"
        reports_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        path = reports_dir / f"quality_{stamp}.json"
        from src.reporting import wrap_report

        payload = wrap_report(
            {
                "data_version": self.cfg.data_version,
                "dataset_version": dataset_version,
                **validation.to_dict(),
            },
            report_type="data_quality",
            methodology="Per-series coverage + cross-series alignment + clock-drift checks over "
            "the coverage window (Section 23); missing/duplicate/misaligned candles fail.",
            limitations="Validates the stored snapshot; live-stream integrity is the data "
            "manager's job.",
            recommendations="Repair safe gaps; quarantine symbols with insufficient history.",
            period={"window": validation.window},
            versions=self.settings.versions(),
        )
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

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
