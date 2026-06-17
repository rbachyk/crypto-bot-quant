"""Immutable, versioned dataset snapshots (Appendix B.5).

A snapshot freezes the exact data a backtest/feature build will read. Its id is
**deterministic** in the coverage window, series set and data content, so a
re-run over the same window reproduces the same id (and is refused as a
duplicate by the immutable store — idempotent re-snapshot). The manifest records
symbols, time range, data types, per-series row counts + checksums, missing
ranges, validation status and source jobs (Appendix B.5 manifest rules).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime

from src.data.config import DataConfig
from src.data.coverage import CoverageReport
from src.data.schema import ms_to_iso
from src.data.store import SeriesStore
from src.storage import DataLake, DatasetManifest


def _deterministic_snapshot_id(cfg: DataConfig, series_checks: dict[str, str]) -> str:
    payload = json.dumps(
        {
            "data_version": cfg.data_version,
            "window": [cfg.window_start_ms, cfg.window_end_ms],
            "series": series_checks,
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{cfg.data_version}_{digest}"


def series_checksums(store: SeriesStore, cfg: DataConfig) -> dict[str, str]:
    start, end = cfg.window_start_ms, cfg.window_end_ms
    return {
        key.label(): store.checksum(key, start, end)
        for symbol in cfg.active_symbols()
        for key in cfg.required_keys(symbol)
    }


def series_row_counts(store: SeriesStore, cfg: DataConfig) -> dict[str, int]:
    start, end = cfg.window_start_ms, cfg.window_end_ms
    return {
        key.label(): store.count(key, start, end)
        for symbol in cfg.active_symbols()
        for key in cfg.required_keys(symbol)
    }


@dataclass(slots=True)
class SnapshotResult:
    snapshot_id: str
    manifest: DatasetManifest
    created: bool  # False if the immutable snapshot already existed (idempotent)
    dataset_checksum: str


def build_dataset_version(
    lake: DataLake,
    store: SeriesStore,
    cfg: DataConfig,
    coverage: CoverageReport,
    validation_status: str,
    source_jobs: list[str],
) -> SnapshotResult:
    checks = series_checksums(store, cfg)
    counts = series_row_counts(store, cfg)
    snapshot_id = _deterministic_snapshot_id(cfg, checks)
    dataset_checksum = hashlib.sha256(
        json.dumps(checks, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]

    missing_ranges = [g.to_dict() for g in coverage.uncovered]
    manifest = DatasetManifest(
        snapshot_id=snapshot_id,
        created_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        symbols=cfg.active_symbols(),
        time_range={"from": ms_to_iso(cfg.window_start_ms), "to": ms_to_iso(cfg.window_end_ms)},
        data_types=cfg.required_series,
        row_counts=counts,
        missing_ranges=missing_ranges,
        validation_status=validation_status,
        source_jobs=source_jobs,
    )

    created = True
    try:
        lake.create_snapshot(manifest)
    except FileExistsError:
        # Immutable snapshot already exists for this exact window+content.
        created = False
        manifest = lake.read_manifest(snapshot_id)

    if created:
        # Per-series checksums as snapshot-local provenance (Appendix B.5).
        checks_path = lake.dataset_dir(snapshot_id) / "series_checksums.json"
        checks_path.write_text(json.dumps(checks, indent=2, sort_keys=True), encoding="utf-8")

    return SnapshotResult(snapshot_id, manifest, created, dataset_checksum)
