"""Immutable, versioned dataset snapshots (Appendix B.5).

A snapshot freezes the exact data a backtest/feature build will read. Its id is
**deterministic** in the coverage window, series set and data content, so a
re-run over the same window reproduces the same id (and is refused as a
duplicate by the immutable store — idempotent re-snapshot). The manifest records
symbols, time range, data types, per-series row counts + checksums, missing
ranges, validation status and source jobs (Appendix B.5 manifest rules).

**Immutability limitation (verification-on-use, not true pinning).** The
snapshot does NOT copy the series data: consumers read the LIVE Parquet store,
so a ``delete_range`` + re-download can change rows inside a snapshotted window
after the fact. :func:`verify_snapshot` re-computes the per-series checksums
over the snapshotted window and compares them against the recorded
``series_checksums.json`` — validation-critical consumers (the lake backtest
service, the DATA-COV gate, the snapshot job) call it so a drifted snapshot
fails loudly instead of silently validating on different data. This is
verification at consumption time; it cannot prevent the drift itself (a true
content-addressed lake would).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime

from src.data.config import DataConfig
from src.data.coverage import CoverageReport
from src.data.schema import SeriesKey, ms_to_iso, parse_utc_ms
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


# --------------------------------------------------------------------------- #
# Verification-on-use (M19): recompute checksums over the snapshotted window   #
# --------------------------------------------------------------------------- #
def _parse_series_label(label: str) -> tuple[str, str, str]:
    """Invert :meth:`SeriesKey.label` (``symbol:data_type:timeframe``). The symbol itself
    may contain ``:`` (``BTC/USDT:USDT``), so split from the right."""
    symbol, data_type, timeframe = label.rsplit(":", 2)
    return symbol, data_type, timeframe


@dataclass(slots=True)
class SnapshotVerification:
    """Result of re-verifying a snapshot's recorded checksums against the live store."""

    snapshot_id: str
    verifiable: bool  # False: no series_checksums.json (legacy snapshot) — cannot verify
    checked: int = 0
    # label -> {"recorded": ..., "actual": ...} for every series whose current store
    # content no longer matches what was snapshotted.
    mismatches: dict[str, dict[str, str]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.verifiable and not self.mismatches

    def summary(self) -> str:
        if not self.verifiable:
            return f"{self.snapshot_id}: unverifiable (no series_checksums.json — legacy snapshot)"
        if self.mismatches:
            sample = ", ".join(list(self.mismatches)[:5])
            return (
                f"{self.snapshot_id}: {len(self.mismatches)}/{self.checked} series changed "
                f"since snapshot (e.g. {sample}) — the lake was modified inside the snapshotted "
                "window (delete_range + re-download?); re-snapshot before validation-critical use"
            )
        return f"{self.snapshot_id}: {self.checked} series checksums verified"


def verify_snapshot(
    lake: DataLake, store: SeriesStore, snapshot_id: str, exchange_id: str
) -> SnapshotVerification:
    """Recompute every recorded per-series checksum over the snapshotted window and compare.

    Snapshots pin metadata, not bytes (see module docstring): this detects post-snapshot
    mutation of the underlying store at consumption time. Raises ``FileNotFoundError`` if the
    snapshot itself does not exist; a snapshot without a checksum sidecar (created before
    checksums were recorded) returns ``verifiable=False`` so callers can warn-and-proceed."""
    manifest = lake.read_manifest(snapshot_id)  # raises if the snapshot is missing
    checks_path = lake.dataset_dir(snapshot_id) / "series_checksums.json"
    if not checks_path.exists():
        return SnapshotVerification(snapshot_id=snapshot_id, verifiable=False)
    recorded: dict[str, str] = json.loads(checks_path.read_text(encoding="utf-8"))
    start = parse_utc_ms(manifest.time_range["from"])
    end = parse_utc_ms(manifest.time_range["to"])

    out = SnapshotVerification(snapshot_id=snapshot_id, verifiable=True)
    for label, expected in sorted(recorded.items()):
        symbol, data_type, timeframe = _parse_series_label(label)
        key = SeriesKey(exchange_id, data_type, symbol, timeframe)
        actual = store.checksum(key, start, end)
        out.checked += 1
        if actual != expected:
            out.mismatches[label] = {"recorded": expected, "actual": actual}
    return out
