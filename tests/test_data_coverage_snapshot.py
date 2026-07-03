"""Coverage computation (DATA-COV) and immutable dataset snapshots (B.5)."""

from __future__ import annotations

import json
import uuid
from dataclasses import replace

import pytest
from src.data.coverage import compute_coverage
from src.data.schema import OHLCV, SeriesKey
from src.data.snapshot import build_dataset_version, verify_snapshot
from src.storage import DataLake, DatasetManifest, manifest_checksum

from tests._data_helpers import fresh_store, populate, small_cfg
from tests.conftest import requires_db


def test_coverage_complete_when_populated(tmp_path) -> None:
    cfg = small_cfg(symbols=("BTC/USDT:USDT", "ETH/USDT:USDT"))
    store = fresh_store(tmp_path)
    populate(store, cfg)
    cov = compute_coverage(store, cfg)
    assert cov.covered
    assert cov.covered_series == cov.required_series


def test_coverage_reports_uncovered_series(tmp_path) -> None:
    cfg = small_cfg()
    store = fresh_store(tmp_path)
    populate(store, cfg)
    key = SeriesKey(cfg.exchange_id, OHLCV, cfg.symbols[0], "5m")
    # An INTERIOR hole (not the leading edge, which is a listing boundary, not a gap).
    store.delete_range(
        key, cfg.window_start_ms + 10 * key.interval_ms, cfg.window_start_ms + 11 * key.interval_ms
    )
    cov = compute_coverage(store, cfg)
    assert not cov.covered
    assert any(g.key.data_type == OHLCV for g in cov.uncovered)


def test_leading_pre_listing_absence_is_not_a_gap(tmp_path) -> None:
    """A contract listed AFTER the window start (ETH/SOL perps vs a multi-year BTC window) has no
    data before its listing — that leading absence must NOT be counted as missing, while an
    interior hole still is. Regression for full-history downloads failing validation."""
    from src.data.gaps import find_gaps

    cfg = small_cfg()
    store = fresh_store(tmp_path)
    populate(store, cfg)
    key = SeriesKey(cfg.exchange_id, OHLCV, cfg.symbols[0], "5m")
    # Simulate the contract listing 20 candles into the window (no data before that point).
    store.delete_range(key, cfg.window_start_ms, cfg.window_start_ms + 20 * key.interval_ms)
    gap = find_gaps(store, key, cfg.window_start_ms, cfg.window_end_ms)
    assert gap.covered and not gap.missing_ts  # leading pre-listing absence ⇒ no gap

    # ...but an INTERIOR hole after listing IS still reported.
    store.delete_range(
        key, cfg.window_start_ms + 30 * key.interval_ms, cfg.window_start_ms + 31 * key.interval_ms
    )
    gap2 = find_gaps(store, key, cfg.window_start_ms, cfg.window_end_ms)
    assert not gap2.covered and gap2.missing_ts


def test_insufficient_history_symbol_is_excluded(tmp_path) -> None:
    cfg = small_cfg(symbols=("BTC/USDT:USDT", "DEAD/USDT:USDT"), insufficient=("DEAD/USDT:USDT",))
    store = fresh_store(tmp_path)
    populate(store, cfg)  # DEAD has no data, but it is excluded from required
    cov = compute_coverage(store, cfg)
    assert cov.covered
    assert "DEAD/USDT:USDT" in cov.insufficient_history


def _lake(tmp_path) -> DataLake:
    return DataLake(tmp_path / "lake", tmp_path / "art")


def test_snapshot_is_deterministic_and_immutable(tmp_path) -> None:
    cfg = small_cfg()
    store = fresh_store(tmp_path)
    populate(store, cfg)
    lake = _lake(tmp_path)
    lake.ensure_ready()
    cov = compute_coverage(store, cfg)

    first = build_dataset_version(lake, store, cfg, cov, "valid", ["test"])
    assert first.created
    assert first.snapshot_id.startswith("data_test_")
    assert first.manifest.row_counts
    assert first.manifest.checksum  # manifest checksum populated on write
    # Re-snapshotting the same window+content reuses the immutable id (idempotent).
    second = build_dataset_version(lake, store, cfg, cov, "valid", ["test"])
    assert not second.created
    assert second.snapshot_id == first.snapshot_id


def test_snapshot_records_missing_ranges_when_uncovered(tmp_path) -> None:
    cfg = small_cfg()
    store = fresh_store(tmp_path)
    populate(store, cfg)
    key = SeriesKey(cfg.exchange_id, OHLCV, cfg.symbols[0], "5m")
    # An INTERIOR hole (a real gap), not the leading listing boundary.
    store.delete_range(
        key, cfg.window_start_ms + 10 * key.interval_ms, cfg.window_start_ms + 12 * key.interval_ms
    )
    lake = _lake(tmp_path)
    lake.ensure_ready()
    cov = compute_coverage(store, cfg)
    result = build_dataset_version(lake, store, cfg, cov, "invalid", ["test"])
    assert result.manifest.validation_status == "invalid"
    assert result.manifest.missing_ranges


# --------------------------------------------------------------------------- #
# M19: snapshots pin nothing — verification-on-use must catch store mutation   #
# --------------------------------------------------------------------------- #
def _snapshot(tmp_path, cfg):
    store = fresh_store(tmp_path)
    populate(store, cfg)
    lake = _lake(tmp_path)
    lake.ensure_ready()
    cov = compute_coverage(store, cfg)
    return store, lake, build_dataset_version(lake, store, cfg, cov, "valid", ["test"])


def test_verify_snapshot_passes_on_untouched_store(tmp_path) -> None:
    cfg = small_cfg()
    store, lake, result = _snapshot(tmp_path, cfg)
    v = verify_snapshot(lake, store, result.snapshot_id, cfg.exchange_id)
    assert v.verifiable and v.ok
    assert v.checked == len(result.manifest.row_counts)
    assert v.mismatches == {}


def test_verify_snapshot_detects_post_snapshot_mutation(tmp_path) -> None:
    """delete_range inside a snapshotted window changes what consumers read while the
    snapshot id/manifest stay put — verification must flag exactly the mutated series."""
    cfg = small_cfg()
    store, lake, result = _snapshot(tmp_path, cfg)
    key = SeriesKey(cfg.exchange_id, OHLCV, cfg.symbols[0], "5m")
    store.delete_range(
        key, cfg.window_start_ms + 5 * key.interval_ms, cfg.window_start_ms + 7 * key.interval_ms
    )
    v = verify_snapshot(lake, store, result.snapshot_id, cfg.exchange_id)
    assert not v.ok
    assert key.label() in v.mismatches
    assert "changed since snapshot" in v.summary()
    # Untouched series still verify — the mismatch set is precise.
    assert all(label == key.label() for label in v.mismatches)


def test_verify_snapshot_legacy_without_sidecar_is_unverifiable(tmp_path) -> None:
    cfg = small_cfg()
    store, lake, result = _snapshot(tmp_path, cfg)
    (lake.dataset_dir(result.snapshot_id) / "series_checksums.json").unlink()
    v = verify_snapshot(lake, store, result.snapshot_id, cfg.exchange_id)
    assert not v.verifiable and not v.ok
    assert "unverifiable" in v.summary()


def test_lake_backtest_refuses_mismatched_snapshot(tmp_path) -> None:
    """The validation-critical consumer: a lake backtest asked to run over a snapshot whose
    underlying rows changed must fail loudly, not silently use different data."""
    from src.backtest.service import _verify_dataset_snapshot
    from src.config import get_settings

    cfg = small_cfg()
    store, lake, result = _snapshot(tmp_path, cfg)
    settings = get_settings().model_copy(
        update={"data_lake_path": tmp_path / "lake", "artifact_path": tmp_path / "art"}
    )
    # Untouched store: verification passes silently.
    _verify_dataset_snapshot(settings, store, cfg, result.snapshot_id)
    # A non-snapshot id (bare DATA_VERSION policy label) is not verifiable → no-op.
    _verify_dataset_snapshot(settings, store, cfg, cfg.data_version)
    # Mutated store: the consumer must refuse.
    key = SeriesKey(cfg.exchange_id, OHLCV, cfg.symbols[0], "5m")
    store.delete_range(key, cfg.window_start_ms, cfg.window_start_ms + 2 * key.interval_ms)
    with pytest.raises(RuntimeError, match="verification FAILED"):
        _verify_dataset_snapshot(settings, store, cfg, result.snapshot_id)


# --------------------------------------------------------------------------- #
# M21: idempotent re-snapshot must not flip validation_status (manifest wins)  #
# --------------------------------------------------------------------------- #
def test_resnapshot_keeps_original_manifest_status(tmp_path) -> None:
    cfg = small_cfg()
    store, lake, first = _snapshot(tmp_path, cfg)  # status "valid"
    cov = compute_coverage(store, cfg)
    # Same content, but the CURRENT run judges it "invalid" (e.g. a transient env problem):
    # identical content cannot change validity — the original manifest status is kept.
    second = build_dataset_version(lake, store, cfg, cov, "invalid", ["test"])
    assert not second.created
    assert second.snapshot_id == first.snapshot_id
    assert second.manifest.validation_status == "valid"


@requires_db
def test_resnapshot_does_not_overwrite_db_status(tmp_path) -> None:
    """DB row and lake manifest must agree after an idempotent re-snapshot: the DB row keeps
    the manifest's (original) status instead of the current run's."""
    from src.data.platform import DataPlatform
    from src.data.validation import CRITICAL, DataValidator, Violation
    from src.db.base import session_scope
    from src.db.models import DatasetVersion

    cfg = replace(small_cfg(), data_version=f"m21_{uuid.uuid4().hex[:8]}")
    store = fresh_store(tmp_path)
    populate(store, cfg)
    platform = DataPlatform(cfg=cfg)
    platform.store = store
    platform.lake = _lake(tmp_path)
    platform.lake.ensure_ready()

    cov = compute_coverage(store, cfg)
    validation = DataValidator(store, cfg).validate()
    assert cov.covered and validation.passed
    first = platform.build_snapshot(cov, validation, ["test"])
    assert first.created

    # Re-snapshot the identical content while the current run reports a critical violation.
    validation.violations.append(Violation("forced", CRITICAL, "current-run-only failure"))
    second = platform.build_snapshot(cov, validation, ["test"])
    assert not second.created and second.snapshot_id == first.snapshot_id
    assert second.manifest.validation_status == "valid"
    try:
        with session_scope() as session:
            row = session.get(DatasetVersion, first.snapshot_id)
            assert row is not None
            assert row.validation_status == "valid"  # pre-fix: overwritten to "invalid"
    finally:
        with session_scope() as session:
            row = session.get(DatasetVersion, first.snapshot_id)
            if row is not None:
                session.delete(row)


# --------------------------------------------------------------------------- #
# L16: manifest checksum is over content (checksum field zeroed) + verified    #
# --------------------------------------------------------------------------- #
def _manifest(snapshot_id: str) -> DatasetManifest:
    return DatasetManifest(
        snapshot_id=snapshot_id,
        created_at="2026-06-01T00:00:00Z",
        symbols=["BTC/USDT:USDT"],
        time_range={"from": "2026-05-31T16:00:00Z", "to": "2026-06-01T00:00:00Z"},
        data_types=["ohlcv"],
        row_counts={"BTC/USDT:USDT:ohlcv:5m": 96},
        validation_status="valid",
    )


def test_manifest_checksum_roundtrip_and_stability(tmp_path) -> None:
    lake = _lake(tmp_path)
    lake.ensure_ready()
    m = _manifest("ds_l16_roundtrip")
    lake.write_manifest(m)
    assert m.checksum == manifest_checksum(m.to_dict())
    back = lake.read_manifest("ds_l16_roundtrip")  # verifies on read
    assert back.checksum == m.checksum
    # Re-writing a manifest that already carries a checksum must NOT drift (the old scheme
    # hashed the JSON containing the previous checksum, changing it on every write).
    lake.write_manifest(back)
    assert lake.read_manifest("ds_l16_roundtrip").checksum == m.checksum


def test_manifest_tampering_detected_on_read(tmp_path) -> None:
    lake = _lake(tmp_path)
    lake.ensure_ready()
    lake.write_manifest(_manifest("ds_l16_tamper"))
    path = lake.dataset_dir("ds_l16_tamper") / "manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["validation_status"] = "valid_totally_trust_me"
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        lake.read_manifest("ds_l16_tamper")


def test_legacy_manifest_without_checksum_accepted(tmp_path) -> None:
    lake = _lake(tmp_path)
    lake.ensure_ready()
    m = _manifest("ds_l16_legacy")
    ddir = lake.dataset_dir("ds_l16_legacy")
    ddir.mkdir(parents=True)
    payload = m.to_dict()
    payload["checksum"] = ""
    (ddir / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    back = lake.read_manifest("ds_l16_legacy")  # warning, not an error
    assert back.snapshot_id == "ds_l16_legacy"
