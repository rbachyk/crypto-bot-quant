"""Data lake / storage tests (AGENTS.md Appendix B.5, STORAGE gate)."""

from __future__ import annotations

import pytest
from src.storage import DataLake, DatasetManifest, new_snapshot_id


def _lake(tmp_path) -> DataLake:
    return DataLake(tmp_path / "lake", tmp_path / "artifacts")


def test_lake_writable(tmp_path) -> None:
    assert _lake(tmp_path).writable() is True


def test_snapshot_ids_are_unique() -> None:
    assert new_snapshot_id() != new_snapshot_id()


def test_create_snapshot_and_read_manifest(tmp_path) -> None:
    lake = _lake(tmp_path)
    lake.ensure_ready()
    sid = new_snapshot_id()
    manifest = DatasetManifest(
        snapshot_id=sid,
        created_at="t",
        symbols=["BTC/USDT:USDT"],
        data_types=["ohlcv"],
        row_counts={"ohlcv": 10},
    )
    lake.create_snapshot(manifest)
    readback = lake.read_manifest(sid)
    assert readback.snapshot_id == sid
    assert readback.symbols == ["BTC/USDT:USDT"]
    assert readback.checksum  # checksum populated on write
    assert sid in lake.list_snapshots()


def test_snapshots_are_immutable(tmp_path) -> None:
    lake = _lake(tmp_path)
    lake.ensure_ready()
    sid = new_snapshot_id()
    m = DatasetManifest(snapshot_id=sid, created_at="t")
    lake.create_snapshot(m)
    with pytest.raises(FileExistsError):
        lake.create_snapshot(m)


def test_write_artifact(tmp_path) -> None:
    lake = _lake(tmp_path)
    path = lake.write_artifact("models/m1/model.bin", b"weights")
    assert path.exists()
    assert path.read_bytes() == b"weights"
