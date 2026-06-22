"""Coverage computation (DATA-COV) and immutable dataset snapshots (B.5)."""

from __future__ import annotations

from src.data.coverage import compute_coverage
from src.data.schema import OHLCV, SeriesKey
from src.data.snapshot import build_dataset_version
from src.storage import DataLake

from tests._data_helpers import fresh_store, populate, small_cfg


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
