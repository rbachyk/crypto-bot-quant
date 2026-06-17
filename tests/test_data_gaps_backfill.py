"""Gap detection, backfill repair, and incremental update (Section 8)."""

from __future__ import annotations

from src.data.gaps import find_gaps
from src.data.ingest import Ingestor
from src.data.schema import OHLCV, SeriesKey
from src.data.source import get_data_source

from tests._data_helpers import fresh_store, populate, small_cfg


def test_full_window_has_zero_gaps(tmp_path) -> None:
    cfg = small_cfg()
    store = fresh_store(tmp_path)
    populate(store, cfg)
    key = SeriesKey(cfg.exchange_id, OHLCV, cfg.symbols[0], "5m")
    report = find_gaps(store, key, cfg.window_start_ms, cfg.window_end_ms)
    assert report.covered
    assert report.expected == report.present


def test_gap_detection_and_range_coalescing(tmp_path) -> None:
    cfg = small_cfg()
    store = fresh_store(tmp_path)
    populate(store, cfg)
    key = SeriesKey(cfg.exchange_id, OHLCV, cfg.symbols[0], "5m")
    iv = key.interval_ms
    start = cfg.window_start_ms
    store.delete_range(key, start + iv, start + 4 * iv)  # 3 contiguous missing bars
    report = find_gaps(store, key, cfg.window_start_ms, cfg.window_end_ms)
    assert len(report.missing_ts) == 3
    assert report.ranges() == [(start + iv, start + 4 * iv)]


def test_repair_closes_gaps_idempotently(tmp_path) -> None:
    cfg = small_cfg()
    store = fresh_store(tmp_path)
    populate(store, cfg)
    key = SeriesKey(cfg.exchange_id, OHLCV, cfg.symbols[0], "5m")
    iv = key.interval_ms
    store.delete_range(key, cfg.window_start_ms + iv, cfg.window_start_ms + 4 * iv)

    ing = Ingestor(get_data_source(cfg.exchange_id), store)
    result = ing.repair(key, cfg.window_start_ms, cfg.window_end_ms)
    assert result.rows_written == 3
    assert result.repaired
    # Repaired values match the original source (pure function of ts).
    assert find_gaps(store, key, cfg.window_start_ms, cfg.window_end_ms).covered
    # Running repair again is a no-op.
    assert ing.repair(key, cfg.window_start_ms, cfg.window_end_ms).rows_written == 0


def test_incremental_update_fetches_only_the_tail(tmp_path) -> None:
    cfg = small_cfg()
    store = fresh_store(tmp_path)
    key = SeriesKey(cfg.exchange_id, OHLCV, cfg.symbols[0], "5m")
    half = cfg.window_start_ms + (cfg.window_end_ms - cfg.window_start_ms) // 2
    ing = Ingestor(get_data_source(cfg.exchange_id), store)
    ing.download(key, cfg.window_start_ms, half)
    before = store.count(key)
    added = ing.update_incremental(key, cfg.window_start_ms, cfg.window_end_ms)
    assert added > 0
    assert store.count(key) == before + added
    # No duplication, fully covered now.
    assert find_gaps(store, key, cfg.window_start_ms, cfg.window_end_ms).covered


def test_missing_symbol_cannot_be_filled(tmp_path) -> None:
    cfg = small_cfg(symbols=("BTC/USDT:USDT",))
    store = fresh_store(tmp_path)
    key = SeriesKey(cfg.exchange_id, OHLCV, "BTC/USDT:USDT", "5m")
    from src.data.source import DeterministicSource

    ing = Ingestor(DeterministicSource(missing_symbols={"BTC/USDT:USDT"}), store)
    result = ing.repair(key, cfg.window_start_ms, cfg.window_end_ms)
    assert result.rows_written == 0
    assert not result.repaired  # exchange genuinely lacks history
