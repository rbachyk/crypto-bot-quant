"""Schema/grid math and the deterministic data source (Phase 2)."""

from __future__ import annotations

import pytest
from src.data.schema import (
    OHLCV,
    SeriesKey,
    expected_grid,
    ms_to_iso,
    parse_utc_ms,
    timeframe_ms,
)
from src.data.source import DeterministicSource


def test_parse_and_roundtrip_utc() -> None:
    ms = parse_utc_ms("2026-06-01T00:00:00Z")
    assert ms == 1780272000000
    assert ms_to_iso(ms) == "2026-06-01T00:00:00Z"


def test_expected_grid_is_regular_and_half_open() -> None:
    grid = expected_grid(0, 600_000, 60_000)
    assert grid == [
        0,
        60_000,
        120_000,
        180_000,
        240_000,
        300_000,
        360_000,
        420_000,
        480_000,
        540_000,
    ]
    assert 600_000 not in grid  # exclusive end


def test_expected_grid_snaps_unaligned_start_up() -> None:
    grid = expected_grid(30_000, 180_000, 60_000)
    assert grid == [60_000, 120_000]  # 30_000 snapped up to the next boundary


def test_timeframe_ms_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        timeframe_ms("3s")


def test_series_key_symbol_path_and_interval() -> None:
    key = SeriesKey("skeleton", OHLCV, "BTC/USDT:USDT", "5m")
    assert key.symbol_path() == "BTC_USDT_USDT"
    assert key.interval_ms == 300_000
    assert key.columns[0] == "ts"


def test_source_is_deterministic_and_range_independent() -> None:
    src = DeterministicSource()
    key = SeriesKey("skeleton", OHLCV, "ETH/USDT:USDT", "5m")
    full = src.fetch(key, 0, 3_000_000)
    sub = src.fetch(key, 600_000, 1_800_000)
    # A sub-range fetch is byte-identical to the same slice of a full fetch
    # (value is a pure function of ts) -> idempotent backfill / reproducible.
    by_ts = {r["ts"]: r for r in full}
    for row in sub:
        assert by_ts[row["ts"]] == row


def test_source_ohlcv_is_internally_consistent() -> None:
    src = DeterministicSource()
    key = SeriesKey("skeleton", OHLCV, "BTC/USDT:USDT", "1m")
    for r in src.fetch(key, 0, 3_600_000):
        assert r["low"] <= r["open"] <= r["high"]
        assert r["low"] <= r["close"] <= r["high"]
        assert r["low"] > 0
        assert r["volume"] >= 0


def test_source_missing_symbol_returns_nothing() -> None:
    src = DeterministicSource(missing_symbols={"DEAD/USDT:USDT"})
    key = SeriesKey("skeleton", OHLCV, "DEAD/USDT:USDT", "5m")
    assert src.has_symbol("DEAD/USDT:USDT") is False
    assert src.fetch(key, 0, 600_000) == []
