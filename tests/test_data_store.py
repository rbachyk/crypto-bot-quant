"""Parquet series store: append-only, dedup, ordering, checksum (Appendix B.5)."""

from __future__ import annotations

from src.data.schema import OHLCV, SeriesKey
from src.data.store import SeriesStore

KEY = SeriesKey("skeleton", OHLCV, "BTC/USDT:USDT", "1m")


def _row(ts: int, close: float = 100.0) -> dict:
    return {
        "ts": ts,
        "open": close,
        "high": close + 1,
        "low": close - 1,
        "close": close,
        "volume": 5.0,
    }


def test_write_read_roundtrip(tmp_path) -> None:
    store = SeriesStore(tmp_path)
    rows = [_row(ts) for ts in (0, 60_000, 120_000)]
    assert store.write(KEY, rows) == 3
    back = store.read(KEY)
    assert [r["ts"] for r in back] == [0, 60_000, 120_000]
    assert back[0]["close"] == 100.0


def test_write_is_append_only_and_dedups(tmp_path) -> None:
    store = SeriesStore(tmp_path)
    store.write(KEY, [_row(0, close=100.0)])
    # Re-writing the same ts with a different value is ignored (append-only).
    assert store.write(KEY, [_row(0, close=999.0)]) == 0
    assert store.read(KEY)[0]["close"] == 100.0


def test_rows_are_stored_sorted(tmp_path) -> None:
    store = SeriesStore(tmp_path)
    store.write(KEY, [_row(120_000), _row(0), _row(60_000)])
    assert [r["ts"] for r in store.read(KEY)] == [0, 60_000, 120_000]


def test_read_range_is_half_open(tmp_path) -> None:
    store = SeriesStore(tmp_path)
    store.write(KEY, [_row(ts) for ts in (0, 60_000, 120_000, 180_000)])
    assert [r["ts"] for r in store.read(KEY, 60_000, 180_000)] == [60_000, 120_000]


def test_checksum_is_stable_and_content_sensitive(tmp_path) -> None:
    s1, s2 = SeriesStore(tmp_path / "a"), SeriesStore(tmp_path / "b")
    s1.write(KEY, [_row(0), _row(60_000)])
    s2.write(KEY, [_row(0), _row(60_000)])
    assert s1.checksum(KEY) == s2.checksum(KEY)
    s2.write(KEY, [_row(120_000)])
    assert s1.checksum(KEY) != s2.checksum(KEY)


def test_delete_range(tmp_path) -> None:
    store = SeriesStore(tmp_path)
    store.write(KEY, [_row(ts) for ts in (0, 60_000, 120_000, 180_000)])
    assert store.delete_range(KEY, 60_000, 180_000) == 2
    assert [r["ts"] for r in store.read(KEY)] == [0, 180_000]


def test_month_partitioning(tmp_path) -> None:
    store = SeriesStore(tmp_path)
    jan = 1_767_225_600_000  # 2026-01-01T00:00:00Z
    feb = 1_769_904_000_000  # 2026-02-01T00:00:00Z
    store.write(KEY, [_row(jan), _row(feb)])
    sdir = tmp_path / "series" / "skeleton" / OHLCV / "BTC_USDT_USDT" / "1m"
    assert (sdir / "2026" / "01.parquet").exists()
    assert (sdir / "2026" / "02.parquet").exists()
    assert store.count(KEY) == 2
