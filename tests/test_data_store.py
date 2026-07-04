"""Parquet series store: append-only, dedup, ordering, checksum (Appendix B.5)."""

from __future__ import annotations

import threading
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

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


def test_latest_ts(tmp_path) -> None:
    store = SeriesStore(tmp_path)
    assert store.latest_ts(KEY) is None  # empty
    # Span two months so the newest-partition shortcut is exercised.
    jan = 1_700_000_000_000 - (1_700_000_000_000 % 60_000)
    feb = jan + 40 * 86_400_000
    store.write(KEY, [_row(jan), _row(jan + 60_000), _row(feb), _row(feb + 60_000)])
    assert store.latest_ts(KEY) == feb + 60_000  # the most recent stored timestamp


def test_earliest_ts(tmp_path) -> None:
    store = SeriesStore(tmp_path)
    assert store.earliest_ts(KEY) is None  # empty
    # Span two months so the oldest-partition shortcut is exercised (mirror of latest_ts).
    jan = 1_700_000_000_000 - (1_700_000_000_000 % 60_000)
    feb = jan + 40 * 86_400_000
    store.write(KEY, [_row(jan), _row(jan + 60_000), _row(feb), _row(feb + 60_000)])
    assert store.earliest_ts(KEY) == jan  # where this series' history begins


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


# --------------------------------------------------------------------------- #
# M18: concurrent writers must not lose rows (per-month flock + unique tmps)   #
# --------------------------------------------------------------------------- #
def _writer_process(args: tuple[str, int, int]) -> int:
    """Top-level worker (picklable for spawn): write ``n`` rows starting at ``offset_ts``,
    interleaved on a 2×60s grid, in many small batches to maximise merge interleaving."""
    root, offset_ts, n = args
    store = SeriesStore(Path(root))
    for i in range(n):
        store.write(KEY, [_row(offset_ts + i * 120_000)])
    return n


def test_concurrent_processes_writing_disjoint_rows_both_survive(tmp_path) -> None:
    """Two PROCESSES writing disjoint rows to the SAME series-month: pre-fix, both read the
    same ``existing``, both rewrote the month file, and the last replace() silently dropped
    the other's rows. The per-month lock serializes the read-merge-replace, so every row
    from both writers must survive."""
    n = 30
    with ProcessPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(_writer_process, [(str(tmp_path), 0, n), (str(tmp_path), 60_000, n)])
        )
    assert results == [n, n]
    store = SeriesStore(tmp_path)
    ts = [r["ts"] for r in store.read(KEY)]
    assert ts == sorted(set(range(0, n * 120_000, 60_000)))  # all 2n rows, none lost


def test_concurrent_threads_writing_disjoint_rows_both_survive(tmp_path) -> None:
    """Same invariant across THREADS (flock acquisitions on separate file descriptions
    exclude each other within one process too)."""
    store = SeriesStore(tmp_path)
    n = 30
    errors: list[Exception] = []

    def work(offset_ts: int) -> None:
        try:
            for i in range(n):
                store.write(KEY, [_row(offset_ts + i * 120_000)])
        except Exception as exc:  # noqa: BLE001 - surface thread failures in the assertion
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(off,)) for off in (0, 60_000)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    ts = [r["ts"] for r in store.read(KEY)]
    assert ts == sorted(set(range(0, n * 120_000, 60_000)))


def test_write_leaves_no_temp_files(tmp_path) -> None:
    store = SeriesStore(tmp_path)
    store.write(KEY, [_row(0), _row(60_000)])
    store.write(KEY, [_row(120_000)])
    assert list(tmp_path.rglob("*.tmp")) == []


def test_month_partitioning(tmp_path) -> None:
    store = SeriesStore(tmp_path)
    jan = 1_767_225_600_000  # 2026-01-01T00:00:00Z
    feb = 1_769_904_000_000  # 2026-02-01T00:00:00Z
    store.write(KEY, [_row(jan), _row(feb)])
    sdir = tmp_path / "series" / "skeleton" / OHLCV / "BTC_USDT_USDT" / "1m"
    assert (sdir / "2026" / "01.parquet").exists()
    assert (sdir / "2026" / "02.parquet").exists()
    assert store.count(KEY) == 2
