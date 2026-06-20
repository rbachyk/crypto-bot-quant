"""Historical download, incremental update, and gap repair (Section 8).

The :class:`Ingestor` is the one component that moves data from a
:class:`~src.data.source.DataSource` into the :class:`~src.data.store.SeriesStore`.
All operations are idempotent (append-only dedup in the store): re-running a
download or a repair never duplicates or corrupts data.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.data.gaps import GapReport, find_gaps
from src.data.schema import SeriesKey
from src.data.source import DataSource
from src.data.store import SeriesStore


@dataclass(slots=True)
class IngestResult:
    key: SeriesKey
    rows_written: int
    gaps_before: int
    gaps_after: int

    @property
    def repaired(self) -> bool:
        return self.gaps_after == 0


class Ingestor:
    def __init__(self, source: DataSource, store: SeriesStore) -> None:
        self.source = source
        self.store = store

    def download(self, key: SeriesKey, start_ms: int, end_ms: int) -> int:
        """Full download of ``[start_ms, end_ms)`` for one series (idempotent)."""
        rows = self.source.fetch(key, start_ms, end_ms)
        return self.store.write(key, rows)

    def update_incremental(self, key: SeriesKey, start_ms: int, end_ms: int) -> int:
        """Fetch only the data that appeared since the last download — the tail past the last
        stored timestamp (an empty store fetches the whole window). Resumes from the newest
        stored ts without reading the whole multi-year series."""
        last = self.store.latest_ts(key)
        resume = (last + key.interval_ms) if (last is not None and last >= start_ms) else start_ms
        if resume >= end_ms:
            return 0
        return self.download(key, resume, end_ms)

    def repair(self, key: SeriesKey, start_ms: int, end_ms: int) -> IngestResult:
        """Detect gaps and fetch only the missing ranges (safe gap repair)."""
        before = find_gaps(self.store, key, start_ms, end_ms)
        written = 0
        for gap_start, gap_end in before.ranges():
            rows = self.source.fetch(key, gap_start, gap_end)
            written += self.store.write(key, rows)
        after = find_gaps(self.store, key, start_ms, end_ms)
        return IngestResult(
            key=key,
            rows_written=written,
            gaps_before=len(before.missing_ts),
            gaps_after=len(after.missing_ts),
        )

    def gap_report(self, key: SeriesKey, start_ms: int, end_ms: int) -> GapReport:
        return find_gaps(self.store, key, start_ms, end_ms)
