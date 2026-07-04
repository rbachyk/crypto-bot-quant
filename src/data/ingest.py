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

    def download(
        self, key: SeriesKey, start_ms: int, end_ms: int, *, record_listing: bool = True
    ) -> int:
        """Full download of ``[start_ms, end_ms)`` for one series (idempotent).

        A fetch that starts at the window start doubles as the listing probe: the first row the
        exchange returns IS the series' listing edge (the earliest available candle at/after
        ``start_ms``), persisted as the listing watermark so gap detection can tell genuine
        pre-listing absence from head-of-series data loss. ``record_listing=False`` skips that
        for fetches that do NOT begin at the window start (the incremental tail resume)."""
        rows = self.source.fetch(key, start_ms, end_ms)
        if rows and record_listing:
            self.store.record_listing_ts(key, rows[0]["ts"])
        return self.store.write(key, rows)

    def update_incremental(self, key: SeriesKey, start_ms: int, end_ms: int) -> int:
        """Fetch only the data that appeared since the last download — the tail past the last
        stored timestamp (an empty store fetches the whole window). Resumes from the newest
        stored ts without reading the whole multi-year series. Only a from-the-start fetch may
        record the listing watermark — a tail resume starts mid-series by construction."""
        last = self.store.latest_ts(key)
        # Backfill the listing watermark for a series that HAS data but never recorded one (legacy
        # series, or one only ever built incrementally) WITHOUT a full re-download: the earliest
        # stored bar is the listing edge whenever the first download began at/before the window
        # start (the norm — the window starts years before most listings). record_listing_ts is
        # monotone-min, so a later full download that recovers earlier data still corrects it. This
        # is data-integrity hygiene (once recorded, future head-of-series loss stays detectable and
        # the panel shows a real listing date); it does NOT change gap counts — grid_start =
        # max(start, earliest) already equals the prior no-watermark first-stored fallback.
        if last is not None and self.store.listing_ts(key) is None:
            earliest = self.store.earliest_ts(key)
            if earliest is not None:
                self.store.record_listing_ts(key, earliest)
        resume = (last + key.interval_ms) if (last is not None and last >= start_ms) else start_ms
        if resume >= end_ms:
            return 0
        return self.download(key, resume, end_ms, record_listing=(resume == start_ms))

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
