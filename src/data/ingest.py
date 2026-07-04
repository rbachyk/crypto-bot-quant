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

        A fetch doubles as the listing probe: when the exchange's first returned row sits WELL ABOVE
        ``start_ms`` (more than one interval), the exchange served nothing earlier, so that row IS
        the series' listing edge — persisted as the watermark so gap detection can tell genuine
        pre-listing absence from head-of-series data loss. When the first row is ≈ ``start_ms`` the
        fetch merely began mid-history (a narrow window), so recording ``start_ms`` as a "listing"
        would falsely mask older data the exchange still has — it is NOT recorded then.
        ``record_listing=False`` skips the probe entirely (a tail resume starts mid-series)."""
        rows = self.source.fetch(key, start_ms, end_ms)
        if rows and record_listing and int(rows[0]["ts"]) - start_ms > key.interval_ms:
            self.store.record_listing_ts(key, int(rows[0]["ts"]))
        return self.store.write(key, rows)

    def update_incremental(self, key: SeriesKey, start_ms: int, end_ms: int) -> int:
        """Bring a series up to date over ``[start_ms, end_ms)`` with minimal fetching; returns the
        count of new rows written.

        * FORWARD: fetch the tail past the newest stored bar (the usual incremental refresh); an
          empty store fetches the whole window in this pass.
        * BACKWARD: if the window START moved BELOW the earliest stored bar (the operator WIDENED
          the history, e.g. 1y → 3y), also fetch the missing older slice ``[start_ms, earliest)``.
          It stops once the store reaches the exchange's listing/retention floor (the recorded
          watermark), so a pre-listing range is never re-scanned every run."""
        written = 0
        last = self.store.latest_ts(key)
        earliest = self.store.earliest_ts(key)
        watermark = self.store.listing_ts(key)
        # BACKWARD backfill for a widened window. Skip when we're already sitting on the floor
        # (earliest at/below the recorded listing watermark) — nothing older exists to fetch.
        if (
            earliest is not None
            and earliest - start_ms >= key.interval_ms
            and (watermark is None or earliest - key.interval_ms > watermark)
        ):
            written += self.download(key, start_ms, earliest, record_listing=True)
        # FORWARD tail from the newest stored bar (or the whole window for an empty store).
        resume = (last + key.interval_ms) if (last is not None and last >= start_ms) else start_ms
        if resume < end_ms:
            written += self.download(key, resume, end_ms, record_listing=(resume == start_ms))
        return written

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
