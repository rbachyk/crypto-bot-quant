"""Gap detection over the expected timestamp grid (Section 8 Historical Data
Manager: "detect gaps").

A gap is any expected grid timestamp absent from the store. Contiguous missing
timestamps are coalesced into ``[start, end)`` ranges so backfill and the
DATA-COV report can speak in ranges, not thousands of points.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.data.schema import SeriesKey, expected_grid, ms_to_iso
from src.data.store import SeriesStore


@dataclass(slots=True)
class GapReport:
    key: SeriesKey
    expected: int
    present: int
    missing_ts: list[int] = field(default_factory=list)
    duplicates: int = 0  # store dedups, so duplicates in storage are always 0

    @property
    def covered(self) -> bool:
        return not self.missing_ts

    def ranges(self) -> list[tuple[int, int]]:
        """Coalesce missing timestamps into contiguous ``[start, end)`` ranges."""
        if not self.missing_ts:
            return []
        iv = self.key.interval_ms
        ordered = sorted(self.missing_ts)
        ranges: list[tuple[int, int]] = []
        run_start = prev = ordered[0]
        for ts in ordered[1:]:
            if ts == prev + iv:
                prev = ts
                continue
            ranges.append((run_start, prev + iv))
            run_start = prev = ts
        ranges.append((run_start, prev + iv))
        return ranges

    def to_dict(self) -> dict:
        return {
            "series": self.key.label(),
            "data_type": self.key.data_type,
            "symbol": self.key.symbol,
            "timeframe": self.key.timeframe,
            "expected": self.expected,
            "present": self.present,
            "missing": len(self.missing_ts),
            "missing_ranges": [
                {"from": ms_to_iso(s), "to": ms_to_iso(e)} for s, e in self.ranges()
            ],
        }


def find_gaps(store: SeriesStore, key: SeriesKey, start_ms: int, end_ms: int) -> GapReport:
    grid = expected_grid(start_ms, end_ms, key.interval_ms)
    present = store.timestamps(key, start_ms, end_ms)
    missing = [ts for ts in grid if ts not in present]
    return GapReport(key=key, expected=len(grid), present=len(present), missing_ts=missing)
