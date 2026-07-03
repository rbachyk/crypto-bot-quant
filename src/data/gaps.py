"""Gap detection over the expected timestamp grid (Section 8 Historical Data
Manager: "detect gaps").

A gap is any expected grid timestamp absent from the store. Contiguous missing
timestamps are coalesced into ``[start, end)`` ranges so backfill and the
DATA-COV report can speak in ranges, not thousands of points.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog

from src.data.schema import FUNDING, SeriesKey, expected_grid, ms_to_iso
from src.data.store import SeriesStore

_log = structlog.get_logger("data.gaps")

# Series already warned (once per process) about a missing listing watermark — legacy series
# downloaded before watermarks existed; the next full download backfills the watermark.
_warned_no_watermark: set[str] = set()


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
        """Coalesce missing timestamps into contiguous ``[start, end)`` ranges.

        Coalescing uses the key's NOMINAL interval; funding settlements missing on a denser
        local cadence each yield their own (possibly overlapping) range — harmless, since
        repair fetches are idempotent (append-only dedup) and each range covers its point."""
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
    """Missing candles for ``key`` over ``[start_ms, end_ms)``.

    A contract listed AFTER the window start (e.g. SOL/ETH perps vs a 5-year BTC window) has no
    data before its listing — that leading absence is NOT a gap we failed to download. The
    listing point is the PERSISTED watermark the ingestor records on first download
    (:meth:`SeriesStore.listing_ts` — the earliest candle the exchange actually served), so the
    expected grid begins at ``max(start_ms, watermark)`` and stored data beginning ABOVE the
    watermark is reported as a leading gap like any other (head-of-series data loss — e.g. a
    ``delete_range`` at the head — stays detectable and repairable instead of being masked as
    "pre-listing"). A legacy series without a watermark falls back to trusting its first stored
    timestamp (the old behaviour; warned once — the next full download backfills the watermark).
    A series with no data at all still reports the full window missing (a genuine failure).

    Funding is an EVENT series (settlements on a per-symbol cadence the exchange adjusts), so it
    is covered by :func:`_funding_gaps` semantics — no fixed-grid assumption."""
    watermark = getattr(store, "listing_ts", lambda _key: None)(key)
    if key.data_type == FUNDING:
        return _funding_gaps(store, key, start_ms, end_ms, watermark)
    present = store.timestamps(key, start_ms, end_ms)
    grid_start = _grid_start(key, start_ms, watermark, min(present) if present else None)
    grid = expected_grid(grid_start, end_ms, key.interval_ms)
    missing = [ts for ts in grid if ts not in present]
    return GapReport(key=key, expected=len(grid), present=len(present), missing_ts=missing)


def _grid_start(key: SeriesKey, start_ms: int, watermark: int | None, first_stored: int | None):
    """Where the expected grid begins: the persisted listing watermark when known; otherwise
    (legacy series) the first stored timestamp — the pre-watermark behaviour, which cannot tell
    pre-listing absence from head data loss, hence the one-time warning."""
    if watermark is not None:
        return max(start_ms, watermark)
    if first_stored is None:
        return start_ms
    label = key.label()
    if label not in _warned_no_watermark:
        _warned_no_watermark.add(label)
        _log.warning(
            "series_missing_listing_watermark",
            series=label,
            detail="leading gaps are masked as pre-listing until a full download "
            "records the listing watermark",
        )
    return max(start_ms, first_stored)


def _funding_gaps(
    store: SeriesStore, key: SeriesKey, start_ms: int, end_ms: int, watermark: int | None
) -> GapReport:
    """Funding coverage: settlements are EVENTS on a per-symbol cadence the exchange adjusts
    dynamically (8h base; 4h/2h/1h on volatile contracts), so a fixed grid would flag denser
    settlements as spurious and silently pass sparser ones. Honest semantics: the series is
    covered iff no spacing exceeds the interval stamped on each row (the spacing OBSERVED at
    fetch time — ``funding_interval_hours``). Missing settlements are enumerated on that local
    cadence so backfill can fetch concrete ranges; ``expected`` = present + inferred missing."""
    rows = store.read(key, start_ms, end_ms)
    if not rows:
        # No data at all: report the full window missing on the nominal grid (a genuine
        # failure — or, with a watermark, everything from the recorded listing edge).
        grid_start = max(start_ms, watermark) if watermark is not None else start_ms
        grid = expected_grid(grid_start, end_ms, key.interval_ms)
        return GapReport(key=key, expected=len(grid), present=0, missing_ts=grid)
    grid_start = _grid_start(key, start_ms, watermark, int(rows[0]["ts"]))
    missing: list[int] = []
    # Leading: settlements between the listing watermark and the first stored one, on the
    # first row's cadence (head-of-series loss; approximate inside the hole, exact on refetch).
    lead_iv = _row_interval_ms(rows[0], key)
    ts = int(rows[0]["ts"]) - lead_iv
    while ts >= grid_start:
        missing.append(ts)
        ts -= lead_iv
    # Interior: each row's stamped interval is the observed spacing to its true predecessor,
    # so any stored spacing beyond it means settlements were lost in between.
    for prev, cur in zip(rows, rows[1:], strict=False):
        iv = _row_interval_ms(cur, key)
        ts = int(prev["ts"]) + iv
        while ts < int(cur["ts"]):
            missing.append(ts)
            ts += iv
    # Trailing: settlements after the last stored one, at the last known cadence.
    tail_iv = _row_interval_ms(rows[-1], key)
    ts = int(rows[-1]["ts"]) + tail_iv
    while ts < end_ms:
        missing.append(ts)
        ts += tail_iv
    missing.sort()
    return GapReport(
        key=key, expected=len(rows) + len(missing), present=len(rows), missing_ts=missing
    )


def _row_interval_ms(row: dict, key: SeriesKey) -> int:
    """A funding row's local cadence (stamped from observed settlement spacing at fetch time);
    the nominal ``key`` interval when unstamped/invalid (defensive: legacy rows always carry 8)."""
    hours = int(row.get("funding_interval_hours") or 0)
    return hours * 3_600_000 if hours > 0 else key.interval_ms
