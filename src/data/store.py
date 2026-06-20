"""Append-only, deduplicated, checksummed Parquet series store (Appendix B.5).

Each series is stored as month-partitioned Parquet files under the data lake,
keyed by the Appendix B.5 partition columns
(``exchange_id / data_type / symbol / timeframe / year / month``). Writes are:

* **append-only** — existing rows for a timestamp are never overwritten by a
  re-download (the source is a pure function of ts, so a re-fetch is identical
  anyway); this makes every download/backfill idempotent (Section 0);
* **deduplicated** — one row per grid timestamp (the primary key);
* **ordered** — rows are stored sorted by ``ts``;
* **checksummed** — a stable content checksum is derivable for any range, for
  the dataset manifest (Appendix B.5: row counts + checksum).

Parquet is the primary historical format (Appendix C). Large history never
lives only in Postgres (Appendix B.4).
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.data.schema import COLUMNS, FUNDING, SeriesKey

# Integer columns (everything else is float64); ts is the primary key.
_INT_COLUMNS = {"ts", "funding_interval_hours"}


def _arrow_schema(data_type: str) -> pa.Schema:
    fields = []
    for col in COLUMNS[data_type]:
        fields.append(pa.field(col, pa.int64() if col in _INT_COLUMNS else pa.float64()))
    return pa.schema(fields)


def _month_of(ts_ms: int) -> tuple[int, int]:
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=UTC)
    return dt.year, dt.month


class SeriesStore:
    """Filesystem-backed Parquet store for time series."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root) / "series"

    # -- paths ----------------------------------------------------------- #
    def _series_dir(self, key: SeriesKey) -> Path:
        return self.root / key.exchange_id / key.data_type / key.symbol_path() / key.timeframe

    def _month_file(self, key: SeriesKey, year: int, month: int) -> Path:
        return self._series_dir(key) / f"{year:04d}" / f"{month:02d}.parquet"

    # -- read ------------------------------------------------------------ #
    def _read_file(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        table = pq.read_table(path)
        return table.to_pylist()

    def read(
        self, key: SeriesKey, start_ms: int | None = None, end_ms: int | None = None
    ) -> list[dict]:
        """Return rows for ``key`` (optionally within ``[start_ms, end_ms)``), sorted by ts."""
        sdir = self._series_dir(key)
        if not sdir.exists():
            return []
        rows: list[dict] = []
        for year_dir in sorted(sdir.iterdir()):
            if not year_dir.is_dir():
                continue
            for mfile in sorted(year_dir.glob("*.parquet")):
                rows.extend(self._read_file(mfile))
        rows.sort(key=lambda r: r["ts"])
        if start_ms is not None:
            rows = [r for r in rows if r["ts"] >= start_ms]
        if end_ms is not None:
            rows = [r for r in rows if r["ts"] < end_ms]
        return rows

    def timestamps(self, key: SeriesKey, start_ms: int, end_ms: int) -> set[int]:
        return {r["ts"] for r in self.read(key, start_ms, end_ms)}

    def latest_ts(self, key: SeriesKey) -> int | None:
        """The most recent stored timestamp for ``key`` (``None`` if empty). Reads only the
        newest partition file, so it stays cheap for multi-year series (used by incremental
        download to resume from where the last download left off)."""
        sdir = self._series_dir(key)
        if not sdir.exists():
            return None
        for year_dir in sorted((d for d in sdir.iterdir() if d.is_dir()), reverse=True):
            for mfile in sorted(year_dir.glob("*.parquet"), reverse=True):
                rows = self._read_file(mfile)
                if rows:
                    return max(int(r["ts"]) for r in rows)
        return None

    def count(self, key: SeriesKey, start_ms: int | None = None, end_ms: int | None = None) -> int:
        return len(self.read(key, start_ms, end_ms))

    # -- write ----------------------------------------------------------- #
    def write(self, key: SeriesKey, rows: list[dict]) -> int:
        """Merge ``rows`` into the store (append-only, dedup by ts). Returns the
        number of genuinely new timestamps written."""
        if not rows:
            return 0
        cols = COLUMNS[key.data_type]
        new_written = 0
        by_month: dict[tuple[int, int], list[dict]] = {}
        for row in rows:
            by_month.setdefault(_month_of(row["ts"]), []).append(row)

        schema = _arrow_schema(key.data_type)
        for (year, month), month_rows in by_month.items():
            path = self._month_file(key, year, month)
            existing = {r["ts"]: r for r in self._read_file(path)}
            before = len(existing)
            for row in month_rows:
                if row["ts"] not in existing:  # append-only: keep first write
                    existing[row["ts"]] = {c: row[c] for c in cols}
            new_written += len(existing) - before
            merged = [existing[ts] for ts in sorted(existing)]
            self._write_file(path, schema, cols, merged, key.data_type)
        return new_written

    def _write_file(
        self, path: Path, schema: pa.Schema, cols: list[str], rows: list[dict], data_type: str
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        columns: dict[str, list] = {c: [] for c in cols}
        for row in rows:
            for c in cols:
                columns[c].append(row[c])
        arrays = [
            pa.array(columns[c], type=(pa.int64() if c in _INT_COLUMNS else pa.float64()))
            for c in cols
        ]
        table = pa.Table.from_arrays(arrays, schema=schema)
        # Atomic-ish replace: write to a temp file then rename.
        tmp = path.with_suffix(".parquet.tmp")
        pq.write_table(table, tmp)
        tmp.replace(path)

    def delete_range(self, key: SeriesKey, start_ms: int, end_ms: int) -> int:
        """Remove rows in ``[start_ms, end_ms)``. Used to re-download a bad range
        or to simulate a gap. Returns rows removed."""
        sdir = self._series_dir(key)
        if not sdir.exists():
            return 0
        removed = 0
        cols = COLUMNS[key.data_type]
        schema = _arrow_schema(key.data_type)
        for year_dir in sorted(sdir.iterdir()):
            if not year_dir.is_dir():
                continue
            for mfile in sorted(year_dir.glob("*.parquet")):
                rows = self._read_file(mfile)
                kept = [r for r in rows if not (start_ms <= r["ts"] < end_ms)]
                removed += len(rows) - len(kept)
                if len(kept) != len(rows):
                    if kept:
                        self._write_file(mfile, schema, cols, kept, key.data_type)
                    else:
                        mfile.unlink()
        return removed

    # -- integrity ------------------------------------------------------- #
    def checksum(
        self, key: SeriesKey, start_ms: int | None = None, end_ms: int | None = None
    ) -> str:
        """Stable sha256 (first 16 hex) over the canonical row content of a range."""
        rows = self.read(key, start_ms, end_ms)
        payload = json.dumps(rows, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def is_funding(key: SeriesKey) -> bool:
    return key.data_type == FUNDING
