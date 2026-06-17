"""``scripts/backfill`` CLI — detect and repair missing data (DATA-COV
remediation step 2: ``scripts/backfill --symbol <s> --series <x> --from <t>``).

Idempotent: fetches only missing grid timestamps from the data source and
writes them into the Parquet store (append-only dedup). With no ``--symbol`` it
repairs the whole required universe over the configured coverage window.
"""

from __future__ import annotations

import argparse
import sys

from src.data.config import load_data_config
from src.data.platform import DataPlatform
from src.data.schema import SeriesKey, parse_utc_ms
from src.observability import configure_logging


def _keys_for(cfg, symbol: str | None, series: str | None) -> list[SeriesKey]:
    symbols = [symbol] if symbol else cfg.active_symbols()
    keys: list[SeriesKey] = []
    for sym in symbols:
        for key in cfg.required_keys(sym):
            if series and key.data_type != series:
                continue
            keys.append(key)
    return keys


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description="Backfill missing market data")
    parser.add_argument("--symbol", help="symbol to repair (default: all active)")
    parser.add_argument("--series", help="data_type to repair, e.g. ohlcv (default: all required)")
    parser.add_argument("--from", dest="from_ts", help="ISO-8601 UTC start (default: window start)")
    parser.add_argument("--snapshot", action="store_true", help="re-snapshot the dataset when done")
    args = parser.parse_args(argv)

    cfg = load_data_config()
    platform = DataPlatform(cfg=cfg)
    start = parse_utc_ms(args.from_ts) if args.from_ts else cfg.window_start_ms
    end = cfg.window_end_ms

    keys = _keys_for(cfg, args.symbol, args.series)
    total_written = 0
    still_missing = 0
    for key in keys:
        result = platform.ingestor.repair(key, start, end)
        total_written += result.rows_written
        still_missing += result.gaps_after
        flag = "ok" if result.repaired else f"MISSING={result.gaps_after}"
        print(f"{key.label():40s} +{result.rows_written:5d} rows  {flag}")

    print(
        f"\nbackfill: wrote {total_written} rows across {len(keys)} series; "
        f"{still_missing} timestamps still missing"
    )
    if args.snapshot:
        run = platform.run_full(repair=False)
        print(f"snapshot: {run.snapshot.snapshot_id} (covered={run.coverage.covered})")

    # Non-zero exit if anything remains unfillable (e.g. insufficient history).
    return 1 if still_missing else 0


if __name__ == "__main__":
    sys.exit(main())
