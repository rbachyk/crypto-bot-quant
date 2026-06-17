# 0003 — Phase 2 data-platform choices

Status: accepted · Date: 2026-06-17 · Phase: 2

Where AGENTS.md leaves a detail unspecified, the most capital-preserving and
operationally-safe option is chosen and recorded here (AGENTS.md Conventions).

## Offline deterministic data source (no exchange access this phase)
Phase 1 ships only an offline `SkeletonExchangeAdapter` (no network). Phase 2
needs real ingestion, gap repair, validation and snapshots to be exercised and
gated, but live/exchange access is out of scope and forbidden by default
(Priority Stack 2; never enable live trading).

Decision: introduce a `DataSource` interface with an offline
`DeterministicSource` that fabricates reproducible, schema-valid market data
(OHLCV / mark / index / funding / open-interest / spread) as a **pure function
of timestamp**. Because `value(ts)` does not depend on the requested range, a
backfill of any sub-range is byte-identical to a full download — making
downloads/backfills idempotent (Section 0) and features reproducible (Phase 3
FEAT). The ccxt + native-SDK source is wired later behind the same interface
(Section 5/6); no strategy/feature/gate touches the venue directly.

## Parquet series store; data lake is regenerable, not committed
Per Appendix C/B.5, Parquet is the primary historical format. Series are stored
month-partitioned under the Appendix B.5 partition keys
(`exchange_id/data_type/symbol/timeframe/year/month`), append-only and
deduplicated by the grid timestamp, with stable content checksums.

The data lake (`var/`) is git-ignored: large history is owned by the platform
and is fully regenerable from the deterministic source via safe gap repair. The
relational **`dataset_versions`** row (Appendix B.4) plus the immutable manifest
are the index/provenance; backtests reference the dataset version, never loose
files (Appendix B.5). Per-run data-quality report dumps under `reports/data/`
are git-ignored like `reports/gates/`; the authoritative copy is the
`data_quality_reports` table.

## Fixed `as_of` coverage window (reproducible snapshots)
DATA-COV requires "data coverage meets configured minimums". The window is
anchored to a fixed `as_of` timestamp in `configs/data.yaml` (not wall-clock
`now`), so the same window always yields the same immutable snapshot id
(deterministic from window + data content). Re-running DATA-COV is therefore
idempotent: it reproduces (or reuses) the same snapshot rather than
accumulating new ones.

## DATA-COV performs safe gap repair (Appendix A "auto-remediation: partial")
The DATA-COV gate attempts safe gap repair from the data source before judging
coverage, matching its declared "Auto-remediation: Partial (safe gap repair
only)". Quarantine of genuinely unfillable symbols
(`insufficient_history` in `configs/data.yaml`) remains a manual decision and
excludes the symbol from the required universe rather than failing the gate
forever (DATA-COV remediation step 3). This keeps the gate self-sufficient on a
fresh checkout while never silently widening coverage.

## Clock-drift check is local; NTP enforced at deploy
The DQ gate verifies the wall clock advances consistently with the monotonic
clock (not frozen/jumping). Absolute NTP synchronisation cannot be checked
offline; it is enforced on the host at deploy time (chrony) and re-verified by
the Phase 13 MON gate. The DQ clock criterion documents this explicitly.

## Liquidation / order-book series are "if available" → not required
The offline source does not provide liquidation or order-book data; per Section
8 these are "if available". `download_liquidation_history` records
unavailability rather than failing, and neither is in `required_series`, so
DATA-COV does not demand them this phase.
