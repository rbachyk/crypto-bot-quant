# 0002 — TimescaleDB is optional in Phase 1

Status: accepted · Date: 2026-06-16 · Phase: 1

Appendix C specifies "PostgreSQL 16 + TimescaleDB". The Phase 1 base schema is
purely operational/relational (jobs, gates, audit, approvals, metadata,
universe) — there are **no hypertables yet**; large time-series storage lives in
the data lake (Appendix B.4/B.5) and TimescaleDB hypertables/continuous
aggregates are introduced in Phase 2 when OHLCV/mark/funding ingestion lands.

Decision:
- The initial migration attempts `CREATE EXTENSION IF NOT EXISTS timescaledb`
  but **tolerates its absence** (a stock PostgreSQL build is acceptable for
  Phase 1 dev/test).
- `docker-compose.yml` uses the `timescale/timescaledb:2.15.3-pg16` image so the
  extension is available where the stack runs as intended.
- The DB gate asserts Postgres reachability, applied migrations, required tables,
  indexes and pooling — it does **not** require TimescaleDB in Phase 1.

This is the most capital-preserving reading: no functionality depends on
Timescale yet, and forcing the extension would block local development without
improving safety.
