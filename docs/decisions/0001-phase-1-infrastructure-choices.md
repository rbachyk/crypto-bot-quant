# 0001 — Phase 1 infrastructure choices

Status: accepted · Date: 2026-06-16 · Phase: 1

Where AGENTS.md leaves a detail unspecified, the most capital-preserving and
operationally-safe option is chosen and recorded here (AGENTS.md Conventions).

## Job queue: in-house Redis+Postgres dispatcher
Appendix C lists "Dramatiq or RQ". Phase 1 implements a thin Redis-list queue
with PostgreSQL job records (`src/jobs`). Rationale:
- Appendix B.6 requires durable job **records, logs, progress, retry, cancel**
  in Postgres — neither RQ nor Dramatiq provides that out of the box; we would
  wrap them anyway.
- The in-house dispatcher is fully deterministic and unit-testable without a
  separate worker daemon, which keeps the QUEUE gate honest and reproducible.
- The broker abstraction is small; swapping to Dramatiq/RQ later is localized to
  `src/jobs/queue.py` + `src/jobs/worker.py`. The job-record contract is stable.

## Single application image, command-per-service
All Python services share one image and differ only by `command`
(docker-compose). Simpler to build/ship; matches the MVP single-node topology
(Appendix B.12). The live engine is a distinct service kept behind the `live`
compose profile so it never starts by default.

## Health check semantics
`make health` runs the CLI health command, which probes DB, Redis, storage and
the kill switch directly. This works identically with or without Docker, so the
INFRA/MON gates are meaningful in local dev and in the container stack.

## Gate runner invocation
`make` cannot accept `--gate` (it is parsed as a make flag — this is exactly why
the orchestrator's first run failed). The canonical Gate Runner invocation is
therefore `python -m src.gates.runner --gate <id> --json`; `configs/gates.yaml`
`meta.rerun_command_pattern` was updated to match, and `make run-gate GATE=<id>`
is a convenience wrapper.
