# Phase Prompt — Phase 1: Infrastructure Foundation
<!-- Worked example for Appendix E.4. The orchestrator generates phases 2-13 from the same
     template + Roadmap (Section 32) + Acceptance Criteria (Appendix D) + gates.yaml. -->

## SYSTEM
You are the **Coding Agent**. Obey `AGENTS.md` exactly; it is the single source of truth.
The **Priority Stack (Section 1)** resolves every conflict (Capital protection > Exchange safety > Data correctness > Operational reliability > … > Speed).
Where a detail is unspecified, choose the most capital-preserving option and record it in `docs/decisions/`.
Capital-agnostic: no absolute-currency assumptions. Live trading must be impossible to enable in this phase.

## CONTEXT (attach)
- Full `AGENTS.md` (all sections + Appendices A–E).
- `configs/gates.yaml`.
- Tech stack: **Appendix C** (Python 3.12+, uv, Postgres16+Timescale, Redis7, Dramatiq/RQ, FastAPI, Caddy, ruff, mypy, pytest, Docker Compose).

## TASK
Implement **Phase 1 — Infrastructure Foundation**.

## DELIVERABLES (Roadmap §32, Appendix B)
- `docker-compose.yml` with: postgres+timescale, redis, backend, dashboard-shell, worker, caddy. Internal network; only 443 exposed.
- `.env.example` with safe placeholders (no real keys); config system with pydantic-settings env validation.
- PostgreSQL schema + Alembic migrations for: `jobs`, `job_logs`, `gates`, `gate_results`, `remediation_actions`, `approvals`, `audit_log`.
- Job orchestration skeleton (enqueue, progress, retry, cancel) with persisted job records + logs.
- Health-check endpoints for every service; dashboard shell **with authentication**.
- Exchange-adapter skeleton + `sync_exchange_metadata` job stub; universe-builder skeleton.
- CLI `make kill` (independent of dashboard); `Makefile` targets: `setup, test, lint, typecheck, docker-up, docker-down, migrate, seed-dev, health, backup-db, restore-test, run-worker-*, run-gate, run-all-gates, kill`.
- Gate scaffolding wired to `configs/gates.yaml`: `INFRA`, `DB`, `QUEUE`, `STORAGE`, plus skeletons for `MON`, `BACKUP`.
- **Gate-runner CLI contract (critical):** `make -s run-gate GATE=<id> FORMAT=json` must run the gate and print a single `GateResult` JSON object to **stdout** (fields: `gate_id`, `overall` ∈ PASS|FAIL|BLOCKED|NOT_RUN, `criteria[]`, `report_path`). Use `@`/`-s` so `make` does not echo recipe lines. `GATE`/`FORMAT` are **make variables**, never flags. Also provide `make -s run-all-gates FORMAT=json` (dependency order per `configs/gates.yaml`).

## ACCEPTANCE CRITERIA (Appendix D, Phase 1 + Global DoD)
- [ ] `make docker-up && make health` → all services green.
- [ ] `make migrate` idempotent; required tables exist.
- [ ] Job skeleton: enqueue/progress/retry/cancel work; records + logs persisted.
- [ ] Dashboard shell requires auth; per-service health endpoints respond.
- [ ] `make kill` works with the dashboard stopped.
- [ ] Gates `INFRA, DB, QUEUE, STORAGE` return `PASS`; `MON, BACKUP` skeletons present.
- [ ] `pytest`, `ruff`, `mypy` green in CI; no secrets committed; no live-mode default.
- [ ] Completion Report (Appendix E.2) filled; assumptions in `docs/decisions/`.

## GATES TO PASS (run via `make run-gate --gate <id>`)
`INFRA`, `DB`, `QUEUE`, `STORAGE`. (`MON`, `BACKUP` skeletons only this phase.)

## CONSTRAINTS
- Forbidden Work (Section 30) + Infrastructure Forbidden Shortcuts (Appendix B.17): no monolith script; no live execution in the dashboard process; no heavy jobs in API handlers; no untracked-file datasets; no unversioned artifacts; no live default; no real keys in `.env.example`; no unauthenticated dashboard outside local.

## OUTPUT
- Complete files at correct paths + the git commands for the commit.
- Then a **Completion Report** (Appendix E.2).
- End with: **Status** (phase/module) · **Tested?** · **Next step** (one line).

## STOP CONDITION
All four gates `PASS` and every acceptance box checked. **Do not begin Phase 2.** Hand off to the Reviewer Agent.
