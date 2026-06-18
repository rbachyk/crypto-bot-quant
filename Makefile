# Makefile / task runner (AGENTS.md Appendix B.16).
# All Python runs go through `uv` for reproducibility (Appendix C).

# Use the docker compose flavour that is available (v2 plugin or v1 binary).
COMPOSE := $(shell docker compose version >/dev/null 2>&1 && echo "docker compose" || echo "docker-compose")
UV := uv
RUN := $(UV) run

.PHONY: help setup test lint typecheck docker-up docker-down migrate seed-dev \
        health backup-db restore-test run-worker-data run-worker-backtest \
        run-worker-ml run-worker-rl run-worker-reports run-paper run-gate \
        run-all-gates kill kill-status

help:
	@echo "Targets: setup test lint typecheck docker-up docker-down migrate seed-dev"
	@echo "         health backup-db restore-test run-worker-* run-paper kill"
	@echo "         run-gate GATE=<id>   run-all-gates"

setup:
	$(UV) sync
	@echo "Setup complete. Copy .env.example to .env and edit secrets."

# --- Quality gates ----------------------------------------------------------
test:
	$(RUN) pytest -q

lint:
	$(RUN) ruff check src tests
	$(RUN) ruff format --check src tests

typecheck:
	$(RUN) mypy src

# --- Docker -----------------------------------------------------------------
docker-up:
	$(COMPOSE) up -d --build
	@echo "Stack starting. trading-engine-live is NOT started (profile 'live')."

docker-down:
	$(COMPOSE) down

# --- Database ---------------------------------------------------------------
migrate:
	$(RUN) alembic upgrade head

seed-dev:
	$(RUN) python -m src.cli.main enqueue sync_exchange_metadata
	$(RUN) python -m src.cli.main enqueue build_symbol_universe

# --- Health -----------------------------------------------------------------
health:
	$(RUN) python -m src.cli.main health

# --- Backup / restore -------------------------------------------------------
backup-db:
	bash scripts/backup_db.sh

restore-test:
	bash scripts/restore_test.sh

config-freeze:
	$(RUN) python -m src.cli.main config-freeze

# --- Workers / engines (dedicated processes; B.13) -------------------------
run-worker-data:
	SERVICE_NAME=worker-data $(RUN) python -m src.cli.main worker

run-worker-backtest:
	SERVICE_NAME=worker-backtest $(RUN) python -m src.cli.main worker

run-worker-ml:
	SERVICE_NAME=worker-ml $(RUN) python -m src.cli.main worker

run-worker-rl:
	SERVICE_NAME=worker-rl $(RUN) python -m src.cli.main worker

run-worker-reports:
	SERVICE_NAME=worker-reports $(RUN) python -m src.cli.main worker

run-paper:
	SERVICE_NAME=trading-engine-paper TRADING_MODE=PAPER $(RUN) python -m src.cli.main worker

# --- Gates ------------------------------------------------------------------
# Gate-runner CLI contract (phase prompt): GATE/FORMAT are make VARIABLES,
# never flags. `make -s run-gate GATE=INFRA FORMAT=json` prints a single
# GateResult JSON object to stdout. The `@` prefix keeps recipe lines off
# stdout even when `-s` is not passed; runner logs go to stderr.
FORMAT ?= text
run-gate:
	@$(RUN) python -m src.gates.runner --gate $(GATE) $(if $(filter json,$(FORMAT)),--json)

run-all-gates:
	@$(RUN) python -m src.gates.runner --all $(if $(filter json,$(FORMAT)),--json)

# --- Kill switch (independent of dashboard; Section 2.2) -------------------
kill:
	$(RUN) python -m src.cli.main kill

kill-status:
	$(RUN) python -m src.cli.main kill-status
