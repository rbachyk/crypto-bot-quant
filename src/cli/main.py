"""``qbot`` CLI.

Critically, the kill switch is reachable here **without** the dashboard or any
web process (AGENTS.md Section 2.2): ``qbot kill`` (wired to ``make kill``)
engages it directly via :class:`src.killswitch.KillSwitch`.
"""

from __future__ import annotations

import json

import typer

from src.config import get_settings
from src.killswitch import KillSwitch
from src.observability import configure_logging

app = typer.Typer(add_completion=False, help="Quant trading bot control CLI")


@app.command()
def kill(reason: str = typer.Option("manual cli kill", help="why the switch was engaged")) -> None:
    """Engage the manual kill switch (independent of the dashboard)."""
    KillSwitch().engage(reason=reason, actor="cli")
    typer.echo("KILL SWITCH ENGAGED — new entries halted; manual reset required.")


@app.command("kill-status")
def kill_status() -> None:
    """Show kill-switch status."""
    typer.echo(json.dumps(KillSwitch().status(), indent=2))


@app.command()
def release(
    confirm: bool = typer.Option(False, "--confirm", help="required to clear the kill switch"),
) -> None:
    """Clear the kill switch (manual reset only)."""
    if not confirm:
        typer.echo("Refusing to release without --confirm.")
        raise typer.Exit(code=1)
    KillSwitch().disengage(actor="cli")
    typer.echo("Kill switch cleared.")


@app.command()
def health() -> None:
    """Print a health report for this node's dependencies."""
    from src.monitoring import check_health

    report = check_health()
    typer.echo(json.dumps(report.to_dict(), indent=2))
    raise typer.Exit(code=0 if report.healthy else 1)


@app.command()
def gate(
    gate_id: str = typer.Argument(..., help="gate id, e.g. INFRA"),
    as_json: bool = typer.Option(True, "--json/--no-json"),
) -> None:
    """Run a single gate and print its result."""
    from src.gates import GateRunner

    result = GateRunner().run(gate_id)
    if as_json:
        typer.echo(json.dumps(result.to_dict(), indent=2))
    else:
        typer.echo(f"{result.gate_id}: {result.overall}")
    raise typer.Exit(code=0 if result.overall == "PASS" else 1)


@app.command()
def worker(
    max_jobs: int = typer.Option(0, help="0 = run forever"),
    queues: str = typer.Option(
        "", "--queues", help="comma-separated queue classes to serve ('' = all / $WORKER_QUEUES)"
    ),
) -> None:
    """Run a job worker process."""
    configure_logging()
    from src.jobs import Worker
    from src.jobs.handlers import ensure_handlers_registered

    ensure_handlers_registered()
    worker = Worker(queues=queues or None)
    typer.echo(f"worker started (queues={worker.queues})")
    worker.run(max_jobs=max_jobs or None)


@app.command()
def enqueue(job_type: str, params_json: str = typer.Option("{}", help="JSON params")) -> None:
    """Enqueue a job by type."""
    from src.jobs import JobQueue
    from src.jobs.handlers import ensure_handlers_registered

    ensure_handlers_registered()
    job_id = JobQueue().enqueue(job_type, json.loads(params_json), requested_by="cli")
    typer.echo(job_id)


@app.command()
def download(
    config_path: str = typer.Option(
        "", "--config", help="data config yaml (default: configs/data.yaml; use data.bybit.yaml)"
    ),
    symbols: str = typer.Option(
        "", "--symbols", help="comma-separated ccxt symbols ('' = config symbols)"
    ),
    exchange: str = typer.Option("", "--exchange", help="ccxt exchange id ('' = config exchange)"),
    timeframes: str = typer.Option(
        "", "--timeframes", help="comma-separated OHLCV timeframes ('' = config default)"
    ),
    days: int = typer.Option(
        0, "--days", help="window length in days back from --as-of (0 = config window length)"
    ),
    as_of: str = typer.Option(
        "", "--as-of", help="ISO8601 UTC window end, e.g. 2026-06-01T00:00:00Z ('' = config)"
    ),
    data_version: str = typer.Option(
        "", "--data-version", help="DATA_VERSION label ('' = config default)"
    ),
    repair: bool = typer.Option(True, "--repair/--no-repair", help="auto-fill safe gaps"),
) -> None:
    """Download real public market data into a versioned DATA_VERSION snapshot.

    Runs the standard coverage→validate→snapshot pipeline over a real exchange so
    backtests/paper/shadow can iterate on real data without losing prior versions
    (each run is an immutable DATA_VERSION snapshot). Pass ``--config
    configs/data.bybit.yaml`` for the real Bybit contract (OI sampled at 1h); the
    default ``configs/data.yaml`` stays on the offline skeleton. CLI flags override
    individual fields of the chosen config.
    """
    from dataclasses import replace

    from src.data.config import load_data_config
    from src.data.platform import DataPlatform
    from src.data.schema import parse_utc_ms

    base = load_data_config(config_path or None)
    end_ms = parse_utc_ms(as_of) if as_of else base.window_end_ms
    if days > 0:
        start_ms = end_ms - days * 86_400_000
    elif as_of:
        start_ms = end_ms - (base.window_end_ms - base.window_start_ms)
    else:
        start_ms = base.window_start_ms
    cfg = replace(
        base,
        exchange_id=exchange or base.exchange_id,
        symbols=([s.strip() for s in symbols.split(",") if s.strip()] if symbols else base.symbols),
        timeframes=(
            [t.strip() for t in timeframes.split(",") if t.strip()]
            if timeframes
            else base.timeframes
        ),
        data_version=data_version or base.data_version,
        window_start_ms=start_ms,
        window_end_ms=end_ms,
    )

    platform = DataPlatform(cfg=cfg)
    written = platform.download_all()
    run = platform.run_full(repair=repair, source_jobs=["cli.download"])
    typer.echo(
        json.dumps(
            {
                "exchange_id": cfg.exchange_id,
                "data_version": cfg.data_version,
                "symbols": cfg.symbols,
                "timeframes": cfg.timeframes,
                "window": [cfg.window_start_ms, cfg.window_end_ms],
                "rows_written": written,
                "snapshot_id": run.snapshot.snapshot_id,
                "coverage_ok": run.coverage.covered,
                "validation_passed": run.validation.passed,
                "report_path": run.report_path,
            },
            indent=2,
        )
    )
    raise typer.Exit(code=0 if (run.coverage.covered and run.validation.passed) else 1)


@app.command(name="backtest-lake")
def backtest_lake(
    config_path: str = typer.Option(
        "configs/data.bybit.yaml", "--config", help="data config yaml (real-data snapshot)"
    ),
    symbols: str = typer.Option("", "--symbols", help="comma-separated symbols ('' = config)"),
    timeframe: str = typer.Option(
        "", "--timeframe", help="decision timeframe ('' = config base_timeframe)"
    ),
    strategy: str = typer.Option(
        "", "--strategy", help="research candidate id (e.g. basis_reversion); '' = reference"
    ),
    label: str = typer.Option("lake", "--label", help="run label"),
    dataset_version: str = typer.Option(
        "", "--dataset-version", help="snapshot id to tag the run ('' = config data_version)"
    ),
) -> None:
    """Run + persist ONE real-data backtest iteration (ranked on the leaderboard).

    Requires a downloaded snapshot for the config window (``qbot download --config ...``).
    Each iteration is an immutable ``backtest_runs`` row tagged with its DATA_VERSION.
    Pass ``--strategy <candidate_id>`` to backtest a real strategy (families A/B/G).
    """
    from src.backtest.service import run_and_persist_lake_backtest
    from src.data.config import load_data_config

    data_cfg = load_data_config(config_path or None)
    syms = [s.strip() for s in symbols.split(",") if s.strip()] or None
    rid, out = run_and_persist_lake_backtest(
        data_cfg,
        timeframe=timeframe or None,
        symbols=syms,
        candidate_id=strategy or None,
        dataset_version=dataset_version or None,
        label=label,
    )
    r = out.report
    typer.echo(
        json.dumps(
            {
                "run_id": rid,
                "dataset_version": dataset_version or data_cfg.data_version,
                "trades": r.trade_count,
                "expectancy_r": r.expectancy_r,
                "profit_factor": min(r.profit_factor, 1e9),
                "total_return": r.total_return,
                "max_drawdown": r.max_drawdown,
            },
            indent=2,
        )
    )


@app.command(name="paper-lake")
def paper_lake(
    config_path: str = typer.Option(
        "configs/data.bybit.yaml", "--config", help="data config yaml (real-data snapshot)"
    ),
    symbols: str = typer.Option("", "--symbols", help="comma-separated symbols ('' = config)"),
    timeframe: str = typer.Option("", "--timeframe", help="decision timeframe ('' = base)"),
    strategy: str = typer.Option(
        "", "--strategy", help="research candidate id (per-row family B); '' = reference"
    ),
    dataset_version: str = typer.Option("", "--dataset-version", help="snapshot id tag"),
) -> None:
    """Run + persist a REAL-DATA (replay) paper session over a snapshot (shadow-only).

    Derives candidates from real lake data and runs the full paper pipeline; trades land
    in ``paper_trades`` and show on the Paper dashboard. Requires a downloaded snapshot.
    """
    from src.data.config import load_data_config
    from src.paper.lake import run_lake_paper_session

    data_cfg = load_data_config(config_path or None)
    syms = [s.strip() for s in symbols.split(",") if s.strip()] or None
    session, _report, sid = run_lake_paper_session(
        data_cfg,
        timeframe=timeframe or None,
        symbols=syms,
        candidate_id=strategy or None,
        dataset_version=dataset_version or None,
    )
    net = sum(t.pnl for t in session.trades)
    typer.echo(
        json.dumps(
            {
                "session_id": sid,
                "executed": session.executed_count,
                "rejected": session.rejected_count,
                "net_pnl": round(net, 2),
            },
            indent=2,
        )
    )


@app.command(name="ml-shadow-lake")
def ml_shadow_lake(
    config_path: str = typer.Option("configs/data.bybit.yaml", "--config"),
    symbols: str = typer.Option("", "--symbols"),
    timeframe: str = typer.Option("", "--timeframe"),
    strategy: str = typer.Option("", "--strategy", help="research candidate id ('' = reference)"),
) -> None:
    """Score REAL lake candidates with the shadow ML meta-labeler (applied=False)."""
    from src.jobs import JobQueue
    from src.jobs.handlers import ensure_handlers_registered

    ensure_handlers_registered()
    params = {
        "config_path": config_path,
        "symbols": [s.strip() for s in symbols.split(",") if s.strip()] or None,
        "timeframe": timeframe or None,
        "candidate_id": strategy or None,
    }
    job_id = JobQueue().enqueue("run_lake_ml_shadow_pass", params, requested_by="cli")
    typer.echo(job_id)


@app.command(name="promote-lake")
def promote_lake(
    config_path: str = typer.Option(
        "configs/data.bybit.yaml", "--config", help="data config yaml (real-data snapshot)"
    ),
    timeframe: str = typer.Option("", "--timeframe", help="decision timeframe ('' = base)"),
) -> None:
    """Validate every enabled candidate on REAL downloaded lake data and persist promote/shelve
    verdicts tagged ``data_source='lake'`` — the strategies that demo/testnet/live are allowed to
    run (Section 13). Runs INLINE (no worker needed). Requires a downloaded snapshot first
    (``qbot download``). Exits non-zero if nothing promotes."""
    from src.data.config import load_data_config
    from src.strategies.lake_research import validate_all_on_lake
    from src.strategies.promotion import persist_validations

    data_cfg = load_data_config(config_path or None)
    validations = validate_all_on_lake(data_cfg, timeframe=timeframe or None)
    written = persist_validations(validations, data_source="lake")
    promoted = [v.candidate_id for v in validations if v.promoted]
    shelved = {
        v.candidate_id: ("; ".join(v.shelved_reasons) or "did not clear gates")
        for v in validations
        if not v.promoted
    }
    typer.echo(
        json.dumps(
            {"validated": written, "promoted": promoted, "shelved": shelved}, indent=2
        )
    )
    if not promoted:
        typer.echo(
            "no candidate promoted on real data — extend the window or retune thresholds "
            "(do NOT loosen the gates)."
        )
        raise typer.Exit(code=1)


@app.command()
def leaderboard(
    kind: str = typer.Option("backtest", "--kind", help="run kind ('all' = every kind)"),
    dataset_version: str = typer.Option("", "--dataset-version", help="filter by snapshot"),
    strategy: str = typer.Option("", "--strategy", help="filter by strategy id"),
    limit: int = typer.Option(20, "--limit"),
    all_iterations: bool = typer.Option(
        False, "--all", help="show every run, not just best-per-iteration"
    ),
) -> None:
    """Print the backtest iteration leaderboard (ranked by the profitability bar)."""
    from src.backtest.leaderboard import build_leaderboard

    entries = build_leaderboard(
        kind=None if kind == "all" else kind,
        dataset_version=dataset_version or None,
        strategy_id=strategy or None,
        limit=limit,
        best_per_iteration=not all_iterations,
    )
    typer.echo(json.dumps([e.to_dict() for e in entries], indent=2, default=str))


@app.command()
def live(
    config_path: str = typer.Option("configs/data.bybit.yaml", "--config"),
    mode: str = typer.Option("paper", "--mode", help="paper | testnet | live"),
    symbols: str = typer.Option("", "--symbols"),
    timeframe: str = typer.Option("", "--timeframe"),
    strategy: str = typer.Option("", "--strategy", help="research candidate id ('' = reference)"),
    transport: str = typer.Option(
        "", "--transport", help="live data feed: ws | rest ('' = none/pure replay)"
    ),
    realtime: bool = typer.Option(
        False, "--realtime", help="drive candidates from the live feed (needs --transport)"
    ),
    multi_strategy: bool = typer.Option(
        False, "--multi-strategy", help="run the active PROMOTED ensemble (not a single --strategy)"
    ),
    poll_sec: float = typer.Option(
        0.0, "--poll-sec", help="real-time poll interval; >0 = CONTINUOUS session (waits for bars)"
    ),
    max_ticks: int = typer.Option(0, "--max-ticks", help="0 = process the whole snapshot"),
) -> None:
    """Run the live trading loop over a snapshot (replay), shadow/paper by default.

    ``--mode paper`` uses the offline SimulatedVenue; ``testnet`` places real orders on
    the exchange sandbox (no real funds); ``live`` is real-money and is refused unless
    the full live-safety condition holds (gates green + sign-off) — orders are gracefully
    rejected otherwise. Requires a downloaded snapshot (``qbot download``).
    """
    from src.data.config import load_data_config
    from src.live.loop import run_replay_session

    data_cfg = load_data_config(config_path or None)
    syms = [s.strip() for s in symbols.split(",") if s.strip()] or None
    result = run_replay_session(
        data_cfg,
        mode=mode,
        timeframe=timeframe or None,
        symbols=syms,
        candidate_id=strategy or None,
        multi_strategy=multi_strategy,
        transport=transport or None,
        realtime=realtime,
        poll_sec=poll_sec,
        max_ticks=max_ticks or None,
    )
    typer.echo(
        json.dumps(
            {
                "mode": result.mode,
                "session_id": result.session.session_id,
                "ticks": len(result.ticks),
                "executed": result.executed,
                "rejected": result.rejected,
                "halted": result.halted,
            },
            indent=2,
        )
    )


@app.command(name="paper-basket")
def paper_basket(
    strategy: str = typer.Option(..., "--strategy", help="cross-sectional candidate id"),
    config_path: str = typer.Option("configs/data.bybit.yaml", "--config"),
    timeframe: str = typer.Option("", "--timeframe"),
    poll_sec: float = typer.Option(60.0, "--poll-sec", help=">0 = continuous (waits for new bars)"),
    max_ticks: int = typer.Option(0, "--max-ticks", help="0 = run until stopped"),
) -> None:
    """Continuous PAPER session for one cross-sectional (basket) strategy on the live REST feed.

    The basket path for carry/factor strategies (funding_carry): holds a dollar-neutral basket,
    rebalanced on the live feed, booking SIMULATED fills (no real orders/funds). The per-symbol
    `qbot live` path is for directional strategies; this is for basket ones.
    """
    from src.data.config import load_data_config
    from src.live.basket import run_basket_paper_session

    data_cfg = load_data_config(config_path or None)
    n = run_basket_paper_session(
        strategy, data_cfg=data_cfg, timeframe=timeframe or None,
        poll_sec=poll_sec, max_ticks=max_ticks or None,
    )
    typer.echo(json.dumps({"strategy": strategy, "paper_trades": n}, indent=2))


@app.command()
def reports(
    name: str = typer.Option("", "--name", help="one report name ('' = all standard reports)"),
) -> None:
    """Generate the Section-34 named reports (live / online_learning / rl / live_readiness / …)."""
    from src.reports import generate_report, generate_standard_reports

    if name:
        typer.echo(generate_report(name))
    else:
        typer.echo(json.dumps(generate_standard_reports(), indent=2))


@app.command(name="demo-readiness")
def demo_readiness() -> None:
    """Pre-flight demo readiness gate — PASS / FAIL / BLOCKED with per-check detail.

    Composes the demo-safety controls (kill switch, ownership, risk caps, verified exchange
    metadata, TP/SL capability, real-data strategy eligibility, reconciliation). Places no
    orders. Exits non-zero unless the verdict is PASS.

    When demo/testnet credentials are configured it connects a (read-only) venue so the
    reconciliation check runs against the REAL exchange book and can PASS; without credentials
    that one check stays BLOCKED (it cannot confirm a clean book up front)."""
    from src.live.demo_guard import DemoReadinessGuard

    settings = get_settings()
    venue = None
    if (
        settings.exchange_env in ("demo", "testnet")
        and settings.exchange_api_key
        and settings.exchange_api_secret
    ):
        try:
            from src.exchange.metadata import load_metadata_for
            from src.execution.live_venue import get_venue

            venue = get_venue(load_metadata_for(settings.exchange_id), settings, live=True)
        except Exception as exc:  # noqa: BLE001 - no creds / no network → recon stays BLOCKED
            typer.echo(f"(reconciliation will stay BLOCKED — could not connect venue: {exc})")

    report = DemoReadinessGuard(settings, venue=venue).evaluate()
    typer.echo(report.report())
    typer.echo(json.dumps(report.to_dict(), indent=2))
    if not report.ok:
        raise typer.Exit(code=1)


@app.command(name="demo-smoke")
def demo_smoke(
    no_cleanup: bool = typer.Option(
        False, "--no-cleanup", help="leave the smoke position/orders open (default: close them)"
    ),
    max_ticks: int = typer.Option(1, "--max-ticks", help="bounded tick cap (1 = one order)"),
) -> None:
    """Safe, bounded demo smoke test: at most one minimal-notional order with mandatory SL/TP.

    Runs the readiness gate first and places NOTHING unless it PASSes; then drives the real
    demo/testnet venue for at most ``--max-ticks`` tick(s), reconciles immediately, and (by
    default) closes the smoke position so the demo book is left flat. Never enables live
    trading. Exits non-zero if readiness blocked or the run halted."""
    from src.live.smoke import run_demo_smoke

    result = run_demo_smoke(cleanup=not no_cleanup, max_ticks=max_ticks)
    typer.echo(result.report())
    typer.echo(json.dumps(result.to_dict(), indent=2))
    if not result.ok:
        raise typer.Exit(code=1)


@app.command()
def config() -> None:
    """Print the active (non-secret) configuration and versions."""
    s = get_settings()
    payload = {
        "app_env": s.app_env.value,
        "trading_mode": s.trading_mode.value,
        "live_trading_allowed": s.live_trading_allowed,
        "exchange_id": s.exchange_id,
        "versions": s.versions(),
    }
    typer.echo(json.dumps(payload, indent=2))


@app.command()
def scheduler(max_ticks: int = typer.Option(0, help="0 = run forever")) -> None:
    """Run the periodic job scheduler (enqueues recurring shadow-only jobs)."""
    configure_logging()
    from src.jobs.handlers import ensure_handlers_registered
    from src.scheduler import Scheduler

    ensure_handlers_registered()
    s = Scheduler()
    typer.echo(f"scheduler started (enabled={s.settings.scheduler_enabled})")
    s.run(max_ticks=max_ticks or None)


@app.command(name="config-freeze")
def config_freeze() -> None:
    """Freeze the current version set into the CONFIG-FREEZE manifest (records git commit)."""
    from src.config_freeze import freeze_config

    path = freeze_config()
    typer.echo(f"config frozen → {path}")


if __name__ == "__main__":
    app()
