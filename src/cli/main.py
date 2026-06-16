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
def worker(max_jobs: int = typer.Option(0, help="0 = run forever")) -> None:
    """Run a job worker process."""
    configure_logging()
    from src.jobs import Worker
    from src.jobs.handlers import ensure_handlers_registered

    ensure_handlers_registered()
    typer.echo("worker started")
    Worker().run(max_jobs=max_jobs or None)


@app.command()
def enqueue(job_type: str, params_json: str = typer.Option("{}", help="JSON params")) -> None:
    """Enqueue a job by type."""
    from src.jobs import JobQueue
    from src.jobs.handlers import ensure_handlers_registered

    ensure_handlers_registered()
    job_id = JobQueue().enqueue(job_type, json.loads(params_json), requested_by="cli")
    typer.echo(job_id)


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


if __name__ == "__main__":
    app()
