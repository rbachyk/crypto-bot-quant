"""Persistent risk-halt state (M9, Section 2.2) — the file is the authority.

A tripped daily-loss / max-drawdown / weekly-loss breaker must halt new entries and SURVIVE a
restart until an operator clears it, and `qbot risk-reset` is the only thing that clears it. Both
halves are load-bearing for incident response, and both involve TWO processes reading one file:
the running session and the operator's CLI. These tests pin that hand-off.
"""

from __future__ import annotations

import json

from src.config import Settings
from src.risk.state import RiskStateStore


def _settings(tmp_path) -> Settings:
    return Settings(_env_file=None, data_lake_path=tmp_path / "lake")


def _state_file(tmp_path, env: str = "demo"):
    return tmp_path / "risk_state" / f"{env}.json"


def test_a_tripped_halt_survives_a_restart(tmp_path) -> None:
    """The whole point of persisting it: a new process still sees the halt."""
    settings = _settings(tmp_path)
    RiskStateStore(settings, env="demo").trip("daily_loss_limit(0.05>=0.05)", ts_ms=123)

    restarted = RiskStateStore(settings, env="demo")
    assert restarted.tripped_reason() == "daily_loss_limit(0.05>=0.05)"
    assert restarted.get("tripped_at") == 123


def test_the_first_trip_wins_even_across_processes(tmp_path) -> None:
    """A later re-evaluation — in this or any other session on the environment — must not
    overwrite the ORIGINAL cause the operator will investigate."""
    settings = _settings(tmp_path)
    a = RiskStateStore(settings, env="demo")
    b = RiskStateStore(settings, env="demo")  # a second session on the same environment
    a.trip("daily_loss_limit(0.05>=0.05)")

    b.trip("max_drawdown_limit(0.10>=0.10)")  # b's in-memory copy predates a's trip

    assert a.tripped_reason() == "daily_loss_limit(0.05>=0.05)"
    assert b.tripped_reason() == "daily_loss_limit(0.05>=0.05)"


def test_an_operator_reset_reaches_the_running_session(tmp_path) -> None:
    """REGRESSION. `qbot risk-reset` runs in its OWN process. While the store cached its state in
    memory, the running session never saw the reset — it kept rejecting every candidate with
    risk_halt_pending_manual_reset — and its next routine persist wrote the stale dict back,
    resurrecting on disk the halt the operator had just cleared. The CLI printed "cleared" either
    way, so nothing surfaced it."""
    settings = _settings(tmp_path)
    running = RiskStateStore(settings, env="demo")  # the live session's store
    running.trip("daily_loss_limit(0.05>=0.05)")

    cleared = RiskStateStore(settings, env="demo").reset(actor="cli")  # the operator's CLI
    assert cleared["tripped_reason"] == "daily_loss_limit(0.05>=0.05)"

    assert running.tripped_reason() is None, "the running session must see the reset immediately"
    running.update(peak_equity=1234.0)  # its next routine persist
    on_disk = json.loads(_state_file(tmp_path).read_text())
    assert "tripped_reason" not in on_disk, "a routine persist must not resurrect a cleared halt"
    assert on_disk["peak_equity"] == 1234.0


def test_a_routine_persist_does_not_clobber_another_writers_field(tmp_path) -> None:
    """Two sessions can share one environment (account partitioning). A write must move only the
    fields it was given, never carry a whole stale snapshot over somebody else's key."""
    settings = _settings(tmp_path)
    session_a = RiskStateStore(settings, env="demo")
    session_b = RiskStateStore(settings, env="demo")
    session_a.update(peak_equity=1000.0)

    session_b.update(guard_orders_placed=3)  # b never saw a's write

    on_disk = json.loads(_state_file(tmp_path).read_text())
    assert on_disk == {"peak_equity": 1000.0, "guard_orders_placed": 3}


def test_environments_never_share_halt_state(tmp_path) -> None:
    """A demo halt must not bind live, or vice versa — separate files, by design."""
    settings = _settings(tmp_path)
    RiskStateStore(settings, env="demo").trip("daily_loss_limit(0.05>=0.05)")
    assert RiskStateStore(settings, env="live").tripped_reason() is None
