"""CLI smoke tests for the demo-readiness + promote-lake commands (Section 13/35).

The CLI is the operator's demo entrypoint; these lock in that the commands run and gate
correctly (exit non-zero when not ready / nothing promoted) without needing network or keys."""

from __future__ import annotations

from typer.testing import CliRunner

from src.cli.main import app

from tests.conftest import requires_db

runner = CliRunner()


@requires_db
def test_demo_readiness_blocks_without_verified_metadata_or_lake_strategy() -> None:
    """With the default Bybit (unverified) metadata and no lake-promoted strategy, the gate is
    BLOCKED and the command exits non-zero — never green by accident."""
    res = runner.invoke(app, ["demo-readiness"])
    assert res.exit_code == 1
    assert "BLOCKED" in res.stdout


@requires_db
def test_promote_lake_exits_nonzero_without_snapshot() -> None:
    """Without a downloaded snapshot nothing can promote on real data → exit non-zero (the
    operator must download first), and it never raises."""
    res = runner.invoke(app, ["promote-lake", "--config", "configs/data.bybit.yaml"])
    assert res.exit_code == 1
    assert "promoted" in res.stdout  # the JSON summary is printed
