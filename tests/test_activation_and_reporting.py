"""Section 27 LiveActivationRequest schema + Section 34 report envelope."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from src.api.stats import GateStats
from src.live.activation import (
    LiveActivationError,
    LiveActivationRequest,
    build_live_activation_request,
)
from src.reporting import REQUIRED_ENVELOPE_KEYS, validate_report_envelope, wrap_report


def _req() -> LiveActivationRequest:
    return LiveActivationRequest(
        request_id="r1",
        requested_by="op1",
        requested_at=datetime.now(UTC),
        gate_results=[{"gate_id": "LIVE", "status": "PASS"}],
        config_version="cfg_0001",
        strategy_versions=["strat_0001"],
        risk_policy_version="risk_0001",
        execution_policy_version="exec_0001",
    )


def test_activation_request_schema_has_all_spec_fields() -> None:
    d = _req().to_dict()
    for key in (
        "request_id",
        "requested_by",
        "requested_at",
        "gate_results",
        "config_version",
        "strategy_versions",
        "model_version",
        "learner_version",
        "risk_policy_version",
        "execution_policy_version",
        "status",
        "approved_by",
        "approved_at",
        "rejection_reason",
    ):
        assert key in d, f"LiveActivationRequest.{key} missing"
    assert d["status"] == "pending"


def test_build_refused_when_gates_not_green(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.api.stats.compute_gate_stats",
        lambda *_a, **_k: GateStats(total_critical_gates=20, critical_gates_passed=18),
    )
    with pytest.raises(LiveActivationError, match="Road to Live < 100%"):
        build_live_activation_request(requested_by="op1")


def test_build_succeeds_when_gates_green(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.api.stats.compute_gate_stats",
        lambda *_a, **_k: GateStats(
            total_critical_gates=20, critical_gates_passed=20, live_readiness_score=100.0
        ),
    )
    req = build_live_activation_request(requested_by="op1")
    assert req.status == "pending"
    assert req.gate_results  # one entry per blocks_live gate
    assert req.config_version and req.risk_policy_version and req.execution_policy_version
    assert req.live_readiness_score == 100.0


# --------------------------------------------------------------------------- #
# Section 34 — report envelope                                                #
# --------------------------------------------------------------------------- #
def test_wrap_report_has_all_envelope_keys() -> None:
    env = wrap_report(
        {"net_pnl": 10.0},
        report_type="backtest",
        methodology="event-based",
        limitations="modelled costs",
        recommendations="retune on real data",
    )
    assert validate_report_envelope(env) == []
    for key in REQUIRED_ENVELOPE_KEYS:
        assert key in env
    assert env["results"] == {"net_pnl": 10.0}


def test_validate_flags_missing_and_empty_fields() -> None:
    assert "report_type" in validate_report_envelope({})
    bad = wrap_report({}, report_type="x", methodology="")  # empty methodology
    assert "methodology" in validate_report_envelope(bad)


def test_write_report_emits_a_valid_envelope(tmp_path) -> None:
    from src.backtest.service import write_report
    from src.config import Settings

    settings = Settings(_env_file=None, reports_path=tmp_path)
    path = write_report(settings, {"label": "t", "net_pnl": 1.0}, kind="backtest")
    payload = json.loads(Path(path).read_text())
    assert validate_report_envelope(payload) == []
    assert payload["report_type"] == "backtest"
    assert payload["results"]["net_pnl"] == 1.0
    assert payload["methodology"]
