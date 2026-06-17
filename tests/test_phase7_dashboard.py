"""Phase 7 — Dashboard and Gate Workflow tests (AGENTS.md Appendix D Phase 7).

Verifies:
- Aggregate stats endpoint with time-period filters
- Per-symbol stats endpoint
- "Road to Live" view with all required fields
- Gate detail endpoint with spec + latest_result + remediation_actions
- Gate remediation endpoint
- Job detail + logs endpoint
- Approvals endpoint
- Audit log endpoint
- Reports endpoint
- Remediation actions endpoint
- Dashboard HTML pages render without error

These tests use FastAPI's synchronous TestClient against the real DB (for DB
tests) and an isolated in-memory settings fixture (for pure-API tests).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from src.api import create_app
from src.config import Settings

from tests.conftest import requires_redis

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_SETTINGS = Settings(
    _env_file=None,
    app_env="paper",
    dashboard_auth_mode="basic",
    dashboard_username="admin",
    dashboard_password="secret",
)
_AUTH = ("admin", "secret")


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app(_SETTINGS))


# ---------------------------------------------------------------------------
# Stats — aggregate with time-period filters
# ---------------------------------------------------------------------------


def test_stats_aggregate_all(client: TestClient) -> None:
    resp = client.get("/api/stats?period=all", auth=_AUTH)
    assert resp.status_code == 200
    body = resp.json()
    assert "period" in body
    assert body["period"] == "all"
    assert "gates" in body
    assert "jobs" in body
    assert "universe" in body
    assert "trading" in body


@pytest.mark.parametrize(
    "period",
    ["today", "yesterday", "last_7d", "last_30d", "current_month", "prev_month", "all"],
)
def test_stats_all_periods(client: TestClient, period: str) -> None:
    resp = client.get(f"/api/stats?period={period}", auth=_AUTH)
    assert resp.status_code == 200
    body = resp.json()
    assert body["period"] == period
    # Gates sub-object must have all expected keys.
    gates = body["gates"]
    for key in ("passed", "failed", "blocked", "not_run", "total", "live_readiness_score"):
        assert key in gates, f"gates.{key} missing for period={period}"
    # Jobs sub-object
    jobs = body["jobs"]
    for key in ("total", "succeeded", "failed", "running", "queued"):
        assert key in jobs, f"jobs.{key} missing for period={period}"


def test_stats_custom_period(client: TestClient) -> None:
    resp = client.get(
        "/api/stats?period=custom&from_ts=2024-01-01T00:00:00+00:00&to_ts=2024-12-31T23:59:59+00:00",
        auth=_AUTH,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["period"] == "custom"
    assert body["window_start"] is not None
    assert body["window_end"] is not None


def test_stats_gate_score_range(client: TestClient) -> None:
    resp = client.get("/api/stats", auth=_AUTH)
    assert resp.status_code == 200
    score = resp.json()["gates"]["live_readiness_score"]
    assert 0.0 <= score <= 100.0


def test_stats_requires_auth(client: TestClient) -> None:
    assert client.get("/api/stats").status_code == 401


# ---------------------------------------------------------------------------
# Stats — per-symbol
# ---------------------------------------------------------------------------


def test_per_symbol_stats_scaffold(client: TestClient) -> None:
    resp = client.get("/api/stats/BTCUSDT", auth=_AUTH)
    assert resp.status_code == 200
    body = resp.json()
    assert body["symbol"] == "BTCUSDT"
    assert "period" in body
    assert "trading" in body
    t = body["trading"]
    for key in ("total_trades", "win_rate", "expectancy_r", "realized_pnl", "max_drawdown_pct"):
        assert key in t, f"trading.{key} missing"


def test_per_symbol_stats_with_period(client: TestClient) -> None:
    resp = client.get("/api/stats/ETHUSDT?period=last_7d", auth=_AUTH)
    assert resp.status_code == 200
    assert resp.json()["symbol"] == "ETHUSDT"
    assert resp.json()["period"] == "last_7d"


def test_symbols_list(client: TestClient) -> None:
    resp = client.get("/api/stats/symbols", auth=_AUTH)
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


# ---------------------------------------------------------------------------
# Road to Live view
# ---------------------------------------------------------------------------


def test_road_to_live_structure(client: TestClient) -> None:
    resp = client.get("/api/gates/road-to-live", auth=_AUTH)
    assert resp.status_code == 200
    body = resp.json()
    assert "live_readiness_score" in body
    assert "critical_gates_passed" in body
    assert "total_critical_gates" in body
    assert "gates" in body
    assert isinstance(body["gates"], list)


def test_road_to_live_gate_fields(client: TestClient) -> None:
    resp = client.get("/api/gates/road-to-live", auth=_AUTH)
    body = resp.json()
    gates = body["gates"]
    assert len(gates) > 0, "Road to Live must list at least one gate"
    required_gate_fields = (
        "gate_id",
        "name",
        "phase",
        "status",
        "blocks_live",
        "next_action",
        "remediation_steps",
    )
    for gate in gates:
        for field in required_gate_fields:
            assert field in gate, f"gates[*].{field} missing"


def test_road_to_live_score_range(client: TestClient) -> None:
    body = client.get("/api/gates/road-to-live", auth=_AUTH).json()
    score = body["live_readiness_score"]
    assert 0.0 <= score <= 100.0


def test_road_to_live_requires_auth(client: TestClient) -> None:
    assert client.get("/api/gates/road-to-live").status_code == 401


# ---------------------------------------------------------------------------
# Gates — list, detail, remediation, run
# ---------------------------------------------------------------------------


def test_gates_list(client: TestClient) -> None:
    resp = client.get("/api/gates", auth=_AUTH)
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_gate_detail_structure(client: TestClient) -> None:
    resp = client.get("/api/gates/INFRA", auth=_AUTH)
    assert resp.status_code == 200
    body = resp.json()
    assert body["gate_id"] == "INFRA"
    assert "spec" in body
    assert "latest_result" in body
    assert "remediation_actions" in body
    spec = body["spec"]
    required_spec_fields = (
        "name",
        "phase",
        "pass_condition",
        "remediation_steps",
        "depends_on",
        "blocks_live",
    )
    for field in required_spec_fields:
        assert field in spec, f"spec.{field} missing"
    result = body["latest_result"]
    assert "status" in result


def test_gate_remediation_endpoint(client: TestClient) -> None:
    resp = client.get("/api/gates/INFRA/remediation", auth=_AUTH)
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


@requires_redis
def test_gate_run_enqueues_job(client: TestClient) -> None:
    resp = client.post("/api/gates/INFRA/run", auth=_AUTH)
    assert resp.status_code == 200
    body = resp.json()
    assert "job_id" in body
    assert body["gate_id"] == "INFRA"


# ---------------------------------------------------------------------------
# Jobs — list, detail, logs
# ---------------------------------------------------------------------------


def test_jobs_list(client: TestClient) -> None:
    resp = client.get("/api/jobs?limit=10", auth=_AUTH)
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_jobs_list_required_fields(client: TestClient) -> None:
    resp = client.get("/api/jobs?limit=5", auth=_AUTH)
    jobs = resp.json()
    if jobs:
        j = jobs[0]
        for field in ("job_id", "job_type", "status", "created_at", "progress"):
            assert field in j, f"jobs[*].{field} missing"


def test_job_detail_404(client: TestClient) -> None:
    resp = client.get("/api/jobs/nonexistent-id-xyz", auth=_AUTH)
    assert resp.status_code == 404


def test_job_logs_404(client: TestClient) -> None:
    resp = client.get("/api/jobs/nonexistent-id-xyz/logs", auth=_AUTH)
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Remediation actions
# ---------------------------------------------------------------------------


def test_remediation_list(client: TestClient) -> None:
    resp = client.get("/api/remediation", auth=_AUTH)
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_remediation_requires_auth(client: TestClient) -> None:
    assert client.get("/api/remediation").status_code == 401


# ---------------------------------------------------------------------------
# Approvals
# ---------------------------------------------------------------------------


def test_approvals_list(client: TestClient) -> None:
    resp = client.get("/api/approvals", auth=_AUTH)
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_approvals_404(client: TestClient) -> None:
    resp = client.post("/api/approvals/999999/approve", auth=_AUTH)
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Audit logs
# ---------------------------------------------------------------------------


def test_audit_logs_list(client: TestClient) -> None:
    resp = client.get("/api/audit-logs", auth=_AUTH)
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_audit_logs_fields(client: TestClient) -> None:
    # Trigger an action that produces an audit log entry.
    client.get("/api/killswitch", auth=_AUTH)
    resp = client.get("/api/audit-logs?limit=100", auth=_AUTH)
    logs = resp.json()
    if logs:
        entry = logs[0]
        for field in ("id", "ts", "actor", "action", "environment"):
            assert field in entry, f"audit_log.{field} missing"


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def test_reports_list(client: TestClient) -> None:
    resp = client.get("/api/reports", auth=_AUTH)
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


# ---------------------------------------------------------------------------
# Dashboard HTML pages
# ---------------------------------------------------------------------------


def test_overview_page_renders(client: TestClient) -> None:
    resp = client.get("/", auth=_AUTH)
    assert resp.status_code == 200
    assert "Control Center" in resp.text or "Overview" in resp.text
    assert "Gate Status" in resp.text


def test_gates_page_renders(client: TestClient) -> None:
    resp = client.get("/dashboard/gates", auth=_AUTH)
    assert resp.status_code == 200
    assert "Gates" in resp.text


def test_road_to_live_page_renders(client: TestClient) -> None:
    resp = client.get("/dashboard/road-to-live", auth=_AUTH)
    assert resp.status_code == 200
    assert "Road to Live" in resp.text
    assert "Live Readiness" in resp.text


def test_stats_page_renders(client: TestClient) -> None:
    resp = client.get("/dashboard/stats", auth=_AUTH)
    assert resp.status_code == 200
    assert "Statistics" in resp.text
    # Time period selector must be present.
    assert "period" in resp.text
    assert "today" in resp.text


def test_per_symbol_stats_page_renders(client: TestClient) -> None:
    resp = client.get("/dashboard/stats/BTCUSDT", auth=_AUTH)
    assert resp.status_code == 200
    assert "BTCUSDT" in resp.text


def test_jobs_page_renders(client: TestClient) -> None:
    resp = client.get("/dashboard/jobs", auth=_AUTH)
    assert resp.status_code == 200
    assert "Jobs" in resp.text


def test_remediation_page_renders(client: TestClient) -> None:
    resp = client.get("/dashboard/remediation", auth=_AUTH)
    assert resp.status_code == 200
    assert "Remediation" in resp.text


def test_approvals_page_renders(client: TestClient) -> None:
    resp = client.get("/dashboard/approvals", auth=_AUTH)
    assert resp.status_code == 200
    assert "Approvals" in resp.text


def test_audit_logs_page_renders(client: TestClient) -> None:
    resp = client.get("/dashboard/audit-logs", auth=_AUTH)
    assert resp.status_code == 200
    assert "Audit" in resp.text


def test_reports_page_renders(client: TestClient) -> None:
    resp = client.get("/dashboard/reports", auth=_AUTH)
    assert resp.status_code == 200
    assert "Reports" in resp.text


def test_gate_detail_page_renders(client: TestClient) -> None:
    resp = client.get("/dashboard/gates/INFRA", auth=_AUTH)
    assert resp.status_code == 200
    assert "INFRA" in resp.text


def test_all_pages_require_auth(client: TestClient) -> None:
    pages = [
        "/",
        "/dashboard/gates",
        "/dashboard/road-to-live",
        "/dashboard/stats",
        "/dashboard/jobs",
        "/dashboard/remediation",
        "/dashboard/approvals",
        "/dashboard/audit-logs",
        "/dashboard/reports",
    ]
    for page in pages:
        resp = client.get(page)
        assert resp.status_code == 401, f"page {page} should require auth"


# ---------------------------------------------------------------------------
# MON gate smoke test (Phase 7 acceptance gate)
# ---------------------------------------------------------------------------


@requires_redis
def test_mon_gate_passes_with_phase7_panels() -> None:
    """MON gate must pass with Phase 7 panel checks — Appendix D Phase 7."""
    from src.gates import GateRunner

    runner = GateRunner()
    result = runner.run("MON")
    assert result.overall == "PASS", (
        f"MON gate FAIL — phase 7 panel smoke test failed:\n{result.note}\n"
        + "\n".join(
            f"  {c.get('id')}: {c.get('status')} — {c.get('detail', '')}" for c in result.criteria
        )
    )
