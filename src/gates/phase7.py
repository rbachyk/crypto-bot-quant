"""Phase 7 gate check: MON smoke test for dashboard panels (Appendix D Phase 7).

The MON gate for Phase 7 verifies that the dashboard API exposes all required
panels with valid, structured responses:
  - Gate status panel (`/api/gates`)
  - Road to Live view (`/api/gates/road-to-live`)
  - Aggregate stats with time-period filter (`/api/stats`)
  - Per-symbol stats (`/api/stats/{symbol}`)
  - Remediation actions panel (`/api/remediation`)
  - Job logs panel (`/api/jobs`)
  - Audit log panel (`/api/audit-logs`)
  - Approvals panel (`/api/approvals`)
  - Reports panel (`/api/reports`)
"""

from __future__ import annotations

from src.config import Settings
from src.gates.result import Criterion


def check_mon_dashboard_panels(settings: Settings) -> list[Criterion]:
    """Smoke-test the Phase 7 dashboard panel API endpoints.

    Uses FastAPI's synchronous TestClient (no network required) against the
    real app wired to the real database/settings — same validation the
    Reviewer Agent performs when re-running the MON gate.
    """
    from fastapi.testclient import TestClient

    from src.api import create_app

    # Auth credentials from settings (Basic auth, paper env).
    app = create_app(settings)
    auth = (settings.dashboard_username, settings.dashboard_password)
    client = TestClient(app, raise_server_exceptions=True)
    out: list[Criterion] = []

    # ---------------------------------------------------------------------- #
    # 1. Gate status panel                                                     #
    # ---------------------------------------------------------------------- #
    try:
        resp = client.get("/api/gates", auth=auth)
        if resp.status_code == 200 and isinstance(resp.json(), list):
            n = len(resp.json())
            out.append(Criterion.ok("gate_status_panel", f"returned {n} gate results"))
        else:
            out.append(
                Criterion.fail(
                    "gate_status_panel",
                    f"status={resp.status_code} body={resp.text[:200]}",
                )
            )
    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("gate_status_panel", f"error: {exc}"))

    # ---------------------------------------------------------------------- #
    # 2. Road to Live view                                                     #
    # ---------------------------------------------------------------------- #
    try:
        resp = client.get("/api/gates/road-to-live", auth=auth)
        body = resp.json()
        required_keys = {
            "live_readiness_score",
            "critical_gates_passed",
            "total_critical_gates",
            "gates",
        }
        if resp.status_code == 200 and required_keys.issubset(body.keys()):
            out.append(
                Criterion.ok(
                    "road_to_live_panel",
                    f"score={body['live_readiness_score']}% "
                    f"({body['critical_gates_passed']}/{body['total_critical_gates']} critical)",
                )
            )
        else:
            out.append(
                Criterion.fail(
                    "road_to_live_panel",
                    f"status={resp.status_code} missing keys={required_keys - set(body.keys())}",
                )
            )
    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("road_to_live_panel", f"error: {exc}"))

    # ---------------------------------------------------------------------- #
    # 3. Aggregate stats with time-period filter                               #
    # ---------------------------------------------------------------------- #
    try:
        for period in ("all", "today", "last_7d", "last_30d"):
            resp = client.get(f"/api/stats?period={period}", auth=auth)
            body = resp.json()
            required = {"period", "gates", "jobs", "universe", "trading"}
            if resp.status_code != 200 or not required.issubset(body.keys()):
                out.append(
                    Criterion.fail(
                        "stats_panel_time_filter",
                        f"period={period} status={resp.status_code} body={str(body)[:200]}",
                    )
                )
                break
        else:
            out.append(
                Criterion.ok(
                    "stats_panel_time_filter",
                    "all/today/last_7d/last_30d periods return valid schema",
                )
            )
    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("stats_panel_time_filter", f"error: {exc}"))

    # ---------------------------------------------------------------------- #
    # 4. Per-symbol stats endpoint                                             #
    # ---------------------------------------------------------------------- #
    try:
        resp = client.get("/api/stats/BTCUSDT", auth=auth)
        body = resp.json()
        required = {"symbol", "period", "trading"}
        if resp.status_code == 200 and required.issubset(body.keys()):
            out.append(Criterion.ok("per_symbol_stats_panel", "per-symbol stats endpoint works"))
        else:
            out.append(
                Criterion.fail(
                    "per_symbol_stats_panel",
                    f"status={resp.status_code} body={str(body)[:200]}",
                )
            )
    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("per_symbol_stats_panel", f"error: {exc}"))

    # ---------------------------------------------------------------------- #
    # 5. Remediation actions panel                                             #
    # ---------------------------------------------------------------------- #
    try:
        resp = client.get("/api/remediation", auth=auth)
        if resp.status_code == 200 and isinstance(resp.json(), list):
            out.append(
                Criterion.ok("remediation_panel", f"returned {len(resp.json())} remediation items")
            )
        else:
            out.append(
                Criterion.fail(
                    "remediation_panel",
                    f"status={resp.status_code} body={resp.text[:200]}",
                )
            )
    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("remediation_panel", f"error: {exc}"))

    # ---------------------------------------------------------------------- #
    # 6. Jobs panel with live progress + logs                                  #
    # ---------------------------------------------------------------------- #
    try:
        resp = client.get("/api/jobs?limit=5", auth=auth)
        if resp.status_code == 200 and isinstance(resp.json(), list):
            out.append(Criterion.ok("jobs_panel", f"returned {len(resp.json())} jobs"))
        else:
            out.append(
                Criterion.fail("jobs_panel", f"status={resp.status_code} body={resp.text[:200]}")
            )
    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("jobs_panel", f"error: {exc}"))

    # ---------------------------------------------------------------------- #
    # 7. Audit log panel                                                       #
    # ---------------------------------------------------------------------- #
    try:
        resp = client.get("/api/audit-logs?limit=5", auth=auth)
        if resp.status_code == 200 and isinstance(resp.json(), list):
            out.append(Criterion.ok("audit_log_panel", "audit log endpoint accessible"))
        else:
            out.append(
                Criterion.fail(
                    "audit_log_panel",
                    f"status={resp.status_code} body={resp.text[:200]}",
                )
            )
    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("audit_log_panel", f"error: {exc}"))

    # ---------------------------------------------------------------------- #
    # 8. Approvals panel                                                       #
    # ---------------------------------------------------------------------- #
    try:
        resp = client.get("/api/approvals?limit=5", auth=auth)
        if resp.status_code == 200 and isinstance(resp.json(), list):
            out.append(Criterion.ok("approvals_panel", "approvals endpoint accessible"))
        else:
            out.append(
                Criterion.fail(
                    "approvals_panel",
                    f"status={resp.status_code} body={resp.text[:200]}",
                )
            )
    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("approvals_panel", f"error: {exc}"))

    # ---------------------------------------------------------------------- #
    # 9. Reports panel                                                         #
    # ---------------------------------------------------------------------- #
    try:
        resp = client.get("/api/reports", auth=auth)
        if resp.status_code == 200 and isinstance(resp.json(), list):
            out.append(Criterion.ok("reports_panel", f"returned {len(resp.json())} report entries"))
        else:
            out.append(
                Criterion.fail(
                    "reports_panel",
                    f"status={resp.status_code} body={resp.text[:200]}",
                )
            )
    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("reports_panel", f"error: {exc}"))

    # ---------------------------------------------------------------------- #
    # 10. Gate result rendering: failure_reason + remediation + re-run button  #
    # ---------------------------------------------------------------------- #
    try:
        resp = client.get("/api/gates/INFRA", auth=auth)
        body = resp.json()
        required = {"gate_id", "spec", "latest_result", "remediation_actions"}
        if resp.status_code == 200 and required.issubset(body.keys()):
            out.append(
                Criterion.ok(
                    "gate_result_rendering",
                    "gate detail endpoint returns spec + result + remediation_actions",
                )
            )
        else:
            out.append(
                Criterion.fail(
                    "gate_result_rendering",
                    f"status={resp.status_code} missing={required - set(body.keys())}",
                )
            )
    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("gate_result_rendering", f"error: {exc}"))

    return out
