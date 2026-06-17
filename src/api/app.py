"""FastAPI application: health endpoints, dashboard, jobs, gates, stats, reports.

Phase 7 expands the Phase 1 skeleton into a full dashboard control centre
(AGENTS.md Section 25, Appendix B.8):
  - Aggregate + per-symbol statistics with time-period selectors (Section 25)
  - Background Gate Runner UI with live progress and remediation panels (B.9)
  - "Road to Live" view (Section 25)
  - Approvals + audit log endpoints (B.4)
  - Reports list (Section 34)

No heavy work runs inside a request handler (B.17). Dangerous actions
require explicit confirmation and are logged to audit_log.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy import desc, select

from src.api.auth import require_dashboard_auth
from src.config import Settings, get_settings
from src.db.base import session_scope
from src.db.models import (
    Approval,
    ApprovalStatus,
    AuditLog,
    GateResult,
    GateStatus,
    Job,
    RemediationAction,
    RemediationStatus,
)
from src.monitoring import Alert, AlertSeverity, check_health, get_alert_sink
from src.observability import configure_logging

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DASHBOARD_PAGES = [
    "Overview",
    "Data Coverage",
    "Universe",
    "Jobs",
    "Gates",
    "Remediation Actions",
    "Backtests",
    "Paper Trading",
    "Live Trading",
    "General Statistics",
    "Per-Symbol Statistics",
    "Strategy Analytics",
    "Regime Analytics",
    "Session Analytics",
    "Execution Quality",
    "Risk",
    "ML Shadow",
    "Online Learning",
    "RL",
    "Reports",
    "Approvals",
    "System Health",
    "Settings",
]

TIME_PERIODS = [
    "today",
    "yesterday",
    "last_7d",
    "last_30d",
    "current_month",
    "prev_month",
    "custom",
    "all",
]

_CSS = """
<style>
body{font-family:monospace;margin:0;background:#0d1117;color:#e6edf3}
nav{background:#161b22;padding:12px 20px;border-bottom:1px solid #30363d;display:flex;gap:16px;flex-wrap:wrap}
nav a{color:#58a6ff;text-decoration:none;font-size:13px}
nav a:hover{text-decoration:underline}
.container{padding:20px;max-width:1200px}
h1{color:#f0f6fc;font-size:20px;margin-top:0}
h2{color:#c9d1d9;font-size:16px;border-bottom:1px solid #30363d;padding-bottom:6px}
.card{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:16px;margin-bottom:16px}
.badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:12px;font-weight:bold}
.pass{background:#1a4731;color:#3fb950}
.fail{background:#4a1421;color:#f85149}
.blocked{background:#3d2b00;color:#e3b341}
.not_run{background:#21262d;color:#8b949e}
.running{background:#0d2137;color:#58a6ff}
table{width:100%;border-collapse:collapse;font-size:13px}
th{background:#21262d;color:#8b949e;text-align:left;padding:8px 10px;border:1px solid #30363d}
td{padding:8px 10px;border:1px solid #30363d;vertical-align:top}
tr:hover td{background:#1c2129}
.meta{color:#8b949e;font-size:12px}
.score{font-size:24px;font-weight:bold;color:#3fb950}
.score-low{color:#f85149}
.score-mid{color:#e3b341}
select,input{background:#21262d;color:#e6edf3;border:1px solid #30363d;padding:6px;border-radius:4px;font-family:monospace}
button,.btn{background:#238636;color:#fff;border:none;padding:6px 14px;border-radius:4px;cursor:pointer;font-family:monospace;text-decoration:none;display:inline-block;font-size:13px}
button:hover,.btn:hover{background:#2ea043}
.btn-danger{background:#b62324}
.btn-danger:hover{background:#da3633}
.btn-neutral{background:#30363d}
.btn-neutral:hover{background:#3c444d}
pre{background:#0d1117;padding:10px;border-radius:4px;overflow-x:auto;font-size:12px;border:1px solid #30363d}
.remediation-step{padding:8px;margin:4px 0;border-left:3px solid #58a6ff;background:#0d1117;font-size:13px}
.form-row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:12px}
</style>
"""

_NAV = """
<nav>
<a href="/">Overview</a>
<a href="/dashboard/gates">Gates</a>
<a href="/dashboard/road-to-live">Road to Live</a>
<a href="/dashboard/jobs">Jobs</a>
<a href="/dashboard/stats">Statistics</a>
<a href="/dashboard/remediation">Remediation</a>
<a href="/dashboard/approvals">Approvals</a>
<a href="/dashboard/audit-logs">Audit Logs</a>
<a href="/dashboard/reports">Reports</a>
<a href="/health">Health</a>
</nav>
"""


def _page(title: str, body: str) -> str:
    return (
        "<html><head>"
        f"<title>{title} — Quant Bot</title>"
        f"{_CSS}"
        "</head><body>"
        f"{_NAV}"
        f'<div class="container">'
        f"<h1>{title}</h1>"
        f"{body}"
        "</div></body></html>"
    )


def _status_badge(status: str) -> str:
    cls = {
        "passed": "pass",
        "failed": "fail",
        "blocked": "blocked",
        "not_run": "not_run",
        "running": "running",
    }.get(status.lower(), "not_run")
    return f'<span class="badge {cls}">{status.upper()}</span>'


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(settings: Settings | None = None) -> FastAPI:
    configure_logging()
    settings = settings or get_settings()
    app = FastAPI(title="Quant Trading Bot — Control Center", version="0.7.0")

    app.dependency_overrides[get_settings] = lambda: settings

    # ----- health (unauthenticated; for orchestration/monitoring) ---------- #
    @app.get("/health")
    def health() -> dict[str, Any]:
        return check_health(settings=settings).to_dict()

    @app.get("/health/{service}")
    def health_service(service: str) -> dict[str, Any]:
        return check_health(service=service, settings=settings).to_dict()

    @app.get("/livez")
    def livez() -> dict[str, str]:
        return {"status": "ok"}

    # ----- dashboard overview (authenticated) ------------------------------ #
    @app.get("/", response_class=HTMLResponse)
    def dashboard(user: str = Depends(require_dashboard_auth)) -> str:
        from src.api.stats import get_aggregate_stats

        try:
            agg = get_aggregate_stats("all")
            g = agg.gates
            score = g.live_readiness_score
            score_cls = (
                "score"
                if score >= 80
                else ("score score-mid" if score >= 50 else "score score-low")
            )
            gate_widget = f"""
<div class="card">
  <h2>Gate Status</h2>
  <div style="display:flex;gap:24px;flex-wrap:wrap;align-items:center">
    <div>
      <div class="{score_cls}">{score:.0f}%</div>
      <div class="meta">Live Readiness Score ({g.critical_gates_passed}/{g.total_critical_gates} critical)</div>
    </div>
    <div>
      {_status_badge("passed")} {g.passed}&nbsp;&nbsp;
      {_status_badge("failed")} {g.failed}&nbsp;&nbsp;
      {_status_badge("blocked")} {g.blocked}&nbsp;&nbsp;
      {_status_badge("not_run")} {g.not_run}
    </div>
  </div>
  {f'<p class="meta" style="margin-top:10px">Next action: {g.next_critical_action}</p>' if g.next_critical_action else ""}
  <p><a href="/dashboard/road-to-live" class="btn">Road to Live →</a>
     <a href="/dashboard/gates" class="btn btn-neutral" style="margin-left:8px">All Gates →</a></p>
</div>"""
            jobs_widget = f"""
<div class="card">
  <h2>Jobs (all-time)</h2>
  <p>Total {agg.jobs.total} &nbsp;|&nbsp;
     ✓ {agg.jobs.succeeded} succeeded &nbsp;|&nbsp;
     ✗ {agg.jobs.failed} failed &nbsp;|&nbsp;
     ↻ {agg.jobs.running} running &nbsp;|&nbsp;
     ⏳ {agg.jobs.queued} queued</p>
  <p><a href="/dashboard/jobs" class="btn btn-neutral">View Jobs →</a></p>
</div>"""
            universe_widget = f"""
<div class="card">
  <h2>Universe</h2>
  <p>{agg.universe.active_symbols} active / {agg.universe.total_symbols} total symbols
     {f"(v: {agg.universe.universe_version})" if agg.universe.universe_version else ""}</p>
  <p>{agg.open_remediation_items} open remediation item(s)
     {'<a href="/dashboard/remediation" class="btn btn-neutral" style="margin-left:8px">View →</a>' if agg.open_remediation_items else ""}</p>
</div>"""
        except Exception as exc:
            gate_widget = f'<div class="card"><p class="meta">Stats unavailable: {exc}</p></div>'
            jobs_widget = ""
            universe_widget = ""

        env_info = (
            f"Signed in as <b>{user}</b> · "
            f"env={settings.app_env.value} · "
            f"mode={settings.trading_mode.value} · "
            f"live_allowed={settings.live_trading_allowed}"
        )
        return _page(
            "Overview — Control Center",
            f"<p class='meta'>{env_info}</p>"
            + gate_widget
            + jobs_widget
            + universe_widget
            + f'<p class="meta">config_version={settings.config_version} · '
            f"data_version={settings.data_version}</p>",
        )

    @app.get("/api/me")
    def me(user: str = Depends(require_dashboard_auth)) -> dict[str, str]:
        return {"user": user, "env": settings.app_env.value}

    # ----- stats ----------------------------------------------------------- #
    @app.get("/api/stats")
    def aggregate_stats(
        period: str = "all",
        from_ts: str | None = None,
        to_ts: str | None = None,
        user: str = Depends(require_dashboard_auth),
    ) -> dict[str, Any]:
        from src.api.stats import get_aggregate_stats

        return get_aggregate_stats(period, from_ts, to_ts).to_dict()

    @app.get("/api/stats/symbols")
    def stats_symbols(user: str = Depends(require_dashboard_auth)) -> list[str]:
        from src.api.stats import get_symbols_list

        return get_symbols_list()

    @app.get("/api/stats/{symbol}")
    def per_symbol_stats(
        symbol: str,
        period: str = "all",
        from_ts: str | None = None,
        to_ts: str | None = None,
        user: str = Depends(require_dashboard_auth),
    ) -> dict[str, Any]:
        from src.api.stats import get_per_symbol_stats

        return get_per_symbol_stats(symbol, period, from_ts, to_ts)

    # ----- stats dashboard pages ------------------------------------------- #
    @app.get("/dashboard/stats", response_class=HTMLResponse)
    def dashboard_stats(
        period: str = "all",
        from_ts: str | None = None,
        to_ts: str | None = None,
        user: str = Depends(require_dashboard_auth),
    ) -> str:
        from src.api.stats import get_aggregate_stats

        try:
            agg = get_aggregate_stats(period, from_ts, to_ts)
            g = agg.gates
            j = agg.jobs
            score_cls = (
                "score"
                if g.live_readiness_score >= 80
                else ("score score-mid" if g.live_readiness_score >= 50 else "score score-low")
            )
            body = f"""
<div class="form-row">
  <label>Period:</label>
  <form method="get">
    <select name="period" onchange="this.form.submit()">
      {"".join(f'<option value="{p}"{" selected" if p == period else ""}>{p}</option>' for p in TIME_PERIODS)}
    </select>
    {f'<input type="text" name="from_ts" value="{from_ts or ""}" placeholder="from (ISO)" style="width:200px">' if period == "custom" else ""}
    {f'<input type="text" name="to_ts" value="{to_ts or ""}" placeholder="to (ISO)" style="width:200px">' if period == "custom" else ""}
    <button type="submit">Apply</button>
  </form>
</div>
<div class="card">
  <h2>Aggregate Statistics — {period}</h2>
  {f'<p class="meta">Window: {agg.window_start} → {agg.window_end}</p>' if agg.window_start else ""}
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Live Readiness Score</td><td><span class="{score_cls}">{g.live_readiness_score:.1f}%</span></td></tr>
    <tr><td>Gates Passed</td><td>{g.passed}</td></tr>
    <tr><td>Gates Failed</td><td>{g.failed}</td></tr>
    <tr><td>Gates Blocked</td><td>{g.blocked}</td></tr>
    <tr><td>Gates Not Run</td><td>{g.not_run}</td></tr>
    <tr><td>Jobs (total)</td><td>{j.total}</td></tr>
    <tr><td>Jobs Succeeded</td><td>{j.succeeded}</td></tr>
    <tr><td>Jobs Failed</td><td>{j.failed}</td></tr>
    <tr><td>Active Symbols</td><td>{agg.universe.active_symbols}</td></tr>
    <tr><td>Open Remediation Items</td><td>{agg.open_remediation_items}</td></tr>
    <tr><td>Total Trades (Phase 8)</td><td>{agg.trading.total_trades}</td></tr>
    <tr><td>Win Rate (Phase 8)</td><td>{agg.trading.win_rate:.1%}</td></tr>
    <tr><td>Expectancy R (Phase 8)</td><td>{agg.trading.expectancy_r:.4f}</td></tr>
    <tr><td>Max Drawdown % (Phase 8)</td><td>{agg.trading.max_drawdown_pct:.1%}</td></tr>
  </table>
</div>
<p class="meta">Trading metrics (PnL, win-rate, drawdown, fees, slippage, funding) populate in Phase 8.</p>
<p><a href="/dashboard/stats/symbol" class="btn btn-neutral">Per-Symbol Stats →</a></p>"""
        except Exception as exc:
            body = f'<div class="card"><p class="meta">Error: {exc}</p></div>'
        return _page("General Statistics", body)

    @app.get("/dashboard/stats/{symbol}", response_class=HTMLResponse)
    def dashboard_per_symbol_stats(
        symbol: str,
        period: str = "all",
        from_ts: str | None = None,
        to_ts: str | None = None,
        user: str = Depends(require_dashboard_auth),
    ) -> str:
        from src.api.stats import get_per_symbol_stats, get_symbols_list

        symbols = get_symbols_list()
        try:
            stats = get_per_symbol_stats(symbol, period, from_ts, to_ts)
            t = stats["trading"]
            body = f"""
<div class="form-row">
  <label>Symbol:</label>
  <form method="get">
    <select name="..." onchange="window.location='/dashboard/stats/'+this.value">
      {"".join(f"<option{'  selected' if s == symbol else ''}>{s}</option>" for s in (symbols or [symbol]))}
    </select>
    <label style="margin-left:8px">Period:</label>
    <select name="period" onchange="this.form.submit()">
      {"".join(f'<option value="{p}"{" selected" if p == period else ""}>{p}</option>' for p in TIME_PERIODS)}
    </select>
    <button type="submit">Apply</button>
  </form>
</div>
<div class="card">
  <h2>Per-Symbol Statistics — {symbol} — {period}</h2>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Total Trades</td><td>{t["total_trades"]}</td></tr>
    <tr><td>Win Rate</td><td>{t["win_rate"]:.1%}</td></tr>
    <tr><td>Expectancy R</td><td>{t["expectancy_r"]:.4f}</td></tr>
    <tr><td>Realized PnL</td><td>{t["realized_pnl"]:.4f}</td></tr>
    <tr><td>Total Fees</td><td>{t["total_fees_paid"]:.4f}</td></tr>
    <tr><td>Total Slippage</td><td>{t["total_slippage"]:.4f}</td></tr>
    <tr><td>Funding Paid</td><td>{t["total_funding_paid"]:.4f}</td></tr>
    <tr><td>Max Drawdown %</td><td>{t["max_drawdown_pct"]:.1%}</td></tr>
  </table>
  <p class="meta">{stats.get("note", "")}</p>
</div>"""
        except Exception as exc:
            body = f'<div class="card"><p class="meta">Error: {exc}</p></div>'
        return _page(f"Per-Symbol Statistics: {symbol}", body)

    # ----- jobs ------------------------------------------------------------ #
    @app.get("/api/jobs")
    def list_jobs(
        limit: int = 50,
        gate_id: str | None = None,
        status: str | None = None,
        job_type: str | None = None,
        user: str = Depends(require_dashboard_auth),
    ) -> list[dict]:
        with session_scope() as session:
            q = select(Job).order_by(desc(Job.created_at)).limit(limit)
            if gate_id:
                q = q.where(Job.related_gate_id == gate_id)
            if status:
                q = q.where(Job.status == status)
            if job_type:
                q = q.where(Job.job_type == job_type)
            rows = session.execute(q).scalars().all()
            return [
                {
                    "job_id": j.job_id,
                    "job_type": j.job_type,
                    "status": j.status.value,
                    "created_at": j.created_at.isoformat(),
                    "started_at": j.started_at.isoformat() if j.started_at else None,
                    "finished_at": j.finished_at.isoformat() if j.finished_at else None,
                    "progress": f"{j.progress_current}/{j.progress_total}",
                    "progress_message": j.progress_message,
                    "failure_reason": j.failure_reason,
                    "next_action_hint": j.next_action_hint,
                    "related_gate_id": j.related_gate_id,
                }
                for j in rows
            ]

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str, user: str = Depends(require_dashboard_auth)) -> dict:
        with session_scope() as session:
            job = session.get(Job, job_id)
            if not job:
                raise HTTPException(status_code=404, detail="job not found")
            logs = [
                {"ts": lg.ts.isoformat(), "level": lg.level, "message": lg.message}
                for lg in job.logs
            ]
            return {
                "job_id": job.job_id,
                "job_type": job.job_type,
                "status": job.status.value,
                "created_at": job.created_at.isoformat(),
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "finished_at": job.finished_at.isoformat() if job.finished_at else None,
                "input_params": job.input_params,
                "progress_current": job.progress_current,
                "progress_total": job.progress_total,
                "progress_message": job.progress_message,
                "failure_reason": job.failure_reason,
                "next_action_hint": job.next_action_hint,
                "related_gate_id": job.related_gate_id,
                "logs": logs,
            }

    @app.get("/api/jobs/{job_id}/logs")
    def get_job_logs(job_id: str, user: str = Depends(require_dashboard_auth)) -> list[dict]:
        with session_scope() as session:
            job = session.get(Job, job_id)
            if not job:
                raise HTTPException(status_code=404, detail="job not found")
            return [
                {"ts": lg.ts.isoformat(), "level": lg.level, "message": lg.message}
                for lg in job.logs
            ]

    @app.post("/api/jobs/{job_type}")
    def enqueue_job(
        job_type: str,
        params: dict | None = None,
        user: str = Depends(require_dashboard_auth),
    ) -> dict[str, str]:
        from src.jobs import JobQueue
        from src.jobs.handlers import ensure_handlers_registered
        from src.jobs.registry import registry

        ensure_handlers_registered()
        if not registry.has(job_type):
            raise HTTPException(status_code=400, detail=f"unknown job_type {job_type}")
        job_id = JobQueue(settings).enqueue(job_type, params or {}, requested_by=user)
        _audit("enqueue_job", target=job_type, actor=user, detail={"job_type": job_type})
        return {"job_id": job_id}

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str, user: str = Depends(require_dashboard_auth)) -> dict[str, bool]:
        from src.jobs import JobQueue

        _audit("cancel_job", target=job_id, actor=user, detail={})
        return {"cancelled": JobQueue(settings).cancel(job_id)}

    @app.post("/api/jobs/{job_id}/retry")
    def retry_job(job_id: str, user: str = Depends(require_dashboard_auth)) -> dict[str, bool]:
        from src.jobs import JobQueue

        _audit("retry_job", target=job_id, actor=user, detail={})
        return {"requeued": JobQueue(settings).retry(job_id)}

    # ----- jobs dashboard page ---------------------------------------------- #
    @app.get("/dashboard/jobs", response_class=HTMLResponse)
    def dashboard_jobs(
        limit: int = 50,
        status: str | None = None,
        user: str = Depends(require_dashboard_auth),
    ) -> str:
        with session_scope() as session:
            q = select(Job).order_by(desc(Job.created_at)).limit(limit)
            if status:
                q = q.where(Job.status == status)
            jobs = session.execute(q).scalars().all()

        rows = ""
        for j in jobs:
            prog = f"{j.progress_current}/{j.progress_total}" if j.progress_total else "-"
            rows += (
                f"<tr>"
                f"<td><a href='/dashboard/jobs/{j.job_id}'>{j.job_id[:12]}…</a></td>"
                f"<td>{j.job_type}</td>"
                f"<td>{_status_badge(j.status.value)}</td>"
                f"<td>{j.created_at.strftime('%Y-%m-%d %H:%M')}</td>"
                f"<td>{prog}</td>"
                f"<td>{j.related_gate_id or '-'}</td>"
                f"<td>{(j.failure_reason or '')[:60]}</td>"
                f"</tr>"
            )
        filter_form = f"""
<form method="get" class="form-row">
  <label>Status:</label>
  <select name="status" onchange="this.form.submit()">
    <option value="">all</option>
    {"".join(f'<option value="{s}"{" selected" if s == status else ""}>{s}</option>' for s in ["queued", "running", "succeeded", "failed", "cancelled"])}
  </select>
  <button type="submit">Filter</button>
</form>"""
        body = (
            filter_form
            + f"""
<div class="card">
  <h2>Background Jobs ({len(jobs)} shown)</h2>
  <table>
    <tr><th>ID</th><th>Type</th><th>Status</th><th>Created</th><th>Progress</th><th>Gate</th><th>Failure</th></tr>
    {rows or '<tr><td colspan="7" class="meta">No jobs found.</td></tr>'}
  </table>
</div>"""
        )
        return _page("Jobs", body)

    @app.get("/dashboard/jobs/{job_id}", response_class=HTMLResponse)
    def dashboard_job_detail(job_id: str, user: str = Depends(require_dashboard_auth)) -> str:
        with session_scope() as session:
            job = session.get(Job, job_id)
            if not job:
                return _page("Job Not Found", '<p class="meta">Job not found.</p>')
            logs_html = (
                "".join(
                    f"<tr><td>{lg.ts.strftime('%H:%M:%S')}</td><td>{lg.level}</td><td>{lg.message}</td></tr>"
                    for lg in job.logs
                )
                or '<tr><td colspan="3" class="meta">No logs.</td></tr>'
            )
            body = f"""
<div class="card">
  <h2>Job: {job.job_id}</h2>
  <table>
    <tr><th>Field</th><th>Value</th></tr>
    <tr><td>Type</td><td>{job.job_type}</td></tr>
    <tr><td>Status</td><td>{_status_badge(job.status.value)}</td></tr>
    <tr><td>Created</td><td>{job.created_at.isoformat()}</td></tr>
    <tr><td>Progress</td><td>{job.progress_current}/{job.progress_total} — {job.progress_message}</td></tr>
    <tr><td>Related Gate</td><td>{job.related_gate_id or "-"}</td></tr>
    <tr><td>Failure Reason</td><td>{job.failure_reason or "-"}</td></tr>
    <tr><td>Next Action</td><td>{job.next_action_hint or "-"}</td></tr>
  </table>
</div>
<div class="card">
  <h2>Logs</h2>
  <table>
    <tr><th>Time</th><th>Level</th><th>Message</th></tr>
    {logs_html}
  </table>
</div>"""
        return _page(f"Job {job_id[:16]}", body)

    # ----- gates ----------------------------------------------------------- #
    @app.get("/api/gates")
    def list_gates(user: str = Depends(require_dashboard_auth)) -> list[dict]:
        with session_scope() as session:
            rows = (
                session.execute(
                    select(GateResult).order_by(GateResult.gate_id, desc(GateResult.id))
                )
                .scalars()
                .all()
            )
            # Only latest result per gate.
            seen: set[str] = set()
            out = []
            for r in rows:
                if r.gate_id in seen:
                    continue
                seen.add(r.gate_id)
                out.append(
                    {
                        "gate_id": r.gate_id,
                        "status": r.status.value,
                        "failure_reason": r.failure_reason,
                        "report_path": r.report_path,
                        "criteria": r.criteria,
                        "started_at": r.started_at.isoformat() if r.started_at else None,
                    }
                )
            return out

    @app.get("/api/gates/road-to-live")
    def road_to_live(user: str = Depends(require_dashboard_auth)) -> dict[str, Any]:
        """Road to Live: all gates with status, blocking criteria, and next action."""
        from src.gates.catalog import load_catalog

        catalog = load_catalog()
        with session_scope() as session:
            rows = (
                session.execute(
                    select(GateResult).order_by(GateResult.gate_id, desc(GateResult.id))
                )
                .scalars()
                .all()
            )
            latest_by_gate: dict[str, GateResult] = {}
            for r in rows:
                if r.gate_id not in latest_by_gate:
                    latest_by_gate[r.gate_id] = r

            gates_out = []
            critical_total = 0
            critical_passed = 0
            for gate_id, spec in catalog.items():
                result = latest_by_gate.get(gate_id)
                status = result.status.value if result else "not_run"
                is_critical = spec.blocks_live == "true"
                if is_critical:
                    critical_total += 1
                    if status == "passed":
                        critical_passed += 1

                blocking = [
                    dep
                    for dep in spec.depends_on
                    if latest_by_gate.get(dep) is None
                    or latest_by_gate[dep].status is not GateStatus.PASSED
                ]

                next_action = ""
                if status == "passed":
                    next_action = "DONE"
                elif blocking:
                    next_action = f"Fix upstream first: {', '.join(blocking)}"
                elif spec.remediation_steps:
                    next_action = spec.remediation_steps[0]
                else:
                    next_action = f"Re-run gate {gate_id}"

                gates_out.append(
                    {
                        "gate_id": gate_id,
                        "name": spec.name,
                        "phase": spec.phase,
                        "status": status,
                        "blocks_live": spec.blocks_live == "true",
                        "depends_on": spec.depends_on,
                        "blocking_dependencies": blocking,
                        "pass_condition": spec.pass_condition,
                        "next_action": next_action,
                        "remediation_steps": spec.remediation_steps,
                        "failure_reason": result.failure_reason if result else None,
                        "last_run": result.started_at.isoformat()
                        if result and result.started_at
                        else None,
                        "report_path": result.report_path if result else None,
                        "rerun_job": spec.rerun_job,
                    }
                )

            score = (critical_passed / critical_total * 100.0) if critical_total else 0.0
            return {
                "live_readiness_score": round(score, 1),
                "critical_gates_passed": critical_passed,
                "total_critical_gates": critical_total,
                "gates": gates_out,
            }

    @app.get("/api/gates/{gate_id}")
    def get_gate(gate_id: str, user: str = Depends(require_dashboard_auth)) -> dict:
        from src.gates.catalog import load_catalog

        catalog = load_catalog()
        spec = catalog.get(gate_id)
        with session_scope() as session:
            result = session.execute(
                select(GateResult)
                .where(GateResult.gate_id == gate_id)
                .order_by(desc(GateResult.id))
                .limit(1)
            ).scalar_one_or_none()
            remediation = (
                session.execute(
                    select(RemediationAction)
                    .where(RemediationAction.gate_id == gate_id)
                    .order_by(RemediationAction.id.desc())
                    .limit(20)
                )
                .scalars()
                .all()
            )

        return {
            "gate_id": gate_id,
            "spec": {
                "name": spec.name if spec else gate_id,
                "phase": spec.phase if spec else "",
                "pass_condition": spec.pass_condition if spec else "",
                "remediation_steps": spec.remediation_steps if spec else [],
                "depends_on": spec.depends_on if spec else [],
                "blocks_live": (spec.blocks_live == "true") if spec else True,
                "rerun_job": spec.rerun_job if spec else "",
            },
            "latest_result": {
                "status": result.status.value if result else "not_run",
                "failure_reason": result.failure_reason if result else None,
                "criteria": result.criteria if result else [],
                "report_path": result.report_path if result else None,
                "started_at": result.started_at.isoformat()
                if result and result.started_at
                else None,
            },
            "remediation_actions": [
                {
                    "id": r.id,
                    "step_index": r.step_index,
                    "description": r.description,
                    "status": r.status.value,
                    "recommended_job": r.recommended_job,
                }
                for r in remediation
            ],
        }

    @app.get("/api/gates/{gate_id}/remediation")
    def gate_remediation(gate_id: str, user: str = Depends(require_dashboard_auth)) -> list[dict]:
        with session_scope() as session:
            actions = (
                session.execute(
                    select(RemediationAction)
                    .where(RemediationAction.gate_id == gate_id)
                    .order_by(RemediationAction.step_index)
                )
                .scalars()
                .all()
            )
            return [
                {
                    "id": a.id,
                    "step_index": a.step_index,
                    "description": a.description,
                    "status": a.status.value,
                    "recommended_job": a.recommended_job,
                    "owner_role": a.owner_role,
                    "created_at": a.created_at.isoformat(),
                }
                for a in actions
            ]

    @app.post("/api/gates/{gate_id}/run")
    def run_gate(gate_id: str, user: str = Depends(require_dashboard_auth)) -> dict:
        from src.jobs import JobQueue

        job_id = JobQueue(settings).enqueue(
            "run_gate", {"gate_id": gate_id}, requested_by=user, related_gate_id=gate_id
        )
        _audit("run_gate", target=gate_id, actor=user, detail={"gate_id": gate_id})
        return {"job_id": job_id, "gate_id": gate_id}

    @app.post("/api/gates/run-all")
    def run_all_gates(user: str = Depends(require_dashboard_auth)) -> dict:
        from src.jobs import JobQueue

        job_id = JobQueue(settings).enqueue("run_all_gates", {}, requested_by=user)
        _audit("run_all_gates", target="gates", actor=user, detail={})
        return {"job_id": job_id}

    # ----- gates dashboard pages ------------------------------------------ #
    @app.get("/dashboard/gates", response_class=HTMLResponse)
    def dashboard_gates(user: str = Depends(require_dashboard_auth)) -> str:
        from src.gates.catalog import load_catalog

        catalog = load_catalog()
        with session_scope() as session:
            rows = (
                session.execute(
                    select(GateResult).order_by(GateResult.gate_id, desc(GateResult.id))
                )
                .scalars()
                .all()
            )
            latest: dict[str, GateResult | None] = {}
            for r in rows:
                if r.gate_id not in latest:
                    latest[r.gate_id] = r

        table_rows = ""
        for gate_id, spec in catalog.items():
            gr = latest.get(gate_id)
            status = gr.status.value if gr else "not_run"
            last_run = gr.started_at.strftime("%Y-%m-%d %H:%M") if gr and gr.started_at else "never"
            table_rows += (
                f"<tr>"
                f"<td><a href='/dashboard/gates/{gate_id}'>{gate_id}</a></td>"
                f"<td>{spec.name}</td>"
                f"<td>{spec.phase}</td>"
                f"<td>{_status_badge(status)}</td>"
                f"<td>{last_run}</td>"
                f"<td><a href='/dashboard/gates/{gate_id}' class='btn btn-neutral' style='padding:2px 8px;font-size:12px'>Detail</a> "
                f"<form method='post' action='/api/gates/{gate_id}/run' style='display:inline'>"
                f"<button type='submit' class='btn' style='padding:2px 8px;font-size:12px'>▶ Run</button></form></td>"
                f"</tr>"
            )
        body = f"""
<div class="form-row">
  <form method="post" action="/api/gates/run-all">
    <button type="submit" class="btn">▶ Run All Gates</button>
  </form>
  <a href="/dashboard/road-to-live" class="btn btn-neutral">Road to Live →</a>
</div>
<div class="card">
  <h2>Gate Catalog ({len(catalog)} gates)</h2>
  <table>
    <tr><th>Gate ID</th><th>Name</th><th>Phase</th><th>Status</th><th>Last Run</th><th>Actions</th></tr>
    {table_rows or '<tr><td colspan="6" class="meta">No gates found.</td></tr>'}
  </table>
</div>"""
        return _page("Gates", body)

    @app.get("/dashboard/gates/{gate_id}", response_class=HTMLResponse)
    def dashboard_gate_detail(gate_id: str, user: str = Depends(require_dashboard_auth)) -> str:
        from src.gates.catalog import load_catalog

        catalog = load_catalog()
        spec = catalog.get(gate_id)
        with session_scope() as session:
            result = session.execute(
                select(GateResult)
                .where(GateResult.gate_id == gate_id)
                .order_by(desc(GateResult.id))
                .limit(1)
            ).scalar_one_or_none()
            actions = (
                session.execute(
                    select(RemediationAction)
                    .where(RemediationAction.gate_id == gate_id)
                    .order_by(RemediationAction.step_index)
                )
                .scalars()
                .all()
            )

        status = result.status.value if result else "not_run"
        criteria_rows = ""
        if result and result.criteria:
            for c in result.criteria:
                c_status = c.get("status", "FAIL")
                c_badge = _status_badge(c_status.lower())
                detail = c.get("detail") or c.get("failure_reason") or ""
                criteria_rows += (
                    f"<tr><td>{c.get('id', '?')}</td><td>{c_badge}</td><td>{detail}</td></tr>"
                )
        remediation_html = ""
        if actions:
            remediation_html = "<h2>Remediation Steps</h2>"
            for a in actions:
                status_badge = _status_badge(a.status.value)
                run_btn = (
                    f'<form method="post" action="/api/jobs/{a.recommended_job}"'
                    f' style="display:inline">'
                    f'<button class="btn" type="submit"'
                    f' style="padding:2px 6px;font-size:11px">'
                    f"&#9654; {a.recommended_job}</button></form>"
                    if a.recommended_job
                    else ""
                )
                remediation_html += (
                    f'<div class="remediation-step">'
                    f"<b>Step {a.step_index + 1}:</b> {a.description} "
                    f"{status_badge} {run_btn}"
                    f"</div>"
                )
        body = f"""
<p><a href="/dashboard/gates" class="btn btn-neutral">← All Gates</a>
   <form method="post" action="/api/gates/{gate_id}/run" style="display:inline;margin-left:8px">
     <button class="btn" type="submit">▶ Re-run Gate</button>
   </form></p>
<div class="card">
  <h2>Gate: {gate_id} — {spec.name if spec else gate_id}</h2>
  <p>Status: {_status_badge(status)}</p>
  <p class="meta">Phase: {spec.phase if spec else "?"} · Blocks live: {(spec.blocks_live if spec else "?")}</p>
  <p><b>Pass condition:</b> {spec.pass_condition if spec else "N/A"}</p>
  {f'<p class="meta"><b>Failure reason:</b> {result.failure_reason}</p>' if result and result.failure_reason else ""}
  {f'<p class="meta">Last run: {result.started_at.isoformat() if result else "never"}</p>'}
  {f'<p class="meta">Report: <code>{result.report_path}</code></p>' if result and result.report_path else ""}
</div>
{(f'<div class="card"><h2>Criteria</h2><table><tr><th>ID</th><th>Status</th><th>Detail</th></tr>{criteria_rows}</table></div>') if criteria_rows else ""}
<div class="card">{remediation_html or '<p class="meta">No remediation actions.</p>'}</div>"""
        return _page(f"Gate: {gate_id}", body)

    @app.get("/dashboard/road-to-live", response_class=HTMLResponse)
    def dashboard_road_to_live(user: str = Depends(require_dashboard_auth)) -> str:
        from src.gates.catalog import load_catalog

        catalog = load_catalog()
        with session_scope() as session:
            rows = (
                session.execute(
                    select(GateResult).order_by(GateResult.gate_id, desc(GateResult.id))
                )
                .scalars()
                .all()
            )
            latest: dict[str, GateResult | None] = {}
            for r in rows:
                if r.gate_id not in latest:
                    latest[r.gate_id] = r

        critical_total = sum(1 for s in catalog.values() if s.blocks_live == "true")
        critical_passed = sum(
            1
            for gid, s in catalog.items()
            if s.blocks_live == "true"
            and (gr2 := latest.get(gid)) is not None
            and gr2.status is GateStatus.PASSED
        )
        score = (critical_passed / critical_total * 100.0) if critical_total else 0.0
        score_cls = (
            "score" if score >= 80 else ("score score-mid" if score >= 50 else "score score-low")
        )

        table_rows = ""
        for gate_id, spec in catalog.items():
            gr = latest.get(gate_id)
            status = gr.status.value if gr else "not_run"
            blocking = [
                dep
                for dep in spec.depends_on
                if (dep_r := latest.get(dep)) is None or dep_r.status is not GateStatus.PASSED
            ]
            if status == "passed":
                next_action = "✓ Done"
            elif blocking:
                next_action = f"Fix upstream: {', '.join(blocking)}"
            elif spec.remediation_steps:
                next_action = spec.remediation_steps[0][:80]
            else:
                next_action = f"Re-run gate {gate_id}"

            table_rows += (
                f"<tr>"
                f"<td><a href='/dashboard/gates/{gate_id}'>{gate_id}</a></td>"
                f"<td>{spec.name}</td>"
                f"<td>{'✓' if spec.blocks_live == 'true' else ''}</td>"
                f"<td>{_status_badge(status)}</td>"
                f"<td style='max-width:300px'>{next_action}</td>"
                f"<td><form method='post' action='/api/gates/{gate_id}/run' style='display:inline'>"
                f"<button type='submit' class='btn' style='padding:2px 8px;font-size:12px'>▶ Run</button></form></td>"
                f"</tr>"
            )

        body = f"""
<div class="card">
  <h2>Live Readiness</h2>
  <div class="{score_cls}">{score:.0f}%</div>
  <p class="meta">{critical_passed} of {critical_total} critical gates passed</p>
  {'<p style="color:#3fb950">All critical gates PASS — system is ready for live activation (still requires manual approval).</p>' if score >= 100 else '<p class="meta">Fix all failed/blocked gates below to reach 100%.</p>'}
</div>
<div class="card">
  <h2>Gate Checklist</h2>
  <table>
    <tr><th>Gate</th><th>Name</th><th>Blocks Live</th><th>Status</th><th>Next Action</th><th>Run</th></tr>
    {table_rows}
  </table>
</div>"""
        return _page("Road to Live", body)

    # ----- remediation ----------------------------------------------------- #
    @app.get("/api/remediation")
    def list_remediation(
        gate_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
        user: str = Depends(require_dashboard_auth),
    ) -> list[dict]:
        with session_scope() as session:
            q = select(RemediationAction).order_by(desc(RemediationAction.id)).limit(limit)
            if gate_id:
                q = q.where(RemediationAction.gate_id == gate_id)
            if status:
                q = q.where(RemediationAction.status == status)
            actions = session.execute(q).scalars().all()
            return [
                {
                    "id": a.id,
                    "gate_id": a.gate_id,
                    "step_index": a.step_index,
                    "description": a.description,
                    "status": a.status.value,
                    "recommended_job": a.recommended_job,
                    "owner_role": a.owner_role,
                    "created_at": a.created_at.isoformat(),
                }
                for a in actions
            ]

    @app.post("/api/remediation/{action_id}/complete")
    def mark_remediation_complete(
        action_id: int, user: str = Depends(require_dashboard_auth)
    ) -> dict:
        with session_scope() as session:
            action = session.get(RemediationAction, action_id)
            if not action:
                raise HTTPException(status_code=404, detail="remediation action not found")
            action.status = RemediationStatus.DONE
        _audit(
            "remediation_complete",
            target=str(action_id),
            actor=user,
            detail={"action_id": action_id},
        )
        return {"id": action_id, "status": "done"}

    @app.get("/dashboard/remediation", response_class=HTMLResponse)
    def dashboard_remediation(user: str = Depends(require_dashboard_auth)) -> str:
        with session_scope() as session:
            actions = (
                session.execute(
                    select(RemediationAction).order_by(desc(RemediationAction.id)).limit(200)
                )
                .scalars()
                .all()
            )

        rows = ""
        for a in actions:
            rows += (
                f"<tr>"
                f"<td>{a.gate_id}</td>"
                f"<td>Step {a.step_index + 1}</td>"
                f"<td>{a.description[:100]}</td>"
                f"<td>{_status_badge(a.status.value)}</td>"
                f"<td>{a.recommended_job or '-'}</td>"
                f"<td>"
                f'<a href="/dashboard/gates/{a.gate_id}" class="btn btn-neutral" style="padding:2px 6px;font-size:11px">Gate</a>'
                f"</td>"
                f"</tr>"
            )
        body = f"""
<div class="card">
  <h2>Remediation Actions ({len(actions)} shown)</h2>
  <p class="meta">These are ordered action items for non-PASS gates. A failed gate is never a dead end.</p>
  <table>
    <tr><th>Gate</th><th>Step</th><th>Description</th><th>Status</th><th>Job</th><th>Link</th></tr>
    {rows or '<tr><td colspan="6" class="meta">No remediation actions found.</td></tr>'}
  </table>
</div>"""
        return _page("Remediation Actions", body)

    # ----- approvals ------------------------------------------------------- #
    @app.get("/api/approvals")
    def list_approvals(
        limit: int = 50,
        status: str | None = None,
        user: str = Depends(require_dashboard_auth),
    ) -> list[dict]:
        with session_scope() as session:
            q = select(Approval).order_by(desc(Approval.id)).limit(limit)
            if status:
                q = q.where(Approval.status == status)
            approvals = session.execute(q).scalars().all()
            return [
                {
                    "id": a.id,
                    "subject_type": a.subject_type,
                    "subject_id": a.subject_id,
                    "status": a.status.value,
                    "requested_by": a.requested_by,
                    "approver": a.approver,
                    "created_at": a.created_at.isoformat(),
                    "decided_at": a.decided_at.isoformat() if a.decided_at else None,
                    "evidence": a.evidence,
                }
                for a in approvals
            ]

    @app.post("/api/approvals/{approval_id}/approve")
    def approve(approval_id: int, user: str = Depends(require_dashboard_auth)) -> dict:
        from datetime import UTC, datetime

        with session_scope() as session:
            approval = session.get(Approval, approval_id)
            if not approval:
                raise HTTPException(status_code=404, detail="approval not found")
            if approval.status is not ApprovalStatus.PENDING:
                raise HTTPException(status_code=400, detail="approval is not pending")
            approval.status = ApprovalStatus.APPROVED
            approval.approver = user
            approval.decided_at = datetime.now(UTC)
        _audit("approval_approved", target=str(approval_id), actor=user, detail={})
        return {"id": approval_id, "status": "approved"}

    @app.post("/api/approvals/{approval_id}/reject")
    def reject(
        approval_id: int,
        reason: str = "rejected by operator",
        user: str = Depends(require_dashboard_auth),
    ) -> dict:
        from datetime import UTC, datetime

        with session_scope() as session:
            approval = session.get(Approval, approval_id)
            if not approval:
                raise HTTPException(status_code=404, detail="approval not found")
            if approval.status is not ApprovalStatus.PENDING:
                raise HTTPException(status_code=400, detail="approval is not pending")
            approval.status = ApprovalStatus.REJECTED
            approval.approver = user
            approval.decided_at = datetime.now(UTC)
            approval.evidence = {**approval.evidence, "rejection_reason": reason}
        _audit("approval_rejected", target=str(approval_id), actor=user, detail={"reason": reason})
        return {"id": approval_id, "status": "rejected"}

    @app.get("/dashboard/approvals", response_class=HTMLResponse)
    def dashboard_approvals(user: str = Depends(require_dashboard_auth)) -> str:
        with session_scope() as session:
            approvals = (
                session.execute(select(Approval).order_by(desc(Approval.id)).limit(100))
                .scalars()
                .all()
            )

        rows = ""
        for a in approvals:
            rows += (
                f"<tr>"
                f"<td>{a.subject_type}</td>"
                f"<td>{a.subject_id[:32]}</td>"
                f"<td>{_status_badge(a.status.value)}</td>"
                f"<td>{a.requested_by}</td>"
                f"<td>{a.approver or '-'}</td>"
                f"<td>{a.created_at.strftime('%Y-%m-%d %H:%M')}</td>"
                f"</tr>"
            )
        body = f"""
<div class="card">
  <h2>Approvals ({len(approvals)} shown)</h2>
  <table>
    <tr><th>Type</th><th>Subject</th><th>Status</th><th>Requested By</th><th>Approver</th><th>Created</th></tr>
    {rows or '<tr><td colspan="6" class="meta">No approvals found.</td></tr>'}
  </table>
</div>"""
        return _page("Approvals", body)

    # ----- audit logs ------------------------------------------------------ #
    @app.get("/api/audit-logs")
    def list_audit_logs(
        limit: int = 100,
        action: str | None = None,
        user: str = Depends(require_dashboard_auth),
    ) -> list[dict]:
        with session_scope() as session:
            q = select(AuditLog).order_by(desc(AuditLog.id)).limit(limit)
            if action:
                q = q.where(AuditLog.action == action)
            logs = session.execute(q).scalars().all()
            return [
                {
                    "id": lg.id,
                    "ts": lg.ts.isoformat(),
                    "actor": lg.actor,
                    "action": lg.action,
                    "target": lg.target,
                    "environment": lg.environment,
                    "detail": lg.detail,
                }
                for lg in logs
            ]

    @app.get("/dashboard/audit-logs", response_class=HTMLResponse)
    def dashboard_audit_logs(user: str = Depends(require_dashboard_auth)) -> str:
        with session_scope() as session:
            logs = (
                session.execute(select(AuditLog).order_by(desc(AuditLog.id)).limit(200))
                .scalars()
                .all()
            )

        rows = ""
        for lg in logs:
            rows += (
                f"<tr>"
                f"<td>{lg.ts.strftime('%Y-%m-%d %H:%M:%S')}</td>"
                f"<td>{lg.actor}</td>"
                f"<td>{lg.action}</td>"
                f"<td>{lg.target or '-'}</td>"
                f"<td>{lg.environment}</td>"
                f"</tr>"
            )
        body = f"""
<div class="card">
  <h2>Audit Log ({len(logs)} entries shown)</h2>
  <p class="meta">Immutable record of all system actions, approvals, config changes, gate runs, manual overrides.</p>
  <table>
    <tr><th>Timestamp</th><th>Actor</th><th>Action</th><th>Target</th><th>Env</th></tr>
    {rows or '<tr><td colspan="5" class="meta">No audit log entries found.</td></tr>'}
  </table>
</div>"""
        return _page("Audit Log", body)

    # ----- reports --------------------------------------------------------- #
    @app.get("/api/reports")
    def list_reports(user: str = Depends(require_dashboard_auth)) -> list[dict]:
        reports_path: Path = settings.reports_path
        if not reports_path.exists():
            return []
        out = []
        for p in sorted(
            reports_path.rglob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True
        )[:100]:
            rel = str(p.relative_to(reports_path))
            stat = p.stat()
            out.append(
                {
                    "path": rel,
                    "size_bytes": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
                }
            )
        return out

    @app.get("/dashboard/reports", response_class=HTMLResponse)
    def dashboard_reports(user: str = Depends(require_dashboard_auth)) -> str:
        reports_path: Path = settings.reports_path
        reports = []
        if reports_path.exists():
            for p in sorted(
                reports_path.rglob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True
            )[:100]:
                rel = str(p.relative_to(reports_path))
                mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=UTC)
                reports.append((rel, mtime.strftime("%Y-%m-%d %H:%M")))

        rows = "".join(
            f"<tr><td><code>{rel}</code></td><td>{mtime}</td></tr>" for rel, mtime in reports
        )
        body = f"""
<div class="card">
  <h2>Reports ({len(reports)} found)</h2>
  <p class="meta">Reports are stored under <code>{reports_path}</code></p>
  <table>
    <tr><th>Path</th><th>Modified</th></tr>
    {rows or '<tr><td colspan="2" class="meta">No reports found.</td></tr>'}
  </table>
</div>"""
        return _page("Reports", body)

    # ----- alerts ---------------------------------------------------------- #
    @app.get("/api/alerts")
    def alerts(limit: int = 50, user: str = Depends(require_dashboard_auth)) -> list[dict]:
        return [a.to_dict() for a in get_alert_sink().recent(limit=limit)]

    # ----- kill switch ----------------------------------------------------- #
    @app.get("/api/killswitch")
    def killswitch_status(user: str = Depends(require_dashboard_auth)) -> dict:
        from src.killswitch import KillSwitch

        return KillSwitch(settings).status()

    @app.post("/api/killswitch/engage")
    def killswitch_engage(
        reason: str = "dashboard manual kill",
        user: str = Depends(require_dashboard_auth),
    ) -> dict:
        from src.killswitch import KillSwitch

        KillSwitch(settings).engage(reason=reason, actor=f"dashboard:{user}")
        get_alert_sink().send(
            Alert(
                title="kill switch engaged",
                severity=AlertSeverity.CRITICAL,
                component="safety",
                environment=settings.app_env.value,
                recommended_action="Trading halted. Resume requires manual review (Section 35).",
            )
        )
        _audit("killswitch_engage", target="killswitch", actor=user, detail={"reason": reason})
        return KillSwitch(settings).status()

    @app.post("/api/killswitch/disengage")
    def killswitch_disengage(
        confirm: bool = False,
        user: str = Depends(require_dashboard_auth),
    ) -> dict:
        from src.killswitch import KillSwitch

        if not confirm:
            raise HTTPException(
                status_code=400, detail="disengage requires confirm=true (manual review)"
            )
        KillSwitch(settings).disengage(actor=f"dashboard:{user}")
        _audit("killswitch_disengage", target="killswitch", actor=user, detail={})
        return KillSwitch(settings).status()

    # ----- helpers --------------------------------------------------------- #
    def _audit(action: str, *, target: str, actor: str, detail: dict) -> None:
        from src.db.models import AuditLog

        try:
            with session_scope() as session:
                session.add(
                    AuditLog(
                        actor=actor,
                        action=action,
                        target=target,
                        environment=settings.app_env.value,
                        detail=detail,
                    )
                )
        except Exception:  # noqa: BLE001
            pass

    return app


# ---------------------------------------------------------------------------
# Module-level app instance (for uvicorn / gunicorn)
# ---------------------------------------------------------------------------
from datetime import UTC, datetime  # noqa: E402 — import at bottom to avoid circular

app = create_app()
