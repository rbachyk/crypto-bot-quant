"""FastAPI application: health endpoints, dashboard shell, jobs & gates API.

This is the Phase 1 control-center skeleton (Appendix B.8). It is deliberately
thin: it exposes health checks for monitoring, an authenticated dashboard shell,
and read/trigger endpoints for jobs and gates. It never runs heavy work in a
request handler and never starts live trading (B.17).
"""

from __future__ import annotations

from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy import desc, select

from src.api.auth import require_dashboard_auth
from src.config import Settings, get_settings
from src.db.base import session_scope
from src.db.models import GateResult, Job
from src.monitoring import Alert, AlertSeverity, check_health, get_alert_sink
from src.observability import configure_logging

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


def create_app(settings: Settings | None = None) -> FastAPI:
    configure_logging()
    settings = settings or get_settings()
    app = FastAPI(title="Quant Trading Bot — Control Center", version="0.1.0")

    # Ensure auth and any `Depends(get_settings)` resolve to *this* app's
    # settings (important when tests construct an app with explicit settings).
    app.dependency_overrides[get_settings] = lambda: settings

    # ----- health (unauthenticated; for orchestration/monitoring) ------- #
    @app.get("/health")
    def health() -> dict[str, Any]:
        return check_health(settings=settings).to_dict()

    @app.get("/health/{service}")
    def health_service(service: str) -> dict[str, Any]:
        return check_health(service=service, settings=settings).to_dict()

    @app.get("/livez")
    def livez() -> dict[str, str]:
        return {"status": "ok"}

    # ----- dashboard shell (authenticated) ------------------------------ #
    @app.get("/", response_class=HTMLResponse)
    def dashboard(user: str = Depends(require_dashboard_auth)) -> str:
        items = "".join(f"<li>{p}</li>" for p in DASHBOARD_PAGES)
        return (
            "<html><head><title>Quant Trading Bot</title></head><body>"
            f"<h1>Quant Trading Bot — Control Center</h1>"
            f"<p>Signed in as <b>{user}</b> · env={settings.app_env.value} · "
            f"mode={settings.trading_mode.value} · live_allowed={settings.live_trading_allowed}</p>"
            "<p>Phase 1 skeleton. Pages (wired in Phase 7):</p>"
            f"<ul>{items}</ul>"
            "</body></html>"
        )

    @app.get("/api/me")
    def me(user: str = Depends(require_dashboard_auth)) -> dict[str, str]:
        return {"user": user, "env": settings.app_env.value}

    # ----- jobs --------------------------------------------------------- #
    @app.get("/api/jobs")
    def list_jobs(user: str = Depends(require_dashboard_auth), limit: int = 50) -> list[dict]:
        with session_scope() as session:
            rows = (
                session.execute(select(Job).order_by(desc(Job.created_at)).limit(limit))
                .scalars()
                .all()
            )
            return [
                {
                    "job_id": j.job_id,
                    "job_type": j.job_type,
                    "status": j.status.value,
                    "progress": f"{j.progress_current}/{j.progress_total}",
                    "failure_reason": j.failure_reason,
                    "next_action_hint": j.next_action_hint,
                }
                for j in rows
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
        return {"job_id": job_id}

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str, user: str = Depends(require_dashboard_auth)) -> dict[str, bool]:
        from src.jobs import JobQueue

        return {"cancelled": JobQueue(settings).cancel(job_id)}

    @app.post("/api/jobs/{job_id}/retry")
    def retry_job(job_id: str, user: str = Depends(require_dashboard_auth)) -> dict[str, bool]:
        from src.jobs import JobQueue

        return {"requeued": JobQueue(settings).retry(job_id)}

    # ----- gates -------------------------------------------------------- #
    @app.get("/api/gates")
    def list_gates(user: str = Depends(require_dashboard_auth)) -> list[dict]:
        with session_scope() as session:
            rows = (
                session.execute(select(GateResult).order_by(desc(GateResult.id)).limit(100))
                .scalars()
                .all()
            )
            return [
                {
                    "gate_id": r.gate_id,
                    "status": r.status.value,
                    "failure_reason": r.failure_reason,
                    "report_path": r.report_path,
                }
                for r in rows
            ]

    @app.post("/api/gates/{gate_id}/run")
    def run_gate(gate_id: str, user: str = Depends(require_dashboard_auth)) -> dict:
        from src.jobs import JobQueue

        # Gates run as background jobs (never in the request handler; B.17).
        job_id = JobQueue(settings).enqueue(
            "run_gate", {"gate_id": gate_id}, requested_by=user, related_gate_id=gate_id
        )
        return {"job_id": job_id, "gate_id": gate_id}

    # ----- alerts ------------------------------------------------------- #
    @app.get("/api/alerts")
    def alerts(user: str = Depends(require_dashboard_auth), limit: int = 50) -> list[dict]:
        return [a.to_dict() for a in get_alert_sink().recent(limit=limit)]

    # ----- kill switch -------------------------------------------------- #
    @app.get("/api/killswitch")
    def killswitch_status(user: str = Depends(require_dashboard_auth)) -> dict:
        from src.killswitch import KillSwitch

        return KillSwitch(settings).status()

    @app.post("/api/killswitch/engage")
    def killswitch_engage(
        reason: str = "dashboard manual kill",
        user: str = Depends(require_dashboard_auth),
    ) -> dict:
        # The dashboard kill switch halts trading; it shares the redundant backend
        # with the CLI so either path halts the bot (Section 2.2, KILL gate).
        from src.killswitch import KillSwitch

        KillSwitch(settings).engage(reason=reason, actor=f"dashboard:{user}")
        get_alert_sink().send(
            Alert(
                title="kill switch engaged",
                severity=AlertSeverity.CRITICAL,
                component="safety",
                environment=settings.app_env.value,
                recommended_action=("Trading halted. Resume requires manual review (Section 35)."),
            )
        )
        _audit("killswitch_engage", target="killswitch", actor=user, detail={"reason": reason})
        return KillSwitch(settings).status()

    @app.post("/api/killswitch/disengage")
    def killswitch_disengage(
        confirm: bool = False,
        user: str = Depends(require_dashboard_auth),
    ) -> dict:
        # Recovery is a deliberate manual action (Section 35): refuse without confirm.
        from src.killswitch import KillSwitch

        if not confirm:
            raise HTTPException(
                status_code=400, detail="disengage requires confirm=true (manual review)"
            )
        KillSwitch(settings).disengage(actor=f"dashboard:{user}")
        _audit("killswitch_disengage", target="killswitch", actor=user, detail={})
        return KillSwitch(settings).status()

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
        except Exception:  # noqa: BLE001 - auditing must never break the safety action
            pass

    return app


app = create_app()
