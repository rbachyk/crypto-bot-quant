"""API & dashboard-auth tests (AGENTS.md Appendix B.8, B.17).

Includes the dashboard permission tests required by Section 31: the dashboard
shell must reject unauthenticated requests.
"""

from __future__ import annotations

from fastapi.testclient import TestClient
from src.api import create_app
from src.config import Settings

from tests.conftest import requires_db, requires_redis

# Force basic-auth even though tests may run in local env.
_settings = Settings(
    _env_file=None,
    app_env="paper",
    dashboard_auth_mode="basic",
    dashboard_username="admin",
    dashboard_password="secret",
)
client = TestClient(create_app(_settings))
AUTH = ("admin", "secret")


def test_livez_ok() -> None:
    assert client.get("/livez").json() == {"status": "ok"}


def test_dashboard_requires_auth() -> None:
    assert client.get("/").status_code == 401


def test_dashboard_rejects_bad_credentials() -> None:
    assert client.get("/", auth=("admin", "wrong")).status_code == 401


def test_dashboard_renders_with_auth() -> None:
    resp = client.get("/", auth=AUTH)
    assert resp.status_code == 200
    assert "Control Center" in resp.text


def test_path_params_are_escaped_no_reflected_xss() -> None:
    """A path param that reaches the rendered HTML (symbol stats form action, gate id) must be
    HTML-escaped — no reflected XSS into the authenticated control plane."""
    payload = '"><script>alert(1)</script>'
    from urllib.parse import quote

    r = client.get(f"/dashboard/stats/{quote(payload, safe='')}", auth=AUTH)
    assert r.status_code == 200
    assert "<script>alert(1)" not in r.text  # escaped, not reflected raw
    g = client.get(f"/dashboard/gates/{quote(payload, safe='')}", auth=AUTH)
    assert "<script>alert(1)" not in g.text


def test_csrf_blocks_cross_site_post() -> None:
    """A browser-marked cross-site POST (Fetch-Metadata) to a state-changing endpoint is
    rejected even with valid credentials — defends the Basic-auth control plane from CSRF."""
    r = client.post(
        "/api/killswitch/engage",
        auth=AUTH,
        headers={"sec-fetch-site": "cross-site"},
    )
    assert r.status_code == 403


def test_csrf_blocks_foreign_origin_post() -> None:
    r = client.post(
        "/api/killswitch/engage",
        auth=AUTH,
        headers={"origin": "https://evil.example.com"},
    )
    assert r.status_code == 403


def test_csrf_allows_same_origin_and_non_browser_post() -> None:
    """Same-origin (sec-fetch-site=same-origin) and non-browser callers (no fetch-metadata,
    no origin — e.g. the test client / CLI) are allowed through the CSRF guard. A non-raising
    client is used so the assertion isolates the MIDDLEWARE result from whatever the endpoint
    does (it must not be a 403), independent of redis/db availability."""
    from fastapi.testclient import TestClient

    nr = TestClient(create_app(_settings), raise_server_exceptions=False)
    same = nr.post("/api/scheduler/resume", auth=AUTH, headers={"sec-fetch-site": "same-origin"})
    assert same.status_code != 403  # same-origin allowed through the guard
    plain = nr.post("/api/scheduler/resume", auth=AUTH)
    assert plain.status_code != 403  # non-browser (no fetch-metadata/origin) allowed


def test_api_me_requires_auth() -> None:
    assert client.get("/api/me").status_code == 401
    assert client.get("/api/me", auth=AUTH).json()["user"] == "admin"


@requires_redis
def test_enqueue_unknown_job_rejected() -> None:
    resp = client.post("/api/jobs/not_a_real_job", auth=AUTH)
    assert resp.status_code == 400


def test_run_basket_rejects_non_cross_sectional_strategy() -> None:
    """The basket-paper start endpoint must reject a strategy that isn't cross-sectional BEFORE
    enqueueing (no redis hit) — only funding_carry / residual_momentum-style baskets run here."""
    resp = client.post(
        "/api/paper/run-basket",
        params={"strategy": "lead_lag", "timeframe": ""},
        auth=AUTH,
        headers={"sec-fetch-site": "same-origin"},
    )
    assert resp.status_code == 400


@requires_db
def test_live_page_offers_offline_paper_start() -> None:
    """The Live control offers an offline-PAPER start (mode=paper → SimulatedVenue: continuous, no
    real orders, no account reconciliation to halt on) distinct from the real-venue start — so the
    per-symbol ensemble can paper-trade on demo data instead of placing demo orders."""
    resp = client.get("/dashboard/live", auth=AUTH)
    assert resp.status_code == 200
    assert "Start paper session" in resp.text
    assert "mode=paper&timeframe=" in resp.text       # paper button passes the offline override
    assert "session (real orders)" in resp.text       # the real-venue start remains available


@requires_db
def test_run_basket_rejects_duplicate_session() -> None:
    """A second Start for a strategy that already has a running basket session is refused (409) —
    so a strategy can't be double-booked once continuous sessions run concurrently."""
    import uuid

    from src.db.base import session_scope
    from src.db.models import Job, JobStatus

    jid = "test-dup-" + uuid.uuid4().hex[:8]
    with session_scope() as s:
        s.add(Job(
            job_id=jid, job_type="run_basket_paper_session", status=JobStatus.RUNNING,
            input_params={"strategy": "funding_carry"},
        ))
    try:
        r = client.post(
            "/api/paper/run-basket",
            params={"strategy": "funding_carry", "timeframe": ""},
            auth=AUTH, headers={"sec-fetch-site": "same-origin"},
        )
        assert r.status_code == 409
    finally:
        with session_scope() as s:
            obj = s.get(Job, jid)
            if obj is not None:
                s.delete(obj)


@requires_db
def test_dashboard_paper_offers_basket_start_form() -> None:
    """The Paper page exposes the basket launch control with the cross-sectional candidates."""
    resp = client.get("/dashboard/paper", auth=AUTH)
    assert resp.status_code == 200
    assert "Start basket paper session" in resp.text
    assert "funding_carry" in resp.text  # a cross-sectional candidate in the dropdown
    # the form must append the selects to the URL as QUERY params (the app's convention — the
    # endpoint reads query params, not the POST body), or the strategy arrives empty.
    assert "/api/paper/run-basket?strategy=" in resp.text
    assert "basket-strat" in resp.text and "basket-tf" in resp.text


def test_dashboard_killswitch_engage_and_recovery(tmp_path) -> None:
    # Isolated kill switch (own data lake, unreachable redis ⇒ file backend) so the
    # test never touches shared state (AGENTS.md Section 2.2, KILL gate).
    iso = Settings(
        _env_file=None,
        app_env="paper",
        dashboard_auth_mode="basic",
        dashboard_username="admin",
        dashboard_password="secret",
        data_lake_path=tmp_path / "dl",
        redis_url="redis://127.0.0.1:1/0",
    )
    c = TestClient(create_app(iso))

    assert c.post("/api/killswitch/engage").status_code == 401  # auth required
    engaged = c.post("/api/killswitch/engage", auth=AUTH)
    assert engaged.status_code == 200 and engaged.json()["engaged"] is True

    # Recovery requires an explicit manual confirmation (Section 35).
    assert c.post("/api/killswitch/disengage", auth=AUTH).status_code == 400
    assert c.get("/api/killswitch", auth=AUTH).json()["engaged"] is True
    cleared = c.post("/api/killswitch/disengage?confirm=true", auth=AUTH)
    assert cleared.status_code == 200 and cleared.json()["engaged"] is False


@requires_db
def test_approvals_create_list_decide_loop(monkeypatch) -> None:
    # The approvals surface is fully wired: an operator can REQUEST an approval, see it
    # pending, and approve it (previously the table was read by the UI but never written).
    import uuid

    from src.api.stats import GateStats

    # Make the activation precondition deterministic regardless of test order: force gates green
    # and persist a real-lake promotion (live activation requires both).
    monkeypatch.setattr(
        "src.api.stats.compute_gate_stats",
        lambda *_a, **_k: GateStats(
            total_critical_gates=20, critical_gates_passed=20, live_readiness_score=100.0
        ),
    )

    # Live activation now requires an active strategy validated on REAL lake data (Section 13);
    # persist one so the activation request can be built.
    from src.strategies.promotion import persist_validations
    from src.strategies.research import CandidateValidation, SideDecision

    sd = SideDecision(
        allow_long=True, allow_short=False, long_expectancy_r=0.2, short_expectancy_r=-0.1,
        long_trades=30, short_trades=5, disabled=["short"],
    )
    persist_validations(
        [
            CandidateValidation(
                candidate_id="basis_reversion", family="B",
                strategy_version=_settings.strategy_version, promoted=True, status="promoted",
                shelved_reasons=[], side_decision=sd, hypothesis={}, report={"expectancy_r": 0.2},
                walk_forward={}, fee_stress={}, slippage_stress={}, noise_control={},
            )
        ],
        data_source="lake",
    )

    sid = f"LIVE-{uuid.uuid4().hex[:8]}"
    created = client.post(
        f"/api/approvals?subject_type=live_activation&subject_id={sid}", auth=AUTH
    )
    assert created.status_code == 200
    aid = created.json()["id"]
    assert created.json()["status"] == "pending"

    # Idempotent per pending subject: a second request returns the same id.
    again = client.post(f"/api/approvals?subject_type=live_activation&subject_id={sid}", auth=AUTH)
    assert again.json()["id"] == aid

    listing = client.get("/api/approvals", auth=AUTH).json()
    assert any(a["id"] == aid and a["status"] == "pending" for a in listing)

    approved = client.post(f"/api/approvals/{aid}/approve", auth=AUTH)
    assert approved.status_code == 200 and approved.json()["status"] == "approved"
    # Re-deciding a non-pending approval is rejected.
    assert client.post(f"/api/approvals/{aid}/approve", auth=AUTH).status_code == 400
