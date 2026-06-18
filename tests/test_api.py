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


def test_api_me_requires_auth() -> None:
    assert client.get("/api/me").status_code == 401
    assert client.get("/api/me", auth=AUTH).json()["user"] == "admin"


@requires_redis
def test_enqueue_unknown_job_rejected() -> None:
    resp = client.post("/api/jobs/not_a_real_job", auth=AUTH)
    assert resp.status_code == 400


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
def test_approvals_create_list_decide_loop() -> None:
    # The approvals surface is fully wired: an operator can REQUEST an approval, see it
    # pending, and approve it (previously the table was read by the UI but never written).
    import uuid

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
