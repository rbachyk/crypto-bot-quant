"""API & dashboard-auth tests (AGENTS.md Appendix B.8, B.17).

Includes the dashboard permission tests required by Section 31: the dashboard
shell must reject unauthenticated requests.
"""

from __future__ import annotations

from fastapi.testclient import TestClient
from src.api import create_app
from src.config import Settings

from tests.conftest import requires_redis

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
