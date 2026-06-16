"""Dashboard authentication (AGENTS.md Appendix C, B.17).

HTTP Basic auth, mandatory outside ``local``. In ``local`` with
``DASHBOARD_AUTH_MODE=none`` auth is skipped to ease development; the config
validator forbids ``none`` in any non-local environment.
"""

from __future__ import annotations

import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from src.config import DashboardAuthMode, Settings, get_settings

_basic = HTTPBasic(auto_error=False)


def require_dashboard_auth(
    credentials: HTTPBasicCredentials | None = Depends(_basic),
    settings: Settings = Depends(get_settings),
) -> str:
    """Return the authenticated username, or raise 401."""
    if settings.dashboard_auth_mode is DashboardAuthMode.NONE:
        return "anonymous"

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Basic"},
        )
    user_ok = secrets.compare_digest(credentials.username, settings.dashboard_username)
    pass_ok = secrets.compare_digest(credentials.password, settings.dashboard_password)
    if not (user_ok and pass_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username
