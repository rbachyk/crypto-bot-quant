"""System health checks (AGENTS.md Section 25, Appendix B.11/B.16).

Each dependency is probed independently so health endpoints can report green
per service and the Infrastructure/Monitoring gates can assert reachability.
Checks are defensive: a probe failure becomes an ``unhealthy`` component, never
an exception that crashes the caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import redis
from sqlalchemy import text

from src.config import Settings, get_settings
from src.db.base import get_engine
from src.killswitch import KillSwitch
from src.storage import DataLake


@dataclass(slots=True)
class ComponentHealth:
    name: str
    healthy: bool
    detail: str = ""


@dataclass(slots=True)
class HealthReport:
    service: str
    healthy: bool
    components: list[ComponentHealth] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "service": self.service,
            "status": "healthy" if self.healthy else "unhealthy",
            "components": [
                {"name": c.name, "healthy": c.healthy, "detail": c.detail} for c in self.components
            ],
        }


def _check_database() -> ComponentHealth:
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return ComponentHealth("database", True, "reachable")
    except Exception as exc:  # noqa: BLE001
        return ComponentHealth("database", False, f"unreachable: {exc}")


def _check_redis(settings: Settings) -> ComponentHealth:
    try:
        client = redis.Redis.from_url(settings.redis_url, socket_connect_timeout=1)
        client.ping()
        return ComponentHealth("redis", True, "reachable")
    except Exception as exc:  # noqa: BLE001
        return ComponentHealth("redis", False, f"unreachable: {exc}")


def _check_storage(settings: Settings) -> ComponentHealth:
    lake = DataLake(settings.data_lake_path, settings.artifact_path)
    if lake.writable():
        return ComponentHealth("storage", True, str(settings.data_lake_path))
    return ComponentHealth("storage", False, f"not writable: {settings.data_lake_path}")


def _check_killswitch(settings: Settings) -> ComponentHealth:
    ks = KillSwitch(settings)
    engaged = ks.engaged()
    # The kill switch being engaged is a valid state, not an unhealthy probe;
    # health here means "the control is observable".
    return ComponentHealth("kill_switch", True, "engaged" if engaged else "clear")


def check_health(
    service: str | None = None,
    settings: Settings | None = None,
    *,
    include_killswitch: bool = True,
) -> HealthReport:
    """Probe all infrastructure dependencies and return a health report."""
    settings = settings or get_settings()
    service = service or settings.service_name
    components = [
        _check_database(),
        _check_redis(settings),
        _check_storage(settings),
    ]
    if include_killswitch:
        components.append(_check_killswitch(settings))
    healthy = all(c.healthy for c in components)
    return HealthReport(service=service, healthy=healthy, components=components)
