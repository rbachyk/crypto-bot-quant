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


def _check_schema() -> ComponentHealth:
    """Is the database at the migration head this code expects?

    Nothing in the deploy path runs ``alembic upgrade`` — it is a manual ``make migrate`` — so new
    code can meet an old database. The symptom is brutal to diagnose from the outside: only the
    pages that touch the changed table break, rendering the generic error shell while the real
    cause (``column X does not exist``) sits in the logs. Reporting it here puts drift on /health,
    where the operator and any monitor already look."""
    try:
        from alembic.config import Config
        from alembic.script import ScriptDirectory

        from src.config.settings import REPO_ROOT

        cfg = Config(str(REPO_ROOT / "alembic.ini"))
        cfg.set_main_option("script_location", str(REPO_ROOT / "migrations"))
        heads = set(ScriptDirectory.from_config(cfg).get_heads())
        with get_engine().connect() as conn:
            current = {r[0] for r in conn.execute(text("SELECT version_num FROM alembic_version"))}
    except Exception as exc:  # noqa: BLE001 - can't determine → don't fail health on a blind spot
        return ComponentHealth("schema", True, f"not determinable ({type(exc).__name__})")
    if not heads or current == heads:
        return ComponentHealth("schema", True, f"at head {sorted(heads) or ['none']}")
    return ComponentHealth(
        "schema",
        False,
        f"database at {sorted(current) or ['<unstamped>']}, code expects {sorted(heads)} — "
        "run `make migrate` (pages touching changed tables will error until you do)",
    )


def _check_redis(settings: Settings) -> ComponentHealth:
    try:
        # Both timeouts matter: connect_timeout bounds a dead host, socket_timeout bounds a
        # hung-but-accepting Redis (without it PING can block forever and hang the probe).
        client = redis.Redis.from_url(
            settings.redis_url, socket_connect_timeout=1, socket_timeout=1
        )
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
        _check_schema(),
        _check_redis(settings),
        _check_storage(settings),
    ]
    if include_killswitch:
        components.append(_check_killswitch(settings))
    healthy = all(c.healthy for c in components)
    return HealthReport(service=service, healthy=healthy, components=components)


def check_readiness(settings: Settings | None = None) -> HealthReport:
    """Readiness probe (``/readyz``): the two hard runtime dependencies only (DB + Redis).

    Complements ``/livez`` (process up): a service is *ready* to take traffic when it can reach
    its database and queue — storage/kill-switch state is reported by the full health check."""
    settings = settings or get_settings()
    components = [_check_database(), _check_redis(settings)]
    return HealthReport(
        service=settings.service_name,
        healthy=all(c.healthy for c in components),
        components=components,
    )
