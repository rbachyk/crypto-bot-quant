"""Manual kill switch — independent of the dashboard (AGENTS.md Section 2.2).

The kill switch is a non-negotiable safety control and must work even if the
UI is down (Section 2.1/2.2, KILL gate). It is implemented with two redundant
backends so it functions whether or not Redis is reachable:

* a Redis flag (``qbot:killswitch``) — shared, fast, observed by all services;
* a local file (``var/KILL_SWITCH``) — survives Redis being unavailable.

``engaged()`` returns True if **either** backend reports engaged (fail-safe:
any signal halts). The CLI (``make kill`` / ``qbot kill``) engages it without
touching the dashboard or any web process.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import redis

from src.config import Settings, get_settings

_REDIS_KEY = "qbot:killswitch"


def _kill_file(settings: Settings) -> Path:
    return settings.data_lake_path.parent / "KILL_SWITCH"


class KillSwitch:
    """Redundant, dashboard-independent kill switch."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._file = _kill_file(self.settings)

    def _redis(self) -> redis.Redis | None:
        try:
            client = redis.Redis.from_url(self.settings.redis_url, socket_connect_timeout=1)
            client.ping()
            return client
        except Exception:
            return None

    def engage(self, reason: str = "manual", actor: str = "cli") -> None:
        """Engage the kill switch on every available backend (fail-safe)."""
        stamp = f"{datetime.now(UTC).isoformat()}|{actor}|{reason}"
        # File backend always works locally and never depends on the UI.
        self._file.parent.mkdir(parents=True, exist_ok=True)
        self._file.write_text(stamp, encoding="utf-8")
        client = self._redis()
        if client is not None:
            client.set(_REDIS_KEY, stamp)

    def disengage(self, actor: str = "cli") -> None:
        """Clear the kill switch (manual reset only)."""
        if self._file.exists():
            self._file.unlink()
        client = self._redis()
        if client is not None:
            client.delete(_REDIS_KEY)

    def engaged(self) -> bool:
        """True if any backend reports the switch engaged."""
        if self._file.exists():
            return True
        client = self._redis()
        return bool(client is not None and client.exists(_REDIS_KEY))

    def status(self) -> dict[str, object]:
        client = self._redis()
        return {
            "engaged": self.engaged(),
            "file_backend": self._file.exists(),
            "redis_backend": bool(client is not None and client.exists(_REDIS_KEY)),
            "redis_reachable": client is not None,
            "detail": self._file.read_text(encoding="utf-8") if self._file.exists() else "",
        }
