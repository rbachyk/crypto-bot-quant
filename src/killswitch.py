"""Manual kill switch — independent of the dashboard (AGENTS.md Section 2.2).

The kill switch is a non-negotiable safety control and must work even if the
UI is down (Section 2.1/2.2, KILL gate). It is implemented with two redundant
backends so it functions whether or not Redis is reachable:

* a Redis flag (``qbot:killswitch``) — shared, fast, observed by all services;
* a local file (``var/KILL_SWITCH``) — survives Redis being unavailable.

``engaged()`` returns True if **either** backend reports engaged (fail-safe:
any signal halts). The CLI (``make kill`` / ``qbot kill``) engages it without
touching the dashboard or any web process.

``engaged()`` is polled at high frequency by the live loops (composed into their
stop conditions so an engage is honoured between bars, not one-bar-late), so it
must be CHEAP: the Redis client is cached on the instance (lazy connect, recreated
on error, with a short back-off after a failed connect so an outage never stalls
the loop on connect timeouts) and the result is cached for a short TTL. The local
file backend is always consulted when the cache expires and stays authoritative.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

import redis

from src.config import Settings, get_settings

_REDIS_KEY = "qbot:killswitch"


def _kill_file(settings: Settings) -> Path:
    return settings.data_lake_path.parent / "KILL_SWITCH"


class KillSwitch:
    """Redundant, dashboard-independent kill switch."""

    # engaged() result cache TTL: the live loops poll roughly every second while sleeping
    # between bars (H6) — this keeps that polling near-free while still bounding the
    # enforcement latency to a couple of seconds.
    _CACHE_TTL_S = 1.5
    # After a failed Redis connect, don't re-attempt for this long. The file backend is
    # still checked on every (uncached) read, so a Redis outage degrades gracefully
    # instead of stalling every poll on a 1s connect timeout (M17).
    _RETRY_BACKOFF_S = 5.0

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._file = _kill_file(self.settings)
        self._client: redis.Redis | None = None  # cached connection (lazy; recreated on error)
        self._down_until: float = 0.0  # monotonic deadline of the connect back-off window
        self._cached: tuple[float, bool] | None = None  # (monotonic ts, engaged) result cache

    def _redis(self) -> redis.Redis | None:
        """The cached Redis client, connecting lazily. Never raises: returns None while
        Redis is unreachable (and backs off between connect attempts)."""
        if self._client is not None:
            return self._client
        if time.monotonic() < self._down_until:
            return None
        try:
            client = redis.Redis.from_url(
                self.settings.redis_url, socket_connect_timeout=1, socket_timeout=1
            )
            client.ping()
        except Exception:
            self._down_until = time.monotonic() + self._RETRY_BACKOFF_S
            return None
        self._client = client
        return client

    def _drop_client(self) -> None:
        """Forget a (presumed broken) cached connection so the next call reconnects."""
        self._client = None

    def _redis_write(self, fn) -> None:
        """Run a Redis mutation with one reconnect-and-retry. Never raises — the file
        backend has already been updated, so a Redis outage can't lose the engage."""
        for _ in range(2):
            client = self._redis()
            if client is None:
                return
            try:
                fn(client)
                return
            except Exception:
                self._drop_client()

    def engage(self, reason: str = "manual", actor: str = "cli") -> None:
        """Engage the kill switch on every available backend (fail-safe)."""
        stamp = f"{datetime.now(UTC).isoformat()}|{actor}|{reason}"
        # File backend always works locally and never depends on the UI.
        self._file.parent.mkdir(parents=True, exist_ok=True)
        self._file.write_text(stamp, encoding="utf-8")
        self._redis_write(lambda c: c.set(_REDIS_KEY, stamp))
        self._cached = None  # bust the result cache: the next engaged() must see it NOW

    def disengage(self, actor: str = "cli") -> None:
        """Clear the kill switch (manual reset only)."""
        if self._file.exists():
            self._file.unlink()
        self._redis_write(lambda c: c.delete(_REDIS_KEY))
        self._cached = None

    def _redis_has_key(self, client: redis.Redis | None) -> bool:
        """Whether redis reports the switch engaged. Never raises — a redis error mid-read
        (e.g. the connection drops after ping) must not propagate up through risk evaluation
        and crash the trading loop. The local file backend stays authoritative regardless.
        A failed read drops the cached client so the next call reconnects."""
        if client is None:
            return False
        try:
            return bool(client.exists(_REDIS_KEY))
        except Exception:
            self._drop_client()
            return False

    def engaged(self) -> bool:
        """True if any backend reports the switch engaged. The local file backend is
        authoritative and always readable, so a redis outage can never read as 'clear' by
        raising — it degrades to the file backend instead of crashing. The result is cached
        for ~1.5s so high-frequency polling (the live loops' stop conditions) is cheap;
        engage()/disengage() on THIS instance bust the cache immediately."""
        now = time.monotonic()
        if self._cached is not None and now - self._cached[0] < self._CACHE_TTL_S:
            return self._cached[1]
        value = self._file.exists() or self._redis_has_key(self._redis())
        self._cached = (now, value)
        return value

    def status(self) -> dict[str, object]:
        client = self._redis()
        return {
            "engaged": self.engaged(),
            "file_backend": self._file.exists(),
            "redis_backend": self._redis_has_key(client),
            "redis_reachable": client is not None,
            "detail": self._file.read_text(encoding="utf-8") if self._file.exists() else "",
        }
