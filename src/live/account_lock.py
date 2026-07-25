"""One trading session per exchange account (audit 2026-07-25e O-1).

An exchange holds ONE net position per symbol. Two sessions that trade the same symbol therefore
share a position whether or not they know it: a basket's rebalance resizes or flattens the
per-symbol strategy's stop-managed leg, its stop then guards a size that no longer exists, and each
session's reconciler reads the other's activity as drift. Account partitioning
(``live_symbols``) is what makes coexistence safe; with no partition declared — the shipped default
— only ONE session may own the account.

The API refuses to enqueue a conflicting session, but the CLI (``qbot demo-basket``) calls the
session function directly, so the interlock has to live where every entry point already passes:
the session's own pre-flight. This is that lock — a Redis key naming the environment, held for the
session's lifetime and refreshed as it runs.

**When the lock store is unreachable** the session STARTS, loudly. Redis being down is not evidence
that another session exists, and refusing to trade because the cache is unavailable would turn an
infrastructure blip into an outage — a strictly worse failure than the one this guards against, and
a regression against today's behaviour (which has no lock at all). The warning names the risk so an
operator can check.
"""

from __future__ import annotations

import contextlib
import os
import socket
from typing import Any

import structlog

_log = structlog.get_logger("live.account_lock")

#: The lock outlives a slow tick but expires soon enough that a killed session frees the account
#: without operator intervention. Refreshed by :meth:`AccountLock.refresh` while the session runs.
_TTL_S = 180


def _key(exchange_env: str) -> str:
    return f"qbot:account-session:{exchange_env}"


class AccountLock:
    """Best-effort exclusive claim on one exchange account, released on exit."""

    def __init__(self, settings: Any, *, owner: str) -> None:
        self.settings = settings
        self.env = str(getattr(settings, "exchange_env", "unknown"))
        self.owner = f"{owner}@{socket.gethostname()}:{os.getpid()}"
        self._client: Any | None = None
        self._held = False

    # -- plumbing --------------------------------------------------------- #
    def _redis(self) -> Any | None:
        if self._client is not None:
            return self._client
        try:
            import redis

            client = redis.Redis.from_url(
                self.settings.redis_url, socket_connect_timeout=2, socket_timeout=2
            )
            client.ping()
        except Exception as exc:  # noqa: BLE001 - unreachable lock store → documented degrade
            _log.warning(
                "account_lock_store_unavailable",
                env=self.env, error=str(exc),
                hint="cannot verify that no other session owns this account — starting anyway; "
                "confirm no second session is running (a shared account nets positions)",
            )
            return None
        self._client = client
        return client

    # -- the claim -------------------------------------------------------- #
    def acquire(self) -> None:
        """Claim the account, or raise ``PermissionError`` naming the current holder."""
        client = self._redis()
        if client is None:
            return  # degraded: documented above, deliberately NOT a refusal
        try:
            if client.set(_key(self.env), self.owner, nx=True, ex=_TTL_S):
                self._held = True
                _log.info("account_lock_acquired", env=self.env, owner=self.owner)
                return
            holder = client.get(_key(self.env))
        except Exception as exc:  # noqa: BLE001 - a lock-store error must not refuse a start
            _log.warning("account_lock_check_failed", env=self.env, error=str(exc))
            return
        held_by = holder.decode() if isinstance(holder, bytes) else str(holder)
        raise PermissionError(
            f"another trading session already owns the {self.env!r} account ({held_by}). An "
            "exchange holds one net position per symbol, so two sessions would trade into each "
            "other's positions. Stop it first, or give the strategies disjoint `live_symbols` in "
            "configs/strategies.yaml and co-host them."
        )

    def refresh(self) -> None:
        """Extend the claim; a session that stops refreshing frees the account by TTL."""
        if not self._held or self._client is None:
            return
        with contextlib.suppress(Exception):
            self._client.set(_key(self.env), self.owner, ex=_TTL_S)

    def release(self) -> None:
        """Release the claim — only if it is still OURS (never free someone else's)."""
        if not self._held or self._client is None:
            return
        self._held = False
        try:
            current = self._client.get(_key(self.env))
            held_by = current.decode() if isinstance(current, bytes) else str(current)
            if held_by == self.owner:
                self._client.delete(_key(self.env))
                _log.info("account_lock_released", env=self.env, owner=self.owner)
        except Exception as exc:  # noqa: BLE001 - the TTL frees it regardless
            _log.warning("account_lock_release_failed", env=self.env, error=str(exc))

    # -- context manager -------------------------------------------------- #
    def __enter__(self) -> AccountLock:
        self.acquire()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()


__all__ = ["AccountLock"]
