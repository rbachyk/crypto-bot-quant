"""Persistent breaker/guard state (M9) — halts survive restarts until a MANUAL reset.

Section 2.2: "Daily-loss and max-drawdown circuit breakers … → halt to manual
reset". Before this store existed the tripped state lived only in process memory
(`PaperTradingEngine` accumulators, `LiveActivationGuard._orders_placed`), so a
restart silently cleared a tripped halt and refilled the bounded-live order
budget. This store persists the minimal state that must outlive the process:

* ``tripped_reason`` / ``tripped_at`` — a manual-reset breaker fired (daily-loss,
  max-drawdown, weekly-loss). The engine refuses ALL new entries while set.
* calendar loss-window anchors (M10): ``day_key`` / ``day_anchor_equity`` and
  ``week_key`` / ``week_anchor_equity`` — real-venue daily/weekly windows anchor
  to UTC-day / ISO-week boundaries, so a mid-day restart doesn't forget the
  day's losses and a multi-day session rolls the daily anchor at midnight.
* ``peak_equity`` — the drawdown breaker's high-water mark.
* ``consecutive_losses`` — the loss-streak cooldown counter.
* ``guard_orders_placed`` — the bounded-live order budget already consumed.

Persistence mechanism (deliberate choice): a JSON file under
``var/risk_state/<env>.json`` — the same operator-visible ``var/`` root as the
kill switch's file backend, keyed by trading environment (demo/testnet/live) so
environments never share halt state. A file was chosen over Redis (flushed by
infra resets; risk state must not vanish with the cache) and over the DB (a
halt must keep binding even when the DB is down — the risk path never blocks on
a DB call). Writes are atomic (tmp + ``os.replace``); reads are best-effort (a
corrupt file degrades to empty state with a warning, never crashes trading —
degrading to "no persisted state" only loses restart continuity, it never
loosens the live, recomputed breakers).

Manual reset path: ``qbot risk-reset --env <env> --confirm`` (src/cli/main.py)
→ :meth:`RiskStateStore.reset`. Nothing else clears a tripped halt.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import structlog

from src.config import Settings, get_settings

_log = structlog.get_logger("risk.state")


class RiskStateStore:
    """File-backed, per-environment breaker/guard state (M9/M10)."""

    def __init__(self, settings: Settings | None = None, *, env: str) -> None:
        self.settings = settings or get_settings()
        self.env = env
        self._path = self.settings.data_lake_path.parent / "risk_state" / f"{env}.json"
        self._state: dict[str, Any] = self._read()

    @property
    def path(self) -> Path:
        return self._path

    def _read(self) -> dict[str, Any]:
        try:
            if self._path.exists():
                data = json.loads(self._path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
                _log.warning("risk_state_not_a_dict", path=str(self._path))
        except Exception:  # noqa: BLE001 - a corrupt file must never crash the trading loop
            _log.warning("risk_state_unreadable", path=str(self._path), exc_info=True)
        return {}

    def _write(self) -> None:
        """Atomic best-effort write (tmp + replace). A write failure is logged, never raised —
        the in-memory state stays correct for this process; only restart continuity is lost."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(f".tmp.{os.getpid()}")
            tmp.write_text(json.dumps(self._state, sort_keys=True), encoding="utf-8")
            os.replace(tmp, self._path)
        except Exception:  # noqa: BLE001
            _log.warning("risk_state_write_failed", path=str(self._path), exc_info=True)

    # ------------------------------------------------------------------ #
    def load(self) -> dict[str, Any]:
        """The current persisted state (in-memory copy)."""
        return dict(self._state)

    def get(self, key: str, default: Any = None) -> Any:
        return self._state.get(key, default)

    def update(self, **fields: Any) -> None:
        """Merge ``fields`` into the state and persist atomically."""
        self._state.update(fields)
        self._write()

    # ------------------------------------------------------------------ #
    def trip(self, reason: str, *, ts_ms: int | None = None) -> None:
        """Record a manual-reset halt. The FIRST trip wins (a later re-evaluation must not
        overwrite the original cause); idempotent while tripped."""
        if self._state.get("tripped_reason"):
            return
        self._state["tripped_reason"] = str(reason)
        if ts_ms is not None:
            self._state["tripped_at"] = int(ts_ms)
        self._write()
        _log.warning("risk_halt_tripped", env=self.env, reason=reason)

    def tripped_reason(self) -> str | None:
        r = self._state.get("tripped_reason")
        return str(r) if r else None

    def reset(self, *, actor: str = "cli") -> dict[str, Any]:
        """MANUAL reset (the only path that clears a tripped halt / refills the order budget).

        Clears the tripped flag, the loss-window anchors and the bounded-live budget; returns
        the state that was cleared so the operator sees what they reset."""
        cleared = dict(self._state)
        self._state = {}
        self._write()
        _log.warning("risk_state_reset", env=self.env, actor=actor, cleared=cleared)
        return cleared
