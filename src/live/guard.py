"""Live-activation guard — the single chokepoint before any real-money order.

A real (mainnet) :class:`~src.execution.live_venue.CcxtLiveVenue` calls
:meth:`LiveActivationGuard.allow_live_order` before placing. The guard refuses unless
EVERY live-safety precondition holds (AGENTS.md Sections 2, 27, 35):

1. ``settings.live_trading_allowed`` — TRADING_MODE=LIVE + APP_ENV=production +
   ENABLE_LIVE_TRADING=true (the single config predicate);
2. every ``blocks_live`` gate currently PASSes (Road to Live = 100%);
3. an APPROVED ``live_activation`` sign-off exists (a second operator, Section 27);
4. the bounded-live caps (``configs/live.yaml``) are not exceeded — per-order notional,
   max orders per session, max concurrent positions.

Each external check is injectable so the guard is unit-testable without a live
deployment. The default checks read the real gate results + approvals from the DB.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from src.config import Settings, get_settings
from src.config.settings import REPO_ROOT
from src.execution.order import OrderPlan

LIVE_YAML = REPO_ROOT / "configs" / "live.yaml"


@dataclass(frozen=True, slots=True)
class LiveLimits:
    max_orders_per_session: int = 5
    max_open_positions: int = 2
    max_order_notional_pct: float = 0.05
    account_equity: float = 10_000.0


@dataclass(frozen=True, slots=True)
class BasketDemoLimits:
    """Caps for the cross-sectional DEMO path (``configs/live.yaml`` → ``basket_demo``).

    Demo/testnet only — virtual funds. These do NOT authorise anything on a real-money account;
    live placement still goes through :class:`LiveActivationGuard`."""

    disaster_stop_frac: float = 0.25
    max_gross_pct: float = 0.50
    min_account_equity: float = 100.0
    require_flat_book: bool = True


@lru_cache
def load_basket_demo_limits(path: str | None = None) -> BasketDemoLimits:
    yaml_path = Path(path) if path else LIVE_YAML
    data = (yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}).get("basket_demo", {})
    return BasketDemoLimits(
        disaster_stop_frac=max(0.0, float(data.get("disaster_stop_frac", 0.25))),
        max_gross_pct=max(0.0, float(data.get("max_gross_pct", 0.50))),
        min_account_equity=max(0.0, float(data.get("min_account_equity", 100.0))),
        require_flat_book=bool(data.get("require_flat_book", True)),
    )


@lru_cache
def load_live_limits(path: str | None = None) -> LiveLimits:
    yaml_path = Path(path) if path else LIVE_YAML
    data = (yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}).get("live", {})
    return LiveLimits(
        max_orders_per_session=int(data.get("max_orders_per_session", 5)),
        max_open_positions=int(data.get("max_open_positions", 2)),
        max_order_notional_pct=float(data.get("max_order_notional_pct", 0.05)),
        account_equity=float(data.get("account_equity", 10_000.0)),
    )


def _gates_all_pass() -> bool:
    """Every ``blocks_live`` gate currently PASSes (Road to Live = 100%)."""
    from src.api.stats import compute_gate_stats, resolve_window

    g = compute_gate_stats(resolve_window("all", None, None))
    return g.total_critical_gates > 0 and g.critical_gates_passed == g.total_critical_gates


def _live_activation_approved() -> bool:
    """An APPROVED ``live_activation`` sign-off exists (Section 27)."""
    from sqlalchemy import select

    from src.db.base import session_scope
    from src.db.models import Approval, ApprovalStatus

    with session_scope() as s:
        row = (
            s.execute(
                select(Approval).where(
                    Approval.subject_type == "live_activation",
                    Approval.subject_id == "LIVE",
                    Approval.status == ApprovalStatus.APPROVED,
                )
            )
            .scalars()
            .first()
        )
        return row is not None


class LiveActivationGuard:
    """Authorises (or refuses) each real-money order; tracks bounded-live usage."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        limits: LiveLimits | None = None,
        gates_pass: Callable[[], bool] = _gates_all_pass,
        approved: Callable[[], bool] = _live_activation_approved,
        state_store=None,
    ) -> None:
        self.settings = settings or get_settings()
        self.limits = limits or load_live_limits()
        self._gates_pass = gates_pass
        self._approved = approved
        # M9: the bounded-live order budget is PERSISTED (RiskStateStore, keyed by env) so a
        # process restart cannot refill it — the count restores here and writes through on every
        # successful placement. Cleared only by the operator's explicit `qbot risk-reset`.
        self._state_store = state_store
        self._orders_placed = (
            int(state_store.get("guard_orders_placed", 0) or 0) if state_store is not None else 0
        )
        self._open_positions = 0
        # Optional source of the CURRENT concurrent owned-position count. When wired (by the live
        # loop, from the reconciled venue mirror) the max_open_positions cap binds to REAL
        # concurrency — a closed position frees a slot — instead of an internal counter that only
        # ever incremented (register_close was never called).
        self._position_source: Callable[[], int] | None = None

    def set_position_source(self, source: Callable[[], int]) -> None:
        """Wire a live concurrent-position counter so the cap reflects real open exposure."""
        self._position_source = source

    def allow_live_order(self, plan: OrderPlan) -> tuple[bool, str]:
        """The four-gate live-safety check + bounded caps. Returns (allowed, reason)."""
        if not self.settings.live_trading_allowed:
            return False, "live trading not enabled (TRADING_MODE/APP_ENV/ENABLE_LIVE_TRADING)"
        if not self._gates_pass():
            return False, "not all blocks_live gates PASS (Road to Live < 100%)"
        if not self._approved():
            return False, "no APPROVED live_activation sign-off (operator review required)"
        if self._orders_placed >= self.limits.max_orders_per_session:
            return (
                False,
                f"bounded-live cap: max_orders_per_session={self.limits.max_orders_per_session}",
            )
        open_now = self._position_source() if self._position_source else self._open_positions
        if open_now >= self.limits.max_open_positions:
            return False, f"bounded-live cap: max_open_positions={self.limits.max_open_positions}"
        notional = self._notional(plan)
        cap = self.limits.account_equity * self.limits.max_order_notional_pct
        if notional > cap + 1e-9:
            return False, f"bounded-live cap: order notional {notional:.2f} > {cap:.2f}"
        # Authorisation only (L13): the budget is consumed by record_order_placed() once the
        # exchange ACCEPTS the order — an authorised-but-rejected placement burns no slot.
        return True, "ok"

    def record_order_placed(self, *, opened: bool = True) -> None:
        """Consume one bounded-live order slot — called by the venue AFTER the exchange accepted
        the entry order (L13: authorisation alone no longer burns budget, so a guard-approved
        order the exchange rejects doesn't shrink the session cap). ``opened`` bumps the fallback
        concurrent-position counter only when a position actually opened (fill observed); wired
        deployments count real concurrency via set_position_source instead. Persists the consumed
        budget (M9) so a restart cannot refill it."""
        self._orders_placed += 1
        if opened:
            self._open_positions += 1
        if self._state_store is not None:
            self._state_store.update(guard_orders_placed=self._orders_placed)

    def register_close(self) -> None:
        """A live position closed — free a concurrent-position slot."""
        self._open_positions = max(0, self._open_positions - 1)

    @staticmethod
    def _notional(plan: OrderPlan) -> float:
        # Market entries have no limit price; the exchange-resident stop price is a close,
        # conservative proxy for the order's notional.
        price = plan.entry.price
        if price is None and plan.stop is not None:
            price = plan.stop.stop_price
        return plan.qty * float(price or 0.0)
