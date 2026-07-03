"""Learner circuit breaker and revert-to-fallback (AGENTS.md Section 21.7).

The :class:`RollbackGuard` monitors rolling performance and triggers a freeze
when any of the five rollback conditions fire:

  1. Realized performance underperforms policy's own shadow projection by
     ≥ ``rollback_margin`` over the last ``rollback_window`` decisions.
  2. Any envelope breaker fires (daily loss, drawdown, heat, beta).
  3. Live-vs-shadow decision divergence exceeds ``max_divergence``.
  4. R8/R7 regime or reconciliation failure (set_regime_flag).
  5. Manual learner kill switch (freeze() call).

On rollback, :meth:`RollbackGuard.revert` performs the full Section 21.7 atomic
sequence: freeze the controller (the frozen fallback snapshot is loaded and becomes the
effective policy in SHADOW mode; no learner action is ever applied again — or, if no
fallback snapshot exists, the learner is honestly reported DISABLED), cancel pending
learner-influenced **new** orders only (never an open position's exchange-side stop),
emit the ``learner_rollback`` alert, and write the event to ``learner_log``.
Recovery FROZEN → LIVE_BOUNDED is MANUAL only.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from src.adaptation.controller import LearnerController
from src.adaptation.scorer import ShadowDecision
from src.monitoring.alerts import Alert, AlertSeverity, AlertSink


@dataclass
class RollbackEvent:
    """Persisted when a rollback trigger fires."""

    ts: datetime = field(default_factory=lambda: datetime.now(UTC))
    trigger: str = ""
    detail: str = ""
    controller_frozen: bool = True
    fallback_active: bool = False  # the frozen fallback policy is now the effective policy
    orders_cancelled: int = 0  # pending learner-influenced NEW orders cancelled (never positions)


@dataclass
class RollbackGuard:
    """Monitors rolling performance and triggers automatic freeze when warranted.

    Instantiate one per :class:`~src.adaptation.controller.LearnerController`
    and call :meth:`check` after every decision+outcome pair. The optional hooks make
    :meth:`revert` perform the full Section 21.7 atomic revert: activate the frozen
    fallback, cancel pending learner orders, alert, and write the ``learner_log`` event.
    """

    rollback_window: int = 20
    rollback_margin: float = 0.05
    max_divergence: float = 0.20
    auto_freeze_on_breaker: bool = True  # IMMUTABLE: cannot be set false
    alert_sink: AlertSink | None = None
    cancel_orders: Callable[[], int] | None = None  # cancels learner NEW orders only
    log_writer: Callable[[RollbackEvent], None] | None = None
    environment: str = "paper"

    _decisions: list[ShadowDecision] = field(default_factory=list, init=False, repr=False)
    _events: list[RollbackEvent] = field(default_factory=list, init=False, repr=False)
    _regime_flag: bool = field(default=False, init=False, repr=False)
    _breaker_flag: bool = field(default=False, init=False, repr=False)
    _divergence_flag: bool = field(default=False, init=False, repr=False)

    def add_decision(
        self,
        projected_outcome: float,
        realized_outcome: float | None,
        *,
        ts: datetime | None = None,
        symbol: str | None = None,
        take: bool = True,
        mode: str = "SHADOW",
    ) -> None:
        self._decisions.append(
            ShadowDecision(
                ts=ts or datetime.now(UTC),
                symbol=symbol,
                projected_outcome=projected_outcome,
                realized_outcome=realized_outcome,
                take=take,
                mode=mode,
            )
        )

    def set_regime_unsafe(self, regime: str) -> None:
        """Signal a R7/R8 or reconciliation-failure regime (trigger 4)."""
        self._regime_flag = regime in ("R7_TOXIC_EXECUTION", "R8_DATA_UNSAFE", "RECON_FAILURE")

    def set_envelope_breaker(self, fired: bool) -> None:
        """Signal an envelope breaker event (trigger 2)."""
        self._breaker_flag = fired

    def set_divergence_flag(self, divergence: float) -> None:
        """Signal a live-vs-shadow divergence flag (trigger 3)."""
        self._divergence_flag = divergence > self.max_divergence

    def check(self, controller: LearnerController) -> RollbackEvent | None:
        """Check all rollback conditions and freeze the controller if any fires.

        Returns the :class:`RollbackEvent` if a rollback was triggered, else None.
        """
        if controller.is_frozen():
            return None

        # Trigger 2: envelope breaker (always auto-freezes; cannot be disabled).
        if self._breaker_flag:
            return self._freeze(controller, "envelope_breaker", "envelope breaker fired")

        # Trigger 4: unsafe regime / reconciliation failure.
        if self._regime_flag:
            return self._freeze(controller, "unsafe_regime", "R7/R8 or reconciliation failure")

        # Trigger 3: live-vs-shadow divergence.
        if self._divergence_flag:
            return self._freeze(
                controller,
                "divergence",
                f"live-vs-shadow divergence > {self.max_divergence}",
            )

        # Trigger 1: underperformance vs own projection, evaluated over the last
        # ``rollback_window`` REALIZED decisions. (Audit M32: filtering the last
        # N decisions down to realized ones required N consecutive realized
        # outcomes — a single pending outcome silently disabled the trigger.)
        realized = [d for d in self._decisions if d.realized_outcome is not None]
        window = realized[-self.rollback_window :]
        if len(window) >= self.rollback_window:
            mean_realized = sum(d.realized_outcome for d in window) / len(window)  # type: ignore[misc]
            mean_projected = sum(d.projected_outcome for d in window) / len(window)
            shortfall = mean_projected - mean_realized
            if shortfall >= self.rollback_margin:
                return self._freeze(
                    controller,
                    "underperformance",
                    f"shortfall {shortfall:.4f} >= margin {self.rollback_margin}",
                )

        return None

    def revert(
        self, controller: LearnerController, *, trigger: str, detail: str
    ) -> RollbackEvent:
        """Atomic revert to the frozen fallback (Section 21.7; audit H17).

        1. Freeze the controller: ``controller.freeze`` loads the last-approved
           snapshot into the fallback slot (via versioning) and the fallback
           runs in SHADOW mode only — no learner action is ever applied again
           until a manual review. If no fallback snapshot exists the controller
           is learner-disabled and this event/alert says so honestly
           (``fallback_active=False``), never claiming a fallback is active.
        2. Cancel pending learner-influenced **new** orders only — never an open
           position's exchange-side stop (via the injected ``cancel_orders`` hook).
        3. Emit the ``learner_rollback`` alert.
        4. Write the event to ``learner_log``.
        Recovery FROZEN → LIVE_BOUNDED is manual only.
        """
        controller.freeze(reason=detail)
        event = RollbackEvent(
            trigger=trigger,
            detail=detail,
            controller_frozen=controller.is_frozen(),
            fallback_active=controller.fallback_active,
        )
        if self.cancel_orders is not None:
            event.orders_cancelled = int(self.cancel_orders())
        if self.alert_sink is not None:
            if event.fallback_active:
                recommended = (
                    "learner FROZEN; approved frozen-fallback policy active in SHADOW mode "
                    f"(never applied); recovery is manual after review — {detail}"
                )
            else:
                recommended = (
                    "learner FROZEN and DISABLED — no frozen-fallback snapshot available; "
                    f"restore a fallback snapshot, then recover manually after review — {detail}"
                )
            self.alert_sink.send(
                Alert(
                    title=f"learner_rollback: {trigger}",
                    severity=AlertSeverity.CRITICAL,
                    component="adaptation.rollback",
                    environment=self.environment,
                    recommended_action=recommended,
                )
            )
        if self.log_writer is not None:
            self.log_writer(event)
        self._events.append(event)
        return event

    # Internal triggers route through the full atomic revert path.
    def _freeze(self, controller: LearnerController, trigger: str, detail: str) -> RollbackEvent:
        return self.revert(controller, trigger=trigger, detail=detail)

    def events(self) -> list[RollbackEvent]:
        return list(self._events)


def write_rollback_log(
    event: RollbackEvent,
    *,
    learner_id: str = "learner",
    learner_version: str = "learner_0001",
) -> None:
    """Default ``log_writer``: persist a rollback event to ``learner_logs`` (applied=False)."""
    from src.db.base import session_scope
    from src.db.models import LearnerLog

    with session_scope() as session:
        session.add(
            LearnerLog(
                learner_id=learner_id,
                learner_version=learner_version,
                mode="FROZEN",
                applied=False,
                rollback_event=f"{event.trigger}: {event.detail}",
            )
        )
