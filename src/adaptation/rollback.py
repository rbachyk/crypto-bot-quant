"""Learner circuit breaker and revert-to-fallback (AGENTS.md Section 21.7).

The :class:`RollbackGuard` monitors rolling performance and triggers a freeze
when any of the five rollback conditions fire:

  1. Realized performance underperforms policy's own shadow projection by
     ≥ ``rollback_margin`` over the last ``rollback_window`` decisions.
  2. Any envelope breaker fires (daily loss, drawdown, heat, beta).
  3. Live-vs-shadow decision divergence exceeds ``max_divergence``.
  4. R8/R7 regime or reconciliation failure (set_regime_flag).
  5. Manual learner kill switch (freeze() call).

On rollback: ``controller.freeze()`` is called; ``applied=False`` on all
subsequent decisions; a ``rollback_event`` is written to the learner log.
Recovery from FROZEN → LIVE_BOUNDED is MANUAL only (Section 21.7).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from src.adaptation.controller import LearnerController
from src.adaptation.scorer import ShadowDecision


@dataclass
class RollbackEvent:
    """Persisted when a rollback trigger fires."""

    ts: datetime = field(default_factory=lambda: datetime.now(UTC))
    trigger: str = ""
    detail: str = ""
    controller_frozen: bool = True


@dataclass
class RollbackGuard:
    """Monitors rolling performance and triggers automatic freeze when warranted.

    Instantiate one per :class:`~src.adaptation.controller.LearnerController`
    and call :meth:`check` after every decision+outcome pair.
    """

    rollback_window: int = 20
    rollback_margin: float = 0.05
    max_divergence: float = 0.20
    auto_freeze_on_breaker: bool = True  # IMMUTABLE: cannot be set false

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

        # Trigger 1: underperformance vs own projection.
        window = [
            d
            for d in self._decisions[-self.rollback_window :]
            if d.realized_outcome is not None
        ]
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

    def _freeze(
        self, controller: LearnerController, trigger: str, detail: str
    ) -> RollbackEvent:
        event = RollbackEvent(trigger=trigger, detail=detail)
        self._events.append(event)
        controller.freeze(reason=detail)
        return event

    def events(self) -> list[RollbackEvent]:
        return list(self._events)
