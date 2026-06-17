"""Strategy interface + the Phase 4 reference strategy (AGENTS.md Section 19).

A :class:`Strategy` consumes a single decision-time feature row (computed by the
ONE feature pipeline — the Parity Rule, Section 10) and either emits a
:class:`Signal` or declines. It never sees future bars; the engine fills any
signal at the NEXT bar's open, so a strategy cannot act on its own bar's close.

Phase 4 ships the *engine*, not a validated trading strategy — real research
candidates (Section 12) arrive in Phase 5. :class:`ReferenceMomentumStrategy` is
a deterministic, fully causal self-test rule used only to drive the engine,
walk-forward and stress machinery and the BT/WF/FEE/SLIP gates. Its hypothesis is
trivial and declared: persistence of short-horizon momentum (``ret_short``). On
the reference series with a planted causal edge it is profitable net of costs; on
a no-structure series it is not — the engine-level look-ahead/leakage guard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from src.backtest.config import ReferenceStrategyConfig


@dataclass(frozen=True, slots=True)
class Signal:
    """A strategy's intent for one symbol at one decision time.

    ``side`` is +1 (long) or -1 (short). ``stop_frac`` / ``tp_frac`` are the stop
    and take-profit distances as a fraction of the entry price; ``stop_frac`` is
    the per-trade risk that anchors position sizing (Section 17).
    """

    side: int
    stop_frac: float
    tp_frac: float
    reason: str = ""


@runtime_checkable
class Strategy(Protocol):
    @property
    def name(self) -> str:
        """Stable strategy identifier (read-only)."""

    @property
    def strategy_version(self) -> str:
        """Versioned strategy parameters (read-only; Section 4)."""

    def evaluate(self, row: dict) -> Signal | None:
        """Return a :class:`Signal` for this decision-time feature row, or None."""


@dataclass(slots=True)
class ReferenceMomentumStrategy:
    """Deterministic causal momentum self-test strategy (Phase 4 engine fixture)."""

    cfg: ReferenceStrategyConfig

    @property
    def name(self) -> str:
        return self.cfg.name

    @property
    def strategy_version(self) -> str:
        return self.cfg.strategy_version

    def evaluate(self, row: dict) -> Signal | None:
        momentum = float(row["ret_short"])
        if abs(momentum) < self.cfg.signal_threshold:
            return None
        side = 1 if momentum > 0 else -1
        if side > 0 and not self.cfg.allow_long:
            return None
        if side < 0 and not self.cfg.allow_short:
            return None

        atr_pct = float(row["atr_pct"])
        stop_frac = max(self.cfg.stop_atr_mult * atr_pct, self.cfg.min_stop_frac)
        tp_frac = max(self.cfg.tp_atr_mult * atr_pct, stop_frac * 1.5)
        return Signal(
            side=side,
            stop_frac=stop_frac,
            tp_frac=tp_frac,
            reason=f"momentum {momentum:+.5f} (|.|>={self.cfg.signal_threshold})",
        )
