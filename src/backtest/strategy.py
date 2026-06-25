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

    ``hold_bars`` overrides the engine's default time-stop for this signal so each
    family can carry its own exit geometry (Section 12: momentum trades hold for
    the tail with no fixed TP; mean-reversion exits quickly). ``None`` falls back
    to the reference time-stop. A very large ``tp_frac`` encodes "no fixed TP"
    (momentum), where the time-stop / trailing logic is the real exit.

    ``trail_frac`` (>0) enables a trailing stop at that fraction of price behind the
    best favorable excursion — so a momentum winner RUNS while the move continues and
    exits when it reverses, instead of being cut at a fixed time-stop (which made
    avgWin ≈ avgLoss). 0 disables trailing (fixed-stop behavior).

    ``maker`` switches entry execution from a taker market fill (cross the spread,
    pay taker fee + adverse slippage) to a PASSIVE limit order: post a limit
    ``limit_offset_frac`` inside the fill-bar open and fill ONLY if the bar trades
    through it (else the order is cancelled and no position opens — fewer trades, by
    design). A maker fill pays the maker fee with zero slippage. The take-profit of a
    maker position is likewise treated as a resting limit (maker); risk exits (stop /
    trailing / time-stop) stay taker — you must cross the spread to get out of a loser.
    ``limit_offset_frac`` is the passive distance (fraction of price) the entry limit
    sits inside the reference price; 0 posts at the open (optimistic — a real maker
    fill should require price to come to you, so set this > 0).
    """

    side: int
    stop_frac: float
    tp_frac: float
    reason: str = ""
    hold_bars: int | None = None
    trail_frac: float = 0.0
    maker: bool = False
    limit_offset_frac: float = 0.0


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


@runtime_checkable
class PortfolioStrategy(Protocol):
    """A cross-asset strategy that sees peer symbols' rows at the SAME decision time.

    Cross-asset families (Section 12 A — lead-lag, G — cross-sectional relative
    strength) cannot decide from one symbol in isolation; they need the universe's
    state at decision time. The engine still preserves causality: ``row`` and every
    entry in ``peers`` are decision-time feature rows for the SAME ``decision_ts``
    (the close of the previous bar), so no peer datum from the fill bar or later is
    ever visible. ``peers`` excludes the evaluated symbol itself.
    """

    @property
    def name(self) -> str:
        """Stable strategy identifier (read-only)."""

    @property
    def strategy_version(self) -> str:
        """Versioned strategy parameters (read-only; Section 4)."""

    def evaluate_portfolio(self, symbol: str, row: dict, peers: dict[str, dict]) -> Signal | None:
        """Return a :class:`Signal` for ``symbol`` given its row + peer rows, or None."""


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
