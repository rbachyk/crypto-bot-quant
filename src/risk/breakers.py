"""Circuit breakers (AGENTS.md Section 17, per-symbol and portfolio).

Portfolio-level halts that stop the bot opening *new* entries. None of these
resize a trade — they BLOCK it (a halt is the capital-preserving action). The
daily-loss and max-drawdown breakers require a manual reset (Section 2.2: "Daily-
loss and max-drawdown circuit breakers … → halt to manual reset"); the risk
manager surfaces them but recovery is an operator action, never automatic.

The breakers read a :class:`BreakerInputs` snapshot (current account state) rather
than owning mutable state, so they are pure and deterministic — the same inputs
always yield the same verdict, which is what the RISK gate asserts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.killswitch import KillSwitch
from src.risk.config import RiskConfig


@dataclass(frozen=True, slots=True)
class BreakerInputs:
    """Account state the breakers judge (all capital-agnostic fractions/counts)."""

    equity: float
    peak_equity: float
    daily_pnl: float  # realized PnL so far today (currency; negative = loss)
    consecutive_losses: int = 0
    abnormal_slippage_active: bool = False
    reconciled: bool = True  # exchange state reconciles with the bot's view
    weekly_pnl: float = 0.0  # realized PnL so far this week (currency; negative = loss)
    cumulative_funding_paid: float = 0.0  # funding paid this period (currency, positive = paid)
    per_symbol_pnl: dict[str, float] = field(default_factory=dict)  # realized PnL per symbol


@dataclass(frozen=True, slots=True)
class BreakerVerdict:
    tripped: bool
    reason: str = ""


def _daily_loss_frac(inp: BreakerInputs) -> float:
    if inp.equity <= 0:
        return 0.0
    return -inp.daily_pnl / inp.equity if inp.daily_pnl < 0 else 0.0


def _drawdown_frac(inp: BreakerInputs) -> float:
    if inp.peak_equity <= 0:
        return 0.0
    return max(0.0, (inp.peak_equity - inp.equity) / inp.peak_equity)


class CircuitBreakers:
    """Deterministic breaker evaluation (Section 17)."""

    def __init__(self, cfg: RiskConfig, kill_switch: KillSwitch | None = None) -> None:
        self.cfg = cfg
        self.kill_switch = kill_switch

    def evaluate(self, inp: BreakerInputs) -> BreakerVerdict:
        """Return the first tripped breaker, in capital-preserving priority order."""
        # 1) Manual kill switch (Section 2.2) — highest priority halt.
        if self.kill_switch is not None and self.kill_switch.engaged():
            return BreakerVerdict(True, "kill_switch_engaged")

        # 2) Reconciliation failure (Section 17: mismatch → halt + alert).
        if self.cfg.breakers.require_reconciled and not inp.reconciled:
            return BreakerVerdict(True, "reconciliation_mismatch")

        # 3) Daily-loss limit (portfolio) → halt to manual reset.
        dl = _daily_loss_frac(inp)
        if dl >= self.cfg.envelope.daily_loss_limit:
            return BreakerVerdict(
                True, f"daily_loss_limit({dl:.4f}>={self.cfg.envelope.daily_loss_limit})"
            )

        # 4) Max-drawdown limit (portfolio) → halt to manual reset.
        dd = _drawdown_frac(inp)
        if dd >= self.cfg.envelope.max_drawdown_limit:
            return BreakerVerdict(
                True, f"max_drawdown_limit({dd:.4f}>={self.cfg.envelope.max_drawdown_limit})"
            )

        # 5) N consecutive max-losses → cooldown.
        if inp.consecutive_losses >= self.cfg.breakers.consecutive_loss_limit:
            return BreakerVerdict(
                True,
                f"consecutive_loss_cooldown({inp.consecutive_losses}>="
                f"{self.cfg.breakers.consecutive_loss_limit})",
            )

        # 6) Weekly-loss limit (portfolio) → halt for the week (manual reset).
        wl = (-inp.weekly_pnl / inp.equity) if (inp.equity > 0 and inp.weekly_pnl < 0) else 0.0
        if wl >= self.cfg.breakers.weekly_loss_limit:
            return BreakerVerdict(
                True, f"weekly_loss_limit({wl:.4f}>={self.cfg.breakers.weekly_loss_limit})"
            )

        # 7) Funding circuit-breaker → halt when funding bleed exceeds the cap.
        fb = (inp.cumulative_funding_paid / inp.equity) if inp.equity > 0 else 0.0
        if fb >= self.cfg.breakers.funding_breaker_limit:
            return BreakerVerdict(
                True, f"funding_breaker({fb:.4f}>={self.cfg.breakers.funding_breaker_limit})"
            )

        # 8) Per-symbol loss breaker → halt only the offending symbol.
        if inp.equity > 0:
            for sym, pnl in inp.per_symbol_pnl.items():
                loss = (-pnl / inp.equity) if pnl < 0 else 0.0
                if loss >= self.cfg.breakers.per_symbol_loss_limit:
                    return BreakerVerdict(
                        True,
                        f"per_symbol_loss[{sym}]({loss:.4f}>="
                        f"{self.cfg.breakers.per_symbol_loss_limit})",
                    )

        # 9) Abnormal-slippage cooldown.
        if self.cfg.breakers.abnormal_slippage_cooldown and inp.abnormal_slippage_active:
            return BreakerVerdict(True, "abnormal_slippage_cooldown")

        return BreakerVerdict(False)

    # -- pre-trade hard blockers (Section 17 required checks) ------------- #
    def liquidation_distance_ok(
        self, entry_price: float, liquidation_price: float, side: int
    ) -> bool:
        """True iff the liquidation price is at least ``min_liquidation_distance`` away.

        A new entry is refused when its liquidation price sits closer than the configured
        fraction — the trade has too little room before forced liquidation (Section 17)."""
        if entry_price <= 0 or liquidation_price <= 0:
            return False
        # For a long, liquidation is below entry; for a short, above. Either way use the gap.
        dist = abs(entry_price - liquidation_price) / entry_price
        return dist >= self.cfg.breakers.min_liquidation_distance

    def margin_available(self, required_margin: float, free_margin: float, equity: float) -> bool:
        """True iff posting ``required_margin`` leaves at least ``min_free_margin_frac`` free."""
        if equity <= 0 or required_margin < 0:
            return False
        remaining = free_margin - required_margin
        return remaining >= self.cfg.breakers.min_free_margin_frac * equity
