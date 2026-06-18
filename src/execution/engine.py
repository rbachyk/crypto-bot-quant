"""Execution Engine (AGENTS.md Section 18) — places orders only after risk approval.

The execution engine is the only component that places orders, and only for a
candidate the risk manager has already approved (Section 5). Before it acts it
**revalidates** the signal (Section 18 "revalidate signal before execution"):
kill switch clear, data fresh, spread not toxic, slippage/latency within caps. It
then builds the bracket and places it **atomically** on the venue so the position
always carries an exchange-resident stop (Section 2.2). Every fill is recorded with
the Section 18 execution-quality fields.

It never sizes or re-prices — that is the risk manager's job; execution is cosmetic
to risk (Section 21.6). It also never touches foreign orders (Section 7).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.exchange.metadata import MetadataConfig
from src.execution.config import ExecutionPolicyConfig
from src.execution.order import OrderBuilder, OrderPlan
from src.execution.ownership import OwnershipPolicy
from src.execution.venue import Fill, Venue, VenuePosition
from src.killswitch import KillSwitch
from src.ranking.candidate import Candidate
from src.risk.manager import RiskDecision


@dataclass(slots=True)
class ExecutionResult:
    placed: bool
    reason: str = ""
    plan: OrderPlan | None = None
    fill: Fill | None = None
    position: VenuePosition | None = None
    fully_filled: bool = True
    remaining_qty: float = 0.0

    def to_dict(self) -> dict:
        return {
            "placed": self.placed,
            "reason": self.reason,
            "plan": self.plan.to_dict() if self.plan else None,
            "fill": self.fill.to_dict() if self.fill else None,
            "fully_filled": self.fully_filled,
            "remaining_qty": self.remaining_qty,
        }


class ExecutionEngine:
    """Revalidate → build → place atomic bracket → measure fill (Section 18)."""

    def __init__(
        self,
        cfg: ExecutionPolicyConfig,
        meta: MetadataConfig,
        ownership: OwnershipPolicy,
        venue: Venue,
        kill_switch: KillSwitch | None = None,
    ) -> None:
        self.cfg = cfg
        self.meta = meta
        self.ownership = ownership
        self.venue = venue
        self.kill_switch = kill_switch
        self.builder = OrderBuilder(cfg, ownership)

    def revalidate(self, candidate: Candidate) -> str | None:
        """Return a reason to abort, or None if the signal is still executable."""
        if self.kill_switch is not None and self.kill_switch.engaged():
            return "kill_switch_engaged"
        if not candidate.data_fresh:
            return "stale_data"
        if candidate.spread_bps > self.cfg.max_spread_bps:
            return f"toxic_spread({candidate.spread_bps:.1f}bps)"
        if candidate.slippage_est > self.cfg.max_slippage_frac:
            return f"slippage_exceeds_cap({candidate.slippage_est:.4f})"
        if candidate.latency_ms > self.cfg.max_latency_ms:
            return f"abnormal_latency({candidate.latency_ms:.0f}ms)"
        return None

    def execute(
        self,
        candidate: Candidate,
        decision: RiskDecision,
        *,
        realized_slippage_frac: float | None = None,
        signal_age_ms: float = 0.0,
        fill_ratio: float = 1.0,
        entry_style: str | None = None,
    ) -> ExecutionResult:
        if not decision.approved:
            return ExecutionResult(False, reason=f"risk_rejected:{decision.action}")

        abort = self.revalidate(candidate)
        if abort is not None:
            return ExecutionResult(False, reason=abort)

        spec = self.meta.spec(candidate.symbol)
        if spec is None:
            return ExecutionResult(False, reason="missing_metadata")

        build = self.builder.build(candidate, decision, spec, entry_style=entry_style)
        if not build.ok or build.plan is None:
            return ExecutionResult(False, reason=f"order_build_failed:{build.reason}")
        plan = build.plan

        # Taker entries realise the estimated slippage; maker entries rest at price.
        slip = candidate.slippage_est if realized_slippage_frac is None else realized_slippage_frac
        try:
            bracket = self.venue.place_bracket(
                plan,
                ref_price=candidate.entry_price,
                realized_slippage_frac=slip,
                latency_ms=self.cfg.simulated_latency_ms,
                spread_bps=candidate.spread_bps,
                signal_age_ms=signal_age_ms,
                fill_ratio=fill_ratio,
            )
        except PermissionError as exc:
            # A live venue refuses unauthorised real-money orders (M8 guard); treat the
            # refusal as a graceful non-placement, never a crash.
            return ExecutionResult(False, reason=f"live_order_refused:{exc}")

        # Section 2.2 invariant: the position must carry exchange-side protection.
        if not bracket.position.has_exchange_side_stop():
            return ExecutionResult(False, reason="position_without_exchange_side_stop")

        return ExecutionResult(
            placed=True,
            plan=plan,
            fill=bracket.fill,
            position=bracket.position,
            fully_filled=bracket.fully_filled,
            remaining_qty=bracket.remaining_qty,
        )
