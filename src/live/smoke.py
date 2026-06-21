"""Safe demo smoke test (AGENTS.md Section 35).

The minimal, bounded end-to-end exercise of the real demo/testnet order path. It is the
LAST manual step before letting the bot run a demo session, and it is safe by construction:

1. Pre-flight: run the :class:`DemoReadinessGuard`. If the verdict is not PASS, place NOTHING
   and return the report — an unverified spec / dirty book / engaged kill switch aborts here.
2. Bounded placement: drive the live loop for at most ``max_ticks`` (default 1) tick — at most
   one order — with a minimal-notional sizing (small equity → the order builder clamps to the
   symbol's min-notional) and the mandatory exchange-resident SL (+ TP/trailing) the builder
   always attaches.
3. Immediate reconciliation after placement (the loop reconciles every tick; we also re-read
   the venue book here).
4. Optional cleanup: cancel our resting orders and close our positions so the demo book is left
   flat (``cleanup=True`` by default).

It never enables live trading and never runs unbounded. With nothing injected it builds the
real (testnet/demo) venue, so it still requires demo credentials + a PASS readiness verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.config import Settings, get_settings
from src.live.demo_guard import DemoReadinessGuard, DemoReadinessReport

# A deliberately small notional account for the smoke order — the order builder sizes from this
# and clamps up to the symbol's min-notional, so the smoke trade is the smallest legal order.
_SMOKE_EQUITY = 200.0


@dataclass(frozen=True, slots=True)
class DemoSmokeResult:
    placed: int
    readiness: DemoReadinessReport
    aborted_reason: str = ""
    reconciliation: dict = field(default_factory=dict)
    cleaned_up: int = 0
    halted: bool = False

    @property
    def ok(self) -> bool:
        """A smoke test 'passes' if readiness was PASS and we did not halt on a foreign book."""
        return self.readiness.verdict == "PASS" and not self.halted and not self.aborted_reason

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "placed": self.placed,
            "aborted_reason": self.aborted_reason,
            "readiness": self.readiness.to_dict(),
            "reconciliation": self.reconciliation,
            "cleaned_up": self.cleaned_up,
            "halted": self.halted,
        }

    def report(self) -> str:
        lines = [self.readiness.report(), ""]
        if self.aborted_reason:
            lines.append(f"Smoke test ABORTED before placement: {self.aborted_reason}")
        else:
            lines.append(
                f"Smoke test placed {self.placed} order(s); halted={self.halted}; "
                f"cleaned_up={self.cleaned_up}"
            )
        return "\n".join(lines)


def run_demo_smoke(
    settings: Settings | None = None,
    *,
    venue: Any | None = None,
    feed: Any | None = None,
    cleanup: bool = True,
    max_ticks: int = 1,
) -> DemoSmokeResult:
    """Run the bounded demo smoke test. See module docstring for the safety contract."""
    from src.exchange.metadata import load_metadata_for
    from src.execution.live_venue import get_venue
    from src.killswitch import KillSwitch
    from src.live.loop import LiveLoop

    settings = settings or get_settings()
    if settings.exchange_env == "live":
        # Hard refusal: the smoke test is a demo/testnet tool, never a live-money path.
        report = DemoReadinessGuard(settings).evaluate()
        return DemoSmokeResult(
            placed=0, readiness=report, aborted_reason="refusing to smoke-test a live environment"
        )

    meta = load_metadata_for(settings.exchange_id)
    mode = "testnet"  # demo + testnet both use the real ccxt venue (virtual funds)
    kill_switch = KillSwitch(settings)
    if venue is None:
        venue = get_venue(meta, settings, live=True)

    # 1) Pre-flight readiness — abort (place nothing) unless PASS.
    readiness = DemoReadinessGuard(settings, kill_switch=kill_switch, venue=venue).evaluate()
    if readiness.verdict != "PASS":
        return DemoSmokeResult(
            placed=0,
            readiness=readiness,
            aborted_reason=f"readiness verdict {readiness.verdict} (not PASS)",
        )

    # 2) Bounded placement: at most one tick / one order, minimal notional.
    loop = LiveLoop(mode=mode, settings=settings, meta=meta, venue=venue, kill_switch=kill_switch)
    if feed is None:
        feed = _smoke_feed(settings, meta)
    result = loop.run(feed, session_name="smoke", max_ticks=max_ticks)
    placed = result.executed

    # 3) Immediate reconciliation snapshot after placement.
    reconciliation = {
        "startup": result.startup_recon.to_dict() if result.startup_recon else {},
        "open_orders": list(getattr(venue, "open_orders", {})),
        "open_positions": list(getattr(venue, "positions", {})),
    }

    # 4) Optional cleanup — leave the demo book flat.
    cleaned = 0
    if cleanup and placed and hasattr(venue, "emergency_close_all"):
        cleaned = venue.emergency_close_all(confirm=True)

    return DemoSmokeResult(
        placed=placed,
        readiness=readiness,
        reconciliation=reconciliation,
        cleaned_up=cleaned,
        halted=result.halted,
    )


def _smoke_feed(settings: Settings, meta: Any) -> Any:
    """A one-tick replay feed carrying a single minimal-notional candidate with a fixed SL/TP.

    Uses the first verified symbol; the order builder attaches the mandatory exchange-resident
    stop and take-profit, and clamps the size up to the symbol's min-notional."""
    from src.live.loop import ReplayFeed
    from src.paper.engine import PaperCandidateInput
    from src.ranking import Candidate

    symbol = meta.symbols()[0]
    spec = meta.spec(symbol)
    # A reference price near the tick grid; the venue fills at the returned average anyway.
    ref = float(spec.fields.get("min_notional", 5.0)) / float(spec.fields.get("min_order_size", 0.001))
    cand = Candidate(
        symbol=symbol,
        strategy="demo_smoke",
        strategy_version="smoke",
        side=1,
        entry_price=ref,
        stop_frac=0.01,
        tp_frac=0.02,  # finite TP → exchange-resident take-profit attached
        regime="low_vol_up",
        session=1,
        features={"atr_pct": 0.003},
        signal_strength=0.85,
        confirmation=0.75,
        expected_edge_frac=0.012,
        spread_bps=3.0,
        slippage_est=0.0005,
        latency_ms=5.0,
        data_fresh=True,
        metadata_verified=True,
        symbol_tradable=True,
        strategy_enabled=True,
        config_live_approved=True,
        decision_ts=1_700_000_000_000,
    )
    return ReplayFeed([PaperCandidateInput(candidate=cand, equity=_SMOKE_EQUITY, exit_move_frac=0.0)])


__all__ = ["run_demo_smoke", "DemoSmokeResult"]
