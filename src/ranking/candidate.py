"""The trade :class:`Candidate` — the unit the decision pipeline carries.

A candidate is the deterministic output of a strategy (Section 5/12) wrapped with
the decision-time context the rest of the pipeline needs:

    strategy → Candidate → ranking (setup quality) → risk approval → execution

Strategies generate candidates, **not** orders (Section 5). A candidate records
which symbol/strategy/regime fired, the explicit initial stop (which anchors
sizing, Section 17), the decision-time features (reproducible setup scoring,
Section 15) and the execution context (spread/slippage/liquidity). It is frozen:
ranking and risk read it but never mutate it.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Candidate:
    """One deterministic trade candidate at one decision time (frozen)."""

    symbol: str
    strategy: str
    strategy_version: str
    side: int  # +1 long / -1 short
    entry_price: float
    stop_frac: float  # |entry − stop| / entry; anchors sizing (Section 17)
    tp_frac: float  # take-profit distance / entry; large ⇒ "no fixed TP" (momentum)
    regime: str
    session: int

    # Decision-time features (reproducible setup scoring; Section 10 Parity Rule).
    features: dict[str, float] = field(default_factory=dict)

    # Strategy conviction inputs (normalised to [0, 1]) — the strategy is the best
    # judge of these two; the scorer derives every other component itself.
    signal_strength: float = 0.0  # normalised |signal| (Section 15 signal strength)
    confirmation: float = 0.0  # cross-signal confirmation strength (Section 15)

    # Expected move (fraction of price) BEFORE costs — the raw edge the strategy
    # claims; net-of-cost EV is computed by the scorer using verified fees.
    expected_edge_frac: float = 0.0

    # Execution context at decision time (Section 18 hard blockers).
    spread_bps: float = 0.0
    slippage_est: float = 0.0  # estimated fill slippage as a fraction of price
    latency_ms: float = 0.0

    # State flags consulted as Section 15 hard blockers.
    data_fresh: bool = True
    metadata_verified: bool = True
    symbol_tradable: bool = True  # status == trading, in active universe
    strategy_enabled: bool = True
    config_live_approved: bool = True

    decision_ts: int = 0

    # Execution geometry carried from the strategy Signal so the LIVE/paper path reproduces the
    # backtest (Parity Rule, Section 10): maker (passive-limit) entry, the trailing-stop offset,
    # the time-stop horizon, and the per-strategy size scale. Defaults reproduce the legacy taker /
    # no-trail / no-time-stop / full-size behaviour, so candidates built without them are unchanged.
    maker: bool = False  # entry posts a passive limit instead of crossing the spread
    limit_offset_frac: float = 0.0  # passive distance inside the reference for the maker limit
    trail_frac: float = 0.0  # trailing-stop offset (fraction of price); 0 = no trailing leg
    hold_bars: int = 0  # active time-stop horizon in bars (0 = no time-stop)
    risk_scale: float = 1.0  # per-strategy size scale (≤1) applied to risk_pct by the risk manager

    @property
    def stop_price(self) -> float:
        return self.entry_price * (1.0 - self.side * self.stop_frac)

    @property
    def tp_price(self) -> float:
        return self.entry_price * (1.0 + self.side * self.tp_frac)
