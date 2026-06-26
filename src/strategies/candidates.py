"""Phase 5 deterministic research candidates (AGENTS.md Section 12 families A/B/G).

Each candidate is a deterministic, fully causal rule that consumes decision-time
feature rows from the ONE feature pipeline (the Parity Rule, Section 10) and emits
a :class:`~src.backtest.strategy.Signal` (side + explicit initial SL for sizing +
exit geometry) or declines. Strategies generate candidates only — they never place
orders (Section 5). Each carries its full :class:`StrategyHypothesis` (Section 13).

Exit geometry matches the edge profile (Section 12):
* mean-reversion (B): near fixed TP, wider SL, asymmetric (high win-rate / low R);
* momentum (A, G): explicit initial SL for sizing, time-stop, no fixed TP
  (``tp_frac`` set unreachable so the tail is the edge).

Long/short sides are config-gated (``allow_long`` / ``allow_short``); the research
harness disables a structurally-losing side before promotion.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.backtest.strategy import ExitDecision, PositionView, Signal
from src.regime.detector import NO_TRADE_REGIMES, detect_regime
from src.strategies.base import StrategyHypothesis
from src.strategies.config import CandidateConfig, StrategyParams


@dataclass(slots=True)
class _BaseCandidate:
    candidate: CandidateConfig
    strategy_version: str
    params: StrategyParams

    @property
    def name(self) -> str:
        return self.candidate.id

    @property
    def risk_scale(self) -> float:
        """Per-strategy position-size scale (≤ 1.0) read by the engine's sizing (Section 17)."""
        return self.params.risk_scale

    @property
    def hypothesis(self) -> StrategyHypothesis:
        """Every candidate declares its full Section 13 hypothesis before it trades."""
        raise NotImplementedError

    def evaluate(self, row: dict) -> Signal | None:
        """Per-symbol decision. Cross-asset families override evaluate_portfolio instead
        (the engine dispatches on hasattr(evaluate_portfolio)); calling this on one is a bug.
        Declared here so a built candidate satisfies the engine's Strategy protocol."""
        raise NotImplementedError

    def _regime_ok(self, row: dict) -> bool:
        """Regime gate (Section 11). Off by default (legacy: trade every bar). ``regimes`` is an
        allow-list (trade ONLY those); ``block_no_trade_regimes`` excludes the live safety regimes.
        Computed from decision-time features already in the row (spread/data handled separately by
        the engine's execution blockers, so spread_bps defaults here)."""
        p = self.params
        if not p.regimes and not p.block_no_trade_regimes:
            return True
        regime = detect_regime(row)
        if p.regimes and regime not in p.regimes:
            return False
        return not (p.block_no_trade_regimes and regime in NO_TRADE_REGIMES)

    def _sided(self, side: int, reason: str, row: dict) -> Signal | None:
        if side > 0 and not self.params.allow_long:
            return None
        if side < 0 and not self.params.allow_short:
            return None
        if not self._regime_ok(row):
            return None  # entry condition fired but the regime is not one this candidate trades
        stop_frac, tp_frac = self._exit_geometry(row)
        atr = float(row.get("atr_pct", 0.0) or 0.0)
        trail_frac = self.params.atr_trail_mult * atr  # 0 when trailing is disabled
        # Maker entry: post the limit ``limit_offset_atr_mult × atr_pct`` inside the open so the
        # passive distance scales with realized volatility (like the stop/TP geometry). 0 when the
        # family fills as a taker market order.
        limit_offset_frac = self.params.limit_offset_atr_mult * atr
        return Signal(
            side=side,
            stop_frac=stop_frac,
            tp_frac=tp_frac,
            hold_bars=self.params.hold_bars,
            trail_frac=trail_frac,
            reason=reason,
            maker=self.params.maker_entry,
            limit_offset_frac=limit_offset_frac,
        )

    def _exit_geometry(self, row: dict) -> tuple[float, float]:
        """Stop/TP fractions for this entry. ``stop_frac``/``tp_frac`` are FLOORS; when an ATR
        multiplier is configured the geometry scales with realized volatility (``k × atr_pct``),
        so the SAME config adapts across decision timeframes instead of being a fixed % of price.
        A fixed 1.2% stop is several 5m bars but only ~1 1h bar, so coarse-timeframe runs stop out
        on ordinary noise; ``max(floor, k × atr)`` widens only on volatile bars / coarser grids and
        never yields a degenerate sub-floor stop (which would explode position sizing)."""
        atr = float(row.get("atr_pct", 0.0) or 0.0)
        stop_frac = max(self.params.stop_frac, self.params.atr_stop_mult * atr)
        if self.params.tp_r_mult > 0.0:
            # Reachable R-multiple target: TP sits tp_r_mult × the effective stop distance, so it
            # is always exactly that many R regardless of which stop term binds (floor vs ATR).
            # Momentum uses it to cap give-back; the trailing stop still exits trades peaking below.
            tp_frac = self.params.tp_r_mult * stop_frac
        else:
            tp_frac = max(self.params.tp_frac, self.params.atr_tp_mult * atr)
        return stop_frac, tp_frac


# --------------------------------------------------------------------------- #
# Family B — Perpetual Premium / Basis Mean Reversion (per-symbol)             #
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class BasisReversionStrategy(_BaseCandidate):
    """Short the rich perp / long the cheap perp when the premium is extreme.

    Reads the decision-time ``premium`` feature (mark-vs-index). An extreme
    positive premium (perp rich) is faded short; an extreme negative premium
    (perp cheap) is faded long. Funding/fees/slippage are charged by the engine,
    so the reversion is only taken when it is large enough to clear them.
    """

    @property
    def hypothesis(self) -> StrategyHypothesis:
        return StrategyHypothesis(
            family="B",
            name=self.candidate.id,
            hypothesis=(
                "Extreme deviations of the perpetual price from index/fair value "
                "(observable as the mark-vs-index premium) mean-revert under stable "
                "liquidity, net of fees, slippage and funding."
            ),
            market_condition="range / stable-liquidity; not during data-unsafe or toxic execution",
            edge_source="structural basis mean-reversion (perp ↔ index convergence)",
            data_requirements=("perp OHLCV", "mark price", "index price", "funding rate"),
            entry="premium > +threshold ⇒ short; premium < -threshold ⇒ long",
            exit="near fixed TP (reversion) or wider SL or time-stop (mean-reversion geometry)",
            invalidation="premium widens through the SL (regime break / one-way repricing)",
            risk_assumptions="explicit initial SL (stop_frac) anchors per-trade sizing",
            cost_assumptions="reversion must exceed 2×taker fee + slippage + funding over the hold",
            failure_modes=(
                "trend repricing (premium keeps widening)",
                "funding flips the carry against the position",
                "thin liquidity inflates slippage past the captured reversion",
            ),
            validation_tests=(
                "walk-forward",
                "fee ×2 stress",
                "slippage +50% stress",
                "noise control",
            ),
            promotion_criteria="WF + FEE + SLIP PASS on the surviving side(s); else shelve",
            exit_profile="mean_reversion",
            notes="Funding may support or block a setup; it never creates one (Section 12.B/C).",
        )

    def evaluate(self, row: dict) -> Signal | None:
        premium = float(row.get("premium", 0.0))
        threshold = self.params.extra["premium_threshold"]
        mag = abs(premium)
        if mag < threshold:
            return None
        # BAND entry: fade a dislocation only when it is in the reversion zone [threshold, cap].
        # An EXTREME premium is usually a one-way repricing (the perp is being re-rated, not
        # temporarily dislocated) — the entry-quality study showed the edge fades to ~+0.01R for
        # |premium| above ~0.0023 vs ~+0.078R in the moderate band. cap <= 0 disables the cap.
        cap = self.params.extra.get("premium_cap", 0.0)
        if cap > 0 and mag > cap:
            return None
        # NOTE: a funding-confirmation filter (fade a cheap perp only when funding is negative — a
        # crowded short / squeeze candidate) was TESTED and REJECTED. It lifted every in-sample
        # metric (deflated 0.555→0.61) but COLLAPSED the locked hold-out (PF 1.07→0.82): the +0.22R
        # aligned-long edge is a training-period artifact, not OOS. Classic overfit — not adopted.
        if premium >= threshold:
            return self._sided(-1, f"premium {premium:+.5f} >= {threshold} ⇒ fade short", row)
        return self._sided(+1, f"premium {premium:+.5f} <= -{threshold} ⇒ fade long", row)

    def manage(self, row: dict, position: PositionView) -> ExitDecision | None:
        """Exit the faded position once the premium it faded has REVERTED — the family's actual
        exit thesis, instead of waiting for a fixed ATR take-profit or the time-stop (which let a
        completed reversion sit on the book and give the edge back). The exit band is
        ``exit_premium_frac × premium_threshold``: a short (faded a rich perp, premium ≥ +thr)
        closes once the premium has fallen back to ``+exit_level``; a long (faded a cheap perp)
        closes once it has risen to ``−exit_level``. ``exit_premium_frac`` 0 ⇒ exit at the
        zero-cross. The exit is posted maker (passive limit, taker-fallback) at the configured
        offset, mirroring the entry."""
        exit_frac = self.params.extra.get("exit_premium_frac", 0.0)
        if exit_frac < 0:
            # Premium-reversion exit DISABLED (sentinel < 0). A controlled A/B on the 20-sym 4h lake
            # showed it is a net negative here: premium reverting ≠ price profit, so it booked small
            # losses (~-0.11R) on trades that had not yet reached their take-profit — holding to
            # TP/stop/time did better (exp +0.038→+0.044, all 5 folds positive). Kept as an opt-in
            # knob (the hook + mechanism are sound) but off for basis on this snapshot.
            return None
        premium = float(row.get("premium", 0.0))
        threshold = self.params.extra["premium_threshold"]
        exit_level = exit_frac * threshold
        reverted = (
            (position.side < 0 and premium <= exit_level)
            or (position.side > 0 and premium >= -exit_level)
        )
        if not reverted:
            return None
        atr = float(row.get("atr_pct", 0.0) or 0.0)
        offset = self.params.limit_offset_atr_mult * atr  # 0 ⇒ taker close (no maker entry)
        return ExitDecision(reason="premium_reverted", limit_offset_frac=offset)


# --------------------------------------------------------------------------- #
# Family A — Cross-Asset Lead-Lag (portfolio)                                  #
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class LeadLagStrategy(_BaseCandidate):
    """Trade a follower in the direction of the leader's last completed move.

    A statistically significant move in the dominant asset (the leader) leads a
    delayed move in related followers. At decision time (close of the previous
    bar) the leader's most recent completed return is known; if it exceeds the
    threshold, the follower is entered in the same direction to capture the lagged
    response. The leader itself is not traded by this candidate.
    """

    @property
    def hypothesis(self) -> StrategyHypothesis:
        return StrategyHypothesis(
            family="A",
            name=self.candidate.id,
            hypothesis=(
                "A significant move in a dominant asset (leader) leads a delayed, "
                "same-direction move in correlated followers within a short horizon."
            ),
            market_condition="impulse/trending leader; adequate follower liquidity",
            edge_source="cross-asset information lag (lead-lag response)",
            data_requirements=("leader OHLCV/returns", "follower OHLCV/returns"),
            entry="|leader last-bar return| > threshold ⇒ follower in the leader's direction",
            exit="time-stop after a few bars (capture the lagged burst); no fixed TP; initial SL",
            invalidation="follower fails to follow and hits the initial SL",
            risk_assumptions="explicit initial SL (stop_frac) anchors per-trade sizing",
            cost_assumptions="lagged response must exceed 2×taker fee + slippage over the hold",
            failure_modes=(
                "lead-lag decays / reverses (followers already repriced)",
                "leader move is noise, not signal",
                "follower-specific shock dominates the lagged response",
            ),
            validation_tests=(
                "walk-forward",
                "fee ×2 stress",
                "slippage +50% stress",
                "noise control",
            ),
            promotion_criteria="WF + FEE + SLIP PASS on the surviving side(s); else shelve",
            exit_profile="momentum",
            notes="Cross-asset family: decided from peer rows at the same decision time (causal).",
        )

    def evaluate_portfolio(self, symbol: str, row: dict, peers: dict[str, dict]) -> Signal | None:
        leader = str(self.candidate.fixture.values["leader"])
        if symbol == leader:
            return None  # the leader is the source, not a tradable follower here
        leader_row = peers.get(leader)
        if leader_row is None:
            return None
        leader_ret = float(leader_row.get("ret_1", 0.0))
        threshold = self.params.extra["leader_ret_threshold"]
        if abs(leader_ret) < threshold:
            return None
        side = 1 if leader_ret > 0 else -1
        return self._sided(side, f"leader {leader} ret_1={leader_ret:+.5f} ⇒ follow", row)


# --------------------------------------------------------------------------- #
# Family G — Cross-Sectional Relative Strength / Dispersion (portfolio)        #
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class CrossSectionalRSStrategy(_BaseCandidate):
    """Long the relative-strength leader / short the laggard across the universe.

    At each decision time, each symbol's short-horizon return is compared with the
    cross-sectional mean. A symbol sufficiently stronger than its peers is bought
    (its idiosyncratic strength is expected to persist); a symbol sufficiently
    weaker is shorted. Trading the dispersion is approximately market-neutral, so
    the persistent idiosyncratic spread — not the common factor — is the edge.
    """

    @property
    def hypothesis(self) -> StrategyHypothesis:
        return StrategyHypothesis(
            family="G",
            name=self.candidate.id,
            hypothesis=(
                "Idiosyncratic cross-sectional relative strength (return vs the "
                "cross-sectional mean) persists, so current leaders keep "
                "outperforming and laggards keep underperforming over a short horizon."
            ),
            market_condition="dispersed universe; market-wide impulse with idiosyncratic spread",
            edge_source="persistent cross-sectional relative strength (dispersion)",
            data_requirements=("returns across the universe",),
            entry="rel. strength vs cross-sectional mean > +thr ⇒ long; < -thr ⇒ short",
            exit="time-stop (relative-strength continuation); no fixed TP; initial SL",
            invalidation="relative strength mean-reverts and the position hits the initial SL",
            risk_assumptions="explicit initial SL (stop_frac) anchors per-trade sizing",
            cost_assumptions="dispersion spread must exceed 2×taker fee + slippage over the hold",
            failure_modes=(
                "relative strength reverses (mean-reversion regime)",
                "single common factor dominates (no dispersion to harvest)",
                "illiquid laggards inflate slippage on the short leg",
            ),
            validation_tests=(
                "walk-forward",
                "fee ×2 stress",
                "slippage +50% stress",
                "noise control",
            ),
            promotion_criteria="WF + FEE + SLIP PASS on the surviving side(s); else shelve",
            exit_profile="momentum",
            notes="Cross-asset: ranks the symbol against its peers at the same decision time.",
        )

    def evaluate_portfolio(self, symbol: str, row: dict, peers: dict[str, dict]) -> Signal | None:
        if not peers:
            return None
        own = float(row.get("ret_short", 0.0))
        rets = [own] + [float(p.get("ret_short", 0.0)) for p in peers.values()]
        mean = sum(rets) / len(rets)
        rel = own - mean
        threshold = self.params.extra["rs_threshold"]
        if rel >= threshold:
            return self._sided(+1, f"rel_strength {rel:+.5f} >= {threshold} ⇒ long leader", row)
        if rel <= -threshold:
            return self._sided(-1, f"rel_strength {rel:+.5f} <= -{threshold} ⇒ short laggard", row)
        return None


# --------------------------------------------------------------------------- #
# Family C — Funding-Dispersion Carry (portfolio, structural cash flow)         #
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class FundingCarryStrategy(_BaseCandidate):
    """Cross-sectional funding carry: LONG the perps PAID funding (most-negative funding), SHORT
    those CHARGED funding (most-positive), ranked by ``funding_z`` vs the cross-sectional mean.

    The edge is the periodic funding cash flow the crowded side pays — a STRUCTURAL carry, not a
    price forecast (the engine's funding model books it on every open position). Held over a carry
    horizon; an ATR stop bounds the idiosyncratic price risk. Diversifies the directional book.
    """

    @property
    def hypothesis(self) -> StrategyHypothesis:
        return StrategyHypothesis(
            family="C",
            name=self.candidate.id,
            hypothesis=(
                "Perpetual funding is a periodic cash flow paid by the crowded side; longing the "
                "most-negative-funding perps and shorting the most-positive collects that flow — a "
                "structural carry edge independent of price direction."
            ),
            market_condition="dispersed funding across the universe; stable liquidity",
            edge_source="funding cash flow (carry harvested from the crowded side)",
            data_requirements=("funding rate", "funding z-score across the universe"),
            entry="funding_z − cross-mean ≤ −thr ⇒ long (paid); ≥ +thr ⇒ short (paid)",
            exit="carry horizon (hold to collect funding) or ATR stop / time-stop",
            invalidation="funding normalizes or an adverse price move overwhelms the carry",
            risk_assumptions="ATR initial SL bounds idiosyncratic risk; sized small (risk_scale)",
            cost_assumptions="net funding must exceed fees + slippage + adverse price drift",
            failure_modes=(
                "a directional price move swamps the collected funding",
                "funding normalizes before enough is collected",
                "crowded-side tail risk realizes (the funding is compensation for it)",
            ),
            validation_tests=("walk-forward", "fee ×2 stress", "slippage +50% stress"),
            promotion_criteria="WF + FEE + SLIP PASS on the surviving side(s); else shelve",
            exit_profile="mean_reversion",
            notes="Carry, not prediction: the funding model books the cash flow (Section 12.C).",
        )

    def evaluate_portfolio(self, symbol: str, row: dict, peers: dict[str, dict]) -> Signal | None:
        if not peers:
            return None
        own = float(row.get("funding_z", 0.0))
        vals = [own] + [float(p.get("funding_z", 0.0)) for p in peers.values()]
        rel = own - sum(vals) / len(vals)  # funding vs the cross-sectional mean
        threshold = self.params.extra["funding_rank_threshold"]
        if rel <= -threshold:  # low/negative funding ⇒ being LONG is paid
            return self._sided(+1, f"funding_z rel {rel:+.3f} <= -{threshold} ⇒ long (paid)", row)
        if rel >= threshold:  # high positive funding ⇒ being SHORT is paid
            return self._sided(-1, f"funding_z rel {rel:+.3f} >= {threshold} ⇒ short (paid)", row)
        return None


# --------------------------------------------------------------------------- #
# Family D — Liquidation / OI-Flush Reversal (per-symbol, event-driven)         #
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class LiquidationReversalStrategy(_BaseCandidate):
    """Fade a forced-flow overshoot: an abnormal short-horizon move WITH an open-interest collapse
    (positions liquidated) AND a volatility spike ⇒ enter AGAINST the flush (down-flush ⇒ long the
    bounce). The gross move is large (liquidation overshoot), so it clears costs even taker;
    mean-reversion exit geometry (near TP, tight stop, short hold)."""

    @property
    def hypothesis(self) -> StrategyHypothesis:
        return StrategyHypothesis(
            family="D",
            name=self.candidate.id,
            hypothesis=(
                "After a liquidation cascade — abnormal displacement + open-interest collapse + "
                "volatility spike — price temporarily overshoots and reverts once forced flow "
                "exhausts."
            ),
            market_condition="liquidation / forced-flow overshoot; not toxic/unsafe execution",
            edge_source="forced-flow exhaustion (liquidation overshoot reversal)",
            data_requirements=("short-horizon return", "open-interest change", "realized vol"),
            entry="|ret_short|≥abnormal_move AND oi_change≤−oi_flush_frac AND rv_short≥vol_spike",
            exit="fast TP (the bounce) / tight stop / short time-stop (mean-reversion geometry)",
            invalidation="the move continues (genuine repricing, not an exhausted cascade)",
            risk_assumptions="tight ATR initial SL; reduced size; strict execution-safety gate",
            cost_assumptions="the overshoot reversal must exceed 2×taker fee + (elevated) slippage",
            failure_modes=(
                "the displacement is genuine repricing, not exhaustion (no reversion)",
                "thin post-cascade liquidity inflates slippage past the captured reversion",
                "a second cascade leg runs the position over",
            ),
            validation_tests=("walk-forward", "fee ×2 stress", "slippage +50% stress"),
            promotion_criteria="WF + FEE + SLIP PASS on the surviving side(s); else shelve",
            exit_profile="mean_reversion",
            notes="Research-grade event edge; trade after the spike, never into it (Section 12.D).",
        )

    def evaluate(self, row: dict) -> Signal | None:
        ret = float(row.get("ret_short", 0.0))
        oi_change = float(row.get("oi_change", 0.0))
        rv = float(row.get("rv_short", 0.0))
        move_thr = self.params.extra["abnormal_move"]
        oi_flush = self.params.extra["oi_flush_frac"]
        vol_thr = self.params.extra["vol_spike"]
        # Require the full flush signature: OI collapse (forced closes) + a volatility spike.
        if oi_change > -oi_flush or rv < vol_thr:
            return None
        if ret <= -move_thr:  # down-flush ⇒ fade long (expect the bounce)
            return self._sided(+1, f"down-flush ret={ret:+.4f} oi={oi_change:+.3f} fade long", row)
        if ret >= move_thr:  # up-flush ⇒ fade short
            return self._sided(-1, f"up-flush ret={ret:+.4f} oi={oi_change:+.3f} ⇒ fade short", row)
        return None


# --------------------------------------------------------------------------- #
# Factory                                                                      #
# --------------------------------------------------------------------------- #
_BY_FAMILY = {
    "A": LeadLagStrategy,
    "B": BasisReversionStrategy,
    "C": FundingCarryStrategy,
    "D": LiquidationReversalStrategy,
    "G": CrossSectionalRSStrategy,
}


def build_strategy(
    candidate: CandidateConfig,
    strategy_version: str,
    params: StrategyParams | None = None,
) -> _BaseCandidate:
    """Instantiate the candidate's strategy, optionally with side-overridden params."""
    cls = _BY_FAMILY.get(candidate.family)
    if cls is None:
        raise ValueError(f"no strategy implemented for family {candidate.family!r}")
    return cls(
        candidate=candidate,
        strategy_version=strategy_version,
        params=params or candidate.params,
    )


def is_portfolio_family(family: str) -> bool:
    return family in {"A", "C", "G"}  # C (funding carry) is cross-sectional like A/G
