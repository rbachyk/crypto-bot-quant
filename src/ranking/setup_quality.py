"""Setup Quality Gate scoring (AGENTS.md Section 15, SETUP gate).

Regime match does **not** authorise a trade — it only allows setup evaluation
(Section 15). Every candidate is scored deterministically across the seven
Section 15 components (max 100) and checked against the hard-blocker list. A
candidate is approved only when its ``setup_quality_score`` clears the configured
threshold **and** no hard blocker is active — a high score can never bypass a
blocker (the SETUP gate asserts exactly this).

Scoring is a pure function of the candidate's decision-time features + verified
metadata + the supplied state context, so it is reproducible: the same inputs
always yield the same score (Section 15 "deterministic, reproducible").
"""

from __future__ import annotations

from dataclasses import dataclass

from src.exchange.metadata import MetadataConfig
from src.ranking.candidate import Candidate
from src.ranking.config import RankingConfig

# Default no-trade / protection regimes (Section 11): no strategy may trade here.
NO_TRADE_REGIMES: frozenset[str] = frozenset(
    {"R8_DATA_UNSAFE", "R7_TOXIC_EXECUTION", "R4_HIGH_VOL_CHOP"}
)


@dataclass(frozen=True, slots=True)
class SetupContext:
    """Account/data state consulted as Section 15 hard blockers."""

    daily_loss_reached: bool = False
    drawdown_reached: bool = False
    data_quality_failed: bool = False
    foreign_order_detected: bool = False
    open_position_conflict: bool = False
    require_config_live_approved: bool = False  # only enforced in live mode


@dataclass(frozen=True, slots=True)
class SetupScore:
    total: float
    components: dict[str, float]
    blockers: tuple[str, ...]
    threshold: float
    expected_value_after_costs: float
    round_trip_cost_frac: float

    @property
    def passed_threshold(self) -> bool:
        return self.total >= self.threshold

    @property
    def approved(self) -> bool:
        # A hard blocker can never be bypassed by a high score (Section 15).
        return not self.blockers and self.passed_threshold

    def to_dict(self) -> dict:
        return {
            "total": round(self.total, 6),
            "components": {k: round(v, 6) for k, v in self.components.items()},
            "blockers": list(self.blockers),
            "threshold": self.threshold,
            "passed_threshold": self.passed_threshold,
            "approved": self.approved,
            "expected_value_after_costs": round(self.expected_value_after_costs, 8),
            "round_trip_cost_frac": round(self.round_trip_cost_frac, 8),
        }


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


class SetupQualityScorer:
    """Deterministic Section 15 setup-quality scoring + hard blockers."""

    def __init__(self, cfg: RankingConfig, meta: MetadataConfig) -> None:
        self.cfg = cfg
        self.meta = meta

    def _taker_fee(self, symbol: str) -> float | None:
        spec = self.meta.spec(symbol)
        if spec is None:
            return None
        fee = spec.fields.get("taker_fee")
        return float(fee) if isinstance(fee, (int, float)) else None

    def round_trip_cost_frac(self, candidate: Candidate) -> float:
        """Round-trip cost as a fraction of price: 2×taker fee + estimated slippage."""
        fee = self._taker_fee(candidate.symbol) or 0.0
        return 2.0 * fee + candidate.slippage_est

    # ------------------------------------------------------------------ #
    def hard_blockers(self, candidate: Candidate, ctx: SetupContext) -> tuple[str, ...]:
        """The active Section 15 hard blockers for this candidate (empty ⇒ none)."""
        out: list[str] = []
        if not candidate.data_fresh:
            out.append("stale_data")
        if not candidate.metadata_verified or self.meta.spec(candidate.symbol) is None:
            out.append("missing_metadata")
        if self._taker_fee(candidate.symbol) is None:
            out.append("missing_fees")
        if candidate.regime in NO_TRADE_REGIMES:
            out.append("no_trade_regime")
        if candidate.spread_bps > self.cfg.max_spread_bps:
            out.append("spread_above_threshold")
        if candidate.slippage_est > self.cfg.max_slippage_frac:
            out.append("slippage_above_threshold")
        if not candidate.symbol_tradable:
            out.append("symbol_halted_or_inactive")
        if not candidate.strategy_enabled:
            out.append("strategy_disabled")
        if candidate.stop_frac <= 0:
            out.append("undefined_stop")
        if self.expected_value_after_costs(candidate) <= 0:
            out.append("negative_ev_after_costs")
        if ctx.daily_loss_reached:
            out.append("daily_loss_limit_reached")
        if ctx.drawdown_reached:
            out.append("drawdown_limit_reached")
        if ctx.data_quality_failed:
            out.append("data_quality_failure")
        if ctx.foreign_order_detected:
            out.append("foreign_order_detected")
        if ctx.open_position_conflict:
            out.append("open_position_conflict")
        if ctx.require_config_live_approved and not candidate.config_live_approved:
            out.append("config_not_live_approved")
        return tuple(out)

    def expected_value_after_costs(self, candidate: Candidate) -> float:
        return candidate.expected_edge_frac - self.round_trip_cost_frac(candidate)

    # ------------------------------------------------------------------ #
    def components(self, candidate: Candidate) -> dict[str, float]:
        """The seven Section 15 component scores (each within its configured max)."""
        c = self.cfg.components
        f = candidate.features

        # 1) Regime alignment: calmer regimes (lower ATR%-rank) suit trading more.
        atr_rank = _clamp01(float(f.get("atr_pct_rank", 0.5)))
        regime_factor = _clamp01(1.0 - 0.5 * atr_rank)
        regime_alignment = c["regime_alignment"] * regime_factor

        # 2) Signal strength: the strategy's normalised conviction.
        signal_strength = c["signal_strength"] * _clamp01(candidate.signal_strength)

        # 3) Cross-signal confirmation.
        cross_conf = c["cross_signal_confirmation"] * _clamp01(candidate.confirmation)

        # 4) Expected move after costs: EV net of costs vs the gross edge.
        edge = candidate.expected_edge_frac
        ev = self.expected_value_after_costs(candidate)
        ev_ratio = _clamp01(ev / edge) if edge > 0 else 0.0
        expected_move = c["expected_move_after_costs"] * ev_ratio

        # 5) Execution quality: spread + slippage headroom vs the caps.
        spread_factor = _clamp01(1.0 - candidate.spread_bps / self.cfg.max_spread_bps)
        slip_factor = _clamp01(1.0 - candidate.slippage_est / self.cfg.max_slippage_frac)
        exec_quality = c["execution_quality"] * (0.5 * spread_factor + 0.5 * slip_factor)

        # 6) Risk/reward quality: reward:risk vs target (momentum's huge TP caps it).
        rr = candidate.tp_frac / candidate.stop_frac if candidate.stop_frac > 0 else 0.0
        rr_quality = c["risk_reward_quality"] * _clamp01(rr / self.cfg.rr_target)

        # 7) Session/context: penalise weekend + pre-funding windows.
        is_weekend = 1.0 if float(f.get("is_weekend", 0.0)) >= 0.5 else 0.0
        pre_funding = 1.0 if float(f.get("pre_funding", 0.0)) >= 0.5 else 0.0
        session_factor = _clamp01(1.0 - 0.3 * is_weekend - 0.3 * pre_funding)
        session_ctx = c["session_context"] * session_factor

        return {
            "regime_alignment": regime_alignment,
            "signal_strength": signal_strength,
            "cross_signal_confirmation": cross_conf,
            "expected_move_after_costs": expected_move,
            "execution_quality": exec_quality,
            "risk_reward_quality": rr_quality,
            "session_context": session_ctx,
        }

    def score(self, candidate: Candidate, ctx: SetupContext | None = None) -> SetupScore:
        ctx = ctx or SetupContext()
        components = self.components(candidate)
        total = sum(components.values())
        return SetupScore(
            total=total,
            components=components,
            blockers=self.hard_blockers(candidate, ctx),
            threshold=self.cfg.threshold,
            expected_value_after_costs=self.expected_value_after_costs(candidate),
            round_trip_cost_frac=self.round_trip_cost_frac(candidate),
        )
