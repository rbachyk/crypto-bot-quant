"""Risk-adjusted, cost-net reward for the RL trading environment (AGENTS.md §21.4).

Reward is fee/slippage/funding-net and risk-adjusted:
  - Raw edge is the candidate's ``expected_edge_frac`` (from deterministic baseline)
  - Fee and slippage costs are subtracted
  - Funding impact is subtracted/added
  - Drawdown and tail-loss penalties are applied
  - Envelope proximity adds a convex penalty (never modifies the envelope itself)

Hard invariants:
  - Reward is never constructed by widening the risk envelope.
  - Size 0.0 (skip) always yields reward 0.0 (neutral signal skip).
  - Reward is bounded to [-10, 10] to prevent gradient explosion.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RewardConfig:
    """Configurable reward parameters (all can be tuned; envelope constants cannot)."""

    taker_fee_rate: float = 0.0004  # 4 bps taker fee
    maker_fee_rate: float = 0.0002  # 2 bps maker fee
    slippage_scale: float = 1.0  # multiplier on slippage_est
    funding_scale: float = 1.0  # multiplier on funding_z contribution
    drawdown_penalty: float = 0.5  # per-unit penalty beyond soft drawdown threshold
    drawdown_soft_threshold: float = 0.04  # 4% soft drawdown threshold
    tail_loss_penalty: float = 2.0  # extra penalty when loss exceeds 3x expected
    envelope_proximity_penalty: float = 1.0  # penalty as heat approaches cap
    max_reward: float = 10.0  # clip bound (prevent gradient explosion)


@dataclass
class RewardState:
    """Tracks episode-level state needed for risk-adjusted reward."""

    cumulative_pnl: float = 0.0
    peak_pnl: float = 0.0
    current_drawdown: float = 0.0
    current_heat: float = 0.0  # current portfolio heat fraction [0, 1]
    n_steps: int = 0
    _daily_pnl: float = field(default=0.0, repr=False)

    def update(self, step_pnl: float, heat_delta: float = 0.0) -> None:
        self.cumulative_pnl += step_pnl
        self._daily_pnl += step_pnl
        self.n_steps += 1
        if self.cumulative_pnl > self.peak_pnl:
            self.peak_pnl = self.cumulative_pnl
        self.current_drawdown = max(0.0, self.peak_pnl - self.cumulative_pnl)
        self.current_heat = max(0.0, min(1.0, self.current_heat + heat_delta))


class RiskAdjustedReward:
    """Computes the per-step reward for the RL trading environment.

    The reward is:
        r = (edge - fees - slippage - funding_impact)
            - drawdown_penalty
            - tail_loss_penalty
            - envelope_proximity_penalty
        clipped to [-max_reward, max_reward]

    A size_bucket of 0.0 (skip) always returns 0.0 (neutral skip).
    """

    def __init__(self, config: RewardConfig | None = None) -> None:
        self.cfg = config or RewardConfig()

    def compute(
        self,
        *,
        expected_edge_frac: float,
        size_bucket: float,
        take: bool,
        exec_style: str,
        spread_bps: float,
        slippage_est: float,
        funding_z: float,
        state: RewardState,
        stochastic_noise: float = 0.0,
    ) -> float:
        """Compute one-step reward.

        Parameters
        ----------
        expected_edge_frac:
            The deterministic baseline's expected edge (net-of-costs estimate from
            the strategy layer). Used as the base PnL draw before RL costs.
        size_bucket:
            The RL policy's chosen size multiplier {0.0, 0.25, 0.5, 1.0}.
        take:
            Whether the RL policy chose to take the trade.
        exec_style:
            Chosen execution style ("maker", "taker", "passive_then_taker").
        spread_bps:
            Current spread in basis points.
        slippage_est:
            Estimated slippage as fraction of notional.
        funding_z:
            Funding rate z-score.
        state:
            Mutable episode-level state (drawdown, heat, cumulative PnL).
        stochastic_noise:
            Optional realized PnL noise for simulation (drawn externally for
            reproducibility — the reward fn is deterministic given inputs).

        Returns
        -------
        float
            Risk-adjusted, cost-net reward clipped to ±max_reward.
        """
        cfg = self.cfg

        # Skip case: neutral (no cost, no gain)
        if not take or size_bucket == 0.0:
            return 0.0

        # --- raw edge (scaled by size) ------------------------------------ #
        raw_edge = expected_edge_frac * size_bucket + stochastic_noise * size_bucket

        # --- fee cost ----------------------------------------------------- #
        fee_rate = cfg.maker_fee_rate if exec_style == "maker" else cfg.taker_fee_rate
        fee_cost = fee_rate * size_bucket

        # --- slippage cost ------------------------------------------------ #
        slippage_cost = slippage_est * size_bucket * cfg.slippage_scale

        # --- funding impact ----------------------------------------------- #
        # Positive funding_z means longs pay shorts; negative = shorts pay longs.
        # Small impact proportional to z-score and position size.
        funding_impact = 0.0001 * funding_z * size_bucket * cfg.funding_scale

        # --- step PnL ----------------------------------------------------- #
        step_pnl = raw_edge - fee_cost - slippage_cost - funding_impact

        # --- update state ------------------------------------------------- #
        heat_delta = size_bucket * 0.01  # 1% heat per full-size trade
        state.update(step_pnl, heat_delta)

        # --- drawdown penalty --------------------------------------------- #
        drawdown_penalty = 0.0
        if state.current_drawdown > cfg.drawdown_soft_threshold:
            excess = state.current_drawdown - cfg.drawdown_soft_threshold
            drawdown_penalty = excess * cfg.drawdown_penalty

        # --- tail loss penalty -------------------------------------------- #
        tail_loss_penalty = 0.0
        if step_pnl < -3.0 * abs(expected_edge_frac + 1e-9) * size_bucket:
            tail_loss_penalty = abs(step_pnl) * cfg.tail_loss_penalty

        # --- envelope proximity penalty ----------------------------------- #
        proximity_penalty = 0.0
        if state.current_heat > 0.8:  # 80% of cap → start penalizing
            proximity_penalty = (state.current_heat - 0.8) * cfg.envelope_proximity_penalty

        reward = step_pnl - drawdown_penalty - tail_loss_penalty - proximity_penalty
        return float(max(-cfg.max_reward, min(cfg.max_reward, reward)))
