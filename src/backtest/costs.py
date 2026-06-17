"""Cost models — realistic fees, slippage and funding (AGENTS.md Section 19).

The backtest is only as honest as its costs, so all three are modelled
explicitly and tied to VERIFIED exchange metadata (Section 19: "realistic fees;
realistic slippage; funding if relevant ... exchange metadata"):

* :class:`FeeModel` — maker/taker fees come from the operator-verified
  ``configs/metadata.yaml`` (the META gate guarantees they exist + are
  consistent); ``fee_multiplier`` is the FEE-stress knob.
* :class:`SlippageModel` — half-spread + a notional-vs-liquidity impact term,
  always adverse to the taker; ``slippage_multiplier`` is the SLIP-stress knob.
* :class:`FundingModel` — perpetual funding charged to the OPEN position at each
  funding timestamp (longs pay positive funding).

Every cost is a pure function of inputs known at or before fill time, so the
engine stays causal.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.backtest.config import CostConfig
from src.exchange.metadata import MetadataConfig

# Buy = +1 (pays the ask / slips up), Sell = -1 (hits the bid / slips down).
BUY = 1
SELL = -1


@dataclass(slots=True)
class FeeModel:
    """Maker/taker fees from verified metadata, scaled by the stress multiplier."""

    meta: MetadataConfig
    costs: CostConfig

    def taker_fee_rate(self, symbol: str) -> float:
        return self._rate(symbol, "taker_fee", self.costs.fallback_taker_fee)

    def maker_fee_rate(self, symbol: str) -> float:
        return self._rate(symbol, "maker_fee", self.costs.fallback_maker_fee)

    def _rate(self, symbol: str, field: str, fallback: float) -> float:
        spec = self.meta.spec(symbol)
        raw = spec.fields.get(field) if spec is not None else None
        base = float(raw) if isinstance(raw, (int, float)) else fallback
        return base * self.costs.fee_multiplier

    def fee(self, symbol: str, notional: float, *, maker: bool) -> float:
        """Absolute fee paid on a fill of ``notional`` (always >= 0)."""
        rate = self.maker_fee_rate(symbol) if maker else self.taker_fee_rate(symbol)
        return abs(notional) * rate


@dataclass(slots=True)
class SlippageModel:
    """Adverse slippage as a fraction of price, scaled by the stress multiplier.

    ``slippage_frac`` is half the spread (the cost of crossing) plus a linear
    impact term in the order's size relative to the bar's traded notional. The
    fill price is always worse for the taker (buys up, sells down).
    """

    costs: CostConfig

    def slippage_frac(self, *, spread_bps: float, notional: float, bar_notional: float) -> float:
        half_spread = max((spread_bps / 1e4) / 2.0, self.costs.min_half_spread_frac)
        impact = 0.0
        if self.costs.impact_coeff > 0.0 and bar_notional > 0.0:
            impact = self.costs.impact_coeff * (abs(notional) / bar_notional)
        return (half_spread + impact) * self.costs.slippage_multiplier

    def fill_price(self, reference_price: float, side: int, slippage_frac: float) -> float:
        """Apply adverse slippage to a reference (mid/open) price for ``side``."""
        return reference_price * (1.0 + side * slippage_frac)


@dataclass(slots=True)
class FundingModel:
    """Perpetual funding charged to an open position (Section 8/19).

    At each funding timestamp the position pays ``funding_rate × notional`` when
    long and receives it when short (positive funding ⇒ longs pay shorts). The
    multiplier supports funding-sensitivity runs.
    """

    costs: CostConfig

    def payment(self, *, side: int, notional: float, funding_rate: float) -> float:
        """Funding cost (>0 = paid by this position, <0 = received)."""
        return side * funding_rate * abs(notional) * self.costs.funding_multiplier
