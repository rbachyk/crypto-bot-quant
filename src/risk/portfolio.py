"""Portfolio state for risk checks (AGENTS.md Section 17, portfolio-level).

A snapshot of currently-open positions used by the risk manager to enforce the
multi-symbol caps: portfolio **heat** (Σ open risk across positions) and net
**beta-to-BTC** (Section 2.2 envelope). Heat is expressed as a fraction of equity
so the whole thing is capital-agnostic (Section 0).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Position:
    """One open position's risk-relevant facts."""

    symbol: str
    side: int  # +1 long / -1 short
    qty: float
    entry_price: float
    risk_amount: float  # currency at risk = qty × |entry − stop|
    beta_to_btc: float  # symbol's beta to BTC (signed contribution uses side)
    regime: str = ""

    @property
    def notional(self) -> float:
        return self.qty * self.entry_price

    def signed_beta_exposure(self, equity: float) -> float:
        """Net-beta contribution as a fraction of equity (Section 17 net-beta cap)."""
        if equity <= 0:
            return 0.0
        return self.side * self.beta_to_btc * self.notional / equity


@dataclass(frozen=True, slots=True)
class PortfolioState:
    """The set of open positions the risk manager reasons over."""

    equity: float
    positions: tuple[Position, ...] = field(default_factory=tuple)

    def heat(self) -> float:
        """Σ open risk as a fraction of equity (the portfolio heat)."""
        if self.equity <= 0:
            return 0.0
        return sum(p.risk_amount for p in self.positions) / self.equity

    def net_beta(self) -> float:
        """Signed net beta-to-BTC across all positions (fraction of equity)."""
        return sum(p.signed_beta_exposure(self.equity) for p in self.positions)

    def count(self) -> int:
        return len(self.positions)

    def count_for_symbol(self, symbol: str) -> int:
        return sum(1 for p in self.positions if p.symbol == symbol)

    def count_for_regime(self, regime: str) -> int:
        return sum(1 for p in self.positions if p.regime == regime)
