"""Backtest integrity guards (AGENTS.md Section 19).

Section 19 requires the backtest to *prevent* look-ahead, survivorship and
future-universe leakage. Two independent guards back the BT gate:

* :func:`noise_expectancy` — the engine-level leakage test. Run the SAME engine +
  strategy on a NO-STRUCTURE reference series (``edge="noise"``). With no real
  edge and realistic costs the strategy cannot be profitable; if it were, the
  engine would be manufacturing edge from nothing (look-ahead). Pass ⇒ expectancy
  is not positive on noise (mirrors the FEAT gate's synthetic test, Section 16).

* :func:`future_universe_violations` — survivorship / future-universe guard. A
  symbol may be traded only at decision times at or after it entered the universe
  (``activation_ts``); any earlier entry is a leak. The reference universe
  activates one symbol partway through specifically to exercise this.

The strongest look-ahead guard is structural and lives in the engine itself
(signals fill at the NEXT bar open) and is asserted directly in the tests.
"""

from __future__ import annotations

from dataclasses import replace

from src.backtest.config import BacktestConfig
from src.backtest.engine import BacktestResult, SymbolInput
from src.backtest.service import build_reference_inputs, run_engine
from src.exchange.metadata import MetadataConfig


def noise_expectancy(
    cfg: BacktestConfig, meta: MetadataConfig, *, tolerance_r: float = 0.05
) -> dict:
    """Run the strategy on a structureless series; it must NOT be profitable."""
    noise_cfg = replace(cfg, reference=replace(cfg.reference, edge="noise"))
    inputs = build_reference_inputs(noise_cfg)
    report = run_engine(noise_cfg, meta, inputs, label="noise_guard").report
    expectancy = report.expectancy_r
    return {
        "expectancy_r": expectancy,
        "net_pnl": report.net_pnl,
        "trade_count": report.trade_count,
        "tolerance_r": tolerance_r,
        # No leakage ⇒ a structureless series yields no positive edge.
        "passed": expectancy <= tolerance_r,
    }


def future_universe_violations(result: BacktestResult, inputs: list[SymbolInput]) -> list[dict]:
    """Trades whose entry preceded their symbol's universe activation (a leak)."""
    activation = {s.symbol: s.activation_ts for s in inputs}
    out: list[dict] = []
    for t in result.trades:
        act = activation.get(t.symbol, 0)
        if t.entry_ts < act:
            out.append({"symbol": t.symbol, "entry_ts": t.entry_ts, "activation_ts": act})
    return out
