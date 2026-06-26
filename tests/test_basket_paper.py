"""Offline proof for the live/paper basket loop: a planted funding-carry edge, fed one snapshot
at a time, books profitable PaperTrades — confirming the live path reuses the engine math and the
Trade→PaperTrade conversion is sound (before it ever touches a live feed)."""

from __future__ import annotations

from src.backtest.config import load_backtest_config
from src.backtest.engine import SymbolInput
from src.exchange.metadata import load_metadata_config
from src.features.pipeline import FeatureFrame
from src.live.basket import BasketPaperLoop
from src.paper.session import PaperSession
from src.strategies.candidates import build_strategy
from src.strategies.config import load_strategies_config

IV = 60_000
N = 240


def _fixture():
    """10 symbols, flat price (carry is the only P&L), constant funding_z, funding rate ALIGNED
    with funding_z (high funding ⇒ shorts collect, low ⇒ longs collect). Returns the per-symbol
    SymbolInput (spread + funding) and the snapshot stream."""
    syms = [f"S{i}/USDT:USDT" for i in range(10)]
    by_symbol: dict[str, SymbolInput] = {}
    fz = {}
    for i, s in enumerate(syms):
        z = -2.0 + i * (4.0 / 9.0)
        fz[s] = z
        funding = [{"ts": k * IV, "funding_rate": 0.001 * z} for k in range(0, N, 8)]
        spread = [{"ts": k * IV, "spread_bps": 2.0} for k in range(N)]
        frame = FeatureFrame(symbol=s, timeframe="1m", feature_names=["funding_z"], rows=[])
        by_symbol[s] = SymbolInput(
            symbol=s, bars=[], frame=frame, spread_samples=spread, funding_events=funding
        )

    def snapshots():
        for k in range(N):
            ts = k * IV
            bars_at = {
                s: {"ts": ts, "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
                    "volume": 1e6}
                for s in syms
            }
            rows_at = {
                s: {"ts": ts, "decision_ts": ts, "funding_z": fz[s], "atr_pct": 0.01,
                    "session_code": 0}
                for s in syms
            }
            yield (ts, bars_at, rows_at)

    return by_symbol, snapshots()


def test_basket_paper_loop_books_profitable_carry_trades():
    cfg = load_backtest_config()
    meta = load_metadata_config()
    sc = load_strategies_config()
    strat = build_strategy(sc.candidate("funding_carry"), sc.strategy_version)
    session = PaperSession(session_id="t")
    by_symbol, snaps = _fixture()

    BasketPaperLoop(cfg, meta, strat, bar_interval_ms=IV, session=session).run(snaps, by_symbol)

    assert session.trades, "the basket loop should book paper trades"
    longs = [t for t in session.trades if t.side > 0]
    shorts = [t for t in session.trades if t.side < 0]
    assert longs and shorts  # dollar-neutral basket trades both sides
    assert sum(t.pnl for t in session.trades) > 0.0  # planted carry is harvested
    assert all(t.execution_route == "maker" for t in session.trades)  # rebalanced as maker
