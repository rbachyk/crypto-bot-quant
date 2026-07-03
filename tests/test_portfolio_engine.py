"""Correctness proof for the cross-sectional (basket) engine: a PLANTED funding-dispersion edge
must show up positive; a no-funding control must net ≈ 0. A new validation engine is exactly where
a subtle bug produces false promotes, so this is the guard before trusting any funding_carry result.
"""

from __future__ import annotations

import pytest
from src.backtest.config import load_backtest_config
from src.backtest.engine import SymbolInput
from src.backtest.metrics import build_report
from src.backtest.portfolio import CrossSectionalEngine
from src.exchange.metadata import load_metadata_config
from src.features.pipeline import FeatureFrame
from src.strategies.candidates import build_strategy
from src.strategies.config import load_strategies_config

IV = 60_000


def _sym(
    symbol: str, funding_z: float, funding_rate: float, n: int = 240, wick: float = 0.0
) -> SymbolInput:
    """Flat close-to-close price (so carry is the ONLY P&L), a constant funding_z, and a funding
    rate paid every 8 bars. ``wick`` adds symmetric intrabar range (high/low ±wick) so passive
    maker limits can genuinely trade through; with wick=0 the basket's realized P&L is exactly
    funding collected − costs."""
    bars = [
        {"ts": k * IV, "open": 100.0, "high": 100.0 + wick, "low": 100.0 - wick, "close": 100.0,
         "volume": 1e6}
        for k in range(n)
    ]
    rows = [
        {"ts": k * IV, "decision_ts": k * IV, "funding_z": funding_z, "atr_pct": 0.01,
         "session_code": 0}
        for k in range(n)
    ]
    frame = FeatureFrame(symbol=symbol, timeframe="1m", feature_names=["funding_z"], rows=rows)
    funding = [{"ts": k * IV, "funding_rate": funding_rate} for k in range(0, n, 8)]
    spread = [{"ts": k * IV, "spread_bps": 2.0} for k in range(n)]
    return SymbolInput(
        symbol=symbol, bars=bars, frame=frame, spread_samples=spread, funding_events=funding
    )


def _universe(aligned: bool, wick: float = 0.0) -> list[SymbolInput]:
    """10 symbols, funding_z spread −2..+2. EDGE: funding_rate aligns with funding_z (high funding_z
    = high positive rate ⇒ shorts collect; low = negative ⇒ longs collect). CONTROL: rate = 0."""
    out = []
    for i in range(10):
        fz = -2.0 + i * (4.0 / 9.0)
        rate = 0.001 * fz if aligned else 0.0
        out.append(_sym(f"S{i}/USDT:USDT", funding_z=fz, funding_rate=rate, wick=wick))
    return out


def _carry_strategy():
    sc = load_strategies_config()
    cand = sc.candidate("funding_carry")
    return build_strategy(cand, sc.strategy_version)


def test_cross_sectional_engine_harvests_a_planted_carry_edge():
    cfg = load_backtest_config()
    meta = load_metadata_config()
    strat = _carry_strategy()
    assert getattr(strat, "cross_sectional", False) is True

    def report(aligned):
        return build_report(CrossSectionalEngine(cfg, meta, strat).run(_universe(aligned))).payload

    edge = report(True)
    ctrl = report(False)

    # The basket is dollar-neutral: roughly equal long and short legs traded.
    sb = edge["side_breakdown"]
    assert sb["long"]["trades"] > 0 and sb["short"]["trades"] > 0

    # PLANTED carry shows up; the no-funding control nets ≈ 0 (only costs) and clearly worse.
    assert edge["expectancy_r"] > 0.0
    assert edge["net_pnl"] > 0.0
    assert ctrl["net_pnl"] <= 0.0  # no carry ⇒ costs only
    assert edge["net_pnl"] > ctrl["net_pnl"]


def test_cross_sectional_beta_mode_runs_and_keeps_the_carry():
    """Beta-neutral mode is wired and computes betas internally (no feature/rebuild). On flat-price
    inputs betas are degenerate ⇒ it falls back to dollar-neutral, so the planted carry still nets
    positive — proving the beta path runs end-to-end without breaking the edge."""
    from dataclasses import replace

    cfg = load_backtest_config()
    meta = load_metadata_config()
    sc = load_strategies_config()
    base = sc.candidate("funding_carry")
    cand = replace(
        base, params=replace(base.params, extra={**base.params.extra, "neutralization": "beta"})
    )
    strat = build_strategy(cand, sc.strategy_version)
    rep = build_report(CrossSectionalEngine(cfg, meta, strat).run(_universe(aligned=True))).payload
    assert rep["net_pnl"] > 0.0


def _carry_variant(**extra_overrides):
    from dataclasses import replace

    sc = load_strategies_config()
    base = sc.candidate("funding_carry")
    extra = {**base.params.extra, **extra_overrides}
    cand = replace(base, params=replace(base.params, extra=extra))
    return build_strategy(cand, sc.strategy_version)


def test_maker_rebalancing_cuts_cost_vs_taker_when_limits_actually_fill():
    """Maker rebalancing nets MORE than taker on the same planted carry edge — but only because
    the bars have real intrabar range, so the passive limits genuinely trade through (maker fee,
    no slippage, price improvement). Turnover cost is the carry's tightest margin."""
    cfg = load_backtest_config()
    meta = load_metadata_config()

    def net(maker: int) -> float:
        strat = _carry_variant(maker_rebalance=maker, rebalance_hours=0, rebalance_bars=8)
        # wick=0.05 (5 bps) > the half-spread limit offset (1 bp) ⇒ every limit trades through.
        return build_report(
            CrossSectionalEngine(cfg, meta, strat).run(_universe(True, wick=0.05))
        ).payload["net_pnl"]

    assert net(maker=1) > net(maker=0)


def test_maker_rebalance_fills_at_the_limit_only_when_bar_trades_through():
    """H2 regression (fill discipline parity with the per-trade engine): a 'maker' rebalance is a
    passive limit posted half the modelled spread inside the bar open. With intrabar range the
    entry fills exactly AT the limit (maker, no slippage); on a FLAT bar that never trades through,
    the leg is NOT granted a free maker fill — it escalates to TAKER at the bar close with
    spread-based slippage (a basket must rebalance), i.e. maker mode on dead bars costs exactly
    what taker mode does."""
    cfg = load_backtest_config()
    meta = load_metadata_config()

    # 1) Range bars: maker entries fill at the limit price (half-spread = 1bp inside the open).
    strat = _carry_variant(maker_rebalance=1, rebalance_hours=0, rebalance_bars=8)
    res = CrossSectionalEngine(cfg, meta, strat).run(_universe(True, wick=0.05))
    longs = [t for t in res.trades if t.side > 0]
    shorts = [t for t in res.trades if t.side < 0]
    assert longs and shorts
    offset = 0.5 * 2.0 / 10_000.0  # half the 2bps modelled spread
    for t in longs:
        assert t.entry_price == pytest.approx(100.0 * (1.0 - offset))
    for t in shorts:
        assert t.entry_price == pytest.approx(100.0 * (1.0 + offset))

    # 2) Flat bars: no bar ever trades through the limit ⇒ every leg escalates to taker; the
    # maker run books the SAME costs as the taker run (slippage paid, taker fee — no free fills).
    def run(maker: int):
        strat = _carry_variant(maker_rebalance=maker, rebalance_hours=0, rebalance_bars=8)
        return CrossSectionalEngine(cfg, meta, strat).run(_universe(True, wick=0.0))

    maker_res, taker_res = run(1), run(0)
    assert sum(t.pnl for t in maker_res.trades) == pytest.approx(
        sum(t.pnl for t in taker_res.trades)
    )
    assert all(t.slippage_cost > 0 for t in maker_res.trades)  # escalation paid real slippage


def test_gapped_leg_closes_on_last_prior_bar_not_a_future_price():
    """H1 regression: a held symbol with NO bar at the rebalance ts (halt / delist / feed gap)
    must be flattened on the last bar AT-OR-BEFORE that rebalance — never on the symbol's final
    bar of the run window, which books a FUTURE price with exit_ts weeks past the decision."""
    cfg = load_backtest_config()
    meta = load_metadata_config()
    universe = _universe(True)

    # S0 (extreme funding_z ⇒ always a basket leg) goes dark for bars [100, 200) and returns.
    gap_lo, gap_hi = 100 * IV, 200 * IV
    s0 = universe[0]
    universe[0] = SymbolInput(
        symbol=s0.symbol,
        bars=[b for b in s0.bars if not (gap_lo <= b["ts"] < gap_hi)],
        frame=FeatureFrame(
            symbol=s0.frame.symbol, timeframe=s0.frame.timeframe,
            feature_names=list(s0.frame.feature_names),
            rows=[r for r in s0.frame.rows if not (gap_lo <= r["ts"] < gap_hi)],
        ),
        spread_samples=s0.spread_samples,
        funding_events=s0.funding_events,
    )

    strat = _carry_variant(maker_rebalance=0, rebalance_hours=0, rebalance_bars=8)
    res = CrossSectionalEngine(cfg, meta, strat).run(universe)

    gap_closes = [t for t in res.trades if t.symbol == s0.symbol and t.exit_reason == "gap_close"]
    assert gap_closes, "the gapped leg must be flattened via the gap_close path"
    for t in gap_closes:
        # Exit is priced on the last bar BEFORE the gap (ts=99·IV) — not the future bar 239·IV.
        assert t.exit_ts == 99 * IV
    # And no basket trade may ever exit at the symbol's end-of-window bar unless the run ended.
    last_ts = 239 * IV
    assert all(t.exit_ts < last_ts or t.exit_reason == "end_of_data" for t in res.trades)


def test_cross_sectional_routing_via_run_engine():
    """run_engine dispatches a cross_sectional strategy to the basket engine (so walk-forward /
    stress route automatically), and a normal strategy still uses the per-trade engine."""
    from src.backtest.service import run_engine

    cfg = load_backtest_config()
    meta = load_metadata_config()
    run = run_engine(cfg, meta, _universe(aligned=True), strategy=_carry_strategy())
    assert run.report.trade_count > 0  # produced a basket report through run_engine


def test_cross_sectional_flag_marks_exactly_the_basket_candidates():
    """The per-symbol live/paper ensemble (resolve_active_strategies) excludes strategies whose
    `cross_sectional` flag is set — they run ONLY through the basket engine. Guard the predicate:
    the basket candidates carry the flag, and the per-symbol portfolio ones (lead_lag, xsection)
    do NOT, so a promoted basket can't leak into paper-live and crash / mis-route it."""
    sc = load_strategies_config()

    def flag(cid: str) -> bool:
        return bool(getattr(build_strategy(sc.candidate(cid), sc.strategy_version),
                            "cross_sectional", False))

    assert flag("funding_carry") is True
    assert flag("residual_momentum") is True
    # cross-asset / cross-sectional-RS run on the per-symbol realtime feed (evaluate_portfolio),
    # NOT the basket engine — they must stay in the per-symbol ensemble.
    assert flag("lead_lag_xasset") is False
    assert flag("xsection_rs") is False
