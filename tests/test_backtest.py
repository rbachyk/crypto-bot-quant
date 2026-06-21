"""Phase 4 Backtest Engine unit tests (AGENTS.md Section 19, Appendix D).

Pure, offline tests of the event-based engine and its harnesses — no DB/Redis
required (the gate-runner integration is covered in ``test_phase4_gates.py``).
They prove the Section 19 contract directly:

* the engine is **event-based** and **structurally look-ahead-free** (a signal
  fills at the NEXT bar's open, never its own bar's close);
* fees, slippage and funding are modelled and adverse to the taker;
* risk sizing follows the Section 17 identity with leverage-as-consequence and
  the metadata min-notional / lot-step gate;
* survivorship / future-universe leakage is prevented (no trade before a symbol
  joins the universe), and a structureless series yields ~0 expectancy;
* the report carries every required output; walk-forward and fee/slippage stress
  behave as specified.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from src.backtest import (
    BacktestEngine,
    SymbolInput,
    build_reference_inputs,
    fee_stress,
    future_universe_violations,
    load_backtest_config,
    max_drawdown,
    noise_expectancy,
    run_engine,
    run_reference_backtest,
    run_walk_forward,
    slippage_stress,
)
from src.backtest.costs import BUY, SELL, FeeModel, FundingModel, SlippageModel
from src.backtest.risk import RiskSimulator
from src.backtest.strategy import ReferenceMomentumStrategy, Signal
from src.exchange.metadata import load_metadata_config

REF_SYMBOL = "BTC/USDT:USDT"


@pytest.fixture(scope="module")
def cfg():
    return load_backtest_config()


@pytest.fixture(scope="module")
def meta():
    return load_metadata_config()


@pytest.fixture(scope="module")
def ref_inputs(cfg):
    return build_reference_inputs(cfg)


# --------------------------------------------------------------------------- #
# Cost models (Section 19: realistic fees / slippage / funding)                #
# --------------------------------------------------------------------------- #
def test_fee_model_uses_verified_metadata(cfg, meta):
    fees = FeeModel(meta, cfg.costs)
    # configs/metadata.yaml pins taker_fee=0.00055, maker_fee=0.0002 for BTC.
    assert fees.taker_fee_rate(REF_SYMBOL) == pytest.approx(0.00055)
    assert fees.maker_fee_rate(REF_SYMBOL) == pytest.approx(0.0002)
    fee = fees.fee(REF_SYMBOL, notional=10_000.0, maker=False)
    assert fee == pytest.approx(10_000.0 * 0.00055)
    assert fee >= 0.0


def test_fee_multiplier_is_the_stress_knob(cfg, meta):
    base = FeeModel(meta, cfg.costs)
    stressed = FeeModel(meta, replace(cfg.costs, fee_multiplier=2.0))
    assert stressed.taker_fee_rate(REF_SYMBOL) == pytest.approx(
        2.0 * base.taker_fee_rate(REF_SYMBOL)
    )


def test_fee_falls_back_when_symbol_unknown(cfg, meta):
    fees = FeeModel(meta, cfg.costs)
    assert fees.taker_fee_rate("NOPE/USDT:USDT") == pytest.approx(cfg.costs.fallback_taker_fee)


def test_slippage_is_adverse_to_the_taker(cfg):
    slip = SlippageModel(cfg.costs)
    frac = slip.slippage_frac(spread_bps=10.0, notional=1_000.0, bar_notional=1e9)
    assert frac > 0.0
    # Buyer pays up, seller is filled down — always worse than the reference.
    assert slip.fill_price(100.0, BUY, frac) > 100.0
    assert slip.fill_price(100.0, SELL, frac) < 100.0


def test_slippage_respects_half_spread_floor(cfg):
    slip = SlippageModel(replace(cfg.costs, impact_coeff=0.0))
    # Tiny spread -> floored at min_half_spread_frac * slippage_multiplier.
    frac = slip.slippage_frac(spread_bps=0.0, notional=1.0, bar_notional=1.0)
    assert frac == pytest.approx(cfg.costs.min_half_spread_frac * cfg.costs.slippage_multiplier)


def test_slippage_multiplier_is_the_stress_knob(cfg):
    base = SlippageModel(cfg.costs).slippage_frac(spread_bps=10.0, notional=1.0, bar_notional=1.0)
    stressed = SlippageModel(replace(cfg.costs, slippage_multiplier=1.5)).slippage_frac(
        spread_bps=10.0, notional=1.0, bar_notional=1.0
    )
    assert stressed == pytest.approx(1.5 * base)


def test_funding_longs_pay_positive_funding(cfg):
    fund = FundingModel(cfg.costs)
    long_pay = fund.payment(side=BUY, notional=10_000.0, funding_rate=0.0001)
    short_pay = fund.payment(side=SELL, notional=10_000.0, funding_rate=0.0001)
    assert long_pay > 0.0  # positive funding => longs pay
    assert short_pay == pytest.approx(-long_pay)  # shorts receive the mirror


# --------------------------------------------------------------------------- #
# Risk simulation (Section 17 sizing identity + hard gates)                    #
# --------------------------------------------------------------------------- #
def test_risk_sizing_follows_section17_identity(cfg, meta):
    risk = RiskSimulator(cfg.account, meta)
    r = risk.size(REF_SYMBOL, equity=100_000.0, entry_price=100.0, stop_frac=0.02)
    assert r.approved
    # size = (equity * risk_pct) / |entry - stop|, before the lot-step rounding.
    expected_qty = (100_000.0 * cfg.account.risk_pct) / (100.0 * 0.02)
    assert r.qty == pytest.approx(expected_qty, rel=0.01)
    assert r.risk_amount == pytest.approx(100_000.0 * cfg.account.risk_pct, rel=0.05)


def test_risk_rejects_invalid_inputs(cfg, meta):
    risk = RiskSimulator(cfg.account, meta)
    assert not risk.size(REF_SYMBOL, equity=0.0, entry_price=100.0, stop_frac=0.02).approved
    assert not risk.size(REF_SYMBOL, equity=1e5, entry_price=100.0, stop_frac=0.0).approved
    assert not risk.size(REF_SYMBOL, equity=1e5, entry_price=0.0, stop_frac=0.02).approved


def test_risk_leverage_is_capped_as_a_consequence(cfg, meta):
    # A tiny stop would size a huge notional; leverage must be clamped, never targeted.
    account = replace(cfg.account, max_leverage=3.0, risk_pct=0.05)
    risk = RiskSimulator(account, meta)
    r = risk.size(REF_SYMBOL, equity=100_000.0, entry_price=100.0, stop_frac=0.0001)
    assert r.approved
    assert r.leverage <= 3.0 + 1e-9


def test_risk_min_notional_gate(cfg, meta):
    # Very small equity -> below the metadata min_notional => rejected.
    risk = RiskSimulator(replace(cfg.account, risk_pct=1e-9), meta)
    r = risk.size(REF_SYMBOL, equity=1.0, entry_price=100.0, stop_frac=0.02)
    assert not r.approved
    assert "min_notional" in r.reason or "min_order_size" in r.reason


# --------------------------------------------------------------------------- #
# Reference data determinism + strategy                                        #
# --------------------------------------------------------------------------- #
def test_reference_inputs_are_deterministic(cfg):
    a = build_reference_inputs(cfg)
    b = build_reference_inputs(cfg)
    assert [s.symbol for s in a] == [s.symbol for s in b]
    assert a[0].bars[0]["close"] == b[0].bars[0]["close"]
    assert a[0].bars[-1]["close"] == b[0].bars[-1]["close"]


def test_strategy_declines_below_threshold(cfg):
    strat = ReferenceMomentumStrategy(cfg.reference_strategy)
    assert strat.evaluate({"ret_short": 0.0, "atr_pct": 0.01}) is None


def test_strategy_emits_directional_signal(cfg):
    strat = ReferenceMomentumStrategy(cfg.reference_strategy)
    thr = cfg.reference_strategy.signal_threshold
    long_sig = strat.evaluate({"ret_short": thr * 3, "atr_pct": 0.01})
    short_sig = strat.evaluate({"ret_short": -thr * 3, "atr_pct": 0.01})
    assert isinstance(long_sig, Signal) and long_sig.side == 1
    assert isinstance(short_sig, Signal) and short_sig.side == -1
    assert long_sig.stop_frac > 0 and long_sig.tp_frac > long_sig.stop_frac


# --------------------------------------------------------------------------- #
# Engine: event-based, structurally look-ahead-free                            #
# --------------------------------------------------------------------------- #
def _grid_bars(prices: list[float], iv: int = 60_000) -> list[dict]:
    bars = []
    for i, p in enumerate(prices):
        prev = prices[i - 1] if i > 0 else p
        bars.append(
            {
                "ts": i * iv,
                "open": prev,
                "high": max(prev, p) * 1.001,
                "low": min(prev, p) * 0.999,
                "close": p,
                "volume": 10_000.0,
            }
        )
    return bars


def test_engine_fills_signal_at_next_bar_open_not_own_close(cfg, meta):
    """The look-ahead guard is structural: a decision on bar k fills at bar k+1's
    open. We craft a one-shot strategy that fires only on bar 0's row and assert
    the entry price equals bar 1's open (adjusted only by adverse slippage)."""
    from src.features.pipeline import FeatureFrame

    iv = 60_000
    bars = _grid_bars([100.0, 110.0, 121.0, 130.0, 130.0, 130.0], iv=iv)

    class FireOnceStrategy:
        name = "fire_once"
        strategy_version = "t1"

        def evaluate(self, row: dict):
            # decision_ts == iv corresponds to bar 0's close -> entry bar 1.
            if row["decision_ts"] == iv:
                return Signal(side=1, stop_frac=0.5, tp_frac=10.0)
            return None

    rows = [{"ts": k * iv, "decision_ts": (k + 1) * iv, "ret_short": 0.0} for k in range(len(bars))]
    frame = FeatureFrame(symbol=REF_SYMBOL, timeframe="1m", feature_names=["ret_short"], rows=rows)
    spread = [{"ts": b["ts"], "spread_bps": 2.0} for b in bars]
    sym = SymbolInput(
        symbol=REF_SYMBOL, bars=bars, frame=frame, spread_samples=spread, funding_events=[]
    )

    engine = BacktestEngine(cfg, meta, FireOnceStrategy())
    result = engine.run([sym])
    assert len(result.trades) == 1
    t = result.trades[0]
    # Entry references bar 1's OPEN (= bar 0's close = 100.0), never bar 0's close
    # read on bar 0, and never bar 1's close (110.0). Adverse slippage nudges it up.
    assert t.entry_price >= 100.0
    assert t.entry_price < 110.0
    assert t.entry_ts == iv  # decision_ts of the firing row == bar 1's ts


def test_engine_charges_costs_on_every_trade(cfg, meta, ref_inputs):
    run = run_engine(cfg, meta, ref_inputs, label="costs")
    assert run.report.trade_count > 0
    for t in run.result.trades:
        assert t.fee > 0.0  # entry + exit taker fees always charged
        assert t.slippage_cost > 0.0  # adverse slippage on entry and exit


def test_engine_logs_rejected_candidates(cfg, meta):
    # A punishing spread cap rejects every entry and logs it (Section 19).
    tight = replace(cfg, execution=replace(cfg.execution, max_spread_bps=0.0))
    inputs = build_reference_inputs(tight)
    run = run_engine(tight, meta, inputs, label="all_rejected")
    assert run.result.rejected, "candidates blocked by hard-blockers must be logged"
    assert any("toxic_spread" in r.reason for r in run.result.rejected)
    assert run.report.payload["rejected_candidates"]["total"] == len(run.result.rejected)


# --------------------------------------------------------------------------- #
# Integrity guards (Section 19: no leakage / survivorship)                     #
# --------------------------------------------------------------------------- #
def test_noise_series_yields_no_edge(cfg, meta):
    """The same engine on a structureless series must not be profitable — the
    engine-level look-ahead/leakage guard (mirror of the FEAT synthetic test)."""
    noise = noise_expectancy(cfg, meta)
    assert noise["passed"]
    assert noise["expectancy_r"] <= noise["tolerance_r"]


def test_no_future_universe_leakage(cfg, meta, ref_inputs):
    run = run_engine(cfg, meta, ref_inputs, label="universe")
    # SOL activates partway through; no symbol may be traded before activation.
    assert future_universe_violations(run.result, ref_inputs) == []
    late = [s for s in ref_inputs if s.activation_ts > 0]
    assert late, "fixture must exercise a point-in-time universe"
    for t in run.result.trades:
        act = {s.symbol: s.activation_ts for s in ref_inputs}[t.symbol]
        assert t.entry_ts >= act


def test_trend_series_has_positive_expectancy(cfg, meta, ref_inputs):
    # The planted causal edge is captured net of realistic costs (drives BT/WF).
    run = run_engine(cfg, meta, ref_inputs, label="trend")
    assert run.report.expectancy_r > 0.0
    assert run.report.net_pnl > 0.0


# --------------------------------------------------------------------------- #
# Report generator (Section 19: required outputs)                              #
# --------------------------------------------------------------------------- #
def test_report_contains_all_required_outputs(cfg, meta, ref_inputs):
    payload = run_engine(cfg, meta, ref_inputs, label="report").report.payload
    required = (
        "total_return",
        "net_pnl",
        "expectancy_r",
        "profit_factor",
        "max_drawdown",
        "trade_count",
        "symbol_breakdown",
        "strategy_breakdown",
        "regime_breakdown",
        "session_breakdown",
        "side_breakdown",
        "cost_breakdown",
        "slippage_breakdown",
        "funding_breakdown",
        "rejected_candidates",
        "worst_trades",
        "stability",
    )
    for key in required:
        assert key in payload, f"missing required output: {key}"
    assert payload["symbol_breakdown"], "per-symbol breakdown must be populated"
    assert payload["side_breakdown"]["long"]["trades"] >= 0
    assert payload["side_breakdown"]["short"]["trades"] >= 0


def test_max_drawdown_is_a_positive_fraction():
    assert max_drawdown([]) == 0.0
    assert max_drawdown([100.0, 110.0, 99.0, 120.0]) == pytest.approx((110.0 - 99.0) / 110.0)
    assert max_drawdown([100.0, 100.0, 100.0]) == 0.0


def test_cost_breakdown_reconciles_with_gross(cfg, meta, ref_inputs):
    run = run_engine(cfg, meta, ref_inputs, label="reconcile")
    cb = run.report.payload["cost_breakdown"]
    # net = gross - fees - funding (slippage is embedded in fill prices).
    assert cb["gross_pnl"] == pytest.approx(
        run.report.net_pnl + cb["total_fees"] + cb["total_funding"], rel=1e-6, abs=1e-3
    )


def test_run_reference_backtest_smoke(cfg, meta):
    run = run_reference_backtest(cfg, meta)
    assert run.report.trade_count > 0
    assert "stability" in run.report.payload


# --------------------------------------------------------------------------- #
# Walk-forward (Section 16) + stress (FEE / SLIP)                              #
# --------------------------------------------------------------------------- #
def test_walk_forward_passes_on_the_reference_edge(cfg, meta, ref_inputs):
    wf = run_walk_forward(cfg, meta, ref_inputs)
    assert len(wf.folds) == cfg.walk_forward.folds
    assert wf.folds_passed >= cfg.walk_forward.kill_criteria.min_folds_passed
    assert wf.holdout is not None and wf.holdout.passed  # locked hold-out evaluated once
    assert wf.passed


def test_walk_forward_fails_clearly_when_folds_are_too_thin(cfg, meta, ref_inputs):
    """A too-short window can't realize enough trades per fold to judge the edge — WF must FAIL
    with a clear trade-based 'insufficient trades' reason (not pass on noise, not a bars guess)."""
    from src.backtest.service import rebase_window

    iv = int(ref_inputs[0].bars[1]["ts"] - ref_inputs[0].bars[0]["ts"])
    tiny = rebase_window(ref_inputs, 0, 12 * iv)  # ~12 bars total → folds far below min trades
    wf = run_walk_forward(cfg, meta, tiny)
    assert not wf.passed
    assert any("insufficient trades" in r for r in wf.reasons), wf.reasons


def test_walk_forward_folds_are_disjoint_and_ordered(cfg, meta, ref_inputs):
    wf = run_walk_forward(cfg, meta, ref_inputs)
    for prev, nxt in zip(wf.folds, wf.folds[1:], strict=False):
        assert prev.hi_ts <= nxt.lo_ts  # no overlap, time-ordered
    # The locked hold-out sits after every fold (never seen during folds).
    assert wf.holdout.lo_ts >= wf.folds[-1].hi_ts


def test_fee_stress_survives_double_fees(cfg, meta, ref_inputs):
    res = fee_stress(cfg, meta, ref_inputs)
    assert res.kind == "fee" and res.multiplier == cfg.stress.fee_multiplier
    assert res.survives  # positive expectancy net of doubled fees
    # Doubling fees cannot improve the edge.
    assert res.stressed_expectancy_r <= res.baseline_expectancy_r + 1e-9


def test_slippage_stress_survives_harsher_slippage(cfg, meta, ref_inputs):
    res = slippage_stress(cfg, meta, ref_inputs)
    assert res.kind == "slippage" and res.multiplier == cfg.stress.slippage_multiplier
    assert res.survives
    assert res.stressed_expectancy_r <= res.baseline_expectancy_r + 1e-9


def test_config_with_cost_overrides_is_isolated(cfg):
    stressed = cfg.with_cost_overrides(fee_multiplier=2.0)
    assert stressed.costs.fee_multiplier == 2.0
    assert cfg.costs.fee_multiplier == 1.0  # original is frozen / untouched
