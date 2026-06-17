"""Setup-quality + ranking unit tests (AGENTS.md Section 15 / Section 7).

Offline tests that scoring is deterministic and reproducible, that every hard
blocker is enforced (a high score never bypasses a blocker), and that the ranking
engine orders candidates and records the full multi-symbol attribution.
"""

from __future__ import annotations

from src.exchange.metadata import load_metadata_config
from src.ranking import (
    NO_TRADE_REGIMES,
    Candidate,
    CandidateRankingEngine,
    SetupContext,
    SetupQualityScorer,
    load_ranking_config,
)

BTC = "BTC/USDT:USDT"


def _scorer():
    return SetupQualityScorer(load_ranking_config(), load_metadata_config())


def _cand(symbol=BTC, **over) -> Candidate:
    base = {
        "symbol": symbol,
        "strategy": "t",
        "strategy_version": "t",
        "side": 1,
        "entry_price": 50_000.0,
        "stop_frac": 0.008,
        "tp_frac": 0.02,
        "regime": "low_vol_up",
        "session": 2,
        "features": {"atr_pct_rank": 0.2, "is_weekend": 0.0, "pre_funding": 0.0},
        "signal_strength": 0.9,
        "confirmation": 0.8,
        "expected_edge_frac": 0.01,
        "spread_bps": 3.0,
        "slippage_est": 0.0005,
        "latency_ms": 40.0,
    }
    base.update(over)
    return Candidate(**base)  # type: ignore[arg-type]


def test_score_is_deterministic_and_reproducible() -> None:
    c = _cand()
    s1 = _scorer().score(c)
    s2 = _scorer().score(c)
    assert s1.total == s2.total
    assert s1.components == s2.components


def test_components_within_max_and_sum_to_total() -> None:
    cfg = load_ranking_config()
    s = _scorer().score(_cand())
    assert len(s.components) == 7
    for name, max_pts in cfg.components.items():
        assert 0.0 <= s.components[name] <= max_pts
    assert abs(sum(s.components.values()) - s.total) < 1e-9
    assert s.total <= cfg.max_score


def test_good_setup_approved() -> None:
    s = _scorer().score(_cand())
    assert s.approved and s.passed_threshold and not s.blockers


def test_hard_blockers_cannot_be_bypassed_by_score() -> None:
    sc = _scorer()
    # A toxic-spread candidate may still score highly but must NOT be approved.
    toxic = _cand(spread_bps=80.0)
    s = sc.score(toxic)
    assert "spread_above_threshold" in s.blockers and not s.approved

    assert "no_trade_regime" in sc.score(_cand(regime=sorted(NO_TRADE_REGIMES)[0])).blockers
    assert "stale_data" in sc.score(_cand(data_fresh=False)).blockers
    assert "strategy_disabled" in sc.score(_cand(strategy_enabled=False)).blockers
    assert "symbol_halted_or_inactive" in sc.score(_cand(symbol_tradable=False)).blockers
    assert "negative_ev_after_costs" in sc.score(_cand(expected_edge_frac=0.0005)).blockers


def test_context_blockers() -> None:
    sc = _scorer()
    s = sc.score(_cand(), SetupContext(daily_loss_reached=True, foreign_order_detected=True))
    assert "daily_loss_limit_reached" in s.blockers
    assert "foreign_order_detected" in s.blockers
    assert not s.approved


def test_expected_value_after_costs() -> None:
    sc = _scorer()
    c = _cand(expected_edge_frac=0.01, slippage_est=0.0005)
    # round-trip cost = 2×taker(0.00055) + slippage(0.0005) = 0.0016
    assert abs(sc.round_trip_cost_frac(c) - 0.0016) < 1e-9
    assert abs(sc.expected_value_after_costs(c) - (0.01 - 0.0016)) < 1e-9


def test_below_threshold_not_approved() -> None:
    weak = _cand(
        signal_strength=0.0,
        confirmation=0.0,
        expected_edge_frac=0.0017,
        spread_bps=24.0,
        tp_frac=0.008,
    )
    s = _scorer().score(weak)
    # Weak setup: no blocker but below threshold ⇒ not approved.
    assert not s.passed_threshold and not s.approved


def test_ranking_orders_and_attributes() -> None:
    cfg = load_ranking_config()
    engine = CandidateRankingEngine(cfg, load_metadata_config())
    best = _cand(signal_strength=0.99, confirmation=0.95)
    mid = _cand(signal_strength=0.7, confirmation=0.6)
    blocked = _cand(spread_bps=90.0)
    result = engine.rank([mid, best, blocked])
    assert result.winner is not None
    assert result.winner.candidate is best
    assert [r.rank for r in result.selected] == [1, 2]
    attr = result.attribution()
    assert attr["winner"]["setup_quality_score"] == result.winner.score.total
    # The blocked candidate and the outranked one are recorded as alternatives.
    reasons = {r.reason.split("(")[0].split(":")[0] for r in result.rejected}
    assert "hard_blocker" in reasons
    assert any("outranked" in r.reason for r in result.rejected)


def test_ranking_is_deterministic() -> None:
    engine = CandidateRankingEngine(load_ranking_config(), load_metadata_config())
    cands = [_cand(signal_strength=x / 10) for x in range(3, 10)]
    r1 = engine.rank(cands)
    r2 = engine.rank(cands)
    assert [c.score.total for c in r1.selected] == [c.score.total for c in r2.selected]
