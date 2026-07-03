"""Section 16 anti-overfitting controls + Section 17 risk-checklist breakers."""

from __future__ import annotations

from src.backtest.overfitting import (
    deflated_sharpe_from_returns,
    deflated_sharpe_ratio,
    effective_sample_size,
    probabilistic_sharpe_ratio,
    purged_kfold_indices,
    sample_adequacy,
    trade_sharpe,
)
from src.risk.breakers import BreakerInputs, CircuitBreakers
from src.risk.config import load_risk_config


# --------------------------------------------------------------------------- #
# Section 16 — anti-overfitting                                               #
# --------------------------------------------------------------------------- #
def test_psr_in_unit_interval_and_monotone() -> None:
    weak = probabilistic_sharpe_ratio(0.1, 0.0, 100)
    strong = probabilistic_sharpe_ratio(1.0, 0.0, 100)
    assert 0.0 <= weak <= strong <= 1.0


def test_deflated_sharpe_penalises_more_trials() -> None:
    few = deflated_sharpe_ratio([1.0, -1.0], 100)
    many = deflated_sharpe_ratio([1.0, -1.0] * 25, 100)  # same best, 25x the trials
    assert 0.0 <= many < few <= 1.0  # more trials searched ⇒ higher bar ⇒ lower DSR


def test_deflated_sharpe_single_strong_trial_is_high() -> None:
    assert deflated_sharpe_ratio([2.5], 200) > 0.9


def test_effective_sample_size_discounts_autocorrelation() -> None:
    iid_like = effective_sample_size([0.01, -0.01] * 50)  # near-zero/neg autocorr
    autocorr = effective_sample_size([1.0] * 10 + [-1.0] * 10)  # strong positive autocorr
    assert autocorr < 5.0  # 20 serially-dependent obs are worth only a handful
    assert iid_like <= 100.0


def test_purged_kfold_has_no_train_test_overlap_and_partitions_test() -> None:
    folds = purged_kfold_indices(100, 5, embargo_frac=0.02)
    assert len(folds) == 5
    all_test: list[int] = []
    for train, test in folds:
        assert set(train).isdisjoint(test)  # purge/embargo removed any overlap
        all_test += test
    assert sorted(all_test) == list(range(100))  # test folds partition the series


def test_deflated_sharpe_from_returns_rewards_sample_size() -> None:
    """H3 regression: the gate statistic is computed from the POOLED per-trade sample, so the
    same per-trade edge on 500 trades must be more significant than on 25 trades — fold means
    alone can no longer fake significance."""
    edge = [0.5, -0.3, 0.4, -0.2, 0.6]  # positive-mean per-trade R pattern
    small = deflated_sharpe_from_returns(edge * 5)  # 25 trades
    large = deflated_sharpe_from_returns(edge * 100)  # 500 trades
    assert 0.0 <= small < large <= 1.0


def test_deflated_sharpe_from_returns_deflates_for_selection_breadth() -> None:
    """The expected-max benchmark of the trials genuinely compared raises the bar: the same
    pooled sample scores LOWER when it was the winner of a wide, dispersed search."""
    returns = [0.5, -0.3, 0.4, -0.2, 0.6] * 40
    single = deflated_sharpe_from_returns(returns)  # one config ⇒ plain PSR
    searched = deflated_sharpe_from_returns(returns, trial_sharpes=[0.08, -0.02, 0.05, -0.06])
    assert 0.0 <= searched < single <= 1.0


def test_deflated_sharpe_from_returns_neutral_and_degenerate_cases() -> None:
    assert deflated_sharpe_from_returns([]) == 0.0
    assert deflated_sharpe_from_returns([0.1]) == 0.0  # one trade proves nothing
    # A zero-mean sample sits at the 0.5 neutral point (no significance either way).
    assert deflated_sharpe_from_returns([0.2, -0.2] * 50) == 0.5


def test_trade_sharpe_basic_properties() -> None:
    assert trade_sharpe([]) == 0.0
    assert trade_sharpe([0.5]) == 0.0
    assert trade_sharpe([0.1, 0.1]) == 0.0  # zero variance guard
    assert trade_sharpe([0.5, -0.3, 0.4]) > 0.0 > trade_sharpe([-0.5, 0.3, -0.4])


def test_sample_adequacy_thresholds() -> None:
    assert sample_adequacy(500) == "robust"
    assert sample_adequacy(150) == "limited"
    assert sample_adequacy(50) == "minimal"
    assert sample_adequacy(10) == "inconclusive"


# --------------------------------------------------------------------------- #
# Section 17 — additional breakers + pre-trade checks                         #
# --------------------------------------------------------------------------- #
def _breakers() -> CircuitBreakers:
    return CircuitBreakers(load_risk_config())


def _inp(**over) -> BreakerInputs:
    base = {"equity": 10_000.0, "peak_equity": 10_000.0, "daily_pnl": 0.0}
    base.update(over)
    return BreakerInputs(**base)


def test_weekly_loss_breaker_trips() -> None:
    cfg = load_risk_config()
    loss = -(cfg.breakers.weekly_loss_limit + 0.01) * 10_000.0
    v = _breakers().evaluate(_inp(weekly_pnl=loss))
    assert v.tripped and "weekly_loss_limit" in v.reason


def test_funding_breaker_trips() -> None:
    cfg = load_risk_config()
    paid = (cfg.breakers.funding_breaker_limit + 0.005) * 10_000.0
    v = _breakers().evaluate(_inp(cumulative_funding_paid=paid))
    assert v.tripped and "funding_breaker" in v.reason


def test_per_symbol_loss_breaker_names_the_symbol() -> None:
    cfg = load_risk_config()
    loss = -(cfg.breakers.per_symbol_loss_limit + 0.01) * 10_000.0
    v = _breakers().evaluate(_inp(per_symbol_pnl={"ETH/USDT:USDT": loss}))
    assert v.tripped and "per_symbol_loss[ETH/USDT:USDT]" in v.reason


def test_no_trip_when_within_limits() -> None:
    assert not _breakers().evaluate(_inp(weekly_pnl=-100.0, cumulative_funding_paid=50.0)).tripped


def test_liquidation_distance_check() -> None:
    b = _breakers()  # min_liquidation_distance default 0.10
    assert b.liquidation_distance_ok(100.0, 80.0, side=1) is True  # 20% away → ok
    assert b.liquidation_distance_ok(100.0, 95.0, side=1) is False  # 5% away → too close


def test_margin_availability_check() -> None:
    b = _breakers()  # min_free_margin_frac default 0.20
    assert b.margin_available(1_000.0, 5_000.0, 10_000.0) is True  # leaves 4000 > 2000
    assert b.margin_available(4_500.0, 5_000.0, 10_000.0) is False  # leaves 500 < 2000


def test_daily_loss_still_takes_priority_over_weekly() -> None:
    cfg = load_risk_config()
    big_daily = -(cfg.envelope.daily_loss_limit + 0.01) * 10_000.0
    v = _breakers().evaluate(_inp(daily_pnl=big_daily, weekly_pnl=big_daily))
    assert v.tripped and "daily_loss_limit" in v.reason  # evaluated before weekly
