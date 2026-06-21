"""Risk Manager unit tests (AGENTS.md Section 17 / Section 2.2).

Offline tests of the capital-critical risk module: deterministic per-trade
sizing, leverage-as-consequence, portfolio heat + net-beta caps, the min-notional
gate, and every circuit breaker — including the deliberate forced-failure trips
the RISK gate relies on. No DB/Redis required.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from src.config import Settings
from src.exchange.metadata import load_metadata_config
from src.killswitch import KillSwitch
from src.ranking import Candidate
from src.risk import (
    AccountState,
    BreakerInputs,
    PortfolioState,
    Position,
    RiskEnvelope,
    RiskManager,
    load_risk_config,
)
from src.risk.envelope import HARD_CEILINGS

BTC = "BTC/USDT:USDT"
EQUITY = 100_000.0


def _meta():
    return load_metadata_config()


def _rm(kill_switch=None):
    return RiskManager(load_risk_config(), _meta(), kill_switch=kill_switch)


def _cand(symbol=BTC, *, side=1, stop_frac=0.02, entry_price=50_000.0, regime="low_vol_up"):
    return Candidate(
        symbol=symbol,
        strategy="t",
        strategy_version="t",
        side=side,
        entry_price=entry_price,
        stop_frac=stop_frac,
        tp_frac=0.04,
        regime=regime,
        session=2,
    )


def _flat(equity=EQUITY, **breaker):
    # daily_pnl defaults to flat (0.0) but stays overridable via **breaker so breaker-trip
    # tests can force it negative without a "multiple values for 'daily_pnl'" collision.
    breaker.setdefault("daily_pnl", 0.0)
    return AccountState(
        portfolio=PortfolioState(equity=equity),
        breakers=BreakerInputs(equity=equity, peak_equity=equity, **breaker),
    )


# --------------------------------------------------------------------------- #
# Envelope (immutable box)                                                     #
# --------------------------------------------------------------------------- #
def test_envelope_config_cannot_widen_past_code_ceilings() -> None:
    env = RiskEnvelope.from_config(
        {
            "max_leverage": 9999,
            "max_risk_pct_per_trade": 5.0,
            "portfolio_heat_cap": 1.0,
            "net_beta_btc_cap": 5.0,
            "daily_loss_limit": 1.0,
            "max_drawdown_limit": 1.0,
        }
    )
    assert env.max_leverage == HARD_CEILINGS["max_leverage"]
    assert env.max_risk_pct_per_trade == HARD_CEILINGS["max_risk_pct_per_trade"]
    assert env.net_beta_btc_cap == HARD_CEILINGS["net_beta_btc_cap"]


def test_envelope_config_can_tighten() -> None:
    env = RiskEnvelope.from_config({"max_leverage": 3, "max_risk_pct_per_trade": 0.005})
    assert env.max_leverage == 3.0
    assert env.max_risk_pct_per_trade == 0.005


def test_envelope_missing_field_fails_closed_not_to_ceiling() -> None:
    # A missing/invalid envelope field must fall back to the conservative SAFE_DEFAULT
    # (tighter), NEVER silently widen to the loosest legal value (the hard ceiling).
    from src.risk.envelope import SAFE_DEFAULTS

    env = RiskEnvelope.from_config({})  # everything missing
    for field, default in SAFE_DEFAULTS.items():
        got = getattr(env, field)
        assert got == default, f"{field}: missing config gave {got}, expected default {default}"
        assert got <= HARD_CEILINGS[field]  # default never exceeds the ceiling
    # Concretely: a missing risk cap must be 1%, not the 2% ceiling.
    assert env.max_risk_pct_per_trade == 0.01
    # Invalid values (non-numeric, <=0) also fail closed to the default.
    bad = RiskEnvelope.from_config({"max_leverage": -5, "daily_loss_limit": "oops"})
    assert bad.max_leverage == SAFE_DEFAULTS["max_leverage"]
    assert bad.daily_loss_limit == SAFE_DEFAULTS["daily_loss_limit"]


def test_base_risk_pct_clamped_to_envelope() -> None:
    cfg = load_risk_config()
    assert cfg.base_risk_pct <= cfg.envelope.max_risk_pct_per_trade


# --------------------------------------------------------------------------- #
# Sizing                                                                       #
# --------------------------------------------------------------------------- #
def test_sizing_identity_when_no_cap_binds() -> None:
    cfg = load_risk_config()
    d = _rm().evaluate(_cand(stop_frac=0.02), _flat())
    assert d.approved
    # qty × |entry − stop| ≈ equity × risk_pct (within lot rounding).
    target = EQUITY * cfg.base_risk_pct
    assert abs(d.risk_amount - target) <= target * 0.05
    assert d.risk_pct_used <= cfg.envelope.max_risk_pct_per_trade + 1e-9


def test_leverage_is_capped_not_targeted() -> None:
    cfg = load_risk_config()
    d = _rm().evaluate(_cand(stop_frac=0.0001), _flat())  # tiny stop → huge notional
    assert d.approved
    assert d.leverage <= cfg.envelope.max_leverage + 1e-9
    assert "leverage_capped" in d.reasons


def test_min_notional_rejected() -> None:
    small = AccountState(
        portfolio=PortfolioState(equity=100.0),
        breakers=BreakerInputs(equity=100.0, peak_equity=100.0, daily_pnl=0.0),
    )
    d = _rm().evaluate(_cand("SOL/USDT:USDT", stop_frac=0.05, entry_price=150.0), small)
    assert not d.approved
    assert any("below_min" in r for r in d.reasons)


def test_risk_pct_never_exceeds_envelope() -> None:
    cfg = load_risk_config()
    for sf in (0.002, 0.005, 0.01, 0.05, 0.1):
        d = _rm().evaluate(_cand(stop_frac=sf), _flat())
        if d.approved:
            assert d.risk_pct_used <= cfg.envelope.max_risk_pct_per_trade + 1e-9


# --------------------------------------------------------------------------- #
# Portfolio caps                                                               #
# --------------------------------------------------------------------------- #
def test_heat_cap_resizes_to_fit() -> None:
    cfg = load_risk_config()
    cap = cfg.envelope.portfolio_heat_cap
    preload = Position(
        symbol="ETH/USDT:USDT",
        side=1,
        qty=0.01,
        entry_price=3000.0,
        risk_amount=EQUITY * (cap - 0.002),
        beta_to_btc=0.85,
    )
    state = AccountState(
        portfolio=PortfolioState(equity=EQUITY, positions=(preload,)),
        breakers=BreakerInputs(equity=EQUITY, peak_equity=EQUITY, daily_pnl=0.0),
    )
    d = _rm().evaluate(_cand(stop_frac=0.02), state)
    assert d.approved and "heat_capped" in d.reasons
    total_heat = state.portfolio.heat() + d.risk_amount / EQUITY
    assert total_heat <= cap + 1e-9


def test_net_beta_cap_rejects_when_full_and_resizes_otherwise() -> None:
    cfg = load_risk_config()
    cap = cfg.envelope.net_beta_btc_cap
    over = Position(
        symbol="ETH/USDT:USDT",
        side=1,
        qty=12.0,
        entry_price=3000.0,
        risk_amount=100.0,
        beta_to_btc=0.85,
    )  # ~0.306 net beta > cap
    full = AccountState(
        portfolio=PortfolioState(equity=EQUITY, positions=(over,)),
        breakers=BreakerInputs(equity=EQUITY, peak_equity=EQUITY, daily_pnl=0.0),
    )
    rejected = _rm().evaluate(_cand(side=1, stop_frac=0.02), full)
    assert not rejected.approved and any("net_beta_cap" in r for r in rejected.reasons)

    half = Position(
        symbol="ETH/USDT:USDT",
        side=1,
        qty=8.0,
        entry_price=3000.0,
        risk_amount=100.0,
        beta_to_btc=0.85,
    )  # ~0.204 net beta
    state = AccountState(
        portfolio=PortfolioState(equity=EQUITY, positions=(half,)),
        breakers=BreakerInputs(equity=EQUITY, peak_equity=EQUITY, daily_pnl=0.0),
    )
    d = _rm().evaluate(_cand(side=1, stop_frac=0.02), state)
    assert d.approved and "beta_capped" in d.reasons
    post_net = state.portfolio.net_beta() + d.notional / EQUITY  # BTC beta 1.0
    assert post_net <= cap + 1e-9


def test_concurrency_caps() -> None:
    cfg = load_risk_config()
    positions = tuple(
        Position(
            symbol=f"S{i}",
            side=1,
            qty=0.001,
            entry_price=100.0,
            risk_amount=1.0,
            beta_to_btc=0.0,
            regime=("a", "a", "b", "b", "c")[i],
        )
        for i in range(cfg.max_concurrent_total)
    )
    state = AccountState(
        portfolio=PortfolioState(equity=EQUITY, positions=positions),
        breakers=BreakerInputs(equity=EQUITY, peak_equity=EQUITY, daily_pnl=0.0),
    )
    d = _rm().evaluate(_cand(regime="z"), state)
    assert not d.approved and "max_concurrent_total" in d.reasons


def test_free_margin_blocker_rejects_when_buffer_breached() -> None:
    """The pre-trade free-margin blocker refuses an order that would breach the minimum free
    margin (active only when the account's free margin is known — a real venue)."""
    cand = _cand()
    tight = AccountState(
        portfolio=PortfolioState(equity=EQUITY),
        breakers=BreakerInputs(equity=EQUITY, peak_equity=EQUITY, daily_pnl=0.0),
        free_margin=100.0,  # almost no free margin
    )
    d = _rm().evaluate(cand, tight)
    assert not d.approved and "insufficient_free_margin" in d.reasons
    # Ample free margin → not blocked by this check.
    ample = AccountState(
        portfolio=PortfolioState(equity=EQUITY),
        breakers=BreakerInputs(equity=EQUITY, peak_equity=EQUITY, daily_pnl=0.0),
        free_margin=EQUITY,
    )
    assert _rm().evaluate(cand, ample).approved


def test_liquidation_distance_blocker_rejects_when_too_close() -> None:
    """The pre-trade liquidation-distance blocker refuses an entry whose (venue-provided)
    liquidation price sits closer than min_liquidation_distance; absent the price it's skipped."""
    cand = _cand(entry_price=50_000.0, side=1)
    close = AccountState(
        portfolio=PortfolioState(equity=EQUITY),
        breakers=BreakerInputs(equity=EQUITY, peak_equity=EQUITY, daily_pnl=0.0),
        liquidation_price=49_000.0,  # 2% away < 10% required
    )
    d = _rm().evaluate(cand, close)
    assert not d.approved and "liquidation_too_close" in d.reasons
    # A far liquidation price passes; no price (paper) skips the check.
    far = AccountState(
        portfolio=PortfolioState(equity=EQUITY),
        breakers=BreakerInputs(equity=EQUITY, peak_equity=EQUITY, daily_pnl=0.0),
        liquidation_price=40_000.0,  # 20% away
    )
    assert _rm().evaluate(cand, far).approved
    assert _rm().evaluate(cand, _flat()).approved  # liquidation_price None → skipped


def test_per_symbol_conflict() -> None:
    existing = Position(
        symbol=BTC,
        side=1,
        qty=0.1,
        entry_price=50000.0,
        risk_amount=10.0,
        beta_to_btc=1.0,
        regime="x",
    )
    state = AccountState(
        portfolio=PortfolioState(equity=EQUITY, positions=(existing,)),
        breakers=BreakerInputs(equity=EQUITY, peak_equity=EQUITY, daily_pnl=0.0),
    )
    d = _rm().evaluate(_cand(BTC), state)
    assert not d.approved and "open_position_conflict" in d.reasons


# --------------------------------------------------------------------------- #
# Circuit breakers (forced trips)                                             #
# --------------------------------------------------------------------------- #
def test_daily_loss_breaker_blocks() -> None:
    cfg = load_risk_config()
    d = _rm().evaluate(_cand(), _flat(daily_pnl=-EQUITY * (cfg.envelope.daily_loss_limit + 0.01)))
    assert d.action == "block" and "daily_loss" in (d.blocker or "")


def test_drawdown_breaker_blocks() -> None:
    cfg = load_risk_config()
    eq = EQUITY * (1.0 - cfg.envelope.max_drawdown_limit - 0.02)
    state = AccountState(
        portfolio=PortfolioState(equity=eq),
        breakers=BreakerInputs(equity=eq, peak_equity=EQUITY, daily_pnl=0.0),
    )
    d = _rm().evaluate(_cand(), state)
    assert d.action == "block" and "drawdown" in (d.blocker or "")


def test_consecutive_loss_cooldown_blocks() -> None:
    cfg = load_risk_config()
    d = _rm().evaluate(_cand(), _flat(consecutive_losses=cfg.breakers.consecutive_loss_limit))
    assert d.action == "block" and "consecutive_loss" in (d.blocker or "")


def test_reconciliation_and_unknown_order_block() -> None:
    recon = _rm().evaluate(_cand(), _flat(reconciled=False))
    assert recon.action == "block" and "reconciliation" in (recon.blocker or "")
    unknown = _rm().evaluate(
        _cand(),
        AccountState(
            portfolio=PortfolioState(equity=EQUITY),
            breakers=BreakerInputs(equity=EQUITY, peak_equity=EQUITY, daily_pnl=0.0),
            unknown_order_present=True,
        ),
    )
    assert unknown.action == "block" and unknown.blocker == "unknown_order_conflict"


def test_kill_switch_blocks_new_entries() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        ks = KillSwitch(
            Settings(
                _env_file=None, data_lake_path=Path(tmp) / "dl", redis_url="redis://127.0.0.1:1/0"
            )
        )
        rm = _rm(kill_switch=ks)
        assert rm.evaluate(_cand(), _flat()).approved
        ks.engage(reason="test", actor="test")
        blocked = rm.evaluate(_cand(), _flat())
        assert blocked.action == "block" and blocked.blocker == "kill_switch_engaged"
        ks.disengage(actor="test")
        assert rm.evaluate(_cand(), _flat()).approved


@pytest.mark.parametrize("equity", [0.0, -1.0])
def test_non_positive_equity_rejected(equity: float) -> None:
    state = AccountState(
        portfolio=PortfolioState(equity=equity),
        breakers=BreakerInputs(equity=equity, peak_equity=max(equity, 1.0), daily_pnl=0.0),
    )
    assert not _rm().evaluate(_cand(), state).approved
