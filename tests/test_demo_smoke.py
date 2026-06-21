"""Safe demo smoke test (Section 35): bounded, readiness-gated, single-order exercise of the
real demo/testnet path. Proves it places NOTHING unless readiness PASSes, refuses live, and on
the happy path places one minimal order, reconciles, and cleans up."""

from __future__ import annotations

import uuid

from src.config import Settings
from src.exchange.metadata import load_metadata_config
from src.execution.live_venue import CcxtLiveVenue
from src.killswitch import KillSwitch
from src.live.smoke import run_demo_smoke

from tests.conftest import requires_db

_PREFIX = "QBOT_LOCAL_v1_"


def _settings(**over) -> Settings:
    base = {
        "_env_file": None,
        "exchange_env": "demo",
        "exchange_api_key": "k",
        "exchange_api_secret": "s",
        "order_client_id_prefix": _PREFIX,
    }
    base.update(over)
    return Settings(**base)


class FakeCcxt:
    def __init__(self) -> None:
        self.orders: list[dict] = []
        self.closed: list[dict] = []

    def create_order(self, symbol, type, side, qty, price, params=None):  # noqa: A002
        rec = {"symbol": symbol, "side": side, "params": params or {}}
        if (params or {}).get("reduceOnly"):
            self.closed.append(rec)
        else:
            self.orders.append(rec)
        return {"average": price or 5_000.0, "filled": qty, "fee": {"cost": 0.0}}

    def cancel_order(self, *a, **k):
        return {}

    def fetch_positions(self):
        return []

    def fetch_open_orders(self):
        return []


def test_smoke_aborts_on_unverified_bybit_metadata() -> None:
    """Bybit demo metadata ships unverified → readiness BLOCKED → smoke places nothing."""
    fake = FakeCcxt()
    settings = _settings(exchange_id="bybit")
    venue = CcxtLiveVenue(load_metadata_config(), settings, client=fake)  # skeleton meta != bybit
    result = run_demo_smoke(settings, venue=venue)
    assert result.placed == 0
    assert "not PASS" in result.aborted_reason
    assert not fake.orders  # nothing sent to the exchange


def test_smoke_refuses_live_environment() -> None:
    settings = _settings(exchange_env="live", exchange_id="bybit")
    result = run_demo_smoke(settings, venue=FakeCcxt())
    assert result.placed == 0
    assert "live" in result.aborted_reason


@requires_db
def test_smoke_happy_path_places_one_order_and_cleans_up() -> None:
    """With all controls green (skeleton verified metadata, real-data strategy, clean book,
    disengaged kill switch) the smoke places exactly one bracket order and closes it."""
    from src.strategies.promotion import persist_validations
    from src.strategies.research import CandidateValidation, SideDecision

    KillSwitch(_settings()).disengage()

    ver = f"strat_test_{uuid.uuid4().hex[:6]}"
    sd = SideDecision(
        allow_long=True, allow_short=False, long_expectancy_r=0.2, short_expectancy_r=-0.1,
        long_trades=30, short_trades=5, disabled=["short"],
    )
    cand = CandidateValidation(
        candidate_id="basis_reversion", family="B", strategy_version=ver, promoted=True,
        status="promoted", shelved_reasons=[], side_decision=sd, hypothesis={},
        report={"expectancy_r": 0.2}, walk_forward={}, fee_stress={}, slippage_stress={},
        noise_control={},
    )
    assert persist_validations([cand], data_source="lake") == 1

    settings = _settings(exchange_id="skeleton", strategy_version=ver)
    fake = FakeCcxt()
    venue = CcxtLiveVenue(load_metadata_config(), settings, client=fake)
    result = run_demo_smoke(settings, venue=venue, cleanup=True)

    assert result.readiness.verdict == "PASS", result.report()
    assert result.placed == 1
    assert len(fake.orders) == 1  # exactly one entry order
    assert fake.orders[0]["params"].get("clientOrderId", "").startswith(_PREFIX)
    # mandatory exchange-resident SL + TP attached
    assert "stopLoss" in fake.orders[0]["params"]
    assert "takeProfit" in fake.orders[0]["params"]
    assert result.cleaned_up >= 1 and fake.closed  # position closed on cleanup
    assert not result.halted


@requires_db
def test_smoke_no_cleanup_leaves_position_open() -> None:
    from src.strategies.promotion import persist_validations
    from src.strategies.research import CandidateValidation, SideDecision

    KillSwitch(_settings()).disengage()
    ver = f"strat_test_{uuid.uuid4().hex[:6]}"
    sd = SideDecision(
        allow_long=True, allow_short=False, long_expectancy_r=0.2, short_expectancy_r=-0.1,
        long_trades=30, short_trades=5, disabled=["short"],
    )
    cand = CandidateValidation(
        candidate_id="basis_reversion", family="B", strategy_version=ver, promoted=True,
        status="promoted", shelved_reasons=[], side_decision=sd, hypothesis={},
        report={"expectancy_r": 0.2}, walk_forward={}, fee_stress={}, slippage_stress={},
        noise_control={},
    )
    persist_validations([cand], data_source="lake")
    settings = _settings(exchange_id="skeleton", strategy_version=ver)
    fake = FakeCcxt()
    venue = CcxtLiveVenue(load_metadata_config(), settings, client=fake)
    result = run_demo_smoke(settings, venue=venue, cleanup=False)
    assert result.placed == 1
    assert result.cleaned_up == 0 and not fake.closed
