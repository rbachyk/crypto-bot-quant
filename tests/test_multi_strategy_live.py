"""Multi-strategy live/demo engine (AGENTS.md §13 / §17).

Demo behaves exactly like live: the engine runs the ACTIVE PROMOTED ensemble — the top-N
promoted strategies by validated expectancy — concurrently, and their signals are arbitrated by
ranking + the one-position-per-symbol cap. These tests cover the top-N selection, the
portfolio-strategy skip, the per-bar multi-strategy grouping, and the empty-when-nothing-promoted
behaviour.
"""

from __future__ import annotations

import uuid

from src.config import Settings, get_settings
from src.data.config import DataConfig, ValidationThresholds
from src.data.schema import OHLCV, SeriesKey, timeframe_ms
from src.data.source import DeterministicSource
from src.live.realtime import LiveCandidateFeed
from src.strategies.promotion import (
    active_strategy_ids,
    persist_validations,
    promoted_strategy_details,
)
from src.strategies.research import CandidateValidation, SideDecision

from tests.conftest import requires_db

EX = "bybit"
SYM = "BTC/USDT:USDT"
TF = "5m"
IV = timeframe_ms(TF)
SEED_END = 300 * IV


# --------------------------------------------------------------------------- #
# Top-N active-strategy selection (the cap that keeps the live ensemble small) #
# --------------------------------------------------------------------------- #
def _promote(
    candidate_id: str, expectancy_r: float, *, version: str, promoted: bool = True
) -> None:
    sd = SideDecision(
        allow_long=True, allow_short=False, long_expectancy_r=expectancy_r,
        short_expectancy_r=0.0, long_trades=40, short_trades=0, disabled=[],
    )
    persist_validations(
        [
            CandidateValidation(
                candidate_id=candidate_id, family="B", strategy_version=version,
                promoted=promoted, status="promoted" if promoted else "shelved",
                shelved_reasons=[], side_decision=sd, hypothesis={},
                report={"expectancy_r": expectancy_r}, walk_forward={}, fee_stress={},
                slippage_stress={}, noise_control={},
            )
        ]
    )


@requires_db
def test_active_set_is_top_n_by_expectancy() -> None:
    # Isolate from the shared dev DB by pinning these promotions to a unique strategy_version,
    # so only our 7 candidates compete for the top-5 active slots (cap = max_active_strategies).
    ver = f"msver_{uuid.uuid4().hex[:8]}"
    ids = [f"ms_{i}" for i in range(7)]
    for i, cid in enumerate(ids):
        _promote(cid, expectancy_r=0.01 * (i + 1), version=ver)  # ascending expectancy
    details = {d.candidate_id: d for d in promoted_strategy_details(ver)}
    assert len(details) == 7
    active = active_strategy_ids(ver)
    assert len(active) == 5  # capped to the top-5
    assert ids[6] in active and ids[2] in active  # highest survive
    assert ids[0] not in active and ids[1] not in active  # lowest two benched
    assert details[ids[6]].active and not details[ids[0]].active


@requires_db
def test_shelved_strategies_never_active() -> None:
    ver = f"msver_{uuid.uuid4().hex[:8]}"
    cid = "shelved_one"
    _promote(cid, expectancy_r=0.5, version=ver, promoted=False)  # high score but NOT promoted
    assert active_strategy_ids(ver) == []
    assert promoted_strategy_details(ver) == []


@requires_db
def test_resolve_active_skips_cross_asset_strategies() -> None:
    """Promoted family-A/G (portfolio) strategies are skipped by the per-row live path."""
    from src.paper.lake import resolve_active_strategies

    ver = get_settings().strategy_version
    # Dominant expectancy so these three are guaranteed in the top-N regardless of any other
    # promoted rows in the shared dev DB. (resolve_active_strategies reads the default version.)
    for cid in ("basis_reversion", "lead_lag_xasset", "xsection_rs"):
        _promote(cid, expectancy_r=99.0, version=ver)
    active, skipped = resolve_active_strategies()
    active_ids = {sid for _s, sid, _v in active}
    assert "basis_reversion" in active_ids  # single-symbol family B runs
    assert "lead_lag_xasset" in skipped and "xsection_rs" in skipped  # cross-asset skipped


# --------------------------------------------------------------------------- #
# Real-time feed runs the whole ensemble per bar (one group, many candidates)  #
# --------------------------------------------------------------------------- #
class _Sig:
    def __init__(self, side: int) -> None:
        self.side, self.stop_frac, self.tp_frac = side, 0.01, 0.02


class _FakeStrat:
    """A per-row strategy that always signals (no evaluate_portfolio → per-row path)."""

    def __init__(self, side: int = 1) -> None:
        self._side = side

    def evaluate(self, row) -> _Sig:
        return _Sig(self._side)


class _ScriptedFeedSource:
    def __init__(self, bars: list[dict]) -> None:
        self._seq = [(int(b["ts"]), b) for b in bars]
        self._i = 0

    def connected(self) -> bool:
        return True

    def latest_bar(self, symbol):
        if self._i >= len(self._seq):
            return self._seq[-1] if self._seq else None
        item = self._seq[self._i]
        self._i += 1
        return item

    def backfill(self, *a, **k):
        return []


def _cfg() -> DataConfig:
    return DataConfig(
        exchange_id=EX, data_version="t", symbols=[SYM], timeframes=[TF], base_timeframe=TF,
        funding_interval_hours=8, required_series=[OHLCV], window_start_ms=0,
        window_end_ms=SEED_END, thresholds=ValidationThresholds(), oi_timeframe="1h",
    )


def test_realtime_feed_runs_every_active_strategy_per_bar() -> None:
    src = DeterministicSource(EX)
    bars = src.fetch(SeriesKey(EX, OHLCV, SYM, TF), SEED_END, SEED_END + 12 * IV)
    strategies = [(_FakeStrat(1), "alpha", "v1"), (_FakeStrat(-1), "beta", "v1")]
    feed = LiveCandidateFeed(
        _cfg(),
        feed_source=_ScriptedFeedSource(bars),
        rest_source=src,
        timeframe=TF,
        symbols=[SYM],
        strategies=strategies,
        settings=Settings(_env_file=None),  # don't depend on the developer's .env
        seed_end_ms=SEED_END,
        max_groups=8,
    )
    groups = list(feed.groups())
    assert groups, "active strategies should fire on streamed bars"
    for _ts, grp in groups:
        # Both active strategies signal on the same bar → one group, two candidates, same symbol.
        assert len(grp) == 2
        assert {g.candidate.strategy for g in grp} == {"alpha", "beta"}
        assert all(g.candidate.symbol == SYM for g in grp)
        assert all(g.candidate.config_live_approved for g in grp)  # active = promoted
