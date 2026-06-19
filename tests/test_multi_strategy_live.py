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
def test_active_set_is_top_n_by_expectancy(monkeypatch) -> None:
    # The registry is filtered to KNOWN config candidates, so use the 3 real ids and force the
    # cap to 2 to exercise the top-N boundary (on a unique version to isolate the shared DB).
    import dataclasses

    from src.strategies.config import load_strategies_config

    capped = dataclasses.replace(load_strategies_config(), max_active_strategies=2)
    monkeypatch.setattr("src.strategies.config.load_strategies_config", lambda *a, **k: capped)

    ver = f"msver_{uuid.uuid4().hex[:8]}"
    scores = {"basis_reversion": 0.10, "xsection_rs": 0.20, "lead_lag_xasset": 0.30}
    for cid, e in scores.items():
        _promote(cid, expectancy_r=e, version=ver)
    details = {d.candidate_id: d for d in promoted_strategy_details(ver)}
    assert set(details) == set(scores)  # all known + promoted; ranked by expectancy
    active = active_strategy_ids(ver)
    assert len(active) == 2  # capped to the top-2
    assert "lead_lag_xasset" in active and "xsection_rs" in active  # highest two
    assert "basis_reversion" not in active  # lowest is benched
    assert details["lead_lag_xasset"].active and not details["basis_reversion"].active


@requires_db
def test_unknown_and_shelved_strategies_never_active() -> None:
    ver = f"msver_{uuid.uuid4().hex[:8]}"
    # Unknown candidate id (not in config) is filtered out even if promoted=True...
    _promote("not_a_real_candidate", expectancy_r=0.9, version=ver, promoted=True)
    # ...and a real candidate that is NOT promoted never appears either.
    _promote("basis_reversion", expectancy_r=0.5, version=ver, promoted=False)
    assert active_strategy_ids(ver) == []
    assert promoted_strategy_details(ver) == []


@requires_db
def test_resolve_active_includes_cross_asset_strategies() -> None:
    """Both per-row AND cross-asset (portfolio) promoted strategies are returned — the realtime
    live feed runs both; only ids no longer in config are skipped."""
    from src.paper.lake import build_active_lake_inputs, resolve_active_strategies

    ver = get_settings().strategy_version
    # Dominant expectancy so these three are guaranteed in the top-N regardless of any other
    # promoted rows in the shared dev DB. (resolve_active_strategies reads the default version.)
    for cid in ("basis_reversion", "lead_lag_xasset", "xsection_rs"):
        _promote(cid, expectancy_r=99.0, version=ver)
    active, skipped = resolve_active_strategies()
    active_ids = {sid for _s, sid, _v in active}
    assert {"basis_reversion", "lead_lag_xasset", "xsection_rs"} <= active_ids  # all run live
    assert "lead_lag_xasset" not in skipped  # cross-asset is no longer skipped

    # The OFFLINE replay builder, by contrast, still includes only per-row strategies (it cannot
    # run cross-asset per-row); with no real lake data it returns empty cleanly.
    portfolio_strats = {
        sid for s, sid, _v in active if hasattr(s, "evaluate_portfolio")
    }
    assert {"lead_lag_xasset", "xsection_rs"} <= portfolio_strats
    inputs, ids = build_active_lake_inputs(
        DataConfig(
            exchange_id=EX, data_version="t", symbols=[SYM], timeframes=[TF], base_timeframe=TF,
            funding_interval_hours=8, required_series=[OHLCV], window_start_ms=0,
            window_end_ms=SEED_END, thresholds=ValidationThresholds(), oi_timeframe="1h",
        ),
        timeframe=TF, symbols=[SYM],
    )
    assert "lead_lag_xasset" not in ids and "xsection_rs" not in ids  # replay skips portfolio


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


class _PortfolioStrat:
    """A cross-asset strategy: signals only when it can see peer symbols (evaluate_portfolio)."""

    def evaluate_portfolio(self, symbol, row, peers):  # noqa: ANN001
        return _Sig(1) if peers else None  # no cross-asset signal without peers


def test_realtime_feed_runs_cross_asset_portfolio_strategy() -> None:
    src = DeterministicSource(EX)
    syms = ["BTC/USDT:USDT", "ETH/USDT:USDT"]
    by_sym = {s: src.fetch(SeriesKey(EX, OHLCV, s, TF), SEED_END, SEED_END + 8 * IV) for s in syms}

    class _MultiScripted:
        def __init__(self, seqs):
            self._seq = {s: [(int(b["ts"]), b) for b in bars] for s, bars in seqs.items()}
            self._i = dict.fromkeys(seqs, 0)

        def connected(self):
            return True

        def latest_bar(self, symbol):
            seq, i = self._seq[symbol], self._i[symbol]
            if i >= len(seq):
                return seq[-1] if seq else None
            self._i[symbol] += 1
            return seq[i]

        def backfill(self, *a, **k):
            return []

    cfg = DataConfig(
        exchange_id=EX, data_version="t", symbols=syms, timeframes=[TF], base_timeframe=TF,
        funding_interval_hours=8, required_series=[OHLCV], window_start_ms=0,
        window_end_ms=SEED_END, thresholds=ValidationThresholds(), oi_timeframe="1h",
    )
    feed = LiveCandidateFeed(
        cfg,
        feed_source=_MultiScripted(by_sym),
        rest_source=src,
        timeframe=TF,
        symbols=syms,
        strategies=[(_PortfolioStrat(), "lead_lag", "v1")],  # cross-asset strategy runs live
        settings=Settings(_env_file=None),
        seed_end_ms=SEED_END,
        max_groups=6,
    )
    groups = list(feed.groups())
    assert groups, "the cross-asset strategy should fire once peer symbols are visible"
    cands = [g.candidate for _ts, grp in groups for g in grp]
    assert cands and all(c.strategy == "lead_lag" for c in cands)
    assert all(c.symbol in syms for c in cands)
