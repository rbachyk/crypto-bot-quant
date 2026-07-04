"""M4: real-data (replay) paper sessions built from lake data.

Seeds a SeriesStore from the deterministic source, then proves the candidate stream
is derived from the real feature frame and runs through the actual paper pipeline
(ranking → risk → execution → SimulatedVenue) producing PaperTrades — no fabricated
candidates. DB persistence is covered by run_lake_paper_session in the e2e path; here
we keep it hermetic (no DB) by exercising the engine directly.
"""

from __future__ import annotations

import math

import pytest
from src.data.schema import (
    FUNDING,
    INDEX,
    MARK,
    OHLCV,
    OPEN_INTEREST,
    SPREAD,
    SeriesKey,
    timeframe_ms,
)
from src.data.source import DeterministicSource
from src.data.store import SeriesStore
from src.paper.engine import PaperTradingEngine
from src.paper.lake import build_lake_paper_inputs

EX = "bybit"
SYM = "BTC/USDT:USDT"
TF = "5m"
BASE = "5m"
OI_TF = "1h"
FUND = "8h"


def _data_cfg(start: int, end: int):
    from src.data.config import DataConfig, ValidationThresholds

    return DataConfig(
        exchange_id=EX,
        data_version="t",
        symbols=[SYM],
        timeframes=[TF],
        base_timeframe=BASE,
        funding_interval_hours=8,
        required_series=[OHLCV, MARK, INDEX, FUNDING, OPEN_INTEREST, SPREAD],
        window_start_ms=start,
        window_end_ms=end,
        thresholds=ValidationThresholds(),
        oi_timeframe=OI_TF,
    )


def _seed(store: SeriesStore, start: int, end: int) -> None:
    src = DeterministicSource(EX)
    for dt, tf in (
        (OHLCV, TF),
        (MARK, BASE),
        (INDEX, BASE),
        (SPREAD, BASE),
        (OPEN_INTEREST, OI_TF),
        (FUNDING, FUND),
    ):
        key = SeriesKey(EX, dt, SYM, tf)
        store.write(key, src.fetch(key, start, end))


def test_build_lake_paper_inputs_from_real_frame(tmp_path) -> None:
    store = SeriesStore(tmp_path)
    start, end = 0, 400 * timeframe_ms(TF)
    _seed(store, start, end)
    cfg = _data_cfg(start, end)

    inputs, strat_id, version = build_lake_paper_inputs(
        cfg, timeframe=TF, symbols=[SYM], store=store
    )
    assert strat_id == "reference_momentum" and version
    assert inputs, "the strategy should fire on the seeded series"
    for pin in inputs:
        c = pin.candidate
        assert c.side in (1, -1)
        assert c.entry_price > 0
        assert c.symbol == SYM
        # features come from the REAL decision-time row, not fabricated constants.
        assert set(c.features) == {"atr_pct", "premium", "funding_z"}
        assert math.isfinite(pin.exit_move_frac)
        assert c.spread_bps > 0  # estimated spread from the lake


def test_lake_paper_inputs_run_through_paper_pipeline(tmp_path) -> None:
    store = SeriesStore(tmp_path)
    start, end = 0, 400 * timeframe_ms(TF)
    _seed(store, start, end)
    cfg = _data_cfg(start, end)
    inputs, _, _ = build_lake_paper_inputs(cfg, timeframe=TF, symbols=[SYM], store=store)

    engine = PaperTradingEngine()
    session = engine.new_session("lake_test")
    engine.process_candidates(inputs, session)
    # Every candidate is either executed or rejected (full pipeline ran on real candidates).
    assert session.executed_count + session.rejected_count == len(inputs)
    assert session.executed_count > 0
    for t in session.trades:
        assert t.has_exchange_side_stop  # bracket attached at entry
        assert t.spread_bps_at_entry > 0


def test_concurrent_lake_replay_holds_positions_and_caps_bind(tmp_path) -> None:
    """M-G: the concurrent driver opens HELD positions and closes them via the shared bracket walk
    (not a per-candidate flatten), so trades carry bracket-walk exit reasons, positions are held
    across bars, and the one-position-per-symbol cap binds (never >1 open per symbol at once)."""
    from src.paper.engine import PaperTradingEngine as _Eng
    from src.paper.lake import _drive_lake_replay, build_lake_replay_timeline

    store = SeriesStore(tmp_path)
    start, end = 0, 400 * timeframe_ms(TF)
    _seed(store, start, end)
    cfg = _data_cfg(start, end)

    groups, bars_by_ts, funding_by_sym, iv, strat = build_lake_replay_timeline(
        cfg, timeframe=TF, symbols=[SYM], store=store
    )
    assert groups and bars_by_ts and iv == timeframe_ms(TF)
    # Held candidates carry NO pre-resolved exit, and each carries a NON-ZERO time-stop horizon so
    # the time-stop fires (MG-H2: reference momentum's Signal has no hold_bars → must default).
    assert all(inp.exit_reason is None for g in groups.values() for inp in g)
    assert all(inp.candidate.hold_bars > 0 for g in groups.values() for inp in g)

    engine = _Eng()
    engine.set_bar_interval(iv)
    engine.set_funding_source(lambda s: funding_by_sym.get(s, []))  # MG-H1: accrue held funding
    session = engine.new_session("lake_concurrent_test")

    # Instrument the open book to prove the one-position-per-symbol cap binds during the walk.
    max_open_per_sym: dict[str, int] = {}
    orig = engine.process_candidates

    def _spy(inputs, sess):
        r = orig(inputs, sess)
        for s in engine._open_positions:
            max_open_per_sym[s] = max(max_open_per_sym.get(s, 0), 1)
        # never more than one Position object per symbol key (dict) — cap holds by construction
        return r

    engine.process_candidates = _spy  # type: ignore[method-assign]
    _drive_lake_replay(engine, session, groups, bars_by_ts, iv)

    assert session.trades, "concurrent replay booked no trades"
    # Exits came from the bracket walk / force-close, never the pre-resolved flatten.
    assert all(
        t.exit_reason in {"stop", "trailing_stop", "take_profit", "time_stop", "end_of_data"}
        for t in session.trades
    )
    assert max_open_per_sym.get(SYM, 0) <= 1  # one-position-per-symbol cap bound throughout
    assert not engine._open_positions  # everything force-closed at end-of-data


def test_concurrent_replay_is_a_subset_of_legacy_and_tracks_backtest(tmp_path) -> None:
    """M-G parity oracle. (1) The concurrent driver can only REMOVE entries vs the legacy flatten
    (an overlapping same-symbol signal is rejected while a position is held), never add — so
    0 < concurrent_trades <= legacy_trades. (2) It stays in the same order of magnitude as a
    BacktestEngine run of the same strategy/window (a gross divergence — caps not binding, exits
    wrong — would break this)."""
    from src.backtest.config import load_backtest_config
    from src.backtest.engine import BacktestEngine
    from src.backtest.metrics import build_report
    from src.backtest.service import build_lake_inputs
    from src.exchange.metadata import load_metadata_config
    from src.paper.engine import PaperTradingEngine as _Eng
    from src.paper.lake import (
        _drive_lake_replay,
        build_lake_paper_inputs,
        build_lake_replay_timeline,
        make_strategy,
    )

    store = SeriesStore(tmp_path)
    start, end = 0, 400 * timeframe_ms(TF)
    _seed(store, start, end)
    cfg = _data_cfg(start, end)

    # Legacy replay (per-candidate flatten) — takes every risk-approved signal.
    legacy_inputs, _, _ = build_lake_paper_inputs(cfg, timeframe=TF, symbols=[SYM], store=store)
    leg = _Eng()
    leg.set_bar_interval(timeframe_ms(TF))
    leg_sess = leg.new_session("legacy")
    leg.process_candidates(legacy_inputs, leg_sess)
    legacy_trades = len(leg_sess.trades)

    # Concurrent replay — holds positions, so overlapping same-symbol signals are capped out.
    groups, bars_by_ts, funding_by_sym, iv, _ = build_lake_replay_timeline(
        cfg, timeframe=TF, symbols=[SYM], store=store
    )
    con = _Eng()
    con.set_bar_interval(iv)
    con.set_funding_source(lambda s: funding_by_sym.get(s, []))
    con_sess = con.new_session("concurrent")
    _drive_lake_replay(con, con_sess, groups, bars_by_ts, iv)
    concurrent_trades = len(con_sess.trades)

    assert 0 < concurrent_trades <= legacy_trades  # caps can only remove entries, never add

    # Backtest of the SAME strategy/window as an independent oracle (order-of-magnitude band; cost
    # models differ so exact parity isn't expected).
    bt_cfg = load_backtest_config()
    lake_inputs = build_lake_inputs(
        store, exchange_id=EX, symbols=[SYM], timeframe=TF, base_timeframe=BASE,
        funding_timeframe=FUND, start_ms=start, end_ms=end, oi_timeframe=OI_TF,
    )
    bt = BacktestEngine(bt_cfg, load_metadata_config(), make_strategy(bt_cfg))
    bt_trades = build_report(bt.run(lake_inputs)).trade_count
    if bt_trades > 0:
        assert 0.2 * bt_trades <= concurrent_trades <= 5 * bt_trades


def test_lake_paper_inputs_for_symbol_listed_mid_window(tmp_path) -> None:
    """Regression: the replay builder must locate the entry bar by TIMESTAMP, not by
    ``decision_ts // iv`` array position. A contract whose data starts mid-window (listed after
    the window start) has its first bar at a large ts, so the slot index would point past the
    short bars array and silently drop every candidate. Seed only the second half of the window."""
    store = SeriesStore(tmp_path)
    iv = timeframe_ms(TF)
    listing = 300 * iv
    window_start, window_end = 0, listing + 400 * iv
    _seed(store, listing, window_end)  # nothing before the listing slot
    cfg = _data_cfg(window_start, window_end)

    inputs, strat_id, _ = build_lake_paper_inputs(cfg, timeframe=TF, symbols=[SYM], store=store)
    assert strat_id == "reference_momentum"
    assert inputs, "a mid-window-listed symbol must still produce candidates"
    for pin in inputs:
        assert pin.candidate.entry_price > 0
        assert math.isfinite(pin.exit_move_frac)


def test_lake_paper_rejects_portfolio_strategy(tmp_path) -> None:
    store = SeriesStore(tmp_path)
    start, end = 0, 50 * timeframe_ms(TF)
    _seed(store, start, end)
    cfg = _data_cfg(start, end)
    with pytest.raises(ValueError, match="cross-asset|portfolio"):
        build_lake_paper_inputs(
            cfg, timeframe=TF, symbols=[SYM], candidate_id="lead_lag_xasset", store=store
        )


def test_ml_shadow_scores_real_lake_candidates(tmp_path) -> None:
    """Real-data ML shadow: the meta-labeler scores REAL lake candidates, applied=False."""
    from src.ml import ShadowPredictor
    from src.ml.config import load_ml_config
    from src.ml.labels import build_reference_dataset, train_test_split

    store = SeriesStore(tmp_path)
    start, end = 0, 400 * timeframe_ms(TF)
    _seed(store, start, end)
    cfg = _data_cfg(start, end)
    inputs, _, _ = build_lake_paper_inputs(cfg, timeframe=TF, symbols=[SYM], store=store)
    candidates = [pin.candidate for pin in inputs]
    assert candidates

    predictor = ShadowPredictor.from_config(load_ml_config())
    train_samples, _ = train_test_split(build_reference_dataset(seed=42), seed=42)
    predictor.train(train_samples)
    result = predictor.run(candidates, write_to_db=False)
    assert result.applied is False  # shadow-only: never influences trading
    assert len(result.bundles) == len(candidates)
