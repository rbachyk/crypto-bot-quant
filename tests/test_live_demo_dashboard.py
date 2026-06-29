"""Dashboard-driven demo/live trading: env-separated sessions, progress + stop, stats reset.

The operator runs everything from the dashboard (no terminal): start a demo session, watch its
progress, stop it, and zero the demo statistics before a clean run. These tests prove the
mechanics behind those controls:

* a real-venue run is tagged by EXCHANGE_ENV (a Bybit **demo** run → ``demo:`` session ids),
  so its statistics stay separated from paper/testnet/live;
* the loop reports progress per tick and stops cleanly when an external Stop is requested;
* ``reset_env_stats`` zeroes one environment only, leaving the others intact;
* ``run_live_session`` / ``reset_env_stats`` route to the dedicated ``live`` worker.
"""

from __future__ import annotations

import uuid

import pytest
from src.config import Settings, get_settings
from src.data.config import DataConfig, ValidationThresholds
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
from src.exchange.metadata import load_metadata_config
from src.execution.venue import SimulatedVenue
from src.jobs.handlers import _live_loop_mode
from src.jobs.routing import queue_class
from src.killswitch import KillSwitch
from src.live.loop import LiveLoop, LiveTick, ReplayFeed, persist_live_run
from src.paper.lake import build_lake_paper_inputs

from tests.conftest import requires_db

EX = "bybit"
SYM = "BTC/USDT:USDT"
TF = "5m"
OI_TF = "1h"


@pytest.fixture(autouse=True)
def _clear_kill_switch():
    KillSwitch().disengage()
    yield
    KillSwitch().disengage()


def _seed(store: SeriesStore, start: int, end: int) -> None:
    src = DeterministicSource(EX)
    for dt, tf in (
        (OHLCV, TF),
        (MARK, TF),
        (INDEX, TF),
        (SPREAD, TF),
        (OPEN_INTEREST, OI_TF),
        (FUNDING, "8h"),
    ):
        key = SeriesKey(EX, dt, SYM, tf)
        store.write(key, src.fetch(key, start, end))


def _cfg(start: int, end: int) -> DataConfig:
    return DataConfig(
        exchange_id=EX,
        data_version="t",
        symbols=[SYM],
        timeframes=[TF],
        base_timeframe=TF,
        funding_interval_hours=8,
        required_series=[OHLCV, MARK, INDEX, FUNDING, OPEN_INTEREST, SPREAD],
        window_start_ms=start,
        window_end_ms=end,
        thresholds=ValidationThresholds(),
        oi_timeframe=OI_TF,
    )


def _feed(tmp_path) -> ReplayFeed:
    store = SeriesStore(tmp_path)
    start, end = 0, 400 * timeframe_ms(TF)
    _seed(store, start, end)
    inputs, _, _ = build_lake_paper_inputs(
        _cfg(start, end), timeframe=TF, symbols=[SYM], store=store
    )
    return ReplayFeed(inputs)


def _demo_settings() -> Settings:
    return Settings(
        _env_file=None,
        exchange_env="demo",
        exchange_api_key="k",
        exchange_api_secret="s",
    )


# --- environment-separated session labelling -------------------------------- #
def test_demo_run_is_tagged_demo_not_testnet(tmp_path) -> None:
    """A real-venue run under EXCHANGE_ENV=demo is labelled ``demo:`` (separated stats)."""
    feed = _feed(tmp_path)
    settings = _demo_settings()
    # Inject the offline venue so the test needs no network/keys; env_label is independent
    # of the injected venue (it comes from mode + EXCHANGE_ENV).
    loop = LiveLoop(
        mode="testnet", settings=settings, venue=SimulatedVenue(load_metadata_config())
    )
    assert loop.env_label == "demo"
    result = loop.run(feed, session_name="t")
    assert result.session.session_id.startswith("demo:")
    assert not result.session.session_id.startswith("testnet:")


def test_paper_mode_label_is_paper_regardless_of_env(tmp_path) -> None:
    feed = _feed(tmp_path)
    loop = LiveLoop(mode="paper", settings=_demo_settings())
    assert loop.env_label == "paper"  # offline SimulatedVenue is always 'paper'
    result = loop.run(feed, session_name="t")
    assert result.session.session_id.startswith("paper:")


# --- progress + clean stop -------------------------------------------------- #
def test_on_tick_reports_progress_each_tick(tmp_path) -> None:
    feed = _feed(tmp_path)
    seen: list[tuple[int, int]] = []
    result = LiveLoop(mode="paper").run(
        feed, session_name="t", on_tick=lambda tick, i: seen.append((i, tick.candidates))
    )
    assert seen  # called at least once
    assert [i for i, _ in seen] == list(range(len(result.ticks)))  # one call per processed tick


def test_should_stop_halts_loop_cleanly(tmp_path) -> None:
    feed = _feed(tmp_path)
    calls = {"n": 0}

    def _stop() -> bool:
        calls["n"] += 1
        return calls["n"] > 2  # allow two ticks, then request stop

    result = LiveLoop(mode="paper").run(feed, session_name="t", should_stop=_stop)
    assert result.halted
    assert len(result.ticks) == 2  # stopped before the third tick
    assert isinstance(result.ticks[0], LiveTick)


# --- persistence: demo run lands in the paper tables with demo: prefix ------- #
@requires_db
def test_persist_live_run_writes_demo_session(tmp_path) -> None:
    from sqlalchemy import select
    from src.db.base import session_scope
    from src.db.models import PaperRun, PaperTradeRecord

    feed = _feed(tmp_path)
    settings = _demo_settings()
    loop = LiveLoop(
        mode="testnet", settings=settings, venue=SimulatedVenue(load_metadata_config())
    )
    result = loop.run(feed, session_name=f"t_{uuid.uuid4().hex[:6]}")
    sid = persist_live_run(result, settings)
    assert sid.startswith("demo:")

    with session_scope() as db:
        run = db.execute(select(PaperRun).where(PaperRun.session_id == sid)).scalars().first()
        trades = (
            db.execute(select(PaperTradeRecord).where(PaperTradeRecord.session_id == sid))
            .scalars()
            .all()
        )
    assert run is not None
    assert run.executed_count == len(trades)


# --- reset zeroes ONE environment only -------------------------------------- #
@requires_db
def test_reset_env_stats_only_touches_target_env(tmp_path) -> None:
    from sqlalchemy import select
    from src.db.base import session_scope
    from src.db.models import PaperRun
    from src.live.admin import reset_env_stats, summarize_env_stats

    feed = _feed(tmp_path)
    meta = load_metadata_config()
    # One demo session and one testnet session.
    demo = LiveLoop(mode="testnet", settings=_demo_settings(), venue=SimulatedVenue(meta))
    demo_res = demo.run(feed, session_name=f"d_{uuid.uuid4().hex[:6]}")
    demo_sid = persist_live_run(demo_res, _demo_settings())

    tn_settings = Settings(
        _env_file=None, exchange_env="testnet", exchange_api_key="k", exchange_api_secret="s"
    )
    tn = LiveLoop(mode="testnet", settings=tn_settings, venue=SimulatedVenue(meta))
    tn_res = tn.run(_feed(tmp_path), session_name=f"n_{uuid.uuid4().hex[:6]}")
    tn_sid = persist_live_run(tn_res, tn_settings)

    assert summarize_env_stats("demo").runs >= 1
    removed = reset_env_stats("demo")
    assert removed.runs >= 1

    with session_scope() as db:
        sids = set(
            db.execute(select(PaperRun.session_id)).scalars().all()
        )
    assert demo_sid not in sids  # demo wiped
    assert tn_sid in sids  # testnet untouched
    assert summarize_env_stats("demo").total == 0


@requires_db
def test_reset_paper_stats_does_not_touch_selftest_rows() -> None:
    """Reset of the PAPER environment must NOT delete self-test runs — they are not paper trading,
    and the dashboard stats already exclude them, so the admin scope must stay aligned (else the
    confirm dialog under-reports what's removed and self-test data is collaterally wiped)."""
    from sqlalchemy import select
    from src.db.base import session_scope
    from src.db.models import PaperRun
    from src.live.admin import reset_env_stats

    paper_sid = f"paper_resetsel_{uuid.uuid4().hex[:6]}"
    self_sid = "selftest:resetsel"
    with session_scope() as db:
        db.add(PaperRun(session_id=paper_sid))
        db.add(PaperRun(session_id=self_sid))
    try:
        reset_env_stats("paper")
        with session_scope() as db:
            sids = set(db.execute(select(PaperRun.session_id)).scalars().all())
        assert paper_sid not in sids  # the real paper run was reset
        assert self_sid in sids  # the self-test run was NOT touched
    finally:
        with session_scope() as db:
            db.query(PaperRun).filter(PaperRun.session_id.in_((paper_sid, self_sid))).delete(
                synchronize_session=False
            )


# --- routing + venue-mode mapping ------------------------------------------- #
def test_live_jobs_route_to_dedicated_live_worker() -> None:
    assert queue_class("run_live_session") == "live"
    assert queue_class("reset_env_stats") == "live"


def test_live_loop_mode_maps_env_to_venue() -> None:
    assert _live_loop_mode("demo", None) == "testnet"  # demo uses the real ccxt venue (virtual)
    assert _live_loop_mode("testnet", None) == "testnet"
    assert _live_loop_mode("live", None) == "live"  # guarded real money
    assert _live_loop_mode("demo", "paper") == "paper"  # explicit override wins


def test_get_settings_singleton_unaffected() -> None:
    # The default process settings still validate (sanity: our new code imports cleanly).
    assert get_settings().exchange_env in ("live", "testnet", "demo")
