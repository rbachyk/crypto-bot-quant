"""Offline proof for the live/paper basket loop: a planted funding-carry edge, fed one snapshot
at a time, books profitable PaperTrades — confirming the live path reuses the engine math and the
Trade→PaperTrade conversion is sound (before it ever touches a live feed)."""

from __future__ import annotations

from src.backtest.config import load_backtest_config
from src.backtest.engine import SymbolInput
from src.config import Settings
from src.exchange.metadata import load_metadata_config
from src.features.pipeline import FeatureFrame
from src.killswitch import KillSwitch
from src.live.basket import BasketPaperLoop, _halt_check
from src.paper.session import PaperSession
from src.strategies.candidates import build_strategy
from src.strategies.config import load_strategies_config

IV = 60_000
N = 240


def _fixture():
    """10 symbols, flat price (carry is the only P&L), constant funding_z, funding rate ALIGNED
    with funding_z (high funding ⇒ shorts collect, low ⇒ longs collect). Returns the per-symbol
    SymbolInput (spread + funding) and the snapshot stream."""
    syms = [f"S{i}/USDT:USDT" for i in range(10)]
    by_symbol: dict[str, SymbolInput] = {}
    fz = {}
    for i, s in enumerate(syms):
        z = -2.0 + i * (4.0 / 9.0)
        fz[s] = z
        funding = [{"ts": k * IV, "funding_rate": 0.001 * z} for k in range(0, N, 8)]
        spread = [{"ts": k * IV, "spread_bps": 2.0} for k in range(N)]
        frame = FeatureFrame(symbol=s, timeframe="1m", feature_names=["funding_z"], rows=[])
        by_symbol[s] = SymbolInput(
            symbol=s, bars=[], frame=frame, spread_samples=spread, funding_events=funding
        )

    def snapshots():
        for k in range(N):
            ts = k * IV
            bars_at = {
                s: {"ts": ts, "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
                    "volume": 1e6}
                for s in syms
            }
            rows_at = {
                s: {"ts": ts, "decision_ts": ts, "funding_z": fz[s], "atr_pct": 0.01,
                    "session_code": 0}
                for s in syms
            }
            yield (ts, bars_at, rows_at)

    return by_symbol, snapshots()


def test_basket_paper_loop_books_profitable_carry_trades():
    cfg = load_backtest_config()
    meta = load_metadata_config()
    sc = load_strategies_config()
    strat = build_strategy(sc.candidate("funding_carry"), sc.strategy_version)
    session = PaperSession(session_id="t")
    by_symbol, snaps = _fixture()

    BasketPaperLoop(cfg, meta, strat, bar_interval_ms=IV, session=session).run(snaps, by_symbol)

    assert session.trades, "the basket loop should book paper trades"
    longs = [t for t in session.trades if t.side > 0]
    shorts = [t for t in session.trades if t.side < 0]
    assert longs and shorts  # dollar-neutral basket trades both sides
    assert sum(t.pnl for t in session.trades) > 0.0  # planted carry is harvested
    assert all(t.execution_route == "maker" for t in session.trades)  # rebalanced as maker


def test_basket_loop_residual_momentum_forms_a_basket() -> None:
    """REGRESSION: the live loop must score residual_momentum via the engine's _residual_score
    (from the rolling bars), NOT strategy.score() — which returns None, so the basket never formed
    and booked 0 legs (the live VPS symptom). With the parity fix it forms a basket and books legs.
    Planted: a common market factor + per-symbol idiosyncratic drift the residual must isolate."""
    import dataclasses
    import math

    cfg = load_backtest_config()
    meta = load_metadata_config()
    sc = load_strategies_config()
    cand = sc.candidate("residual_momentum")
    # small windows so the test needs few bars (defaults are 24/120)
    extra = {**cand.params.extra, "signal_window": 6.0, "beta_window": 20.0}
    params = dataclasses.replace(cand.params, extra=extra)
    strat = build_strategy(cand, sc.strategy_version, params=params)

    nsym, nbars, iv = 10, 60, 3_600_000
    syms = [f"S{i}/USDT:USDT" for i in range(nsym)]
    drift = [(-1.0 + 2.0 * i / (nsym - 1)) * 0.002 for i in range(nsym)]  # ±, symmetric

    def price(i: int, k: int) -> float:
        return 100.0 * math.exp(0.03 * math.sin(2 * math.pi * k / 40) + drift[i] * k)

    bars = {s: [{"ts": k * iv, "open": price(i, k), "high": price(i, k), "low": price(i, k),
                 "close": price(i, k), "volume": 1e6} for k in range(nbars)]
            for i, s in enumerate(syms)}
    by_symbol = {s: SymbolInput(
        symbol=s, bars=bars[s],
        frame=FeatureFrame(symbol=s, timeframe="1h", feature_names=[], rows=[]),
        spread_samples=[{"ts": k * iv, "spread_bps": 2.0} for k in range(nbars)], funding_events=[],
    ) for s in syms}

    def snaps():
        for k in range(40, nbars):  # enough history before each ts for the 6+20 windows
            ts = k * iv
            yield (ts, {s: bars[s][k] for s in syms},
                   {s: {"ts": ts, "decision_ts": ts, "atr_pct": 0.01, "session_code": 0}
                    for s in syms})

    session = PaperSession(session_id="t")
    BasketPaperLoop(cfg, meta, strat, bar_interval_ms=iv, session=session).run(snaps(), by_symbol)

    assert session.trades, "residual_momentum must FORM a basket in the live loop (was 0 legs)"
    assert any(t.side > 0 for t in session.trades) and any(t.side < 0 for t in session.trades)


def test_basket_halt_check_honours_global_kill_switch(tmp_path) -> None:
    """The basket loop's stop condition must trip on the GLOBAL kill switch (emergency halt), not
    just the caller's Stop/job-cancel — or a kill switch would stop directional trading but leave
    the carry/factor baskets running. Isolated file-backed switch (unreachable redis)."""
    iso = Settings(
        _env_file=None, app_env="paper",
        data_lake_path=tmp_path / "dl", redis_url="redis://127.0.0.1:1/0",
    )
    kill = KillSwitch(iso)
    kill.disengage()  # clean slate

    # caller never asks to stop; only the kill switch governs.
    halt = _halt_check(lambda: False, kill)
    assert halt() is False  # not engaged → keep running
    kill.engage(reason="test")
    assert halt() is True  # engaged → halt the basket loop
    kill.disengage()
    assert halt() is False

    # caller's own stop still works independently of the kill switch.
    assert _halt_check(lambda: True, kill)() is True
    assert _halt_check(None, kill)() is False
