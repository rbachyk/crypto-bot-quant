"""Portfolio-state integration (AGENTS.md Section 17): the paper/live engine must feed the
risk manager the CURRENTLY-OPEN positions, not an empty portfolio — otherwise the Section-17
portfolio caps (per-symbol concurrency, total concurrency, portfolio heat, net beta-to-BTC)
are dead. These tests prove those caps actually bind against positions already open in the
engine, and that a closed (round-tripped) position frees its slot.

This is the regression guard for the demo-readiness review: a real demo account holds real
positions across ticks, so the risk caps must see them. Before the fix the engine always
passed ``PortfolioState(equity=...)`` with an empty positions tuple, so none of these caps
could ever trip in paper/demo.
"""

from __future__ import annotations

from src.config import Settings
from src.exchange.metadata import load_metadata_config
from src.paper.engine import PaperCandidateInput, PaperTradingEngine
from src.ranking import Candidate

BTC = "BTC/USDT:USDT"
ETH = "ETH/USDT:USDT"
_REF = {BTC: 50_000.0, ETH: 3_000.0}


def _cand(symbol: str, *, side: int = 1, regime: str = "low_vol_up") -> Candidate:
    return Candidate(
        symbol=symbol,
        strategy="basis_reversion_v1",
        strategy_version="v1.0.0",
        side=side,
        entry_price=_REF[symbol],
        stop_frac=0.008,
        tp_frac=0.02,
        regime=regime,
        session=1,
        features={"atr_pct": 0.003},
        signal_strength=0.85,
        confirmation=0.75,
        expected_edge_frac=0.012,
        spread_bps=3.0,
        slippage_est=0.0005,
        latency_ms=5.0,
        data_fresh=True,
        metadata_verified=True,
        symbol_tradable=True,
        strategy_enabled=True,
        config_live_approved=True,
        decision_ts=1_700_000_000_000,
    )


def _engine() -> PaperTradingEngine:
    return PaperTradingEngine(meta=load_metadata_config(), settings=Settings(_env_file=None))


def _held(symbol: str, *, side: int = 1, **over) -> PaperCandidateInput:
    """A candidate whose simulated trade stays OPEN (exit_move_frac=0) — i.e. it carries a
    live position across to the next bar, exactly as a real demo position would."""
    return PaperCandidateInput(
        candidate=_cand(symbol, side=side, **over), equity=100_000.0, exit_move_frac=0.0
    )


def test_open_position_blocks_second_same_symbol_entry() -> None:
    """With a position already open on BTC, a second BTC candidate is rejected — proving the
    per-symbol concurrency cap (max_concurrent_per_symbol=1) now sees the open position
    instead of an empty portfolio."""
    eng = _engine()
    session = eng.new_session()
    eng.process_candidates([_held(BTC), _held(BTC)], session)

    assert session.executed_count == 1  # only the first BTC entry placed
    assert [r.symbol for r in session.rejected] == [BTC]
    assert session.rejected[0].reason.startswith("risk_")


def test_per_symbol_cap_is_direction_agnostic() -> None:
    """An open BTC *long* still blocks a BTC *short* — the cap counts positions on the
    symbol regardless of side (you cannot stack a second position on a held symbol)."""
    eng = _engine()
    session = eng.new_session()
    eng.process_candidates([_held(BTC, side=1), _held(BTC, side=-1)], session)
    assert session.executed_count == 1


def test_different_symbol_not_blocked_by_per_symbol_cap() -> None:
    """An open BTC position must not block an entry on ETH via the per-symbol cap. (A
    beta-reducing short keeps the net-beta envelope satisfied so we isolate the per-symbol
    behaviour.)"""
    eng = _engine()
    session = eng.new_session()
    eng.process_candidates([_held(BTC, side=1), _held(ETH, side=-1)], session)
    assert session.executed_count == 2  # BTC and ETH open concurrently


def test_net_beta_cap_binds_against_open_position() -> None:
    """A second correlated long (ETH) on top of an open BTC long is rejected by the net
    beta-to-BTC envelope — a portfolio cap that was completely dead when the risk manager
    saw an empty portfolio. This proves heat/beta now reason over real exposure."""
    eng = _engine()
    session = eng.new_session()
    eng.process_candidates([_held(BTC, side=1), _held(ETH, side=1)], session)
    assert session.executed_count == 1  # ETH long rejected (net_beta_cap)
    assert session.rejected and session.rejected[0].symbol == ETH


def test_daily_loss_breaker_trips_from_accumulated_realized_losses() -> None:
    """Realized losses from CLOSED trades accumulate and trip the daily-loss breaker, halting
    new entries — the loss breakers were previously dead (hardcoded 0) in a real run."""
    eng = _engine()
    session = eng.new_session()
    # Each trade's stop is hit (exit_move beyond −0.8% stop) → a realized loss is booked.
    losers = [
        PaperCandidateInput(candidate=_cand(BTC), equity=100_000.0, exit_move_frac=-0.03)
        for _ in range(5)
    ]
    eng.process_candidates(losers, session)
    assert eng._realized_pnl < 0  # losses accumulated
    assert session.executed_count < 5  # the breaker halted entries before all 5
    assert any("risk_block" in r.reason for r in session.rejected)  # blocked by a breaker


def test_real_account_equity_drives_the_loss_breakers_in_live() -> None:
    """When the venue reports real account equity (live/demo), the daily-loss breaker trips on a
    falling account — the engine's simulated exit path is inert when exits are exchange-side, so
    this is what makes the loss breakers actually work in production."""
    from src.execution.venue import SimulatedVenue

    class _EquityVenue(SimulatedVenue):
        def __init__(self, meta, equities):
            super().__init__(meta)
            self._equities = list(equities)
            self._i = 0

        def fetch_account_equity(self):
            v = self._equities[min(self._i, len(self._equities) - 1)]
            self._i += 1
            return v

    meta = load_metadata_config()
    # Batch 1 at 100k (sets the session-start equity), batch 2 at 96k = −4% > 3% daily limit.
    venue = _EquityVenue(meta, [100_000.0, 96_000.0])
    eng = PaperTradingEngine(meta=meta, settings=Settings(_env_file=None), venue=venue)
    session = eng.new_session()
    eng.process_candidates([_held(BTC)], session)  # batch 1: equity flat → trades
    before = session.executed_count
    eng.process_candidates([_held(ETH, side=-1)], session)  # batch 2: −4% account → breaker
    assert session.executed_count == before  # nothing new executed
    assert session.rejected and session.rejected[-1].reason.startswith("risk_")


def test_closed_position_frees_the_concurrency_slot() -> None:
    """A BTC trade that round-trips (hits TP) releases its slot, so the next BTC entry is
    allowed — the cap binds on *currently* open positions, not historical ones."""
    eng = _engine()
    session = eng.new_session()
    winner = PaperCandidateInput(candidate=_cand(BTC), equity=100_000.0, exit_move_frac=0.03)
    eng.process_candidates([winner, _held(BTC)], session)
    assert session.executed_count == 2  # first closed (TP) → second BTC allowed
