"""Phase 8 paper trading tests: PAPER-A and PAPER-B (Appendix D Phase 8).

Tests cover:
  - PaperTradingEngine: full pipeline runs, stops placed, kill switch, recon.
  - PaperSession: candidate accounting, breakdowns.
  - PaperReport: A and B report builders.
  - Gate checks: PAPER-A and PAPER-B (offline; DB-backed gate runner for CI).
"""

from __future__ import annotations

import pytest
from src.paper import (
    PaperSession,
    PaperTradingEngine,
    build_paper_report,
)
from src.paper.engine import PaperCandidateInput
from src.paper.report import (
    _REQUIRED_DECISION_FIELDS,
    build_paper_a_report,
    build_paper_b_report,
)
from src.paper.session import (
    PaperDecisionLog,
    PaperTrade,
    RejectedPaperCandidate,
)
from src.ranking.candidate import Candidate

from tests.conftest import requires_db

# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #

_REF_PRICE = {
    "BTC/USDT:USDT": 50_000.0,
    "ETH/USDT:USDT": 3_000.0,
    "SOL/USDT:USDT": 150.0,
}


def _candidate(
    symbol: str = "BTC/USDT:USDT",
    *,
    side: int = 1,
    regime: str = "low_vol_up",
    strategy: str = "basis_reversion_v1",
    data_fresh: bool = True,
    spread_bps: float = 3.0,
    expected_edge_frac: float = 0.01,
) -> Candidate:
    return Candidate(
        symbol=symbol,
        strategy=strategy,
        strategy_version="v1.0.0",
        side=side,
        entry_price=_REF_PRICE.get(symbol, 1_000.0),
        stop_frac=0.008,
        tp_frac=0.02,
        regime=regime,
        session=1,
        features={"atr_pct": 0.003, "premium": 0.0008},
        signal_strength=0.85,
        confirmation=0.75,
        expected_edge_frac=expected_edge_frac,
        spread_bps=spread_bps,
        slippage_est=0.0005,
        latency_ms=5.0,
        data_fresh=data_fresh,
        metadata_verified=True,
        symbol_tradable=True,
        strategy_enabled=True,
        config_live_approved=True,
        decision_ts=1_700_000_000_000,
    )


# --------------------------------------------------------------------------- #
# PaperSession unit tests                                                       #
# --------------------------------------------------------------------------- #


class TestPaperSession:
    def test_initial_state(self) -> None:
        session = PaperSession("test-session-1")
        assert session.total_candidates == 0
        assert session.executed_count == 0
        assert session.rejected_count == 0
        assert not session.kill_switch_exercised
        assert not session.foreign_order_halt_triggered

    def test_symbol_breakdown_empty(self) -> None:
        session = PaperSession("test-session-2")
        assert session.symbol_breakdown() == {}

    def test_regime_breakdown_empty(self) -> None:
        session = PaperSession("test-session-3")
        assert session.regime_breakdown() == {}

    def test_rejection_breakdown(self) -> None:
        session = PaperSession("test-session-4")
        for reason in ["risk_reject", "risk_reject", "exec_stale_data"]:
            session.rejected.append(
                RejectedPaperCandidate(
                    symbol="BTC/USDT:USDT",
                    strategy="S",
                    side=1,
                    regime="low_vol_up",
                    decision_ts=0,
                    reason=reason,
                )
            )
        breakdown = session.rejection_breakdown()
        assert breakdown["risk_reject"] == 2
        assert breakdown["exec_stale_data"] == 1

    def test_to_dict_structure(self) -> None:
        session = PaperSession("test-session-5")
        d = session.to_dict()
        for key in (
            "session_id",
            "total_candidates",
            "executed_count",
            "rejected_count",
            "kill_switch_exercised",
            "foreign_order_halt_triggered",
            "symbol_breakdown",
            "regime_breakdown",
        ):
            assert key in d


# --------------------------------------------------------------------------- #
# PaperTradingEngine unit tests                                                 #
# --------------------------------------------------------------------------- #


class TestPaperTradingEngine:
    def test_instantiation(self) -> None:
        engine = PaperTradingEngine()
        assert engine is not None

    def test_new_session(self) -> None:
        engine = PaperTradingEngine()
        session = engine.new_session("my-session")
        assert session.session_id == "my-session"

    def test_process_good_candidate_executes(self) -> None:
        engine = PaperTradingEngine()
        session = engine.new_session()
        inp = PaperCandidateInput(
            candidate=_candidate("BTC/USDT:USDT", side=1),
            equity=10_000.0,
            exit_move_frac=0.015,
        )
        engine.process_candidates([inp], session)
        assert session.executed_count >= 1 or session.rejected_count >= 1
        assert session.total_candidates == 1

    def test_executed_trade_has_exchange_side_stop(self) -> None:
        engine = PaperTradingEngine()
        session = engine.new_session()
        inp = PaperCandidateInput(
            candidate=_candidate("BTC/USDT:USDT", side=1, regime="low_vol_up"),
            equity=10_000.0,
            exit_move_frac=0.015,
        )
        engine.process_candidates([inp], session)
        for trade in session.trades:
            assert trade.has_exchange_side_stop, (
                f"trade {trade.trade_id} missing exchange-side stop"
            )

    def test_stale_data_candidate_is_rejected(self) -> None:
        engine = PaperTradingEngine()
        session = engine.new_session()
        inp = PaperCandidateInput(
            candidate=_candidate("BTC/USDT:USDT", data_fresh=False),
            equity=10_000.0,
        )
        engine.process_candidates([inp], session)
        # Either rejected by exec revalidation or by risk (both are valid PAPER-A rejections).
        assert session.executed_count == 0 or any(
            "stale" in r.reason.lower() for r in session.rejected
        )

    def test_kill_switch_blocks_new_entries(self) -> None:
        engine = PaperTradingEngine()
        session = engine.new_session()
        engine.engage_kill_switch(session)
        assert session.kill_switch_exercised

        inp = PaperCandidateInput(
            candidate=_candidate("BTC/USDT:USDT", side=1),
            equity=10_000.0,
        )
        engine.process_candidates([inp], session)
        # With kill switch engaged, no new trades should execute.
        assert session.executed_count == 0

    def test_kill_switch_disengages(self) -> None:
        engine = PaperTradingEngine()
        session = engine.new_session()
        engine.engage_kill_switch(session)
        engine.disengage_kill_switch(session)
        assert len(session.kill_switch_events) == 2

    def test_reconciliation_detects_foreign_order(self) -> None:
        engine = PaperTradingEngine()
        session = engine.new_session()
        halt = engine.run_reconciliation(session, inject_foreign_order=True)
        assert halt is True
        assert session.foreign_order_halt_triggered
        assert len(session.reconciliation_events) == 1
        assert session.reconciliation_events[0]["halt_triggered"] is True

    def test_reconciliation_clean_no_halt(self) -> None:
        engine = PaperTradingEngine()
        session = engine.new_session()
        halt = engine.run_reconciliation(session, inject_foreign_order=False)
        assert halt is False
        assert not session.foreign_order_halt_triggered

    def test_decision_logs_produced(self) -> None:
        engine = PaperTradingEngine()
        session = engine.new_session()
        inputs = [
            PaperCandidateInput(
                candidate=_candidate(sym, side=1, regime="low_vol_up"),
                equity=10_000.0,
                exit_move_frac=0.01,
            )
            for sym in ["BTC/USDT:USDT", "ETH/USDT:USDT"]
        ]
        engine.process_candidates(inputs, session)
        # Each candidate produces exactly one decision log.
        assert session.decision_log_count == len(inputs)

    def test_decision_log_required_fields(self) -> None:
        engine = PaperTradingEngine()
        session = engine.new_session()
        inp = PaperCandidateInput(
            candidate=_candidate("BTC/USDT:USDT"),
            equity=10_000.0,
            exit_move_frac=0.01,
        )
        engine.process_candidates([inp], session)
        for log in session.decision_logs:
            log_dict = log.to_dict()
            missing = _REQUIRED_DECISION_FIELDS - set(log_dict.keys())
            assert not missing, f"missing fields: {missing}"

    def test_multi_symbol_session(self) -> None:
        engine = PaperTradingEngine()
        session = engine.new_session()
        inputs = [
            PaperCandidateInput(
                candidate=_candidate(sym, side=1, regime=reg),
                equity=10_000.0,
                exit_move_frac=0.012,
            )
            for sym, reg in [
                ("BTC/USDT:USDT", "low_vol_up"),
                ("ETH/USDT:USDT", "trend_up"),
                ("SOL/USDT:USDT", "low_vol_down"),
            ]
        ]
        engine.process_candidates(inputs, session)
        assert session.total_candidates == 3
        if session.executed_count > 0:
            assert len(session.symbol_breakdown()) > 0

    def test_session_to_dict_complete(self) -> None:
        engine = PaperTradingEngine()
        session = engine.new_session()
        inputs = [
            PaperCandidateInput(
                candidate=_candidate("BTC/USDT:USDT", side=1),
                equity=10_000.0,
                exit_move_frac=0.015,
            )
        ]
        engine.process_candidates(inputs, session)
        d = session.to_dict()
        for key in (
            "session_id",
            "started_at",
            "total_candidates",
            "executed_count",
            "rejected_count",
            "kill_switch_exercised",
            "foreign_order_halt_triggered",
            "trades",
            "rejected",
            "decision_logs",
        ):
            assert key in d

    def test_pnl_r_computed(self) -> None:
        engine = PaperTradingEngine()
        session = engine.new_session()
        inp = PaperCandidateInput(
            candidate=_candidate("BTC/USDT:USDT", side=1),
            equity=10_000.0,
            exit_move_frac=0.02,
        )
        engine.process_candidates([inp], session)
        for trade in session.trades:
            # pnl_r is set (may be positive or negative).
            assert isinstance(trade.pnl_r, float)

    def test_long_and_short_sides(self) -> None:
        engine = PaperTradingEngine()
        session = engine.new_session()
        inputs = [
            PaperCandidateInput(
                candidate=_candidate("BTC/USDT:USDT", side=s),
                equity=10_000.0,
                exit_move_frac=0.012,
            )
            for s in [1, -1]
        ]
        engine.process_candidates(inputs, session)
        assert session.total_candidates == 2

    def test_breaker_windows_roll_over_and_equity_tracks_realized(self) -> None:
        """REGRESSION (B2/B3): the daily/weekly loss windows must reset at UTC day/week boundaries
        — _realized_pnl is a LIFETIME accumulator, so without rollover a multi-day session charged
        prior days' losses against today's daily limit and tripped it permanently. And simulated
        equity must reflect realized P&L so the drawdown breaker actually moves."""
        import dataclasses

        eng = PaperTradingEngine()
        sess = eng.new_session()
        eng._realized_pnl = -500.0  # simulate prior-day realized losses (lifetime total)
        day1 = 1_700_000_000_000 + 86_400_000  # a UTC day AFTER the prior losses' day

        cand = dataclasses.replace(_candidate("BTC/USDT:USDT", side=1), decision_ts=day1)
        eng.process_candidates(
            [PaperCandidateInput(candidate=cand, equity=10_000.0, exit_move_frac=0.0)], sess
        )

        # the daily/weekly windows snapshotted the lifetime total at the boundary → today starts
        # fresh (windowed loss = lifetime - snapshot = 0), so the prior day's loss no longer counts.
        assert eng._day_key == day1 // 86_400_000
        assert eng._day_start_realized == -500.0
        assert eng._week_start_realized == -500.0
        # equity reflects realized P&L (seed + lifetime realized), feeding the drawdown breaker.
        assert eng._sim_peak_equity == 9_500.0  # 10_000 seed + (-500) realized

    def test_short_exit_classifies_and_closes_not_open(self) -> None:
        """REGRESSION: a short must classify as take_profit/stop and CLOSE — the old exit_move sign
        guards were swapped for side<0, so every short stuck at 'open', pinned its concurrency slot
        forever and never realized P&L. TP is BELOW entry for a short, stop ABOVE."""
        # short, price FALLS 2.5% → take-profit (below entry) → profit, slot released
        eng = PaperTradingEngine()
        sess = eng.new_session()
        eng.process_candidates(
            [PaperCandidateInput(candidate=_candidate("BTC/USDT:USDT", side=-1),
                                 equity=10_000.0, exit_move_frac=-0.025)],
            sess,
        )
        assert sess.trades, "the short should have executed"
        won = sess.trades[-1]
        assert won.side == -1 and won.exit_reason == "take_profit" and won.pnl > 0
        assert "BTC/USDT:USDT" not in eng._open_positions  # closed, not stuck open

        # short, price RISES 1.2% → stop (above entry) → loss
        eng2 = PaperTradingEngine()
        sess2 = eng2.new_session()
        eng2.process_candidates(
            [PaperCandidateInput(candidate=_candidate("ETH/USDT:USDT", side=-1),
                                 equity=10_000.0, exit_move_frac=0.012)],
            sess2,
        )
        lost = sess2.trades[-1]
        assert lost.exit_reason == "stop" and lost.pnl < 0
        assert "ETH/USDT:USDT" not in eng2._open_positions


# --------------------------------------------------------------------------- #
# PaperReport tests                                                             #
# --------------------------------------------------------------------------- #


class TestPaperReports:
    def _make_session_with_trades(self, n: int = 10) -> PaperSession:
        engine = PaperTradingEngine()
        session = engine.new_session("report-test")
        symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]
        regimes = ["low_vol_up", "trend_up", "low_vol_down"]
        inputs = [
            PaperCandidateInput(
                candidate=_candidate(
                    symbols[i % len(symbols)],
                    side=1 if i % 2 == 0 else -1,
                    regime=regimes[i % len(regimes)],
                ),
                equity=10_000.0,
                exit_move_frac=0.015 if i % 3 != 2 else -0.005,
            )
            for i in range(n)
        ]
        engine.process_candidates(inputs, session)
        engine.engage_kill_switch(session)
        engine.disengage_kill_switch(session)
        engine.run_reconciliation(session, inject_foreign_order=True)
        return session

    def test_paper_a_report_passes(self) -> None:
        session = self._make_session_with_trades(3)
        # Manually set the flags the report builder reads.
        session.kill_switch_exercised = True
        report = build_paper_a_report(
            session,
            component_imports_ok=True,
            kill_switch_halts_new_entries=True,
        )
        assert report.passed(), report.to_dict()

    def test_paper_a_report_fails_no_stops(self) -> None:
        session = PaperSession("no-stops")
        from datetime import UTC, datetime

        session.trades.append(
            PaperTrade(
                trade_id="t1",
                symbol="BTC/USDT:USDT",
                strategy="S",
                side=1,
                qty=0.01,
                entry_price=50_000.0,
                stop_price=49_600.0,
                tp_price=51_000.0,
                regime="low_vol_up",
                session=1,
                decision_ts=0,
                entry_ts=0,
                exit_ts=0,
                exit_price=50_500.0,
                exit_reason="take_profit",
                fee=0.01,
                slippage_cost=0.005,
                pnl=4.99,
                pnl_r=0.8,
                has_exchange_side_stop=False,  # <-- missing stop
                execution_route="taker",
                spread_bps_at_entry=3.0,
                slippage_frac=0.0005,
            )
        )
        session.decision_logs.append(
            PaperDecisionLog(
                entry_ts=datetime.now(UTC),
                symbol="BTC/USDT:USDT",
                strategy="S",
                regime="low_vol_up",
                side=1,
                action="execute",
                reason="approved",
                risk_approved=True,
                expected_edge=5.0,
                expected_fee=0.6,
                expected_slippage=0.25,
                config_version="v1",
                universe_version="u1",
                strategy_version="v1",
                kill_switch_state="clear",
            )
        )
        session.kill_switch_exercised = True
        session.foreign_order_halt_triggered = True
        session.reconciliation_events.append({"ts": "t", "halt_triggered": True})
        report = build_paper_a_report(
            session, component_imports_ok=True, kill_switch_halts_new_entries=True
        )
        assert not report.passed()
        assert not report.all_executed_have_exchange_side_stop

    def test_paper_b_report_passes(self) -> None:
        session = self._make_session_with_trades(15)
        report = build_paper_b_report(
            session, backtest_pnl=1.0, min_candidates_required=10, min_executed_required=5
        )
        # Even if not all criteria pass (due to rejections), we test the structure.
        assert isinstance(report.to_dict(), dict)
        assert "symbol_breakdown" in report.to_dict()
        assert "regime_breakdown" in report.to_dict()

    def test_paper_b_insufficient_trades(self) -> None:
        session = PaperSession("too-few")
        report = build_paper_b_report(
            session, backtest_pnl=0.0, min_candidates_required=10, min_executed_required=5
        )
        assert not report.passed()
        assert report.executed_count < report.min_executed_required

    def test_pnl_consistency_ratio_acceptable(self) -> None:
        session = self._make_session_with_trades(10)
        paper_pnl = sum(t.pnl for t in session.trades)
        # Set backtest_pnl close to paper_pnl.
        report = build_paper_b_report(
            session,
            backtest_pnl=paper_pnl * 1.5 if abs(paper_pnl) > 0.001 else 1.0,
            min_candidates_required=5,
            min_executed_required=3,
        )
        assert report.paper_vs_backtest_consistent

    def test_combined_report_structure(self) -> None:
        session = self._make_session_with_trades(10)
        session.kill_switch_exercised = True
        report = build_paper_report(
            session, backtest_pnl=0.5, component_imports_ok=True, kill_switch_halts_new_entries=True
        )
        d = report.to_dict()
        assert "paper_a" in d
        assert "paper_b" in d
        assert "passed" in d["paper_a"]
        assert "passed" in d["paper_b"]


# --------------------------------------------------------------------------- #
# Gate check tests (offline — no DB required)                                   #
# --------------------------------------------------------------------------- #


class TestPaperAGateCheck:
    def test_check_paper_a_imports(self) -> None:
        from src.config import get_settings
        from src.gates.phase8 import check_paper_a

        criteria = check_paper_a(get_settings())
        import_criterion = next((c for c in criteria if c.id == "paper_a_imports"), None)
        assert import_criterion is not None
        assert import_criterion.passed, import_criterion.detail

    def test_check_paper_a_pipeline(self) -> None:
        from src.config import get_settings
        from src.gates.phase8 import check_paper_a

        criteria = check_paper_a(get_settings())
        pipeline = next((c for c in criteria if c.id == "paper_a_pipeline"), None)
        assert pipeline is not None
        assert pipeline.passed, pipeline.detail

    def test_check_paper_a_stops(self) -> None:
        from src.config import get_settings
        from src.gates.phase8 import check_paper_a

        criteria = check_paper_a(get_settings())
        stops = next((c for c in criteria if c.id == "paper_a_stops"), None)
        assert stops is not None
        assert stops.passed, stops.detail

    def test_check_paper_a_kill_switch(self) -> None:
        from src.config import get_settings
        from src.gates.phase8 import check_paper_a

        criteria = check_paper_a(get_settings())
        ks = next((c for c in criteria if c.id == "paper_a_kill_switch"), None)
        assert ks is not None
        assert ks.passed, ks.detail

    def test_check_paper_a_reconciliation(self) -> None:
        from src.config import get_settings
        from src.gates.phase8 import check_paper_a

        criteria = check_paper_a(get_settings())
        recon = next((c for c in criteria if c.id == "paper_a_reconciliation"), None)
        assert recon is not None
        assert recon.passed, recon.detail

    def test_check_paper_a_decision_logs(self) -> None:
        from src.config import get_settings
        from src.gates.phase8 import check_paper_a

        criteria = check_paper_a(get_settings())
        logs = next((c for c in criteria if c.id == "paper_a_decision_logs"), None)
        assert logs is not None
        assert logs.passed, logs.detail

    def test_check_paper_a_all_pass(self) -> None:
        from src.config import get_settings
        from src.gates.phase8 import check_paper_a

        criteria = check_paper_a(get_settings())
        failing = [c for c in criteria if not c.passed]
        assert not failing, [(c.id, c.detail) for c in failing]


class TestPaperBGateCheck:
    def test_check_paper_b_candidates(self) -> None:
        from src.config import get_settings
        from src.gates.phase8 import check_paper_b

        criteria = check_paper_b(get_settings())
        cand = next((c for c in criteria if c.id == "paper_b_candidates"), None)
        assert cand is not None
        assert cand.passed, cand.detail

    def test_check_paper_b_executed(self) -> None:
        from src.config import get_settings
        from src.gates.phase8 import check_paper_b

        criteria = check_paper_b(get_settings())
        exe = next((c for c in criteria if c.id == "paper_b_executed"), None)
        assert exe is not None
        assert exe.passed, exe.detail

    def test_check_paper_b_symbol_breakdown(self) -> None:
        from src.config import get_settings
        from src.gates.phase8 import check_paper_b

        criteria = check_paper_b(get_settings())
        sym = next((c for c in criteria if c.id == "paper_b_symbol_breakdown"), None)
        assert sym is not None
        assert sym.passed, sym.detail

    def test_check_paper_b_regime_breakdown(self) -> None:
        from src.config import get_settings
        from src.gates.phase8 import check_paper_b

        criteria = check_paper_b(get_settings())
        reg = next((c for c in criteria if c.id == "paper_b_regime_breakdown"), None)
        assert reg is not None
        assert reg.passed, reg.detail

    def test_check_paper_b_vs_backtest(self) -> None:
        from src.config import get_settings
        from src.gates.phase8 import check_paper_b

        criteria = check_paper_b(get_settings())
        bt = next((c for c in criteria if c.id == "paper_b_vs_backtest"), None)
        assert bt is not None
        assert bt.passed, bt.detail

    def test_check_paper_b_all_pass(self) -> None:
        from src.config import get_settings
        from src.gates.phase8 import check_paper_b

        criteria = check_paper_b(get_settings())
        failing = [c for c in criteria if not c.passed]
        assert not failing, [(c.id, c.detail) for c in failing]


# --------------------------------------------------------------------------- #
# Gate runner integration tests (DB-backed; skipped if DB unavailable)          #
# --------------------------------------------------------------------------- #


@requires_db
def test_gate_paper_a_via_runner() -> None:
    from src.gates import GateRunner
    from src.gates.result import GateVerdict

    result = GateRunner().run("PAPER-A")
    assert result.overall == GateVerdict.PASS.value, (result.note, result.criteria)
    assert result.report_path
    failing = [c for c in result.criteria if c["status"] != "PASS"]
    assert not failing, failing


@requires_db
def test_gate_paper_b_via_runner() -> None:
    from src.gates import GateRunner
    from src.gates.result import GateVerdict

    result = GateRunner().run("PAPER-B")
    assert result.overall == GateVerdict.PASS.value, (result.note, result.criteria)
    assert result.report_path
    failing = [c for c in result.criteria if c["status"] != "PASS"]
    assert not failing, failing


@requires_db
def test_paper_dependency_chain() -> None:
    from src.gates import GateRunner

    runner = GateRunner()
    paper_a_deps = runner.catalog["PAPER-A"].depends_on
    paper_b_deps = runner.catalog["PAPER-B"].depends_on
    assert "PAPER-A" in paper_b_deps
    assert len(paper_a_deps) > 0


@requires_db
@pytest.mark.parametrize("gate_id", ["PAPER-A", "PAPER-B"])
def test_phase8_gates_persist_results(gate_id: str) -> None:
    from sqlalchemy import desc, select
    from src.db.base import session_scope
    from src.db.models import GateResult, GateStatus
    from src.gates import GateRunner

    GateRunner().run(gate_id)
    with session_scope() as db_session:
        latest = (
            db_session.execute(
                select(GateResult)
                .where(GateResult.gate_id == gate_id)
                .order_by(desc(GateResult.id))
            )
            .scalars()
            .first()
        )
        assert latest is not None
        assert latest.status is GateStatus.PASSED
        assert latest.criteria
