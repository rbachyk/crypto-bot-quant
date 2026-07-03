"""Phase 8 gate checks: PAPER-A and PAPER-B (AGENTS.md Appendix A, Phase 8).

PAPER-A — Technical Paper validation:
  Exercises the FULL pipeline end-to-end in paper mode with deterministic
  inputs (same approach as Phase 6 gate checks). Verifies:
  - full pipeline runs without error (candidate → risk → exec sim);
  - simulated stops are placed on every position;
  - kill switch halts new entries when engaged;
  - reconciliation detects foreign orders and sets halt_required;
  - decision logs are complete (all required fields present).

PAPER-B — Strategy Paper validation:
  Evaluates the REAL persisted paper results (``paper_runs`` / ``paper_trades``
  written by ``run_paper_session`` / ``run_lake_paper_session`` / the basket
  paper loop) — never a fabricated in-process session — and verifies:
  - at least one qualifying paper session exists in the lookback window;
  - sufficient candidates evaluated (≥ configured threshold);
  - sufficient executed paper trades (≥ configured threshold);
  - per-symbol and per-regime breakdowns derivable from the persisted trades;
  - win/loss breakdown available;
  - rejection stats available;
  - paper expectancy is consistent with the persisted backtest reference
    (``backtest_runs``) for the strategies actually paper-traded.

  Scope: the "paper" environment — the SAME definition the dashboard stats and
  env-reset admin use (src/live/admin.py): every session whose session_id does
  NOT carry a real-venue (``demo:``/``testnet:``/``live:``) or ``selftest:``
  prefix. That includes plain ``run_paper_session`` runs, ``lake:`` replay
  sessions and ``paper:basket:`` sessions, and deliberately excludes real-venue
  results (those belong to the LIVE gates) and self-test noise.

  FAIL-CLOSED: no qualifying session / unreachable DB / no backtest reference
  → the gate FAILS with a clear remediation reason. It never synthesises data.

PAPER-A uses the REAL pipeline components (PaperTradingEngine, RiskManager,
ExecutionEngine, SimulatedVenue, KillSwitch, Reconciler) on deterministic
reference inputs. A failed criterion carries full detail for dashboard
remediation.
"""

from __future__ import annotations

from src.config import Settings
from src.gates.result import Criterion
from src.paper.engine import PaperCandidateInput
from src.ranking.candidate import Candidate

# --------------------------------------------------------------------------- #
# Reference candidate builders                                                  #
# --------------------------------------------------------------------------- #

_REF_PRICE: dict[str, float] = {
    "BTC/USDT:USDT": 50_000.0,
    "ETH/USDT:USDT": 3_000.0,
    "SOL/USDT:USDT": 150.0,
}


def _make_candidate(
    symbol: str,
    *,
    side: int = 1,
    stop_frac: float = 0.008,
    tp_frac: float = 0.02,
    spread_bps: float = 3.0,
    slippage_est: float = 0.0005,
    expected_edge_frac: float = 0.01,
    signal_strength: float = 0.85,
    confirmation: float = 0.75,
    regime: str = "low_vol_up",
    strategy: str = "basis_reversion_v1",
    strategy_version: str = "v1.0.0",
    data_fresh: bool = True,
) -> Candidate:
    return Candidate(
        symbol=symbol,
        strategy=strategy,
        strategy_version=strategy_version,
        side=side,
        entry_price=_REF_PRICE.get(symbol, 1_000.0),
        stop_frac=stop_frac,
        tp_frac=tp_frac,
        regime=regime,
        session=1,
        features={"atr_pct": 0.003, "premium": 0.0008, "funding_z": 0.5},
        signal_strength=signal_strength,
        confirmation=confirmation,
        expected_edge_frac=expected_edge_frac,
        spread_bps=spread_bps,
        slippage_est=slippage_est,
        latency_ms=5.0,
        data_fresh=data_fresh,
        metadata_verified=True,
        symbol_tradable=True,
        strategy_enabled=True,
        config_live_approved=True,
        decision_ts=1_700_000_000_000,
    )


# --------------------------------------------------------------------------- #
# PAPER-A — Technical Paper validation                                          #
# --------------------------------------------------------------------------- #


def check_paper_a(settings: Settings) -> list[Criterion]:  # noqa: ARG001
    """PAPER-A: full pipeline in paper, stops simulated, kill/recon exercised."""
    out: list[Criterion] = []

    # ------------------------------------------------------------------ #
    # 1. Component imports (all pipeline layers importable)               #
    # ------------------------------------------------------------------ #
    try:
        from src.execution import ExecutionEngine, Reconciler, SimulatedVenue  # noqa: F401
        from src.killswitch import KillSwitch  # noqa: F401
        from src.paper import PaperTradingEngine  # noqa: F401
        from src.risk import RiskManager  # noqa: F401

        out.append(Criterion.ok("paper_a_imports", "all pipeline component imports successful"))
    except ImportError as exc:
        out.append(Criterion.fail("paper_a_imports", f"import error: {exc}"))
        return out

    # ------------------------------------------------------------------ #
    # 2. Full pipeline end-to-end run (candidate → risk → exec sim)       #
    # ------------------------------------------------------------------ #
    try:
        engine = PaperTradingEngine(
            config_version="v1.0.0-paper-a",
            universe_version="u_paper_a_001",
        )
        session = engine.new_session("paper_a_technical_test")

        good_inputs = [
            PaperCandidateInput(
                candidate=_make_candidate("BTC/USDT:USDT", side=1, regime="low_vol_up"),
                equity=10_000.0,
                exit_move_frac=0.015,
            ),
            PaperCandidateInput(
                candidate=_make_candidate("ETH/USDT:USDT", side=-1, regime="low_vol_down"),
                equity=10_000.0,
                exit_move_frac=-0.015,
            ),
            PaperCandidateInput(
                candidate=_make_candidate("SOL/USDT:USDT", side=1, regime="trend_up"),
                equity=10_000.0,
                exit_move_frac=0.008,
            ),
        ]
        engine.process_candidates(good_inputs, session)

        if session.executed_count == 0:
            out.append(
                Criterion.fail(
                    "paper_a_pipeline",
                    f"expected >=1 executed trade; got {session.executed_count} "
                    f"(rejected: {session.rejected_count})",
                )
            )
        else:
            out.append(
                Criterion.ok(
                    "paper_a_pipeline",
                    f"pipeline ran: {session.executed_count} executed, "
                    f"{session.rejected_count} rejected of {session.total_candidates} candidates",
                )
            )
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
        out.append(Criterion.fail("paper_a_pipeline", f"pipeline raised: {err}"))
        return out

    # ------------------------------------------------------------------ #
    # 3. Simulated stops placed (exchange-resident stop on every position) #
    # ------------------------------------------------------------------ #
    stops_ok = all(t.has_exchange_side_stop for t in session.trades)
    if stops_ok:
        out.append(
            Criterion.ok(
                "paper_a_stops",
                f"all {session.executed_count} executed trades have exchange-side stop",
            )
        )
    else:
        missing = [t.trade_id for t in session.trades if not t.has_exchange_side_stop]
        out.append(
            Criterion.fail(
                "paper_a_stops",
                f"positions without exchange-side stop: {missing}",
            )
        )

    # ------------------------------------------------------------------ #
    # 4. Kill switch halts new entries when engaged                        #
    # ------------------------------------------------------------------ #
    try:
        ks_session = engine.new_session("paper_a_killswitch_test")
        # Engage kill switch then try to submit a candidate.
        engine.engage_kill_switch(ks_session)
        ks_inputs = [
            PaperCandidateInput(
                candidate=_make_candidate("BTC/USDT:USDT", side=1),
                equity=10_000.0,
            )
        ]
        engine.process_candidates(ks_inputs, ks_session)

        if ks_session.executed_count == 0:
            out.append(
                Criterion.ok(
                    "paper_a_kill_switch",
                    "kill switch engaged: candidate correctly blocked (0 executed)",
                )
            )
        else:
            out.append(
                Criterion.fail(
                    "paper_a_kill_switch",
                    f"kill switch engaged but {ks_session.executed_count} trade(s) still executed",
                )
            )
        engine.disengage_kill_switch(ks_session)
    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("paper_a_kill_switch", f"kill switch test raised: {exc}"))

    # ------------------------------------------------------------------ #
    # 5. Reconciliation detects foreign orders → halt_required            #
    # ------------------------------------------------------------------ #
    try:
        recon_session = engine.new_session("paper_a_recon_test")
        halt = engine.run_reconciliation(recon_session, inject_foreign_order=True)
        if halt:
            out.append(
                Criterion.ok(
                    "paper_a_reconciliation",
                    "reconciliation detected foreign order and set halt_required=True",
                )
            )
        else:
            out.append(
                Criterion.fail(
                    "paper_a_reconciliation",
                    "reconciliation ran but did NOT set halt_required for injected foreign order",
                )
            )
    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("paper_a_reconciliation", f"reconciliation raised: {exc}"))

    # ------------------------------------------------------------------ #
    # 6. Decision logs complete (all required fields present)             #
    # ------------------------------------------------------------------ #
    from src.paper.report import _REQUIRED_DECISION_FIELDS

    logs_ok = True
    missing_fields: list[str] = []
    for log in session.decision_logs:
        log_dict = log.to_dict()
        diff: set[str] = _REQUIRED_DECISION_FIELDS - set(log_dict.keys())
        if diff:
            logs_ok = False
            missing_fields = sorted(diff)
            break

    if logs_ok and session.decision_log_count > 0:
        out.append(
            Criterion.ok(
                "paper_a_decision_logs",
                f"{session.decision_log_count} decision log entries; all required fields present",
            )
        )
    elif session.decision_log_count == 0:
        out.append(Criterion.fail("paper_a_decision_logs", "no decision log entries produced"))
    else:
        out.append(
            Criterion.fail(
                "paper_a_decision_logs",
                f"missing fields in decision log: {missing_fields}",
            )
        )

    return out


# --------------------------------------------------------------------------- #
# PAPER-B — Strategy Paper validation (REAL persisted paper results)            #
# --------------------------------------------------------------------------- #

_PAPER_B_MIN_CANDIDATES = 10
_PAPER_B_MIN_EXECUTED = 5
# Only sessions from the recent past qualify — a blocks_live gate must reflect the
# CURRENT strategy set, not a paper run from months ago.
_PAPER_B_LOOKBACK_DAYS = 30
# Paper-vs-backtest expectancy consistency bounds (per-R units). The lower bound is
# generous because the paper engine understates the backtest (it does not simulate the
# trailing stop); the upper bound catches a paper sim wildly out of line with research.
# The absolute tolerance keeps tiny reference expectancies from making the ratio degenerate.
_PAPER_B_MIN_EXPECTANCY_RATIO = 0.25
_PAPER_B_MAX_EXPECTANCY_RATIO = 4.0
_PAPER_B_EXPECTANCY_ABS_TOL = 0.2

# Real-venue / self-test session_id prefixes excluded from the paper scope — MUST stay
# aligned with src.live.admin._REAL_ENV_PREFIXES + _SELFTEST_PREFIX (dashboard stats scope).
_PAPER_B_EXCLUDED_PREFIXES = ("demo:", "testnet:", "live:", "selftest:")


def check_paper_b(settings: Settings) -> list[Criterion]:  # noqa: ARG001
    """PAPER-B: REAL persisted paper sessions — trades, breakdowns, backtest consistency.

    Reads ``paper_runs`` / ``paper_trades`` (paper-environment scope, recent window) and
    ``backtest_runs`` (consistency reference). Fails closed on missing data — it NEVER
    fabricates a session to evaluate.
    """
    out: list[Criterion] = []

    # ------------------------------------------------------------------ #
    # 0. Load the persisted paper sessions (fail-closed on any DB issue)   #
    # ------------------------------------------------------------------ #
    from datetime import UTC, datetime, timedelta

    try:
        from sqlalchemy import select

        from src.db.base import session_scope
        from src.db.models import BacktestRun, DecisionLog, PaperRun, PaperTradeRecord

        cutoff = datetime.now(UTC) - timedelta(days=_PAPER_B_LOOKBACK_DAYS)
        with session_scope() as db:
            run_q = select(PaperRun).where(PaperRun.created_at >= cutoff)
            for pfx in _PAPER_B_EXCLUDED_PREFIXES:
                run_q = run_q.where(~PaperRun.session_id.like(f"{pfx}%"))
            runs = [
                {
                    "session_id": r.session_id,
                    "executed": int(r.executed_count),
                    "rejected": int(r.rejected_count),
                }
                for r in db.execute(run_q).scalars()
            ]
            session_ids = [r["session_id"] for r in runs]

            trades: list[dict] = []
            rejection_reasons: dict[str, int] = {}
            if session_ids:
                trade_rows = db.execute(
                    select(PaperTradeRecord).where(PaperTradeRecord.session_id.in_(session_ids))
                ).scalars()
                trades = [
                    {
                        "symbol": t.symbol,
                        "strategy": t.strategy,
                        "regime": t.regime,
                        "pnl": float(t.pnl),
                        "pnl_r": float(t.pnl_r),
                    }
                    for t in trade_rows
                ]
                for action, reason in db.execute(
                    select(DecisionLog.action, DecisionLog.reason).where(
                        DecisionLog.session_id.in_(session_ids)
                    )
                ):
                    if action != "execute":
                        rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

            # Latest PASSED backtest reference per strategy actually paper-traded.
            paper_strategies = sorted({t["strategy"] for t in trades})
            bt_refs: dict[str, dict] = {}
            if paper_strategies:
                bt_rows = db.execute(
                    select(BacktestRun)
                    .where(
                        BacktestRun.passed.is_(True),
                        BacktestRun.strategy_id.in_(paper_strategies),
                    )
                    .order_by(BacktestRun.created_at.desc())
                ).scalars()
                for b in bt_rows:
                    bt_refs.setdefault(
                        b.strategy_id,
                        {
                            "run_id": b.run_id,
                            "expectancy_r": float(b.expectancy_r),
                            "trade_count": int(b.trade_count),
                        },
                    )
    except Exception as exc:  # noqa: BLE001
        out.append(
            Criterion.fail(
                "paper_b_sessions",
                f"could not read persisted paper results (fail-closed): "
                f"{type(exc).__name__}: {exc}",
            )
        )
        return out

    # ------------------------------------------------------------------ #
    # 1. Qualifying real paper sessions exist                              #
    # ------------------------------------------------------------------ #
    if not runs:
        out.append(
            Criterion.fail(
                "paper_b_sessions",
                f"no persisted paper session in the last {_PAPER_B_LOOKBACK_DAYS} days "
                "(paper scope: session_id without demo:/testnet:/live:/selftest: prefix) — "
                "run `run_paper_session` / `run_lake_paper_session` and re-run this gate; "
                "the gate never fabricates a session",
            )
        )
        return out
    out.append(
        Criterion.ok(
            "paper_b_sessions",
            f"{len(runs)} persisted paper session(s) in the last {_PAPER_B_LOOKBACK_DAYS} days: "
            f"{sorted(session_ids)[:10]}",
        )
    )

    total_candidates = sum(r["executed"] + r["rejected"] for r in runs)
    total_rejected = sum(r["rejected"] for r in runs)

    # ------------------------------------------------------------------ #
    # 2. Sufficient candidates                                             #
    # ------------------------------------------------------------------ #
    if total_candidates >= _PAPER_B_MIN_CANDIDATES:
        out.append(
            Criterion.ok(
                "paper_b_candidates",
                f"{total_candidates} candidates ≥ {_PAPER_B_MIN_CANDIDATES} required "
                f"(executed+rejected across {len(runs)} session(s))",
            )
        )
    else:
        out.append(
            Criterion.fail(
                "paper_b_candidates",
                f"only {total_candidates} candidates evaluated across persisted sessions; "
                f"need ≥ {_PAPER_B_MIN_CANDIDATES} (extend paper window)",
            )
        )

    # ------------------------------------------------------------------ #
    # 3. Sufficient executed trades                                        #
    # ------------------------------------------------------------------ #
    if len(trades) >= _PAPER_B_MIN_EXECUTED:
        out.append(
            Criterion.ok(
                "paper_b_executed",
                f"{len(trades)} persisted paper trades ≥ {_PAPER_B_MIN_EXECUTED} required",
            )
        )
    else:
        out.append(
            Criterion.fail(
                "paper_b_executed",
                f"only {len(trades)} persisted paper trades; "
                f"need ≥ {_PAPER_B_MIN_EXECUTED} (do not loosen filters — extend window)",
            )
        )

    # ------------------------------------------------------------------ #
    # 4. Per-symbol breakdown derivable                                    #
    # ------------------------------------------------------------------ #
    symbols = sorted({t["symbol"] for t in trades})
    if len(symbols) >= 2:
        out.append(Criterion.ok("paper_b_symbol_breakdown", f"per-symbol breakdown: {symbols}"))
    elif len(symbols) == 1:
        out.append(
            Criterion.ok(
                "paper_b_symbol_breakdown",
                f"per-symbol breakdown: {symbols} (single symbol; multi-symbol recommended)",
            )
        )
    else:
        out.append(
            Criterion.fail(
                "paper_b_symbol_breakdown",
                "no per-symbol breakdown (no persisted executed trades)",
            )
        )

    # ------------------------------------------------------------------ #
    # 5. Per-regime breakdown derivable                                    #
    # ------------------------------------------------------------------ #
    regimes = sorted({t["regime"] for t in trades if t["regime"]})
    if len(regimes) >= 2:
        out.append(Criterion.ok("paper_b_regime_breakdown", f"per-regime breakdown: {regimes}"))
    elif len(regimes) == 1:
        out.append(
            Criterion.ok(
                "paper_b_regime_breakdown",
                f"per-regime breakdown: {regimes} "
                "(single regime; multi-regime recommended — extend paper window)",
            )
        )
    else:
        out.append(
            Criterion.fail(
                "paper_b_regime_breakdown",
                "no regime attribution on persisted trades (no executed trades or regime "
                "column empty)",
            )
        )

    # ------------------------------------------------------------------ #
    # 6. Win/loss breakdown available                                      #
    # ------------------------------------------------------------------ #
    wins = sum(1 for t in trades if t["pnl"] > 0)
    losses = len(trades) - wins
    if trades:
        win_rate = wins / len(trades)
        out.append(
            Criterion.ok(
                "paper_b_win_loss",
                f"win/loss breakdown: {wins} wins / {losses} losses "
                f"(win_rate={win_rate:.2%}, net_pnl={sum(t['pnl'] for t in trades):.4f})",
            )
        )
    else:
        out.append(
            Criterion.fail(
                "paper_b_win_loss",
                "no persisted trades — win/loss breakdown unavailable",
            )
        )

    # ------------------------------------------------------------------ #
    # 7. Rejection stats available                                         #
    # ------------------------------------------------------------------ #
    if rejection_reasons:
        out.append(
            Criterion.ok(
                "paper_b_rejection_analysis",
                f"{total_rejected} rejected candidates; per-reason breakdown: "
                f"{dict(sorted(rejection_reasons.items()))}",
            )
        )
    elif total_rejected > 0:
        out.append(
            Criterion.ok(
                "paper_b_rejection_analysis",
                f"{total_rejected} rejected candidates recorded on paper_runs "
                "(no per-reason decision logs persisted — mid-run sessions write logs "
                "only at session end)",
            )
        )
    else:
        out.append(
            Criterion.ok(
                "paper_b_rejection_analysis",
                "0 rejections recorded across persisted sessions (nothing to analyse)",
            )
        )

    # ------------------------------------------------------------------ #
    # 8. Paper expectancy consistent with the persisted backtest reference #
    # ------------------------------------------------------------------ #
    if not trades:
        out.append(
            Criterion.fail(
                "paper_b_vs_backtest",
                "no persisted paper trades — consistency vs backtest cannot be verified "
                "(fail-closed)",
            )
        )
    elif not bt_refs:
        out.append(
            Criterion.fail(
                "paper_b_vs_backtest",
                f"no PASSED backtest_runs reference for the paper-traded strategies "
                f"{paper_strategies} — persist a passing backtest for these strategies "
                "before promotion (fail-closed; the gate does not compare against "
                "fabricated numbers)",
            )
        )
    else:
        paper_exp_r = sum(t["pnl_r"] for t in trades) / len(trades)
        # Reference expectancy: weight each matched strategy's backtest expectancy by how
        # much of the paper sample that strategy contributes.
        matched = [t for t in trades if t["strategy"] in bt_refs]
        if not matched:
            out.append(
                Criterion.fail(
                    "paper_b_vs_backtest",
                    f"backtest references exist but none match the paper-traded strategies "
                    f"{paper_strategies} (fail-closed)",
                )
            )
        else:
            bt_exp_r = sum(bt_refs[t["strategy"]]["expectancy_r"] for t in matched) / len(matched)
            diff = paper_exp_r - bt_exp_r
            if bt_exp_r > 0:
                ratio = paper_exp_r / bt_exp_r
                within = (
                    _PAPER_B_MIN_EXPECTANCY_RATIO <= ratio <= _PAPER_B_MAX_EXPECTANCY_RATIO
                ) or abs(diff) <= _PAPER_B_EXPECTANCY_ABS_TOL
                detail = (
                    f"paper expectancy={paper_exp_r:.3f}R vs backtest reference="
                    f"{bt_exp_r:.3f}R (ratio={ratio:.2f}, bounds "
                    f"[{_PAPER_B_MIN_EXPECTANCY_RATIO}, {_PAPER_B_MAX_EXPECTANCY_RATIO}], "
                    f"abs_tol={_PAPER_B_EXPECTANCY_ABS_TOL}R); refs="
                    f"{ {k: v['run_id'] for k, v in bt_refs.items()} }"
                )
            else:
                # A non-positive reference expectancy is not a promotable baseline.
                within = False
                detail = (
                    f"backtest reference expectancy is non-positive ({bt_exp_r:.3f}R) — "
                    "not a valid promotion baseline (fail-closed)"
                )
            out.append(
                Criterion.ok("paper_b_vs_backtest", detail)
                if within
                else Criterion.fail(
                    "paper_b_vs_backtest", f"paper vs backtest INCONSISTENT: {detail}"
                )
            )

    return out
