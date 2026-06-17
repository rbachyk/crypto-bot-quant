"""Phase 6 gate checks: SETUP, RISK, EXEC, KILL, ORDER-OWN (Appendix A).

Each check exercises the REAL Phase 6 components (ranking, risk manager, order
builder, simulated venue, reconciler, kill switch) on deterministic inputs and
asserts the Appendix A pass conditions, including the **deliberate forced-failure
tests** Appendix D Phase 6 requires (a toxic-spread setup must be blocked, a
breaker trip must halt, a foreign order must be detected, the kill switch must
flatten new entries). A failed criterion carries enough detail for the dashboard
to point at exactly what broke.
"""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from src.config import Settings
from src.exchange.metadata import MetadataConfig, load_metadata_config
from src.execution import (
    ExecutionEngine,
    OrderBuilder,
    OwnershipPolicy,
    Reconciler,
    SimulatedVenue,
    load_execution_config,
)
from src.execution.order import BUY, SELL, Order, OrderType
from src.gates.result import Criterion
from src.killswitch import KillSwitch
from src.monitoring import get_alert_sink
from src.ranking import (
    NO_TRADE_REGIMES,
    Candidate,
    CandidateRankingEngine,
    SetupQualityScorer,
    load_ranking_config,
)
from src.risk import (
    AccountState,
    BreakerInputs,
    PortfolioState,
    Position,
    RiskEnvelope,
    RiskManager,
    load_risk_config,
)

# Representative reference prices for the offline `skeleton` venue symbols.
_REF_PRICE: dict[str, float] = {
    "BTC/USDT:USDT": 50_000.0,
    "ETH/USDT:USDT": 3_000.0,
    "SOL/USDT:USDT": 150.0,
}


def _good_candidate(
    symbol: str,
    *,
    side: int = 1,
    stop_frac: float = 0.008,
    tp_frac: float = 0.02,
    spread_bps: float = 3.0,
    slippage_est: float = 0.0005,
    expected_edge_frac: float = 0.01,
    signal_strength: float = 0.9,
    confirmation: float = 0.8,
    regime: str = "low_vol_up",
    entry_price: float | None = None,
    **flags: object,
) -> Candidate:
    """A clean, high-quality candidate (no blockers) used as the gate baseline."""
    return Candidate(
        symbol=symbol,
        strategy="ranking_selftest",
        strategy_version="rank_selftest",
        side=side,
        entry_price=entry_price if entry_price is not None else _REF_PRICE.get(symbol, 100.0),
        stop_frac=stop_frac,
        tp_frac=tp_frac,
        regime=regime,
        session=2,
        features={"atr_pct_rank": 0.2, "is_weekend": 0.0, "pre_funding": 0.0, "trend_slope": 1.0},
        signal_strength=signal_strength,
        confirmation=confirmation,
        expected_edge_frac=expected_edge_frac,
        spread_bps=spread_bps,
        slippage_est=slippage_est,
        latency_ms=40.0,
        **flags,  # type: ignore[arg-type]
    )


def _report(settings: Settings, kind: str, payload: dict) -> str:
    out_dir = settings.reports_path / kind
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    path = out_dir / f"{kind}_{stamp}.json"
    path.write_text(
        json.dumps({"versions": settings.versions(), **payload}, indent=2, default=str),
        encoding="utf-8",
    )
    return str(path)


# --------------------------------------------------------------------------- #
# SETUP — Setup Quality Gate (Section 15)                                      #
# --------------------------------------------------------------------------- #
def check_setup(settings: Settings) -> list[Criterion]:
    cfg = load_ranking_config()
    meta = load_metadata_config()
    scorer = SetupQualityScorer(cfg, meta)
    out: list[Criterion] = []

    symbol = meta.symbols()[0]
    good = _good_candidate(symbol)

    # 1) Deterministic + reproducible: same inputs ⇒ identical score (two scorers).
    s1 = scorer.score(good)
    s2 = SetupQualityScorer(cfg, meta).score(good)
    deterministic = abs(s1.total - s2.total) < 1e-12 and s1.components == s2.components
    out.append(
        Criterion.ok("deterministic_reproducible", f"score={s1.total:.4f} reproduced exactly")
        if deterministic
        else Criterion.fail("deterministic_reproducible", "score not reproducible")
    )

    # 2) All seven components documented and within their configured maxima.
    comps = s1.components
    in_range = all(
        name in comps and 0.0 <= comps[name] <= cfg.components[name] for name in cfg.components
    )
    sum_ok = abs(sum(comps.values()) - s1.total) < 1e-9
    out.append(
        Criterion.ok(
            "components_documented",
            f"{len(comps)}/7 components, each within max; sum=={s1.total:.4f}",
        )
        if in_range and sum_ok and len(comps) == 7
        else Criterion.fail("components_documented", f"component issue: {comps}")
    )

    # 3) Threshold is a validated tunable within range (Section 15 / SETUP gate).
    threshold_ok = 0.0 < cfg.threshold <= cfg.max_score
    out.append(
        Criterion.ok("threshold_validated", f"threshold={cfg.threshold} of {cfg.max_score}")
        if threshold_ok
        else Criterion.fail("threshold_validated", f"threshold out of range: {cfg.threshold}")
    )

    # 4) A clean high-quality setup is approved (score clears threshold, no blocker).
    out.append(
        Criterion.ok("good_setup_approved", f"score={s1.total:.1f} >= {cfg.threshold}, approved")
        if s1.approved
        else Criterion.fail("good_setup_approved", f"clean setup not approved: {s1.to_dict()}")
    )

    # 5) Hard blockers are NEVER bypassed by a high score (forced-failure tests).
    toxic = _good_candidate(symbol, spread_bps=80.0)  # toxic spread blocker
    no_trade = _good_candidate(symbol, regime=sorted(NO_TRADE_REGIMES)[0])  # no-trade regime
    neg_ev = _good_candidate(symbol, expected_edge_frac=0.0005)  # EV < costs
    stale = _good_candidate(symbol, data_fresh=False)
    blocked_ok = (
        not scorer.score(toxic).approved
        and "spread_above_threshold" in scorer.score(toxic).blockers
        and not scorer.score(no_trade).approved
        and not scorer.score(neg_ev).approved
        and not scorer.score(stale).approved
    )
    out.append(
        Criterion.ok(
            "hard_blockers_enforced",
            "toxic-spread / no-trade-regime / negative-EV / stale-data all rejected "
            "despite passing score",
        )
        if blocked_ok
        else Criterion.fail("hard_blockers_enforced", "a hard blocker was bypassed")
    )

    # 6) Ranking is deterministic and attributes the decision (Section 15).
    engine = CandidateRankingEngine(cfg, meta)
    cands = [
        _good_candidate(symbol, signal_strength=0.95, confirmation=0.9),  # best
        _good_candidate(symbol, signal_strength=0.6, confirmation=0.5),
        toxic,  # blocked → rejected alternative
    ]
    r1 = engine.rank(cands)
    r2 = engine.rank(cands)
    winner = r1.winner
    rank_ok = (
        winner is not None
        and winner.rank == 1
        and [x.score.total for x in r1.selected] == [x.score.total for x in r2.selected]
        and any("hard_blocker" in r.reason for r in r1.rejected)
    )
    out.append(
        Criterion.ok(
            "ranking_attributes_decision",
            f"winner score={winner.score.total:.1f}; "
            f"{len(r1.rejected)} rejected alternatives recorded"
            if winner is not None
            else "no winner",
        )
        if rank_ok
        else Criterion.fail("ranking_attributes_decision", "ranking not deterministic/attributed")
    )

    _report(
        settings,
        "ranking",
        {
            "gate": "SETUP",
            "ranking_version": cfg.ranking_version,
            "threshold": cfg.threshold,
            "good_setup": s1.to_dict(),
            "attribution": r1.attribution(),
        },
    )
    return out


# --------------------------------------------------------------------------- #
# RISK — Risk Policy Gate (Section 17)                                         #
# --------------------------------------------------------------------------- #
def check_risk(settings: Settings) -> list[Criterion]:
    rcfg = load_risk_config()
    meta = load_metadata_config()
    env = rcfg.envelope
    rm = RiskManager(rcfg, meta)
    out: list[Criterion] = []

    equity = 100_000.0
    btc = "BTC/USDT:USDT"
    flat = AccountState(
        portfolio=PortfolioState(equity=equity),
        breakers=BreakerInputs(equity=equity, peak_equity=equity, daily_pnl=0.0),
    )

    # 1) Per-trade sizing identity: qty × |entry − stop| ≈ equity × risk_pct.
    #    stop_frac=0.02 keeps notional small enough that no portfolio cap binds,
    #    so the pure deterministic formula is observed.
    base = _good_candidate(btc, stop_frac=0.02)
    d = rm.evaluate(base, flat)
    target = equity * rcfg.base_risk_pct
    sizing_ok = d.approved and abs(d.risk_amount - target) <= target * 0.05
    out.append(
        Criterion.ok(
            "sizing_formula_correct",
            f"risk_amount={d.risk_amount:.2f} ≈ equity×risk_pct={target:.2f} "
            f"(qty={d.qty}, lev={d.leverage:.3f})",
        )
        if sizing_ok
        else Criterion.fail("sizing_formula_correct", f"sizing off: {d.to_dict()}")
    )

    # 2) risk_pct never exceeds the envelope cap, and config can only tighten it.
    clamp = RiskEnvelope.from_config({"max_risk_pct_per_trade": 99.0, "max_leverage": 999})
    risk_cap_ok = (
        d.risk_pct_used <= env.max_risk_pct_per_trade + 1e-9
        and rcfg.base_risk_pct <= env.max_risk_pct_per_trade
        and clamp.max_risk_pct_per_trade <= 0.02
        and clamp.max_leverage <= 10.0
    )
    out.append(
        Criterion.ok(
            "risk_pct_within_envelope",
            f"risk_pct_used={d.risk_pct_used:.4f} <= cap {env.max_risk_pct_per_trade}; "
            "config clamps to code ceilings",
        )
        if risk_cap_ok
        else Criterion.fail("risk_pct_within_envelope", "risk_pct/envelope clamp violated")
    )

    # 3) Leverage is a consequence, capped (forced: a tiny stop wants huge notional).
    lev = rm.evaluate(_good_candidate(btc, stop_frac=0.0001), flat)
    lev_ok = (
        lev.approved
        and lev.leverage <= env.max_leverage + 1e-6
        and "leverage_capped" in lev.reasons
    )
    out.append(
        Criterion.ok(
            "leverage_capped",
            f"tiny-stop candidate resized: leverage={lev.leverage:.2f} <= {env.max_leverage}",
        )
        if lev_ok
        else Criterion.fail("leverage_capped", f"leverage not capped: {lev.to_dict()}")
    )

    # 4) Min-notional gate rejects a sub-minimum order (Section 17).
    small_state = AccountState(
        portfolio=PortfolioState(equity=100.0),
        breakers=BreakerInputs(equity=100.0, peak_equity=100.0, daily_pnl=0.0),
    )
    tiny = rm.evaluate(_good_candidate("SOL/USDT:USDT", stop_frac=0.05), small_state)
    out.append(
        Criterion.ok("min_notional_gate", f"sub-minimum order rejected: {tiny.reasons}")
        if not tiny.approved and any("below_min" in r for r in tiny.reasons)
        else Criterion.fail(
            "min_notional_gate", f"sub-minimum order not rejected: {tiny.to_dict()}"
        )
    )

    # 5) Portfolio heat cap enforced: a near-full book resizes the new trade to fit.
    heat_preload = Position(
        symbol="ETH/USDT:USDT",
        side=1,
        qty=0.01,
        entry_price=3000.0,
        risk_amount=equity * (env.portfolio_heat_cap - 0.002),
        beta_to_btc=0.85,
        regime="calm",
    )
    heat_state = AccountState(
        portfolio=PortfolioState(equity=equity, positions=(heat_preload,)),
        breakers=BreakerInputs(equity=equity, peak_equity=equity, daily_pnl=0.0),
    )
    hd = rm.evaluate(_good_candidate(btc, stop_frac=0.01), heat_state)
    new_heat = heat_state.portfolio.heat() + (hd.risk_amount / equity if hd.approved else 0.0)
    heat_ok = (
        hd.approved and "heat_capped" in hd.reasons and new_heat <= env.portfolio_heat_cap + 1e-9
    )
    out.append(
        Criterion.ok(
            "portfolio_heat_cap_enforced",
            f"resized to fit heat: total heat={new_heat:.4f} <= {env.portfolio_heat_cap}",
        )
        if heat_ok
        else Criterion.fail("portfolio_heat_cap_enforced", f"heat cap not enforced: {hd.to_dict()}")
    )

    # 6) Net beta-to-BTC cap enforced. (a) A book already at the cap rejects a new
    #    same-direction trade (no headroom); (b) a partially-loaded book RESIZES the
    #    new trade so |net beta| stays within the cap.
    beta_preload = Position(
        symbol="ETH/USDT:USDT",
        side=1,
        qty=12.0,
        entry_price=3000.0,  # ~0.306 net beta
        risk_amount=100.0,
        beta_to_btc=0.85,
        regime="calm",
    )
    beta_full = AccountState(
        portfolio=PortfolioState(equity=equity, positions=(beta_preload,)),
        breakers=BreakerInputs(equity=equity, peak_equity=equity, daily_pnl=0.0),
    )
    bd = rm.evaluate(_good_candidate(btc, side=1, stop_frac=0.02), beta_full)

    half_preload = Position(
        symbol="ETH/USDT:USDT",
        side=1,
        qty=8.0,
        entry_price=3000.0,  # ~0.204 net beta
        risk_amount=100.0,
        beta_to_btc=0.85,
        regime="calm",
    )
    beta_half = AccountState(
        portfolio=PortfolioState(equity=equity, positions=(half_preload,)),
        breakers=BreakerInputs(equity=equity, peak_equity=equity, daily_pnl=0.0),
    )
    rbd = rm.evaluate(_good_candidate(btc, side=1, stop_frac=0.02), beta_half)
    post_net = beta_half.portfolio.net_beta() + (
        rbd.notional / equity if rbd.approved else 0.0
    )  # BTC beta = 1.0
    reject_ok = not bd.approved and any("net_beta_cap" in r for r in bd.reasons)
    resize_ok = (
        rbd.approved and "beta_capped" in rbd.reasons and post_net <= env.net_beta_btc_cap + 1e-9
    )
    out.append(
        Criterion.ok(
            "net_beta_cap_enforced",
            f"over-cap book rejects ({bd.reasons}); partial book resizes to net={post_net:.3f} "
            f"<= {env.net_beta_btc_cap}",
        )
        if reject_ok and resize_ok
        else Criterion.fail(
            "net_beta_cap_enforced",
            f"beta cap not enforced: reject={bd.to_dict()} resize={rbd.to_dict()}",
        )
    )

    # 7) Max-concurrent-positions cap enforced.
    full_positions = tuple(
        Position(
            symbol=f"S{i}",
            side=1,
            qty=0.001,
            entry_price=100.0,
            risk_amount=1.0,
            beta_to_btc=0.0,
            regime=("a", "a", "b", "b", "c")[i],
        )
        for i in range(rcfg.max_concurrent_total)
    )
    full_state = AccountState(
        portfolio=PortfolioState(equity=equity, positions=full_positions),
        breakers=BreakerInputs(equity=equity, peak_equity=equity, daily_pnl=0.0),
    )
    fd = rm.evaluate(_good_candidate(btc, regime="z"), full_state)
    out.append(
        Criterion.ok("max_positions_enforced", f"full book rejects new entry: {fd.reasons}")
        if not fd.approved and "max_concurrent_total" in fd.reasons
        else Criterion.fail("max_positions_enforced", f"concurrency not enforced: {fd.to_dict()}")
    )

    # 8) Daily-loss circuit breaker halts new entries (forced trip).
    dl_state = AccountState(
        portfolio=PortfolioState(equity=equity),
        breakers=BreakerInputs(
            equity=equity, peak_equity=equity, daily_pnl=-equity * (env.daily_loss_limit + 0.005)
        ),
    )
    dld = rm.evaluate(base, dl_state)
    out.append(
        Criterion.ok("daily_loss_breaker", f"halted: {dld.blocker}")
        if dld.action == "block" and dld.blocker is not None and "daily_loss" in dld.blocker
        else Criterion.fail(
            "daily_loss_breaker", f"daily-loss breaker did not halt: {dld.to_dict()}"
        )
    )

    # 9) Max-drawdown circuit breaker halts new entries (forced trip).
    dd_equity = equity * (1.0 - env.max_drawdown_limit - 0.02)
    dd_state = AccountState(
        portfolio=PortfolioState(equity=dd_equity),
        breakers=BreakerInputs(equity=dd_equity, peak_equity=equity, daily_pnl=0.0),
    )
    ddd = rm.evaluate(base, dd_state)
    out.append(
        Criterion.ok("drawdown_breaker", f"halted: {ddd.blocker}")
        if ddd.action == "block" and ddd.blocker is not None and "drawdown" in ddd.blocker
        else Criterion.fail("drawdown_breaker", f"drawdown breaker did not halt: {ddd.to_dict()}")
    )

    # 10) Reconciliation mismatch / unknown order halts new entries (Section 17).
    recon_state = AccountState(
        portfolio=PortfolioState(equity=equity),
        breakers=BreakerInputs(equity=equity, peak_equity=equity, daily_pnl=0.0, reconciled=False),
    )
    rr = rm.evaluate(base, recon_state)
    unknown = rm.evaluate(
        base,
        AccountState(
            portfolio=PortfolioState(equity=equity),
            breakers=BreakerInputs(equity=equity, peak_equity=equity, daily_pnl=0.0),
            unknown_order_present=True,
        ),
    )
    out.append(
        Criterion.ok("reconciliation_halts", "unreconciled state and unknown order both halt")
        if rr.action == "block" and unknown.action == "block"
        else Criterion.fail(
            "reconciliation_halts", f"recon={rr.to_dict()} unknown={unknown.to_dict()}"
        )
    )

    # 11) Kill switch halts new entries (forced trip on an ISOLATED switch).
    out.append(_risk_kill_switch_criterion(rcfg, meta, base, equity))

    _report(
        settings,
        "risk",
        {
            "gate": "RISK",
            "risk_policy_version": rcfg.risk_policy_version,
            "envelope": env.to_dict(),
            "baseline_decision": d.to_dict(),
        },
    )
    return out


def _risk_kill_switch_criterion(
    rcfg: object, meta: MetadataConfig, base: Candidate, equity: float
) -> Criterion:
    with tempfile.TemporaryDirectory() as tmp:
        ks = KillSwitch(
            Settings(
                _env_file=None,
                data_lake_path=Path(tmp) / "datalake",
                redis_url="redis://127.0.0.1:1/0",  # unreachable: file backend only
            )
        )
        rm = RiskManager(rcfg, meta, kill_switch=ks)  # type: ignore[arg-type]
        state = AccountState(
            portfolio=PortfolioState(equity=equity),
            breakers=BreakerInputs(equity=equity, peak_equity=equity, daily_pnl=0.0),
        )
        before = rm.evaluate(base, state)
        ks.engage(reason="risk-gate-selftest", actor="gate")
        during = rm.evaluate(base, state)
        ks.disengage(actor="gate")
        after = rm.evaluate(base, state)
    if (
        before.approved
        and during.action == "block"
        and during.blocker == "kill_switch_engaged"
        and after.approved
    ):
        return Criterion.ok("kill_switch_blocks", "kill switch halts new entries; clears on reset")
    return Criterion.fail(
        "kill_switch_blocks",
        f"kill switch did not halt: before={before.approved} during={during.to_dict()}",
    )


# --------------------------------------------------------------------------- #
# EXEC — Execution Gate (Section 18)                                          #
# --------------------------------------------------------------------------- #
def check_exec(settings: Settings) -> list[Criterion]:
    ecfg = load_execution_config()
    rcfg = load_risk_config()
    meta = load_metadata_config()
    ownership = OwnershipPolicy(settings)
    rm = RiskManager(rcfg, meta)
    out: list[Criterion] = []

    equity = 100_000.0
    flat = AccountState(
        portfolio=PortfolioState(equity=equity),
        breakers=BreakerInputs(equity=equity, peak_equity=equity, daily_pnl=0.0),
    )
    btc = "BTC/USDT:USDT"
    builder = OrderBuilder(ecfg, ownership)

    # 1) Every exchange-supported order type is implemented.
    impl = {t.value for t in OrderType}
    missing = set(meta.supported_order_types) - impl
    out.append(
        Criterion.ok(
            "order_types_supported", f"all {len(meta.supported_order_types)} types implemented"
        )
        if not missing
        else Criterion.fail("order_types_supported", f"unimplemented order types: {missing}")
    )

    # 2) Order builder respects tick / lot / min-notional for every symbol.
    builder_ok = True
    detail = []
    for sym in meta.symbols():
        spec = meta.spec(sym)
        assert spec is not None
        cand = _good_candidate(sym, stop_frac=0.01)
        dec = rm.evaluate(cand, flat)
        bres = builder.build(cand, dec, spec)
        if not bres.ok or bres.plan is None:
            builder_ok = False
            detail.append(f"{sym}:build_failed:{bres.reason}")
            continue
        tick = float(spec.fields["tick_size"])
        step = float(spec.fields["qty_step"])
        e = bres.plan.entry
        price = e.price if e.price is not None else cand.entry_price
        on_tick = abs((price / tick) - round(price / tick)) < 1e-6
        on_step = abs((bres.plan.qty / step) - round(bres.plan.qty / step)) < 1e-6
        notion_ok = bres.plan.qty * price >= float(spec.fields["min_notional"])
        if not (on_tick and on_step and notion_ok):
            builder_ok = False
            detail.append(f"{sym}:tick={on_tick},step={on_step},notional={notion_ok}")
    # Forced failure: a candidate the risk manager won't approve cannot be built.
    small_state = AccountState(
        portfolio=PortfolioState(equity=100.0),
        breakers=BreakerInputs(equity=100.0, peak_equity=100.0, daily_pnl=0.0),
    )
    sol = "SOL/USDT:USDT"
    rejected_dec = rm.evaluate(_good_candidate(sol, stop_frac=0.05), small_state)
    sol_spec = meta.spec(sol)
    assert sol_spec is not None
    reject_build = builder.build(_good_candidate(sol, stop_frac=0.05), rejected_dec, sol_spec)
    out.append(
        Criterion.ok(
            "builder_respects_constraints", "tick/lot/min-notional respected; bad size rejected"
        )
        if builder_ok and not reject_build.ok
        else Criterion.fail("builder_respects_constraints", f"constraint issues: {detail}")
    )

    # 3) Atomic exchange-side SL/TP on entry (Section 2.2).
    venue = SimulatedVenue(meta)
    engine = ExecutionEngine(ecfg, meta, ownership, venue)
    cand = _good_candidate(btc, stop_frac=0.01, tp_frac=0.02)  # finite TP
    res = engine.execute(cand, rm.evaluate(cand, flat), realized_slippage_frac=0.0005)
    pos = res.position
    atomic_ok = (
        res.placed
        and pos is not None
        and pos.stop_order_id is not None
        and pos.tp_order_id is not None
        and venue.order_status(pos.stop_order_id) == "open"
        and venue.order_status(pos.tp_order_id) == "open"
    )
    out.append(
        Criterion.ok(
            "atomic_exchange_side_sl_tp", "entry attaches exchange-resident SL + TP atomically"
        )
        if atomic_ok
        else Criterion.fail("atomic_exchange_side_sl_tp", f"SL/TP not atomic: {res.to_dict()}")
    )

    # 4) Native trailing for no-fixed-TP (momentum) exits, resting on the venue.
    venue2 = SimulatedVenue(meta)
    engine2 = ExecutionEngine(ecfg, meta, ownership, venue2)
    mom = _good_candidate(btc, stop_frac=0.01, tp_frac=1.0)  # tp_frac>=0.5 ⇒ no fixed TP
    mres = engine2.execute(mom, rm.evaluate(mom, flat), realized_slippage_frac=0.0005)
    trail_ok = (
        mres.placed
        and mres.position is not None
        and mres.position.trail_order_id is not None
        and venue2.open_orders[mres.position.trail_order_id].order_type is OrderType.TRAILING_STOP
        and venue2.open_orders[mres.position.trail_order_id].reduce_only
    )
    out.append(
        Criterion.ok("native_trailing_stop", "momentum exit uses an exchange-native trailing stop")
        if trail_ok
        else Criterion.fail("native_trailing_stop", f"no native trailing: {mres.to_dict()}")
    )

    # 5) Cancel + cancel/replace work (own orders only).
    stop_id = res.position.stop_order_id if res.position else None
    replaced = None
    cancel_ok = False
    if stop_id is not None:
        new_stop = Order(
            client_id=ownership.new_client_id("stop"),
            symbol=btc,
            side=SELL,
            qty=res.plan.qty if res.plan else 1.0,
            order_type=OrderType.STOP_MARKET,
            role="stop",
            stop_price=49000.0,
            reduce_only=True,
            tags=ownership.tags(),
        )
        replaced = venue.cancel_replace(stop_id, new_stop)
        cancel_ok = (
            replaced == new_stop.client_id
            and venue.order_status(stop_id) == "cancelled"
            and venue.order_status(new_stop.client_id) == "open"
            and venue.cancel(new_stop.client_id) is True
        )
    out.append(
        Criterion.ok("cancel_and_replace", "cancel + atomic cancel/replace work on own orders")
        if cancel_ok
        else Criterion.fail("cancel_and_replace", f"cancel/replace failed (replaced={replaced})")
    )

    # 6) Partial-fill handling.
    venue3 = SimulatedVenue(meta)
    engine3 = ExecutionEngine(ecfg, meta, ownership, venue3)
    pcand = _good_candidate(btc, stop_frac=0.01, tp_frac=0.02)
    pdec = rm.evaluate(pcand, flat)
    pres = engine3.execute(
        pcand, pdec, realized_slippage_frac=0.0005, fill_ratio=ecfg.simulated_partial_fill_ratio
    )
    partial_ok = (
        pres.placed
        and not pres.fully_filled
        and pres.position is not None
        and abs(pres.position.qty - pdec.qty * ecfg.simulated_partial_fill_ratio) < 1e-9
        and pres.remaining_qty > 0
    )
    out.append(
        Criterion.ok(
            "partial_fill_handled",
            f"partial fill {ecfg.simulated_partial_fill_ratio:g}: "
            f"filled={pdec.qty * ecfg.simulated_partial_fill_ratio:g}, "
            f"remaining={pres.remaining_qty}",
        )
        if partial_ok
        else Criterion.fail("partial_fill_handled", f"partial fill mishandled: {pres.to_dict()}")
    )

    # 7) Slippage measured (expected vs actual fill price) on a TAKER entry.
    venue5 = SimulatedVenue(meta)
    engine5 = ExecutionEngine(ecfg, meta, ownership, venue5)
    tcand = _good_candidate(btc, stop_frac=0.01, tp_frac=0.02)
    tfill_res = engine5.execute(
        tcand, rm.evaluate(tcand, flat), realized_slippage_frac=0.0008, entry_style="taker"
    )
    fill = tfill_res.fill
    slip_ok = (
        fill is not None
        and fill.actual_price != fill.expected_price
        and fill.slippage_frac > 0
        and fill.slippage_cost > 0
        and not fill.maker
    )
    out.append(
        Criterion.ok(
            "slippage_measured",
            f"taker fill expected={fill.expected_price:.2f} actual={fill.actual_price:.2f} "
            f"slippage_frac={fill.slippage_frac:.5f}"
            if fill
            else "no fill",
        )
        if slip_ok
        else Criterion.fail(
            "slippage_measured", f"slippage not measured: {fill.to_dict() if fill else None}"
        )
    )

    # 8) Reconciliation works on a clean own-only book.
    recon = Reconciler(ownership)
    known_orders = set(venue.open_orders.keys())
    known_positions = set(venue.positions.keys())
    rr = recon.reconcile(
        venue.open_orders,
        venue.positions,
        known_order_ids=known_orders,
        known_position_symbols=known_positions,
    )
    out.append(
        Criterion.ok("reconciliation_works", "own-only book reconciles cleanly (no false halt)")
        if rr.ok and not rr.halt_required
        else Criterion.fail("reconciliation_works", f"false reconciliation halt: {rr.to_dict()}")
    )

    # 9) Revalidate-before-execute: a toxic-spread signal is not placed.
    venue4 = SimulatedVenue(meta)
    engine4 = ExecutionEngine(ecfg, meta, ownership, venue4)
    toxic = _good_candidate(btc, spread_bps=80.0, stop_frac=0.01)
    tdec = rm.evaluate(toxic, flat)
    tres = engine4.execute(toxic, tdec, realized_slippage_frac=0.0005)
    out.append(
        Criterion.ok("revalidate_before_execute", f"toxic-spread signal not placed: {tres.reason}")
        if not tres.placed and "toxic_spread" in tres.reason
        else Criterion.fail(
            "revalidate_before_execute", f"toxic signal was placed: {tres.to_dict()}"
        )
    )

    _report(
        settings,
        "execution",
        {
            "gate": "EXEC",
            "execution_policy_version": ecfg.execution_policy_version,
            "sample_plan": res.plan.to_dict() if res.plan else None,
            "sample_fill": fill.to_dict() if fill else None,
        },
    )
    return out


# --------------------------------------------------------------------------- #
# KILL — Kill Switch Verification Gate (Section 2.2)                           #
# --------------------------------------------------------------------------- #
def check_kill(settings: Settings) -> list[Criterion]:
    from fastapi.testclient import TestClient

    from src.api import create_app

    ecfg = load_execution_config()
    rcfg = load_risk_config()
    meta = load_metadata_config()
    out: list[Criterion] = []

    btc = "BTC/USDT:USDT"
    cand = _good_candidate(btc, stop_frac=0.01)
    equity = 100_000.0
    state = AccountState(
        portfolio=PortfolioState(equity=equity),
        breakers=BreakerInputs(equity=equity, peak_equity=equity, daily_pnl=0.0),
    )

    with tempfile.TemporaryDirectory() as tmp:
        iso = Settings(
            _env_file=None,
            app_env="paper",
            dashboard_auth_mode="basic",
            dashboard_username="admin",
            dashboard_password="secret",
            data_lake_path=Path(tmp) / "datalake",
            redis_url="redis://127.0.0.1:1/0",  # dashboard DOWN ⇒ file backend only
        )
        ks = KillSwitch(iso)

        # 1) CLI / file backend works independent of any web process (Section 2.2).
        cli_ok = False
        if not ks.engaged():
            ks.engage(reason="kill-gate-cli", actor="cli")
            cli_ok = ks.engaged() and ks.status()["file_backend"] is True
            ks.disengage(actor="cli")
            cli_ok = cli_ok and not ks.engaged()
        out.append(
            Criterion.ok("cli_kill_independent", "CLI/file kill switch toggles with no web process")
            if cli_ok
            else Criterion.fail("cli_kill_independent", "CLI kill switch did not toggle")
        )

        # 2) Kill switch halts new entries (risk + execution both refuse).
        rm = RiskManager(rcfg, meta, kill_switch=ks)
        venue = SimulatedVenue(meta)
        engine = ExecutionEngine(ecfg, meta, OwnershipPolicy(iso), venue, kill_switch=ks)
        ks.engage(reason="kill-gate-halt", actor="cli")
        risk_blocked = rm.evaluate(cand, state).action == "block"
        exec_blocked = not engine.execute(cand, rm.evaluate(cand, state)).placed
        flattened = len(venue.positions) == 0  # no new position opened while halted
        ks.disengage(actor="cli")
        out.append(
            Criterion.ok("halts_trading", "risk blocks + execution refuses while engaged")
            if risk_blocked and exec_blocked and flattened
            else Criterion.fail("halts_trading", "kill switch did not halt new entries")
        )

        # 3) Dashboard kill switch works (and fires an alert) — via the API.
        app = create_app(iso)
        client = TestClient(app)
        sink = get_alert_sink()
        before = len(sink.recent(limit=1000))
        engaged_resp = client.post("/api/killswitch/engage", auth=("admin", "secret"))
        status_resp = client.get("/api/killswitch", auth=("admin", "secret"))
        dash_engaged = (
            engaged_resp.status_code == 200
            and status_resp.json()["engaged"] is True
            and KillSwitch(iso).engaged() is True
        )
        alert_fired = len(sink.recent(limit=1000)) > before

        # 4) Recovery requires a manual, explicit action (Section 35).
        refused = client.post("/api/killswitch/disengage", auth=("admin", "secret"))
        still_engaged = KillSwitch(iso).engaged() is True  # refused without confirm
        cleared = client.post("/api/killswitch/disengage?confirm=true", auth=("admin", "secret"))
        recovered = cleared.status_code == 200 and KillSwitch(iso).engaged() is False

        out.append(
            Criterion.ok(
                "dashboard_kill_works", "dashboard engage halts and reflects engaged state"
            )
            if dash_engaged
            else Criterion.fail(
                "dashboard_kill_works", f"dashboard kill failed: {engaged_resp.status_code}"
            )
        )
        out.append(
            Criterion.ok("alert_on_activation", "kill-switch activation fires a critical alert")
            if alert_fired
            else Criterion.fail("alert_on_activation", "no alert on kill-switch activation")
        )
        out.append(
            Criterion.ok(
                "recovery_requires_manual_review",
                "disengage refused without confirm; cleared only on explicit manual confirm",
            )
            if refused.status_code == 400 and still_engaged and recovered
            else Criterion.fail(
                "recovery_requires_manual_review", "recovery did not require explicit confirmation"
            )
        )

    # 5) Verified in a non-live (paper) environment by deliberate trip.
    out.append(
        Criterion.ok("tested_in_paper", f"deliberate trip in mode={settings.trading_mode.value}")
        if not settings.live_trading_allowed
        else Criterion.fail("tested_in_paper", "must be tested outside live")
    )

    _report(settings, "execution", {"gate": "KILL", "criteria": [c.id for c in out]})
    return out


# --------------------------------------------------------------------------- #
# ORDER-OWN — Order Ownership Gate (Section 7)                                 #
# --------------------------------------------------------------------------- #
def check_order_own(settings: Settings) -> list[Criterion]:
    ecfg = load_execution_config()
    rcfg = load_risk_config()
    meta = load_metadata_config()
    ownership = OwnershipPolicy(settings)
    rm = RiskManager(rcfg, meta)
    out: list[Criterion] = []

    equity = 100_000.0
    flat = AccountState(
        portfolio=PortfolioState(equity=equity),
        breakers=BreakerInputs(equity=equity, peak_equity=equity, daily_pnl=0.0),
    )
    btc = "BTC/USDT:USDT"

    # 1) Ownership identifiers configured (Section 7).
    out.append(
        Criterion.ok(
            "ownership_configured",
            f"prefix={ownership.prefix!r} instance={ownership.bot_instance_id!r}",
        )
        if ownership.configured()
        else Criterion.fail(
            "ownership_configured", "ORDER_CLIENT_ID_PREFIX / BOT_INSTANCE_ID unset"
        )
    )

    # 2) Every bot order leg carries the prefix + provenance tags.
    venue = SimulatedVenue(meta)
    engine = ExecutionEngine(ecfg, meta, ownership, venue)
    cand = _good_candidate(btc, stop_frac=0.01, tp_frac=0.02)
    res = engine.execute(cand, rm.evaluate(cand, flat), realized_slippage_frac=0.0005)
    legs = res.plan.legs() if res.plan else []
    prefixed = bool(legs) and all(ownership.is_own(o.client_id) for o in legs)
    tagged = all(o.tags.get("bot_instance_id") and o.tags.get("config_version") for o in legs)
    out.append(
        Criterion.ok("all_orders_prefixed", f"{len(legs)} legs carry the prefix + provenance tags")
        if prefixed and tagged
        else Criterion.fail("all_orders_prefixed", "an order leg lacked the prefix/tags")
    )

    # 3) Unknown order detected → halt + alert (forced foreign order).
    recon = Reconciler(ownership)
    sink = get_alert_sink()
    before = len(sink.recent(limit=1000))
    foreign = Order(
        client_id="MANUAL_humantrader_42",
        symbol=btc,
        side=BUY,
        qty=1.0,
        order_type=OrderType.LIMIT,
        role="entry",
        price=49000.0,
        tags={},
    )
    venue.inject_foreign_order(foreign)
    known_orders = {oid for oid in venue.open_orders if ownership.is_own(oid)}
    rr = recon.reconcile(
        venue.open_orders,
        venue.positions,
        known_order_ids=known_orders,
        known_position_symbols=set(venue.positions),
    )
    alerted = len(sink.recent(limit=1000)) > before
    out.append(
        Criterion.ok(
            "unknown_order_detected",
            f"foreign order flagged + halt: {rr.unknown_orders}; alert fired={alerted}",
        )
        if rr.halt_required and "MANUAL_humantrader_42" in rr.unknown_orders and alerted
        else Criterion.fail("unknown_order_detected", f"foreign order not detected: {rr.to_dict()}")
    )

    # 4) Unknown position detected → halt (forced foreign position).
    venue2 = SimulatedVenue(meta)
    venue2.inject_foreign_position("XRP/USDT:USDT", side=1, qty=100.0, price=0.5)
    rr2 = recon.reconcile(
        venue2.open_orders,
        venue2.positions,
        known_order_ids=set(),
        known_position_symbols=set(),
    )
    out.append(
        Criterion.ok(
            "unknown_position_detected", f"foreign position flagged + halt: {rr2.unknown_positions}"
        )
        if rr2.halt_required and "XRP/USDT:USDT" in rr2.unknown_positions
        else Criterion.fail(
            "unknown_position_detected", f"foreign position not detected: {rr2.to_dict()}"
        )
    )

    # 5) Cleanup touches only own orders (never foreign).
    cannot_touch = venue.cancel("MANUAL_humantrader_42", owned_only=True) is False
    own_stop = res.position.stop_order_id if res.position else None
    can_touch_own = own_stop is not None and venue.cancel(own_stop, owned_only=True) is True
    out.append(
        Criterion.ok(
            "cleanup_touches_only_own", "refuses to cancel the foreign order; cancels its own"
        )
        if cannot_touch and can_touch_own
        else Criterion.fail("cleanup_touches_only_own", "ownership boundary not enforced on cancel")
    )

    # 6) Emergency close requires explicit confirmation (Section 7/35).
    venue3 = SimulatedVenue(meta)
    venue3.inject_foreign_position("DOGE/USDT:USDT", side=1, qty=1.0, price=0.1)
    refused = False
    try:
        venue3.emergency_close_all(confirm=False)
    except PermissionError:
        refused = True
    confirmed = (
        venue3.emergency_close_all(confirm=True) >= 1
        if ecfg.emergency_close_requires_confirmation
        else False
    )
    out.append(
        Criterion.ok(
            "emergency_close_requires_confirmation", "refused without confirm; ran with confirm"
        )
        if refused and confirmed
        else Criterion.fail(
            "emergency_close_requires_confirmation", "emergency close not gated by confirmation"
        )
    )

    _report(
        settings,
        "execution",
        {
            "gate": "ORDER-OWN",
            "prefix": ownership.prefix,
            "bot_instance_id": ownership.bot_instance_id,
            "sample_legs": [o.to_dict() for o in legs],
        },
    )
    return out
