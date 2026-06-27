"""Real-data (replay) paper sessions over a downloaded DATA_VERSION snapshot.

The research-stage counterpart of the synthetic paper session (``src/paper/run.py``):
instead of fabricated candidates, it derives the candidate stream from REAL lake data
— a strategy is run over the snapshot's feature frame (the one Parity-Rule pipeline),
each signal becomes a :class:`Candidate` from that decision-time row, and the realized
post-entry price move is read forward from the real bars. Those candidates are pushed
through the SAME paper pipeline (ranking → risk → execution → SimulatedVenue), so the
resulting :class:`PaperTrade` records carry full execution-quality fields and persist to
``paper_runs``/``paper_trades`` (shadow-only; no live influence). Until a live/replay
feed exists (later milestone), this is how paper forward-tests on real data.
"""

from __future__ import annotations

from datetime import UTC, datetime

from src.backtest.service import build_lake_inputs, lake_candidate_strategy, make_strategy
from src.config import Settings, get_settings
from src.data.config import DataConfig, load_data_config
from src.data.store import SeriesStore
from src.paper.engine import PaperCandidateInput, PaperTradingEngine
from src.paper.report import PaperReport, build_paper_report
from src.paper.run import persist_paper_session
from src.paper.session import PaperSession
from src.ranking import Candidate
from src.regime import detect_regime
from src.strategies.promotion import is_strategy_promoted

_DEFAULT_HOLD_BARS = 12  # forward horizon for the realized move when a signal omits one


def build_candidate(
    symbol: str,
    row: dict,
    sig,
    *,
    strat_id: str,
    strat_ver: str,
    entry_price: float,
    spread_bps: float,
    promoted: bool,
    data_ok: bool = True,
    risk_scale: float = 1.0,
) -> Candidate:
    """Build a ranking Candidate from a decision-time feature row + a strategy signal.

    Shared by the snapshot paper builder and the real-time live feed so both go through the
    identical Candidate construction + Section-11 regime labelling (the Parity Rule)."""
    return Candidate(
        symbol=symbol,
        strategy=strat_id,
        strategy_version=strat_ver,
        side=sig.side,
        entry_price=entry_price,
        stop_frac=sig.stop_frac,
        tp_frac=sig.tp_frac,
        # Execution geometry → live/paper parity (Section 10). maker/limit/trail/hold come off the
        # Signal; risk_scale is a per-strategy property the caller reads off the strategy object.
        maker=bool(getattr(sig, "maker", False)),
        limit_offset_frac=float(getattr(sig, "limit_offset_frac", 0.0)),
        trail_frac=float(getattr(sig, "trail_frac", 0.0)),
        hold_bars=int(getattr(sig, "hold_bars", 0) or 0),
        risk_scale=float(risk_scale),
        regime=detect_regime(row, spread_bps=spread_bps, data_ok=data_ok),
        session=int(row.get("session_code", 0)),
        features={
            "atr_pct": float(row.get("atr_pct", 0.0)),
            "premium": float(row.get("premium", 0.0)),
            "funding_z": float(row.get("funding_z", 0.0)),
        },
        signal_strength=min(1.0, abs(float(row.get("ret_short", 0.0))) / 0.02),
        confirmation=0.6,
        expected_edge_frac=sig.tp_frac,
        spread_bps=spread_bps,
        slippage_est=0.0005,
        latency_ms=5.0,
        # Research/shadow context flags; live eligibility is judged by the gates, not a run.
        data_fresh=data_ok,
        metadata_verified=True,
        symbol_tradable=True,
        strategy_enabled=True,
        config_live_approved=promoted,
        decision_ts=int(row["decision_ts"]),
    )


def build_lake_paper_inputs(
    data_cfg: DataConfig,
    *,
    timeframe: str,
    symbols: list[str],
    candidate_id: str | None = None,
    settings: Settings | None = None,
    store: SeriesStore | None = None,
    equity: float = 10_000.0,
    hold_bars: int = _DEFAULT_HOLD_BARS,
) -> tuple[list[PaperCandidateInput], str, str]:
    """Build paper candidate inputs from REAL lake data → (inputs, strategy_id, version).

    A per-row strategy (reference momentum or a family-B candidate) is evaluated over the
    snapshot's feature frame; each signal yields a candidate priced at the entry bar's open
    with its realized signed move read ``hold_bars`` bars forward. Cross-asset families
    (A/G) need a multi-symbol portfolio path and are not supported here.
    """
    from src.backtest.config import load_backtest_config

    settings = settings or get_settings()
    bt_cfg = load_backtest_config()
    if candidate_id:
        strategy, strat_id, strat_ver = lake_candidate_strategy(candidate_id)
    else:
        strategy = make_strategy(bt_cfg)
        strat_id = bt_cfg.reference_strategy.name
        strat_ver = bt_cfg.reference_strategy.strategy_version
    if hasattr(strategy, "evaluate_portfolio"):
        raise ValueError(
            f"candidate {candidate_id!r} is a cross-asset (portfolio) strategy; lake paper "
            "supports per-row strategies (reference momentum or family B)"
        )

    series_store = store if store is not None else SeriesStore(settings.data_lake_path)
    lake_inputs = build_lake_inputs(
        series_store,
        exchange_id=data_cfg.exchange_id,
        symbols=symbols,
        timeframe=timeframe,
        base_timeframe=data_cfg.base_timeframe,
        funding_timeframe=data_cfg.funding_timeframe,
        start_ms=data_cfg.window_start_ms,
        end_ms=data_cfg.window_end_ms,
        oi_timeframe=data_cfg.oi_grid,
    )
    promoted = is_strategy_promoted(strat_id, strat_ver)
    out = _eval_strategy_over_lake(
        strategy, strat_id, strat_ver, lake_inputs, hold_bars=hold_bars,
        equity=equity, promoted=promoted,
    )
    return out, strat_id, strat_ver


def _eval_strategy_over_lake(
    strategy,
    strat_id: str,
    strat_ver: str,
    lake_inputs,
    *,
    hold_bars: int,
    equity: float,
    promoted: bool,
) -> list[PaperCandidateInput]:
    """Evaluate ONE per-row strategy over pre-built lake feature frames → candidate inputs.

    Factored out so the single-strategy and multi-strategy (active-set) builders share the
    exact same Candidate construction and forward-move accounting (Parity Rule)."""
    out: list[PaperCandidateInput] = []
    for si in lake_inputs:
        n = len(si.bars)
        # Locate the entry bar by its TIMESTAMP, not by ``decision_ts // iv`` array position:
        # a contract listed after the window start has its first bar at a large ts (not index 0),
        # so the slot index would point at the wrong bar (or past the end) and silently drop every
        # signal. Time is the shared coordinate; the forward hold then walks N array bars from
        # there (preserving the "hold N bars" semantics across any interior gaps).
        pos_by_ts = {int(b["ts"]): i for i, b in enumerate(si.bars)}
        for row in si.frame.rows:
            if row["decision_ts"] < si.activation_ts:
                continue
            sig = strategy.evaluate(row)
            if sig is None:
                continue
            entry_bar = pos_by_ts.get(int(row["decision_ts"]))
            if entry_bar is None:
                continue  # no tradable bar at this timestamp (pre-listing or interior gap)
            entry_price = float(si.bars[entry_bar]["open"])
            exit_bar = min(entry_bar + hold_bars, n - 1)
            exit_price = float(si.bars[exit_bar]["close"])
            exit_move_frac = exit_price / entry_price - 1.0 if entry_price > 0 else 0.0
            spread_bps = si.spread_bps_at(int(row["decision_ts"]))
            cand = build_candidate(
                si.symbol,
                row,
                sig,
                strat_id=strat_id,
                strat_ver=strat_ver,
                entry_price=entry_price,
                spread_bps=spread_bps,
                promoted=promoted,
                risk_scale=float(getattr(strategy, "risk_scale", 1.0)),
            )
            out.append(
                PaperCandidateInput(
                    candidate=cand,
                    equity=equity,
                    exit_move_frac=exit_move_frac,
                    hold_bars=min(hold_bars, exit_bar - entry_bar),
                )
            )
    return out


def resolve_active_strategies(
    settings: Settings | None = None, *, require_real_data: bool = False
) -> tuple[list[tuple], list[str]]:
    """Resolve the ACTIVE promoted strategy set the live/demo engine runs (Section 13).

    Returns ``([(strategy, strat_id, strat_ver), ...], skipped_ids)`` — the top-N promoted
    candidates by validated expectancy_r. Both per-row (family B / reference) AND cross-asset
    (families A/G, portfolio) strategies are included; the REAL-TIME live feed runs both, while
    the offline replay builder filters portfolio ones out. ``skipped_ids`` are only ids no longer
    present in configs/strategies.yaml (stale promotions).

    ``require_real_data=True`` (demo/testnet/live) restricts the set to strategies validated on
    real lake data — synthetic/reference-only promotions are blocked from trading a real
    account (Section 13)."""
    from src.strategies.promotion import active_strategy_ids

    settings = settings or get_settings()
    out: list[tuple] = []
    skipped: list[str] = []
    for cid in active_strategy_ids(settings.strategy_version, require_real_data=require_real_data):
        try:
            strategy, sid, ver = lake_candidate_strategy(cid)
        except ValueError:
            # A promoted id no longer in configs/strategies.yaml (config changed under it) must
            # not crash the live engine — skip it. It re-appears once re-validated.
            skipped.append(cid)
            continue
        # Cross-sectional (basket) strategies — funding_carry (C), residual_momentum (I) — run only
        # through the CrossSectionalEngine (the `paper-basket` path), never the per-symbol live loop.
        # If a promoted basket leaked in here it would either run through the wrong per-symbol
        # vehicle (funding_carry has a degraded evaluate_portfolio) or crash the feed entirely
        # (residual_momentum has no per-symbol evaluate at all → NotImplementedError every tick).
        if getattr(strategy, "cross_sectional", False):
            continue
        out.append((strategy, sid, ver))
    return out, skipped


def build_active_lake_inputs(
    data_cfg: DataConfig,
    *,
    timeframe: str,
    symbols: list[str],
    settings: Settings | None = None,
    store: SeriesStore | None = None,
    equity: float = 10_000.0,
    hold_bars: int = _DEFAULT_HOLD_BARS,
    require_real_data: bool = False,
) -> tuple[list[PaperCandidateInput], list[str]]:
    """Build candidate inputs for the ACTIVE promoted strategy set over one snapshot.

    Runs every active promoted strategy over the same feature frames and concatenates their
    candidates; the engine then arbitrates via ranking + the one-position-per-symbol cap. This
    is the multi-strategy ensemble the live/demo loop uses so demo mirrors live. Returns
    ``(inputs, active_strategy_ids)``; an empty list means nothing is promoted yet."""
    settings = settings or get_settings()
    active, _skipped = resolve_active_strategies(settings, require_real_data=require_real_data)
    # The offline replay path is per-row only; cross-asset (portfolio) strategies run on the
    # real-time live feed, not here.
    active = [(s, sid, ver) for (s, sid, ver) in active if not hasattr(s, "evaluate_portfolio")]
    if not active:
        return [], []
    series_store = store if store is not None else SeriesStore(settings.data_lake_path)
    lake_inputs = build_lake_inputs(
        series_store,
        exchange_id=data_cfg.exchange_id,
        symbols=symbols,
        timeframe=timeframe,
        base_timeframe=data_cfg.base_timeframe,
        funding_timeframe=data_cfg.funding_timeframe,
        start_ms=data_cfg.window_start_ms,
        end_ms=data_cfg.window_end_ms,
        oi_timeframe=data_cfg.oi_grid,
    )
    out: list[PaperCandidateInput] = []
    for strategy, sid, ver in active:
        out.extend(
            _eval_strategy_over_lake(
                strategy, sid, ver, lake_inputs, hold_bars=hold_bars,
                equity=equity, promoted=True,
            )
        )
    return out, [sid for _s, sid, _v in active]


def run_lake_paper_session(
    data_cfg: DataConfig | None = None,
    *,
    settings: Settings | None = None,
    timeframe: str | None = None,
    symbols: list[str] | None = None,
    candidate_id: str | None = None,
    multi_strategy: bool = False,
    dataset_version: str | None = None,
    session_name: str | None = None,
) -> tuple[PaperSession, PaperReport, str]:
    """Run + persist a real-data backtest/replay over a snapshot. Returns (session, report, id).

    ``multi_strategy=True`` runs the ACTIVE PROMOTED ENSEMBLE (all top-N per-row strategies) in
    ONE run — the offline real-data twin of the live engine, with ranking + one-position-per-
    symbol arbitration — and tags the session ``lakebt:…:ensemble`` so its combined statistics
    are viewable on their own (Statistics → Session filter). Otherwise a single ``candidate_id``
    (or the reference) is replayed. Requires the snapshot to be downloaded first."""
    settings = settings or get_settings()
    data_cfg = data_cfg or load_data_config()
    tf = timeframe or data_cfg.base_timeframe
    syms = symbols or data_cfg.active_symbols()
    dsv = dataset_version or data_cfg.data_version
    if multi_strategy:
        inputs, ids = build_active_lake_inputs(
            data_cfg, timeframe=tf, symbols=syms, settings=settings
        )
        strat_id = "ensemble"
        name = session_name or f"lakebt:{data_cfg.exchange_id}:{dsv}:{tf}:ensemble"
    else:
        inputs, strat_id, _ = build_lake_paper_inputs(
            data_cfg, timeframe=tf, symbols=syms, candidate_id=candidate_id, settings=settings
        )
        name = session_name or f"lake:{data_cfg.exchange_id}:{dsv}:{tf}:{strat_id}"
    engine = PaperTradingEngine(
        config_version=settings.config_version,
        universe_version=settings.universe_version,
        settings=settings,
    )
    session = engine.new_session(name)
    engine.process_candidates(inputs, session)
    session.ended_at = datetime.now(UTC)
    report = build_paper_report(session)
    persist_paper_session(session, report, settings)
    return session, report, session.session_id
