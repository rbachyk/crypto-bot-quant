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
from src.data.schema import timeframe_ms
from src.data.store import SeriesStore
from src.paper.engine import PaperCandidateInput, PaperTradingEngine
from src.paper.report import PaperReport, build_paper_report
from src.paper.run import persist_paper_session
from src.paper.session import PaperSession
from src.ranking import Candidate
from src.strategies.promotion import is_strategy_promoted

_DEFAULT_HOLD_BARS = 12  # forward horizon for the realized move when a signal omits one


def _regime(row: dict) -> str:
    slope = float(row.get("trend_slope", 0.0))
    if slope > 1e-4:
        return "trend_up"
    if slope < -1e-4:
        return "trend_down"
    return "range"


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
    iv = timeframe_ms(timeframe)
    promoted = is_strategy_promoted(strat_id, strat_ver)
    out: list[PaperCandidateInput] = []
    for si in lake_inputs:
        n = len(si.bars)
        for row in si.frame.rows:
            if row["decision_ts"] < si.activation_ts:
                continue
            sig = strategy.evaluate(row)  # type: ignore[union-attr]
            if sig is None:
                continue
            entry_bar = row["decision_ts"] // iv
            if entry_bar >= n:
                continue
            entry_price = float(si.bars[entry_bar]["open"])
            exit_bar = min(entry_bar + hold_bars, n - 1)
            exit_price = float(si.bars[exit_bar]["close"])
            exit_move_frac = exit_price / entry_price - 1.0 if entry_price > 0 else 0.0
            cand = Candidate(
                symbol=si.symbol,
                strategy=strat_id,
                strategy_version=strat_ver,
                side=sig.side,
                entry_price=entry_price,
                stop_frac=sig.stop_frac,
                tp_frac=sig.tp_frac,
                regime=_regime(row),
                session=int(row.get("session_code", 0)),
                features={
                    "atr_pct": float(row.get("atr_pct", 0.0)),
                    "premium": float(row.get("premium", 0.0)),
                    "funding_z": float(row.get("funding_z", 0.0)),
                },
                signal_strength=min(1.0, abs(float(row.get("ret_short", 0.0))) / 0.02),
                confirmation=0.6,
                expected_edge_frac=sig.tp_frac,
                spread_bps=si.spread_bps_at(int(row["decision_ts"])),
                slippage_est=0.0005,
                latency_ms=5.0,
                # Research/shadow context flags (mirrors the synthetic paper session); live
                # eligibility is judged by the gates, not by a paper run.
                data_fresh=True,
                metadata_verified=True,
                symbol_tradable=True,
                strategy_enabled=True,
                config_live_approved=promoted,
                decision_ts=int(row["decision_ts"]),
            )
            out.append(
                PaperCandidateInput(
                    candidate=cand,
                    equity=equity,
                    exit_move_frac=exit_move_frac,
                    hold_bars=min(hold_bars, exit_bar - entry_bar),
                )
            )
    return out, strat_id, strat_ver


def run_lake_paper_session(
    data_cfg: DataConfig | None = None,
    *,
    settings: Settings | None = None,
    timeframe: str | None = None,
    symbols: list[str] | None = None,
    candidate_id: str | None = None,
    dataset_version: str | None = None,
    session_name: str | None = None,
) -> tuple[PaperSession, PaperReport, str]:
    """Run + persist a real-data paper session over a snapshot. Returns (session, report, id).

    Requires the snapshot to be downloaded first (``qbot download --config ...``). The
    session id encodes the snapshot + timeframe so real-data runs are identifiable on the
    Paper dashboard; trades persist to ``paper_trades`` (shadow-only)."""
    settings = settings or get_settings()
    data_cfg = data_cfg or load_data_config()
    tf = timeframe or data_cfg.base_timeframe
    syms = symbols or data_cfg.active_symbols()
    dsv = dataset_version or data_cfg.data_version
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
