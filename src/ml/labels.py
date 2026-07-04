"""Meta-label generation for the ML shadow layer (AGENTS.md Section 20).

Meta-labeling: given a deterministic candidate, generate a binary label
indicating whether to take (1) or skip (0) the trade, based on the eventual
outcome.  Labels are generated from paper trade outcomes — profitable trades
within the hold window get label=1; losses get label=0.

Two label sources (audit H18):
  * :func:`build_labels_from_paper_outcomes` — the REAL pipeline: joins the
    decision-time feature vectors persisted in ``decision_logs`` to realized
    ``paper_trades`` outcomes per paper-money session. This is the production
    training source once paper trades with persisted features accumulate.
  * :func:`synthetic_labels` / :func:`build_reference_dataset` — a deterministic
    reference dataset from candidate features, for PLUMBING self-checks only
    (it is constructed so a correct meta-labeler wins, so it is never evidence
    of edge). :func:`build_training_dataset` prefers real and falls back to it.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, replace

import structlog

from src.ranking.candidate import Candidate

_log = structlog.get_logger("ml.labels")


@dataclass(slots=True)
class LabeledSample:
    """One training sample for the meta-labeler."""

    candidate: Candidate
    label: int  # 1=take, 0=skip
    realized_pnl: float  # normalized R (e.g., +1.0R = full stop-distance gain)
    hold_bars: int


def label_from_outcome(realized_pnl: float, *, threshold: float = 0.0) -> int:
    """Convert realized PnL (R) to take/skip label."""
    return 1 if realized_pnl > threshold else 0


def synthetic_labels(
    candidates: list[Candidate],
    *,
    seed: int = 42,
    good_threshold_strength: float = 0.75,
    good_threshold_edge: float = 0.009,
) -> list[LabeledSample]:
    """Generate deterministic synthetic labels for a set of candidates.

    Label = 1 (take) when the candidate has both high signal strength AND
    sufficient expected edge.  PnL is drawn from a distribution parameterised
    by quality so that good candidates produce a positive expectancy and bad
    ones produce a negative expectancy — giving the meta-labeler a learnable
    signal.

    Uses a deterministic hash-based RNG seeded from the candidate so that
    labels are reproducible given the same inputs (Parity Rule, Section 10).
    """
    rng = random.Random(seed)
    samples: list[LabeledSample] = []
    for cand in candidates:
        good = (
            cand.signal_strength >= good_threshold_strength
            and cand.expected_edge_frac >= good_threshold_edge
        )
        # Add a bit of noise per-candidate via a seeded hash so each sample
        # is slightly different while keeping the dataset deterministic.
        h = int(hashlib.md5(f"{cand.symbol}:{cand.decision_ts}".encode()).hexdigest(), 16)
        local_rng = random.Random(seed + (h % 10_000))

        pnl = (0.8 + local_rng.gauss(0.0, 0.4)) if good else (-0.5 + local_rng.gauss(0.0, 0.35))

        label = label_from_outcome(pnl)
        hold = rng.randint(1, 5)
        samples.append(LabeledSample(candidate=cand, label=label, realized_pnl=pnl, hold_bars=hold))
    return samples


def build_reference_dataset(
    n_good: int = 40,
    n_bad: int = 30,
    n_neutral: int = 30,
    *,
    seed: int = 42,
) -> list[LabeledSample]:
    """Build a synthetic labeled dataset for gate checks.

    The dataset has three classes:
    * **good** (n_good): high signal_strength + high expected_edge → label=1
    * **bad** (n_bad): low signal_strength + low expected_edge → label=0
    * **neutral** (n_neutral): mid-range features → label=0

    Good candidates have positive PnL; bad/neutral have negative PnL.  This
    ensures the meta-labeler, if trained, will achieve higher expectancy than
    the always-take baseline.
    """
    rng = random.Random(seed)
    _SYMBOLS = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]
    _STRATEGIES = ["basis_reversion_v1", "lead_lag_v1", "cross_strength_v1"]
    _REGIMES = ["low_vol_up", "low_vol_down", "trend_up"]

    def _make(
        i: int,
        signal_strength: float,
        expected_edge_frac: float,
        spread_bps: float,
        slippage_est: float,
        pnl: float,
        label: int,
    ) -> LabeledSample:
        sym = _SYMBOLS[i % len(_SYMBOLS)]
        strat = _STRATEGIES[i % len(_STRATEGIES)]
        regime = _REGIMES[i % len(_REGIMES)]
        price = {"BTC/USDT:USDT": 50_000.0, "ETH/USDT:USDT": 3_000.0, "SOL/USDT:USDT": 150.0}[sym]
        cand = Candidate(
            symbol=sym,
            strategy=strat,
            strategy_version="v1.0.0",
            side=1 if i % 2 == 0 else -1,
            entry_price=price,
            stop_frac=0.008,
            tp_frac=0.02,
            regime=regime,
            session=1,
            features={
                "atr_pct": round(0.003 + rng.uniform(-0.001, 0.001), 6),
                "premium": round(rng.uniform(-0.002, 0.002), 6),
                "funding_z": round(rng.uniform(-1.5, 1.5), 4),
                "rv_short": round(rng.uniform(0.001, 0.005), 6),
                "ret_1": round(rng.uniform(-0.01, 0.01), 6),
            },
            signal_strength=round(signal_strength + rng.gauss(0, 0.02), 4),
            confirmation=round(signal_strength * 0.9 + rng.gauss(0, 0.02), 4),
            expected_edge_frac=round(expected_edge_frac + rng.gauss(0, 0.001), 6),
            spread_bps=round(spread_bps + rng.gauss(0, 0.3), 2),
            slippage_est=round(slippage_est + rng.gauss(0, 0.0001), 6),
            latency_ms=5.0,
            data_fresh=True,
            metadata_verified=True,
            symbol_tradable=True,
            strategy_enabled=True,
            config_live_approved=True,
            decision_ts=1_700_000_000_000 + i * 60_000,
        )
        pnl_r = pnl + rng.gauss(0, 0.15)
        return LabeledSample(
            candidate=cand,
            label=label_from_outcome(pnl_r),
            realized_pnl=pnl_r,
            hold_bars=rng.randint(1, 5),
        )

    samples: list[LabeledSample] = []
    for i in range(n_good):
        samples.append(_make(i, 0.88, 0.013, 2.0, 0.0003, 0.9, 1))
    for i in range(n_bad):
        j = n_good + i
        samples.append(_make(j, 0.45, 0.004, 7.5, 0.0009, -0.65, 0))
    for i in range(n_neutral):
        k = n_good + n_bad + i
        samples.append(_make(k, 0.65, 0.008, 4.0, 0.0006, -0.2, 0))

    # Shuffle deterministically so train/test split doesn't track good→bad order,
    # then reassign decision_ts in shuffled order: train_test_split sorts
    # chronologically (audit L33), so timestamps must follow the shuffled order
    # or sorting would restore the good→bad class ordering.
    rng.shuffle(samples)
    for idx, s in enumerate(samples):
        s.candidate = replace(s.candidate, decision_ts=1_700_000_000_000 + idx * 60_000)
    return samples


# Sessions with these prefixes are NOT paper-money runs (demo/testnet/live are real-venue; selftest
# is a gate self-check) — they are excluded from the real training set, matching the PAPER-B scope.
_NON_PAPER_SESSION_PREFIXES: tuple[str, ...] = ("demo:", "testnet:", "live:", "selftest:")

# ML feature-row keys (kept local to avoid importing the features module at label-build time).
_CANDIDATE_FEATURE_KEYS = ("signal_strength", "expected_edge_frac", "spread_bps", "slippage_est")
_PIPELINE_FEATURE_KEYS = ("atr_pct", "premium", "funding_z", "rv_short", "ret_1")


def build_labels_from_paper_outcomes(*, lookback_days: int = 120) -> list[LabeledSample]:
    """REAL ML training labels from persisted paper trades (AGENTS.md Section 20; audit H18).

    Joins the decision-time feature vectors persisted in ``decision_logs`` (action=execute, with
    non-empty ``features``) to realized outcomes in ``paper_trades``, per paper-money session. As a
    paper session holds at most one position per symbol at a time, executed decisions and closed
    trades for a given ``(session, symbol, strategy, side)`` occur in lockstep, so they are paired
    positionally in chronological order (robust to ``paper_trades`` carrying only a persist-time
    ``created_at``, not the entry ts). Each pair becomes a :class:`LabeledSample` whose label is the
    realized sign of ``pnl_r``.

    Returns ``[]`` when no real paper data with persisted features exists yet — the caller then
    falls back to the synthetic reference dataset (PLUMBING only, never evidence of edge)."""
    from collections import defaultdict
    from datetime import UTC, datetime, timedelta

    from src.db.base import session_scope
    from src.db.models import DecisionLog, PaperTradeRecord

    cutoff = datetime.now(UTC) - timedelta(days=lookback_days)

    def _is_paper(session_id: str | None) -> bool:
        # POSITIVE rule, not a denylist (L-J): a paper-money session is either a bare id with no
        # scheme prefix (the default paper session — a uuid, `paper/engine.py`) or the explicit
        # `paper:` scheme (the basket paper loop). Any OTHER scheme (`demo:`/`testnet:`/`live:`/
        # `selftest:`, or a future `backtest:`/`replay:`) is NOT paper — so a new subsystem writing
        # decision_logs/paper_trades under its own scheme is never silently swept into the training
        # set. Kept consistent with the PAPER-B gate's non-paper prefixes.
        sid = session_id or ""
        if not sid or sid.startswith(_NON_PAPER_SESSION_PREFIXES):
            return False
        return ":" not in sid or sid.startswith("paper:")

    with session_scope() as session:
        decisions = (
            session.query(DecisionLog)
            .filter(DecisionLog.action == "execute", DecisionLog.ts >= cutoff)
            .order_by(DecisionLog.ts)
            .all()
        )
        dkey_rows: dict[tuple, list] = defaultdict(list)
        for d in decisions:
            feats = dict(d.features or {})
            if not feats or not _is_paper(d.session_id):
                continue  # no persisted feature vector (old rows / H18 not yet live), or not paper
            dkey_rows[(d.session_id, d.symbol, d.strategy, int(d.side))].append(
                {
                    "symbol": d.symbol,
                    "strategy": d.strategy,
                    "strategy_version": d.strategy_version or "",
                    "side": int(d.side),
                    # regime is filled from the paired TRADE row (DecisionLog has no regime column).
                    "features": feats,
                    "ts": d.ts,
                }
            )

        trades = (
            session.query(
                PaperTradeRecord.session_id,
                PaperTradeRecord.symbol,
                PaperTradeRecord.strategy,
                PaperTradeRecord.side,
                PaperTradeRecord.pnl_r,
                PaperTradeRecord.regime,
                PaperTradeRecord.created_at,
            )
            .filter(PaperTradeRecord.created_at >= cutoff)
            .order_by(PaperTradeRecord.created_at, PaperTradeRecord.id)
            .all()
        )
        # Store (realized R, realized regime): the regime comes from the TRADE row — DecisionLog has
        # no regime column, so without this every real sample's regime is "" and the regime
        # classifier collapses to a single class and never trains on real data.
        tkey_out: dict[tuple, list[tuple[float, str]]] = defaultdict(list)
        for sid, sym, strat, side, pnl_r, regime, _created in trades:
            if _is_paper(sid):
                tkey_out[(sid, sym, strat, int(side))].append((float(pnl_r), regime or ""))

    samples: list[LabeledSample] = []
    skipped_keys = 0
    for key, drows in dkey_rows.items():
        outcomes = tkey_out.get(key, [])
        # Positional lockstep is only sound when the executed decisions and closed trades for this
        # (session,symbol,strategy,side) are 1:1 — true when every execute-decision produced exactly
        # one trade in the same order (a paper session holds ≤1 position per symbol). The COMMON
        # end-state has one MORE decision than trade — the most-recent position is still open — so
        # tolerate exactly one trailing extra decision (pair the first ``len(outcomes)`` in order,
        # drop the open one). Any LARGER/other mismatch (a post-execute reject, a non-FIFO close)
        # can't be aligned safely, so skip the whole key rather than mislabel — paper_trades carries
        # no decision link to join on precisely.
        if len(drows) - len(outcomes) not in (0, 1):
            skipped_keys += 1
            continue
        for drow, (pnl_r, regime) in zip(drows, outcomes, strict=False):
            samples.append(_labeled_sample_from_real({**drow, "regime": regime}, pnl_r))
    if skipped_keys:
        _log.warning(
            "ml_real_labels_key_count_mismatch",
            skipped_keys=skipped_keys,
            note="decision/trade counts disagreed for these keys; skipped to avoid mislabeling",
        )
    # Chronological order for the downstream chronological train/test split (no temporal leakage).
    samples.sort(key=lambda s: s.candidate.decision_ts)
    return samples


def _labeled_sample_from_real(drow: dict, pnl_r: float) -> LabeledSample:
    """Reconstruct a training :class:`LabeledSample` from a persisted decision row + realized R."""
    from datetime import datetime

    feats = drow["features"]
    ts = drow["ts"]
    decision_ms = int(ts.timestamp() * 1000) if isinstance(ts, datetime) else 0
    cand = Candidate(
        symbol=drow["symbol"],
        strategy=drow["strategy"],
        strategy_version=drow["strategy_version"],
        side=drow["side"],
        entry_price=0.0,
        # Reconstruct the persisted stop_frac (H-C): the exec_quality target expresses execution
        # cost in R via cost/stop_frac; with stop_frac=0 it collapsed to the take/skip label.
        stop_frac=float(feats.get("stop_frac", 0.0)),
        tp_frac=0.0,
        regime=drow.get("regime") or "",
        session=0,
        signal_strength=float(feats.get("signal_strength", 0.0)),
        expected_edge_frac=float(feats.get("expected_edge_frac", 0.0)),
        spread_bps=float(feats.get("spread_bps", 0.0)),
        slippage_est=float(feats.get("slippage_est", 0.0)),
        features={k: float(feats.get(k, 0.0)) for k in _PIPELINE_FEATURE_KEYS},
        decision_ts=decision_ms,
    )
    return LabeledSample(
        candidate=cand,
        label=label_from_outcome(pnl_r),
        realized_pnl=float(pnl_r),
        hold_bars=0,
    )


def build_training_dataset(
    *, prefer_real: bool = True, min_real: int = 50, seed: int = 42
) -> tuple[list[LabeledSample], str]:
    """Return ``(samples, source)`` for ML training/eval (audit H18).

    Prefers REAL labels from paper outcomes when at least ``min_real`` exist; otherwise falls back
    to the synthetic reference dataset (``source="synthetic"``) so the pipeline still runs for
    plumbing while real data accumulates. ``source`` is one of ``"real"`` / ``"synthetic"`` so the
    caller can tag/log which was used and never present synthetic numbers as evidence of edge."""
    if prefer_real:
        try:
            real = build_labels_from_paper_outcomes()
        except Exception:  # noqa: BLE001 - DB unavailable / schema drift → fall back to synthetic
            real = []
        if len(real) >= min_real:
            return real, "real"
    return build_reference_dataset(seed=seed), "synthetic"


def train_test_split(
    samples: list[LabeledSample], test_fraction: float = 0.25, seed: int = 42
) -> tuple[list[LabeledSample], list[LabeledSample]]:
    """CHRONOLOGICAL split: train on the earlier ``1-test_fraction``, test on the most-recent
    ``test_fraction``. NEVER a random shuffle — a random split leaks future trade outcomes into
    training (temporal leakage that inflates the meta-labeler's apparent skill once real paper
    trades feed it). Samples are explicitly sorted by ``candidate.decision_ts`` first (audit
    L33): an unsorted real-outcome batch would otherwise reintroduce the leakage the front/back
    split exists to prevent. The synthetic reference dataset assigns timestamps in its shuffled
    order, so its class balance survives the sort. ``seed`` is accepted for call-site
    compatibility but unused (the split is deterministic)."""
    _ = seed
    ordered = sorted(samples, key=lambda s: s.candidate.decision_ts)
    split = max(1, int(len(ordered) * (1.0 - test_fraction)))
    return ordered[:split], ordered[split:]


def count_positives(samples: list[LabeledSample]) -> int:
    return sum(s.label for s in samples)


def baseline_expectancy(samples: list[LabeledSample]) -> float:
    """Always-take baseline: mean PnL over all samples."""
    if not samples:
        return 0.0
    return sum(s.realized_pnl for s in samples) / len(samples)


def filtered_expectancy(samples: list[LabeledSample], predictions: list[int]) -> float:
    """Expectancy over samples where model predicted take (prediction=1)."""
    taken = [s.realized_pnl for s, p in zip(samples, predictions, strict=False) if p == 1]
    if not taken:
        return 0.0
    return sum(taken) / len(taken)


def profit_factor(samples: list[LabeledSample], predictions: list[int] | None = None) -> float:
    """Gross profit / gross loss for a set of trades.

    If *predictions* is given, only samples where ``prediction == 1`` are counted.
    """
    pnls = [s.realized_pnl for s in samples]
    if predictions is not None:
        pnls = [s.realized_pnl for s, pr in zip(samples, predictions, strict=False) if pr == 1]
    gains = sum(p for p in pnls if p > 0)
    losses = abs(sum(p for p in pnls if p < 0))
    if losses == 0:
        return math.inf if gains > 0 else 1.0
    return gains / losses


def worst_trade(samples: list[LabeledSample], predictions: list[int] | None = None) -> float:
    """Worst (most negative) PnL; 0.0 if no trades taken."""
    pnls = [s.realized_pnl for s in samples]
    if predictions is not None:
        pnls = [s.realized_pnl for s, pr in zip(samples, predictions, strict=False) if pr == 1]
    return min(pnls) if pnls else 0.0


def best_n_trades(samples: list[LabeledSample], n: int) -> list[LabeledSample]:
    """Return the top-N samples by realized_pnl (descending)."""
    return sorted(samples, key=lambda s: s.realized_pnl, reverse=True)[:n]


def worst_n_trades(samples: list[LabeledSample], n: int) -> list[LabeledSample]:
    """Return the bottom-N samples by realized_pnl (ascending)."""
    return sorted(samples, key=lambda s: s.realized_pnl)[:n]
