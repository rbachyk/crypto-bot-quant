"""Meta-label generation for the ML shadow layer (AGENTS.md Section 20).

Meta-labeling: given a deterministic candidate, generate a binary label
indicating whether to take (1) or skip (0) the trade, based on the eventual
outcome.  Labels are generated from paper trade outcomes — profitable trades
within the hold window get label=1; losses get label=0.

For the Phase 9 gate check (no real paper trade history yet) we expose a
:func:`synthetic_labels` function that generates a deterministic reference
dataset from candidate features, enabling gate checks to verify the full
pipeline works end-to-end before real observations accumulate.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass

from src.ranking.candidate import Candidate


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

    # Shuffle deterministically so train/test split doesn't track good→bad order.
    rng.shuffle(samples)
    return samples


def train_test_split(
    samples: list[LabeledSample], test_fraction: float = 0.25, seed: int = 42
) -> tuple[list[LabeledSample], list[LabeledSample]]:
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    split = max(1, int(len(shuffled) * (1.0 - test_fraction)))
    return shuffled[:split], shuffled[split:]


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
