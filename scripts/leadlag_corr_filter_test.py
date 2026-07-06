"""Would a CORRELATION FILTER rescue lead_lag? Test the one claim the idea rests on: does the
TRAILING lead-lag correlation (known at decision time) PREDICT the forward short-edge payoff — and
does it still predict in the decayed era B?

For each short trigger we compute the trailing lead-lag corr over the prior W_CORR bars — corr of
(leader_ret[j], follower_ret[j+1]), using ONLY past data — then bucket the trigger's forward payoff
(-follower fwd return, the short's P&L proxy) by that trailing corr. If high-trailing-corr triggers
pay forward AND still pay in era B, a corr gate has real predictive value. If era B loses even in
the top corr bucket, the gate is only a lagging kill-switch (turns off after the edge already died,
eating the transition) — it cannot revive the edge. That distinction decides whether it is worth
building + re-validating a filter, or whether it is the same overfit that broke the hold-out.

Run: docker compose exec worker-backtest python scripts/leadlag_corr_filter_test.py
"""
from __future__ import annotations

import math
import os
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.data.schema import OHLCV, SeriesKey  # noqa: E402
from src.data.store import SeriesStore  # noqa: E402
from src.strategies.config import load_strategies_config  # noqa: E402

TF = "4h"
SHORT = 12  # rv_short window (configs/features.yaml windows.short)
ABS_FLOOR = 0.0025
VOL_MULT = 2.0
HORIZON = 6  # forward capture (bars); the era diagnostic showed the edge is ~1-day/6-bar
W_CORR = 60  # trailing window for the lead-lag correlation the filter would read (~10d on 4h)
ERA_BOUNDARY = "2026-02-25T07:36"  # fold3->fold4: edge-present (A) vs edge-gone (B)
WINDOW = ("2025-07-07T06:00", "2026-07-06T10:00")


def _ms(iso: str) -> int:
    return int(datetime.fromisoformat(iso).replace(tzinfo=UTC).timestamp() * 1000)


def _store() -> SeriesStore:
    root = os.environ.get("DATA_LAKE_PATH") or str(_REPO_ROOT / "var" / "datalake")
    return SeriesStore(Path(root))


def _closes(store: SeriesStore, sym: str, lo: int, hi: int) -> dict[int, float]:
    rows = store.read(SeriesKey("bybit", OHLCV, sym, TF), lo, hi)
    return {int(r["ts"]): float(r["close"]) for r in rows}


def _corr(pairs: list[tuple[float, float]]) -> float:
    n = len(pairs)
    if n < 8:
        return float("nan")
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return float("nan")
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    return sxy / math.sqrt(sxx * syy)


def _summ(vals: list[float]) -> str:
    if not vals:
        return f"{'n=0':>22}"
    m = statistics.mean(vals)
    wr = sum(1 for v in vals if v > 0) / len(vals)
    return f"n={len(vals):4}  mean={m:+.4f}  win={wr:.2f}"


def main() -> None:
    from src.data.config import load_data_config

    store = _store()
    sc = load_strategies_config()
    cand = sc.candidate("lead_lag_xasset")
    leader = str(cand.fixture.values["leader"])
    # The REAL lake validation trades EVERY non-leader universe symbol as a follower (the fixture's
    # `followers` list is only for the synthetic 5m fixture), so pool all of them, not just 2.
    # Load the SAME config the validation uses (configs/data.bybit.yaml, 20 symbols) — NOT the
    # default configs/data.yaml (the 3-symbol skeleton), or this would silently see only ETH/SOL.
    universe = load_data_config("configs/data.bybit.yaml").active_symbols()
    followers = [s for s in universe if s != leader]
    lo, hi = _ms(WINDOW[0]), _ms(WINDOW[1])
    boundary = _ms(ERA_BOUNDARY)

    lc = _closes(store, leader, lo, hi)
    fcloses = {f: _closes(store, f, lo, hi) for f in followers}
    ts = sorted(lc)
    if len(ts) < W_CORR + SHORT + HORIZON + 5:
        raise SystemExit(f"insufficient {TF} data: {len(ts)} bars")

    lret = {ts[i]: lc[ts[i]] / lc[ts[i - 1]] - 1.0 for i in range(1, len(ts))}
    llog = {ts[i]: math.log(lc[ts[i]] / lc[ts[i - 1]]) for i in range(1, len(ts))}
    def _rets(fc: dict[int, float]) -> dict[int, float]:
        return {
            ts[i]: fc[ts[i]] / fc[ts[i - 1]] - 1.0
            for i in range(1, len(ts))
            if ts[i - 1] in fc and ts[i] in fc
        }

    fret = {f: _rets(fc) for f, fc in fcloses.items()}

    # Each observation: a short trigger on one follower, tagged with era + trailing lead-lag corr.
    obs: list[tuple[str, float, float]] = []  # (era, trailing_corr, fwd_payoff)
    start = max(W_CORR + 1, SHORT)
    for i in range(start, len(ts) - HORIZON - 1):
        t = ts[i]
        if t not in lret:
            continue
        window = [llog[ts[j]] for j in range(i - SHORT + 1, i + 1) if ts[j] in llog]
        rv = statistics.pstdev(window) if len(window) > 1 else 0.0
        gate = max(ABS_FLOOR, VOL_MULT * rv)
        if lret[t] >= -gate:  # only significant leader DOWN moves fire the short
            continue
        era = "A" if t < boundary else "B"
        for f in followers:
            fc = fcloses[f]
            fr = fret[f]
            # trailing lead-lag corr over [i-W_CORR, i-1], using ONLY past bars.
            pairs = [
                (lret[ts[j]], fr[ts[j + 1]])
                for j in range(i - W_CORR, i)
                if ts[j] in lret and ts[j + 1] in fr
            ]
            c = _corr(pairs)
            if math.isnan(c):
                continue
            if ts[i + 1] in fc and ts[i + 1 + HORIZON] in fc:
                fwd = fc[ts[i + 1 + HORIZON]] / fc[ts[i + 1]] - 1.0
                obs.append((era, c, -fwd))  # short payoff = -(forward return)

    if not obs:
        raise SystemExit("no short triggers with a valid trailing correlation in the window")

    # Global terciles of trailing corr (buckets are defined on the WHOLE sample, then read per era).
    corrs = sorted(o[1] for o in obs)
    q1, q2 = corrs[len(corrs) // 3], corrs[2 * len(corrs) // 3]

    def bucket(c: float) -> str:
        return "low" if c < q1 else ("mid" if c < q2 else "high")

    print(f"\nlead_lag CORR-FILTER predictive test  ({TF}, followers={followers})")
    print(f"trailing corr window={W_CORR} bars, forward payoff @{HORIZON} bars, terciles cut at "
          f"{q1:+.3f} / {q2:+.3f}\n")
    print("Does HIGH trailing lead-lag corr predict a positive forward short payoff — in each era?")
    print(f"{'trailing-corr bucket':22}  {'ERA A (edge present)':>30}  {'ERA B (edge gone)':>30}")
    print("-" * 88)
    for b in ("low", "mid", "high"):
        a = [o[2] for o in obs if o[0] == "A" and bucket(o[1]) == b]
        bb = [o[2] for o in obs if o[0] == "B" and bucket(o[1]) == b]
        print(f"{b:22}  {_summ(a):>30}  {_summ(bb):>30}")

    # The filter, made concrete: only trade triggers whose trailing corr clears a threshold.
    print("\nHypothetical FILTER: only fire when trailing corr > c   (payoff per era)")
    for label, c in (("c = median(A)", statistics.median([o[1] for o in obs if o[0] == "A"])),
                     ("c = 0.05", 0.05), ("c = 0.10", 0.10)):
        a = [o[2] for o in obs if o[0] == "A" and o[1] > c]
        bb = [o[2] for o in obs if o[0] == "B" and o[1] > c]
        print(f"  {label:14} (c={c:+.3f})   A: {_summ(a):>28}   B: {_summ(bb):>28}")

    print("\nRead:")
    print("  HIGH bucket positive in A AND B  -> corr predicts forward edge; a filter has legs.")
    print("  HIGH bucket (and filtered B) still NEGATIVE -> corr does NOT predict the recent edge;")
    print("     the gate is only a lagging kill-switch, cannot revive it (same overfit risk).")


if __name__ == "__main__":
    main()
