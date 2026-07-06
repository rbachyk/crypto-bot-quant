"""Why does lead_lag's edge die in the recent regime? Decompose the MARKET STATE (not the trades)
across the profitable era (folds 0-3) vs the loss era (fold 4 + hold-out), to tell apart the three
mechanisms that kill a short-only cross-asset momentum edge:

  1. Directional headwind -> follower DRIFT flips up (short-only fights a rising tape).
  2. Lag compression      -> realized LAGGED BETA (follower_ret[t] ~ leader_ret[t-1]) decays to ~0.
                             The information lag closed; the edge is GONE. Shelve is correct.
  3. Impulse -> chop       -> leader DIRECTIONAL EFFICIENCY drops (moves stop continuing). ROTATES.

The decisive column is the SHORT EDGE PROXY: E[ -follower forward return | short trigger ] — the raw
per-signal payoff of the strategy's actual entry, measured per era. If it is positive in era A and
negative in era B, the other columns explain WHY.

Run in-container:  docker compose exec worker-backtest python scripts/leadlag_market_state.py
"""
from __future__ import annotations

import math
import os
import statistics
from datetime import UTC, datetime
from pathlib import Path

from src.data.schema import OHLCV, SeriesKey
from src.data.store import SeriesStore
from src.strategies.config import load_strategies_config

TF = "4h"
SHORT = 12  # rv_short window (configs/features.yaml windows.short)
ABS_FLOOR = 0.0025
VOL_MULT = 2.0
HORIZONS = (6, 12)  # forward-return capture windows in bars (1d, 2d on 4h)

# Fold / era boundaries from the 4h validation report (UTC). Era A = folds 0-3 (edge present),
# Era B = fold 4 + hold-out (edge gone). Kept explicit so the split matches what you saw.
FOLD_EDGES_ISO = [
    "2025-07-07T06:00", "2025-09-03T12:24", "2025-10-31T18:48", "2025-12-29T01:12",
    "2026-02-25T07:36", "2026-04-24T14:00", "2026-07-06T10:00",
]


def _ms(iso: str) -> int:
    dt = datetime.fromisoformat(iso).replace(tzinfo=UTC)
    return int(dt.timestamp() * 1000)


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=UTC).strftime("%Y-%m-%d")


def _store() -> SeriesStore:
    # SeriesStore appends "/series" itself, so pass the datalake ROOT (the dir CONTAINING series/).
    for root in (os.environ.get("DATA_LAKE_PATH"), "var/datalake", "/app/var/datalake"):
        if root and (Path(root) / "series").exists():
            return SeriesStore(Path(root))
    raise SystemExit("no datalake found (set DATA_LAKE_PATH)")


def _closes(store: SeriesStore, sym: str, lo: int, hi: int) -> dict[int, float]:
    rows = store.read(SeriesKey("bybit", OHLCV, sym, TF), lo, hi)
    return {int(r["ts"]): float(r["close"]) for r in rows}


def _slope_corr(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """OLS slope of ys~xs and Pearson corr. Returns (0,0) on degenerate input."""
    n = len(xs)
    if n < 3:
        return 0.0, 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    if sxx <= 0 or syy <= 0:
        return 0.0, 0.0
    return sxy / sxx, sxy / math.sqrt(sxx * syy)


def _dir_efficiency(rets: list[float]) -> float:
    """|net move| / sum|bar moves|: ~1 = clean trend, ~0 = chop (mean-reverting)."""
    denom = sum(abs(r) for r in rets)
    return abs(sum(rets)) / denom if denom > 0 else 0.0


def _mean(xs: list[float]) -> float:
    return statistics.mean(xs) if xs else 0.0


def _pstdev(xs: list[float]) -> float:
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0


def main() -> None:
    store = _store()
    sc = load_strategies_config()
    cand = sc.candidate("lead_lag_xasset")
    leader = str(cand.fixture.values["leader"])
    followers = list(cand.fixture.values["followers"])
    lo, hi = _ms(FOLD_EDGES_ISO[0]), _ms(FOLD_EDGES_ISO[-1])

    lc = _closes(store, leader, lo, hi)
    fcloses = {f: _closes(store, f, lo, hi) for f in followers}
    ts = sorted(lc)
    if len(ts) < SHORT + max(HORIZONS) + 5:
        raise SystemExit(f"insufficient {TF} leader data: {len(ts)} bars (need the full window)")

    # Leader per-bar returns + rv_short + gate, indexed by bar position.
    lret = {ts[i]: (lc[ts[i]] / lc[ts[i - 1]] - 1.0) for i in range(1, len(ts))}
    llog = {ts[i]: math.log(lc[ts[i]] / lc[ts[i - 1]]) for i in range(1, len(ts))}

    def era(t: int) -> str:
        a, b = _ms(FOLD_EDGES_ISO[0]), _ms(FOLD_EDGES_ISO[4])
        c = _ms(FOLD_EDGES_ISO[6])
        if a <= t < b:
            return "A edge-present (folds 0-3)"
        if b <= t < c:
            return "B edge-gone (fold4+holdout)"
        return "?"

    # Accumulators per era.
    buckets: dict[str, dict] = {}
    for name in ("A edge-present (folds 0-3)", "B edge-gone (fold4+holdout)"):
        buckets[name] = {
            "lrets": [],
            "lag_pairs": {f: [] for f in followers},  # (leader_ret[t], follower_ret[t+1]) pairs
            "fdrift": {f: [] for f in followers},
            "short_fwd": {h: [] for h in HORIZONS},  # follower fwd return AFTER a short trigger
            "n_short": 0,
        }

    for i in range(SHORT, len(ts) - max(HORIZONS) - 1):
        t = ts[i]
        e = era(t)
        if e not in buckets:
            continue
        bk = buckets[e]
        bk["lrets"].append(lret[t])
        # rv_short of leader log returns over trailing SHORT window; gate as the strategy uses it.
        window = [llog[ts[j]] for j in range(i - SHORT + 1, i + 1) if ts[j] in llog]
        rv = statistics.pstdev(window) if len(window) > 1 else 0.0
        gate = max(ABS_FLOOR, VOL_MULT * rv)
        # Lagged beta samples: follower bar-(i+1) return vs leader bar-i return.
        for f in followers:
            fc = fcloses[f]
            if ts[i] in fc and ts[i + 1] in fc:
                fret_next = fc[ts[i + 1]] / fc[ts[i]] - 1.0
                bk["fdrift"][f].append(fret_next)
                bk["lag_pairs"][f].append((lret[t], fret_next))
        # SHORT trigger: significant leader DOWN move -> short the follower next bar.
        if lret[t] < -gate:
            bk["n_short"] += 1
            for f in followers:
                fc = fcloses[f]
                for h in HORIZONS:
                    if ts[i + 1] in fc and ts[i + 1 + h] in fc:
                        fwd = fc[ts[i + 1 + h]] / fc[ts[i + 1]] - 1.0
                        bk["short_fwd"][h].append(-fwd)  # short payoff = -(forward return)

    # ---- report ----
    print(f"\nlead_lag MARKET-STATE decomposition  ({TF}, leader={leader}, followers={followers})")
    print(f"window {_iso(lo)} -> {_iso(hi)}   bars={len(ts)}\n")
    print(f"{'metric':42}  {'A edge-present':>18}  {'B edge-gone':>18}")
    print("-" * 84)

    def row(label, a, b, fmt="{:+.4f}"):
        print(f"{label:42}  {fmt.format(a):>18}  {fmt.format(b):>18}")

    A, B = buckets["A edge-present (folds 0-3)"], buckets["B edge-gone (fold4+holdout)"]

    # Leader trend / vol / chop
    row("leader NET return (era)", sum(A["lrets"]), sum(B["lrets"]))
    row("leader mean bar return", _mean(A["lrets"]), _mean(B["lrets"]))
    row("leader bar vol (stdev)", _pstdev(A["lrets"]), _pstdev(B["lrets"]))
    de_a, de_b = _dir_efficiency(A["lrets"]), _dir_efficiency(B["lrets"])
    row("leader DIRECTIONAL EFFICIENCY", de_a, de_b, "{:.3f}")
    print()
    # Follower drift (short-only headwind)
    for f in followers:
        row(f"follower DRIFT net  {f.split('/')[0]}", sum(A["fdrift"][f]), sum(B["fdrift"][f]))
    print()
    # Lagged beta / corr (signal decay)
    for f in followers:
        ax = [p[0] for p in A["lag_pairs"][f]]
        ay = [p[1] for p in A["lag_pairs"][f]]
        bx = [p[0] for p in B["lag_pairs"][f]]
        by = [p[1] for p in B["lag_pairs"][f]]
        sa, ca = _slope_corr(ax, ay)
        sb, cb = _slope_corr(bx, by)
        row(f"LAGGED BETA  {f.split('/')[0]} (foll~lead[-1])", sa, sb, "{:+.3f}")
        row(f"  lagged corr {f.split('/')[0]}", ca, cb, "{:+.3f}")
    print()
    # The decisive column: raw short-signal payoff per era.
    print(f"{'short triggers (n)':42}  {A['n_short']:>18}  {B['n_short']:>18}")
    for h in HORIZONS:
        a = statistics.mean(A["short_fwd"][h]) if A["short_fwd"][h] else 0.0
        b = statistics.mean(B["short_fwd"][h]) if B["short_fwd"][h] else 0.0
        row(f"SHORT EDGE PROXY  E[payoff] @{h}bar", a, b)
    print("\nRead: B leader NET up + follower DRIFT up  => #1 directional headwind (rotates back).")
    print("      LAGGED BETA/corr collapses to ~0 in B  => #2 lag compression (edge gone; shelve).")
    print("      DIRECTIONAL EFFICIENCY drops in B       => #3 impulse->chop (rotates back).")
    print("      SHORT EDGE PROXY positive in A, negative in B => loss is in the signal.")


if __name__ == "__main__":
    main()
