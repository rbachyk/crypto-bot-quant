"""Operator aid for the Section-6 metadata review (the META / demo-readiness blocker).

`configs/metadata.<exchange>.yaml` ships `verified: false`, so the venue refuses every order and
demo-readiness reports BLOCKED until an operator confirms each symbol's contract spec against the
exchange's reference and flips the flag. This script turns that review from manual cross-checking
into reading a diff: it fetches the exchange's LIVE public spec (ccxt, no keys, no orders) and
compares it field-by-field against the config.

It does NOT flip `verified` — that is the operator's sign-off and the one thing the code refuses to
launder (src/exchange/metadata.py). It only surfaces the evidence the sign-off rests on.

CAVEAT ON FEES: exchanges publish DEFAULT (VIP0) maker/taker rates; your account's actual rates
depend on its fee tier and any promotions. ccxt reports the published default. Put YOUR account's
real taker fee in the config — the basket pays taker on every leg it can't fill passively, so this
value is the cost model the whole demo run exists to test. Confirm it in the Bybit fee-tier page,
not just here.

Run: docker compose exec worker-live python scripts/verify_metadata.py
     docker compose exec worker-live python scripts/verify_metadata.py --exchange bybit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.exchange.metadata import load_metadata_for  # noqa: E402

# Fields compared, and how close counts as a match. Fees are floats with representation wobble;
# tick/step must be exact. None on the live side means ccxt did not expose it (Bybit does not map
# min_notional into limits.cost.min) — confirm those from the exchange docs, not here.
_NUMERIC = {
    "tick_size": 1e-12,
    "qty_step": 1e-12,
    "lot_size": 1e-12,
    "min_order_size": 1e-12,
    "min_notional": 1e-9,
    "maker_fee": 1e-9,
    "taker_fee": 1e-9,
}
_EXACT = ("max_leverage", "funding_interval_hours")


def _close(a, b, tol: float) -> bool:
    return a is not None and b is not None and abs(float(a) - float(b)) <= tol


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exchange", default="bybit")
    args = ap.parse_args()

    # load_metadata_for prefers configs/metadata.<exchange>.yaml (the per-venue spec the demo gate
    # reads) over the skeleton default — load_metadata_config() alone returns the skeleton.
    cfg = load_metadata_for(args.exchange)
    if cfg.exchange_id != args.exchange:
        print(f"no per-venue metadata for {args.exchange!r} (loaded {cfg.exchange_id!r}); "
              f"expected configs/metadata.{args.exchange}.yaml")
        return 2

    try:
        from src.exchange.ccxt_adapter import CcxtExchangeAdapter
    except Exception as exc:  # noqa: BLE001
        print(f"cannot import the ccxt adapter: {exc}")
        return 2
    adapter = CcxtExchangeAdapter(args.exchange)

    print(f"metadata: {cfg.metadata_version}  verified={cfg.verified}  "
          f"({len(cfg.specs)} symbols)\n")
    mismatches = 0
    unconfirmable = 0
    for symbol, spec in cfg.specs.items():
        try:
            live = adapter.fetch_metadata(symbol)
        except Exception as exc:  # noqa: BLE001 - a bad symbol shouldn't abort the whole review
            print(f"[{symbol}]  LIVE FETCH FAILED: {exc}")
            mismatches += 1
            continue
        live_fields = {
            "tick_size": live.tick_size, "qty_step": live.qty_step, "lot_size": live.lot_size,
            "min_order_size": live.min_order_size, "min_notional": live.min_notional,
            "max_leverage": live.max_leverage, "maker_fee": live.maker_fee,
            "taker_fee": live.taker_fee, "funding_interval_hours": live.funding_interval_hours,
        }
        rows = []
        for field, cfg_val in spec.fields.items():
            if field not in live_fields:
                continue
            live_val = live_fields[field]
            if field in _EXACT:
                ok = cfg_val == live_val if live_val is not None else None
            else:
                ok = _close(cfg_val, live_val, _NUMERIC.get(field, 1e-12))
            if live_val is None:
                mark, note = "?", "live n/a — confirm from docs"
                unconfirmable += 1
            elif ok:
                mark, note = "ok", ""
            else:
                mark, note = "MISMATCH", f"config {cfg_val} != live {live_val}"
                mismatches += 1
            if mark != "ok":
                rows.append((field, cfg_val, live_val, mark, note))
        status = "all fields match live" if not rows else f"{len(rows)} field(s) to review"
        print(f"[{symbol}]  {status}")
        for field, _cfg_val, _live_val, mark, note in rows:
            print(f"    {mark:>9}  {field:<22} {note}")

    print()
    if mismatches:
        print(f"→ {mismatches} MISMATCH(es): correct the config to the exchange's real spec "
              "(and your account's real fee tier) before verifying.")
    if unconfirmable:
        print(f"→ {unconfirmable} field(s) ccxt could not confirm (e.g. min_notional): check "
              "the exchange's contract page directly.")
    if not mismatches and not unconfirmable:
        print("→ every field matches the live spec. If you have also confirmed the fee TIER is "
              "your account's, you may sign off:")
    print(
        "\nTo verify (operator sign-off — Section 6): in configs/metadata."
        f"{cfg.exchange_id}.yaml set the corrected values, then\n"
        "  verified_against: \"<exchange contract ref / date you checked>\"\n"
        "  verified_at: \"<ISO timestamp>\"\n"
        "  verified: true\n"
        "and BUMP metadata_version (any spec change is a new version). Then re-run the META gate "
        "so the DB mirrors it, and re-run demo-readiness.")
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
