# Deploy paper-trading on a VPS

Continuous **paper-trading** of the promoted strategy ensemble on **live Bybit data**, with no real
funds. Paper mode uses the offline `SimulatedVenue` — it needs **no API keys** and can never place a
real order. The loop polls Bybit's public REST for newly-closed bars, builds candidates through the
**same** feature pipeline / risk / execution path as a real run (the Parity Rule), and books
simulated fills to `paper_trades`, visible on the dashboard.

## What runs

The core stack (`docker compose up -d`): `postgres`, `redis`, `backend` (API), `dashboard`,
the `worker-*` services, `scheduler`, `trading-engine-paper`. Plus the opt-in **`paper-live`**
service (compose profile `paper`) that runs the continuous loop:

```
qbot live --mode paper --realtime --transport rest --multi-strategy \
          --poll-sec $PAPER_POLL_SEC --timeframe $PAPER_TIMEFRAME --config $PAPER_DATA_CONFIG
```

`--multi-strategy` runs only **real-data-validated promoted** strategies (a reference/synthetic
candidate can never trade here). With nothing promoted the feed simply has no candidates (a safe
no-op) — so **promotions must exist** before paper-live does anything (step 4).

## Prerequisites

- A VPS with Docker + Docker Compose, ports as you wish (the dashboard binds loopback by default;
  expose it via the `proxy` profile + Caddy, or an SSH tunnel).
- This repo checked out on the VPS.

## Steps

**1. Configure `.env`** (copy the template, set real values):
```
cp .env.example .env
# edit .env:
#   DASHBOARD_PASSWORD=<a real password>        # REQUIRED: compose refuses to start without it
#   EXCHANGE_ID=bybit
#   EXCHANGE_ENV=live                           # public mainnet DATA only; paper places no orders
#   PAPER_TIMEFRAME=4h                          # the timeframe whose promoted strategies to run
#   PAPER_POLL_SEC=60
# leave EXCHANGE_API_KEY / SECRET BLANK — paper needs none.
```

**2. Bring up the core stack:**
```
docker compose up -d
docker compose ps           # postgres/redis/backend/dashboard/workers healthy
```

**3. Verify metadata** (the venue refuses unverified specs): ensure `configs/metadata.bybit.yaml`
has `verified: true` for your universe (the META gate). If it ships `verified: false`, review and
flip it after confirming the specs.

**4. Promote a strategy** — paper-live runs the *promoted* set, so validate on real lake data first.
Promotions are keyed by `(candidate_id, strategy_version)`, so re-run the validation for the
timeframe you'll paper-trade (e.g. lead_lag is the promoted edge on **4h**):
```
docker compose exec worker-backtest python -m src.cli.main download --config configs/data.bybit.yaml   # if no lake yet
docker compose exec worker-backtest python -m src.cli.main promote-lake --config configs/data.bybit.yaml --timeframe 4h
# → expect e.g. {"promoted": ["lead_lag_xasset"], ...}
```
(Or run it from the dashboard.) Confirm with `…promote-lake` output or the dashboard's Leaderboard /
Road-to-Live page. **`PAPER_TIMEFRAME` must match the timeframe you promoted on** — a 4h-validated
strategy must run on 4h bars.

**5. Start the paper loop:**
```
docker compose --profile paper up -d paper-live
docker compose logs -f paper-live      # watch ticks: seeds the window via REST, then polls
```

**6. Monitor:** the dashboard (`:8001`, or behind Caddy with `docker compose --profile proxy up -d`)
— Overview shows live paper performance (win rate, expectancy R, equity curve) from `paper_trades`;
Control Center shows session/gate/kill-switch status. Stop with
`docker compose stop paper-live`; the kill switch (`qbot kill` / dashboard) halts trading without
stopping the container.

## Basket (cross-sectional) paper-trading

Carry / factor baskets (**funding_carry**, **residual_momentum**) run through the
`CrossSectionalEngine` — a dollar-neutral basket rebalanced on the live feed — which the per-symbol
`paper-live` loop cannot drive. They paper-trade through a **separate** path, and crucially **do not
require promotion**: a basket strategy is named explicitly and paper is simulated (no real account
to protect), so a validated-but-not-yet-promoted carry/momentum edge can be paper-traded today.
Their booked legs persist to `paper_trades` like every session, so they show up on the dashboard
**Overview** and **Paper Trading** pages.

Three ways to run them (pick one):

1. **Managed compose services (recommended on the VPS)** — restart-on-failure, one per strategy:
   ```
   docker compose --profile paper up -d paper-basket-funding-carry paper-basket-residual
   docker compose logs -f paper-basket-funding-carry
   ```
   Timeframe/cadence come from `.env` (`PAPER_BASKET_TIMEFRAME=1h`, `PAPER_BASKET_POLL_SEC=60`);
   the strategy ids default to `funding_carry` / `residual_momentum` (override via
   `PAPER_BASKET_FUNDING_STRATEGY` / `PAPER_BASKET_RESIDUAL_STRATEGY`). These run **independently**
   of `paper-live`, so lead_lag on 4h and the baskets on 1h all run at once.

2. **Dashboard (recommended for hands-on control)** — **Paper Trading** page → *Basket
   (cross-sectional) paper sessions* → pick a strategy + timeframe → **Start basket paper session**;
   the same row has a **Stop** button. The `live` worker runs sessions **concurrently**
   (`LIVE_WORKER_CONCURRENCY`, default 4), so you can Start lead_lag's live session **and** both
   basket sessions at once — each runs in parallel with its own Stop, no queueing. A second Start
   for a strategy that's already running is refused (no double-booking). (Needs the `worker-live`
   service — part of `docker compose up -d`.)

3. **Manual CLI** (ad-hoc):
   ```
   docker compose exec worker-live python -m src.cli.main \
     paper-basket --strategy funding_carry --timeframe 1h --poll-sec 60
   ```

All three drive the same `CrossSectionalEngine` rebalance / leg / funding math as the backtest (the
Parity Rule). The loop math is offline-proven (`tests/test_basket_paper.py`); the live REST feed is
network-dependent and VPS-validated. PAPER only — simulated fills, no real orders/funds.

## Basket DEMO trading (real orders, virtual funds)

Paper books **simulated** fills, so it can tell you whether the *edge* is there but never whether
the **cost model** is. `qbot demo-basket` runs the identical basket math against the Bybit
**demo** account: every leg open/close is a real order, booked at the **observed** fill.

```
docker compose exec worker-live python -m src.cli.main \
  demo-readiness --strategy funding_carry          # pre-flight; places nothing

# BOTH strategies on the ONE demo account — repeat --strategy:
docker compose exec worker-live python -m src.cli.main \
  demo-basket --strategy funding_carry --strategy residual_momentum \
              --timeframe 1h --poll-sec 60
```

Or from the dashboard: the `run_basket_demo_session` job (same `live` worker, same Stop button),
which takes a `strategies` list. Sessions are tagged `demo:basket:<strategy>:…` — one per
strategy, so their statistics stay separate from each other AND from `paper:` history.

### One account, several baskets

An exchange holds **one position per symbol**. Two baskets that both trade BTC share it whether
or not they know — so running them as two independent processes gives each a private mirror of a
book they actually share: each sizes against exposure it does not own, and a whole-symbol close
by one silently flattens the other's leg.

Co-hosting solves it. Naming several strategies in one `demo-basket` run puts them behind a
shared **`NetPositionManager`** (`src/live/basket_exec.py`): each strategy states its own desired
position, the manager owns the aggregate and sends only the position **delta** needed to keep
`Σ(strategy intents) == exchange position`. So:

- a long from one basket and a short from another **net**, exactly as the exchange nets them;
- a close is this strategy's own delta, never a whole-symbol flatten — nobody can close anybody
  else's leg;
- reduce-only is set only for a strict reduction (an order crossing zero would be rejected);
- reconciliation compares the **aggregate** against the real book each tick. A leg that vanished
  (disaster stop, liquidation, manual close) is booked at its observed exit and un-mirrored;
  exposure *larger* than our intent is reported as CRITICAL and never adopted;
- the account equity is **split** across the strategies, so the aggregate gross still respects
  `basket_demo.max_gross_pct`. This is the shared capital allocator Section 17 requires before
  several strategies trade one account — without it, N strategies each sizing off the full
  balance is N× the intended exposure.

Per-strategy accounting is unchanged: each keeps its own engine, equity slice and PaperSession,
so the dashboard still shows funding_carry and residual_momentum separately.

`basket_demo.require_flat_book` still refuses to START on a non-flat account: a leftover position
cannot be attributed to any strategy in the new session, so adopting it would corrupt the
aggregate from tick 1. Start the strategies **together**; don't add a second process later.

**Known limitation — no intra-tick coalescing.** Each strategy's rebalance is sent as its own
order, in sequence. When two strategies trade the same symbol in the same tick, that is two
orders (and two spreads) where a batched implementation would net them into one. The resulting
position is always correct; only the cost is higher than optimal. Since each order is the
strategy's own turnover, per-strategy costs stay directly comparable to its backtest.

**It refuses to start unless all of these hold** — each is a deliberate refusal, not a warning:

| Requirement | Why |
|---|---|
| `EXCHANGE_ENV=demo` or `testnet` + API keys | `live` is refused outright — this path never reaches real money |
| `demo-readiness --strategy <id>` PASSes | verified metadata, ownership prefix, risk caps, clean book |
| **every** named candidate is **lake-validated** | Section 13 — a synthetic/reference-only edge never touches an account. It need NOT be *promoted*: a basket is structurally excluded from the promoted ensemble, and measuring execution is the point |
| the demo account is **FLAT** | see *One account, several baskets* below |
| the balance is readable and ≥ `min_account_equity` | a basket sizes its whole book off equity; guessing it is not acceptable |

**Legs carry a wide disaster stop, not a normal one.** A basket holds delta-neutral and exits on
the rebalance cadence — a per-leg stop would knife the hedge and leave the other legs naked. But
an unprotected position on a real venue is what Section 2.2 forbids. The compromise
(`configs/live.yaml` → `basket_demo.disaster_stop_frac`, default 25%) is an exchange-resident stop
far outside normal basket behaviour: invisible in operation, but it bounds the loss if this
process dies with legs open. Set it to `0` to place bare legs — a deliberate, logged deviation.
If a disaster stop *does* fire, the next tick books that leg at its observed exit (it is a real
exit) and drops it from the mirror.

**Do not mix in a separate process.** The netting only works for strategies co-hosted in one
session. `paper-live` (per-symbol, e.g. lead_lag) against the SAME demo account is a second
independent mirror and would reintroduce exactly the collision described above — run it on its
own demo account, or leave it in paper.

**What a demo run is for.** It answers one question paper cannot: does the backtest's fee/slippage
model survive a real book? The session logs an execution summary (fill rate, average entry
slippage, maker rate, total fees, rejects) — compare it against the backtest's assumed costs
before any of this is considered for real money. It is **not** a promotion, and grants nothing.

## Risk & position management across the parallel strategies

The three strategies run as **three separate processes**, each with its **own** equity pool
(`account.initial_equity`) and its **own** `PaperSession`. Risk is enforced *within* each process by
**two different models**, and — importantly — there is **no portfolio layer across them**:

- **`lead_lag` (per-symbol path)** runs the full `RiskManager` (`src/risk/manager.py`): per-trade
  sizing `equity × base_risk_pct × risk_scale / |entry − stop|` (capped at the envelope), plus
  portfolio **heat** (Σ open risk), net **beta-to-BTC**, and **concurrency** caps, plus the circuit
  breakers. Positions are managed per tick by exchange-style stop / reachable TP / ATR trail /
  time-stop.
- **The baskets (`funding_carry`, `residual_momentum`)** do **not** use `RiskManager`. Sizing is
  `gross = equity × portfolio_gross × risk_scale`, split dollar-neutral across legs; `stop_frac` is
  an **accounting R-unit only**. Legs are held delta-neutral and **exit only on the rebalance
  cadence** (or at session end) — there is **no protective stop monitored between rebalances** (a
  stop would knife the hedge). This is correct for a carry/factor basket but means a leg's mark can
  drift until the next rebalance.

**Cross-process caveats (matter for live, not for isolated paper measurement):**

- **No aggregation.** The heat / beta / concurrency caps apply *within* `lead_lag` only; the basket
  processes don't see each other or `lead_lag`. Each process also sizes off the **full**
  `initial_equity`, so three processes ≈ **3× the intended account exposure**, and a
  `lead_lag`-short-BTC vs basket-long-BTC position does **not** net. Fine for measuring each edge's
  standalone Sharpe in paper; **wrong for a single real account** — a shared capital allocator /
  aggregate-exposure layer is required before live.
- **Kill switch (shared).** The global `KillSwitch` (redis-backed, also a local file) is the one
  control all three honour: `lead_lag` checks it per tick, and the basket loop now halts on it too
  (`_halt_check` in `src/live/basket.py` — flattens every leg and persists on engage). Engage from
  the dashboard or `qbot kill`; it stops *all* three processes (redis backend is shared across
  containers).

## Notes & current limitations

- **One timeframe per `paper-live`.** The loop runs a single timeframe and `--multi-strategy`
  resolves *all* promoted strategies (not filtered by their validation timeframe). Today only
  lead_lag (4h) is promoted, so one `paper-live` on 4h is correct. When strategies promote on other
  timeframes (e.g. funding_carry on 1h), run a **second** `paper-live` with `PAPER_TIMEFRAME=1h`
  (override per service) — until per-timeframe filtering of the promoted set lands.
- **Real-time feed = REST polling of closed bars** (no websocket yet). `--poll-sec` is the cadence;
  the loop waits for a new closed bar before each tick, so on 4h it acts a few times a day.
- **No real funds, ever, in paper mode.** Going to testnet/live is a separate, gated path
  (`--profile live`, `ENABLE_LIVE_TRADING`, sign-off, real keys) — out of scope here.
- **Parity:** maker fills, trailing/TP brackets, `risk_scale`, and the time-stop are honored in the
  live/paper path. Cross-sectional (basket) strategies (funding_carry, residual_momentum) do **not**
  run through `qbot live` (a per-symbol directional loop) — see *Basket (cross-sectional)
  paper-trading* above for their dedicated path (managed services / dashboard button / CLI).
- **Promotion vs paper:** `paper-live --multi-strategy` runs the **promoted** ensemble only (the gate
  protects real accounts). The basket path is **not** promotion-gated — paper is simulated, so a
  3/5-fold edge like funding_carry/residual_momentum can be paper-traded by name while it's still
  short of promotion. Promotion is written *only* by the validation gate (`promote-lake`); there is
  no manual force-promote.
