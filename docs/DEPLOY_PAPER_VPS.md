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
#   DASHBOARD_PASSWORD=<a real password>        # mandatory outside local
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
  live/paper path; cross-sectional (basket) strategies like funding_carry run through their engine
  only in the backtest today — wire basket execution into the live path before paper-trading carry.
