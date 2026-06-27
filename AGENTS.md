# AGENTS.md — Quant + ML + Adaptive Crypto Trading Bot
> Master instruction for building a production-grade, quant-first, ML-assisted, multi-symbol crypto perpetual futures trading system on one centralized exchange.
>
> The system is data-owned, research-driven, risk-first, explainable, and designed for staged progression from deterministic quant strategies to ML, online learning, and reinforcement learning only after sufficient validation.
>
> **Capital-agnostic**: every limit is expressed in % equity / R / portfolio heat, never in absolute currency.
>
> **Single source of truth for any coding agent.** Where a detail is unspecified, choose the most capital-preserving option and record the assumption in `docs/decisions/`.

---

## Conventions & Definitions (read first)

- **Audience & authority:** this file is the single source of truth for any coding agent or engineer building the system. Implement strictly to it. Where a detail is unspecified, choose the most capital-preserving option and record the assumption in `docs/decisions/`.
- **Status labels:** every code artifact in a delivery is labelled `tested` / `written-not-run` / `pseudocode`.
- **Version vocabulary:** **v1 / deterministic baseline** = the first production system (Roadmap phases 1–7), deterministic and pre-ML. **v2** = shadow/assisted ML (phases 8–10). **v3** = bounded adaptation (phases 11–13).
- **`risk_cap`** ≡ `risk_envelope.max_risk_pct_per_trade` (Section 17). **`equity`** = current account equity read from the exchange.
- **Gate:** a named automated pass/fail check with an explicit pass condition **and** concrete remediation steps. Every gate is catalogued in **Appendix A**, declared in `configs/gates.yaml`, executed by the **Gate Runner** (Section 25), and must surface remediation on failure. A failed gate is never a dead end — it always carries concrete action items.
- **Objective:** drive every gate to **PASS** and go live, safely. The whole workflow is designed so the operator always sees what remains and exactly how to clear it.
- **Idempotency:** all jobs (data download, backfill, gate runs, retrain, deploy) must be safely re-runnable.
- **Cross-references** use section numbers; keep them updated if sections are reordered.
- **`[VERIFIED]`** vs **`[UNVERIFIED]`:** any exchange-specific value (fee, min-notional, tick size, funding schedule) must be marked `[VERIFIED]` against current exchange docs, or `[UNVERIFIED]` if not yet confirmed. No live trading with `[UNVERIFIED]` metadata.

---

## 0. Mission

Build a disciplined crypto perpetual futures trading system that can automatically:

- discover and track the tradable symbol universe on one exchange;
- download, backfill, validate, store, and serve all data required for research, backtesting, paper trading, and live trading;
- research and validate quant strategy hypotheses across many symbols;
- generate candidate trades across the active universe;
- rank symbols and strategies when multiple candidates are available;
- execute only approved trades through strict risk and execution controls;
- monitor performance at portfolio, strategy, regime, session, and symbol level;
- run all validation gates as background jobs from the dashboard;
- show exact remediation steps when gates fail;
- progress toward live trading only when all required gates pass;
- use ML, online learning, and RL only after deterministic baselines and staged validation prove they add value.

The goal is to build a system that can eventually trade live, but not by bypassing safety. The correct path to live is to pass every required gate with auditable evidence, fix every failed gate with clear action items, and promote components only through versioned approval.

Profit alone is not success.

The system is successful only if its behavior is:

- understandable;
- repeatable;
- data-valid;
- statistically defensible;
- risk-controlled;
- operationally safe;
- explainable at every trade decision;
- capable of showing exactly what must be fixed when it is not ready.

---

## 1. Priority Stack

When requirements conflict, resolve them in this order:

1. Capital protection
2. Exchange safety
3. Data correctness
4. Operational reliability
5. Honest validation
6. Explainability
7. Risk-controlled progression toward live
8. Strategy performance
9. ML / online learning / RL complexity
10. Convenience
11. Speed of implementation

If a request violates a higher-priority rule, do not implement it. Implement the safest practical alternative and explain what gate or safety rule would be violated.

---

## 2. Non-Negotiable Safety Rules & the Immutable Risk Envelope

### 2.1 Non-negotiable rules (every phase, every version)

- No live trading without explicit manual approval, verified exchange metadata, exchange-side stop protection (where supported), and startup reconciliation.
- No strategy, ML model, or learning component may bypass the Risk Layer.
- No hidden parameter updates during a live session. Any change → halt → version → manual approval (exception: the bounded autonomous learner of Section 21, which still cannot touch the envelope).
- No martingale, no grid averaging-down, no revenge logic, no uncontrolled leverage.
- No trading on stale data, missing/inconsistent metadata, unreconcilable positions, or foreign/unknown orders.
- No trade the system cannot **explain** and whose post-fee/slippage expected cost it cannot compute.
- On any critical safety failure: **halt**, cancel only the bot's own orders where safe, alert.

### 2.2 The Immutable Risk Envelope (the box no learner may ever modify)

The following are version-controlled constants. **No deterministic optimizer, ML model, online learner, or RL policy may widen, disable, or exceed them — at any maturity stage, ever.** Learning may only act *inside* this box.

- Exchange-resident stop-loss on every position, placed atomically at entry.
- Max leverage cap (leverage is a *consequence* of sizing, never a target).
- Max risk % of equity per trade (`risk_cap`).
- **Portfolio heat cap** (Σ open risk across positions).
- **Net portfolio beta-to-BTC cap.**
- Daily-loss and max-drawdown circuit breakers (portfolio level) → halt to manual reset.
- Manual kill switch (CLI + dashboard), implemented independently of the dashboard so it works even if the UI is down.
- Hard blockers: stale data, missing metadata/fees, toxic execution, illiquidity below threshold, no-trade regimes.
- A **frozen fallback policy**: the last manually-approved deterministic configuration the system reverts to instantly on any learner anomaly.

A learner may change *which* validated strategies are weighted, bet size within `[0, risk_cap]`, skip/allow decisions, execution routing, and parameter values **within pre-declared bounded ranges** — and nothing else.

**Example envelope values (configurable, but never widened by learner):**
```yaml
risk_envelope:
  max_leverage: 5
  max_risk_pct_per_trade: 0.01        # 1% equity per trade
  portfolio_heat_cap: 0.05             # 5% total open risk
  net_beta_btc_cap: 0.30               # 30% net beta to BTC
  daily_loss_limit: 0.03               # 3% equity
  max_drawdown_limit: 0.10             # 10% from peak
```

---

## 3. Scope

### Included

- One centralized exchange active at a time
- Multi-symbol crypto perpetual futures
- Dynamic symbol universe management
- Automated historical data download and backfill
- Automated live data collection
- Data validation and data quality gates
- Event-based backtesting
- Walk-forward validation
- Paper trading
- Live trading after gates and approval
- Quant strategy research
- Cross-symbol signal evaluation and ranking
- ML-assisted trade filtering and strategy/symbol selection
- Online learning after sufficient validated observations
- RL after mature simulation, shadow, and recommendation performance
- Dashboard-controlled background gates
- Dashboard remediation workflow for failed gates
- General and per-symbol analytics
- Time-period selectable dashboard reporting
- Full explainability and audit logs

### Excluded by default

- Multiple exchanges active at the same time
- Options
- Spot/perp hedged arbitrage unless explicitly enabled later
- HFT or co-location claims
- Unbounded market making
- Martingale
- Uncontrolled grid averaging down
- Fully autonomous ML/RL before staged validation
- Manual trading mixed with bot-managed orders unless isolated by account/sub-account and order prefix

---

## 4. Configuration and Versioning

All runtime behavior must be controlled through versioned configuration.

Required identifiers:

- `EXCHANGE_ID`
- `EXCHANGE_ENV`
- `EXCHANGE_ACCOUNT_TYPE`
- `BOT_INSTANCE_ID`
- `ORDER_CLIENT_ID_PREFIX`
- `CONFIG_VERSION`
- `UNIVERSE_VERSION`
- `DATA_VERSION`
- `STRATEGY_VERSION`
- `FEATURE_SET_VERSION`
- `RISK_POLICY_VERSION`
- `EXECUTION_POLICY_VERSION`
- `MODEL_VERSION`, if ML is active
- `ONLINE_LEARNER_VERSION`, if online learning is active
- `RL_POLICY_VERSION`, if RL is active

Rules:

- No live trading without a frozen `CONFIG_VERSION`.
- No live trading with unversioned strategy, data, risk, execution, or model artifacts.
- No live session may silently change parameters.
- Any material change creates a new version.
- Every report must link to the exact versions used.
- Every trade decision must store all relevant versions.

### Config Freeze Procedure

Before live activation:
1. Stage config changes in a new versioned copy.
2. Run `gate:config-freeze` (Appendix A) to verify all versions are locked.
3. Manual review and approval.
4. Git tag the frozen configuration.
5. Only then enable live trading.

---

## 5. System Layers

Strict separation. Higher layers may not reach around lower ones.

1. **Exchange Adapter** — market metadata, symbols, order types, balances, positions, websockets, historical data
2. **Data Platform** — collect, validate, store, serve market + account data; own the full historical dataset
3. **Universe Manager** — dynamic symbol universe tracking, filtering, versioning
4. **Feature Pipeline** — one code path for backtest and live; only data available at decision time
5. **Regime Engine** — deterministic regime detection (v1); ML shadow only until promoted
6. **Strategy Engine** — deterministic candidate generation; never places orders
7. **Candidate Ranking Engine** — rank candidates before risk approval
8. **Risk Manager** — approve / reject / resize / block; absolute authority over strategies, ML, and learners
9. **Execution Engine** — order construction, routing, placement, cancellation, reconciliation, execution-quality measurement
10. **Backtest Engine** — event-based; same feature pipeline as live
11. **Paper Trading Engine** — Phase A technical validation; Phase B strategy validation
12. **ML Layer** — shadow predictions, meta-labeling, regime classification, strategy selection, execution modeling
13. **Online Learning Layer** — bounded adaptation within the envelope
14. **RL Layer** — bounded decision policies; research/shadow first
15. **Monitoring and Dashboard** — control center, not just statistics viewer
16. **Reporting and Review** — process-quality separation from outcome
17. **Job Orchestration Layer** — background jobs, gates, remediation
18. **Model and Artifact Registry** — versioned artifacts only

Rules:

- Research, backtest, paper, and live must use the same feature code path.
- Strategies generate candidates, not orders.
- The risk manager approves or rejects candidates.
- The execution engine places orders only after risk approval.
- ML, online learning, and RL may not bypass deterministic safety checks.
- Dashboard actions create jobs or approvals; they do not directly mutate live behavior without versioned promotion.
- No strategy, model, or learner may call exchange APIs directly — only through the adapter.

---

## 6. Exchange Scope

The system runs on one exchange at a time.

All exchange-specific logic must be isolated behind an exchange adapter.

No strategy, feature pipeline, ML model, risk module, RL policy, dashboard component, or notebook may call exchange APIs directly.

### Exchange Adapter Responsibilities

The exchange adapter must provide:

- symbol list;
- contract metadata;
- symbol status;
- tick size;
- lot size;
- quantity step;
- price precision;
- minimum order size;
- minimum notional;
- leverage limits;
- margin mode;
- position mode;
- maker/taker fees;
- funding schedule;
- funding history;
- mark price;
- index price;
- open interest;
- liquidation data if available;
- historical OHLCV;
- historical trades if available;
- order book snapshots if enabled;
- websocket market data;
- websocket private data;
- account balances;
- open positions;
- open orders;
- order placement;
- order cancellation;
- order status;
- fills;
- reconciliation.

All metadata must be stored with timestamps and versioned.

If metadata is missing, stale, contradictory, or unverified, trading must halt or skip affected symbols.

All exchange-specific behavior lives here. Verify signatures/fees/limits against current docs; mark anything unverifiable `[UNVERIFIED]` until confirmed, then `[VERIFIED]`.

### Metadata Verification Workflow

1. Adapter fetches metadata from exchange API.
2. Store with timestamp and `[UNVERIFIED]` flag.
3. Operator reviews against exchange docs and marks `[VERIFIED]`.
4. Gate `META` (Appendix A) checks that all active symbols have `[VERIFIED]` metadata.
5. Any metadata change → new version → re-verification required.

---

## 7. Order Ownership

The bot may only manage orders it created.

Required configuration:

- `BOT_INSTANCE_ID`
- `ORDER_CLIENT_ID_PREFIX`

The prefix must include the bot instance identity, for example:

- `QBOT_MAIN_v1_`
- `QBOT_RESEARCH_v3_`
- `QBOT_PAPER_v2_`

Rules:

- Every bot-created order must include the configured prefix where the exchange supports client order IDs.
- The bot may only cancel, replace, or close orders with its own prefix.
- If unknown open orders are detected, the bot must halt new entries and alert.
- If unknown positions are detected, the bot must halt new entries and alert.
- The bot must never modify orders from another bot, manual trading, or another account process unless emergency mode is explicitly enabled.
- Emergency close mode must require explicit confirmation and be fully audited.

---

## 8. Data Ownership

The bot is solely responsible for acquiring and persisting every piece of data its backtests and live decisions need. No reliance on external manual exports.

No validated backtest may depend on manually prepared files unless those files are imported into the data store, validated, timestamped, and assigned a `DATA_VERSION`.

### Required Data Types

The system must collect and store:

- OHLCV candles for configured timeframes;
- mark price;
- index price;
- funding rates;
- funding timestamps;
- open interest;
- traded volume;
- symbol metadata;
- maker/taker fees;
- spread snapshots;
- order book snapshots if enabled;
- liquidation data if available;
- historical trades if available;
- account balances;
- positions;
- open orders;
- placed orders;
- canceled orders;
- fills;
- rejected candidate trades;
- accepted candidate trades;
- decision logs;
- model predictions;
- strategy outputs;
- risk decisions;
- execution quality metrics;
- dashboard gate results;
- reports and artifacts.

### Historical Data Manager

The historical data manager must:

- discover missing data ranges;
- backfill missing OHLCV candles;
- backfill mark/index prices where available;
- backfill funding history;
- backfill open interest where available;
- backfill liquidation data where available;
- backfill historical trades where available;
- store raw and normalized data;
- deduplicate records;
- detect gaps;
- repair safe gaps automatically;
- quarantine unsafe data;
- produce data quality reports;
- track data source;
- track download timestamp;
- track checksum or row count where possible;
- support re-download;
- support reproducible `DATA_VERSION` snapshots.

### Live Data Manager

The live data manager must:

- subscribe to websocket streams;
- detect stale streams;
- detect disconnects;
- backfill via REST after reconnect;
- compare websocket and REST data where appropriate;
- prevent feature calculation from stale data;
- halt affected symbols if critical live data is stale;
- halt all trading if exchange-wide live data integrity fails.

### Data Quality Gate

Before research, backtest, paper, or live, the system must verify:

- no critical missing candles;
- no duplicate records;
- no out-of-order timestamps;
- no future timestamps;
- no impossible prices;
- no extreme unexplained gaps;
- funding timestamps are aligned;
- mark/index/perp data are aligned;
- exchange metadata is verified;
- feature inputs are reproducible;
- data coverage meets configured minimums;
- symbol universe can be reconstructed for the tested period.

If data quality fails, the run must be blocked or marked invalid.

The dashboard must show exact remediation steps for each failed data check.

---

## 9. Dynamic Symbol Universe

The bot must track available symbols and decide where signals are valid.

There must be no hardcoded single-symbol assumption.

### Universe Manager Responsibilities

The universe manager must:

- sync available perpetual futures symbols from exchange metadata;
- filter inactive or non-tradable symbols;
- filter symbols with insufficient history;
- filter symbols with missing metadata;
- filter symbols with unacceptable spread or liquidity;
- filter symbols with unstable contract specifications;
- filter newly listed symbols until they have sufficient history;
- maintain universe versions;
- store universe membership history;
- support backtest-time universe reconstruction;
- support scheduled live universe refresh;
- mark symbols as active, disabled, quarantined, or research-only;
- prevent trading on symbols that do not pass gates.

**Implemented (`src/universe/manager.py`, `src/universe/filters.py`, `configs/universe.yaml`, UNIV gate).** The `UniverseManager` scores the config candidate set against the Section-9 filters (using owned data + `[VERIFIED]` metadata) and records each symbol `active` / `research_only` / `quarantined` with a per-filter reason (never silently dropped). The build is **content-addressed** (version id `univ_0001_<hash>` = pure function of membership + statuses + filter policy, so an identical universe re-uses the version) and **history-logged** (the diff vs the previous version is written to `universe_changes` — membership history). Backtest survivorship / future-universe leakage is prevented in the engine via `activation_ts` (a symbol is tradable only at decision times at/after it entered the universe).

**Known gap (NOT yet implemented):** liquidity-**ranked** dynamic selection of the top-N universe and **scheduled monthly rotation** are not wired. The current real-data validation universe is a **hardcoded point-in-time 20-symbol list** in `configs/data.bybit.yaml` (the 20 most-liquid Bybit USDT perps that listed before mid-2022, chosen manually to prove edge with breadth), NOT the manager's output; `configs/universe.yaml` still carries the skeleton/test candidate set. So "no hardcoded single-symbol assumption" holds (the system is multi-symbol throughout), but a survivorship-free, liquidity-rotated universe is outstanding work.

### Default Universe Filters

Configurable filters should include:

- quote currency;
- contract type;
- minimum daily notional volume;
- minimum historical data length;
- maximum missing data percentage;
- maximum median spread;
- minimum order book depth if order book data is used;
- funding history availability if funding is used;
- open interest availability if open interest is used;
- liquidation data availability if liquidation features are used;
- minimum listing age;
- maximum metadata-change frequency;
- maximum abnormal gap frequency.

### Signal Routing and Symbol Selection

The bot must evaluate candidate signals across the active universe.

For every signal scan, the system must log:

- universe version;
- symbols evaluated;
- symbols skipped;
- skip reasons;
- candidate signals;
- ranking score;
- selected symbol;
- rejected alternatives;
- final decision;
- risk allocation;
- execution route.

A trade may be opened only on a symbol where:

- symbol is active;
- metadata is verified;
- data is fresh;
- strategy is enabled for that symbol;
- setup quality passes;
- ranking gate passes;
- risk limits allow exposure;
- execution constraints pass;
- expected edge after costs is positive.

---

## 10. Feature Pipeline & Parity Rule

- One feature-computation code path for **both** backtest and live. **No feature that does not exist in backtest may be critical to a live decision.** The only difference between modes is the data-reading adapter.
- Every feature uses **only data available at decision time** (no close-before-candle-close, no funding/liquidation data before its timestamp, no misaligned mark/index).
- Features are stored and reproducible; a feature is not used unless its timestamp behavior is understood.

Feature groups:

- **Market:** returns, RV, ATR-percentile, directional efficiency, trend slope, structure, volume
- **Cross-asset:** leader returns, rolling beta/correlation, relative strength, lead-lag gap, dispersion, exchange-local index proxy
- **Derivatives:** funding, funding z-score, premium vs mark/index, OI change, liquidation proxy, basis state
- **Execution:** spread, depth, slippage estimate, latency, recent fill quality, adverse selection
- **Context:** hour UTC, day, session, weekend, pre/post-funding window, recent strategy/symbol/market volatility

---

## 11. Regime Detection

v1 is **deterministic, per-symbol**. ML regime classification is **shadow-only** until promoted.

Use normalized volatility (**ATR% = ATR/close**, percentile over a rolling window — never raw ATR). Dimensions: volatility, directional efficiency, trend strength, spread/liquidity, market-wide impulse, correlation state, funding state, premium state, session state, **data-quality state**.

Regimes (the numeric suffix `Rn` is a stable **identifier, not a priority rank**). **Resolution priority** when several match — the safest wins, in this exact order (expose it as one `regime_priority` list in config so detection and tests share a single source):

1. `R8_DATA_UNSAFE`
2. `R7_TOXIC_EXECUTION`
3. `R6_LIQUIDATION_EVENT`
4. `R3_HIGH_VOL_EXPANSION`
5. `R4_HIGH_VOL_CHOP`
6. `R5_MARKET_WIDE_IMPULSE`
7. `R2_TREND`
8. `R1_LOW_VOL_RANGE`

No strategy may trade in `R8_DATA_UNSAFE`, `R7_TOXIC_EXECUTION`, or `R4_HIGH_VOL_CHOP` (the default no-trade/protection regimes).

- **Stability:** a regime must persist a minimum number of bars before switching (anti-whipsaw).
- **In-flight rule (mandatory):** if the regime changes mid-trade, the open position is managed to exit by the **strategy that opened it** (exit logic frozen at entry). Only **new** entries respond to the new regime. A symbol leaving the universe likewise does not force-close; it is managed to exit.

---

## 12. Preferred Quant Strategy Families

The strategy layer must focus on quant and algo edges.

Retail-style indicators may be used as features, but they must not be the primary edge claim.

Strategies are **testable hypotheses, not proven edges.** A strategy never trades because a pattern "looks good." Structural edges have a higher prior than pure TA.

Each strategy declares: hypothesis; market condition; expected edge source; data requirements; entry; exit; invalidation; risk assumptions; cost assumptions; expected failure modes; required validation tests; promotion criteria.

**Per-strategy exit geometry must match the edge profile:**
- Mean-reversion: near fixed TP, wider SL, asymmetric (high win-rate / low R).
- Momentum/trend: **no fixed TP** (the tail is the edge), exchange-native trailing + time stop, explicit initial SL for sizing.
- Volatility/liquidation: fast short TP, strictest filters, harsher slippage assumption.

**Implemented per-strategy execution model (`configs/strategies.yaml`, `src/strategies/candidates.py`; current `STRATEGY_VERSION strat_0007`).** Each candidate emits a `Signal` carrying its OWN entry + exit geometry, so families no longer share one engine geometry:
- **Entry execution per strategy:** `maker` (passive-limit entry posted `limit_offset_atr_mult × atr_pct` inside the reference; fills only if the bar trades through it, else no-fill; maker fee, zero slippage) vs taker market. Basis uses maker; momentum families use taker (need immediate fills).
- **Exit geometry per strategy:** volatility-scaled stop `max(stop_frac, atr_stop_mult × atr_pct)`; momentum uses a **reachable R-multiple take-profit** `tp_r_mult × stop` (e.g. 1.3R) PLUS an ATR trailing stop (`atr_trail_mult`) — whichever fires first; mean-reversion uses an ATR take-profit. A per-bar **`manage()` hook** lets a strategy exit on its own thesis (mechanism present; basis's premium-reversion variant was tested and disabled as a net negative — see Section 16 anti-overfitting). A **time-stop** (`hold_bars`) backstops all.
- **Entry quality filters:** basis fades only a **band** `[premium_threshold, premium_cap]` (an extreme premium is one-way repricing, not reversion).
- **Per-strategy `risk_scale`** (≤ 1, Section 17): a real-but-hot edge is sized DOWN so its drawdown fits the envelope without changing its `expectancy_r`.
- Current real-data state (20-symbol 4h `bybit_0002`): **Family A lead_lag — PROMOTED** (the first to clear the gates); Family B basis — improved but shelved on hold-out economic viability (recent-period edge decay); Family G xsection — shelved (cross-sectional relative-strength carries no OOS edge on this universe). State is a point-in-time validation snapshot, not a permanent claim.

**New structural candidates (Phase 13+, `lake_only` — validated on real lake data, no synthetic fixture).** The promoted edge (lead_lag) is *directional prediction*, which decays with both speed and time. To grow capital via a higher-Sharpe DIVERSIFIED book (uncorrelated edges combine to `Sharpe·√N`), add **structural** edges (cash flows / relative value / forced flow) that don't depend on forecasting price and so don't decay the same way. Built without changing existing candidates:
- **Family C — Funding-Dispersion Carry (portfolio, market-neutral-ish).** Cross-sectionally LONG the perps whose funding is most negative (paid to be long) and SHORT those whose funding is most positive (paid to be short), ranked by `funding_z` vs the cross-sectional mean (`funding_rank_threshold`). The edge is the funding *cash flow* (the engine's funding model credits it), not a price forecast — robust and diversifying vs the directional book. Held over a carry horizon (`hold_bars` spanning funding settlements); ATR stop bounds the idiosyncratic price risk. Entry filters out unrelated dislocations; sized small (`risk_scale`) because the basket is only partially beta-hedged in a per-trade engine.
- **Family D — Liquidation / OI-Flush Reversal (per-symbol, event-driven).** Fade a forced-flow overshoot: an abnormal short-horizon return (`ret_short`) WITH an open-interest collapse (`oi_change` ≤ −`oi_flush_frac`, positions liquidated) AND a volatility spike (`rv_short`) ⇒ enter AGAINST the flush (down-flush ⇒ long the bounce). The gross move is large (liquidation overshoot), so it clears costs even taker; mean-reversion exit geometry (near TP, tight stop, short hold). The one genuinely *directional* edge expected to survive at faster horizons (the move dwarfs cost).
- **Family I — Beta-Residual Cross-Sectional Momentum (portfolio).** The correct fix for the dead raw-return xsection: rank the universe by the cumulative **beta-residual** return `Σ(r − β·r_mkt)` over `signal_window` (the idiosyncratic move with the common market factor stripped), long the top basket / short the bottom through the `CrossSectionalEngine`. Beta + the market factor are computed **inside the engine** from the bars already in hand (rolling cov/var vs the equal-weight universe return) — **no new feature column, no rebuild** (the originally-proposed approach below predicted a feature was needed; computing it in-engine avoided that). Built to test *reversion* (over-reaction); on the real 1h lake the residual **trends** — reversion is net-negative on both sides at every (window, cadence) swept, momentum is net-positive on both (the residual-momentum factor: Blitz-Huij-Martens). `score_mode=residual_momentum`; `residual_reversion` is retained as a tested-negative control. Real-data state (20-sym 1h): exp +0.066R over 10.6k legs, hold-out PF 1.138 ✓, deflated-Sharpe 0.528 ✓, stress ✓ — **blocked only on the 4th directional fold (3/5)**, the same clean spot as funding_carry. Params `sw=24/24h` chosen a priori for robustness (both sides positive, most trades), NOT the grid-max corner (`sw96/72h` hits +0.22R but concentrates the edge entirely on the long side — fragile). (Conceptual taxonomy letter `I` to avoid colliding with the conditioning families E/F/H below; thematically a refinement of Family G.)
- **Proposed next:** cointegration/pairs spread reversion and multi-day cross-sectional momentum. (Beta-residual, listed here previously as needing a new feature, is now Family E — done **in-engine** with no feature/rebuild.) Cointegration spread would still need a new feature (the rolling hedge ratio + spread).

### Family A — Cross-Asset Lead-Lag

Hypothesis: A statistically significant move in a dominant asset or market cluster can lead delayed moves in related assets under specific correlation, volatility, liquidity, and session regimes.

Inputs: returns across universe; BTC/ETH returns if present; rolling beta; rolling correlation; relative response gap; market impulse strength; volatility regime; spread and liquidity; session context.

Output: candidate symbol; direction; expected response; invalidation; expected cost; confidence score.

### Family B — Perpetual Premium / Basis Mean Reversion

Hypothesis: Extreme deviations between perpetual price, mark price, index price, and fair value may mean-revert under specific trend, funding, volatility, and liquidity conditions.

Inputs: perp mid price; mark price; index price; premium; premium z-score; funding rate; funding z-score; realized volatility; spread; trend state; open interest.

Rules: Funding may support or block a setup. Funding may not create a trade by itself. Premium reversion must be measured net of fees, slippage, and funding.

### Family C — Funding / Carry Conditional Bias

Hypothesis: Extreme funding may indicate crowded positioning and may improve expectancy when aligned with independent price, flow, or premium-reversion signals.

Allowed uses: cost filter; trade blocker; directional bias modifier; holding filter; exit urgency modifier; ML feature.

Forbidden early use: funding-only contrarian trades.

Rule: Funding can support a setup, but funding cannot create a setup.

### Family D — Liquidation Cascade Reversal / Exhaustion

Hypothesis: After abnormal displacement, volume spike, open interest shift, liquidation event, and spread normalization, price may temporarily overshoot and revert once forced flow exhausts.

Inputs: abnormal 1m/5m return; volume spike; open interest change; liquidation data if available; spread normalization; volatility spike; post-cascade stabilization.

Rules: No trade during the first uncontrolled spike unless explicitly validated. Wait for stabilization. Use reduced risk. Require strict slippage checks. Research-only until robustly validated.

### Family E — Volatility Regime Switching

Hypothesis: The same signal has different expectancy depending on realized volatility, volatility-of-volatility, spread, liquidity, and market structure.

Allowed uses: allow or block strategy families; reduce size during high-risk regimes; modify holding period; modify order type; modify stop distance; block toxic volatility chop.

### Family F — Intraday / Session Conditional Expectancy

Hypothesis: Signal quality varies across Asia, Europe, US session, pre/post funding windows, weekends, and low-liquidity periods.

Allowed uses: strategy permission; risk scaling; no-trade windows; signal quality adjustment.

This family should condition other strategies, not create trades alone.

### Family G — Cross-Sectional Relative Strength / Dispersion

Hypothesis: During market-wide impulses, some assets lead, lag, overreact, or underreact relative to beta-adjusted expected movement.

Inputs: returns across universe; beta-adjusted returns; correlation clusters; dispersion; market index proxy; leader/follower ranking; volume participation.

Rules: Must account for liquidity. Must account for symbol-specific costs. Must avoid newly listed or unstable symbols unless research-only.

### Family H — Execution Alpha

Hypothesis: For short-horizon crypto futures strategies, realized edge depends heavily on execution quality, spread, slippage, latency, and adverse selection.

Allowed decisions: limit vs market; post-only vs taker; price offset; cancel timing; entry urgency; skip toxic execution conditions.

Execution alpha is part of the strategy stack, not an afterthought.

### Family I — Beta-Residual Cross-Sectional Momentum

Hypothesis: After regressing out the common market factor (β·r_mkt), the residual idiosyncratic return persists over a multi-day horizon — a market-neutral momentum edge that raw-return ranking cannot capture because the common factor dominates raw cross-sectional dispersion (Family G's raw variant is dead on real data for exactly this reason).

Inputs: returns across the universe; rolling beta to the equal-weight universe factor; cumulative residual return over `signal_window`. All computed **inside the `CrossSectionalEngine`** from bars (no feature column / no rebuild).

Rules: dollar-neutral basket through the portfolio engine; maker rebalancing (turnover is the thin edge's tightest margin); pick window/cadence a priori for both-sides robustness, not the grid-max corner (overfitting). The `residual_reversion` mode is retained as a tested-negative control — the residual *trends*, it does not revert.

---

## 13. Strategy Candidate Lifecycle

Every strategy must move through this lifecycle.

### Stage 1 — Draft

No trading.

Required: hypothesis; expected edge source; data requirements; assumptions; failure modes; validation plan.

### Stage 2 — Research Candidate

Allowed: feature analysis; exploratory testing; synthetic tests; preliminary backtests.

Not allowed: paper trading; live trading.

### Stage 3 — Backtest Candidate

Required: clean implementation; unit tests; event-based backtest; cost model; slippage model; no look-ahead bias; no survivorship bias; universe version; data version; config version.

### Stage 4 — Validated Candidate

Required: walk-forward validation; out-of-sample testing; fee stress; slippage stress; parameter sensitivity; market-regime breakdown; symbol-level breakdown; long/short breakdown; worst-trade review; degradation analysis.

### Stage 5 — Paper Candidate

Allowed: paper trading; candidate logging; rejected trade analysis; paper vs backtest comparison.

### Stage 6 — Active Strategy

Allowed in live only after manual approval and all required gates pass.

Required: strategy report; paper report; risk report; execution report; monitoring dashboard; rollback plan.

### Stage 7 — Disabled

Must be disabled if: live behavior diverges from expectations; drawdown exceeds threshold; execution costs exceed assumptions; edge disappears; data quality fails; repeated risk violations occur.

---

## 14. Parameter Selection Rules

Parameters are hypotheses until validated.

Rules:

- Do not select parameters by maximum PnL.
- Do not select parameters by a single best backtest.
- Prefer robust parameter zones over isolated best values.
- Use walk-forward validation.
- Use out-of-sample validation.
- Use parameter sensitivity analysis.
- Reject parameters that only work in one symbol.
- Reject parameters that only work in one month.
- Reject parameters that only work in one market regime.
- Reject parameters that fail fee or slippage stress.
- Reject parameters where most profit comes from a few outlier trades.
- Prefer lower variance over higher unstable return.
- Prefer simpler parameters unless complexity is justified.
- Any parameter change requires a new config version.
- **Parameters are frozen only after walk-forward + locked hold-out (Section 16).** Any change → new config version; never change mid-session without halt + approval.

No parameter may change during a live session unless:

- the bot is halted;
- the change is reviewed;
- the config version is incremented;
- rollback is available;
- approval is logged.

---

## 15. Setup Quality Gate & Multi-Symbol Attribution

Regime match does not authorize a trade.

Regime match only allows setup evaluation.

Every candidate must pass a setup quality gate.

Default score components:

- regime alignment: 0–20;
- signal strength: 0–20;
- cross-signal confirmation: 0–15;
- expected move after costs: 0–15;
- execution quality: 0–15;
- risk/reward quality: 0–10;
- session/context quality: 0–5.

Trade may be approved only if:

- setup_quality_score is above configured threshold;
- no hard blocker is active;
- expected value is positive after fees and slippage;
- stop and invalidation are defined;
- order size is valid;
- symbol is active and tradable;
- strategy is enabled;
- universe version is valid;
- config version is active;
- data is fresh;
- risk manager approves.

Hard blockers:

- stale data;
- missing metadata;
- missing fees;
- missing funding schedule if funding is used;
- spread above threshold;
- slippage estimate above threshold;
- symbol halted or inactive;
- open position conflict;
- foreign order detected;
- daily loss limit reached;
- drawdown limit reached;
- strategy disabled;
- ML-only signal without deterministic candidate unless explicitly approved at later ML stage;
- model outside approved version;
- exchange reconciliation failure;
- data quality failure;
- config version not live-approved.

**Multi-symbol attribution (the bot decides where it acts):** signals are evaluated across the whole universe each cycle. When a trade is opened, the system records the full attribution in `decision_log`: which **symbol** fired, which **strategy/setup**, which **regime**, the signal features, expected edge & cost, the rejected alternatives (other symbols/setups considered and why they lost), `config_version`, and `model_version` if ML was involved. The system never opens a trade it cannot attribute this way.

---

## 16. Validation & Anti-Overfitting

- **Walk-forward** with broad history coverage; **out-of-sample** mandatory.
- **Locked hold-out:** the most recent segment is untouched during all tuning and is looked at **exactly once** at the end.
- **Kill-criteria declared up front** (before optimization): e.g. OOS expectancy > X R after costs in ≥K folds; otherwise the strategy is **shelved**, not tuned further. Validation exists to reject.
- **Multiple-testing control:** many setups × symbols × sides × parameter grids → deflated Sharpe / correction for number of hypotheses.
- **Effective sample size << trade count:** sequential and cross-symbol-correlated trades are not independent; compute effective N.
- **Fee/slippage stress** (e.g. ×2 fees, +50% slippage) — does the edge survive?
- **Look-ahead guard:** synthetic/shuffled data must yield ≈0 expectancy.
- **Sample minimums (configurable; below them = *inconclusive*, not successful):** 300+ candidate trades for preliminary conclusions; 100+ executed for strategy-level; 30+ per symbol; 30+ per regime; 30+ per side.

**Kill-criteria example (declare before any optimization):**
```yaml
kill_criteria:
  min_oos_expectancy_r: 0.15
  min_oos_profit_factor: 1.3
  max_oos_drawdown: 0.08
  min_folds_passed: 4
  min_executed_trades: 100
  min_per_symbol: 30
  min_per_regime: 30
  min_per_side: 30
```
If any criterion fails → **shelve the strategy**, do not tune further to force a pass.

**Implemented walk-forward gate (`src/backtest/walkforward.py`, `configs/backtest.yaml`).** The walk-forward asks two DISTINCT questions and judges them separately — applying one economic bar to both was found to commit a Type II error on thin-but-real edges:
- **Per-fold = STABILITY** (`fold_criterion: directional`, the default): a fold passes when the edge is PRESENT — `expectancy_r > 0`, enough trades, drawdown within the risk cap. The economic-magnitude bar is NOT applied per fold. (`fold_criterion: economic` restores the legacy full-kill-criteria-per-fold behaviour.)
- **Locked hold-out = ECONOMIC VIABILITY:** the most-recent untouched segment must clear the FULL kill-criteria (`min_oos_expectancy_r`, `min_oos_profit_factor`, `max_oos_drawdown`), evaluated exactly once.
- **Multiple-testing significance:** the deflated Sharpe (PSR over the fold trials) must clear `min_deflated_sharpe` (0.5 = "more likely than not a real edge net of multiple testing"; a genuinely edgeless strategy averages below 0.5 — this is the anti-luck guard that stops directional folds passing by chance).
- **Verdict = all three** (≥ `min_folds_passed` folds pass the fold test AND the hold-out clears the economic criteria AND the deflated Sharpe clears its floor). The change is monotonically looser for the fold COUNT (so nothing previously promoted regresses) but adds the deflated-Sharpe floor (net-new guard). Validated against controls: it promotes a thin-but-real edge while still rejecting a real no-edge strategy on all three counts. The actual operating thresholds live in `configs/backtest.yaml` (`min_oos_expectancy_r: 0.03`, `min_oos_profit_factor: 1.10`, etc.), not the illustrative example above.
- **Anti-overfit discipline observed in practice:** parameter values are chosen on the TRAIN folds, the hold-out is read once; changes that lift in-sample metrics but collapse the locked hold-out (e.g. basis funding-confirmation, an over-tight band cap) are rejected as overfits, not adopted.

---

## 17. Risk Management (capital-agnostic)

The risk manager has absolute authority.

It must approve every order.

**Per-trade sizing (deterministic in early versions):**
```
size = (equity × risk_pct) / |entry − stop_loss|
```
Leverage is the resulting notional/equity, **capped** by the envelope; if it would exceed the cap, reduce size or no-trade. If `size < min_qty/min_notional` for the symbol → no-trade (logged).

**Implemented per-strategy `risk_scale` (`src/backtest/risk.py`, `src/risk/manager.py`).** A per-strategy size scale (clamped to `(0, 1]`) multiplies the per-trade risk: `size = (equity × risk_pct × risk_scale) / |entry − stop|`. It can only scale DOWN, never above the account `risk_pct` (cannot loosen the control). Purpose: a real-but-hot edge whose drawdown exceeds the envelope at the account standard is sized down so its drawdown fits — `expectancy_r`/profit-factor are size-invariant, only the equity drawdown scales. The backtest `RiskSimulator` and the live/paper `RiskManager` apply it identically (Parity Rule).

**Implemented basket (cross-sectional) risk/position model — DISTINCT from the per-symbol path (`src/backtest/portfolio.py`, `src/live/basket.py`).** The `CrossSectionalEngine` does **not** use `RiskManager`: a dollar-neutral basket is sized as `gross = equity × portfolio_gross × risk_scale` split across legs, with `stop_frac` an **accounting R-unit only** — legs exit on the **rebalance cadence** (or session end), with **no protective stop between rebalances** (a stop would knife the hedge). This is correct for carry/factor (the directional variance is hedged, not stopped), but means the heat / beta / concurrency caps and per-position stops of the per-symbol path do NOT apply to baskets. Both paths honour the global **kill switch** (the basket loop via `_halt_check`, which flattens + persists on engage). **Cross-process gap (known):** the per-symbol loop and each basket run as SEPARATE processes with SEPARATE equity pools (each off the full `initial_equity`), so there is **no portfolio-level aggregation across strategies** — combined exposure is not netted or capped, and N parallel processes ≈ N× the intended account exposure. Acceptable for isolated PAPER measurement of each edge; a shared capital allocator / aggregate-exposure layer is REQUIRED before multiple strategies trade one real account.

**Portfolio-level (multi-symbol, per-symbol path):** account/symbol/strategy/correlation/market-wide exposure; **portfolio heat cap**; **net beta-to-BTC cap**; max positions total, per symbol (1), per regime; margin & liquidation-distance checks.

**Circuit breakers (per-symbol and portfolio):** daily-loss limit; weekly limit if configured; max-drawdown; N consecutive max-losses → cooldown; abnormal-slippage cooldown; funding circuit; kill-switch status.

Kelly forbidden until explicitly approved after long-term validation. No martingale; no averaging-down (unless a separately-risk-managed, validated inventory strategy). 

**Reconciliation & order ownership:** on every start, reconcile all positions/orders with the exchange; mismatch → halt + alert. Every order carries `ORDER_CLIENT_ID_PREFIX` (+ `BOT_INSTANCE_ID`, `STRATEGY_VERSION`, `CONFIG_VERSION`); any position/order not confidently attributable to this instance → halt + alert; never touch others' orders except in explicit emergency-close mode.

Required checks:

- account-level exposure;
- symbol-level exposure;
- strategy-level exposure;
- correlation exposure;
- market-wide exposure;
- leverage limit;
- position size;
- stop distance;
- liquidation distance;
- margin availability;
- daily loss limit;
- weekly loss limit if configured;
- max drawdown limit;
- max trades per strategy;
- max trades per symbol;
- max trades per regime;
- cooldown after loss;
- cooldown after abnormal slippage;
- kill switch status;
- open order conflicts;
- existing position conflicts;
- unknown order conflicts.

Rules:

- No strategy may bypass risk.
- No ML model may bypass risk.
- No online learner may bypass risk.
- No RL policy may bypass risk.
- Risk sizing must be deterministic until specifically approved otherwise.
- Kelly sizing is forbidden until explicitly approved after long-term validation.
- Martingale is forbidden.
- Uncontrolled averaging down is forbidden.

---

## 18. Execution Rules

Execution is part of edge.

The execution engine must support:

- market orders;
- limit orders;
- post-only orders if supported;
- reduce-only orders;
- stop orders;
- take-profit orders;
- cancel/replace;
- order status reconciliation;
- partial fill handling;
- slippage measurement;
- adverse selection measurement.

Default execution preference:

- maker-first when urgency is low;
- taker only when expected edge justifies cost;
- no market order during toxic spread;
- no order if slippage estimate exceeds threshold;
- no order if exchange latency is abnormal;
- cancel stale passive orders;
- revalidate signal before execution.

Stops/TP are **exchange-resident**; trend trailing uses **exchange-native trailing** so it survives bot downtime.

**Implemented execution / backtest↔live parity (`src/execution/order.py`, `src/execution/live_venue.py`, `src/live/loop.py`).** The strategy `Signal` carries the execution intent (maker, limit offset, trail, time-stop, risk_scale) onto the `Candidate`, and BOTH the backtest engine and the live `OrderBuilder` honour it identically:
- **Maker entry:** a `POST_ONLY` passive limit posted `limit_offset_frac` inside the reference (rests at price); taker is a market order.
- **Bracket:** the exchange-resident stop + a reachable take-profit + an exchange-native trailing stop coexist on the position simultaneously (Bybit holds all three) — whichever triggers first exits, matching the backtest's stop/TP/trail OR-of-exits. The trailing offset comes from the strategy's own `trail_frac` (floored at the initial stop).
- **Time-stop:** `hold_bars` has no exchange-native equivalent, so the live loop tracks each owned position's entry time + horizon and flattens it once aged (`Venue.close_position`) — the exchange legs keep protecting it meanwhile, so this is an optimization exit, not protection.
This closes the backtest↔live parity gap: a gate-validated edge executes the same live.

Every fill must store:

- expected price;
- actual price;
- slippage;
- fee;
- maker/taker flag;
- latency;
- order type;
- signal age;
- spread at order time;
- post-fill adverse movement.

**Execution quality metrics (tracked per symbol, regime, session):**
- avg slippage vs estimate;
- maker ratio;
- fill rate;
- adverse selection (price movement after fill);
- latency distribution;
- toxic execution frequency.

---

## 19. Event-Based Backtesting

Backtesting must be event-based.

Vectorized backtests may be used for exploration only, not validation.

The backtest engine must use the same feature pipeline as paper/live.

Backtest must include:

- realistic fees;
- realistic slippage;
- funding if relevant;
- spread assumptions;
- exchange metadata;
- order size constraints;
- tick size;
- lot size;
- minimum notional;
- order latency assumptions;
- partial fill assumptions if relevant;
- rejected trades;
- execution constraints;
- risk constraints;
- universe selection as of test time.

Backtest must prevent:

- look-ahead bias;
- future universe leakage;
- survivorship bias;
- using unavailable features;
- using close price before candle close;
- using funding data before it was known;
- using liquidation data with wrong timestamp;
- using mark/index data misaligned with trade data.

Backtest output must include:

- total return;
- net PnL;
- expectancy;
- profit factor;
- max drawdown;
- trade count;
- symbol breakdown;
- strategy breakdown;
- regime breakdown;
- session breakdown;
- long/short breakdown;
- cost breakdown;
- slippage breakdown;
- funding breakdown;
- rejected candidate breakdown;
- worst trades;
- stability metrics;
- **per-trade MAE/MFE excursion** (max adverse / max favorable, in R) with averages split by win/loss — the trade-quality lens for diagnosing whether a problem is the entry (no move to capture) or the exit (move captured then given back).

**Implemented engine behaviour (`src/backtest/engine.py`).** The engine walks one epoch-time grid shared by all symbols (symbols looked up by timestamp, robust to mid-window listings). It fills per the strategy's execution intent — **maker** passive-limit fills (maker fee, zero slippage, no-fill if the bar doesn't trade through) vs taker market fills (taker fee + adverse slippage) — and exits via the OR of: exchange-style stop / reachable take-profit / ATR trailing stop / per-bar **`manage()`** strategy hook (consulted after stop/TP, before the time-stop) / **time-stop** / end-of-data. The same feature pipeline, sizing (`risk_scale`), and exit geometry drive the live path (Parity Rule, Section 10), so backtest and live agree.

---
## 20. ML Layer

ML is **not** the first source of edge; it is allowed only on top of a deterministic baseline, and it starts in **shadow**.

**Best first use: meta-labeling.** The deterministic system proposes a candidate; ML estimates whether to take or skip (and, later, size within `[0, risk_cap]`). ML answers "should we," never "which direction."

**Staged use cases:**
1. Offline research (no paper/live decisions).
2. **Shadow** (predictions/labels/recommendations logged; behavior unchanged).
3. **Trade filter** (may block deterministic candidates; may not create trades, increase risk, or override no-trade rules).
4. **Strategy selector** (rank/recommend permission/reduce activity; may not enable unvalidated or trade disabled strategies).
5. **Execution optimizer** (order type / passive-aggressive / cancel timing / toxic-execution flagging; may not override risk).
6. **Conservative risk reduction** (only after long-term validation; choose among predefined buckets `0.25× / 0.5× / 1.0×` base risk; never increase beyond the cap; no Kelly, no martingale).

**Model requirements:** versioned with `model_id`, data/feature versions, label & target definitions, train/validation/OOS periods, performance + calibration + explainability reports, known failure modes, promotion status. **Allowed early classes:** logistic regression, random forest, gradient boosting / XGBoost / LightGBM / CatBoost, calibrated classifiers, simple clustering, GMM, HMM (research). Deep learning only if simpler models fail and complexity is justified.

**Promotion gates (shadow → live influence):** improves expectancy net of costs; preserves/raises profit factor; preserves/reduces max DD; does **not** remove most of the best trades; reduces worst-trade frequency/tail risk; stable across folds, symbols, regimes, OOS; calibrated; explainable; no data/target/availability leakage; manually reviewed. If it fails to beat the deterministic baseline in shadow, it stays shadow-only.

### ML Stages

#### ML Stage 0 — Forbidden from Decisions
No ML affects trading decisions.

#### ML Stage 1 — Offline Research
Allowed: feature engineering; clustering; exploratory modeling; meta-label experiments; regime analysis; execution quality analysis.
Not allowed: paper decisions; live decisions.

#### ML Stage 2 — Shadow Mode
Allowed: shadow predictions; shadow trade approval/rejection; shadow regime labels; shadow strategy ranking; shadow symbol ranking; shadow execution recommendation.
Not allowed: changing actual bot behavior.

#### ML Stage 3 — Recommendation Mode
ML recommendations are shown in dashboard and reports.
Allowed: recommend skip/take; recommend preferred strategy; recommend preferred symbol; recommend execution route; recommend risk bucket reduction.
Not allowed: automatic behavior changes; risk increases; order placement.

#### ML Stage 4 — Constrained Live Filter
ML may block deterministic candidate trades.
Allowed: block weak candidates; reduce strategy activity; reduce risk bucket; block toxic execution.
Not allowed: create trades; increase risk; trade disabled strategies; override hard blockers.

#### ML Stage 5 — Strategy and Symbol Selector
ML may rank strategies and symbols among deterministic candidates.
Allowed: choose among validated candidates; rank active symbols; reduce or block weak symbols; prefer strategies with better current context.
Not allowed: enable unvalidated strategies; trade outside active universe; bypass risk manager.

#### ML Stage 6 — Conservative Risk Adjustment
Allowed only after long-term validation.
Allowed risk buckets: 0x; 0.25x; 0.5x; 1.0x base risk.
ML may reduce risk before it may increase risk.
ML may not exceed configured base risk without separate approval.

#### ML Stage 7 — Full ML Live Control
Allowed only after extensive validation and manual approval.
Full ML live control means ML may: approve candidates; rank symbols; rank strategies; choose allowed risk bucket; choose execution preference.
Still forbidden: bypass risk manager; bypass hard blockers; trade inactive symbols; trade without explainability; change model online without approved process; use unversioned features or models.

---

## 21. Online Learning & Reinforcement Learning (gated path to live)

This is the **highest-risk component.** It is permitted because the operator has explicitly chosen it, **but only inside the Immutable Risk Envelope (Section 2.2), which it can never modify, and only through the staged rollout below.** The note here is deliberate: small/non-stationary samples make naive online adaptation overfit and amplify losses — every guardrail below exists to contain that.

### 21.1 Eligibility (before *any* adaptation is allowed)
- A deterministic baseline is live (or paper-validated) and profitable net of costs.
- A large, configurable **minimum sample** of logged signals/trades across regimes and symbols (online learning trains on *every logged signal*, not only executed trades).
- Leakage-safe validation in place: **purged + embargoed CV**, locked hold-out, kill-criteria.
- A **frozen fallback policy** exists and is tested.

### 21.2 What it may and may not touch
- **May** (within pre-declared bounded ranges): weighting among already-validated strategies; bet size within `[0, risk_cap]`; skip/allow filtering; execution routing; bounded parameter values.
- **May never:** create new strategies, enable unvalidated ones, widen/disable any envelope constant, change stop placement beyond bounds, raise leverage/heat/beta caps, trade in no-trade regimes, or act on unavailable features.

### 21.3 Staged rollout (exactly: shadow → recommend → live)
1. **Shadow (observe-only).** The online/RL policy runs continuously, logs the decisions it *would* make to `learner_log`, and is scored offline against the live deterministic system and against its own prior projections. No effect on real trades. Promotion requires beating the baseline on walk-forward **and** locked hold-out.
2. **Recommend (human-in-the-loop).** The policy's proposed adjustments surface on the dashboard as **recommendations**; nothing changes until a human approves. Approved changes are versioned and git-committed with evidence. This stage must accumulate a track record of approved-and-correct recommendations.
3. **Bounded live (autonomous within the envelope).** Only after stages 1–2 pass, the policy may act autonomously **inside bounded ranges**, with:
   - **Bounded updates:** max change per update and a max change-rate; no large jumps.
   - **Learner circuit breaker / auto-rollback:** if live decisions underperform the policy's shadow projection by a configured margin over N trades, or any envelope breaker fires, **freeze to the last-good version and revert to the frozen fallback policy**, then alert.
   - **Separate learner kill switch:** disables adaptation while continuing to trade the last-approved frozen policy (independent of the trading kill switch).
   - **Continuous re-validation:** periodic offline retrain + OOS check; models decay — a frozen learner is never trusted indefinitely.

### 21.4 RL specifics
RL remains research/shadow until it has passed 21.1–21.3 like any other learner. Reward must be **fee/slippage/funding-net and risk-adjusted** (not raw PnL), with explicit penalties for drawdown, tail loss, and envelope-proximity. The action space is constrained to the bounded set in 21.6; the RL policy cannot emit any action outside it by construction.

### 21.5 Module layout (`src/adaptation/`)

```
src/adaptation/
  __init__.py
  action_space.py     # BoundedAction schema + validators; clamps to declared ranges
  envelope_guard.py   # immutable-envelope enforcement; rejects any out-of-box action
  policy_base.py      # Policy interface (decide / update / snapshot / load)
  policies/
    online_logreg.py  # incremental classifier (e.g. SGD) for meta-filter weighting
    bandit.py         # contextual bandit over validated strategies (Gaussian TS)
    rl_policy.py      # RL agent (research/shadow first); bounded action head
  scorer.py           # shadow scoring: realized vs projected; promotion metrics
  controller.py       # state machine (SHADOW→RECOMMEND→LIVE), update gating
  rollback.py         # learner circuit breaker + revert-to-fallback
  versioning.py       # learner_version, frozen snapshots, git-commit helper
  store.py            # learner_log + recommendation + snapshot persistence
```

The learner is a **subordinate advisor**: it emits a `BoundedAction`; the Risk Layer (Section 17) still independently approves the resulting order. The learner never calls execution or exchange APIs.

### 21.6 Bounded action space (the only things a learner may emit)

Every action passes through `action_space.validate()` then `envelope_guard.enforce()` before use. Anything outside the declared bounds is **clamped or rejected** (configurable per field; reject is the safe default) and logged.

```python
class BoundedAction(BaseModel):
    # 1) strategy weighting — among ALREADY-VALIDATED, ENABLED strategies only
    strategy_weights: dict[str, float]      # keys ⊆ active strategies; each ∈ [w_min, w_max]; renormalized

    # 2) bet-size multiplier — bucketed, never continuous-unbounded
    size_bucket: Literal[0.0, 0.25, 0.5, 1.0]   # × base risk_pct; 0.0 = skip. Never > 1.0

    # 3) trade filter — may only BLOCK a deterministic candidate, never create one
    take: bool

    # 4) execution routing — cosmetic to risk; cannot change price/size/stop
    exec_style: Literal["maker", "taker", "passive_then_taker"]

    # 5) bounded parameter nudges — only params explicitly registered as tunable,
    #    each constrained to a pre-declared [lo, hi] zone validated in Section 16
    param_nudges: dict[str, float]          # key ∈ registered_tunables; value ∈ its [lo, hi]

    # provenance (mandatory)
    learner_id: str
    learner_version: str
    mode: Literal["SHADOW", "RECOMMEND", "LIVE_BOUNDED"]
    rationale: str                          # human-readable; stored for explainability
```

**Hard invariants enforced by `envelope_guard` (cannot be overridden by config):**
- `size_bucket` ≤ 1.0 and the resulting trade still passes the per-trade `risk_pct` cap, leverage cap, portfolio heat cap, and beta cap.
- `param_nudges` keys must be in the `registered_tunables` allow-list; values clamped to each param's validated zone. Stops, leverage, heat, beta, breaker thresholds, and no-trade regimes are **never** registered as tunable.
- `take=True` cannot resurrect a candidate the deterministic system did not produce, nor one a hard blocker rejected.
- Any action referencing a disabled/unvalidated strategy → reject + alert.

### 21.7 Learner state machine, update gating & rollback

```
        eligibility(21.1) pass
INIT ────────────────────────► SHADOW
                                  │  scorer beats baseline on WF + locked hold-out
                                  ▼
                               RECOMMEND ──── human approves track record ───► LIVE_BOUNDED
                                  ▲                                               │
                                  │   learner circuit breaker / kill switch       │
                                  └───────────────── FROZEN ◄─────────────────────┘
FROZEN = trading continues on last-approved frozen policy; adaptation disabled.
```

**Update gating (LIVE_BOUNDED):**
- Online updates apply only at safe points (no open position for the affected symbol/param).
- `|Δ| ≤ max_change_per_update`; cumulative drift over a window ≤ `max_change_rate`.
- Every accepted update writes a new `learner_version` snapshot (immutable) so any version is restorable.

**Rollback triggers (any one → revert to `frozen_fallback_policy`, set FROZEN, alert):**
1. Realized performance underperforms the policy's own shadow projection by ≥ `rollback_margin` over the last `rollback_window` decisions.
2. Any envelope breaker fires (daily loss, drawdown, heat, beta) while the learner is active.
3. Live-vs-shadow decision divergence exceeds `max_divergence` (policy acting differently than it claimed it would).
4. Data-quality `R8`/toxic-execution `R7` regime, or reconciliation failure.
5. Manual **learner kill switch** (separate from the trading kill switch).

`rollback.revert()` is atomic: load `frozen_fallback_policy`, cancel only pending learner-influenced *new* orders (never touch open positions' exchange-side stops), set state FROZEN, emit `learner_rollback` alert, write the event to `learner_log`. Recovery from FROZEN back to LIVE_BOUNDED is **manual only**, after review.

### 21.8 `learner_log` & recommendation schema

```python
class LearnerLogEntry(BaseModel):
    ts: datetime
    learner_id: str
    learner_version: str
    mode: Literal["SHADOW", "RECOMMEND", "LIVE_BOUNDED"]
    symbol: str | None
    context_features: dict          # inputs at decision time (reproducible)
    proposed_action: BoundedAction
    projected_outcome: float        # policy's own expectation (for divergence/scoring)
    realized_outcome: float | None  # filled in post-trade
    applied: bool                   # did it affect a real order?
    clamped_fields: list[str]       # what envelope_guard adjusted
    rollback_event: str | None      # set on FROZEN transitions
    config_version: str

class Recommendation(BaseModel):       # RECOMMEND mode → dashboard
    id: str
    created_ts: datetime
    learner_version: str
    change_set: dict                  # proposed weights / nudges (bounded)
    evidence: dict                    # WF + hold-out deltas, fee-adjusted, per-symbol/regime
    status: Literal["pending", "approved", "rejected"]
    approved_by: str | None
    git_commit: str | None            # set when approved + committed
```

### 21.9 Config keys (envelope + learner)

```yaml
risk_envelope:           # IMMUTABLE at runtime; change only via halt + approval + git
  max_leverage: <int>
  max_risk_pct_per_trade: <float>
  portfolio_heat_cap: <float>
  net_beta_btc_cap: <float>
  daily_loss_limit: <float>
  max_drawdown_limit: <float>

adaptation:
  enabled: false                 # master off-switch
  mode: "SHADOW"                 # SHADOW | RECOMMEND | LIVE_BOUNDED (promotion is manual)
  min_samples_to_start: <int>    # large; per 21.1
  registered_tunables:           # allow-list; each with a validated zone
    <param_name>: { lo: <float>, hi: <float> }
  bounds:
    strategy_weight: { w_min: 0.0, w_max: <float> }
    size_buckets: [0.0, 0.25, 0.5, 1.0]
    max_change_per_update: <float>
    max_change_rate: <float>     # per rolling window
  rollback:
    rollback_window: <int>
    rollback_margin: <float>     # realized vs projected shortfall
    max_divergence: <float>
    auto_freeze_on_breaker: true # cannot be set false
  retrain:
    schedule: "<cron>"           # periodic offline retrain + OOS revalidation
  frozen_fallback_policy: "<path-or-version-id>"
```

`risk_envelope` and the `rollback.auto_freeze_on_breaker: true` invariant are enforced in code regardless of config edits; the config cannot disable them.

### 21.10 Minimal interfaces

```python
class Policy(Protocol):
    def decide(self, ctx: Context) -> BoundedAction: ...
    def update(self, ctx: Context, action: BoundedAction, outcome: Outcome) -> None: ...  # no-op in SHADOW/RECOMMEND
    def snapshot(self) -> bytes: ...
    def load(self, blob: bytes) -> None: ...

# Decision path (live): deterministic candidate → learner advises → guard → RISK approves → execute
action = policy.decide(ctx)
action = envelope_guard.enforce(action_space.validate(action))   # clamp/reject + log
if controller.mode == "LIVE_BOUNDED" and action.take:
    order = build_order(candidate, action)
    if risk_manager.approve(order):        # Risk Layer still has final, independent veto
        execution.submit(order)
store.write_learner_log(ctx, action, applied=(controller.mode == "LIVE_BOUNDED"))
```

---

## 22. Supervised Improvement Loop

1. The bot logs every signal, regime, decision, gate-rejection, and outcome.
2. `backtest/optimizer.py` runs **offline, on demand**: walk-forward, proposing parameter changes with in/out-of-sample, fee-adjusted evidence.
3. Proposals appear on the dashboard as recommendations; **nothing changes without human approval.**
4. Approved changes → versioned config + git commit (evidence in message) + tag.

Rules-based regime switching and the bounded learner of Section 21 are the only autonomous adaptations; both are subordinate to the envelope.

---

## 23. Data Quality Rules

Data quality is a trading-safety issue. Detect: missing/duplicate/out-of-order candles; stale websocket; inconsistent mark/index/perp timestamps; funding timestamp mismatch; missing metadata; symbol-status changes; abnormal spreads/gaps; corrupted records; clock drift (run NTP). On critical failure → halt. Generate a data-validation report before every research, paper, or live run.

---

## 24. Persistence, Observability, Explainability

Single database is the source of truth for trades, orders, equity snapshots, regime states, funding, OI, errors, and the logs below. Back it up regularly with a tested restore (the logs are the platform's whole value; losing them resets validation).

### Required Logs

- **`decision_log`:** per signal — chosen action, **rejected alternatives with reasons** (which gate/symbol/setup lost), features at decision time, expected edge/cost, `config_version`, `model_version`, `universe_version`. Writes are **asynchronous and never block execution**.
- **`shadow_log`:** ML/experimental recommendations for offline comparison.
- **`learner_log`:** online/RL would-be decisions, bounds, and rollback events.
- **`audit_log`:** immutable record of all system actions, approvals, config changes, gate runs, manual overrides.

### Trade Explainability Schema

Every live trade must be explainable as:

```python
class TradeExplainability(BaseModel):
    trade_id: str
    timestamp: datetime
    symbol: str
    strategy_id: str
    setup_type: str
    regime: str
    signal_features: dict
    expected_edge_after_costs: float
    expected_fees: float
    expected_slippage: float
    expected_funding_impact: float | None
    stop_price: float
    invalidation_conditions: list[str]
    execution_route: str
    risk_approved: bool
    risk_reason: str
    model_version: str | None
    learner_version: str | None
    config_version: str
    universe_version: str
    why_selected: str           # why this symbol over alternatives
    why_rejected_others: list[dict]  # symbol, reason
```

If the system cannot populate this schema for a trade, the trade is not taken.

---

## 25. Dashboard & Monitoring

A FastAPI app (API + web UI) behind a TLS reverse proxy; the rest of the stack stays on an internal network.

The dashboard is not only a statistics viewer. It is the operational control center for: data coverage; universe status; jobs; gates; remediation actions; reports; approvals; paper/live monitoring; ML/RL maturity; live readiness.

### Statistics Views (required)

- An **aggregate / portfolio** view, **and** a separate **per-symbol** view — selectable per symbol.
- A **time-range selector** for every stats view: day / week / month / all-time / **custom range**. All metrics (equity, PnL realized/unrealized, drawdown, win rate, expectancy, profit factor, fees, funding paid/received, slippage, trade counts) recompute for the chosen symbol scope and period.
- Further breakdowns: by strategy, regime, session; execution quality; rejected trades; ML shadow performance; learner status; risk-limit usage (heat, beta, exposure); data freshness; exchange health.

### Dashboard Pages

1. Overview
2. Data Coverage
3. Universe
4. Jobs
5. Gates
6. Remediation Actions
7. Backtests
8. Paper Trading
9. Live Trading
10. General Statistics
11. Per-Symbol Statistics
12. Strategy Analytics
13. Regime Analytics
14. Session Analytics
15. Execution Quality
16. Risk
17. ML Shadow
18. Online Learning
19. RL
20. Reports
21. Approvals
22. System Health
23. Settings

Every statistics page must include a time period selector.

Required time filters:

- today;
- yesterday;
- last 7 days;
- last 30 days;
- current month;
- previous month;
- custom date range;
- by backtest run;
- by paper session;
- by live session;
- by config version;
- by universe version;
- by strategy version;
- by model version.

### Background Gate Runner (required)

Every gate in the catalogue (**Appendix A**) can be **launched as a background job from the dashboard** (single gate, a group, or "run all"). Jobs are idempotent, show live progress + streamed logs, never block trading or the UI, and persist a reviewable report.

Each gate run returns a structured `GateResult`. The dashboard renders it and, **for any criterion that is not PASS, displays the failure reason, the concrete ordered remediation action items, and a one-click "re-run this gate" button**:

```python
class CriterionResult(BaseModel):
    id: str                       # e.g. "LIVE-3", "VAL-holdout"
    title: str
    status: Literal["PASS", "FAIL", "BLOCKED", "NOT_RUN", "WAIVED"]
    measured: dict                # actual values observed
    threshold: dict               # required values
    failure_reason: str | None
    remediation_steps: list[str]  # concrete, ordered action items (from configs/gates.yaml)
    rerun_job: str                # job id the "re-run" button triggers
    auto_remediation: bool        # can system fix automatically?
    manual_review_required: bool  # must human review?
    last_run_ts: datetime | None

class GateResult(BaseModel):
    gate_id: str
    overall: Literal["PASS", "FAIL", "BLOCKED", "NOT_RUN", "EXPIRED", "NEEDS_APPROVAL"]
    criteria: list[CriterionResult]
    report_path: str
    affected_downstream: list[str]  # which gates are blocked if this fails
    next_action: str              # human-readable "what to do now"
```

### Gate Status Widget (every page)

A persistent widget shows:
- Total gates: X passed / Y failed / Z blocked / W not run
- Live readiness score: % of critical gates passed
- Next critical action: the single most important thing to do now
- Click-through to full "Road to Live" view

### "Road to Live" View (required)

A single dashboard screen lists every gate with its current status, the blocking criteria, and the **next action** for each — so the operator always sees exactly what remains to reach live and how to clear it, and can re-run any gate after fixing it.

Gates, thresholds, and remediation text are declared once in `configs/gates.yaml` (a single source shared by the runner, the dashboard, and the tests). The intended loop is explicit: **run gate → read remediation → fix → re-run → green → advance**, repeated until all gates pass and live is enabled.

**Live Readiness Score:**
```
score = (critical_gates_passed / total_critical_gates) × 100
```
Critical gates are those marked `blocks_live: true` in `configs/gates.yaml`. The dashboard shows this score prominently. Live activation is enabled only at 100%.

### Remediation Workflow

When a gate fails:
1. Dashboard shows `GateResult` with all `CriterionResult` details.
2. Each failed criterion shows ordered `remediation_steps`.
3. Dashboard offers buttons: **Run Repair Job**, **Re-run Gate**, **View Logs**, **View Report**, **Create Remediation Task**, **Mark Manual Check Complete** (where applicable).
4. Remediation task is created with: task_id, gate_id, criterion_id, assigned_to, created_ts, status, action_items, evidence_links, completed_ts.
5. When remediation is complete, operator clicks **Re-run Gate**.
6. If gate passes, downstream blocked gates are automatically marked for re-run.

### Dashboard Actions

Allowed actions:

- start background gate job;
- start data backfill;
- refresh universe;
- run backtest;
- run validation;
- run ML training;
- run RL simulation;
- generate report;
- stage config;
- compare config versions;
- approve or reject strategy promotion;
- approve or reject model promotion;
- approve or reject online learner promotion;
- approve or reject RL promotion;
- activate paper mode;
- request live readiness review;
- approve live activation;
- trigger kill switch;
- pause bot;
- resume bot after review.

Dangerous actions must require confirmation and be logged.

### Alerts

Required alert channels:

- dashboard alert center;
- Telegram or equivalent push channel;
- email optional.

Required alerts:

- service unhealthy;
- exchange disconnected;
- websocket stale;
- data gap detected;
- job failed;
- gate failed;
- gate expired;
- live activation requested;
- live activation approved/rejected;
- live engine started;
- live engine stopped;
- kill switch triggered;
- order failed;
- stop placement failed;
- unknown order detected;
- position mismatch;
- abnormal slippage;
- drawdown limit reached;
- daily loss reached;
- model/config mismatch;
- backup failed;
- restore test failed;
- learner rollback;
- remediation task overdue.

Every alert must have:

- severity (critical / warning / info);
- timestamp;
- component;
- environment;
- run/session id if applicable;
- recommended action;
- dashboard link if available;
- escalation path (e.g., if unacknowledged in 15 min → escalate).

---

## 26. Paper Trading

Paper trading has two phases.

### Phase A — Technical Paper Validation

Purpose: Validate infrastructure and execution simulation.

Required: data ingestion works; universe manager works; candidate generation works; risk approval/rejection works; execution simulator works; dashboard works; alerts work; kill switch works; reconciliation works; decision logs are complete.

Phase A does not prove edge.

### Phase B — Strategy Paper Validation

Purpose: Validate strategy behavior across symbols, regimes, sessions, and time.

Required: sufficient candidate trades; sufficient executed paper trades; strategy reports; symbol reports; regime reports; session reports; execution reports; risk reports; rejected candidate analysis; paper vs backtest comparison; ML shadow comparison if ML exists; manual review.

No strategy may move live from Phase A alone.

---

## 27. Live Deployment and Activation

Deployment means the system is technically ready.

Deployment does not mean live activation.

### Pre-Live Gate Checklist (all must PASS)

Before live activation, the following gates must all be `PASS`:

| # | Gate | Section | Blocks Live? |
|---|------|---------|-------------|
| 1 | DATA-COV — Data Coverage | Appendix A | Yes |
| 2 | DQ — Data Quality | Appendix A | Yes |
| 3 | UNIV — Universe Validity | Appendix A | Yes |
| 4 | META — Exchange Metadata | Appendix A | Yes |
| 5 | FEAT — Feature Reproducibility | Appendix A | Yes |
| 6 | BT — Backtest | Appendix A | Yes |
| 7 | WF — Walk-Forward | Appendix A | Yes |
| 8 | FEE — Fee Stress | Appendix A | Yes |
| 9 | SLIP — Slippage Stress | Appendix A | Yes |
| 10 | SETUP — Setup Quality | Appendix A | Yes |
| 11 | RISK — Risk Policy | Appendix A | Yes |
| 12 | EXEC — Execution | Appendix A | Yes |
| 13 | PAPER-A — Technical Paper | Appendix A | Yes |
| 14 | PAPER-B — Strategy Paper | Appendix A | Yes |
| 15 | ML-PROMO — ML Promotion | Appendix A | Yes (if ML active) |
| 16 | LEARN-PROMO-S — Learner Shadow | Appendix A | Yes (if learner active) |
| 17 | LEARN-PROMO-L — Learner Live | Appendix A | Yes (if learner active) |
| 18 | SEC — Security | Appendix A | Yes |
| 19 | DEPLOY — Deployment | Appendix A | Yes |
| 20 | BACKUP — Backup & Restore | Appendix A | Yes |
| 21 | MON — Monitoring | Appendix A | Yes |
| 22 | KILL — Kill Switch | Appendix A | Yes |
| 23 | ORDER-OWN — Order Ownership | Appendix A | Yes |
| 24 | CONFIG-FREEZE — Config Freeze | Appendix A | Yes |
| 25 | LIVE — Live Gate | Appendix A | Yes |

Additionally required:
- config frozen;
- strategy versions frozen;
- model versions frozen if used;
- RL policy version frozen if used;
- order ownership configured;
- kill switch tested;
- reconciliation tested;
- dashboard available;
- alerting available;
- rollback plan ready;
- live activation manually approved.

Live activation must be logged.

The dashboard may show live readiness and launch required gate jobs, but it must not bypass approval.

Goal orientation:

- Failed gates are not dead ends.
- Failed gates generate action items.
- Action items are executed through dashboard jobs or code/config changes.
- The intended path is to resolve failures, re-run gates, and progress toward live readiness.
- Live is allowed only after evidence shows the system is ready.

### Live Activation Request Workflow

1. Operator clicks "Request Live Activation" on dashboard.
2. System runs `gate:live` (all criteria).
3. If any criterion fails, dashboard shows remediation steps; request is blocked.
4. If all criteria pass, system creates `LiveActivationRequest` record:
   ```python
   class LiveActivationRequest(BaseModel):
       request_id: str
       requested_by: str
       requested_at: datetime
       gate_results: list[GateResult]
       config_version: str
       strategy_versions: list[str]
       model_version: str | None
       learner_version: str | None
       risk_policy_version: str
       execution_policy_version: str
       status: Literal["pending", "approved", "rejected"]
       approved_by: str | None
       approved_at: datetime | None
       rejection_reason: str | None
   ```
5. Second operator (or designated approver) reviews and approves/rejects.
6. On approval: system logs activation, starts live engine, sends alerts.
7. On rejection: system logs rejection reason, returns to paper mode.

---

## 28. Review Process

Every trading day or session must produce a review.

Review must separate:

- valid losses;
- invalid losses;
- strategy errors;
- execution errors;
- data errors;
- risk errors;
- ML errors;
- online learning errors;
- RL policy errors;
- missed opportunities;
- rejected trades that would have worked;
- taken trades that should have been blocked.

The system must distinguish:

- bad outcome from good process;
- good outcome from bad process;
- edge degradation;
- execution degradation;
- market regime mismatch;
- overfitting risk.

Never judge by PnL alone; never confuse luck with edge.

### Daily Review Job

A scheduled job (`run_daily_review`) produces:
- Session summary (trades, PnL, costs, drawdown);
- Valid vs invalid loss classification;
- Strategy performance vs expectation;
- Execution quality report;
- Rejected trade analysis;
- Gate status changes;
- Recommended actions for next session.

---

## 29. Agent Roles

### Quant Research Agent

Responsible for: hypothesis design; feature research; strategy research; statistical testing; backtest analysis; robustness testing; strategy reports.

Must not: judge by PnL alone; ignore costs; ignore sample size; optimize on one period; promote strategies without validation.

### Trading Systems Engineer Agent

Responsible for: exchange adapter; data platform; universe manager; backtest engine; risk manager; execution engine; dashboard; job orchestration; deployment safety.

Must not: allow strategies to bypass risk; create unsafe live defaults; hardcode exchange assumptions outside adapter.

### ML Engineer Agent

Responsible for: feature pipelines; labels; model training; leakage prevention; shadow mode; model reports; calibration; model registry; ML promotion process.

Must not: deploy ML directly to live control; use unavailable features; optimize only for accuracy; ignore financial metrics.

### Adaptation / RL Agent

Responsible for: online/RL policy design; bounded action spaces; reward shaping (risk-adjusted, cost-net); shadow scoring; rollback logic.

Must never: design an action that can touch the envelope; keep the frozen fallback current.

### Risk Manager Agent

Responsible for: risk rules; exposure/heat/beta limits; drawdown controls; halt conditions; approval logic; post-trade risk review.

Must not: raise risk on unvalidated ML/learners; allow positions without invalidation; allow unknown orders/positions.

### Dashboard / UX Agent

Responsible for: general dashboard; per-symbol dashboard; time-period selectors; gate panels; remediation action items; background job controls; approvals; audit logs.

Must not: allow unsafe bypasses; hide failed gates; show failed gates without next steps; perform dangerous actions without confirmation.

### Infrastructure Agent

Responsible for: Docker Compose services; database migrations; Redis queues; data lake; backups; health checks; monitoring stack; deployment topology; security hardening.

Must not: allow live trading without backups; allow live trading without health checks; allow live trading without alerting; allow live trading without order ownership checks.

### Review Agent

Responsible for: daily review; strategy degradation analysis; execution review; rejected-trade analysis; live-vs-backtest comparison; process quality review.

Must not: judge by PnL alone; ignore invalid trades with positive outcomes.

---

## 30. Forbidden Work

Do not implement: live trading before paper validation; ML-controlled live before shadow validation; **any online/RL live influence before its shadow → recommend → bounded-live gates pass, or any learner action that modifies the Immutable Risk Envelope**; Kelly sizing in early versions; martingale; grid averaging-down; HFT claims without infrastructure; order-book strategy without historical order-book data; liquidation strategy without timestamp-safe data; funding-only strategy; strategy selection by best backtest PnL; **unbounded** parameter auto-update during live; hidden config changes; exchange calls inside strategy logic; notebooks as production code; optimistic backtests without realistic costs.

---

## 31. Development Standards & Project Structure

Code must be: modular; typed; tested; reproducible; observable; configurable; deterministic where required; safe by default.

Required tests: unit tests; integration tests; exchange adapter tests; data validation tests; feature reproducibility tests; leakage tests; backtest consistency tests; universe reconstruction tests; risk manager tests; execution tests; dashboard permission tests; gate job tests; infrastructure gate tests; job queue tests; backup and restore tests; environment safety tests; ML leakage tests; online learning drift tests; RL environment tests; config validation tests.

Forbidden engineering shortcuts:

- hardcoded secrets;
- hardcoded live mode;
- hardcoded symbol assumptions;
- exchange calls inside strategy logic;
- notebooks as production code;
- optimistic backtests without realistic costs;
- unversioned model artifacts;
- unversioned config changes;
- dashboard actions without audit logs.

Recommended structure:

```
src/
  config/ data/ exchange/ universe/ features/ regimes/ strategies/
  risk/ execution/ backtest/ ml/ adaptation/ monitoring/ reporting/ storage/ cli/
tests/  notebooks/  reports/  configs/  migrations/  scripts/  docs/
```

Notebooks are allowed for research only.
Production logic must live in tested source code.

---

## 32. Roadmap

### Phase 1 — Infrastructure Foundation

Deliver: `docker-compose.yml` with required services (Appendix B.3); `.env.example` with safe defaults and no live keys; config system with environment validation; PostgreSQL database and Alembic migrations; Redis queue integration; job orchestration skeleton; job records, job logs, retry, cancellation, and progress tracking; health check endpoints for every service; dashboard skeleton with authentication; gate records and remediation action records; Infrastructure Gate; Database Gate; Queue Gate; Storage Gate; Monitoring Gate; Backup and Restore Gate skeleton; exchange adapter skeleton; metadata sync job; universe builder skeleton; local test environment; Makefile/task runner commands for setup, test, docker-up, migrate, health, backup, and restore-test.

See Appendix B for full infrastructure contract.

### Phase 2 — Data Platform

Deliver: OHLCV ingestion; mark/index price ingestion; funding ingestion; open interest ingestion; spread snapshots if available; historical downloader; automatic backfill; data validation; storage schema; data quality reports; Data Quality Gate (DATA-COV, DQ).

### Phase 3 — Universe and Features

Deliver: dynamic universe manager; universe versioning; symbol filters; feature pipeline; feature reproducibility tests; Universe Gate (UNIV); Feature Reproducibility Gate (FEAT); Exchange Metadata Gate (META).

### Phase 4 — Backtest Engine

Deliver: event-based backtest; fee model; slippage model; funding model; rejected candidate logging; risk simulation; execution simulation; report generator; Backtest Gate (BT); Walk-Forward Gate (WF); Fee Stress Gate (FEE); Slippage Stress Gate (SLIP).

### Phase 5 — Deterministic Quant Strategies

Deliver research candidates: cross-asset lead-lag; premium/basis mean reversion; funding conditional bias; liquidation exhaustion research; volatility regime filter; session expectancy filter; cross-sectional relative strength; execution quality filter.

### Phase 6 — Ranking, Risk, and Execution Core

Deliver: candidate ranking engine; risk manager; order builder; execution adapter; reconciliation; exchange-side stop handling; kill switch; order ownership; Setup Quality Gate (SETUP); Risk Gate (RISK); Execution Gate (EXEC); Kill Switch Verification Gate (KILL); Order Ownership Gate (ORDER-OWN).

### Phase 7 — Dashboard and Gate Workflow

Deliver: general dashboard; per-symbol dashboard; time-period selection; background job panel; gate status panel; remediation action items; approvals; audit logs; reports linked from dashboard; "Road to Live" view; Gate Runner UI.

### Phase 8 — Paper Trading

Deliver: Phase A technical validation; Phase B strategy validation; paper vs backtest comparison; rejected trade analysis; Paper Technical Gate (PAPER-A); Paper Strategy Gate (PAPER-B).

### Phase 9 — Shadow ML

Deliver: regime classifier shadow; meta-labeling shadow; execution quality model shadow; strategy selector shadow; symbol ranking shadow; ML Shadow Gate (ML-PROMO).

### Phase 10 — ML Recommendation and Constrained Filtering

Deliver only if shadow validation passes: ML recommendation mode; ML trade filter; ML cannot create trades; ML cannot increase risk; ML cannot override risk manager.

### Phase 11 — Online Learning Shadow

Deliver only after sufficient observations: online learning shadow; drift monitoring; calibration monitoring; recommendation mode; constrained live filter only after gate pass; rollback process; Learner Shadow Gate (LEARN-PROMO-S).

### Phase 12 — RL Research and Shadow Policy

Deliver only after mature simulator: RL environment; reward function; action-space constraints; simulation training; simulation stress tests; shadow policy; recommendation mode; RL Simulation Gate; RL Shadow Gate.

### Phase 13 — Controlled Live Readiness

Deliver: Security Gate (SEC); Deployment Gate (DEPLOY); Backup & Restore Gate (BACKUP); Monitoring Gate (MON); Config Freeze Gate (CONFIG-FREEZE); Live Gate (LIVE); deployment checklist; frozen config; frozen strategy versions; frozen model versions; frozen RL policy versions if used; rollback plan; live readiness dashboard; manual approval gate.

If behind schedule: cut dashboard polish and the second strategy before anything in Phases 1–5. **Never cut risk, validation, or tests.**

---

## 33. Success Criteria & Final Operating Principle

The project succeeds only if it demonstrates: clean data; reliable automated backfill; reproducible feature generation; dynamic universe tracking; realistic backtests; robust strategy validation; explainable candidate selection; safe execution; controlled risk; dashboard-visible gates; actionable remediation when gates fail; stable paper behavior; ML improvement over deterministic baseline in shadow mode before live use; online learning stability before live influence; RL simulation and shadow superiority before live influence; live readiness without unsafe assumptions.

**Profit alone — or a profitable backtest or paper run — is not sufficient.**

A failed gate without action items is not acceptable.

The system must never become a black box that trades because a model says so. Edge is small, capacity-limited, regime-dependent, and decays. **Capital does not create edge; the strategy does. ML and adaptation are multipliers on a proven edge, never its source — and never permitted to touch the risk envelope.**

The goal is to go live, but only by becoming ready for live.
When a gate fails, do not bypass it.
Create the remediation plan, execute the action items, regenerate evidence, re-run the gate, and continue progressing.
The safest path to live is the fastest acceptable path.

---

## 34. Reporting Requirements

The system must generate and store:

- data quality report;
- universe report;
- feature reproducibility report;
- backtest report;
- walk-forward report;
- parameter sensitivity report;
- fee stress report;
- slippage stress report;
- strategy report;
- risk report;
- execution report;
- paper technical report;
- paper strategy report;
- live report;
- ML report;
- online learning report;
- RL simulation report;
- RL shadow report;
- dashboard gate report;
- live readiness report;
- daily review report.

Reports must be stored, versioned, and linked from the dashboard.
Every report must include: versions used (config, universe, data, strategy, model); time period; methodology; results; limitations; recommendations.

---

## 35. Emergency Procedures

### Critical Safety Failure

On any of: stale data; reconciliation mismatch; unknown order/position; kill switch triggered; envelope breaker fired; learner rollback; exchange disconnect > threshold; data quality critical failure:

1. **HALT** — stop generating new candidates immediately.
2. Cancel only bot's own pending orders where safe (never touch open positions' exchange-side stops).
3. Log the event to `audit_log` with full context.
4. Send critical alert via all channels.
5. Dashboard shows emergency status with "Resume requires manual review" button.
6. Operator must review, fix root cause, re-run affected gates, and manually resume.

### Emergency Close Mode

If enabled (requires explicit confirmation):
- Close all open positions using market orders.
- Cancel all bot orders.
- Log all actions.
- Alert operator.
- Disable live trading until full review.

### Recovery Checklist After Halt

1. Identify root cause from logs/alerts.
2. Fix the issue (data, code, config, exchange).
3. Re-run affected gates.
4. Verify all gates PASS.
5. Reconcile positions/orders with exchange.
6. Manual approval to resume.
7. Log recovery action.

---
## Appendix A — Gate Catalog & Remediation Playbook

This is the authoritative list of gates. Each is declared in `configs/gates.yaml` with the same `id`, pass condition, and `remediation_steps`, and is executed by the Gate Runner (Section 25). On any non-`PASS` criterion the dashboard shows the failure reason and these ordered action items, plus a re-run button. Goal: clear every gate to `PASS`, then enable live.

### Gate Catalog Summary Table

| ID | Gate | Phase | Blocks Live? | Auto-Remediation? | Manual Review? |
|----|------|-------|-------------|------------------|----------------|
| DATA-COV | Data Coverage & Integrity | 2 | Yes | Partial | No |
| DQ | Data Quality | 2+ | Yes | No | Yes |
| UNIV | Universe Validity | 1–2 | Yes | No | Yes |
| META | Exchange Metadata | 1–3 | Yes | No | Yes |
| FEAT | Feature Reproducibility | 3 | Yes | No | Yes |
| BT | Backtest | 4 | Yes | No | Yes |
| WF | Walk-Forward | 4 | Yes | No | Yes |
| FEE | Fee Stress | 4 | Yes | No | Yes |
| SLIP | Slippage Stress | 4 | Yes | No | Yes |
| SETUP | Setup Quality | 5–6 | Yes | No | Yes |
| RISK | Risk Policy | 6 | Yes | No | Yes |
| EXEC | Execution | 6 | Yes | No | Yes |
| PAPER-A | Technical Paper | 8 | Yes | No | Yes |
| PAPER-B | Strategy Paper | 8 | Yes | No | Yes |
| ML-PROMO | ML Promotion | 9–10 | Yes (if ML) | No | Yes |
| LEARN-PROMO-S | Learner Shadow | 11 | Yes (if learner) | No | Yes |
| LEARN-PROMO-L | Learner Live | 13 | Yes (if learner) | No | Yes |
| SEC | Security | 13 | Yes | No | Yes |
| DEPLOY | Deployment | 13 | Yes | Partial | Yes |
| BACKUP | Backup & Restore | 13 | Yes | No | Yes |
| MON | Monitoring | 13 | Yes | Partial | Yes |
| KILL | Kill Switch | 6,13 | Yes | No | Yes |
| ORDER-OWN | Order Ownership | 6,13 | Yes | No | Yes |
| CONFIG-FREEZE | Config Freeze | 13 | Yes | No | Yes |
| LIVE | Live Gate | 13 | Yes | No | Yes |

### Gate Dependency Rules

- **DATA-COV** must pass before DQ, UNIV, META, FEAT, BT.
- **DQ** must pass before BT, WF, PAPER-A, PAPER-B, LIVE.
- **UNIV** must pass before FEAT, BT, WF, PAPER-A, PAPER-B, LIVE.
- **META** must pass before BT, EXEC, PAPER-A, LIVE.
- **FEAT** must pass before BT, WF, PAPER-A, LIVE.
- **BT** must pass before WF, FEE, SLIP, PAPER-B, LIVE.
- **WF** must pass before FEE, SLIP, PAPER-B, LIVE.
- **FEE + SLIP** must pass before PAPER-B, LIVE.
- **SETUP** must pass before PAPER-A, PAPER-B, LIVE.
- **RISK** must pass before PAPER-A, PAPER-B, LIVE.
- **EXEC** must pass before PAPER-A, PAPER-B, LIVE.
- **PAPER-A** must pass before PAPER-B, LIVE.
- **PAPER-B** must pass before LIVE.
- **ML-PROMO** must pass before LIVE (if ML is enabled).
- **LEARN-PROMO-S** must pass before LEARN-PROMO-L (if learner is enabled).
- **LEARN-PROMO-L** must pass before LIVE (if learner is enabled).
- **SEC + DEPLOY + BACKUP + MON + KILL + ORDER-OWN + CONFIG-FREEZE** must pass before LIVE.
- **LIVE** is the final gate; it checks all upstream gates and adds live-specific criteria.

When a gate fails, all downstream gates are marked `BLOCKED` with a clear message: "Blocked because upstream gate X failed. Fix X first, then re-run this gate."

---

### DATA-COV — Data Coverage & Integrity (Phase 2)
- **Pass:** for every universe symbol, all required series (OHLCV per TF, funding, OI, mark/index, liquidation if used) cover the required window with 0 unfilled gaps; an immutable dataset snapshot id is produced.
- **Failure signals:** missing history, gaps/duplicates/out-of-order rows, missing series for a symbol, checksum/row-count mismatch.
- **Remediation:** 1) open the data-quality report, identify symbols/series/timestamps flagged; 2) run `scripts/backfill --symbol <s> --series <x> --from <t>`; 3) if the exchange lacks history, mark the symbol `insufficient_history` and exclude it from the universe; 4) re-snapshot the dataset; 5) re-run.
- **Dashboard actions:** Run Backfill Job | View Missing Ranges | Quarantine Symbol | Re-snapshot Dataset | Re-run Gate
- **Auto-remediation:** Partial (safe gap repair only; manual review for quarantine decisions).
- **Manual review required:** Yes for quarantine decisions.
- **Re-run:** `gate:data-cov`

### DQ — Data Quality (before every research/paper/live run)
- **Pass:** no critical data-quality violation active (Section 23); clocks within NTP tolerance.
- **Failure signals:** stale websocket, mark/index/perp timestamp inconsistency, funding-timestamp mismatch, abnormal spread/gap, clock drift.
- **Remediation:** 1) inspect which check failed and the symbol/stream; 2) restart the affected ingestion stream / reconnect websocket; 3) fix clock (enable/repair NTP) if drift; 4) if exchange-side outage, wait and let auto-retry clear it; 5) re-run.
- **Dashboard actions:** View Data Quality Report | Restart Stream | Fix NTP | Re-run Gate
- **Auto-remediation:** No (requires operator judgment).
- **Manual review required:** Yes.
- **Re-run:** `gate:dq`

### UNIV — Universe Validity (Phase 1–2)
- **Pass:** every selected symbol meets all universe filters (volume, spread, depth, history length, missing-data %, data availability) and has stable metadata; universe is versioned.
- **Failure signals:** symbol below liquidity/volume threshold, unstable/missing metadata, too little history, newly listed.
- **Remediation:** 1) review the per-symbol filter report; 2) tighten/relax filter thresholds in `universe.yaml` *only if justified* and re-validate; 3) drop failing symbols; 4) bump universe version; 5) re-run.
- **Dashboard actions:** View Filter Report | Edit Universe Config | Drop Symbol | Bump Universe Version | Re-run Gate
- **Auto-remediation:** No.
- **Manual review required:** Yes.
- **Re-run:** `gate:universe`

### META — Exchange Metadata (Phase 1–3)
- **Pass:** all active symbols have `[VERIFIED]` metadata (tick size, lot size, min notional, fees, funding schedule, leverage limits, order types) against current exchange docs; no contradictions; no `[UNVERIFIED]` flags for active symbols.
- **Failure signals:** missing metadata for active symbol; `[UNVERIFIED]` metadata; contradictory values; stale metadata older than configured threshold.
- **Remediation:** 1) run `sync_exchange_metadata` job; 2) review new metadata against exchange docs; 3) mark verified symbols as `[VERIFIED]`; 4) quarantine symbols with unresolvable metadata issues; 5) bump metadata version; 6) re-run.
- **Dashboard actions:** Sync Metadata | View Metadata Report | Mark Verified | Quarantine Symbol | Re-run Gate
- **Auto-remediation:** No (requires human verification against docs).
- **Manual review required:** Yes.
- **Re-run:** `gate:meta`

### FEAT — Feature Reproducibility (Phase 3)
- **Pass:** features computed from stored raw data are 100% reproducible; no look-ahead leakage; no unavailable features used; timestamp alignment verified; feature store builds cleanly from dataset snapshot.
- **Failure signals:** non-reproducible features; look-ahead detected (synthetic data yields non-zero expectancy); timestamp misalignment; feature build fails from snapshot.
- **Remediation:** 1) identify non-reproducible features; 2) check timestamp alignment in feature pipeline; 3) check look-ahead leakage (run synthetic data test); 4) rebuild feature store from raw data snapshot; 5) fix feature code if necessary; 6) re-run feature reproducibility tests.
- **Dashboard actions:** View Feature Report | Run Leakage Test | Rebuild Feature Store | Re-run Gate
- **Auto-remediation:** No.
- **Manual review required:** Yes.
- **Re-run:** `gate:feat`

### BT — Backtest Gate (Phase 4)
- **Pass:** event-based backtest completes without errors; includes realistic fees, slippage, funding; no look-ahead; no survivorship bias; universe as of test time; outputs all required metrics; passes basic sanity checks (e.g., no impossible returns).
- **Failure signals:** backtest errors/crashes; missing cost model; look-ahead detected; survivorship bias; universe leakage; impossible results.
- **Remediation:** 1) review backtest logs for errors; 2) verify cost model (fees, slippage, funding) matches exchange metadata; 3) fix look-ahead or survivorship bias in code; 4) re-run backtest with clean dataset snapshot; 5) verify outputs.
- **Dashboard actions:** View Backtest Logs | Fix Cost Model | Run Leakage Test | Re-run Gate
- **Auto-remediation:** No.
- **Manual review required:** Yes.
- **Re-run:** `gate:backtest --strategy <id>`

### WF — Walk-Forward Gate (Phase 4)
- **Pass:** walk-forward validation completes with ≥K folds passing kill-criteria; edge is not isolated to one period; no overfitting detected; results are stable across folds.
- **Failure signals:** <K folds pass; edge only in isolated folds; high variance across folds; parameter instability.
- **Remediation:** 1) identify failing folds; 2) check if edge exists only in isolated market periods; 3) reduce parameter overfitting; 4) prefer robust parameter zones; 5) restrict strategy to regimes/symbols where stable, or keep research-only; 6) re-run walk-forward.
- **Dashboard actions:** View Fold Breakdown | Run Parameter Sensitivity | Restrict Strategy | Re-run Gate
- **Auto-remediation:** No.
- **Manual review required:** Yes.
- **Re-run:** `gate:walk-forward --strategy <id>`

### FEE — Fee Stress Gate (Phase 4)
- **Pass:** strategy survives ×2 fees (or configured multiplier) with positive expectancy net of costs; edge is not fee-dependent.
- **Failure signals:** expectancy turns negative under fee stress; most profit from low-fee periods; edge disappears with realistic costs.
- **Remediation:** 1) review cost sensitivity report; 2) reduce turnover; 3) improve execution route (maker-first); 4) increase minimum expected edge threshold; 5) disable strategy if edge disappears under realistic fees.
- **Dashboard actions:** View Cost Report | Run Execution Optimization | Adjust Edge Threshold | Disable Strategy | Re-run Gate
- **Auto-remediation:** No.
- **Manual review required:** Yes.
- **Re-run:** `gate:fee-stress --strategy <id>`

### SLIP — Slippage Stress Gate (Phase 4)
- **Pass:** strategy survives +50% slippage (or configured multiplier) with positive expectancy; slippage is not the primary risk.
- **Failure signals:** expectancy turns negative under slippage stress; strategy concentrated in toxic symbols/sessions; high adverse selection.
- **Remediation:** 1) review slippage by symbol, session, regime, and order type; 2) disable toxic symbols or time windows; 3) add spread/slippage hard blockers; 4) prefer passive execution where possible; 5) reduce strategy activity in high-volatility regimes; 6) re-run slippage stress.
- **Dashboard actions:** View Slippage Report | Add Execution Blocker | Disable Symbol/Session | Re-run Gate
- **Auto-remediation:** No.
- **Manual review required:** Yes.
- **Re-run:** `gate:slippage-stress --strategy <id>`

### SETUP — Setup Quality Gate (Phase 5–6)
- **Pass:** setup quality scoring is deterministic, reproducible, and validated; threshold is justified via walk-forward; no hard blocker is bypassed; all score components are documented.
- **Failure signals:** non-deterministic scoring; threshold not validated; hard blockers bypassed; score components not documented.
- **Remediation:** 1) review setup quality logic for determinism; 2) validate threshold via walk-forward (it is a tunable); 3) ensure all hard blockers are enforced in code; 4) document score components; 5) re-run setup quality tests.
- **Dashboard actions:** View Setup Logic | Validate Threshold | Test Hard Blockers | Re-run Gate
- **Auto-remediation:** No.
- **Manual review required:** Yes.
- **Re-run:** `gate:setup-quality`

### RISK — Risk Policy Gate (Phase 6)
- **Pass:** all risk rules are implemented and tested; exposure limits enforced; drawdown controls active; circuit breakers tested; sizing formula correct; reconciliation logic tested; kill switch tested.
- **Failure signals:** missing risk rule; exposure limit not enforced; drawdown control inactive; circuit breaker not tested; sizing formula incorrect; reconciliation gap; kill switch unresponsive.
- **Remediation:** 1) review risk policy implementation; 2) add missing rules; 3) fix exposure/drawdown/circuit breaker logic; 4) verify sizing formula against spec; 5) test reconciliation with exchange; 6) test kill switch (deliberate trip); 7) re-run risk tests.
- **Dashboard actions:** View Risk Report | Edit Risk Config | Test Circuit Breaker | Test Kill Switch | Re-run Gate
- **Auto-remediation:** No.
- **Manual review required:** Yes.
- **Re-run:** `gate:risk`

### EXEC — Execution Gate (Phase 6)
- **Pass:** all order types supported by exchange are implemented; order builder respects tick/lot/min-notional; stop/TP placement works; cancel/replace works; partial fill handling works; slippage measurement works; reconciliation works.
- **Failure signals:** unsupported order type; order builder violates constraints; stop/TP fails; cancel/replace fails; partial fill mishandled; slippage not measured; reconciliation gap.
- **Remediation:** 1) review execution adapter against exchange docs; 2) fix order builder constraints; 3) test stop/TP placement; 4) test cancel/replace; 5) test partial fill handling; 6) verify slippage measurement; 7) fix reconciliation logic; 8) re-run execution tests.
- **Dashboard actions:** View Execution Report | Test Order Types | Test Stops | Test Reconciliation | Re-run Gate
- **Auto-remediation:** No.
- **Manual review required:** Yes.
- **Re-run:** `gate:execution`

### PAPER-A — Technical Paper (Phase 8)
- **Pass:** infrastructure works end-to-end in paper (ingestion, candidate gen, risk approve/reject, execution sim, order/stop logging, dashboard, alerts, kill switch, reconciliation).
- **Failure signals:** any pipeline step errors, stop not simulated, kill switch or reconciliation not exercised.
- **Remediation:** 1) open the failing component's logs; 2) fix and add/repair the corresponding test (`tests/`); 3) re-run paper-A.
- **Dashboard actions:** View Technical Logs | Fix Component | Add Test | Re-run Gate
- **Auto-remediation:** No.
- **Manual review required:** Yes.
- **Re-run:** `gate:paper-a`

### PAPER-B — Strategy Paper (Phase 8)
- **Pass:** sufficient candidate + executed paper trades; strategy/execution/risk reports produced; per-symbol & per-regime breakdowns; paper-vs-backtest consistent; manual review done.
- **Failure signals:** too few trades (→ `BLOCKED`), paper materially worse than backtest, regimes/symbols unrepresented.
- **Remediation:** 1) if too few trades, extend the paper window (do not loosen filters); 2) if paper ≪ backtest, investigate slippage/fees/latency assumptions and execution quality, reconcile the cost model with realized fills; 3) if a regime is unseen, wait for it or document the limitation; 4) re-run.
- **Dashboard actions:** View Paper Report | Extend Paper Run | Reconcile Costs | Re-run Gate
- **Auto-remediation:** No.
- **Manual review required:** Yes.
- **Re-run:** `gate:paper-b`

### ML-PROMO — ML Promotion (Phase 9–10)
- **Pass:** in shadow the model improves expectancy net of costs, preserves/raises profit factor, preserves/reduces max DD, does not remove most best trades, reduces tail risk; stable across folds/symbols/regimes/OOS; calibrated; explainable; no leakage; manually reviewed.
- **Failure signals:** no improvement over deterministic baseline, unstable across slices, miscalibrated, leakage detected.
- **Remediation:** 1) if no improvement, keep model **shadow-only** (do not promote); 2) if leakage, fix features/labels and re-train; 3) if instability, simplify the model class or add data, re-validate; 4) re-run shadow scoring.
- **Dashboard actions:** View ML Report | Run Leakage Test | Retrain Model | Keep Shadow Only | Re-run Gate
- **Auto-remediation:** No.
- **Manual review required:** Yes.
- **Re-run:** `gate:ml-promo --model <id>`

### LEARN-PROMO-S — Learner Shadow → Recommend (Phase 11)
- **Pass:** eligibility (21.1) met; shadow policy beats baseline on walk-forward **and** locked hold-out; bounded actions only; frozen fallback exists and tested.
- **Failure signals:** insufficient samples (→ `BLOCKED`), no shadow edge, any envelope-touching action proposed.
- **Remediation:** 1) accumulate `min_samples_to_start` (keep in shadow); 2) if no shadow edge, keep shadow-only / revise policy; 3) if an action touched the envelope, fix `envelope_guard` bounds (it must be rejected) before anything else; 4) re-run.
- **Dashboard actions:** View Learner Report | Fix Guard Bounds | Extend Shadow | Re-run Gate
- **Auto-remediation:** No.
- **Manual review required:** Yes.
- **Re-run:** `gate:learn-promo-s`

### LEARN-PROMO-L — Learner Recommend → Bounded Live (Phase 13)
- **Pass:** a track record of approved-and-correct recommendations; bounded-update/rollback config present and tested; learner kill switch verified; auto-freeze-on-breaker enforced.
- **Failure signals:** thin/negative recommendation track record, rollback path untested, kill switch unverified.
- **Remediation:** 1) extend RECOMMEND mode until the approved-recommendation track record is sufficient; 2) test `rollback.revert()` (deliberately trip a trigger) and confirm revert-to-fallback; 3) verify learner kill switch independently; 4) re-run.
- **Dashboard actions:** View Recommendation Track Record | Test Rollback | Verify Kill Switch | Re-run Gate
- **Auto-remediation:** No.
- **Manual review required:** Yes.
- **Re-run:** `gate:learn-promo-l`

### SEC — Security Gate (Phase 13)
- **Pass:** no secrets in repository; dashboard authentication enabled; HTTPS configured in non-local environments; exchange keys validated; withdrawal permissions disabled; IP whitelist checked; live keys absent outside production; audit logs enabled; security checklist completed.
- **Failure signals:** secrets in repo; auth disabled; no HTTPS; invalid API keys; withdrawal enabled; live keys in wrong environment; audit logs disabled.
- **Remediation:** 1) remove secrets from repo, use env vars/secrets manager; 2) enable dashboard auth; 3) configure HTTPS; 4) validate API key permissions (no withdrawal); 5) remove live keys from non-prod environments; 6) enable audit logs; 7) complete security checklist; 8) re-run.
- **Dashboard actions:** View Security Checklist | Fix Secrets | Enable Auth | Configure HTTPS | Validate Keys | Re-run Gate
- **Auto-remediation:** No.
- **Manual review required:** Yes.
- **Re-run:** `gate:security`

### DEPLOY — Deployment Gate (Phase 13)
- **Pass:** all Docker services start cleanly; health checks pass; environment variables validate; safe defaults enforced; live service disabled by default; database migrations apply; queue works; storage reachable; monitoring active.
- **Failure signals:** service fails to start; health check fails; env var missing/invalid; unsafe default detected; live service enabled unexpectedly; migration fails; queue unreachable; storage error; monitoring inactive.
- **Remediation:** 1) check `docker-compose logs` for failing service; 2) fix health check endpoint; 3) validate `.env` against `.env.example`; 4) enforce safe defaults; 5) ensure `ENABLE_LIVE_TRADING=false` by default; 6) fix migration; 7) verify Redis connectivity; 8) verify storage path/MinIO; 9) start monitoring stack; 10) re-run.
- **Dashboard actions:** View Deployment Logs | Fix Service | Validate Env | Enforce Defaults | Fix Migration | Re-run Gate
- **Auto-remediation:** Partial (some service restarts).
- **Manual review required:** Yes.
- **Re-run:** `gate:deploy`

### BACKUP — Backup & Restore Gate (Phase 13)
- **Pass:** database backup configured and tested; data lake backup configured (or explicit MVP exception); config backup configured; model/report backup configured; restore test executed successfully; restore report linked.
- **Failure signals:** no backup job; backup fails; restore test not run; restore test fails; no restore report.
- **Remediation:** 1) configure automated DB backup (e.g., `pg_dump` cron); 2) configure data lake backup or document MVP exception; 3) config backup via git tags; 4) model/report backup to object storage; 5) run restore test job; 6) review restore report; 7) fix any restore issues; 8) re-run.
- **Dashboard actions:** View Backup Config | Run Backup Job | Run Restore Test | View Restore Report | Re-run Gate
- **Auto-remediation:** No.
- **Manual review required:** Yes.
- **Re-run:** `gate:backup`

### MON — Monitoring Gate (Phase 13)
- **Pass:** health checks active for all services; system status visible in dashboard; alerts configured; alert test delivered end-to-end; stale data alert works; failed job alert works; kill switch alert works; Prometheus/Grafana or equivalent running (optional but recommended).
- **Failure signals:** health check missing; dashboard not showing status; alert not configured; alert test failed; stale data alert not firing; job failure alert not firing; kill switch alert not firing.
- **Remediation:** 1) add missing health check endpoints; 2) verify dashboard status page; 3) configure alert channels (Telegram, email); 4) send test alert for each type; 5) verify stale data detection and alert; 6) verify failed job detection and alert; 7) test kill switch alert (deliberate trip); 8) re-run.
- **Dashboard actions:** View Health Checks | Configure Alerts | Send Test Alert | Test Kill Switch Alert | Re-run Gate
- **Auto-remediation:** Partial (alert configuration).
- **Manual review required:** Yes.
- **Re-run:** `gate:monitoring`

### KILL — Kill Switch Verification Gate (Phase 6, 13)
- **Pass:** CLI kill switch works independently of dashboard; dashboard kill switch works; both halt trading within configured timeout; alerts fire on activation; recovery requires manual review; tested by deliberate trip in paper mode.
- **Failure signals:** kill switch unresponsive; does not halt trading; no alert; auto-recovery without review; not tested in paper.
- **Remediation:** 1) test CLI kill switch (`make kill` or equivalent); 2) test dashboard kill switch button; 3) verify trading halts within timeout; 4) verify alert delivery; 5) verify manual review is required to resume; 6) document kill switch procedure; 7) re-run.
- **Dashboard actions:** Test CLI Kill Switch | Test Dashboard Kill Switch | Verify Alert | Document Procedure | Re-run Gate
- **Auto-remediation:** No.
- **Manual review required:** Yes.
- **Re-run:** `gate:kill-switch`

### ORDER-OWN — Order Ownership Gate (Phase 6, 13)
- **Pass:** `ORDER_CLIENT_ID_PREFIX` and `BOT_INSTANCE_ID` configured; all bot orders carry prefix; reconciliation detects unknown orders/positions; halt on unknown order works; emergency close mode requires explicit confirmation.
- **Failure signals:** prefix not configured; orders missing prefix; unknown orders not detected; unknown positions not detected; halt not triggered; emergency close too easy to trigger.
- **Remediation:** 1) configure `ORDER_CLIENT_ID_PREFIX` and `BOT_INSTANCE_ID`; 2) verify all order placement includes prefix; 3) test reconciliation with manual order (should detect and halt); 4) test position mismatch detection; 5) verify halt logic; 6) add confirmation for emergency close; 7) re-run.
- **Dashboard actions:** View Ownership Config | Test Reconciliation | Test Unknown Order Detection | Test Emergency Close | Re-run Gate
- **Auto-remediation:** No.
- **Manual review required:** Yes.
- **Re-run:** `gate:order-ownership`

### CONFIG-FREEZE — Config Freeze Gate (Phase 13)
- **Pass:** `CONFIG_VERSION` is frozen and tagged in git; all sub-versions (strategy, model, risk, execution, data, universe, feature) are frozen and linked; no unversioned artifacts; config change log complete; rollback to previous version tested.
- **Failure signals:** config not tagged; versions not frozen; unversioned artifact detected; missing change log; rollback untested.
- **Remediation:** 1) git tag current config as `CONFIG_VERSION`; 2) freeze and tag all sub-artifacts; 3) verify no unversioned files in production paths; 4) complete change log; 5) test rollback to previous config version; 6) re-run.
- **Dashboard actions:** View Config Versions | Git Tag Config | Freeze Artifacts | Test Rollback | Re-run Gate
- **Auto-remediation:** No.
- **Manual review required:** Yes.
- **Re-run:** `gate:config-freeze`

### LIVE — Live Gate (Phase 13)
For each, **Re-run** is `gate:live --criterion <id>` (or `gate:live` for all).
- **LIVE-0 (all upstream gates):** *Pass:* all upstream critical gates (DATA-COV through CONFIG-FREEZE) are `PASS` and not expired. *Remediation:* run the specific upstream gate that is not PASS; follow its remediation steps; do not proceed until all upstream gates are green.
- **LIVE-1 (72h soak):** *Pass:* ≥72h continuous testnet/demo on VPS, no unhandled crash. *Remediation:* fix the crash from logs, add a regression test, restart the 72h clock.
- **LIVE-2 (breakers/kill):** *Pass:* stop-loss, daily-loss, drawdown, kill switch (CLI+dashboard) each verified by deliberate trip. *Remediation:* run each forced-failure test; fix any breaker that did not halt/flatten; re-verify.
- **LIVE-3 (restart-reconciliation):** *Pass:* kill process with an open position, restart, state correct. *Remediation:* fix reconciliation logic so exchange is source of truth; ensure mismatch → halt+alert; re-test.
- **LIVE-4 (portfolio limits):** *Pass:* heat cap, beta cap, max-concurrent enforced under deliberate load. *Remediation:* fix the limit that breached; add a test that loads the portfolio past each cap; re-verify.
- **LIVE-5 (alerts E2E):** *Pass:* every alert type delivered end-to-end. *Remediation:* fix the notifier/credentials/route for the failing alert; send a test event; re-verify.
- **LIVE-6 (validation):** *Pass:* strategy passed `VAL-*` with declared kill-criteria, costs first-class. *Remediation:* see VAL gate; do not go live until VAL is green.
- **LIVE-7 (metadata verified):** *Pass:* fees, funding schedule, min-notional, precision `[VERIFIED]` against current exchange docs. *Remediation:* re-sync metadata, confirm against docs, flip `[UNVERIFIED]`→`[VERIFIED]`; re-run META gate.
- **LIVE-8 (paper-B):** *Pass:* Paper Phase B passed per strategy. *Remediation:* see PAPER-B gate.
- **LIVE-9 (frozen versions + rollback):** *Pass:* strategy/model/config versions frozen, rollback plan ready, security checklist reviewed. *Remediation:* see CONFIG-FREEZE and BACKUP gates; tag and freeze versions; write/verify rollback runbook; complete security checklist; re-run.
- **LIVE-10 (operator sign-off):** *Pass:* operator confirms capital is fully loseable and confirms risk/leverage/drawdown numbers. *Remediation:* present the numbers on the dashboard for explicit confirmation; record the sign-off.

When **all** criteria are `PASS`, the dashboard enables the manual **"Go Live"** action (still a deliberate human step). Going live never bypasses any gate; it is the result of clearing all of them.

---
## Appendix B — Infrastructure Contract for Coding Agents

This section is mandatory implementation guidance for coding agents. It supplements the main architecture sections (5, 25, 32) with concrete implementation details.

The system must be implemented as a small quant trading platform, not as a single trading script.

Infrastructure must support four independent workloads:

1. data acquisition and validation;
2. research, backtesting, ML, online learning, and RL jobs;
3. paper/live trading execution;
4. dashboard, monitoring, reporting, gates, approvals, and remediation workflows.

Heavy research jobs must not block paper/live trading.

Live execution must remain lightweight, stable, observable, and isolated from expensive background jobs.

### B.1 Required Runtime Environments

Implement support for these environments from the beginning:

- `local` — local development, unit tests, small backtests, mocked exchange;
- `research` — historical data, backtests, walk-forward, ML/RL training, no live keys;
- `paper` — exchange testnet/demo or internal paper execution, dashboard, gates, alerts;
- `production` — live keys, frozen config, live engine, monitoring, no heavy research jobs;
- `staging` — optional production-like validation before production deployment.

Rules:

- Each environment must have its own config file or config profile.
- Each environment must have explicit `TRADING_MODE`.
- `TRADING_MODE=LIVE` must never be the default.
- Research environment must not have withdrawal permissions or live trading keys.
- Production environment must not run heavy backtests, ML training, or RL training on the live execution process.
- Environment mismatch must fail startup.

Required config values:

- `APP_ENV`
- `TRADING_MODE`
- `EXCHANGE_ID`
- `EXCHANGE_ENV`
- `BOT_INSTANCE_ID`
- `ORDER_CLIENT_ID_PREFIX`
- `DATABASE_URL`
- `REDIS_URL`
- `OBJECT_STORAGE_URL` or local data lake path
- `DASHBOARD_AUTH_MODE`
- `ENABLE_LIVE_TRADING`
- `ENABLE_BACKGROUND_RESEARCH_JOBS`
- `ENABLE_ML_SHADOW`
- `ENABLE_ONLINE_LEARNING_SHADOW`
- `ENABLE_RL_SHADOW`

### B.2 Required Services

The coding agent must implement the system as separate services, even if they initially run on one machine through Docker Compose.

Required services:

- `postgres` — relational database;
- `redis` — queues, locks, cache, job state;
- `backend` — FastAPI API for dashboard, jobs, reports, gates, approvals;
- `dashboard` — web UI for monitoring, statistics, gates, jobs, reports, approvals;
- `worker-data` — historical download, live data persistence, repair jobs;
- `worker-backtest` — backtests, walk-forward, parameter sensitivity, stress tests;
- `worker-ml` — ML training, ML shadow evaluation, model reports;
- `worker-rl` — RL simulation, RL training, RL shadow/recommendation evaluation;
- `worker-reports` — scheduled reports, strategy reports, gate reports, daily reviews;
- `scheduler` — scheduled jobs and recurring maintenance tasks;
- `trading-engine-paper` — paper execution loop;
- `trading-engine-live` — live execution loop, disabled by default;
- `caddy` or `nginx` — HTTPS reverse proxy;
- optional `prometheus` and `grafana` — metrics stack;
- optional `minio` — S3-compatible object storage for local deployments.

Rules:

- Services must be independently startable.
- Live engine must not start unless Live Activation Gate is passed and manual approval exists (Section 27).
- Trading engines must not run database migrations on startup.
- Workers must use queues and job records, not ad hoc background threads hidden inside API handlers.
- Long-running jobs must be resumable or safely restartable.
- Each service must expose health checks.

### B.3 Recommended Initial Docker Compose Services

Initial implementation should provide `docker-compose.yml` with at least:

- `postgres`
- `redis`
- `backend`
- `dashboard`
- `worker-data`
- `worker-backtest`
- `worker-ml`
- `worker-rl`
- `worker-reports`
- `scheduler`
- `trading-engine-paper`
- `trading-engine-live`
- `caddy`

`trading-engine-live` must be present but disabled by default.

The coding agent must add a `.env.example` with safe non-live defaults.

Required safe defaults:

- `TRADING_MODE=PAPER`
- `ENABLE_LIVE_TRADING=false`
- `ENABLE_BACKGROUND_RESEARCH_JOBS=true`
- `ENABLE_ML_SHADOW=false`
- `ENABLE_ONLINE_LEARNING_SHADOW=false`
- `ENABLE_RL_SHADOW=false`
- no real API keys;
- dashboard password placeholder;
- local database URL;
- local Redis URL.

### B.4 Database Responsibilities

PostgreSQL is the system of record for operational, relational, auditable state.

Store in PostgreSQL:

- exchange metadata snapshots;
- symbol metadata;
- universe versions;
- universe membership;
- dataset versions;
- config versions;
- strategy versions;
- risk policy versions;
- execution policy versions;
- model registry;
- RL policy registry;
- job records;
- gate records;
- remediation action records;
- approvals;
- audit logs;
- candidate trades;
- rejected candidates;
- approved trades;
- orders;
- fills;
- positions;
- reconciliations;
- paper trades;
- live trades;
- reports metadata;
- dashboard saved filters;
- live activation requests.

Do not store very large historical candle datasets only in relational tables unless the dataset is small.

PostgreSQL may store metadata and recent operational slices, but large history must be stored in the data lake as partitioned files.

### B.5 Data Lake and Object Storage

The bot owns its data.

It must download, backfill, validate, store, and version all datasets required for backtesting, paper trading, live validation, ML, online learning, and RL.

Use Parquet as the primary historical research storage format.

Required data lake partitions:

- `exchange_id`
- `data_type`
- `symbol`
- `timeframe` if applicable
- `year`
- `month`
- `day` if needed

Required data types:

- OHLCV;
- mark price;
- index price;
- funding rates;
- open interest;
- liquidations if available;
- spread snapshots;
- order book snapshots if enabled;
- trades if available;
- engineered features;
- backtest datasets;
- ML training datasets;
- RL training episodes;
- model artifacts;
- report artifacts.

Allowed storage backends:

- local filesystem for MVP;
- MinIO for local S3-compatible storage;
- S3-compatible cloud storage later.

Rules:

- Every dataset must have a dataset version.
- Every dataset version must have a manifest.
- Every manifest must include symbols, time range, data types, row counts, missing ranges, validation status, and source jobs.
- Backtests must reference dataset versions, not loose files.
- ML/RL training must reference dataset versions, not loose files.

### B.6 Background Job System

The system must implement a proper background job model.

Every long-running operation triggered from the dashboard must create a job record.

Required job fields:

- `job_id`
- `job_type`
- `status`
- `created_at`
- `started_at`
- `finished_at`
- `requested_by`
- `environment`
- `input_params`
- `related_gate_id`
- `related_dataset_version`
- `related_universe_version`
- `related_strategy_version`
- `related_model_version`
- `progress_current`
- `progress_total`
- `progress_message`
- `logs_uri`
- `artifact_uri`
- `failure_reason`
- `next_action_hint`

Required statuses:

- `queued`
- `running`
- `succeeded`
- `failed`
- `cancelled`
- `blocked`
- `expired`

Required dashboard actions:

- start job;
- cancel job;
- retry failed job;
- view logs;
- view artifacts;
- re-run related gate;
- create remediation task;
- mark manual check complete only where manual checks are allowed.

### B.7 Required Job Types

Implement these job types as first-class job definitions:

Data jobs:

- `sync_exchange_metadata`
- `build_symbol_universe`
- `download_ohlcv_history`
- `download_mark_index_history`
- `download_funding_history`
- `download_open_interest_history`
- `download_liquidation_history`
- `download_spread_snapshots`
- `repair_missing_data`
- `validate_data_quality`
- `build_dataset_version`
- `build_feature_dataset`

Research jobs:

- `run_single_strategy_backtest`
- `run_multi_symbol_backtest`
- `run_walk_forward_validation`
- `run_parameter_sensitivity`
- `run_fee_stress_test`
- `run_slippage_stress_test`
- `run_symbol_breakdown_report`
- `run_regime_breakdown_report`
- `run_session_breakdown_report`

Paper/live validation jobs:

- `run_paper_technical_report`
- `run_paper_strategy_report`
- `run_execution_quality_report`
- `run_risk_report`
- `run_live_readiness_check`
- `run_daily_review`

ML jobs:

- `build_ml_dataset`
- `train_ml_model`
- `evaluate_ml_model`
- `run_ml_shadow_backtest`
- `run_ml_shadow_paper_report`
- `calibrate_ml_model`
- `promote_ml_model_request`

Online learning jobs:

- `build_online_learning_state`
- `run_online_learning_shadow_eval`
- `run_online_learning_recommendation_eval`
- `promote_online_learner_request`

RL jobs:

- `build_rl_environment_dataset`
- `train_rl_policy_simulation`
- `evaluate_rl_policy_simulation`
- `run_rl_shadow_eval`
- `run_rl_recommendation_eval`
- `promote_rl_policy_request`

Security/deployment jobs:

- `run_security_checklist`
- `run_config_freeze_check`
- `run_order_ownership_check`
- `run_exchange_permissions_check`
- `run_backup_check`
- `run_restore_test_check`
- `create_live_activation_request`

Gate runner jobs:

- `run_gate` (generic, dispatches to specific gate logic)
- `run_all_gates` (runs all gates in dependency order)
- `run_upstream_gates` (runs all gates upstream of a target)

### B.8 Dashboard as Control Center

The dashboard is not only a statistics viewer. It is the operational control center for: data coverage; universe status; jobs; gates; remediation actions; reports; approvals; paper/live monitoring; ML/RL maturity; live readiness.

The dashboard must include these pages:

1. Overview
2. Data Coverage
3. Universe
4. Jobs
5. Gates
6. Remediation Actions
7. Backtests
8. Paper Trading
9. Live Trading
10. General Statistics
11. Per-Symbol Statistics
12. Strategy Analytics
13. Regime Analytics
14. Session Analytics
15. Execution Quality
16. Risk
17. ML Shadow
18. Online Learning
19. RL
20. Reports
21. Approvals
22. System Health
23. Settings

Every statistics page must include a time period selector.

Required time filters:

- today;
- yesterday;
- last 7 days;
- last 30 days;
- current month;
- previous month;
- custom date range;
- by backtest run;
- by paper session;
- by live session;
- by config version;
- by universe version;
- by strategy version;
- by model version.

### B.9 Dashboard Gate Remediation UX

Every failed, blocked, or expired gate must show clear action items.

A failed gate must never be shown as a dead end.

For each failed gate, dashboard must show:

- gate name;
- status;
- last run time;
- failed check;
- failure reason;
- affected symbols;
- affected strategies;
- affected datasets;
- severity;
- why it blocks live;
- required action items (ordered, from `configs/gates.yaml`);
- recommended next job;
- button to start recommended job;
- button to view logs;
- button to view report;
- button to create remediation task;
- button to rerun gate after remediation;
- owner role;
- estimated scope of remediation, if known;
- auto-remediation available? (yes/no);
- manual review required? (yes/no).

Example failed gate remediation:

Gate: Data Quality Gate
Status: Failed
Reason: Missing 1m OHLCV data for multiple active symbols in universe version `u_2026_06_16_001`.
Why this blocks live: Backtest and feature datasets are incomplete. Strategy validation would be biased.
Required actions:
1. Run `repair_missing_data` for affected symbols.
2. Re-run `validate_data_quality`.
3. Build a new dataset version.
4. Re-run affected backtests.
5. Re-run Data Quality Gate.
Dashboard buttons:
- Run Repair Job
- Re-run Data Validation
- Build Dataset Version
- Re-run Gate
- View Missing Ranges
- Create Remediation Task

### B.10 Gate State Machine

All gates must use the same state machine.

Allowed gate statuses:

- `not_run`
- `running`
- `passed`
- `failed`
- `blocked`
- `expired`
- `needs_manual_approval`
- `approved`
- `rejected`

Rules:

- `not_run` blocks downstream gates.
- `running` blocks downstream gates.
- `failed` blocks downstream gates and must produce remediation actions.
- `blocked` must show the upstream dependency that blocks it.
- `expired` must show what changed and why the gate must be re-run.
- `needs_manual_approval` must show who must approve and what evidence is required.
- `approved` must store approver, timestamp, evidence links, and approved versions.
- `rejected` must store rejection reason and next actions.

Gate results expire when relevant versions change.

Examples:

- New dataset version expires Backtest Gate and ML Dataset Gate.
- New universe version expires Universe Gate, Backtest Gate, Paper Strategy Gate, and Live Readiness Gate.
- New strategy version expires Backtest Gate and Paper Strategy Gate.
- New model version expires ML Shadow Gate.
- New risk policy version expires Risk Gate and Live Readiness Gate.
- New execution policy version expires Execution Gate and Live Readiness Gate.
- New config version expires Config Freeze Gate and Live Readiness Gate.

### B.11 Infrastructure Gates

Implement these infrastructure gates in addition to research/trading gates (see Appendix A for full details):

Infrastructure Gate:

- Docker services defined;
- required services can start;
- health checks exist;
- environment variables validate;
- safe defaults are enforced;
- live service disabled by default;
- dashboard authentication configured.

Database Gate:

- migrations apply cleanly;
- required tables exist;
- indexes exist for critical queries;
- connection pooling works;
- backup job exists;
- restore test job exists.

Queue Gate:

- Redis reachable;
- job creation works;
- workers pick up jobs;
- failed jobs are visible;
- retry works;
- cancellation works;
- job logs are linked.

Storage Gate:

- data lake path or object storage reachable;
- manifests can be written;
- artifacts can be written;
- reports can be written;
- model artifacts can be written;
- permission errors are surfaced.

Monitoring Gate:

- health checks active;
- system status visible;
- alerts configured;
- alert test works;
- stale data alert works;
- failed job alert works;
- kill switch alert works.

Security Gate:

- no secrets in repository;
- dashboard authentication enabled;
- HTTPS configured in non-local environments;
- exchange keys validated;
- withdrawal permission rejected;
- IP whitelist recommended or checked if supported;
- live keys are absent outside production;
- audit logs enabled.

Backup and Restore Gate:

- database backup configured;
- data lake backup configured or explicitly accepted as local-only MVP risk;
- config backup configured;
- model/report backup configured;
- restore test executed;
- restore result report linked.

### B.12 Deployment Topology

The coding agent must support two deployment topologies.

#### MVP Single-Node Topology

Use for local, research MVP, and early paper trading.

One machine runs:

- PostgreSQL;
- Redis;
- backend;
- dashboard;
- data workers;
- backtest workers;
- ML/RL workers;
- paper trading engine;
- reverse proxy.

Rules:

- Live trading is disabled by default.
- Resource-heavy jobs must have concurrency limits.
- Dashboard must show system load.
- Paper/live execution must have higher priority than research jobs.

#### Recommended Split Topology

Use for mature paper/live and ongoing research.

Production node:

- backend;
- dashboard;
- Redis or queue client;
- trading-engine-paper;
- trading-engine-live;
- light monitoring workers;
- light reporting workers;
- reverse proxy.

Research node:

- data lake;
- data download workers;
- backtest workers;
- walk-forward workers;
- ML workers;
- RL workers;
- heavy report workers.

Database/storage may be shared or replicated depending on deployment maturity.

Rules:

- Heavy research jobs must not run inside the live execution process.
- Production node must be able to continue safe trading or safe halt if research node is unavailable.
- Live engine must not depend on active ML/RL training jobs.
- Only promoted and frozen artifacts may be used by production.

### B.13 Resource Limits and Scheduling

Workers must have explicit concurrency limits.

Recommended initial concurrency:

- data worker: 1–4 jobs;
- backtest worker: 1–2 jobs;
- ML worker: 1 job;
- RL worker: 1 job;
- reports worker: 1–2 jobs;
- paper trading engine: dedicated process;
- live trading engine: dedicated process.

**Implemented per-worker concurrency (`src/jobs/worker.py`, `WORKER_CONCURRENCY` / `--concurrency`).** A worker process runs `concurrency` jobs in parallel (a thread pool over its served queues; each job still claims atomically + fences, so overlap is safe). Default 1 (classic one-job-at-a-time). The **`live` worker runs at `LIVE_WORKER_CONCURRENCY` (default 4)** so the dashboard-started CONTINUOUS sessions — `run_live_session` (the per-symbol promoted ensemble) + each `run_basket_paper_session` (one basket) — run **simultaneously** instead of head-of-line blocking on a single loop. Each has its own dashboard Stop (cooperative cancel); a duplicate Start for a strategy already running is refused. Transient-job workers stay at 1.

Rules:

- Live engine must not share a worker process with research jobs.
- CPU-heavy jobs must not starve websocket/event handling.
- Dashboard must show running job resource usage where available.
- The scheduler must avoid launching heavy jobs during live-critical windows if configured.

### B.14 Monitoring and Alerts Infrastructure

Minimum required alert channels:

- dashboard alert center;
- Telegram or equivalent push channel;
- email optional.

Required alerts:

- service unhealthy;
- exchange disconnected;
- websocket stale;
- data gap detected;
- job failed;
- gate failed;
- gate expired;
- live activation requested;
- live activation approved/rejected;
- live engine started;
- live engine stopped;
- kill switch triggered;
- order failed;
- stop placement failed;
- unknown order detected;
- position mismatch;
- abnormal slippage;
- drawdown limit reached;
- daily loss reached;
- model/config mismatch;
- backup failed;
- restore test failed;
- learner rollback;
- remediation task overdue.

Every alert must have:

- severity (critical / warning / info);
- timestamp;
- component;
- environment;
- run/session id if applicable;
- recommended action;
- dashboard link if available;
- escalation path (e.g., if unacknowledged in 15 min → escalate).

### B.15 Backup and Restore Requirements

Backups are mandatory before production live readiness.

Required backups:

- PostgreSQL daily backup;
- config/version backup;
- strategy/model/RL artifact backup;
- report artifact backup;
- data lake backup or explicit MVP-local-only exception;
- dashboard approval/audit log backup.

Rules:

- Backup without restore test does not pass the gate.
- Restore test must be runnable from dashboard as background job.
- Restore test report must be linked from Backup and Restore Gate.
- Production live activation is blocked if backup/restore gate fails.

### B.16 Infrastructure Implementation Deliverables

A coding agent implementing infrastructure must deliver:

- `docker-compose.yml`;
- `.env.example` with safe defaults;
- `Makefile` or task runner commands;
- database migrations;
- health check endpoints;
- basic dashboard shell;
- job queue implementation;
- job records and logs;
- gate records;
- remediation action records;
- initial infrastructure gates;
- backup script;
- restore test script;
- README with local setup;
- README with deployment setup;
- README with live safety checklist.

Required commands:

- `make setup`
- `make test`
- `make lint`
- `make typecheck`
- `make docker-up`
- `make docker-down`
- `make migrate`
- `make seed-dev`
- `make health`
- `make backup-db`
- `make restore-test`
- `make run-worker-data`
- `make run-worker-backtest`
- `make run-worker-ml`
- `make run-worker-rl`
- `make run-paper`
- `make run-gate GATE=<id>` (run a specific gate)
- `make run-all-gates` (run all gates in dependency order)
- `make kill` (CLI kill switch, independent of dashboard)

No command may start live trading unless it is explicitly named and protected.

If a live command is added, it must require explicit environment variables and a passed Live Activation Gate.

### B.17 Infrastructure Forbidden Shortcuts

Do not implement:

- single monolithic script that downloads data, trains models, and trades live;
- live execution inside the dashboard process;
- heavy backtests inside API request handlers;
- hidden threads for long-running jobs;
- untracked local files as datasets;
- unversioned model artifacts;
- unversioned dataset artifacts;
- live mode as default;
- real API keys in `.env.example`;
- dashboard without authentication in non-local environments;
- live trading without backups;
- live trading without health checks;
- live trading without alerting;
- live trading without order ownership checks;
- live trading without dashboard-visible gate status.

---

## Appendix C — Tech Stack & Pinned Dependencies

Concrete, opinionated choices so any coding agent builds the same system. Pin exact versions in a committed lockfile (`uv.lock` / `poetry.lock`); the majors below are the contract. GPU is **not** required (tabular ML/RL run on CPU).

**Language & tooling**
- Python **3.12+**. Dependency manager **uv** (preferred) or Poetry, with a committed lockfile and pinned hashes.
- Lint+format **ruff**; type-check **mypy** (`strict` on `risk/`, `execution/`, `adaptation/`); pre-commit hooks.
- Tests **pytest** + **pytest-asyncio** + **coverage**; **hypothesis** (property tests for risk/sizing math); **freezegun** (time). Coverage gate: ≥90% on `risk/`, `execution/`, `adaptation/`, `data/validation`.

**Exchange & data**
- **ccxt** (+ `ccxt.pro` for websockets) as the unified adapter, with a **native venue SDK fallback** (e.g. `pybit` for Bybit) wherever the unified layer cannot guarantee venue-specific behavior — specifically atomic exchange-side SL/TP attachment and native trailing stops (verified by `EXEC`/`LIVE` gates).
- **PostgreSQL 16 + TimescaleDB** (hypertables/continuous aggregates power per-symbol + time-range analytics). ORM **SQLAlchemy 2.x**; migrations **Alembic**.
- Optional fast columnar reads: **polars**; core math **numpy**/**pandas**; stats **statsmodels**, **scipy**.

**Compute, queue, scheduling**
- **Redis 7** for the independent kill-switch flag, ephemeral state, pub/sub, and as the task broker.
- Task queue **Dramatiq** or **RQ** (simple) — Celery only if needed; scheduler **APScheduler**. Live trading engine and paper engine run as **dedicated processes**, never inside a research worker or the API process (Appendix B).

**Quant / ML / RL**
- Indicators: **custom, single code path** shared by backtest & live (Parity Rule) — avoid library indicators that differ between modes.
- ML: **scikit-learn**, **XGBoost**/**LightGBM**/**CatBoost**, calibrated classifiers; experiment/model registry **MLflow** (or a versioned artifact store) keyed by `MODEL_VERSION`.
- RL (research/shadow only): **gymnasium** + **Stable-Baselines3** for prototyping; the bounded action head is custom (Section 21). Reward is risk-adjusted, cost-net.

**Service & ops**
- API/dashboard **FastAPI** + **Uvicorn**; UI **React + Vite** (or HTMX for a thinner build); charts **Recharts**/**Plotly**.
- Reverse proxy **Caddy** (auto-HTTPS); containers **Docker** + **Docker Compose**.
- Logging **structlog** (JSON); metrics **prometheus-client** + **Grafana** (optional); error tracking **Sentry** (optional).
- Auth on dashboard mandatory in non-local envs; prefer placing it behind a VPN (Tailscale/WireGuard). Host clock via **chrony/NTP**.
- CI **GitHub Actions**: runs `ruff`, `mypy`, `pytest`, and the leakage/data gates on every PR; `main` always green.

**Determinism & secrets**
- Pinned RNG seeds for backtests/training; dataset and model artifacts immutable and versioned.
- Secrets via `.env` (never committed) or a secrets manager; `.env.example` only with safe placeholders. API key scoped to trading-only, IP-whitelisted.

Any exchange-specific numeric (fee, tick, min-notional, funding interval) stays `[UNVERIFIED]` until confirmed against current docs, then `[VERIFIED]` (Conventions; `META` gate).

---

## Appendix D — Per-Phase Acceptance Criteria (Definition of Done)

Each phase is **done** only when **all** boxes below are checked. The Reviewer Agent verifies these against the diff, tests, and a **re-run of the listed gates** (it must not trust the Completion Report alone). Gate ids reference Appendix A; forbidden shortcuts reference Appendix B.17 / Section 30.

**Global DoD (applies to every phase):**
- [ ] All listed gates for the phase return `PASS` on a Reviewer re-run.
- [ ] New/changed code has tests; `pytest`, `ruff`, `mypy` green in CI; coverage thresholds met on critical modules.
- [ ] No forbidden shortcut introduced; no live-mode default; no secrets committed.
- [ ] All new runtime behavior is config-driven and versioned (Section 4).
- [ ] Completion Report (Appendix E) filled, linking exact versions and gate report paths.
- [ ] `docs/decisions/` updated for any assumption made under unspecified detail.

### Phase 1 — Infrastructure Foundation
- [ ] `docker-compose.yml` brings up postgres+timescale, redis, backend, dashboard-shell, worker, caddy; `make docker-up && make health` all green.
- [ ] Alembic migrations create base schema (jobs, job_logs, gates, gate_results, remediation_actions, audit/approvals); `make migrate` idempotent.
- [ ] Job orchestration skeleton: enqueue, progress, retry, cancel; job records + logs persisted.
- [ ] Dashboard shell with auth; health endpoints per service; CLI `make kill` works independent of dashboard.
- [ ] Exchange-adapter skeleton + `sync_exchange_metadata` job stub; universe-builder skeleton.
- [ ] Gates present & wired: `INFRA`, `DB`, `QUEUE`, `STORAGE`, `MON` (skeleton), `BACKUP` (skeleton).

### Phase 2 — Data Platform
- [ ] Ingestion for OHLCV (all TFs), mark/index, funding, OI, spread snapshots; historical downloader + incremental updater.
- [ ] Automatic gap detection + backfill (`scripts/backfill`); append-only, deduplicated, checksummed; versioned dataset snapshots.
- [ ] Data-validation report generated before runs.
- [ ] Gates `PASS`: `DATA-COV`, `DQ`.

### Phase 3 — Universe and Features
- [ ] Dynamic universe manager with filters + versioning; symbols entering/leaving logged.
- [ ] Feature pipeline = single code path (Parity Rule); only decision-time data; reproducible from snapshot.
- [ ] Leakage test: synthetic/shuffled data ⇒ ≈0 expectancy.
- [ ] Gates `PASS`: `UNIV`, `FEAT`, `META`.

### Phase 4 — Backtest Engine
- [ ] Event-based engine with fee/slippage/funding models tied to verified metadata; rejected-candidate logging; risk+execution simulation; report generator.
- [ ] Look-ahead / survivorship / future-universe guards enforced and tested.
- [ ] Gates `PASS`: `BT`, `WF`, `FEE`, `SLIP`.

### Phase 5 — Deterministic Quant Strategies
- [ ] At least families A, B, G implemented as research candidates with full hypothesis declaration (Section 12/13); per-strategy exit geometry correct; explicit initial SL for sizing.
- [ ] Each candidate produces a Strategy Report (Section 13) with per-symbol/regime/side breakdowns and fee/slippage stress.
- [ ] Long/short expectancy computed separately; losing side disabled.
- [ ] Gates `PASS`: `WF`, `FEE`, `SLIP` for each promoted candidate (or candidate is shelved per kill-criteria).

### Phase 6 — Ranking, Risk, and Execution Core
- [ ] Cross-symbol candidate ranking; risk manager with per-trade sizing, portfolio heat cap, net-beta cap, leverage-as-consequence, min-notional gate.
- [ ] Order builder + execution adapter with **atomic exchange-side SL/TP** and **native trailing**; reconciliation; order ownership (`ORDER_CLIENT_ID_PREFIX`).
- [ ] Kill switch (CLI + dashboard) and every circuit breaker verified by **deliberate forced-failure tests**.
- [ ] Gates `PASS`: `SETUP`, `RISK`, `EXEC`, `KILL`, `ORDER-OWN`.

### Phase 7 — Dashboard and Gate Workflow
- [ ] Aggregate **and** per-symbol stats with day/week/month/all-time/custom range; all metrics recompute per scope.
- [ ] Background Gate Runner UI: run single/group/all; live progress + logs; `GateResult` rendering with failure reason + remediation + re-run button.
- [ ] "Road to Live" view shows every gate, blocking criteria, and next action; approvals + audit logs; reports linked.
- [ ] Gates `PASS`: dashboard renders all existing gate results correctly (smoke gate `MON` for panels).

### Phase 8 — Paper Trading
- [ ] **Phase A (technical):** full pipeline runs in paper; simulated stops; kill switch + reconciliation exercised. Gate `PAPER-A` `PASS`.
- [ ] **Phase B (strategy):** sufficient candidate + executed paper trades; per-symbol/regime breakdown; paper-vs-backtest consistency reviewed. Gate `PAPER-B` `PASS`.
- [ ] No strategy advanced to live from Phase A alone.

### Phase 9 — Shadow ML
- [ ] Regime classifier, meta-labeling, execution-quality, strategy/symbol selector run **shadow-only**, logged to `shadow_log`; offline scoring vs deterministic baseline.
- [ ] No live influence. Gate `ML-PROMO` evaluated (may remain shadow-only if it fails to beat baseline).

### Phase 10 — ML Recommendation & Constrained Filtering
- [ ] ML may **block** candidates / recommend; cannot create trades, raise risk, or override risk manager.
- [ ] Promotion only if `ML-PROMO` `PASS`; otherwise stays shadow.

### Phase 11 — Online Learning Shadow
- [ ] Bounded learner in **shadow** (`learner_log`); drift + calibration monitoring; eligibility (21.x) met; frozen fallback exists and tested.
- [ ] `envelope_guard` rejects any out-of-box action (tested). Gate `LEARN-PROMO-S` evaluated.

### Phase 12 — RL Research & Shadow Policy
- [ ] RL env + risk-adjusted/cost-net reward; bounded action head; simulation training + stress tests; shadow policy logged.
- [ ] No live influence. RL Simulation + RL Shadow gates evaluated.

### Phase 13 — Controlled Live Readiness
- [ ] Frozen `CONFIG_VERSION` + strategy/model/RL versions; rollback plan; security checklist; backup **with tested restore**; monitoring/alerting verified end-to-end.
- [ ] Gates `PASS`: `SEC`, `DEPLOY`, `BACKUP`, `MON`, `CONFIG-FREEZE`, plus all upstream gates; finally `LIVE` `PASS`.
- [ ] Operator sign-off recorded (capital loseable; risk/leverage/drawdown confirmed). **"Go Live" remains a manual action.**

---

## Appendix E — Agent Workflow Artifacts (Orchestrator · Coder · Reviewer)

### E.1 The loop (authoritative protocol)
1. **Orchestrator** composes the Phase N prompt (E.4) and dispatches it to the **Coding Agent**.
2. Coding Agent implements Phase N, runs `make test`/`make lint`/`make typecheck` and the phase's gates, writes a **Completion Report** (E.2).
3. **Reviewer Agent** reads: this AGENTS.md, the Phase N prompt, the diff, the tests, the Completion Report — **and independently re-runs the phase gates** (`make run-gate GATE=<id>`); it never trusts the report alone.
4. Reviewer writes a **Review Report** (E.3): `PASS` / `FAIL` / `BLOCKED`, critical issues, required fixes, optional improvements, and the **exact acceptance criteria (Appendix D) / gate ids that failed**.
5. Coding Agent fixes **only** the required issues (no scope creep).
6. Reviewer re-verifies fixes and re-runs the affected gates.
7. **Human** approves advancing to Phase N+1 (mandatory at risk/execution/paper/live boundaries: Phases 6, 8, 13).

Hard rules: a phase cannot advance with any required gate not `PASS`; `BLOCKED` upstream gates block downstream automatically (Appendix A dependency graph); the Reviewer must be a **different model family** than the Coder (independence); the loop iterates Coder↔Reviewer until `PASS`, then stops for human approval.

### E.2 Completion Report template (`reports/phase_<N>/completion_report.md`)
```markdown
# Completion Report — Phase <N>: <name>
- Coder: <agent/model>            Date: <ts>
- Branch/commit: <hash>           CONFIG_VERSION: <v>  DATA_VERSION: <v>  ...
## Scope implemented
- <bullet list mapped to Roadmap deliverables>
## Acceptance criteria (Appendix D)
- [x] <criterion> — evidence: <path/log/test>
- [ ] <criterion> — NOT met because <reason>
## Gates run (self-check)
| gate_id | result | report_path |
|---------|--------|-------------|
## Tests
- pytest: <pass/fail, counts>  coverage(risk/exec/adaptation): <%>
- ruff: <ok>  mypy: <ok>
## Assumptions & decisions
- <links to docs/decisions/*>
## Known gaps / risks
- <list>
## Status / Tested? / Next step
- Status: <phase/module>   Tested?: <yes/partial/no>   Next step: <one line>
```

### E.3 Review Report template (`reports/phase_<N>/review_report.md`)
```markdown
# Review Report — Phase <N>: <name>
- Reviewer: <agent/model, different family than coder>   Date: <ts>
- Verdict: PASS | FAIL | BLOCKED
## Gate re-run (independent)
| gate_id | reviewer_result | matches_coder? | report_path |
|---------|-----------------|----------------|-------------|
## Critical issues (must fix to pass)
1. <issue> — violates: <Appendix D criterion / gate id / safety rule> — file:line
## Required fixes (exact, minimal)
- [ ] <actionable fix>
## Exact acceptance criteria failed
- <Appendix D items / gate ids>
## Optional improvements (non-blocking)
- <list>
## Evidence reviewed
- diff: <hash>  tests: <paths>  completion_report: <path>
## Decision rationale
- <why PASS/FAIL/BLOCKED, tied to Priority Stack & gates>
```

### E.4 Phase Prompt template (orchestrator composes per phase)
The orchestrator assembles each Phase N prompt **deterministically** from existing sections, so prompts never drift from the spec:
```
SYSTEM: You are the Coding Agent. Obey AGENTS.md exactly. Priority Stack (Section 1) wins all conflicts.
CONTEXT (attach): full AGENTS.md.
TASK: Implement Phase <N> — <name>.
DELIVERABLES: <Roadmap Section 32 Phase N "Deliver:" list>.
ACCEPTANCE CRITERIA: <Appendix D Phase N checklist> + Global DoD.
GATES TO PASS: <Appendix A ids for this phase> (run via `make run-gate`).
CONSTRAINTS: Forbidden Work (Section 30) + Infrastructure Forbidden Shortcuts (Appendix B.17). Capital-agnostic. No live defaults.
OUTPUT: complete files at correct paths + git commands; then a Completion Report (Appendix E.2). End with Status / Tested? / Next step.
STOP CONDITION: all listed gates PASS and acceptance criteria met; do not start Phase <N+1>.
```
A worked example for Phase 1 lives at `phase_prompts/phase_01.md`; the remaining phase prompts are generated from this template + Roadmap + Appendix D at dispatch time.

### E.5 Reviewer independence & anti-gaming safeguards
- Reviewer **re-runs gates itself**; a green Completion Report with a red Reviewer gate run ⇒ `FAIL`.
- Reviewer checks that tests actually exercise the acceptance criteria (not trivially passing); weak/Tautological tests ⇒ required fix.
- Reviewer confirms no forbidden shortcut and no envelope weakening (diff `risk_envelope`, `envelope_guard`, breakers, stop placement).
- For Phases 6/8/13, Reviewer output is advisory to the **human** approver, who holds final sign-off.

---

*Output format for the agent: complete files with paths matching the structure, plus the git commands for the commit. Each delivery ends with **Status** (phase/module), **Tested?**, and **Next step** (one line). No design essays unless asked. Never claim or imply future profitability; most retail bots lose money.*
