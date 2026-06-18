# Training the ML and RL / Online-Learning layers

This explains **when**, **how**, and **with what data** the learning components are trained,
and the guardrails that keep them inside the Immutable Risk Envelope (AGENTS.md §2.2). The
single rule that overrides everything: **learning may only act *inside* the envelope — it can
never widen, disable, or exceed a risk limit, at any maturity stage.**

Everything below is **shadow-first**: a model/policy is trained and scored against frozen
kill-criteria *before* it is allowed to influence anything, and even then only through
bounded, audited actions. Promotion is staged and each step requires a passing gate plus
manual sign-off.

> Status note. The learning *machinery* (models, trainers, scorers, gates, guards) is fully
> implemented and gate-verified. Two things are deliberately **synthetic/prototype** today and
> are the first items to wire to real data before live use — see [Known gaps](#known-gaps).

---

## 1. Supervised ML (the "shadow" layer)

### 1.1 What the models predict
Five small sklearn models (`src/ml/models/`, configured in `configs/ml.yaml`):

| Model | Predicts | Estimator |
|---|---|---|
| **Meta-labeler** (primary) | take (1) / skip (0) a deterministic candidate | LogisticRegression |
| **Regime classifier** | market-regime class | RandomForestClassifier |
| **Exec-quality** | good / poor execution conditions | LogisticRegression |
| **Strategy-selector** | preferred / not-preferred strategy | LogisticRegression |
| **Symbol-ranker** | preferred / not-preferred symbol | LogisticRegression |

The meta-labeler is the one that matters: it only ever *filters* (says "skip") deterministic
candidates — it can never create a trade, size up, or override a hard blocker.

### 1.2 What data trains them
- **Features** (`src/ml/features.py`) — the same causal features the strategies use (feature
  parity / Parity Rule): `signal_strength, expected_edge_frac, spread_bps, slippage_est` off
  the candidate, plus `atr_pct, premium, funding_z, rv_short, ret_1` from the feature pipeline.
  The *exact same* `candidate_to_row()` path is used for training and inference, so there is no
  train/serve skew.
- **Labels** (`src/ml/labels.py`) — `label = 1 if realized_pnl > 0 else 0`, where `realized_pnl`
  is the normalised R-multiple outcome of that candidate.
- **Sample population** — should be **realised paper-trade outcomes** (the candidate + what
  actually happened when it was — or would have been — taken). ⚠️ Today `build_ml_dataset`
  uses a deterministic **synthetic** dataset (`build_reference_dataset()`); wiring it to the
  `shadow_log` / `paper_trades` tables is the main pre-live task (see [Known gaps](#known-gaps)).
- **Split** — seeded 75/25 train/test (`train_test_split(..., seed=42)`); evaluation is on the
  held-out test split only.

### 1.3 When training runs
- **Trigger:** manual / on-demand via the job queue or the gate run — there is **no scheduler**.
  Enqueue the jobs (dashboard → Jobs, or `make`/CLI) or run the **ML-PROMO** gate, which drives
  the pipeline. Retrain whenever you have a meaningfully larger/newer batch of realised paper
  outcomes, after a strategy/feature version bump, or on a regime shift.
- **Cadence (recommended once wired to real data):** retrain on a rolling window of recent
  paper outcomes; re-score against the baseline before any promotion.
- **Always shadow:** `shadow.applied_to_live = false` (`configs/ml.yaml`). The gate
  `ml_no_live_influence` **fails** if any `ShadowLog.applied == True` exists, so a model that is
  not yet promoted can never affect a live decision.

### 1.4 How — the pipeline and promotion bar
Pipeline (job handlers, run in order; consumed by the **ml** worker):
`build_ml_dataset → train_ml_models → evaluate_ml_models → run_ml_shadow_pass →
run_ml_recommendation_pass → run_ml_filter_evaluation`.

The **ML-PROMO** gate (`src/gates/phase9.py`/`phase10.py`) promotes a model only if it clears
every kill-criterion (thresholds in `configs/ml.yaml`, scored by `src/ml/scorer.py`):

| Criterion | Threshold |
|---|---|
| Expectancy improvement over the always-take baseline | ≥ 0 |
| Profit factor preserved (ratio vs baseline) | ≥ 1.0 |
| Tail loss not worsened | ≤ 1.0 |
| Best trades not over-filtered (top-10 removed) | ≤ 20% |
| **Leakage check** — improvement on *shuffled-label* noise | ≤ 0.10 R |

Promotion also requires a registered, **manually-reviewed** model artifact
(`src/ml/registry.py`). Versioning: `model_version: ml_shadow_0001`, `ml_stage` in `ml.yaml`.

The four lifecycle stages a model passes through: **(1)** trained & evaluated → **(2)** SHADOW
(logged, `applied=False`) → **(3)** RECOMMEND (`applied=False`) → **(4)** CONSTRAINED_FILTER
(may *block* a candidate, never create/size one; `min_confidence_to_take: 0.40`).

---

## 2. RL / Online learning (the "adaptation" layer)

Two pieces: an offline-trained bounded **policy** (`src/rl/` training env + `src/adaptation/`
controller/guards) and **online** learners that update from realised outcomes.

### 2.1 What it decides (bounded actions only)
A policy outputs a `BoundedAction`: a **size bucket** ∈ {0, 0.25, 0.5, 1.0}, a **take** flag, and
an **exec style** — plus optional small strategy-weight nudges. Hard bounds
(`configs/adaptation.yaml`, `src/adaptation/action_space.py`): `strategy_weight ∈ [0, 2]`,
`max_change_per_update = 0.10`, `max_change_rate = 0.25`. Out-of-bound actions are clamped or
rejected. Policies: `RLPolicy` (offline linear, trained by CEM), `OnlineLogRegPolicy`
(`SGDClassifier.partial_fit` per outcome), `GaussianTSBandit` (Thompson sampling).

### 2.2 What data / environment trains it
- **Offline RL policy** — trained in a **simulation** (`src/rl/environment.py`, a Gymnasium env;
  reward in `src/rl/reward.py` is cost-net with drawdown/tail/envelope-proximity penalties,
  clipped to ±10). The trainer is Cross-Entropy-Method (`src/rl/trainer.py`). The env is
  **never** connected to live data/exchange; it generates synthetic signals across stress modes
  (`normal | no_edge | high_vol | toxic`). So RL training data = **simulation**, not live history.
- **Online learners** — update from **realised outcomes of their own shadow decisions**.
  `controller.record_outcome()` forwards the realised result to `policy.update()` *even in
  SHADOW/RECOMMEND*, so the learner trains continuously while it is guaranteed the action was
  never applied. Decisions + outcomes are logged to `learner_logs` and can be replayed offline.

### 2.3 When — the staged promotion ladder
Mode path **SHADOW → RECOMMEND → LIVE_BOUNDED**, manual approval at each step
(`configs/adaptation.yaml`, `mode: SHADOW`, `min_samples_to_start: 50`). Gates:

| Gate | What it proves |
|---|---|
| **RL-SIM** (`phase12`) | env loads, full episode no crash, reward finite & bounded, every action passes the envelope guard, CEM training completes, trained policy emits valid SHADOW actions |
| **RL-SHADOW** (`phase12`) | all RL decisions are `mode=SHADOW`, every `learner_log.applied == False`, envelope-touching actions are rejected |
| **LEARN-PROMO-S** (`phase11`) | eligibility (≥ min samples, frozen fallback exists, leakage-safe) + shadow beats baseline on walk-forward **and** the locked hold-out + bounded-only + calibrated |
| **LEARN-PROMO-L** (`phase13`) | a real recommendation track record, rollback tested, **independent** learner kill switch, `auto_freeze_on_breaker` enforced, frozen-fallback revert verified |

Scorer thresholds (`src/adaptation/scorer.py`, mirrored in `adaptation.yaml`): walk-forward
folds (`min_wf_folds_positive`), `min_holdout_edge ≥ 0.0`, `calibration_max_brier ≤ 0.30`,
`max_drift_per_window`, baseline mean.

### 2.4 The guards (why this is safe)
- **envelope_guard** (`src/adaptation/envelope_guard.py`) — loads the envelope read-only from
  `configs/risk.yaml`; a frozenset of `_FORBIDDEN_TUNABLES` (leverage, risk %, heat cap, beta
  cap, daily/drawdown limits, stop placement, …) is **hard-blocked**. Any action that touches a
  forbidden key is rejected; size is clamped to envelope risk — regardless of what
  `adaptation.yaml` says.
- **RollbackGuard** (`src/adaptation/rollback.py`) — `auto_freeze_on_breaker` is immutable-True;
  triggers (envelope breaker, unsafe regime, divergence > 0.20, underperformance, manual kill)
  **freeze** the learner and revert to a frozen fallback snapshot.
- **Independent kill switch** — the learner's freeze is separate from the trading kill switch.
- Versions: `learner_version: learner_0001` (`adaptation.yaml`); frozen fallbacks are immutable,
  checksum-named snapshots under `var/adaptation/`.

---

## 3. Quick how-to

```bash
# Run the ML promotion pipeline (trains, evaluates, shadow-scores against the kill-criteria):
make run-gate GATE=ML-PROMO          # or enqueue the build/train/evaluate jobs individually

# RL simulation training + shadow verification:
make run-gate GATE=RL-SIM
make run-gate GATE=RL-SHADOW

# Online-learner promotion checks:
make run-gate GATE=LEARN-PROMO-S     # shadow -> recommend readiness
make run-gate GATE=LEARN-PROMO-L     # recommend -> bounded-live readiness (Phase 13)
```

All of the above run **shadow-only** and cannot affect live trading; promotion to any
applied mode additionally requires the gate to pass **and** explicit human approval.

---

## Known gaps

These are implemented-but-not-yet-real and should be wired before relying on the learned layer
in live:

1. **ML training data is synthetic.** `build_ml_dataset` calls `build_reference_dataset()`
   (deterministic synthetic samples). Point it at realised paper-trade outcomes
   (`shadow_log` / `paper_trades`) before promoting any model on real edge.
2. **RL trainer is prototype-grade.** The Cross-Entropy-Method `LinearRLTrainer` is intentionally
   simple (no PyTorch/GPU); it can be replaced by an SB3/PyTorch trainer without changing the
   bounded-action contract or the guards.
3. **No scheduler.** Training/promotion is manual (job queue / gate run). Add a scheduled trigger
   if you want periodic retraining once (1) is wired to real data.
