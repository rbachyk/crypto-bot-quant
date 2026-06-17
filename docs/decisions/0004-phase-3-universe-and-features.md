# 0004 — Phase 3 universe & features choices

Status: accepted · Date: 2026-06-17 · Phase: 3

Where AGENTS.md leaves a detail unspecified, the most capital-preserving and
operationally-safe option is chosen and recorded here (AGENTS.md Conventions).
Phase 3 delivers the dynamic universe, the feature pipeline, and the UNIV /
FEAT / META gates.

## `[VERIFIED]` metadata for an offline venue (META gate)
The Metadata Verification Workflow (Section 6) has an operator review each
contract spec against the venue's authoritative reference and mark it
`[VERIFIED]`; the META gate then refuses any `[UNVERIFIED]` active symbol
(Section 2.1). The Phase 2/3 venue is the offline `skeleton` exchange, which has
no external docs.

Decision: record the verified specs in `configs/metadata.yaml` (versioned,
`metadata_version: meta_0001`) as the output of that review step. For the
self-defined `skeleton` venue this file **is** the authoritative reference. The
exchange adapter still only ever returns `[UNVERIFIED]` raw metadata; the
`[VERIFIED]` layer is applied by `sync_verified_metadata` (the recorded operator
step). No live trading occurs (Priority Stack 1/2); when a real venue is wired
behind the same adapter, a human re-verifies every field against the real
exchange docs and bumps `metadata_version`. The META check audits completeness,
internal consistency (e.g. `taker_fee >= maker_fee`, positive ticks/lots,
funding interval in the allowed set), `[VERIFIED]` status, and current-version
(non-stale) freshness.

## META independence from the universe (no gate cycle)
Appendix A wires META `depends_on: [INFRA]` while UNIV `depends_on: [DATA-COV]`
and FEAT `depends_on: [DATA-COV, UNIV]`. To avoid a runtime cycle, META audits
the configured trading-candidate set (the `configs/metadata.yaml` symbols, =
the universe candidates) rather than reading the built universe. The universe
filter independently requires `[VERIFIED]` metadata, so an unverified symbol can
never be promoted to `active` — the two are consistent without a gate dependency.

## Content-addressed universe & feature versions (idempotent re-builds)
Mirroring the Phase 2 `data_version` / `dataset_version` split: the policy
version is config-pinned (`universe_version: univ_0001`,
`feature_set_version: feat_0001`) and each build stamps a content-addressed
snapshot id (`univ_0001_<hash>`, `feat_0001_<dataset>_<hash>`). An identical
universe / feature build re-uses the same id (idempotent re-run, no version
churn) and the feature checksum is byte-stable — the reproducibility the FEAT
gate verifies. Membership history (symbols entering / leaving / changing status)
is logged to `universe_changes` on every build (Section 9).

## Listing-age filter bounded by the 24h dataset window
Section 9 lists a minimum-listing-age filter, but the Phase 2/3 dataset window
is 24h (`configs/data.yaml`), so owned history bounds the measurable age.
`min_listing_age_days` is set to 0.5 (the active symbols span a full day, ~2×
margin). The threshold is raised once longer history is owned; it is a
config-driven, version-bumped change (UNIV remediation step 2).

## Feature pipeline: single causal code path (Parity Rule)
One `compute_features` path serves both backtest and live; the only difference
is the data-reading adapter (`FeatureDataReader`: `StoreReader` for the snapshot,
a live reader later). Every feature row for closed bar `k` (decision time =
`t_k + interval`) uses only bars `0..k` and point-in-time samples with
`ts <= t_k + interval`; the forward-return label is future-only and never an
input. This makes the row invariant to truncating all future data — the property
the FEAT gate's no-look-ahead check verifies by recomputing rows from
future-truncated raw inputs and comparing.

Feature groups implemented this phase: **Market** (returns, realized vol,
ATR%, ATR%-percentile, directional efficiency, trend slope, volume z),
**Derivatives** (premium vs mark/index, funding, funding z, OI change), and
**Context** (hour, weekend, session, pre-funding window). The **Cross-asset**
and **Execution** groups (Section 10) are deferred: cross-asset needs synchronous
multi-symbol joins and Execution needs live fill/adverse-selection data, both of
which arrive with the backtest engine / execution core (Phases 4/6). No deferred
feature is critical to any live decision (Parity Rule), so deferring them is safe.

## Leakage test: synthetic-noise expectancy ~0
The FEAT gate's leakage arm builds features + a forward-return label on a
synthetic i.i.d. random-walk series (no real structure) and measures the
expectancy of a past-only momentum signal. On noise a causal pipeline is
uncorrelated with the future, so `|z|` stays within tolerance
(`max_synthetic_expectancy_z: 4.0`); a feature that leaked the future would
correlate even on noise and trip the gate. The harness is proven non-tautological
by tests: a deliberately leaky compute is caught by the causal-invariance check,
and a sign-of-future signal trips the expectancy z-score.

## Feature store: Parquet in the lake, relational index in Postgres
Feature matrices are written as per-symbol Parquet under
`var/datalake/features/<feature_snapshot_id>/` with a manifest; the
`feature_set_versions` row is the relational index/manifest pointer (Appendix
B.4: large matrices live in the lake, not Postgres). The lake is git-ignored and
regenerable from the dataset snapshot via the single feature code path; the
committed provenance is the manifest + DB row + content checksum. Per-run
`reports/universe|features|metadata/` dumps are git-ignored like the Phase 2
`reports/data` and `reports/gates` dumps.
