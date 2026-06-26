# Project context for Claude Code
The single source of truth is **AGENTS.md** in this repo root. Read it fully and obey it.
Priority Stack (Section 1) resolves every conflict. Never enable live trading.
Per-phase tasks come from the orchestrator (phase prompts) + Appendix D acceptance criteria.

As-built implementation state is kept in AGENTS.md inline, marked **Implemented** in the relevant
sections (current `STRATEGY_VERSION strat_0007`): the per-strategy execution model (Section 12),
the refined walk-forward gate — directional folds + economic hold-out + deflated-Sharpe floor
(Section 16), per-strategy `risk_scale` (Section 17), maker/bracket/time-stop backtest↔live parity
(Section 18), and the event engine's fill/exit model + MAE/MFE (Section 19). Config: `configs/strategies.yaml`, `configs/backtest.yaml`.
