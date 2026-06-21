"""Demo-account readiness gate (AGENTS.md Section 6/7/13/17/35).

A single pre-flight check that answers one question: *is it safe to run real demo/testnet
orders right now?* It composes the individual safety controls built across the demo-readiness
review into one verdict — PASS / FAIL / BLOCKED — with a per-check explanation:

* kill switch        — must be disengaged;
* order ownership    — the clientOrderId prefix + bot instance id must be configured (Section 7);
* risk caps          — the risk envelope + concurrency caps must be bounded and sane (Section 17);
* exchange metadata  — verified, exchange-matched spec for every symbol (Section 6);
* TP/SL capability   — the venue must support both a stop and a take-profit/trailing exit;
* strategy eligibility — at least one strategy validated on REAL lake data (Section 13);
* reconciliation     — the real exchange book must be clean (no foreign/manual order/position).

Verdict precedence: any FAIL → FAIL (a misconfiguration to fix); else any BLOCKED → BLOCKED
(a precondition not yet met — do the work, then re-check); else PASS. This guard NEVER places
an order and never enables live trading; it only reports readiness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.config import Settings, get_settings

PASS = "PASS"
FAIL = "FAIL"
BLOCKED = "BLOCKED"


@dataclass(frozen=True, slots=True)
class ReadinessCheck:
    name: str
    status: str  # PASS | FAIL | BLOCKED
    detail: str

    def to_dict(self) -> dict:
        return {"name": self.name, "status": self.status, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class DemoReadinessReport:
    verdict: str  # PASS | FAIL | BLOCKED
    environment: str
    checks: tuple[ReadinessCheck, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return self.verdict == PASS

    def blockers(self) -> list[ReadinessCheck]:
        return [c for c in self.checks if c.status != PASS]

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "environment": self.environment,
            "checks": [c.to_dict() for c in self.checks],
        }

    def report(self) -> str:
        lines = [f"Demo readiness [{self.environment}] — {self.verdict}"]
        for c in self.checks:
            mark = {PASS: "✓", FAIL: "✗", BLOCKED: "■"}.get(c.status, "?")
            lines.append(f"  {mark} {c.status:<7} {c.name}: {c.detail}")
        if self.verdict != PASS:
            lines.append(
                "  → NOT ready for demo execution. Resolve the FAIL/BLOCKED items above and "
                "re-run the readiness check."
            )
        return "\n".join(lines)


def _worst(statuses: list[str]) -> str:
    if FAIL in statuses:
        return FAIL
    if BLOCKED in statuses:
        return BLOCKED
    return PASS


class DemoReadinessGuard:
    """Composes the demo-safety controls into one PASS/FAIL/BLOCKED verdict + report."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        kill_switch: Any | None = None,
        venue: Any | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._kill_switch = kill_switch
        self._venue = venue
        self.environment = self.settings.exchange_env

    def evaluate(self) -> DemoReadinessReport:
        checks = [
            self._check_environment(),
            self._check_kill_switch(),
            self._check_ownership_prefix(),
            self._check_risk_caps(),
            self._check_metadata(),
            self._check_tp_sl_capability(),
            self._check_strategy_eligibility(),
            self._check_reconciliation(),
        ]
        verdict = _worst([c.status for c in checks])
        return DemoReadinessReport(
            verdict=verdict, environment=self.environment, checks=tuple(checks)
        )

    # -- individual checks ---------------------------------------------- #
    def _check_environment(self) -> ReadinessCheck:
        """This is the DEMO gate; real-money live is authorised by the LiveActivationGuard."""
        if self.environment == "live":
            return ReadinessCheck(
                "environment", FAIL,
                "EXCHANGE_ENV=live — the demo readiness guard does NOT authorise real-money "
                "trading; live requires the LiveActivationGuard (gates + sign-off).",
            )
        if self.environment not in ("demo", "testnet"):
            return ReadinessCheck(
                "environment", FAIL, f"EXCHANGE_ENV={self.environment!r} is not demo/testnet"
            )
        return ReadinessCheck("environment", PASS, f"virtual-funds environment '{self.environment}'")

    def _check_kill_switch(self) -> ReadinessCheck:
        from src.killswitch import KillSwitch

        ks = self._kill_switch or KillSwitch(self.settings)
        if ks.engaged():
            return ReadinessCheck("kill_switch", FAIL, "kill switch is ENGAGED — trading halted")
        return ReadinessCheck("kill_switch", PASS, "disengaged")

    def _check_ownership_prefix(self) -> ReadinessCheck:
        from src.execution.ownership import OwnershipPolicy

        own = OwnershipPolicy(self.settings)
        if not own.configured():
            return ReadinessCheck(
                "order_ownership", FAIL,
                "ORDER_CLIENT_ID_PREFIX / BOT_INSTANCE_ID not configured — orders would be "
                "un-attributable (Section 7)",
            )
        return ReadinessCheck(
            "order_ownership", PASS, f"clientOrderId prefix '{own.prefix}' (instance {own.bot_instance_id})"
        )

    def _check_risk_caps(self) -> ReadinessCheck:
        from src.risk.config import load_risk_config
        from src.risk.envelope import HARD_CEILINGS

        cfg = load_risk_config()
        env = cfg.envelope
        problems: list[str] = []
        if cfg.max_concurrent_per_symbol < 1 or cfg.max_concurrent_total < 1:
            problems.append("concurrency caps must be >= 1")
        if not (0 < env.max_risk_pct_per_trade <= HARD_CEILINGS["max_risk_pct_per_trade"]):
            problems.append("max_risk_pct_per_trade out of bounds")
        if not (0 < env.portfolio_heat_cap <= HARD_CEILINGS["portfolio_heat_cap"]):
            problems.append("portfolio_heat_cap out of bounds")
        if not (0 < env.net_beta_btc_cap <= HARD_CEILINGS["net_beta_btc_cap"]):
            problems.append("net_beta_btc_cap out of bounds")
        if not (1 <= env.max_leverage <= HARD_CEILINGS["max_leverage"]):
            problems.append("max_leverage out of bounds")
        if problems:
            return ReadinessCheck("risk_caps", FAIL, "; ".join(problems))
        return ReadinessCheck(
            "risk_caps", PASS,
            f"risk≤{env.max_risk_pct_per_trade:.1%}/trade heat≤{env.portfolio_heat_cap:.0%} "
            f"lev≤{env.max_leverage:g}x perSymbol={cfg.max_concurrent_per_symbol} "
            f"total={cfg.max_concurrent_total}",
        )

    def _meta(self):
        from src.exchange.metadata import load_metadata_for

        return load_metadata_for(self.settings.exchange_id)

    def _active_symbols(self) -> list[str]:
        meta = self._meta()
        try:
            from src.data.config import load_data_config

            syms = load_data_config().active_symbols()
        except Exception:  # noqa: BLE001 - fall back to the metadata's symbols
            syms = []
        return syms or meta.symbols()

    def _check_metadata(self) -> ReadinessCheck:
        meta = self._meta()
        blockers = []
        for sym in self._active_symbols():
            b = meta.tradable_blocker(sym, exchange_id=self.settings.exchange_id)
            if b is not None:
                blockers.append(f"{sym}: {b}")
        if blockers:
            return ReadinessCheck("exchange_metadata", BLOCKED, "; ".join(blockers))
        return ReadinessCheck(
            "exchange_metadata", PASS,
            f"verified spec '{meta.metadata_version}' for {self.settings.exchange_id}",
        )

    def _check_tp_sl_capability(self) -> ReadinessCheck:
        ot = set(self._meta().supported_order_types)
        has_sl = bool(ot & {"stop_market", "stop_limit"})
        has_tp = bool(ot & {"take_profit_market", "trailing_stop"})
        if has_sl and has_tp:
            return ReadinessCheck(
                "tp_sl_capability", PASS, "venue supports exchange-resident stop + take-profit/trailing"
            )
        missing = []
        if not has_sl:
            missing.append("stop")
        if not has_tp:
            missing.append("take-profit/trailing")
        return ReadinessCheck(
            "tp_sl_capability", BLOCKED,
            f"venue metadata lacks order types for: {missing} — cannot attach mandatory SL/TP",
        )

    def _check_strategy_eligibility(self) -> ReadinessCheck:
        from src.strategies.promotion import active_strategy_ids, reference_only_active_ids

        ver = self.settings.strategy_version
        eligible = active_strategy_ids(ver, require_real_data=True)
        blocked = reference_only_active_ids(ver)
        if eligible:
            note = f" (blocked reference-only: {blocked})" if blocked else ""
            return ReadinessCheck(
                "strategy_eligibility", PASS,
                f"{len(eligible)} strategy(ies) validated on real lake data: {eligible}{note}",
            )
        if blocked:
            return ReadinessCheck(
                "strategy_eligibility", BLOCKED,
                f"all active promotions are reference-only ({blocked}); re-validate on real lake "
                "data before demo/live",
            )
        return ReadinessCheck(
            "strategy_eligibility", BLOCKED,
            "no promoted strategy is active — validate + promote at least one on real lake data",
        )

    def _check_reconciliation(self) -> ReadinessCheck:
        if self._venue is None:
            return ReadinessCheck(
                "reconciliation", BLOCKED,
                "exchange book not yet reconciled — runs automatically at session start; provide "
                "a connected venue to confirm a clean book up front",
            )
        from src.execution.ownership import OwnershipPolicy
        from src.execution.reconciliation import reconcile_startup

        res = reconcile_startup(
            self._venue, OwnershipPolicy(self.settings), environment=self.environment, adopt=False
        )
        if res.halt_required:
            return ReadinessCheck(
                "reconciliation", BLOCKED,
                f"foreign/manual items on the exchange: orders={list(res.foreign_orders)} "
                f"positions={list(res.foreign_positions)} — halt until the book is clean",
            )
        return ReadinessCheck(
            "reconciliation", PASS,
            f"clean book (owned orders={len(res.owned_orders)} positions={len(res.owned_positions)})",
        )


__all__ = ["DemoReadinessGuard", "DemoReadinessReport", "ReadinessCheck", "PASS", "FAIL", "BLOCKED"]
