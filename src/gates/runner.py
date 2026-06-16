"""Gate Runner (AGENTS.md Section 25).

Runs a gate (or all gates in dependency order), resolving upstream dependencies
first: a gate whose dependencies are not all ``PASS`` is reported ``BLOCKED``
(Appendix A dependency rules; Appendix B.10). Every run is persisted as a
``GateResult`` and, on any non-PASS verdict, ordered ``RemediationAction`` rows
are created from ``configs/gates.yaml`` — a failed gate is never a dead end.

The module is also the CLI invoked by the orchestrator:

    python -m src.gates.runner --gate INFRA --json
    python -m src.gates.runner --all --json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime

from src.config import Settings, get_settings
from src.db.base import session_scope
from src.db.models import (
    Gate,
    GateResult,
    GateStatus,
    RemediationAction,
    RemediationStatus,
)
from src.gates import checks
from src.gates.catalog import GateSpec, load_catalog
from src.gates.result import GateRunResult, GateVerdict
from src.observability import configure_logging

_VERDICT_TO_STATUS = {
    GateVerdict.PASS: GateStatus.PASSED,
    GateVerdict.FAIL: GateStatus.FAILED,
    GateVerdict.BLOCKED: GateStatus.BLOCKED,
    GateVerdict.NOT_RUN: GateStatus.NOT_RUN,
}


class GateRunner:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.catalog: dict[str, GateSpec] = load_catalog()
        self._seed_catalog()

    # -- catalog seeding ------------------------------------------------- #
    def _seed_catalog(self) -> None:
        with session_scope() as session:
            for spec in self.catalog.values():
                gate = session.get(Gate, spec.gate_id)
                if gate is None:
                    session.add(
                        Gate(
                            gate_id=spec.gate_id,
                            name=spec.name,
                            phase=spec.phase,
                            depends_on=spec.depends_on,
                            blocks_live=spec.blocks_live,
                            pass_condition=spec.pass_condition,
                            remediation_steps=spec.remediation_steps,
                        )
                    )
                else:
                    gate.name = spec.name
                    gate.phase = spec.phase
                    gate.depends_on = spec.depends_on
                    gate.blocks_live = spec.blocks_live
                    gate.pass_condition = spec.pass_condition
                    gate.remediation_steps = spec.remediation_steps

    # -- running --------------------------------------------------------- #
    def run(self, gate_id: str) -> GateRunResult:
        return self._evaluate(gate_id, {})

    def run_all(self) -> list[GateRunResult]:
        cache: dict[str, GateRunResult] = {}
        results = []
        for gate_id in self._topological_order():
            results.append(self._evaluate(gate_id, cache))
        return results

    def _topological_order(self) -> list[str]:
        order: list[str] = []
        seen: set[str] = set()

        def visit(gid: str) -> None:
            if gid in seen:
                return
            seen.add(gid)
            for dep in self.catalog.get(gid, GateSpec(gid, gid, "")).depends_on:
                if dep in self.catalog:
                    visit(dep)
            order.append(gid)

        for gid in self.catalog:
            visit(gid)
        return order

    def _evaluate(self, gate_id: str, cache: dict[str, GateRunResult]) -> GateRunResult:
        if gate_id in cache:
            return cache[gate_id]

        spec = self.catalog.get(gate_id)
        if spec is None:
            result = GateRunResult(gate_id, GateVerdict.NOT_RUN.value, note="unknown gate id")
            cache[gate_id] = result
            return result

        # Resolve dependencies first (Appendix A dependency rules).
        blocking: list[str] = []
        for dep in spec.depends_on:
            dep_result = self._evaluate(dep, cache)
            if dep_result.overall != GateVerdict.PASS.value:
                blocking.append(f"{dep}={dep_result.overall}")

        if blocking:
            result = self._blocked(spec, blocking)
            cache[gate_id] = result
            return result

        if not checks.has_check(gate_id):
            result = GateRunResult(
                gate_id,
                GateVerdict.NOT_RUN.value,
                note=f"no check implemented for {gate_id} (introduced after Phase 1)",
            )
            self._persist(spec, result, GateStatus.NOT_RUN)
            cache[gate_id] = result
            return result

        try:
            criteria = checks.run_check(gate_id, self.settings)
        except Exception as exc:  # noqa: BLE001
            result = GateRunResult(
                gate_id,
                GateVerdict.FAIL.value,
                note=f"gate check raised: {type(exc).__name__}: {exc}",
            )
            self._persist(spec, result, GateStatus.FAILED)
            cache[gate_id] = result
            return result

        overall = GateVerdict.PASS if all(c.passed for c in criteria) else GateVerdict.FAIL
        failed = [c.name for c in criteria if not c.passed]
        result = GateRunResult(
            gate_id,
            overall.value,
            criteria=[{"name": c.name, "status": c.status, "detail": c.detail} for c in criteria],
            note="" if overall is GateVerdict.PASS else f"failed criteria: {failed}",
        )
        self._persist(spec, result, _VERDICT_TO_STATUS[overall])
        cache[gate_id] = result
        return result

    def _blocked(self, spec: GateSpec, blocking: list[str]) -> GateRunResult:
        result = GateRunResult(
            spec.gate_id,
            GateVerdict.BLOCKED.value,
            note=f"blocked by upstream gate(s): {blocking}",
        )
        self._persist(spec, result, GateStatus.BLOCKED, blocking=blocking)
        return result

    # -- persistence ----------------------------------------------------- #
    def _persist(
        self,
        spec: GateSpec,
        result: GateRunResult,
        status: GateStatus,
        *,
        blocking: list[str] | None = None,
    ) -> None:
        report_path = self._write_report(result)
        result.report_path = str(report_path)
        with session_scope() as session:
            gr = GateResult(
                gate_id=spec.gate_id,
                status=status,
                started_at=datetime.now(UTC),
                finished_at=datetime.now(UTC),
                run_by="gate-runner",
                environment=self.settings.app_env.value,
                criteria=result.criteria,
                failure_reason=result.note or None,
                report_path=str(report_path),
                related_versions=self.settings.versions(),
            )
            session.add(gr)
            session.flush()

            # Remediation actions on any non-PASS verdict (Section 0 / B.9).
            if status is not GateStatus.PASSED:
                if blocking:
                    session.add(
                        RemediationAction(
                            gate_result_id=gr.id,
                            gate_id=spec.gate_id,
                            step_index=0,
                            description=("Resolve upstream gate(s) first: " + ", ".join(blocking)),
                            status=RemediationStatus.OPEN,
                            recommended_job="run_upstream_gates",
                        )
                    )
                for idx, step in enumerate(spec.remediation_steps):
                    session.add(
                        RemediationAction(
                            gate_result_id=gr.id,
                            gate_id=spec.gate_id,
                            step_index=idx + (1 if blocking else 0),
                            description=step,
                            status=RemediationStatus.OPEN,
                            recommended_job=spec.rerun_job or None,
                        )
                    )

    def _write_report(self, result: GateRunResult) -> str:
        gates_dir = self.settings.reports_path / "gates"
        gates_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        path = gates_dir / f"{result.gate_id}_{stamp}.json"
        path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        return str(path)


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description="Quant bot Gate Runner")
    parser.add_argument("--gate", help="gate id to run (e.g. INFRA)")
    parser.add_argument("--all", action="store_true", help="run all gates in dependency order")
    parser.add_argument("--json", action="store_true", help="emit JSON to stdout")
    args = parser.parse_args(argv)

    runner = GateRunner()
    payload: list[dict[str, object]] | dict[str, object]
    if args.all:
        results = runner.run_all()
        payload = [r.to_dict() for r in results]
    elif args.gate:
        payload = runner.run(args.gate).to_dict()
    else:
        parser.error("one of --gate <id> or --all is required")
        return 2

    if args.json:
        # Clean JSON on stdout (logs go to stderr).
        print(json.dumps(payload, indent=2))
    else:
        rows = payload if isinstance(payload, list) else [payload]
        for item in rows:
            print(f"{item['gate_id']}: {item['overall']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
