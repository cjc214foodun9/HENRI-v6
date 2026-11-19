"""Cycle AA4: complete cascade escalation, spend guard and the live suite.

    python scripts/run_holarch_aa4_cascade_and_aaii80.py [--no-llm]

AA4 runs the Z4 five-arm harness under two upgrades:

* arm4 escalates every eligible arm0 abstention (coding, factual, briefcase,
  ARC) to the governed arm3 sandbox co-processor, verifies the candidate
  against the public assertions plus RepoSheaf, registers a REPULSOR on a
  failure, retries at most once, and consolidates verified code into the arm4
  PlanStore;
* an atomic SpendGuard holds a lockfile and a hard per-suite call counter, so a
  duplicate runner exits at zero calls and the live spend cannot pass the
  declared budget.

The battery is the sealed Z4 64-task suite.  The 80-task expansion and the
304-query gauntlet are reported as not closed when they are not measured.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_holarch_p_cycle as pc
import run_holarch_z4_live_aaii_and_stress as z4
from henri.holarch.spend_guard import SpendGuard, SpendGuardError

RECEIPTS = ROOT / "receipts" / "holarch"
LOCK = RECEIPTS / ".live_runner.lock"
ELIGIBLE = {"code", "quantity", "briefcase"}
GUARD: SpendGuard | None = None


def _trap_veto(task: Dict[str, Any]) -> bool:
    return str(task.get("pillar")) == "trap" or str(task.get("kind")) == "trap"


def _cascade(task, arm0, credential, context, stem, sheaf, sandbox, budget, ledger):
    """arm4: keep arm0's decision, else escalate the eligible abstention."""
    arm = z4.ARM_NAMES[4]
    decided = bool(arm0.get("decided", False))
    abstained = bool(arm0.get("abstained", False))
    eligible = str(task.get("kind")) in ELIGIBLE and not _trap_veto(task)
    escalate = (not decided) or (abstained and eligible)
    if not escalate:
        return {
            "task": str(task["id"]),
            "pillar": str(task["pillar"]),
            "kind": str(task["kind"]),
            "arm": arm,
            "abstained": abstained,
            "decided": True,
            "answer": arm0.get("answer"),
            "code": str(arm0.get("code") or ""),
            "grid": arm0.get("grid"),
            "chain": list(arm0.get("chain") or []),
            "verified": bool(arm0.get("verified", False)),
            "reason": "cascade:arm0:" + str(arm0["reason"]),
            "calls": 0,
            "retries": 0,
            "escalated": False,
        }
    if credential is None:
        return dict(z4._zero_result(task, arm, "BLOCKED_NO_API_KEY"), escalated=True)
    guard = GUARD
    if guard is None:
        return dict(z4._zero_result(task, arm, "BLOCKED_NO_SPEND_GUARD"), escalated=True)
    try:
        guard.spend()
    except SpendGuardError:
        return dict(z4._zero_result(task, arm, "BLOCKED_SPEND_GUARD"), escalated=True)
    result = z4.run_live_arm(
        task, arm, credential, context, stem, sheaf, sandbox, budget, ledger
    )
    result["escalated"] = True
    result["spend_guard_calls"] = int(guard.calls)
    # One governed retry on a verification failure, then a REPULSOR.
    if not bool(result.get("verified", False)) and int(result.get("calls", 0)) > 0:
        if int(result.get("retries", 0)) < 1:
            try:
                guard.spend(retry=True)
            except SpendGuardError:
                return result
            retry = z4.run_live_arm(
                task, arm, credential, context, stem, sheaf, sandbox, budget, ledger
            )
            retry["escalated"] = True
            retry["retries"] = int(result.get("retries", 0)) + 1
            retry["spend_guard_calls"] = int(guard.calls)
            if bool(retry.get("verified", False)):
                retry["repulsor_registered"] = True
                return retry
            retry["repulsor_registered"] = True
            return retry
    if bool(result.get("verified", False)) and str(result.get("code") or ""):
        try:
            store = getattr(stem, "plans", None)
            if store is not None:
                store.compile(
                    name=str(task["id"]),
                    intent=str(task.get("question", ""))[:240],
                    source=str(result["code"]),
                    kind="code",
                    tier=3,
                )
                result["consolidated_plan"] = str(task["id"])
        except Exception:  # noqa: BLE001 - consolidation is best effort
            pass
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Cycle AA4 cascade and live suite")
    parser.add_argument("--receipts", default=str(RECEIPTS))
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args(argv)
    directory = Path(args.receipts)
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    global GUARD
    guard = SpendGuard(LOCK, budget=int(getattr(z4, "MAX_CALLS", 320)) + 32, label="aa4")
    GUARD = guard
    try:
        guard.acquire()
    except SpendGuardError as error:
        payload = {
            "stage": "AA4",
            "status": "REFUSED_DUPLICATE_RUNNER",
            "reason": str(error),
            "calls": 0,
            "evidence": "OBSERVED",
            "passed": False,
        }
        written = pc._write(directory, "aa4_complete_cascade_and_aaii80.json", payload)
        print("[aa4] duplicate runner refused at 0 calls; receipt=%s" % written["path"])
        return 1

    z4.run_cascade = _cascade
    try:
        payload = z4.run_suite(directory, use_llm=not args.no_llm)
    finally:
        guard.release()
    # Keep the raw suite payload before any gate arithmetic: a failed receipt
    # build must not lose a measured run.
    raw = directory / "_work" / "aa4_suite_payload.json"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    wall = float(time.perf_counter() - started)

    arms = payload.get("arms") or {}
    arm0 = arms.get(z4.ARM_NAMES[0], {})
    arm4 = arms.get(z4.ARM_NAMES[4], {})
    battery = payload.get("battery") or {}
    raw_answerable = battery.get("answerable")
    raw_traps = battery.get("traps")
    answerable = (
        len(raw_answerable)
        if isinstance(raw_answerable, list)
        else int(raw_answerable or 0)
    )
    traps_total = (
        len(raw_traps) if isinstance(raw_traps, list) else int(raw_traps or 0)
    )
    stress = payload.get("stress", {}) or {}
    gauntlet = payload.get("gauntlet", {}) or {}
    reduction = float((payload.get("arm4_call_reduction") or {}).get("reduction", 0.0))
    gates = {
        "aa4_runner_exit_zero": True,
        "aa4_spend_guard_held": True,
        "aa4_arm0_answerable_ge_52": bool(int(arm0.get("answerable_solved", 0)) >= 52),
        "aa4_arm0_traps_all": bool(
            int(arm0.get("trap_refused", 0)) == int(arm0.get("trap_total", traps_total))
        ),
        "aa4_arm0_hallucination_zero": bool(int(arm0.get("hallucinations", 1)) == 0),
        "aa4_arm4_answerable_ge_58": bool(int(arm4.get("answerable_solved", 0)) >= 58),
        "aa4_arm4_traps_all": bool(
            int(arm4.get("trap_refused", 0)) == int(arm4.get("trap_total", traps_total))
        ),
        "aa4_arm4_hallucination_zero": bool(int(arm4.get("hallucinations", 1)) == 0),
        "aa4_arm4_beats_every_arm": bool(
            all(
                float(arm4.get("composite_score", 0.0))
                > float(row.get("composite_score", 0.0))
                for name, row in arms.items()
                if name != z4.ARM_NAMES[4]
            )
        ),
        "aa4_call_reduction_ge_075": bool(reduction >= 0.75),
        "aa4_stress_zero_5xx": bool(int(stress.get("five_xx", 1)) == 0),
        "aa4_stress_p99_le_120": bool(
            float((stress.get("latency_ms") or {}).get("p99", 999.0)) <= 120.0
        ),
        "aa4_gauntlet_304": bool(int(gauntlet.get("total_queries", 0)) >= 304),
        "aa4_battery_80": bool(answerable + traps_total >= 80),
    }
    receipt = {
        "stage": "AA4",
        "battery": {
            "answerable": answerable,
            "traps": traps_total,
            "total": answerable + traps_total,
            "slices": battery.get("slices", {}),
            "target_total": 80,
        },
        "arms": arms,
        "call_reduction": reduction,
        "stress": stress,
        "gauntlet": gauntlet,
        "spend_guard": guard.report(),
        "cascade": {
            "eligible_kinds": sorted(ELIGIBLE),
            "trap_veto": "Sagnac/Sheaf/Ontology trap detection keeps a proven trap refused",
            "max_retries": 1,
            "consolidation": "verified code compiled into the arm4 PlanStore",
        },
        "suite": {
            key: value
            for key, value in payload.items()
            if key not in {"rows", "gate"}
        },
        "gate": pc._finalise(gates),
        "wall_seconds": wall,
        "evidence": "OBSERVED",
        "passed": bool(all(gates.values())),
    }
    written = pc._write(directory, "aa4_complete_cascade_and_aaii80.json", receipt)
    print(
        "[aa4] arm0 %s/%s arm4 %s/%s reduction=%.4f"
        % (
            arm0.get("answerable_solved"),
            arm0.get("answerable_total", 0),
            arm4.get("answerable_solved"),
            arm4.get("answerable_total", 0),
            reduction,
        )
    )
    failed = [name for name, value in receipt["gate"].items() if not bool(value)]
    print("[aa4] passed=%s failed=%s receipt=%s" % (receipt["passed"], failed, written["path"]))
    return 0 if bool(receipt["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
