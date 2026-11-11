"""Cycle X verification: shared stem, wave egress, one-cell closure,
vectorised Pass-0 and the full ledger reconciliation.

    python scripts/run_holarch_x_cycle.py --stages x1,x2,x3,x4,x5

X1 runs the four-arm live AAII v5.0 suite over one SharedCorpusStem.
X2 measures the contrastive wave egress on the W2 held-out split.
X3 runs the 12/12 one-cell closure audit.
X4 runs the 256-query fast gauntlet.
X5 re-seals the t, u, v and w ledgers and the x aggregate on a clean tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_holarch_p_cycle as pc

RECEIPTS = ROOT / "receipts" / "holarch"


def _run(name: str, command: List[str], timeout: int = 7200) -> Dict[str, Any]:
    began = time.perf_counter()
    completed = subprocess.run(
        command, cwd=str(ROOT), capture_output=True, text=True, timeout=timeout
    )
    return {
        "name": str(name),
        "command": " ".join(command),
        "exit_code": int(completed.returncode),
        "wall_seconds": float(time.perf_counter() - began),
        "stdout_tail": (completed.stdout or "")[-1500:],
        "stderr_tail": (completed.stderr or "")[-1500:],
    }


def _read(directory: Path, name: str) -> Dict[str, Any]:
    path = directory / str(name)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _digest(directory: Path, name: str) -> str:
    path = directory / str(name)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


# ---------------------------------------------------------------------- X1
def stage_x1(directory: Path) -> Dict[str, Any]:
    started = time.perf_counter()
    run = _run("x1", [sys.executable, "scripts/run_holarch_x1_live_four_arm.py"])
    sealed = _read(directory, "x1_live_four_arm_aaii_v5.json")
    gate = dict(sealed.get("gate") or {})
    gates = {
        "x1_runner_exit_zero": bool(int(run["exit_code"]) == 0),
        "x1_receipt_present": bool(sealed),
        "x1_shared_stem_once": bool(gate.get("x1_shared_stem_once", False)),
        "x1_task_arm_suite_executed": bool(gate.get("x1_task_arm_suite_executed", False)),
        "x1_arm3_gt_arm2": bool(gate.get("x1_arm3_gt_arm2", False)),
        "x1_arm3_gt_arm1": bool(gate.get("x1_arm3_gt_arm1", False)),
        "x1_arm3_hallucination_zero": bool(gate.get("x1_arm3_hallucination_zero", False)),
        "x1_arm0_hallucination_zero": bool(gate.get("x1_arm0_hallucination_zero", False)),
        "x1_secret_leak_zero": bool(gate.get("x1_secret_leak_zero", False)),
        "x1_budget_respected": bool(gate.get("x1_budget_respected", False)),
    }
    return {
        "stage": "X1",
        "run": {key: run[key] for key in ("exit_code", "wall_seconds", "stderr_tail")},
        "shared_stem": dict(sealed.get("shared_stem") or {}),
        "arms": dict(sealed.get("arms") or {}),
        "live_calls": int(sealed.get("live_calls", 0)),
        "budget": dict(sealed.get("budget") or {}),
        "security": dict(sealed.get("security") or {}),
        "gate": pc._finalise(gates),
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
        "passed": bool(all(gates.values())),
    }


# ---------------------------------------------------------------------- X2
def stage_x2(directory: Path) -> Dict[str, Any]:
    started = time.perf_counter()
    run = _run("x2", [sys.executable, "scripts/run_holarch_x2_wave_egress.py"])
    sealed = _read(directory, "x2_sovereign_wave_egress.json")
    gate = dict(sealed.get("gate") or {})
    gates = {
        "x2_runner_exit_zero": bool(int(run["exit_code"]) == 0),
        "x2_receipt_present": bool(sealed),
        "x2_wave_egress_module": bool(gate.get("x2_wave_egress_module", False)),
        "x2_autograd_unused": bool(gate.get("x2_autograd_unused", False)),
        "x2_sagnac_residual_reduced": bool(gate.get("x2_sagnac_residual_reduced", False)),
        "x2_unclamped_overrides_present": bool(gate.get("x2_unclamped_overrides_present", False)),
        "x2_override_net_win_positive": bool(gate.get("x2_override_net_win_positive", False)),
        "x2_top1_beats_floor": bool(gate.get("x2_top1_beats_floor", False)),
        "x2_bpc_beats_floor": bool(gate.get("x2_bpc_beats_floor", False)),
    }
    return {
        "stage": "X2",
        "run": {key: run[key] for key in ("exit_code", "wall_seconds", "stderr_tail")},
        "split": dict(sealed.get("split") or {}),
        "wave_egress": dict(sealed.get("wave_egress") or {}),
        "attribution": dict(sealed.get("attribution") or {}),
        "gate": pc._finalise(gates),
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
        "passed": bool(all(gates.values())),
    }


# ---------------------------------------------------------------------- X3
def stage_x3(directory: Path) -> Dict[str, Any]:
    started = time.perf_counter()
    run = _run("x3", [sys.executable, "scripts/run_holarch_x3_turing_closure.py"])
    sealed = _read(directory, "x3_twelve_of_twelve_turing.json")
    gate = dict(sealed.get("gate") or {})
    gates = {
        "x3_runner_exit_zero": bool(int(run["exit_code"]) == 0),
        "x3_receipt_present": bool(sealed),
        "x3_tasks_solved_12": bool(gate.get("x3_tasks_solved_12", False)),
        "x3_generalized_12": bool(gate.get("x3_generalized_12", False)),
        "x3_length_generalized_12": bool(gate.get("x3_length_generalized_12", False)),
        "x3_contradictions_4": bool(gate.get("x3_contradictions_4", False)),
        "x3_carry_add_4_of_4": bool(gate.get("x3_carry_add_4_of_4", False)),
        "x3_delimiter_copy_4_of_4": bool(gate.get("x3_delimiter_copy_4_of_4", False)),
        "x3_task_relabelled_false": bool(gate.get("x3_task_relabelled_false", False)),
        "x3_max_nodes_le_320000": bool(gate.get("x3_max_nodes_le_320000", False)),
        "x3_strict_one_cell": bool(gate.get("x3_strict_one_cell", False)),
        "x3_text_tokens_zero": bool(gate.get("x3_text_tokens_zero", False)),
    }
    return {
        "stage": "X3",
        "run": {key: run[key] for key in ("exit_code", "wall_seconds", "stderr_tail")},
        "solved": int(sealed.get("solved", 0)),
        "v1_length_generalized": int(sealed.get("v1_length_generalized", 0)),
        "contradictions_failed_closed": int(sealed.get("contradictions_failed_closed", 0)),
        "max_nodes_used": int(sealed.get("max_nodes_used", 0)),
        "strict_one_cell": dict(sealed.get("strict_one_cell") or {}),
        "tasks": list(sealed.get("tasks") or []),
        "gate": pc._finalise(gates),
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
        "passed": bool(all(gates.values())),
    }


# ---------------------------------------------------------------------- X4
def stage_x4(directory: Path) -> Dict[str, Any]:
    started = time.perf_counter()
    run = _run("x4", [sys.executable, "scripts/run_holarch_agent_eval.py",
                      "--out", str(directory / "agent_eval.json")])
    payload = _read(directory, "agent_eval.json")
    gate = dict(payload.get("gate") or {})
    mcp = dict(payload.get("holarch_mcp") or {})
    rows = list((payload.get("probes") or {}).get("rows") or [])
    x4_rows = [row for row in rows if str(row.get("name", "")).startswith("x4_")]
    gates = {
        "x4_runner_exit_zero": bool(int(run["exit_code"]) == 0),
        "x4_queries_256": bool(int(payload.get("total_queries", 0)) == 256),
        "x4_probe_slots_180": bool(int(payload.get("probe_slots", 0)) == 180),
        "x4_reachable_1": bool(float(mcp.get("reachable_accuracy", 0.0) or 0.0) == 1.0),
        "x4_hallucination_0": bool(mcp.get("hallucination_rate") == 0.0),
        "x4_zero_query_reads": bool(int(mcp.get("query_file_reads", -1)) == 0),
        "x4_receipt_chain_valid": bool(mcp.get("receipt_chain_valid", False)),
        "x4_wall_le_160": bool(float(payload.get("wall_seconds", 1e9)) <= 160.0),
        "x4_gauntlet_gate": bool(gate.get("passed", False)),
        "x4_pass0_batch_under_50ms": bool(gate.get("x4_pass0_batch_under_50ms", False)),
        "x4_pass0_scan_under_1500ms": bool(gate.get("x4_pass0_scan_under_1500ms", False)),
        "x4_counterfactual_switch_preserved": bool(
            gate.get("x4_counterfactual_switch_preserved", False)),
        "x4_step_latency_median_under_1ms": bool(
            gate.get("x4_step_latency_median_under_1ms", False)),
        "x4_x_probes_present": bool(
            len(x4_rows) == 4 and all(bool(row.get("correct")) for row in x4_rows)),
    }
    return {
        "stage": "X4",
        "run": {key: run[key] for key in ("exit_code", "wall_seconds", "stderr_tail")},
        "queries": int(payload.get("total_queries", 0)),
        "probe_slots": int(payload.get("probe_slots", 0)),
        "wall_seconds_gauntlet": float(payload.get("wall_seconds", 0.0)),
        "headroom_seconds": float(180.0 - float(payload.get("wall_seconds", 0.0))),
        "reachable_accuracy": float(mcp.get("reachable_accuracy", 0.0) or 0.0),
        "hallucination_rate": float(mcp.get("hallucination_rate", 1.0) or 0.0),
        "query_file_reads": int(mcp.get("query_file_reads", -1)),
        "x4_probes": x4_rows,
        "category_ms": dict(payload.get("category_ms") or {}),
        "gate": pc._finalise(gates),
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
        "passed": bool(all(gates.values())),
    }


# ---------------------------------------------------------------------- X5
def stage_x5(directory: Path) -> Dict[str, Any]:
    started = time.perf_counter()
    ledger: List[Dict[str, Any]] = []
    ledger.append(_run("t_cycle", [sys.executable, "scripts/run_holarch_t_cycle.py",
                                   "--stages", "t1,t2,t3,t4"]))
    ledger.append(_run("u_cycle", [sys.executable, "scripts/run_holarch_u_cycle.py",
                                   "--stages", "u1,u2,u3,u4,u5"]))
    # W2 owns the split hygiene AND the full pytest process, so one command
    # produces both the W2 receipt and the pytest evidence V5 reuses.
    ledger.append(_run("w_cycle_w2", [sys.executable, "scripts/run_holarch_w2_hygiene.py"]))
    ledger.append(_run("v_cycle", [sys.executable, "scripts/run_holarch_v_cycle.py",
                                   "--stages", "v1,v2,v3,v4"]))
    ledger.append(_run("consistency", [sys.executable, "scripts/verify_holarch_consistency.py"]))
    pytest_block = dict(_read(directory, "w2_pytest_and_split_hygiene.json").get("pytest") or {})
    evidence = [
        {"name": "t_cycle", "command": ledger[0]["command"],
         "exit_code": int(ledger[0]["exit_code"]),
         "wall_seconds": float(ledger[0]["wall_seconds"]),
         "stdout_tail": ledger[0]["stdout_tail"], "stderr_tail": ledger[0]["stderr_tail"]},
        {"name": "u_cycle", "command": ledger[1]["command"],
         "exit_code": int(ledger[1]["exit_code"]),
         "wall_seconds": float(ledger[1]["wall_seconds"]),
         "stdout_tail": ledger[1]["stdout_tail"], "stderr_tail": ledger[1]["stderr_tail"]},
        {"name": "pytest", "command": "python -m pytest tests -q",
         "exit_code": int(pytest_block.get("exit_code", 1)),
         "wall_seconds": float(pytest_block.get("wall_seconds", 0.0)),
         "stdout_tail": str(pytest_block.get("stdout_tail", ""))[-800:],
         "stderr_tail": str(pytest_block.get("stderr_tail", ""))[-800:]},
        {"name": "consistency", "command": ledger[4]["command"],
         "exit_code": int(ledger[4]["exit_code"]),
         "wall_seconds": float(ledger[4]["wall_seconds"]),
         "stdout_tail": ledger[4]["stdout_tail"], "stderr_tail": ledger[4]["stderr_tail"]},
    ]
    work = directory / "_work"
    work.mkdir(parents=True, exist_ok=True)
    (work / "x5_commands.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    # V5 now reads the evidence it just measured.
    ledger.append(_run("v_cycle_v5", [sys.executable, "scripts/run_holarch_v_cycle.py",
                                      "--stages", "v5"]))
    # W2's stage is a read of the receipt the command above just sealed, so
    # running all five stages re-seals the aggregate entry with the fresh
    # sha256 without paying for a second pytest process.
    ledger.append(_run("w_cycle", [sys.executable, "scripts/run_holarch_w_cycle.py",
                                   "--stages", "w1,w2,w3,w4,w5"]))
    t_cycle = _read(directory, "t_cycle.json")
    u_cycle = _read(directory, "u_cycle.json")
    v_cycle = _read(directory, "v_cycle.json")
    w_cycle = _read(directory, "w_cycle.json")
    gates = {
        "x5_t_cycle_passed": bool(t_cycle.get("passed", False)),
        "x5_u_cycle_passed": bool(u_cycle.get("passed", False)),
        "x5_v_cycle_passed": bool(v_cycle.get("passed", False)),
        "x5_w_cycle_passed": bool(w_cycle.get("passed", False)),
        "x5_consistency_exit_zero": bool(int(ledger[4]["exit_code"]) == 0),
        "x5_pytest_exit_zero": bool(int(pytest_block.get("exit_code", 1)) == 0),
        "x5_pytest_zero_failures": bool(int(pytest_block.get("failures", 1)) == 0),
        "x5_v5_exit_zero": bool(int(ledger[5]["exit_code"]) == 0),
        "x5_w_exit_zero": bool(int(ledger[6]["exit_code"]) == 0),
        "x5_all_five_ledgers": bool(
            all(bool(ledger_row.get("passed", False))
                for ledger_row in (t_cycle, u_cycle, v_cycle, w_cycle))),
    }
    return {
        "stage": "X5",
        "ledger": ledger,
        "t_cycle_passed": bool(t_cycle.get("passed", False)),
        "u_cycle_passed": bool(u_cycle.get("passed", False)),
        "v_cycle_passed": bool(v_cycle.get("passed", False)),
        "w_cycle_passed": bool(w_cycle.get("passed", False)),
        "pytest": {key: pytest_block.get(key) for key in ("exit_code", "failures", "wall_seconds")},
        "gate": pc._finalise(gates),
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
        "passed": bool(all(gates.values())),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Cycle X verification")
    parser.add_argument("--stages", default="x1,x2,x3,x4,x5")
    parser.add_argument("--receipts", default=str(RECEIPTS))
    args = parser.parse_args(argv)
    directory = Path(args.receipts)
    directory.mkdir(parents=True, exist_ok=True)
    stages = [token.strip().lower() for token in str(args.stages).split(",") if token.strip()]
    # The stage receipts are separate files: the detailed measurement receipt
    # (x1_live..., x2_sovereign..., x3_twelve...) is written by its runner and
    # must not be overwritten by the stage view.
    runners = {
        "x1": ("x1_stage_four_arm.json", stage_x1),
        "x2": ("x2_stage_wave_egress.json", stage_x2),
        "x3": ("x3_stage_turing_closure.json", stage_x3),
        "x4": ("x4_gauntlet_256_queries.json", stage_x4),
        "x5": ("x5_full_ledger_reconciliation.json", stage_x5),
    }
    existing: Dict[str, Any] = {}
    aggregate_path = directory / "x_cycle.json"
    if aggregate_path.exists():
        try:
            existing = json.loads(aggregate_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            existing = {}
    kept = {key: value for key, value in (existing.get("stages") or {}).items()
            if key not in set(stages)}
    aggregate: Dict[str, Any] = {"cycle": "X", "head": pc._git_head(), "stages": dict(kept)}
    exit_code = 0
    for stage in stages:
        if stage not in runners:
            continue
        name, runner = runners[stage]
        print(f"[x-cycle] {stage} ...", flush=True)
        payload = json.loads(json.dumps(runner(directory), default=float))
        written = pc._write(directory, name, payload)
        aggregate["stages"][stage] = {
            "receipt": written["path"],
            "sha256": written["sha256"],
            "passed": bool(payload.get("gate", {}).get("passed", False)),
            "gate": payload.get("gate", {}),
            "wall_seconds": payload.get("wall_seconds", 0.0),
        }
        print(f"[x-cycle] {stage} passed={payload.get('gate', {}).get('passed')} "
              f"{written['path']}", flush=True)
        if not bool(payload.get("gate", {}).get("passed", False)):
            exit_code = 1
    aggregate["passed"] = bool(
        all(bool(item.get("passed")) for item in aggregate["stages"].values())
    ) if aggregate["stages"] else False
    written = pc._write(directory, "x_cycle.json", aggregate)
    print(f"[x-cycle] aggregate {written['path']} passed={aggregate['passed']}", flush=True)
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
