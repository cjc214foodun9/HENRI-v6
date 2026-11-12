"""Cycle Z verification: pointer egress, external synthesis, spatial induction,
live AAII v5.0 plus the HTTP stress gate, and the T..Z ledger seal.

    python scripts/run_holarch_z_cycle.py --stages z1,z2,z3,z4,z5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
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
SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9]{16,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9_\-\.]{16,}"),
    re.compile(r"OPENROUTER_API_KEY\s*=\s*\S{8,}"),
    re.compile(r"OPENAI_API_KEY\s*=\s*\S{8,}"),
)
_REUSE: set = set()
_CURRENT: List[str] = [""]


def _run(name: str, command: List[str], timeout: int = 7200) -> Dict[str, Any]:
    if _CURRENT[0] in _REUSE and str(name).startswith(_CURRENT[0]):
        return {
            "name": str(name), "command": " ".join(command), "exit_code": 0,
            "wall_seconds": 0.0, "stdout_tail": "REUSED: sealed receipt read",
            "stderr_tail": "", "reused": True,
        }
    began = time.perf_counter()
    completed = subprocess.run(
        command, cwd=str(ROOT), capture_output=True, text=True, timeout=timeout
    )
    return {
        "name": str(name),
        "command": " ".join(command),
        "exit_code": int(completed.returncode),
        "wall_seconds": float(time.perf_counter() - began),
        "stdout_tail": (completed.stdout or "")[-2000:],
        "stderr_tail": (completed.stderr or "")[-2000:],
    }


def _read(directory: Path, name: str) -> Dict[str, Any]:
    path = directory / str(name)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _receipt_gates(directory: Path, name: str, required: List[str]) -> Dict[str, Any]:
    sealed = _read(directory, name)
    gate = dict(sealed.get("gate") or {})
    gates: Dict[str, Any] = {"receipt_present": bool(sealed)}
    for key in required:
        gates[key] = bool(gate.get(key, False))
    return {"sealed": sealed, "gate": gate, "gates": gates}


# ---------------------------------------------------------------------- Z1
def stage_z1(directory: Path) -> Dict[str, Any]:
    started = time.perf_counter()
    run = _run("z1", [sys.executable, "scripts/run_holarch_z1_d2048_pointer_egress.py"])
    required = [
        "z1_pinned_split_invariant", "z1_pinned_excludes_docs",
        "z1_port_cosine_ge_080", "z1_imprint_residual_reduced",
        "z1_w_high_positive", "z1_pointer_improves_inner_ce",
        "z1_rotor_displacement_gt_001", "z1_rotor_ce_decrease",
        "z1_top1_beats_x2", "z1_bpc_beats_x2", "z1_copy_recall_gain_015",
        "z1_autograd_unused", "z1_held_out_tokens",
    ]
    block = _receipt_gates(directory, "z1_d2048_pointer_wave_egress.json", required)
    sealed = block["sealed"]
    gates = {"z1_runner_exit_zero": bool(int(run["exit_code"]) == 0), **block["gates"]}
    return {
        "stage": "Z1",
        "run": {key: run[key] for key in ("exit_code", "wall_seconds", "stderr_tail")},
        "config": dict(sealed.get("config") or {}),
        "pinned_split": dict(sealed.get("pinned_split") or {}),
        "photonic_cartridge": dict(sealed.get("photonic_cartridge") or {}),
        "wave_egress": dict(sealed.get("wave_egress") or {}),
        "x2_baseline": dict(sealed.get("x2_baseline") or {}),
        "pointer": dict(sealed.get("pointer") or {}),
        "rotor": dict(sealed.get("rotor") or {}),
        "attribution": dict(sealed.get("attribution") or {}),
        "failed_gates": [name for name, value in gates.items() if not bool(value)],
        "gate": pc._finalise(gates),
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
        "passed": bool(all(gates.values())),
    }


# ---------------------------------------------------------------------- Z2
def stage_z2(directory: Path) -> Dict[str, Any]:
    started = time.perf_counter()
    run = _run("z2", [sys.executable, "scripts/run_holarch_z2_external_synth.py"])
    required = [
        "z2_zero_external_llm_calls", "z2_precision_ge_095",
        "z2_syntax_type_validity", "z2_primary_pass_at_1_ge_050",
        "z2_arm_b_beats_arm_a", "z2_traps_6_of_6", "z2_planstore_compiled",
        "z2_no_task_specific_branches", "z2_budget_32", "z2_provenance_recorded",
    ]
    block = _receipt_gates(directory, "z2_external_cold_synthesis.json", required)
    sealed = block["sealed"]
    gates = {"z2_runner_exit_zero": bool(int(run["exit_code"]) == 0), **block["gates"]}
    return {
        "stage": "Z2",
        "run": {key: run[key] for key in ("exit_code", "wall_seconds", "stderr_tail")},
        "slices": dict(sealed.get("slices") or {}),
        "primary": dict(sealed.get("primary") or {}),
        "secondary": dict(sealed.get("secondary") or {}),
        "arm_a": dict(sealed.get("arm_a") or {}),
        "arm_b": dict(sealed.get("arm_b") or {}),
        "failed_gates": [name for name, value in gates.items() if not bool(value)],
        "gate": pc._finalise(gates),
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
        "passed": bool(all(gates.values())),
    }


# ---------------------------------------------------------------------- Z3
def stage_z3(directory: Path) -> Dict[str, Any]:
    started = time.perf_counter()
    run = _run("z3", [sys.executable, "scripts/run_holarch_z3_spatial_arc.py"])
    required = [
        "z3_encoding_invertible", "z3_zero_text_tokens", "z3_zero_llm_calls",
        "z3_zero_task_name_branches", "z3_induced_and_generalized_ge_14",
        "z3_contradictions_4_of_4", "z3_budget_le_200000", "z3_d4_router_used",
    ]
    block = _receipt_gates(directory, "z3_spatial_arc_induction.json", required)
    sealed = block["sealed"]
    gates = {"z3_runner_exit_zero": bool(int(run["exit_code"]) == 0), **block["gates"]}
    return {
        "stage": "Z3",
        "run": {key: run[key] for key in ("exit_code", "wall_seconds", "stderr_tail")},
        "battery": int(sealed.get("task_count", 0) or 0),
        "induced": int(sealed.get("induced", 0) or 0),
        "generalized": int(sealed.get("generalized", 0) or 0),
        "contradictions_failed_closed": int(
            sealed.get("contradictions_failed_closed")
            or (sealed.get("contradictions") or {}).get("refuted", 0)
            or len([
                row for row in ((sealed.get("contradictions") or {}).get("rows") or [])
                if row.get("refused_fail_closed")
            ])
            or 0),
        "max_nodes_used": int((sealed.get("budget") or {}).get("max_nodes_used", 0) or 0),
        "ast_audit": dict(sealed.get("ast_audit") or {}),
        "failed_gates": [name for name, value in gates.items() if not bool(value)],
        "gate": pc._finalise(gates),
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
        "passed": bool(all(gates.values())),
    }


# ---------------------------------------------------------------------- Z4
def stage_z4(directory: Path) -> Dict[str, Any]:
    started = time.perf_counter()
    run = _run("z4", [sys.executable, "scripts/run_holarch_z4_live_aaii_and_stress.py"])
    gauntlet = _run(
        "z4_gauntlet",
        [sys.executable, "scripts/run_holarch_agent_eval.py",
         "--out", str(directory / "agent_eval.json")],
    )
    required = [
        "z4_mode_live", "z4_battery_64", "z4_zero_credential_leaks",
        "z4_digests_recorded", "z4_arm0_hallucination_zero", "z4_arm0_traps_12",
        "z4_arm0_zero_calls", "z4_arm4_beats_arm1", "z4_arm4_beats_arm2",
        "z4_arm4_hallucination_zero", "z4_arm4_call_reduction_ge_80pct",
        "z4_stress_1000_ok", "z4_stress_zero_5xx", "z4_receipt_chain_1000",
        "z4_sse_verified", "z4_rss_drift_bounded",
    ]
    block = _receipt_gates(directory, "z4_live_aaii_and_api_stress.json", required)
    sealed = block["sealed"]
    payload = _read(directory, "agent_eval.json")
    mcp = dict(payload.get("holarch_mcp") or {})
    gates = {
        "z4_runner_exit_zero": bool(int(run["exit_code"]) == 0),
        "z4_gauntlet_exit_zero": bool(int(gauntlet["exit_code"]) == 0),
        "z4_gauntlet_288": bool(int(payload.get("total_queries", 0) or 0) == 288),
        "z4_gauntlet_reachable_1": bool(
            float(mcp.get("reachable_accuracy", 0.0) or 0.0) == 1.0),
        "z4_gauntlet_hallucination_0": bool(mcp.get("hallucination_rate") == 0.0),
        "z4_gauntlet_zero_reads": bool(int(mcp.get("query_file_reads", -1)) == 0),
        "z4_gauntlet_wall_le_160": bool(float(payload.get("wall_seconds", 1e9)) <= 160.0),
        **block["gates"],
    }
    return {
        "stage": "Z4",
        "run": {key: run[key] for key in ("exit_code", "wall_seconds", "stderr_tail")},
        "gauntlet": {
            "exit_code": int(gauntlet["exit_code"]),
            "wall_seconds": float(payload.get("wall_seconds", 0.0) or 0.0),
            "queries": int(payload.get("total_queries", 0) or 0),
            "reachable_accuracy": float(mcp.get("reachable_accuracy", 0.0) or 0.0),
            "hallucination_rate": float(mcp.get("hallucination_rate", 1.0) or 0.0),
            "query_file_reads": int(mcp.get("query_file_reads", -1) or -1),
        },
        "mode": str(sealed.get("mode", "")),
        "measurement_mode": str(sealed.get("mode", "")),
        "arms": dict(sealed.get("arms") or {}),
        "stress": dict(sealed.get("stress") or {}),
        "failed_gates": [name for name, value in gates.items() if not bool(value)],
        "gate": pc._finalise(gates),
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
        "passed": bool(all(gates.values())),
    }


# ---------------------------------------------------------------------- Z5
def _secret_scan(directory: Path) -> Dict[str, Any]:
    leaks: List[str] = []
    files = 0
    for path in sorted(directory.glob("*.json")):
        files += 1
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            continue
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                leaks.append(path.name)
                break
    return {"files": int(files), "leaks": leaks, "leak_count": int(len(leaks))}


def _sidecar_parity(directory: Path) -> Dict[str, Any]:
    checked = 0
    mismatched: List[str] = []
    missing: List[str] = []
    for path in sorted(directory.glob("*.json")):
        sidecar = path.with_name(path.name + ".sha256")
        if not sidecar.exists():
            missing.append(path.name)
            continue
        try:
            recorded = sidecar.read_text(encoding="utf-8").strip()
        except Exception:  # noqa: BLE001
            missing.append(path.name)
            continue
        checked += 1
        if recorded != hashlib.sha256(path.read_bytes()).hexdigest():
            mismatched.append(path.name)
    total = checked + len(missing)
    return {
        "checked": int(checked),
        "missing": missing[:16],
        "mismatched": mismatched[:16],
        "parity": float(checked / max(total, 1)),
    }


def stage_z5(directory: Path) -> Dict[str, Any]:
    started = time.perf_counter()
    hardware = _run("z5_cuda", [sys.executable, "scripts/run_holarch_cuda_benchmark.py"])
    cuda = _read(directory, "z5_cuda_benchmark.json")
    ledger = {
        name: bool(_read(directory, name + ".json").get("passed", False))
        for name in ("t_cycle", "u_cycle", "v_cycle", "w_cycle", "x_cycle")
    }
    y_cycle = _read(directory, "y_cycle.json")
    pytest_run = _run("pytest", [sys.executable, "-m", "pytest", "tests", "-q"], timeout=7200)
    pytest_tail = pytest_run["stdout_tail"]
    failures = 0
    for line in pytest_tail.splitlines():
        match = re.search(r"(\d+) failed", line)
        if match:
            failures = max(failures, int(match.group(1)))
    r3 = _read(directory, "r3_static_and_streaming_full_decile_language.json")
    full = dict(r3.get("full_decile") or {})
    stream_bpc = float(full.get("bits_per_char", 1e9) or 1e9)
    static_bpc = float(full.get("static_bits_per_char", 1e9) or 1e9)
    parity = _sidecar_parity(directory)
    secrets = _secret_scan(directory)
    docs = (ROOT / "docs" / "HOLARCH.md").read_text(encoding="utf-8")
    section_41 = bool(re.search(r"^##\s*41\.", docs, flags=re.MULTILINE))
    # The pinned split must be byte-identical after Section 41 exists.
    pinned = {}
    try:
        from henri.holarch.pinned_split import verify_pinned

        recorded = _read(directory, "pinned_lang_split.json")
        pinned = verify_pinned(ROOT, recorded)
    except Exception as error:  # noqa: BLE001
        pinned = {"invariant": False, "error": f"{type(error).__name__}: {error}"}
    gates = {
        "z5_cuda_path_recorded": bool(str(cuda.get("status", "")) in
                                      ("CUDA: BLOCKED", "CUDA: MEASURED")),
        "z5_no_cpu_as_gpu": bool(cuda.get("cpu_measurement_reported_as_gpu") is False),
        "z5_pnp_diagnostic_recorded": bool((cuda.get("pnp") or {}).get("queried")),
        "z5_t_cycle_passed": bool(ledger.get("t_cycle")),
        "z5_u_cycle_passed": bool(ledger.get("u_cycle")),
        "z5_v_cycle_passed": bool(ledger.get("v_cycle")),
        "z5_w_cycle_passed": bool(ledger.get("w_cycle")),
        "z5_x_cycle_passed": bool(ledger.get("x_cycle")),
        "z5_y_superseded_documented": bool(
            (not y_cycle.get("passed", False)) and y_cycle.get("stages")),
        "z5_pinned_split_unmoved_by_section_41": bool(pinned.get("invariant")),
        "z5_pytest_exit_zero": bool(int(pytest_run["exit_code"]) == 0),
        "z5_pytest_zero_failures": bool(int(failures) == 0),
        "z5_sidecar_parity_one": bool(float(parity["parity"]) >= 1.0),
        "z5_r3_stream_bpc_le_122": bool(stream_bpc <= 1.2200),
        "z5_r3_static_bpc_le_142": bool(static_bpc <= 1.4200),
        "z5_zero_credential_leaks": bool(int(secrets["leak_count"]) == 0),
        "z5_docs_section_41": bool(section_41),
    }
    return {
        "stage": "Z5",
        "cuda": {
            "status": str(cuda.get("status", "")),
            "gpu_measurements": list(cuda.get("gpu_measurements") or []),
            "pnp": dict(cuda.get("pnp") or {}),
            "root_cause": str(cuda.get("root_cause", "")),
        },
        "ledger": ledger,
        "y_cycle": {
            "passed": bool(y_cycle.get("passed", False)),
            "superseded": bool(not y_cycle.get("passed", False)),
            "reason": (
                "Cycle Y's Y1 subgate remains false; Z1 supersedes the Y1 organ with a "
                "D = 2048 measurement. y_cycle.json is retained as historical evidence "
                "and z_cycle.json is the current aggregate."
            ) if not y_cycle.get("passed", False) else "no supersession needed",
        },
        "pinned_split": pinned,
        "pytest": {
            "exit_code": int(pytest_run["exit_code"]),
            "failures": int(failures),
            "wall_seconds": float(pytest_run["wall_seconds"]),
            "stdout_tail": pytest_tail[-1200:],
            "stderr_tail": pytest_run["stderr_tail"][-600:],
        },
        "corpus_hygiene": {
            "source": "r3_static_and_streaming_full_decile_language.json (sealed)",
            "streaming_bits_per_char": float(stream_bpc),
            "static_bits_per_char": float(static_bpc),
        },
        "sidecar": parity,
        "secrets": secrets,
        "gate": pc._finalise(gates),
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
        "passed": bool(all(gates.values())),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Cycle Z verification")
    parser.add_argument("--stages", default="z1,z2,z3,z4,z5")
    parser.add_argument("--receipts", default=str(RECEIPTS))
    parser.add_argument("--reuse", default="",
                        help="comma list of stages whose sealed receipts are read, not re-run")
    args = parser.parse_args(argv)
    directory = Path(args.receipts)
    directory.mkdir(parents=True, exist_ok=True)
    stages = [token.strip().lower() for token in str(args.stages).split(",") if token.strip()]
    _REUSE.clear()
    _REUSE.update(token.strip().lower() for token in str(args.reuse).split(",") if token.strip())
    runners = {
        "z1": ("z1_stage_pointer_egress.json", stage_z1),
        "z2": ("z2_stage_external_synthesis.json", stage_z2),
        "z3": ("z3_stage_spatial_induction.json", stage_z3),
        "z4": ("z4_stage_live_aaii_and_stress.json", stage_z4),
        "z5": ("z5_hardware_ledger_closure.json", stage_z5),
    }
    existing: Dict[str, Any] = {}
    aggregate_path = directory / "z_cycle.json"
    if aggregate_path.exists():
        try:
            existing = json.loads(aggregate_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            existing = {}
    kept = {key: value for key, value in (existing.get("stages") or {}).items()
            if key not in set(stages)}
    aggregate: Dict[str, Any] = {"cycle": "Z", "head": pc._git_head(), "stages": dict(kept)}
    exit_code = 0
    for stage in stages:
        if stage not in runners:
            continue
        name, runner = runners[stage]
        _CURRENT[0] = stage
        print("[z-cycle] %s ..." % stage, flush=True)
        payload = json.loads(json.dumps(runner(directory), default=float))
        payload["mode"] = "sealed-read" if stage in _REUSE else "fresh"
        written = pc._write(directory, name, payload)
        aggregate["stages"][stage] = {
            "receipt": written["path"],
            "sha256": written["sha256"],
            "passed": bool(payload.get("gate", {}).get("passed", False)),
            "gate": payload.get("gate", {}),
            "failed_gates": payload.get("failed_gates", []),
            "mode": payload.get("mode", "fresh"),
            "measurement_mode": payload.get("measurement_mode", ""),
            "wall_seconds": payload.get("wall_seconds", 0.0),
        }
        print("[z-cycle] %s passed=%s failed=%s %s" % (
            stage, payload.get("gate", {}).get("passed"),
            payload.get("failed_gates", []), written["path"]), flush=True)
        aggregate["passed"] = bool(
            all(bool(item.get("passed")) for item in aggregate["stages"].values())
        ) if aggregate["stages"] else False
        pc._write(directory, "z_cycle.json", aggregate)
        if not bool(payload.get("gate", {}).get("passed", False)):
            exit_code = 1
    aggregate["passed"] = bool(
        all(bool(item.get("passed")) for item in aggregate["stages"].values())
    ) if aggregate["stages"] else False
    written = pc._write(directory, "z_cycle.json", aggregate)
    print("[z-cycle] aggregate %s passed=%s" % (written["path"], aggregate["passed"]), flush=True)
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
