"""Cycle Y verification: photonic egress, cold synthesis, unified induction,
scaled AAII v5.0 five-arm benchmark, hardware probe and the full ledger seal.

    python scripts/run_holarch_y_cycle.py --stages y1,y2,y3,y4,y5

Every stage re-runs its own measurement script as a subprocess, reads the
receipt that script sealed, and re-gates on the receipt's own keys.  A stage
whose runner exits non-zero is recorded with the failure, never masked.
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
        # --reuse: the sealed measurement in this tree is read, not repeated.
        # The stage payload records mode = sealed-read so the evidence label
        # cannot be mistaken for a fresh run.
        return {
            "name": str(name),
            "command": " ".join(command),
            "exit_code": 0,
            "wall_seconds": 0.0,
            "stdout_tail": "REUSED: sealed receipt read from this tree",
            "stderr_tail": "",
            "reused": True,
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


def _digest(directory: Path, name: str) -> str:
    path = directory / str(name)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def _receipt_gates(directory: Path, name: str, required: List[str]) -> Dict[str, Any]:
    sealed = _read(directory, name)
    gate = dict(sealed.get("gate") or {})
    gates: Dict[str, Any] = {"receipt_present": bool(sealed)}
    for key in required:
        gates[key] = bool(gate.get(key, False))
    return {"sealed": sealed, "gate": gate, "gates": gates}


# ---------------------------------------------------------------------- Y1
def stage_y1(directory: Path) -> Dict[str, Any]:
    started = time.perf_counter()
    run = _run("y1", [sys.executable, "scripts/run_holarch_y1_photonic_wave_egress.py"])
    required = [
        "y1_autograd_unused", "y1_quadrature_enabled", "y1_photonic_weight_positive",
        "y1_photonic_improves_inner_ce", "y1_phase_displacement",
        "y1_phase_attribution_nonzero", "y1_held_out_top1",
        "y1_bits_per_char_beat_x2", "y1_net_overrides", "y1_top1_beats_floor",
        "y1_bpc_beats_floor", "y1_gap_purged", "y1_held_out_tokens",
        "y1_cartridge_positive_cosine", "y1_capacity_probe_ran",
        "y1_capacity_probe_monotone", "y1_capacity_suffix_blind",
        "y1_capacity_deployed_retrieval",
    ]
    block = _receipt_gates(directory, "y1_photonic_wave_egress.json", required)
    sealed = block["sealed"]
    gates = {"y1_runner_exit_zero": bool(int(run["exit_code"]) == 0), **block["gates"]}
    return {
        "stage": "Y1",
        "run": {key: run[key] for key in ("exit_code", "wall_seconds", "stderr_tail")},
        "split": dict(sealed.get("split") or {}),
        "wave_egress": dict(sealed.get("wave_egress") or {}),
        "x2_side_by_side": dict(sealed.get("x2_side_by_side") or {}),
        "photonic_cartridge": dict(sealed.get("photonic_cartridge") or {}),
        "inner_ce_ablation": dict(sealed.get("inner_ce_ablation") or {}),
        "phase": {key: sealed.get("phase", {}).get(key) for key in
                  ("displacement", "inner_nll_initial", "inner_nll_final", "updates")},
        "attribution": dict(sealed.get("attribution") or {}),
        "capacity_probe": dict(sealed.get("capacity_probe") or {}),
        "failed_gates": [name for name, value in gates.items() if not bool(value)],
        "gate": pc._finalise(gates),
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
        "passed": bool(all(gates.values())),
    }


# ---------------------------------------------------------------------- Y2
def stage_y2(directory: Path) -> Dict[str, Any]:
    started = time.perf_counter()
    run = _run("y2", [sys.executable, "scripts/run_holarch_y2_cold_synthesis.py"])
    required = [
        "y2_planstore_empty_at_start", "y2_zero_llm_calls", "y2_standalone_pass_at_1",
        "y2_hidden_precision", "y2_syntax_type_error_rate_zero", "y2_traps_refused_4",
        "y2_budget_24", "y2_effective_branching_le_8", "y2_repulsor_registered",
        "y2_sandbox_out_of_process", "y2_emitted_solutions_present",
    ]
    block = _receipt_gates(directory, "y2_cold_start_synthesis.json", required)
    sealed = block["sealed"]
    gates = {"y2_runner_exit_zero": bool(int(run["exit_code"]) == 0), **block["gates"]}
    return {
        "stage": "Y2",
        "run": {key: run[key] for key in ("exit_code", "wall_seconds", "stderr_tail")},
        "suite": dict(sealed.get("suite") or {}),
        "pass_at_1": float(sealed.get("pass_at_1", 0.0) or 0.0),
        "solved": int(sealed.get("solved", 0) or 0),
        "hidden_precision": float(sealed.get("hidden_precision", 0.0) or 0.0),
        "traps_refused": int(sealed.get("traps_refused", 0) or 0),
        "max_effective_branching": int(sealed.get("max_effective_branching", 0) or 0),
        "repulsors_registered": int(sealed.get("repulsors_registered", 0) or 0),
        "rows": list(sealed.get("rows") or []),
        "traps": list(sealed.get("traps") or []),
        "failed_gates": [name for name, value in gates.items() if not bool(value)],
        "gate": pc._finalise(gates),
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
        "passed": bool(all(gates.values())),
    }


# ---------------------------------------------------------------------- Y3
def stage_y3(directory: Path) -> Dict[str, Any]:
    started = time.perf_counter()
    run = _run("y3", [sys.executable, "scripts/run_holarch_y3_unified_induction.py"])
    required = [
        "y3_twelve_retained", "y3_battery_16", "y3_induced_and_generalized_ge_14",
        "y3_contradictions_4", "y3_strict_one_cell_all_rows", "y3_text_tokens_zero",
        "y3_no_task_name_branches", "y3_budget_le_320000", "y3_new_tasks_4_present",
        "y3_router_signature_used",
    ]
    block = _receipt_gates(directory, "y3_unified_wave_induction.json", required)
    sealed = block["sealed"]
    gates = {"y3_runner_exit_zero": bool(int(run["exit_code"]) == 0), **block["gates"]}
    return {
        "stage": "Y3",
        "run": {key: run[key] for key in ("exit_code", "wall_seconds", "stderr_tail")},
        "battery": int(sealed.get("task_count", 0) or 0),
        "induced_and_generalized": int(sealed.get("induced_and_generalized", 0) or 0),
        "contradictions_failed_closed": int(
            sealed.get("contradictions_failed_closed")
            or (sealed.get("contradictions") or {}).get("refuted_by_construction", 0)
            or 0),
        "max_nodes_used": int((sealed.get("budget") or {}).get("max_nodes_used", 0) or 0),
        "strict_one_cell": dict(sealed.get("strict_one_cell") or {}),
        "ast_audit": {
            key: (sealed.get("ast_audit") or {}).get(key)
            for key in ("task_name_branches", "r1_name_attribute_comparisons",
                        "r2_name_literal_comparisons_in_routing",
                        "r3_name_keyed_dicts_in_routing")
        },
        "router_evidence": dict(sealed.get("router_evidence") or {}),
        "new_tasks": dict(sealed.get("new_tasks") or {}),
        "tasks": int(sealed.get("task_count", 0) or 0),
        "task_rows": [
            {
                "name": str(row.get("name", "")),
                "status": str(row.get("status", "")),
                "mechanism": str(row.get("mechanism_used", "")),
                "nodes": int(row.get("nodes", 0) or 0),
                "v1_exact": int(row.get("v1_exact", 0) or 0),
                "v1_total": int(row.get("v1_total", 0) or 0),
            }
            for row in (sealed.get("tasks") or [])
            if isinstance(row, dict)
        ],
        "failed_gates": [name for name, value in gates.items() if not bool(value)],
        "gate": pc._finalise(gates),
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
        "passed": bool(all(gates.values())),
    }


# ---------------------------------------------------------------------- Y4
def stage_y4(directory: Path) -> Dict[str, Any]:
    started = time.perf_counter()
    run = _run("y4", [sys.executable, "scripts/run_holarch_y4_aaii_v5.py"])
    gauntlet = _run(
        "y4_gauntlet",
        [sys.executable, "scripts/run_holarch_agent_eval.py",
         "--out", str(directory / "agent_eval.json")],
    )
    required = [
        "y4_battery_48", "y4_five_arms_mounted", "y4_arm0_cold_coding_gt_0",
        "y4_arm0_hallucination_zero", "y4_arm3_beats_arm1", "y4_arm3_beats_arm2",
        "y4_arm4_beats_arm1", "y4_arm4_beats_arm2", "y4_arm4_call_reduction_ge_35pct",
        "y4_arm4_matches_arm3", "y4_arm4_hallucination_le_002", "y4_zero_credential_leaks",
        "y4_mount_under_5ms",
    ]
    block = _receipt_gates(directory, "y4_scaled_aaii_v5.json", required)
    sealed = block["sealed"]
    payload = _read(directory, "agent_eval.json")
    mcp = dict(payload.get("holarch_mcp") or {})
    rows = list((payload.get("probes") or {}).get("rows") or [])
    y4_rows = [
        row for row in rows
        if str(row.get("name", "")).startswith(("y1_", "y2_", "y3_", "y4_"))
    ]
    gates = {
        "y4_runner_exit_zero": bool(int(run["exit_code"]) == 0),
        "y4_gauntlet_exit_zero": bool(int(gauntlet["exit_code"]) == 0),
        "y4_gauntlet_272": bool(int(payload.get("total_queries", 0) or 0) == 272),
        "y4_gauntlet_reachable_1": bool(
            float(mcp.get("reachable_accuracy", 0.0) or 0.0) == 1.0),
        "y4_gauntlet_hallucination_0": bool(mcp.get("hallucination_rate") == 0.0),
        "y4_gauntlet_zero_query_reads": bool(int(mcp.get("query_file_reads", -1)) == 0),
        "y4_gauntlet_wall_le_160": bool(float(payload.get("wall_seconds", 1e9)) <= 160.0),
        "y4_gauntlet_y_probes_present": bool(
            len(y4_rows) == 16 and all(bool(row.get("correct")) for row in y4_rows)),
        **block["gates"],
    }
    return {
        "stage": "Y4",
        "run": {key: run[key] for key in ("exit_code", "wall_seconds", "stderr_tail")},
        "gauntlet": {
            "exit_code": int(gauntlet["exit_code"]),
            "wall_seconds": float(payload.get("wall_seconds", 0.0) or 0.0),
            "queries": int(payload.get("total_queries", 0) or 0),
            "reachable_accuracy": float(mcp.get("reachable_accuracy", 0.0) or 0.0),
            "hallucination_rate": float(mcp.get("hallucination_rate", 1.0) or 0.0),
            "query_file_reads": int(mcp.get("query_file_reads", -1) or -1),
            "y_probes": y4_rows,
        },
        "arms": dict(sealed.get("arms") or {}),
        "scoring": dict(sealed.get("scoring") or {}),
        "cold_path": dict(sealed.get("cold_path") or {}),
        "shared_stem": dict(sealed.get("shared_stem") or {}),
        "arm4_call_reduction": dict(sealed.get("arm4_call_reduction") or {}),
        "failed_gates": [name for name, value in gates.items() if not bool(value)],
        "gate": pc._finalise(gates),
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
        "passed": bool(all(gates.values())),
    }


# ---------------------------------------------------------------------- Y5
def hardware_probe(directory: Path) -> Dict[str, Any]:
    """The deterministic hardware gate.

    On a host with a bound CUDA device the probe runs the production D = 4096
    profile forward / imprint / egress benchmark.  On a host without one it
    records the exact driver diagnostics and the string CUDA: BLOCKED, and it
    never reports a CPU number as a GPU measurement.
    """
    import torch

    diagnostics: Dict[str, Any] = {
        "torch_version": str(torch.__version__),
        "cuda_build": str(torch.version.cuda),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
        "python": sys.version.split()[0],
    }
    try:
        completed = subprocess.run(
            ["nvidia-smi", "-q"], capture_output=True, text=True, timeout=20
        )
        text = completed.stdout or ""
        fields = ("Driver Version", "CUDA Version", "Product Name")
        diagnostics["nvidia_smi_exit"] = int(completed.returncode)
        for field in fields:
            match = re.search(field + r"\s*:\s*(.+)", text)
            if match:
                diagnostics[field.lower().replace(" ", "_")] = match.group(1).strip()
        diagnostics["nvidia_smi_present"] = True
    except Exception as error:  # noqa: BLE001 - a missing probe is a diagnostic
        diagnostics["nvidia_smi_present"] = False
        diagnostics["nvidia_smi_error"] = type(error).__name__

    if not bool(diagnostics["cuda_available"]):
        return {
            "status": "CUDA: BLOCKED",
            "path": "B",
            "diagnostics": diagnostics,
            "cpu_measurement_reported_as_gpu": False,
            "production_benchmark": {},
            "measured": False,
            "reason": (
                "torch.cuda.is_available() is False: no NVIDIA adapter is bound to "
                "the driver, so no D = 4096 production forward/imprint/egress number "
                "can be reported. A CPU number would not be a GPU measurement."
            ),
        }

    from henri.holarch.config import HolarchConfig
    from henri.holarch.pipeline import HenriHolarch

    cfg = HolarchConfig.production()
    model = HenriHolarch(cfg).to("cuda")
    model.eval()
    generator = torch.Generator(device="cuda").manual_seed(20261111)
    psi = torch.randn(1, int(cfg.dim), dtype=torch.complex64, device="cuda", generator=generator)
    psi = psi / psi.norm()
    target = torch.randn(1, int(cfg.dim), dtype=torch.complex64, device="cuda", generator=generator)
    target = target / target.norm()
    codebook = torch.randn(2048, int(cfg.dim), dtype=torch.complex64, device="cuda",
                           generator=generator)
    codebook = codebook / codebook.norm(dim=-1, keepdim=True)

    def timed(fn, repeats: int = 5) -> float:
        torch.cuda.synchronize()
        began = time.perf_counter()
        for _ in range(int(repeats)):
            fn()
        torch.cuda.synchronize()
        return float((time.perf_counter() - began) / max(int(repeats), 1) * 1000.0)

    with torch.no_grad():
        forward_ms = timed(lambda: model.zone_b(psi))
        imprint_ms = timed(lambda: model.zone_b.imprint(psi, target, rate=0.05), repeats=3)
        state = model.zone_b(psi)
        readout_ms = timed(lambda: codebook @ state.bright[0].conj())
    return {
        "status": "CUDA: MEASURED",
        "path": "A",
        "diagnostics": diagnostics,
        "cpu_measurement_reported_as_gpu": False,
        "production_benchmark": {
            "dim": int(cfg.dim),
            "photonic_layers": int(cfg.photonic_layers),
            "forward_ms": forward_ms,
            "imprint_ms": imprint_ms,
            "egress_readout_ms": readout_ms,
            "vram_allocated_mb": float(torch.cuda.memory_allocated() / 2**20),
            "vram_reserved_mb": float(torch.cuda.memory_reserved() / 2**20),
        },
        "measured": True,
        "reason": "a bound CUDA device was available and the production profile ran on it",
    }


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


def stage_y5(directory: Path) -> Dict[str, Any]:
    started = time.perf_counter()
    hardware = hardware_probe(directory)
    ledger = {
        name: bool(_read(directory, name + ".json").get("passed", False))
        for name in ("t_cycle", "u_cycle", "v_cycle", "w_cycle", "x_cycle")
    }
    for name in ("t_cycle", "u_cycle", "v_cycle", "w_cycle", "x_cycle"):
        sidecar = directory / (name + ".json.sha256")
        ledger[name + "_sidecar"] = bool(sidecar.exists())
    pytest_run = _run("pytest", [sys.executable, "-m", "pytest", "tests", "-q"], timeout=7200)
    pytest_tail = pytest_run["stdout_tail"]
    failures = 0
    passes = 0
    for line in pytest_tail.splitlines():
        match = re.search(r"(\d+) failed", line)
        if match:
            failures = max(failures, int(match.group(1)))
        match = re.search(r"(\d+) passed", line)
        if match:
            passes = max(passes, int(match.group(1)))
    r3 = _read(directory, "r3_static_and_streaming_full_decile_language.json")
    full = dict(r3.get("full_decile") or {})
    stream_bpc = float(full.get("bits_per_char", 1e9) or 1e9)
    static_bpc = float(full.get("static_bits_per_char", 1e9) or 1e9)
    parity = _sidecar_parity(directory)
    secrets = _secret_scan(directory)
    docs = (ROOT / "docs" / "HOLARCH.md").read_text(encoding="utf-8")
    section_40 = bool(re.search(r"^##\s*40\.", docs, flags=re.MULTILINE))
    gates = {
        "y5_hardware_path_recorded": bool(
            str(hardware.get("status")) in ("CUDA: BLOCKED", "CUDA: MEASURED")),
        "y5_no_cpu_reported_as_gpu": bool(
            hardware.get("cpu_measurement_reported_as_gpu") is False),
        "y5_production_dim_4096": bool(
            str(hardware.get("status")) == "CUDA: BLOCKED"
            or int(hardware["production_benchmark"].get("dim", 0)) == 4096),
        "y5_t_cycle_passed": bool(ledger.get("t_cycle")),
        "y5_u_cycle_passed": bool(ledger.get("u_cycle")),
        "y5_v_cycle_passed": bool(ledger.get("v_cycle")),
        "y5_w_cycle_passed": bool(ledger.get("w_cycle")),
        "y5_x_cycle_passed": bool(ledger.get("x_cycle")),
        "y5_pytest_exit_zero": bool(int(pytest_run["exit_code"]) == 0),
        "y5_pytest_zero_failures": bool(int(failures) == 0),
        "y5_sidecar_parity_one": bool(float(parity["parity"]) >= 1.0),
        "y5_r3_stream_bpc_le_122": bool(stream_bpc <= 1.2200),
        "y5_r3_static_bpc_le_142": bool(static_bpc <= 1.4200),
        "y5_zero_credential_leaks": bool(int(secrets["leak_count"]) == 0),
        "y5_docs_section_40": bool(section_40),
    }
    return {
        "stage": "Y5",
        "hardware": hardware,
        "ledger": ledger,
        "pytest": {
            "exit_code": int(pytest_run["exit_code"]),
            "failures": int(failures),
            "passes": int(passes),
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
    parser = argparse.ArgumentParser(description="Cycle Y verification")
    parser.add_argument("--stages", default="y1,y2,y3,y4,y5")
    parser.add_argument("--receipts", default=str(RECEIPTS))
    parser.add_argument("--reuse", default="",
                        help="comma list of stages whose sealed receipts are read, not re-run")
    args = parser.parse_args(argv)
    directory = Path(args.receipts)
    directory.mkdir(parents=True, exist_ok=True)
    stages = [token.strip().lower() for token in str(args.stages).split(",") if token.strip()]
    _REUSE.clear()
    _REUSE.update(token.strip().lower() for token in str(args.reuse).split(",") if token.strip())
    # The stage receipt is a separate file from the measurement receipt the
    # runner sealed.  Writing the stage view over the detailed receipt would
    # destroy the numbers the stage is supposed to re-gate.
    runners = {
        "y1": ("y1_stage_photonic_wave_egress.json", stage_y1),
        "y2": ("y2_stage_cold_synthesis.json", stage_y2),
        "y3": ("y3_stage_unified_induction.json", stage_y3),
        "y4": ("y4_stage_scaled_aaii_v5.json", stage_y4),
        "y5": ("y5_hardware_and_ledger_closure.json", stage_y5),
    }
    existing: Dict[str, Any] = {}
    aggregate_path = directory / "y_cycle.json"
    if aggregate_path.exists():
        try:
            existing = json.loads(aggregate_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            existing = {}
    kept = {key: value for key, value in (existing.get("stages") or {}).items()
            if key not in set(stages)}
    aggregate: Dict[str, Any] = {"cycle": "Y", "head": pc._git_head(), "stages": dict(kept)}
    exit_code = 0
    for stage in stages:
        if stage not in runners:
            continue
        name, runner = runners[stage]
        _CURRENT[0] = stage
        print("[y-cycle] %s ..." % stage, flush=True)
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
            "wall_seconds": payload.get("wall_seconds", 0.0),
        }
        print("[y-cycle] %s passed=%s failed=%s %s" % (
            stage, payload.get("gate", {}).get("passed"),
            payload.get("failed_gates", []), written["path"]), flush=True)
        # An incremental aggregate write: a later stage that raises must not
        # erase the stages that already ran and sealed.
        aggregate["passed"] = bool(
            all(bool(item.get("passed")) for item in aggregate["stages"].values())
        ) if aggregate["stages"] else False
        pc._write(directory, "y_cycle.json", aggregate)
        if not bool(payload.get("gate", {}).get("passed", False)):
            exit_code = 1
    aggregate["passed"] = bool(
        all(bool(item.get("passed")) for item in aggregate["stages"].values())
    ) if aggregate["stages"] else False
    written = pc._write(directory, "y_cycle.json", aggregate)
    print("[y-cycle] aggregate %s passed=%s" % (written["path"], aggregate["passed"]), flush=True)
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
