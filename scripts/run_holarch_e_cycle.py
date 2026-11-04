"""Cycle E verification ledger: E1, E2, and E3.

Evidence label: DERIVED for the verdicts, OBSERVED for every measured value
they read. Each verdict is computed from a receipt that a verification
command wrote; the SHA-256 of that receipt is embedded here, so editing a
receipt breaks the chain.

    python scripts/run_holarch_e_cycle.py

E1  closed-loop action at D=1024 with the 1-step latent wave lookahead
E2  converged checkpoint mounted in the sidecar and the 44-query benchmark
E3  hardware provenance: Path B, CUDA BLOCKED on the AMD host
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = "receipts/holarch/e_cycle.json"


def _read(path: Path) -> Optional[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def _entry(directory: Path, name: str, checks: Dict[str, bool], measured: Dict[str, Any]) -> Dict[str, Any]:
    path = directory / name
    return {
        "receipt": name,
        "present": path.exists(),
        "sha256": _digest(path),
        "checks": {key: bool(value) for key, value in checks.items()},
        "passed": bool(all(checks.values())) if checks else False,
        "measured": measured,
    }


def build_ledger(directory: Path) -> Dict[str, Any]:
    medium = _read(directory / "holarch_medium.json")
    agent = _read(directory / "agent_eval.json")
    budget = _read(directory / "budget.json")
    directives: Dict[str, Any] = {}

    # ---------------------------------------------------------------- E1
    if medium:
        test = medium.get("test", {})
        vision = float(test.get("vision_accuracy", 0.0))
        relational = float(test.get("relational_accuracy", 0.0))
        action = float(test.get("action_accuracy", 0.0))
        closed = float(medium.get("action_success", 0.0))
        lookahead = medium.get("closed_loop_lookahead", {})
        reactive = float(medium.get("action_success_reactive", 0.0))
        directives["E1"] = _entry(
            directory,
            "holarch_medium.json",
            {
                "closed_loop_at_least_0_54": closed >= 0.54,
                "closed_loop_target_0_56": closed >= 0.56,
                "one_step_at_least_0_78": action >= 0.78,
                "relational_at_least_0_76": relational >= 0.76,
                "vision_at_least_0_975": vision >= 0.975,
                "lookahead_beats_reactive": closed > reactive,
            },
            {
                "vision_accuracy": vision,
                "relational_accuracy": relational,
                "one_step_accuracy": action,
                "closed_loop_success": closed,
                "closed_loop_target": 0.56,
                "reactive_success": reactive,
                "lookahead": lookahead,
                "dim": float(medium.get("config", {}).get("dim", 0.0)),
                "parameters": float(medium.get("parameters", 0.0)),
                "dagger_rounds": float(medium.get("dagger_rounds", 0.0)),
                "coupling": float(
                    medium.get("train_config", {}).get("relational_bonus_coupling", 0.0)
                ),
                "checkpoint_sha256": str(medium.get("checkpoint", {}).get("sha256", "")),
            },
        )
    else:
        directives["E1"] = {"receipt": "holarch_medium.json", "present": False, "passed": False}

    # ---------------------------------------------------------------- E2
    if agent:
        composite = agent.get("composite", {})
        mcp = agent.get("holarch_mcp", {})
        checkpoint = agent.get("checkpoint", {})
        directives["E2"] = _entry(
            directory,
            "agent_eval.json",
            {
                "checkpoint_mounted": bool(checkpoint.get("mounted", False)),
                "perceptual_truth_3_of_3": int(composite.get("perceptual_truth_match", 0)) == 3,
                "service_consistency_3_of_3": int(composite.get("perceptual_consistency", 0)) == 3,
                "composite_accuracy_1_0": float(composite.get("mcp_accuracy", 0.0)) >= 0.999,
                "traps_refused_2_of_2": int(composite.get("mcp_refused", 0))
                == int(composite.get("traps_total", -1))
                == 2,
                "zero_query_reads": float(mcp.get("query_file_reads", 1)) == 0.0,
                "receipt_chain_valid": bool(mcp.get("receipt_chain_valid", False)),
                "receipts_at_least_44": int(mcp.get("receipts", 0)) >= 44,
                "queries_44": int(agent.get("queries_total", 0)) >= 44,
            },
            {
                "queries_total": int(agent.get("queries_total", 0)),
                "checkpoint": {
                    "mounted": bool(checkpoint.get("mounted", False)),
                    "path": str(checkpoint.get("path", "")),
                    "sha256": str(checkpoint.get("sha256", "")),
                    "parameters": int(checkpoint.get("parameters", 0)),
                },
                "perceptual_truth_match": int(composite.get("perceptual_truth_match", 0)),
                "perceptual_total": int(composite.get("perceptual_total", 0)),
                "perceptual_consistency": int(composite.get("perceptual_consistency", 0)),
                "composite_accuracy": float(composite.get("mcp_accuracy", 0.0)),
                "traps_total": int(composite.get("traps_total", 0)),
                "traps_refused": int(composite.get("mcp_refused", 0)),
                "query_file_reads": float(mcp.get("query_file_reads", 0.0)),
                "receipts": int(mcp.get("receipts", 0)),
                "receipt_chain_valid": bool(mcp.get("receipt_chain_valid", False)),
            },
        )
    else:
        directives["E2"] = {"receipt": "agent_eval.json", "present": False, "passed": False}

    # ---------------------------------------------------------------- E3
    if budget:
        hardware = budget.get("hardware", {})
        profiles = budget.get("profiles", {})
        cuda = bool(hardware.get("cuda_available", False))
        nvidia = bool(hardware.get("host_has_nvidia_gpu", False))
        accelerators = [str(item) for item in hardware.get("detected_accelerators", [])]
        amd = any("radeon" in item.lower() or "amd" in item.lower() for item in accelerators)
        directives["E3"] = _entry(
            directory,
            "budget.json",
            {
                "cuda_blocked": not cuda,
                "cuda_evidence_blocked": str(hardware.get("evidence", "")) == "BLOCKED",
                "adapters_enumerated": len(accelerators) >= 1,
                "amd_host_named": amd,
                "three_profiles_measured": all(
                    name in profiles for name in ("small", "medium", "production")
                ),
                "all_profiles_fit": bool(budget.get("all_profiles_fit", False)),
                "no_gpu_number_claimed": str(budget.get("device", "")) == "cpu"
                and float(budget.get("cuda_peak_allocated_gib", 0.0)) == 0.0,
            },
            {
                "path": "B",
                "path_meaning": "local AMD host; CUDA build cannot see a device; CPU measurements only",
                "cuda_available": cuda,
                "cuda_device_count": int(hardware.get("device_count", 0)),
                "host_has_nvidia_gpu": nvidia,
                "cuda_evidence": str(hardware.get("evidence", "")),
                "torch_version": str(hardware.get("torch_version", "")),
                "cuda_compiled_version": str(hardware.get("cuda_compiled_version", "")),
                "detected_accelerators": accelerators,
                "device": str(budget.get("device", "")),
                "profiles": {
                    name: {
                        "parameters": float(profile.get("parameters", 0.0)),
                        "training_gib": float(profile.get("training_gib", 0.0)),
                        "forward_ms": float(profile.get("forward_ms", 0.0)),
                        "backward_ms": float(profile.get("backward_ms", 0.0)),
                        "fits": bool(profile.get("fits", False)),
                    }
                    for name, profile in profiles.items()
                },
            },
        )
    else:
        directives["E3"] = {"receipt": "budget.json", "present": False, "passed": False}

    return {
        "cycle": "E1-E3",
        "directives": directives,
        "summary": {
            "passed": sum(1 for item in directives.values() if item.get("passed")),
            "total": len(directives),
            "blocked": [
                name
                for name, item in directives.items()
                if not item.get("passed") and name == "E3" and not item.get("measured", {}).get("cuda_available", True)
            ],
        },
        "evidence": "DERIVED",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="receipts/holarch")
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()
    ledger = build_ledger(ROOT / args.dir)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ledger, indent=2, default=float), encoding="utf-8")
    (out.parent / (out.name + ".sha256")).write_text(
        hashlib.sha256(out.read_bytes()).hexdigest() + "\n", encoding="utf-8"
    )
    print(json.dumps(ledger["summary"], indent=2))
    for name, item in ledger["directives"].items():
        print(name, "PASS" if item.get("passed") else "FAIL", item.get("measured", {}))
    return 0 if ledger["summary"]["passed"] == ledger["summary"]["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
