"""Cycle D verification ledger: C1-C5 style aggregation for D1-D5.

Evidence label: DERIVED for the verdicts, OBSERVED for every measured value
they read. Each verdict is computed from a receipt that a verification
command wrote, and the SHA-256 of that receipt is embedded here. Editing a
receipt breaks the chain.

    python scripts/run_holarch_d_cycle.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = "receipts/holarch/d_cycle.json"


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
    small = _read(directory / "holarch_small.json")
    production = _read(directory / "holarch_production.json")
    budget = _read(directory / "budget.json")
    symbolic = _read(directory / "symbolic.json")
    agent = _read(directory / "agent_eval.json")

    directives: Dict[str, Any] = {}

    if small:
        ood = float(small.get("ood_veto_rate", 0.0))
        false_veto = float(small.get("in_distribution_veto_rate", 1.0))
        diagnostics = small.get("veto_diagnostics", {})
        clean_max = float(diagnostics.get("clean_residual_max", 0.0))
        noise_min = float(diagnostics.get("noise_residual_min", 0.0))
        directives["D1"] = _entry(
            directory,
            "holarch_small.json",
            {
                "ood_at_least_0_999": ood >= 0.999,
                "false_veto_at_most_0_005": false_veto <= 0.005,
                "clean_below_noise": clean_max < noise_min,
            },
            {
                "ood_veto_rate": ood,
                "false_veto_rate": false_veto,
                "clean_residual_max": clean_max,
                "noise_residual_min": noise_min,
                "tau_residual": float(diagnostics.get("tau_residual", 0.0)),
                "parameters": float(small.get("parameters", 0.0)),
            },
        )
    else:
        directives["D1"] = {"receipt": "holarch_small.json", "present": False, "passed": False}

    if production:
        test = production.get("test", {})
        vision = float(test.get("vision_accuracy", 0.0))
        relational = float(test.get("relational_accuracy", 0.0))
        action = float(test.get("action_accuracy", 0.0))
        success = float(production.get("action_success", 0.0))
        directives["D2"] = _entry(
            directory,
            "holarch_production.json",
            {
                "vision_at_least_0_97": vision >= 0.97,
                "relational_at_least_0_75": relational >= 0.75,
                "action_at_least_0_7875": action >= 0.7875,
                "closed_loop_at_least_0_54": success >= 0.54,
            },
            {
                "vision_accuracy": vision,
                "relational_accuracy": relational,
                "action_accuracy": action,
                "action_success": success,
                "dim": float(production.get("config", {}).get("dim", 0.0)),
                "coupling": float(production.get("train_config", {}).get("relational_bonus_coupling", 0.0)),
                "dagger_rounds": float(production.get("dagger_rounds", 0.0)),
                "parameters": float(production.get("parameters", 0.0)),
            },
        )
    else:
        directives["D2"] = {"receipt": "holarch_production.json", "present": False, "passed": False}

    if budget:
        hardware = budget.get("hardware", {})
        profiles = budget.get("profiles", {})
        directives["D3"] = _entry(
            directory,
            "budget.json",
            {
                "three_profiles": all(name in profiles for name in ("small", "medium", "production")),
                "all_profiles_fit": bool(budget.get("all_profiles_fit", False)),
                "provenance_complete": all(
                    key in hardware
                    for key in ("host_platform", "torch_version", "cuda_available",
                                "detected_accelerators", "host_has_nvidia_gpu")
                ),
            },
            {
                "cuda_available": bool(hardware.get("cuda_available", False)),
                "host_has_nvidia_gpu": bool(hardware.get("host_has_nvidia_gpu", False)),
                "detected_accelerators": hardware.get("detected_accelerators", []),
                "profiles": {
                    name: {
                        "parameters": float(profile.get("parameters", 0.0)),
                        "training_gib": float(profile.get("training_gib", 0.0)),
                        "fits": bool(profile.get("fits", False)),
                        "forward_ms": float(profile.get("forward_ms", 0.0)),
                        "backward_ms": float(profile.get("backward_ms", 0.0)),
                    }
                    for name, profile in profiles.items()
                },
            },
        )
    else:
        directives["D3"] = {"receipt": "budget.json", "present": False, "passed": False}

    composite = (agent or {}).get("composite", {})
    if symbolic and agent:
        pair_sum = float(
            (symbolic.get("composition", {}) or {}).get("pair_sum", {}).get("accuracy", 0.0)
        )
        pair_diff = float(
            (symbolic.get("composition", {}) or {}).get("pair_diff", {}).get("accuracy", 0.0)
        )
        directives["D4"] = _entry(
            directory,
            "agent_eval.json",
            {
                "composite_mcp_at_least_0_90": float(composite.get("mcp_accuracy", 0.0)) >= 0.90,
                "composite_traps_refused": int(composite.get("mcp_refused", 0))
                == int(composite.get("traps_total", -1)),
                "symbolic_pair_at_least_0_85": pair_sum >= 0.85 and pair_diff >= 0.85,
            },
            {
                "composite_mcp_accuracy": float(composite.get("mcp_accuracy", 0.0)),
                "composite_baseline_accuracy": float(composite.get("baseline_accuracy", 0.0)),
                "ingested": float(composite.get("ingested", 0.0)),
                "tau_composite": float(composite.get("tau_composite", 0.0)),
                "traps_total": int(composite.get("traps_total", 0)),
                "mcp_refused": int(composite.get("mcp_refused", 0)),
                "perceptual_consistency": int(composite.get("perceptual_consistency", 0)),
                "perceptual_total": int(composite.get("perceptual_total", 0)),
                "pair_sum": pair_sum,
                "pair_diff": pair_diff,
            },
        )
    else:
        directives["D4"] = {"receipt": "agent_eval.json", "present": False, "passed": False}

    if agent:
        mcp = agent.get("holarch_mcp", {})
        directives["D5"] = _entry(
            directory,
            "agent_eval.json",
            {
                "forty_four_queries": int(agent.get("queries_total", 0)) >= 44,
                "zero_query_reads": float(mcp.get("query_file_reads", 1)) == 0.0,
                "reachable_accuracy_1": float(mcp.get("reachable_accuracy", 0.0)) >= 0.999,
                "hallucination_zero": float(mcp.get("hallucination_rate", 1.0)) == 0.0,
                "receipt_chain_valid": bool(mcp.get("receipt_chain_valid", False)),
                "agent_gate_passed": bool(agent.get("gate", {}).get("passed", False)),
            },
            {
                "queries_total": int(agent.get("queries_total", 0)),
                "query_file_reads": float(mcp.get("query_file_reads", 0.0)),
                "reachable_accuracy": float(mcp.get("reachable_accuracy", 0.0)),
                "hallucination_rate": float(mcp.get("hallucination_rate", 0.0)),
                "receipts": float(mcp.get("receipts", 0.0)),
                "compression": float(agent.get("context_compression_ratio", 0.0)),
            },
        )
    else:
        directives["D5"] = {"receipt": "agent_eval.json", "present": False, "passed": False}

    return {
        "cycle": "D1-D5",
        "directives": directives,
        "summary": {
            "passed": sum(1 for item in directives.values() if item.get("passed")),
            "total": len(directives),
            "blocked": [
                name
                for name, item in directives.items()
                if not item.get("passed")
                and not item.get("measured", {}).get("cuda_available", True)
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
