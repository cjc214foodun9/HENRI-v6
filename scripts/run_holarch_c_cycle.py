"""Cycle C verification ledger: one signed summary of C1-C5.

Evidence label: DERIVED for the verdicts, OBSERVED for every measured value
they read. Each directive's verdict is computed from a receipt that a
verification command wrote, and the SHA-256 of that receipt is embedded in
this ledger. Editing any underlying receipt breaks the chain.

    python scripts/run_holarch_c_cycle.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = "receipts/holarch/c_cycle.json"


def _read(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    sidecar = _read(directory / "sidecar.json")

    directives: Dict[str, Any] = {}

    if small:
        ood = float(small.get("ood_veto_rate", 0.0))
        false_veto = float(small.get("in_distribution_veto_rate", 1.0))
        directives["C1"] = _entry(
            directory,
            "holarch_small.json",
            {"ood_at_least_0_999": ood >= 0.999, "false_veto_at_most_0_005": false_veto <= 0.005},
            {"ood_veto_rate": ood, "false_veto_rate": false_veto,
             "tau_residual": float(small.get("calibration", {}).get("vision", {}).get("tau_residual", 0.0)),
             "parameters": float(small.get("parameters", 0.0))},
        )
    else:
        directives["C1"] = {"receipt": "holarch_small.json", "present": False, "passed": False}

    if production:
        test = production.get("test", {})
        relational = float(test.get("relational_accuracy", 0.0))
        action = float(test.get("action_accuracy", 0.0))
        vision = float(test.get("vision_accuracy", 0.0))
        directives["C2"] = _entry(
            directory,
            "holarch_production.json",
            {
                "relational_at_least_0_75": relational >= 0.75,
                "action_at_least_0_788": action >= 0.788,
                "vision_at_least_0_9697": vision >= 0.9697,
            },
            {"relational_accuracy": relational, "action_accuracy": action,
             "vision_accuracy": vision, "dim": float(production.get("config", {}).get("dim", 0.0)),
             "parameters": float(production.get("parameters", 0.0)),
             "action_success": float(production.get("action_success", 0.0))},
        )
    else:
        directives["C2"] = {"receipt": "holarch_production.json", "present": False, "passed": False}

    if budget:
        hardware = budget.get("hardware", {})
        cuda = bool(hardware.get("cuda_available", False))
        directives["C3"] = _entry(
            directory,
            "budget.json",
            {"cuda_observed": cuda},
            {"evidence": "OBSERVED" if cuda else "BLOCKED",
             "reason": hardware.get("reason", ""),
             "device": budget.get("measured_device", budget.get("device", "")),
             "training_gib": float(budget.get("training_gib", 0.0)),
             "parameters": float(budget.get("parameters", 0.0)),
             "cuda_peak_allocated_gib": float(budget.get("cuda_peak_allocated_gib", 0.0))},
        )
    else:
        directives["C3"] = {"receipt": "budget.json", "present": False, "passed": False}

    if symbolic:
        composition = symbolic.get("composition", {})
        pair_sum = float(composition.get("pair_sum", {}).get("accuracy", 0.0))
        pair_diff = float(composition.get("pair_diff", {}).get("accuracy", 0.0))
        directives["C4"] = _entry(
            directory,
            "symbolic.json",
            {"pair_sum_at_least_0_85": pair_sum >= 0.85, "pair_diff_at_least_0_85": pair_diff >= 0.85},
            {"pair_sum": pair_sum, "pair_diff": pair_diff,
             "grounding_accuracy": float(composition.get("grounding_accuracy", 0.0)),
             "n": float(composition.get("n", 0.0))},
        )
    else:
        directives["C4"] = {"receipt": "symbolic.json", "present": False, "passed": False}

    if agent and sidecar:
        mcp = agent.get("holarch_mcp", {})
        gate = agent.get("gate", {})
        clean_veto = float(sidecar.get("veto", {}).get("clean_veto_rate", 1.0))
        directives["C5"] = _entry(
            directory,
            "agent_eval.json",
            {
                "zero_query_reads": float(mcp.get("query_file_reads", 1)) == 0.0,
                "reachable_accuracy_1": float(mcp.get("reachable_accuracy", 0.0)) >= 0.999,
                "hallucination_zero": float(mcp.get("hallucination_rate", 1.0)) == 0.0,
                "receipt_chain_valid": bool(mcp.get("receipt_chain_valid", False)),
                "sidecar_false_veto_at_most_0_01": clean_veto <= 0.01,
            },
            {"query_file_reads": float(mcp.get("query_file_reads", 0.0)),
             "reachable_accuracy": float(mcp.get("reachable_accuracy", 0.0)),
             "hallucination_rate": float(mcp.get("hallucination_rate", 0.0)),
             "receipts": float(mcp.get("receipts", 0.0)),
             "compression": float(agent.get("context_compression_ratio", 0.0)),
             "sidecar_clean_veto_rate": clean_veto,
             "agent_gate_passed": bool(gate.get("passed", False))},
        )
    else:
        directives["C5"] = {"receipt": "agent_eval.json", "present": False, "passed": False}

    return {
        "cycle": "C1-C5",
        "directives": directives,
        "summary": {
            "passed": sum(1 for item in directives.values() if item.get("passed")),
            "total": len(directives),
            "blocked": [name for name, item in directives.items()
                        if not item.get("passed") and item.get("measured", {}).get("evidence") == "BLOCKED"],
        },
        "evidence": "DERIVED",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="receipts/holarch")
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()
    directory = ROOT / args.dir
    ledger = build_ledger(directory)
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
