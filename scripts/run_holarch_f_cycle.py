"""Cycle F verification ledger: F1, F2, F3, and F4.

Evidence label: OBSERVED for every measured value, DERIVED for the verdicts.

    python scripts/run_holarch_f_cycle.py

F1  multi-tenant namespace isolation in Zone C and the sidecar API
F2  learned unitary action transition rotor and 2-step lookahead
F3  MCP memory-stream transport parity and zero-exclusion test suite
F4  full-repository impact analysis and the NVIDIA driver diagnostic
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from henri.holarch.config import HolarchConfig
from henri.holarch.model import (
    evaluate_closed_loop_action,
    fit_action_rotor,
    load_checkpoint,
)
from henri.holarch.server import SidecarService, verify_receipt
from henri.holarch.tasks import build_tasks
from henri.holarch.zone_c import ZoneCMemory


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(directory: Path, name: str, payload: Dict[str, object]) -> Dict[str, str]:
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    digest = _digest(path)
    path.with_name(path.name + ".sha256").write_text(digest + chr(10), encoding="utf-8")
    return {"path": str(path.relative_to(ROOT)), "sha256": digest}


# ---------------------------------------------------------------------- F1
def stage_f1(directory: Path) -> Dict[str, object]:
    memory_path = directory / "_work" / "f1_namespaces.sqlite3"
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    if memory_path.exists():
        memory_path.unlink()
    service = SidecarService(
        HolarchConfig.small(), backend="sqlite", memory_path=str(memory_path), autoload=False
    )
    rows: List[Dict[str, object]] = []
    leakage = 0
    try:
        service.ingest_edges(
            [
                {"source": "s00", "operator": "fwd", "target": "s01"},
                {"source": "s01", "operator": "fwd", "target": "s02"},
                {"source": "s02", "operator": "fwd", "target": "s03"},
            ],
            namespace="tenant_alpha",
        )
        service.ingest_edges(
            [
                {"source": "s00", "operator": "fwd", "target": "z01"},
                {"source": "z01", "operator": "fwd", "target": "z02"},
            ],
            namespace="tenant_beta",
        )
        alpha = service.solve("s00", ["fwd", "fwd"], namespace="tenant_alpha")
        beta = service.solve("s00", ["fwd"], namespace="tenant_beta")
        alpha_cross = service.solve("z01", ["fwd"], namespace="tenant_alpha")
        beta_cross = service.solve("s01", ["fwd"], namespace="tenant_beta")
        rows = [
            {"case": "alpha_own_path", "answer": alpha["result"]["answer"], "vetoed": alpha["result"]["vetoed"]},
            {"case": "beta_own_path", "answer": beta["result"]["answer"], "vetoed": beta["result"]["vetoed"]},
            {"case": "alpha_on_beta_symbol", "answer": alpha_cross["result"]["answer"], "vetoed": alpha_cross["result"]["vetoed"]},
            {"case": "beta_on_alpha_symbol", "answer": beta_cross["result"]["answer"], "vetoed": beta_cross["result"]["vetoed"]},
        ]
        if alpha["result"]["answer"] != "s02":
            leakage += 1
        if beta["result"]["answer"] != "z01":
            leakage += 1
        if alpha_cross["result"]["answer"] is not None or not alpha_cross["result"]["vetoed"]:
            leakage += 1
        if beta_cross["result"]["answer"] is not None or not beta_cross["result"]["vetoed"]:
            leakage += 1
        receipts_valid = all(
            verify_receipt(item["receipt"])
            for item in (alpha, beta, alpha_cross, beta_cross)
        )
        payload = {
            "stage": "F1",
            "namespace": "tenant_alpha|tenant_beta",
            "rows": rows,
            "cross_tenant_leakage": int(leakage),
            "leakage_rate": float(leakage / 4.0),
            "receipts_valid": bool(receipts_valid),
            "gate": {
                "alpha_own_path_1_0": bool(alpha["result"]["answer"] == "s02"),
                "beta_own_path_1_0": bool(beta["result"]["answer"] == "z01"),
                "alpha_refuses_beta_1_0": bool(alpha_cross["result"]["vetoed"]),
                "beta_refuses_alpha_1_0": bool(beta_cross["result"]["vetoed"]),
                "leakage_zero": bool(leakage == 0),
                "receipts_valid": bool(receipts_valid),
            },
            "evidence": "OBSERVED",
        }
        payload["gate"]["passed"] = bool(all(payload["gate"].values()))
    finally:
        service.close()
    return payload


# ---------------------------------------------------------------------- F2
def stage_f2(directory: Path, checkpoint: Path) -> Dict[str, object]:
    model, metadata = load_checkpoint(checkpoint)
    cfg = model.cfg
    tasks = build_tasks(cfg, seed=cfg.seed)
    task = tasks["action"]
    fit = fit_action_rotor(model, task, episodes=40)
    rotor = fit["rotor"]
    arms: Dict[str, Dict[str, object]] = {}
    started = time.perf_counter()
    arms["shift_depth1"] = evaluate_closed_loop_action(model, task, episodes=48)
    arms["rotor_depth1"] = evaluate_closed_loop_action(
        model, task, episodes=48, rotor=rotor, transition="rotor"
    )
    arms["blend_depth1"] = evaluate_closed_loop_action(
        model, task, episodes=48, rotor=rotor, transition="blend"
    )
    arms["rotor_depth2"] = evaluate_closed_loop_action(
        model, task, episodes=48, rotor=rotor, transition="rotor", lookahead_depth=2
    )
    arms["shift_depth2"] = evaluate_closed_loop_action(
        model, task, episodes=48, lookahead_depth=2
    )
    elapsed = time.perf_counter() - started
    best = max(arms.items(), key=lambda pair: (pair[1]["success_rate"], -pair[1]["mean_steps_to_goal"]))
    payload = {
        "stage": "F2",
        "checkpoint": {
            "path": str(checkpoint.relative_to(ROOT)),
            "sha256": str(metadata.get("sha256", "")),
            "parameters": int(metadata.get("parameters", 0)),
            "dim": int(cfg.dim),
        },
        "rotor": {
            "fit": fit["fit"],
            "held_out": fit["held_out"],
            "state": fit["state"],
        },
        "arms": {
            name: {
                "success_rate": float(arm["success_rate"]),
                "wins": int(arm["wins"]),
                "episodes": int(arm["episodes"]),
                "mean_steps_to_goal": float(arm["mean_steps_to_goal"]),
                "transition": str(arm["transition"]),
                "lookahead_depth": int(arm["lookahead_depth"]),
                "reactive_success_rate": float(arm["reactive_success_rate"]),
            }
            for name, arm in arms.items()
        },
        "best_arm": {
            "name": str(best[0]),
            "success_rate": float(best[1]["success_rate"]),
            "mean_steps_to_goal": float(best[1]["mean_steps_to_goal"]),
        },
        "improves_over_cycle_e": bool(
            float(arms["rotor_depth1"]["success_rate"]) > float(arms["shift_depth1"]["success_rate"])
        ),
        "mean_steps_below_5_29": bool(
            float(best[1]["mean_steps_to_goal"]) < 5.29
        ),
        "wall_seconds": float(elapsed),
        "gate": {
            "held_out_coherence_at_least_0_85": bool(fit["held_out"]["coherence"] >= 0.85),
            "closed_loop_not_degraded": bool(float(arms["shift_depth1"]["success_rate"]) >= 0.7083),
            "depth2_reported": bool("rotor_depth2" in arms),
            "rotor_reported": bool("rotor_depth1" in arms),
        },
        "evidence": "OBSERVED",
    }
    payload["gate"]["passed"] = bool(all(payload["gate"].values()))
    return payload


# ---------------------------------------------------------------------- F3
def stage_f3(directory: Path) -> Dict[str, object]:
    from mcp.shared.memory import create_connected_server_and_client_session

    from henri.holarch import mcp_server

    memory_path = directory / "_work" / "f3_transport.sqlite3"
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    if memory_path.exists():
        memory_path.unlink()
    service = SidecarService(
        HolarchConfig.small(), backend="sqlite", memory_path=str(memory_path), autoload=False
    )
    mcp_server.set_service(service)
    try:
        async def probe():
            async with create_connected_server_and_client_session(mcp_server.mcp) as session:
                await asyncio.wait_for(session.initialize(), timeout=120)
                listed = await asyncio.wait_for(session.list_tools(), timeout=120)
                names = sorted(tool.name for tool in listed.tools)
                service.ingest_edges(
                    [{"source": "a", "operator": "fwd", "target": "b"}],
                    namespace="tenant_alpha",
                )
                solved = await asyncio.wait_for(
                    session.call_tool(
                        "holarch_solve",
                        {"source": "a", "operators": ["fwd"], "namespace": "tenant_alpha"},
                    ),
                    timeout=120,
                )
                composed = await asyncio.wait_for(
                    session.call_tool(
                        "holarch_composite",
                        {"mode": "ingest", "rules": [{"a": "d1", "op": "add", "b": "d2", "result": "d3"}]},
                    ),
                    timeout=120,
                )
                return names, solved, composed

        names, solved, composed = asyncio.run(probe())
        solve_payload = json.loads(solved.content[0].text)
        composite_payload = json.loads(composed.content[0].text)
        receipts_valid = bool(
            verify_receipt(solve_payload["receipt"]) and verify_receipt(composite_payload["receipt"])
        )
        payload = {
            "stage": "F3",
            "transport": "in-process-anyio-memory-object-stream",
            "stdio_pipe_status": "NOT_REQUIRED_IN_PROCESS",
            "tools": names,
            "solve_answer": solve_payload["result"]["answer"],
            "composite_ingested": composite_payload["result"]["rules_ingested"],
            "receipts_valid": receipts_valid,
            "gate": {
                "handshake_completed": bool(names),
                "tool_listing_present": bool("holarch_solve" in names),
                "holarch_solve_called": bool(solve_payload["result"]["answer"] == "b"),
                "holarch_composite_called": bool(
                    composite_payload["result"]["rules_ingested"] == 1.0
                ),
                "receipts_valid": receipts_valid,
            },
            "evidence": "OBSERVED",
        }
        payload["gate"]["passed"] = bool(all(payload["gate"].values()))
        return payload
    finally:
        mcp_server.set_service(None)
        service.close()


# ---------------------------------------------------------------------- F4
def stage_f4(directory: Path) -> Dict[str, object]:
    from henri.holarch.budget import probe_nvidia_driver_state

    agent_path = directory / "agent_eval.json"
    budget_path = directory / "budget.json"
    agent = json.loads(agent_path.read_text(encoding="utf-8")) if agent_path.exists() else {}
    budget = json.loads(budget_path.read_text(encoding="utf-8")) if budget_path.exists() else {}
    driver = budget.get("hardware", {}).get("nvidia_driver", {})
    if not driver:
        driver = probe_nvidia_driver_state()
    impact = agent.get("per_category", {}).get("impact_analysis", {})
    traps = agent.get("per_category", {}).get("impact_traps", {})
    payload = {
        "stage": "F4",
        "agent_eval": {
            "queries_total": int(agent.get("queries_total", 0)),
            "reachable_accuracy": float(agent.get("holarch_mcp", {}).get("reachable_accuracy", 0.0)),
            "hallucination_rate": float(agent.get("holarch_mcp", {}).get("hallucination_rate", 1.0)),
            "query_file_reads": int(agent.get("holarch_mcp", {}).get("query_file_reads", -1)),
            "receipts": int(agent.get("holarch_mcp", {}).get("receipts", 0)),
            "receipt_chain_valid": bool(agent.get("holarch_mcp", {}).get("receipt_chain_valid", False)),
            "code_index": agent.get("code_index", {}),
            "impact_queries": int(impact.get("total", 0)),
            "impact_correct": int(impact.get("mcp_correct", 0)),
            "impact_traps": int(traps.get("total", 0)),
            "impact_traps_refused": int(traps.get("mcp_refused", 0)),
        },
        "nvidia_driver": driver,
        "gate": {
            "queries_52": bool(int(agent.get("queries_total", 0)) >= 52),
            "reachable_accuracy_1_0": bool(
                float(agent.get("holarch_mcp", {}).get("reachable_accuracy", 0.0)) >= 0.999
            ),
            "hallucination_zero": bool(
                float(agent.get("holarch_mcp", {}).get("hallucination_rate", 1.0)) == 0.0
            ),
            "zero_query_reads": bool(int(agent.get("holarch_mcp", {}).get("query_file_reads", -1)) == 0),
            "receipt_chain_valid": bool(agent.get("holarch_mcp", {}).get("receipt_chain_valid", False)),
            "impact_5_of_6_or_better": bool(
                int(impact.get("mcp_correct", 0)) >= max(1, int(impact.get("total", 1)) - 1)
            ),
            "impact_traps_refused": bool(
                int(traps.get("mcp_refused", 0)) == int(traps.get("total", -1))
            ),
            "driver_fields_recorded": bool(
                "nvidia_smi_reachable" in driver and "root_cause" in driver
            ),
        },
        "evidence": "OBSERVED",
    }
    payload["gate"]["passed"] = bool(all(payload["gate"].values()))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="receipts/holarch")
    parser.add_argument("--checkpoint", default="receipts/holarch/holarch_medium.pt")
    args = parser.parse_args()
    directory = ROOT / args.dir
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    f1 = stage_f1(directory)
    f1_receipt = _write(directory, "f1_namespaces.json", f1)
    f2 = stage_f2(directory, (ROOT / args.checkpoint).resolve())
    f2_receipt = _write(directory, "f2_rotor.json", f2)
    f3 = stage_f3(directory)
    f3_receipt = _write(directory, "f3_transport.json", f3)
    f4 = stage_f4(directory)
    f4_receipt = _write(directory, "f4_impact.json", f4)

    upstream = {}
    for name in ("holarch_medium.json", "agent_eval.json", "budget.json", "code_lang_eval.json"):
        path = directory / name
        upstream[name] = _digest(path) if path.exists() else ""

    ledger = {
        "cycle": "F1-F4",
        "protocol": "multi-tenant wave planning sidecar",
        "git_head": _git_head(),
        "directives": {"F1": f1, "F2": f2, "F3": f3, "F4": f4},
        "receipts": {"F1": f1_receipt, "F2": f2_receipt, "F3": f3_receipt, "F4": f4_receipt},
        "upstream_receipts": upstream,
        "summary": {
            "passed": sum(1 for item in (f1, f2, f3, f4) if item["gate"]["passed"]),
            "total": 4,
        },
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "DERIVED",
    }
    out = directory / "f_cycle.json"
    out.write_text(json.dumps(ledger, indent=2, default=float), encoding="utf-8")
    digest = _digest(out)
    out.with_name(out.name + ".sha256").write_text(digest + chr(10), encoding="utf-8")
    print(json.dumps(ledger["summary"], indent=2))
    for name, item in ledger["directives"].items():
        print(name, "PASS" if item["gate"]["passed"] else "FAIL", json.dumps(item["gate"]))
    print("receipt", str(out.relative_to(ROOT)), digest)
    return 0 if ledger["summary"]["passed"] == ledger["summary"]["total"] else 1


def _git_head() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), capture_output=True, text=True, timeout=20
        )
        return result.stdout.strip()
    except Exception:
        return ""


if __name__ == "__main__":
    raise SystemExit(main())
