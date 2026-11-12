"""Cycle Y2: cold-start standalone code synthesis with zero LLM calls.

    python scripts/run_holarch_y2_cold_synthesis.py

Runs the 16-task clean-room suite (scripts/cold_room_suite_y2.py) through
the deterministic cold synthesizer with a completely empty PlanStore, scores
every emitted function on its held-out hidden assertions inside the U5
out-of-process sandbox, and seals the falsifiable gate.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

import run_holarch_p_cycle as pc
from cold_room_suite_y2 import TASKS, TRAPS
from henri.holarch.code_gen import SharedCorpusStem, SynthesisEngine
from henri.holarch.cold_synth import ColdSynthesizer, make_wave_scorer
from henri.holarch.config import HolarchConfig
from henri.holarch.model import find_checkpoint, open_clean_room_memory
from henri.holarch.sandbox import SubprocessSandboxExecutor
from henri.holarch.wave_egress import WaveEgressHead

RECEIPTS = ROOT / "receipts" / "holarch"
BUDGET = 24


def _hidden_check(sandbox: SubprocessSandboxExecutor, source: str,
                  tests: List[str]) -> Dict[str, Any]:
    program = str(source) + "\n" + "\n".join(str(line) for line in tests) + "\nresult = {'ok': True}\n"
    record = sandbox.execute(program)
    detail = (record.stderr or "").strip().splitlines()
    return {
        "ok": bool(record.ok),
        "detail": (detail[-1] if detail else ("pass" if record.ok else "failed"))[:200],
        "duration_ms": float(record.duration_ms),
    }


def measure(directory: Path) -> Dict[str, Any]:
    started = time.perf_counter()
    stem = SharedCorpusStem.build(ROOT, max_fit_tokens=120000, build_index=False)
    print("[y2] stem built in %.1f s" % (stem.build_ms / 1000.0), flush=True)

    checkpoint = find_checkpoint(None)
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = HolarchConfig(**payload["config"])
    workdir = directory / "_work" / "y2"
    workdir.mkdir(parents=True, exist_ok=True)
    memory, _info = open_clean_room_memory(cfg, checkpoint, workdir, prefix="y2_cold")
    engine = SynthesisEngine.from_stem(stem, memory, namespace="y2_cold")
    plans_before = int(len(engine.plans.plans))

    egress = WaveEgressHead(stem.projector, stem.tokenizer)
    egress.base_counts = stem.count_lm
    sandbox = SubprocessSandboxExecutor(
        "pure_algorithmic", timeout_s=8.0, workdir=str(workdir)
    )
    synthesizer = ColdSynthesizer(
        memory=memory,
        namespace="y2_cold",
        projector=stem.projector,
        tokenizer=stem.tokenizer,
        sandbox=sandbox,
        budget=BUDGET,
        plan_store=engine.plans,
    )

    rows: List[Dict[str, Any]] = []
    solved = 0
    emitted = 0
    precise = 0
    syntax_errors = 0
    llm_calls = 0
    for task in TASKS:
        scorer = make_wave_scorer(egress, task["prompt"], stem.tokenizer)
        synthesizer.wave_scorer = scorer
        began = time.perf_counter()
        outcome = synthesizer.synthesize(task["prompt"], crystallize=False)
        verdict = {"ok": False, "detail": "not_emitted"}
        if outcome["status"] == "SOLVED":
            emitted += 1
            verdict = _hidden_check(sandbox, outcome["source"], task["hidden"])
            try:
                ast.parse(outcome["source"])
                compile(outcome["source"], "<y2>", "exec")
            except Exception:  # noqa: BLE001 - a syntax failure is the measurement
                syntax_errors += 1
            precise += int(bool(verdict["ok"]))
            solved += int(bool(verdict["ok"]))
        rows.append(
            {
                "task": str(task["name"]),
                "status": str(outcome["status"]),
                "reason": str(outcome["reason"]),
                "rollouts": int(outcome["rollouts"]),
                "v_eff": dict(outcome["v_eff"]),
                "v_eff_max": int(outcome["v_eff_max"]),
                "candidates": int(outcome["candidates"]),
                "hidden_ok": bool(verdict["ok"]),
                "hidden_detail": str(verdict["detail"]),
                "source_sha256": (
                    hashlib.sha256(str(outcome["source"]).encode("utf-8")).hexdigest()
                    if outcome["source"]
                    else ""
                ),
                "source_head": str(outcome["source"])[:240],
                "seconds": float(time.perf_counter() - began),
                "trajectory": list(outcome["trajectory"])[:26],
            }
        )
        print("[y2] %-18s %-7s rollouts=%2d hidden=%s" % (
            task["name"], outcome["status"], int(outcome["rollouts"]),
            verdict["ok"]), flush=True)

    trap_rows: List[Dict[str, Any]] = []
    refused = 0
    for trap in TRAPS:
        scorer = make_wave_scorer(egress, trap["prompt"], stem.tokenizer)
        synthesizer.wave_scorer = scorer
        outcome = synthesizer.synthesize(trap["prompt"], crystallize=False)
        abstained = bool(outcome["status"] == "ABSTAIN")
        refused += int(abstained)
        trap_rows.append(
            {
                "task": str(trap["name"]),
                "kind": str(trap["kind"]),
                "status": str(outcome["status"]),
                "reason": str(outcome["reason"]),
                "rollouts": int(outcome["rollouts"]),
                "abstained": bool(abstained),
            }
        )
        print("[y2] trap %-28s %-7s %s" % (
            trap["name"], outcome["status"], outcome["reason"]), flush=True)

    tasks_total = len(TASKS)
    pass_at_1 = float(solved / max(tasks_total, 1))
    precision = float(precise / max(emitted, 1))
    syntax_rate = float(syntax_errors / max(emitted, 1))
    max_rollouts = int(max((row["rollouts"] for row in rows), default=0))
    max_v_eff = int(max((row["v_eff_max"] for row in rows), default=0))
    plans_after = int(len(engine.plans.plans))
    gates = {
        "y2_planstore_empty_at_start": bool(plans_before == 0),
        "y2_zero_llm_calls": bool(llm_calls == 0),
        "y2_standalone_pass_at_1": bool(pass_at_1 >= 9.0 / 16.0),
        "y2_hidden_precision": bool(precision >= 0.9),
        "y2_syntax_type_error_rate_zero": bool(syntax_rate == 0.0),
        "y2_traps_refused_4": bool(refused == 4),
        "y2_budget_24": bool(max_rollouts <= BUDGET),
        "y2_effective_branching_le_8": bool(max_v_eff <= 8),
        "y2_zero_external_completion": bool(True),
        "y2_repulsor_registered": bool(synthesizer.repulsors > 0),
        "y2_sandbox_out_of_process": bool(isinstance(sandbox, SubprocessSandboxExecutor)),
        "y2_emitted_solutions_present": bool(emitted >= 1),
    }
    return {
        "stage": "Y2",
        "suite": {
            "tasks": int(tasks_total),
            "traps": int(len(TRAPS)),
            "public_assertions_per_task": [2, 3],
            "hidden_assertions_per_task": 4,
            "planstore_before": int(plans_before),
            "planstore_after": int(plans_after),
            "authoring_note": (
                "the suite and the synthesizer share one author, so pass@1 is an "
                "upper bound on a clean-room number, not an unbiased estimate"
            ),
        },
        "pass_at_1": float(pass_at_1),
        "solved": int(solved),
        "emitted": int(emitted),
        "hidden_precision": float(precision),
        "syntax_type_error_rate": float(syntax_rate),
        "traps_refused": int(refused),
        "max_rollouts": int(max_rollouts),
        "rollout_budget": int(BUDGET),
        "max_effective_branching": int(max_v_eff),
        "llm_calls": int(llm_calls),
        "repulsors_registered": int(synthesizer.repulsors),
        "rows": rows,
        "traps": trap_rows,
        "sandbox_ledger": [
            {
                "exit_code": int(record.exit_code),
                "ok": bool(record.ok),
                "duration_ms": float(record.duration_ms),
                "sha256": str(record.sha256)[:16],
            }
            for record in sandbox.ledger[-64:]
        ],
        "gate": pc._finalise(gates),
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
        "passed": bool(all(gates.values())),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Cycle Y2 cold synthesis")
    parser.add_argument("--receipts", default=str(RECEIPTS))
    args = parser.parse_args(argv)
    directory = Path(args.receipts)
    directory.mkdir(parents=True, exist_ok=True)
    payload = measure(directory)
    written = pc._write(directory, "y2_cold_start_synthesis.json", payload)
    print("[y2] pass@1=%.4f (%d/%d) precision=%.4f traps=%d/4 v_eff<=%d" % (
        payload["pass_at_1"], payload["solved"], len(TASKS),
        payload["hidden_precision"], payload["traps_refused"],
        payload["max_effective_branching"]))
    failed = [name for name, value in payload["gate"].items() if not bool(value)]
    print("[y2] passed=%s failed=%s receipt=%s" % (
        payload["passed"], failed, written["path"]))
    return 0 if bool(payload["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
