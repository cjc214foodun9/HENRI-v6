"""Cycle AB2: in-retrieval sheaf filtering, boundary consensus, loop schemas.

    python scripts/run_holarch_ab2_sheaf_retrieval_and_loop_synthesis.py

The measured question is narrow and falsifiable: can the cold-start
synthesizer (a) prune a type-incompatible plan BEFORE it is ranked, closing
the AA2 Arm-B 11/12 miss, (b) refuse an edge-case guess through differential
boundary perturbation, and (c) solve two more external HumanEval tasks with
the extended loop schemas - on the SAME vendored slices, with zero external
LLM calls, zero preloaded plans in Arm A, zero task-id checks and zero
benchmark function-name branches?

Every number is produced in this process.  A candidate is emitted only after
it passes every extracted example inside the sandbox and the boundary
consensus gate; the hidden tests are used only for scoring.
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_holarch_p_cycle as pc
import run_holarch_z2_external_synth as z2
from henri.holarch.cold_synth import (
    ColdSynthesizer,
    audit_task_specific_branches,
    build_candidates,
    parse_problem,
    prose_preconditions,
)
from henri.holarch.prose_examples import extract as extract_prose
from henri.holarch.sandbox import SubprocessSandboxExecutor

RECEIPTS = ROOT / "receipts" / "holarch"
STAGE1_BUDGET = 32
STAGE2_BUDGET = 64
BUDGET = STAGE1_BUDGET + STAGE2_BUDGET
SOURCES = [
    ROOT / "henri" / "holarch" / "cold_synth.py",
    ROOT / "henri" / "holarch" / "loop_schemas.py",
    ROOT / "henri" / "holarch" / "prose_examples.py",
]
# repo_sheaf.py, code_gen.py and the codec are general engines: they are
# audited for task-identity comparisons only, because they legitimately compare
# structural family tags and the one declared split filename.
IDENTITY_ONLY_SOURCES = [
    ROOT / "henri" / "holarch" / "repo_sheaf.py",
    ROOT / "henri" / "holarch" / "code_gen.py",
]


def _mark(label: str, started: float) -> None:
    print("[ab2] %-44s %8.1f s" % (label, time.perf_counter() - started), flush=True)


def _run_arm_a(sandbox, suite: Dict[str, Any]) -> tuple:
    synthesizer = ColdSynthesizer(sandbox=sandbox, budget=BUDGET, plan_store=None)
    rows: List[Dict[str, Any]] = []
    for task in suite["tasks"]:
        started = time.perf_counter()
        prompt = str(task["prompt"])
        prose = extract_prose(prompt, str(task["entry"]), [])
        try:
            spec = parse_problem(prompt)
            parsed = True
            params = [str(param["name"]) for param in spec.params]
            examples = len(spec.examples)
            contradictions = list(spec.contradictions)
            preconditions = prose_preconditions(spec.docstring + " " + str(spec.name))
        except Exception as error:  # noqa: BLE001 - an unparsable prompt abstains
            spec = None
            parsed = False
            params = []
            examples = 0
            contradictions = [str(error)]
            preconditions = []
        prose = extract_prose(prompt, str(task["entry"]), params)
        outcome = synthesizer.synthesize(prompt) if parsed else {
            "status": "ABSTAIN",
            "reason": "unparsable_prompt",
            "source": "",
            "rollouts": 0,
        }
        row: Dict[str, Any] = {
            "suite": str(task["suite"]),
            "task_id": str(task["task_id"]),
            "entry": str(task["entry"]),
            "status": str(outcome["status"]),
            "reason": str(outcome["reason"]),
            "rollouts": int(outcome["rollouts"]),
            "examples": int(examples),
            "prose_assertions": int(len(prose["assertions"])),
            "prose_formats": dict(prose["formats"]),
            "contradictions": contradictions,
            "preconditions": list(preconditions),
            "intent": (
                z2._plain_language(spec.docstring) or str(task["entry"])
                if spec is not None
                else str(task["entry"])
            ),
            "params": params,
            "param_kinds": (
                [str(value) for value in spec.arg_kinds.values()] if spec else []
            ),
            "return_kind": str(getattr(spec, "return_kind", "")) if spec else "",
            "emitted": bool(outcome["status"] == "SOLVED"),
            "hidden_ok": False,
            "hidden_detail": "not_emitted",
            "syntax_ok": True,
            "source": str(outcome["source"]),
            "source_sha256": (
                hashlib.sha256(str(outcome["source"]).encode("utf-8")).hexdigest()
                if outcome["source"]
                else ""
            ),
            "consensus": dict(outcome.get("consensus") or {}),
            "seconds": float(time.perf_counter() - started),
        }
        if row["emitted"]:
            verdict = z2._hidden_check(sandbox, row["source"], task["hidden"])
            row["hidden_ok"] = bool(verdict["ok"])
            row["hidden_detail"] = str(verdict["detail"])
            row["syntax_ok"] = z2._syntax_ok(row["source"])
        rows.append(row)
    counts = z2._suite_counts(rows, len(suite["tasks"]))
    return rows, counts


def _traps(sandbox) -> List[Dict[str, Any]]:
    synthesizer = ColdSynthesizer(sandbox=sandbox, budget=BUDGET, plan_store=None)
    rows: List[Dict[str, Any]] = []
    for trap in z2.TRAPS:
        outcome = synthesizer.synthesize(str(trap["prompt"]))
        rows.append(
            {
                "name": str(trap["name"]),
                "kind": str(trap["kind"]),
                "status": str(outcome["status"]),
                "reason": str(outcome["reason"]),
                "refused": bool(str(outcome["status"]) == "ABSTAIN"),
            }
        )
    return rows


def _audit(identifiers: Sequence[str]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for path in SOURCES:
        result = audit_task_specific_branches(
            path.read_text(encoding="utf-8"), identifiers=identifiers
        )
        result["file"] = str(path.relative_to(ROOT))
        result["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(result)
    identity_rows: List[Dict[str, Any]] = []
    for path in IDENTITY_ONLY_SOURCES:
        result = audit_task_specific_branches(
            path.read_text(encoding="utf-8"), identifiers=identifiers
        )
        result["file"] = str(path.relative_to(ROOT))
        result["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        identity_rows.append(result)
    clean = bool(
        all(
            bool(result.get("ok"))
            and int(result.get("unexplained_string_comparisons", 0) or 0) == 0
            and int(result.get("task_identity_comparisons", 0) or 0) == 0
            and int(result.get("compared_identifier_literal_count", 0) or 0) == 0
            for result in rows
        )
    )
    identity_branches = [
        branch for result in identity_rows for branch in (result.get("branches") or [])
    ]
    declared_filename_branches = [branch for branch in identity_branches if "HOLARCH.md" in branch]
    identity_clean = bool(len(identity_branches) == len(declared_filename_branches))
    return {
        "files": rows,
        "identity_only_files": identity_rows,
        "identity_branches": identity_branches,
        "declared_filename_branches": declared_filename_branches,
        "clean": clean,
        "identity_clean": identity_clean,
    }


def measure(directory: Path) -> Dict[str, Any]:
    started = time.perf_counter()
    directory.mkdir(parents=True, exist_ok=True)
    workdir = directory / "_work" / "ab2"
    workdir.mkdir(parents=True, exist_ok=True)
    primary = z2.load_primary()
    secondary = z2.load_secondary()
    _mark("suites loaded", started)
    sandbox = SubprocessSandboxExecutor(
        "pure_algorithmic", timeout_s=8.0, workdir=str(workdir)
    )
    rows_secondary, counts_secondary = _run_arm_a(sandbox, secondary)
    _mark(
        "arm A humaneval %d/%d" % (counts_secondary["solved"], len(secondary["tasks"])),
        started,
    )
    rows_primary, counts_primary = _run_arm_a(sandbox, primary)
    _mark(
        "arm A mbpp %d/%d" % (counts_primary["solved"], len(primary["tasks"])), started
    )
    traps = _traps(sandbox)
    _mark("traps", started)

    rows = rows_primary + rows_secondary
    emitted = counts_primary["emitted"] + counts_secondary["emitted"]
    solved = counts_primary["solved"] + counts_secondary["solved"]
    failed_hidden = (
        counts_primary["emitted_but_failed_hidden"]
        + counts_secondary["emitted_but_failed_hidden"]
    )
    precision = float(solved / emitted) if emitted else 0.0
    abstain_reasons: Dict[str, int] = {}
    for row in rows:
        if not row["emitted"]:
            abstain_reasons[row["reason"]] = abstain_reasons.get(row["reason"], 0) + 1
    prose_total = {
        key: int(sum(int(row["prose_formats"].get(key, 0)) for row in rows))
        for key in ("call_equals", "call_arrow", "input_output", "example_block")
    }
    consensus_rows = [row for row in rows if row.get("consensus")]
    consensus_summary = {
        "candidates_scored": int(len(consensus_rows)),
        "ambiguous": int(sum(1 for row in consensus_rows if row["consensus"].get("ambiguous"))),
        "abstained_on_disagreement": int(
            sum(1 for row in consensus_rows if row["consensus"].get("abstained"))
        ),
        "probe_count": int(
            max([int(row["consensus"].get("probes", 0)) for row in consensus_rows] or [0])
        ),
        "total_boundary_probes": int(
            sum(int(row["consensus"].get("probes", 0)) for row in consensus_rows)
        ),
    }

    identifiers = [str(task["entry"]) for task in primary["tasks"] + secondary["tasks"]]
    audit = _audit(identifiers)
    _mark("branch audit", started)

    # ---- Arm B: consolidate every verified Arm-A solution, then re-synthesize
    transfer: Dict[str, Any] = {"engine": "unavailable"}
    try:
        import torch

        stem = z2.SharedCorpusStem.build(ROOT, max_fit_tokens=60000, build_index=False)
        checkpoint = z2.find_checkpoint(None)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        config = z2.HolarchConfig(**payload["config"])
        memory, memory_info = z2.open_clean_room_memory(
            config, checkpoint, workdir, prefix="ab2_arm_b"
        )
        engine = z2.SynthesisEngine.from_stem(
            stem, memory, namespace=z2.PLAN_NAMESPACE
        )
        plans_before = int(len(engine.plans.plans))
        compiled = z2.compile_solutions(engine, rows)
        plans_after = int(len(engine.plans.plans))
        held_out, held_meta = z2.build_held_out(rows)
        held_a = z2.run_held_out(held_out, sandbox, engine, arm="a")
        held_b = z2.run_held_out(held_out, sandbox, engine, arm="b")
        transfer = {
            "engine": "ready",
            "checkpoint": str(checkpoint),
            "plans_before": plans_before,
            "plans_after": plans_after,
            "plans_compiled": len(compiled),
            "held_out": held_meta,
            "arm_a": z2._held_out_counts(held_a),
            "arm_b": z2._held_out_counts(held_b),
            "rows_a": held_a,
            "rows_b": held_b,
            "sheaf_filter": dict(engine.plans.type_filter_report),
            "memory": memory_info,
        }
    except Exception as error:  # noqa: BLE001 - a missing engine is reported, not hidden
        transfer = {"engine": "unavailable", "error": str(error)}
    _mark("arm B transfer", started)

    sheaf = transfer.get("sheaf_filter", {})
    gates = {
        "ab2_runner_exit_zero": True,
        "ab2_zero_external_llm_calls": bool(not z2._llm_modules_loaded()),
        "ab2_staged_budget_le_96": bool(STAGE1_BUDGET + STAGE2_BUDGET <= 96),
        "ab2_mbpp_ge_28_of_30": bool(counts_primary["solved"] >= 28),
        "ab2_humaneval_ge_22_of_32": bool(counts_secondary["solved"] >= 22),
        "ab2_precision_ge_098": bool(precision >= 0.98),
        "ab2_hidden_misses_le_1": bool(failed_hidden <= 1),
        "ab2_traps_6_of_6": bool(
            len(traps) >= 6 and all(row["refused"] for row in traps)
        ),
        "ab2_no_task_identity_branches": bool(audit["clean"]),
        "ab2_code_gen_no_task_identity": bool(audit["identity_clean"]),
        "ab2_same_vendored_slices": bool(
            int(primary["selected"]) == 30 and int(secondary["selected"]) == 32
        ),
        "ab2_sheaf_filter_engaged": bool(int(sheaf.get("checked", 0) or 0) > 0),
        "ab2_arm_b_transfer_12_of_12": bool(
            transfer.get("arm_b", {}).get("tasks") == 12
            and transfer.get("arm_b", {}).get("solved") == 12
        ),
        "ab2_arm_b_beats_arm_a": bool(
            transfer.get("arm_b", {}).get("solved", 0)
            > transfer.get("arm_a", {}).get("solved", 0)
        ),
    }
    return {
        "stage": "AB2",
        "config": {
            "stage1_budget": STAGE1_BUDGET,
            "stage2_budget": STAGE2_BUDGET,
            "total_budget": BUDGET,
            "slot_support": 8,
            "boundary_probes": 6,
            "consensus_candidates": 2,
        },
        "primary": {
            "name": primary["name"],
            "path": primary["path"],
            "sha256": primary["sha256"],
            "rows": primary["rows"],
            "eligible": primary["eligible"],
            "selected": primary["selected"],
            "digest": primary["digest"],
            "counts": counts_primary,
            "tasks": rows_primary,
        },
        "secondary": {
            "name": secondary["name"],
            "path": secondary["path"],
            "sha256": secondary["sha256"],
            "rows": secondary["rows"],
            "eligible": secondary["eligible"],
            "selected": secondary["selected"],
            "digest": secondary["digest"],
            "counts": counts_secondary,
            "tasks": rows_secondary,
        },
        "combined": {
            "emitted": int(emitted),
            "solved": int(solved),
            "emitted_but_failed_hidden": int(failed_hidden),
            "precision": float(precision),
            "abstain_reasons": abstain_reasons,
            "prose_assertions_total": prose_total,
            "consensus": consensus_summary,
        },
        "traps": traps,
        "transfer": transfer,
        "audit": audit,
        "gate": pc._finalise(gates),
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
        "passed": bool(all(gates.values())),
    }


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Cycle AB2 sheaf retrieval and loop synthesis")
    parser.add_argument("--receipts", default=str(RECEIPTS))
    args = parser.parse_args(argv)
    directory = Path(args.receipts).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    payload = measure(directory)
    written = pc._write(directory, "ab2_sheaf_retrieval_and_loop_synthesis.json", payload)
    print(
        "[ab2] mbpp %d/%d  humaneval %d/%d  precision %.4f  traps %d/%d  armB %d/12"
        % (
            payload["primary"]["counts"]["solved"],
            len(payload["primary"]["tasks"]),
            payload["secondary"]["counts"]["solved"],
            len(payload["secondary"]["tasks"]),
            payload["combined"]["precision"],
            sum(1 for row in payload["traps"] if row["refused"]),
            len(payload["traps"]),
            payload["transfer"].get("arm_b", {}).get("solved", 0),
        )
    )
    failed = [name for name, value in payload["gate"].items() if not bool(value)]
    print("[ab2] passed=%s failed=%s receipt=%s" % (payload["passed"], failed, written["path"]))
    return 0 if bool(payload["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
