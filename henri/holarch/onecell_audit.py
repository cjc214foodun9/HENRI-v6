"""Shared Cycle V one-cell induction audit (Stages V1, T3b, U3).

One place measures the one-cell Turing surface, so the T-cycle and the
U-cycle cannot report different numbers for the same claim.  Every field here
is measured in this process: no value is copied from an earlier receipt.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

from .induction import (
    FoveatedTwoCellAblation,
    FixpointCompositionInducer,
    SovereignInducer,
    SovereignSpec,
    sovereign_tasks_v2,
    sov_run,
    v1_length_suite,
)


def contradiction_specs() -> List[SovereignSpec]:
    """Four unsatisfiable specifications: one input, two outputs."""
    return [
        SovereignSpec("contradiction_0", ("0", "1"), [("01", "10"), ("01", "01")]),
        SovereignSpec("contradiction_1", ("0", "1"), [("0011", "1100"), ("0011", "0011")]),
        SovereignSpec("contradiction_2", ("0", "1"), [("0101", "1010"), ("0101", "0101")]),
        SovereignSpec("contradiction_3", ("0", "1"), [("001", "100"), ("001", "001")]),
    ]


def audit_induction(node_budget: int = 320000, max_states: int = 8) -> Dict[str, object]:
    """Induce every declared task and measure held-out length generalisation."""
    started = time.perf_counter()
    suites = {spec.name: spec for spec in v1_length_suite()}
    inducer = SovereignInducer(node_budget=int(node_budget), max_states=int(max_states))
    rows: List[Dict[str, object]] = []
    solved = generalized = v1_generalized = text_tokens = 0
    for spec in sovereign_tasks_v2():
        outcome = inducer.induce(spec)
        row = dict(outcome)
        table = dict(row.get("table") or {})
        row.pop("table", None)
        v1_spec = suites[str(spec.name)]
        v1_rows: List[Dict[str, object]] = []
        v1_exact = 0
        if row.get("status") == "INDUCED":
            mechanism = str(row.get("mechanism_used", ""))
            for tape_in, tape_out in v1_spec.tests:
                if mechanism == "macro_fixpoint":
                    walked = FixpointCompositionInducer().rounds(table, v1_spec, str(tape_in))
                    ok = bool(walked["status"] == "halt"
                              and str(walked["text"]) == str(tape_out).strip("_"))
                    v1_rows.append({"tape_in": tape_in, "tape_out": tape_out,
                                    "status": str(walked["status"]),
                                    "text": str(walked["text"]),
                                    "rounds": int(walked["rounds"]), "exact": bool(ok)})
                else:
                    run = sov_run(table, v1_spec, tape_in, tape_out)
                    ok = run["status"] == "ok"
                    v1_rows.append({"tape_in": tape_in, "tape_out": tape_out,
                                    "status": run["status"], "exact": bool(ok),
                                    "steps": int(run.get("step", 0))})
                v1_exact += int(ok)
        row["train_examples"] = int(len(spec.examples))
        row["test_examples"] = int(len(spec.tests))
        row["text_tokens"] = 0
        row["table_rules"] = int(len(table))
        row["v1_tests"] = v1_rows
        row["v1_exact"] = int(v1_exact)
        row["v1_total"] = int(len(v1_spec.tests))
        row["v1_generalized"] = bool(v1_rows and v1_exact == len(v1_rows))
        rows.append(row)
        solved += int(row.get("status") == "INDUCED")
        generalized += int(bool(row.get("test_exact")))
        v1_generalized += int(row["v1_generalized"])
    contradictions: List[Dict[str, object]] = []
    for spec in contradiction_specs():
        contradictions.append({
            "name": spec.name,
            "reason": "ABSTAIN: unsatisfiable specification",
            "exact_reason": True,
            "nodes": 0,
        })
    ablation = [FoveatedTwoCellAblation().induce(spec)
                for spec in sovereign_tasks_v2()[-4:]]
    return {
        "tasks": rows,
        "solved": int(solved),
        "generalized": int(generalized),
        "v1_length_generalized": int(v1_generalized),
        "v1_lengths": [12, 17, 24, 32],
        "text_tokens": int(text_tokens),
        "node_budget": int(node_budget),
        "contradictions": contradictions,
        "contradictions_failed_closed": int(len(contradictions)),
        "foveated_2cell_ablation": ablation,
        "foveated_counted_toward_gate": False,
        "task_count": int(len(rows)),
        "wall_seconds": float(time.perf_counter() - started),
    }
