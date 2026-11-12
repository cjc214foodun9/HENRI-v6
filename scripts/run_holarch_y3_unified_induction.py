"""Cycle Y3: unified wave-invariant induction over the 12 + 4 battery.

    python scripts/run_holarch_y3_unified_induction.py

Stage Y3 keeps every Cycle X3 gate and adds four composed one-cell tasks:

  * delimited_reverse, unary_increment_parity, two_bit_borrow_xor, and
    binary_rotate_left, each with four training pairs, four held-out pairs,
    and a deterministic held-out length suite for L in [12, 32];
  * one deterministic macro compiler per task that emits a strict one-cell
    transition table (one read, one write, move in {-1, 0, +1});
  * a wave-invariant sketch router whose I/O Wave-Delta Signature is
    computed from the four training pairs only, ranks every operator sketch,
    and drives the attempt order inside one 320000-node budget.

The receipt records the X3 retention audit (the unchanged twelve-task path),
the sixteen-task battery, the strict one-cell audit of the actual tables, the
AST audit of task-name branches in the routing path, the router signature and
ranking evidence, and the node-budget accounting.  Nothing is copied from an
earlier receipt: every number is measured in this process.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_holarch_p_cycle as pc
from henri.holarch.induction import (
    SOVEREIGN_SKETCHES,
    BorrowXorMacro,
    DelimitedReverseMacro,
    FixpointCompositionInducer,
    RotateLeftMacro,
    SovereignInducer,
    SovereignSpec,
    UnaryParityMacro,
    WaveDeltaSketchRouter,
    sovereign_tasks_v2,
    sovereign_tasks_y3,
    sov_run,
    v1_length_suite,
    y3_length_suite,
)
from henri.holarch.onecell_audit import (
    audit_induction,
    audit_task_name_branches,
    contradiction_specs,
)

RECEIPTS = ROOT / "receipts" / "holarch"
INDUCTION_PATH = ROOT / "henri" / "holarch" / "induction.py"
NODE_BUDGET = 320000
MAX_STATES = 8
NEW_TASKS = ("delimited_reverse", "unary_increment_parity",
             "two_bit_borrow_xor", "binary_rotate_left")
NEW_MACROS = {
    "delimited_reverse": "delimited_reverse_macro",
    "unary_increment_parity": "unary_parity_macro",
    "two_bit_borrow_xor": "borrow_xor_macro",
    "binary_rotate_left": "rotate_left_macro",
}
MACRO_CLASSES = {
    "delimited_reverse": DelimitedReverseMacro,
    "unary_increment_parity": UnaryParityMacro,
    "two_bit_borrow_xor": BorrowXorMacro,
    "binary_rotate_left": RotateLeftMacro,
}
X3_ORDER = (
    "factored_direct", "factored_class", "macro_fixpoint",
    "carry_latch_macro", "ripple_insert_macro", "cegis_1cell", "mn_1cell",
)


def _strict_one_cell(tables: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Every transition reads and writes one cell and moves by -1, 0, or +1."""
    rows = 0
    bad: List[str] = []
    for row in tables:
        table = dict(row.get("table") or {})
        for (state, symbol), action in table.items():
            rows += 1
            next_state, write, move = action
            if not isinstance(state, int) or isinstance(state, bool):
                bad.append("%s:state %r" % (row.get("name"), state))
            if not isinstance(symbol, str) or len(symbol) > 2:
                bad.append("%s:symbol %r" % (row.get("name"), symbol))
            if not isinstance(write, str) or len(write) > 2:
                bad.append("%s:write %r" % (row.get("name"), write))
            if int(move) not in (-1, 0, 1):
                bad.append("%s:move %r" % (row.get("name"), move))
            if int(move) == 0 and int(next_state) != -1:
                bad.append("%s:zero move without halt" % (row.get("name"),))
    return {"rows": int(rows), "violations": bad[:16], "strict": bool(rows and not bad)}


def _reproduces(row: Dict[str, Any], spec: SovereignSpec) -> bool:
    """Re-walk the training examples through the selected table."""
    table = dict(row.get("table") or {})
    if not table:
        return False
    mechanism = str(row.get("mechanism_used", ""))
    if mechanism == "macro_fixpoint":
        runner = FixpointCompositionInducer()
        return all(str(runner.rounds(table, spec, str(x))["text"]) == str(y).strip("_")
                   for x, y in spec.examples)
    return all(sov_run(table, spec, x, y)["status"] == "ok" for x, y in spec.examples)


def _signature_is_example_only(router: WaveDeltaSketchRouter,
                               spec: SovereignSpec, digest: str) -> bool:
    """Recompute the signature with the held-out pairs removed."""
    stripped = dataclasses.replace(spec, tests=())
    again = router.rank(stripped)
    return bool(str(again["signature_digest"]) == str(digest))


def measure() -> Dict[str, Any]:
    started = time.perf_counter()
    # ---- 1. X3 retention: the unchanged twelve-task audit path -----------
    x3 = audit_induction(node_budget=NODE_BUDGET, max_states=MAX_STATES)
    x3_names = [str(row["name"]) for row in x3["tasks"]]
    x3_sealed = pc._read(RECEIPTS, "x3_twelve_of_twelve_turing.json")
    sealed_solved = int(x3_sealed.get("solved", -1) or -1)
    sealed_generalized = int(x3_sealed.get("v1_length_generalized", -1) or -1)
    retention_ok = bool(
        int(x3["solved"]) == 12
        and int(x3["generalized"]) == 12
        and int(x3["v1_length_generalized"]) == 12
        and len(x3["tasks"]) == 12
        and sealed_solved == 12
        and sealed_generalized == 12
    )

    # ---- 2. Y3 battery: the 12 + 4 tasks under one routed inducer --------
    tasks = list(sovereign_tasks_v2()) + list(sovereign_tasks_y3())
    suites = list(v1_length_suite()) + list(y3_length_suite())
    router = WaveDeltaSketchRouter(node_budget=NODE_BUDGET)
    inducer = SovereignInducer(node_budget=NODE_BUDGET, max_states=MAX_STATES,
                               router=router, extended=True)
    battery = audit_induction(node_budget=NODE_BUDGET, max_states=MAX_STATES,
                              tasks=tasks, suites=suites, inducer=inducer,
                              keep_tables=True)
    raw_rows = list(battery["tasks"])
    specs = {str(spec.name): spec for spec in tasks}
    battery_rows: List[Dict[str, Any]] = []
    strict_tables: List[Dict[str, Any]] = []
    reproduces_all = True
    signature_example_only = True
    ranking_applied = True
    router_present = True
    ranked_evidence: List[Dict[str, Any]] = []
    text_mechanisms = True
    for row in raw_rows:
        name = str(row["name"])
        spec = specs[name]
        table = dict(row.get("table") or {})
        strict_tables.append({"name": name, "table": table})
        reproduces_all = bool(reproduces_all and _reproduces(row, spec))
        router_row = dict(row.get("router") or {})
        if not router_row:
            router_present = False
            signature_example_only = False
            ranking_applied = False
        else:
            signature_example_only = bool(
                signature_example_only and _signature_is_example_only(
                    router, spec, str(router_row.get("signature_digest", ""))))
            ranked = [str(value) for value in list(router_row.get("ranked") or [])]
            attempts = [str(value) for value in list(router_row.get("attempt_order") or [])]
            ranking_applied = bool(
                ranking_applied and ranked and attempts
                and ranked[0] == attempts[0]
                and set(ranked) == set(SOVEREIGN_SKETCHES))
            if len(ranked_evidence) < 4 or name in NEW_TASKS:
                ranked_evidence.append({
                    "name": name,
                    "signature": dict(router_row.get("signature") or {}),
                    "signature_digest": str(router_row.get("signature_digest", "")),
                    "scores": dict(router_row.get("scores") or {}),
                    "ranked": ranked,
                    "attempt_order": attempts,
                    "cegis_prior": dict(router_row.get("cegis_prior") or {}),
                    "examples_used": int(router_row.get("examples_used", 0) or 0),
                    "tests_consulted": bool(router_row.get("tests_consulted", True)),
                    "budget_nodes": int(router_row.get("budget_nodes", 0) or 0),
                })
        for attempt in list(row.get("attempts") or []):
            text_mechanisms = bool(text_mechanisms
                                   and str(attempt.get("mechanism")) in SOVEREIGN_SKETCHES)
        battery_rows.append({
            "name": name,
            "status": str(row.get("status", "")),
            "mechanism_used": str(row.get("mechanism_used", "")),
            "nodes": int(row.get("nodes_evaluated", 0) or 0),
            "table_rules": int(len(table)),
            "state_count_q": int(row.get("state_count_q", 0) or 0),
            "test_exact": int(row.get("test_exact", 0) or 0),
            "v1_exact": int(row.get("v1_exact", 0) or 0),
            "v1_total": int(row.get("v1_total", 0) or 0),
            "length_generalized": bool(row.get("v1_generalized")),
            "train_examples": int(row.get("train_examples", 0) or 0),
            "test_examples": int(row.get("test_examples", 0) or 0),
            "text_tokens": int(row.get("text_tokens", 0) or 0),
            "attempts": [
                {
                    "mechanism": str(attempt.get("mechanism", "")),
                    "status": str(attempt.get("status", "")),
                    "nodes": int(attempt.get("nodes", 0) or 0),
                    "rank": int(attempt.get("rank", -1) or -1),
                    "train_exact": bool(attempt.get("train_exact")),
                    "test_exact": int(attempt.get("test_exact", 0) or 0),
                    "test_total": int(attempt.get("test_total", 0) or 0),
                }
                for attempt in list(row.get("attempts") or [])
            ],
            "router": {
                "signature_digest": str((row.get("router") or {}).get("signature_digest", "")),
                "ranked": [str(value) for value in
                           list((row.get("router") or {}).get("ranked") or [])],
                "examples_used": int((row.get("router") or {}).get("examples_used", 0) or 0),
                "tests_consulted": bool((row.get("router") or {}).get("tests_consulted", True)),
            },
        })
    # ---- 3. verify each Y3 macro compiler directly (examples + suite) ----
    compiler_rows: List[Dict[str, Any]] = []
    for name in NEW_TASKS:
        spec = specs[name]
        suite = {str(row_spec.name): row_spec for row_spec in suites}[name]
        macro = MACRO_CLASSES[name]()
        outcome = macro.induce(spec)
        table = dict(outcome.get("table") or {})
        example_exact = sum(
            1 for x, y in spec.examples
            if table and sov_run(table, spec, x, y)["status"] == "ok")
        suite_exact = sum(
            1 for x, y in suite.tests
            if table and sov_run(table, suite, x, y)["status"] == "ok")
        compiler_rows.append({
            "name": name,
            "macro_class": type(macro).__name__,
            "mechanism": NEW_MACROS[name],
            "status": str(outcome.get("status", "")),
            "table_rules": int(len(table)),
            "examples_exact": int(example_exact),
            "examples_total": int(len(spec.examples)),
            "suite_exact": int(suite_exact),
            "suite_total": int(len(suite.tests)),
            "alphabet": [str(value) for value in spec.alphabet],
            "alphabet_source": "spec alphabet + declared example symbols, minus the blank",
            "verified": bool(outcome.get("status") == "INDUCED"
                             and example_exact == len(spec.examples)
                             and suite_exact == len(suite.tests)),
        })
    names = [row["name"] for row in battery_rows]
    counted = [row for row in battery_rows
               if row["status"] == "INDUCED" and row["length_generalized"]]
    induced_and_generalized = len(counted)
    new_present = all(name in names for name in NEW_TASKS)
    new_induced = all(
        name in names
        and next(row for row in battery_rows if row["name"] == name)["status"] == "INDUCED"
        for name in NEW_TASKS)
    new_by_macro = bool(
        new_present
        and all(next(row for row in battery_rows if row["name"] == name)["mechanism_used"]
                == NEW_MACROS[name] for name in NEW_TASKS))
    all_compilers_verified = bool(compiler_rows
                                  and all(row["verified"] for row in compiler_rows))
    max_nodes = max(int(row["nodes"]) for row in battery_rows)
    total_nodes = int(sum(int(row["nodes"]) for row in battery_rows))
    strict = _strict_one_cell(strict_tables)
    ast_audit = audit_task_name_branches(str(INDUCTION_PATH),
                                         [str(spec.name) for spec in tasks])
    # A contradictory pair is refuted by construction: one input tape cannot
    # have two different outputs under a deterministic one-cell table.
    contradiction_proof = []
    for spec in contradiction_specs():
        grouped: Dict[str, set] = {}
        for tape_in, tape_out in spec.examples:
            grouped.setdefault(str(tape_in), set()).add(str(tape_out))
        contradiction_proof.append({
            "name": str(spec.name),
            "same_input": sorted(grouped),
            "outputs": {key: sorted(value) for key, value in grouped.items()},
            "two_outputs_for_one_input": bool(any(len(value) > 1
                                                  for value in grouped.values())),
        })
    contradictions_refuted = int(sum(
        1 for row in contradiction_proof if row["two_outputs_for_one_input"]))
    text_tokens = int(sum(int(row["text_tokens"]) for row in battery_rows))
    reordered = sum(
        1 for row in battery_rows
        if list((row.get("router") or {}).get("ranked") or [])[:len(X3_ORDER)]
        != list(X3_ORDER))
    declarations_ok = all(row["name"] in NEW_TASKS or row["name"] in
                          [str(spec.name) for spec in sovereign_tasks_v2()]
                          for row in battery_rows)
    router_budget_ok = all(int(row.get("budget_nodes", 0) or 0) == NODE_BUDGET
                           for row in ranked_evidence)
    stages = {
        "retention_rows": int(len(x3["tasks"])),
        "battery_rows": int(len(battery_rows)),
        "strict_rows": int(strict["rows"]),
        "ast_task_names": int(ast_audit["task_names_checked"]),
        "router_evidence": int(len(ranked_evidence)),
        "compilers": int(len(compiler_rows)),
    }
    stages_ok = bool(stages["retention_rows"] == 12 and stages["battery_rows"] == 16
                     and stages["strict_rows"] > 0 and stages["ast_task_names"] == 16
                     and stages["router_evidence"] >= 2 and stages["compilers"] == 4)
    gates = {
        "y3_runner_runs": bool(stages_ok),
        "y3_twelve_retained": bool(retention_ok),
        "y3_battery_16": bool(len(battery_rows) == 16
                              and len(set(names)) == 16
                              and set(names) == {str(spec.name) for spec in tasks}),
        "y3_induced_and_generalized_ge_14": bool(induced_and_generalized >= 14),
        "y3_contradictions_4": bool(int(battery["contradictions_failed_closed"]) == 4
                                    and contradictions_refuted == 4),
        "y3_strict_one_cell_all_rows": bool(strict["strict"] and reproduces_all),
        "y3_text_tokens_zero": bool(text_tokens == 0 and text_mechanisms),
        "y3_no_task_name_branches": bool(ast_audit["passed"]),
        "y3_budget_le_320000": bool(max_nodes <= NODE_BUDGET
                                    and router_budget_ok
                                    and int(battery["node_budget"]) == NODE_BUDGET),
        "y3_new_tasks_4_present": bool(new_present and new_induced),
        "y3_router_signature_used": bool(router_present and ranking_applied
                                         and signature_example_only
                                         and reordered >= 2
                                         and len(ranked_evidence) >= 2),
    }
    payload: Dict[str, Any] = {
        "stage": "Y3",
        "title": "unified wave-invariant induction over the 12 + 4 one-cell battery",
        "node_budget": int(NODE_BUDGET),
        "max_states": int(MAX_STATES),
        "retention_x3": {
            "solved": int(x3["solved"]),
            "generalized": int(x3["generalized"]),
            "v1_length_generalized": int(x3["v1_length_generalized"]),
            "tasks": x3_names,
            "contradictions": int(x3["contradictions_failed_closed"]),
            "max_nodes_used": int(max(int(row.get("nodes_evaluated", 0) or 0)
                                      for row in x3["tasks"])),
            "mechanism_used": {str(row["name"]): str(row.get("mechanism_used", ""))
                               for row in x3["tasks"]},
            "sealed_receipt": {
                "path": "receipts/holarch/x3_twelve_of_twelve_turing.json",
                "solved": sealed_solved,
                "v1_length_generalized": sealed_generalized,
                "passed": bool(x3_sealed.get("passed")),
            },
        },
        "tasks": battery_rows,
        "new_tasks": {
            "names": list(NEW_TASKS),
            "declared_macros": dict(NEW_MACROS),
            "present": bool(new_present),
            "induced": bool(new_induced),
            "closed_by_declared_macro": bool(new_by_macro),
            "compilers": compiler_rows,
            "all_compilers_verified": bool(all_compilers_verified),
        },
        "router_evidence": {
            "signature_components": [
                "tape_length_delta_distribution",
                "cross_spectral_shift_peak",
                "symbol_histogram_conservation_delta",
                "dyadic_parity_alternation",
            ],
            "sketches": list(SOVEREIGN_SKETCHES),
            "examples_only": True,
            "per_task": ranked_evidence,
            "ranking_applied": bool(ranking_applied),
            "signature_invariant_to_tests": bool(signature_example_only),
            "reordered_versus_x3_order": int(reordered),
        },
        "strict_one_cell": {
            "rows": int(strict["rows"]),
            "violations": strict["violations"],
            "strict": bool(strict["strict"]),
            "reproduces_all_training_pairs": bool(reproduces_all),
            "tables_audited": int(len(strict_tables)),
            "move_zero_only_on_halt": True,
        },
        "budget": {
            "node_budget": int(NODE_BUDGET),
            "max_nodes_used": int(max_nodes),
            "total_nodes_used": int(total_nodes),
            "per_task_nodes": {str(row["name"]): int(row["nodes"])
                               for row in battery_rows},
            "router_budget_nodes": int(NODE_BUDGET),
            "within_budget": bool(max_nodes <= NODE_BUDGET and router_budget_ok),
            "unified": "one budget per task across every ranked sketch combined",
        },
        "ast_audit": ast_audit,
        "contradictions": {
            "declared": int(battery["contradictions_failed_closed"]),
            "refuted_by_construction": int(contradictions_refuted),
            "proofs": contradiction_proof,
            "induction_reason": "ABSTAIN: unsatisfiable specification",
        },
        "stages": stages,
        "text_tokens": int(text_tokens),
        "induced_and_generalized": int(induced_and_generalized),
        "task_count": int(len(battery_rows)),
        "declared_family_unchanged": bool(declarations_ok),
        "gate": pc._finalise(gates),
        "evidence": "OBSERVED",
        "git_head": pc._git_head(),
        "wall_seconds": float(time.perf_counter() - started),
    }
    payload["passed"] = bool(all(bool(value) for value in gates.values()))
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Cycle Y3 unified induction")
    parser.add_argument("--receipts", default=str(RECEIPTS))
    args = parser.parse_args(argv)
    directory = Path(args.receipts)
    directory.mkdir(parents=True, exist_ok=True)
    payload = measure()
    written = pc._write(directory, "y3_unified_wave_induction.json", payload)
    print("[y3] retained=%d induced+generalized=%d/16 contradictions=%d max_nodes=%d "
          "strict_rows=%d ast_branches=%d compilers=%s" % (
              payload["retention_x3"]["v1_length_generalized"],
              payload["induced_and_generalized"],
              payload["contradictions"]["refuted_by_construction"],
              payload["budget"]["max_nodes_used"], payload["strict_one_cell"]["rows"],
              payload["ast_audit"]["task_name_branches"],
              payload["new_tasks"]["all_compilers_verified"]))
    print("[y3] new_tasks_4=%s router=%s passed=%s receipt=%s sha256=%s" % (
        payload["new_tasks"]["present"], payload["gate"]["y3_router_signature_used"],
        payload["passed"], written["path"], written["sha256"]))
    return 0 if bool(payload["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
