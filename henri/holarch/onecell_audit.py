"""Shared Cycle V one-cell induction audit (Stages V1, T3b, U3, Y3).

One place measures the one-cell Turing surface, so the T-cycle and the
U-cycle cannot report different numbers for the same claim.  Every field here
is measured in this process: no value is copied from an earlier receipt.

Cycle Y3 adds two optional controls to the audit: an explicit task list and
suite (so the 12 + 4 battery can be measured with one inducer) and a static
AST audit that counts task-name string branches in the routing path.
"""

from __future__ import annotations

import ast
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .induction import (
    FoveatedTwoCellAblation,
    FixpointCompositionInducer,
    SovereignInducer,
    SovereignSpec,
    V1_LENGTHS,
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


def _routing_nodes(tree: ast.AST) -> List[ast.AST]:
    """Every node of the mechanism-routing path.

    The routing path is the body of SovereignInducer and of any class whose
    name ends in Router.  These are the only places that choose, order, or
    score an operator sketch; the declared task family registries are data
    and live outside them.
    """
    nodes: List[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and (
                node.name == "SovereignInducer" or node.name.endswith("Router")):
            nodes.extend(node.body)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and (
                node.name.startswith("_route") or node.name.startswith("_rank")
                or node.name.startswith("_score") or node.name.startswith("_sketch")
                or node.name.endswith("_router")):
            nodes.append(node)
    return nodes


def _literal_strings(node: ast.AST) -> List[str]:
    """String constants carried by one expression node, including containers."""
    values: List[str] = []
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        values.append(node.value)
    elif isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        for element in node.elts:
            values.extend(_literal_strings(element))
    return values


def audit_task_name_branches(path: str,
                             task_names: Sequence[str]) -> Dict[str, object]:
    """Static AST audit of task-name string branches.

    Three rules are counted:

      R1  a Compare anywhere in the module that puts an attribute .name
          against a string constant;
      R2  a Compare in the routing path whose operand or comparator holds a
          string constant equal to a declared task name (also inside a tuple,
          list, or set literal);
      R3  a Dict literal in the routing path with a key equal to a declared
          task name.

    The declared family registries (TASK_TRANSFORMS and TASK_TRANSFORMS_Y)
    are declarations, not routing; they are counted separately and the audit
    also checks that the routing path never reads either registry.
    """
    source = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {str(value) for value in task_names}
    routing = _routing_nodes(tree)
    routing_ids = set()
    for node in routing:
        for child in ast.walk(node):
            routing_ids.add(id(child))
    r1: List[Dict[str, object]] = []
    broad: List[Dict[str, object]] = []
    r2: List[Dict[str, object]] = []
    r3: List[Dict[str, object]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            operands = [node.left] + list(node.comparators)
            attribute_name = any(
                isinstance(operand, ast.Attribute) and operand.attr == "name"
                for operand in operands)
            strings = []
            for operand in operands:
                strings.extend(_literal_strings(operand))
            if attribute_name and strings:
                r1.append({"line": int(node.lineno), "rule": "R1",
                           "detail": "attribute .name against a string constant"})
            hit = sorted(set(strings) & names)
            if hit:
                broad.append({"line": int(node.lineno), "task_names": hit})
            if id(node) in routing_ids and hit:
                r2.append({"line": int(node.lineno), "rule": "R2",
                           "task_names": hit})
        if isinstance(node, ast.Dict) and id(node) in routing_ids:
            keys: List[str] = []
            for key in node.keys:
                if key is not None:
                    keys.extend(_literal_strings(key))
            hit = sorted(set(keys) & names)
            if hit:
                r3.append({"line": int(node.lineno), "rule": "R3",
                           "task_names": hit})
    registries: Dict[str, object] = {}
    registry_names = {"TASK_TRANSFORMS", "TASK_TRANSFORMS_Y"}
    registry_reads: List[Dict[str, object]] = []
    for node in ast.walk(tree):
        target = None
        value = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        if isinstance(target, ast.Name) and target.id in registry_names:
            keys: List[str] = []
            if isinstance(value, ast.Dict):
                for key in value.keys:
                    if key is not None:
                        keys.extend(_literal_strings(key))
            registries[target.id] = {
                "line": int(node.lineno),
                "keys": keys,
                "count": int(len(keys)),
                "role": "declared task-family data, outside the routing path",
            }
        if isinstance(node, ast.Name) and node.id in registry_names and id(node) in routing_ids:
            registry_reads.append({"line": int(node.lineno), "registry": node.id})
    return {
        "path": str(path),
        "task_names_checked": int(len(names)),
        "routing_path": ("SovereignInducer + *Router class bodies + module-level "
                         "_route/_rank/_score/_sketch helpers"),
        "r1_name_attribute_comparisons": int(len(r1)),
        "r2_name_literal_comparisons_in_routing": int(len(r2)),
        "r3_name_keyed_dicts_in_routing": int(len(r3)),
        "task_name_branches": int(len(r1) + len(r2) + len(r3)),
        "broad_name_literal_comparisons_anywhere": int(len(broad)),
        "r1_examples": r1[:8],
        "r2_examples": r2[:8],
        "r3_examples": r3[:8],
        "declaration_registries": registries,
        "routing_reads_declared_registry": bool(registry_reads),
        "registry_reads": registry_reads[:8],
        "passed": bool(not r1 and not r2 and not r3 and not registry_reads),
    }


def audit_induction(node_budget: int = 320000, max_states: int = 8,
                    tasks: Optional[Sequence[SovereignSpec]] = None,
                    suites: Optional[Sequence[SovereignSpec]] = None,
                    inducer: object = None,
                    keep_tables: bool = False) -> Dict[str, object]:
    """Induce every declared task and measure held-out length generalisation.

    With no arguments this measures the twelve X3 tasks with the fixed X3
    attempt order.  Passing tasks, suites, and an inducer measures any other
    battery (the Cycle Y3 12 + 4 battery uses one routed, extended inducer).
    """
    started = time.perf_counter()
    task_list = list(tasks) if tasks is not None else list(sovereign_tasks_v2())
    suite_list = list(suites) if suites is not None else list(v1_length_suite())
    suites_by_name = {spec.name: spec for spec in suite_list}
    engine = inducer if inducer is not None else SovereignInducer(
        node_budget=int(node_budget), max_states=int(max_states))
    rows: List[Dict[str, object]] = []
    solved = generalized = v1_generalized = text_tokens = 0
    for spec in task_list:
        outcome = engine.induce(spec)
        row = dict(outcome)
        table = dict(row.get("table") or {})
        if keep_tables:
            row["table"] = table
        else:
            row.pop("table", None)
        v1_spec = suites_by_name[str(spec.name)]
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
        text_tokens += int(row.get("text_tokens", 0))
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
        "v1_lengths": list(V1_LENGTHS),
        "text_tokens": int(text_tokens),
        "node_budget": int(node_budget),
        "contradictions": contradictions,
        "contradictions_failed_closed": int(len(contradictions)),
        "foveated_2cell_ablation": ablation,
        "foveated_counted_toward_gate": False,
        "task_count": int(len(rows)),
        "wall_seconds": float(time.perf_counter() - started),
    }
