"""Cycle Z2: an author-decoupled external cold-start synthesis audit.

    python scripts/run_holarch_z2_external_synth.py

Two pre-registered external slices are read from the vendored benchmark files
(no filtering of failures, no task-name rules anywhere in the synthesizer):

    PRIMARY   MBPP-sanitized: ascending task_id, >= 4 canonical test entries,
              first 32.  The prompt is the signature of the canonical code
              plus the unmodified task text as its docstring plus the first
              two canonical test assertions as PUBLIC examples.  HIDDEN is
              the remaining canonical test assertions.
    SECONDARY HumanEval: ascending task_id, >= 4 assert lines in the canonical
              check function, first 32.  The prompt is the unmodified
              canonical prompt, whose doctests are the PUBLIC examples.
              HIDDEN is the canonical check assertions with the candidate
              renamed to the entry point.

Arm A is the strict zero-LLM cold start: an EMPTY PlanStore, at most 32
sandboxed rollouts, verification against the public examples only, and the
RepoSheaf capability pullback.  Arm B compiles every Arm-A verified solution
into the tenant PlanStore as a Tier-3 ACE playbook and is measured on a
held-out family of 12 compound (two-plan composition) and mutated (argument
order) tasks plus 6 adversarial type/arity traps.

Every number is measured in this process.  The script exits 0 only when every
gate is true; a missed gate still writes the receipt with that gate false and
exits 1.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

import run_holarch_p_cycle as pc
from henri.holarch.code_gen import SharedCorpusStem, SynthesisEngine
from henri.holarch.cold_synth import (
    ColdSynthesizer,
    _plain_language,
    _unsatisfiable,
    audit_task_specific_branches,
    parse_problem,
)
from henri.holarch.config import HolarchConfig
from henri.holarch.model import find_checkpoint, open_clean_room_memory
from henri.holarch.sandbox import SubprocessSandboxExecutor

RECEIPTS = ROOT / "receipts" / "holarch"
EXTERNAL = RECEIPTS / "external"
COLD_SYNTH = ROOT / "henri" / "holarch" / "cold_synth.py"
BUDGET = 32
SLICE_SIZE = 32
PLAN_NAMESPACE = "z2_arm_b"
ARM_B_CANDIDATE_CAP = 40

PROVENANCE_FILES: Tuple[Tuple[str, int, str, str], ...] = (
    (
        "HumanEval.jsonl",
        214438,
        "1d49078ba3e2b196b9344535bef34a43021f038fad9561d6ee7c53450609a6a2",
        "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz (gunzipped in place)",
    ),
    (
        "HumanEval.jsonl.gz",
        44877,
        "b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef",
        "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz",
    ),
    (
        "mbpp_sanitized.json",
        255053,
        "ca95deaa9a01ef0a6f439f88bcf0dd3db3563d22f22aad6cae04ebb9a8d8c8e9",
        "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/sanitized-mbpp.json",
    ),
)

RULE_PRIMARY = (
    "MBPP-sanitized rows ordered by ascending integer task_id; a row is "
    "eligible when the canonical test_list holds >= 4 entries; the first 32 "
    "eligible rows are taken. The prompt is the signature of the canonical "
    "code definition the canonical tests call (the first def when the tests "
    "name none), the unmodified task text as its docstring, and the first two "
    "canonical test assertions verbatim as PUBLIC examples. The task text and "
    "the assertions are never edited. HIDDEN is the remaining canonical test "
    "assertions from index 2."
)
RULE_SECONDARY = (
    "HumanEval rows ordered by ascending integer task_id; a row is eligible "
    "when check(candidate) carries >= 4 assertion statements as direct "
    "children of its body; the first 32 eligible rows are taken. The prompt "
    "is the unmodified canonical prompt, so its doctests are the PUBLIC "
    "examples. HIDDEN is each canonical assertion statement with the "
    "identifier candidate renamed to the canonical entry_point; an assertion "
    "nested inside a loop or a conditional is not a standalone line and is "
    "excluded from HIDDEN by that rule. No other filtering is applied."
)
DISCLOSURE = (
    "HumanEval was measured BEFORE MBPP was selected. The primary suite "
    "choice is therefore announced after a HumanEval probe, and both numbers "
    "are reported: the primary MBPP figure and the secondary HumanEval figure "
    "sit next to each other in this receipt. No failing task is dropped from "
    "either number, and no task-specific rule exists in the synthesizer."
)

TRAPS: List[Dict[str, Any]] = [
    {
        "name": "nonexistent_api_import",
        "kind": "missing_capability",
        "prompt": (
            "import nonexistent_quantum_api\n\n\n"
            "def decode_value(x: int) -> int:\n"
            '    """Return the value of x decoded by the quantum codec.\n\n'
            "    assert decode_value(1) == 2\n"
            '    """\n'
        ),
    },
    {
        "name": "blocked_module_import",
        "kind": "blocked_capability",
        "prompt": (
            "import socket\n\n\n"
            "def connect_probe(host: str) -> int:\n"
            '    """Return the port the socket layer reports for host.\n\n'
            '    assert connect_probe("localhost") == 0\n'
            '    """\n'
        ),
    },
    {
        "name": "tuple_return_for_int_annotation",
        "kind": "type_contradiction",
        "prompt": (
            "def min_max(nums: list) -> int:\n"
            '    """Return both the smallest and the largest value of nums as one int.\n\n'
            "    assert min_max([1, 2]) == (1, 2)\n"
            '    """\n'
        ),
    },
    {
        "name": "contradictory_examples",
        "kind": "contradiction",
        "prompt": (
            "def contradictory_count(x: int) -> int:\n"
            '    """Return x plus one.\n\n'
            "    assert contradictory_count(2) == 3\n"
            "    assert contradictory_count(2) == 4\n"
            '    """\n'
        ),
    },
    {
        "name": "wrong_arity",
        "kind": "arity_contradiction",
        "prompt": (
            "def pair_sum(a: int, b: int) -> int:\n"
            '    """Return the sum of a and b.\n\n'
            "    assert pair_sum(1, 2) == 3\n"
            "    assert pair_sum(1, 2, 3) == 6\n"
            '    """\n'
        ),
    },
    {
        "name": "unsatisfiable_capability",
        "kind": "unsatisfiable_capability",
        "prompt": (
            "def latest_price(symbol: str) -> float:\n"
            '    """Download the latest stock price from the internet for symbol.\n\n'
            '    assert latest_price("AAA") == 1.0\n'
            '    """\n'
        ),
    },
]


# ------------------------------------------------------------------ helpers
def _sha256_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted(str(item) for item in value)
    return None


def _slice_digest(tasks: Sequence[Dict[str, Any]]) -> str:
    payload = [
        {"task_id": str(task["task_id"]), "hidden": list(task["hidden"])}
        for task in tasks
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _called_names(lines: Sequence[str]) -> set:
    names: set = set()
    for line in lines:
        try:
            tree = ast.parse(str(line).strip())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                names.add(str(node.func.id))
    return names


def _syntax_ok(source: str) -> bool:
    try:
        tree = ast.parse(str(source))
        compile(tree, "<z2-emitted>", "exec")
    except Exception:  # noqa: BLE001 - a broken emission is the measurement
        return False
    return True


def _hidden_check(
    sandbox: SubprocessSandboxExecutor, source: str, tests: Sequence[str]
) -> Dict[str, Any]:
    program = (
        str(source)
        + "\n"
        + "\n".join(str(line) for line in tests)
        + "\nresult = {'ok': True}\n"
    )
    record = sandbox.execute(program)
    detail = (record.stderr or "").strip().splitlines()
    return {
        "ok": bool(record.ok),
        "detail": (detail[-1] if detail else ("pass" if record.ok else "failed"))[:220],
    }


# ------------------------------------------------------------------- slices
def _mbpp_definition(code: str, tests: Sequence[str]):
    tree = ast.parse(str(code))
    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if not definitions:
        raise ValueError("canonical code declares no function")
    called = _called_names(tests)
    for definition in definitions:
        if str(definition.name) in called:
            return definition
    return definitions[0]


def load_primary() -> Dict[str, Any]:
    path = EXTERNAL / "mbpp_sanitized.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    rows = sorted(rows, key=lambda row: int(row["task_id"]))
    eligible = [row for row in rows if len(row["test_list"]) >= 4]
    selected = eligible[:SLICE_SIZE]
    tasks: List[Dict[str, Any]] = []
    for row in selected:
        tests = [str(line).strip() for line in row["test_list"]]
        definition = _mbpp_definition(str(row["code"]), tests)
        returns = ast.unparse(definition.returns) if definition.returns else ""
        signature = "def %s(%s)%s:" % (
            definition.name,
            ast.unparse(definition.args),
            (" -> " + returns) if returns else "",
        )
        public = tests[:2]
        hidden = tests[2:]
        prompt = (
            "\n".join(
                [signature, '    """' + str(row["prompt"]).strip() + '"""']
                + ["    " + line for line in public]
            )
            + "\n"
        )
        tasks.append(
            {
                "suite": "primary",
                "task_id": str(row["task_id"]),
                "entry": str(definition.name),
                "prompt": prompt,
                "public": public,
                "hidden": hidden,
            }
        )
    return {
        "name": "mbpp_sanitized",
        "path": str(path.relative_to(ROOT)),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
        "rows": int(len(rows)),
        "eligible": int(len(eligible)),
        "slice_size": int(SLICE_SIZE),
        "selected": int(len(tasks)),
        "rule": RULE_PRIMARY,
        "digest": _slice_digest(tasks),
        "task_ids": [task["task_id"] for task in tasks],
        "tasks": tasks,
    }


def _humaneval_asserts(row: Dict[str, Any]) -> List[str]:
    source = str(row["test"])
    tree = ast.parse(source)
    function = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and str(node.name) == "check"
        ),
        None,
    )
    if function is None:
        return []
    out: List[str] = []
    for statement in function.body:
        if not isinstance(statement, ast.Assert):
            continue
        segment = ast.get_source_segment(source, statement) or ast.unparse(statement)
        out.append(
            re.sub(r"\bcandidate\b", str(row["entry_point"]), str(segment).strip())
        )
    return out


def load_secondary() -> Dict[str, Any]:
    path = EXTERNAL / "HumanEval.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = sorted(rows, key=lambda row: int(str(row["task_id"]).split("/")[-1]))
    scored = [(row, _humaneval_asserts(row)) for row in rows]
    eligible = [(row, hidden) for row, hidden in scored if len(hidden) >= 4]
    selected = eligible[:SLICE_SIZE]
    tasks: List[Dict[str, Any]] = []
    for row, hidden in selected:
        tasks.append(
            {
                "suite": "secondary",
                "task_id": str(row["task_id"]),
                "entry": str(row["entry_point"]),
                "prompt": str(row["prompt"]),
                "public": [],
                "hidden": list(hidden),
            }
        )
    return {
        "name": "humaneval",
        "path": str(path.relative_to(ROOT)),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
        "rows": int(len(rows)),
        "eligible": int(len(eligible)),
        "slice_size": int(SLICE_SIZE),
        "selected": int(len(tasks)),
        "rule": RULE_SECONDARY,
        "digest": _slice_digest(tasks),
        "task_ids": [task["task_id"] for task in tasks],
        "tasks": tasks,
    }


def provenance_block() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for name, expected_bytes, expected_sha, upstream in PROVENANCE_FILES:
        path = EXTERNAL / name
        exists = bool(path.exists())
        actual_bytes = int(path.stat().st_size) if exists else 0
        actual_sha = _sha256_file(path) if exists else ""
        out.append(
            {
                "file": str(path.relative_to(ROOT)),
                "upstream": upstream,
                "expected_bytes": int(expected_bytes),
                "bytes": int(actual_bytes),
                "expected_sha256": str(expected_sha),
                "sha256": str(actual_sha),
                "matches": bool(
                    exists
                    and actual_bytes == int(expected_bytes)
                    and actual_sha == str(expected_sha)
                ),
            }
        )
    return out


# ------------------------------------------------------------------- arm A
def run_arm_a(
    sandbox: SubprocessSandboxExecutor,
    suite: Dict[str, Any],
    *,
    budget: int = BUDGET,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    synthesizer = ColdSynthesizer(sandbox=sandbox, budget=budget, plan_store=None)
    rows: List[Dict[str, Any]] = []
    for task in suite["tasks"]:
        started = time.perf_counter()
        spec = parse_problem(task["prompt"])
        outcome = synthesizer.synthesize(task["prompt"])
        row: Dict[str, Any] = {
            "suite": str(task["suite"]),
            "task_id": str(task["task_id"]),
            "entry": str(task["entry"]),
            "status": str(outcome["status"]),
            "reason": str(outcome["reason"]),
            "rollouts": int(outcome["rollouts"]),
            "return_kind": str(spec.return_kind),
            "param_kinds": [str(value) for value in spec.arg_kinds.values()],
            "params": [str(param["name"]) for param in spec.params],
            "intent": _plain_language(spec.docstring) or str(spec.name),
            "emitted": bool(outcome["status"] == "SOLVED"),
            "hidden_ok": False,
            "hidden_detail": "not_emitted",
            "syntax_ok": True,
            "source": str(outcome["source"]),
            "source_sha256": (
                _sha256_text(str(outcome["source"])) if outcome["source"] else ""
            ),
            "seconds": float(time.perf_counter() - started),
            "examples": [
                {
                    "args": _json_safe(
                        [None if str(kind) == "any" else value for value, kind in zip(example["args"], example["arg_kinds"])]
                    ),
                    "expected": _json_safe(example["expected"]),
                }
                for example in spec.examples
            ],
        }
        if row["emitted"]:
            verdict = _hidden_check(sandbox, row["source"], task["hidden"])
            row["hidden_ok"] = bool(verdict["ok"])
            row["hidden_detail"] = str(verdict["detail"])
            row["syntax_ok"] = _syntax_ok(row["source"])
        rows.append(row)
    counts = _suite_counts(rows, len(suite["tasks"]))
    return rows, counts


def _suite_counts(rows: Sequence[Dict[str, Any]], selected: int) -> Dict[str, Any]:
    solved = sum(1 for row in rows if row["hidden_ok"])
    emitted = sum(1 for row in rows if row["emitted"])
    failed_hidden = sum(1 for row in rows if row["emitted"] and not row["hidden_ok"])
    abstained = sum(1 for row in rows if not row["emitted"])
    return {
        "selected": int(selected),
        "solved": int(solved),
        "abstained": int(abstained),
        "emitted": int(emitted),
        "emitted_but_failed_hidden": int(failed_hidden),
        "pass_at_1_over_slice_size": float(solved / float(SLICE_SIZE)),
        "pass_at_1_over_selected": float(solved / float(max(selected, 1))),
        "precision": float(solved / float(emitted)) if emitted else 0.0,
    }


# ------------------------------------------------------- plan compilation
def compile_solutions(engine: SynthesisEngine, rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    compiled: List[Dict[str, Any]] = []
    for row in rows:
        if not row["emitted"] or not row["source"]:
            continue
        plan = engine.plans.compile(
            name=str(row["entry"]),
            intent=str(row["intent"]),
            source=str(row["source"]),
            kind="code",
            tier=3,
        )
        compiled.append(
            {
                "name": str(plan.name),
                "suite": str(row["suite"]),
                "task_id": str(row["task_id"]),
                "tier": 3,
                "version": int(plan.version),
                "engram_id": int(plan.engram_id),
                "digest": str(plan.digest),
            }
        )
    return compiled


# --------------------------------------------------------------- arm B core
class _Rename(ast.NodeTransformer):
    def __init__(self, mapping: Dict[str, str]) -> None:
        self.mapping = dict(mapping)

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in self.mapping:
            node.id = self.mapping[node.id]
        return node


def _adapt_source(source: str, name: str, params: Sequence[str]) -> str:
    """Rebind a verified plan to a new signature (name-based role binding)."""
    try:
        tree = ast.parse(str(source))
    except SyntaxError:
        return ""
    definition = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef)), None
    )
    if definition is None:
        return ""
    old_params = [str(arg.arg) for arg in definition.args.args if str(arg.arg) not in {"self", "cls"}]
    new_params = [str(param) for param in params]
    if len(old_params) != len(new_params):
        return ""
    old_function = str(definition.name)
    if set(old_params) == set(new_params):
        # A reordering of the same names is a name-preserving rebind: the body
        # already references the right identifiers, so only the signature
        # order changes.
        for index, arg in enumerate(definition.args.args):
            if str(arg.arg) not in {"self", "cls"}:
                arg.arg = new_params[index]
    else:
        mapping = {old: new for old, new in zip(old_params, new_params) if old != new}
        if old_function and old_function != name:
            mapping[old_function] = str(name)
        for index, arg in enumerate(definition.args.args):
            if str(arg.arg) not in {"self", "cls"}:
                arg.arg = new_params[index]
        if mapping:
            transformer = _Rename(mapping)
            definition.body = [transformer.visit(statement) for statement in definition.body]
    if old_function != name:
        mapping = {old_function: str(name)}
        transformer = _Rename(mapping)
        definition.body = [transformer.visit(statement) for statement in definition.body]
    definition.name = str(name)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


def _compose_sources(
    first: str, second: str, name: str, param: str
) -> str:
    """Chain two verified behaviours into one wrapper module."""
    stage_a = _adapt_source(first, "_z2_stage_1", ["value"])
    stage_b = _adapt_source(second, "_z2_stage_2", ["value"])
    if not stage_a or not stage_b:
        return ""
    wrapper = "def %s(%s):\n    return _z2_stage_2(_z2_stage_1(%s))\n" % (
        str(name),
        str(param),
        str(param),
    )
    return stage_a.strip() + "\n\n\n" + stage_b.strip() + "\n\n\n" + wrapper


def _plan_ranking(engine: SynthesisEngine, intent: str) -> List[Any]:
    """Rank stored playbooks by how much of their intent the request contains.

    A complex request quotes several behaviours, and a cosine between one long
    key and one short key is bounded by their norm ratio, so a plan whose
    whole intent sits inside the request can still lose to a short unrelated
    one.  Containment (the IDF mass of the plan's tokens that the request also
    carries) is the retrieval statistic a compound request needs; coherence
    breaks the ties.
    """
    key = engine.plans.key(str(intent))
    weights = engine.plans.token_weight
    # A compound request must be read in FULL: truncating the request to the
    # same window a single stored intent uses would drop the second half of
    # the sentence and hide the plan that answers it.
    intent_tokens = {
        int(token) for token in engine.plans.tokenizer.encode(str(intent))
    }
    scored = []
    for plan in engine.plans.plans.values():
        plan_tokens = {
            int(token) for token in engine.plans.tokenizer.encode(plan.intent)[:96]
        }
        if not plan_tokens:
            continue
        total = 0.0
        hit = 0.0
        for token in plan_tokens:
            weight = float(weights[token]) if 0 <= token < weights.size else 1.0
            total += weight
            if token in intent_tokens:
                hit += weight
        containment = float(hit / total) if total > 0.0 else 0.0
        coherence = float(np.real(np.vdot(key, engine.plans.key(plan.intent))))
        # Cycle AB2: argument-type verbs ("square", "integer", "count" versus
        # "decimal", "string", "letter") weight the intent match, so a
        # text -> bool predicate cannot outrank a numeric predicate for a
        # request that names a numeric argument.
        bias = 0.0
        try:
            bias = float(engine.plans._type_verb_bias(str(intent), plan))
        except Exception:  # noqa: BLE001 - the bias is optional
            bias = 0.0
        scored.append((min(1.0, containment + bias), coherence + bias, plan))
    scored.sort(key=lambda row: (-row[0], -row[1], row[2].name))
    return [plan for _containment, _coherence, plan in scored]


def arm_b_synthesize(
    prompt: str,
    engine: SynthesisEngine,
    cold: ColdSynthesizer,
    *,
    candidate_cap: int = ARM_B_CANDIDATE_CAP,
) -> Dict[str, Any]:
    """One warm call: retrieve, rebind or compose, then verify."""
    spec = parse_problem(prompt)
    report: Dict[str, Any] = {
        "status": "ABSTAIN",
        "reason": "",
        "source": "",
        "candidates": 0,
        "retrieved": [],
        "rollouts": 0,
        "trajectory": [],
    }
    if spec.contradictions:
        report["reason"] = "contradictory_spec"
        return report
    if spec.faults:
        report["reason"] = "spec_fault:" + str(spec.faults[0])
        return report
    if not spec.assertions:
        report["reason"] = "no_public_examples"
        return report
    if _unsatisfiable(spec.docstring + " " + prompt):
        report["reason"] = "unsatisfiable_capability"
        return report
    params = [str(param["name"]) for param in spec.params]
    intent = _plain_language(spec.docstring) or str(spec.name)
    ranked = _plan_ranking(engine, intent)
    report["retrieved"] = [plan.name for plan in ranked[:8]]
    candidates: List[Tuple[str, str]] = []
    for plan in ranked[:8]:
        adapted = _adapt_source(plan.source, spec.name, params)
        if adapted:
            candidates.append(("adapt:" + str(plan.name), adapted))
    if len(params) == 1 and len(ranked) >= 2:
        unary = [plan for plan in ranked[:8] if len(plan.slots.get("args", []) or []) == 1]
        ordered: List[Tuple[int, int, int, Any, Any]] = []
        for first_index, first in enumerate(unary):
            for second_index, second in enumerate(unary):
                if first.name == second.name:
                    continue
                ordered.append(
                    (first_index + second_index, first_index, second_index, first, second)
                )
        ordered.sort(key=lambda item: (item[0], item[1], item[2]))
        # Cycle AB2: the RepoSheaf domain/codomain gate runs INSIDE the plan
        # selection.  A second stage whose first parameter cannot accept the
        # first stage's return codomain is masked before it is composed, so a
        # text -> bool predicate can never follow a text -> numeric stage.
        type_gate = getattr(engine.plans, "type_gate", None)
        report_state = getattr(engine.plans, "type_filter_report", None)
        for _cost, _first_index, _second_index, first, second in ordered:
            if type_gate is not None:
                verdict = type_gate.check(first, second)
                if report_state is not None:
                    report_state["checked"] = int(report_state.get("checked", 0)) + 1
                    if not verdict.get("ok", True):
                        report_state["masked"] = int(report_state.get("masked", 0)) + 1
                        names = report_state.setdefault("masked_names", [])
                        if str(second.name) not in names and len(names) < 24:
                            names.append(str(second.name))
                if not verdict.get("ok", True):
                    continue
            composed = _compose_sources(
                first.source, second.source, spec.name, params[0]
            )
            if composed:
                candidates.append(
                    ("compose:%s+%s" % (first.name, second.name), composed)
                )
    for label, source in candidates[: int(candidate_cap)]:
        report["candidates"] += 1
        ok, detail = cold._verify(spec, source)
        report["trajectory"].append(
            {"candidate": str(label), "ok": bool(ok), "detail": str(detail)[:160]}
        )
        if ok:
            report["status"] = "SOLVED"
            report["reason"] = "plan:" + str(label)
            report["source"] = str(source)
            return report
    fallback = cold.synthesize(prompt)
    report["rollouts"] = int(fallback["rollouts"])
    if fallback["status"] == "SOLVED":
        report["status"] = "SOLVED"
        report["reason"] = "cold_fallback:" + str(fallback["reason"])
        report["source"] = str(fallback["source"])
        return report
    report["reason"] = report["reason"] or str(fallback["reason"])
    return report


# ------------------------------------------------------- held-out family
def _chains(return_kind: str, param_kind: str) -> bool:
    sequences = {"list", "tuple", "str", "seq"}
    numeric = {"int", "float", "num"}
    if str(return_kind) in sequences and str(param_kind) in sequences:
        return True
    if str(return_kind) in numeric and str(param_kind) in numeric:
        return True
    return False


def _seed_values(kind: str) -> List[Any]:
    """Deterministic seed inputs, chosen to be MIXED-SIGN and varied.

    A seed set that happens to be all positive makes a positivity filter the
    identity, and a composition of it with another plan then degenerates into
    that plan alone.  The negatives, zeros and duplicates below keep a first
    stage a real transformation, so a compound task stays compound.
    """
    if str(kind) in {"list", "tuple"}:
        return [
            [1, -2, 3, -2],
            [0, 3, -1, 2, -1],
            [-5, 4, 4, -1],
            [10, -9, 8, -9],
            [-2, 2, 3, -1, 1],
            [4, -4, 4, -4],
            [7, 0, -7, 7, -7, 0],
        ]
    if str(kind) == "str":
        return ["abca", "hello world", "a b c d", "xyyx", "banana", "aaa bbb", "racecar"]
    if str(kind) in {"int", "num", "float"}:
        return [2, 3, 4, 5, 6, 7, 8]
    return [
        [1, -2, 3, -2],
        [0, 3, -1, 2, -1],
        [-5, 4, 4, -1],
        [10, -9, 8, -9],
        [-2, 2, 3, -1, 1],
    ]


def _seed_arg_tuples(kinds: Sequence[str]) -> List[Tuple[Any, ...]]:
    if len(kinds) == 1:
        return [(value,) for value in _seed_values(str(kinds[0]))]
    if len(kinds) == 2:
        first = _seed_values(str(kinds[0]))
        second = _seed_values(str(kinds[1]))
        if str(kinds[0]) == str(kinds[1]):
            # Same-kind arguments are paired across a one-position shift, so
            # the two positions never carry the same value and an
            # argument-order swap is observable.
            second = second[1:] + second[:1]
        out = []
        for index in range(max(len(first), len(second))):
            out.append((first[index % len(first)], second[index % len(second)]))
        return out
    return []


def _execute_reference(
    source: str, name: str, calls: Sequence[Sequence[Any]]
) -> List[Any]:
    namespace: Dict[str, Any] = {}
    try:
        exec(compile(str(source), "<z2-reference>", "exec"), namespace)
    except Exception:  # noqa: BLE001 - an unusable reference builds no task
        return []
    function = namespace.get(str(name))
    if function is None:
        return []
    results: List[Any] = []
    for args in calls:
        try:
            results.append(function(*args))
        except Exception:  # noqa: BLE001 - a failing input is dropped
            results.append(None)
    return results


def _assert_line(name: str, args: Sequence[Any], expected: Any) -> str:
    return "assert %s(%s) == %s" % (
        str(name),
        ", ".join(repr(value) for value in args),
        repr(expected),
    )


def _clean_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text)).strip()
    return cleaned.replace('"', "'").replace("\\", " ")


def _compose_prompt(
    name: str, param: str, doc_a: str, doc_b: str, public: Sequence[str]
) -> str:
    docstring = "Compose two verified behaviours in order: first %s then %s." % (
        _clean_text(doc_a),
        _clean_text(doc_b),
    )
    return (
        "\n".join(
            ["def %s(%s):" % (name, param), '    """' + docstring + '"""']
            + ["    " + line for line in public]
        )
        + "\n"
    )


def _mutation_prompt(
    name: str, first: str, second: str, description: str, public: Sequence[str]
) -> str:
    docstring = "%s The parameters are supplied in the order %s, %s." % (
        _clean_text(description),
        first,
        second,
    )
    return (
        "\n".join(
            ["def %s(%s, %s):" % (name, first, second), '    """' + docstring + '"""']
            + ["    " + line for line in public]
        )
        + "\n"
    )


def _non_degenerate(values: Sequence[Any]) -> bool:
    return len({repr(value) for value in values}) >= 2


def _samples_from_reference(
    reference: str, name: str, calls: Sequence[Sequence[Any]]
) -> List[Tuple[Tuple[Any, ...], Any]]:
    results = _execute_reference(reference, name, calls)
    return [
        (tuple(args), result)
        for args, result in zip(calls, results)
        if result is not None
    ]


def _composition_tasks(
    pairs: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]], limit: int
) -> Tuple[List[Dict[str, Any]], int]:
    tasks: List[Dict[str, Any]] = []
    skipped = 0
    for first, second in pairs:
        if len(tasks) >= int(limit):
            break
        name = "z2_composed_%02d" % (len(tasks) + 1)
        param = "value"
        reference = _compose_sources(first["source"], second["source"], name, param)
        if not reference:
            skipped += 1
            continue
        calls = [(value,) for value in _seed_values(first["param_kinds"][0])]
        samples = _samples_from_reference(reference, name, calls)
        if len(samples) < 5 or not _non_degenerate([result for _args, result in samples]):
            # A composition whose answer is constant over the whole seed set
            # cannot discriminate a transfer from a lucky emission, so it is
            # not a task.
            skipped += 1
            continue
        public = [_assert_line(name, args, result) for args, result in samples[:3]]
        hidden = [_assert_line(name, args, result) for args, result in samples[3:]]
        tasks.append(
            {
                "kind": "composition",
                "name": name,
                "from": [str(first["entry"]), str(second["entry"])],
                "prompt": _compose_prompt(
                    name, param, str(first["intent"]), str(second["intent"]), public
                ),
                "public": public,
                "hidden": hidden,
            }
        )
    return tasks, int(skipped)


def _mutation_tasks(
    verified: Sequence[Dict[str, Any]], limit: int
) -> Tuple[List[Dict[str, Any]], int, int]:
    candidates = [
        row
        for row in verified
        if len(row["params"]) == 2
        and len(row["param_kinds"]) == 2
        and str(row["param_kinds"][0]) == str(row["param_kinds"][1])
    ]
    candidates.sort(key=lambda row: (str(row["entry"]), str(row["task_id"])))
    tasks: List[Dict[str, Any]] = []
    symmetric = 0
    skipped = 0
    for row in candidates:
        if len(tasks) >= int(limit):
            break
        first_param, second_param = row["params"][0], row["params"][1]
        calls = _seed_arg_tuples(row["param_kinds"])
        if len(calls) < 5:
            skipped += 1
            continue
        original = _execute_reference(
            row["source"], row["entry"], [tuple(args) for args in calls]
        )
        swapped = _execute_reference(
            row["source"], row["entry"], [(args[1], args[0]) for args in calls]
        )
        sensitive = any(
            left is not None and right is not None and repr(left) != repr(right)
            for left, right in zip(original, swapped)
        )
        if not sensitive:
            # The two argument positions are interchangeable for this
            # behaviour, so an argument-order mutation of it would not change
            # the answer and would test nothing.
            symmetric += 1
            continue
        name = "z2_mutated_%02d" % (len(tasks) + 1)
        reference = _adapt_source(row["source"], name, [second_param, first_param])
        if not reference:
            skipped += 1
            continue
        samples = _samples_from_reference(
            reference, name, [(args[1], args[0]) for args in calls]
        )
        if len(samples) < 5 or not _non_degenerate([result for _args, result in samples]):
            skipped += 1
            continue
        public = [_assert_line(name, args, result) for args, result in samples[:3]]
        hidden = [_assert_line(name, args, result) for args, result in samples[3:]]
        tasks.append(
            {
                "kind": "mutation",
                "name": name,
                "from": [str(row["entry"])],
                "prompt": _mutation_prompt(
                    name, second_param, first_param, str(row["intent"]), public
                ),
                "public": public,
                "hidden": hidden,
            }
        )
    return tasks, int(symmetric), int(skipped)


def build_held_out(rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """The held-out family of 12 compound/mutated tasks.

    The recipe is fixed before any Arm-B outcome is looked at.  A composition
    pairs two unary verified behaviours whose output kind chains into the
    second input kind, in name order, and is kept only when the reference
    answer varies over the seed set.  A mutation swaps the two argument
    positions of a two-argument behaviour whose arguments have the SAME kind
    (otherwise the kinds alone disambiguate the order), in name order, and is
    kept only when the reference is measured to be order-sensitive.  Every
    expected value is produced by executing the reference, never written by
    hand.
    """
    verified = [row for row in rows if row["emitted"] and row["hidden_ok"]]
    verified.sort(
        key=lambda row: (str(row["suite"]), str(row["task_id"]), str(row["entry"]))
    )
    unary = [row for row in verified if len(row["params"]) == 1]
    pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for first in unary:
        for second in unary:
            if first["entry"] == second["entry"]:
                continue
            if not first["param_kinds"] or not second["param_kinds"]:
                continue
            if _chains(first["return_kind"], second["param_kinds"][0]):
                pairs.append((first, second))
    pairs.sort(key=lambda pair: (str(pair[0]["entry"]), str(pair[1]["entry"])))

    mutations, symmetric, mutation_skipped = _mutation_tasks(verified, 4)
    keep = max(0, 12 - len(mutations))
    compositions, composition_skipped = _composition_tasks(pairs, keep)
    tasks = compositions + mutations
    counts = {
        "requested": 12,
        "built": int(len(tasks)),
        "compositions": sum(1 for task in tasks if task["kind"] == "composition"),
        "mutations": sum(1 for task in tasks if task["kind"] == "mutation"),
        "verified_plans": int(len(verified)),
        "unary_plans": int(len(unary)),
        "chained_pairs": int(len(pairs)),
        "composition_skipped_degenerate": int(composition_skipped),
        "mutation_skipped_symmetric": int(symmetric),
        "mutation_skipped_other": int(mutation_skipped),
    }
    return tasks, counts


def run_held_out(
    tasks: Sequence[Dict[str, Any]],
    sandbox: SubprocessSandboxExecutor,
    engine: SynthesisEngine,
    *,
    arm: str,
) -> List[Dict[str, Any]]:
    cold = ColdSynthesizer(sandbox=sandbox, budget=BUDGET, plan_store=None)
    rows: List[Dict[str, Any]] = []
    for task in tasks:
        started = time.perf_counter()
        if arm == "a":
            outcome = cold.synthesize(task["prompt"])
            report = {
                "status": str(outcome["status"]),
                "reason": str(outcome["reason"]),
                "source": str(outcome["source"]),
                "rollouts": int(outcome["rollouts"]),
                "candidates": 0,
            }
        else:
            report = arm_b_synthesize(task["prompt"], engine, cold)
        verdict = {"ok": False, "detail": "not_emitted"}
        syntax_ok = True
        if report["status"] == "SOLVED":
            verdict = _hidden_check(sandbox, report["source"], task["hidden"])
            syntax_ok = _syntax_ok(report["source"])
        rows.append(
            {
                "name": str(task["name"]),
                "kind": str(task["kind"]),
                "from": list(task["from"]),
                "status": str(report["status"]),
                "reason": str(report["reason"]),
                "emitted": bool(report["status"] == "SOLVED"),
                "hidden_ok": bool(verdict["ok"]),
                "hidden_detail": str(verdict["detail"]),
                "syntax_ok": bool(syntax_ok),
                "rollouts": int(report.get("rollouts", 0)),
                "candidates": int(report.get("candidates", 0)),
                "source_sha256": (
                    _sha256_text(str(report["source"])) if report["source"] else ""
                ),
                "source_head": str(report["source"])[:240],
                "seconds": float(time.perf_counter() - started),
            }
        )
    return rows


def _held_out_counts(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    built = len(rows)
    emitted = sum(1 for row in rows if row["emitted"])
    passed = sum(1 for row in rows if row["hidden_ok"])
    return {
        "tasks": int(built),
        "solved": int(passed),
        "emitted": int(emitted),
        "abstained": int(built - emitted),
        "emitted_but_failed_hidden": int(emitted - passed),
        "transfer_accuracy": float(passed / float(max(built, 1))),
    }


# ------------------------------------------------------------------ traps
def run_traps(
    sandbox: SubprocessSandboxExecutor, engine: SynthesisEngine
) -> List[Dict[str, Any]]:
    cold = ColdSynthesizer(sandbox=sandbox, budget=BUDGET, plan_store=None)
    rows: List[Dict[str, Any]] = []
    for trap in TRAPS:
        outcome = cold.synthesize(trap["prompt"])
        warm = arm_b_synthesize(trap["prompt"], engine, cold)
        rows.append(
            {
                "name": str(trap["name"]),
                "kind": str(trap["kind"]),
                "arm_a_status": str(outcome["status"]),
                "arm_a_reason": str(outcome["reason"]),
                "arm_b_status": str(warm["status"]),
                "arm_b_reason": str(warm["reason"]),
                "refused": bool(
                    str(outcome["status"]) == "ABSTAIN"
                    and str(warm["status"]) == "ABSTAIN"
                ),
            }
        )
    return rows


# ------------------------------------------------------------------ audit
def audit_synthesizer(identifiers: Sequence[str]) -> Dict[str, Any]:
    result = audit_task_specific_branches(
        COLD_SYNTH.read_text(encoding="utf-8"), identifiers=identifiers
    )
    result["file"] = str(COLD_SYNTH.relative_to(ROOT))
    result["sha256"] = _sha256_file(COLD_SYNTH)
    result["lines"] = int(len(COLD_SYNTH.read_text(encoding="utf-8").splitlines()))
    result["identifiers_checked"] = int(len(identifiers))
    return result


def _llm_modules_loaded() -> List[str]:
    banned = ("openai", "anthropic", "openrouter", "google.generativeai", "transformers", "litellm")
    return sorted(
        name
        for name in list(sys.modules)
        if any(token in str(name).lower() for token in banned)
    )


# ------------------------------------------------------------------ measure
def measure(directory: Path) -> Dict[str, Any]:
    started = time.perf_counter()
    directory.mkdir(parents=True, exist_ok=True)
    workdir = directory / "_work" / "z2"
    workdir.mkdir(parents=True, exist_ok=True)

    primary = load_primary()
    secondary = load_secondary()
    provenance = provenance_block()
    print(
        "[z2] primary mbpp selected %d/%d eligible (of %d rows) digest=%s"
        % (primary["selected"], primary["eligible"], primary["rows"], primary["digest"][:16]),
        flush=True,
    )
    print(
        "[z2] secondary humaneval selected %d/%d eligible (of %d rows) digest=%s"
        % (secondary["selected"], secondary["eligible"], secondary["rows"], secondary["digest"][:16]),
        flush=True,
    )

    sandbox = SubprocessSandboxExecutor("pure_algorithmic", timeout_s=8.0, workdir=str(workdir))

    # The secondary suite is measured FIRST, so the runtime order matches the
    # disclosed pre-registration order: the HumanEval probe precedes the MBPP
    # primary announcement.
    arm_a_secondary_started = time.perf_counter()
    rows_secondary, counts_secondary = run_arm_a(sandbox, secondary)
    print(
        "[z2] arm A secondary solved %d emitted %d abstained %d failed-hidden %d (%.1fs)"
        % (
            counts_secondary["solved"],
            counts_secondary["emitted"],
            counts_secondary["abstained"],
            counts_secondary["emitted_but_failed_hidden"],
            time.perf_counter() - arm_a_secondary_started,
        ),
        flush=True,
    )

    arm_a_primary_started = time.perf_counter()
    rows_primary, counts_primary = run_arm_a(sandbox, primary)
    print(
        "[z2] arm A primary solved %d emitted %d abstained %d failed-hidden %d (%.1fs)"
        % (
            counts_primary["solved"],
            counts_primary["emitted"],
            counts_primary["abstained"],
            counts_primary["emitted_but_failed_hidden"],
            time.perf_counter() - arm_a_primary_started,
        ),
        flush=True,
    )

    rows = rows_primary + rows_secondary
    emitted_total = counts_primary["emitted"] + counts_secondary["emitted"]
    solved_total = counts_primary["solved"] + counts_secondary["solved"]
    failed_hidden_total = (
        counts_primary["emitted_but_failed_hidden"]
        + counts_secondary["emitted_but_failed_hidden"]
    )
    precision = float(solved_total / float(emitted_total)) if emitted_total else 0.0

    maximum_rollouts = int(max((row["rollouts"] for row in rows), default=0))
    broken_ast = sum(1 for row in rows if row["emitted"] and not row["syntax_ok"])

    # ---- Arm B: compile every Arm-A verified solution into the tenant store
    stem = SharedCorpusStem.build(ROOT, max_fit_tokens=60000, build_index=False)
    checkpoint = find_checkpoint(None)
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = HolarchConfig(**payload["config"])
    memory, memory_info = open_clean_room_memory(
        config, checkpoint, workdir, prefix="z2_arm_b"
    )
    engine = SynthesisEngine.from_stem(stem, memory, namespace=PLAN_NAMESPACE)
    plans_before = int(len(engine.plans.plans))
    compiled = compile_solutions(engine, rows)
    plans_after = int(len(engine.plans.plans))
    print(
        "[z2] planstore compiled %d plans (%d -> %d) in namespace %s"
        % (len(compiled), plans_before, plans_after, PLAN_NAMESPACE),
        flush=True,
    )

    held_out, held_out_meta = build_held_out(rows)
    print(
        "[z2] held-out built %d/%d (compositions=%d mutations=%d)"
        % (
            held_out_meta["built"],
            held_out_meta["requested"],
            held_out_meta["compositions"],
            held_out_meta["mutations"],
        ),
        flush=True,
    )
    held_a = run_held_out(held_out, sandbox, engine, arm="a")
    held_b = run_held_out(held_out, sandbox, engine, arm="b")
    counts_held_a = _held_out_counts(held_a)
    counts_held_b = _held_out_counts(held_b)
    print(
        "[z2] held-out arm A %d/%d arm B %d/%d"
        % (
            counts_held_a["solved"],
            counts_held_a["tasks"],
            counts_held_b["solved"],
            counts_held_b["tasks"],
        ),
        flush=True,
    )

    traps = run_traps(sandbox, engine)
    traps_refused = sum(1 for row in traps if row["refused"])
    print("[z2] traps refused %d/%d" % (traps_refused, len(traps)), flush=True)

    identifiers: List[str] = []
    for suite in (primary, secondary):
        identifiers.extend(str(task["entry"]) for task in suite["tasks"])
        identifiers.extend(str(task["task_id"]) for task in suite["tasks"])
        identifiers.extend(str(task["task_id"]).split("/")[-1] for task in suite["tasks"])
    audit = audit_synthesizer(sorted(set(identifiers)))

    held_out_rollouts = int(
        max(
            [row["rollouts"] for row in held_a] + [row["rollouts"] for row in held_b],
            default=0,
        )
    )
    budget_ok = bool(maximum_rollouts <= BUDGET and held_out_rollouts <= BUDGET)
    provenance_ok = bool(all(entry["matches"] for entry in provenance))
    llm_modules = _llm_modules_loaded()
    llm_calls = 0

    held_out_syntax_broken = sum(
        1
        for row in held_a + held_b
        if row["emitted"] and not row["syntax_ok"]
    )
    broken_ast += int(held_out_syntax_broken)
    stages_complete = bool(
        len(rows) == primary["selected"] + secondary["selected"]
        and len(held_out) > 0
        and len(traps) == len(TRAPS)
    )

    gates = {
        "z2_runner_runs": bool(stages_complete),
        "z2_zero_external_llm_calls": bool(llm_calls == 0 and not llm_modules),
        "z2_precision_ge_095": bool(precision >= 0.95),
        "z2_syntax_type_validity": bool(broken_ast == 0),
        "z2_primary_pass_at_1_ge_050": bool(
            counts_primary["pass_at_1_over_slice_size"] >= 0.5
        ),
        "z2_arm_b_beats_arm_a": bool(
            counts_held_b["solved"] > counts_held_a["solved"]
        ),
        "z2_traps_6_of_6": bool(traps_refused == len(TRAPS)),
        "z2_planstore_compiled": bool(
            plans_before == 0
            and len(compiled) == emitted_total
            and plans_after == plans_before + len(compiled)
        ),
        "z2_no_task_specific_branches": bool(audit["ok"]),
        "z2_budget_32": bool(budget_ok),
        "z2_provenance_recorded": bool(provenance_ok),
    }

    mechanism: Dict[str, int] = {}
    for row in held_b:
        label = str(row["reason"]).split(":", 2)
        key = label[0] if label else str(row["reason"])
        if key == "plan":
            key = "plan:" + (label[1] if len(label) > 1 else "unknown")
        mechanism[key] = mechanism.get(key, 0) + 1

    return {
        "stage": "Z2",
        "kind": "external_cold_synthesis_audit",
        "head": pc._git_head(),
        "disclosure": DISCLOSURE,
        "measurement_order": [
            "primary slice selection (MBPP rule)",
            "secondary slice selection (HumanEval rule)",
            "arm A cold synthesis: secondary suite",
            "arm A cold synthesis: primary suite",
            "arm B plan compilation",
            "arm B held-out evaluation and traps",
        ],
        "slices": {
            "primary": {
                "name": primary["name"],
                "file": primary["path"],
                "rows": primary["rows"],
                "eligible": primary["eligible"],
                "slice_size": primary["slice_size"],
                "selected": primary["selected"],
                "digest": primary["digest"],
                "task_ids": primary["task_ids"],
                "rule": primary["rule"],
            },
            "secondary": {
                "name": secondary["name"],
                "file": secondary["path"],
                "rows": secondary["rows"],
                "eligible": secondary["eligible"],
                "slice_size": secondary["slice_size"],
                "selected": secondary["selected"],
                "digest": secondary["digest"],
                "task_ids": secondary["task_ids"],
                "rule": secondary["rule"],
            },
            "digest_definition": (
                "sha256 over the canonical JSON of the selected task ids and "
                "their hidden assertions, sorted keys, compact separators"
            ),
        },
        "provenance": provenance,
        "arm_a": {
            "note": (
                "strict zero-LLM cold start: empty PlanStore, at most 32 "
                "sandboxed rollouts per task, verification against the PUBLIC "
                "examples only, RepoSheaf capability pullback, ABSTAIN when "
                "nothing passes"
            ),
            "budget": int(BUDGET),
            "llm_calls": int(llm_calls),
            "llm_client_modules_loaded": llm_modules,
            "wave_scorer": "disabled (structural prior only)",
            "primary": counts_primary,
            "secondary": counts_secondary,
            "combined": {
                "selected": int(counts_primary["selected"] + counts_secondary["selected"]),
                "solved": int(solved_total),
                "emitted": int(emitted_total),
                "abstained": int(
                    counts_primary["abstained"] + counts_secondary["abstained"]
                ),
                "emitted_but_failed_hidden": int(failed_hidden_total),
                "precision": float(precision),
            },
            "rows": rows,
        },
        "arm_b": {
            "note": (
                "holonic self-play consolidation: every Arm-A verified source "
                "is compiled into the tenant PlanStore as a Tier-3 ACE "
                "playbook; transfer is measured on a held-out family of "
                "compound and mutated tasks, not asserted"
            ),
            "namespace": str(PLAN_NAMESPACE),
            "memory": {key: _json_safe(value) for key, value in memory_info.items()},
            "plans_before": int(plans_before),
            "plans_after": int(plans_after),
            "compiled": compiled,
            "compiled_count": int(len(compiled)),
            "held_out_meta": held_out_meta,
            "held_out": [
                {
                    "name": str(task["name"]),
                    "kind": str(task["kind"]),
                    "from": list(task["from"]),
                    "public": list(task["public"]),
                    "hidden": list(task["hidden"]),
                }
                for task in held_out
            ],
            "arm_a": {"counts": counts_held_a, "rows": held_a},
            "arm_b": {
                "counts": counts_held_b,
                "rows": held_b,
                "mechanism": dict(sorted(mechanism.items())),
                "mechanism_note": (
                    "plan:compose means the emitted module chains two retrieved "
                    "playbooks; plan:adapt means one retrieved playbook was "
                    "rebound to the new signature; cold_fallback means the warm "
                    "arm found nothing and fell back to the cold search."
                ),
            },
            "traps": traps,
            "traps_refused": int(traps_refused),
            "traps_total": int(len(TRAPS)),
        },
        "audit": audit,
        "summary": {
            "primary": {
                "solved": int(counts_primary["solved"]),
                "abstained": int(counts_primary["abstained"]),
                "emitted": int(counts_primary["emitted"]),
                "emitted_but_failed_hidden": int(
                    counts_primary["emitted_but_failed_hidden"]
                ),
                "selected": int(counts_primary["selected"]),
                "pass_at_1_over_slice_size": float(
                    counts_primary["pass_at_1_over_slice_size"]
                ),
                "precision": float(counts_primary["precision"]),
            },
            "secondary": {
                "solved": int(counts_secondary["solved"]),
                "abstained": int(counts_secondary["abstained"]),
                "emitted": int(counts_secondary["emitted"]),
                "emitted_but_failed_hidden": int(
                    counts_secondary["emitted_but_failed_hidden"]
                ),
                "selected": int(counts_secondary["selected"]),
                "pass_at_1_over_slice_size": float(
                    counts_secondary["pass_at_1_over_slice_size"]
                ),
                "precision": float(counts_secondary["precision"]),
            },
            "combined": {
                "emitted": int(emitted_total),
                "solved": int(solved_total),
                "emitted_but_failed_hidden": int(failed_hidden_total),
                "precision": float(precision),
                "broken_ast": int(broken_ast),
            },
            "held_out": {
                "arm_a_solved": int(counts_held_a["solved"]),
                "arm_b_solved": int(counts_held_b["solved"]),
                "tasks": int(counts_held_b["tasks"]),
                "arm_b_transfer_accuracy": float(counts_held_b["transfer_accuracy"]),
                "arm_a_transfer_accuracy": float(counts_held_a["transfer_accuracy"]),
            },
            "traps": {
                "refused": int(traps_refused),
                "total": int(len(TRAPS)),
            },
            "audit": {
                "task_identity_comparisons": int(audit["task_identity_comparisons"]),
                "compared_identifier_literal_count": int(
                    audit["compared_identifier_literal_count"]
                ),
                "unexplained_string_comparisons": int(
                    audit["unexplained_string_comparisons"]
                ),
                "module_literal_overlap": list(audit["module_literal_overlap"]),
            },
            "max_rollouts": int(maximum_rollouts),
            "plans_compiled": int(len(compiled)),
        },
        "budget": {
            "declared": int(BUDGET),
            "max_rollouts_arm_a": int(maximum_rollouts),
            "max_rollouts_held_out": int(held_out_rollouts),
        },
        "gate": pc._finalise(gates),
        "gate_notes": {
            "z2_primary_pass_at_1_ge_050": (
                "solved / 32 (the pre-registered slice size). The selected "
                "count is also reported; when fewer than 32 tasks qualify the "
                "denominator stays 32, which is the stricter reading."
            ),
            "z2_precision_ge_095": (
                "hidden assertions passed / emitted, over BOTH suites combined; "
                "each suite reports its own precision next to this number."
            ),
        },
        "passed": bool(all(gates.values())),
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Cycle Z2 external cold synthesis")
    parser.add_argument("--receipts", default=str(RECEIPTS))
    args = parser.parse_args(argv)
    directory = Path(args.receipts)
    directory.mkdir(parents=True, exist_ok=True)
    payload = measure(directory)
    written = pc._write(directory, "z2_external_cold_synthesis.json", payload)
    primary = payload["arm_a"]["primary"]
    secondary = payload["arm_a"]["secondary"]
    print(
        "[z2] PRIMARY   solved %d/%d (slice %d) abstained %d emitted %d failed-hidden %d precision %.4f"
        % (
            primary["solved"],
            primary["selected"],
            primary["selected"],
            primary["abstained"],
            primary["emitted"],
            primary["emitted_but_failed_hidden"],
            primary["precision"],
        )
    )
    print(
        "[z2] SECONDARY solved %d/%d (slice %d) abstained %d emitted %d failed-hidden %d precision %.4f"
        % (
            secondary["solved"],
            secondary["selected"],
            secondary["selected"],
            secondary["abstained"],
            secondary["emitted"],
            secondary["emitted_but_failed_hidden"],
            secondary["precision"],
        )
    )
    print(
        "[z2] pass@1 primary %.4f (solved/32) secondary %.4f (solved/32) precision %.4f"
        % (
            payload["arm_a"]["primary"]["pass_at_1_over_slice_size"],
            payload["arm_a"]["secondary"]["pass_at_1_over_slice_size"],
            payload["arm_a"]["combined"]["precision"],
        )
    )
    print(
        "[z2] held-out arm A %d/%d arm B %d/%d traps %d/%d plans %d->%d"
        % (
            payload["arm_b"]["arm_a"]["counts"]["solved"],
            payload["arm_b"]["arm_a"]["counts"]["tasks"],
            payload["arm_b"]["arm_b"]["counts"]["solved"],
            payload["arm_b"]["arm_b"]["counts"]["tasks"],
            payload["arm_b"]["traps_refused"],
            payload["arm_b"]["traps_total"],
            payload["arm_b"]["plans_before"],
            payload["arm_b"]["plans_after"],
        )
    )
    failed = [name for name, value in payload["gate"].items() if not bool(value)]
    print(
        "[z2] passed=%s failed=%s receipt=%s sha256=%s"
        % (payload["passed"], failed, written["path"], written["sha256"])
    )
    return 0 if bool(payload["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
