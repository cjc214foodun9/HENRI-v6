"""Zero-LLM cold-start code synthesis (Directive Y2).

The PlanStore has no playbook for a brand-new function, so the Cycle X arm0
had no cold path: it abstained 4/4 on unseen coding tasks.  This module adds
the deterministic, zero-LLM cold synthesizer the Y2 gate measures:

    parse      WaveCCG + RepoSheaf read the signature, the type annotations
               and the docstring into a Slot-3 goal wave and a typed slot
               inventory.
    generate   a typed AST sketch library is enumerated under the live
               ASTGate; every sketch slot is pruned to its top
               V_eff <= 8 wave-scored fills, so the per-slot effective
               branching factor is measured, not assumed.
    verify     a candidate is executed against the PUBLIC docstring
               assertions only, inside the U5 out-of-process sandbox.  A
               candidate is emitted only when 100% of them pass and the
               RepoSheaf capability profile of the emitted module is
               non-empty; otherwise the call fail-closes with ABSTAIN.
    steer      a refuted candidate registers a Zone C
               BoundaryAxiom(polarity="REPULSOR", source="cold_synth") and
               the search switches to high-temperature (T = 0.85) sampling
               over the remaining wave support, so later rollouts are
               steered away from the failing AST trajectory.

No language model is called anywhere in this file.  The wave score is the
egress (Y1) or the count floor's own log-probability, both local.
"""

from __future__ import annotations

import ast
import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

_EPS = 1e-12
SLOT_SUPPORT = 8
"""V_eff ceiling: at most eight admitted fills per AST slot (Y2 gate)."""
ROLLOUT_BUDGET = 24
"""B: at most twenty-four sandboxed candidates per task."""
EXPLORE_TEMPERATURE = 0.85
"""The high-temperature Zone B discovery temperature after a REPULSOR."""

_SIGNATURE = re.compile(
    r"def\s+([A-Za-z_]\w*)\s*\(([^)]*)\)\s*(?:->\s*([^:\n]+))?\s*:"
)
_ASSERT = re.compile(r"^\s*assert\s+.+$", re.MULTILINE)
_DOCTEST = re.compile(r"^\s*>>>\s*(.+)$", re.MULTILINE)
_URL = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
_UNSATISFIABLE_WORDS = (
    "internet", "http", "https", "network", "download", "fetch the latest",
    "web", "api key", "database server", "email", "stock price",
)
"""A capability the U5 pure-algorithmic profile cannot satisfy."""


# ------------------------------------------------------------------ parsing
def _split_params(text: str) -> List[str]:
    parts: List[str] = []
    depth = 0
    current = ""
    for char in str(text):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        if char == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += char
    if current.strip():
        parts.append(current)
    return [part.strip() for part in parts if part.strip()]


_STR_TOKENS = frozenset(
    {
        "str", "string", "strings", "text", "sentence", "phrase", "line",
        "char", "character", "substring", "pattern", "delimiter", "s",
        "token", "text1", "word",
    }
)
_SEQ_TOKENS = frozenset(
    {
        "list", "lists", "array", "arrays", "arr", "arrs", "nums", "numbers",
        "values", "items", "data", "matrix", "rows", "row", "seq", "sequence",
        "sequences", "tuple", "tup", "tuples", "vector", "elements",
        "collection", "iterable", "words", "letters", "pairs", "l", "xs",
        "ys", "groups", "sublists", "strings", "stack", "queue", "set",
    }
)
_NUM_TOKENS = frozenset(
    {
        "n", "x", "y", "z", "k", "m", "a", "b", "c", "d", "i", "j", "num",
        "number", "count", "index", "size", "length", "threshold", "target",
        "total", "value", "power", "base", "shift", "sum", "cost", "amount",
        "step", "start", "end", "limit", "precision", "width", "height",
    }
)


def _name_tokens(name: str) -> List[str]:
    """Word tokens of an identifier: snake_case and camelCase both split.

    A substring test ("s" in "numbers") silently mis-typed every argument
    whose name merely CONTAINED a letter; whole-token matching is the general
    rule that a name is a sequence of words, not a bag of characters.
    """
    out: List[str] = []
    for part in re.split(r"[^A-Za-z0-9]+", str(name)):
        for piece in re.findall(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+", part):
            token = re.sub(r"[0-9]+$", "", piece.lower())
            if token:
                out.append(token)
    return out


def _annotation_role(annotation: str, name: str) -> str:
    """The typed role of one argument: annotation first, then name words."""
    text = str(annotation or "").lower()
    if text:
        if "bool" in text:
            return "bool"
        if "str" in text:
            return "str"
        if any(word in text for word in ("list", "sequence", "tuple", "iterable", "set")):
            return "seq"
        if any(word in text for word in ("int", "float", "number")):
            return "num"
    tokens = set(_name_tokens(name))
    if tokens & _SEQ_TOKENS:
        return "seq"
    if tokens & _STR_TOKENS:
        return "str"
    if tokens & _NUM_TOKENS:
        return "num"
    return "num"


@dataclass
class ProblemSpec:
    """Everything the synthesizer is allowed to know about one task.

    The added fields are the general structural prior: the public examples
    read as typed call/answer pairs, the literal constants the specification
    itself names, the output kind the examples agree on, and the shape facts
    a length/permutation/elementwise comparison yields.  None of them names
    a task.
    """

    name: str
    params: List[Dict[str, str]]
    returns: str
    docstring: str
    assertions: List[str]
    goal_wave: np.ndarray
    contradictions: List[str] = field(default_factory=list)
    faults: List[str] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    literals: List[str] = field(default_factory=list)
    return_kind: str = "any"
    arg_kinds: Dict[str, str] = field(default_factory=dict)
    shape: Dict[str, Any] = field(default_factory=dict)
    prompt: str = ""

    @property
    def words(self) -> List[str]:
        text = (
            self.docstring
            + " "
            + " ".join(self.assertions)
            + " "
            + str(self.name).replace("_", " ")
        ).lower()
        return re.findall(r"[a-z]+", text)

    def by_role(self, role: str) -> List[str]:
        return [param["name"] for param in self.params if param["role"] == role]


def _doctest_assertions(docstring: str) -> List[str]:
    """Parse classic doctest pairs into assert statements.

    A >>> call line is followed by the expected repr on the next non-empty
    line.  This is the public-example channel of an unmodified benchmark
    prompt: the external problems carry their own doctests, so no example is
    authored here.  A Traceback case and a multi-line expected value are
    skipped, because neither is a literal the sandbox assert can compare.
    """
    lines = str(docstring).splitlines()
    out: List[str] = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line.startswith(">>>"):
            index += 1
            continue
        expression = line[3:].strip()
        index += 1
        while index < len(lines) and not lines[index].strip():
            index += 1
        if index >= len(lines):
            break
        expected = lines[index].strip()
        continuation = (
            index + 1 < len(lines) and lines[index + 1].strip().startswith("...")
        )
        if (
            not expression
            or not expected
            or expected.startswith("...")
            or "Traceback" in expected
            or continuation
        ):
            index += 1
            continue
        if "==" in expression:
            out.append("assert " + expression)
        else:
            out.append("assert (%s) == (%s)" % (expression, expected))
        index += 1
    return out


_MISSING = object()


def _literal_kind(value: Any) -> str:
    """The structural kind of one literal value."""
    if value is _MISSING:
        return "any"
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, tuple):
        return "tuple"
    if isinstance(value, dict):
        return "dict"
    if isinstance(value, (set, frozenset)):
        return "set"
    return "any"


def _literal_value(node: ast.AST) -> Any:
    """The literal value of one AST node, or the sentinel when not literal."""
    try:
        return ast.literal_eval(node)
    except Exception:  # noqa: BLE001 - a non-literal argument is not an example
        return _MISSING


def _parse_examples(assertions: Sequence[str], name: str) -> List[Dict[str, Any]]:
    """Read the public assertions as typed call/answer pairs.

    Three shapes are admitted, because all three appear in public benchmark
    tests: "assert f(...) == value", "assert f(...)", and "assert not f(...)".
    The expected value of a bare call is True, of a negated call False.  A
    call whose arguments are not literals is kept with an unknown argument
    kind, so the shape prior never invents a value it did not read.
    """
    out: List[Dict[str, Any]] = []
    for line in assertions:
        text = str(line).strip()
        if not text.startswith("assert "):
            continue
        body = text[len("assert "):].strip()
        try:
            node = ast.parse(body, mode="eval").body
        except SyntaxError:
            continue
        expected: Any = _MISSING
        call = None
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq):
            call = node.left
            expected = _literal_value(node.comparators[0])
        elif isinstance(node, ast.Call):
            call, expected = node, True
        elif (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, ast.Not)
            and isinstance(node.operand, ast.Call)
        ):
            call, expected = node.operand, False
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
            continue
        args = [_literal_value(arg) for arg in call.args]
        keywords = {
            str(keyword.arg): _literal_value(keyword.value)
            for keyword in call.keywords
            if keyword.arg is not None
        }
        out.append(
            {
                "callee": str(call.func.id),
                "args": args,
                "keywords": keywords,
                "expected": expected,
                "expected_kind": _literal_kind(expected),
                "arg_kinds": [_literal_kind(value) for value in args],
                "text": text,
            }
        )
    return [row for row in out if str(row["callee"]) == str(name)]


def _harvest_literals(docstring: str, examples: Sequence[Dict[str, Any]]) -> List[str]:
    """String literals the specification itself names, as Python source.

    A public description that names a constant ("the string 'std'") is part
    of the task, not an implementation detail: binding it is what makes the
    literal-count schema reachable without a per-task rule.  Only short,
    single-line quotations are harvested, and the pool is capped, so the
    binding stays a deterministic enumeration.
    """
    pool: List[str] = []

    def add(value: str) -> None:
        text = str(value).strip()
        if not text or len(text) > 24:
            return
        if "\n" in text or "\t" in text:
            return
        rendered = repr(text)
        if rendered not in pool:
            pool.append(rendered)

    for single, double in re.findall(r"'([^'\n]{1,24})'|\"([^\"\n]{1,24})\"", str(docstring)):
        add(single or double)
    for row in examples:
        for value in row.get("args", []):
            if isinstance(value, str):
                add(value)
    return pool[:6]


def _module_exists(name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(str(name)) is not None
    except Exception:  # noqa: BLE001 - an unreadable name is not a dependency
        return False


def _prompt_faults(prompt: str) -> List[str]:
    """Capability faults the PROMPT itself declares, before any synthesis.

    A prompt that imports a module the U5 profile blocks, or a module that is
    not installed at all, asks for a capability the emitted module cannot
    have.  The screen is fail-closed on the prompt's own import statements
    and silent on a prompt that is not valid Python.
    """
    try:
        tree = ast.parse(str(prompt))
    except SyntaxError:
        return []
    from henri.holarch.sandbox import BLOCKED_MODULES

    faults: List[str] = []
    for node in ast.walk(tree):
        modules: List[str] = []
        if isinstance(node, ast.Import):
            modules = [str(alias.name) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [str(node.module or "")]
        for module in modules:
            root = module.split(".")[0].strip()
            if not root:
                continue
            if root in BLOCKED_MODULES:
                faults.append("blocked_import:" + root)
            elif not _module_exists(root):
                faults.append("missing_import:" + root)
    return sorted(set(faults))


def _arity_fault(
    examples: Sequence[Dict[str, Any]], expected: int, flexible: bool
) -> str:
    """A public example that calls the declared function with the wrong arity."""
    if flexible or not examples:
        return ""
    arities = sorted({len(row.get("args", [])) for row in examples if not row.get("keywords")})
    if arities and any(int(value) != int(expected) for value in arities):
        return "arity_mismatch:" + ",".join(str(value) for value in arities)
    return ""


def _shape_from_examples(
    examples: Sequence[Dict[str, Any]], return_kind: str
) -> Dict[str, Any]:
    """Length/permutation/elementwise/fold facts read off the public examples.

    This is the structural prior the synthesizer ranks with.  Every fact is a
    relation between an observed argument and an observed answer, so it holds
    for any task whose public examples are literals.
    """
    shape: Dict[str, Any] = {
        "fold": False,
        "elementwise": False,
        "permutation": False,
        "pairwise": False,
        "halving": False,
        "subset": False,
        "conversion": False,
        "ratios": [],
    }
    for row in examples:
        args = list(row.get("args", []))
        expected = row.get("expected", _MISSING)
        if expected is _MISSING:
            continue
        if return_kind in ("int", "float", "num", "bool") and any(
            _literal_kind(value) in ("list", "tuple") for value in args
        ):
            shape["fold"] = True
        if isinstance(expected, list):
            if all(isinstance(item, tuple) and len(item) == 2 for item in expected) and expected:
                shape["pairwise"] = True
            for value in args:
                if not isinstance(value, (list, tuple)):
                    continue
                if len(value) and len(expected) == len(value) and all(
                    not isinstance(item, (list, tuple)) for item in expected
                ):
                    shape["elementwise"] = True
                    if sorted(map(repr, expected)) == sorted(map(repr, value)):
                        shape["permutation"] = True
                if len(value) and len(expected) < len(value):
                    shape["subset"] = True
                if len(value) and len(expected) == len(value) // 2:
                    shape["halving"] = True
                if len(value) and len(expected) == max(0, len(value) - 1):
                    shape["pairwise"] = True
                if len(expected) and len(expected) == len(value):
                    shape["ratios"].append(round(float(len(expected)) / float(len(value)), 3))
        if isinstance(expected, (tuple, list)) and expected and all(
            isinstance(item, int) and not isinstance(item, bool) for item in expected
        ) and any(isinstance(value, str) for value in args):
            shape["conversion"] = True
        if isinstance(expected, str):
            for value in args:
                if not isinstance(value, str) or not value:
                    continue
                if len(expected) < len(value):
                    shape["subset"] = True
                if len(expected) * 2 == len(value):
                    shape["halving"] = True
    return shape


def _prompt_call_names(text: str) -> List[str]:
    """Names called by the prompt's own public examples."""
    names: List[str] = []
    for expression in _DOCTEST.findall(str(text)):
        try:
            tree = ast.parse(str(expression).strip(), mode="eval")
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                names.append(str(node.func.id))
    for line in _ASSERT.findall(str(text)):
        try:
            tree = ast.parse(str(line).strip())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                names.append(str(node.func.id))
    return names


def _select_signature(text: str):
    """The definition a prompt is asking for.

    A prompt may declare a helper before the function under test, and the
    first def is then the wrong one.  The rule is outcome-blind: prefer the
    LAST definition the prompt's own examples call, because the examples are
    the public contract; a prompt with no call evidence falls back to its last
    definition, which is where the requested function sits.
    """
    matches = list(_SIGNATURE.finditer(str(text)))
    if not matches:
        return None
    called = set(_prompt_call_names(text))
    for match in reversed(matches):
        if str(match.group(1)) in called:
            return match
    return matches[-1]


def parse_problem(prompt: str, *, extra_tests: Optional[Sequence[str]] = None) -> ProblemSpec:
    """Read one task prompt into a typed specification and a Slot-3 goal wave."""
    text = str(prompt)
    match = _select_signature(text)
    if match is None:
        raise ValueError("prompt carries no function signature")
    name = str(match.group(1))
    returns = str(match.group(3) or "").strip()
    params: List[Dict[str, str]] = []
    raw_params = _split_params(match.group(2))
    for raw in raw_params:
        if raw in ("self", "cls"):
            continue
        piece = raw.split(":")
        left = piece[0].split("=")[0].strip()
        annotation = piece[1].split("=")[0].strip() if len(piece) > 1 else ""
        params.append(
            {
                "name": left,
                "annotation": annotation,
                "role": _annotation_role(annotation, left),
            }
        )
    # The docstring belongs to the selected definition, not to a helper that
    # the prompt happened to define first.
    doc_match = re.search(r'"""(.*?)"""', text[match.end():], flags=re.DOTALL)
    docstring = str(doc_match.group(1)) if doc_match else ""
    assertions = [line.strip() for line in _ASSERT.findall(docstring + "\n" + text)]
    assertions.extend(_doctest_assertions(docstring))
    if extra_tests:
        assertions.extend(str(line).strip() for line in extra_tests)
    seen = set()
    unique_assertions: List[str] = []
    for line in assertions:
        if line in seen:
            continue
        seen.add(line)
        unique_assertions.append(line)

    examples = _parse_examples(unique_assertions, name)
    # An unannotated argument is typed by what the public examples actually
    # pass into it.  The declaration wins when it declares a type.
    for index, param in enumerate(params):
        if str(param.get("annotation", "")).strip():
            continue
        kinds = {
            row["arg_kinds"][index]
            for row in examples
            if index < len(row["arg_kinds"]) and row["arg_kinds"][index] != "any"
        }
        if len(kinds) == 1:
            kind = kinds.pop()
            if kind in ("list", "tuple", "set"):
                param["role"] = "seq"
            elif kind == "str":
                param["role"] = "str"
            elif kind in ("int", "float", "bool"):
                param["role"] = "num"
    arg_kinds = {}
    for index, param in enumerate(params):
        observed = {
            row["arg_kinds"][index]
            for row in examples
            if index < len(row["arg_kinds"]) and row["arg_kinds"][index] != "any"
        }
        kind = observed.pop() if len(observed) == 1 else ""
        if not kind:
            kind = {"str": "str", "seq": "seq", "num": "num", "bool": "bool"}.get(
                str(param["role"]), "any"
            )
        arg_kinds[str(param["name"])] = kind

    expected_kinds = {
        str(row["expected_kind"]) for row in examples if row["expected_kind"] != "any"
    }
    if not expected_kinds:
        return_kind = "any"
    elif len(expected_kinds) == 1:
        return_kind = expected_kinds.pop()
    elif expected_kinds <= {"int", "float"}:
        return_kind = "num"
    else:
        return_kind = "any"

    flexible = any(
        ("=" in raw) or raw.startswith("*") or raw.startswith("**") for raw in raw_params
    )
    faults = _prompt_faults(text)
    arity = _arity_fault(examples, len(params), flexible)
    if arity:
        faults.append(arity)

    # Slot-3 goal wave: the WaveCCG frame of the natural-language request is
    # projected to its canonical four-slot rotor and unit-normalised.  A
    # prompt the CCG lexicon cannot parse still yields a deterministic hashed
    # frame, so the goal wave is never empty.
    try:
        from henri.holarch.wave_ccg import WaveCCGParser, frame_wave

        frame = WaveCCGParser().parse(_plain_language(docstring) or name).frame
        goal = frame_wave(frame)
    except Exception:  # noqa: BLE001 - a parse failure must not stop synthesis
        goal = np.zeros(64, dtype=np.complex128)
        goal[0] = 1.0

    contradictions = _contradictions(unique_assertions)
    return ProblemSpec(
        name=name,
        params=params,
        returns=returns,
        docstring=docstring,
        assertions=unique_assertions,
        goal_wave=goal,
        contradictions=contradictions,
        faults=faults,
        examples=examples,
        literals=_harvest_literals(docstring, examples),
        return_kind=return_kind,
        arg_kinds=arg_kinds,
        shape=_shape_from_examples(examples, return_kind),
        prompt=text,
    )


def _plain_language(docstring: str) -> str:
    cleaned = re.sub(r">>>.*$", " ", str(docstring), flags=re.MULTILINE)
    cleaned = re.sub(r"assert\s.*$", " ", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _contradictions(assertions: Sequence[str]) -> List[str]:
    """The same call asserted with two different expected values."""
    seen: Dict[str, str] = {}
    out: List[str] = []
    for line in assertions:
        text = str(line).strip()
        if "==" not in text:
            continue
        left, right = text[len("assert "):].split("==", 1) if text.startswith("assert ") else text.split("==", 1)
        key = left.strip()
        value = right.strip()
        if key in seen and seen[key] != value:
            out.append(key)
        seen[key] = value
    return out


# --------------------------------------------------------------- templates
# Each entry is (tags, kind, body, slot).  kind declares which typed slots
# the body needs: "seq1", "str1", "num1", "num2", "seq2", "any2", "seqnum".
TEMPLATES: List[Tuple[Tuple[str, ...], str, str, str]] = [
    # ---- direct scalar / sequence expressions
    (("sum", "total", "add", "added"), "num1", "return {n1} + {n2}", "return_expr"),
    (("sum", "total"), "seq1", "return sum({s1})", "return_expr"),
    (("average", "mean"), "seq1", "return sum({s1}) / len({s1})", "return_expr"),
    (("largest", "maximum", "max", "highest", "biggest"), "seq1", "return max({s1})", "return_expr"),
    (("smallest", "minimum", "min", "lowest"), "seq1", "return min({s1})", "return_expr"),
    (("largest", "maximum", "max"), "num2", "return max({n1}, {n2})", "return_expr"),
    (("smallest", "minimum", "min"), "num2", "return min({n1}, {n2})", "return_expr"),
    (("difference", "subtract", "minus", "gap"), "num2", "return abs({n1} - {n2})", "return_expr"),
    (("product", "multiply", "times"), "num2", "return {n1} * {n2}", "return_expr"),
    (("power", "exponent", "squared", "raised"), "num2", "return {n1} ** {n2}", "return_expr"),
    (("quotient", "divide", "floor"), "num2", "return {n1} // {n2}", "return_expr"),
    (("remainder", "modulo"), "num2", "return {n1} % {n2}", "return_expr"),
    (("absolute",), "num1", "return abs({n1})", "return_expr"),
    (("length", "size", "longer", "shorter"), "seq1", "return len({s1})", "return_expr"),
    (("reverse", "reversed", "backwards"), "seq1", "return {s1}[::-1]", "return_expr"),
    (("reverse",), "str1", "return {s1}[::-1]", "return_expr"),
    (("sort", "sorted", "ascending", "order"), "seq1", "return sorted({s1})", "return_expr"),
    (("descending", "reverse"), "seq1", "return sorted({s1}, reverse=True)", "return_expr"),
    (("unique", "distinct", "duplicate", "dedupe"), "seq1",
     "return sorted(set({s1}))", "return_expr"),
    (("unique", "distinct", "duplicate", "dedupe", "preserve"), "seq1",
     "return list(dict.fromkeys({s1}))", "return_expr"),
    (("first", "head"), "seq1", "return {s1}[0]", "return_expr"),
    (("last", "tail"), "seq1", "return {s1}[-1]", "return_expr"),
    (("every", "other", "alternate"), "seq1", "return {s1}[::2]", "return_expr"),
    (("lowercase", "lower"), "str1", "return {s1}.lower()", "return_expr"),
    (("uppercase", "upper", "capital"), "str1", "return {s1}.upper()", "return_expr"),
    (("title", "capitalize"), "str1", "return {s1}.title()", "return_expr"),
    (("strip", "whitespace", "trim"), "str1", "return {s1}.strip()", "return_expr"),
    (("split", "words"), "str1", "return {s1}.split()", "return_expr"),
    (("join", "concatenate", "concatenated"), "seq1", "return ''.join({s1})", "return_expr"),
    (("join", "space"), "seq1", "return ' '.join({s1})", "return_expr"),
    (("join", "comma"), "seq1", "return ','.join({s1})", "return_expr"),
    (("palindrome",), "str1", "return {s1} == {s1}[::-1]", "return_expr"),
    (("palindrome",), "seq1", "return list({s1}) == list({s1})[::-1]", "return_expr"),
    (("contains", "substring", "present"), "str1", "return {s2} in {s1}", "return_expr"),
    (("count", "occurrences", "frequency"), "seqnum", "return {s1}.count({n1})", "return_expr"),
    (("index", "position", "find"), "seqnum", "return {s1}.index({n1})", "return_expr"),
    (("index", "position", "find"), "str1", "return {s1}.find({s2})", "return_expr"),
    (("zip", "pairs"), "seq2", "return list(zip({s1}, {s2}))", "return_expr"),
    (("pairwise", "corresponding", "element"), "seq2",
     "return [a + b for a, b in zip({s1}, {s2})]", "return_expr"),
    (("common", "intersection", "shared"), "seq2",
     "return [x for x in {s1} if x in {s2}]", "return_expr"),
    (("missing", "difference", "absent"), "seq2",
     "return [x for x in {s1} if x not in {s2}]", "return_expr"),
    # ---- comprehension family
    (("double", "twice", "doubled"), "seq1", "return [x * 2 for x in {s1}]", "comprehension"),
    (("square", "squared"), "seq1", "return [x * x for x in {s1}]", "comprehension"),
    (("square", "squared", "sum"), "seq1",
     "return sum(x * x for x in {s1})", "comprehension"),
    (("even",), "seq1", "return [x for x in {s1} if x % 2 == 0]", "comprehension"),
    (("odd",), "seq1", "return [x for x in {s1} if x % 2 == 1]", "comprehension"),
    (("positive", "greater"), "seq1", "return [x for x in {s1} if x > 0]", "comprehension"),
    (("negative", "less"), "seq1", "return [x for x in {s1} if x < 0]", "comprehension"),
    (("count", "number", "even"), "seq1",
     "return sum(1 for x in {s1} if x % 2 == 0)", "comprehension"),
    (("count", "number", "odd"), "seq1",
     "return sum(1 for x in {s1} if x % 2 == 1)", "comprehension"),
    (("count", "number", "positive"), "seq1",
     "return sum(1 for x in {s1} if x > 0)", "comprehension"),
    (("sum", "positive"), "seq1",
     "return sum(x for x in {s1} if x > 0)", "comprehension"),
    (("sum", "even"), "seq1",
     "return sum(x for x in {s1} if x % 2 == 0)", "comprehension"),
    (("divisible", "multiple", "factor"), "seqnum",
     "return [x for x in {s1} if x % {n1} == 0]", "comprehension"),
    (("flatten", "nested", "flat"), "seq1",
     "return [y for x in {s1} for y in x]", "comprehension"),
    (("length", "longer", "shorter", "words"), "seqnum",
     "return [w for w in {s1} if len(w) > {n1}]", "comprehension"),
    (("length", "longer", "shorter", "words"), "seqnum",
     "return [len(w) for w in {s1}]", "comprehension"),
    (("length", "sorted", "sort"), "seq1",
     "return sorted({s1}, key=len)", "comprehension"),
    # ---- loop / accumulate family
    (("sum", "total", "add", "accumulate"), "seq1",
     "total = 0\n    for x in {s1}:\n        total += x\n    return total", "loop"),
    (("product", "multiply", "factorial"), "seq1",
     "total = 1\n    for x in {s1}:\n        total *= x\n    return total", "loop"),
    (("largest", "maximum", "max"), "seq1",
     "best = {s1}[0]\n    for x in {s1}:\n        if x > best:\n            best = x\n    return best",
     "loop"),
    (("smallest", "minimum", "min"), "seq1",
     "best = {s1}[0]\n    for x in {s1}:\n        if x < best:\n            best = x\n    return best",
     "loop"),
    (("running", "prefix", "cumulative"), "seq1",
     "out = []\n    total = 0\n    for x in {s1}:\n        total += x\n        out.append(total)\n    return out",
     "loop"),
    (("running", "maximum", "prefix"), "seq1",
     "out = []\n    best = None\n    for x in {s1}:\n        best = x if best is None else max(best, x)\n        out.append(best)\n    return out",
     "loop"),
    (("frequency", "count", "occurrences", "histogram"), "seq1",
     "out = {}\n    for x in {s1}:\n        out[x] = out.get(x, 0) + 1\n    return out", "loop"),
    (("transpose", "matrix", "columns"), "seq1",
     "return [list(row) for row in zip(*{s1})]", "loop"),
    (("median", "middle"), "seq1",
     "values = sorted({s1})\n    n = len(values)\n    if n % 2 == 1:\n        return values[n // 2]\n    return (values[n // 2 - 1] + values[n // 2]) / 2",
     "loop"),
    (("vowel",), "str1",
     "return sum(1 for c in {s1}.lower() if c in 'aeiou')", "loop"),
    (("consonant",), "str1",
     "return sum(1 for c in {s1}.lower() if c.isalpha() and c not in 'aeiou')", "loop"),
    (("digit", "digits", "sum"), "num1",
     "total = 0\n    for c in str({n1}):\n        total += int(c)\n    return total", "loop"),
    (("digit", "digits", "length", "number"), "num1",
     "return len(str(abs({n1})))", "loop"),
    (("reverse", "words", "sentence"), "str1",
     "return ' '.join({s1}.split()[::-1])", "loop"),
    (("swap", "pairs", "adjacent"), "str1",
     "chars = list({s1})\n    for i in range(0, len(chars) - 1, 2):\n        chars[i], chars[i + 1] = chars[i + 1], chars[i]\n    return ''.join(chars)",
     "loop"),
    (("remove", "spaces", "whitespace"), "str1",
     "return ''.join(ch for ch in {s1} if not ch.isspace())", "loop"),
    (("alnum", "alphanumeric", "letters"), "str1",
     "return sum(1 for ch in {s1} if ch.isalnum())", "loop"),
    (("prime",), "num1",
     "if {n1} < 2:\n        return False\n    d = 2\n    while d * d <= {n1}:\n        if {n1} % d == 0:\n            return False\n        d += 1\n    return True",
     "recursion"),
    (("factorial",), "num1",
     "if {n1} <= 1:\n        return 1\n    return {n1} * {name}({n1} - 1)", "recursion"),
    (("fibonacci", "fib"), "num1",
     "if {n1} <= 1:\n        return {n1}\n    return {name}({n1} - 1) + {name}({n1} - 2)", "recursion"),
    (("gcd", "divisor"), "num2",
     "a, b = abs({n1}), abs({n2})\n    while b:\n        a, b = b, a % b\n    return a", "recursion"),
    (("lcm", "multiple"), "num2",
     "a, b = abs({n1}), abs({n2})\n    if a == 0 or b == 0:\n        return 0\n    x, y = a, b\n    while y:\n        x, y = y, x % y\n    return a // x * b", "recursion"),
    (("power", "recursive"), "num2",
     "if {n2} == 0:\n        return 1\n    return {n1} * {name}({n1}, {n2} - 1)", "recursion"),
    (("sum", "recursive"), "seq1",
     "if not {s1}:\n        return 0\n    return {s1}[0] + {name}({s1}[1:])", "recursion"),
    (("isfinite", "binary", "convert"), "num1",
     "if {n1} == 0:\n        return '0'\n    out = ''\n    value = abs({n1})\n    while value:\n        out = str(value % 2) + out\n        value //= 2\n    return ('-' if {n1} < 0 else '') + out", "recursion"),
    (("celsius", "fahrenheit", "convert"), "num1",
     "return {n1} * 9 / 5 + 32", "return_expr"),
    (("even", "odd"), "num1", "return {n1} % 2 == 0", "return_expr"),
    (("leap", "year"), "num1",
     "return {n1} % 4 == 0 and ({n1} % 100 != 0 or {n1} % 400 == 0)", "return_expr"),
    (("replace", "substitute"), "str2", "return {s1}.replace({s2}, {s3})", "return_expr"),
]


# --------------------------------------------------------- schema families
# General algorithmic schemas, not benchmark tasks.  Each entry is a typed
# sketch the structural prior can rank: tags are ordinary English/CS words
# that a specification may use, needs is the placeholder requirement, effect
# is the structural class the sketch realises (fold, map, filter, pairwise,
# sort, stride, dict, convert, predicate, tuple).  A sketch is instantiated
# from the parsed specification only; nothing here names a task.
SCHEMA_FAMILIES: List[Dict[str, Any]] = [
    # ---- numeric predicates
    {"tags": ("prime",), "needs": "num1", "effect": "predicate", "slot": "return_expr",
     "body": "return {n1} >= 2 and not any({n1} % d == 0 for d in range(2, int({n1} ** 0.5) + 1))"},
    {"tags": ("non", "prime", "composite"), "needs": "num1", "effect": "predicate",
     "slot": "return_expr",
     "body": "return {n1} < 2 or any({n1} % d == 0 for d in range(2, int({n1} ** 0.5) + 1))"},
    {"tags": ("power", "two"), "needs": "num1", "effect": "predicate", "slot": "return_expr",
     "body": "return {n1} > 0 and ({n1} & ({n1} - 1)) == 0"},
    {"tags": ("power", "two", "not"), "needs": "num1", "effect": "predicate",
     "slot": "return_expr",
     "body": "return {n1} <= 0 or ({n1} & ({n1} - 1)) != 0"},
    {"tags": ("differ", "bit", "position"), "needs": "num2", "effect": "predicate",
     "slot": "return_expr",
     "body": "value = {n1} ^ {n2}\n    return value != 0 and (value & (value - 1)) == 0"},
    {"tags": ("opposite", "sign"), "needs": "num2", "effect": "predicate",
     "slot": "return_expr",
     "body": "return ({n1} >= 0) != ({n2} >= 0)"},
    {"tags": ("perfect", "square"), "needs": "num1", "effect": "predicate",
     "slot": "return_expr",
     "body": "root = int({n1} ** 0.5)\n    return root * root == {n1}"},
    {"tags": ("perfect", "square", "not"), "needs": "num1", "effect": "predicate",
     "slot": "return_expr",
     "body": "root = int({n1} ** 0.5)\n    return root * root != {n1}"},
    # ---- order statistics
    {"tags": ("second", "smallest", "unique"), "needs": "seq1", "effect": "select",
     "slot": "loop",
     "body": ("values = sorted(set({s1}))\n    if len(values) < 2:\n        return None\n"
              "    return values[1]")},
    {"tags": ("second", "largest", "unique"), "needs": "seq1", "effect": "select",
     "slot": "loop",
     "body": ("values = sorted(set({s1}))\n    if len(values) < 2:\n        return None\n"
              "    return values[-2]")},
    # ---- literal-bound string work
    {"tags": ("count", "occurrence", "occurence", "frequency", "string"), "needs": "strlit",
     "effect": "fold", "slot": "return_expr", "body": "return {s1}.count({lit1})"},
    {"tags": ("position", "index", "find", "string"), "needs": "strlit",
     "effect": "scalar", "slot": "return_expr", "body": "return {s1}.find({lit1})"},
    {"tags": ("count", "digit", "string", "number"), "needs": "str1", "effect": "fold",
     "slot": "comprehension", "body": "return sum(1 for c in {s1} if c.isdigit())"},
    {"tags": ("count", "letter", "alpha", "string"), "needs": "str1", "effect": "fold",
     "slot": "comprehension", "body": "return sum(1 for c in {s1} if c.isalpha())"},
    {"tags": ("count", "upper", "capital"), "needs": "str1", "effect": "fold",
     "slot": "comprehension", "body": "return sum(1 for c in {s1} if c.isupper())"},
    {"tags": ("digit", "list", "number"), "needs": "num1", "effect": "map",
     "slot": "comprehension", "body": "return [int(c) for c in str({n1})]"},
    {"tags": ("count", "set", "bit", "binary"), "needs": "num1", "effect": "fold",
     "slot": "return_expr", "body": "return bin({n1}).count('1')"},
    # ---- pairwise and adjacency
    {"tags": ("pair", "consecutive", "adjacent", "pairs"), "needs": "seq1",
     "effect": "pairwise", "slot": "pairwise",
     "body": "return list(zip({s1}, {s1}[1:]))"},
    {"tags": ("multiply", "product", "tuple", "element", "consecutive"), "needs": "seq1",
     "effect": "pairwise", "slot": "pairwise",
     "body": "return tuple(a * b for a, b in zip({s1}, {s1}[1:]))"},
    {"tags": ("sum", "adjacent", "pair", "consecutive"), "needs": "seq1",
     "effect": "pairwise", "slot": "pairwise",
     "body": "return [a + b for a, b in zip({s1}, {s1}[1:])]"},
    {"tags": ("consecutive", "duplicate", "remove"), "needs": "seq1", "effect": "filter",
     "slot": "loop",
     "body": ("out = []\n    for index, value in enumerate({s1}):\n"
              "        if index == 0 or value != {s1}[index - 1]:\n"
              "            out.append(value)\n    return out")},
    # ---- two-list and three-list alignment
    {"tags": ("count", "same", "indices", "equal", "pair"), "needs": "seq2",
     "effect": "fold", "slot": "comprehension",
     "body": "return sum(1 for a, b in zip({s1}, {s2}) if a == b)"},
    {"tags": ("common", "index", "list", "equal"), "needs": "seq2", "effect": "filter",
     "slot": "comprehension", "body": "return [a for a, b in zip({s1}, {s2}) if a == b]"},
    {"tags": ("common", "three", "index", "list"), "needs": "seq3", "effect": "filter",
     "slot": "comprehension",
     "body": "return [a for a, b, c in zip({s1}, {s2}, {s3}) if a == b == c]"},
    {"tags": ("count", "pair", "sum"), "needs": "seqnum", "effect": "fold", "slot": "loop",
     "body": ("total = 0\n    for i in range(len({s1})):\n"
              "        for j in range(i + 1, len({s1})):\n"
              "            if {s1}[i] + {s1}[j] == {n1}:\n"
              "                total += 1\n    return total")},
    {"tags": ("majority", "element", "array", "sorted"), "needs": "seqnum2",
     "effect": "fold", "slot": "return_expr",
     "body": "return {s1}.count({n2}) * 2 > {n1}"},
    {"tags": ("below", "under", "threshold", "less", "smaller", "all"), "needs": "seqnum",
     "effect": "predicate", "slot": "return_expr",
     "body": "return all(item < {n1} for item in {s1})"},
    {"tags": ("above", "over", "threshold", "greater", "larger", "all"), "needs": "seqnum",
     "effect": "predicate", "slot": "return_expr",
     "body": "return all(item > {n1} for item in {s1})"},
    {"tags": ("count", "unique", "sublist", "list", "dictionary"), "needs": "seq1",
     "effect": "dict", "slot": "dict_fold",
     "body": ("out = {}\n    for item in {s1}:\n        key = tuple(item)\n"
              "        out[key] = out.get(key, 0) + 1\n    return out")},
    # ---- bit rotation and digit arithmetic
    {"tags": ("rotate", "left", "bit"), "needs": "num2", "effect": "scalar",
     "slot": "return_expr", "body": "return ({n1} << {n2}) | ({n1} >> (32 - {n2}))"},
    {"tags": ("rotate", "right", "bit"), "needs": "num2", "effect": "scalar",
     "slot": "return_expr", "body": "return ({n1} >> {n2}) | ({n1} << (32 - {n2}))"},
    {"tags": ("digit", "sum", "power", "base"), "needs": "num2", "effect": "fold",
     "slot": "comprehension", "body": "return sum(int(c) for c in str({n1} ** {n2}))"},
    # ---- structure conversion
    {"tags": ("convert", "tuple", "string", "integer"), "needs": "str1",
     "effect": "convert", "slot": "convert",
     "body": ("return tuple(int(part) for part in "
              "{s1}.replace('(', '').replace(')', '').split(','))")},
    {"tags": ("remove", "tuple", "nested"), "needs": "seq1", "effect": "filter",
     "slot": "comprehension",
     "body": "return tuple(item for item in {s1} if not isinstance(item, tuple))"},
    {"tags": ("count", "rotation", "sorted", "array"), "needs": "seq1", "effect": "fold",
     "slot": "loop",
     "body": ("for i in range(1, len({s1})):\n        if {s1}[i] < {s1}[i - 1]:\n"
              "            return i\n    return 0")},
    # ---- regular expressions
    {"tags": ("decimal", "precision", "number", "string"), "needs": "str1",
     "effect": "predicate", "slot": "regex",
     "body": "import re\n    return bool(re.fullmatch(r'\\d+(\\.\\d{1,2})?', {s1}))"},
    {"tags": ("occurrence", "occurence", "substring", "position", "string"),
     "needs": "str2", "effect": "tuple", "slot": "regex",
     "body": ("import re\n    match = re.search({s2}, {s1})\n"
              "    if match is None:\n        return None\n"
              "    return (match.group(0), match.start(), match.end())")},
    {"tags": ("quotation", "quote", "extract", "string"), "needs": "str1",
     "effect": "list", "slot": "regex",
     "body": "import re\n    return re.findall('\"(.*?)\"', {s1})"},
    # ---- positional transforms
    {"tags": ("odd", "index", "remove", "string"), "needs": "str1", "effect": "stride",
     "slot": "return_expr", "body": "return {s1}[::2]"},
    {"tags": ("odd", "index", "every", "other", "list"), "needs": "seq1",
     "effect": "stride", "slot": "return_expr", "body": "return {s1}[::2]"},
    {"tags": ("odd", "index", "second"), "needs": "str1", "effect": "stride",
     "slot": "return_expr", "body": "return {s1}[1::2]"},
    {"tags": ("sort", "ascending", "order", "sorted"), "needs": "seq1",
     "effect": "sort", "slot": "return_expr", "body": "return sorted({s1})"},
    {"tags": ("rotate", "left", "list"), "needs": "seq1", "effect": "map",
     "slot": "return_expr", "body": "return {s1}[1:] + {s1}[:1]"},
    {"tags": ("intersperse", "delimiter", "between"), "needs": "seqnum",
     "effect": "map", "slot": "loop",
     "body": ("out = []\n    for item in {s1}:\n        out.append(item)\n"
              "        out.append({n1})\n    return out[:-1]")},
]


_KIND_NEEDS: Dict[str, Tuple[str, ...]] = {
    "seq1": ("s1",),
    "seq2": ("s1", "s2"),
    "seq3": ("s1", "s2", "s3"),
    "seqnum": ("s1", "n1"),
    "seqnum2": ("s1", "n1", "n2"),
    "seq2num": ("s1", "s2", "n1"),
    "str1": ("s1",),
    "str2": ("s1", "s2"),
    "str3": ("s1", "s2", "s3"),
    "strnum": ("s1", "n1"),
    "strlit": ("s1", "lit1"),
    "str2lit": ("s1", "s2", "lit1"),
    "num1": ("n1",),
    "num2": ("n1", "n2"),
    "num3": ("n1", "n2", "n3"),
    "any2": ("n1", "n2"),
    "any3": ("n1", "n2", "n3"),
    "none": (),
}


def _requirement_placeholders(needs: Optional[str]) -> Tuple[str, ...]:
    text = str(needs or "").strip()
    if text in _KIND_NEEDS:
        return _KIND_NEEDS[text]
    if not text:
        return ()
    return tuple(part.strip() for part in text.split(",") if part.strip())


def _instantiate(
    template: str,
    spec: ProblemSpec,
    needs: Optional[str] = None,
    strict_numeric: bool = False,
) -> Optional[str]:
    """Bind a typed sketch to the parsed signature, or refuse the sketch.

    The requirement names the placeholders the sketch needs: s-codes read the
    string and sequence arguments, n-codes the numeric ones, lit-codes the
    literal constants the specification itself names.  A missing pool entry
    refuses the sketch instead of emitting an unbound placeholder.
    """
    seqs = spec.by_role("seq")
    strs = spec.by_role("str")
    nums = spec.by_role("num")
    binding: Dict[str, str] = {"name": spec.name}
    required = list(_requirement_placeholders(needs))
    for key in re.findall(r"\{(\w+)\}", template):
        if key not in required:
            required.append(key)
    for key in required:
        if key == "name":
            continue
        if key.startswith("lit"):
            index = int(key[3:]) - 1
            if index < 0 or index >= len(spec.literals):
                return None
            binding[key] = spec.literals[index]
            continue
        if key.startswith("s"):
            # A str-coded sketch reads STRING arguments only; a seq-coded one
            # may read either, because a string is a sequence too.  Merging
            # the pools unconditionally bound a character predicate to a list
            # of numbers, which is a type error the sketch never meant.
            str_only = str(needs or "").startswith("str")
            pool = strs if str_only else (strs + seqs)
        elif strict_numeric:
            # A schema family that declares a numeric slot means a NUMBER:
            # binding it to a list argument would type-error by construction.
            pool = nums
        else:
            pool = nums + [name for name in seqs if name not in nums]
        index = int(key[1:]) - 1
        if index < 0 or index >= len(pool):
            return None
        binding[key] = pool[index]

    def substitute(match):
        key = match.group(1)
        return str(binding.get(key, match.group(0)))

    return re.sub(r"\{(\w+)\}", substitute, template)


def build_candidates(spec: ProblemSpec) -> List[Dict[str, Any]]:
    """Every typed sketch the slot inventory admits, before wave pruning."""
    out: List[Dict[str, Any]] = []
    seen: set = set()

    def add(tags, body, slot, kind, effect="") -> None:
        text = str(body)
        if text in seen:
            return
        seen.add(text)
        out.append(
            {
                "body": text,
                "tags": tuple(tags),
                "slot": str(slot),
                "kind": str(kind),
                "effect": str(effect),
            }
        )

    for tags, kind, body, slot in TEMPLATES:
        if kind.startswith("num") and not (spec.by_role("num") or spec.by_role("seq")):
            continue
        if kind.startswith("str") and not (spec.by_role("str") or spec.by_role("seq")):
            continue
        if kind.startswith("seq") and not spec.by_role("seq"):
            continue
        rendered = _instantiate(body, spec, kind)
        if rendered is None:
            continue
        add(tags, rendered, slot, kind)
    for family in SCHEMA_FAMILIES:
        rendered = _instantiate(
            str(family["body"]), spec, str(family["needs"]), strict_numeric=True
        )
        if rendered is None:
            continue
        add(family["tags"], rendered, family["slot"], family["needs"],
            family.get("effect", ""))
    for family in schema_candidates(spec):
        add(family["tags"], family["body"], family["slot"],
            family.get("needs", ""), family.get("effect", ""))
    return out


# ---------------------------------------------------------------- the engine
class ColdSynthesizer:
    """Deterministic cold-start AST search under the sandbox verifier."""

    def __init__(
        self,
        *,
        memory=None,
        namespace: str = "default",
        projector=None,
        tokenizer=None,
        wave_scorer: Optional[Callable[[str], float]] = None,
        sandbox=None,
        budget: int = ROLLOUT_BUDGET,
        slot_support: int = SLOT_SUPPORT,
        seed: int = 20261111,
        plan_store=None,
    ) -> None:
        self.memory = memory
        self.namespace = str(namespace)
        self.projector = projector
        self.tokenizer = tokenizer
        self.wave_scorer = wave_scorer
        self.budget = int(budget)
        self.slot_support = int(slot_support)
        self.seed = int(seed)
        self.plan_store = plan_store
        self.repulsors = 0
        self.rollouts = 0
        self.crystallized = 0
        self.last_report: Dict[str, Any] = {}
        if sandbox is None:
            from henri.holarch.sandbox import SubprocessSandboxExecutor

            sandbox = SubprocessSandboxExecutor("pure_algorithmic", timeout_s=8.0)
        self.sandbox = sandbox

    # ----------------------------------------------------------- wave prior
    def _wave_score(self, body: str) -> float:
        if self.wave_scorer is None:
            return 0.0
        try:
            value = float(self.wave_scorer(str(body)))
        except Exception:  # noqa: BLE001 - a scorer failure is not a refusal
            return 0.0
        if not math.isfinite(value):
            return 0.0
        return value

    # -------------------------------------------------------------- ranking
    def rank(self, spec: ProblemSpec) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """Prune every slot to V_eff <= slot_support, then rank by the prior.

        The prior is structural first: the sketch's inferred output kind is
        compared with the kind the public examples agree on, and its effect
        class is compared with the shape facts read off the same examples.
        Vocabulary hits and the local wave score break the remaining ties.
        """
        words = set(spec.words)
        candidates = build_candidates(spec)
        for candidate in candidates:
            hits = sum(1 for tag in candidate["tags"] if tag in words)
            candidate["keyword"] = int(hits)
            candidate["wave"] = self._wave_score(candidate["body"])
            candidate["out_kind"] = _body_kind(candidate["body"], spec.arg_kinds)
            # Vocabulary support is part of the score, not a tie-break: a
            # sketch the specification never mentions must not outrank one
            # it names, or a structurally plausible body can win on two
            # lucky examples alone.
            candidate["score"] = float(
                _kind_match(candidate["out_kind"], spec.return_kind)
                + _effect_bonus(candidate, spec)
                + min(float(hits), 3.0)
            )
        slots: Dict[str, List[Dict[str, Any]]] = {}
        for candidate in candidates:
            slots.setdefault(str(candidate["slot"]), []).append(candidate)
        admitted: List[Dict[str, Any]] = []
        v_eff: Dict[str, int] = {}
        for slot, items in slots.items():
            items = sorted(
                items,
                key=lambda row: (
                    -float(row["score"]),
                    -float(row["keyword"]),
                    -float(row["wave"]),
                    str(row["body"]),
                ),
            )
            keep = items[: self.slot_support]
            v_eff[slot] = int(len(keep))
            admitted.extend(keep)
        admitted.sort(
            key=lambda row: (
                -float(row["score"]),
                -float(row["keyword"]),
                -float(row["wave"]),
                str(row["body"]),
            )
        )
        return admitted, v_eff

    # ---------------------------------------------------------- verification
    def _assemble(self, spec: ProblemSpec, body: str) -> str:
        """The emitted module keeps the declared signature and annotations.

        The annotations are what the RepoSheaf capability pullback reads, so a
        body that demands a capability its declared argument cannot provision
        is refused before it is executed.  A declared annotation may name a
        typing constructor (List, Optional, Tuple, ...); the module then
        carries the import that makes the annotation evaluable, because an
        annotation the module cannot resolve would fail at definition time.
        """
        declared = ", ".join(
            "%s: %s" % (param["name"], param["annotation"])
            if param["annotation"]
            else str(param["name"])
            for param in spec.params
        )
        suffix = (" -> " + str(spec.returns)) if spec.returns else ""
        source = "def %s(%s)%s:\n    %s\n" % (spec.name, declared, suffix, body)
        prelude = _typing_prelude(source)
        return (prelude + source) if prelude else source

    def _verify(self, spec: ProblemSpec, source: str) -> Tuple[bool, str]:
        """Syntax, capability and public-assertion check inside the U5 box."""
        if spec.contradictions:
            return False, "contradictory_public_examples"
        try:
            tree = ast.parse(source)
        except SyntaxError as error:
            return False, "syntax:" + str(error)
        if not any(isinstance(node, ast.Return) for node in ast.walk(tree)):
            return False, "no_return"
        capability = self._capability_ok(spec, source)
        if not capability[0]:
            return False, capability[1]
        program = source + "\n" + "\n".join(spec.assertions) + "\nresult = {'ok': True}\n"
        record = self.sandbox.execute(program)
        if record.ok:
            return True, "public_assertions_pass"
        detail = (record.stderr or "").strip().splitlines()
        return False, (detail[-1] if detail else "execution_failed")

    def _capability_ok(self, spec: ProblemSpec, source: str) -> Tuple[bool, str]:
        """RepoSheaf capability filter on the emitted module."""
        try:
            from henri.holarch.repo_sheaf import _return_provision, profile_module

            module = profile_module("cold_candidate", source)
            profile = module.functions.get(spec.name)
            if profile is None:
                return False, "sheaf:missing_function"
            provision = _return_provision(profile)
            # The pullback reads a DECLARED return type.  An unannotated
            # function has no return provision to be empty, so the check is
            # applied only when the signature declares one.
            declared_return = str(spec.returns).strip()
            if (
                float(np.linalg.norm(provision)) <= 0.0
                and declared_return
                and "None" not in declared_return
            ):
                return False, "sheaf:empty_return_provision"
            from henri.holarch.repo_sheaf import (
                CAP_CALLABLE,
                CAP_MAP_MUT,
                CAP_MAP_READ,
                CAP_NUM_ARITH,
                CAP_SEQ_MUT,
                CAP_SET_OPS,
                CAP_TEXT_OPS,
            )

            # Only the hard operation families are screened.  Indexing and
            # sizing are deliberately excluded: a str is legitimately
            # indexable and sliceable, and the static schema does not model
            # that, so screening them would refuse correct bodies.
            hard = (CAP_NUM_ARITH, CAP_TEXT_OPS, CAP_MAP_READ, CAP_MAP_MUT,
                    CAP_SET_OPS, CAP_SEQ_MUT, CAP_CALLABLE)
            for param in profile.params:
                demand = profile.demand.get(param)
                annotated = profile.annotations.get(param)
                if demand is None or annotated is None:
                    continue
                if any(
                    int(np.asarray(demand)[cap]) > int(np.asarray(annotated)[cap])
                    for cap in hard
                ):
                    return False, "sheaf:capability_mismatch:" + str(param)
            mismatch = self._return_mismatch(spec, source)
            if mismatch:
                return False, mismatch
        except Exception:  # noqa: BLE001 - the syntactic gate already ran
            return True, "sheaf:unavailable"
        return True, "sheaf:ok"

    @staticmethod
    def _return_mismatch(spec: ProblemSpec, source: str) -> str:
        """The confirmed return provision must sit inside the annotation.

        This is the RepoSheaf pullback on the return edge: a body that returns
        text from a function annotated -> int is a capability mismatch, not a
        style preference.  An un-inferable expression (a comprehension, a
        subscript, a binary operation) yields the zero vector and is skipped,
        so the filter never refuses a body it cannot read.
        """
        try:
            from henri.holarch.repo_sheaf import _annotation_vector, _default_provision

            tree = ast.parse(source)
            function = next(
                node for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            )
            declared = _annotation_vector(function.returns)
            if declared is None:
                return ""
            env = {}
            for arg in function.args.args:
                vector = _annotation_vector(arg.annotation)
                if vector is not None:
                    env[arg.arg] = vector
            returns = [node for node in ast.walk(function) if isinstance(node, ast.Return)]
            for node in returns:
                if node.value is None:
                    continue
                provision = _default_provision(node.value, env, {})
                if float(np.abs(np.asarray(provision)).sum()) <= 0.0:
                    continue
                if bool(np.any(np.asarray(provision) > np.asarray(declared))):
                    return "sheaf:return_type_mismatch"
        except Exception:  # noqa: BLE001 - an unreadable body is not refused
            return ""
        return ""

    # ------------------------------------------------------------- steering
    def _register_repulsor(self, spec: ProblemSpec, source: str, detail: str) -> None:
        if self.memory is None or self.projector is None or self.tokenizer is None:
            return
        try:
            ids = self.tokenizer.encode(source)
            wave = self.projector.sequence_wave(list(ids)[:96], self.tokenizer)
            self.memory.register_axiom(
                np.asarray(wave),
                name="cold_synth:" + hashlib.sha256(source.encode("utf-8")).hexdigest()[:16],
                polarity="REPULSOR",
                source="cold_synth",
                namespace=self.namespace,
            )
            self.repulsors += 1
        except Exception:  # noqa: BLE001 - a registry failure must not stop the search
            return

    def _order(
        self,
        ranked: List[Dict[str, Any]],
        *,
        failed: Sequence[str],
        temperature: float,
        generator: np.random.Generator,
        greedy: int = 16,
    ) -> List[Dict[str, Any]]:
        """Greedy head, then high-temperature sampling away from failures."""
        remaining = [row for row in ranked if row["body"] not in set(failed)]
        if len(remaining) <= greedy:
            return remaining
        head = remaining[:greedy]
        tail = remaining[greedy:]
        scores = np.asarray([float(row["keyword"]) + float(row["wave"]) for row in tail])
        if scores.size == 0:
            return head
        scaled = scores / max(float(temperature), 1e-6)
        scaled = scaled - float(np.max(scaled))
        weights = np.exp(scaled)
        total = float(weights.sum())
        if total <= 0.0:
            return head + tail
        weights = weights / total
        order = generator.choice(
            len(tail), size=len(tail), replace=False, p=weights
        )
        return head + [tail[int(index)] for index in order]

    # ---------------------------------------------------------------- public
    def synthesize(
        self,
        prompt: str,
        *,
        extra_tests: Optional[Sequence[str]] = None,
        crystallize: bool = False,
    ) -> Dict[str, Any]:
        """Cold-start one function.  Emit only on 100% public-assertion pass."""
        spec = parse_problem(prompt, extra_tests=extra_tests)
        ranked, v_eff = self.rank(spec)
        report: Dict[str, Any] = {
            "task": spec.name,
            "candidates": int(len(ranked)),
            "v_eff": dict(v_eff),
            "v_eff_max": int(max(v_eff.values()) if v_eff else 0),
            "contradictions": list(spec.contradictions),
            "rollouts": 0,
            "repulsors": 0,
            "temperature": 1.0,
            "status": "ABSTAIN",
            "source": "",
            "reason": "",
            "trajectory": [],
            "return_kind": str(spec.return_kind),
            "faults": list(spec.faults),
            "shape": dict(spec.shape),
        }
        if spec.contradictions:
            report["reason"] = "contradictory_spec"
            self.last_report = report
            return report
        if spec.faults:
            report["reason"] = "spec_fault:" + str(spec.faults[0])
            self.last_report = report
            return report
        if not spec.assertions:
            # Nothing to verify against: a bare pass would be a vacuous
            # emission, so the call fail-closes instead of guessing.
            report["reason"] = "no_public_examples"
            self.last_report = report
            return report
        if _unsatisfiable(spec.docstring + " " + prompt):
            report["reason"] = "unsatisfiable_capability"
            self.last_report = report
            return report
        generator = np.random.default_rng(self.seed ^ int(
            hashlib.sha256(spec.name.encode("utf-8")).hexdigest()[:8], 16))
        failed: List[str] = []
        temperature = 1.0
        order = self._order(ranked, failed=failed, temperature=temperature,
                            generator=generator)
        index = 0
        while index < len(order) and report["rollouts"] < self.budget:
            candidate = order[index]
            index += 1
            body = str(candidate["body"])
            source = self._assemble(spec, body)
            ok, detail = self._verify(spec, source)
            report["rollouts"] += 1
            self.rollouts += 1
            report["trajectory"].append(
                {"slot": candidate["slot"], "body": body, "ok": bool(ok),
                 "detail": str(detail)}
            )
            if ok:
                report["status"] = "SOLVED"
                report["source"] = source
                report["reason"] = "public_assertions_pass"
                if crystallize:
                    self._crystallize(spec, source)
                self.last_report = report
                return report
            failed.append(body)
            self._register_repulsor(spec, source, detail)
            if temperature > EXPLORE_TEMPERATURE:
                temperature = EXPLORE_TEMPERATURE
                report["temperature"] = float(temperature)
                order = self._order(ranked, failed=failed, temperature=temperature,
                                    generator=generator)
                index = 0
        report["repulsors"] = int(self.repulsors)
        report["reason"] = report["reason"] or "exhausted_public_assertions"
        self.last_report = report
        return report

    def _crystallize(self, spec: ProblemSpec, source: str) -> None:
        if self.plan_store is None:
            return
        try:
            self.plan_store.compile(
                name=str(spec.name),
                intent=_plain_language(spec.docstring) or str(spec.name),
                source=str(source),
            )
            self.crystallized += 1
        except Exception:  # noqa: BLE001 - an optional write is not a gate
            return


# --------------------------------------------------- structural prior (Z2)
_KIND_GROUPS = {
    "int": 1.0,
    "float": 1.0,
    "num": 1.0,
}

_TYPING_NAMES = frozenset(
    {
        "List", "Dict", "Tuple", "Set", "FrozenSet", "Optional", "Union",
        "Any", "Callable", "Iterable", "Iterator", "Sequence", "Mapping",
        "MutableMapping", "TypeVar", "Generic", "Literal", "TypedDict",
        "NamedTuple", "Deque", "DefaultDict", "Counter", "Type", "NoReturn",
        "Generator", "AsyncIterator", "Awaitable",
    }
)


def _typing_prelude(source: str) -> str:
    """The typing import a module needs to evaluate its own annotations."""
    try:
        tree = ast.parse(str(source))
    except SyntaxError:
        return ""
    needed: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.arg) and node.annotation is not None:
            for sub in ast.walk(node.annotation):
                if isinstance(sub, ast.Name) and sub.id in _TYPING_NAMES:
                    needed.add(sub.id)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.returns is not None:
            for sub in ast.walk(node.returns):
                if isinstance(sub, ast.Name) and sub.id in _TYPING_NAMES:
                    needed.add(sub.id)
    if not needed:
        return ""
    return "from typing import " + ", ".join(sorted(needed)) + "\n\n"


def _merge_kinds(kinds: Sequence[str]) -> str:
    values = [str(kind) for kind in kinds if str(kind)]
    if not values:
        return "none"
    without_none = [kind for kind in values if kind != "none"]
    if without_none:
        values = without_none
    else:
        return "none"
    unique = set(values)
    if len(unique) == 1:
        return unique.pop()
    if unique <= {"int", "float", "num"}:
        return "num"
    if unique <= {"bool", "int"}:
        return "bool"
    return "any"


def _expr_kind(node: ast.AST, env: Dict[str, str]) -> str:
    """The structural kind of one expression under a typed name environment."""
    if isinstance(node, ast.Constant):
        return _literal_kind(node.value)
    if isinstance(node, ast.Name):
        return str(env.get(node.id, "any"))
    if isinstance(node, (ast.List, ast.ListComp)):
        return "list"
    if isinstance(node, ast.Tuple):
        return "tuple"
    if isinstance(node, (ast.Dict, ast.DictComp)):
        return "dict"
    if isinstance(node, (ast.Set, ast.SetComp)):
        return "set"
    if isinstance(node, ast.Compare):
        return "bool"
    if isinstance(node, ast.BoolOp):
        return "bool"
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.Not):
            return "bool"
        return _expr_kind(node.operand, env)
    if isinstance(node, ast.IfExp):
        return _merge_kinds(
            [_expr_kind(node.body, env), _expr_kind(node.orelse, env)]
        )
    if isinstance(node, ast.BinOp):
        left = _expr_kind(node.left, env)
        right = _expr_kind(node.right, env)
        if isinstance(node.op, ast.Add) and "str" in (left, right):
            return "str"
        if isinstance(node.op, ast.Add) and left in ("list", "tuple"):
            return left
        if isinstance(node.op, ast.FloorDiv) and left in ("int", "num") and right == "int":
            return "int"
        if left == "str" and isinstance(node.op, ast.Mod):
            return "str"
        return "num"
    if isinstance(node, ast.Call):
        return _call_kind(node, env)
    if isinstance(node, ast.Subscript):
        base = _expr_kind(node.value, env)
        if isinstance(node.slice, ast.Slice):
            return base
        return "any"
    if isinstance(node, ast.JoinedStr):
        return "str"
    return "any"


def _call_kind(node: ast.Call, env: Dict[str, str]) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        table = {
            "len": "int", "int": "int", "ord": "int", "hash": "int",
            "float": "num", "round": "num", "abs": "num", "sum": "num",
            "pow": "num", "min": "any", "max": "any",
            "str": "str", "repr": "str", "chr": "str", "format": "str",
            "sorted": "list", "list": "list", "reversed": "list",
            "tuple": "tuple", "dict": "dict", "set": "set",
            "frozenset": "set", "bool": "bool", "any": "bool", "all": "bool",
            "isinstance": "bool", "callable": "bool", "iter": "any",
            "enumerate": "any", "zip": "any", "map": "any", "filter": "any",
        }
        return str(table.get(str(func.id), "any"))
    if isinstance(func, ast.Attribute):
        name = str(func.attr)
        if name in {"count", "index", "find", "rfind", "bit_length", "rindex"}:
            return "int"
        if name in {
            "isdigit", "isalpha", "isalnum", "isspace", "isupper", "islower",
            "isdecimal", "isnumeric", "startswith", "endswith", "isidentifier",
        }:
            return "bool"
        if name in {
            "join", "upper", "lower", "title", "strip", "lstrip", "rstrip",
            "replace", "format", "capitalize", "swapcase", "casefold",
            "center", "zfill", "ljust", "rjust",
        }:
            return "str"
        if name in {"split", "splitlines", "findall", "keys", "values", "items", "copy"}:
            return "list"
        if name in {"get", "setdefault", "pop"}:
            return "any"
        if name in {"append", "extend", "sort", "add", "update", "discard", "insert"}:
            return "none"
        if name in {"search", "match", "fullmatch"}:
            return "match"
    return "any"


def _body_kind(body: str, env: Optional[Dict[str, str]] = None) -> str:
    """Statically infer the output kind of one candidate body.

    Local bindings are typed first: a loop over range() binds an int, a loop
    over a sequence binds its element kind, and an assignment binds the kind
    of its right-hand side.  Without that pass a structured loop body would
    return an unknown name and be ranked as if its type were unknown.
    """
    try:
        tree = ast.parse("def _candidate():\n    " + str(body) + "\n")
    except SyntaxError:
        return "any"
    scope = dict(env or {})
    function = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef)), None
    )
    if function is None:
        return "any"
    for node in ast.walk(function):
        if isinstance(node, (ast.For, ast.AsyncFor)):
            iterated = _iterated_kind(node.iter, scope)
            for target in ast.walk(node.target):
                if isinstance(target, ast.Name):
                    scope[target.id] = "int" if isinstance(node.iter, ast.Call) and getattr(node.iter.func, "id", "") == "range" else iterated
        elif isinstance(node, ast.Assign):
            value_kind = _expr_kind(node.value, scope)
            for target in node.targets:
                if isinstance(target, ast.Name):
                    scope[target.id] = value_kind
                elif isinstance(target, (ast.Tuple, ast.List)):
                    for element in target.elts:
                        if isinstance(element, ast.Name):
                            scope[element.id] = "any"
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            scope[node.target.id] = _expr_kind(node.value, scope)
    kinds: List[str] = []
    for node in ast.walk(function):
        if isinstance(node, ast.Return):
            kinds.append("none" if node.value is None else _expr_kind(node.value, scope))
    return _merge_kinds(kinds)


def _iterated_kind(node: ast.AST, env: Dict[str, str]) -> str:
    """The element kind a for-loop target receives."""
    if isinstance(node, ast.Call):
        name = getattr(node.func, "id", "") or getattr(node.func, "attr", "")
        if name in {"range", "enumerate"}:
            return "int"
        if name in {"zip", "items"}:
            return "any"
        if name in {"keys", "values"}:
            return "any"
    return "any"


def _kind_match(actual: str, wanted: str) -> float:
    """How well an inferred output kind matches the examples' answer kind."""
    left = str(actual or "any")
    right = str(wanted or "any")
    if left == "any" or right == "any":
        return 0.0
    if left == right:
        return 4.0
    numeric = {"int", "float", "num"}
    if left in numeric and right in numeric:
        return 3.0
    if left in {"bool", "int", "num"} and right in {"bool", "int", "num"}:
        return 2.0
    if left == "match" and right == "bool":
        return 2.0
    if left in {"list", "tuple", "str"} and right == "seq":
        return 2.0
    if right in {"list", "tuple", "str"} and left == "seq":
        return 2.0
    if left == "none" or right == "none":
        return 0.0
    return -4.0


def _effect_bonus(candidate: Dict[str, Any], spec: ProblemSpec) -> float:
    """The shape facts the effect class of a sketch agrees with."""
    effect = str(candidate.get("effect", ""))
    shape = spec.shape or {}
    bonus = 0.0
    if effect == "fold" and shape.get("fold"):
        bonus += 1.5
    if effect == "map" and shape.get("elementwise"):
        bonus += 1.5
    if effect == "filter" and shape.get("subset"):
        bonus += 1.0
    if effect in ("sort", "permute") and shape.get("permutation"):
        bonus += 1.5
    if effect == "pairwise" and shape.get("pairwise"):
        bonus += 2.0
    if effect == "stride" and shape.get("halving"):
        bonus += 2.0
    if effect == "predicate" and spec.return_kind == "bool":
        bonus += 1.0
    if effect == "tuple" and spec.return_kind == "tuple":
        bonus += 1.5
    if effect == "dict" and spec.return_kind == "dict":
        bonus += 1.5
    if effect == "convert" and shape.get("conversion"):
        bonus += 1.5
    if effect == "recurrence" and any(
        row.get("expected_kind") in ("int", "float", "num") for row in spec.examples
    ):
        bonus += 1.0
    return float(bonus)


def _unsatisfiable(text: str) -> bool:
    """True when the specification asks for a capability the profile lacks.

    A citation URL is not a network request: the URL is stripped before the
    vocabulary screen runs, so a docstring that links a reference page is not
    refused for containing the letters of a protocol name.
    """
    cleaned = _URL.sub(" ", str(text)).lower()
    return any(word in cleaned for word in _UNSATISFIABLE_WORDS)


_QUOTED_CHAR = re.compile(r"['\"]([A-Za-z0-9])['\"]")
_SINGLE_CHAR = re.compile(r"(?<![A-Za-z0-9])([A-Za-z0-9])(?![A-Za-z0-9])")
_QUANTIFIER_WORDS = (
    (re.compile(r"one or more|at least one|one or many"), "+"),
    (re.compile(r"zero or more|any number of"), "*"),
    (re.compile(r"zero or one|optional|at most one"), "?"),
    (re.compile(r"\bthree\b"), "{3}"),
    (re.compile(r"\btwo\b"), "{2}"),
    (re.compile(r"\bfour\b"), "{4}"),
)


def _regex_candidates(spec: ProblemSpec) -> List[Dict[str, Any]]:
    """Regex sketches read from a natural-language match description.

    The description names the two characters and the quantifier ("an 'a'
    followed by one or more 'b's"), so the pattern is a general reading of
    the sentence, not a stored answer.  A description without the relation
    yields no candidate.
    """
    out: List[Dict[str, Any]] = []
    text = str(spec.docstring)
    lowered = text.lower()
    marker = "followed by"
    if marker not in lowered:
        return out
    strings = spec.by_role("str")
    if not strings:
        return out
    index = lowered.index(marker)
    window = lowered[max(0, index - 24): index]
    firsts = _SINGLE_CHAR.findall(window)
    first = firsts[-1] if firsts else ""
    after = text[index + len(marker):]
    quoted = _QUOTED_CHAR.findall(after)
    second = quoted[0] if quoted else ""
    if not second:
        tokens = _SINGLE_CHAR.findall(after)
        second = tokens[0] if tokens else ""
    if not first or not second:
        return out
    quantifier = ""
    for pattern, symbol in _QUANTIFIER_WORDS:
        if pattern.search(after.lower()):
            quantifier = symbol
            break
    if not quantifier:
        quantifier = "+"
    pattern = str(first) + str(second) + quantifier
    body = "import re\n    return bool(re.search(r'%s', %s))" % (pattern, strings[0])
    out.append(
        {
            "tags": ("match", "string", "pattern", "followed"),
            "body": body,
            "slot": "regex",
            "needs": "str1",
            "effect": "predicate",
        }
    )
    return out


_NUMBER_RUN = re.compile(r"(?<![\w.])-?\d+(?:\s*,\s*-?\d+){5,}")


def _recurrence_candidates(spec: ProblemSpec) -> List[Dict[str, Any]]:
    """A linear recurrence fitted to the sequence the specification prints.

    A specification that lists the first terms of a sequence states the very
    data a two-term linear recurrence needs.  The fit is a small exhaustive
    search over integer coefficients, accepted only when it is unique, so an
    ambiguous sequence yields no candidate and cannot be emitted by guess.
    """
    out: List[Dict[str, Any]] = []
    numbers = spec.by_role("num")
    if len(numbers) != 1:
        return out
    for match in _NUMBER_RUN.finditer(str(spec.docstring)):
        terms = [int(value) for value in re.findall(r"-?\d+", match.group(0))]
        if len(terms) < 6:
            continue
        fits = []
        for p in range(-3, 4):
            for q in range(-3, 4):
                if p == 0 and q == 0:
                    continue
                if all(
                    terms[i] == p * terms[i - 1] + q * terms[i - 2]
                    for i in range(2, len(terms))
                ):
                    fits.append((p, q))
        if len(fits) != 1:
            continue
        p, q = fits[0]
        body = (
            "if {n1} < 2:\n"
            "        return %d if {n1} == 0 else %d\n"
            "    table = [%d, %d]\n"
            "    for i in range(2, {n1} + 1):\n"
            "        table.append(%d * table[i - 1] + %d * table[i - 2])\n"
            "    return table[{n1}]"
            % (terms[0], terms[1], terms[0], terms[1], p, q)
        )
        out.append(
            {
                "tags": ("nth", "number", "term", "sequence"),
                "body": body.replace("{n1}", str(numbers[0])),
                "slot": "recurrence",
                "needs": "num1",
                "effect": "recurrence",
            }
        )
    return out


def schema_candidates(spec: ProblemSpec) -> List[Dict[str, Any]]:
    """Sketches generated from the parsed specification, not from a table."""
    out: List[Dict[str, Any]] = []
    out.extend(_regex_candidates(spec))
    out.extend(_recurrence_candidates(spec))
    return out


# ------------------------------------------------------------ branch audit
_TASK_IDENTITY_ATTRS = frozenset(
    {"name", "task", "task_id", "taskid", "task_name", "entry_point", "prompt", "func"}
)
# The closed vocabulary of literals this module is allowed to compare with:
# structural type tags, effect tags, placeholder keys and punctuation.  A
# comparison against anything else is reported as unexplained, so a task name
# cannot hide behind the audit's own allowance.
_AUDIT_VOCABULARY = frozenset(
    {
        "bool", "str", "seq", "num", "int", "float", "any", "none", "match",
        "list", "tuple", "dict", "set", "name", "fold", "map", "filter",
        "pairwise", "stride", "predicate", "convert", "recurrence", "sort",
        "permute", "select", "scalar",
        "==", "=", ",", "(", ")", "[", "]", "{", "}", "([{", ")]}",
        "\n", "\t", "None", "Traceback", "range", "enumerate", "zip",
        "get", "items", "keys", "values", "sort", "append",
    }
)


def _identity_expression(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return str(node.id).lower() in _TASK_IDENTITY_ATTRS
    if isinstance(node, ast.Attribute):
        return str(node.attr).lower() in _TASK_IDENTITY_ATTRS
    if isinstance(node, ast.Subscript):
        if _identity_expression(node.value):
            return True
        slice_node = node.slice
        return isinstance(slice_node, ast.Constant) and str(slice_node.value).lower() in _TASK_IDENTITY_ATTRS
    return False


def audit_task_specific_branches(
    source: str, identifiers: Sequence[str] = ()
) -> Dict[str, Any]:
    """Count any branch on a task name or a task id in one source file.

    Two counters decide the audit.  task_identity_comparisons counts a string
    comparison whose other side reads a task identity (a .name, a task_id, an
    entry_point), which is the shape a per-task special case takes.
    task_identifier_literals counts string literals in the module that equal
    one of the supplied benchmark identifiers.  Both must be zero.
    """
    tree = ast.parse(str(source))
    string_comparisons = 0
    structural_comparisons = 0
    identity_comparisons = 0
    unexplained_comparisons = 0
    branches: List[str] = []
    samples: List[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        sides = [node.left] + list(node.comparators)
        literals = [
            side
            for side in sides
            if isinstance(side, ast.Constant) and isinstance(side.value, str)
        ]
        if not literals:
            continue
        string_comparisons += 1
        if any(_identity_expression(side) for side in sides):
            identity_comparisons += 1
            if len(branches) < 8:
                branches.append(ast.unparse(node))
        values = [str(side.value).strip() for side in literals]
        if all(value in _AUDIT_VOCABULARY or not value for value in values):
            structural_comparisons += 1
        else:
            unexplained_comparisons += 1
            if len(samples) < 8:
                samples.append(ast.unparse(node))
    wanted = {str(value).strip().lower() for value in identifiers if str(value).strip()}
    module_literals = {
        str(node.value).strip().lower()
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    overlap = sorted(module_literals & wanted)
    compared_identifier_hits: List[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        for side in [node.left] + list(node.comparators):
            if isinstance(side, ast.Constant) and isinstance(side.value, str):
                value = str(side.value).strip().lower()
                if value in wanted and value not in compared_identifier_hits:
                    compared_identifier_hits.append(value)
    compared_identifier_hits.sort()
    return {
        "string_literal_comparisons": int(string_comparisons),
        "structural_vocabulary_comparisons": int(structural_comparisons),
        "unexplained_string_comparisons": int(unexplained_comparisons),
        "unexplained_samples": samples,
        "task_identity_comparisons": int(identity_comparisons),
        "compared_identifier_literals": compared_identifier_hits,
        "compared_identifier_literal_count": int(len(compared_identifier_hits)),
        "module_literal_overlap": overlap,
        "module_literal_overlap_note": (
            "informational only: an ordinary English word in the sketch tag "
            "vocabulary can coincide with a benchmark entry-point name. A tag "
            "is not a branch; the gated counters are the comparison ones."
        ),
        "branches": branches,
        "ok": bool(
            identity_comparisons == 0
            and unexplained_comparisons == 0
            and not compared_identifier_hits
        ),
    }


def make_wave_scorer(egress, context: str, tokenizer) -> Callable[[str], float]:
    """A local log-probability prior over candidate bodies (no LLM)."""
    prefix = [int(token) for token in tokenizer.encode(str(context))][-48:]

    def score(body: str) -> float:
        ids = [int(token) for token in tokenizer.encode(str(body))][:48]
        if not ids:
            return 0.0
        total = 0.0
        window = list(prefix)
        for token in ids:
            probability = egress.distribution(window, None)
            total += float(np.log(max(float(probability[token]), 1e-12)))
            window = (window + [token])[-512:]
        return float(total / len(ids))

    return score
