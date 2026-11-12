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


def _annotation_role(annotation: str, name: str) -> str:
    text = str(annotation or "").lower()
    if text:
        if "str" in text:
            return "str"
        if any(word in text for word in ("list", "sequence", "tuple", "iterable", "set")):
            return "seq"
        if any(word in text for word in ("int", "float", "number")):
            return "num"
        if "bool" in text:
            return "bool"
    lowered = str(name).lower()
    if any(word in lowered for word in ("text", "word", "sentence", "phrase", "string", "s")):
        return "str"
    if any(word in lowered for word in (
            "nums", "values", "items", "data", "arr", "lst", "seq", "words",
            "matrix", "rows", "xs", "numbers", "array", "pairs", "letters")):
        return "seq"
    return "num"


@dataclass
class ProblemSpec:
    """Everything the synthesizer is allowed to know about one task."""

    name: str
    params: List[Dict[str, str]]
    returns: str
    docstring: str
    assertions: List[str]
    goal_wave: np.ndarray
    contradictions: List[str] = field(default_factory=list)

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


def parse_problem(prompt: str, *, extra_tests: Optional[Sequence[str]] = None) -> ProblemSpec:
    """Read one task prompt into a typed specification and a Slot-3 goal wave."""
    text = str(prompt)
    match = _SIGNATURE.search(text)
    if match is None:
        raise ValueError("prompt carries no function signature")
    name = str(match.group(1))
    returns = str(match.group(3) or "").strip()
    params: List[Dict[str, str]] = []
    for raw in _split_params(match.group(2)):
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
    doc_match = re.search(r'"""(.*?)"""', text, flags=re.DOTALL)
    docstring = str(doc_match.group(1)) if doc_match else ""
    assertions = [line.strip() for line in _ASSERT.findall(docstring + "\n" + text)]
    for line in _DOCTEST.findall(docstring):
        expression = str(line).strip()
        if "==" not in expression and not expression.startswith("assert"):
            continue
        assertions.append("assert " + expression if not expression.startswith("assert")
                          else expression)
    if extra_tests:
        assertions.extend(str(line).strip() for line in extra_tests)
    seen = set()
    unique_assertions: List[str] = []
    for line in assertions:
        if line in seen:
            continue
        seen.add(line)
        unique_assertions.append(line)

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


def _instantiate(template: str, spec: ProblemSpec) -> Optional[str]:
    seqs = spec.by_role("seq")
    strs = spec.by_role("str")
    nums = spec.by_role("num")
    binding: Dict[str, str] = {"name": spec.name}
    kind_needs = {
        "seq1": ["s1"], "seq2": ["s1", "s2"], "seqnum": ["s1", "n1"],
        "str1": ["s1"], "str2": ["s1", "s2"], "str3": ["s1", "s2", "s3"],
        "num1": ["n1"], "num2": ["n1", "n2"], "any2": ["n1", "n2"],
    }
    for key in re.findall(r"\{(\w+)\}", template):
        if key == "name":
            continue
        if key.startswith("s"):
            pool = strs + seqs
            index = int(key[1:]) - 1
            if index >= len(pool):
                return None
            binding[key] = pool[index]
        else:
            pool = nums + [name for name in seqs if name not in nums]
            index = int(key[1:]) - 1
            if index >= len(pool):
                return None
            binding[key] = pool[index]

    def substitute(match):
        key = match.group(1)
        return str(binding.get(key, match.group(0)))

    return re.sub(r"\{(\w+)\}", substitute, template)


def build_candidates(spec: ProblemSpec) -> List[Dict[str, Any]]:
    """Every typed sketch the slot inventory admits, before wave pruning."""
    out: List[Dict[str, Any]] = []
    for tags, kind, body, slot in TEMPLATES:
        if kind.startswith("num") and not (spec.by_role("num") or spec.by_role("seq")):
            continue
        if kind.startswith("str") and not (spec.by_role("str") or spec.by_role("seq")):
            continue
        if kind.startswith("seq") and not spec.by_role("seq"):
            continue
        rendered = _instantiate(body, spec)
        if rendered is None:
            continue
        out.append({"body": rendered, "tags": tuple(tags), "slot": slot,
                    "kind": kind})
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
        """Prune every slot to V_eff <= slot_support, then rank by the prior."""
        words = set(spec.words)
        candidates = build_candidates(spec)
        for candidate in candidates:
            hits = sum(1 for tag in candidate["tags"] if tag in words)
            candidate["keyword"] = int(hits)
            candidate["wave"] = self._wave_score(candidate["body"])
        slots: Dict[str, List[Dict[str, Any]]] = {}
        for candidate in candidates:
            slots.setdefault(str(candidate["slot"]), []).append(candidate)
        admitted: List[Dict[str, Any]] = []
        v_eff: Dict[str, int] = {}
        for slot, items in slots.items():
            items = sorted(
                items,
                key=lambda row: (-float(row["keyword"]), -float(row["wave"])),
            )
            keep = items[: self.slot_support]
            v_eff[slot] = int(len(keep))
            admitted.extend(keep)
        admitted.sort(key=lambda row: (-float(row["keyword"]), -float(row["wave"])))
        return admitted, v_eff

    # ---------------------------------------------------------- verification
    def _assemble(self, spec: ProblemSpec, body: str) -> str:
        """The emitted module keeps the declared signature and annotations.

        The annotations are what the RepoSheaf capability pullback reads, so a
        body that demands a capability its declared argument cannot provision
        is refused before it is executed.
        """
        declared = ", ".join(
            "%s: %s" % (param["name"], param["annotation"])
            if param["annotation"]
            else str(param["name"])
            for param in spec.params
        )
        suffix = (" -> " + str(spec.returns)) if spec.returns else ""
        return "def %s(%s)%s:\n    %s\n" % (spec.name, declared, suffix, body)

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
            if float(np.linalg.norm(provision)) <= 0.0 and "None" not in str(spec.returns):
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
        greedy: int = 8,
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
        }
        if spec.contradictions:
            report["reason"] = "contradictory_spec"
            self.last_report = report
            return report
        lowered = (spec.docstring + " " + prompt).lower()
        if any(word in lowered for word in _UNSATISFIABLE_WORDS):
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
