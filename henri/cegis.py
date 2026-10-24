"""Counterfactual execution-guided AST repair (CEGIS). Phase 10 Step 2.

WHY THIS MODULE EXISTS
----------------------
`henri/code_synth.py` generates one open-loop pool of AST candidates and
runs them. If the pool fails, it stops and reports `NO_CANDIDATES_PASS`.
The failure signal -- what the program actually returned versus what the
assertion demanded -- is discarded at the sandbox boundary.

`robotics_loop.py` does not throw that signal away. When the Sagnac delta
exceeds the veto it uses the residual to drive six viscoelastic steps.
This module is the software analogue: read the failing assertion, classify
the residual, and synthesize a TARGETED repair rather than a blind mutation.

WHY THE EXISTING EXECUTOR CANNOT SUPPLY THIS
--------------------------------------------
`scripts/mbpp_wave_eval.py::run_candidate` returns only
`{"ok": bool, "reason": str}`. It runs code + tests in a child process and
reports exit status. It cannot say "returned `[3, 2, 1]` where `(3, 2, 1)`
was expected". It also cannot distinguish a SyntaxError from an
AssertionError: both arrive as `nonzero_exit`.

That file is the registered Phase 6/G-AST2 control and is NOT modified
here. This module is additive. It runs its own child process that
instruments each assertion to capture BOTH sides.

HOW ONE CHARACTER DECIDES THE WHOLE DESIGN
------------------------------------------
The instrumented child must never be trusted to report a pass. It reports
OBSERVATIONS only -- actual value, expected value, exception type -- and
the verdict stays where it already lives, in the existing sandbox. A repair
that "wins" in the probe process but fails in `run_candidate` is not a win,
so `code_verify` re-runs every repaired candidate through the real
executor. Provenance `origin="cegis_repair"` marks the winners.

WHAT IS NOT CLAIMED
-------------------
No search over arbitrary programs. Three residual classes only, each a
bounded AST rewrite of an ALREADY-PARSING candidate:

    type_mismatch   wrap the Return expression in the target constructor
    numeric_offset  shift a range bound or the returned scalar by +/-1
    sign_flip       flip the primary operator, or reverse a sequence

Anything else is `unclassified` and produces NO repair. An empty repair
list is a real answer: this candidate offers no targeted fix, and inventing
one would spend sandbox budget on noise.

TWO MEASURED FAILURE MODES I BUILT IN GUARDS AGAINST
----------------------------------------------------
1. `actual == -expected` is TRUE when both are 0. A sign flip synthesized
   from a 0-versus-0 residual is a fabricated repair. Guarded.
2. `abs(actual - expected) in (1, 2)` is TRUE across types (`True - 0 == 1`
   because `bool` is an `int` subclass). The offset rule therefore requires
   BOTH sides to be real numbers and NOT booleans. Guarded.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

REPO = Path(__file__).resolve().parents[1]

CEGIS_ORIGIN = "cegis_repair"
MAX_CEGIS_CANDIDATES = 4

# -- G-SYNTH3 ablation switch ------------------------------------------------
#
# THIS IS A DISABLE FLAG, NOT A DEFAULT-OFF FEATURE FLAG. `name_error` repair
# shipped ON in `4ceabd3`, so default MUST stay "1" or that commit's receipt
# stops describing its own revision.
#
# READ FRESH ON EVERY CALL. An import-cached constant would make both ablation
# arms execute identical code, so the gate could never detect an inert
# mechanism. The probe sets the variable between arms in one process; only a
# live read makes the comparison real.
NAME_ERROR_REPAIR_ENV = "HENRI_CEGIS_NAME_ERROR_REPAIR"


def name_error_repair_enabled() -> bool:
    """False only when the operator disables it explicitly."""
    return os.environ.get(NAME_ERROR_REPAIR_ENV, "1").strip() != "0"
"""Bounded repair budget per failing candidate. Each repair costs one
sandboxed execution, so an unbounded list would turn a synthesis gate into
a search gate without saying so."""

PROBE_TIMEOUT_S = 6
"""Child-process timeout for the diagnostic probe. Matches the executor's
`EXEC_TIMEOUT` so the probe cannot observe a program the real sandbox would
have killed."""


@dataclass
class FailureResidual:
    """One structured observation from a failing assertion."""

    index: int
    test: str
    exception_type: Optional[str] = None
    actual_repr: Optional[str] = None
    expected_repr: Optional[str] = None
    actual_type: Optional[str] = None
    expected_type: Optional[str] = None
    actual_value: object = None
    expected_value: object = None
    exception_msg: Optional[str] = None

    @property
    def kind(self) -> str:
        """Classify into the repair classes, or `unclassified`."""
        # PHASE 11 STEP 3. `NameError` is split OUT of `exception` because it
        # is the one exception class with a bounded AST repair. Without this
        # branch it fell through to `exception`, `synthesise_repairs` returned
        # [], and the most common MBPP synthesis crash earned no budget.
        if self.exception_type == "NameError":
            return "name_error"
        if self.exception_type and self.actual_repr is None:
            return "exception"
        if self.actual_type is None or self.expected_type is None:
            return "unclassified"
        if self.actual_type != self.expected_type:
            return "type_mismatch"
        if _is_real_number(self.actual_value) and _is_real_number(
                self.expected_value):
            delta = self.actual_value - self.expected_value
            if delta != 0 and abs(delta) in (1, 2):
                return "numeric_offset"
            if self.actual_value != 0 and self.actual_value == -self.expected_value:
                return "sign_flip"
            return "unclassified"
        if isinstance(self.actual_value, (list, str)) and \
                isinstance(self.expected_value, (list, str)):
            if _reversed_equal(self.actual_value, self.expected_value):
                return "sign_flip"
        if isinstance(self.actual_value, bool) and \
                isinstance(self.expected_value, bool):
            if self.actual_value != self.expected_value:
                return "sign_flip"
        return "unclassified"

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "test": self.test[:200],
            "kind": self.kind,
            "exception_type": self.exception_type,
            "actual_repr": _clip(self.actual_repr),
            "expected_repr": _clip(self.expected_repr),
            "actual_type": self.actual_type,
            "expected_type": self.expected_type,
            # PHASE 11 STEP 3. The name is the only thing that drives a
            # NameError repair, so a receipt that omits it cannot be audited
            # for whether the mechanism had a target.
            "exception_msg": _clip(self.exception_msg, 160),
        }


def _clip(text: Optional[str], n: int = 120) -> Optional[str]:
    if text is None:
        return None
    return text if len(text) <= n else text[:n] + "..."


def _is_real_number(v: object) -> bool:
    """True for int/float but NOT bool. `isinstance(True, int)` is True."""
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _reversed_equal(a, b) -> bool:
    if isinstance(a, str) and isinstance(b, str):
        return a != b and a == b[::-1]
    if isinstance(a, list) and isinstance(b, list):
        return a != b and a == list(reversed(b))
    return False


# -- diagnostic probe -----------------------------------------------------

_CHILD = r'''
import ast, json, sys

SOURCE = {source!r}
TESTS = {tests!r}


def _safe(v):
    """JSON-safe view of a runtime value. Types preserved for the parent."""
    if isinstance(v, (bool, int, float, str)) or v is None:
        return v
    if isinstance(v, tuple):
        return {"__tuple__": [_safe(x) for x in v]}
    if isinstance(v, (list, set, frozenset)):
        return [_safe(x) for x in v]
    if isinstance(v, dict):
        return {"__dict__": [[_safe(k), _safe(x)] for k, x in v.items()]}
    return None


def _sides(test, ns):
    """Actual/expected from `assert L == R`, evaluated at module scope."""
    try:
        tree = ast.parse(test)
    except SyntaxError:
        return None
    if not tree.body or not isinstance(tree.body[0], ast.Assert):
        return None
    expr = tree.body[0].test
    if not (isinstance(expr, ast.Compare) and len(expr.ops) == 1
            and isinstance(expr.ops[0], ast.Eq)):
        return None
    out = []
    for side in (expr.left, expr.comparators[0]):
        node = ast.Expression(side)
        ast.fix_missing_locations(node)
        try:
            out.append(eval(compile(node, "<side>", "eval"), ns))
        except BaseException:
            return None
    return out[0], out[1]


ns = {"__name__": "__candidate__"}
report = {"loaded": True, "load_error": None, "failures": []}
try:
    exec(compile(SOURCE, "<candidate>", "exec"), ns)
except BaseException as exc:
    report["loaded"] = False
    # MEASURED NEED: the type alone cannot drive a NameError repair. The name
    # is only present in `str(exc)`. Without it `synthesise_repairs` has no
    # target to import or rebind, so the residual is unrepairable by
    # construction and the arm under-reports for a reason unrelated to code.
    report["load_error"] = type(exc).__name__
    report["load_error_msg"] = str(exc)[:200]
    print("__CEGIS__" + json.dumps(report))
    raise SystemExit(0)

for i, t in enumerate(TESTS):
    try:
        exec(compile(t, "<test>", "exec"), ns)
    except BaseException as exc:
        rec = {"index": i, "test": t, "exception_type": type(exc).__name__,
               "exception_msg": str(exc)[:200],
               "actual": None, "expected": None,
               "actual_type": None, "expected_type": None,
               "actual_repr": None, "expected_repr": None}
        pair = _sides(t, ns)
        if pair is not None:
            a, e = pair
            rec["actual"] = _safe(a)
            rec["expected"] = _safe(e)
            rec["actual_type"] = type(a).__name__
            rec["expected_type"] = type(e).__name__
            rec["actual_repr"] = repr(a)
            rec["expected_repr"] = repr(e)
        report["failures"].append(rec)
        break

print("__CEGIS__" + json.dumps(report))
'''


def probe_failure(code: str, tests: Sequence[str],
                  timeout: int = PROBE_TIMEOUT_S) -> Optional[FailureResidual]:
    """Run one candidate against `tests` and return the FIRST failure.

    Returns `None` when the candidate passes every assertion, when the
    candidate fails to load (a SyntaxError is not a repairable residual),
    or when the probe itself cannot run. A `None` is a real answer and the
    caller must spend no repair budget on it.
    """
    if not code.strip() or not tests:
        return None
    body = _CHILD.replace("{source!r}", repr(code)).replace(
        "{tests!r}", repr(list(tests)))
    path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix="_cegis.py", delete=False,
                                         encoding="utf-8") as f:
            f.write(body)
            path = f.name
        p = subprocess.run([sys.executable, path], capture_output=True,
                           text=True, timeout=timeout, cwd=str(REPO))
        marker = "__CEGIS__"
        line = next((ln for ln in (p.stdout or "").splitlines()
                     if ln.startswith(marker)), None)
        if line is None:
            return None
        rep = json.loads(line[len(marker):])
        # MEASURED NEED: a LOAD-TIME NameError (`sqrt` used before any import
        # was ever added) sets loaded=False and failures=[]. The guard below
        # returned None, so the single most common MBPP synthesis crash earned
        # zero repair budget. Split the case and carry it as a residual.
        if not rep.get("loaded"):
            if rep.get("load_error") == "NameError":
                return FailureResidual(
                    index=-1, test="", exception_type="NameError",
                    exception_msg=rep.get("load_error_msg"))
            return None
        if not rep.get("failures"):
            return None
        rec = rep["failures"][0]
        return FailureResidual(
            index=int(rec["index"]), test=str(rec["test"]),
            exception_type=rec.get("exception_type"),
            actual_repr=rec.get("actual_repr"),
            expected_repr=rec.get("expected_repr"),
            actual_type=rec.get("actual_type"),
            expected_type=rec.get("expected_type"),
            actual_value=_unsafe(rec.get("actual")),
            expected_value=_unsafe(rec.get("expected")),
            exception_msg=rec.get("exception_msg"),
        )
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError,
            KeyError, ValueError):
        return None
    finally:
        if path:
            Path(path).unlink(missing_ok=True)


def _unsafe(v):
    """Inverse of the child's `_safe`. Restores tuples and sets."""
    if isinstance(v, dict) and "__tuple__" in v:
        return tuple(_unsafe(x) for x in v["__tuple__"])
    if isinstance(v, dict) and "__dict__" in v:
        return {_unsafe(k): _unsafe(x) for k, x in v["__dict__"]}
    if isinstance(v, list):
        return [_unsafe(x) for x in v]
    return v


# -- repair synthesizers --------------------------------------------------

_TYPE_CASTERS = {
    "tuple": "tuple", "list": "list", "set": "set", "frozenset": "frozenset",
    "bool": "bool", "int": "int", "float": "float", "str": "str",
}


def _parse(source: str):
    try:
        return ast.parse(source)
    except SyntaxError:
        return None


def _unparse(tree) -> Optional[str]:
    ast.fix_missing_locations(tree)
    try:
        return ast.unparse(tree)
    except Exception:                     # noqa: BLE001
        return None


def repair_return_type(source: str, target_type: str) -> Optional[str]:
    """Wrap every `Return` value in the constructor of `target_type`.

    For `str` the wrap is `"".join(...)`, because `str([1, 2])` returns
    `"[1, 2]"` and would satisfy the TYPE while failing the VALUE. That
    distinction matters: a cast that satisfies the type check but cannot
    satisfy the assertion wastes a sandbox execution and hides the real
    residual.
    """
    tree = _parse(source)
    if tree is None:
        return None
    caster = _TYPE_CASTERS.get(target_type)
    if caster is None:
        return None

    hits = [0]

    class Wrap(ast.NodeTransformer):
        def visit_Return(self, node):     # noqa: N802
            self.generic_visit(node)
            if node.value is None:
                return node
            if caster == "str":
                call = ast.Call(
                    func=ast.Attribute(value=ast.Constant(value=""),
                                       attr="join", ctx=ast.Load()),
                    args=[node.value], keywords=[])
            else:
                call = ast.Call(func=ast.Name(id=caster, ctx=ast.Load()),
                                args=[node.value], keywords=[])
            node.value = call
            hits[0] += 1
            return node

    out = Wrap().visit(tree)
    if hits[0] == 0:
        return None
    return _unparse(out)


def repair_numeric_offset(source: str, delta: int,
                          limit: int = 2) -> List[str]:
    """Shift range bounds and returned scalars by `delta`.

    Two independent rewrites, each emitted separately so a receipt can
    attribute a win to the bound or to the return value rather than to
    "the offset family".
    """
    tree = _parse(source)
    if tree is None:
        return []
    out: List[str] = []

    # (a) every `range(...)` last bound moves by delta.
    t2 = _parse(source)
    seen = [0]

    class RangeShift(ast.NodeTransformer):
        def visit_Call(self, node):       # noqa: N802
            self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id == "range" \
                    and node.args:
                node.args[-1] = ast.BinOp(
                    left=node.args[-1],
                    op=ast.Add() if delta > 0 else ast.Sub(),
                    right=ast.Constant(value=abs(int(delta))))
                seen[0] += 1
            return node

    t2 = RangeShift().visit(t2)
    if seen[0]:
        text = _unparse(t2)
        if text and text != source:
            out.append(text)

    # (b) the returned expression moves by delta.
    t3 = _parse(source)
    seen3 = [0]

    class ReturnShift(ast.NodeTransformer):
        def visit_Return(self, node):     # noqa: N802
            self.generic_visit(node)
            if node.value is None or isinstance(node.value, ast.Constant):
                return node
            node.value = ast.BinOp(
                left=node.value,
                op=ast.Add() if delta > 0 else ast.Sub(),
                right=ast.Constant(value=abs(int(delta))))
            seen3[0] += 1
            return node

    t3 = ReturnShift().visit(t3)
    if seen3[0]:
        text = _unparse(t3)
        if text and text not in out and text != source:
            out.append(text)
    return out[:limit]


def repair_sign_flip(source: str) -> List[str]:
    """Flip the primary operator, or reverse the returned sequence.

    The operator swap reuses `code_synth._swap_one` so there is ONE
    definition of an antipode in the tree. A second table would drift.
    """
    from henri.code_synth import OPPOSITE, _swap_one

    out: List[str] = []
    tree = _parse(source)
    if tree is None:
        return out

    kinds = []
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp):
            kinds.append(type(node.op))
        elif isinstance(node, ast.Compare):
            kinds.extend(type(o) for o in node.ops)
        elif isinstance(node, ast.BoolOp):
            kinds.append(type(node.op))
    for cls in kinds:
        if cls not in OPPOSITE:
            continue
        text = _swap_one(source, cls)
        if text and text != source and text not in out:
            out.append(text)
        if len(out) >= 2:
            break

    # Reverse the returned sequence. This is the antipode for Axis 1
    # (Order) from the Step-1 morphism basis, applied to code.
    t2 = _parse(source)
    seen = [0]

    class ReverseReturn(ast.NodeTransformer):
        def visit_Return(self, node):     # noqa: N802
            self.generic_visit(node)
            if node.value is None:
                return node
            node.value = ast.Subscript(
                value=node.value,
                slice=ast.Slice(lower=None, upper=None,
                                step=ast.UnaryOp(op=ast.USub(),
                                                 operand=ast.Constant(value=1))),
                ctx=ast.Load())
            seen[0] += 1
            return node

    t2 = ReverseReturn().visit(t2)
    if seen[0]:
        text = _unparse(t2)
        if text and text not in out:
            out.append(text)
    return out[:2]


# -- NameError / missing-import repair (Phase 11 Step 3) -------------------
#
# MEASURED NEED, from `G-CEGIS1` (`704d623`). Round-2 value-residual repair
# lifted held-out MBPP 0.2250 -> 0.2750, and the remaining failures split into
# two classes that carry NO repairable value residual:
#
#   NameError   a template body references a stdlib symbol that was never
#               imported (`sqrt` from `math`), or a donor variable that was
#               renamed away. Execution aborts before `return`, so there is no
#               `actual` value to compare against `expected`.
#   TypeError   call arity differs from the assertion's call. Handled upstream
#               in `code_synth.adapt_arity`, not here.
#
# A value-residual repair cannot reach either class. These two mechanisms are
# the only bounded repairs available, and both are AST-exact: no string
# rewriting, no guessing at a target name.
#
# The import table is EXPLICIT and closed. An unexpected name returns no
# candidate rather than a speculative import, because a wrong import turns a
# clean NameError into a different, harder failure.

STDLIB_IMPORTS = {
    "math": "import math",
    "re": "import re",
    "heapq": "import heapq",
    "itertools": "import itertools",
    "collections": "import collections",
    "functools": "import functools",
    "sys": "import sys",
    "operator": "import operator",
    "string": "import string",
    "Counter": "from collections import Counter",
    "defaultdict": "from collections import defaultdict",
    "deque": "from collections import deque",
    "OrderedDict": "from collections import OrderedDict",
    "reduce": "from functools import reduce",
    "gcd": "from math import gcd",
    "lcm": "from math import lcm",
    "sqrt": "from math import sqrt",
    "ceil": "from math import ceil",
    "floor": "from math import floor",
    "log": "from math import log",
    "log2": "from math import log2",
    "factorial": "from math import factorial",
    "pi": "from math import pi",
    "e": "from math import e",
    "inf": "from math import inf",
    "perm": "from itertools import permutations as perm",
    "permutations": "from itertools import permutations",
    "combinations": "from itertools import combinations",
    "product": "from itertools import product",
    "accumulate": "from itertools import accumulate",
    "chain": "from itertools import chain",
    "heappush": "from heapq import heappush",
    "heappop": "from heapq import heappop",
    "heapify": "from heapq import heapify",
}

_NAMEERROR_RE = re.compile(r"name '([^']+)' is not defined")


def _missing_name(message):
    """Recover `X` from `NameError: name 'X' is not defined`.

    Returns None for any other message shape. A miss must not invent a name.
    """
    if not message:
        return None
    m = _NAMEERROR_RE.search(message)
    return m.group(1) if m else None


def repair_missing_import(source: str, name: str) -> Optional[str]:
    """Prepend the stdlib import for `name`, if the table names one.

    The import goes FIRST in the module body so it precedes every reference.
    Returns None when the table has no entry: an unknown name is not a stdlib
    symbol, so an import would be a fabrication.
    """
    stmt = STDLIB_IMPORTS.get(name)
    if not stmt:
        return None
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    # Already imported? Then the NameError came from somewhere else and this
    # repair would be a no-op that reports as a mechanism.
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if (a.asname or a.name.split(".")[0]) == name:
                    return None
        if isinstance(node, ast.ImportFrom):
            for a in node.names:
                if (a.asname or a.name) == name:
                    return None
    imp = ast.parse(stmt).body[0]
    tree.body.insert(0, imp)
    ast.fix_missing_locations(tree)
    try:
        return ast.unparse(tree)
    except Exception:                     # noqa: BLE001
        return None


def repair_unbound_name(source: str, name: str) -> Optional[str]:
    """Rebind a Load-context `name` to the entry function's first parameter.

    Only fires when the name is genuinely unbound: not a parameter, not
    assigned anywhere in the function, not a builtin, not the function's own
    name. Otherwise the rebinding would be a semantic change disguised as a
    repair.
    """
    import builtins

    if not name or not name.isidentifier():
        return None
    if hasattr(builtins, name):
        return None
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not fns:
        return None
    fn = fns[0]
    params = [a.arg for a in list(getattr(fn.args, "posonlyargs", []))
              + list(fn.args.args)]
    if not params:
        return None
    if name in params or name == fn.name:
        return None
    # Assigned inside the body? Then it is a local, and the NameError is a
    # use-before-assignment bug that rebinding would silently paper over.
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) \
                and node.id == name:
            return None
    target = params[0]
    hits = 0
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) \
                and node.id == name:
            node.id = target
            hits += 1
    if hits == 0:
        return None
    ast.fix_missing_locations(tree)
    try:
        return ast.unparse(tree)
    except Exception:                     # noqa: BLE001
        return None


def synthesise_repairs(source: str, residual: FailureResidual,
                       max_candidates: int = MAX_CEGIS_CANDIDATES
                       ) -> List[dict]:
    """Repairs for ONE residual, each tagged with its mechanism.

    Returns `[]` when the residual is `unclassified` or `exception`. An
    empty list is honest: there is no targeted fix for "raised IndexError",
    and a blind mutation is not a counterfactual repair.
    """
    kind = residual.kind
    rows: List[dict] = []

    def push(text: Optional[str], mechanism: str) -> None:
        if not text or text == source:
            return
        if any(r["text"] == text for r in rows):
            return
        rows.append({"text": text, "origin": CEGIS_ORIGIN,
                     "mechanism": mechanism,
                     "residual_kind": kind})

    if kind == "name_error":
        # G-SYNTH3 ABLATION GATE. Read the switch HERE, not at import, so the
        # two arms of the ablation differ in exactly one flag. When disabled,
        # NO name_error candidate is emitted, so the round budget is spent only
        # on the other residual classes. That isolates this branch's marginal
        # contribution at a MATCHED round count.
        if not name_error_repair_enabled():
            # MEASURED DEFECT, fixed here: this line read `return repairs` and
            # the accumulator is `rows`. Python does not check that at compile
            # time, so the ablation arm died with `NameError: name 'repairs'`
            # instead of returning an empty candidate list.
            return rows[:int(max(1, max_candidates))]
        # MUTUALLY EXCLUSIVE MECHANISMS. Measured defect in my first version:
        # for `sqrt` both fired, and `rebind_sqrt` rewrote `sqrt(x)` to
        # `x(x)`, which executes and raises `'int' object is not callable`.
        # A repair that replaces a clean NameError with a confusing TypeError
        # is worse than no repair, because it mislabels the residual class.
        #
        # A name in the closed stdlib table has a correct import. A name
        # outside it cannot be imported without fabricating a module. So the
        # first match wins and the second is not attempted.
        nm = _missing_name(residual.exception_msg)
        if nm:
            if nm in STDLIB_IMPORTS:
                push(repair_missing_import(source, nm), "import_%s" % nm)
            else:
                push(repair_unbound_name(source, nm), "rebind_%s" % nm)

    elif kind == "type_mismatch" and residual.expected_type:
        push(repair_return_type(source, residual.expected_type),
             "return_type_%s" % residual.expected_type)

    elif kind == "numeric_offset":
        delta = int(round(residual.actual_value - residual.expected_value))
        # Move the program TOWARD the expected value, so the sign is the
        # negative of the measured residual.
        for i, text in enumerate(repair_numeric_offset(source, -delta)):
            push(text, "numeric_offset_%+d_%d" % (-delta, i + 1))

    elif kind == "sign_flip":
        for i, text in enumerate(repair_sign_flip(source)):
            push(text, "sign_flip_%d" % (i + 1))

    return rows[:int(max(1, max_candidates))]


__all__ = ["FailureResidual", "probe_failure", "synthesise_repairs",
           "repair_return_type", "repair_numeric_offset", "repair_sign_flip",
           "repair_missing_import", "repair_unbound_name", "STDLIB_IMPORTS",
           "CEGIS_ORIGIN", "MAX_CEGIS_CANDIDATES", "PROBE_TIMEOUT_S"]
