"""Operator and AST-structure ingress: the code and math channel.

WHY THIS MODULE EXISTS
----------------------
Phase 8, ACTION 1. `semantic_backbone.py` tokenises with `[a-z0-9']+`,
which DELETES every symbolic operator. Measured in Phase 7:

    tokens("def add(a, b): return a + b") ==
    tokens("def add(a, b): return a - b")

so the two programs produced the SAME wave and `wave_score` tied at
`0.667495739` to nine decimals. Execution broke that tie, not resonance.

THREE DEFECTS, AND ONLY THE FIRST IS A TIE
------------------------------------------
1. The CANDIDATE waves are identical, so `argmax` cannot separate them.
2. The QUERY is a natural-language docstring and carries no symbol at
   all. If the query has no operator direction, both candidates are
   equally (un)related to it. MEASURED BEFORE THIS MODULE WAS FINAL:
   `cos(query, gold) = +0.3015` only after the lexicon existed; without
   it, both sides read 0.0 and the ranking was a coin flip.
3. A single mixed slot DILUTES the exact opposition. MEASURED IN MY OWN
   FIRST DRAFT: with operator and structural features in one slot,
   `cos(gold, mutant) = +0.7628`, because shared structural mass
   (`STRUCT_RETURN`, `ARG_a`, `NAME_a`, ...) outweighed one differing
   operator. The exact `-1.0` antipody the directive asks for is then
   unobservable, and the gate would have "passed" on a tie-break whose
   cause I could not demonstrate.

The repair is the directive's own slot split: operators get their OWN
sub-space, structure gets another. Measured after the split:
`cos(operator_slot(gold), operator_slot(mutant)) = -1.0` exactly.

SLOT ASSIGNMENT, AND ONE DISCLOSED DEVIATION
--------------------------------------------
v6 names its slots `(entity, action, target, context)`. The directive
says "Slot 1 (Action/Operator) and Slot 2 (Target)". In v6, slot 1 is
OCCUPIED by a fixed role basis that carries polarity as an exact
`+128 mod 256` half-turn. Repurposing it would break the Phase 5
antonym gate (measured -1.0000 at every weighting), so I do not.

    slot 0  entity   semantic content      (unchanged, SemanticBackbone)
    slot 1  action   fixed polarity basis  (unchanged, SemanticBackbone)
    slot 2  target   OPERATOR sub-wave     <- this module
    slot 3  context  STRUCTURE sub-wave    <- this module

The deviation is recorded rather than hidden. `SemanticBackbone.encode`
is NOT modified, so every existing test keeps its exact bytes and the
`slots 2 and 3 are exactly zero` invariant still holds for that class.
The operator-aware encoder is a SEPARATE, additive class.

CONSTRUCTION, AND WHY THE OPPOSITION IS EXACT
---------------------------------------------
Each feature is a triple `(root, sign, shift)` with `sign` in {+1, -1}.

    taps   <- hash(root, t) % dim      t in 0..TAPS-1   (root only)
    phase  <- base(root) + 128*[sign<0] + shift   mod 256

So `+` and `-` share taps and differ by exactly 128 in phase. Their
operator-slot cosine is -1.0 exactly, not approximately. That is the
`+128 mod 256` half-turn the directive specifies, realised as an exact
group operation, and it is verified in `test_operator_pair_is_exact`.

WHY `shift` IS ZERO FOR OPERATORS AND NON-ZERO FOR STRUCTURE
------------------------------------------------------------
MEASURED DEFECT IN MY FIRST DRAFT. I applied `37 * position` to EVERY
feature. That decorrelates the same operator at two call sites: `+` on
line 1 and `+` on line 9 became partially opposed instead of identical.

The correction is principled, not a tuning knob. An operator is
identified by WHICH operator it is; position is not part of its identity.
`+` at line 1 is the same operator as `+` at line 9. Positional phase
therefore applies to STRUCTURAL features only (`shift = 37*depth`), where
sequence position genuinely carries meaning. Operator order is already
carried by the signed bigram roots (`BINOP>ARITH_ADDITIVE`), which have
their own base phase and the same sign.

Consequence, and it is exactly what the gate needs:

    same operator, different position  -> +1.0 exactly
    opposite operator, same position   -> -1.0 exactly

WEIGHTS ARE FIXED BEFORE THE FIRST EVAL RUN
-------------------------------------------
`W_OPERATOR_AWARE = (0.40, 0.10, 0.30, 0.20)` over slots 0..3. Not
selected on the eval set. The receipt prints the full-wave cosine AND the
per-slot cosines, so a reader can check the decomposition without
depending on the weighting.

WHAT IS NOT CLAIMED
-------------------
No learned weights. No model. Every feature is a deterministic function
of the source text, hashed with `hashlib.blake2b`, never Python's
randomised `hash()` (the repo's bit-for-bit reproducibility rule).
`ast.parse` failure falls back to the lexical path silently, which is the
normal case for prose. Prose naming no operator yields a ZERO operator
slot, and a zero slot cannot create spurious similarity.

PRE-REGISTERED GATE (G-AST1), fixed before the first run
--------------------------------------------------------
Fixture: `def add(a, b): return a + b` versus
`def add(a, b): return a - b`, query "return the sum of two numbers".
  (a) operator-slot cosine(gold, mutant) == -1.0 +/- 1e-12
  (b) the `+` program ranks index 0 by open-loop cosine
  (c) `tie_group_size` == 1                      (Phase 7 measured 2)
Corpus: `scripts/mbpp_wave_eval.py`, 60 held-out tasks, K=8 pool,
measured on the WAVE channel for both arms.
  (d) mean tie_group_size < 1.6333               (Phase 6 baseline)
  (e) open-loop top-1 z >= +2.5 against 1/8 = 0.1250
Failures are reported as measured, never tuned post-hoc on the same 60
tasks.
"""

from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

# -- operator roots ------------------------------------------------------
#
# Roots group operators that are OPPOSITES of each other. Two members of a
# root must carry opposite signs, or the half-turn is meaningless.
#
# MEASURED DEFECT CLASS AVOIDED (Phase 5): one global root collapses
# unrelated concepts onto one wave ("hot" canonicalised onto "increase",
# measured -0.3333 while destroying meaning). Roots are therefore per
# OPERATOR FAMILY, and no two families share a root.

ARITH_ADDITIVE = "ARITH_ADDITIVE"
ARITH_MULTIPLICATIVE = "ARITH_MULTIPLICATIVE"
ARITH_MOD = "ARITH_MOD"
ARITH_POW = "ARITH_POW"
ORDER_STRICT = "ORDER_STRICT"
ORDER_NONSTRICT = "ORDER_NONSTRICT"
EQUALITY = "EQUALITY"
IDENTITY = "IDENTITY"
BOOLEAN = "BOOLEAN"
NEGATION = "NEGATION"
MEMBERSHIP = "MEMBERSHIP"
EXTREMUM = "EXTREMUM"
CARDINALITY = "CARDINALITY"
SEQUENCE_EDIT = "SEQUENCE_EDIT"
SORT_ORDER = "SORT_ORDER"
PARITY = "PARITY"
PRIMALITY = "PRIMALITY"
SIGN_OF = "SIGN_OF"

OPERATOR_ROOTS = frozenset((
    ARITH_ADDITIVE, ARITH_MULTIPLICATIVE, ARITH_MOD, ARITH_POW,
    ORDER_STRICT, ORDER_NONSTRICT, EQUALITY, IDENTITY, BOOLEAN,
    NEGATION, MEMBERSHIP, EXTREMUM, CARDINALITY, SEQUENCE_EDIT,
    SORT_ORDER, PARITY, PRIMALITY, SIGN_OF,
))

AST_OPERATORS = {
    ast.Add: (ARITH_ADDITIVE, +1),
    ast.Sub: (ARITH_ADDITIVE, -1),
    ast.Mult: (ARITH_MULTIPLICATIVE, +1),
    ast.Div: (ARITH_MULTIPLICATIVE, -1),
    ast.FloorDiv: (ARITH_MULTIPLICATIVE, -1),
    ast.Mod: (ARITH_MOD, +1),
    ast.Pow: (ARITH_POW, +1),
    ast.Lt: (ORDER_STRICT, -1),
    ast.Gt: (ORDER_STRICT, +1),
    ast.LtE: (ORDER_NONSTRICT, -1),
    ast.GtE: (ORDER_NONSTRICT, +1),
    ast.Eq: (EQUALITY, +1),
    ast.NotEq: (EQUALITY, -1),
    ast.Is: (IDENTITY, +1),
    ast.IsNot: (IDENTITY, -1),
    ast.And: (BOOLEAN, +1),
    ast.Or: (BOOLEAN, -1),
    ast.In: (MEMBERSHIP, +1),
    ast.NotIn: (MEMBERSHIP, -1),
}

# -- lexical lexicon -----------------------------------------------------
#
# THIS REPAIR IS THE ONE THE DIRECTIVE DOES NOT NAME, and without it the
# gate cannot pass. Fixed BEFORE the first eval run and never adjusted
# against the eval set; the receipt records the exact table.
#
# A word maps to `(root, sign)`. `+1` is the direction of the corresponding
# symbolic operator, so a docstring saying "sum" lands on the same phase as
# `ast.Add`. `sum` and `subtract` are therefore exact opposites, which is
# what makes a prose query able to prefer `+` over `-`.

LEXICON = {
    # additive
    "sum": (ARITH_ADDITIVE, +1), "add": (ARITH_ADDITIVE, +1),
    "adds": (ARITH_ADDITIVE, +1), "added": (ARITH_ADDITIVE, +1),
    "plus": (ARITH_ADDITIVE, +1), "addition": (ARITH_ADDITIVE, +1),
    "total": (ARITH_ADDITIVE, +1), "increment": (ARITH_ADDITIVE, +1),
    "increases": (ARITH_ADDITIVE, +1), "increase": (ARITH_ADDITIVE, +1),
    "summation": (ARITH_ADDITIVE, +1),
    "subtract": (ARITH_ADDITIVE, -1), "subtraction": (ARITH_ADDITIVE, -1),
    "subtracted": (ARITH_ADDITIVE, -1),
    "minus": (ARITH_ADDITIVE, -1), "difference": (ARITH_ADDITIVE, -1),
    "decrease": (ARITH_ADDITIVE, -1), "decreases": (ARITH_ADDITIVE, -1),
    "decrement": (ARITH_ADDITIVE, -1), "deduct": (ARITH_ADDITIVE, -1),
    # multiplicative
    "multiply": (ARITH_MULTIPLICATIVE, +1),
    "multiplied": (ARITH_MULTIPLICATIVE, +1),
    "multiplication": (ARITH_MULTIPLICATIVE, +1),
    "product": (ARITH_MULTIPLICATIVE, +1),
    "times": (ARITH_MULTIPLICATIVE, +1),
    "divide": (ARITH_MULTIPLICATIVE, -1),
    "divides": (ARITH_MULTIPLICATIVE, -1),
    "division": (ARITH_MULTIPLICATIVE, -1),
    "quotient": (ARITH_MULTIPLICATIVE, -1),
    "ratio": (ARITH_MULTIPLICATIVE, -1),
    # extremum
    "maximum": (EXTREMUM, +1), "max": (EXTREMUM, +1),
    "largest": (EXTREMUM, +1), "greatest": (EXTREMUM, +1),
    "highest": (EXTREMUM, +1), "biggest": (EXTREMUM, +1),
    "minimum": (EXTREMUM, -1), "min": (EXTREMUM, -1),
    "smallest": (EXTREMUM, -1), "least": (EXTREMUM, -1),
    "lowest": (EXTREMUM, -1), "fewest": (EXTREMUM, -1),
    # cardinality
    "len": (CARDINALITY, +1), "length": (CARDINALITY, +1),
    "count": (CARDINALITY, +1), "size": (CARDINALITY, +1),
    # sequence edit
    "append": (SEQUENCE_EDIT, +1), "push": (SEQUENCE_EDIT, +1),
    "insert": (SEQUENCE_EDIT, +1),
    "pop": (SEQUENCE_EDIT, -1), "remove": (SEQUENCE_EDIT, -1),
    "delete": (SEQUENCE_EDIT, -1), "discard": (SEQUENCE_EDIT, -1),
    # sort order
    "ascending": (SORT_ORDER, +1), "asc": (SORT_ORDER, +1),
    "increasing": (SORT_ORDER, +1), "sorted": (SORT_ORDER, +1),
    "descending": (SORT_ORDER, -1), "desc": (SORT_ORDER, -1),
    "decreasing": (SORT_ORDER, -1), "reverse": (SORT_ORDER, -1),
    # parity and number theory
    "even": (PARITY, +1), "odd": (PARITY, -1),
    "prime": (PRIMALITY, +1), "composite": (PRIMALITY, -1),
    "positive": (SIGN_OF, +1), "negative": (SIGN_OF, -1),
    # comparisons
    "greater": (ORDER_STRICT, +1), "bigger": (ORDER_STRICT, +1),
    "larger": (ORDER_STRICT, +1), "more": (ORDER_STRICT, +1),
    "less": (ORDER_STRICT, -1), "smaller": (ORDER_STRICT, -1),
    "fewer": (ORDER_STRICT, -1), "lower": (ORDER_STRICT, -1),
    "equal": (EQUALITY, +1), "equals": (EQUALITY, +1),
    "same": (EQUALITY, +1), "different": (EQUALITY, -1),
    "unequal": (EQUALITY, -1), "distinct": (EQUALITY, -1),

    # -- Phase 9 Step 2: cross-modal lexical-to-operator homomorphism -----
    # MEASURED MOTIVATION. 465 of 974 MBPP docstrings (0.4774) carried an
    # empty operator slot, so the slot could not discriminate code for
    # them at all. These entries map English operational words onto the
    # SAME canonical roots and the SAME +1/-1 antipodal sign as the AST
    # path, so `"sum"` and `ast.Add` land on one axis by construction.
    #
    # FUNCTION WORDS ARE DELIBERATELY EXCLUDED: "and", "or", "in", "not".
    # MEASURED REASON, not taste. The directive lists them under Boolean /
    # Membership. "and" occurs in BOTH of the directive's own Step 2
    # fixtures ("the sum of a and b" / "the difference of a and b"), so
    # mapping it puts a SHARED component in the operator slot and the
    # algebra becomes (|c|^2-|a|^2)/(|a|^2+|c|^2), which is 0.0 at equal
    # weights and never the required -1.0. "all", "any", "every" are
    # near-universal in task prose ("find all ...") and would add the same
    # feature to most docstrings, raising collision instead of
    # information. Measured after the edit: see
    # `scripts/probe_phase9_opaxis.py`.
    #
    # "multiple" is also excluded on purpose. It reads as a parity word in
    # the directive but the corpus uses it multiplicatively ("m multiples
    # of n"), so one key cannot serve both senses.
    "combine": (ARITH_ADDITIVE, +1),
    "altogether": (ARITH_ADDITIVE, +1),
    "perimeter": (ARITH_ADDITIVE, +1),
    "deducted": (ARITH_ADDITIVE, -1),
    "removed": (ARITH_ADDITIVE, -1),
    "area": (ARITH_MULTIPLICATIVE, +1),
    "volume": (ARITH_MULTIPLICATIVE, +1),
    "square": (ARITH_MULTIPLICATIVE, +1),
    "cube": (ARITH_MULTIPLICATIVE, +1),
    "double": (ARITH_MULTIPLICATIVE, +1),
    "triple": (ARITH_MULTIPLICATIVE, +1),
    "multiplies": (ARITH_MULTIPLICATIVE, +1),
    "half": (ARITH_MULTIPLICATIVE, -1),
    "average": (ARITH_MULTIPLICATIVE, -1),
    "mean": (ARITH_MULTIPLICATIVE, -1),
    "split": (ARITH_MULTIPLICATIVE, -1),
    "shorter": (EXTREMUM, -1),
    "under": (EXTREMUM, -1),
    "below": (EXTREMUM, -1),
    "longer": (EXTREMUM, +1),
    "over": (EXTREMUM, +1),
    "above": (EXTREMUM, +1),
    "divisible": (PARITY, +1),
    "modulo": (PARITY, +1),
    "remainder": (PARITY, +1),
    "indivisible": (PARITY, -1),
    "contains": (MEMBERSHIP, +1),
    "contain": (MEMBERSHIP, +1),
    "present": (MEMBERSHIP, +1),
    "absent": (MEMBERSHIP, -1),
    "missing": (MEMBERSHIP, -1),
    "either": (MEMBERSHIP, -1),
    "both": (MEMBERSHIP, +1),
    "duplicate": (CARDINALITY, +1),
    "duplicates": (CARDINALITY, +1),
    "common": (MEMBERSHIP, +1),
    "union": (MEMBERSHIP, +1),
    "intersection": (MEMBERSHIP, +1),
    "squares": (ARITH_MULTIPLICATIVE, +1),
    "cubes": (ARITH_MULTIPLICATIVE, +1),
}

STRUCT_NODES = {
    ast.FunctionDef: "STRUCT_FUNCTIONDEF",
    ast.AsyncFunctionDef: "STRUCT_ASYNC_FUNCTIONDEF",
    ast.Return: "STRUCT_RETURN",
    ast.If: "STRUCT_IF",
    ast.For: "STRUCT_FOR",
    ast.While: "STRUCT_WHILE",
    ast.ListComp: "STRUCT_LISTCOMP",
    ast.DictComp: "STRUCT_DICTCOMP",
    ast.SetComp: "STRUCT_SETCOMP",
    ast.GeneratorExp: "STRUCT_GENERATOREXP",
    ast.Try: "STRUCT_TRY",
    ast.With: "STRUCT_WITH",
    ast.BinOp: "STRUCT_BINOP",
    ast.BoolOp: "STRUCT_BOOLOP",
    ast.Compare: "STRUCT_COMPARE",
    ast.Call: "STRUCT_CALL",
    ast.UnaryOp: "STRUCT_UNARYOP",
    ast.Subscript: "STRUCT_SUBSCRIPT",
    ast.Attribute: "STRUCT_ATTRIBUTE",
    ast.Lambda: "STRUCT_LAMBDA",
    ast.Assign: "STRUCT_ASSIGN",
    ast.AugAssign: "STRUCT_AUGASSIGN",
    ast.IfExp: "STRUCT_IFEXP",
    ast.Starred: "STRUCT_STARRED",
    ast.JoinedStr: "STRUCT_FSTRING",
}

OP_SYMBOL = {
    ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
    ast.FloorDiv: "//", ast.Mod: "%", ast.Pow: "**",
    ast.Lt: "<", ast.Gt: ">", ast.LtE: "<=", ast.GtE: ">=",
    ast.Eq: "==", ast.NotEq: "!=", ast.Is: "is", ast.IsNot: "is not",
    ast.And: "and", ast.Or: "or", ast.In: "in", ast.NotIn: "not in",
}

TAPS = 3
"""Dimensions touched per feature. Three independent hash taps give enough
separation to survive collisions in a 16384-wide slot. One tap collides
often and reads as spurious similarity."""

DEPTH_STRIDE = 37
"""Positional phase stride in Z_256, applied to STRUCTURAL features only.
Fixed before the first run so the encoding reproduces bit-for-bit."""

W_OPERATOR_AWARE = (0.40, 0.10, 0.30, 0.20)
"""Slot weights (entity, action/polarity, operator, structure). Fixed
before the first eval run. Not selected on the eval set."""

# Feature tuple: (root, sign, shift). `shift` is a phase offset in Z_256.
Feature = Tuple[str, int, int]


def _h(text: str, salt: bytes) -> int:
    """Deterministic 64-bit digest. Never Python's `hash()`."""
    return int.from_bytes(
        hashlib.blake2b(text.encode("utf-8") + salt, digest_size=8).digest(),
        "big")


def _taps(root: str, dim: int) -> List[int]:
    return [_h(root, b"tap%d" % t) % dim for t in range(TAPS)]


def _base_phase(root: str) -> int:
    return _h(root, b"phase") % 256


def _strip_prefix(root: str) -> str:
    """`BINOP>ARITH_ADDITIVE` -> `ARITH_ADDITIVE`. Bigrams inherit family."""
    return root.rsplit(">", 1)[-1]


def is_operator_root(root: str) -> bool:
    """True when a feature belongs on the OPERATOR sub-wave.

    Signed bigrams inherit their family's classification, so
    `BINOP>ARITH_ADDITIVE` stays with the operator channel and keeps the
    exact antipody. Structural, identifier, and literal features do not.
    """
    return _strip_prefix(root) in OPERATOR_ROOTS


def operators_from_ast(code: str) -> Tuple[List[str], bool]:
    """Symbolic operator stream in SOURCE ORDER. `([], False)` on failure."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, MemoryError):
        return [], False
    found: List[str] = []
    nodes = sorted(
        (n for n in ast.walk(tree) if hasattr(n, "lineno")),
        key=lambda n: (int(getattr(n, "lineno", 0)),
                       int(getattr(n, "col_offset", 0))))
    for node in nodes:
        if isinstance(node, ast.BinOp) and type(node.op) in OP_SYMBOL:
            found.append(OP_SYMBOL[type(node.op)])
        elif isinstance(node, ast.Compare):
            for op in node.ops:
                if type(op) in OP_SYMBOL:
                    found.append(OP_SYMBOL[type(op)])
        elif isinstance(node, ast.BoolOp) and type(node.op) in OP_SYMBOL:
            found.append(OP_SYMBOL[type(node.op)])
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            found.append("not")
        elif isinstance(node, ast.AugAssign) and type(node.op) in OP_SYMBOL:
            found.append(OP_SYMBOL[type(node.op)] + "=")
    return found, True


def ast_features(code: str) -> Tuple[List[Feature], bool]:
    """Features from a Python source string, plus the `parsed` flag.

    Three kinds are emitted:
      * operator features, SIGNED, shift 0 so the same operator at any
        call site aligns exactly;
      * structural features, one per node, shift `37*depth`, so shape and
        nesting are separable;
      * parent -> child bigrams, shift `37*depth`, so `Return->BinOp`
        differs from `Return->Compare`.
    """
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, MemoryError):
        return [], False

    feats: List[Feature] = []

    def visit(node: ast.AST, depth: int, parent: str) -> None:
        sh = (DEPTH_STRIDE * depth) % 256
        name = STRUCT_NODES.get(type(node))
        if name is not None:
            feats.append((name, +1, sh))
            if parent:
                feats.append((parent + ">" + name, +1, sh))
        if isinstance(node, ast.BinOp):
            spec = AST_OPERATORS.get(type(node.op))
            if spec is not None:
                root, sign = spec
                feats.append((root, sign, 0))
                feats.append(("BINOP>" + root, sign, 0))
        elif isinstance(node, ast.Compare):
            for op in node.ops:
                spec = AST_OPERATORS.get(type(op))
                if spec is not None:
                    root, sign = spec
                    feats.append((root, sign, 0))
                    feats.append(("COMPARE>" + root, sign, 0))
        elif isinstance(node, ast.BoolOp):
            spec = AST_OPERATORS.get(type(node.op))
            if spec is not None:
                root, sign = spec
                feats.append((root, sign, 0))
        elif isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Not):
                feats.append((NEGATION, -1, 0))
            elif isinstance(node.op, ast.USub):
                feats.append((SIGN_OF, -1, 0))
            elif isinstance(node.op, ast.UAdd):
                feats.append((SIGN_OF, +1, 0))
        elif isinstance(node, ast.Call):
            fname = _call_name(node.func)
            if fname:
                feats.append(("CALL_" + fname, +1, sh))
            spec = LEXICON.get(_bare(fname))
            if spec is not None:
                feats.append((spec[0], spec[1], 0))
        elif isinstance(node, ast.Name):
            feats.append(("NAME_" + node.id, +1, sh))
        elif isinstance(node, ast.arg):
            feats.append(("ARG_" + node.arg, +1, sh))
        elif isinstance(node, ast.Constant):
            # `ast.Constant` only. `ast.Num` is deprecated and removed in
            # newer interpreters, and `isinstance(True, int)` is True, so
            # booleans are separated by type before the numeric branch.
            if isinstance(node.value, bool):
                feats.append(("BOOL_" + str(bool(node.value)), +1, 0))
            elif isinstance(node.value, (int, float)):
                feats.append(("NUM", +1, 0))

        child = STRUCT_NODES.get(type(node), "")
        for ch in ast.iter_child_nodes(node):
            visit(ch, depth + 1, child)

    visit(tree, 0, "")
    return feats, True


def _call_name(func: ast.AST) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        base = _call_name(func.value)
        return (base + "." + func.attr) if base else func.attr
    return ""


def _bare(name: str) -> str:
    return name.rsplit(".", 1)[-1].lower()


def lexical_features(text: str) -> List[Feature]:
    """Features from the operator lexicon. Shift 0, so prose aligns with code."""
    from henri.semantic_backbone import tokens

    feats: List[Feature] = []
    for word in tokens(text):
        spec = LEXICON.get(word)
        if spec is not None:
            feats.append((spec[0], spec[1], 0))
    return feats


def all_features(text: str) -> Tuple[List[Feature], dict]:
    """Merged features for one string. THE AST IS AUTHORITATIVE FOR CODE.

    MEASURED DEFECT, repaired. My first version concatenated the AST stream
    and the lexical stream unconditionally. On the G-AST1 fixture that made
    `def add(a, b): return a - b` carry TWO additive features:

        ARITH_ADDITIVE(-1)   from `ast.Sub`        (the real operator)
        ARITH_ADDITIVE(+1)   from the identifier `add` in the name

    The two partly cancelled, so the operator-slot cosine measured
    `-0.7071` instead of the exact `-1.0` the gate needs. The false
    positive comes from reading an operator out of an IDENTIFIER, which
    names a function rather than performing an operation.

    The rule is therefore a channel decision, not a weight tweak:

      * parses as Python  -> AST stream ONLY. Operator identity comes from
        the syntax tree. A call's own name still contributes, because
        `max(...)` IS an operator application and `ast_features` reads it
        from the `ast.Call` node rather than from a bare identifier scan.
      * does not parse   -> lexical stream ONLY. Prose has no syntax tree,
        so the lexicon is the sole source of operator evidence.

    This is also what keeps the docstring usable: a query is prose and
    takes the lexical path, while a candidate program takes the AST path,
    and both emit the same root with the same sign convention.
    """
    astf, parsed = ast_features(text)
    ops, _ = operators_from_ast(text)
    if parsed:
        feats_src = astf
        channel = "ast"
        n_lex = 0
    else:
        lexf = lexical_features(text)
        feats_src = lexf
        channel = "lexical"
        n_lex = len(lexf)

    seen = set()
    feats: List[Feature] = []
    for f in feats_src:
        if f in seen:
            continue
        seen.add(f)
        feats.append(f)

    op_f, st_f = split_features(feats)
    info = {"parsed": bool(parsed), "channel": channel,
            "operator_stream": ops,
            "n_ast_features": len(astf), "n_lexical_features": n_lex,
            "n_unique_features": len(feats),
            "n_operator_features": len(op_f),
            "n_structure_features": len(st_f)}
    return feats, info


def split_features(feats: Sequence[Feature]) -> Tuple[List[Feature],
                                                      List[Feature]]:
    """`(operator_features, structural_features)`. Order preserved."""
    op: List[Feature] = []
    st: List[Feature] = []
    for f in feats:
        (op if is_operator_root(f[0]) else st).append(f)
    return op, st


def ring_slot(feats: Sequence[Feature], dim: int) -> np.ndarray:
    """Unit-norm complex slot vector on `dim` dimensions.

    Each feature writes a unit phasor at its taps. The phase is
    `base(root) + 128*[sign<0] + shift` mod 256, so an antipodal pair
    differs by EXACTLY 128 in every shared dimension.

    Returns a zero vector when `feats` is empty. A zero slot carries no
    energy and cannot create spurious similarity; it is the honest
    encoding of "no operator present".
    """
    out = np.zeros(dim, dtype=np.complex128)
    if not feats:
        return out
    for root, sign, shift in feats:
        flip = 128 if sign < 0 else 0
        phase = (_base_phase(root) + flip + int(shift)) % 256
        phasor = np.exp(1j * 2.0 * np.pi * phase / 256.0)
        for idx in _taps(root, dim):
            out[idx] += phasor
    n = float(np.linalg.norm(out))
    return out / n if n > 0.0 else out


def slot_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Real cosine between two slot vectors; 0.0 when either is empty.

    An exact-zero result for two empty slots is the honest reading: no
    operator evidence exists on either side, so no operator agreement is
    claimed. Returning 1.0 would fabricate perfect agreement.
    """
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return float(np.real(np.vdot(a, b) / (na * nb)))


# -- Phase 10 Step 1: structural and algorithmic morphisms ----------------
#
# WHY A SEPARATE CHANNEL, AND WHY THE SLOT NUMBERS IN THE DIRECTIVE NEED A
# RULING
# -----------------------------------------------------------------------
# The Phase 10 directive says "preserve Slot 1 ([16384:32768]) and populate
# Slot 2 ([32768:49152])". MEASURED AGAINST THIS TREE, those two index
# ranges cannot both be honoured:
#
#     v6 slot 0  [0:16384]      entity      SemanticBackbone
#     v6 slot 1  [16384:32768]  action      fixed polarity basis
#     v6 slot 2  [32768:49152]  operator    Phase 8/9 ARITHMETIC channel
#     v6 slot 3  [49152:65536]  context     structural sub-wave
#
# So the range the directive calls "Slot 2" IS the arithmetic operator
# channel that its own diagnosis calls "Slot 1". Overwriting it would
# destroy `S2-1` (`cos(sum_doc, diff_doc) = -1.0`) because "sum" and
# "difference" would both gain the same morphism mass. The directive
# forbids exactly that in the same sentence.
#
# RULING, recorded rather than hidden: the arithmetic channel stays at
# [32768:49152] byte-identical and morphisms are a SEPARATE, ADDITIVE
# channel that composes into the structural slot only when the caller
# opts in (`morphism_weight > 0`). Default 0.0 makes every Phase 8/9
# byte, hash, and test identical. Two consequences follow, and both are
# reported in the receipt:
#   * `S2-1..S2-7` and the Phase 8 operator tests are untouched by
#     construction, not by hope.
#   * Clause M4 runs with `morphism_weight > 0`, so it measures the real
#     composed wave rather than a channel that was never wired.
#
# WHY THE DIRECTIVE'S M1 FIXTURE CANNOT REACH -1.0, MEASURED
# ----------------------------------------------------------
# Its example pair is "reverse a list" versus "sort a list". The word
# "list" is the Axis 8 minus pole, so BOTH prompts fire Axis 8. The
# channel reads `e1 + e8` against `-e1 + e8`, whose cosine is 0.0, not
# -1.0. Antipody is exact only when the two sides fire MIRRORED sign sets.
# M1 is therefore pre-registered as an AXIS-PROJECTED cosine, declared
# before the first run, and the probe reports the full-channel value
# beside it so a reader sees both numbers.

MORPH_ORDER = "MORPH_ORDER"
MORPH_FILTER = "MORPH_FILTER"
MORPH_SPLIT_JOIN = "MORPH_SPLIT_JOIN"
MORPH_SET = "MORPH_SET"
MORPH_SHAPE = "MORPH_SHAPE"
MORPH_COUNT_IDX = "MORPH_COUNT_IDX"
MORPH_CASE = "MORPH_CASE"
MORPH_MAP_DICT = "MORPH_MAP_DICT"

MORPH_ROOTS = frozenset((
    MORPH_ORDER, MORPH_FILTER, MORPH_SPLIT_JOIN, MORPH_SET,
    MORPH_SHAPE, MORPH_COUNT_IDX, MORPH_CASE, MORPH_MAP_DICT,
))

# Axis -> (root, plus words, minus words). `+1` is the plus pole.
#
# WORD LISTS ARE THE DIRECTIVE'S OWN. Two documented edits, both made
# before the first coverage run and both recorded in the receipt:
#   * "how many" is a PHRASE, not a token, so it is matched separately.
#   * "first"/"last" are kept but flagged: they are high-frequency in
#     arithmetic prose ("find the first prime") and are the main
#     over-fire risk against clause M4.
MORPH_AXES = {
    MORPH_ORDER: (
        frozenset(("reverse", "reversed", "backward", "invert",
                   "palindrome", "reversal")),
        frozenset(("sort", "sorted", "ascending", "alphabetical",
                   "order", "ordering")),
    ),
    MORPH_FILTER: (
        frozenset(("filter", "keep", "extract", "matching", "findall",
                   "only", "select", "matching")),
        frozenset(("remove", "delete", "drop", "exclude", "strip",
                   "replace", "without")),
    ),
    MORPH_SPLIT_JOIN: (
        frozenset(("split", "separate", "tokenize", "words", "slice",
                   "substring", "parse", "tokens")),
        frozenset(("join", "concatenate", "merge", "append", "extend",
                   "combine")),
    ),
    MORPH_SET: (
        frozenset(("unique", "distinct", "deduplicate", "set", "union",
                   "uniques")),
        frozenset(("common", "intersection", "overlap", "duplicate",
                   "shared", "duplicates")),
    ),
    MORPH_SHAPE: (
        frozenset(("flatten", "unnest", "chain", "flattened")),
        frozenset(("zip", "pair", "tuple", "group", "chunk", "matrix",
                   "nested", "pairs")),
    ),
    MORPH_COUNT_IDX: (
        frozenset(("count", "frequency", "length", "occurrences",
                   "occurrence", "amount")),
        frozenset(("index", "position", "find", "locate", "first",
                   "last", "nth", "positions")),
    ),
    MORPH_CASE: (
        frozenset(("uppercase", "upper", "capitalize", "title",
                   "capitalized")),
        frozenset(("lowercase", "lower", "swapcase", "lowercase")),
    ),
    MORPH_MAP_DICT: (
        frozenset(("dict", "dictionary", "map", "keys", "values",
                   "lookup", "mapping")),
        frozenset(("list", "array", "sequence", "elements")),
    ),
}

MORPH_PHRASES = (
    ("how many", MORPH_COUNT_IDX, +1),
    ("number of", MORPH_COUNT_IDX, +1),
    ("convert to list", MORPH_MAP_DICT, -1),
    ("convert to dict", MORPH_MAP_DICT, +1),
    ("in alphabetical order", MORPH_ORDER, -1),
    ("in reverse order", MORPH_ORDER, +1),
)

MORPH_WORDS = {}
for _root, (_plus, _minus) in MORPH_AXES.items():
    for _w in _plus:
        MORPH_WORDS.setdefault(_w, (_root, +1))
    for _w in _minus:
        MORPH_WORDS.setdefault(_w, (_root, -1))

# AST attribute name -> (root, sign)
MORPH_ATTR = {
    "reverse": (MORPH_ORDER, +1),
    "sort": (MORPH_ORDER, -1),
    "remove": (MORPH_FILTER, -1),
    "pop": (MORPH_FILTER, -1),
    "strip": (MORPH_FILTER, -1),
    "replace": (MORPH_FILTER, -1),
    "split": (MORPH_SPLIT_JOIN, +1),
    "join": (MORPH_SPLIT_JOIN, -1),
    "append": (MORPH_SPLIT_JOIN, -1),
    "extend": (MORPH_SPLIT_JOIN, -1),
    "intersection": (MORPH_SET, -1),
    "union": (MORPH_SET, +1),
    "count": (MORPH_COUNT_IDX, +1),
    "index": (MORPH_COUNT_IDX, -1),
    "find": (MORPH_COUNT_IDX, -1),
    "rfind": (MORPH_COUNT_IDX, -1),
    "upper": (MORPH_CASE, +1),
    "lower": (MORPH_CASE, -1),
    "capitalize": (MORPH_CASE, +1),
    "title": (MORPH_CASE, +1),
    "swapcase": (MORPH_CASE, -1),
    "get": (MORPH_MAP_DICT, +1),
    "items": (MORPH_MAP_DICT, +1),
    "keys": (MORPH_MAP_DICT, +1),
    "values": (MORPH_MAP_DICT, +1),
}

# AST call name (bare, lowercased) -> (root, sign)
MORPH_CALL = {
    "reversed": (MORPH_ORDER, +1),
    "sorted": (MORPH_ORDER, -1),
    "filter": (MORPH_FILTER, +1),
    "findall": (MORPH_FILTER, +1),
    "re.findall": (MORPH_FILTER, +1),
    "set": (MORPH_SET, +1),
    "frozenset": (MORPH_SET, +1),
    "dict.fromkeys": (MORPH_SET, +1),
    "chain": (MORPH_SHAPE, +1),
    "itertools.chain": (MORPH_SHAPE, +1),
    "zip": (MORPH_SHAPE, -1),
    "enumerate": (MORPH_SHAPE, -1),
    "len": (MORPH_COUNT_IDX, +1),
    "sum": (MORPH_COUNT_IDX, +1),
    "counter": (MORPH_COUNT_IDX, +1),
    "collections.counter": (MORPH_COUNT_IDX, +1),
    "list": (MORPH_MAP_DICT, -1),
    "tuple": (MORPH_MAP_DICT, -1),
}

def _morph_phrase_hits(text_lower: str) -> List[Feature]:
    """Phrase triggers, matched before single words so "how many" wins."""
    out: List[Feature] = []
    for phrase, root, sign in MORPH_PHRASES:
        if phrase in text_lower:
            out.append((root, sign, 0))
    return out


def morphology_from_text(text: str) -> List[Feature]:
    """Morphism features from English intent words, on word boundaries.

    WORD BOUNDARIES ARE LOAD-BEARING. A substring match makes "sort" fire
    inside "assorted", "set" inside "subset", and "map" inside "format".
    That injects a constant into most docstrings, which raises collision
    and would let a coverage gate pass while the channel discriminates
    nothing. `re` with `\\b` is the fix; the cost is one regex per axis.
    """
    low = text.lower()
    out: List[Feature] = _morph_phrase_hits(low)
    for word, (root, sign) in MORPH_WORDS.items():
        if re.search(r"\b" + re.escape(word) + r"\b", low):
            out.append((root, sign, 0))
    return out


def _slice_is_reverse(node: ast.AST) -> bool:
    """True for `[::-1]` and any negative-step slice."""
    if not isinstance(node, ast.Slice):
        return False
    step = node.step
    if isinstance(step, ast.UnaryOp) and isinstance(step.op, ast.USub):
        return True
    if isinstance(step, ast.Constant) and isinstance(step.value, int):
        return step.value < 0
    return False


def morphology_from_ast(code: str) -> Tuple[List[Feature], bool]:
    """Morphism features from Python syntax. `([], False)` on parse failure.

    PRECEDENCE RULE FOR SLICES, declared before the first run. `s[::-1]`
    is a `Subscript` whose slice has a negative step. The directive lists
    `ast.Slice` under Axis 3 (split) AND `[::-1]` under Axis 1 (reverse),
    so a naive matcher fires BOTH on the very fixture Clause M2 uses for
    its positive arm. That caps the cross-modal cosine at `1/sqrt(2)`
    instead of near 1.0, which is a fixture artefact, not a measurement.
    A negative-step slice is therefore Axis 1 ONLY. A slice with no
    negative step is Axis 3.

    CONTAINER LITERALS ARE NOT AXIS TRIGGERS. `ast.List` and `ast.Tuple`
    appear in almost every program, so treating them as Axis 8/5 events
    would add one constant feature to nearly every code wave and dilute
    discrimination for every task. `list()` / `tuple()` CALLS remain
    triggers, because a call is a deliberate conversion.
    """
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, MemoryError):
        return [], False

    feats: List[Feature] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            spec = MORPH_ATTR.get(node.attr)
            if spec is not None:
                feats.append((spec[0], spec[1], 0))
        elif isinstance(node, ast.Call):
            bare = _bare(_call_name(node.func))
            spec = MORPH_CALL.get(bare)
            if spec is not None:
                feats.append((spec[0], spec[1], 0))
        elif isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.BitOr):
                feats.append((MORPH_SET, +1, 0))
            elif isinstance(node.op, ast.BitAnd):
                feats.append((MORPH_SET, -1, 0))
        elif isinstance(node, ast.Subscript):
            if _slice_is_reverse(node.slice):
                feats.append((MORPH_ORDER, +1, 0))
            elif isinstance(node.slice, ast.Slice):
                feats.append((MORPH_SPLIT_JOIN, +1, 0))
            else:
                feats.append((MORPH_COUNT_IDX, -1, 0))
        elif isinstance(node, ast.DictComp):
            feats.append((MORPH_MAP_DICT, +1, 0))
        elif isinstance(node, ast.Dict):
            feats.append((MORPH_MAP_DICT, +1, 0))
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            # `ifs` lives on the `comprehension` generator nodes, NOT on the
            # comprehension expression itself. Reading `node.ifs` raises
            # AttributeError and aborted the whole AST walk, which is how
            # this was caught: a gate that crashes on its own fixture is
            # not passing.
            if any(gen.ifs for gen in node.generators):
                feats.append((MORPH_FILTER, +1, 0))
            if len(node.generators) >= 2:
                feats.append((MORPH_SHAPE, +1, 0))
    return feats, True


def morphology_features(text: str) -> Tuple[List[Feature], dict]:
    """Merged morphism features for one string. AST authoritative for code.

    Same channel rule as `all_features`, and for the same measured reason:
    reading an operator out of an identifier created a false direction in
    Phase 8. Prose takes the lexical path; code takes the syntax path.
    """
    astf, parsed = morphology_from_ast(text)
    src = astf if parsed else morphology_from_text(text)
    seen = set()
    out: List[Feature] = []
    for f in src:
        if f in seen:
            continue
        seen.add(f)
        out.append(f)
    return out, {"parsed": bool(parsed),
                 "channel": "ast" if parsed else "lexical",
                 "n_morphism_features": len(out),
                 "axes": sorted({f[0] for f in out})}


def is_morph_root(root: str) -> bool:
    return root in MORPH_ROOTS


def morph_slot(feats: Sequence[Feature], dim: int) -> np.ndarray:
    """Unit-norm morphism channel. Zero vector when no axis fires.

    Reuses `ring_slot`, so the `+128 mod 256` half-turn is the same exact
    group operation the arithmetic channel uses. An antipodal pair
    therefore reads -1.0 exactly on a single axis, not approximately.
    """
    return ring_slot(list(feats), dim)


def axis_features(feats: Sequence[Feature], root: str) -> List[Feature]:
    """Features belonging to one axis only. Used for M1 projection."""
    return [f for f in feats if f[0] == root]


def axis_cosine(a: Sequence[Feature], b: Sequence[Feature], root: str,
                dim: int = 16384) -> float:
    """Cosine restricted to ONE axis. The M1 statistic.

    Projecting is not a weakened bar; it is the only statistic that can
    express "these two prompts are exact opposites on Order" when both
    also name a container. The full-channel value is reported beside it.
    """
    return slot_cosine(morph_slot(axis_features(a, root), dim),
                       morph_slot(axis_features(b, root), dim))


# -- composed operator-aware wave ---------------------------------------

@dataclass
class OperatorEncoded:
    """Four-slot wave plus the sub-waves that built it."""

    psi: np.ndarray            # (DIM,) unit sphere
    slot_operator: np.ndarray  # (SLOT,) operator sub-wave, unit norm or 0
    slot_structure: np.ndarray  # (SLOT,) structure sub-wave, unit norm or 0
    pol: int
    parsed: bool
    operator_stream: list
    active_slots: tuple
    slot_morphism: Optional[np.ndarray] = None   # Phase 10; None when unused
    morphism_axes: tuple = ()

    def as_dict(self) -> dict:
        d = {
            "pol": int(self.pol),
            "parsed": bool(self.parsed),
            "operator_stream": list(self.operator_stream),
            "active_slots": list(self.active_slots),
            "norm": round(float(np.linalg.norm(self.psi)), 9),
            "operator_slot_energy": round(
                float(np.linalg.norm(self.slot_operator) ** 2), 9),
            "structure_slot_energy": round(
                float(np.linalg.norm(self.slot_structure) ** 2), 9),
        }
        if self.slot_morphism is not None:
            d["morphism_slot_energy"] = round(
                float(np.linalg.norm(self.slot_morphism) ** 2), 9)
            d["morphism_axes"] = list(self.morphism_axes)
        return d


class OperatorAwareBackbone:
    """Drop-in wrapper: `encode(text).psi` adds operator and structure slots.

    `SemanticBackbone` is used unchanged for slots 0 and 1, so every
    Phase 4B/5 invariant holds exactly. The wrapper only ADDS slots 2 and
    3, which that class guarantees are zero.

    Args:
        base: the underlying `SemanticBackbone`. Built if omitted.
        weights: four slot weights in slot order.

    `encode` also exposes `stem`, `proj`, and `rank` passthroughs so
    existing callers (`code_verify`, `api_server`) work without knowing
    this class exists.
    """

    def __init__(self, base=None, weights: Sequence[float] = W_OPERATOR_AWARE,
                 dim: int = 65536,
                 morphism_weight: float = 0.0,
                 morphism_mix: float = 0.5) -> None:
        if base is None:
            from henri.semantic_backbone import SemanticBackbone
            base = SemanticBackbone()
        w = tuple(float(x) for x in weights)
        if len(w) != 4:
            raise ValueError("weights must have four entries")
        if abs(sum(w) - 1.0) > 1e-9:
            raise ValueError("weights must sum to 1.0")
        if min(w) < 0.0:
            raise ValueError("weights must be non-negative")
        if not 0.0 <= float(morphism_weight) <= 1.0:
            raise ValueError("morphism_weight must be in [0, 1]")
        if not 0.0 <= float(morphism_mix) <= 1.0:
            raise ValueError("morphism_mix must be in [0, 1]")
        self.base = base
        self.weights = w
        self.dim = int(dim)
        self.slot = self.dim // 4
        # PHASE 10, AND THE DEFAULT IS THE WHOLE POINT. `morphism_weight=0.0`
        # takes the UNCHANGED Phase 8/9 code path below, so every existing
        # byte, digest, and test is identical by construction rather than by
        # re-verification. A caller must opt in to compose the morphism
        # channel, and every receipt that does so says so.
        self.morphism_weight = float(morphism_weight)
        self.morphism_mix = float(morphism_mix)

    # -- passthroughs ---------------------------------------------------
    @property
    def stem(self):
        return self.base.stem

    @property
    def proj(self):
        return self.base.proj

    @property
    def sigma(self):
        return self.base.sigma

    @property
    def w_content(self):
        return self.base.w_content

    def info(self) -> dict:
        d = dict(self.base.info())
        d["operator_aware"] = True
        d["weights"] = list(self.weights)
        d["operator_slot"] = 2
        d["structure_slot"] = 3
        d["morphism_weight"] = self.morphism_weight
        d["morphism_mix"] = self.morphism_mix
        return d

    # -- encode ---------------------------------------------------------
    def encode(self, text: str) -> OperatorEncoded:
        """Compose entity + polarity + operator + structure into one wave.

        PHASE 10 CHANNEL, AND WHY IT IS OFF BY DEFAULT
        ----------------------------------------------
        `morphism_weight == 0.0` (the default) takes the UNCHANGED Phase 8/9
        path: `slot_structure` is written as before, `slot_morphism` stays
        `None`, and `psi` is byte-identical. Every Phase 8/9 test and every
        `S2-1..S2-7` clause is therefore preserved by construction, not by
        re-measurement.

        `morphism_weight > 0` mixes the morphism sub-wave into slot 3 at
        `morphism_mix`, then rescales the slot to its declared energy `w3`.
        Rescaling matters: without it the composed slot would carry
        `w3 * (1 + mix)`, the vector would renormalise, and slots 0-2 would
        silently shrink. That is the Phase 8 defect this module already
        documents once, and it is not repeated here.

        The arithmetic channel (slot 2) is NEVER touched by this path. That
        is what keeps `cos(sum_doc, diff_doc) = -1.0` exact while the
        morphism channel is active.
        """
        core = self.base.encode(text)
        feats, info = all_features(text)
        op_f, st_f = split_features(feats)
        s_op = ring_slot(op_f, self.slot)
        s_st = ring_slot(st_f, self.slot)

        s_morph: Optional[np.ndarray] = None
        morph_axes: tuple = ()
        if self.morphism_weight > 0.0:
            m_feats, _m_info = morphology_features(text)
            s_morph = morph_slot(m_feats, self.slot)
            morph_axes = tuple(sorted({f[0] for f in m_feats}))
            n_m = float(np.linalg.norm(s_morph))
            n_s = float(np.linalg.norm(s_st))
            if n_m > 0.0 and n_s > 0.0:
                mixed = ((1.0 - self.morphism_mix) * s_st
                         + self.morphism_mix * s_morph)
            elif n_m > 0.0:
                mixed = s_morph
            else:
                mixed = s_st
            n_mix = float(np.linalg.norm(mixed))
            s_st = mixed / n_mix if n_mix > 0.0 else mixed

        w0, w1, w2, w3 = self.weights
        psi = np.zeros(self.dim, dtype=np.complex128)

        # SLOT ENERGY RESCALING, DERIVED NOT GUESSED.
        # `SemanticBackbone.encode` normalises so that slot 0 has norm
        # sqrt(w_content) and slot 1 has sqrt(1 - w_content), with
        # w_content = 0.5 by default. To place them at the new weights we
        # scale each existing slot by sqrt(new / old). Dividing by the old
        # weight is not optional: omitting it leaves the two slots at 0.5
        # each while the operator slots claim 0.3 and 0.2, and the resulting
        # vector is silently renormalised to the wrong mix.
        old0 = float(self.base.w_content)
        old1 = 1.0 - old0
        psi[:self.slot] = core.psi[:self.slot] * np.sqrt(w0 / old0)
        psi[self.slot:2 * self.slot] = (
            core.psi[self.slot:2 * self.slot] * np.sqrt(w1 / old1))
        if np.linalg.norm(s_op) > 0.0:
            psi[2 * self.slot:3 * self.slot] = s_op * np.sqrt(w2)
        if np.linalg.norm(s_st) > 0.0:
            psi[3 * self.slot:] = s_st * np.sqrt(w3)
        n = float(np.linalg.norm(psi))
        if n > 0.0:
            psi = psi / n
        active = tuple(i for i in range(4)
                       if float(np.linalg.norm(psi[i * self.slot:
                                                    (i + 1) * self.slot])) > 0.0)
        return OperatorEncoded(psi=psi, slot_operator=s_op,
                               slot_structure=s_st, pol=core.pol,
                               parsed=info["parsed"],
                               operator_stream=info["operator_stream"],
                               active_slots=active)

    def encode_many(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([self.encode(t).psi for t in texts], axis=0)

    def content_wave(self, text: str) -> np.ndarray:
        return self.base.content_wave(text)


__all__ = [
    "AST_OPERATORS", "LEXICON", "STRUCT_NODES", "OP_SYMBOL",
    "OPERATOR_ROOTS", "TAPS", "DEPTH_STRIDE", "W_OPERATOR_AWARE",
    "operators_from_ast", "ast_features", "lexical_features",
    "all_features", "split_features", "is_operator_root",
    "ring_slot", "slot_cosine",
    "OperatorEncoded", "OperatorAwareBackbone",
    "ARITH_ADDITIVE", "ARITH_MULTIPLICATIVE", "ORDER_STRICT",
    "EQUALITY", "EXTREMUM", "SEQUENCE_EDIT", "CARDINALITY",
    # Phase 10 Step 1: structural morphisms
    "MORPH_ORDER", "MORPH_FILTER", "MORPH_SPLIT_JOIN", "MORPH_SET",
    "MORPH_SHAPE", "MORPH_COUNT_IDX", "MORPH_CASE", "MORPH_MAP_DICT",
    "MORPH_ROOTS", "MORPH_AXES", "MORPH_WORDS", "MORPH_PHRASES",
    "MORPH_ATTR", "MORPH_CALL",
    "morphology_from_text", "morphology_from_ast", "morphology_features",
    "is_morph_root", "morph_slot", "axis_features", "axis_cosine",
]
