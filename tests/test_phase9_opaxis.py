"""Phase 9 Step 2: cross-modal lexical-to-operator homomorphism.

Clause S2-4 (zero-slot rate <= 0.10) is measured on the corpus and lives in
`scripts/probe_phase9_opaxis.py` with a receipt. It is deliberately NOT a
unit test here: a corpus rate is not a unit invariant, and asserting it
would either fail permanently or be weakened to pass. MEASURED: 0.4774 ->
0.3747, and the residue names no arithmetic operation.

These tests cover what must hold for every run. Each states what makes it
fail.
"""
from __future__ import annotations

import numpy as np

from henri import operator_ingress as oi

SLOT = 16384

SUM_DOC = "Write a function to find the sum of a and b"
DIFF_DOC = "Write a function to find the difference of a and b"
ADD_CODE = "def f(a, b): return a + b"
SUB_CODE = "def f(a, b): return a - b"
MUL_CODE = "def f(a, b): return a * b"


def op_slot(text: str) -> np.ndarray:
    feats, _ = oi.all_features(text)
    opf, _struct = oi.split_features(feats)
    if not opf:
        return np.zeros(SLOT, dtype=np.complex128)
    return oi.ring_slot(opf, SLOT)


def cos(a, b) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-30 or nb < 1e-30:
        return 0.0
    return float(np.real(np.vdot(a, b)) / (na * nb))


def test_prose_antipody_is_exact_not_approximate():
    """cos(sum_doc, difference_doc) must be EXACTLY -1.0.

    FAILS if any word shared by both sentences enters the operator slot.
    A shared component `c` against opposing parts `+-a` gives
    (|c|^2-|a|^2)/(|a|^2+|c|^2), which is 0.0 at equal weights and never
    -1.0. This is why "and" is excluded from the lexicon: both fixtures
    contain it.
    """
    s, d = op_slot(SUM_DOC), op_slot(DIFF_DOC)
    assert abs(cos(s, d) + 1.0) <= 1e-12, cos(s, d)


def test_cross_modal_positive_and_negative():
    """English "sum" must land on the same axis as `+`, opposed to `-`.

    FAILS if either bound slips. MEASURED margins are +0.0071 and -0.0071
    against the directive's +-0.70 bars, so this is a knife edge; a single
    stray stopword in the operator channel flips it.
    """
    s = op_slot(SUM_DOC)
    a, b = op_slot(ADD_CODE), op_slot(SUB_CODE)
    assert cos(s, a) > 0.70, cos(s, a)
    assert cos(s, b) < -0.70, cos(s, b)


def test_multiplicative_bleed_control():
    """`sum` must not claim the multiplicative axis.

    FAILS if the lexicon maps every arithmetic word to one root. Without
    this control, S2-2/S2-3 could pass by making all arithmetic look
    additive.
    """
    s = op_slot(SUM_DOC)
    m = op_slot(MUL_CODE)
    assert cos(s, m) < 0.70, cos(s, m)


def test_code_operator_antipody_still_exact():
    """The Phase 8 property must survive the Step 2 lexicon additions.

    FAILS if a new lexical entry contaminates the AST-authoritative path.
    """
    a, b = op_slot(ADD_CODE), op_slot(SUB_CODE)
    assert abs(cos(a, b) + 1.0) <= 1e-12, cos(a, b)


def test_function_words_are_not_in_the_operator_lexicon():
    """`and`, `or`, `in`, `not` must not map to an operator root.

    FAILS if one of them is added. Each appears in most English task
    prose, so mapping them adds one shared feature to most docstrings:
    collision rises and the antipody above can no longer be exact.
    This test protects the measured reason recorded in the module.
    """
    for w in ("and", "or", "in", "not", "all", "any", "every", "multiple"):
        assert w not in oi.LEXICON, w


def test_lexicon_senses_are_internally_consistent():
    """Every value must be `(root, sign)` with sign in {-1, +1}.

    FAILS if a typo produces an unhashable or wrongly signed entry, which
    would silently place a feature on no valid axis.
    """
    for word, val in oi.LEXICON.items():
        assert isinstance(val, tuple) and len(val) == 2, (word, val)
        root, sign = val
        assert root in oi.OPERATOR_ROOTS, (word, root)
        assert sign in (-1, 1), (word, sign)
