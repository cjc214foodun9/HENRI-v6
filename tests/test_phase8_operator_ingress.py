"""Phase 8 ACTION 1 tests: operator and AST ingress. Every gate can fail.

G-AST1 as pre-registered in `henri/operator_ingress.py`, fixed before the
first run:

  (a) operator-slot cosine(gold, mutant) == -1.0 +/- 1e-12
  (b) the `+` program ranks index 0 by open-loop cosine
  (c) `tie_group_size` == 1   (Phase 7 measured 2 at 0.667495739)

The NEGATIVE CONTROLS below are the load-bearing part. A gate that cannot
fail is not a gate, and this repository has caught three such controls.

  control 1: with the operator channel REMOVED, the two programs must
             return to a tie. If the removal does not restore the tie,
             the operator channel is not what broke it.
  control 2: a query naming NO operator must not prefer either program.
             If it did, the preference would come from content leakage,
             not from the operator channel.
  control 3: the antipody must require OPPOSITE signs. Two different
             operator FAMILIES must NOT read -1.0, or the encoder is
             collapsing every operator onto one axis and the "exact
             opposition" is an artefact of that collapse.
"""

from __future__ import annotations

import numpy as np
import pytest

from henri import operator_ingress as oi
from henri.semantic_backbone import SLOT, SemanticBackbone

GOLD = "def add(a, b):\n    return a + b\n"
MUTANT = "def add(a, b):\n    return a - b\n"
QUERY_SUM = "return the sum of two numbers"
QUERY_DIFF = "return the difference of two numbers"
QUERY_NEUTRAL = "return a function of two values"


@pytest.fixture(scope="module")
def bb():
    return SemanticBackbone()


@pytest.fixture(scope="module")
def aware(bb):
    return oi.OperatorAwareBackbone(base=bb)


def _op(aware, text):
    feats, _ = oi.all_features(text)
    op, _ = oi.split_features(feats)
    return oi.ring_slot(op, aware.slot)


def _scores(aware, query, candidates):
    q = aware.encode(query).psi
    return np.array([float(np.real(np.vdot(q, aware.encode(c).psi)))
                     for c in candidates])


# -- (a) exact antipody --------------------------------------------------

def test_operator_pair_is_exact(aware):
    """G-AST1(a). Opposite operators are EXACTLY -1, not approximately."""
    c = oi.slot_cosine(_op(aware, GOLD), _op(aware, MUTANT))
    assert abs(c + 1.0) < 1e-12, f"measured {c!r}"


def test_same_operator_at_two_positions_is_exactly_one(aware):
    """Position is not part of an operator's identity.

    MEASURED DEFECT THIS PINS: my first draft applied `37*position` to
    every feature, which decorrelated `+` at line 1 from `+` at line 9.
    """
    a = "def f(x, y):\n    return x + y\n"
    b = ("def f(x, y):\n    p = 1\n    q = 2\n    r = 3\n\n"
         "    return x + y\n")
    c = oi.slot_cosine(_op(aware, a), _op(aware, b))
    assert abs(c - 1.0) < 1e-12, f"measured {c!r}"


def test_ast_is_authoritative_for_code(aware):
    """An IDENTIFIER must not inject an operator direction.

    MEASURED DEFECT THIS PINS: concatenating the AST and lexical streams
    made `def add(...): return a - b` carry both `+` (from the name `add`)
    and `-` (from `ast.Sub`), which cancelled to -0.7071 instead of -1.0.
    """
    feats, info = oi.all_features(MUTANT)
    assert info["channel"] == "ast"
    assert info["n_lexical_features"] == 0
    roots = [f[0] for f in feats]
    assert oi.ARITH_ADDITIVE in roots
    signs = [f[1] for f in feats if f[0] == oi.ARITH_ADDITIVE]
    assert signs == [-1], f"expected only the negative member, got {signs}"


def test_control_1_removing_operator_channel_restores_the_tie(aware):
    """CONTROL 1. A control that cannot fail is not a control.

    With the operator channel deleted, both programs must tie again,
    because they share every identifier and structure node. If this does
    NOT tie, the separation is coming from somewhere other than the
    operator channel and gate (a) proves nothing.
    """
    a = oi.OperatorAwareBackbone(base=aware.base, weights=(0.5, 0.5, 0.0, 0.0))
    s = _scores(a, QUERY_SUM, [GOLD, MUTANT])
    assert int(np.isclose(s, s.max(), atol=1e-12).sum()) == 2, \
        f"channel removed but scores differ: {s!r}"


def test_control_3_different_family_is_not_antipodal(aware):
    """CONTROL 3. Antipody must need OPPOSITE SIGNS, not merely difference.

    `+` versus `*` are different ROOTS, so they must NOT read -1.0. If
    they did, every operator would be collapsing onto one axis.
    """
    mult = "def mul(a, b):\n    return a * b\n"
    c = oi.slot_cosine(_op(aware, GOLD), _op(aware, mult))
    assert c > -0.9, f"different families collapsed onto one axis: {c!r}"


# -- (b) correct ranking --------------------------------------------------

def test_gold_ranks_first_on_the_phase7_fixture(aware):
    """G-AST1(b). The correct program ranks index 0 BEFORE execution."""
    s = _scores(aware, QUERY_SUM, [GOLD, MUTANT])
    assert int(np.argmax(s)) == 0, f"scores {s!r}"


def test_control_2_neutral_query_prefers_neither(aware):
    """CONTROL 2. A query naming no operator must not pick a winner.

    If it did, the preference would come from content, not operators.
    """
    s = _scores(aware, QUERY_NEUTRAL, [GOLD, MUTANT])
    o = _op(aware, QUERY_NEUTRAL)
    assert float(np.linalg.norm(o)) == 0.0, "neutral query has no operator"
    assert abs(s[0] - s[1]) < 1e-12, f"neutral query discriminated: {s!r}"


def test_difference_query_prefers_the_minus_program(aware):
    """SPECIFICITY. The channel must point both ways, not just favour `+`."""
    s = _scores(aware, QUERY_DIFF, [GOLD, MUTANT])
    assert int(np.argmax(s)) == 1, f"scores {s!r}"


# -- (c) tie group -------------------------------------------------------

def test_tie_group_size_is_one(aware):
    """G-AST1(c). Phase 7 measured 2 at wave_score 0.667495739."""
    s = _scores(aware, QUERY_SUM, [GOLD, MUTANT])
    group = int(np.isclose(s, s.max(), atol=1e-12).sum())
    assert group == 1, f"scores {s!r}"


# -- invariants that must NOT regress -----------------------------------

def test_phase5_antonym_coherence_survives(bb):
    """The -1.0000 antonym path must be untouched by the new module."""
    a, b = "the rate increases", "the rate decreases"
    c = float(np.real(np.vdot(bb.encode(a).psi, bb.encode(b).psi)))
    assert abs(c + 1.0) < 1e-9, f"measured {c!r}"


def test_slots_two_and_three_stay_zero_in_the_base_backbone(bb):
    """`SemanticBackbone` is unmodified, so its invariant still holds."""
    e = bb.encode("entropy always increases")
    assert float(np.max(np.abs(e.psi[2 * SLOT:]))) == 0.0
    assert e.active_slots == (0, 1)


def test_operator_aware_wave_is_unit_norm(aware):
    for text in (GOLD, MUTANT, QUERY_SUM, QUERY_NEUTRAL):
        assert abs(float(np.linalg.norm(aware.encode(text).psi)) - 1.0) < 1e-12


def test_operator_aware_slot_energies_match_the_weights(aware):
    """DERIVED. Each populated slot must carry exactly its declared weight."""
    e = aware.encode(GOLD)
    for idx, w in enumerate(aware.weights):
        energy = float(np.linalg.norm(e.psi[idx * aware.slot:
                                            (idx + 1) * aware.slot]) ** 2)
        assert abs(energy - w) < 1e-9, (idx, energy, w)


def test_prose_without_operators_yields_a_zero_slot(aware):
    """No fabrication: absent operator evidence is exact zero, not a guess."""
    e = aware.encode("the cat sat on the mat")
    assert float(np.linalg.norm(e.slot_operator)) == 0.0
    assert 2 not in e.active_slots


def test_lexicon_is_symmetric_and_each_root_has_both_signs():
    """DERIVED. Every antipodal root must carry at least one +1 and one -1.

    A root with one sign only cannot express opposition, so the half-turn
    would be unusable on that family.
    """
    by_root: dict = {}
    for word, (root, sign) in oi.LEXICON.items():
        by_root.setdefault(root, set()).add(sign)
    single = {r: s for r, s in by_root.items() if len(s) < 2}
    # These families are single-direction by construction: they name a
    # construct, not an opposition. Listed so the exception is explicit.
    allowed = {oi.CARDINALITY, oi.PARITY, oi.PRIMALITY, oi.SORT_ORDER}
    unexpected = {r: s for r, s in single.items() if r not in allowed}
    assert not unexpected, f"root with one sign only: {unexpected}"


def test_parse_failure_falls_back_to_the_lexicon():
    """A docstring is not Python; the fallback must engage, not raise."""
    feats, info = oi.all_features("sum these values, then subtract one")
    assert info["parsed"] is False
    assert info["channel"] == "lexical"
    roots = {f[0] for f in feats}
    assert oi.ARITH_ADDITIVE in roots
    signs = {f[1] for f in feats if f[0] == oi.ARITH_ADDITIVE}
    assert signs == {+1, -1}


def test_determinism_across_processes():
    """Bit-for-bit. `hash()` is randomised per process; blake2b is not."""
    import subprocess
    import sys
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    code = (
        "import sys; sys.path.insert(0, r'%s');"
        "import numpy as np;"
        "from henri import operator_ingress as oi;"
        "f,_=oi.all_features(%r);"
        "v=oi.ring_slot(f, 1024);"
        "print(np.abs(v[:8]).round(12).tolist(), v[0].real)"
    ) % (str(repo), GOLD)
    outs = []
    for seed in ("0", "12345"):
        p = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, env={"PYTHONHASHSEED": seed,
                                           "PATH": ""})
        outs.append(p.stdout.strip())
    assert outs[0] and outs[0] == outs[1], outs
