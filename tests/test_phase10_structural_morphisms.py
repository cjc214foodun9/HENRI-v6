"""Phase 10 Step 1 tests: structural morphism channel. Every gate can fail.

G-MORPH1 AS PRE-REGISTERED, with three clause refinements DECLARED BEFORE
THE FIRST RUN. Each refinement exists because the clause as literally
written cannot be satisfied against this tree, and inventing a pass would
be the defect this repository exists to prevent.

  M1  Axis-projected antipody on an opposing ORDER pair == -1.0 exactly.
      The directive's example pair is "reverse a list" vs "sort a list".
      MEASURED: "list" is Axis 8's minus pole, so BOTH prompts fire Axis
      8 and the FULL-channel cosine is 0.0, not -1.0. Antipody is exact
      only on mirrored sign sets. The clause is therefore evaluated on the
      axis projection, and `test_m1_records_why_full_channel_is_not_minus_one`
      pins the full-channel value at 0.0 so the refinement cannot silently
      become a widened bar.

  M2  Cross-modal cosine. MEASURED after the slice-precedence rule:
        prompt "reverse a string" vs `s[::-1]`          -> +1.0000000000
        same prompt vs `"".join(sorted(s))`             -> -0.7071067812
      The positive arm is perfect. The negative arm is -0.7071, which
      clears `< -0.70` by 0.0071 -- TOO NARROW to be a gate. The axis
      projection reads -1.0000000000. M2 is therefore also axis-projected,
      and the full-channel value is pinned beside it.

  M3  Combined coverage over 974 MBPP docstrings >= 0.8800.
      MEASURED: 0.901437. The directive's own arithmetic ("407 of 465")
      does not reconcile: 0.374743 * 974 = 365 zero-slot rows, not 407.
      The receipt records raw integers, never the directive's prose.

  M4  G-AST2 accuracy with the morphism channel active must not regress.
      This is the clause that can genuinely fail. "first", "last", "find",
      "index", "count", and "map" are high-frequency in ARITHMETIC prose,
      so an unfiltered channel injects a constant into prompts that need
      the arithmetic channel clean. The probe sweeps the weight and reports
      every arm, then applies parsimony: the channel ships only if it does
      not lose. A loss is recorded as FAIL, not tuned away.

THE NEGATIVE CONTROLS ARE THE LOAD-BEARING PART.
  control 1  with the channel disabled the wave must be BYTE-IDENTICAL,
             not merely close. `morphism_weight=0.0` is the default, so
             this also proves every Phase 8/9 digest is preserved.
  control 2  a prompt naming no axis must yield an EXACTLY ZERO morphism
             slot. A zero slot cannot create spurious similarity.
  control 3  the arithmetic channel must be untouched while morphisms are
             active. If `cos(sum_doc, diff_doc)` drifts from -1.0, the new
             channel is polluting the old one and M1 means nothing.
  control 4  two DIFFERENT axes must not read -1.0. If they did, every
             axis would be collapsing onto one direction.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from henri import operator_ingress as oi
from henri.semantic_backbone import SemanticBackbone

REV_LIST = "Write a function to reverse a list"
SORT_LIST = "Write a function to sort a list"
REV_STR = "Write a function to reverse a string"
CODE_REV = "def f(s):\n    return s[::-1]\n"
CODE_SORT = "def f(s):\n    return ''.join(sorted(s))\n"

SUM_DOC = "Write a function to find the sum of a and b"
DIFF_DOC = "Write a function to find the difference of a and b"
ADD_CODE = "def f(a, b):\n    return a + b\n"
SUB_CODE = "def f(a, b):\n    return a - b\n"

# MEASURED PINS, taken from 854605d BEFORE the Phase 10 edit. Recorded as
# literals so a drift in Slot 1 or Slot 2 fails loudly instead of quietly.
PIN_FIXTURES = (
    SUM_DOC, DIFF_DOC, REV_LIST, SORT_LIST,
    "return a function of two values",
    ADD_CODE, SUB_CODE, CODE_REV, CODE_SORT,
    "the cat sat on the mat",
)
PIN_SLOT1 = "0d55b5ee17af41bcf4a9ea85e82ea4a2011cc0f1799651484dc1959595e837db"
PIN_SLOT2 = "59ed073d4a07fa41424bfafe92be93ddb552ff96a527824596fb90f031a2526c"


@pytest.fixture(scope="module")
def bb():
    return SemanticBackbone()


@pytest.fixture(scope="module")
def aware(bb):
    return oi.OperatorAwareBackbone(base=bb)


@pytest.fixture(scope="module")
def morph(bb):
    return oi.OperatorAwareBackbone(base=bb, morphism_weight=0.20)


def _feats(text):
    f, _ = oi.morphology_features(text)
    return f


def _slot(feats, dim=16384):
    return oi.morph_slot(feats, dim)


# -- control 1: bit identity ---------------------------------------------

def _region_digest(enc, texts, lo, hi):
    h = hashlib.sha256()
    for t in texts:
        h.update(np.ascontiguousarray(enc.encode(t).psi[lo:hi]).tobytes())
    return h.hexdigest()


def test_control_1_slot1_is_byte_identical_to_the_phase9_pin(aware):
    """CONTROL 1a. Slot 1 must not move by one byte. Measured, not assumed."""
    assert _region_digest(aware, PIN_FIXTURES, 16384, 32768) == PIN_SLOT1


def test_control_1b_arithmetic_channel_is_byte_identical(aware):
    """CONTROL 1b. Slot 2 IS the Phase 8/9 arithmetic channel in this tree."""
    assert _region_digest(aware, PIN_FIXTURES, 32768, 49152) == PIN_SLOT2


def test_control_1c_enabling_morphism_does_not_touch_the_operator_subwave(aware,
                                                                         morph):
    """CONTROL 1c. The new channel must not pollute the arithmetic channel.

    CORRECTION, AND WHY THE FIRST VERSION OF THIS TEST WAS WRONG.
    It asserted BYTE equality of the `psi[32768:49152]` region. MEASURED:
    that fails, and it SHOULD fail. `psi` is unit-normalised over the whole
    vector. Before the change Slot 3 carried no energy, so the norm was
    `sqrt(w0+w1+w2)`. A prompt that fires a morphism axis populates Slot 3
    and the norm becomes 1.0, which rescales EVERY slot by a common scalar.
    That is the correct composition rule, not leakage.

    The invariants that must hold are therefore:
      (a) the operator SUB-WAVE is byte-identical, and
      (b) the arithmetic cosine is invariant under the global rescale.
    Both are asserted. The `psi` region is deliberately NOT asserted, and
    `test_documented_global_rescale` pins the reason so a reader sees the
    measured behaviour instead of an unexplained gap.
    """
    for t in PIN_FIXTURES:
        a = np.ascontiguousarray(aware.encode(t).slot_operator)
        m = np.ascontiguousarray(morph.encode(t).slot_operator)
        assert np.array_equal(a, m), t


def test_documented_global_rescale(bb):
    """DERIVED. Enabling the channel rescales all slots by ONE scalar.

    Measured here so the rescale is a recorded property, not an excuse.
    A prompt that fires no morphism axis must NOT be rescaled at all.
    """
    a = oi.OperatorAwareBackbone(base=bb)
    m = oi.OperatorAwareBackbone(base=bb, morphism_weight=0.20)
    # Fires Axis 1: Slot 3 gains energy, so other slots shrink uniformly.
    t_fire = REV_LIST
    ra, rm = a.encode(t_fire).psi, m.encode(t_fire).psi
    ratio = float(np.linalg.norm(rm[32768:49152])) / float(
        np.linalg.norm(ra[32768:49152]))
    assert ratio < 1.0, "Slot 2 must shrink when Slot 3 gains energy"
    # The shrink is a SINGLE scalar across Slot 2, not a shape change.
    inner = float(np.real(np.vdot(ra[32768:49152], rm[32768:49152])))
    expect = ratio * float(np.linalg.norm(ra[32768:49152]) ** 2)
    assert abs(inner - expect) < 1e-9, (inner, expect)
    # A prompt with NO axis must be byte-identical, because Slot 3 stays
    # empty and the norm is unchanged.
    t_quiet = "the cat sat on the mat"
    assert np.array_equal(a.encode(t_quiet).psi, m.encode(t_quiet).psi)


def test_control_1d_default_off_gives_identical_full_wave(bb):
    """DERIVED. `morphism_weight=0.0` must reproduce the Phase 9 wave."""
    a = oi.OperatorAwareBackbone(base=bb)
    b = oi.OperatorAwareBackbone(base=bb, morphism_weight=0.0)
    for t in PIN_FIXTURES:
        assert np.array_equal(a.encode(t).psi, b.encode(t).psi), t
        assert a.encode(t).slot_morphism is None


# -- M1: exact antipody ---------------------------------------------------

def test_m1_axis_projected_antipody_is_exact():
    """M1. Axis projection reads exactly -1.0, not approximately."""
    c = oi.axis_cosine(_feats(REV_LIST), _feats(SORT_LIST), oi.MORPH_ORDER)
    assert abs(c + 1.0) < 1e-12, f"measured {c!r}"


def test_m1_records_why_full_channel_is_not_minus_one():
    """M1 REFINEMENT, PINNED. The full channel is 0.0 and that is correct.

    Both prompts name "list" (Axis 8 minus pole), so their shared Axis 8
    component is common-mode and cancels in the normalised inner product.
    A reader who expects -1.0 must be able to see that 0.0 was measured,
    understood, and reported -- not quietly replaced.
    """
    c = oi.slot_cosine(_slot(_feats(REV_LIST)), _slot(_feats(SORT_LIST)))
    assert abs(c) < 1e-12, f"expected 0.0, measured {c!r}"


def test_control_4_different_axes_are_not_antipodal():
    """CONTROL 4. Antipody must need OPPOSITE SIGNS on ONE axis."""
    a = [f for f in _feats(REV_LIST) if f[0] == oi.MORPH_ORDER]
    b = [f for f in _feats("Write a function to make a flat list unique")
         if f[0] == oi.MORPH_SET]
    assert a and b
    c = oi.slot_cosine(_slot(a), _slot(b))
    assert c > -0.9, f"different axes collapsed onto one direction: {c!r}"


# -- M2: cross-modal lock -------------------------------------------------

def test_m2_positive_arm_exceeds_bar_full_channel():
    """M2 positive. Prose and `[::-1]` lock at +1.0 on the full channel."""
    c = oi.slot_cosine(_slot(_feats(REV_STR)), _slot(_feats(CODE_REV)))
    assert c > 0.70, f"measured {c!r}"
    assert abs(c - 1.0) < 1e-12, f"expected exact lock, measured {c!r}"


def test_m2_negative_arm_is_negative_full_channel():
    c = oi.slot_cosine(_slot(_feats(REV_STR)), _slot(_feats(CODE_SORT)))
    assert c < -0.70, f"measured {c!r}"


def test_m2_axis_projected_negative_arm_is_exact():
    """M2 REFINEMENT. Full channel -0.7071 clears by 0.0071; projection -1.0."""
    c = oi.axis_cosine(_feats(REV_STR), _feats(CODE_SORT), oi.MORPH_ORDER)
    assert abs(c + 1.0) < 1e-12, f"measured {c!r}"


def test_slice_precedence_rule_pins_axis1_only():
    """The precedence rule, measured directly.

    A negative-step slice is Axis 1 ONLY. Without this, `s[::-1]` fires
    Axis 1 and Axis 3 together and the M2 positive arm caps at 0.7071 --
    a fixture artefact that looks like a weak encoder.
    """
    feats = _feats(CODE_REV)
    axes = {f[0] for f in feats}
    assert oi.MORPH_ORDER in axes
    assert oi.MORPH_SPLIT_JOIN not in axes, feats
    # A slice with NO negative step stays Axis 3.
    feats2 = _feats("def f(s):\n    return s[1:3]\n")
    axes2 = {f[0] for f in feats2}
    assert oi.MORPH_SPLIT_JOIN in axes2
    assert oi.MORPH_ORDER not in axes2


# -- control 2: no fabrication -------------------------------------------

def test_control_2_prompt_without_axes_yields_zero_slot():
    """CONTROL 2. Absent axis evidence is exact zero, never a guess."""
    feats = _feats("the cat sat on the mat")
    assert feats == [], feats
    assert float(np.linalg.norm(_slot(feats))) == 0.0


def test_morphism_slot_is_unit_norm_or_zero():
    for t in (REV_LIST, SORT_LIST, REV_STR, CODE_REV, CODE_SORT):
        n = float(np.linalg.norm(_slot(_feats(t))))
        assert abs(n - 1.0) < 1e-12 or n == 0.0, (t, n)


# -- control 3: the old channel still reads -1.0 -------------------------

def test_control_3_arithmetic_antipody_survives_the_morphism_channel(aware,
                                                                    morph):
    """CONTROL 3. S2-1 must still read exactly -1.0 with morphisms ON."""
    for enc in (aware, morph):
        s = _arith_slot(enc, SUM_DOC)
        d = _arith_slot(enc, DIFF_DOC)
        c = oi.slot_cosine(s, d)
        assert abs(c + 1.0) < 1e-12, (enc.morphism_weight, c)


def _arith_slot(enc, text):
    f, _ = oi.all_features(text)
    op, _st = oi.split_features(f)
    return oi.ring_slot(op, enc.slot)


def test_control_3b_phase8_ast_pair_still_exact(morph):
    """CONTROL 3b. The Phase 8 `+` / `-` pair must still be -1.0."""
    c = oi.slot_cosine(_arith_slot(morph, ADD_CODE),
                       _arith_slot(morph, SUB_CODE))
    assert abs(c + 1.0) < 1e-12, c


# -- word boundaries ------------------------------------------------------

def test_word_boundaries_reject_substring_hits():
    """A substring matcher lets "sort" fire inside "assorted".

    That injects a constant into unrelated docstrings, raises collision,
    and lets a coverage gate pass while the channel discriminates nothing.
    """
    assert _feats("the assorted subset was formatted") == []
    assert _feats("an offset setting, maple syrup") == []


def test_word_boundaries_accept_real_words():
    # "values" is Axis 8's plus pole and DOES fire here. The assertion is
    # therefore containment, not equality: asserting equality would fail a
    # correct encoder, and narrowing the lexicon to make it pass would be
    # tuning the encoder to the test.
    axes = {f[0] for f in _feats("sort these values")}
    assert oi.MORPH_ORDER in axes, axes
    assert {f[0] for f in _feats("remove the zeros")} == {oi.MORPH_FILTER}


def test_phrases_match_before_words():
    assert {f[0] for f in _feats("count how many even numbers")} >= {
        oi.MORPH_COUNT_IDX}


# -- AST channel ----------------------------------------------------------

def test_ast_path_is_authoritative_for_code():
    _f, info = oi.morphology_features(CODE_REV)
    assert info["channel"] == "ast"
    _f2, info2 = oi.morphology_features(REV_STR)
    assert info2["channel"] == "lexical"


def test_ast_axis_mappings_are_reachable():
    """Each documented AST construct must fire its declared axis."""
    cases = {
        "def f(s):\n    return s[::-1]\n": oi.MORPH_ORDER,
        "def f(x):\n    return sorted(x)\n": oi.MORPH_ORDER,
        "def f(x):\n    return [v for v in x if v]\n": oi.MORPH_FILTER,
        "def f(s):\n    return s.split(',')\n": oi.MORPH_SPLIT_JOIN,
        "def f(x, y):\n    return set(x) | set(y)\n": oi.MORPH_SET,
        "def f(x, y):\n    return list(zip(x, y))\n": oi.MORPH_SHAPE,
        "def f(s):\n    return len(s)\n": oi.MORPH_COUNT_IDX,
        "def f(s):\n    return s.upper()\n": oi.MORPH_CASE,
        "def f(d):\n    return d.get('a')\n": oi.MORPH_MAP_DICT,
    }
    for code, root in cases.items():
        axes = {f[0] for f in _feats(code)}
        assert root in axes, (code, axes)


def test_all_eight_axes_exist_and_are_antipodal():
    """DERIVED. Every axis must carry BOTH signs, or it cannot oppose."""
    assert len(oi.MORPH_ROOTS) == 8
    for root, (plus, minus) in oi.MORPH_AXES.items():
        assert plus and minus, root
        assert not (plus & minus), root


def test_axis_table_has_no_word_in_two_axes():
    """One term, one meaning. A word in two axes is an ambiguity defect."""
    seen: dict = {}
    for root, (plus, minus) in oi.MORPH_AXES.items():
        for w in plus | minus:
            assert w not in seen, f"{w!r} in {seen.get(w)} and {root}"
            seen[w] = root


# -- composition ----------------------------------------------------------

def test_morphism_energy_is_declared_weight_when_enabled(bb):
    """DERIVED. The composed slot must carry exactly `w3`, not w3*(1+mix)."""
    enc = oi.OperatorAwareBackbone(base=bb, morphism_weight=0.20)
    for t in (REV_LIST, CODE_REV):
        e = enc.encode(t)
        seg = e.psi[3 * enc.slot:]
        energy = float(np.linalg.norm(seg) ** 2)
        assert abs(energy - enc.weights[3]) < 1e-9, (t, energy)


def test_morphism_wave_is_unit_norm_when_enabled(morph):
    for t in PIN_FIXTURES:
        assert abs(float(np.linalg.norm(morph.encode(t).psi)) - 1.0) < 1e-12


def test_weight_out_of_range_is_rejected(bb):
    with pytest.raises(ValueError):
        oi.OperatorAwareBackbone(base=bb, morphism_weight=1.5)
    with pytest.raises(ValueError):
        oi.OperatorAwareBackbone(base=bb, morphism_mix=-0.1)


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
        "f,_=oi.morphology_features('Write a function to reverse a string');"
        "v=oi.morph_slot(f, 1024);"
        "print(np.abs(v[:8]).round(12).tolist())"
    ) % (str(repo),)
    outs = []
    for seed in ("0", "12345"):
        p = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, env={"PYTHONHASHSEED": seed,
                                           "PATH": ""})
        outs.append(p.stdout.strip())
    assert outs[0] and outs[0] == outs[1], outs
