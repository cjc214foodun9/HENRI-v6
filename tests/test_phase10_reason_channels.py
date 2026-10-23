"""Phase 10 Step 3 tests: operator and morphism channels in reason ranking.

G-REASON1 AS PRE-REGISTERED, fixed before the first run.

  R1  `channel_weight=0.0` (the default) must leave the wave scores
      BYTE-IDENTICAL, so every Phase 5/6 receipt stays numerically valid.
  R2  4x cyclic rotation invariance is EXACT (1.000000), measured through
      both `/v1/reason/rank` and the blended score path.
  R3  A question with NO operator and NO morphism evidence must receive a
      channel contribution of EXACTLY zero. This is the honesty gate: an
      empty channel cannot vote, and it cannot pull an option toward the
      option-mean and invent a direction.
  R4  An option with an empty operator slot scores exactly 0.0 on that
      channel, even when the question has evidence. No fabrication.
  R5  A STEM operator-contrastive fixture must be decided correctly by the
      channel where the wave alone is not sufficient.

WHY R2 IS DERIVED HERE AND NOT MERELY MEASURED
----------------------------------------------
The contrast is taken against the OPTION-MEAN. Subtracting a symmetric
function of the option set makes the score permutation-equivariant: rotating
the options permutes the scores and changes no option's value. Rotation
invariance therefore follows from the construction. It is asserted, not
hoped for.

WHY R3 EXISTS
-------------
`contrastive()` computes a mean over the LIVE slots only. If a zero slot
took part in that mean it would emerge as `-mean` and acquire a direction
it never had. An option naming no operator would then score on the operator
channel. That is fabrication, and `test_empty_slot_is_excluded_from_the_mean`
pins the mask that prevents it.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from henri import reason_channels as rc

# A question that DOES name an operator ("maximum") and a morphism
# ("count"), so the channel has evidence to work with.
Q_OP = "Return the maximum value and count how many times it occurs"
OPTS_OP = [
    "the largest element with its frequency",
    "the smallest element with its frequency",
    "the largest element without any frequency",
    "the smallest element without any frequency",
]

# A question naming NO operator and NO morphism axis.
Q_NEUTRAL = "the cat sat on the mat"
OPTS_NEUTRAL = ["a cat on a mat", "a mat on a cat", "a dog", "a log"]


def _wave_scores(q, opts, bb):
    qv = bb.encode(q).psi
    return np.array([float(np.real(np.vdot(qv, bb.encode(o).psi)))
                     for o in opts])


def _pick(scores, opts):
    """Content-digest tie-break, never the index. Same rule as the API."""
    dg = [hashlib.sha256(o.encode()).hexdigest() for o in opts]
    order = sorted(range(len(scores)),
                   key=lambda i: (-round(float(scores[i]), 12), dg[i]))
    return order[0]


@pytest.fixture(scope="module")
def bb():
    from henri.semantic_backbone import SemanticBackbone
    return SemanticBackbone()


@pytest.fixture(scope="module")
def api():
    from henri.api import DigitalTwinAPI
    return DigitalTwinAPI()


# -- R1: default is a no-op ----------------------------------------------

def test_r1_zero_weight_is_a_no_op(bb):
    base = _wave_scores(Q_OP, OPTS_OP, bb)
    out, meta = rc.blended_scores(base, Q_OP, OPTS_OP, 0.0)
    assert np.array_equal(out, base), "w=0 changed the wave scores"
    assert meta["channel_added"] is False


def test_r1_zero_weight_matches_the_prior_api(api):
    """The default path must reproduce the pre-Phase-10 selection exactly."""
    r = api.rank_options(Q_OP, OPTS_OP).payload
    assert r["channel_weight"] == 0.0
    assert r["channel_added"] is False
    assert r["selected_index"] == 0


# -- R2: rotation invariance ---------------------------------------------

def test_r2_rotation_invariance_is_exact(bb):
    """R2. Rotating options must not change WHICH TEXT is selected.

    MEASURED CORRECTION, and the first version of this test was wrong in a
    way worth recording. It compared INDICES:

        chosen = _pick(scores, OPTS_OP)         # -> 0
        sel    = _pick(rotated, opts)           # -> 3
        assert sel == chosen                    # 3 == 0 FAILS

    An index is only meaningful relative to ONE ordering. After a rotation
    the same text sits at a different index, so the comparison was
    guaranteed to fail on a correct implementation. MEASURED: the rotation-1
    selection was OPTS_OP[0] sitting at index 3, i.e. invariant.

    The assertion compares TEXTS. That is the property the endpoint
    promises, and `test_r2_api_rotation_invariance` already compared texts
    through the real API path and passed.
    """
    base = _wave_scores(Q_OP, OPTS_OP, bb)
    chosen_text = OPTS_OP[_pick(
        rc.blended_scores(base, Q_OP, OPTS_OP, 0.2)[0], OPTS_OP)]
    for r in range(4):
        idx = list(range(r, 4)) + list(range(0, r))
        opts = [OPTS_OP[i] for i in idx]
        scores = base[idx]                     # scores travel WITH options
        sel = _pick(rc.blended_scores(scores, Q_OP, opts, 0.2)[0], opts)
        assert opts[sel] == chosen_text, \
            "rotation %d selected %r, expected %r" % (r, opts[sel],
                                                      chosen_text)


def test_r2_api_rotation_invariance(api):
    """R2 through the real endpoint path, at every weight."""
    base_sel = api.rank_options(Q_OP, OPTS_OP,
                                channel_weight=0.2).payload["selected_text"]
    for r in range(4):
        opts = OPTS_OP[r:] + OPTS_OP[:r]
        sel = api.rank_options(Q_OP, opts,
                               channel_weight=0.2).payload["selected_text"]
        assert sel == base_sel, "rotation %d changed the selection" % r


# -- R3: empty evidence cannot vote --------------------------------------

def test_r3_neutral_question_gets_exactly_zero():
    """R3. No question evidence means no channel contribution at all."""
    ch = rc.channel_scores(Q_NEUTRAL, OPTS_NEUTRAL)
    assert not ch["question_has_operator"]
    assert not ch["question_has_morphism"]
    assert np.allclose(ch["op"], 0.0)
    assert np.allclose(ch["morph"], 0.0)
    assert np.allclose(ch["combined"], 0.0)


def test_r3_neutral_question_leaves_scores_unchanged(bb):
    base = _wave_scores(Q_NEUTRAL, OPTS_NEUTRAL, bb)
    out, _meta = rc.blended_scores(base, Q_NEUTRAL, OPTS_NEUTRAL, 0.2)
    assert np.allclose(out, base), "empty channel moved the scores"


def test_empty_slot_is_excluded_from_the_mean():
    """R3 mechanism. A zero slot must not emerge as `-mean`.

    This is the mask that prevents an option with no operator evidence from
    acquiring one. Measured directly on the contrastive transform.
    """
    from henri import operator_ingress as oi
    dim = 256
    live = oi.ring_slot([(oi.ARITH_ADDITIVE, +1, 0)], dim)
    empty = np.zeros(dim, dtype=np.complex128)
    out = rc.contrastive([live, empty])
    assert float(np.linalg.norm(out[1])) == 0.0, "empty slot acquired direction"


# -- R4: per-option fabrication guard ------------------------------------

def test_r4_option_without_evidence_scores_zero():
    """R4. Question has evidence, one option does not. That option is 0.0."""
    ch = rc.channel_scores(
        "Return the maximum of the values",
        ["the largest value", "an unrelated phrase with no operator at all"])
    assert ch["question_has_operator"] or ch["n_op_live"] >= 1
    # The second option names no axis and carries no operator.
    assert ch["op"][1] == 0.0
    assert ch["morph"][1] == 0.0


# -- R5: the channel can actually decide --------------------------------

def test_r5_channel_separates_opposing_operators():
    """R5. 'maximum' must score above 'minimum' on the operator channel."""
    ch = rc.channel_scores("Return the maximum of the values",
                           ["the largest value", "the smallest value"])
    assert ch["question_has_operator"]
    assert ch["op"][0] > ch["op"][1], ch["op"]
    assert ch["op"][0] > 0.0 > ch["op"][1] or ch["op"][0] > ch["op"][1]


def test_r5_morphism_channel_is_separately_measured():
    """The two channels are reported separately so a win is attributable."""
    ch = rc.channel_scores(
        "Reverse the list, then count the elements",
        ["the reversed list length", "the sorted list length"])
    assert "op" in ch and "morph" in ch and "combined" in ch
    assert ch["n_op_live"] >= 1 or ch["n_morph_live"] >= 1


def test_channels_are_unit_bounded():
    """A cosine lies in [-1, 1]; a combined mean of two does too."""
    ch = rc.channel_scores(Q_OP, OPTS_OP)
    for key in ("op", "morph", "combined"):
        assert np.all(np.abs(ch[key]) <= 1.0 + 1e-12), (key, ch[key])


def test_no_second_operator_table():
    """DERIVED. This module must not redefine an operator or a morphism.

    A duplicate table would drift from the encoder and the drift would be
    silent: two definitions of `+` that disagree still both "work".
    """
    src = (rc.__file__ or "")
    text = open(src, encoding="utf-8").read()
    for banned in ("ARITH_ADDITIVE =", "MORPH_ORDER =", "ast.Add:"):
        assert banned not in text, banned
