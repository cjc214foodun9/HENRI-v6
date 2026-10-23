"""Phase 11: pin the tie-break repair so a refactor cannot undo it.

THE DEFECT THIS FILE PINS
=========================
`DigitalTwinAPI.rank_options` orders candidates by
`(-round(score, 12), digest_of_STRIPPED_text)`. `strip_common_option_tokens`
deletes tokens shared by >= ceil(K/2) options and present in the prompt. On
25 of 198 GPQA items that makes two options STRIPPED-IDENTICAL, so their
digests are equal AND their wave scores are equal (identical encoded input).
Two equal primary keys leave the sort no content rule; Python's stable sort
then keeps whatever order the caller supplied. MEASURED consequence before
the repair: rotating the option order changed the selected option on 14 of
198 GPQA items, and `gpqa_w0.2` rotation invariance read 0.950000 instead of
1.000000.

THE REPAIR
==========
The ORIGINAL option text stays distinct under rotation, so its digest is a
permutation-equivariant secondary key, placed before the stripped digest.
Scores are not touched; only the choice among exactly-tied scores changes.

WHY THIS TEST CAN FAIL
======================
Every assertion below is a property that a correct implementation has and a
regressed one lacks. `test_fixture_reaches_the_defect` runs FIRST in spirit:
if a future change to the stripper stops collapsing this fixture, the tie
never occurs and the invariance test would pass vacuously. That test fails
loudly in that case, so this file can never become a silent no-op.
"""
from __future__ import annotations

import numpy as np
import pytest

from henri.semantic_backbone import strip_common_option_tokens

# MEASURED fixture. The prompt names both tokens, so `ascending` and
# `descending` each appear in 2 of 4 options (>= ceil(4/2) = 2) AND in the
# prompt, so both are classified boilerplate and removed. Every option
# collapses to a single shared token.
Q_COLLIDE = "Sort the ascending list and the descending list"
OPTS_COLLIDE = ["ascending alpha", "descending alpha",
                "ascending beta", "descending beta"]


def _rotations(opts):
    out = []
    for r in range(len(opts)):
        idx = list(range(r, len(opts))) + list(range(0, r))
        out.append([opts[i] for i in idx])
    return out


def test_fixture_reaches_the_defect():
    """The fixture must actually produce duplicate stripped texts.

    Without this guard the invariance test below could pass because the tie
    never happens, which would be a gate that cannot fail. Measured value:
    2 distinct stripped texts of 4.
    """
    stripped = strip_common_option_tokens(Q_COLLIDE, OPTS_COLLIDE)
    assert len(stripped) == 4
    assert len(set(stripped)) == 2, (
        "fixture no longer collapses to 2 distinct stripped texts, got %r"
        % (stripped,))
    assert stripped[0] == stripped[1], (
        "the first two options must collide, got %r" % (stripped,))
    assert stripped[2] == stripped[3], (
        "the last two options must collide, got %r" % (stripped,))


@pytest.fixture(scope="module")
def api():
    from henri.api import DigitalTwinAPI
    return DigitalTwinAPI()


def test_colliding_options_tie_exactly(api):
    """Identical stripped text must produce an exactly zero margin.

    MEASURED CORRECTION: my first version read `payload["ranking"]`, a key
    the endpoint does not emit. It raised `KeyError: 'ranking'`. The real
    payload carries the top-2 scores as scalars `t` and `second`, plus
    `margin`. Measured on this fixture: `t == second == 0.576919369`,
    `margin == 0.0`. The tie is exact, so the ordering rule is consulted.
    """
    p = api.rank_options(Q_COLLIDE, OPTS_COLLIDE).payload
    assert {"t", "second", "margin", "selected_text"} <= set(p), \
        "payload schema changed: %r" % (sorted(p),)
    assert abs(p["t"] - p["second"]) < 1e-12, (
        "expected an exact top-2 tie, got t=%r second=%r"
        % (p["t"], p["second"]))
    assert abs(p["margin"]) < 1e-12, (
        "expected margin 0.0 for colliding options, got %r" % (p["margin"],))


def test_rotation_is_invariant_with_duplicate_stripped_text(api):
    """The repair's property: rotation must not change WHICH TEXT wins.

    Before the repair this returned a foreign option on some rotations
    because the sort fell back to insertion order.
    """
    base = api.rank_options(Q_COLLIDE, OPTS_COLLIDE).payload["selected_text"]
    for r, opts in enumerate(_rotations(OPTS_COLLIDE)):
        sel = api.rank_options(Q_COLLIDE, opts).payload["selected_text"]
        assert sel == base, (
            "rotation %d selected %r, expected %r (opts=%r)"
            % (r, sel, base, opts))


def test_rotation_is_invariant_at_every_channel_weight(api):
    """The property must hold with the additive channel engaged too."""
    for w in (0.0, 0.1, 0.2, 0.3):
        base = api.rank_options(
            Q_COLLIDE, OPTS_COLLIDE, channel_weight=w
        ).payload["selected_text"]
        for r, opts in enumerate(_rotations(OPTS_COLLIDE)):
            sel = api.rank_options(
                Q_COLLIDE, opts, channel_weight=w
            ).payload["selected_text"]
            assert sel == base, (
                "weight %.2f rotation %d selected %r, expected %r"
                % (w, r, sel, base))


def test_tie_break_key_uses_original_text_not_stripped(api):
    """The repair's mechanism: the tie is broken by ORIGINAL text digest.

    MEASURED CORRECTION: the first version read `payload["ranking"]`, which
    the endpoint does not emit. The grouping is reconstructed from the
    stripper instead. That is not a re-implementation of the rule: it is the
    same function the endpoint calls, so the only thing under test is which
    member of a genuinely tied group wins.

    Among options whose STRIPPED texts are identical (and which therefore
    tie on score), the winner must be the member with the smallest
    original-text sha256. A refactor that drops `orig_digests` makes the
    winner depend on caller order, and this assertion fails.
    """
    import hashlib

    stripped = strip_common_option_tokens(Q_COLLIDE, OPTS_COLLIDE)
    p = api.rank_options(Q_COLLIDE, OPTS_COLLIDE).payload
    win = p["selected_text"]
    i = OPTS_COLLIDE.index(win)
    group = [j for j, s in enumerate(stripped) if s == stripped[i]]
    assert len(group) > 1, (
        "winner %r shares no stripped text with another option, so the"
        " tie-break was never exercised; stripped=%r" % (win, stripped))
    expected = min(
        group,
        key=lambda j: hashlib.sha256(
            OPTS_COLLIDE[j].encode("utf-8")).hexdigest())
    assert OPTS_COLLIDE[expected] == win, (
        "tie group %r resolved to %r; original-digest rule predicts %r"
        % ([OPTS_COLLIDE[j] for j in group], win, OPTS_COLLIDE[expected]))
