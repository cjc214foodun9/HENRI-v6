"""Phase 11 Step 1: pin the pruned lexicon and the null-space gate.

WHY THIS FILE EXISTS
Phase 11 Step 1 makes two claims that a later refactor could silently
reverse: that the pruned lexicon stops firing on arithmetic prose, and that
the gated channel returns the base score UNTOUCHED when Slot 1 is active and
untied. Both are cheap to assert and expensive to lose, because a lost
invariant shows up as a quiet accuracy drop in a probe that runs for
seventeen minutes.

THE FOUR DEFECTS THESE TESTS PIN
1. Broad prose words firing a "structural" axis. Measured in Phase 10:
   `MORPH_COUNT_IDX` on 558/974 docstrings, `MORPH_MAP_DICT` on 384/974.
2. The AST side staying live after the word list is pruned. `len(x)` and
   `sum(x)` must contribute nothing, or the channel is dense in code exactly
   where it was pruned in prose.
3. `ast.Attribute` has no `keywords` attribute. A first version of
   `structural_morphology_from_ast` read keywords off the Attribute node, so
   `x.sort(reverse=True)` lost its polar flip and looked identical to
   `x.sort()`. Verified: `hasattr(node, "keywords")` is False.
4. A z-score over a constant pool divides by zero. The contract is
   "contribute nothing", not NaN.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from henri import operator_ingress as oi          # noqa: E402
from henri import reason_channels as rc           # noqa: E402

# Measured in Phase 10 Step 1 as the false-trigger set. Every one of these
# appears inside ordinary arithmetic problem statements.
ARITHMETIC_PROSE = (
    "count", "values", "find", "first", "last", "list", "set",
    "number", "array", "length", "index",
)

# The transformations the pruned lexicon must keep.
STRUCTURAL_PROSE = (
    "reverse", "sort", "flatten", "split", "join", "unique",
    "deduplicate", "uppercase", "dictionary", "tuple",
)


class TestPrunedLexicon:
    def test_arithmetic_prose_words_are_pruned(self):
        """Each arithmetic-prose word must fire NO structural feature."""
        fired = {}
        for w in ARITHMETIC_PROSE:
            feats = oi.structural_morphology_from_text(
                "write a function to %s the things" % w)
            if feats:
                fired[w] = feats
        assert not fired, fired

    def test_structural_words_are_retained(self):
        """Each structural transformation must fire at least one feature."""
        missing = []
        for w in STRUCTURAL_PROSE:
            feats = oi.structural_morphology_from_text(
                "write a function to %s the things" % w)
            if not feats:
                missing.append(w)
        assert not missing, missing

    def test_word_boundaries_hold(self):
        """`sort` must not fire inside `assorted`; `set` is gone entirely."""
        assert not oi.structural_morphology_from_text("assorted items")
        assert not oi.structural_morphology_from_text("a subset of them")

    def test_no_count_axis_in_pruned_table(self):
        """Axis 6 is absent by design: every one of its tokens is arithmetic."""
        roots = {root for root, _pm in oi.STRUCTURAL_ONLY_AXES.items()}
        assert oi.MORPH_COUNT_IDX not in roots

    def test_phase10_tables_untouched(self):
        """The pruned table is ADDITIVE. Phase 10 pins depend on this.

        `tests/test_phase10_structural_morphisms.py` pins Slot 1 and Slot 2
        byte-identical. Those pins only hold if MORPH_AXES and MORPH_WORDS
        still contain their original entries.
        """
        for w in ("count", "find", "values", "first", "last", "list", "set"):
            assert w in oi.MORPH_WORDS, w


class TestAstLexicon:
    def test_len_and_sum_contribute_nothing(self):
        """The AST-side trap: pruning words alone leaves these live."""
        for code in ("def f(x):\n    return len(x)",
                     "def f(x):\n    return sum(x)",
                     "def f(x):\n    return max(x)"):
            feats, ok = oi.structural_morphology_from_ast(code)
            assert ok
            assert feats == [], (code, feats)

    def test_sort_keeps_polar_flip(self):
        """`x.sort(reverse=True)` must differ from `x.sort()`.

        DEFECT PINNED: keywords live on the Call node, never on the
        Attribute node. Reading them off the Attribute silently dropped the
        flip, so the two programs scored identically.
        """
        plain, _ = oi.structural_morphology_from_ast(
            "def f(x):\n    x.sort()\n    return x")
        flip, _ = oi.structural_morphology_from_ast(
            "def f(x):\n    x.sort(reverse=True)\n    return x")
        assert plain == [(oi.MORPH_ORDER, -1, 1)]
        assert (oi.MORPH_ORDER, +1, 1) in flip

    def test_builtin_sorted_flip(self):
        flip, _ = oi.structural_morphology_from_ast(
            "def f(x):\n    return sorted(x, reverse=True)")
        assert (oi.MORPH_ORDER, +1, 1) in flip

    def test_reverse_slice_fires(self):
        feats, ok = oi.structural_morphology_from_ast(
            "def f(s):\n    return s[::-1]")
        assert ok
        assert (oi.MORPH_ORDER, +1, 1) in feats

    def test_syntax_error_is_distinguishable(self):
        """A bad parse must not be confused with 'no structure'."""
        feats, ok = oi.structural_morphology_from_ast("def f(:")
        assert (feats, ok) == ([], False)


class TestPooledZ:
    def test_constant_pool_returns_zeros(self):
        z = rc.pooled_z([0.7, 0.7, 0.7, 0.7])
        assert z.shape == (4,)
        assert not np.any(z)
        assert np.all(np.isfinite(z))

    def test_spread_pool_is_standardised(self):
        z = rc.pooled_z([1.0, 2.0, 3.0, 4.0])
        assert abs(float(z.mean())) < 1e-12
        assert abs(float(z.std()) - 1.0) < 1e-12

    def test_empty_pool(self):
        assert rc.pooled_z([]).size == 0


class TestPoolTied:
    def test_rounding_matches_the_sort_key(self):
        """A 13th-digit difference is a tie at 12 decimals, as the ranker sees it."""
        assert rc.pool_tied([0.5000000000001, 0.5000000000000, 0.1])
        assert not rc.pool_tied([0.51, 0.50, 0.1])

    def test_single_candidate_is_not_a_tie(self):
        assert not rc.pool_tied([0.9])


class TestNullspaceGate:
    def test_opens_when_slot1_silent(self):
        open_gate, why = rc.nullspace_gate([0.9, 0.4, 0.1], slot1_active=False)
        assert open_gate and why == "slot1_silent"

    def test_opens_on_primary_tie(self):
        open_gate, why = rc.nullspace_gate([0.5, 0.5, 0.1], slot1_active=True)
        assert open_gate and why == "primary_tie"

    def test_shuts_when_slot1_active_and_untied(self):
        open_gate, why = rc.nullspace_gate([0.9, 0.4, 0.1], slot1_active=True)
        assert not open_gate and why == "slot1_active_untied"


class TestGatedScores:
    BASE = np.array([0.900, 0.400, 0.100, 0.050])
    OPTS = ["reverse the list", "sort the list",
            "join the strings", "count them"]
    Q = "reverse the string"

    def test_weight_zero_returns_base_untouched(self):
        out, meta = rc.blended_scores_gated(
            self.BASE, self.Q, self.OPTS, 0.0, slot1_active=True)
        assert np.array_equal(out, self.BASE)
        assert meta["channel_added"] is False
        assert meta["gate"] == "parsimony_w0"

    def test_shut_gate_returns_base_untouched(self):
        """THE ZERO-DEGRADATION GUARANTEE, asserted structurally.

        On a slot1-active, untied pool the returned vector must equal the
        base vector EXACTLY. Not approximately: equal. This is the clause that
        makes arithmetic accuracy impossible to lose.
        """
        out, meta = rc.blended_scores_gated(
            self.BASE, self.Q, self.OPTS, 0.5, slot1_active=True)
        assert np.array_equal(out, self.BASE)
        assert meta["gate"] == "slot1_active_untied"
        assert meta["channel_added"] is False

    def test_open_gate_changes_scores(self):
        out, meta = rc.blended_scores_gated(
            self.BASE, self.Q, self.OPTS, 0.5, slot1_active=False)
        assert meta["channel_added"] is True
        assert meta["gate"] == "slot1_silent"
        assert not np.array_equal(out, self.BASE)

    def test_no_channel_signal_leaves_base(self):
        out, meta = rc.blended_scores_gated(
            self.BASE, "unrelated prompt", ["alpha", "beta", "gamma"],
            0.5, slot1_active=False)
        assert np.array_equal(out, self.BASE)
        assert meta["channel_added"] is False
        assert meta["gate"].endswith("_no_channel_signal")

    def test_zero_variance_leaves_base(self):
        """Equal channel scores across a live pool must contribute nothing.

        MEASURED CORRECTION: my first fixture was four IDENTICAL options
        (`["sort it"] * 4`). That never reaches the zero-variance branch --
        `contrastive` subtracts the option-mean, so duplicates cancel to zero
        first and `any_live` is already False. Measured: n_live 0.

        The reachable fixture is ORTHOGONAL options whose cosines against the
        question are all equal. Measured on the pair below: n_live 4,
        any_live True, scores [0.0, 0.0, 0.0, 0.0] -- so the spread really is
        zero while the channel really is live. That is the case the guard
        exists for.
        """
        opts = ["split the text", "join the text", "split it", "join it"]
        out, meta = rc.blended_scores_gated(
            self.BASE, self.Q, opts, 0.5, slot1_active=False)
        assert meta["z_zero_variance"] is True, meta["gate"]
        assert meta["channel_added"] is False
        assert np.array_equal(out, self.BASE)

    def test_identical_options_are_caught_by_any_live_first(self):
        """Duplicate options are rejected earlier, not by the spread guard.

        Worth pinning separately: the two guards are not interchangeable. A
        reader who assumes identical options exercise the zero-variance path
        will write a test that passes without testing it.
        """
        out, meta = rc.blended_scores_gated(
            self.BASE, self.Q, ["sort it"] * 4, 0.5, slot1_active=False)
        assert meta["z_zero_variance"] is False
        assert meta["gate"].endswith("_no_channel_signal")
        assert np.array_equal(out, self.BASE)

    def test_sole_live_option_cannot_be_scored(self):
        """A DESIGN LIMIT, recorded so nobody mistakes it for a bug.

        `contrastive` centres on the option-mean. With exactly one live
        option, that option minus itself is the zero vector, so the channel
        reports `any_live` False and the gate declines. Measured: n_live 0
        with a single structural candidate among inert ones.

        The consequence is honest: this channel needs at least TWO distinct
        structural candidates before it can rank. It never fabricates a
        direction for a lone candidate.
        """
        opts = ["def f(s):\n    return s[::-1]",
                "def f(s):\n    return s",
                "def f(s):\n    return len(s)"]
        st = rc.structural_channel_scores(self.Q, opts, options_are_code=True)
        assert st["n_live"] == 0
        assert st["any_live"] is False
        assert not np.any(st["score"])

    def test_code_options_use_the_ast_reader(self):
        """Candidates in the G-AST2 pool are Python, not prose.

        Two DISTINCT structural candidates are required for contrast -- see
        `test_sole_live_option_cannot_be_scored`. Here `s[::-1]` (ORDER +1)
        and `sorted(s)` (ORDER -1) oppose each other, so the channel
        discriminates at full magnitude.
        """
        opts = ["def f(s):\n    return s[::-1]",
                "def f(s):\n    return sorted(s)",
                "def f(s):\n    return len(s)"]
        st = rc.structural_channel_scores(
            "reverse the string", opts, options_are_code=True)
        assert st["any_live"] is True
        assert st["n_live"] == 2
        assert abs(float(st["score"][0]) - 1.0) < 1e-3
        assert abs(float(st["score"][1]) + 1.0) < 1e-3


class TestCacheContract:
    def test_repeat_calls_are_bit_identical(self):
        a = rc.structural_slot("reverse and sort this")
        b = rc.structural_slot("reverse and sort this")
        assert np.array_equal(a, b)

    def test_caller_mutation_cannot_poison_the_cache(self):
        first = rc.structural_slot("deduplicate the list")
        before = rc.structural_slot("deduplicate the list").copy()
        first += 99.0
        after = rc.structural_slot("deduplicate the list")
        assert np.array_equal(after, before), "cache handed out a live reference"
