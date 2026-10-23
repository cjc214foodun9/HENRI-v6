"""Contrastive operator and morphism channels for multiple-choice ranking.

WHY THIS MODULE EXISTS
----------------------
`/v1/reason/rank` and `benchmarks/wave_rank.py` score options with the
semantic backbone only. A question about "proportional" versus "inversely
proportional" can therefore be decided by stem words that both options
share, because the operator distinction carries no weight at all.

Phase 8/9 built that distinction for CODE (`operator_ingress`). Phase 10
Step 1 built the structural half (`morphology_features`). Neither was wired
into the multiple-choice path. This module wires them.

THE MEASURED PROBLEM THIS DESIGN SOLVES, AND WHY IT IS NOT A REWEIGHTING
-----------------------------------------------------------------------
Step 1 measured a zero-sum failure. `psi` is unit-normalised across four
slots, so populating Slot 3 STEALS mass from slots 0-2:

    slots 0-2 rescale 0.707107 -> 0.632456 when Slot 3 gains energy

On the 40-task `G-AST2` pool that moved firing prompts `12 -> 9` correct
while non-firing prompts stayed `4 -> 4`. The transfer is zero-sum BY
CONSTRUCTION. No lexicon edit fixes it, and narrowing the lexicon to pass
that clause would be gate-fitting.

This module therefore does not touch the wave. It adds a SECOND, ADDITIVE
score term:

    score(q, c) = <psi_q, psi_c>              # the wave, unchanged
                + w * (op_contrast + morph_contrast)

Three consequences follow, each testable rather than asserted:

  1. NO MASS TRANSFER. The wave keeps full energy. Nothing is diluted.
  2. EMPTY EVIDENCE STAYS EMPTY. An option whose operator slot is zero
     scores exactly 0.0 on that channel. It is never pulled toward the
     option-mean and given a direction it does not have.
  3. ROTATION INVARIANCE IS EXACT, NOT APPROXIMATE. The contrast is built
     from the option-mean, so it is permutation-equivariant: rotating the
     options permutes the scores and changes no option's value. A claim of
     "1.000000" is a derived property here, not a lucky measurement.

WHY CONTRAST AT ALL, AND WHY THE MEAN IS THE RIGHT CONTRAST
------------------------------------------------------------
Options in a multiple-choice item share stem words. Those shared words
inject the SAME feature into every option's operator slot, so they cancel
in the inner product and the discriminating part is drowned. Subtracting
the option-mean removes exactly the common component and leaves the
residual that differs between options.

The mean is chosen because it is the unique contrast that makes the design
permutation-equivariant. A contrast against the TOP-scoring option would
break rotation invariance, and a contrast against a stored reference would
reintroduce a prior.

WHAT IS NOT CLAIMED
-------------------
No learned weights. No model. `w` is selected on a STRATIFIED split by the
parsimony rule and is only kept if it does not degrade the held-out
accuracy. `w = 0.0` is the default and reproduces the Phase 5/6 behaviour
exactly, so every existing receipt stays valid.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import hashlib

import numpy as np

from henri import operator_ingress as oi

DEFAULT_SLOT = 16384

# -- MEASURED NEED: per-text encoding memoisation ---------------------------
#
# The GPQA 4x rotation arm re-encodes the same stem and the same four options
# on every one of the four cyclic shifts, so identical long paragraphs are
# encoded 4x. `channel_slots` is a PURE FUNCTION of (text, dim), so caching it
# changes speed and nothing else: the cached value is byte-identical to a
# recomputation. Floating-point operations are unaffected, by construction.
#
# Copies are returned so a caller cannot mutate a cached array in place and
# poison later reads.
_SLOT_CACHE: dict = {}
_CACHE_STATS = {"hits": 0, "misses": 0}


def _cached_channel_slots(text: str, dim: int = DEFAULT_SLOT):
    key = (hashlib.sha256(text.encode("utf-8")).hexdigest(), int(dim))
    hit = _SLOT_CACHE.get(key)
    if hit is None:
        _CACHE_STATS["misses"] += 1
        q_op, q_mo = channel_slots(text, dim)
        hit = (np.array(q_op, copy=True), np.array(q_mo, copy=True))
        _SLOT_CACHE[key] = hit
    else:
        _CACHE_STATS["hits"] += 1
    return np.array(hit[0], copy=True), np.array(hit[1], copy=True)


def cache_stats() -> dict:
    """Measured hit/miss counts. Reported in receipts, never inferred."""
    return dict(_CACHE_STATS, size=len(_SLOT_CACHE))
"""Operator and morphism sub-wave width at D = 65536."""

W_GRID = (0.0, 0.1, 0.2, 0.3)
"""Fixed before the first eval run. Not selected on held-out rows."""


def channel_slots(text: str, dim: int = DEFAULT_SLOT
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """`(operator_slot, morphism_slot)` for one text. Zero when absent.

    Both sub-waves come from `operator_ingress`, so this module holds no
    second definition of an operator or a morphism. A duplicate table would
    drift from the encoder and the drift would be silent.
    """
    from henri import operator_ingress as oi

    feats, _info = oi.all_features(text)
    op_feats, _struct = oi.split_features(feats)
    morph_feats, _minfo = oi.morphology_features(text)
    return (oi.ring_slot(op_feats, dim), oi.morph_slot(morph_feats, dim))


def _unit(v: np.ndarray) -> Tuple[np.ndarray, bool]:
    n = float(np.linalg.norm(v))
    if n <= 1e-30:
        return v, False
    return v / n, True


def contrastive(vecs: Sequence[np.ndarray]) -> List[np.ndarray]:
    """Remove the option-mean, then renormalise. Zero slots stay zero.

    THE MASK IS THE LOAD-BEARING PART. If a zero slot took part in the
    mean it would come out as `-mean` and acquire a direction it never had.
    An option naming no operator would then score on the operator channel,
    which is fabrication. Only NON-EMPTY slots enter the mean, and only
    non-empty slots leave with a value.
    """
    units: List[np.ndarray] = []
    present: List[bool] = []
    for v in vecs:
        u, has = _unit(np.asarray(v))
        units.append(u)
        present.append(has)
    live = [u for u, has in zip(units, present) if has]
    if not live:
        return [u for u in units]
    mean = np.mean(np.stack(live, axis=0), axis=0)
    out: List[np.ndarray] = []
    for u, has in zip(units, present):
        if not has:
            out.append(u)                 # exactly zero, by construction
            continue
        c = u - mean
        n = float(np.linalg.norm(c))
        out.append(c / n if n > 1e-30 else np.zeros_like(c))
    return out


def _masked_cos(q: np.ndarray, cands: Sequence[np.ndarray],
                has_q: bool) -> np.ndarray:
    """Cosine of `q` against each candidate, 0.0 where either side is empty."""
    if not has_q:
        return np.zeros(len(cands), dtype=float)
    qn, _ = _unit(q)
    out = np.zeros(len(cands), dtype=float)
    for i, c in enumerate(cands):
        cn, has_c = _unit(np.asarray(c))
        if not has_c:
            out[i] = 0.0
            continue
        out[i] = float(np.real(np.vdot(qn, cn)))
    return out


def channel_scores(question: str, options: Sequence[str],
                   dim: int = DEFAULT_SLOT) -> dict:
    """Contrastive operator and morphism scores for every option.

    Returns `{op, morph, combined, n_op_live, n_morph_live}`. `combined` is
    the mean of the two channels, and a channel with no live option reports
    all zeros rather than a fabricated ranking.
    """
    q_op, q_mo = _cached_channel_slots(question, dim)
    q_op_u, q_op_live = _unit(q_op)
    q_mo_u, q_mo_live = _unit(q_mo)

    opts = list(options)
    # One cached call per option, reused for BOTH channels. The Phase 10 code
    # called `channel_slots(o, dim)` twice per option (once for [0], once for
    # [1]), so it encoded every option twice. Same values, half the work.
    per_opt = [_cached_channel_slots(o, dim) for o in opts]
    c_op = contrastive([v[0] for v in per_opt])
    c_mo = contrastive([v[1] for v in per_opt])

    op = _masked_cos(q_op_u, c_op, q_op_live)
    mo = _masked_cos(q_mo_u, c_mo, q_mo_live)
    return {
        "op": op,
        "morph": mo,
        "combined": 0.5 * (op + mo),
        "n_op_live": int(sum(1 for v in c_op
                             if float(np.linalg.norm(v)) > 1e-30)),
        "n_morph_live": int(sum(1 for v in c_mo
                                if float(np.linalg.norm(v)) > 1e-30)),
        "question_has_operator": bool(q_op_live),
        "question_has_morphism": bool(q_mo_live),
    }


def blended_scores(base_scores: np.ndarray, question: str,
                   options: Sequence[str], weight: float,
                   dim: int = DEFAULT_SLOT) -> Tuple[np.ndarray, dict]:
    """Add the channel term to the wave scores at `weight`.

    `weight = 0.0` returns `base_scores` UNTOUCHED, so the parsimony rule
    can hold the channel off and every prior numeric result stays valid.
    """
    base = np.asarray(base_scores, dtype=float)
    if float(weight) == 0.0:
        return base.copy(), {"channel_weight": 0.0, "channel_added": False}
    ch = channel_scores(question, options, dim)
    return base + float(weight) * ch["combined"], {
        "channel_weight": float(weight), "channel_added": True,
        "n_op_live": ch["n_op_live"], "n_morph_live": ch["n_morph_live"],
        "question_has_operator": ch["question_has_operator"],
        "question_has_morphism": ch["question_has_morphism"],
    }


# -- PHASE 11 STEP 1: null-space gated structural channel --------------------
#
# THE DEFECT THIS REPLACES
# Phase 10 Step 1 mixed a fifth slot into `psi` and measured M4 falling
# 16/40 -> 13/40. The cause was arithmetic, not statistical: `psi` lives on
# the unit sphere, so a fifth slot rescales the others by
# 1/sqrt(2)/1/sqrt(2.5) = 0.8000. Nothing here touches `psi`. The structural
# channel is an ADDITIVE term on the SCORE, so `WaveState` and every pin stay
# byte-identical by construction.
#
# FALSIFIED PREMISE, RECORDED
# The directive asked for the channel to be "normalized by the number of
# active structural axes". Measured: `ring_slot`/`morph_slot` already return
# unit vectors -- 1, 2 and 3 active axes all give ||slot|| = 1.000000 -- so
# inactive axes contribute exactly nothing and the divisor would always be 1.
# Active-axis normalisation is a no-op here and is NOT implemented. The real
# levers are pool standardisation (below) and the null-space gate.

STRUCT_TIE_DECIMALS = 12
_STRUCT_CACHE: dict = {}
_STRUCT_STATS = {"hits": 0, "misses": 0}


def _cached_structural_slot(text: str, dim: int = DEFAULT_SLOT) -> np.ndarray:
    """Pruned-lexicon slot for one text, memoised by text digest."""
    key = (hashlib.sha256(text.encode("utf-8")).hexdigest(), int(dim))
    got = _STRUCT_CACHE.get(key)
    if got is not None:
        _STRUCT_STATS["hits"] += 1
        return got.copy()
    _STRUCT_STATS["misses"] += 1
    feats = oi.structural_morphology_from_text(text)
    v = oi.morph_slot(feats, int(dim))
    _STRUCT_CACHE[key] = v
    return v.copy()


def structural_slot(text: str, dim: int = DEFAULT_SLOT) -> np.ndarray:
    """Unit slot from the PRUNED structural lexicon only."""
    return _cached_structural_slot(text, dim)


def structural_cache_stats() -> dict:
    return dict(_STRUCT_STATS)


def structural_channel_scores(question: str, options: Sequence[str],
                              dim: int = DEFAULT_SLOT,
                              options_are_code: bool = False) -> dict:
    """Contrastive structural score per option.

    Same masking discipline as `channel_scores`: an option that names no
    structural transformation returns 0.0, never `-mean`. A zero slot that
    entered the mean would come out with a direction it never had.

    `options_are_code` selects the AST reader for the candidates. MEASURED
    NEED, not a convenience: the G-AST2 pool holds Python snippets, and the
    prose lexicon fires on English words. Reading code as prose would score
    every candidate near zero and the channel would look inert for a reason
    unrelated to its signal. The question stays prose in both cases.
    """
    q = _cached_structural_slot(question, dim)
    q_u, q_live = _unit(q)
    opts = list(options)
    if options_are_code:
        per_opt = []
        for o in opts:
            feats, _ok = oi.structural_morphology_from_ast(o)
            per_opt.append(oi.morph_slot(feats, int(dim)))
    else:
        per_opt = [_cached_structural_slot(o, dim) for o in opts]
    c = contrastive(per_opt)
    s = _masked_cos(q_u, c, q_live)
    n_live = int(sum(1 for v in c if float(np.linalg.norm(v)) > 1e-30))
    return {
        "score": s,
        "n_live": n_live,
        "question_has_structure": bool(q_live),
        "any_live": bool(q_live and n_live > 0),
    }


def pooled_z(x: Sequence[float], eps: float = 1e-12) -> np.ndarray:
    """Standardise across the candidate pool.

    ZERO-VARIANCE FALLBACK IS EXPLICIT. If every candidate scores the same on
    the channel, the term carries no ranking information and dividing by a
    zero spread would produce NaN or inf. The contract is "contribute
    nothing", so this returns exact zeros. That is a decision, not an
    accident: a silent division here would poison every downstream score.
    """
    a = np.asarray(x, dtype=float)
    if a.size == 0:
        return a
    sd = float(np.std(a))
    if not np.isfinite(sd) or sd <= eps:
        return np.zeros_like(a)
    return (a - float(np.mean(a))) / sd


def pool_tied(base_scores: Sequence[float],
              decimals: int = STRUCT_TIE_DECIMALS) -> bool:
    """True when the top TWO pool scores are equal after the sort rounding.

    THE ROUNDING MUST MATCH THE SORT KEY. `rank_options` orders by
    `-round(score, 12)`; a gate that used exact float equality would open on
    differences the ranker treats as ties, or stay shut on differences it
    treats as distinct. One rounding rule, one place.
    """
    s = sorted((round(float(v), decimals) for v in base_scores), reverse=True)
    return len(s) >= 2 and s[0] == s[1]


def nullspace_gate(base_scores: Sequence[float], slot1_active: bool,
                   decimals: int = STRUCT_TIE_DECIMALS) -> Tuple[bool, str]:
    """Decide whether the structural channel may enter the score.

    Opens in exactly two situations:
      * `slot1_active` is False -- the primary arithmetic operator channel is
        silent, so there is no primary margin to dilute.
      * the top candidates tie on the primary score -- the wave cannot choose,
        so a secondary channel breaks the tie instead of overriding a choice.

    Shuts when Slot 1 is active AND the primary pool is untied. This is what
    makes the zero-degradation clause structural rather than hoped for: on
    such a prompt the returned score is `base` untouched.
    """
    if not bool(slot1_active):
        return True, "slot1_silent"
    if pool_tied(base_scores, decimals):
        return True, "primary_tie"
    return False, "slot1_active_untied"


def blended_scores_gated(base_scores: np.ndarray, question: str,
                         options: Sequence[str], weight: float,
                         slot1_active: bool,
                         dim: int = DEFAULT_SLOT,
                         decimals: int = STRUCT_TIE_DECIMALS,
                         options_are_code: bool = False
                         ) -> Tuple[np.ndarray, dict]:
    """Null-space gated additive structural score.

    `weight = 0.0` returns `base` untouched, so the parsimony rule keeps its
    meaning. When the gate shuts, the return is also `base` untouched -- the
    meta block records which reason applied, so a null result says WHY.
    """
    base = np.asarray(base_scores, dtype=float)
    meta = {"channel_weight": 0.0, "channel_added": False,
            "gate": "", "z_zero_variance": False}

    if float(weight) == 0.0:
        meta["gate"] = "parsimony_w0"
        return base.copy(), meta

    open_gate, reason = nullspace_gate(base, slot1_active, decimals)
    meta["gate"] = reason
    if not open_gate:
        return base.copy(), meta

    st = structural_channel_scores(question, options, dim,
                                   options_are_code=options_are_code)
    if not st["any_live"]:
        meta["gate"] = reason + "_no_channel_signal"
        return base.copy(), meta

    z = pooled_z(st["score"])
    if not np.any(z):
        meta["z_zero_variance"] = True
        meta["gate"] = reason + "_zero_variance"
        return base.copy(), meta

    meta.update(channel_weight=float(weight), channel_added=True,
                n_live=st["n_live"])
    return base + float(weight) * z, meta


__all__ = ["DEFAULT_SLOT", "W_GRID", "channel_slots", "contrastive",
           "channel_scores", "blended_scores",
           "structural_slot", "structural_channel_scores",
           "structural_cache_stats", "pooled_z", "pool_tied",
           "nullspace_gate", "blended_scores_gated"]
