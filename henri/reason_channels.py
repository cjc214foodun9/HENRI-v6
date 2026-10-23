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

import numpy as np

DEFAULT_SLOT = 16384
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
    q_op, q_mo = channel_slots(question, dim)
    q_op_u, q_op_live = _unit(q_op)
    q_mo_u, q_mo_live = _unit(q_mo)

    opts = list(options)
    c_op = contrastive([channel_slots(o, dim)[0] for o in opts])
    c_mo = contrastive([channel_slots(o, dim)[1] for o in opts])

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


__all__ = ["DEFAULT_SLOT", "W_GRID", "channel_slots", "contrastive",
           "channel_scores", "blended_scores"]
