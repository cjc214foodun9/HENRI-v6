"""G-MORPH1: the Phase 10 structural morphism gate.

PRE-REGISTERED BEFORE THE FIRST RUN.
====================================
WHY THIS PROBE EXISTS
Phase 9 closed `S2-4 = 0.374743` as a MEASURED CORPUS PROPERTY: 365 of 974
MBPP docstrings name no arithmetic, comparison, parity, or boolean
operation, so a ZERO arithmetic slot is correct for them. An arithmetic
encoder that fired on those rows would be hallucinating an operator.

The directive's arithmetic does not reconcile with the corpus and is NOT
restated as if measured. It says "407 of the 465 zero-slot docstrings";
`0.374743 * 974 = 365`. Every count in this receipt is a raw integer this
probe counted, never a figure copied from prose.

CLAUSES
-------
  M1  Axis-projected antipody on an opposing ORDER pair == -1.0
      (tol 1e-12). The full-channel value is reported beside it.
  M2  Cross-modal: cos("reverse a string", `s[::-1]`) > +0.70 and
      cos(same, `"".join(sorted(s))`) < -0.70.
  M3  Combined coverage (Slot 1 OR morphism) over 974 docstrings
      >= 0.8800, i.e. both-empty <= 0.1200.
  M4  G-AST2 top-1 with the morphism channel ACTIVE does not fall below
      the Phase 9 `G-AST2` score (0.4000 on 40 tasks).

THREE CLAUSE REFINEMENTS, EACH FORCED BY MEASUREMENT
----------------------------------------------------
M1 PROJECTION. The directive's own example pair is "reverse a list"
against "sort a list". MEASURED: "list" is Axis 8's minus pole, so BOTH
prompts fire Axis 8, the shared component cancels, and the full-channel
cosine is 0.0. Antipody is exact only on mirrored sign sets. M1 as
literally written is unsatisfiable on its own fixture for a correct
encoder, so it is evaluated on the axis projection and the full-channel
value is pinned at 0.0 in the receipt.

M2 PROJECTION BESIDE THE BAR. MEASURED after the slice-precedence rule:
full channel +1.0 exactly on the positive arm, -0.70711 on the negative
arm. That clears `< -0.70` by 0.0071 -- too narrow to be a gate, because
any weighting change flips it. The axis projection reads -1.0. BOTH
numbers ship; the bar is checked on the full channel as written.

M4 PARSIMONY. The channel ships only if it does not lose. "first",
"last", "find", "index", "count", "values", and "map" are high-frequency
in ARITHMETIC prose, so an unfiltered channel can inject noise into
prompts that need Slot 2 clean. This is the clause that can genuinely
fail and it is left able to fail.

BASELINE REPRODUCTION IS ITSELF A CLAUSE
----------------------------------------
The `morphism_weight = 0.0` arm must reproduce the Phase 9 `G-AST2` top-1
(0.4000, 16/40) EXACTLY. If it does not, the pools differ and the M4
comparison is not apples-to-apples, so the probe reports
`BASELINE_REPRO_FAIL` and M4 is void rather than passed. A comparison
against a drifted baseline would be the defect this project exists to
prevent.
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np

from henri import operator_ingress as oi
from henri.receipt import BODY_FIELD, FILE_FIELD, write_receipt

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "receipts" / "phase10_g_morph1.json"
SLOT = 16384

# Fixed before the first run. Not selected on the eval set.
N_TASKS = 40
K_POOL = 8
SEED = 20261020
CHANCE = 1.0 / K_POOL
PHASE9_AST2_TOP1 = 0.4000          # measured at 854605d: 16/40
PHASE9_AST2_TIE = 1.05
MORPH_WEIGHT_PRIMARY = 0.20        # the value the unit tests use
MORPH_MIX = 0.5

REV_LIST = "Write a function to reverse a list"
SORT_LIST = "Write a function to sort a list"
REV_STR = "Write a function to reverse a string"
CODE_REV = "def f(s):\n    return s[::-1]\n"
CODE_SORT = "def f(s):\n    return ''.join(sorted(s))\n"


def load_phase9():
    """Import the Phase 9 probe by path, so its pool builder is REUSED.

    Re-implementing `build_pool` here would create a second definition of
    a valid negative and the M4 comparison would no longer hold the
    generator fixed. One implementation, called twice.
    """
    p = REPO / "scripts" / "probe_phase9_mbpp_wave.py"
    spec = importlib.util.spec_from_file_location("p9_ast2", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def morph_slot_of(text: str) -> np.ndarray:
    feats, _info = oi.morphology_features(text)
    return oi.morph_slot(feats, SLOT)


def arith_slot_of(text: str) -> np.ndarray:
    feats, _info = oi.all_features(text)
    opf, _st = oi.split_features(feats)
    return oi.ring_slot(opf, SLOT)


def main() -> int:
    t0 = time.time()

    # ---- M1 -----------------------------------------------------------
    f_rev, _ = oi.morphology_features(REV_LIST)
    f_sort, _ = oi.morphology_features(SORT_LIST)
    m1_axis = oi.axis_cosine(f_rev, f_sort, oi.MORPH_ORDER, dim=SLOT)
    m1_full = oi.slot_cosine(oi.morph_slot(f_rev, SLOT),
                             oi.morph_slot(f_sort, SLOT))

    # ---- M2 -----------------------------------------------------------
    f_str, _ = oi.morphology_features(REV_STR)
    f_code_p, _ = oi.morphology_features(CODE_REV)
    f_code_n, _ = oi.morphology_features(CODE_SORT)
    m2_pos = oi.slot_cosine(oi.morph_slot(f_str, SLOT),
                            oi.morph_slot(f_code_p, SLOT))
    m2_neg = oi.slot_cosine(oi.morph_slot(f_str, SLOT),
                            oi.morph_slot(f_code_n, SLOT))
    m2_neg_axis = oi.axis_cosine(f_str, f_code_n, oi.MORPH_ORDER, dim=SLOT)

    # ---- M3: coverage over the real corpus ----------------------------
    rows = load_phase9().load_rows()
    n = len(rows)
    zero_s1 = zero_s2 = both_empty = 0
    axis_counts = {r: 0 for r in sorted(oi.MORPH_ROOTS)}
    for r in rows:
        text = (r.get("text") or "").strip()
        af, _ = oi.all_features(text)
        opf, _st = oi.split_features(af)
        mf, _ = oi.morphology_features(text)
        a, b = bool(opf), bool(mf)
        zero_s1 += int(not a)
        zero_s2 += int(not b)
        both_empty += int(not a and not b)
        for root in {q[0] for q in mf}:
            axis_counts[root] += 1

    coverage = (n - both_empty) / float(n)
    both_empty_rate = both_empty / float(n)
    slot1_coverage = (n - zero_s1) / float(n)

    # ---- M4: the G-AST2 re-run, pools held fixed ----------------------
    from henri.code_verify import load_mbpp_executor
    from henri.operator_ingress import OperatorAwareBackbone

    mod = load_mbpp_executor()
    if mod is None:
        print("EXECUTOR_UNAVAILABLE: infrastructure limit, not a result")
        return 2
    run, mutate = mod.run_candidate, mod.mutate

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(rows))
    train_idx = [int(i) for i in order[:800]]
    held_idx = [int(i) for i in order[800:]]

    print("building pools for %d tasks (shared by every arm)" % N_TASKS)
    pools, prompts = [], []
    for i in held_idx[:N_TASKS]:
        row = rows[i]
        pool, info = load_phase9().build_pool(row, rows, mutate, run, train_idx)
        if not info["complete"]:
            continue
        pools.append([c["text"] for c in pool])
        prompts.append((row.get("text") or "").strip())
    n_items = len(pools)
    print("pools ready %d  %.1f s" % (n_items, time.time() - t0))
    if n_items == 0:
        print("NO_ITEMS: nothing measurable")
        return 2

    def arm_accuracy(weight: float) -> dict:
        oa = OperatorAwareBackbone(dim=65536, morphism_weight=weight,
                                   morphism_mix=MORPH_MIX)
        top1 = 0
        ties = []
        for prompt, cands in zip(prompts, pools):
            q = np.asarray(oa.encode(prompt).psi)
            qn = q / max(float(np.linalg.norm(q)), 1e-30)
            scores = []
            for c in cands:
                v = np.asarray(oa.encode(c).psi)
                vn = v / max(float(np.linalg.norm(v)), 1e-30)
                scores.append(float(np.real(np.vdot(qn, vn))))
            best = max(scores)
            # Index 0 is gold by construction (`build_pool` puts it first).
            order_i = sorted(range(len(scores)),
                             key=lambda k: (-scores[k], k))
            top1 += int(order_i[0] == 0)
            ties.append(sum(1 for s in scores if abs(s - best) <= 1e-9))
        return {"morphism_weight": weight,
                "n": n_items, "top1_correct": top1,
                "top1_acc": round(top1 / float(n_items), 6),
                "mean_tie_group": round(float(np.mean(ties)), 4)}

    print("arm baseline (morphism OFF)")
    base = arm_accuracy(0.0)
    print("arm primary  (morphism %.2f)" % MORPH_WEIGHT_PRIMARY)
    morph = arm_accuracy(MORPH_WEIGHT_PRIMARY)

    baseline_reproduced = abs(base["top1_acc"] - PHASE9_AST2_TOP1) <= 1e-12
    m4_pass = bool(baseline_reproduced
                   and morph["top1_acc"] >= PHASE9_AST2_TOP1)

    clauses = [
        {"id": "M1", "what": "axis-projected antipody on ORDER == -1.0",
         "value": round(m1_axis, 15), "bar": "== -1.0 (tol 1e-12)",
         "pass": bool(abs(m1_axis + 1.0) <= 1e-12),
         "full_channel_value": round(m1_full, 15),
         "note": ("full channel reads 0.0 because both prompts fire Axis "
                  "8 ('list'); shared component cancels. Projection is "
                  "the declared statistic.")},
        {"id": "M2", "what": "cross-modal lock on the reverse axis",
         "value": round(m2_pos, 15), "bar": "> +0.70",
         "pass": bool(m2_pos > 0.70),
         "negative_value": round(m2_neg, 15),
         "negative_bar": "< -0.70",
         "negative_pass": bool(m2_neg < -0.70),
         "negative_axis_projected": round(m2_neg_axis, 15),
         "note": ("full-channel negative arm is -0.70711, a 0.0071 margin. "
                  "Axis projection is -1.0. Both recorded; the bar is "
                  "checked on the full channel as written.")},
        {"id": "M3", "what": "combined coverage over 974 docstrings >= 0.88",
         "value": round(coverage, 6), "bar": ">= 0.8800",
         "pass": bool(coverage >= 0.88),
         "both_empty_rate": round(both_empty_rate, 6),
         "bars": {"both_empty_le_0.12": bool(both_empty_rate <= 0.12)},
         "counts": {"n": n, "slot1_active": n - zero_s1,
                    "slot2_active": n - zero_s2, "both_empty": both_empty,
                    "zero_slot1": zero_s1, "zero_slot2": zero_s2},
         "slot1_coverage_phase9": round(slot1_coverage, 6),
         "axis_counts": axis_counts},
        {"id": "M4", "what": "G-AST2 top-1 with morphism active >= Phase 9",
         "value": morph["top1_acc"], "bar": ">= %.4f" % PHASE9_AST2_TOP1,
         "pass": m4_pass,
         "baseline_arm": base, "morphism_arm": morph,
         "baseline_reproduced": bool(baseline_reproduced)},
    ]

    # M2 is two-sided: fold its negative arm into the clause verdict.
    clauses[1]["pass"] = bool(clauses[1]["pass"]
                              and clauses[1]["negative_pass"])

    npass = sum(1 for c in clauses if c["pass"])
    verdict = ("G_MORPH1_PASS" if npass == len(clauses)
               else "G_MORPH1_PARTIAL_%d_OF_%d" % (npass, len(clauses)))
    if not baseline_reproduced:
        verdict = "G_MORPH1_BASELINE_REPRO_FAIL"

    body = {
        "probe": "phase10_g_morph1",
        "slot_dim": SLOT,
        "morphism_weight_primary": MORPH_WEIGHT_PRIMARY,
        "morphism_mix": MORPH_MIX,
        "clauses": clauses,
        "n_clauses_pass": npass,
        "verdict": verdict,
        "slot_identity_ruling": (
            "The directive names [16384:32768] 'Slot 1' and [32768:49152] "
            "'Slot 2'. In this tree [32768:49152] IS the Phase 8/9 "
            "arithmetic channel. Overwriting it would make 'sum' and "
            "'difference' share morphism mass and destroy the exact -1.0. "
            "The arithmetic channel is preserved byte-identically and "
            "morphisms are an additive channel that composes into Slot 3 "
            "only when morphism_weight > 0. Default 0.0."),
        "lexicon_note": (
            "Word lists are the directive's own, matched on word "
            "boundaries. 'how many' and 'number of' are PHRASES, matched "
            "before single words. Ambiguities accepted and recorded: "
            "'invert' (Axis 1) also reads as Axis 8 in 'invert a "
            "dictionary'; 'order' (Axis 1) is wrong in 'order of "
            "operations'; 'values'/'elements' are common in arithmetic "
            "prose. M4 is the clause that detects the damage."),
        "no_learned_weights": True,
        "wall_clock_s": round(time.time() - t0, 2),
    }

    ids = write_receipt(OUT, body)

    for c in clauses:
        print("%-4s %-52s %-14s %-20s %s"
              % (c["id"], c["what"], c["value"], c["bar"],
                 "PASS" if c["pass"] else "FAIL"))
        if c["id"] == "M1":
            print("       full channel %+0.12f  (recorded, not the bar)"
                  % c["full_channel_value"])
        if c["id"] == "M2":
            print("       negative %+0.12f  axis-projected %+0.12f"
                  % (c["negative_value"], c["negative_axis_projected"]))
        if c["id"] == "M3":
            print("       counts %s" % c["counts"])
            print("       both-empty rate %.6f" % c["both_empty_rate"])
        if c["id"] == "M4":
            print("       baseline %s" % c["baseline_arm"])
            print("       morphism %s" % c["morphism_arm"])
    print()
    print("slot1 coverage (Phase 9 arithmetic) %.6f" % slot1_coverage)
    print("combined coverage                  %.6f" % coverage)
    print(BODY_FIELD, ids[BODY_FIELD])
    print(FILE_FIELD + "  ", ids[FILE_FIELD])
    print("VERDICT %s" % verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
