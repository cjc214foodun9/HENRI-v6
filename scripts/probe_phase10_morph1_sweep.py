"""M4 follow-up: does ANY morphism weight help G-AST2? Pre-registered.

STATUS OF THIS FILE
-------------------
`receipts/phase10_g_morph1.json` records `G_MORPH1_PARTIAL_3_OF_4` with M4
measured FAIL at `0.3250` against the Phase 9 `0.4000`. That receipt is the
registered result and is NOT edited. This script is an ADDITIVE follow-up
that answers one question the first run could not: is the failure a WRONG
WEIGHT or a channel with no ranking value?

MEASURED MECHANISM THE FIRST RUN EXPOSED
----------------------------------------
Of the 40 held-out prompts, 31 fire at least one morphism axis and 9 fire
none. At `morphism_weight=0.20`:
    firing prompts     12 -> 9 correct   (-3)
    non-firing prompts  4 -> 4 correct   (unchanged)
Slots 0-2 rescale by `0.632/0.707 = 0.894` when Slot 3 gains energy,
because `psi` is unit-normalised over the whole vector. So the channel
takes 20% of the ranking mass away from the semantic and arithmetic
content and replaces it with features that are noisy on arithmetic prose
("count", "values", "find", "index", "first" are high-frequency there).

DECISION RULE, FIXED BEFORE THE FIRST RUN OF THIS FILE
------------------------------------------------------
Selection happens on the TRAIN split. Evaluation happens ONCE on the
held-out 40. This ordering is the whole point: sweeping on the held-out
set and reporting the best arm would be selection-on-test, and the Phase 8
`lam` result was retracted for exactly that defect.

  t   morphism weight. `t=0` disables the channel; `t=0.20` reproduces the
      registered run. `(w0,w1,w2) = (1-t) * (0.40,0.10,0.30)/0.80` and
      `w3 = t`, so `t=0.20` gives exactly `W_OPERATOR_AWARE`.
  grid  t in {0.00, 0.05, 0.10, 0.15, 0.20}
  pick  argmax TRAIN top-1; ties break toward the SMALLER t, so the
        channel must earn its mass rather than win a coin toss.

THE EXACT-DECOMPOSITION SELF-CHECK
----------------------------------
Scoring is done from cached per-slot unit sub-waves, not by re-encoding
per weight:
    psi = sum_i s_i * sqrt(w_i)   (s_i unit, empty slots contribute 0)
    cos = sum_i w_i c_i / sqrt( (sum_{i in Q} w_i)(sum_{i in C} w_i) )
This is exact, because each slot is scaled independently and the norm is a
sum. It is only valid if it reproduces the two numbers the encoder already
produced. The script therefore ASSERTS `t=0 -> 0.4000` and
`t=0.20 -> 0.3250`. If either assertion fails, the decomposition is wrong,
`DECOMPOSITION_MISMATCH` is reported, and no sweep number is trusted.

VERDICTS
--------
  M4_RECOVERED            held-out at the selected t >= 0.4000
  M4_CHANNEL_NO_GAIN      held-out at the selected t <  0.4000
  DECOMPOSITION_MISMATCH  the self-check failed; nothing else is claimed
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
from henri.semantic_backbone import SemanticBackbone

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "receipts" / "phase10_g_morph1_m4_sweep.json"
SLOT = 16384

N_TASKS = 40
SEED = 20261020
GRID = (0.00, 0.05, 0.10, 0.15, 0.20)
PHASE9_TOP1 = 0.4000
REGISTERED_T020 = 0.3250
MIX = 0.5
BASE_P = (0.40, 0.10, 0.30)


def load_phase9():
    p = REPO / "scripts" / "probe_phase9_mbpp_wave.py"
    spec = importlib.util.spec_from_file_location("p9_sweep", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0.0 else v


class SlotCache:
    """Per-slot unit sub-waves for one text. Encoded ONCE, scored per weight."""

    __slots__ = ("s", "has")

    def __init__(self, text: str, bb) -> None:
        core = bb.encode(text)
        s0 = _unit(np.asarray(core.psi[0:SLOT]))
        s1 = _unit(np.asarray(core.psi[SLOT:2 * SLOT]))
        feats, _ = oi.all_features(text)
        op, st = oi.split_features(feats)
        s2 = oi.ring_slot(op, SLOT)
        s3_struct = oi.ring_slot(st, SLOT)
        mfeats, _ = oi.morphology_features(text)
        s_m = oi.morph_slot(mfeats, SLOT)
        n_m, n_s = float(np.linalg.norm(s_m)), float(np.linalg.norm(s3_struct))
        if n_m > 0.0 and n_s > 0.0:
            s3 = _unit((1.0 - MIX) * s3_struct + MIX * s_m)
        elif n_m > 0.0:
            s3 = _unit(s_m)
        else:
            s3 = s3_struct
        self.s = (s0, s1, s2, s3)
        self.has = tuple(float(np.linalg.norm(x)) > 0.0 for x in self.s)


def weights_for(t: float) -> tuple:
    rem = 1.0 - float(t)
    tot = sum(BASE_P)
    return tuple(rem * p / tot for p in BASE_P) + (float(t),)


def slot_dots(q: SlotCache, c: SlotCache) -> tuple:
    return tuple(float(np.real(np.vdot(q.s[i], c.s[i])))
                 if (q.has[i] and c.has[i]) else 0.0
                 for i in range(4))


def score(dots, q_has, c_has, w) -> float:
    num = sum(w[i] * dots[i] for i in range(4))
    nq = sum(w[i] for i in range(4) if q_has[i])
    nc = sum(w[i] for i in range(4) if c_has[i])
    den = float(np.sqrt(nq * nc))
    return num / den if den > 0.0 else 0.0


def top1_for(items, w) -> dict:
    """Gold is index 0 by construction in `build_pool`."""
    correct = 0
    ties = []
    for q, cands in items:
        dots = [slot_dots(q, c) for c in cands]
        sc = [score(d, q.has, c.has, w) for d, c in zip(dots, cands)]
        best = max(sc)
        order = sorted(range(len(sc)), key=lambda k: (-sc[k], k))
        correct += int(order[0] == 0)
        ties.append(sum(1 for s in sc if abs(s - best) <= 1e-9))
    n = len(items)
    return {"n": n, "top1_correct": correct,
            "top1_acc": round(correct / float(n), 6),
            "mean_tie_group": round(float(np.mean(ties)), 4)}


def build_items(rows, index_list, donors, p9, run, mutate, bb, tag):
    items = []
    for i in index_list:
        row = rows[i]
        pool, info = p9.build_pool(row, rows, mutate, run, donors)
        if not info["complete"]:
            continue
        prompt = (row.get("text") or "").strip()
        q = SlotCache(prompt, bb)
        cands = [SlotCache(t, bb) for t in [c["text"] for c in pool]]
        items.append((q, cands))
    print("  %s items %d" % (tag, len(items)))
    return items


def main() -> int:
    t0 = time.time()
    from henri.code_verify import load_mbpp_executor

    p9 = load_phase9()
    mod = load_mbpp_executor()
    if mod is None:
        print("EXECUTOR_UNAVAILABLE: infrastructure limit, not a result")
        return 2
    run, mutate = mod.run_candidate, mod.mutate
    rows = p9.load_rows()
    bb = SemanticBackbone()

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(rows))
    train_idx = [int(i) for i in order[:800]]
    held_idx = [int(i) for i in order[800:]]

    print("building HELD-OUT pools (%d tasks)" % N_TASKS)
    held = build_items(rows, held_idx[:N_TASKS], train_idx, p9, run, mutate,
                       bb, "held")
    print("building TRAIN pools (%d tasks)" % N_TASKS)
    # Donors come from the held split here so the train arms cannot lean on
    # their own rows for distractors.
    train = build_items(rows, train_idx[:N_TASKS], held_idx, p9, run, mutate,
                       bb, "train")

    # ---- exactness self-check, BEFORE any sweep is believed -------------
    check = {}
    for t, expect, label in ((0.00, PHASE9_TOP1, "t0_reproduces_phase9"),
                             (0.20, REGISTERED_T020, "t20_reproduces_registered")):
        got = top1_for(held, weights_for(t))["top1_acc"]
        check[label] = {"t": t, "expected": expect, "measured": got,
                        "match": bool(abs(got - expect) <= 1e-9)}
    ok = all(v["match"] for v in check.values())

    # ---- sweep: select on TRAIN, evaluate on HELD-OUT -------------------
    train_table, held_table = [], []
    for t in GRID:
        w = weights_for(t)
        train_table.append({"t": t, "weights": [round(x, 6) for x in w],
                            **top1_for(train, w)})
        held_table.append({"t": t, "weights": [round(x, 6) for x in w],
                           **top1_for(held, w)})

    best = max(train_table, key=lambda r: (r["top1_acc"], -r["t"]))
    selected_t = best["t"]
    held_at_selected = next(r for r in held_table if r["t"] == selected_t)

    if not ok:
        verdict = "DECOMPOSITION_MISMATCH"
    elif selected_t == 0.0:
        # THE LABEL MUST NOT FLATTER THE RESULT. The bar is met at `t=0.0`,
        # and `t=0.0` DISABLES the channel. Calling that "recovered" would
        # report a baseline as a channel win -- the exact Phase 9 defect
        # where a body digest was printed as a file hash. Parsimony holds
        # the channel OFF, mirroring the `G-KOOP1` precedent.
        verdict = "M4_PARSIMONY_HELD_CHANNEL_OFF"
    elif held_at_selected["top1_acc"] >= PHASE9_TOP1:
        verdict = "M4_RECOVERED"
    else:
        verdict = "M4_CHANNEL_NO_GAIN"

    body = {
        "probe": "phase10_g_morph1_m4_sweep",
        "status": "ADDITIVE FOLLOW-UP to the registered G_MORPH1 receipt",
        "registered_result": {"file": "receipts/phase10_g_morph1.json",
                              "m4_value": REGISTERED_T020,
                              "m4_bar": PHASE9_TOP1,
                              "m4_pass": False,
                              "verdict": "G_MORPH1_PARTIAL_3_OF_4"},
        "decision_rule": ("select argmax TRAIN top-1, ties toward smaller t; "
                          "evaluate ONCE on held-out. Selection on train "
                          "prevents selection-on-test."),
        "grid": list(GRID),
        "decomposition_self_check": check,
        "decomposition_exact": bool(ok),
        "train_table": train_table,
        "heldout_table": held_table,
        "selected_t_on_train": selected_t,
        "heldout_at_selected": held_at_selected,
        "verdict": verdict,
        "mechanism": ("The channel takes ranking mass from slots 0-2 in "
                      "proportion to t. It reads strongly as an ENCODER "
                      "(M1 -1.0, M2 +1.0, M3 0.9014 coverage) but its "
                      "features are noisy on arithmetic prose, so it does "
                      "not rank the G-AST2 pool better."),
        "wall_clock_s": round(time.time() - t0, 2),
    }

    ids = write_receipt(OUT, body)

    print("\nexactness self-check")
    for k, v in check.items():
        print("  %-28s t=%.2f expected %.4f measured %.4f  %s"
              % (k, v["t"], v["expected"], v["measured"],
                 "OK" if v["match"] else "MISMATCH"))
    print("\n  t     train    held-out   (held-out is the evaluation)")
    for tr, he in zip(train_table, held_table):
        mark = "  <- selected on train" if tr["t"] == selected_t else ""
        print("  %.2f  %.4f   %.4f%s" % (tr["t"], tr["top1_acc"],
                                         he["top1_acc"], mark))
    print("\nselected t=%.2f  held-out %.4f  bar %.4f" %
          (selected_t, held_at_selected["top1_acc"], PHASE9_TOP1))
    print(BODY_FIELD, ids[BODY_FIELD])
    print(FILE_FIELD + "  ", ids[FILE_FIELD])
    print("VERDICT %s" % verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
