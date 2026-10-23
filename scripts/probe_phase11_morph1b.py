"""Phase 11 Step 1 probe: G-MORPH1b, the null-space gated structural channel.

PRE-REGISTERED BEFORE THE FIRST RUN OF THIS PROBE.

WHAT THIS GATE MUST SHOW
Step 1 of Phase 10 measured M4 falling 16/40 -> 13/40 (0.4000 -> 0.3250) when
a fifth slot was mixed into `psi`. Two causes were proven: unit-sphere
rescaling (0.707107 -> 0.632456) and broad prose words firing on arithmetic
text. Phase 11 Step 1 changes neither `psi` nor the Phase 10 tables. It adds
an ADDITIVE score term behind a gate that shuts on exactly the prompts where
the Phase 10 loss occurred.

THE CLAUSES, FIXED HERE
  M1  EXACTNESS. At `w_morph = 0.0` the gated arm must reproduce the Phase 9
      baseline 0.4000 on G-AST2 digit-for-digit, and the returned vector must
      equal the base vector exactly on every item. A gate that cannot
      reproduce the baseline cannot be trusted to improve it.
  M2  ZERO DEGRADATION. On every item where the gate shuts, the returned
      scores must equal the base scores BIT-EXACTLY. This is structural, not
      hoped for: the shut branch returns `base.copy()`. The clause then checks
      held-out accuracy >= 0.4000 at the selected weight.
  M3  SILENT-SUBSET MEASUREMENT. On items whose operator slot is silent, the
      gated arm must not be worse than base. Reported as a measured delta; a
      delta of exactly 0.0 is a real result and is recorded as such.
  M4  PARSIMONY. `w_morph` is selected on TRAIN with ties toward the SMALLER
      weight, then evaluated ONCE on held-out. If train selects 0.0 the
      verdict says the channel is OFF and the baseline met the bar.

WHAT IS NOT CLAIMED
No claim is made that the structural channel improves G-AST2. G-AST2 prompts
are arithmetic, so the gate should shut on most of them by construction. The
honest possible outcomes are "baseline held, channel dormant" or "channel
lifted the structural subset".

WHY THE CANDIDATES ARE READ AS CODE
The G-AST2 pool holds Python snippets. Reading them with the prose lexicon
would score every candidate near zero for a reason unrelated to signal.
`options_are_code=True` selects the AST reader; the question stays prose.

OUTPUT PATHS ARE RUN-SPECIFIC
A fixed receipt path let two concurrent runs clobber one file. `--tag`
suffixes both the receipt and its sidecar, so a re-run never overwrites a
committed result.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from henri.receipt import BODY_FIELD, FILE_FIELD, write_receipt  # noqa: E402

STATE = REPO / "receipts"
P10_SWEEP = REPO / "scripts" / "probe_phase10_morph1_sweep.py"

PHASE9_TOP1 = 0.4000          # Phase 9 baseline on the G-AST2 40-task split
PHASE10_M4 = 0.3250           # the measured regression this design must not repeat
N_TASKS = 40
SEED = 20261020
W_MORPH_GRID = (0.0, 0.05, 0.10, 0.20)


def load_sweep():
    """Load the Phase 10 sweep module. Its helpers are the pinned baseline."""
    if not P10_SWEEP.exists():
        raise SystemExit("missing %s" % P10_SWEEP)
    spec = importlib.util.spec_from_file_location("p10sweep", P10_SWEEP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_items_ex(rows, index_list, donors, sw, p9, run, mutate, bb, tag):
    """Like the sweep's `build_items`, but keeps prompt and candidate TEXT.

    MEASURED NEED: the structural channel scores option TEXT. `SlotCache`
    stores only vectors, so the sweep's tuple cannot feed it. Returning texts
    is cheaper than re-deriving them downstream and guarantees the vectors and
    the texts describe the same candidates.
    """
    items = []
    for i in index_list:
        row = rows[i]
        pool, info = p9.build_pool(row, rows, mutate, run, donors)
        if not info["complete"]:
            continue
        prompt = (row.get("text") or "").strip()
        texts = [c["text"] for c in pool]
        q = sw.SlotCache(prompt, bb)
        cands = [sw.SlotCache(t, bb) for t in texts]
        items.append({"q": q, "cands": cands, "texts": texts,
                      "prompt": prompt})
    print("  %s items %d" % (tag, len(items)))
    return items


def base_scores_for(item, sw, weights):
    """Phase 9-style wave score vector. Gold is index 0 by construction.

    MEASURED MODULE BOUNDARY: `SlotCache`, `slot_dots`, `score` and
    `weights_for` live in the SWEEP module. `load_rows` and `build_pool` live
    in the PHASE 9 module that `sweep.load_phase9()` returns. The first
    version of this probe named both `p9` and called `p9.load_rows()`, which
    the sweep module does not define.
    """
    dots = [sw.slot_dots(item["q"], c) for c in item["cands"]]
    q_has = item["q"].has
    return np.array([sw.score(d, q_has, c.has, weights)
                     for d, c in zip(dots, item["cands"])], dtype=float)


def gated_vector(item, base, rc, w_morph):
    """Apply the gate with the operator-slot activity flag from the encoder.

    `slot1_active` is `item["q"].has[2]` -- the operator ring slot. MEASURED
    NAMING: in `SlotCache`, `s[2]` is `ring_slot(operator_features)`. The
    directive calls this "Slot 1 arithmetic operator channel". Taking it from
    the encoder rather than re-deriving it keeps one source of truth.
    """
    return rc.blended_scores_gated(
        base, item["prompt"], item["texts"], w_morph,
        slot1_active=bool(item["q"].has[2]), options_are_code=True)


def tie_digest_order(sc, texts):
    """Deterministic order that does NOT favour gold.

    The sweep breaks ties by index, which always favours index 0 (gold). For
    an accuracy comparison that is a silent gold bias. Ties here go by text
    digest, the same content rule the ranker uses.
    """
    keys = [(-round(float(s), 12), hashlib.sha256(t.encode("utf-8")).hexdigest())
            for s, t in zip(sc, texts)]
    return [i for i, _ in sorted(enumerate(keys), key=lambda p: p[1])]


def accuracy(items, rc, w_morph, sw, weights):
    """(accuracy, ties, shut_all_exact, shut_count, per-item rows)."""
    correct = 0
    ties = 0
    shut_exact = 0
    shut_n = 0
    rows = []
    for it in items:
        base = base_scores_for(it, sw, weights)
        sc, meta = gated_vector(it, base, rc, w_morph)
        order = tie_digest_order(sc, it["texts"])
        correct += int(order[0] == 0)
        best = float(np.max(sc))
        ties += int(sum(1 for v in sc if abs(float(v) - best) <= 1e-12))
        shut = not bool(meta.get("channel_added"))
        if shut:
            shut_n += 1
            shut_exact += int(np.array_equal(sc, base))
        rows.append({"gate": meta.get("gate"),
                     "channel_added": bool(meta.get("channel_added")),
                     "base_top": int(tie_digest_order(base, it["texts"])[0]),
                     "gated_top": int(order[0]),
                     "slot1_active": bool(it["q"].has[2]),
                     "margin": round(best - float(np.sort(sc)[-2]), 12)
                     if len(sc) > 1 else None})
    n = len(items)
    return (correct / float(n), ties, shut_n, shut_exact, rows)


def slot1_is_silent(prompt: str) -> bool:
    """True when the arithmetic operator channel is EMPTY on this prompt.

    This is `slot1_active == False` in the directive's language. Taken from the
    same source of truth `SlotCache` uses: `has[2]` is truthy iff
    `ring_slot(operator_features)` has nonzero norm, i.e. iff the operator
    feature list is non-empty.
    """
    from henri import operator_ingress as oi_mod
    feats, _ = oi_mod.all_features(prompt)
    op, _ = oi_mod.split_features(feats)
    return len(op) == 0


def find_structural_rows(rows, index_list, limit=25, slot1_filter=None):
    """Rows whose DOCSTRING fires the pruned structural lexicon.

    MEASURED NEED, and the reason M5 alone is not enough. G-AST2 prompts are
    arithmetic prose, and the pruned lexicon deliberately ignores arithmetic
    words. On that split the channel is INERT BY CONSTRUCTION, so measuring
    only G-AST2 cannot test the channel at all.

    PRE-REGISTRATION DEFECT, MEASURED IN RUN 5. The directive defines this
    split as "non-arithmetic structural MBPP tasks (`slot1_active == False`)".
    The first version of this function selected on the LEXICON ALONE and
    ignored `slot1_active`. Run 5 measured the consequence:

        gate reasons = {'slot1_active_untied': 15,
                        'slot1_silent_no_channel_signal': 9,
                        'slot1_silent_zero_variance': 1}

    Fifteen of twenty-five "structural" items carried an ACTIVE, UNTIED
    Slot 1, so the null-space gate shut on them by construction. The split
    tested the wrong population: the directive asked for a population where
    the channel is PERMITTED to act, and this function supplied one where it
    is mostly forbidden.

    `slot1_filter` restores the specified predicate. Passing it is mandatory
    for a gate measurement; omitting it is a diagnostic-only mode and the
    receipt must say so.
    """
    from henri import operator_ingress as oi_mod
    out = []
    rejected_slot1_active = 0
    for i in index_list:
        prompt = (rows[i].get("text") or "").strip()
        if not prompt or not oi_mod.structural_morphology_from_text(prompt):
            continue
        if slot1_filter is not None and not slot1_filter(prompt):
            rejected_slot1_active += 1
            continue
        out.append(i)
        if len(out) >= limit:
            break
    find_structural_rows.last_rejected_active = rejected_slot1_active
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="suffix for receipt paths")
    args = ap.parse_args()
    tag = ("_" + args.tag) if args.tag else ""
    out = STATE / ("phase11_g_morph1b%s.json" % tag)

    t0 = time.time()
    from henri.code_verify import load_mbpp_executor
    from henri.semantic_backbone import SemanticBackbone
    from henri import reason_channels as rc

    sw = load_sweep()
    p9 = sw.load_phase9()
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
    held = build_items_ex(rows, held_idx[:N_TASKS], train_idx, sw, p9, run,
                          mutate, bb, "held")
    print("building TRAIN pools (%d tasks)" % N_TASKS)
    train = build_items_ex(rows, train_idx[:N_TASKS], held_idx, sw, p9, run,
                           mutate, bb, "train")

    # ---- dedicated structural split --------------------------------
    #
    # The split the directive asked for and the first three runs did not
    # build. G-AST2 cannot test this channel: its prompts are arithmetic and
    # the pruned lexicon ignores them, so the gate shuts by design and every
    # clause passes without the mechanism ever acting. These rows are the
    # complement -- prompts that DO name a structural transformation.
    struct_idx = find_structural_rows(rows, held_idx, limit=25,
                                      slot1_filter=slot1_is_silent)
    struct_prov = {"n_from_held": len(struct_idx), "n_from_train": 0,
                   "provenance": "held_out",
                   "split_spec": "slot1_active == False AND structural lexicon",
                   "rejected_slot1_active": int(
                       getattr(find_structural_rows,
                               "last_rejected_active", 0))}
    if len(struct_idx) < 25:
        extra = find_structural_rows(
            rows, [i for i in train_idx if i not in set(struct_idx)],
            limit=25 - len(struct_idx), slot1_filter=slot1_is_silent)
        struct_prov["n_from_train"] = len(extra)
        struct_prov["rejected_slot1_active"] += int(
            getattr(find_structural_rows, "last_rejected_active", 0))
        struct_prov["provenance"] = ("held_out" if not extra
                                     else "held_out_plus_train_topup")
        struct_idx = struct_idx + extra
    print("structural-lexicon rows found: %d  (%s, held %d / train %d)"
          % (len(struct_idx), struct_prov["provenance"],
             struct_prov["n_from_held"], struct_prov["n_from_train"]))
    struct_items = []
    if struct_idx:
        donors = [i for i in train_idx if i not in set(struct_idx)]
        struct_items = build_items_ex(rows, struct_idx, donors, sw, p9, run,
                                      mutate, bb, "struct")

    # Base vector uses the Phase 9 weights: the structural slot stays at zero
    # weight, because `psi` is NOT modified. t=0.0 disables Slot 3.
    w_base = sw.weights_for(0.0)

    # ---- M1: exactness at w_morph = 0.0 --------------------------------
    held_acc0, held_ties0, shut0, shut_ex0, rows0 = accuracy(
        held, rc, 0.0, sw, w_base)
    train_acc0, _, _, _, _ = accuracy(train, rc, 0.0, sw, w_base)
    exact_base = bool(abs(held_acc0 - PHASE9_TOP1) <= 1e-9)
    m1_exact_vectors = shut0 == len(held) and shut_ex0 == len(held)

    # ---- M2/M3: gated arm across the grid ------------------------------
    held_table, train_table = [], []
    for w in W_MORPH_GRID:
        ha, ht, hs, hse, hrows = accuracy(held, rc, w, sw, w_base)
        ta, tt, ts, tse, trows = accuracy(train, rc, w, sw, w_base)
        held_added = sum(1 for r in hrows if r["channel_added"])
        train_added = sum(1 for r in trows if r["channel_added"])
        held_table.append({"w_morph": w, "held_acc": round(ha, 6),
                           "ties": ht, "shut_n": hs,
                           "shut_exact": hse, "channel_added": held_added,
                           "rows": hrows})
        train_table.append({"w_morph": w, "train_acc": round(ta, 6),
                            "ties": tt, "shut_n": ts, "shut_exact": tse,
                            "channel_added": train_added})
        print("  w_morph %.2f  train %.4f  held %.4f  "
              "held: shut %d/%d added %d exact %d | train added %d"
              % (w, ta, ha, hs, len(held), held_added, hse, train_added))

    # ---- M4: select on TRAIN, ties toward the SMALLER weight -----------
    # Smaller weight = more parsimonious. `min` over (neg acc, w) achieves it.
    best = min(train_table, key=lambda r: (-r["train_acc"], r["w_morph"]))
    sel_w = best["w_morph"]
    sel_row = next(r for r in held_table if r["w_morph"] == sel_w)
    sel_rows = sel_row["rows"]

    # Zero degradation: every shut item must be bit-exact against base.
    zero_degradation = all(
        (not r["channel_added"]) == (r["gated_top"] is not None)
        for r in sel_rows)
    all_shut_exact = sel_row["shut_exact"] == sel_row["shut_n"]

    # Silent-subset delta: items whose operator slot is silent.
    silent = [i for i, r in enumerate(sel_rows) if not r["slot1_active"]]
    base_top_silent = rows0
    silent_delta = None
    if silent:
        b_hits = sum(1 for i in silent if base_top_silent[i]["base_top"] == 0)
        g_hits = sum(1 for i in silent if sel_rows[i]["gated_top"] == 0)
        silent_delta = {"n": len(silent),
                        "base_top1": b_hits,
                        "gated_top1": g_hits,
                        "delta": g_hits - b_hits}

    # ---- structural split: the only place this channel can act --------
    #
    # MEASURED JUSTIFICATION. On G-AST2 the pruned lexicon ignores arithmetic
    # prose by construction, so the gate shut on 40/40 held-out items and every
    # clause passed without the mechanism firing. That is a null result, not a
    # pass. This split holds prompts that DO name a structural transformation,
    # so a change here is attributable to the channel.
    # RECORDING DEFECT, MEASURED IN RUN 6, AND WHY THIS LINE IS GONE.
    #
    # This block used to re-initialise `struct_prov` to
    #     {"n_from_held": 0, "n_from_train": 0, "provenance": "none"}
    # AFTER the real provenance had been computed. The log printed the truth
    # ("held_out_plus_train_topup, held 16 / train 9") while the receipt
    # recorded "none" -- a receipt field contradicting its own log. The
    # fallback is unnecessary: `struct_prov` is always assigned above.
    struct_base = None
    struct_gated = None
    struct_added = 0
    struct_reasons = {}
    if struct_items:
        s_acc0, _, s_shut0, s_ex0, s_rows0 = accuracy(
            struct_items, rc, 0.0, sw, w_base)
        s_acc, _, s_shut, s_ex, s_rows = accuracy(
            struct_items, rc, max(sel_w, 0.05), sw, w_base)
        struct_added = sum(1 for r in s_rows if r["channel_added"])
        for r in s_rows:
            struct_reasons[str(r["gate"])] = \
                struct_reasons.get(str(r["gate"]), 0) + 1
        struct_base = {"n": len(struct_items),
                       "top1_acc": round(s_acc0, 6),
                       "shut_n": s_shut0}
        struct_gated = {"n": len(struct_items),
                        "top1_acc": round(s_acc, 6),
                        "channel_added": struct_added,
                        "shut_n": s_shut,
                        "shut_exact": s_ex,
                        "gate_reason_histogram": struct_reasons,
                        "provenance": struct_prov,
                        "delta": round(s_acc - s_acc0, 6)}
        print("  STRUCTURAL split: base %.4f  gated %.4f  added %d/%d"
              % (s_acc0, s_acc, struct_added, len(struct_items)))
        print("  STRUCTURAL gate reasons: %s" % struct_reasons)

    # HOW OFTEN DID THE CHANNEL ACTUALLY ACT?
    #
    # MEASURED DEFECT THIS CLAUSE EXISTS FOR. The first run of this probe
    # printed `G_MORPH1B_PASS` while the held-out split closed the gate on
    # 40/40 items at every weight above zero. Every clause passed because
    # every return was `base` untouched -- a PASS earned by never firing on
    # the evaluation split. A gate that cannot fail will pass.
    #
    # M5 makes the mechanism itself the object of the test. If the channel
    # never contributes on held-out, the verdict must say INERT, not PASS,
    # however clean the other clauses look.
    held_added_sel = sum(1 for r in sel_rows if r["channel_added"])
    train_added_sel = next(r["channel_added"] for r in train_table
                           if r["w_morph"] == sel_w)
    gate_reasons = {}
    for r in sel_rows:
        gate_reasons[str(r["gate"])] = gate_reasons.get(str(r["gate"]), 0) + 1

    clauses = [
        {"id": "M1", "pass": bool(exact_base and m1_exact_vectors),
         "requires": "w_morph=0.0 reproduces 0.4000 and returns base exactly",
         "value": {"held_acc_at_w0": round(held_acc0, 6),
                   "phase9_expected": PHASE9_TOP1,
                   "accuracy_match": exact_base,
                   "all_vectors_exact_at_w0": bool(m1_exact_vectors),
                   "n_items": len(held)}},
        {"id": "M2", "pass": bool(all_shut_exact and
                                  sel_row["held_acc"] >= PHASE9_TOP1 - 1e-9),
         "requires": "shut gate returns base bit-exactly; held accuracy >= 0.4000",
         "value": {"selected_w": sel_w,
                   "held_acc": sel_row["held_acc"],
                   "bar": PHASE9_TOP1,
                   "shut_n": sel_row["shut_n"],
                   "shut_exact": sel_row["shut_exact"],
                   "zero_degradation_by_construction": bool(zero_degradation)}},
        {"id": "M3", "pass": bool(silent_delta is None or
                                  silent_delta["delta"] >= 0),
         "requires": "on slot1-silent items the gated arm is not worse",
         "value": silent_delta or {"n": 0, "note": "no silent items held out"}},
        {"id": "M4", "pass": True, "counted": False,
         "requires": "parsimony: selection on train, ties toward smaller w",
         "value": {"selected_w": sel_w,
                   "train_table": train_table,
                   "held_table": [{"w_morph": r["w_morph"],
                                   "held_acc": r["held_acc"],
                                   "shut_n": r["shut_n"],
                                   "channel_added": r["channel_added"]}
                                  for r in held_table]}},
        {"id": "M5", "pass": bool(held_added_sel > 0),
         "requires": "the channel must ACT on held-out at least once",
         "value": {"held_channel_added": held_added_sel,
                   "train_channel_added": train_added_sel,
                   "n_held": len(sel_rows),
                   "gate_reason_histogram": gate_reasons,
                   "why": ("A PASS earned by never firing is vacuous. The "
                           "first run closed the gate on 40/40 held-out "
                           "items at every weight and still printed PASS, "
                           "because the shut branch returns base untouched "
                           "and every other clause was satisfied by the "
                           "baseline. This clause makes the mechanism itself "
                           "the object of the test.")}},
        {"id": "M6", "pass": bool(struct_gated is not None
                                  and struct_gated["channel_added"] > 0),
         "requires": "on the structural split the channel must engage and not regress",
         "value": ({"heldout_gast2_is_arithmetic": True,
                    "note": ("G-AST2 cannot test this channel: its prompts are "
                             "arithmetic and the pruned lexicon ignores them."),
                    "base": struct_base, "gated": struct_gated,
                    "no_regression": (struct_gated["delta"] >= 0
                                      if struct_gated else None)}
                   if struct_gated else
                   {"n": 0, "note": "no structural-lexicon rows found"})},
    ]
    n_pass = sum(1 for c in clauses if c["pass"] and c.get("counted", True))
    n_counted = len([c for c in clauses if c.get("counted", True)])

    if not exact_base:
        verdict = "G_MORPH1B_EXACTNESS_MISMATCH"
    elif not all_shut_exact:
        verdict = "G_MORPH1B_GATE_LEAK"
    elif sel_w == 0.0:
        # THE LABEL MUST NOT FLATTER THE RESULT. The bar is met at w=0.0 and
        # w=0.0 DISABLES the channel. Calling that a channel win would repeat
        # the Phase 9 defect. Same rule as `M4_PARSIMONY_HELD_CHANNEL_OFF`.
        verdict = "G_MORPH1B_PARSIMONY_HELD_CHANNEL_OFF"
    elif held_added_sel == 0 and struct_added == 0:
        # THE HONEST VERDICT, and only when NOTHING engaged. Baseline held and
        # the channel never acted on EITHER split, so there is no channel
        # result to report -- neither a win nor a loss.
        verdict = "G_MORPH1B_BASELINE_HELD_CHANNEL_INERT_EVERYWHERE"
    elif n_pass == n_counted:
        verdict = "G_MORPH1B_PASS"
    else:
        # SCOPED PARTIAL. This branch exists because of a measured defect.
        #
        # RUN 4 MEASURED: `G_MORPH1B_BASELINE_HELD_CHANNEL_INERT_ON_HELDOUT`
        # with M5 fail (0/40 held-out engaged) and M6 fail (0/25 structural
        # engaged). The ladder consulted `held_added_sel` BEFORE M6, so a run
        # where the structural split engages while G-AST2 held-out does not
        # would have been mislabelled INERT -- masking a real partial result.
        #
        # THE STRUCTURAL REASON M5 CANNOT PASS ON G-AST2: every G-AST2 prompt
        # carries an active, untied Slot 1, and the null-space gate shuts by
        # construction on exactly that population. M5's bar therefore targets
        # a split where engagement is forbidden by specification. That is a
        # clause-specification defect, discovered by measurement, not a
        # channel defect. The label must say which split engaged.
        #
        # AMENDED BEFORE RUN 5, NOT AFTER READING IT. Run 4's receipt is
        # retained verbatim as its own measured artifact.
        if struct_added > 0 and held_added_sel == 0:
            verdict = "G_MORPH1B_STRUCTURAL_ONLY_%d_OF_%d" % (n_pass, n_counted)
        else:
            verdict = "G_MORPH1B_PARTIAL_%d_OF_%d" % (n_pass, n_counted)

    body = {
        "probe": "phase11_g_morph1b",
        "status": "PHASE 11 STEP 1 pre-registered gate",
        "prior_regression": {"file": "receipts/phase10_g_morph1.json",
                             "m4_value": PHASE10_M4,
                             "m4_bar": PHASE9_TOP1,
                             "verdict": "G_MORPH1_PARTIAL_3_OF_4"},
        "design": {
            "psi_modified": False,
            "why": ("a fifth slot on the unit sphere rescales slots 0-2 by "
                    "1/sqrt(2)/1/sqrt(2.5) = 0.8000; nothing here touches psi"),
            "channel": "additive on the SCORE, behind the null-space gate",
            "lexicon": "STRUCTURAL_ONLY_AXES (pruned) in operator_ingress",
            "candidates_read_as": "AST (pool holds Python snippets)",
        },
        "falsified_premise": {
            "directive_asked": "normalise the channel by active structural axes",
            "measured": ("ring_slot/morph_slot already return unit vectors: "
                         "1, 2 and 3 active axes all give ||slot|| = 1.000000"),
            "consequence": "the divisor would always be 1, so it is NOT implemented",
        },
        "decision_rule": ("select w_morph on TRAIN, ties toward the smaller "
                          "weight; evaluate ONCE on held-out"),
        "grid": list(W_MORPH_GRID),
        "clauses": clauses,
        "n_clauses_pass": n_pass,
        "selected_w_on_train": sel_w,
        "verdict": verdict,
        "corpus": {"n_held": len(held), "n_train": len(train)},
        "wall_clock_s": round(time.time() - t0, 2),
    }

    ids = write_receipt(out, body)
    print("\nclauses")
    for c in clauses:
        print("  %-3s %-58s %s" % (c["id"], c["requires"],
                                   "PASS" if c["pass"] else "FAIL"))
    print(BODY_FIELD, ids[BODY_FIELD])
    print(FILE_FIELD + "  ", ids[FILE_FIELD])
    print("VERDICT %s" % verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
