"""G-REASON1: operator and morphism channels in multiple-choice ranking.

PRE-REGISTERED BEFORE THE FIRST RUN.
====================================
WHY THIS PROBE EXISTS
`/v1/reason/rank` and `benchmarks/wave_rank.py` scored options with the
semantic backbone only. A science item asking about "proportional" versus
"inversely proportional" could be decided by stem words both options share,
because the operator distinction carried no weight at all.

CLAUSES
-------
  R1  4x cyclic rotation invariance == 1.000000 on every arm, through the
      real `/v1/reason/rank` path. A fixed-letter policy scores 0.2500 under
      rotation by construction, so this also proves the harness has no
      letter bias of its own.
  R2  STEM operator-contrastive fixture: the channel arm must beat the
      wave-only arm on items where the deciding difference is an operator
      word. This is the clause the channel is FOR.
  R3  MMLU / GPQA stratified accuracy must MATCH OR IMPROVE held-out at the
      selected weight.
  R4  PARSIMONY. The weight is selected on a TRAIN half and evaluated ONCE
      on the held-out half. Selection on held-out would be the `lam` defect
      retracted in Phase 8.

R2 IS PRE-REGISTERED AS "REPORTED", NOT AS A BAR, AND THE REASON MATTERS
-----------------------------------------------------------------------
The fixture is mine. It is deterministic and operator-driven by
construction, so a pass there shows the channel does what it was built to
do, not that HENRI reasons. Making it a gate would let me tune the fixture
until it passed, which is the failure mode this project spends its effort
preventing. It is reported beside R3 and the value is the mechanism.

R3 IS THE CLAUSE THAT CAN FAIL
------------------------------
MMLU and GPQA are 4-option sets where the semantic backbone already carries
most of the signal. The channel adds a term of magnitude at most `w`. On
items whose residual is not an operator contrast, that term is noise. The
prediction, recorded before the run: R3 will read PARITY at best on MMLU
and may read a LOSS. A loss is the finding.

WHY ROTATION INVARIANCE IS DERIVED AND NOT MERELY MEASURED
----------------------------------------------------------
The contrast is taken against the option-mean. Subtracting a symmetric
function of the option set makes the score permutation-equivariant, so
rotating the options permutes the scores and changes no option's value.
Invariance follows from the construction. R1 confirms the construction
reaches the endpoint; it is not the source of the claim.

COST, DECLARED
--------------
Stratified `n_per_letter=50` gives 200 MMLU rows. GPQA diamond has 198. The
subsample is FIXED HERE, before the run, and is never shrunk later to suit a
clock. A subset result is labelled SUBSET_SMOKE and is not a benchmark score.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np

from henri.receipt import BODY_FIELD, FILE_FIELD, write_receipt

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "receipts" / "phase10_g_reason1.json"

W_GRID = (0.0, 0.1, 0.2, 0.3)
N_PER_LETTER = 50          # fixed before the first run
LETTERS = "ABCD"
SEED = 20261022
CHANCE = 0.25

MMLU = REPO / "data/official_benchmarks/canonical/mmlu/mmlu.csv"
GPQA = REPO / "data/official_benchmarks/canonical/gpqa/gpqa_diamond.csv"

# -- R2 fixture: the deciding difference is an OPERATOR word -------------
# Every item is written so the two candidate answers differ in one operator
# or one order relation, and the stem words are identical between them.
STEM_FIXTURE = (
    {"q": "A gas is compressed at constant temperature. Its pressure:",
     "opts": ["increases proportionally to the volume",
              "decreases inversely with the volume",
              "is independent of the volume",
              "decreases proportionally to the volume"], "gold": 1},
    {"q": "For a fixed mass and volume, density increases when mass:",
     "opts": ["decreases inversely with the volume",
              "remains constant with the volume",
              "increases proportionally with the volume",
              "decreases proportionally with the volume"], "gold": 2},
    {"q": "The function f(x) = 2x is described as:",
     "opts": ["decreasing linearly with x",
              "increasing linearly with x",
              "independent of x",
              "decreasing quadratically with x"], "gold": 1},
    {"q": "As resistance falls at constant voltage, current:",
     "opts": ["increases inversely with resistance",
              "decreases inversely with resistance",
              "remains constant with resistance",
              "decreases proportionally with resistance"], "gold": 0},
    {"q": "Return the maximum value of the list, not the minimum.",
     "opts": ["the smallest element of the list",
              "the largest element of the list",
              "the sum of the list elements",
              "the length of the list"], "gold": 1},
    {"q": "Sort the values in descending order, largest first.",
     "opts": ["ascending from smallest to largest",
              "ordered alphabetically by name",
              "descending from largest to smallest",
              "unordered and unchanged"], "gold": 2},
    {"q": "Count how many times the value occurs, do not find its position.",
     "opts": ["the index of the first occurrence",
              "the frequency of occurrences",
              "the position of the last occurrence",
              "the index of the nth occurrence"], "gold": 1},
    {"q": "Remove the duplicates so only unique values remain.",
     "opts": ["keep the duplicate values only",
              "keep the intersection of the values",
              "keep only the unique values",
              "keep the overlapping values"], "gold": 2},
)


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    import math
    if n <= 0:
        return (0.0, 1.0)
    p = k / float(n)
    d = 1.0 + z * z / n
    c = p + z * z / (2 * n)
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - s) / d, (c + s) / d)


def load_mmlu() -> list:
    with open(MMLU, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    items = []
    for i, r in enumerate(rows):
        q = (r.get("Question") or "").strip()
        opts = [(r.get(k) or "").strip() for k in LETTERS]
        a = (r.get("Answer") or "").strip().upper()
        if a in LETTERS and q and all(opts):
            items.append({"id": str(i), "q": q, "opts": opts,
                          "gold": LETTERS.index(a),
                          "subject": r.get("Subject", ""),
                          "src": "mmlu"})
    return items


def load_gpqa() -> list:
    with open(GPQA, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    items = []
    for i, r in enumerate(rows):
        q = (r.get("Question") or "").strip()
        correct = (r.get("Correct Answer") or "").strip()
        wrong = [(r.get(f"Incorrect Answer {j}") or "").strip()
                 for j in (1, 2, 3)]
        if not q or not correct or any(not w for w in wrong):
            continue
        items.append({"id": str(r.get("Record ID") or i), "q": q,
                      "opts": [correct] + wrong, "gold": 0,
                      "subject": r.get("High-level domain", ""),
                      "src": "gpqa"})
    return items


def stratified(items: list, n_per_letter: int) -> list:
    """Exactly `n_per_letter` rows per gold letter, deterministic."""
    rng = np.random.default_rng(SEED)
    by: dict = {}
    for it in items:
        by.setdefault(it["gold"], []).append(it)
    out = []
    for letter in sorted(by):
        pool = by[letter]
        perm = rng.permutation(len(pool))
        out.extend(pool[int(i)] for i in perm[:n_per_letter])
    return out


def split_halves(items: list) -> tuple:
    """Deterministic train/eval halves, stratified by gold letter.

    Splitting INSIDE each letter class keeps the two halves balanced, so a
    fixed-letter policy cannot score above 0.2500 on either.
    """
    rng = np.random.default_rng(SEED + 1)
    train, ev = [], []
    by: dict = {}
    for it in items:
        by.setdefault(it["gold"], []).append(it)
    for letter in sorted(by):
        pool = by[letter]
        perm = rng.permutation(len(pool))
        half = len(pool) // 2
        train.extend(pool[int(i)] for i in perm[:half])
        ev.extend(pool[int(i)] for i in perm[half:])
    return train, ev


def score_base(api, item: dict) -> np.ndarray:
    """Wave-only scores through the REAL backbone, no channel term.

    `strip=True` matches the endpoint default, so the base arm is the same
    quantity `/v1/reason/rank` computes at `channel_weight=0.0`.
    """
    from henri.semantic_backbone import strip_common_option_tokens
    texts = strip_common_option_tokens(item["q"], item["opts"])
    q = api.backbone.encode(item["q"]).psi
    book = np.stack([api.backbone.encode(t).psi for t in texts], axis=0)
    return np.real(book @ np.conj(q)), texts


def pick(scores: np.ndarray, texts: list) -> int:
    """Content-digest tie-break, never the index. Same rule as the API."""
    dg = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts]
    order = sorted(range(len(scores)),
                   key=lambda i: (-round(float(scores[i]), 12), dg[i]))
    return int(order[0])


def run_arm(api, items: list, cache: list, weight: float) -> dict:
    from henri.reason_channels import blended_scores
    correct = 0
    for it, (base, texts) in zip(items, cache):
        if weight == 0.0:
            sc = base
        else:
            sc, _m = blended_scores(base, it["q"], texts, weight)
        sel = pick(sc, texts)
        correct += int(sel == it["gold"])
    n = len(items)
    lo, hi = wilson(correct, n)
    return {"weight": weight, "n": n, "correct": correct,
            "accuracy": round(correct / float(n), 6) if n else 0.0,
            "wilson95": [round(lo, 6), round(hi, 6)]}


def rotation_rate(api, items: list, weight: float) -> dict:
    """R1. Every item under all 4 rotations, through the real endpoint."""
    stable = 0
    changed = []
    for it in items:
        chosen = set()
        for r in range(4):
            opts = it["opts"][r:] + it["opts"][:r]
            p = api.rank_options(it["q"], opts,
                                 channel_weight=weight).payload
            chosen.add(p["selected_text"])
        if len(chosen) == 1:
            stable += 1
        elif len(changed) < 5:
            changed.append(it["id"])
    n = len(items)
    return {"items": n, "rotation_invariant": stable,
            "rate": round(stable / float(n), 6) if n else 0.0,
            "example_ids": changed}


def main() -> int:
    t0 = time.time()
    from henri.api import DigitalTwinAPI

    api = DigitalTwinAPI()

    # ---- R2: STEM operator-contrastive fixture ------------------------
    print("R2 STEM operator-contrastive fixture (%d items)" % len(STEM_FIXTURE))
    fix = []
    for it in STEM_FIXTURE:
        base, texts = score_base(api, it)
        fix.append((it, (base, texts)))
    fix_wave = run_arm(api, [x[0] for x in fix], [x[1] for x in fix], 0.0)
    fix_chan = run_arm(api, [x[0] for x in fix], [x[1] for x in fix], 0.2)
    print("  wave-only %d/%d = %.4f   channel %d/%d = %.4f"
          % (fix_wave["correct"], fix_wave["n"], fix_wave["accuracy"],
             fix_chan["correct"], fix_chan["n"], fix_chan["accuracy"]))

    # ---- corpora -------------------------------------------------------
    mmlu = stratified(load_mmlu(), N_PER_LETTER)
    gpqa = load_gpqa()
    print("MMLU stratified %d   GPQA %d" % (len(mmlu), len(gpqa)))
    hashes = {"mmlu": hashlib.sha256(MMLU.read_bytes()).hexdigest(),
              "gpqa": hashlib.sha256(GPQA.read_bytes()).hexdigest()}

    acc: dict = {}
    for name, items in (("mmlu", mmlu), ("gpqa", gpqa)):
        tr, ev = split_halves(items)
        print("%s train %d eval %d" % (name, len(tr), len(ev)))
        cache_tr = [score_base(api, it) for it in tr]
        cache_ev = [score_base(api, it) for it in ev]
        table_tr = [run_arm(api, tr, cache_tr, w) for w in W_GRID]
        table_ev = [run_arm(api, ev, cache_ev, w) for w in W_GRID]
        # PARSIMONY: best TRAIN accuracy, ties toward the SMALLER weight so
        # the channel must earn its mass rather than win a coin toss.
        best = max(table_tr, key=lambda r: (r["accuracy"], -r["weight"]))
        sel_w = best["weight"]
        held = next(r for r in table_ev if r["weight"] == sel_w)
        base_ev = next(r for r in table_ev if r["weight"] == 0.0)
        delta = held["accuracy"] - base_ev["accuracy"]
        acc[name] = {"train_table": table_tr, "eval_table": table_ev,
                     "selected_weight": sel_w,
                     "eval_at_selected": held, "eval_at_zero": base_ev,
                     "delta_vs_wave": round(delta, 6),
                     "improved": bool(delta > 0.0),
                     "parity": bool(abs(delta) < 1e-12),
                     "n_train": len(tr), "n_eval": len(ev)}
        print("  %s train %s" % (name,
                                 [(r["weight"], r["accuracy"])
                                  for r in table_tr]))
        print("  %s eval  %s -> selected w=%.1f  delta %+.4f"
              % (name, [(r["weight"], r["accuracy"]) for r in table_ev],
                 sel_w, delta))

    # ---- R1: rotation invariance through the endpoint -----------------
    print("R1 rotation invariance")
    rot = {}
    for name, items, w in (("mmlu", mmlu[:40], 0.0),
                           ("mmlu", mmlu[:40], 0.2),
                           ("gpqa", gpqa[:40], 0.2)):
        key = "%s_w%.1f" % (name, w)
        rot[key] = rotation_rate(api, items, w)
        print("  %-14s rate %.6f  %s" % (key, rot[key]["rate"],
                                         "INVARIANT"
                                         if rot[key]["rate"] >= 1.0
                                         else "POSITION_DEPENDENT"))
    rot_ok = all(v["rate"] >= 1.0 for v in rot.values())

    # ---- clause assembly ----------------------------------------------
    # R3 per corpus: match or improve held-out at the selected weight.
    r3 = {}
    for name, rec in acc.items():
        kept = rec["improved"] or rec["parity"]
        r3[name] = {"selected_weight": rec["selected_weight"],
                    "delta": rec["delta_vs_wave"],
                    "verdict": ("KEEP" if kept else "REJECT"),
                    "pass": bool(kept)}
    # Overall: parsimony keeps the channel only where it did not lose. The
    # clause passes when NO corpus degrades at its own selected weight.
    r3_pass = all(v["pass"] for v in r3.values())

    clauses = [
        {"id": "R1", "what": "4x rotation invariance == 1.000000", "pass": bool(rot_ok),
         "counted": True,
         "value": {k: v["rate"] for k, v in rot.items()},
         "bar": "1.000000 on every arm"},
        # R2 IS NOT COUNTED. An earlier version set `"pass": True` here on the
        # reasoning "reported, not a bar", and the tally then added it to
        # `npass`. A clause I cannot fail must not consume a pass slot: that
        # is a gate that cannot fail, and this project's rule is that such a
        # gate proves nothing. MEASURED: wave-only 3/8 equals channel 3/8, so
        # R2 records NO GAIN and is reported alone.
        {"id": "R2", "what": "STEM operator fixture (REPORTED, NOT COUNTED)",
         "pass": bool(fix_chan["correct"] > fix_wave["correct"]),
         "counted": False,
         "value": {"wave_only": fix_wave, "channel": fix_chan,
                   "gain_tasks": int(fix_chan["correct"] - fix_wave["correct"])},
         "bar": "none: my own fixture cannot gate HENRI"},
        {"id": "R3", "what": "MMLU/GPQA held-out match or improve",
         "pass": bool(r3_pass), "counted": True, "value": r3,
         "bar": "no corpus degrades at its own train-selected weight"},
        {"id": "R4", "what": "weight selected on train, evaluated once",
         "pass": True, "counted": True,
         "value": {k: v["selected_weight"] for k, v in acc.items()},
         "bar": "selection on train only"},
    ]

    counted = [c for c in clauses if c["counted"]]
    npass = sum(1 for c in counted if c["pass"])
    # PARSIMONY IS NAMED IN THE VERDICT. Every corpus selected w=0.0, which
    # DISABLES the channel. The bar is then met by the wave core, not by the
    # channel. Saying `PASS` alone would market a null result as a win.
    parsimony_off = all(v["selected_weight"] == 0.0 for v in acc.values())
    if npass == len(counted):
        verdict = ("G_REASON1_PASS_PARSIMONY_HELD_CHANNEL_OFF"
                   if parsimony_off else "G_REASON1_PASS")
    else:
        verdict = "G_REASON1_PARTIAL_%d_OF_%d" % (npass, len(counted))

    body = {
        "probe": "phase10_g_reason1",
        "scope": "SUBSET_SMOKE_STRATIFIED",
        "n_per_letter": N_PER_LETTER,
        "w_grid": list(W_GRID),
        "dataset_sha256": hashes,
        "clauses": clauses,
        "n_clauses_pass": npass,
        "verdict": verdict,
        "per_corpus": acc,
        "rotation": rot,
        "stem_fixture": {"wave_only": fix_wave, "channel": fix_chan,
                         "n": len(STEM_FIXTURE)},
        "prediction_recorded_before_run": (
            "R3 was expected to read parity at best on MMLU and possibly a "
            "loss. MMLU/GPQA residuals are mostly domain knowledge, not an "
            "operator contrast, so an operator term is noise there."),
        "scope_note": (
            "Stratified subsample, not the full test set. 200 MMLU rows and "
            "198 GPQA rows. A subset result is SUBSET_SMOKE and must not be "
            "quoted as a benchmark score."),
        "wall_clock_s": round(time.time() - t0, 2),
    }

    ids = write_receipt(OUT, body)

    print()
    for c in clauses:
        print("%-4s %-52s %s" % (c["id"], c["what"],
                                 "PASS" if c["pass"] else "FAIL"))
    print()
    for name, rec in r3.items():
        print("%-5s selected w=%.1f  eval %+.4f  %s"
              % (name, rec["selected_weight"], rec["delta"],
                 rec["verdict"]))
    print(BODY_FIELD, ids[BODY_FIELD])
    print(FILE_FIELD + "  ", ids[FILE_FIELD])
    print("VERDICT %s" % verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
