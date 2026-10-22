"""Phase 8 ACTION 1 gate probe: operator ingress against its controls.

PRE-REGISTERED BEFORE THE FIRST RUN, and not changed afterwards.

WHY THIS PROBE EXISTS SEPARATELY FROM `scripts/mbpp_wave_eval.py`
-----------------------------------------------------------------
MEASURED DEFECT IN THE PHASE 6 HARNESS. `mbpp_wave_eval.py` scores

    q  = bb.stem.encode(r["text"])
    cs = [bb.stem.encode(c[:800]) for c in pool]
    scores = cs @ q

That is a cosine between 384-dimensional SENTENCE EMBEDDINGS. It never
calls `SemanticBackbone.encode`, so it never builds a wave. The Phase 6
receipt therefore describes the STEM, and no change to wave ingress can
move its numbers. Feeding an operator-aware encoder into that harness
would have "failed" the gate for a reason unrelated to the mechanism.

This probe measures three arms on the SAME 60 held-out tasks and the SAME
shuffled K=8 pools, so the comparison is paired:

  A  stem_cosine    the Phase 6 harness, reproduced as the historical
                    baseline. Expected near 0.0 top-1.
  B  wave_v6        `SemanticBackbone` full wave (slots 0+1 only). This is
                    the static control the directive names.
  C  wave_operator  `OperatorAwareBackbone` (slots 0..3). The candidate.

ARMS B AND C SKIP `strip`. `strip_common_option_tokens` removes tokens
shared by a majority of options and present in the prompt. That is correct
for near-duplicate multiple-choice options and wrong for programs, where a
shared identifier is often the discriminating one. Stripping here would
have changed two variables at once.

PRE-REGISTERED THRESHOLDS (G-AST1 corpus clauses)
-------------------------------------------------
n = 60, K = 8, chance top-1 = 0.1250.

  (d) mean tie_group_size(C) < 1.6333        Phase 6 baseline
  (e) top-1(C) z >= +2.5 against 0.1250      requires >= 14/60 = 0.2333
  (f) top-1(C) > top-1(B)                    the mechanism must beat the
                                             static control, not merely
                                             differ from it
  (g) mutation validity >= 0.90              otherwise the closed-loop
                                             number measures the pool

VERDICT RULE, fixed now. `G_AST1_PASS` requires (d) AND (e) AND (f) AND
(g). Anything else is reported as the measured number with the failing
clause named. No threshold is moved after seeing the result. In
particular (e) is NOT relaxed to "above the static control": a mechanism
that beats a broken control while staying at chance has not demonstrated
ranking.

CLOSED LOOP is reported for every arm and is NOT a gate. A verified pass
proves the candidate executes; it does not prove the ranker chose well.

Evidence labels: every number printed here is OBSERVED.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.mbpp_wave_eval import (K_POOL, load_mbpp, mutate,  # noqa: E402
                                    run_candidate, split_mbpp)

N_EVAL = 60
CHANCE = 1.0 / K_POOL
Z_BAR = 2.5
TIE_BAR = 1.6333
MUT_BAR = 0.90


def _tie(scores: np.ndarray) -> tuple:
    mx = float(scores.max())
    group = np.flatnonzero(np.isclose(scores, mx, atol=1e-12))
    return int(group.size), group


def _paired(a_ok: int, b_ok: int, n: int) -> dict:
    """Paired z on the DISCORDANT pairs, plus the raw counts."""
    # McNemar on discordant cells. Reported because the repo's standard is
    # a paired test, and an unpaired z on two arms that share the pool
    # overstates independence.
    return {"a_correct": int(a_ok), "b_correct": int(b_ok), "n": int(n)}


def main() -> int:
    started = time.time()
    rows = load_mbpp()
    _train, held = split_mbpp(rows)
    n = min(len(held), N_EVAL)

    from henri.operator_ingress import OperatorAwareBackbone
    from henri.semantic_backbone import SemanticBackbone

    bb = SemanticBackbone()
    aware = OperatorAwareBackbone(base=bb)

    arms = ("stem_cosine", "wave_v6", "wave_operator")
    top1 = {a: 0 for a in arms}
    top3 = {a: 0 for a in arms}
    execv = {a: 0 for a in arms}
    groups = {a: [] for a in arms}
    ingroup = {a: 0 for a in arms}
    ctrl = {a: 0.0 for a in arms}
    picks = {a: [] for a in arms}
    gold_positions = []
    mut_runs = mut_pass = 0
    gold_pass = 0
    reasons: dict = {}
    disagreement = 0

    for r in held[:n]:
        gold = r["code"]
        pool = [gold] + [mutate(gold, k) for k in range(K_POOL - 1)]
        prng = np.random.default_rng(r["task_id"] % (2 ** 32))
        order = prng.permutation(len(pool))
        pool = [pool[int(i)] for i in order]
        gold_pos = int(np.where(order == 0)[0][0])
        gold_positions.append(gold_pos)

        # Execute every candidate ONCE. Two purposes: the closed-loop column
        # for each arm, and the mutation-validity gate (g). A mutation that
        # still passes is not a negative.
        passed = np.zeros(len(pool), dtype=bool)
        for j, cand in enumerate(pool):
            out = run_candidate(cand, r["test_list"])
            passed[j] = bool(out.get("ok", False))
            reasons[out["reason"]] = reasons.get(out["reason"], 0) + 1
        gold_pass += int(passed[gold_pos])
        mut_pass += int(passed.sum()) - int(passed[gold_pos])
        mut_runs += len(pool) - 1

        # -- arm A: the Phase 6 stem harness, reproduced ------------------
        qs = bb.stem.encode(r["text"])
        qs = qs / max(float(np.linalg.norm(qs)), 1e-30)
        cs = np.stack([bb.stem.encode(c[:800]) for c in pool], axis=0)
        cs = cs / np.maximum(np.linalg.norm(cs, axis=1, keepdims=True), 1e-30)
        sa = np.round(cs @ qs, 12)

        # -- arm B: static v6 wave (slots 0+1 only) -----------------------
        qb = bb.encode(r["text"]).psi
        sb = np.round(np.array([
            float(np.real(np.vdot(qb, bb.encode(c[:2000]).psi)))
            for c in pool]), 12)

        # -- arm C: operator-aware wave (slots 0..3) ----------------------
        qc = aware.encode(r["text"]).psi
        sc = np.round(np.array([
            float(np.real(np.vdot(qc, aware.encode(c[:2000]).psi)))
            for c in pool]), 12)

        for arm, s in (("stem_cosine", sa), ("wave_v6", sb),
                       ("wave_operator", sc)):
            ranked = list(np.argsort(-s))
            top1[arm] += int(ranked[0] == gold_pos)
            top3[arm] += int(gold_pos in ranked[:3])
            execv[arm] += int(bool(passed[ranked[0]]))
            g, group = _tie(s)
            groups[arm].append(g)
            gi = bool(gold_pos in group)
            ingroup[arm] += int(gi)
            # SIZE-MATCHED control, CONDITIONED on gold being in the group.
            # The Phase 6 defect: adding 1/|G| unconditionally counted a hit
            # when gold was absent and produced a fake lift of 0.2353.
            if group.size and gi:
                ctrl[arm] += 1.0 / float(group.size)
            picks[arm].append(int(ranked[0]))

        if int(np.argmax(sc)) != int(np.argmax(sb)):
            disagreement += 1

    rec: dict = {
        "probe": "phase8_ast_ingress",
        "n_eval": n, "k_pool": K_POOL, "chance_top1": round(CHANCE, 4),
        "z_bar": Z_BAR, "tie_bar": TIE_BAR, "mutation_bar": MUT_BAR,
        "preregistration": "encoded in the module docstring before run 1",
        "arms": {},
    }

    for arm in arms:
        p = top1[arm] / n
        # One-proportion z against 1/K. Two-sided reported as one-sided
        # because the claim is directional ("beats chance").
        se = float(np.sqrt(CHANCE * (1.0 - CHANCE) / n))
        z = (p - CHANCE) / se
        rec["arms"][arm] = {
            "top1": round(p, 4), "top1_count": int(top1[arm]),
            "top3": round(top3[arm] / n, 4),
            "z_vs_chance": round(float(z), 4),
            "mean_tie_group": round(float(np.mean(groups[arm])), 4),
            "gold_in_max_group": round(ingroup[arm] / n, 4),
            "size_matched_control": round(ctrl[arm] / n, 4),
            "lift_vs_control": round((ingroup[arm] / n)
                                     / max(1e-9, ctrl[arm] / n), 4),
            "exec_verified_top1": round(execv[arm] / n, 4),
        }

    # Paired comparison C vs B, using the stored picks.
    pb, pc = picks["wave_v6"], picks["wave_operator"]
    gp = gold_positions
    only_c = sum(1 for a, b, g in zip(pb, pc, gp) if b == g and a != g)
    only_b = sum(1 for a, b, g in zip(pb, pc, gp) if a == g and b != g)
    rec["paired_c_vs_b"] = {
        "c_only_correct": int(only_c), "b_only_correct": int(only_b),
        "disagreements": int(only_c + only_b),
        "picks_differ": int(disagreement),
        "note": "McNemar counts on the discordant cells only.",
    }

    mut_rate = mut_pass / max(1, mut_runs)
    rec["G_P6E6_mutation_validity"] = {
        "runs": int(mut_runs), "passes": int(mut_pass),
        "pass_rate": round(mut_rate, 4),
        "valid_rate": round(1.0 - mut_rate, 4),
    }
    rec["gold_oracle_pass_rate"] = round(gold_pass / n, 4)
    rec["exec_reasons"] = reasons

    c = rec["arms"]["wave_operator"]
    d = c["mean_tie_group"] < TIE_BAR
    e = c["z_vs_chance"] >= Z_BAR
    f = c["top1"] > rec["arms"]["wave_v6"]["top1"]
    g = (1.0 - mut_rate) >= MUT_BAR
    rec["gates"] = {
        "G_AST1_d_tie_group": {"measured": c["mean_tie_group"],
                               "bar": TIE_BAR, "pass": bool(d)},
        "G_AST1_e_z": {"measured": c["z_vs_chance"], "bar": Z_BAR,
                       "pass": bool(e)},
        "G_AST1_f_beats_static": {"measured_c": c["top1"],
                                  "measured_b": rec["arms"]["wave_v6"]["top1"],
                                  "pass": bool(f)},
        "G_AST1_g_mutation_validity": {"measured": round(1.0 - mut_rate, 4),
                                       "bar": MUT_BAR, "pass": bool(g)},
    }
    rec["verdict"] = ("G_AST1_PASS" if (d and e and f and g)
                      else "G_AST1_FAIL")
    rec["wall_clock_sec"] = round(time.time() - started, 1)

    out = REPO / "receipts" / "phase8_ast_ingress.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    # BYTES, not text. The repo's own defect record: `write_text` on
    # Windows translates newlines, so the bytes hashed are not the bytes
    # stored. Readback follows.
    out.write_bytes(json.dumps(rec, indent=2).encode("utf-8") + b"\n")

    for k in ("arms", "gates", "G_P6E6_mutation_validity",
              "gold_oracle_pass_rate", "paired_c_vs_b", "verdict"):
        print(f"{k}: {json.dumps(rec[k])}")
    print(f"receipt: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
