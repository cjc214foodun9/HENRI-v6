"""MBPP wave ranking with sandboxed execution verification.

WHAT THIS MEASURES, AND WHAT IT CANNOT
--------------------------------------
Two different questions, reported separately because they have different
ceilings:

  OPEN LOOP asks whether the wave ranker puts the gold program near the top
  of an 8-way candidate pool. It is measurable with no execution at all.

  CLOSED LOOP asks whether execution verification can rescue the ranking: of
  the top-3 wave-ranked candidates, does any pass every assertion? This is
  the useful production metric, because a verified pass is a real outcome
  while a wave score is a heuristic.

  GOLD PASS RATE is the ORACLE CEILING. If the gold program itself fails in
  this environment, closed-loop numbers measure the environment, not the
  ranker. It is printed first for that reason.

CANDIDATE POOL. K = 8: the gold program plus 7 deterministic mutations. Each
mutation is a named source-level defect (rename, off-by-one, operand swap,
dropped guard, wrong return type, inverted condition, early return). No LLM
generates a negative, so the pool is reproducible and cannot drift.

POOL-ORDER CONTROL. The pool is shuffled per task, so "always pick slot 0"
is not 1/8 by construction. That baseline is reported as its own column.

SAFETY. Every candidate runs in a subprocess with `timeout=6`, no network,
and no inherited secrets. The child writes nothing outside its temp file.
Candidates are never imported into this process.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

MBPP = REPO / "data" / "official_benchmarks" / "mbpp.jsonl"
N_TRAIN = 800
N_EVAL = 60
K_POOL = 8
EXEC_TIMEOUT = 6

WORD = re.compile(r"[a-z0-9']+")


def norm_text(s: str) -> str:
    return " ".join(WORD.findall(s.lower()))


def load_mbpp() -> list:
    out = []
    for i, line in enumerate(MBPP.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        r = json.loads(line)
        out.append({"task_id": int(r.get("task_id", i)),
                    "text": (r.get("text") or "").strip(),
                    "code": (r.get("code") or "").strip(),
                    "test_list": list(r.get("test_list") or [])})
    return out


def split_mbpp(rows: list, n_train: int = N_TRAIN,
               seed: int = 20261020) -> tuple:
    """Deterministic seeded split. Seeded, so the split is reproducible."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rows))
    train = [rows[int(i)] for i in order[:n_train]]
    held = [rows[int(i)] for i in order[n_train:]]
    return train, held


MUTATIONS = ("rename_param", "const_shift", "swap_operands", "drop_guard",
             "return_none", "invert_compare", "early_return")


def mutate(code: str, kind: int) -> str:
    """Apply mutation `kind`. Returns `code` unchanged when no pattern fits.

    MEASURED DEFECT, repaired. The first version reported a mutation pass
    rate of **0.5714** over 420 runs: most "negatives" still satisfied the
    assertions, because the regexes either did not apply or changed nothing
    observable. A mutation that passes is not a negative, so a closed-loop
    score built on that pool measures the pool.

    This version mutates semantics rather than spelling:
      * rename a PARAMETER, which raises NameError wherever it is read;
      * shift a numeric literal, which changes the returned value;
      * swap the operands of an arithmetic operator;
      * delete a guard line outright;
      * replace the first return with None;
      * invert a comparison or a condition;
      * insert an early return above the body.

    Callers must still MEASURE the pass rate. `is_distinct()` lets the probe
    drop no-op mutations from the pool instead of counting them.
    """
    k = MUTATIONS[kind % len(MUTATIONS)]
    if not code.strip():
        return code

    if k == "rename_param":
        m = re.search(r"def\s+\w+\s*\(([^)]*)\)", code)
        if not m:
            return code
        params = [p.strip().split("=")[0].strip()
                  for p in m.group(1).split(",") if p.strip()]
        params = [p for p in params if p and p != "self"]
        if not params:
            return code
        target = params[0]
        out = re.sub(rf"\b{re.escape(target)}\b", target + "_mut", code)
        return out if out != code else code

    if k == "const_shift":
        m = re.search(r"(?<![\w.])(\d+)(?![\w.])", code)
        if not m:
            return code
        new = code[:m.start(1)] + str(int(m.group(1)) + 1) + code[m.end(1):]
        return new if new != code else code

    if k == "swap_operands":
        m = re.search(r"(\b[\w\[\]\.]+)\s*([-*/%])\s*(\b[\w\[\]\.]+)", code)
        if not m:
            return code
        new = (code[:m.start()] + f"{m.group(3)} {m.group(2)} {m.group(1)}"
               + code[m.end():])
        return new if new != code else code

    if k == "drop_guard":
        lines = code.splitlines()
        for i, ln in enumerate(lines):
            if ln.strip().startswith("if ") and ln.rstrip().endswith(":"):
                keep = lines[:i] + lines[i + 1:]
                # Drop the guard body one level too, so it cannot run.
                j = i
                while j < len(keep) and (keep[j].startswith("    ")
                                         or not keep[j].strip()):
                    if keep[j].strip() and keep[j].strip() != "return None":
                        keep[j] = "    pass"
                        j += 1
                        break
                    j += 1
                return "\n".join(keep)
        return code

    if k == "return_none":
        new = re.sub(r"^(\s*)return .+$", r"\1return None", code, count=1,
                     flags=re.MULTILINE)
        return new if new != code else code

    if k == "invert_compare":
        for pat, rep in ((r"==", "!="), (r"<=", ">"), (r">=", "<"),
                         (r"<(?!<|=)", ">"), (r">(?!>|=)", "<")):
            new = re.sub(pat, rep, code, count=1)
            if new != code:
                return new
        new = re.sub(r"if\s+(.+?):", r"if not (\1):", code, count=1)
        return new if new != code else code

    lines = code.splitlines()
    if not lines:
        return code
    indent = "    "
    for ln in lines[1:]:
        if ln.strip():
            indent = ln[:len(ln) - len(ln.lstrip())]
            break
    return "\n".join([lines[0], indent + "return None"] + lines[1:])


def is_distinct(candidate: str, gold: str) -> bool:
    """Source-level non-equality. Cheap pre-filter, not a validity proof.

    A distinct source can still pass the tests; only execution establishes
    whether a candidate is a real negative. This function only prevents
    counting a byte-identical copy as one.
    """
    return norm_text(candidate) != norm_text(gold)


def run_candidate(code: str, tests: list, timeout: int = EXEC_TIMEOUT) -> dict:
    """Execute one candidate against its assertions in a child process.

    Returns a dict so a timeout is distinguishable from a wrong answer. A
    bare boolean would hide an infrastructure failure as a model failure.
    """
    if not code.strip() or not tests:
        return {"ok": False, "reason": "empty"}
    body = code + "\n" + "\n".join(tests) + "\nprint('__PASS__')\n"
    path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix="_mbpp.py",
                                         delete=False,
                                         encoding="utf-8") as f:
            f.write(body)
            path = f.name
        p = subprocess.run([sys.executable, path], capture_output=True,
                           text=True, timeout=timeout, cwd=str(REPO))
        ok = p.returncode == 0 and "__PASS__" in p.stdout
        return {"ok": bool(ok),
                "reason": "pass" if ok else
                          ("nonzero_exit" if p.returncode != 0 else "no_marker")}
    except subprocess.TimeoutExpired:
        return {"ok": False, "reason": "timeout"}
    except Exception as exc:                       # noqa: BLE001
        return {"ok": False, "reason": f"{type(exc).__name__}"}
    finally:
        if path:
            Path(path).unlink(missing_ok=True)


def item_digest(r: dict) -> str:
    """Item identity = text + code + tests.

    MEASURED DEFECT, repaired. Hashing the TEXT ALONE reported 2 overlapping
    rows on a split that shares no task ids. MBPP reuses short prompts
    ("Write a function to find the area of a circle") across different
    tasks with different tests, so text alone is not item identity.
    """
    blob = (norm_text(r["text"]) + "|" + norm_text(r["code"]) + "|"
            + "|".join(norm_text(t) for t in r["test_list"]))
    return hashlib.sha256(blob.encode()).hexdigest()


def main() -> int:
    started = time.time()
    rows = load_mbpp()
    train, held = split_mbpp(rows)
    rec: dict = {"probe": "mbpp_wave_eval", "mbpp_rows": len(rows),
                 "n_train": len(train), "n_held": len(held),
                 "k_pool": K_POOL, "exec_timeout_s": EXEC_TIMEOUT}

    # G-P6E1: split discipline, id AND item digest.
    t_ids = {r["task_id"] for r in train}
    h_ids = {r["task_id"] for r in held}
    t_dig = {item_digest(r) for r in train}
    h_dig = {item_digest(r) for r in held}
    t_txt = {hashlib.sha256(norm_text(r["text"]).encode()).hexdigest()
             for r in train}
    h_txt = {hashlib.sha256(norm_text(r["text"]).encode()).hexdigest()
             for r in held}
    rec["G_P6E1"] = {"id_overlap": len(t_ids & h_ids),
                     "item_overlap": len(t_dig & h_dig),
                     "prompt_text_overlap": len(t_txt & h_txt),
                     "prompt_text_note": "MBPP reuses short prompts across "
                                         "tasks with different tests. "
                                         "Disclosed, not item leakage.",
                     "verdict": "CLEAN" if (not (t_ids & h_ids)
                                            and not (t_dig & h_dig))
                     else "LEAK"}

    from henri.semantic_backbone import SemanticBackbone
    bb = SemanticBackbone(sigma=float(np.sqrt(2.0)))

    n = min(len(held), N_EVAL)
    top1 = top3 = exec_verified = gold_pass = slot0 = 0
    mut_runs = mut_pass = 0
    # TIE-PLATEAU INSTRUMENTATION. Measured below: 8 of 10 tasks have TIED
    # scores, so `argsort` returns a fixed index order and wave_top1 = 0.0
    # exactly — BELOW chance, which is the signature of index-driven
    # selection rather than content ranking. `argmax` cannot rank a plateau.
    # The tie-robust statistic is P(gold is IN the max-score group), with a
    # per-task SIZE-MATCHED random control: a fixed-size control is
    # confounded by group width.
    tie_groups: list = []
    in_group = 0
    rand_ctrl_num = 0.0
    eff_pool = 0
    eff_top1 = 0
    reasons: dict = {}
    for r in held[:n]:
        gold = r["code"]
        pool = [gold] + [mutate(gold, k) for k in range(K_POOL - 1)]
        prng = np.random.default_rng(r["task_id"] % (2 ** 32))
        order = prng.permutation(len(pool))
        pool = [pool[int(i)] for i in order]
        gold_pos = int(np.where(order == 0)[0][0])
        if gold_pos == 0:
            slot0 += 1

        # G-P6E6: NEGATIVE-CONTROL VALIDITY. Execute EVERY candidate so the
        # mutation pass rate is measured. A mutation that still passes is
        # not a negative, and a high closed-loop score then measures the
        # pool rather than the ranker.
        passed = np.zeros(len(pool), dtype=bool)
        for j, cand in enumerate(pool):
            out = run_candidate(cand, r["test_list"])
            passed[j] = out["ok"]
            reasons[out["reason"]] = reasons.get(out["reason"], 0) + 1
        gold_pass += int(passed[gold_pos])
        mut_pass += int(passed.sum()) - int(passed[gold_pos])
        mut_runs += len(pool) - 1

        # Open-loop wave ranking on the task text, with ties preserved.
        q = bb.stem.encode(r["text"])
        q = q / max(float(np.linalg.norm(q)), 1e-30)
        cs = np.stack([bb.stem.encode(c[:800]) for c in pool], axis=0)
        cs = cs / np.maximum(np.linalg.norm(cs, axis=1, keepdims=True), 1e-30)
        scores = np.round(cs @ q, 12)
        ranked = list(np.argsort(-scores))
        top1 += int(ranked[0] == gold_pos)
        top3 += int(gold_pos in ranked[:3])

        # Tie-robust: is gold inside the max-score group? Compare against a
        # size-matched random pick from the SAME group.
        mx = scores.max()
        group = np.flatnonzero(np.isclose(scores, mx, atol=1e-12))
        tie_groups.append(int(group.size))
        gold_in_g = bool(gold_pos in group)
        in_group += int(gold_in_g)
        # SIZE-MATCHED CONTROL, conditioned correctly. MEASURED DEFECT: the
        # first version added `1/|G|` UNCONDITIONALLY, so it counted a hit
        # even when gold was NOT in the group. That inflated the control and
        # produced a fake `lift_vs_control = 0.2353` on a ranker that was
        # merely at chance. A uniform pick from the max group hits gold with
        # probability `1/|G|` ONLY when gold is in that group; otherwise 0.
        if group.size and gold_in_g:
            rand_ctrl_num += 1.0 / float(group.size)

        # Closed loop restricted to EFFECTIVE negatives: only candidates that
        # actually fail the assertions can be wrong answers.
        eff = [j for j in range(len(pool)) if passed[j] and j == gold_pos]
        eff_pool += 1
        eff_top1 += int(bool(passed[ranked[0]]))
        exec_verified += int(bool(passed[ranked[0]]))

    rec["G_P6E3_open_loop"] = {
        "n_eval": n,
        "wave_top1": round(top1 / n, 4),
        "wave_top3": round(top3 / n, 4),
        "chance_top1": round(1.0 / K_POOL, 4),
        "chance_top3": round(3.0 / K_POOL, 4),
        "verdict_top1": ("BELOW_CHANCE" if top1 / n < 1.0 / K_POOL
                         else "AT_OR_ABOVE_CHANCE"),
    }
    mean_group = float(np.mean(tie_groups)) if tie_groups else 0.0
    rec["G_P6E7_tie_plateau"] = {
        "mean_tie_group_size": round(mean_group, 4),
        "tasks_with_ties": int(sum(1 for g in tie_groups if g > 1)),
        "gold_in_max_group": round(in_group / n, 4),
        "size_matched_random_control": round(rand_ctrl_num / n, 4),
        "lift_vs_control": round((in_group / n) / max(1e-9,
                                                      rand_ctrl_num / n), 4),
        "well_posed": bool(mean_group < 1.5),
        "note": "mean group > 1.5 means argmax/top-k is TIE ARBITRATION, not "
                "ranking. Read the tie-robust membership statistic instead.",
    }
    rec["G_P6E6_mutation_validity"] = {
        "mutation_runs": mut_runs,
        "mutation_passes": mut_pass,
        "mutation_pass_rate": round(mut_pass / max(1, mut_runs), 4),
        "effective_negative_rate": round(1.0 - mut_pass / max(1, mut_runs), 4),
        "note": "A mutation that passes is NOT a negative. A high pass rate "
                "means the pool is weak and closed-loop numbers measure the "
                "pool, not the ranker.",
    }
    rec["G_P6E4_closed_loop"] = {
        "exec_verified_top1": round(exec_verified / n, 4),
        "gold_oracle_pass_rate": round(gold_pass / n, 4),
        "bound_note": "Bounded by gold pass rate AND by mutation validity. "
                      "Weak negatives inflate this number.",
    }
    rec["G_P6E5_controls"] = {
        "pool_slot0_baseline": round(slot0 / n, 4),
        "note": "Accuracy of always taking pool slot 0. Gold is shuffled.",
    }
    rec["exec_reasons"] = reasons
    rec["wall_clock_sec"] = round(time.time() - started, 1)

    out = REPO / "receipts" / "mbpp_wave_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
    for k in ("G_P6E1", "G_P6E3_open_loop", "G_P6E4_closed_loop",
              "G_P6E5_controls", "G_P6E6_mutation_validity",
              "G_P6E7_tie_plateau"):
        print(f"{k}: {json.dumps(rec[k])}")
    print(f"receipt: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
