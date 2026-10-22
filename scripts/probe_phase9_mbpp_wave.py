"""G-AST2: the real wave path, on a pool of verified-failing negatives.

PRE-REGISTERED BEFORE THE FIRST RUN.
------------------------------------
WHY THIS PROBE EXISTS
Phase 8 measured two defects that made the old MBPP number unreadable:
`scripts/mbpp_wave_eval.py` scores `bb.stem.encode` (384-d) and never
builds a wave; and 48.81 percent of its mutated "negatives" still PASS
`test_list`, so the random base rate of any candidate passing was 0.5521.
Ranking on that pool is ill-posed. `mbpp_wave_eval.py` stays UNMODIFIED as
the registered Phase 6 control; this file is additive.

MEASURED DEFECT IN MY FIRST DRAFT OF THIS FILE, repaired
It recorded which distractors passed but did not REJECT them. Smoke run on
two tasks gave validity 0.714 and 0.857, so the "filtered" arm was a subset
of a still-degenerate pool. A negative is now admitted only when the
sandbox reports `ok = False`. The generator is retried across mutation
kinds and then across tasks until 7 verified-failing negatives exist.

THE CENTRAL HONESTY CONSTRAINT
Filtering a distractor by running it is ANSWER KNOWLEDGE. A pool built that
way is a DISCRIMINATION benchmark, not a generation benchmark. It can show
whether a ranker separates right from wrong. It CANNOT show that HENRI can
produce a program from a prompt. Therefore this probe reports, always and
separately:

    generator quality   raw attempt statistics, unfiltered
    pool validity       1.0000 BY CONSTRUCTION after rejection, labelled
    ranking             measured only on the verified-failing pool

No generation claim is made anywhere in this file.

ARMS
  wave       OperatorAwareBackbone, `encode(...).psi`, D = 65536
  stem_384d  `bb.stem.encode(...)`, the Phase 6 control, 384-d

PRE-REGISTERED CLAUSES
  A2-1  pool validity == 1.0000 after rejection, and reported as
        constructed rather than discovered
  A2-2  raw generator validity reported (a property of MBPP, no bar)
  A2-3  wave open-loop top-1 beats 0.1250 with z >= +2.5
  A2-4  wave top-1 > stem_384d top-1
  A2-5  closed-loop exec rate reported against the pool's own base rate

SAMPLE SIZE, FIXED NOW
  N_TASKS = 40 held-out tasks, `EXEC_TIMEOUT` 6 s per candidate. 40 is the
  pre-registered subsample. It is not shrunk later to suit a clock. If the
  run is slow it is backgrounded, not cut.
"""
from __future__ import annotations

import ast
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "receipts" / "phase9_g_ast2.json"

N_TASKS = 40
K_POOL = 8
N_NEG = K_POOL - 1
SEED = 20261020
CHANCE = 1.0 / K_POOL
MAX_ATTEMPTS = 60


def load_rows():
    p = REPO / "data" / "official_benchmarks" / "mbpp.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def entry_name(tests):
    """Function name from the first parseable assert, by AST.

    Raises rather than defaulting. A guessed name or arity would score the
    wrong function and look like a ranking failure.
    """
    for t in tests:
        try:
            tree = ast.parse(t.strip())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                return node.func.id
    raise ValueError("no callable name in test_list")


def rename_def(source, name):
    """Rename the first FunctionDef. Returns None when it cannot."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == name:
                return source
            node.name = name
            try:
                return ast.unparse(tree)
            except Exception:
                return None
    return None


def build_pool(row, rows, mutate, run, n_tasks_others):
    """Gold plus N_NEG distractors, each VERIFIED to fail `test_list`.

    Returns (pool, info). `pool` always holds 8 entries or the caller drops
    the task. `info` separates generator quality from pool validity.
    """
    gold = row["code"]
    tests = list(row.get("test_list") or [])
    seen = {gold.strip()}
    attempts = []

    def attempt(text, origin):
        t = (text or "").strip()
        if not t or t in seen:
            return
        seen.add(t)
        res = run(text, tests)
        attempts.append({"text": text, "origin": origin,
                         "ok": bool(res.get("ok", False))})

    for kind in range(7):
        if len(attempts) >= MAX_ATTEMPTS:
            break
        attempt(mutate(gold, kind), "mutate_%d" % kind)

    try:
        name = entry_name(tests)
    except ValueError:
        name = None

    idx = 0
    while (sum(1 for a in attempts if not a["ok"]) < N_NEG
           and name is not None and idx < len(n_tasks_others)
           and len(attempts) < MAX_ATTEMPTS):
        other = rows[n_tasks_others[idx]]
        idx += 1
        if not other.get("code"):
            continue
        attempt(rename_def(other["code"], name), "cross_task")

    # Raw generator quality: the first N_NEG attempts, accepted blindly.
    # This is the pool the Phase 6 harness effectively built.
    raw = attempts[:N_NEG]
    raw_pass = sum(1 for a in raw if a["ok"])
    raw_base = (1.0 + raw_pass) / float(K_POOL)

    valid = [a for a in attempts if not a["ok"]][:N_NEG]
    pool = [{"text": gold, "origin": "gold", "ok": True}] + valid

    info = {
        "n_pool": len(pool),
        "n_neg": len(valid),
        "n_valid_neg": len(valid),
        # 1.0 BY CONSTRUCTION, not by luck. Labelled as such in the receipt.
        "validity": (len(valid) / float(N_NEG)) if valid else 0.0,
        "validity_is_constructed": True,
        "complete": len(pool) == K_POOL,
        "n_attempts": len(attempts),
        "n_attempts_passing": sum(1 for a in attempts if a["ok"]),
        # What the UNFILTERED generator would have produced.
        "raw_n_neg": len(raw),
        "raw_n_passing": raw_pass,
        "raw_validity": ((len(raw) - raw_pass) / float(len(raw)))
        if raw else 0.0,
        "raw_base_rate": raw_base,
        "origins": sorted({a["origin"] for a in valid}),
    }
    return pool, info


def rank_pool(vectors, gold_index):
    """Cosine rank of every candidate against the docstring wave."""
    q = vectors["query"]
    scores = [float(np.real(np.vdot(q, v))) for v in vectors["cand"]]
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    best = scores[order[0]]
    tie = sum(1 for s in scores if abs(s - best) <= 1e-9)
    return {"order": order, "scores": scores, "tie_group": tie,
            "top1_is_gold": order[0] == gold_index}


def main() -> int:
    t0 = time.time()
    from henri.code_verify import load_mbpp_executor
    from henri.operator_ingress import OperatorAwareBackbone

    mod = load_mbpp_executor()
    if mod is None:
        print("EXECUTOR_UNAVAILABLE: infrastructure limit, not a result")
        return 2
    run, mutate = mod.run_candidate, mod.mutate

    rows = load_rows()
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(rows))
    train_idx = [int(i) for i in order[:800]]
    held_idx = [int(i) for i in order[800:]]
    overlap = set(train_idx) & set(held_idx)
    print("corpus %d  train %d  held %d  overlap %d  N_TASKS %d"
          % (len(rows), len(train_idx), len(held_idx), len(overlap), N_TASKS))

    print("building wave encoder D = 65536")
    oa = OperatorAwareBackbone(dim=65536)
    from henri.api_server import get_api
    stem = get_api().backbone.stem
    print("encoders ready %.1f s" % (time.time() - t0))

    items, dropped = [], 0
    for n, i in enumerate(held_idx[:N_TASKS]):
        row = rows[i]
        pool, info = build_pool(row, rows, mutate, run, train_idx)
        if not info["complete"]:
            dropped += 1
            print("  skip task %s: %d/%d negatives"
                  % (row.get("task_id"), info["n_neg"], N_NEG))
            continue

        prompt = (row.get("text") or "").strip()
        wq = np.asarray(oa.encode(prompt).psi)
        wc = [np.asarray(oa.encode(c["text"]).psi) for c in pool]
        sq = np.asarray(stem.encode(prompt))
        sc = [np.asarray(stem.encode(c["text"])) for c in pool]

        def unit(v):
            nv = float(np.linalg.norm(v))
            return v / nv if nv > 1e-30 else v

        items.append({
            "task_id": row.get("task_id"), "info": info,
            "wave": {"query": unit(wq), "cand": [unit(v) for v in wc]},
            "stem": {"query": unit(sq), "cand": [unit(v) for v in sc]},
        })
        if (n + 1) % 10 == 0:
            print("  encoded %d  %.1f s" % (n + 1, time.time() - t0))

    n = len(items)
    print("\nitems %d  dropped %d  %.1f s" % (n, dropped, time.time() - t0))
    if n == 0:
        print("NO_ITEMS: nothing measurable")
        return 2

    def agg(key):
        rs = [rank_pool(it[key], 0) for it in items]
        top1 = sum(1 for r in rs if r["top1_is_gold"])
        return {"n": len(rs), "top1_correct": top1,
                "top1_acc": top1 / float(len(rs)),
                "mean_tie_group": round(float(np.mean(
                    [r["tie_group"] for r in rs])), 4)}

    wave = agg("wave")
    stem_arm = agg("stem")

    raw_validity = float(np.mean([it["info"]["raw_validity"] for it in items]))
    raw_base = float(np.mean([it["info"]["raw_base_rate"] for it in items]))
    pool_validity = float(np.mean([it["info"]["validity"] for it in items]))
    n_attempts = float(np.mean([it["info"]["n_attempts"] for it in items]))
    frac_passing_attempt = float(np.mean([
        it["info"]["n_attempts_passing"] / max(1, it["info"]["n_attempts"])
        for it in items]))

    def z_of(p, nn):
        return (p - CHANCE) / float(np.sqrt(CHANCE * (1 - CHANCE)
                                           / max(1, nn)))

    clauses = [
        {"id": "A2-1", "what": "pool validity after rejection == 1.0000",
         "value": round(pool_validity, 6), "bar": "== 1.0",
         "pass": abs(pool_validity - 1.0) <= 1e-12},
        {"id": "A2-2", "what": "raw generator validity (MBPP property)",
         "value": round(raw_validity, 6), "bar": "reported, no bar",
         "raw_pass_rate_of_attempts": round(frac_passing_attempt, 6),
         "pass": True},
        {"id": "A2-3", "what": "wave top-1 > 0.1250 and z >= +2.5",
         "value": round(wave["top1_acc"], 6),
         "z": round(z_of(wave["top1_acc"], wave["n"]), 4),
         "bar": "> 0.1250, z >= +2.5",
         "pass": bool(wave["top1_acc"] > CHANCE
                      and z_of(wave["top1_acc"], wave["n"]) >= 2.5)},
        {"id": "A2-4", "what": "wave top-1 > stem_384d top-1",
         "value": round(wave["top1_acc"], 6),
         "control": round(stem_arm["top1_acc"], 6), "bar": "wave > stem",
         "pass": bool(wave["top1_acc"] > stem_arm["top1_acc"])},
        {"id": "A2-5", "what": "closed-loop rate vs the pool's own base rate",
         "value": round(wave["top1_acc"], 6),
         "pool_base_rate_filtered": round(CHANCE, 6),
         "raw_pool_base_rate": round(raw_base, 6),
         "bar": "reported, no bar", "pass": True},
    ]

    verdict = ("G_AST2_PASS" if all(c["pass"] for c in clauses)
               else "G_AST2_PARTIAL_%d_OF_%d"
               % (sum(1 for c in clauses if c["pass"]), len(clauses)))

    body = {
        "probe": "phase9_g_ast2",
        "n_tasks_target": N_TASKS, "n_items": n, "dropped": dropped,
        "k_pool": K_POOL, "chance": round(CHANCE, 6),
        "split_seed": SEED, "train_held_overlap": len(overlap),
        "dim_wave": 65536, "dim_stem": 384,
        "wave": wave, "stem_384d": stem_arm,
        "pool_validity_constructed": round(pool_validity, 6),
        "raw_generator_validity": round(raw_validity, 6),
        "raw_pool_base_rate": round(raw_base, 6),
        "mean_attempts_per_pool": round(n_attempts, 2),
        "frac_attempts_passing": round(frac_passing_attempt, 6),
        "clauses": clauses, "verdict": verdict,
        "labelling": ("DISCRIMINATION benchmark. Distractors are rejected by "
                      "running them, so the pool carries answer knowledge. It "
                      "cannot support a generation claim. The raw column "
                      "reports the unfiltered generator."),
        "wall_clock_s": round(time.time() - t0, 2),
    }

    ser = json.dumps(body, indent=2, sort_keys=True, default=float)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(ser.encode("utf-8"))
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    body["receipt_sha256"] = digest
    OUT.write_bytes(json.dumps(body, indent=2, sort_keys=True,
                               default=float).encode("utf-8"))

    for c in clauses:
        extra = ""
        if "z" in c:
            extra += " z=%+0.3f" % c["z"]
        if "control" in c:
            extra += " ctrl=%.4f" % c["control"]
        print("%-6s %-46s %-10s %s%s"
              % (c["id"], c["what"], c["value"],
                 "PASS" if c["pass"] else "FAIL", extra))
    print("\nwave  %s" % wave)
    print("stem  %s" % stem_arm)
    print("pool validity %.6f (constructed)  raw generator %.6f  raw base %.6f"
          % (pool_validity, raw_validity, raw_base))
    print("mean attempts/pool %.2f  frac attempts passing %.4f"
          % (n_attempts, frac_passing_attempt))
    print("receipt %s sha256 %s" % (OUT.name, digest[:16]))
    print("VERDICT %s" % verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
