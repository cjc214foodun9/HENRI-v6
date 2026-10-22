"""G-KOOP1: does the Koopman + cartridge arm beat the static control?

PRE-REGISTERED BEFORE THE FIRST RUN.
------------------------------------
Split      seeded 800/174, as `scripts/mbpp_wave_eval.py` uses. Fit rows
           and held-out rows are disjoint. Fit is capped for runtime.
Weights    grid `W_GRID x W_MEM_GRID`, `0.0` in both, so the static arm is a
           member of the search space.
Selection  best FIT accuracy; ties prefer the SMALLER total weight. Without
           that tie-break a positive weight that adds nothing could be
           installed and reported as a win.
Encoders   TWO arms, reported separately:
             `strip`  SemanticBackbone only  (the Phase 7 control)
             `opast`  OperatorAwareEncoder   (ACTION 1's channel)
           MEASURED WHY: on `strip`, `a + b` and `a - b` both scored
           `s_static = 0.69013433` to eight decimals in the smoke run, so no
           transition weight can separate them. That is the Phase 7 tie,
           still present wherever the operator channel is absent.

G-KOOP1 VERDICT RULE
  PASS  held-out top-1 at the fitted weights is STRICTLY greater than at
        `w_trans = w_mem = 0`, and the paired McNemar test does not
        contradict it (`p < 0.05` on the arm-better side).
  FAIL  otherwise. On FAIL the installed weights stay `0.0` for that arm
        and the coupling is not claimed. A transition that does not beat
        doing nothing is not installed.

COMPONENT CACHING, AND WHY IT IS EXACT
`rank()` recomputes each term per weight pair. The transition term depends
only on (spec, domain) and the static term only on (spec, candidate), so the
components are computed ONCE per (item, candidate) and the grid is
arithmetic over the cache. This is the same score `rank()` produces, not an
approximation of it.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np

from henri.config import WaveConfig
from henri.koopman_router import (
    W_GRID,
    W_MEM_GRID,
    CartridgeRecall,
    domain_of,
    mcnemar,
)

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "receipts" / "phase8_g_koop1.json"

N_FIT = 80
N_HELD = 40
K_POOL = 8
SEED = 20261020
TAU_VETO = 0.0431


def load_mbpp() -> list:
    rows = []
    path = REPO / "data" / "official_benchmarks" / "mbpp.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rows.append({"task_id": int(r.get("task_id", len(rows))),
                     "text": (r.get("text") or "").strip(),
                     "code": (r.get("code") or "").strip(),
                     "test_list": list(r.get("test_list") or [])})
    return rows


def split_rows(rows: list, n_train: int = 800, seed: int = SEED):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rows))
    return ([rows[int(i)] for i in order[:n_train]],
            [rows[int(i)] for i in order[n_train:]])


def build_pools(rows: list, mutate) -> list:
    """K=8 pool per task: gold plus K-1 mutations of the gold source.

    `mutate(code, kind)` is the repo's own mutator. A mutation that returns
    the source unchanged is NOT a negative, so it is recorded rather than
    silently counted. That defect was measured in Phase 6 (validity 0.5714
    on the first run) and must not regress here.
    """
    pools = []
    for r in rows:
        gold = r["code"]
        cands = [gold]
        for k in range(K_POOL - 1):
            m = mutate(gold, k)
            cands.append(m if m and m != gold else gold + "\n# mutant_noop")
        pools.append({"item": r, "candidates": cands, "gold_index": 0,
                      "mutation_validity": sum(
                          1 for c in cands[1:]
                          if c and c != gold) / float(K_POOL - 1)})
    return pools


def components_for(router, pool: dict, encoder_name: str) -> dict:
    """Every score term for one item, cached across the weight grid."""
    item = pool["item"]
    spec = router._psi(item["text"])
    domain = domain_of(item.get("subject", ""), router.domains.n_domains)
    moved = router.domains.step(spec, domain).psi
    moved = moved / max(float(np.linalg.norm(moved)), 1e-30)

    hit = None
    if router.recall_bank is not None and router.recall_bank.rows:
        hit = router.recall_bank.recall(spec)

    rows = []
    for i, text in enumerate(pool["candidates"]):
        psi = router._psi(text)
        v = psi.psi / max(float(np.linalg.norm(psi.psi)), 1e-30)
        s_static = float(np.real(np.vdot(spec.psi, v)))
        s_trans = float(np.real(np.vdot(moved, v)))
        s_mem = 0.0
        if hit is not None and hit.hit:
            s_mem = router.recall_bank.coherence(hit.value, psi)
        rows.append({"index": i, "s_static": s_static,
                     "s_trans": s_trans, "s_mem": s_mem,
                     "hit": bool(hit.hit) if hit is not None else False})
    return {"domain": domain, "rows": rows, "gold_index": pool["gold_index"],
            "encoder": encoder_name,
            "recall_hit": bool(hit.hit) if hit is not None else False}


def evaluate(cached: list, w_trans: float, w_mem: float) -> dict:
    hits, ties = [], []
    for c in cached:
        scored = [(r["s_static"] + w_trans * r["s_trans"]
                   + w_mem * r["s_mem"], r["index"]) for r in c["rows"]]
        scored.sort(key=lambda t: (-t[0], t[1]))
        best = scored[0][0]
        tie = sum(1 for s, _ in scored if abs(s - best) <= 1e-12)
        ties.append(tie)
        hits.append(int(scored[0][1]) == int(c["gold_index"]))
    n = len(hits)
    return {"n": n, "correct": int(sum(hits)),
            "acc": (sum(hits) / n) if n else 0.0,
            "hits": hits,
            "mean_tie_group": float(np.mean(ties)) if ties else 0.0}


def fit(cached: list) -> dict:
    table = []
    best = None
    for w in W_GRID:
        for wm in W_MEM_GRID:
            r = evaluate(cached, w, wm)
            row = {"w_trans": w, "w_mem": wm,
                   "fit_acc": round(r["acc"], 6),
                   "fit_correct": r["correct"],
                   "mean_tie_group": round(r["mean_tie_group"], 4)}
            table.append(row)
            key = (-r["acc"], w + wm, w, wm)
            if best is None or key < best[0]:
                best = (key, row)
    return {"chosen": best[1], "grid": table}


def run_arm(name, backbone, cfg, trains, helds, encoder, mutate) -> dict:
    from henri.koopman_router import KoopmanRouter

    recall = CartridgeRecall(dim=cfg.dim, rank=32, max_rows=N_FIT)
    for r in trains:
        recall.add(backbone.encode(r["text"]).psi,
                   backbone.encode(r["code"]).psi)
    rows_written = recall.finalize()

    router = KoopmanRouter(backbone, cfg, recall=recall,
                           n_domains=4, seed=7, encoder=encoder)

    fit_pools = build_pools(trains[:N_FIT], mutate)
    held_pools = build_pools(helds[:N_HELD], mutate)
    fit_cached = [components_for(router, p, name) for p in fit_pools]
    held_cached = [components_for(router, p, name) for p in held_pools]

    f = fit(fit_cached)
    ch = f["chosen"]
    arm = evaluate(held_cached, ch["w_trans"], ch["w_mem"])
    ctrl = evaluate(held_cached, 0.0, 0.0)
    mc = mcnemar(arm["hits"], ctrl["hits"])

    passed = bool(arm["acc"] > ctrl["acc"] and mc["verdict"] == "ARM_BETTER")
    installed = ({"w_trans": ch["w_trans"], "w_mem": ch["w_mem"]}
                 if passed else {"w_trans": 0.0, "w_mem": 0.0})

    return {
        "arm": name,
        "recall_rows_written": rows_written,
        "fit_n": len(fit_cached),
        "held_n": len(held_cached),
        "fit_chosen": ch,
        "held_arm": {k: arm[k] for k in ("n", "correct", "acc",
                                         "mean_tie_group")},
        "held_control_w0": {k: ctrl[k] for k in ("n", "correct", "acc",
                                                 "mean_tie_group")},
        "delta_acc": round(arm["acc"] - ctrl["acc"], 6),
        "mcnemar": mc,
        "mutation_validity_fit": round(float(np.mean(
            [p["mutation_validity"] for p in fit_pools])), 4),
        "mutation_validity_held": round(float(np.mean(
            [p["mutation_validity"] for p in held_pools])), 4),
        "recall_hit_rate_fit": round(float(np.mean(
            [1.0 if c["recall_hit"] else 0.0 for c in fit_cached])), 4),
        "top1_vs_chance": {
            "p0": round(1.0 / K_POOL, 4),
            "z": round((arm["acc"] - 1.0 / K_POOL)
                       / float(np.sqrt((1.0 / K_POOL)
                                       * (1 - 1.0 / K_POOL)
                                       / max(1, arm["n"]))), 4),
        },
        "verdict": "G_KOOP1_PASS" if passed else "G_KOOP1_FAIL",
        "installed_weights": installed,
        "grid": f["grid"],
    }


def main() -> int:
    t0 = time.time()
    from henri.api_server import get_api
    from henri.operator_ingress import OperatorAwareBackbone
    from scripts.mbpp_wave_eval import mutate

    backbone = get_api().backbone
    probe = backbone.encode("probe").psi
    dim = int(np.asarray(probe).shape[0])
    cfg = WaveConfig(dim=dim)
    print("wave width %d  r_koopman %d" % (dim, cfg.r_koopman))

    rows = load_mbpp()
    trains, helds = split_rows(rows)
    print("corpus %d  train %d  held %d  use fit %d / held %d"
          % (len(rows), len(trains), len(helds), N_FIT, N_HELD))
    overlap = {r["task_id"] for r in trains} & {r["task_id"] for r in helds}
    print("fit/held task_id overlap %d" % len(overlap))

    # MEASURED, not guessed. `dir(henri.operator_ingress)` exposes
    # `OperatorAwareBackbone` and `OperatorEncoded`; there is no
    # `OperatorAwareEncoder` and no `OperatorIngress`. The ctor is
    # `(base=None, weights=(0.4,0.1,0.3,0.2), dim=65536)` and `encode`
    # returns an `OperatorEncoded` DATACLASS whose fields are
    # `psi (65536,) complex128`, `slot_operator (16384,)`,
    # `slot_structure (16384,)`, `pol`, `parsed`, `operator_stream`,
    # `active_slots`.
    #
    # Two defects hid here, not one:
    #   1. wrong class name -> ImportError at import time
    #   2. passing the dataclass to the router -> `np.asarray(dataclass)`
    #      gives a 0-d object array, so `_psi` would raise
    #      `ValueError: encoder emits width 0`
    # The adapter extracts `.psi` so the encoder emits a real vector.
    oa = OperatorAwareBackbone(dim=dim)
    opast = lambda text: oa.encode(text).psi                   # noqa: E731
    arms = []

    for name, enc in (("strip", None), ("opast", opast)):
        print("-- arm %s" % name)
        a = run_arm(name, backbone, cfg, trains, helds, enc, mutate)
        arms.append(a)
        print("   fit %s  held %.4f  ctrl %.4f  d %+0.4f  %s  z %.2f  %s"
              % (a["fit_chosen"], a["held_arm"]["acc"],
                 a["held_control_w0"]["acc"], a["delta_acc"],
                 a["mcnemar"]["verdict"], a["top1_vs_chance"]["z"],
                 a["verdict"]))

    body = {
        "probe": "phase8_g_koop1",
        "dim": dim,
        "tau_veto": TAU_VETO,
        "k_pool": K_POOL,
        "fit_n_target": N_FIT,
        "held_n_target": N_HELD,
        "split_seed": SEED,
        "fit_held_task_overlap": len(overlap),
        "w_grid": list(W_GRID),
        "w_mem_grid": list(W_MEM_GRID),
        "arms": arms,
        "wall_clock_s": round(time.time() - t0, 2),
        "policy": ("weights chosen on FIT only; ties prefer the smaller "
                   "weight so w=0 wins every tie; on FAIL the installed "
                   "weights stay 0.0 and the coupling is not claimed"),
    }
    ser = json.dumps(body, indent=2, sort_keys=True, default=float)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(ser.encode("utf-8"))
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    body["receipt_sha256"] = digest
    OUT.write_bytes(json.dumps(body, indent=2, sort_keys=True,
                               default=float).encode("utf-8"))

    back = json.loads(OUT.read_text(encoding="utf-8"))
    ok = back["arms"][0]["held_arm"]["acc"] == body["arms"][0]["held_arm"]["acc"]
    print("receipt %s sha256 %s readback %s"
          % (OUT, digest[:16], "OK" if ok else "MISMATCH"))
    for a in arms:
        print("VERDICT %-6s %s  installed %s"
              % (a["arm"], a["verdict"], a["installed_weights"]))
    print("wall %.1f s" % body["wall_clock_s"])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
