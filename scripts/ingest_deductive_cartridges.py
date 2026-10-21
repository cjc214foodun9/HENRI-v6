"""Action C: deductive Q->A operator and cartridge ingestion, leakage-checked.

PRE-REGISTERED BEFORE ANY FIT
-----------------------------
Written before `W_deduce` was fitted, so the bar cannot move to meet the
result. Recorded in `docs/EVIDENCE_REGISTER.md` before the run.

G-P5C1  SPLIT PROVENANCE. The held-out set must be exactly the rows that
        `wave_rank.load_items` reads: `items[:400]` in file order. The
        ingestion bank is every other row. Overlap must be ZERO by record
        id AND by normalized-text digest.
G-P5C2  NEAR-DUPLICATE AUDIT. Text digests only catch exact repeats. Report
        the count of paraphrase pairs with token-set Jaccard > 0.8 across
        the split. This is a disclosure, not a gate: MMLU reuses concepts.
G-P5C3  W_deduce must beat a NO-TRANSITION control on held-out Q->A
        retrieval, AND a SHUFFLED-PAIR control. Beating only the first
        means it memorized topic, not deduction.
G-P5C4  Report the ridge fit R^2. A value near zero is the expected result
        and must not be narrated as a deductive operator.
G-P5C5  Cartridge recall is RETRIEVAL, not deduction. The recall path
        returns the answer to the nearest PREMISE, which is a different
        question when the bank is disjoint from the query. Report it as
        retrieval and let the harness decide.

FOOTPRINT. One cartridge at `D = 65536`, `rank = 64`, complex128 is
`2 * 65536 * 64 * 16 = 134,217,728 B = 128 MiB`. The 13,642-row ingestion
bank needs 214 cartridges, which is **27.4 GiB** at complex128 and
**13.7 GiB** at complex64, before anything else. The probe therefore runs
at a bounded scale and reports the projection rather than allocating it.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from henri.cartridge_bank import cartridge_bytes          # noqa: E402
from henri.semantic_backbone import (SemanticBackbone,    # noqa: E402
                                     tokens)

CANON = REPO / "data" / "official_benchmarks" / "canonical"
EVAL_N = 400
LETTERS = ["A", "B", "C", "D"]


def read_mmlu() -> list:
    """Read MMLU with a UNIQUE row key.

    MEASURED DEFECT, repaired. The CSV carries a `''` header column, but it
    holds only 1534 distinct values across 14042 rows: it restarts per
    subject (`0` appears 57 times). Using it as a record id made the
    leakage check report 152 overlapping ids on a split that shares no
    rows. The unique key is the FILE INDEX, which is also what
    `wave_rank.load_items` iterates over.
    """
    import csv
    rows = list(csv.DictReader(open(CANON / "mmlu" / "mmlu.csv",
                                   encoding="utf-8")))
    items = []
    for idx, r in enumerate(rows):
        q = (r.get("Question") or "").strip()
        opts = [(r.get(k) or "").strip() for k in LETTERS]
        letter = (r.get("Answer") or "").strip().upper()
        if letter not in LETTERS or not q or any(not o for o in opts):
            continue
        items.append({
            "record_id": f"mmlu:{idx}",
            "csv_index": r.get("", ""),
            "question": q,
            "options": opts,
            "answer_idx": LETTERS.index(letter),
            "gold": opts[LETTERS.index(letter)],
            "subject": r.get("Subject", ""),
        })
    return items


def norm_text(s: str) -> str:
    return " ".join(tokens(s))


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def fit_w(u_q: np.ndarray, u_a: np.ndarray, lam: float = 1.0) -> dict:
    """Closed-form ridge: minimise ||W u_q - (u_a - u_q)|| + lam ||W||^2.

    Args:
        u_q: `(n, d)` premise embeddings, unit rows.
        u_a: `(n, d)` consequent embeddings, unit rows.
        lam: ridge penalty. Fixed before the run, not tuned on held-out.

    DERIVATION. Stack rows: `U_q W^T = Delta`, with `Delta = U_a - U_q`.
    Gradient of `||U_q W^T - Delta||_F^2 + lam ||W||_F^2` gives

        (U_q^T U_q + lam I) W^T = U_q^T Delta

    which is a `(d, d)` system. The first version wrote `U_q U_q^T`, an
    `(n, n)` matrix, and failed on `n=512, d=384`. The shape error is the
    reason transposes matter here; both systems are valid ridge but the
    primal one is `(d, d)` and is what this fit reports.
    """
    d = int(u_q.shape[1])
    delta = u_a - u_q                              # (n, d)
    gram = u_q.T @ u_q                             # (d, d)
    rhs = u_q.T @ delta                            # (d, d)
    w = np.linalg.solve(gram + lam * np.eye(d), rhs).T   # (d, d)
    pred = u_q + u_q @ w.T                         # (n, d)
    pred /= np.maximum(np.linalg.norm(pred, axis=1, keepdims=True), 1e-30)
    resid = float(np.linalg.norm(pred - u_a) ** 2)
    total = float(np.linalg.norm(u_a - u_a.mean(axis=0)) ** 2)
    r2 = 1.0 - resid / max(total, 1e-30)
    return {"W": w, "r2": r2, "pred": pred, "lam": lam}


def retrieval_acc(pred: np.ndarray, gold: np.ndarray) -> float:
    """Top-1 accuracy of matching predicted consequent to the gold set."""
    sims = pred @ gold.T
    return float(np.mean(np.argmax(sims, axis=1) == np.arange(len(gold))))


def main() -> int:
    started = time.perf_counter()
    rec: dict = {"probe": "ingest_deductive_cartridges",
                 "status": "EVALUATED"}
    dim = int(sys.argv[1]) if len(sys.argv) > 1 else 384
    max_rows = int(sys.argv[2]) if len(sys.argv) > 2 else 512

    items = read_mmlu()
    rec["mmlu_rows"] = len(items)
    # EXACT split reproduction: load_items reads items[:limit] in file order.
    held = items[:EVAL_N]
    bank_rows = items[EVAL_N:]
    rec["held_out"] = EVAL_N
    rec["ingestion_bank"] = len(bank_rows)

    # G-P5C1: zero overlap by id AND by ITEM digest.
    #
    # MEASURED DEFECT, repaired. Hashing the QUESTION ALONE reported one
    # overlap on a split that shares no rows. The cause: the stem "Which of
    # the following statements is true?" appears in many MMLU rows across
    # different subjects, and only the OPTIONS distinguish them. Item
    # identity is (question, options, answer) together, so that is what the
    # digest covers. The bare-stem count is reported separately, because a
    # reader should see it rather than have it silently excluded.
    held_ids = {r["record_id"] for r in held}
    bank_ids = {r["record_id"] for r in bank_rows}
    rec["id_overlap"] = len(held_ids & bank_ids)

    def item_digest(r: dict) -> str:
        blob = norm_text(r["question"]) + "|" + "|".join(
            norm_text(o) for o in r["options"]) + "|" + str(r["answer_idx"])
        return hashlib.sha256(blob.encode()).hexdigest()

    held_txt = {item_digest(r) for r in held}
    bank_txt = {item_digest(r) for r in bank_rows}
    rec["item_overlap"] = len(held_txt & bank_txt)

    stem_held = {hashlib.sha256(norm_text(r["question"]).encode()).hexdigest()
                 for r in held}
    stem_bank = {hashlib.sha256(norm_text(r["question"]).encode()).hexdigest()
                 for r in bank_rows}
    rec["question_stem_overlap"] = len(stem_held & stem_bank)
    rec["question_stem_note"] = (
        "Generic stems recur across MMLU subjects. Not item leakage, since "
        "the options differ. Disclosed, not asserted away.")

    rec["leakage_verdict"] = ("CLEAN" if rec["id_overlap"] == 0
                              and rec["item_overlap"] == 0 else "LEAK")

    # G-P5C2: paraphrase audit. Disclosure, not a gate.
    bank_sets = [set(norm_text(r["question"])) for r in bank_rows[:400]]
    near = 0
    for r in held[:120]:
        h = set(norm_text(r["question"]))
        for b in bank_sets:
            if jaccard(h, b) > 0.8:
                near += 1
                break
    rec["near_duplicate_hits"] = near
    rec["near_duplicate_probe"] = "held[:120] vs bank[:400], Jaccard > 0.8"

    # -- embeddings -----------------------------------------------------
    bb = SemanticBackbone()
    tr = bank_rows[:max_rows]
    t0 = time.perf_counter()
    u_q = np.stack([bb.stem.encode(r["question"]) for r in tr])
    u_a = np.stack([bb.stem.encode(r["gold"]) for r in tr])
    rec["embed_sec"] = round(time.perf_counter() - t0, 2)
    rec["dim_sem"] = int(u_q.shape[1])

    he_q = np.stack([bb.stem.encode(r["question"]) for r in held])
    he_a = np.stack([bb.stem.encode(r["gold"]) for r in held])

    # G-P5C3/C4: fit, then two controls that can fail.
    fit = fit_w(u_q, u_a)
    rec["ridge"] = {"lam": fit["lam"], "r2": round(fit["r2"], 6)}

    def eval_transition(w: Optional[np.ndarray]) -> dict:
        if w is None:
            pred = he_q.copy()
        else:
            pred = he_q + he_q @ w.T
            pred /= np.maximum(np.linalg.norm(pred, axis=1, keepdims=True),
                               1e-30)
        # Retrieval over the HELD-OUT gold set: does the transition put the
        # query nearest its OWN answer more often than the identity does?
        sims = pred @ he_a.T
        own = np.diag(sims)
        better = float(np.mean(sims.argmax(axis=1) == np.arange(len(he_a))))
        return {"retrieval_top1": round(better, 4),
                "mean_own_sim": round(float(own.mean()), 6)}

    rec["arm_w_deduce"] = eval_transition(fit["W"])
    rec["control_no_transition"] = eval_transition(None)

    rng = np.random.default_rng(20261020)
    perm = rng.permutation(len(u_q))
    fit_shuf = fit_w(u_q, u_a[perm])
    rec["control_shuffled_pairs"] = eval_transition(fit_shuf["W"])
    rec["shuffled_ridge_r2"] = round(fit_shuf["r2"], 6)

    d = rec["arm_w_deduce"]["retrieval_top1"]
    n = rec["control_no_transition"]["retrieval_top1"]
    s = rec["control_shuffled_pairs"]["retrieval_top1"]
    rec["verdict"] = ("PASS_TRANSITION_BEATS_BOTH" if d > n and d > s
                      else "FALSIFIED_DOES_NOT_BEAT_CONTROLS")

    # -- G-P5C5: cartridge recall is retrieval --------------------------
    try:
        from henri.cartridge_bank import build_bank
    except Exception as e:                                    # pragma: no cover
        rec["cartridge_error"] = str(e)
    else:
        D = 4096
        waves_q = np.stack([bb.encode(r["question"]).psi[:D] for r in tr[:64]])
        waves_a = np.stack([bb.encode(r["gold"]).psi[:D] for r in tr[:64]])
        bk = build_bank(waves_q, waves_a, dim=D, rank=64)
        res = bk.query(waves_q[7], k_route=1)
        rec["cartridge_probe"] = {
            "rows": 64, "dim_probe": D,
            "exact_recall_error": round(float(
                np.linalg.norm(res.value - bk.cartridges[res.cartridge]
                               .values[res.row])), 9),
            "router_hit": int(res.cartridge == 0),
            "note": "RETRIEVAL over nearest premise, not deduction",
        }
        rec["footprint"] = {
            "bytes_per_cartridge_c128": cartridge_bytes(65536, 64,
                                                        np.complex128),
            "cartridges_for_bank": int(np.ceil(len(bank_rows) / 64)),
            "gib_c128": round(cartridge_bytes(65536, 64, np.complex128)
                              * np.ceil(len(bank_rows) / 64) / 2**30, 2),
            "gib_c64": round(cartridge_bytes(65536, 64, np.complex64)
                             * np.ceil(len(bank_rows) / 64) / 2**30, 2),
        }

    rec["wall_clock_sec"] = round(time.perf_counter() - started, 2)
    out = REPO / "receipts" / "ingest_deductive.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
    for k in ("leakage_verdict", "id_overlap", "item_overlap",
              "question_stem_overlap", "near_duplicate_hits", "ridge",
              "arm_w_deduce", "control_no_transition",
              "control_shuffled_pairs", "shuffled_ridge_r2", "verdict",
              "cartridge_probe", "footprint"):
        if k in rec:
            print(f"{k}: {rec[k]}")
    print(f"receipt: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
