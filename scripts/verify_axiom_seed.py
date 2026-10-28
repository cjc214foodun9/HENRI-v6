#!/usr/bin/env python3
"""Zone C seeding gate. Every counted clause states how it can FAIL.

Framing: this measures a SEEDED AXIOM BANK with retrieval. It is not world
knowledge and not a benchmark.

  SEED-1  bank builds and keeps Stiefel coherence: c_max < 1e-6
  SEED-2  in-bank recall: query (subject, relation), recover the stored object
  SEED-3  novelty separation: unseen (subject, relation) pairs score a LOWER
          margin than known pairs by a stated gap. This is the clause a
          collapse-to-bank defect would fail.
  SEED-4  kill control: a random query must score a margin below 0.10
"""
from __future__ import annotations

import sys
import time

import numpy as np

from henri.axiom_seed import SeededBank, build_bank, default_axioms, novel_pairs, query_pair

RESULTS = []


def record(gate: str, ok: bool, detail: str) -> None:
    RESULTS.append((gate, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {gate}: {detail}")


def main() -> int:
    t0 = time.time()
    seed = 20261026
    np.random.seed(seed)
    print("=== ZONE C SEEDING (MiniLM -> phase lift -> HHRG ledger, CPU) ===")

    axioms = default_axioms()
    bank = build_bank(d_model=16384, seed=42, axioms=axioms)
    cmax = bank.ledger.max_coherence()

    record(
        "SEED-1 bank + coherence",
        bank.ledger.size == len(axioms) and cmax < 1e-6,
        f"engrams={bank.ledger.size}/{len(axioms)}, c_max={cmax:.2e} (bar 1e-6)",
    )

    hits = 0
    margins_known = []
    for subj, rel, obj in axioms:
        ret = query_pair(bank, subj, rel)
        labels = bank.ledger.memory.charts[ret.chart_id].labels
        got = labels[ret.local_index]
        if got.split("|")[2] == obj:
            hits += 1
        margins_known.append(ret.margin)
    recall = hits / len(axioms)
    record(
        "SEED-2 in-bank recall",
        recall >= 0.90,
        f"top1_object={hits}/{len(axioms)} = {recall:.3f} (bar 0.90), "
        f"mean_margin={float(np.mean(margins_known)):.4f}",
    )

    margins_novel = []
    for subj, rel in novel_pairs():
        ret = query_pair(bank, subj, rel)
        margins_novel.append(ret.margin)
    gap = float(np.mean(margins_known)) - float(np.mean(margins_novel))
    record(
        "SEED-3 novelty separation",
        gap >= 0.10,
        f"known={float(np.mean(margins_known)):.4f} novel={float(np.mean(margins_novel)):.4f} "
        f"gap={gap:.4f} (bar >=0.10). A collapse-to-bank defect makes this gap ~0.",
    )

    rng = np.random.default_rng(7)
    random_wave = rng.standard_normal(16384) + 1j * rng.standard_normal(16384)
    random_wave /= np.linalg.norm(random_wave)
    miss = bank.ledger.query(random_wave)
    record(
        "SEED-4 kill control",
        miss.margin < 0.10,
        f"random-query margin={miss.margin:.4f} (bar <0.10)",
    )

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print()
    print(f"=== {passed}/{len(RESULTS)} seeding gates passed in {time.time() - t0:.1f}s ===")
    print("[INFO] This is a seeded axiom bank with measured retrieval.")
    print("[INFO] It is NOT world knowledge, NOT a benchmark score, NOT generalization.")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
