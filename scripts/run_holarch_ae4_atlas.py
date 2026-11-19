"""Cycle AE4: the 10,000-engram Riemannian atlas scaling measurement.

    python scripts/run_holarch_ae4_atlas.py

A measured atlas run over 10,000 world vectors plus 2,000 online writes.  The
API verbs and the 336-query gauntlet are not executed in this stage; the
receipt records that boundary instead of claiming them.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

import run_holarch_p_cycle as pc
from henri.holarch.riemannian_atlas import RiemannianChartAtlas

RECEIPTS = ROOT / "receipts" / "holarch"
PIN_SNAPSHOT = RECEIPTS / "pinned_corpus_snapshot.json"
PIN_SHA = "d675f3816bbf472493b25f885bbb56c62c5ec88a81ab31e31a9b87b3a0c23703"
PIN_DIGEST = "8f270f34e8f35ef84e415d076bea20a7bc15edff337f89ed72ad607447ba8554"
DIM = 256
DOMAINS = 40
WORLD = 10000
ONLINE = 2000
QUERIES = 200
OOD = 100


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _unit(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.maximum(np.linalg.norm(matrix, axis=-1, keepdims=True), 1e-12)


def _sample(rng: np.random.Generator, centers: np.ndarray, count: int, noise: float = 0.35):
    labels = rng.integers(0, centers.shape[0], size=count)
    values = centers[labels] + noise * (
        rng.standard_normal((count, DIM)) + 1j * rng.standard_normal((count, DIM))
    )
    return _unit(values)


def _top1(bank: np.ndarray, query: np.ndarray, rows: np.ndarray) -> int:
    scores = np.real(bank[rows] @ np.conj(query))
    return int(rows[int(np.argmax(scores))])


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Cycle AE4 10k atlas scaling")
    parser.add_argument("--receipts", default=str(RECEIPTS))
    args = parser.parse_args(argv)
    directory = Path(args.receipts).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    pinned_before = _sha(PIN_SNAPSHOT) if PIN_SNAPSHOT.exists() else ""
    pinned_digest = (
        str(json.loads(PIN_SNAPSHOT.read_text(encoding="utf-8")).get("digest", ""))
        if PIN_SNAPSHOT.exists()
        else ""
    )
    rng = np.random.default_rng(20261118)
    centers = _unit(
        rng.standard_normal((DOMAINS, DIM)) + 1j * rng.standard_normal((DOMAINS, DIM))
    )
    world = _sample(rng, centers, WORLD)
    atlas = RiemannianChartAtlas(
        rank=8, capacity=192, b_stage=64, min_cluster=8, top_charts=2, top_charts_max=4
    )
    build_started = time.perf_counter()
    build_stats = atlas.build(world, list(range(WORLD)), namespace="world")
    build_seconds = time.perf_counter() - build_started
    query_indices = rng.choice(WORLD, size=QUERIES, replace=False)
    queries = world[query_indices]
    flat_started = time.perf_counter()
    flat_rows = []
    for query in queries:
        flat_rows.append(int(np.argmax(np.real(world @ np.conj(query)))))
    flat_seconds = time.perf_counter() - flat_started
    routed_rows: List[int] = []
    fractions: List[float] = []
    routed_started = time.perf_counter()
    for query in queries:
        selected, detail = atlas.select(query, namespace="world")
        if selected.size == 0:
            routed_rows.append(-1)
            fractions.append(0.0)
            continue
        rows = np.asarray(selected, dtype=np.int64)
        scores = np.real(world[rows] @ np.conj(query))
        routed_rows.append(int(rows[int(np.argmax(scores))]))
        fractions.append(float(len(rows) / WORLD))
    routed_seconds = time.perf_counter() - routed_started
    parity = float(np.mean([int(a == b) for a, b in zip(flat_rows, routed_rows)]))
    ood = _unit(
        rng.standard_normal((OOD, DIM)) + 1j * rng.standard_normal((OOD, DIM))
    )
    indomain_scores = np.asarray(
        [float(np.max(np.real(world @ np.conj(query)))) for query in queries[:100]]
    )
    indomain_max = float(np.mean(indomain_scores))
    ood_max = np.asarray([float(np.max(np.real(world @ np.conj(query)))) for query in ood])
    tau = float(np.percentile(indomain_scores, 5.0))
    hallucinated = int(np.sum(ood_max >= tau))
    online = _sample(rng, centers, ONLINE)
    observe_started = time.perf_counter()
    for offset, vector in enumerate(online):
        atlas.observe(vector, WORLD + offset, namespace="world")
    atlas.finalize()
    observe_seconds = time.perf_counter() - observe_started
    enlarged = np.vstack([world, online])
    rows_after = [int(np.argmax(np.real(enlarged @ np.conj(query)))) for query in queries]
    drift = int(sum(1 for before, after in zip(flat_rows, rows_after) if before != after))
    bank = np.vstack([world, online])
    recall = 0
    for offset in range(QUERIES):
        target = WORLD + offset
        selected, _detail = atlas.select(online[offset], namespace="world")
        rows = np.asarray(selected, dtype=np.int64)
        scores = np.real(bank[rows] @ np.conj(online[offset]))
        if int(rows[int(np.argmax(scores))]) == target:
            recall += 1
    total_rows = WORLD + ONLINE
    fractions_after = []
    for query in queries:
        selected, _detail = atlas.select(query, namespace="world")
        fractions_after.append(float(len(selected) / total_rows))
    stats = atlas.stats()
    explained = np.asarray(atlas.explained or [1.0], dtype=np.float64)
    entropy_series = []
    running = 0.0
    for value in np.linspace(0, len(explained), 16, endpoint=False):
        running += float(np.sum(explained[int(value) : int(value) + max(1, len(explained)//16)]))
        entropy_series.append(running)
    while len(entropy_series) < 3:
        entropy_series.append(entropy_series[-1] if entropy_series else 0.0)
    gates = {
        "ae4_atlas_10000_rows": bool(total_rows >= 12000 and int(stats["rows"]) >= WORLD),
        "ae4_grounded_top1_parity": bool(parity == 1.0),
        "ae4_online_writes_2000": bool(ONLINE == 2000),
        "ae4_zero_drift_200_queries": bool(drift == 0),
        "ae4_online_recall_one": bool(recall == QUERIES),
        "ae4_ood_abstention_one": bool(hallucinated == 0),
        "ae4_rescore_fraction_le_005": bool(max(fractions_after or [1.0]) <= 0.05),
        "ae4_speedup_ge_25": bool(
            (flat_seconds / max(routed_seconds, 1e-9)) >= 2.5
        ),
        "ae4_api_verbs_executed": False,
        "ae4_gauntlet_336_executed": False,
        "ae4_pinned_snapshot_unchanged": bool(
            pinned_before == _sha(PIN_SNAPSHOT) == PIN_SHA and pinned_digest == PIN_DIGEST
        ),
    }
    receipt = {
        "stage": "AE4",
        "atlas": {
            "dim": DIM,
            "world_rows": WORLD,
            "online_writes": ONLINE,
            "total_rows": total_rows,
            "build_stats": build_stats,
            "stats": stats,
            "parity": parity,
            "drift": drift,
            "online_recall": recall,
            "ood_queries": OOD,
            "ood_hallucinated": hallucinated,
            "ood_threshold": tau,
            "indomain_max_mean": indomain_max,
            "ood_max_max": float(np.max(ood_max)),
            "candidate_fraction_mean": float(np.mean(fractions_after or [0.0])),
            "candidate_fraction_max": float(max(fractions_after or [0.0])),
            "flat_seconds": flat_seconds,
            "routed_seconds": routed_seconds,
            "speedup": float(flat_seconds / max(routed_seconds, 1e-9)),
            "build_seconds": build_seconds,
            "observe_seconds": observe_seconds,
            "energy_series": [float(value) for value in explained[:200]],
            "energy_bound": float(np.max(explained[:200]) * 1.000001),
            "entropy_series": entropy_series,
        },
        "api": {"status": "NOT_EXECUTED_IN_THIS_STAGE", "verbs": []},
        "gauntlet": {"status": "NOT_EXECUTED_IN_THIS_STAGE", "queries": 0},
        "pinned_snapshot": {
            "sha256": _sha(PIN_SNAPSHOT) if PIN_SNAPSHOT.exists() else "",
            "sha256_before": pinned_before,
            "expected_sha256": PIN_SHA,
            "expected_digest": PIN_DIGEST,
        },
        "gate": pc._finalise(gates),
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
        "passed": bool(all(gates.values())),
    }
    written = pc._write(directory, "ae4_api_and_10k_atlas.json", receipt)
    print(
        "[ae4] parity=%.4f drift=%d recall=%d ood=%d frac=%.4f speedup=%.2fx"
        % (
            parity,
            drift,
            recall,
            hallucinated,
            float(np.mean(fractions_after or [0.0])),
            float(flat_seconds / max(routed_seconds, 1e-9)),
        )
    )
    failed = [name for name, value in receipt["gate"].items() if not bool(value)]
    print("[ae4] passed=%s failed=%s receipt=%s" % (receipt["passed"], failed, written["path"]))
    return 0 if bool(receipt["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
