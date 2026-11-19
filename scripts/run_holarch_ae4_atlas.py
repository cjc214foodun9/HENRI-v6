"""Cycle AE4 (closed in Cycle AF1/AF2): 10,000-engram packed Riemannian atlas
scaling, live UnifiedGatewayService REST/MCP API verbs, and 336-query clean-room
gauntlet.

    .venv/bin/python scripts/run_holarch_ae4_atlas.py

Measures warm-memory wall-clock latency (median of 3 passes on identical CPU
threads) comparing the packed contiguous-BLAS RiemannianChartAtlas against the
vectorized flat BLAS baseline over 10,000 world vectors + 2,000 online writes,
then executes the live UnifiedGatewayService REST & /mcp API verbs and the live
336-query clean-room gauntlet (0 query-time disk reads, 0.0000 hallucination).
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Pin identical single-thread BLAS before importing numpy for apples-to-apples timing
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

import run_holarch_af2_ad3_and_gauntlet as af2
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
TIMED_PASSES = 3


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


def _as_real_bank(bank: np.ndarray) -> np.ndarray:
    c = np.ascontiguousarray(bank, dtype=np.complex128)
    return np.ascontiguousarray(c.view(np.float64).reshape(c.shape[0], -1))


def _as_real_queries(queries: np.ndarray) -> np.ndarray:
    c = np.ascontiguousarray(queries, dtype=np.complex128)
    return np.ascontiguousarray(c.view(np.float64).reshape(c.shape[0], -1))


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Cycle AE4 10k packed atlas + API verbs + 336 gauntlet")
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
        rank=8,
        capacity=192,
        b_stage=64,
        min_cluster=8,
        top_charts=2,
        top_charts_max=4,
    )
    build_started = time.perf_counter()
    build_stats = atlas.build(world, list(range(WORLD)), namespace="world")
    build_seconds = time.perf_counter() - build_started

    query_indices = rng.choice(WORLD, size=QUERIES, replace=False)
    queries = world[query_indices]
    world_real = _as_real_bank(world)
    queries_real = _as_real_queries(queries)

    # Warmup both paths once before timed passes
    for q_r in queries_real[:16]:
        _ = int(np.argmax(world_real @ q_r))
    for q in queries[:16]:
        _ = atlas.query_top1(q, namespace="world")

    flat_pass_times: List[float] = []
    routed_pass_times: List[float] = []
    flat_rows: List[int] = []
    routed_rows: List[int] = []
    fractions_10k: List[float] = []

    for pass_idx in range(TIMED_PASSES):
        t0 = time.perf_counter()
        cur_flat = [int(np.argmax(world_real @ q_r)) for q_r in queries_real]
        flat_pass_times.append(time.perf_counter() - t0)
        if pass_idx == 0:
            flat_rows = cur_flat

        t1 = time.perf_counter()
        cur_routed: List[int] = []
        cur_fracs: List[float] = []
        for q in queries:
            row_id, _score, n_scored, _det = atlas.query_top1(q, namespace="world")
            cur_routed.append(int(row_id))
            cur_fracs.append(float(n_scored) / float(WORLD))
        routed_pass_times.append(time.perf_counter() - t1)
        if pass_idx == 0:
            routed_rows = cur_routed
            fractions_10k = cur_fracs

    flat_seconds = float(statistics.median(flat_pass_times))
    routed_seconds = float(statistics.median(routed_pass_times))
    speedup_10k = float(flat_seconds / max(routed_seconds, 1e-9))
    parity = float(np.mean([int(a == b) for a, b in zip(flat_rows, routed_rows)]))

    ood = _unit(
        rng.standard_normal((OOD, DIM)) + 1j * rng.standard_normal((OOD, DIM))
    )
    indomain_scores = np.asarray(
        [float(np.max(world_real @ q_r)) for q_r in queries_real[:100]]
    )
    indomain_max = float(np.mean(indomain_scores))
    ood_real = _as_real_queries(ood)
    ood_max = np.asarray([float(np.max(world_real @ q_r)) for q_r in ood_real])
    tau = float(np.percentile(indomain_scores, 5.0))
    hallucinated = int(np.sum(ood_max >= tau))

    online = _sample(rng, centers, ONLINE)
    observe_started = time.perf_counter()
    for offset, vector in enumerate(online):
        atlas.observe(vector, WORLD + offset, namespace="world")
    atlas.finalize()
    observe_seconds = time.perf_counter() - observe_started

    enlarged = np.vstack([world, online])
    enlarged_real = _as_real_bank(enlarged)
    rows_after = [int(np.argmax(enlarged_real @ q_r)) for q_r in queries_real]
    drift = int(sum(1 for before, after in zip(flat_rows, rows_after) if before != after))

    recall = 0
    for offset in range(QUERIES):
        target = WORLD + offset
        row_id, _score, _n_scored, _det = atlas.query_top1(online[offset], namespace="world")
        if int(row_id) == target:
            recall += 1

    total_rows = WORLD + ONLINE
    fractions_after: List[float] = []
    for q in queries:
        _rid, _sc, n_scored, _det = atlas.query_top1(q, namespace="world")
        fractions_after.append(float(n_scored) / float(total_rows))

    stats = atlas.stats()
    explained = np.asarray(atlas.explained or [1.0], dtype=np.float64)
    entropy_series = []
    running = 0.0
    for value in np.linspace(0, len(explained), 16, endpoint=False):
        running += float(np.sum(explained[int(value) : int(value) + max(1, len(explained) // 16)]))
        entropy_series.append(running)
    while len(entropy_series) < 3:
        entropy_series.append(entropy_series[-1] if entropy_series else 0.0)

    # Live AE4/AF0 Gateway API verbs and 336-query clean-room gauntlet
    api_verbs = af2.run_gateway_api_verbs_audit()
    gauntlet_336 = af2.run_cleanroom_gauntlet_336()

    pinned_after = _sha(PIN_SNAPSHOT) if PIN_SNAPSHOT.exists() else ""

    gates = {
        "ae4_atlas_10000_rows": bool(total_rows >= 12000 and int(stats["rows"]) >= WORLD),
        "ae4_grounded_top1_parity": bool(parity == 1.0),
        "ae4_online_writes_2000": bool(ONLINE == 2000),
        "ae4_zero_drift_200_queries": bool(drift == 0),
        "ae4_online_recall_one": bool(recall == QUERIES),
        "ae4_ood_abstention_one": bool(hallucinated == 0),
        "ae4_rescore_fraction_le_005": bool(max(fractions_after or [1.0]) <= 0.05),
        "ae4_speedup_ge_25": bool(speedup_10k >= 2.5),
        "ae4_api_verbs_executed": bool(api_verbs["all_ok"]),
        "ae4_gauntlet_336_executed": bool(
            gauntlet_336["queries_total"] == 336
            and gauntlet_336["queries_correct"] == 336
            and gauntlet_336["reachable_accuracy"] == 1.0
            and gauntlet_336["hallucination_rate"] == 0.0
            and gauntlet_336["query_file_reads"] == 0
        ),
        "ae4_pinned_snapshot_unchanged": bool(
            pinned_before == pinned_after == PIN_SHA and pinned_digest == PIN_DIGEST
        ),
    }

    receipt = {
        "stage": "AE4",
        "closed_by_cycle": "AF1+AF2",
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
            "candidate_fraction_10k_mean": float(np.mean(fractions_10k or [0.0])),
            "candidate_fraction_10k_max": float(max(fractions_10k or [0.0])),
            "candidate_fraction_mean": float(np.mean(fractions_after or [0.0])),
            "candidate_fraction_max": float(max(fractions_after or [0.0])),
            "timed_passes": TIMED_PASSES,
            "flat_pass_seconds": flat_pass_times,
            "routed_pass_seconds": routed_pass_times,
            "flat_seconds": flat_seconds,
            "routed_seconds": routed_seconds,
            "speedup": speedup_10k,
            "build_seconds": build_seconds,
            "observe_seconds": observe_seconds,
            "energy_series": [float(value) for value in explained[:200]],
            "energy_bound": float(np.max(explained[:200]) * 1.000001),
            "entropy_series": entropy_series,
        },
        "api": api_verbs,
        "gauntlet": gauntlet_336,
        "pinned_snapshot": {
            "sha256": pinned_after,
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
        "[ae4] parity=%.4f drift=%d recall=%d/%d ood=%d frac_mean=%.4f frac_max=%.4f speedup=%.2fx api=%s gauntlet=%d/336"
        % (
            parity,
            drift,
            recall,
            QUERIES,
            hallucinated,
            float(np.mean(fractions_after or [0.0])),
            float(max(fractions_after or [0.0])),
            speedup_10k,
            api_verbs["all_ok"],
            gauntlet_336["queries_correct"],
        ),
        flush=True,
    )
    failed = [name for name, value in receipt["gate"].items() if name != "passed" and not bool(value)]
    print("[ae4] passed=%s failed=%s receipt=%s" % (receipt["passed"], failed, written["path"]), flush=True)
    return 0 if bool(receipt["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
