"""Cycle AF1: Packed contiguous-BLAS RiemannianChartAtlas & 200/200 online recall.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        python scripts/run_holarch_af1_fast_atlas.py

Measures warm-memory wall-clock latency (median of 5 timed passes on identical
single-thread CPU BLAS for both flat BLAS baselines and the packed charted
atlas), grounded top-1 parity, post-write drift, OOD abstention, online recall,
and candidate rescore fraction on N = 10,000 world rows + 2,000 online writes.
Seals `receipts/holarch/af1_fast_riemannian_atlas.json` + `.sha256`.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Pin identical CPU BLAS thread count before importing NumPy.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

import run_holarch_p_cycle as pc
from henri.holarch.riemannian_atlas import RiemannianChartAtlas, _as_real_view

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
TIMED_PASSES = 5


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _unit(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.maximum(np.linalg.norm(matrix, axis=-1, keepdims=True), 1e-12)


def _sample(rng: np.random.Generator, centers: np.ndarray, count: int, noise: float = 0.35) -> np.ndarray:
    labels = rng.integers(0, centers.shape[0], size=count)
    values = centers[labels] + noise * (
        rng.standard_normal((count, DIM)) + 1j * rng.standard_normal((count, DIM))
    )
    return np.ascontiguousarray(_unit(values), dtype=np.complex128)


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Cycle AF1 fast packed Riemannian atlas benchmark")
    parser.add_argument("--receipts", default=str(RECEIPTS))
    parser.add_argument("--passes", type=int, default=TIMED_PASSES)
    args = parser.parse_args(argv)
    directory = Path(args.receipts).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    passes = max(3, int(args.passes))

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
    world_real = _as_real_view(world)

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
    build_seconds = float(time.perf_counter() - build_started)

    query_indices = rng.choice(WORLD, size=QUERIES, replace=False)
    queries = np.ascontiguousarray(world[query_indices], dtype=np.complex128)
    queries_real = [q.view(np.float64) for q in queries]

    # Warm-memory warmup pass before timed passes.
    for q, qr in zip(queries[:16], queries_real[:16]):
        _ = int(np.argmax(world_real @ qr))
        _ = atlas.query_top1(q, namespace="world")

    flat_complex_10k_runs: List[float] = []
    flat_real_10k_runs: List[float] = []
    atlas_10k_runs: List[float] = []
    flat_rows_10k: List[int] = []
    routed_rows_10k: List[int] = []
    fractions_10k: List[float] = []

    for pass_idx in range(passes):
        t0 = time.perf_counter()
        f_rows = [int(np.argmax(np.real(world @ np.conj(q)))) for q in queries]
        flat_complex_10k_runs.append(float(time.perf_counter() - t0))

        t0 = time.perf_counter()
        _fr_rows = [int(np.argmax(world_real @ qr)) for qr in queries_real]
        flat_real_10k_runs.append(float(time.perf_counter() - t0))

        t0 = time.perf_counter()
        r_rows: List[int] = []
        r_fracs: List[float] = []
        for q in queries:
            row_id, _score, scored, _detail = atlas.query_top1(q, namespace="world")
            r_rows.append(int(row_id))
            r_fracs.append(float(scored) / float(WORLD))
        atlas_10k_runs.append(float(time.perf_counter() - t0))

        if pass_idx == 0:
            flat_rows_10k = f_rows
            routed_rows_10k = r_rows
            fractions_10k = r_fracs

    parity_10k = float(np.mean([int(a == b) for a, b in zip(flat_rows_10k, routed_rows_10k)]))

    # OOD abstention check using routed coherence threshold.
    ood = _unit(rng.standard_normal((OOD, DIM)) + 1j * rng.standard_normal((OOD, DIM)))
    indomain_scores = np.asarray(
        [atlas.query_top1(q, namespace="world")[1] for q in queries[:100]], dtype=np.float64
    )
    ood_scores = np.asarray(
        [atlas.query_top1(q, namespace="world")[1] for q in ood], dtype=np.float64
    )
    tau = float(np.percentile(indomain_scores, 5.0))
    hallucinated = int(np.sum(ood_scores >= tau))

    # 2,000 online writes with coherent condensation & bounded Voronoi dual-membership.
    online = _sample(rng, centers, ONLINE)
    observe_started = time.perf_counter()
    for offset, vector in enumerate(online):
        atlas.observe(vector, WORLD + offset, namespace="world")
    atlas.finalize()
    observe_seconds = float(time.perf_counter() - observe_started)

    total_rows = WORLD + ONLINE
    bank = np.ascontiguousarray(np.vstack([world, online]), dtype=np.complex128)
    bank_real = _as_real_view(bank)

    # Warmup on 12k bank.
    for q, qr in zip(queries[:16], queries_real[:16]):
        _ = int(np.argmax(bank_real @ qr))
        _ = atlas.query_top1(q, namespace="world")

    flat_complex_12k_runs: List[float] = []
    flat_real_12k_runs: List[float] = []
    atlas_12k_runs: List[float] = []
    rows_after_flat: List[int] = []
    rows_after_atlas: List[int] = []
    fractions_12k: List[float] = []

    for pass_idx in range(passes):
        t0 = time.perf_counter()
        f12_rows = [int(np.argmax(np.real(bank @ np.conj(q)))) for q in queries]
        flat_complex_12k_runs.append(float(time.perf_counter() - t0))

        t0 = time.perf_counter()
        _fr12 = [int(np.argmax(bank_real @ qr)) for qr in queries_real]
        flat_real_12k_runs.append(float(time.perf_counter() - t0))

        t0 = time.perf_counter()
        a12_rows: List[int] = []
        a12_fracs: List[float] = []
        for q in queries:
            row_id, _score, scored, _detail = atlas.query_top1(q, namespace="world")
            a12_rows.append(int(row_id))
            a12_fracs.append(float(scored) / float(total_rows))
        atlas_12k_runs.append(float(time.perf_counter() - t0))

        if pass_idx == 0:
            rows_after_flat = f12_rows
            rows_after_atlas = a12_rows
            fractions_12k = a12_fracs

    drift_flat = int(sum(1 for a, b in zip(flat_rows_10k, rows_after_flat) if a != b))
    drift_atlas = int(sum(1 for a, b in zip(flat_rows_10k, rows_after_atlas) if a != b))
    parity_12k = float(np.mean([int(a == b) for a, b in zip(rows_after_flat, rows_after_atlas)]))

    # Online recall over 200 newly condensed online writes.
    recall = 0
    online_fractions: List[float] = []
    for offset in range(QUERIES):
        target = WORLD + offset
        row_id, _score, scored, _detail = atlas.query_top1(online[offset], namespace="world")
        online_fractions.append(float(scored) / float(total_rows))
        if int(row_id) == target:
            recall += 1

    # Summary statistics (median warm-memory latency across passes).
    med_flat_c_10k = float(np.median(flat_complex_10k_runs))
    med_flat_r_10k = float(np.median(flat_real_10k_runs))
    med_atlas_10k = float(np.median(atlas_10k_runs))
    speedup_10k_vs_complex = float(med_flat_c_10k / max(med_atlas_10k, 1e-9))
    speedup_10k_vs_real = float(med_flat_r_10k / max(med_atlas_10k, 1e-9))

    med_flat_c_12k = float(np.median(flat_complex_12k_runs))
    med_flat_r_12k = float(np.median(flat_real_12k_runs))
    med_atlas_12k = float(np.median(atlas_12k_runs))
    speedup_12k_vs_complex = float(med_flat_c_12k / max(med_atlas_12k, 1e-9))
    speedup_12k_vs_real = float(med_flat_r_12k / max(med_atlas_12k, 1e-9))

    all_12k_fracs = fractions_12k + online_fractions
    frac_12k_mean = float(np.mean(all_12k_fracs))
    frac_12k_max = float(np.max(all_12k_fracs))
    frac_10k_mean = float(np.mean(fractions_10k))
    frac_10k_max = float(np.max(fractions_10k))

    stats = atlas.stats()
    explained = np.asarray(atlas.explained or [1.0], dtype=np.float64)
    entropy_series = []
    running = 0.0
    for value in np.linspace(0, len(explained), 16, endpoint=False):
        running += float(
            np.sum(explained[int(value) : int(value) + max(1, len(explained) // 16)])
        )
        entropy_series.append(running)
    while len(entropy_series) < 3:
        entropy_series.append(entropy_series[-1] if entropy_series else 0.0)

    gates = {
        "af1_total_rows_12000": bool(total_rows >= 12000 and int(stats["rows"]) == total_rows),
        "af1_grounded_top1_parity_one": bool(parity_10k == 1.0 and parity_12k == 1.0),
        "af1_post_write_drift_zero": bool(drift_flat == 0 and drift_atlas == 0),
        "af1_ood_hallucination_zero": bool(hallucinated == 0),
        "af1_online_recall_200_of_200": bool(recall == QUERIES),
        "af1_rescore_fraction_mean_le_004": bool(frac_12k_mean <= 0.0400 and frac_10k_mean <= 0.0400),
        "af1_rescore_fraction_max_le_005": bool(frac_12k_max <= 0.0500 and frac_10k_max <= 0.0500),
        "af1_speedup_vs_flat_real_blas_ge_25": bool(
            speedup_10k_vs_real >= 2.50 and speedup_12k_vs_real >= 2.50
        ),
        "af1_speedup_vs_flat_complex_blas_ge_25": bool(
            speedup_10k_vs_complex >= 2.50 and speedup_12k_vs_complex >= 2.50
        ),
        "af1_boundary_ratio_le_008": bool(float(stats["boundary_ratio"]) <= 0.08),
        "af1_pinned_snapshot_unchanged": bool(
            pinned_before == _sha(PIN_SNAPSHOT) == PIN_SHA and pinned_digest == PIN_DIGEST
        ),
    }

    receipt = {
        "stage": "AF1",
        "kind": "fast_packed_riemannian_atlas",
        "thread_pin": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", ""),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", ""),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", ""),
        },
        "compute_device": "cpu",
        "timed_passes": passes,
        "atlas": {
            "dim": DIM,
            "domains": DOMAINS,
            "world_rows": WORLD,
            "online_writes": ONLINE,
            "total_rows": total_rows,
            "build_stats": build_stats,
            "stats": stats,
            "parity_10k": parity_10k,
            "parity_12k": parity_12k,
            "drift_flat": drift_flat,
            "drift_atlas": drift_atlas,
            "online_recall": recall,
            "online_recall_total": QUERIES,
            "ood_queries": OOD,
            "ood_hallucinated": hallucinated,
            "ood_threshold": tau,
            "indomain_max_mean": float(np.mean(indomain_scores)),
            "ood_max_max": float(np.max(ood_scores)),
            "candidate_fraction_10k_mean": frac_10k_mean,
            "candidate_fraction_10k_max": frac_10k_max,
            "candidate_fraction_12k_mean": frac_12k_mean,
            "candidate_fraction_12k_max": frac_12k_max,
            "latency_10k_median_s": {
                "flat_complex_blas": med_flat_c_10k,
                "flat_real_blas": med_flat_r_10k,
                "packed_atlas": med_atlas_10k,
            },
            "latency_12k_median_s": {
                "flat_complex_blas": med_flat_c_12k,
                "flat_real_blas": med_flat_r_12k,
                "packed_atlas": med_atlas_12k,
            },
            "speedup_10k_vs_flat_complex": speedup_10k_vs_complex,
            "speedup_10k_vs_flat_real": speedup_10k_vs_real,
            "speedup_12k_vs_flat_complex": speedup_12k_vs_complex,
            "speedup_12k_vs_flat_real": speedup_12k_vs_real,
            "build_seconds": build_seconds,
            "observe_seconds": observe_seconds,
            "energy_series": [float(v) for v in explained[:200]],
            "energy_bound": float(np.max(explained[:200]) * 1.000001),
            "entropy_series": entropy_series,
        },
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
    written = pc._write(directory, "af1_fast_riemannian_atlas.json", receipt)
    print(
        "[af1] parity=%.4f/%.4f drift=%d recall=%d/%d ood=%d/%d frac12k=%.4f(max=%.4f) "
        "speedup10k=%.2fx(real=%.2fx) speedup12k=%.2fx(real=%.2fx)"
        % (
            parity_10k,
            parity_12k,
            drift_atlas,
            recall,
            QUERIES,
            hallucinated,
            OOD,
            frac_12k_mean,
            frac_12k_max,
            speedup_10k_vs_complex,
            speedup_10k_vs_real,
            speedup_12k_vs_complex,
            speedup_12k_vs_real,
        )
    )
    failed = [name for name, val in receipt["gate"].items() if not bool(val)]
    print("[af1] passed=%s failed=%s receipt=%s" % (receipt["passed"], failed, written["path"]))
    return 0 if bool(receipt["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
