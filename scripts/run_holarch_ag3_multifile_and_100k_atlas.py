"""Cycle AG3: Multi-File Cross-Module RepoPatch & 100,000-Engram Persistent Atlas.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
        python scripts/run_holarch_ag3_multifile_and_100k_atlas.py

Executes both parts of Stage AG3 cold at zero external LLM calls:
  Part A (Multi-File Cross-Module RepoPatch + QuixBugs Non-Regression):
    - Verifies SHA-256 of every case and trap in
      receipts/holarch/external/repo_patch_multifile/MANIFEST.json.
    - Runs RepoPatchEngine across 12 multi-file (3-5 module) defect repositories
      plus unseen hidden regression suites and 4 cross-module RepoSheaf traps.
    - Audits zero task-name branching and zero forbidden LLM imports.
    - Re-runs AF3 unmodified external QuixBugs (14 tasks + 4 traps) to verify
      non-regression (>= 12/14 @ 1.0000 precision, 4/4 traps refused).
  Part B (100,000-Engram Persistent Hierarchical Packed RiemannianChartAtlas):
    - Benchmarks N = 100,000 world rows + 5,000 online writes (D = 256,
      1 pinned CPU BLAS thread, 5 timed passes, median warm-memory latency).
    - Verifies 1.0000 grounded top-1 parity (100k and 105k), 0/200 drift,
      200/200 online recall, 0/100 OOD hallucination, candidate rescore
      fraction mean <= 0.0250 (max <= 0.0500), wall-clock speedup >= 15.0x
      vs flat complex BLAS (>= 12.0x vs flat real BLAS).
    - Persists the full 105,000-engram packed atlas via binary slab + SQLite
      checkpoint and spawns a fresh cold Python subprocess mounting
      UnifiedGatewayService(GatewayConfig(atlas_persist_dir=...)), asserting
      mount latency < 2.0 s, rebuilt_charts == 0, and 1.0000 post-restart
      query parity over 200 queries.

Seals:
  - receipts/holarch/ag3_multifile_repo_and_100k_atlas.json (+ .sha256)
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

# Pin identical single-thread CPU BLAS before importing NumPy.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

import run_holarch_af3_external_and_arc as af3
import run_holarch_p_cycle as pc
from henri.holarch.cold_synth import audit_task_specific_branches
from henri.holarch.repo_patch import (
    RepoCase,
    RepoPatchEngine,
    RepoSubprocessSandbox,
    materialise,
)
from henri.holarch.riemannian_atlas import RiemannianChartAtlas, _as_real_view

RECEIPTS = ROOT / "receipts" / "holarch"
EXT_MULTIFILE = RECEIPTS / "external" / "repo_patch_multifile"
PIN_SNAPSHOT = RECEIPTS / "pinned_corpus_snapshot.json"
PIN_SHA = "d675f3816bbf472493b25f885bbb56c62c5ec88a81ab31e31a9b87b3a0c23703"
PIN_DIGEST = "8f270f34e8f35ef84e415d076bea20a7bc15edff337f89ed72ad607447ba8554"
FORBIDDEN_IMPORTS = ("openai", "anthropic", "requests", "urllib", "transformers", "tokenizers")

# Part B constants (100k world + 5k online writes at D = 256).
DIM = 256
DOMAINS = 128
WORLD = 100000
ONLINE = 5000
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


def run_part_a_multifile_and_quixbugs() -> Dict[str, Any]:
    started = time.perf_counter()
    manifest_path = EXT_MULTIFILE / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_sha = _sha(manifest_path)

    digest_mismatches: List[str] = []
    case_rows: List[Dict[str, Any]] = []
    llm_calls = 0

    for entry in manifest["cases"]:
        cpath = EXT_MULTIFILE / entry["file"]
        if _sha(cpath) != entry["sha256"]:
            digest_mismatches.append(entry["file"])
        raw = json.loads(cpath.read_text(encoding="utf-8"))
        cid = str(raw["id"])
        case = RepoCase(
            name=cid,
            issue=str(raw["issue"]),
            files=dict(raw["files"]),
            public=dict(raw["public"]),
            hidden=dict(raw["hidden"]),
            test_path=str(raw.get("test_path", "tests/test_public.py")),
        )
        engine = RepoPatchEngine(rollouts=48, timeout_s=2.0)
        outcome = engine.solve(case)
        llm_calls += int(outcome.get("llm_calls", 0))

        hidden_ok = False
        if outcome["status"] == "emitted":
            with tempfile.TemporaryDirectory(prefix="ag3_mf_hid_") as td:
                materialise(case, Path(td), include_hidden=True, patched=outcome["patched_files"])
                hrun = RepoSubprocessSandbox(timeout_s=2.0).run_tests(
                    Path(td), "tests/test_hidden.py"
                )
                hidden_ok = bool(hrun["passed"])

        patch_info = outcome.get("patch") or {}
        case_rows.append(
            {
                "id": cid,
                "module_count": int(raw["module_count"]),
                "defect_family": str(raw["defect_family"]),
                "call_chain": str(raw["call_chain"]),
                "sha256": str(entry["sha256"]),
                "status": str(outcome["status"]),
                "reason": str(outcome.get("reason", "")),
                "hidden_passed": bool(hidden_ok),
                "baseline_sheaf_energy": float((outcome.get("sheaf_baseline") or {}).get("energy", 0.0)),
                "sheaf_after_energy": float((outcome.get("sheaf_after") or {}).get("energy", 0.0)),
                "rollouts": int(len(outcome.get("rollouts") or [])),
                "repulsors": int(len(outcome.get("repulsors") or [])),
                "blocked_by_sheaf": int(outcome.get("blocked_by_sheaf", 0)),
                "patched_qualname": str(patch_info.get("qualname", "")),
                "patch_kind": str(patch_info.get("kind", "")),
                "patch_note": str(patch_info.get("note", "")),
                "seconds": float(outcome.get("seconds", 0.0)),
            }
        )
        print(
            "[ag3.A case] %-24s mods=%d status=%-8s hidden=%-5s sheaf=%.1f->%.1f rep=%2d qn=%s (%s)"
            % (
                cid,
                int(raw["module_count"]),
                outcome["status"],
                hidden_ok,
                float((outcome.get("sheaf_baseline") or {}).get("energy", 0.0)),
                float((outcome.get("sheaf_after") or {}).get("energy", 0.0)),
                len(outcome.get("repulsors") or []),
                patch_info.get("qualname", ""),
                patch_info.get("note", outcome.get("reason", "")),
            ),
            flush=True,
        )

    trap_rows: List[Dict[str, Any]] = []
    for entry in manifest["traps"]:
        tpath = EXT_MULTIFILE / entry["file"]
        if _sha(tpath) != entry["sha256"]:
            digest_mismatches.append(entry["file"])
        raw = json.loads(tpath.read_text(encoding="utf-8"))
        tid = str(raw["id"])
        case = RepoCase(
            name=tid,
            issue=str(raw["issue"]),
            files=dict(raw["files"]),
            public=dict(raw["public"]),
            hidden=dict(raw["hidden"]),
            test_path=str(raw.get("test_path", "tests/test_public.py")),
        )
        engine = RepoPatchEngine(rollouts=48, timeout_s=2.0)
        outcome = engine.solve(case)
        llm_calls += int(outcome.get("llm_calls", 0))
        baseline_energy = float((outcome.get("sheaf_baseline") or {}).get("energy", 0.0))
        refused = bool(outcome["status"] == "abstained" and not outcome.get("patched_files"))
        trap_rows.append(
            {
                "id": tid,
                "module_count": int(raw["module_count"]),
                "kind": str(raw["kind"]),
                "expected_sheaf_violation_kind": str(raw.get("expected_sheaf_violation_kind", "")),
                "sha256": str(entry["sha256"]),
                "baseline_sheaf_energy": baseline_energy,
                "blocked_by_sheaf": int(outcome.get("blocked_by_sheaf", 0)),
                "status": str(outcome["status"]),
                "reason": str(outcome.get("reason", "")),
                "refused": bool(refused),
            }
        )
        print(
            "[ag3.A trap] %-32s refused=%-5s sheaf0=%.1f reason=%s"
            % (tid, refused, baseline_energy, outcome.get("reason", "")),
            flush=True,
        )

    emitted = sum(1 for r in case_rows if r["status"] == "emitted")
    passed_hidden = sum(1 for r in case_rows if r["hidden_passed"])
    precision = float(passed_hidden) / float(emitted) if emitted else 0.0
    traps_refused = sum(1 for r in trap_rows if r["refused"])
    all_emitted_sheaf_zero = all(
        r["sheaf_after_energy"] == 0.0 for r in case_rows if r["status"] == "emitted"
    )
    all_traps_sheaf_nonzero = all(r["baseline_sheaf_energy"] > 0.0 for r in trap_rows)

    ids = [r["id"] for r in case_rows] + [r["id"] for r in trap_rows]
    rp_text = (ROOT / "henri" / "holarch" / "repo_patch.py").read_text(encoding="utf-8")
    rs_text = (ROOT / "henri" / "holarch" / "repo_sheaf.py").read_text(encoding="utf-8")
    branch_audit_rp = audit_task_specific_branches(rp_text, identifiers=ids)
    branch_audit_rs = audit_task_specific_branches(rs_text, identifiers=ids)
    forbidden = [
        name for name in FORBIDDEN_IMPORTS if ("import " + name) in (rp_text + "\n" + rs_text)
    ]

    # Run QuixBugs non-regression directly through AF3 verifier.
    print("\n[ag3.A] Running AF3 unmodified external QuixBugs non-regression...", flush=True)
    quixbugs = af3.run_af3_repo_patch()

    return {
        "multifile": {
            "slice_id": str(manifest["slice_id"]),
            "manifest_path": str(manifest_path.relative_to(ROOT)),
            "manifest_sha256": manifest_sha,
            "digest_mismatches": digest_mismatches,
            "total_cases": len(case_rows),
            "emitted": emitted,
            "passed_hidden": passed_hidden,
            "pass_at_1": float(passed_hidden) / float(max(1, len(case_rows))),
            "emitted_precision": precision,
            "all_emitted_sheaf_zero": bool(all_emitted_sheaf_zero),
            "traps_total": len(trap_rows),
            "traps_refused": traps_refused,
            "all_traps_sheaf_nonzero": bool(all_traps_sheaf_nonzero),
            "total_repulsors_registered": int(sum(r["repulsors"] for r in case_rows)),
            "llm_calls": llm_calls,
            "branch_audit": {
                "repo_patch": branch_audit_rp,
                "repo_sheaf": branch_audit_rs,
                "zero_task_name_branches": bool(
                    branch_audit_rp["task_identity_comparisons"] == 0
                    and branch_audit_rs["task_identity_comparisons"] == 0
                    and not branch_audit_rp["module_literal_overlap"]
                    and not branch_audit_rs["module_literal_overlap"]
                ),
                "forbidden_imports": forbidden,
            },
            "cases": case_rows,
            "traps": trap_rows,
        },
        "quixbugs_non_regression": {
            "slice_id": quixbugs["slice_id"],
            "manifest_sha256": quixbugs["manifest_sha256"],
            "digest_mismatches": quixbugs["digest_mismatches"],
            "total_tasks": quixbugs["total_tasks"],
            "emitted": quixbugs["emitted"],
            "passed_hidden": quixbugs["passed_hidden"],
            "pass_at_1": quixbugs["pass_at_1"],
            "emitted_precision": quixbugs["emitted_precision"],
            "traps_total": quixbugs["traps_total"],
            "traps_refused": quixbugs["traps_refused"],
            "llm_calls": quixbugs["llm_calls"],
            "zero_task_name_branches": quixbugs["branch_audit"]["zero_task_name_branches"],
            "seconds": quixbugs["seconds"],
        },
        "seconds": float(time.perf_counter() - started),
    }


def _cold_restart_gateway_probe(
    persist_dir: Path, queries_path: Path, expected_rows_path: Path
) -> Dict[str, Any]:
    """Spawn a fresh Python subprocess that mounts UnifiedGatewayService cold."""
    probe_code = """
import json, os, sys, tempfile, time
from pathlib import Path
import numpy as np

sys.path.insert(0, os.environ["HENRI_ROOT"])
from henri.gateway.config import GatewayConfig
from henri.gateway.service import UnifiedGatewayService

persist_dir = Path(os.environ["AG3_PERSIST_DIR"])
queries = np.load(os.environ["AG3_QUERIES_NPY"])
expected = json.loads(Path(os.environ["AG3_EXPECTED_JSON"]).read_text(encoding="utf-8"))

with tempfile.TemporaryDirectory(prefix="ag3_cold_gw_") as gw_td:
    gw_root = Path(gw_td)
    cfg = GatewayConfig(
        zone_c_backend="sqlite",
        sqlite_path=str(gw_root / "gw.sqlite3"),
        zone_c_sqlite_path=str(gw_root / "zonec.sqlite3"),
        signing_key_path=str(gw_root / "ed25519.pem"),
        holarch_autoload=False,
        atlas_persist_dir=str(persist_dir),
    )
    t0 = time.perf_counter()
    svc = UnifiedGatewayService(config=cfg)
    mount_seconds = float(time.perf_counter() - t0)

    health = svc.healthz()
    atlas_stats = svc.atlas_stats(namespace="world")["result"]["stats"]
    persist_meta = health.get("atlas_persistence") or {}
    world_atlas = svc._atlases["world"]

    t_q = time.perf_counter()
    matches = 0
    for q, exp_row in zip(queries, expected):
        row_id, _score, _scored, _detail = world_atlas.query_top1(q, namespace="world")
        if int(row_id) == int(exp_row):
            matches += 1
    query_seconds = float(time.perf_counter() - t_q)
    svc.shutdown()

print(json.dumps({
    "cold_mount_seconds": mount_seconds,
    "rebuilt_charts_total": int(persist_meta.get("rebuilt_charts_total", -1)),
    "mounted_namespaces": list(persist_meta.get("loaded_namespaces", [])),
    "loaded_rows": int(atlas_stats.get("rows", 0)),
    "loaded_charts": int(atlas_stats.get("charts", 0)),
    "post_restart_matches": int(matches),
    "post_restart_queries": int(len(expected)),
    "post_restart_parity": float(matches) / float(max(1, len(expected))),
    "post_restart_query_seconds": query_seconds,
}))
"""
    env = dict(os.environ)
    env["HENRI_ROOT"] = str(ROOT)
    env["AG3_PERSIST_DIR"] = str(persist_dir)
    env["AG3_QUERIES_NPY"] = str(queries_path)
    env["AG3_EXPECTED_JSON"] = str(expected_rows_path)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[var] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", probe_code],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=60.0,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"cold gateway probe failed (rc={proc.returncode}):\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    return json.loads(lines[-1])


def run_part_b_100k_persistent_atlas(passes: int = TIMED_PASSES) -> Dict[str, Any]:
    started = time.perf_counter()
    print(
        "\n[ag3.B] Benchmarking N=%d + %d online writes (D=%d, domains=%d, passes=%d, BLAS=1)..."
        % (WORLD, ONLINE, DIM, DOMAINS, passes),
        flush=True,
    )
    rng = np.random.default_rng(20261119)
    centers = _unit(
        rng.standard_normal((DOMAINS, DIM)) + 1j * rng.standard_normal((DOMAINS, DIM))
    )
    world = _sample(rng, centers, WORLD)
    world_real = _as_real_view(world)

    atlas = RiemannianChartAtlas(
        rank=8,
        capacity=448,
        b_stage=128,
        min_cluster=16,
        top_charts=2,
        top_charts_max=4,
        max_rescore_fraction=0.045,
    )
    build_started = time.perf_counter()
    build_stats = atlas.build(world, list(range(WORLD)), namespace="world")
    build_seconds = float(time.perf_counter() - build_started)
    print(
        "[ag3.B] built 100k atlas in %.2fs charts=%d eff_cap=%d boundary_ratio=%.4f"
        % (
            build_seconds,
            int(build_stats["charts"]),
            int(build_stats["effective_capacity"]),
            float(build_stats["boundary_ratio"]),
        ),
        flush=True,
    )

    query_indices = rng.choice(WORLD, size=QUERIES, replace=False)
    queries = np.ascontiguousarray(world[query_indices], dtype=np.complex128)
    queries_real = [q.view(np.float64) for q in queries]

    # Warm-memory warmup pass before timed passes.
    for q, qr in zip(queries[:16], queries_real[:16]):
        _ = int(np.argmax(world_real @ qr))
        _ = atlas.query_top1(q, namespace="world")

    flat_complex_100k_runs: List[float] = []
    flat_real_100k_runs: List[float] = []
    atlas_100k_runs: List[float] = []
    flat_rows_100k: List[int] = []
    routed_rows_100k: List[int] = []
    fractions_100k: List[float] = []

    for pass_idx in range(passes):
        t0 = time.perf_counter()
        f_rows = [int(np.argmax(np.real(world @ np.conj(q)))) for q in queries]
        flat_complex_100k_runs.append(float(time.perf_counter() - t0))

        t0 = time.perf_counter()
        _fr_rows = [int(np.argmax(world_real @ qr)) for qr in queries_real]
        flat_real_100k_runs.append(float(time.perf_counter() - t0))

        t0 = time.perf_counter()
        r_rows: List[int] = []
        r_fracs: List[float] = []
        for q in queries:
            row_id, _score, scored, _detail = atlas.query_top1(q, namespace="world")
            r_rows.append(int(row_id))
            r_fracs.append(float(scored) / float(WORLD))
        atlas_100k_runs.append(float(time.perf_counter() - t0))

        if pass_idx == 0:
            flat_rows_100k = f_rows
            routed_rows_100k = r_rows
            fractions_100k = r_fracs

    parity_100k = float(
        np.mean([int(a == b) for a, b in zip(flat_rows_100k, routed_rows_100k)])
    )

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

    # 5,000 online writes with coherent condensation & bounded Voronoi dual-membership.
    online = _sample(rng, centers, ONLINE)
    observe_started = time.perf_counter()
    for offset, vector in enumerate(online):
        atlas.observe(vector, WORLD + offset, namespace="world")
    atlas.finalize()
    observe_seconds = float(time.perf_counter() - observe_started)

    total_rows = WORLD + ONLINE
    bank = np.ascontiguousarray(np.vstack([world, online]), dtype=np.complex128)
    bank_real = _as_real_view(bank)

    # Warmup on 105k bank.
    for q, qr in zip(queries[:16], queries_real[:16]):
        _ = int(np.argmax(bank_real @ qr))
        _ = atlas.query_top1(q, namespace="world")

    flat_complex_105k_runs: List[float] = []
    flat_real_105k_runs: List[float] = []
    atlas_105k_runs: List[float] = []
    rows_after_flat: List[int] = []
    rows_after_atlas: List[int] = []
    fractions_105k: List[float] = []

    for pass_idx in range(passes):
        t0 = time.perf_counter()
        f105_rows = [int(np.argmax(np.real(bank @ np.conj(q)))) for q in queries]
        flat_complex_105k_runs.append(float(time.perf_counter() - t0))

        t0 = time.perf_counter()
        _fr105 = [int(np.argmax(bank_real @ qr)) for qr in queries_real]
        flat_real_105k_runs.append(float(time.perf_counter() - t0))

        t0 = time.perf_counter()
        a105_rows: List[int] = []
        a105_fracs: List[float] = []
        for q in queries:
            row_id, _score, scored, _detail = atlas.query_top1(q, namespace="world")
            a105_rows.append(int(row_id))
            a105_fracs.append(float(scored) / float(total_rows))
        atlas_105k_runs.append(float(time.perf_counter() - t0))

        if pass_idx == 0:
            rows_after_flat = f105_rows
            rows_after_atlas = a105_rows
            fractions_105k = a105_fracs

    drift_flat = int(sum(1 for a, b in zip(flat_rows_100k, rows_after_flat) if a != b))
    drift_atlas = int(sum(1 for a, b in zip(flat_rows_100k, rows_after_atlas) if a != b))
    parity_105k = float(
        np.mean([int(a == b) for a, b in zip(rows_after_flat, rows_after_atlas)])
    )

    # Online recall over 200 newly condensed online writes.
    recall = 0
    online_fractions: List[float] = []
    for offset in range(QUERIES):
        target = WORLD + offset
        row_id, _score, scored, _detail = atlas.query_top1(online[offset], namespace="world")
        online_fractions.append(float(scored) / float(total_rows))
        if int(row_id) == target:
            recall += 1

    # Binary + SQLite persistence and fresh cold subprocess UnifiedGatewayService mount.
    with tempfile.TemporaryDirectory(prefix="ag3_atlas_persist_") as td:
        persist_root = Path(td) / "persist"
        ns_dir = persist_root / "world"
        save_started = time.perf_counter()
        save_meta = atlas.save(ns_dir)
        save_seconds = float(time.perf_counter() - save_started)

        # Measure full checksum verification load time separately from fast cold mount.
        full_verify_started = time.perf_counter()
        verified_atlas = RiemannianChartAtlas.load(ns_dir, mmap=True, verify_checksum="full")
        full_verify_seconds = float(time.perf_counter() - full_verify_started)
        verified_parity = float(
            np.mean(
                [
                    int(verified_atlas.query_top1(q, namespace="world")[0] == exp)
                    for q, exp in zip(queries[:32], rows_after_atlas[:32])
                ]
            )
        )
        verified_atlas.close()

        queries_path = Path(td) / "queries.npy"
        expected_path = Path(td) / "expected.json"
        np.save(queries_path, queries)
        expected_path.write_text(json.dumps(rows_after_atlas), encoding="utf-8")

        cold_probe = _cold_restart_gateway_probe(persist_root, queries_path, expected_path)

    # Summary statistics (median warm-memory latency across passes).
    med_flat_c_100k = float(np.median(flat_complex_100k_runs))
    med_flat_r_100k = float(np.median(flat_real_100k_runs))
    med_atlas_100k = float(np.median(atlas_100k_runs))
    speedup_100k_vs_complex = float(med_flat_c_100k / max(med_atlas_100k, 1e-9))
    speedup_100k_vs_real = float(med_flat_r_100k / max(med_atlas_100k, 1e-9))

    med_flat_c_105k = float(np.median(flat_complex_105k_runs))
    med_flat_r_105k = float(np.median(flat_real_105k_runs))
    med_atlas_105k = float(np.median(atlas_105k_runs))
    speedup_105k_vs_complex = float(med_flat_c_105k / max(med_atlas_105k, 1e-9))
    speedup_105k_vs_real = float(med_flat_r_105k / max(med_atlas_105k, 1e-9))

    all_105k_fracs = fractions_105k + online_fractions
    frac_105k_mean = float(np.mean(all_105k_fracs))
    frac_105k_max = float(np.max(all_105k_fracs))
    frac_100k_mean = float(np.mean(fractions_100k))
    frac_100k_max = float(np.max(fractions_100k))

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

    # Bandwidth / working-set footprint metrics for AG4 JEV L3-cache scaling proof.
    bytes_per_vec = DIM * 16  # complex128 = 16 bytes/dim = 4096 B/engram
    flat_100k_bytes = int(WORLD * bytes_per_vec)
    flat_105k_bytes = int(total_rows * bytes_per_vec)
    avg_rescore_105k_bytes = int(round(frac_105k_mean * total_rows * bytes_per_vec))
    energy_series = [
        float(med_flat_c_105k * 1e3 / QUERIES),
        float(med_flat_r_105k * 1e3 / QUERIES),
        float(med_atlas_105k * 1e3 / QUERIES),
    ]

    print(
        "[ag3.B] 100k: parity=%.4f frac_mean=%.4f frac_max=%.4f speedup=%.2fx(complex)/%.2fx(real)"
        % (parity_100k, frac_100k_mean, frac_100k_max, speedup_100k_vs_complex, speedup_100k_vs_real),
        flush=True,
    )
    print(
        "[ag3.B] 105k: parity=%.4f drift=%d recall=%d/%d ood=%d/%d frac_mean=%.4f frac_max=%.4f speedup=%.2fx(complex)/%.2fx(real)"
        % (
            parity_105k,
            drift_atlas,
            recall,
            QUERIES,
            hallucinated,
            OOD,
            frac_105k_mean,
            frac_105k_max,
            speedup_105k_vs_complex,
            speedup_105k_vs_real,
        ),
        flush=True,
    )
    bytes_total = (
        int(save_meta["vectors_bytes"])
        + int(save_meta["rows_bytes"])
        + int(save_meta["geom_bytes"])
    )
    print(
        "[ag3.B] persist: save=%.3fs (%.1f MB) cold_mount=%.4fs rebuilt_charts=%d cold_parity=%.4f"
        % (
            save_seconds,
            float(bytes_total) / (1024.0 * 1024.0),
            float(cold_probe["cold_mount_seconds"]),
            int(cold_probe["rebuilt_charts_total"]),
            float(cold_probe["post_restart_parity"]),
        ),
        flush=True,
    )

    return {
        "benchmark_config": {
            "dim": DIM,
            "domains": DOMAINS,
            "world_rows": WORLD,
            "online_writes": ONLINE,
            "total_rows": total_rows,
            "queries": QUERIES,
            "ood_queries": OOD,
            "timed_passes": passes,
            "blas_threads": {
                k: os.environ.get(k, "")
                for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
            },
            "capacity": int(atlas.capacity),
            "effective_capacity": int(build_stats["effective_capacity"]),
            "b_stage": int(atlas.b_stage),
            "top_charts": int(atlas.top_charts),
            "top_charts_max": int(atlas.top_charts_max),
            "max_rescore_fraction": float(atlas.max_rescore_fraction),
        },
        "build": {
            "seconds": build_seconds,
            "stats": build_stats,
        },
        "world_100k": {
            "grounded_top1_parity": parity_100k,
            "rescore_fraction_mean": frac_100k_mean,
            "rescore_fraction_max": frac_100k_max,
            "flat_complex_seconds_runs": flat_complex_100k_runs,
            "flat_real_seconds_runs": flat_real_100k_runs,
            "atlas_seconds_runs": atlas_100k_runs,
            "flat_complex_median_seconds": med_flat_c_100k,
            "flat_real_median_seconds": med_flat_r_100k,
            "atlas_median_seconds": med_atlas_100k,
            "flat_complex_ms_per_query": float(med_flat_c_100k * 1e3 / QUERIES),
            "flat_real_ms_per_query": float(med_flat_r_100k * 1e3 / QUERIES),
            "atlas_ms_per_query": float(med_atlas_100k * 1e3 / QUERIES),
            "speedup_vs_flat_complex_blas": speedup_100k_vs_complex,
            "speedup_vs_flat_real_blas": speedup_100k_vs_real,
        },
        "ood_abstention": {
            "queries": OOD,
            "threshold_tau": tau,
            "indomain_p05": tau,
            "ood_max_score": float(np.max(ood_scores)),
            "ood_mean_score": float(np.mean(ood_scores)),
            "hallucinated": hallucinated,
            "hallucination_rate": float(hallucinated) / float(OOD),
        },
        "online_105k": {
            "observe_and_finalize_seconds": observe_seconds,
            "grounded_top1_parity": parity_105k,
            "post_write_drift_flat": drift_flat,
            "post_write_drift_atlas": drift_atlas,
            "online_recall_hits": recall,
            "online_recall_total": QUERIES,
            "online_recall_rate": float(recall) / float(QUERIES),
            "rescore_fraction_mean": frac_105k_mean,
            "rescore_fraction_max": frac_105k_max,
            "flat_complex_seconds_runs": flat_complex_105k_runs,
            "flat_real_seconds_runs": flat_real_105k_runs,
            "atlas_seconds_runs": atlas_105k_runs,
            "flat_complex_median_seconds": med_flat_c_105k,
            "flat_real_median_seconds": med_flat_r_105k,
            "atlas_median_seconds": med_atlas_105k,
            "flat_complex_ms_per_query": float(med_flat_c_105k * 1e3 / QUERIES),
            "flat_real_ms_per_query": float(med_flat_r_105k * 1e3 / QUERIES),
            "atlas_ms_per_query": float(med_atlas_105k * 1e3 / QUERIES),
            "speedup_vs_flat_complex_blas": speedup_105k_vs_complex,
            "speedup_vs_flat_real_blas": speedup_105k_vs_real,
        },
        "persistence_and_cold_restart": {
            "save_seconds": save_seconds,
            "bytes_total": int(bytes_total),
            "vectors_sha256": str(save_meta["vectors_sha256"]),
            "rows_sha256": str(save_meta["rows_sha256"]),
            "geom_sha256": str(save_meta["geom_sha256"]),
            "full_checksum_load_seconds": full_verify_seconds,
            "full_checksum_sample_parity": verified_parity,
            "cold_gateway_subprocess": cold_probe,
        },
        "bandwidth_scaling": {
            "bytes_per_engram": bytes_per_vec,
            "flat_100k_working_set_bytes": flat_100k_bytes,
            "flat_105k_working_set_bytes": flat_105k_bytes,
            "atlas_105k_mean_rescore_bytes": avg_rescore_105k_bytes,
            "working_set_reduction_ratio": float(flat_105k_bytes)
            / float(max(avg_rescore_105k_bytes, 1)),
        },
        "atlas_stats": stats,
        "energy_series": energy_series,
        "entropy_series": entropy_series,
        "seconds": float(time.perf_counter() - started),
    }


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Cycle AG3 multi-file RepoPatch & 100k persistent atlas")
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

    part_a = run_part_a_multifile_and_quixbugs()
    part_b = run_part_b_100k_persistent_atlas(passes=passes)

    mf = part_a["multifile"]
    qb = part_a["quixbugs_non_regression"]
    w100 = part_b["world_100k"]
    o105 = part_b["online_105k"]
    ood = part_b["ood_abstention"]
    cold = part_b["persistence_and_cold_restart"]["cold_gateway_subprocess"]
    stats = part_b["atlas_stats"]
    pinned_after = _sha(PIN_SNAPSHOT) if PIN_SNAPSHOT.exists() else ""

    gates = {
        # Part A gates
        "ag3_multifile_manifest_digests_valid": bool(not mf["digest_mismatches"]),
        "ag3_multifile_pass_ge_9_of_12": bool(mf["passed_hidden"] >= 9 and mf["total_cases"] == 12),
        "ag3_multifile_precision_ge_095": bool(mf["emitted_precision"] >= 0.9500),
        "ag3_multifile_sheaf_zero_on_all_emitted": bool(mf["all_emitted_sheaf_zero"]),
        "ag3_multifile_traps_4_of_4_refused": bool(
            mf["traps_refused"] == 4 and mf["traps_total"] == 4 and mf["all_traps_sheaf_nonzero"]
        ),
        "ag3_multifile_zero_task_branches_and_llm": bool(
            mf["branch_audit"]["zero_task_name_branches"]
            and not mf["branch_audit"]["forbidden_imports"]
            and mf["llm_calls"] == 0
        ),
        "ag3_quixbugs_non_regression_12_of_14": bool(
            not qb["digest_mismatches"]
            and qb["passed_hidden"] >= 12
            and qb["emitted_precision"] == 1.0
            and qb["traps_refused"] == 4
            and qb["llm_calls"] == 0
        ),
        # Part B gates
        "ag3_atlas_total_rows_105000": bool(
            int(stats["rows"]) == WORLD + ONLINE and int(cold["loaded_rows"]) == WORLD + ONLINE
        ),
        "ag3_atlas_grounded_top1_parity_one": bool(
            w100["grounded_top1_parity"] == 1.0 and o105["grounded_top1_parity"] == 1.0
        ),
        "ag3_atlas_post_write_drift_zero": bool(
            o105["post_write_drift_flat"] == 0 and o105["post_write_drift_atlas"] == 0
        ),
        "ag3_atlas_online_recall_200_of_200": bool(
            o105["online_recall_hits"] == QUERIES
        ),
        "ag3_atlas_ood_hallucination_zero": bool(ood["hallucinated"] == 0),
        "ag3_atlas_rescore_fraction_mean_le_0025": bool(
            w100["rescore_fraction_mean"] <= 0.0250 and o105["rescore_fraction_mean"] <= 0.0250
        ),
        "ag3_atlas_rescore_fraction_max_le_0050": bool(
            w100["rescore_fraction_max"] <= 0.0500 and o105["rescore_fraction_max"] <= 0.0500
        ),
        "ag3_atlas_speedup_vs_complex_ge_15x": bool(
            w100["speedup_vs_flat_complex_blas"] >= 15.0
            and o105["speedup_vs_flat_complex_blas"] >= 15.0
        ),
        "ag3_atlas_speedup_vs_real_ge_12x": bool(
            w100["speedup_vs_flat_real_blas"] >= 12.0
            and o105["speedup_vs_flat_real_blas"] >= 12.0
        ),
        "ag3_atlas_boundary_ratio_le_008": bool(float(stats["boundary_ratio"]) <= 0.0800),
        "ag3_cold_restart_mount_lt_2s": bool(float(cold["cold_mount_seconds"]) < 2.0),
        "ag3_cold_restart_zero_rebuilt_charts": bool(int(cold["rebuilt_charts_total"]) == 0),
        "ag3_cold_restart_parity_one": bool(
            float(cold["post_restart_parity"]) == 1.0
            and float(part_b["persistence_and_cold_restart"]["full_checksum_sample_parity"]) == 1.0
        ),
        "ag3_pinned_corpus_unchanged": bool(
            pinned_before == PIN_SHA and pinned_after == PIN_SHA and pinned_digest == PIN_DIGEST
        ),
    }

    receipt = {
        "schema": "henri.holarch.ag3_multifile_repo_and_100k_atlas.v1",
        "spec_id": "HENRI-SPEC-CYCLE-AG-BLIND-HOLDOUT-RAW-PROSE-SCIENCE-LIVE-CASCADE-100K-ATLAS-AND-BLACKWELL-SCALE-V1",
        "passed": bool(all(gates.values())),
        "gates": gates,
        "part_a_repo_patch": part_a,
        "part_b_100k_persistent_atlas": part_b,
        "pinned_corpus": {
            "sha256_before": pinned_before,
            "sha256_after": pinned_after,
            "digest": pinned_digest,
            "unchanged": bool(pinned_before == PIN_SHA and pinned_after == PIN_SHA),
        },
        "total_seconds": float(time.perf_counter() - started),
    }
    pc._write(directory, "ag3_multifile_repo_and_100k_atlas.json", receipt)

    print("\n=== CYCLE AG3 SUMMARY ===", flush=True)
    print(
        "  Part A Multi-File : %d/%d hidden (prec=%.4f) | traps=%d/%d | QuixBugs=%d/%d (prec=%.4f)"
        % (
            mf["passed_hidden"],
            mf["total_cases"],
            mf["emitted_precision"],
            mf["traps_refused"],
            mf["traps_total"],
            qb["passed_hidden"],
            qb["total_tasks"],
            qb["emitted_precision"],
        ),
        flush=True,
    )
    print(
        "  Part B 100k+5k    : parity=%.4f/%.4f | drift=%d | recall=%d/%d | ood=%d/%d | frac=%.4f(max=%.4f)"
        % (
            w100["grounded_top1_parity"],
            o105["grounded_top1_parity"],
            o105["post_write_drift_atlas"],
            o105["online_recall_hits"],
            QUERIES,
            ood["hallucinated"],
            OOD,
            o105["rescore_fraction_mean"],
            o105["rescore_fraction_max"],
        ),
        flush=True,
    )
    print(
        "  Speedup & Persist : 100k=%.2fx/%.2fx | 105k=%.2fx/%.2fx | cold_mount=%.4fs (rebuilt=%d, parity=%.4f)"
        % (
            w100["speedup_vs_flat_complex_blas"],
            w100["speedup_vs_flat_real_blas"],
            o105["speedup_vs_flat_complex_blas"],
            o105["speedup_vs_flat_real_blas"],
            float(cold["cold_mount_seconds"]),
            int(cold["rebuilt_charts_total"]),
            float(cold["post_restart_parity"]),
        ),
        flush=True,
    )
    print(
        "  Gates passed      : %d/%d -> passed=%s"
        % (sum(1 for v in gates.values() if v), len(gates), receipt["passed"]),
        flush=True,
    )
    if not receipt["passed"]:
        for k, v in gates.items():
            if not v:
                print("  FAILED GATE:", k, flush=True)
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
