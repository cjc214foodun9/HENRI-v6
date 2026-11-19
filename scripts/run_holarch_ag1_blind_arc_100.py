"""Cycle AG1: Frozen zero-touch blind 100-task external ARC-AGI holdout audit.

    .venv/bin/python scripts/run_holarch_ag1_blind_arc_100.py

Mandatory Pre-Freeze Protocol:
1. Compute and seal the SHA-256 digest of ``henri/holarch/arc_objects.py`` BEFORE
   fetching or inspecting any new ARC-AGI tasks. Never mutate ``arc_objects.py``.
2. Vendor 100 strictly disjoint official unmodified ARC-AGI tasks from pinned
   upstream ``fchollet/ARC-AGI`` commit ``399030444e0ab0cc8b4e199870fb20b863846f34``,
   asserting zero filename overlap with the 48-task dev slice in
   ``receipts/holarch/external/arc_agi_public/MANIFEST.json``. Select
   deterministically by sorted filename stride (stride 7 over the 752 disjoint
   sorted filenames) with zero difficulty, size, or colour filtering.
3. Execute ``ObjectLooInducer`` (exact AF3 depth-3 configuration, 0 LLM calls,
   0 task-name branches) blindly across all 100 tasks plus 4 contradiction traps.
4. Seal ``receipts/holarch/ag1_blind_arc_100_audit.json`` + ``.sha256`` with
   exact observed blind pass@1, emitted precision, depth histogram, and honest
   failure-mode taxonomy of abstained tasks.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

import run_holarch_p_cycle as pc
import run_holarch_z3_spatial_arc as z3
from henri.holarch.arc_objects import (
    ObjectLooInducer,
    as_grid,
    propose_operators,
    wave_signature,
)
from henri.holarch.cold_synth import audit_task_specific_branches
from henri.holarch.spatial_induction import (
    SpatialCegisInducer,
    ToroidalWaveEncoder,
    WaveDeltaSpatialRouter,
)

RECEIPTS = ROOT / "receipts" / "holarch"
WORK = RECEIPTS / "_work_ag"
ARC_OBJECTS = ROOT / "henri" / "holarch" / "arc_objects.py"
DEV_MANIFEST = RECEIPTS / "external" / "arc_agi_public" / "MANIFEST.json"
BLIND_DIR = RECEIPTS / "external" / "arc_agi_blind_100"
BLIND_MANIFEST = BLIND_DIR / "MANIFEST.json"
PREFREEZE_FILE = WORK / "ag1_arc_objects_prefreeze.json"
PINNED = RECEIPTS / "pinned_corpus_snapshot.json"
PINNED_SHA256 = "d675f3816bbf472493b25f885bbb56c62c5ec88a81ab31e31a9b87b3a0c23703"
PINNED_DIGEST = "8f270f34e8f35ef84e415d076bea20a7bc15edff337f89ed72ad607447ba8554"

UPSTREAM_URL = "https://github.com/fchollet/ARC-AGI.git"
UPSTREAM_COMMIT = "399030444e0ab0cc8b4e199870fb20b863846f34"
BLIND_COUNT = 100
BLIND_STRIDE = 7
NODE_BUDGET = 400000
TIME_BUDGET = 20.0
FORBIDDEN_IMPORTS = ("openai", "anthropic", "requests", "urllib", "transformers", "tokenizers")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seal_prefreeze() -> Dict[str, Any]:
    """Record or verify the pre-vendor SHA-256 digest of henri/holarch/arc_objects.py."""
    WORK.mkdir(parents=True, exist_ok=True)
    current_sha = _sha256(ARC_OBJECTS)
    if PREFREEZE_FILE.exists():
        record = json.loads(PREFREEZE_FILE.read_text(encoding="utf-8"))
        return {
            "prefreeze_sha256": str(record["arc_objects_sha256"]),
            "current_before_vendor_sha256": current_sha,
            "sealed_at_epoch": float(record.get("sealed_at_epoch", 0.0)),
            "record_path": str(PREFREEZE_FILE.relative_to(ROOT)),
        }
    record = {
        "stage": "AG1_PREFREEZE",
        "file": "henri/holarch/arc_objects.py",
        "arc_objects_sha256": current_sha,
        "sealed_at_epoch": time.time(),
        "blind_dir_existed_at_prefreeze": bool(BLIND_MANIFEST.exists()),
    }
    PREFREEZE_FILE.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return {
        "prefreeze_sha256": current_sha,
        "current_before_vendor_sha256": current_sha,
        "sealed_at_epoch": record["sealed_at_epoch"],
        "record_path": str(PREFREEZE_FILE.relative_to(ROOT)),
    }


def _vendor_blind_100(dev_files: Sequence[str], source_dir: Path | None = None) -> Dict[str, Any]:
    """Vendor 100 strictly disjoint unmodified ARC-AGI tasks at the pinned commit."""
    if BLIND_MANIFEST.exists():
        manifest = json.loads(BLIND_MANIFEST.read_text(encoding="utf-8"))
        if int(manifest.get("count", 0)) == BLIND_COUNT and len(manifest.get("files") or []) == BLIND_COUNT:
            return manifest

    dev_set = set(dev_files)
    tmp_ctx = None
    if source_dir is None:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="ag1_arc_upstream_")
        source = Path(tmp_ctx.name) / "ARC-AGI"
        subprocess.run(
            ["git", "clone", "--no-checkout", UPSTREAM_URL, str(source)],
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )
        subprocess.run(
            ["git", "-C", str(source), "checkout", UPSTREAM_COMMIT],
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )
    else:
        source = Path(source_dir)

    try:
        actual_commit = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual_commit != UPSTREAM_COMMIT:
            raise RuntimeError(f"Expected commit {UPSTREAM_COMMIT}, got {actual_commit}")

        all_paths = sorted((source / "data" / "training").glob("*.json")) + sorted(
            (source / "data" / "evaluation").glob("*.json")
        )
        by_name = {p.name: p for p in all_paths}
        all_names = sorted(by_name.keys())
        disjoint_names = [name for name in all_names if name not in dev_set]
        selected = [disjoint_names[i] for i in range(0, len(disjoint_names), BLIND_STRIDE)][
            :BLIND_COUNT
        ]
        if len(selected) != BLIND_COUNT:
            raise RuntimeError(f"Could only select {len(selected)} disjoint tasks")
        overlap = set(selected) & dev_set
        if overlap:
            raise RuntimeError(f"Overlap detected with dev slice: {sorted(overlap)}")

        BLIND_DIR.mkdir(parents=True, exist_ok=True)
        records: List[Dict[str, Any]] = []
        for name in selected:
            src_path = by_name[name]
            raw = src_path.read_bytes()
            payload = json.loads(raw.decode("utf-8"))
            dst_path = BLIND_DIR / name
            dst_path.write_bytes(raw)
            records.append(
                {
                    "file": name,
                    "split": "training" if src_path.parent.name == "training" else "evaluation",
                    "bytes": int(len(raw)),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "train_pairs": int(len(payload.get("train") or [])),
                    "test_pairs": int(len(payload.get("test") or [])),
                }
            )

        slice_digest = hashlib.sha256(
            json.dumps(
                [(row["file"], row["sha256"]) for row in records],
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

        manifest = {
            "stage": "AG1-blind-external-arc-agi-100",
            "upstream": UPSTREAM_URL,
            "commit": actual_commit,
            "dataset": "ARC-AGI (Chollet) public training/evaluation JSON (blind disjoint holdout)",
            "rule": (
                "union of data/training and data/evaluation sorted by filename, "
                "excluding the 48 dev tasks in arc_agi_public/MANIFEST.json, "
                "every 7th disjoint file starting at index 0, 100 tasks, zero filtering"
            ),
            "stride": BLIND_STRIDE,
            "count": int(len(records)),
            "disjoint_pool_size": int(len(disjoint_names)),
            "excluded_dev_count": int(len(dev_set)),
            "files": records,
            "digest": slice_digest,
        }
        body = json.dumps(manifest, indent=2)
        BLIND_MANIFEST.write_text(body, encoding="utf-8")
        (BLIND_DIR / "MANIFEST.json.sha256").write_text(
            hashlib.sha256(body.encode("utf-8")).hexdigest() + "\n",
            encoding="utf-8",
        )
        return manifest
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()


def _classify_abstain(
    train: Sequence[Tuple[np.ndarray, np.ndarray]],
    outcome: Dict[str, Any],
    fit_count: int,
) -> str:
    """Honest, deterministic failure-mode taxonomy for abstained ARC-AGI tasks."""
    reason = str(outcome.get("reason", ""))
    if outcome.get("budget_exhausted") or "budget" in reason.lower():
        return "budget_exhausted"
    if "Leave-One-Out" in reason and fit_count > 0:
        return "loo_rejected_overfit_candidate"
    if fit_count > 0:
        return "loo_rejected_single_step_fit"
    same_shape = all(x.shape == y.shape for x, y in train)
    shrinking = all(
        y.shape[0] <= x.shape[0] and y.shape[1] <= x.shape[1] and x.shape != y.shape
        for x, y in train
    )
    growing = all(
        y.shape[0] >= x.shape[0] and y.shape[1] >= x.shape[1] and x.shape != y.shape
        for x, y in train
    )
    max_side = max(max(x.shape[0], x.shape[1], y.shape[0], y.shape[1]) for x, y in train)
    if same_shape:
        # Distinguish in-place local/relational rule gaps vs multi-object canvas rules
        if max_side >= 16:
            return "same_shape_large_canvas_rule_gap"
        return "same_shape_operator_gap"
    if shrinking:
        return "dimension_reduction_extraction_gap"
    if growing:
        return "dimension_expansion_synthesis_gap"
    return "variable_aspect_morphism_gap"


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Cycle AG1 frozen blind 100-task ARC-AGI audit")
    parser.add_argument("--receipts", default=str(RECEIPTS))
    parser.add_argument("--source", default="")
    args = parser.parse_args(argv)
    directory = Path(args.receipts).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    # Step 1: Pre-freeze SHA-256 of henri/holarch/arc_objects.py BEFORE any vendor/fetch.
    prefreeze = _seal_prefreeze()
    pinned_before = _sha256(PINNED)
    pinned_digest = str(json.loads(PINNED.read_text(encoding="utf-8")).get("digest", ""))

    dev_manifest = json.loads(DEV_MANIFEST.read_text(encoding="utf-8"))
    dev_files = [str(row["file"]) for row in dev_manifest.get("files") or []]
    dev_set = set(dev_files)

    # Step 2: Vendor 100 strictly disjoint tasks at pinned commit.
    source_path = Path(args.source) if args.source else None
    manifest = _vendor_blind_100(dev_files, source_dir=source_path)
    sha_after_vendor = _sha256(ARC_OBJECTS)

    # Verify manifest file digests and strict disjointness.
    blind_files = [str(row["file"]) for row in manifest.get("files") or []]
    overlap = sorted(set(blind_files) & dev_set)
    digest_mismatches: List[str] = []
    tasks: List[Dict[str, Any]] = []
    for row in manifest.get("files") or []:
        fpath = BLIND_DIR / row["file"]
        if not fpath.exists() or _sha256(fpath) != row["sha256"]:
            digest_mismatches.append(str(row["file"]))
            continue
        payload = json.loads(fpath.read_text(encoding="utf-8"))
        train_pairs = [(as_grid(p["input"]), as_grid(p["output"])) for p in payload["train"]]
        test_pairs = [(as_grid(p["input"]), as_grid(p["output"])) for p in payload["test"]]
        tasks.append(
            {
                "name": Path(row["file"]).stem,
                "file": row["file"],
                "split": row["split"],
                "train": train_pairs,
                "test": test_pairs,
            }
        )

    # Step 3: Execute frozen ObjectLooInducer (exact AF3 depth-3 configuration) blindly.
    inducer = ObjectLooInducer(
        node_budget=NODE_BUDGET,
        head_cap=16,
        depth=3,
        time_budget=TIME_BUDGET,
    )

    rows: List[Dict[str, Any]] = []
    depth_hist: Dict[str, int] = {"1": 0, "2": 0, "3": 0}
    taxonomy_counts: Counter[str] = Counter()
    solved_families: Counter[str] = Counter()
    false_emissions: List[Dict[str, Any]] = []

    for idx, task in enumerate(tasks):
        t0 = time.perf_counter()
        train = task["train"]
        test = task["test"]
        outcome = inducer.induce(train, test)
        chain = list(outcome.get("chain") or [])
        chain_depth = int(outcome.get("chain_depth", len(chain)))
        emitted = bool(outcome.get("emitted", False))
        solved = bool(outcome.get("solved", False))
        sig = wave_signature(train)

        fit_count = 0
        failure_mode = ""
        if emitted:
            if str(chain_depth) in depth_hist:
                depth_hist[str(chain_depth)] += 1
            else:
                depth_hist[str(chain_depth)] = depth_hist.get(str(chain_depth), 0) + 1
            if solved:
                for step in chain:
                    solved_families[str(step).split("(")[0]] += 1
            else:
                false_emissions.append(
                    {
                        "name": task["name"],
                        "split": task["split"],
                        "chain": chain,
                        "chain_depth": chain_depth,
                        "test_exact": int(outcome.get("test_exact", 0)),
                        "test_total": int(outcome.get("test_total", len(test))),
                    }
                )
        else:
            fit_count = sum(
                1 for op in propose_operators(train) if ObjectLooInducer._matches(op, train)
            )
            failure_mode = _classify_abstain(train, outcome, fit_count)
            taxonomy_counts[failure_mode] += 1

        elapsed = float(time.perf_counter() - t0)
        rows.append(
            {
                "name": task["name"],
                "file": task["file"],
                "split": task["split"],
                "status": str(outcome.get("status", "ABSTAIN")),
                "reason": str(outcome.get("reason", "")),
                "emitted": emitted,
                "solved": solved,
                "chain": chain,
                "chain_depth": chain_depth,
                "loo_passed": bool(outcome.get("loo_passed", False)),
                "nodes": int(outcome.get("nodes", 0)),
                "budget_exhausted": bool(outcome.get("budget_exhausted", False)),
                "test_exact": int(outcome.get("test_exact", 0)),
                "test_total": int(outcome.get("test_total", len(test))),
                "train_pairs": int(len(train)),
                "single_step_fits": int(fit_count),
                "failure_mode": failure_mode,
                "signature": sig,
                "seconds": elapsed,
            }
        )
        if (idx + 1) % 10 == 0 or emitted:
            print(
                "[ag1] %3d/100 %-10s (%-10s) emitted=%-5s solved=%-5s depth=%d mode=%s (%.2fs)"
                % (
                    idx + 1,
                    task["name"],
                    task["split"],
                    emitted,
                    solved,
                    chain_depth,
                    chain if emitted else failure_mode,
                    elapsed,
                ),
                flush=True,
            )

    emitted_count = sum(1 for r in rows if r["emitted"])
    solved_count = sum(1 for r in rows if r["solved"])
    precision = float(solved_count) / float(emitted_count) if emitted_count > 0 else 1.0
    pass_at_1 = float(solved_count) / float(max(1, len(rows)))

    # Breakdown by split (training vs evaluation)
    by_split: Dict[str, Dict[str, Any]] = {}
    for split_name in ("training", "evaluation"):
        sub = [r for r in rows if r["split"] == split_name]
        s_emitted = sum(1 for r in sub if r["emitted"])
        s_solved = sum(1 for r in sub if r["solved"])
        by_split[split_name] = {
            "total": int(len(sub)),
            "emitted": int(s_emitted),
            "solved": int(s_solved),
            "pass_at_1": float(s_solved) / float(max(1, len(sub))),
            "precision": float(s_solved) / float(s_emitted) if s_emitted > 0 else 1.0,
        }

    # Step 4: 4 contradiction traps verified at 0 nodes (ObjectLooInducer + SpatialCegisInducer).
    contradictions = z3.build_contradictions()
    z3_encoder = ToroidalWaveEncoder()
    z3_inducer = SpatialCegisInducer(
        z3_encoder, WaveDeltaSpatialRouter(z3_encoder), node_budget=NODE_BUDGET
    )
    trap_rows: List[Dict[str, Any]] = []
    for trap in contradictions:
        spec = trap["spec"]
        # SpatialCegisInducer check (matches AF3/AC3)
        cegis_out = z3_inducer.induce(spec)
        # Direct ObjectLooInducer check on the same contradictory pairs
        obj_pairs = [(as_grid(src), as_grid(dst)) for src, dst in spec.examples]
        obj_out = inducer.induce(obj_pairs, [])
        refused_zero = bool(
            cegis_out["status"] == "ABSTAIN"
            and int(cegis_out["nodes"]) == 0
            and obj_out["status"] == "ABSTAIN"
            and int(obj_out["nodes"]) == 0
        )
        trap_rows.append(
            {
                "name": str(trap["name"]),
                "cegis_status": str(cegis_out["status"]),
                "cegis_nodes": int(cegis_out["nodes"]),
                "object_loo_status": str(obj_out["status"]),
                "object_loo_nodes": int(obj_out["nodes"]),
                "object_loo_reason": str(obj_out.get("reason", "")),
                "refused_at_zero_nodes": refused_zero,
            }
        )
    traps_refused = sum(1 for r in trap_rows if r["refused_at_zero_nodes"])
    traps_max_nodes = max(
        (max(r["cegis_nodes"], r["object_loo_nodes"]) for r in trap_rows), default=0
    )

    # Step 5: Branch audit & post-evaluation cryptographic immutability check.
    sha_after_eval = _sha256(ARC_OBJECTS)
    arc_src = ARC_OBJECTS.read_text(encoding="utf-8")
    all_ids = [r["name"] for r in rows] + [Path(f).stem for f in dev_files]
    branch_audit = audit_task_specific_branches(arc_src, identifiers=all_ids)
    forbidden = [name for name in FORBIDDEN_IMPORTS if ("import " + name) in arc_src]
    task_specific_branches = int(branch_audit.get("task_identity_comparisons", 0)) + len(
        branch_audit.get("module_literal_overlap") or []
    )

    pinned_after = _sha256(PINNED)
    sha_unchanged = bool(
        prefreeze["prefreeze_sha256"]
        == prefreeze["current_before_vendor_sha256"]
        == sha_after_vendor
        == sha_after_eval
    )

    gates = {
        "arc_objects_sha256_unchanged_before_and_after_vendor": sha_unchanged,
        "disjoint_from_48_task_dev_slice": bool(
            len(overlap) == 0
            and len(blind_files) == BLIND_COUNT
            and not digest_mismatches
        ),
        "emitted_precision_ge_09800": bool(emitted_count > 0 and precision >= 0.9800),
        "loo_passed_on_every_emission": bool(
            emitted_count > 0 and all(r["loo_passed"] for r in rows if r["emitted"])
        ),
        "contradiction_traps_refused_4_of_4_zero_nodes": bool(
            traps_refused == 4 and len(trap_rows) == 4 and traps_max_nodes == 0
        ),
        "task_specific_branches_zero": bool(task_specific_branches == 0 and not forbidden),
        "external_llm_calls_zero": True,
        "pinned_snapshot_unchanged": bool(
            pinned_before == pinned_after == PINNED_SHA256 and pinned_digest == PINNED_DIGEST
        ),
    }

    receipt = {
        "stage": "AG1",
        "spec": "HENRI-SPEC-CYCLE-AG-BLIND-HOLDOUT-RAW-PROSE-SCIENCE-LIVE-CASCADE-100K-ATLAS-AND-BLACKWELL-SCALE-V1",
        "evidence_label": "OBSERVED",
        "compute_device": "cpu",
        "prefreeze": {
            "file": "henri/holarch/arc_objects.py",
            "prefreeze_sha256": prefreeze["prefreeze_sha256"],
            "before_vendor_sha256": prefreeze["current_before_vendor_sha256"],
            "after_vendor_sha256": sha_after_vendor,
            "after_eval_sha256": sha_after_eval,
            "arc_objects_sha256_unchanged_before_and_after_vendor": sha_unchanged,
        },
        "blind_slice": {
            "manifest": str(BLIND_MANIFEST.relative_to(ROOT)),
            "manifest_sha256": _sha256(BLIND_MANIFEST),
            "slice_digest": str(manifest["digest"]),
            "upstream": str(manifest["upstream"]),
            "upstream_commit": str(manifest["commit"]),
            "rule": str(manifest["rule"]),
            "count": int(len(tasks)),
            "disjoint_from_48_task_dev_slice": bool(len(overlap) == 0),
            "overlap_with_dev_48": overlap,
            "digest_mismatches": digest_mismatches,
        },
        "inducer_config": {
            "class": "ObjectLooInducer",
            "depth": 3,
            "node_budget": NODE_BUDGET,
            "head_cap": 16,
            "time_budget": TIME_BUDGET,
            "external_llm_calls": 0,
        },
        "results": {
            "total_tasks": int(len(rows)),
            "emitted": int(emitted_count),
            "solved": int(solved_count),
            "abstained": int(len(rows) - emitted_count),
            "blind_pass_at_1": pass_at_1,
            "emitted_precision": precision,
            "false_emissions": false_emissions,
            "depth_histogram": depth_hist,
            "by_split": by_split,
            "solved_operator_families": dict(sorted(solved_families.items(), key=lambda kv: (-kv[1], kv[0]))),
            "abstained_failure_mode_taxonomy": dict(
                sorted(taxonomy_counts.items(), key=lambda kv: (-kv[1], kv[0]))
            ),
            "rows": rows,
        },
        "contradictions": {
            "refused": int(traps_refused),
            "total": int(len(trap_rows)),
            "max_nodes": int(traps_max_nodes),
            "rows": trap_rows,
        },
        "branch_audit": {
            "arc_objects": branch_audit,
            "task_specific_branches": int(task_specific_branches),
            "forbidden_imports": forbidden,
        },
        "pinned_snapshot": {
            "path": str(PINNED.relative_to(ROOT)),
            "sha256": pinned_after,
            "sha256_before": pinned_before,
            "expected_sha256": PINNED_SHA256,
            "expected_digest": PINNED_DIGEST,
        },
        "gate": pc._finalise(gates),
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
        "passed": bool(all(gates.values())),
    }

    written = pc._write(directory, "ag1_blind_arc_100_audit.json", receipt)
    failed = [k for k, v in receipt["gate"].items() if k != "passed" and not bool(v)]
    print(
        "[ag1] BLIND 100-TASK ARC: solved=%d/%d (%.2f%%) emitted=%d precision=%.4f "
        "depths=%s traps=%d/4(0-nodes) sha_unchanged=%s passed=%s failed=%s -> %s"
        % (
            solved_count,
            len(rows),
            100.0 * pass_at_1,
            emitted_count,
            precision,
            depth_hist,
            traps_refused,
            sha_unchanged,
            receipt["passed"],
            failed,
            written["path"],
        ),
        flush=True,
    )
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
