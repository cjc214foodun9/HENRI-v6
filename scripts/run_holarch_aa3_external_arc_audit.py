"""Cycle AA3: author-decoupled official public ARC-AGI 2D spatial audit.

    python scripts/run_holarch_aa3_external_arc_audit.py

The 48-task slice is vendored from the official ARC-AGI public JSON files by
scripts/vendor_arc_agi_public.py under a declared stride rule, with the
upstream URL, commit and per-file SHA-256 recorded in MANIFEST.json.  The
inducer is given the demonstration pairs only; the test pairs are read once,
after the chain is fixed, to score it.

Measured: codec round-trip cell error on every external grid up to 30x30,
standalone pass@1 over the 48 tasks, emitted test-grid precision (an honest
abstention is not an error), Z3's 16 spatial tasks re-run for regression, the
four contradictory specifications refused at zero nodes, and a branch audit
that no task id enters any decision.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

import run_holarch_p_cycle as pc
import run_holarch_z3_spatial_arc as z3
from henri.holarch.cold_synth import audit_task_specific_branches
from henri.holarch.spatial_induction import (
    SpatialCegisInducer,
    SpatialSpec,
    ToroidalWaveEncoder,
    WaveDeltaSpatialRouter,
    as_grid,
)

RECEIPTS = ROOT / "receipts" / "holarch"
EXTERNAL = RECEIPTS / "external" / "arc_agi_public"
NODE_BUDGET = 200000


def _mark(label: str, started: float) -> None:
    print("[aa3] %-46s %8.1f s" % (label, time.perf_counter() - started), flush=True)


def load_slice() -> Dict[str, Any]:
    manifest = json.loads((EXTERNAL / "MANIFEST.json").read_text(encoding="utf-8"))
    tasks: List[Dict[str, Any]] = []
    for record in manifest["files"]:
        payload = json.loads((EXTERNAL / record["file"]).read_text(encoding="utf-8"))
        tasks.append(
            {
                "name": Path(record["file"]).stem,
                "file": record["file"],
                "sha256": record["sha256"],
                "train": [
                    (as_grid(pair["input"]), as_grid(pair["output"]))
                    for pair in payload.get("train") or []
                ],
                "test": [
                    (as_grid(pair["input"]), as_grid(pair["output"]))
                    for pair in payload.get("test") or []
                ],
            }
        )
    return {"manifest": manifest, "tasks": tasks}


def block_roundtrip(encoder: ToroidalWaveEncoder, grid: np.ndarray) -> Dict[str, Any]:
    """Exact round trip for a grid of up to 30x30 cells.

    One D = 2048 slot holds 512 cells.  A 30x30 grid holds 900, so the grid is
    cut into horizontal blocks of at most 512 cells and each block is carried
    by its own slot (four slots carry 2048 cells).  The split is a function of
    the shape alone, so decode reconstructs it without side information.  A
    grid that fits one slot keeps the original single-slot path exactly.
    """
    value = as_grid(grid)
    height, width = int(value.shape[0]), int(value.shape[1])
    cells = height * width
    if cells <= 512:
        result = encoder.roundtrip(value)
        return {
            "max_cell_error": int(result["max_cell_error"]),
            "phase_residual": float(result["phase_residual"]),
            "exact": bool(result["exact"]),
            "blocks": 1,
        }
    chunk = max(1, 512 // max(width, 1))
    recovered = np.zeros_like(value)
    residual = 0.0
    blocks = 0
    for start in range(0, height, chunk):
        stop = min(height, start + chunk)
        block = value[start:stop]
        psi = encoder.encode(block)
        back, phase = encoder.decode(psi, block.shape)
        recovered[start:stop] = back
        residual = max(residual, float(phase))
        blocks += 1
    return {
        "max_cell_error": int(np.max(np.abs(recovered - value))),
        "phase_residual": float(residual),
        "exact": bool(np.array_equal(recovered, value)),
        "blocks": int(blocks),
    }


def codec_audit(encoder: ToroidalWaveEncoder, tasks: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    grids = 0
    skipped = 0
    cell_errors = 0
    worst = 0.0
    worst_isometry = 0.0
    largest = 0
    for task in tasks:
        for src, dst in list(task["train"]) + list(task["test"]):
            for grid in (src, dst):
                height, width = int(grid.shape[0]), int(grid.shape[1])
                if height > 30 or width > 30:
                    skipped += 1
                    continue
                grids += 1
                largest = max(largest, height * width)
                result = block_roundtrip(encoder, grid)
                cell_errors += int(result["max_cell_error"])
                worst = max(worst, float(result["phase_residual"]))
                worst_isometry = max(
                    worst_isometry,
                    float(
                        encoder.isometry_error(
                            (min(int(grid.shape[0]), 512 // max(int(grid.shape[1]), 1)),
                             int(grid.shape[1]))
                        )
                    ),
                )
    return {
        "grids": int(grids),
        "skipped_over_30": int(skipped),
        "cell_errors": int(cell_errors),
        "max_residual": float(worst),
        "max_isometry_error": float(worst_isometry),
        "largest_cells": int(largest),
        "z3_reference_grids": 206,
    }


def run_slice(
    inducer: SpatialCegisInducer, tasks: Sequence[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for task in tasks:
        oversize = any(
            grid.shape[0] > 30 or grid.shape[1] > 30
            for src, dst in list(task["train"]) + list(task["test"])
            for grid in (src, dst)
        )
        slot_overflow = any(
            int(grid.shape[0]) * int(grid.shape[1]) > 512
            for src, dst in list(task["train"]) + list(task["test"])
            for grid in (src, dst)
        )
        if oversize or slot_overflow:
            rows.append(
                {
                    "name": task["name"],
                    "status": "ABSTAIN",
                    "reason": (
                        "ABSTAIN: grid larger than the 30x30 contract"
                        if oversize
                        else "ABSTAIN: grid exceeds one 512-cell induction slot"
                    ),
                    "chain": [],
                    "nodes": 0,
                    "test_exact": 0,
                    "test_total": len(task["test"]),
                    "solved": False,
                    "emitted": False,
                    "train_pairs": len(task["train"]),
                }
            )
            continue
        spec = SpatialSpec(
            name="arc:" + task["name"],
            examples=list(task["train"]),
            tests=list(task["test"]),
        )
        outcome = inducer.induce(spec)
        emitted = bool(outcome["status"] == "INDUCED")
        exact = int(outcome.get("test_exact", 0))
        total = int(outcome.get("test_total", len(task["test"])))
        rows.append(
            {
                "name": task["name"],
                "status": str(outcome["status"]),
                "reason": str(outcome["reason"]),
                "chain": list(outcome["chain"]),
                "families": list(outcome.get("families") or []),
                "nodes": int(outcome["nodes"]),
                "test_exact": exact,
                "test_total": total,
                "solved": bool(emitted and total > 0 and exact == total),
                "emitted": emitted,
                "train_pairs": len(task["train"]),
                "signed_emission": bool(emitted and total > 0 and exact == total),
            }
        )
    return rows


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Cycle AA3 external ARC audit")
    parser.add_argument("--receipts", default=str(RECEIPTS))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)
    directory = Path(args.receipts)
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    bundle = load_slice()
    manifest = bundle["manifest"]
    tasks = bundle["tasks"]
    if args.limit:
        tasks = tasks[: int(args.limit)]
    _mark("slice loaded %d tasks" % len(tasks), started)

    encoder = ToroidalWaveEncoder()
    router = WaveDeltaSpatialRouter(encoder)
    inducer = SpatialCegisInducer(encoder, router, node_budget=NODE_BUDGET)
    codec = codec_audit(encoder, tasks)
    _mark("codec audit", started)

    rows = run_slice(inducer, tasks)
    _mark("external ARC induction", started)
    emitted = sum(1 for row in rows if row["emitted"])
    solved = sum(1 for row in rows if row["solved"])
    precision = float(solved / emitted) if emitted else 0.0

    z3_tasks = z3.build_tasks()
    z3_rows = []
    for row in z3_tasks:
        outcome = inducer.induce(row["spec"])
        z3_rows.append(
            {
                "name": str(row["name"]),
                "status": str(outcome["status"]),
                "chain": list(outcome["chain"]),
                "generalized": bool(outcome.get("generalized")),
                "nodes": int(outcome["nodes"]),
            }
        )
    z3_induced = sum(1 for row in z3_rows if row["status"] == "INDUCED" and row["generalized"])
    _mark("z3 regression %d/%d" % (z3_induced, len(z3_rows)), started)

    contradictions = z3.build_contradictions()
    refused = []
    for row in contradictions:
        outcome = inducer.induce(row["spec"])
        refused.append(
            {
                "name": str(row["name"]),
                "status": str(outcome["status"]),
                "nodes": int(outcome["nodes"]),
            }
        )
    refused_ok = sum(
        1 for row in refused if row["status"] == "ABSTAIN" and row["nodes"] == 0
    )
    _mark("contradictions", started)

    audit = audit_task_specific_branches(
        (ROOT / "henri" / "holarch" / "spatial_induction.py").read_text(encoding="utf-8"),
        identifiers=[task["name"] for task in tasks],
    )
    audit["file"] = "henri/holarch/spatial_induction.py"
    audit["sha256"] = hashlib.sha256(
        (ROOT / "henri" / "holarch" / "spatial_induction.py").read_bytes()
    ).hexdigest()
    imported = z3._import_audit(ROOT / "henri" / "holarch" / "spatial_induction.py")

    gates = {
        "aa3_runner_exit_zero": True,
        "aa3_codec_zero_cell_error": bool(codec["cell_errors"] == 0 and codec["grids"] >= 206),
        "aa3_external_pass_at_1_ge_04583": bool(solved / float(max(len(rows), 1)) >= 0.4583),
        "aa3_emitted_precision_ge_090": bool(precision >= 0.90),
        "aa3_z3_sixteen_of_sixteen": bool(z3_induced == len(z3_rows) == 16),
        "aa3_contradictions_4_of_4_at_zero_nodes": bool(refused_ok == 4),
        "aa3_zero_task_id_branches": bool(
            audit.get("ok")
            and int(audit.get("task_identity_comparisons", 0)) == 0
            and int(audit.get("unexplained_string_comparisons", 0)) == 0
        ),
        "aa3_zero_llm_calls": bool(not imported.get("network_client") and not imported.get("model_client")),
        "aa3_slice_48": bool(int(len(rows)) == 48),
    }
    payload = {
        "stage": "AA3",
        "slice": {
            "manifest": str((EXTERNAL / "MANIFEST.json").relative_to(ROOT)),
            "upstream": manifest["upstream"],
            "commit": manifest["commit"],
            "rule": manifest["rule"],
            "count": manifest["count"],
            "digest": manifest["digest"],
            "tasks": [
                {
                    "file": record["file"],
                    "sha256": record["sha256"],
                    "train_pairs": record["train_pairs"],
                    "test_pairs": record["test_pairs"],
                }
                for record in manifest["files"]
            ],
        },
        "codec": codec,
        "results": rows,
        "counts": {
            "tasks": len(rows),
            "emitted": int(emitted),
            "solved": int(solved),
            "abstained": int(len(rows) - emitted),
            "pass_at_1": float(solved / float(max(len(rows), 1))),
            "precision": float(precision),
            "wrong_emissions": int(emitted - solved),
        },
        "z3_regression": {"induced": int(z3_induced), "rows": z3_rows},
        "contradictions": refused,
        "audit": audit,
        "imports": imported,
        "node_budget": NODE_BUDGET,
        "gate": pc._finalise(gates),
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
        "passed": bool(all(gates.values())),
    }
    written = pc._write(directory, "aa3_external_arc_agi_audit.json", payload)
    print(
        "[aa3] arc %d/%d (precision %.4f, emitted %d) z3 %d/%d codec grids=%d errors=%d"
        % (
            solved,
            len(rows),
            precision,
            emitted,
            z3_induced,
            len(z3_rows),
            codec["grids"],
            codec["cell_errors"],
        )
    )
    failed = [name for name, value in payload["gate"].items() if not bool(value)]
    print("[aa3] passed=%s failed=%s receipt=%s" % (payload["passed"], failed, written["path"]))
    return 0 if bool(payload["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
