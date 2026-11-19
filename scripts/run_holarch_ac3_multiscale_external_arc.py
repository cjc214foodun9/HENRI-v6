"""Cycle AC3: multiscale external ARC-AGI audit (descendant of AB3).

    python scripts/run_holarch_ac3_multiscale_external_arc.py

The 48-task slice is the exact vendored official public ARC-AGI bundle
(MANIFEST.json digest e9b030a233bf7f46).  The inducer sees the demonstration
pairs only; the test pairs are read once, after the chain is fixed, to score
it.  Every admitted program passed the strict Leave-One-Out gate: for every
fold k a setting derived from the other K-1 pairs reproduced pair k exactly.

Cycle AC3 adds four operator families on top of the AB3 object vocabulary:

  A  dimension morphisms   mirror tiling, self-Kronecker fractal tiling,
                           block-mode downscale, rank-predicate crops and
                           edge replication padding;
  B  divider panel algebra mesh template propagation, mesh rank selection,
                           and the OR / AND / XOR / NOR / DIFF / occlusion
                           panel operators with a dihedral action;
  C  topology              corner rays, glyph rays, collinear bridges,
                           cavity fill, unique shortest path, despeckle,
                           contact gravity and ring completion;
  D  signature routing     the Wave invariant orders the candidate families
                           and is recorded per task; it never gates one.

Measured: codec round-trip cell error on all 396 external grids, official
pass@1, emitted test precision, the 16 Z3 spatial tasks (regression), the four
contradictory specifications refused at zero nodes, a branch audit that no
task id enters any decision, and the fact that the vendored manifest and the
pinned corpus snapshot are byte-identical before and after the run.
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
import run_holarch_aa3_external_arc_audit as aa3
import run_holarch_z3_spatial_arc as z3
from henri.holarch.arc_objects import ObjectLooInducer, propose_operators, wave_signature
from henri.holarch.cold_synth import audit_task_specific_branches
from henri.holarch.spatial_induction import (
    SpatialCegisInducer,
    WaveDeltaSpatialRouter,
    ToroidalWaveEncoder,
)

RECEIPTS = ROOT / "receipts" / "holarch"
EXTERNAL = RECEIPTS / "external" / "arc_agi_public"
MANIFEST = EXTERNAL / "MANIFEST.json"
PINNED = RECEIPTS / "pinned_corpus_snapshot.json"
PINNED_SHA256 = "d675f3816bbf472493b25f885bbb56c62c5ec88a81ab31e31a9b87b3a0c23703"
NODE_BUDGET = 400000
TIME_BUDGET = 20.0

AC3_FAMILIES = (
    "mirror_tile",
    "kron_fractal",
    "downscale_blocks",
    "crop_rank",
    "mesh_template",
    "mesh_select_fill",
    "panel_op",
    "corner_ray",
    "glyph_rays",
    "ring_fill",
    "delete_isolated_colour",
    "shortest_path",
    "ray_bridge",
    "cavity_object_fill",
    "gravity_colour",
    "fill_holes_by_size",
    "pad_edges",
    "swatch",
)


def _mark(label: str, started: float) -> None:
    print("[ac3] %-52s %7.1f s" % (label, time.perf_counter() - started), flush=True)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _families(chain: Sequence[str]) -> List[str]:
    out: List[str] = []
    for step in chain:
        name = str(step).split("(")[0]
        if name not in out:
            out.append(name)
    return out


def run_slice(
    inducer: ObjectLooInducer, tasks: Sequence[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for task in tasks:
        started = time.perf_counter()
        train = list(task["train"])
        outcome = inducer.induce(train, list(task["test"]))
        signature = wave_signature(train)
        routed = [op.family for op in propose_operators(train)]
        route_order: List[str] = []
        for name in routed:
            if name not in route_order:
                route_order.append(name)
            if len(route_order) >= 6:
                break
        chain = list(outcome["chain"])
        families = _families(chain)
        fitted = [
            str(op.label())[:80]
            for op in propose_operators(train)
            if ObjectLooInducer._matches(op, train)
        ]
        rows.append(
            {
                "name": str(task["name"]),
                "status": str(outcome["status"]),
                "reason": str(outcome["reason"]),
                "chain": chain,
                "families": families,
                "ac3_families": [name for name in families if name in AC3_FAMILIES],
                "signature": signature,
                "route_order": route_order,
                "fit_candidates": int(len(fitted)),
                "fit_samples": fitted[:4],
                "nodes": int(outcome["nodes"]),
                "folds": int(outcome.get("folds", 0)),
                "loo_passed": bool(outcome.get("loo_passed", False)),
                "test_exact": int(outcome.get("test_exact", 0)),
                "test_total": int(outcome.get("test_total", len(task["test"]))),
                "test_rows": list(outcome.get("test_rows") or []),
                "solved": bool(outcome.get("solved", False)),
                "emitted": bool(outcome.get("emitted", False)),
                "train_pairs": int(len(train)),
                "seconds": float(time.perf_counter() - started),
            }
        )
    return rows


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Cycle AC3 multiscale external ARC audit")
    parser.add_argument("--receipts", default=str(RECEIPTS))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)
    directory = Path(args.receipts).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    manifest_before = _sha256(MANIFEST)
    pinned_before = _sha256(PINNED)

    bundle = aa3.load_slice()
    manifest = bundle["manifest"]
    tasks = bundle["tasks"]
    if args.limit:
        tasks = tasks[: int(args.limit)]
    _mark("slice loaded %d tasks" % len(tasks), started)

    encoder = ToroidalWaveEncoder()
    codec = aa3.codec_audit(encoder, tasks)
    _mark("codec audit (%d grids)" % int(codec["grids"]), started)

    inducer = ObjectLooInducer(
        node_budget=NODE_BUDGET,
        head_cap=16,
        depth=2,
        time_budget=TIME_BUDGET,
    )
    rows = run_slice(inducer, tasks)
    _mark("multiscale LOO induction", started)
    emitted = sum(1 for row in rows if row["emitted"])
    solved = sum(1 for row in rows if row["solved"])
    precision = float(solved / emitted) if emitted else 0.0

    z3_encoder = ToroidalWaveEncoder()
    z3_inducer = SpatialCegisInducer(
        z3_encoder, WaveDeltaSpatialRouter(z3_encoder), node_budget=NODE_BUDGET
    )
    z3_tasks = z3.build_tasks()
    z3_rows = []
    for row in z3_tasks:
        outcome = z3_inducer.induce(row["spec"])
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
        outcome = z3_inducer.induce(row["spec"])
        refused.append(
            {
                "name": str(row["name"]),
                "status": str(outcome["status"]),
                "nodes": int(outcome["nodes"]),
            }
        )
    refused_ok = sum(1 for row in refused if row["status"] == "ABSTAIN" and row["nodes"] == 0)
    _mark("contradictions refused %d/%d" % (refused_ok, len(refused)), started)

    source_path = ROOT / "henri" / "holarch" / "arc_objects.py"
    source_text = source_path.read_text(encoding="utf-8")
    audit = audit_task_specific_branches(
        source_text,
        identifiers=[task["name"] for task in tasks],
    )
    audit["file"] = "henri/holarch/arc_objects.py"
    audit["sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    forbidden_imports = [
        name
        for name in ("openai", "anthropic", "requests", "urllib", "transformers", "tokenizers")
        if ("import " + name) in source_text
    ]

    manifest_after = _sha256(MANIFEST)
    pinned_after = _sha256(PINNED)
    loo_rows = [row for row in rows if row["emitted"]]
    family_hits: Dict[str, List[str]] = {}
    for row in rows:
        if not row["solved"]:
            continue
        for name in row["ac3_families"]:
            family_hits.setdefault(name, []).append(row["name"])

    gates = {
        "external_manifest_unchanged": bool(
            manifest_before == manifest_after
            and "e9b030a2" in str(manifest.get("digest", ""))
            and int(len(tasks)) == 48
        ),
        "codec_cell_errors_zero": bool(int(codec["cell_errors"]) == 0 and int(codec["grids"]) == 396),
        "z3_internal_regression_16_of_16": bool(z3_induced == 16 and len(z3_rows) == 16),
        "ac3_contradictions_4_of_4_at_zero_nodes": bool(refused_ok == 4 and len(refused) == 4),
        "ac3_external_pass_at_1_ge_16_of_48": bool(solved >= 16),
        "ac3_emitted_precision_ge_095": bool(precision >= 0.95),
        "ac3_loo_gate_on_every_emission": bool(
            all(bool(row["loo_passed"]) for row in loo_rows) and bool(loo_rows)
        ),
        "ac3_zero_task_id_branches": bool(
            int(audit.get("task_identity_comparisons", 1) or 0) == 0
            and int(audit.get("compared_identifier_literal_count", 1) or 0) == 0
            and not (audit.get("module_literal_overlap") or [])
        ),
        "ac3_pinned_snapshot_unchanged": bool(
            pinned_before == pinned_after == PINNED_SHA256
        ),
        "ac3_zero_llm_imports": bool(not forbidden_imports),
    }

    ab3_path = RECEIPTS / "ab3_object_centric_external_arc.json"
    ab3 = {}
    if ab3_path.exists():
        try:
            ab3 = json.loads(ab3_path.read_text(encoding="utf-8"))
        except Exception:
            ab3 = {}
    ab3_names = {
        str(row["name"]) for row in (ab3.get("external") or {}).get("rows") or [] if row.get("solved")
    }
    ac3_names = {row["name"] for row in rows if row["solved"]}
    delta = {
        "ab3_solved": int(len(ab3_names)),
        "ac3_solved": int(len(ac3_names)),
        "newly_solved": sorted(ac3_names - ab3_names),
        "newly_lost": sorted(ab3_names - ac3_names),
        "ab3_precision": float((ab3.get("external") or {}).get("precision") or 0.0),
        "ac3_precision": float(precision),
        "ab3_emitted": int((ab3.get("external") or {}).get("emitted") or 0),
        "ac3_emitted": int(emitted),
    }

    receipt = {
        "stage": "AC3",
        "slice": {
            "manifest": str(MANIFEST.relative_to(ROOT)),
            "manifest_sha256": manifest_after,
            "manifest_sha256_before": manifest_before,
            "digest": str(manifest.get("digest", "")),
            "tasks": int(len(tasks)),
            "grids": int(codec["grids"]),
            "source": "official public ARC-AGI bundle vendored by scripts/vendor_arc_agi_public.py",
        },
        "pinned_snapshot": {
            "path": str(PINNED.relative_to(ROOT)),
            "sha256": pinned_after,
            "sha256_before": pinned_before,
            "expected": PINNED_SHA256,
        },
        "codec": codec,
        "external": {
            "tasks": int(len(rows)),
            "emitted": int(emitted),
            "solved": int(solved),
            "pass_at_1": float(solved / max(len(rows), 1)),
            "precision": float(precision),
            "rows": rows,
        },
        "families": {
            "ac3_vocabulary": list(AC3_FAMILIES),
            "fired_on_solved": {name: sorted(names) for name, names in sorted(family_hits.items())},
            "counts": {
                name: int(len(names)) for name, names in sorted(family_hits.items())
            },
        },
        "delta_vs_ab3": delta,
        "z3_regression": {"induced": int(z3_induced), "total": int(len(z3_rows)), "rows": z3_rows},
        "contradictions": {
            "refused_at_zero_nodes": int(refused_ok),
            "total": int(len(refused)),
            "rows": refused,
        },
        "audit": audit,
        "forbidden_imports": forbidden_imports,
        "budgets": {"node_budget": NODE_BUDGET, "time_budget": TIME_BUDGET},
        "gate": pc._finalise(gates),
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
        "passed": bool(all(gates.values())),
    }
    written = pc._write(directory, "ac3_multiscale_external_arc.json", receipt)
    print(
        "[ac3] external %d/%d precision %.4f emitted %d codec errors %d z3 %d/%d refused %d/%d"
        % (
            solved,
            len(rows),
            precision,
            emitted,
            int(codec["cell_errors"]),
            z3_induced,
            len(z3_rows),
            refused_ok,
            len(refused),
        )
    )
    failed = [name for name, value in receipt["gate"].items() if not bool(value)]
    print("[ac3] passed=%s failed=%s receipt=%s" % (receipt["passed"], failed, written["path"]))
    return 0 if bool(receipt["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
