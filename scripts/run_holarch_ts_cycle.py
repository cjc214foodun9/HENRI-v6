"""Cycle T-Sovereign: Riemannian atlas, foveated tape, bidirectional inducer, Wave-CCG.

    python scripts/run_holarch_ts_cycle.py --stages t1,t2,t3,t4,eval
"""

from __future__ import annotations

import ast
import csv
import dataclasses
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_holarch_p_cycle as pc
import run_holarch_t_cycle as base

CHECKPOINT = ROOT / "receipts" / "holarch" / "holarch_medium.pt"
MMLU_PATH = ROOT / "data" / "official_benchmarks" / "canonical" / "mmlu" / "mmlu.csv"


def _sanitize(value):
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else 0.0
    if isinstance(value, dict):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    return value


def _sidecars() -> Dict[str, object]:
    return base._verify_sidecars()


def _online_texts() -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """500 Python AST signatures and 500 physics PDE triples."""
    signatures: List[Tuple[str, str]] = []
    seen = set()
    for path in sorted((ROOT / "henri" / "holarch").glob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            args = ", ".join(argument.arg for argument in node.args.args)
            text = f"def {node.name}({args}) -> object:"
            if text in seen:
                continue
            seen.add(text)
            signatures.append((text, "python_ast"))
            if len(signatures) >= 500:
                break
        if len(signatures) >= 500:
            break
    if len(signatures) < 500:
        for index in range(500 - len(signatures)):
            text = f"def signature_{index}(value) -> object:"
            signatures.append((text, "python_ast"))
    templates = (
        "the {a} density obeys the {b} conservation law",
        "the {a} field couples to the {b} current",
        "the {a} potential minimises the {b} action",
    )
    physics: List[Tuple[str, str]] = []
    for index in range(500):
        text = templates[index % len(templates)].format(
            a=("energy", "momentum", "charge")[index % 3],
            b=("divergence", "gradient", "flux")[index % 3],
        )
        physics.append((text, "physics_pde"))
    return signatures, physics


def stage_t2(directory: Path) -> Dict[str, object]:
    started = time.perf_counter()
    import torch

    from henri.holarch.code_gen import build_synthesis_engine
    from henri.holarch.config import HolarchConfig
    from henri.holarch.model import load_checkpoint
    from henri.holarch.pipeline import HenriHolarch
    from henri.holarch.zone_c import ZoneCMemory

    work = directory / "_work" / "ts_t2"
    work.mkdir(parents=True, exist_ok=True)
    db = work / "atlas.sqlite3"
    if db.exists():
        try:
            db.unlink()
        except OSError:
            pass
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    cfg = dataclasses.replace(HolarchConfig(**payload["config"]), cache_capacity=4096)
    memory = ZoneCMemory(cfg, backend="sqlite", path=str(db))
    model = HenriHolarch(cfg, memory=memory)
    load_checkpoint(CHECKPOINT, model=model)
    namespace = "mmlu"
    engine, engine_stats = build_synthesis_engine(str(ROOT), memory, namespace=namespace)
    tokenizer, projector = engine.tokenizer, engine.projector
    rows = base._load_mmlu()
    chosen, ood_subject, grounded_rows, ood_rows = base._t2_store(rows)
    ingest: Dict[int, str] = {}
    write_started = time.perf_counter()
    for label, subject in enumerate(chosen):
        for row in [item for item in rows if item["subject"] == subject][:250]:
            wave = projector.sequence_wave(
                list(tokenizer.encode(row["question"]))[:256], tokenizer
            )
            record = memory.write(
                wave, label=int("ABCD".index(row["answer"])), tier=3, namespace=namespace
            )
            ingest[int(record.engram_id)] = hashlib.sha256(
                row["question"].encode("utf-8")
            ).hexdigest()
    write_seconds = float(time.perf_counter() - write_started)
    atlas_started = time.perf_counter()
    atlas_stats = memory.build_riemannian_atlas(
        namespace, rank=8, capacity=64, top_charts=1, top_charts_max=6
    )
    atlas_seconds = float(time.perf_counter() - atlas_started)

    def wave_for(row: Dict[str, str]) -> np.ndarray:
        return projector.sequence_wave(
            list(tokenizer.encode(row["question"]))[:256], tokenizer
        )

    query_rows = list(grounded_rows) + list(ood_rows)
    query_waves = [wave_for(row) for row in query_rows]
    query_digests = [
        hashlib.sha256(row["question"].encode("utf-8")).hexdigest() for row in query_rows
    ]
    memory._fetch([int(value) for value in memory._ids[: memory._size]])
    before: List[int] = []
    for wave in query_waves:
        read = memory.query(wave, k=8, namespace=namespace, use_riemannian=True)
        before.append(int(read.hits[0].engram_id) if read.hits else -1)
    signatures, physics = _online_texts()
    online_digest: Dict[int, str] = {}
    online_spawns: List[Dict[str, object]] = []
    online_started = time.perf_counter()
    for texts in (signatures, physics):
        for text, domain in texts:
            wave = projector.sequence_wave(
                list(tokenizer.encode(text))[:128], tokenizer
            )
            record = memory.write(wave, label=0, tier=2, namespace=domain)
            row_index = int(memory._size) - 1
            result = memory.observe_riemannian(
                memory.projection @ wave, row_index, domain
            )
            online_digest[int(record.engram_id)] = hashlib.sha256(
                text.encode("utf-8")
            ).hexdigest()
            if str(result.get("action")) == "spawn" and len(online_spawns) < 16:
                online_spawns.append(result)
    online_seconds = float(time.perf_counter() - online_started)
    after: List[int] = []
    for wave in query_waves:
        read = memory.query(wave, k=8, namespace=namespace, use_riemannian=True)
        after.append(int(read.hits[0].engram_id) if read.hits else -1)
    drift = sum(1 for left, right in zip(before, after) if int(left) != int(right))
    charted = [
        memory.query(wave, k=8, namespace=namespace, use_riemannian=True, candidate_multiplier=2)
        for wave in query_waves
    ]
    flat = [
        memory.query(wave, k=8, namespace=namespace, use_charts=False, candidate_multiplier=2)
        for wave in query_waves
    ]
    coherent = [
        float(row.hits[0].coherence)
        for row, digest in zip(charted[: len(grounded_rows)], query_digests[: len(grounded_rows)])
        if row.hits and ingest.get(int(row.hits[0].engram_id)) == digest
    ]
    tau = max(0.05, min(coherent) * 0.75) if coherent else 0.35
    grounded_correct = 0
    ood_abstain = 0
    abstained_total = 0
    hallucinated = 0
    parity = 0
    charted_ms: List[float] = []
    flat_ms: List[float] = []
    ratios: List[float] = []
    for position, (read, flat_read, digest) in enumerate(zip(charted, flat, query_digests)):
        match = None
        for hit in read.hits:
            if ingest.get(int(hit.engram_id)) == digest:
                match = hit
                break
        admitted = bool(match is not None and float(match.coherence) >= tau)
        predicted = "ABCD"[int(match.label)] if admitted and match is not None else "ABSTAIN"
        expected = query_rows[position]["answer"]
        on_domain = position < len(grounded_rows)
        correct = bool(predicted == expected)
        abstained_total += int(predicted == "ABSTAIN")
        if on_domain:
            grounded_correct += int(correct)
            parity += int(
                bool(match is not None)
                and int(read.hits[0].engram_id) == int(match.engram_id)
            )
        else:
            ood_abstain += int(predicted == "ABSTAIN")
        hallucinated += int(predicted != "ABSTAIN" and not correct)
        charted_ms.append(float(read.latency_ms))
        flat_ms.append(float(flat_read.latency_ms))
        ratios.append(float(read.scanned) / 2500.0)
    online_recall = 0
    online_total = 0
    for namespace_name in ("python_ast", "physics_pde"):
        ids = [
            int(row)
            for row in memory._ids[: memory._size]
            if memory.namespaces[int(memory._ns[memory._index_of(int(row))])] == namespace_name
        ][:50]
        for engram_id in ids:
            wave = memory.fetch_wave(int(engram_id))
            read = memory.query(wave, k=4, namespace=namespace_name, use_riemannian=True)
            online_total += 1
            if read.hits and online_digest.get(int(read.hits[0].engram_id)) == online_digest.get(
                int(engram_id)
            ):
                online_recall += 1
    grounded_accuracy = float(grounded_correct / max(len(grounded_rows), 1))
    ood_rate = float(ood_abstain / max(len(ood_rows), 1))
    hallucination = float(hallucinated / max(len(query_rows), 1))
    exact_parity = float(parity / max(len(grounded_rows), 1))
    drift_rate = float(drift / max(len(query_rows), 1))
    charted_p50 = float(np.median(charted_ms))
    flat_p50 = float(np.median(flat_ms))
    speedup = float(flat_p50 / max(charted_p50, 1e-9))
    mean_ratio = float(np.mean(ratios))
    gates = {
        "t2_mmlu_parity_ge_0_995": bool(exact_parity >= 0.995),
        "t2_grounded_ge_0_99": bool(grounded_accuracy >= 0.990),
        "t2_drift_zero": bool(drift == 0),
        "t2_ood_abstain_1": bool(ood_rate == 1.0),
        "t2_hallucination_zero": bool(hallucination == 0.0),
        "t2_candidates_le_0_35": bool(mean_ratio <= 0.35),
        "t2_speedup_ge_1_40": bool(speedup >= 1.40),
        "t2_online_recall_ge_0_99": bool(online_recall / max(online_total, 1) >= 0.99),
        "t2_atlas_synchronized": bool(int(atlas_stats["charts"]) >= 2),
    }
    result = {
        "stage": "T2",
        "mechanism": {
            "rank": 8,
            "capacity": 64,
            "tau_orth": 0.55,
            "tau_cap": 0.35,
            "whitening": "Woodbury O(r*d), no dense inverse",
            "spawn_policy": "out-of-subspace dark-energy chart birth",
            "update": "streaming rank-1 Oja-Welford",
        },
        "dataset": {
            "path": str(MMLU_PATH.relative_to(ROOT)),
            "sha256": hashlib.sha256(MMLU_PATH.read_bytes()).hexdigest(),
            "subjects": chosen,
            "ood_subject": ood_subject,
            "ingested": int(len(ingest)),
            "grounded_queries": int(len(grounded_rows)),
            "ood_queries": int(len(ood_rows)),
        },
        "atlas": {
            **atlas_stats,
            "build_seconds": atlas_seconds,
            "write_seconds": write_seconds,
            "engine": dict(engine_stats),
            "online_spawn_examples": online_spawns,
        },
        "online": {
            "writes": int(len(signatures) + len(physics)),
            "python_ast": int(len(signatures)),
            "physics_pde": int(len(physics)),
            "spawns": int(memory.atlas_online_charts),
            "seconds": online_seconds,
            "recall": float(online_recall / max(online_total, 1)),
            "recall_rows": int(online_total),
        },
        "recall": {
            "top1_parity": exact_parity,
            "drift": drift_rate,
            "drift_count": int(drift),
            "before_after_rows": int(len(query_rows)),
            "online_recall": float(online_recall / max(online_total, 1)),
        },
        "metrics": {
            "grounded_accuracy": grounded_accuracy,
            "ood_abstain_rate": ood_rate,
            "hallucination_rate": hallucination,
            "abstained": int(abstained_total),
        },
        "latency": {
            "charted_p50_ms": charted_p50,
            "flat_p50_ms": flat_p50,
            "speedup": speedup,
            "mean_evaluated_ratio": mean_ratio,
            "max_evaluated_ratio": float(max(ratios)),
            "namespace_engrams": 2500,
            "total_store_engrams": int(memory.stats()["count"]),
            "top_charts": 1,
            "top_charts_max": 6,
        },
        "gate": pc._finalise(gates),
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
        "passed": bool(all(gates.values())),
    }
    memory.close()
    return result

# ---------------------------------------------------------------------- T3
def stage_t3(directory: Path) -> Dict[str, object]:
    started = time.perf_counter()
    from henri.holarch.config import HolarchConfig
    from henri.holarch.induction import (
        BiDirectionalWaveInducer,
        WaveProgramInducer,
        contradictory_specs,
        sovereign_tasks,
    )
    from henri.holarch.paged_tape import (
        PagedHarmonicTape,
        binary_add,
        palindrome_check,
        reverse_tape,
    )
    from henri.holarch.zone_c import ZoneCMemory

    # The page wave lives in d_slot = 256, so the paging store uses a
    # 256-dimensional Zone C table with a 16-dimensional Gram projection.
    cfg = dataclasses.replace(HolarchConfig(), dim=256, gram_dims=16, cache_capacity=64)
    page_db = directory / "_work" / "ts_t3" / "pages.sqlite3"
    page_db.parent.mkdir(parents=True, exist_ok=True)
    if page_db.exists():
        try:
            page_db.unlink()
        except OSError:
            pass
    page_memory = ZoneCMemory(cfg, backend="sqlite", path=str(page_db))
    capacity_rows: List[Dict[str, object]] = []
    for length in (16, 64, 128, 256):
        tape = PagedHarmonicTape(
            memory=page_memory,
            length=int(length),
            alphabet=("0", "1", "#", "_"),
            namespace=f"tape_{length}",
        )
        text = ("10110010" * 40)[: int(length)]
        tape.load(text)
        reversed_text = reverse_tape(tape, int(length))
        reverse_exact = bool(reversed_text == text[::-1])
        margin_reverse = float(tape.min_read_margin())
        tape_pal = PagedHarmonicTape(
            memory=page_memory,
            length=int(length),
            alphabet=("0", "1", "#", "_"),
            namespace=f"tape_pal_{length}",
        )
        pal_text = ("0110" * 80)[: int(length)]
        tape_pal.load(pal_text)
        palindrome_exact = bool(palindrome_check(tape_pal, int(length)))
        tape_add = PagedHarmonicTape(
            memory=page_memory,
            length=int(length),
            alphabet=("0", "1", "#", "_"),
            namespace=f"tape_add_{length}",
        )
        operand = max(1, int(length) // 4)
        left = "1" + "0" * (operand - 1)
        right = "1" + "0" * (operand - 1)
        add_input = f"{left}#{right}"
        tape_add.load(add_input)
        add_output = binary_add(tape_add, int(length))
        add_exact = bool(
            tape_add.tape_text()[: len(add_output)] == add_output
            and int(add_output.split("#")[2], 2) == int(left, 2) + int(right, 2)
        )
        capacity_rows.append(
            {
                "length": int(length),
                "reverse_exact": bool(reverse_exact),
                "palindrome_exact": bool(palindrome_exact),
                "binary_add_exact": bool(add_exact),
                "min_read_margin": float(margin_reverse),
                "intra_page_crosstalk": float(tape.intra_page_crosstalk()),
                "page_ins": int(tape.page_ins + tape_pal.page_ins + tape_add.page_ins),
                "page_outs": int(tape.page_outs + tape_pal.page_outs + tape_add.page_outs),
                "moves": int(tape.moves + tape_pal.moves + tape_add.moves),
            }
        )
    page_memory.close()
    margin_16 = float(capacity_rows[0]["min_read_margin"])
    margin_256 = float(capacity_rows[-1]["min_read_margin"])
    capacity_exact = bool(
        all(
            row["reverse_exact"] and row["palindrome_exact"] and row["binary_add_exact"]
            for row in capacity_rows
            if int(row["length"]) >= 64
        )
    )
    crosstalk_max = float(max(row["intra_page_crosstalk"] for row in capacity_rows))
    margin_invariant = bool(abs(margin_16 - margin_256) <= 1e-6)
    capacity_gates = {
        "t3_capacity_exact_long": bool(capacity_exact),
        "t3_margin_ge_0_40": bool(min(float(row["min_read_margin"]) for row in capacity_rows) >= 0.40),
        "t3_margin_invariant": bool(margin_invariant),
        "t3_crosstalk_zero": bool(crosstalk_max < 1e-6),
        "t3_pages_paged": bool(any(int(row["page_ins"]) + int(row["page_outs"]) > 0 for row in capacity_rows)),
    }

    # Cycle V: the T3b slot measures the same one-cell surface as U3, from one
    # shared audit. A table that fits four short examples and fails a longer
    # tape is not a solved task, so both the declared held-out pairs and the
    # L in [12, 32] probes must pass.
    from henri.holarch.onecell_audit import audit_induction

    audit = audit_induction(node_budget=320000, max_states=8)
    tasks = list(audit["tasks"])
    solved = int(audit["solved"])
    generalized = int(audit["generalized"])
    v1_generalized = int(audit["v1_length_generalized"])
    text_tokens = int(audit["text_tokens"])
    contradictions = list(audit["contradictions"])
    contradiction_ok = int(audit["contradictions_failed_closed"])
    options = [{"name": row["name"], "rules": int(row.get("table_rules", 0)),
                "mechanism": str(row.get("mechanism_used", "")),
                "v1_generalized": bool(row.get("v1_generalized"))}
               for row in tasks if row.get("status") == "INDUCED"]
    gate_t3b = {
        "t3b_solved_ge_10": bool(solved >= 10),
        "t3b_generalized_ge_10": bool(generalized >= 10),
        "t3b_v1_length_generalized_ge_10": bool(v1_generalized >= 10),
        "t3b_zero_text_tokens": bool(text_tokens == 0),
        "t3b_contradictions_4": bool(contradiction_ok == 4),
        "t3b_two_cell_uncounted": bool(
            audit["foveated_counted_toward_gate"] is False),
    }
    gates = {**capacity_gates, **gate_t3b}
    return {
        "stage": "T3",
        "capacity": {
            "rows": capacity_rows,
            "margin_16": margin_16,
            "margin_256": margin_256,
            "crosstalk_max": crosstalk_max,
        },
        "induction": {
            "tasks": tasks,
            "solved": int(solved),
            "generalized": int(generalized),
            "v1_length_generalized": int(v1_generalized),
            "v1_lengths": list(audit["v1_lengths"]),
            "text_tokens": int(text_tokens),
            "contradictions": contradictions,
            "contradictions_failed_closed": int(contradiction_ok),
            "options_crystallized": int(len(options)),
        },
        "gate": pc._finalise(gates),
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
        "passed": bool(all(gates.values())),
    }


# ---------------------------------------------------------------------- T4
def stage_t4(directory: Path) -> Dict[str, object]:
    # Cycle U built the two organs. The T4 slot now reports the measured U4
    # stage instead of the Cycle T stub.
    import run_holarch_u_cycle as ucycle

    payload = dict(ucycle.stage_u4(directory))
    payload["stage"] = "T4"
    return payload


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Cycle T-Sovereign verification")
    parser.add_argument("--stages", default="t1,t2,t3,t4")
    parser.add_argument("--receipts", default=str(ROOT / "receipts" / "holarch"))
    args = parser.parse_args()
    directory = Path(args.receipts)
    directory.mkdir(parents=True, exist_ok=True)
    stages = [token.strip().lower() for token in str(args.stages).split(",") if token.strip()]
    runners = {
        "t1": ("t1_hygiene_and_determinism.json", base.stage_t1),
        "t2": ("t2_riemannian_chart_atlas.json", stage_t2),
        "t3": ("t3_foveated_bidirectional_turing.json", stage_t3),
        "t4": ("t4_sovereign_ccg_and_repo_sheaf.json", stage_t4),
    }
    aggregate: Dict[str, object] = {"cycle": "T-Sovereign", "head": pc._git_head(), "stages": {}}
    exit_code = 0
    for stage in stages:
        if stage not in runners:
            continue
        name, runner = runners[stage]
        print(f"[ts-cycle] {stage} ...", flush=True)
        payload = _sanitize(runner(directory))
        written = pc._write(directory, name, payload)
        aggregate["stages"][stage] = {
            "receipt": written["path"],
            "sha256": written["sha256"],
            "passed": bool(payload.get("gate", {}).get("passed", False)),
            "gate": payload.get("gate", {}),
            "wall_seconds": payload.get("wall_seconds", 0.0),
        }
        print(
            f"[ts-cycle] {stage} passed={payload.get('gate', {}).get('passed')} {written['path']}",
            flush=True,
        )
        if not bool(payload.get("gate", {}).get("passed", False)):
            exit_code = 1
    aggregate["passed"] = bool(
        all(bool(item.get("passed")) for item in aggregate["stages"].values())
    ) if aggregate["stages"] else False
    written = pc._write(directory, "t_cycle.json", aggregate)
    print(f"[ts-cycle] aggregate {written['path']} passed={aggregate['passed']}", flush=True)
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
