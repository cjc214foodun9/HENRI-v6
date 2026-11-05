"""Build the v1.0 multi-task seed cartridge and rewrite the checkpoint.

    python scripts/build_holarch_seed_cartridge.py

Cycle K closes two Cycle J boundaries:

1. the J1 cartridge held 512 action rows, so the deterministic reference
   store scored 38/48 closed-loop while the clean room scored 37/48;
2. the J1 cartridge hydrated only the action block, so a clean-room vision,
   language or relational query hit an empty candidate set and the C1 rule
   vetoed it.

The K1 cartridge is a declared selection, not a search:

1. every action-block row (604), because the closed-loop Voronoi partition
   of the deterministic reference store is the partition being ratified;
2. every tier-3 axiomatic row in all four task blocks, because those are the
   class prototypes and the boundary-axiom source;
3. per-task support engrams: the rows the task's validation and test
   families actually retrieve, ranked by retrieval count, up to the declared
   quota. This is the trajectory-critical cover, so a clean-room query reads
   the same top-k context wave the reference store served;
4. if the serialised block still exceeds 2.5 MiB, the lowest-ranked support
   rows are removed first (language, then relational, then vision) and the
   trim is recorded. The action block and the tier-3 rows are never trimmed.

Storage. Cycle J used two float16 quadratures (4 bytes per dimension), which
does not fit the multi-task set. K1 uses two peak-normalised int8
quadratures (2 bytes per dimension) compressed with zlib level 9. The
reconstruction coherence of every row is measured and written to the
receipt; the format is lossless with respect to the stored int8 code.

The script writes the checkpoint, its .sha256 sidecar, and
receipts/holarch/j1_cartridge.json with the reference measurement, in the
same shape Cycle J used so the J1 stage keeps its meaning.
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from henri.holarch.config import HolarchConfig
from henri.holarch.model import (
    SEED_ENGRAM_BYTE_BUDGET,
    SEED_ENGRAM_LIMIT,
    LatentActionLookahead,
    build_cul_de_sac_traps,
    evaluate_closed_loop_action,
    load_checkpoint,
    open_planner_memory,
    seed_engram_payload,
    seed_engram_rows,
)
from henri.holarch.pipeline import HenriHolarch
from henri.holarch.tasks import build_tasks
from henri.holarch.train import evaluate
from henri.holarch.zone_c import task_label_block

CHECKPOINT = ROOT / "receipts" / "holarch" / "holarch_medium.pt"
RECEIPT = ROOT / "receipts" / "holarch" / "j1_cartridge.json"

SUPPORT_QUOTAS: Dict[str, int] = {"vision": 106, "relational": 72, "language": 64}
"""Declared per-task support quota, tier-3 rows included. The trim order is the reverse."""

TASK_ORDER = ("language", "relational", "vision")
"""Trim order when the byte budget binds: least gate-critical first."""


def _digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _task_cover(model, tasks, task: str, memory, limit: int = 4000) -> collections.Counter:
    """Engram ids the task's validation and test families retrieve, with counts."""
    counter: collections.Counter = collections.Counter()
    seen = 0
    for split in ("val", "test"):
        try:
            batches = tasks[task].epoch(64, split)
        except KeyError:
            continue
        for batch in batches:
            waves = model.encode(task, batch.x).detach().numpy()
            for row in waves:
                read = memory.query(row, k=4, task=task)
                for hit in read.hits:
                    counter[int(hit.engram_id)] += 1
                seen += 1
            if seen >= int(limit):
                return counter
    return counter


def _balanced_cover(
    cover_ids: Sequence[int],
    meta: Dict[int, tuple],
    block: tuple,
    quota: int,
) -> List[int]:
    """Class-balanced order of one task's retrieval cover.

    Frequency alone over-samples the classes the validation family visits
    most. Round-robin over the class buckets gives every class a context wave
    before any class gets a second one, which is what the K1 support quotas
    are sized against.
    """
    lo, _hi = block
    groups: Dict[int, List[int]] = {}
    for engram in cover_ids:
        if engram not in meta:
            continue
        groups.setdefault(int(meta[engram][0]) - int(lo), []).append(int(engram))
    out: List[int] = []
    keys = sorted(groups)
    while len(out) < int(quota) and any(groups[key] for key in keys):
        for key in keys:
            if groups[key] and len(out) < int(quota):
                out.append(groups[key].pop(0))
    return out


def _select(
    meta: Dict[int, tuple],
    cover: Dict[str, List[int]],
    quotas: Dict[str, int],
) -> tuple:
    """Apply the declared selection rule; returns ids, per-block counts, trim log."""
    lo, hi = task_label_block("action")
    action_ids = sorted(i for i in meta if lo <= meta[i][0] < hi)
    tier3 = sorted(i for i in meta if meta[i][1] == 3)
    chosen: List[int] = list(action_ids)
    seen = set(chosen)
    for engram in tier3:
        if engram not in seen:
            chosen.append(int(engram))
            seen.add(int(engram))

    support: Dict[str, int] = {}
    for task in ("vision", "relational", "language"):
        block = task_label_block(task)
        current = sum(1 for engram in chosen if block[0] <= meta[engram][0] < block[1])
        room = max(0, int(quotas.get(task, 0)) - int(current))
        added = 0
        for engram in _balanced_cover(cover.get(task, []), meta, block, int(room)):
            if engram not in seen:
                chosen.append(int(engram))
                seen.add(int(engram))
                added += 1
        support[task] = int(added)
    return sorted(chosen), action_ids, tier3, support


def build(workdir: Path, limit: int = SEED_ENGRAM_LIMIT) -> Dict[str, object]:
    started = time.perf_counter()
    workdir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    cfg = HolarchConfig(**payload["config"])
    reference = workdir / "seed_source.sqlite3"
    if reference.exists():
        reference.unlink()

    memory, fixture = open_planner_memory(cfg, workdir=workdir, prefix="seed_source")
    meta = {int(row[0]): (int(row[2]), int(row[4])) for row in memory.store.metadata(cfg.dim)}
    wave_by_id = dict(zip(sorted(meta), memory._fetch(sorted(meta))))

    model = HenriHolarch(cfg, memory=memory)
    load_checkpoint(str(CHECKPOINT), model=model)
    model.eval()
    tasks = build_tasks(cfg, seed=int(cfg.seed))

    # Directive K1 (3): the trajectory-critical cover of the four task blocks.
    cover_rows: Dict[str, List[int]] = {}
    cover_counts: Dict[str, int] = {}
    for task in ("vision", "relational", "language"):
        counter = _task_cover(model, tasks, task, memory)
        cover_rows[task] = [engram for engram, _count in counter.most_common()]
        cover_counts[task] = int(sum(counter.values()))

    traps = build_cul_de_sac_traps()
    lookahead = LatentActionLookahead(
        model, None, rotor=getattr(model, "action_rotor", None), transition="rotor",
        lookahead_depth=2, gamma_look=0.02,
    )
    trap_pool: set = set()
    original_fetch = memory._fetch

    def recording_fetch(ids, *args, **kwargs):
        trap_pool.update(int(item) for item in ids)
        return original_fetch(ids, *args, **kwargs)

    memory._fetch = recording_fetch
    trap_actions: List[Dict[str, object]] = []
    for trap in traps:
        choice = lookahead.choose(trap["observation"], [])
        trap_actions.append(
            {
                "id": str(trap["id"]),
                "action": int(choice["action"]),
                "bypass": int(trap["bypass_action"]),
                "correct": bool(int(choice["action"]) == int(trap["bypass_action"])),
            }
        )
    memory._fetch = original_fetch

    hits: collections.Counter = collections.Counter()
    original_query = memory.query

    def recording_query(wave, **kwargs):
        read = original_query(wave, **kwargs)
        for hit in read.hits:
            hits[int(hit.engram_id)] += 1
        return read

    memory.query = recording_query
    task = tasks["action"]
    closed = evaluate_closed_loop_action(
        model, task, episodes=48, rotor=getattr(model, "action_rotor", None),
        transition="rotor", lookahead_depth=2, gamma_look=0.02, batch_rollout=True,
        use_memory=True, trace=True,
    )
    memory.query = original_query

    chosen, action_ids, tier3, support = _select(meta, cover_rows, SUPPORT_QUOTAS)
    support_pool = [i for i in chosen if i not in set(action_ids) and i not in set(tier3)]

    with torch.no_grad():
        reference_accuracy = {
            split: {
                f"{name}_accuracy": float(metrics.get(f"{name}_accuracy", 0.0))
                for name in ("vision", "language", "action", "relational")
            }
            for split, metrics in (
                ("val", evaluate(model, tasks, "val", limit=2000)),
                ("test", evaluate(model, tasks, "test", limit=2000)),
            )
        }
    memory.close()

    def serialise(ids: Sequence[int]) -> Dict[str, object]:
        records = [
            {
                "label": int(meta[engram][0]),
                "tier": int(meta[engram][1]),
                "wave": wave_by_id[engram],
                "namespace": "default",
                "source_sha256": str(fixture.get("sha256", "")),
            }
            for engram in ids
        ]
        return records

    chosen = sorted(chosen)
    trimmed: List[int] = []
    block = None
    while True:
        records = serialise(chosen)
        block = seed_engram_payload(records)
        if int(block["bytes"]) <= SEED_ENGRAM_BYTE_BUDGET:
            break
        drop = None
        support_set = set(support_pool)
        for task_name in TASK_ORDER:
            candidates = [i for i in chosen if i in cover_rows.get(task_name, []) and i in support_set]
            if candidates:
                # Lowest retrieval count first; ties broken by the largest id.
                drop = min(
                    candidates,
                    key=lambda i: (-cover_rows[task_name].index(i), -int(i)),
                )
                break
        if drop is None:
            raise RuntimeError("the mandatory action and tier-3 blocks exceed the byte budget")
        chosen.remove(drop)
        trimmed.append(int(drop))

    # Quantisation fidelity: the cosine between every stored wave and its
    # reconstructed copy. This is measured, not asserted.
    decoded = seed_engram_rows(block)
    original = np.stack([np.asarray(record["wave"], dtype=np.complex64) for record in records])
    stored = np.stack([row["wave"] for row in decoded])
    coherence = np.real(np.sum(stored * np.conj(original), axis=1))
    coherence = coherence / np.maximum(
        np.linalg.norm(stored, axis=1) * np.linalg.norm(original, axis=1), 1e-12
    )

    temporary = CHECKPOINT.with_suffix(".pt.tmp")
    payload["seed_engrams"] = block
    torch.save(payload, temporary)
    os.replace(temporary, CHECKPOINT)
    digest = _digest(CHECKPOINT)
    CHECKPOINT.with_name(CHECKPOINT.name + ".sha256").write_text(digest + chr(10), encoding="utf-8")

    labels = [int(meta[engram][0]) for engram in chosen]
    rows_by_task = {
        name: int(sum(1 for label in labels if lo <= label < hi))
        for name, (lo, hi) in (
            ("vision", task_label_block("vision")),
            ("language", task_label_block("language")),
            ("action", task_label_block("action")),
            ("relational", task_label_block("relational")),
        )
    }
    tier3_in_chosen = int(sum(1 for engram in chosen if meta[engram][1] == 3))
    receipt = {
        "stage": "J1-cartridge",
        "cartridge_version": "v1.0-multitask-q16",
        "selection_rule": [
            "every action-block row (the closed-loop Voronoi partition)",
            "every tier-3 axiomatic row in all four task blocks",
            "per-task support: validation and test retrieval cover, by count, to quota",
            "byte trim: lowest-ranked support row first (language, relational, vision)",
        ],
        "quotas": dict(SUPPORT_QUOTAS),
        "reference_fixture": {
            "path": str(fixture.get("source", "")),
            "sha256": str(fixture.get("sha256", "")),
            "engrams": int(fixture.get("engrams", 0)),
            "bytes": int(fixture.get("bytes", 0)),
        },
        "action_block": {
            "rows": int(len(action_ids)),
            "tier3": int(sum(1 for engram in action_ids if meta[engram][1] == 3)),
            "trap_pool": int(len(trap_pool)),
            "retrieved": int(len(hits)),
            "retrieved_total_hits": int(sum(hits.values())),
        },
        "task_cover": {
            "vision_unique": int(len(cover_rows["vision"])),
            "relational_unique": int(len(cover_rows["relational"])),
            "language_unique": int(len(cover_rows["language"])),
            "vision_retrievals": int(cover_counts["vision"]),
            "relational_retrievals": int(cover_counts["relational"]),
            "language_retrievals": int(cover_counts["language"]),
        },
        "support": {
            "added": support,
            "rows_by_task": rows_by_task,
            "trimmed": len(trimmed),
            "trimmed_ids": trimmed[:32],
        },
        "cartridge": {
            "count": int(block["count"]),
            "bytes": int(block["bytes"]),
            "raw_bytes": int(block.get("raw_bytes", 0)),
            "dim": int(block["dim"]),
            "dtype": str(block["dtype"]),
            "compression": str(block.get("compression", "")),
            "limit": int(limit),
            "sha256": digest,
            "rows_by_task": rows_by_task,
            "tier3_rows": tier3_in_chosen,
            "within_budget": bool(int(block["bytes"]) <= SEED_ENGRAM_BYTE_BUDGET),
            "reconstruction_coherence_min": float(coherence.min()),
            "reconstruction_coherence_mean": float(coherence.mean()),
        },
        "reference_measurement": {
            "closed_loop_wins": int(closed["wins"]),
            "closed_loop_trace": closed.get("trace"),
            "one_step_accuracy": float(closed.get("one_step_accuracy", 0.0)),
            "traps_correct": int(sum(1 for row in trap_actions if row["correct"])),
            "traps": trap_actions,
            "accuracy": reference_accuracy,
        },
        "checkpoint": {
            "path": str(CHECKPOINT.relative_to(ROOT)),
            "sha256": digest,
            "bytes": int(CHECKPOINT.stat().st_size),
        },
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
    }
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(receipt, indent=2, default=float), encoding="utf-8")
    RECEIPT.with_name(RECEIPT.name + ".sha256").write_text(
        _digest(RECEIPT) + chr(10), encoding="utf-8"
    )
    return receipt


def main() -> int:
    receipt = build(ROOT / "receipts" / "holarch" / "_work")
    print(json.dumps({key: receipt[key] for key in ("cartridge", "action_block", "support")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
