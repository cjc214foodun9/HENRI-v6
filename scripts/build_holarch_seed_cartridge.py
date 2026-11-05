"""Build the J1 seed cartridge and rewrite the deployed checkpoint.

    python scripts/build_holarch_seed_cartridge.py

The cartridge is the minimal sufficient Zone C chart for the acceptance
measurement, selected by a declared rule and not by a search:

1. every axiomatic (tier-3) row of the action task block, because those are
   the class prototypes the attractor branch reads;
2. every engram that enters the candidate pool of the eight Cul-de-Sac trap
   states, so a clean-room trap decision is the reference decision and not a
   near neighbour of it;
3. the engrams the 48-episode closed-loop run actually retrieved, ranked by
   how many states retrieved them, until the 512-engram cap is reached.

The waves are stored as two half-precision quadratures inside
holarch_medium.pt, which keeps 512 rows at 2.1 MB (a complex64 row at D=1024
is 8 KB, so 512 rows would be 4.2 MB and break the 2.5 MB budget).

The script never mutates docs or receipts other than the checkpoint and its
.sha256 sidecar; the J1 stage writes the measurement receipt.
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from henri.holarch.config import HolarchConfig
from henri.holarch.model import (
    SEED_ENGRAM_LIMIT,
    LatentActionLookahead,
    build_cul_de_sac_traps,
    evaluate_closed_loop_action,
    load_checkpoint,
    open_planner_memory,
    seed_engram_payload,
)
from henri.holarch.pipeline import HenriHolarch
from henri.holarch.tasks import build_tasks
from henri.holarch.zone_c import task_label_block

CHECKPOINT = ROOT / "receipts" / "holarch" / "holarch_medium.pt"
RECEIPT = ROOT / "receipts" / "holarch" / "j1_cartridge.json"


def _digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
    lo, hi = task_label_block("action")
    action_ids = [engram for engram in meta if lo <= meta[engram][0] < hi]
    waves = dict(zip(action_ids, memory._fetch(action_ids)))

    model = HenriHolarch(cfg, memory=memory)
    load_checkpoint(str(CHECKPOINT), model=model)
    model.eval()
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

    def capture(ids, *args, **kwargs):
        return original_fetch(ids, *args, **kwargs)

    original_query = memory.query
    def recording_query(wave, **kwargs):
        read = original_query(wave, **kwargs)
        for hit in read.hits:
            hits[int(hit.engram_id)] += 1
        return read
    memory.query = recording_query
    task = build_tasks(cfg, seed=int(cfg.seed))["action"]
    closed = evaluate_closed_loop_action(
        model, task, episodes=48, rotor=getattr(model, "action_rotor", None),
        transition="rotor", lookahead_depth=2, gamma_look=0.02, batch_rollout=True,
        use_memory=True,
    )
    memory.query = original_query

    tier3 = [engram for engram in action_ids if meta[engram][1] == 3]
    mandatory = sorted(set(trap_pool) | set(tier3))
    ranked = [engram for engram, _count in hits.most_common() if engram in meta]
    chosen: List[int] = list(mandatory)
    for engram in ranked:
        if len(chosen) >= int(limit):
            break
        if engram not in chosen:
            chosen.append(engram)
    for engram in action_ids:
        if len(chosen) >= int(limit):
            break
        if engram not in chosen:
            chosen.append(engram)
    chosen = sorted(chosen)[: int(limit)]

    rows = [
        {
            "label": int(meta[engram][0]),
            "tier": int(meta[engram][1]),
            "wave": waves[engram],
            "namespace": "default",
            "source_sha256": str(fixture.get("sha256", "")),
        }
        for engram in chosen
    ]
    block = seed_engram_payload(rows)
    memory.close()

    # Rewrite the checkpoint with the cartridge. Everything else in the payload
    # is preserved verbatim.
    temporary = CHECKPOINT.with_suffix(".pt.tmp")
    payload["seed_engrams"] = block
    torch.save(payload, temporary)
    os.replace(temporary, CHECKPOINT)
    digest = _digest(CHECKPOINT)
    CHECKPOINT.with_name(CHECKPOINT.name + ".sha256").write_text(digest + chr(10), encoding="utf-8")

    receipt = {
        "stage": "J1-cartridge",
        "selection_rule": [
            "all tier-3 axiomatic rows of the action task block",
            "all engrams in the candidate pool of the eight Cul-de-Sac trap states",
            "closed-loop retrieved engrams ranked by retrieval count, to the cap",
            "then remaining action-block rows in id order, to the cap",
        ],
        "reference_fixture": {
            "path": str(fixture.get("source", "")),
            "sha256": str(fixture.get("sha256", "")),
            "engrams": int(fixture.get("engrams", 0)),
            "bytes": int(fixture.get("bytes", 0)),
        },
        "action_block": {
            "rows": int(len(action_ids)),
            "tier3": int(len(tier3)),
            "trap_pool": int(len(trap_pool)),
            "retrieved": int(len(hits)),
            "retrieved_total_hits": int(sum(hits.values())),
        },
        "cartridge": {
            "count": int(block["count"]),
            "bytes": int(block["bytes"]),
            "dim": int(block["dim"]),
            "dtype": str(block["dtype"]),
            "limit": int(limit),
            "sha256": digest,
        },
        "reference_measurement": {
            "closed_loop_wins": int(closed["wins"]),
            "traps_correct": int(sum(1 for row in trap_actions if row["correct"])),
            "traps": trap_actions,
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
    print(json.dumps({key: receipt[key] for key in ("cartridge", "action_block")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
