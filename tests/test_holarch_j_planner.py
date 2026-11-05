"""Directive J1: clean-room checkpoint planning and thread invariance.

Every assertion here is a measured property of the artifact that ships, not a
restatement of the design: the seed cartridge is read out of the checkpoint,
the hydration path is exercised on a brand-new SQLite file, and the thread
invariance test runs the same forward pass under one and eight intra-op
threads.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from henri.holarch.config import HolarchConfig
from henri.holarch.model import (
    LatentActionLookahead,
    build_cul_de_sac_traps,
    evaluate_closed_loop_action,
    load_checkpoint,
    open_clean_room_memory,
    save_checkpoint,
    seed_engram_payload,
)
from henri.holarch.pipeline import HenriHolarch
from henri.holarch.tasks import build_tasks
from henri.holarch.zone_c import ZoneCMemory

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "receipts" / "holarch" / "holarch_medium.pt"


def checkpoint_config() -> HolarchConfig:
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    return HolarchConfig(**payload["config"])


def test_checkpoint_carries_a_bounded_seed_cartridge():
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    block = payload.get("seed_engrams")
    assert block and block.get("format") == "henri-seed-engrams-v1"
    assert 0 < int(block["count"]) <= 512
    assert int(block["bytes"]) < 2_500_000
    assert tuple(block["real"].shape) == (int(block["count"]), int(block["dim"]))
    assert block["real"].dtype == torch.float16


def test_mount_hydrates_an_empty_store_exactly_once(tmp_path):
    cfg = checkpoint_config()
    memory, info = open_clean_room_memory(cfg, CHECKPOINT, tmp_path, "hydrate")
    try:
        assert info["pre_mount_rows"] == 0
        assert int(memory.stats()["count"]) == int(info["hydrated"]) > 0
        block = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)["seed_engrams"]
        assert int(info["hydrated"]) == int(block["count"])
        model = HenriHolarch(cfg, memory=memory)
        _, metadata = load_checkpoint(str(CHECKPOINT), model=model)
        # A second mount must not duplicate a single row.
        assert int(memory.stats()["count"]) == int(block["count"])
        assert metadata["seed_engrams"]["hydration"]["written"] == 0
    finally:
        memory.close()


def test_seed_payload_refuses_an_oversized_cartridge():
    cfg = HolarchConfig.small()
    wave = np.ones(int(cfg.dim), dtype=np.complex64) / np.sqrt(float(cfg.dim))
    rows = [{"label": 0, "tier": 1, "wave": wave} for _ in range(513)]
    with pytest.raises(ValueError):
        seed_engram_payload(rows)


def test_forward_pass_is_bit_identical_at_one_and_eight_threads():
    cfg = HolarchConfig.small(seed=7)
    torch.manual_seed(cfg.seed)
    model = HenriHolarch(cfg, backend="sqlite", memory_path=None)
    model.eval()
    task = build_tasks(cfg, seed=cfg.seed)["action"]
    observation = torch.as_tensor(
        np.asarray(task.val_pairs[0][0], dtype=np.float32)
    ).reshape(1, -1)
    before = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.no_grad():
            single = model("action", observation, training=False, use_memory=False).task_logits[0]
        torch.set_num_threads(8)
        with torch.no_grad():
            many = model("action", observation, training=False, use_memory=False).task_logits[0]
    finally:
        torch.set_num_threads(before)
    assert torch.equal(single, many)


def test_clean_room_traps_are_solved_and_thread_invariant(tmp_path):
    cfg = checkpoint_config()
    memory, _info = open_clean_room_memory(cfg, CHECKPOINT, tmp_path, "traps")
    try:
        model = HenriHolarch(cfg, memory=memory)
        load_checkpoint(str(CHECKPOINT), model=model)
        model.eval()
        traps = build_cul_de_sac_traps()
        rotor = getattr(model, "action_rotor", None)
        before = torch.get_num_threads()
        try:
            results = {}
            for threads in (1, 8):
                torch.set_num_threads(threads)
                lookahead = LatentActionLookahead(
                    model, None, rotor=rotor, transition="rotor",
                    lookahead_depth=2, gamma_look=0.02,
                )
                rows = []
                for trap in traps:
                    choice = lookahead.choose(trap["observation"], [])
                    rows.append(
                        (
                            int(choice["action"]),
                            [None if not np.isfinite(float(v)) else float(v) for v in choice["scores"]],
                        )
                    )
                results[threads] = rows
        finally:
            torch.set_num_threads(before)
        eight = [action for action, _ in results[8]]
        bypass = [int(trap["bypass_action"]) for trap in traps]
        correct = sum(1 for action, truth in zip(eight, bypass) if action == truth)
        assert correct == 8, (eight, bypass)
        assert [action for action, _ in results[1]] == eight
        for (_, left), (_, right) in zip(results[8], results[1]):
            for a, b in zip(left, right):
                if a is None or b is None:
                    assert (a is None) == (b is None)
                else:
                    assert abs(a - b) < 1e-6
    finally:
        memory.close()


def test_clean_room_closed_loop_trace_matches_across_threads(tmp_path):
    cfg = checkpoint_config()
    memory, _info = open_clean_room_memory(cfg, CHECKPOINT, tmp_path, "closed")
    try:
        model = HenriHolarch(cfg, memory=memory)
        load_checkpoint(str(CHECKPOINT), model=model)
        model.eval()
        task = build_tasks(cfg, seed=cfg.seed)["action"]
        rotor = getattr(model, "action_rotor", None)
        before = torch.get_num_threads()
        try:
            traces = {}
            for threads in (1, 8):
                torch.set_num_threads(threads)
                result = evaluate_closed_loop_action(
                    model, task, episodes=4, rotor=rotor, transition="rotor",
                    lookahead_depth=2, gamma_look=0.02, use_memory=True, trace=True,
                )
                traces[threads] = (int(result["wins"]), result["trace"])
        finally:
            torch.set_num_threads(before)
        assert traces[1] == traces[8]
        assert len(traces[8][1]) == 4
    finally:
        memory.close()
