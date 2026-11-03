"""Continuous-learning probe: distribution shift and recovery.

Evidence label: OBSERVED. One trained checkpoint is copied, then adapted
to a pixel-permuted version of the same digits by two different paths:

    fast   Zone A delta-rule writes only, no backpropagation
    replay optimizer steps on a bounded replay buffer of the new domain

Retention is measured by re-evaluating the original domain after each
adaptation. A method that gains on the new domain and loses on the old
one is not a continual learner, and the table shows both columns.

    python scripts/run_holarch_continual.py --steps 150
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from henri.holarch.config import HolarchConfig
from henri.holarch.pipeline import HenriHolarch
from henri.holarch.tasks import build_tasks
from henri.holarch.train import Trainer, TrainConfig


def accuracy(model, task, permutation=None, limit=300) -> float:
    correct = 0
    total = 0
    for batch in task.epoch(64, "val"):
        x = batch.x
        if permutation is not None:
            x = x.reshape(x.shape[0], -1)[:, permutation].reshape(x.shape)
        output = model("vision", x, training=False, use_memory=False)
        correct += int((output.predicted() == batch.y).sum())
        total += int(batch.y.numel())
        if total >= limit:
            break
    return correct / max(total, 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--batches", type=int, default=24)
    parser.add_argument("--out", default="receipts/holarch/continual_vision.json")
    args = parser.parse_args()

    cfg = HolarchConfig.small()
    torch.manual_seed(cfg.seed)
    path = Path(tempfile.gettempdir()) / "henri_continual.sqlite3"
    if path.exists():
        path.unlink()
    tasks = build_tasks(cfg, seed=cfg.seed)
    model = HenriHolarch(cfg, backend="sqlite", memory_path=str(path))
    trainer = Trainer(model, tasks, TrainConfig(steps_per_task=int(args.steps), batch_size=32))
    trainer.flags["engram_write"] = False
    trainer.fit(steps_per_task=int(args.steps), order=["vision"])
    model.eval()

    anchors = {name: value.detach().clone() for name, value in model.state_dict().items()}
    permutation = torch.randperm(64, generator=torch.Generator().manual_seed(99))

    original_before = accuracy(model, tasks["vision"])
    shifted_before = accuracy(model, tasks["vision"], permutation)
    rows = {"no_adaptation": {"shifted": shifted_before, "original": original_before}}

    def shifted_batches():
        for _ in range(int(args.batches)):
            batch = tasks["vision"].sample(32, "train")
            x = batch.x.reshape(batch.x.shape[0], -1)[:, permutation].reshape(batch.x.shape)
            yield x, batch.y

    # fast path only
    model.load_state_dict(anchors)
    for x, y in shifted_batches():
        output = model("vision", x, training=False, use_memory=False)
        model.fast_learn(output, y, lr=0.2)
    rows["fast_weights_only"] = {
        "shifted": accuracy(model, tasks["vision"], permutation),
        "original": accuracy(model, tasks["vision"]),
    }

    # optical phase-conjugate imprint only (Zone B, no backprop)
    model.load_state_dict(anchors)
    for x, y in shifted_batches():
        model.imprint("vision", x, y, rate=0.2)
    rows["imprint_only"] = {
        "shifted": accuracy(model, tasks["vision"], permutation),
        "original": accuracy(model, tasks["vision"]),
    }

    # coordinated plasticity: Zone B imprint plus Zone A fast weights
    model.load_state_dict(anchors)
    for x, y in shifted_batches():
        output = model("vision", x, training=False, use_memory=False)
        model.fast_learn(output, y, lr=0.2)
        model.imprint("vision", x, y, rate=0.2)
    rows["imprint_plus_fast"] = {
        "shifted": accuracy(model, tasks["vision"], permutation),
        "original": accuracy(model, tasks["vision"]),
    }

    # plain fine-tuning on the new domain
    model.load_state_dict(anchors)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    for x, y in shifted_batches():
        output = model("vision", x, training=True, use_memory=False)
        loss = F.cross_entropy(output.task_logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    rows["finetune_only"] = {
        "shifted": accuracy(model, tasks["vision"], permutation),
        "original": accuracy(model, tasks["vision"]),
    }

    # experience replay: half old domain, half new domain, every step
    old_features = []
    old_labels = []
    for batch in tasks["vision"].epoch(64, "train"):
        old_features.append(batch.x)
        old_labels.append(batch.y)
        if sum(len(part) for part in old_labels) >= 512:
            break
    old_x = torch.cat(old_features)
    old_y = torch.cat(old_labels)

    def mixed_batches():
        generator = torch.Generator().manual_seed(7)
        for x, y in shifted_batches():
            index = torch.randint(0, old_x.shape[0], (x.shape[0],), generator=generator)
            yield torch.cat([x, old_x[index]]), torch.cat([y, old_y[index]]), x, y

    model.load_state_dict(anchors)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    for mix_x, mix_y, _, _ in mixed_batches():
        output = model("vision", mix_x, training=True, use_memory=False)
        loss = F.cross_entropy(output.task_logits, mix_y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    rows["replay_mixed"] = {
        "shifted": accuracy(model, tasks["vision"], permutation),
        "original": accuracy(model, tasks["vision"]),
    }

    model.load_state_dict(anchors)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    for mix_x, mix_y, x, y in mixed_batches():
        output = model("vision", mix_x, training=True, use_memory=False)
        loss = F.cross_entropy(output.task_logits, mix_y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        fresh = model("vision", x, training=False, use_memory=False)
        model.fast_learn(fresh, y, lr=0.2)
    rows["fast_plus_replay_mixed"] = {
        "shifted": accuracy(model, tasks["vision"], permutation),
        "original": accuracy(model, tasks["vision"]),
    }

    baseline = rows["no_adaptation"]["shifted"]
    for name, row in rows.items():
        row["shifted_gain"] = row["shifted"] - baseline
        row["retention"] = row["original"] / max(original_before, 1e-9)

    # Associative rule shift: the pixels stay, the labels rotate by three.
    # Fast weights are an associative map and should adapt without backprop.
    shift = 3

    def rule_accuracy(model, offset: int) -> float:
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in tasks["vision"].epoch(64, "val"):
                labels = (batch.y + offset) % 10
                output = model("vision", batch.x, training=False, use_memory=False)
                correct += int((output.predicted() == labels).sum())
                total += int(labels.numel())
        return correct / max(total, 1)

    def rule_batches():
        generator = torch.Generator().manual_seed(31)
        for _ in range(int(args.batches)):
            batch = tasks["vision"].sample(32, "train")
            yield batch.x, (batch.y + shift) % 10

    rule = {}
    model.load_state_dict(anchors)
    rule["no_adaptation"] = {
        "shifted": rule_accuracy(model, shift),
        "original": rule_accuracy(model, 0),
    }
    model.load_state_dict(anchors)
    for x, y in rule_batches():
        output = model("vision", x, training=False, use_memory=False)
        model.fast_learn(output, y, lr=0.2)
    rule["fast_weights_only"] = {
        "shifted": rule_accuracy(model, shift),
        "original": rule_accuracy(model, 0),
    }
    model.load_state_dict(anchors)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    for x, y in rule_batches():
        index = torch.randint(0, old_x.shape[0], (x.shape[0],), generator=torch.Generator().manual_seed(5))
        mix_x = torch.cat([x, old_x[index]])
        mix_y = torch.cat([y, old_y[index]])
        output = model("vision", mix_x, training=True, use_memory=False)
        loss = F.cross_entropy(output.task_logits, mix_y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    rule["replay_mixed"] = {
        "shifted": rule_accuracy(model, shift),
        "original": rule_accuracy(model, 0),
    }
    rule_baseline = rule["no_adaptation"]["shifted"]
    for name, row in rule.items():
        row["shifted_gain"] = row["shifted"] - rule_baseline
        row["retention"] = row["original"] / max(original_before, 1e-9)
    rows["rule_shift"] = rule

    payload = {
        "task": "vision-pixel-permutation",
        "steps": int(args.steps),
        "adaptation_batches": int(args.batches),
        "permutation_seed": 99,
        "original_before_shift": original_before,
        "variants": rows,
        "rule_shift": rule,
        "evidence": "OBSERVED",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    (out.parent / (out.name + ".sha256")).write_text(
        hashlib.sha256(out.read_bytes()).hexdigest() + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, default=float))
    model.memory.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
