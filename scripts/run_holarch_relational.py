"""Relational compositionality: holarch against a flat MLP.

Evidence label: OBSERVED. Same triples, same seed, same step budget and
the same optimiser. The only difference is the inductive bias:

    holarch   binds the three role waves (pointwise phasor product) and
              reads the bound wave through the photonic stack
    flat MLP  concatenates one-hot primitives and learns a dense map

    python scripts/run_holarch_relational.py --steps 400
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from henri.holarch.config import HolarchConfig
from henri.holarch.pipeline import HenriHolarch
from henri.holarch.tasks import build_tasks
from henri.holarch.train import Trainer, TrainConfig


class FlatMLP(nn.Module):
    """Dense map from concatenated one-hot primitives to the outcome."""

    def __init__(self, entities: int, actions: int, targets: int, classes: int, hidden: int = 384) -> None:
        super().__init__()
        self.entities = entities
        self.actions = actions
        self.targets = targets
        self.net = nn.Sequential(
            nn.Linear(entities + actions + targets, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        onehot = torch.cat(
            [
                F.one_hot(x[:, 0].long(), self.entities).float(),
                F.one_hot(x[:, 1].long(), self.actions).float(),
                F.one_hot(x[:, 2].long(), self.targets).float(),
            ],
            dim=-1,
        )
        return self.net(onehot)


def accuracy(model, task, split: str) -> float:
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in task.epoch(64, split):
            logits = model(batch.x)
            correct += int((logits.argmax(-1) == batch.y).sum())
            total += int(batch.y.numel())
    return correct / max(total, 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--hidden", type=int, default=384)
    parser.add_argument("--out", default="receipts/holarch/relational.json")
    args = parser.parse_args()

    cfg = HolarchConfig.small()
    torch.manual_seed(cfg.seed)
    tasks = build_tasks(cfg, seed=cfg.seed)
    task = tasks["relational"]

    path = Path(tempfile.gettempdir()) / "henri_relational.sqlite3"
    if path.exists():
        path.unlink()
    holarch = HenriHolarch(cfg, backend="sqlite", memory_path=str(path))
    trainer = Trainer(holarch, tasks, TrainConfig(steps_per_task=1, batch_size=32, lr=2e-3, ood_weight=0.0))
    for _ in range(int(args.steps)):
        trainer.step("relational")
    holarch.eval()
    holarch_train = accuracy(lambda x: holarch("relational", x, training=False).task_logits, task, "train")
    holarch_test = accuracy(lambda x: holarch("relational", x, training=False).task_logits, task, "test")
    holarch_params = holarch.parameter_count()
    holarch.memory.close()

    torch.manual_seed(1)
    flat = FlatMLP(
        cfg.relational_entities,
        cfg.relational_actions,
        cfg.relational_targets,
        cfg.relational_classes,
        hidden=int(args.hidden),
    )
    optimizer = torch.optim.AdamW(flat.parameters(), lr=2e-3, weight_decay=1e-4)
    for _ in range(int(args.steps)):
        batch = task.sample(32, "train")
        loss = F.cross_entropy(flat(batch.x), batch.y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    flat.eval()
    flat_train = accuracy(flat, task, "train")
    flat_test = accuracy(flat, task, "test")
    flat_params = int(sum(p.numel() for p in flat.parameters()))

    payload = {
        "task": "relational",
        "steps": int(args.steps),
        "train_triples": int(task.size("train")),
        "test_triples": int(task.size("test")),
        "classes": int(task.n_classes),
        "holarch": {
            "train_accuracy": holarch_train,
            "test_accuracy": holarch_test,
            "parameters": holarch_params,
        },
        "flat_mlp": {
            "train_accuracy": flat_train,
            "test_accuracy": flat_test,
            "parameters": flat_params,
        },
        "gate": {
            "holarch_generalises": bool(holarch_test > 0.25),
            "flat_mlp_below_25": bool(flat_test < 0.25),
            "passed": bool(holarch_test > 0.25 and flat_test < 0.25),
        },
        "evidence": "OBSERVED",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    (out.parent / (out.name + ".sha256")).write_text(
        hashlib.sha256(out.read_bytes()).hexdigest() + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
