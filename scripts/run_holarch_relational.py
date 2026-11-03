"""Relational compositionality: holarch against a flat MLP.

Evidence label: OBSERVED. Two holdout densities are measured:

    standard   25 percent of triples held out
    low-data   75 percent held out, 25 percent training triples

The flat MLP is parameter matched (159k against the holarch's 172k) and
uses the same optimiser, seed and step budget. The low-data case is the
Directive 1 gate: a CP-decomposed readout should generalise from few
triples, a dense one-hot MLP should not.

    python scripts/run_holarch_relational.py --steps 300
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
from henri.holarch.tasks import RelationalTask
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


def accuracy(logits_fn, task, split: str) -> float:
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in task.epoch(64, split):
            logits = logits_fn(batch.x)
            correct += int((logits.argmax(-1) == batch.y).sum())
            total += int(batch.y.numel())
    return correct / max(total, 1)


def run_case(cfg, task, steps: int, hidden: int, tag: str):
    torch.manual_seed(cfg.seed)
    path = Path(tempfile.gettempdir()) / f"henri_relational_{tag}.sqlite3"
    if path.exists():
        path.unlink()
    tasks = {"relational": task}
    holarch = HenriHolarch(cfg, backend="sqlite", memory_path=str(path))
    trainer = Trainer(holarch, tasks, TrainConfig(steps_per_task=1, batch_size=32, lr=2e-3, ood_weight=0.0))
    for _ in range(int(steps)):
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
        hidden=int(hidden),
    )
    optimizer = torch.optim.AdamW(flat.parameters(), lr=2e-3, weight_decay=1e-4)
    for _ in range(int(steps)):
        batch = task.sample(32, "train")
        loss = F.cross_entropy(flat(batch.x), batch.y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    flat.eval()
    return {
        "train_triples": int(task.size("train")),
        "test_triples": int(task.size("test")),
        "holarch": {
            "train_accuracy": holarch_train,
            "test_accuracy": holarch_test,
            "parameters": holarch_params,
        },
        "flat_mlp": {
            "train_accuracy": accuracy(flat, task, "train"),
            "test_accuracy": accuracy(flat, task, "test"),
            "parameters": int(sum(p.numel() for p in flat.parameters())),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--hidden", type=int, default=384)
    parser.add_argument("--out", default="receipts/holarch/relational.json")
    args = parser.parse_args()

    cfg = HolarchConfig.small()
    standard = RelationalTask(seed=cfg.seed, holdout=0.25)
    low_data = RelationalTask(seed=cfg.seed, holdout=0.75)

    cases = {}
    for tag, task in (("standard", standard), ("low_data", low_data)):
        result = run_case(cfg, task, int(args.steps), int(args.hidden), tag)
        result["holarch_generalises"] = bool(result["holarch"]["test_accuracy"] > 0.25)
        result["flat_mlp_below_0_30"] = bool(result["flat_mlp"]["test_accuracy"] < 0.30)
        cases[tag] = result
        print(f"{tag:9s} holarch test={result['holarch']['test_accuracy']:.3f} "
              f"mlp test={result['flat_mlp']['test_accuracy']:.3f}")

    payload = {
        "task": "relational",
        "steps": int(args.steps),
        "classes": int(cfg.relational_classes),
        "cases": cases,
        "gate": {
            "standard_passed": bool(
                cases["standard"]["holarch_generalises"]
                and cases["standard"]["flat_mlp_below_0_30"]
            ),
            "low_data_passed": bool(
                cases["low_data"]["holarch_generalises"]
                and cases["low_data"]["flat_mlp_below_0_30"]
            ),
            "passed": bool(
                cases["low_data"]["holarch_generalises"]
                and cases["low_data"]["flat_mlp_below_0_30"]
            ),
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
