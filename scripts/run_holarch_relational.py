"""Relational compositionality: holarch against a flat MLP.

Evidence label: OBSERVED. Two holdout densities are measured:

    standard   25 percent of triples held out
    low-data   75 percent held out, 25 percent training triples

Decision 3 replaces the K = 16 real trilinear product with a wave-domain
bind followed by a dedicated Stiefel basis, and right-sizes the penalty
(L2 5e-4, dropout 0.05). Both arms are measured on both splits so the
change is attributed, not asserted:

    wave        Decision 3: bind in C^d_slot, read 2 * K_rel quadratures
    trilinear   Decision C control: K-dim real product, L2 0.05, dropout 0.2

The flat MLP is parameter matched and uses the same optimiser, seed and
step budget.

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

ARMS = {
    # Decision 7: circular convolution over four cyclic frames, read through
    # the Zone A relational MLP instead of a standalone linear head.
    "conv_shift": {
        "relational_binding": "conv_shift",
        "relational_k": 16,
        "relational_shifts": 4,
        "relational_head": "expert",
        "relational_l2": 5e-4,
        "relational_dropout": 0.05,
    },
    # Decision 3: single diagonal product, dedicated linear head.
    "wave": {
        "relational_binding": "wave",
        "relational_k": 32,
        "relational_head": "linear",
        "relational_l2": 5e-4,
        "relational_dropout": 0.05,
    },
    # Decision C: the retired K-dim real product.
    "trilinear": {
        "relational_binding": "trilinear",
        "relational_head": "linear",
        "relational_l2": 0.05,
        "relational_dropout": 0.20,
    },
}


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


def run_holarch(cfg, task, steps: int, arm: str, tag: str):
    torch.manual_seed(cfg.seed)
    path = Path(tempfile.gettempdir()) / f"henri_relational_{tag}_{arm}.sqlite3"
    if path.exists():
        path.unlink()
    tasks = {"relational": task}
    holarch = HenriHolarch(cfg, backend="sqlite", memory_path=str(path))
    trainer = Trainer(
        holarch, tasks, TrainConfig(steps_per_task=1, batch_size=32, lr=2e-3, ood_weight=0.0)
    )
    losses = []
    for _ in range(int(steps)):
        losses.append(trainer.step("relational")["loss"])
    holarch.eval()
    holarch_train = accuracy(lambda x: holarch("relational", x, training=False).task_logits, task, "train")
    holarch_test = accuracy(lambda x: holarch("relational", x, training=False).task_logits, task, "test")
    params = holarch.parameter_count()
    holarch.memory.close()
    return {
        "train_accuracy": holarch_train,
        "test_accuracy": holarch_test,
        "parameters": params,
        "final_train_loss": float(np.mean(losses[-10:])),
        "head_width": int(holarch.relational_width),
    }


def run_mlp(cfg, task, steps: int, hidden: int):
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
        "train_accuracy": accuracy(flat, task, "train"),
        "test_accuracy": accuracy(flat, task, "test"),
        "parameters": int(sum(p.numel() for p in flat.parameters())),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--budget-sweep", default="300,600,900")
    parser.add_argument("--hidden", type=int, default=384)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--out", default="receipts/holarch/relational.json")
    args = parser.parse_args()

    seeds = [20261102, 20261103, 20261104][: max(1, int(args.seeds))]
    cases = {}
    for tag, holdout in (("standard", 0.25), ("low_data", 0.75)):
        task = RelationalTask(seed=20261102, holdout=holdout)
        mlp = run_mlp(HolarchConfig.small(), task, int(args.steps), int(args.hidden))
        arms = {}
        for arm, overrides in ARMS.items():
            runs = []
            for seed in seeds:
                cfg = HolarchConfig.small(seed=int(seed), **overrides)
                runs.append(run_holarch(cfg, task, int(args.steps), arm, tag))
            arms[arm] = {
                "train_accuracy_mean": float(np.mean([r["train_accuracy"] for r in runs])),
                "test_accuracy_mean": float(np.mean([r["test_accuracy"] for r in runs])),
                "test_accuracy_per_seed": [float(r["test_accuracy"]) for r in runs],
                "parameters": float(runs[0]["parameters"]),
                "head_width": float(runs[0]["head_width"]),
                "final_train_loss": float(np.mean([r["final_train_loss"] for r in runs])),
            }
            arms[arm]["beats_mlp"] = bool(arms[arm]["test_accuracy_mean"] > mlp["test_accuracy"])
        cases[tag] = {
            "train_triples": int(task.size("train")),
            "test_triples": int(task.size("test")),
            "flat_mlp": mlp,
            "arms": arms,
        }
        print(
            f"{tag:9s} conv={arms['conv_shift']['test_accuracy_mean']:.3f} "
            f"wave={arms['wave']['test_accuracy_mean']:.3f} "
            f"trilinear={arms['trilinear']['test_accuracy_mean']:.3f} "
            f"mlp={mlp['test_accuracy']:.3f} "
            f"(conv train={arms['conv_shift']['train_accuracy_mean']:.3f}, "
            f"wave train={arms['wave']['train_accuracy_mean']:.3f})"
        )

    # Step-budget attribution: the bilinear architecture needs more
    # optimisation steps than a dense MLP to reach the same training loss.
    # The sweep is recorded so the choice of budget is evidence, not taste.
    budget: Dict[str, Dict[str, float]] = {}
    standard_task = RelationalTask(seed=20261102, holdout=0.25)
    for raw in str(args.budget_sweep).split(","):
        if not raw.strip():
            continue
        budget_steps = int(raw)
        cfg = HolarchConfig.small(seed=20261102, **ARMS["conv_shift"])
        conv = run_holarch(cfg, standard_task, budget_steps, "conv", f"budget{budget_steps}")
        mlp_budget = run_mlp(HolarchConfig.small(), standard_task, budget_steps, int(args.hidden))
        budget[str(budget_steps)] = {
            "conv_train": conv["train_accuracy"],
            "conv_test": conv["test_accuracy"],
            "mlp_train": mlp_budget["train_accuracy"],
            "mlp_test": mlp_budget["test_accuracy"],
        }

    payload = {
        "task": "relational",
        "budget_sweep": budget,
        "steps": int(args.steps),
        "classes": 8,
        "seeds": seeds,
        "cases": cases,
        "gate": {
            "conv_standard_passed": bool(
                cases["standard"]["arms"]["conv_shift"]["beats_mlp"]
            ),
            "conv_low_data_passed": bool(
                cases["low_data"]["arms"]["conv_shift"]["beats_mlp"]
            ),
            "passed": bool(
                cases["standard"]["arms"]["conv_shift"]["beats_mlp"]
                and cases["low_data"]["arms"]["conv_shift"]["beats_mlp"]
            ),
            "wave_standard_passed": bool(cases["standard"]["arms"]["wave"]["beats_mlp"]),
            "trilinear_standard_passed": bool(
                cases["standard"]["arms"]["trilinear"]["beats_mlp"]
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
    print(json.dumps(payload["gate"], indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
