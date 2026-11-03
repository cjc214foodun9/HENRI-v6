"""Ablation of the three zones on the vision task.

Evidence label: OBSERVED. Every row is a separate training run with the
same seed, the same data order and the same step budget. Only the named
component is switched off.

    python scripts/run_holarch_ablation.py --steps 150
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from henri.holarch.config import HolarchConfig
from henri.holarch.pipeline import HenriHolarch
from henri.holarch.tasks import build_tasks
from henri.holarch.train import Trainer, TrainConfig


VARIANTS = {
    "full": {},
    "identity_optics": {"use_photonic": False},
    "no_zone_b_features": {"use_optical_features": False},
    "no_swarm": {"use_swarm": False},
    "no_memory": {"use_memory": False, "engram_write": False},
    "no_fast_weights": {"fast_learn": False},
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--out", default="receipts/holarch/ablation_vision.json")
    args = parser.parse_args()

    rows = {}
    for name, flags in VARIANTS.items():
        cfg = HolarchConfig.small()
        torch.manual_seed(cfg.seed)
        path = Path(tempfile.gettempdir()) / f"henri_ablation_{name}.sqlite3"
        if path.exists():
            path.unlink()
        tasks = build_tasks(cfg, seed=cfg.seed)
        model = HenriHolarch(cfg, backend="sqlite", memory_path=str(path))
        trainer = Trainer(model, tasks, TrainConfig(steps_per_task=int(args.steps), batch_size=32))
        trainer.flags.update(flags)
        history = trainer.fit(steps_per_task=int(args.steps), order=["vision"])
        # Vision only: this receipt is a vision ablation, and evaluating the
        # other three tasks would triple the wall clock for no extra claim.
        correct = 0
        total = 0
        loss_sum = 0.0
        batches = 0
        with torch.no_grad():
            for batch in tasks["vision"].epoch(64, "val"):
                output = model(
                    "vision",
                    batch.x,
                    training=False,
                    use_photonic=trainer.flags["use_photonic"],
                    use_swarm=trainer.flags["use_swarm"],
                    use_memory=trainer.flags["use_memory"],
                    use_optical_features=trainer.flags["use_optical_features"],
                )
                correct += int((output.predicted() == batch.y).sum())
                total += int(batch.y.numel())
                loss_sum += float(torch.nn.functional.cross_entropy(output.task_logits, batch.y))
                batches += 1
                if total >= 300:
                    break
        metrics = {
            "vision_accuracy": correct / max(total, 1),
            "vision_loss": loss_sum / max(batches, 1),
        }
        rows[name] = {
            "vision_accuracy": metrics["vision_accuracy"],
            "vision_loss": metrics["vision_loss"],
            "final_train_loss": history.steps[-1]["loss"],
            "parameters": model.parameter_count(),
            "memory_count": model.memory.stats()["count"],
        }
        model.memory.close()
        print(f"{name:16s} acc={rows[name]['vision_accuracy']:.4f} loss={rows[name]['vision_loss']:.4f}")

    baseline = rows["full"]["vision_accuracy"]
    for name, row in rows.items():
        row["delta_vs_full"] = row["vision_accuracy"] - baseline
    payload = {
        "task": "vision",
        "steps": int(args.steps),
        "seed": HolarchConfig.small().seed,
        "variants": rows,
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
