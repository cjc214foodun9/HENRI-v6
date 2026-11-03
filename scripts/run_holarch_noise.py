"""Noisy-observation probe for the thermodynamic relaxation loop.

Evidence label: OBSERVED. A trained checkpoint is evaluated on the same
validation split twice: once with the relaxation loop enabled and once
with it disabled. The inputs carry Gaussian sensor noise, which is the
regime the loop is supposed to repair.

    python scripts/run_holarch_noise.py --steps 200 --sigma 0.25
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


def accuracy(model, task, sigma: float, limit: int = 400, seed: int = 0) -> float:
    generator = torch.Generator().manual_seed(int(seed))
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in task.epoch(64, "val"):
            noise = torch.randn(batch.x.shape, generator=generator) * float(sigma)
            output = model("vision", batch.x + noise, training=False)
            correct += int((output.predicted() == batch.y).sum())
            total += int(batch.y.numel())
            if total >= int(limit):
                break
    return correct / max(total, 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--sigma", type=float, default=0.25)
    parser.add_argument("--out", default="receipts/holarch/noise_vision.json")
    args = parser.parse_args()

    cfg = HolarchConfig.small()
    torch.manual_seed(cfg.seed)
    tasks = build_tasks(cfg)
    path = Path(tempfile.gettempdir()) / "henri_noise.sqlite3"
    if path.exists():
        path.unlink()
    model = HenriHolarch(cfg, backend="sqlite", memory_path=str(path))
    trainer = Trainer(model, tasks, TrainConfig(steps_per_task=1, batch_size=32, lr=2e-3))
    for _ in range(int(args.steps)):
        trainer.step("vision")
    batches = [tasks["vision"].sample(32, "val").x for _ in range(6)]
    model.calibrate_chance("vision", batches, percentile=1.0, margin=0.02)
    model.calibrate_veto("vision", batches, ceiling=0.98)
    model.eval()

    clean = accuracy(model, tasks["vision"], 0.0)
    noisy_relax = accuracy(model, tasks["vision"], float(args.sigma))
    model.cfg = cfg.replace(relax_steps=0)
    noisy_plain = accuracy(model, tasks["vision"], float(args.sigma))
    model.cfg = cfg

    payload = {
        "task": "vision",
        "steps": int(args.steps),
        "sigma": float(args.sigma),
        "clean_accuracy": clean,
        "noisy_accuracy_relax": noisy_relax,
        "noisy_accuracy_no_relax": noisy_plain,
        "delta_from_relax": noisy_relax - noisy_plain,
        "gate": {
            "relax_gain_at_least_0_05": bool(noisy_relax - noisy_plain >= 0.05),
            "passed": bool(noisy_relax - noisy_plain >= 0.05),
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
    model.memory.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
