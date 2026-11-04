"""Noisy-observation probe for the thermodynamic relaxation loop.

Evidence label: OBSERVED. One trained checkpoint is evaluated on the same
validation split under additive Gaussian sensor noise.

Decision 2 replaced the blind pull toward a noisy context bundle with a
pull toward clean post-photonic prototypes plus a Stiefel nullspace
penalty. The probe therefore measures, for every noise level:

    model accuracy      the trained readout on the (possibly relaxed) wave
    attractor accuracy  the nearest clean prototype of the same wave
    Lyapunov energy     (1 - max alignment) + lambda_dark * dark energy
    dark energy         the fraction of the field left outside U_s

The attractor row matters because the relaxation objective is a
classification energy: its minimum IS a prototype, so the readout that
matches the physics is the prototype alignment, not a head trained on
unrelaxed waves.

    python scripts/run_holarch_noise.py --steps 300 --sigma 0.25
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from henri.holarch.config import HolarchConfig
from henri.holarch.pipeline import HenriHolarch
from henri.holarch.tasks import build_tasks
from henri.holarch.train import Trainer, TrainConfig


def _batches(task, sigma: float, split: str = "val", seed: int = 0):
    generator = torch.Generator().manual_seed(int(seed))
    for batch in task.epoch(64, split):
        noise = torch.randn(batch.x.shape, generator=generator) * float(sigma)
        yield batch.x + noise, batch.y


def evaluate(model, task, sigma: float, relax: bool, limit: int = 400, seed: int = 0) -> dict:
    """Model accuracy, attractor accuracy, energy and dark energy."""
    lambda_dark = float(model.cfg.prototype_lambda_dark)
    bank = model.prototype_waves("vision")
    labels = torch.from_numpy(model.prototypes["vision"].labels.astype(np.int64))
    correct = 0
    attractor_correct = 0
    total = 0
    energy_sum = 0.0
    dark_sum = 0.0
    with torch.no_grad():
        for x, y in _batches(task, sigma, seed=seed):
            output = model("vision", x, training=False, relax=relax)
            correct += int((output.predicted() == y).sum())
            bright = output.photonic.bright
            if bank.numel() > 0:
                align = model.zone_b.alignment(bright, bank)
                attractor_correct += int((labels[align.argmax(dim=-1)] == y).sum())
                free, _ = model.zone_b.free_energy(bright, bank, lambda_dark)
                energy_sum += float(free.sum())
                dark_sum += float(model.zone_b.dark_energy(bright).sum())
            total += int(y.numel())
            if total >= int(limit):
                break
    count = max(total, 1)
    return {
        "model_accuracy": correct / count,
        "attractor_accuracy": attractor_correct / count,
        "energy": energy_sum / count,
        "dark_energy": dark_sum / count,
        "n": float(total),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--sigma", type=float, default=0.25)
    parser.add_argument("--sweep", default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--seeds", type=int, default=5, help="independent noise draws")
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
    batches = [tasks["vision"].sample(32, "val").x for _ in range(8)]
    # Decision A3: one joint calibration for both veto conditions.
    veto_report = model.calibrate_two_condition("vision", batches, ceiling=0.98)
    # Decision 2: clean attractors, built before any noise is applied.
    prototype_stats = model.calibrate_prototypes(
        "vision", [tasks["vision"].sample(64, "train") for _ in range(4)]
    )
    # Heat only where the wave is unfamiliar, so the loop cannot disturb a
    # confident clean input.
    gate_stats = model.calibrate_gate(
        "vision", [tasks["vision"].sample(32, "val").x for _ in range(4)]
    )
    model.eval()

    rows = {}
    seeds = int(args.seeds)
    for raw in str(args.sweep).split(","):
        sigma = float(raw)
        deltas = {"model": [], "attractor": [], "energy": [], "dark": []}
        off_model: List[float] = []
        on_model: List[float] = []
        off_attractor: List[float] = []
        on_attractor: List[float] = []
        for seed in range(seeds):
            off = evaluate(model, tasks["vision"], sigma, relax=False, seed=seed)
            on = evaluate(model, tasks["vision"], sigma, relax=True, seed=seed)
            off_model.append(off["model_accuracy"])
            on_model.append(on["model_accuracy"])
            off_attractor.append(off["attractor_accuracy"])
            on_attractor.append(on["attractor_accuracy"])
            deltas["model"].append(on["model_accuracy"] - off["model_accuracy"])
            deltas["attractor"].append(on["attractor_accuracy"] - off["attractor_accuracy"])
            deltas["energy"].append(on["energy"] - off["energy"])
            deltas["dark"].append(on["dark_energy"] - off["dark_energy"])

        def mean(values: List[float]) -> float:
            return float(np.mean(values)) if values else float("nan")

        def stderr(values: List[float]) -> float:
            if len(values) < 2:
                return 0.0
            return float(np.std(values, ddof=1) / np.sqrt(len(values)))

        rows[f"{sigma:.2f}"] = {
            "seeds": seeds,
            "no_relax_model_accuracy": mean(off_model),
            "relax_model_accuracy": mean(on_model),
            "no_relax_attractor_accuracy": mean(off_attractor),
            "relax_attractor_accuracy": mean(on_attractor),
            "model_delta": mean(deltas["model"]),
            "model_delta_stderr": stderr(deltas["model"]),
            "attractor_delta": mean(deltas["attractor"]),
            "attractor_delta_stderr": stderr(deltas["attractor"]),
            "energy_delta": mean(deltas["energy"]),
            "dark_delta": mean(deltas["dark"]),
        }

    primary = f"{float(args.sigma):.2f}"
    row = rows[primary]
    payload = {
        "task": "vision",
        "steps": int(args.steps),
        "sigma": float(args.sigma),
        "conditions": rows,
        "clean_accuracy": rows.get("0.00", {}).get("relax_model_accuracy"),
        "noisy_accuracy_relax": row["relax_model_accuracy"],
        "noisy_accuracy_no_relax": row["no_relax_model_accuracy"],
        "delta_from_relax": row["model_delta"],
        "attractor_delta_from_relax": row["attractor_delta"],
        "dark_energy_reduction": -row["dark_delta"],
        "energy_reduction": -row["energy_delta"],
        "prototypes": prototype_stats,
        "gate_calibration": gate_stats,
        "veto_calibration": veto_report,
        "gate": {
            "relax_gain_at_sigma_at_least_0_05": bool(row["model_delta"] >= 0.05),
            "relax_gain_positive": bool(row["model_delta"] > 0.0),
            "relax_gain_two_stderr_positive": bool(
                row["model_delta"] - 2.0 * row["model_delta_stderr"] > 0.0
            ),
            "attractor_gain_positive": bool(row["attractor_delta"] > 0.0),
            "energy_reduced": bool(row["energy_delta"] < 0.0),
            "dark_energy_reduced": bool(row["dark_delta"] < 0.0),
            "passed": bool(row["model_delta"] >= 0.05),
            "a2_clean_invariant": bool(
                abs(rows.get("0.00", {}).get("model_delta", float("nan"))) <= 1e-12
            ),
            "a2_all_sigma_positive": bool(
                all(rows[key]["model_delta"] > 0.0 for key in ("0.25", "0.50", "0.75", "1.00") if key in rows)
            ),
            "a2_passed": bool(
                abs(rows.get("0.00", {}).get("model_delta", float("nan"))) <= 1e-12
                and all(rows[key]["model_delta"] > 0.0 for key in ("0.25", "0.50", "0.75", "1.00") if key in rows)
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
    model.memory.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
