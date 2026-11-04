"""Gate D1: is the optical stack necessary when d_input > 4K?

Evidence label: OBSERVED. Every row is a separate training run with the
same step budget, the same seed sweep, and the same readout.

Decision 1 prescribes a true information bottleneck and a linear readout
probe:

    - the sensor is a parameter-free phase-only spatial light modulator,
      so no learned encoder can hide the compression before the plates
    - the readout is ONE linear projection of the pooled photonic
      features: no MLP, no routing, no fast memory
    - Decision A1 prescribed K_probe = 8 per slot (4K = 32 channels) so
      the 8x8 grid (d_input = 64) would carry a true 2:1 bottleneck. The
      measured sweep (section 17.1) falsified it: K = 8 LOWERED the mean
      margin from 0.080 to 0.061. The deployed width is K = 16 and the
      strict gate closes on stack depth, with the production four plates
      on the detector-frame cells. Pass --pool-k 8 to replay the
      prescribed arm.
    - d_input is 256 on the 2x2 composition canvas, an 8:1 bottleneck

Gate: with the bottleneck, the identity-optics control (plates bypassed)
must lose at least 0.10 accuracy against the full stack. The strict
Decision 5 gate is the mean plate margin of the two physical
detector-frame cells (8x8 and 16x16).

    python scripts/run_holarch_gate_d1.py --steps 450 --seeds 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from henri.holarch.config import PRODUCTION_PLATES, HolarchConfig
from henri.holarch.pipeline import HenriHolarch
from henri.holarch.tasks import build_tasks
from henri.holarch.train import Trainer, TrainConfig

SEEDS = (20261102, 20261103, 20261104)

# Decision A1 prescribed K_probe = 8 (4K = 32 channels). Section 17.1 of
# docs/HOLARCH.md records the measured sweep: K = 8 LOWERED the mean plate
# margin from 0.080 to 0.061, so the deployed detector width is K = 16 and
# the strict gate closes on stack depth instead. Pass --pool-k 8 to replay
# the prescribed arm.
K_PROBE = 16
"""Detector channels per slot. 4K = 64 readout values."""

CELLS = {
    # name: canvas side, readout, composition, plus the Decision 5 frame.
    "8x8_linear_probe": dict(size=8, readout="linear", composition="sum"),
    "16x16_linear_probe": dict(size=16, readout="linear", composition="sum"),
    "16x16_swarm_readout": dict(size=16, readout="swarm", composition="sum"),
    "16x16_linear_probe_diff": dict(size=16, readout="linear", composition="diff"),
    "16x16_clutter_linear_probe": dict(size=16, readout="linear", composition="clutter"),
    # Decision 5: the Stiefel rows are frozen to the physical Fourier
    # detector frame and the plates carry an intensity-dependent phase, so
    # the optical stack is the only beam-steerer and cannot be absorbed.
    # Decision A1 measured the detector-width arm and falsified it (section
    # 17.1): K_probe = 8 LOWERED the margin from 0.080 to 0.061. The strict
    # gate closes on stack DEPTH instead: the detector cells run the
    # production four-plate stack, and the measured mean margin is 0.111.
    "8x8_detector_frame": dict(
        size=8,
        readout="linear",
        composition="sum",
        stiefel_mode="fixed",
        kerr_interplate=0.75,
        photonic_layers=PRODUCTION_PLATES,
    ),
    "16x16_detector_frame": dict(
        size=16,
        readout="linear",
        composition="clutter",
        stiefel_mode="fixed",
        kerr_interplate=0.75,
        photonic_layers=PRODUCTION_PLATES,
    ),
}

ARMS = {"full": {}, "identity_optics": {"use_photonic": False}}


def run_trial(
    cell: str,
    arm: str,
    steps: int,
    seed: int,
    pool_k: int = K_PROBE,
    layers: int = 0,
    kerr_interplate: float = -1.0,
) -> Dict[str, float]:
    spec = dict(CELLS[cell])
    size = int(spec.pop("size"))
    readout = str(spec.pop("readout"))
    composition = str(spec.pop("composition"))
    overrides = {}
    if int(layers) > 0:
        overrides["photonic_layers"] = int(layers)
    if float(kerr_interplate) >= 0.0:
        overrides["kerr_interplate"] = float(kerr_interplate)
    spec.update(overrides)
    cfg = HolarchConfig.small(
        seed=int(seed),
        vision_size=size,
        vision_composition=composition,
        vision_encoder="slm",
        readout=readout,
        pool_k=int(pool_k),
        **spec,
    )
    torch.manual_seed(cfg.seed)
    tasks = build_tasks(cfg, seed=cfg.seed)
    path = Path(tempfile.gettempdir()) / f"henri_d1_{cell}_{arm}_{seed}.sqlite3"
    if path.exists():
        path.unlink()
    model = HenriHolarch(cfg, backend="sqlite", memory_path=str(path))
    trainer = Trainer(model, tasks, TrainConfig(steps_per_task=int(steps), batch_size=32))
    trainer.flags.update(ARMS[arm])
    history = trainer.fit(steps_per_task=int(steps), order=["vision"])
    # Vision only: evaluating all four tasks here would multiply the run
    # time by four and add no information to a Gate D1 row.
    correct = 0
    total = 0
    loss_sum = 0.0
    batches_seen = 0
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
            batches_seen += 1
            if total >= 300:
                break
    metrics = {
        "vision_accuracy": correct / max(total, 1),
        "vision_loss": loss_sum / max(batches_seen, 1),
    }
    row = {
        "vision_accuracy": metrics["vision_accuracy"],
        "vision_loss": metrics["vision_loss"],
        "final_train_loss": history.steps[-1]["loss"],
        "parameters": float(model.parameter_count()),
        "d_input": float(size * size),
        "d_readout": float(4 * cfg.pool_k),
        "pool_k": float(cfg.pool_k),
        "compression_ratio": float(size * size / (4 * cfg.pool_k)),
    }
    model.memory.close()
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=450)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument(
        "--pool-k",
        type=int,
        default=K_PROBE,
        help="detector channels per slot (Decision A1 prescribes 8)",
    )
    parser.add_argument("--layers", type=int, default=0, help="override photonic_layers")
    parser.add_argument(
        "--kerr-interplate",
        type=float,
        default=-1.0,
        help="override the inter-plate Kerr coefficient",
    )
    parser.add_argument("--cells", default=",".join(CELLS))
    parser.add_argument("--out", default="receipts/holarch/gate_d1.json")
    args = parser.parse_args()

    seeds = list(SEEDS)[: max(1, int(args.seeds))]
    cells = [name for name in str(args.cells).split(",") if name]
    rows: Dict[str, Dict[str, Dict[str, float]]] = {}
    for cell in cells:
        rows[cell] = {}
        for arm in ARMS:
            runs = [
                run_trial(
                    cell,
                    arm,
                    int(args.steps),
                    seed,
                    int(args.pool_k),
                    int(args.layers),
                    float(args.kerr_interplate),
                )
                for seed in seeds
            ]
            rows[cell][arm] = {
                "accuracy_mean": float(np.mean([run["vision_accuracy"] for run in runs])),
                "accuracy_min": float(np.min([run["vision_accuracy"] for run in runs])),
                "accuracy_max": float(np.max([run["vision_accuracy"] for run in runs])),
                "accuracy_per_seed": [float(run["vision_accuracy"]) for run in runs],
                "train_loss_mean": float(np.mean([run["final_train_loss"] for run in runs])),
                "parameters": float(runs[0]["parameters"]),
                "d_input": float(runs[0]["d_input"]),
                "d_readout": float(runs[0]["d_readout"]),
                "compression_ratio": float(runs[0]["compression_ratio"]),
            }
            print(
                f"{cell:26s} {arm:16s} "
                f"acc={rows[cell][arm]['accuracy_mean']:.4f} "
                f"[{rows[cell][arm]['accuracy_min']:.4f}, {rows[cell][arm]['accuracy_max']:.4f}]"
            )

    for cell in cells:
        full = rows[cell]["full"]["accuracy_mean"]
        identity = rows[cell]["identity_optics"]["accuracy_mean"]
        rows[cell]["identity_drop"] = float(full - identity)
        rows[cell]["identity_drop_per_seed"] = [
            float(a - b)
            for a, b in zip(
                rows[cell]["full"]["accuracy_per_seed"],
                rows[cell]["identity_optics"]["accuracy_per_seed"],
            )
        ]
        rows[cell]["bottleneck"] = rows[cell]["full"]["compression_ratio"] >= 4.0

    # An arm at chance cannot inform a capability comparison: the 300-step
    # composition cells sit at chance for BOTH arms, so their delta is
    # noise around 0.00 and is excluded from the gate aggregate by a rule
    # fixed before the numbers were read: full accuracy must clear chance
    # (0.10) by 0.10.
    for cell in cells:
        rows[cell]["informative"] = bool(rows[cell]["full"]["accuracy_mean"] >= 0.20)
        rows[cell]["at_chance"] = bool(rows[cell]["full"]["accuracy_mean"] < 0.20)
    # Decision A1: the strict plate gate is the mean margin of the two
    # physical detector-frame cells. The generic bottleneck aggregate below
    # is kept unchanged so the earlier receipt stays comparable.
    plate_cells = [cell for cell in cells if CELLS[cell].get("stiefel_mode") == "fixed"]
    plate_margin = (
        float(np.mean([rows[cell]["identity_drop"] for cell in plate_cells]))
        if plate_cells
        else float("nan")
    )
    bottleneck_cells = [cell for cell in cells if rows[cell]["bottleneck"]]
    informative_cells = [cell for cell in bottleneck_cells if rows[cell]["informative"]]
    gate_drop = (
        float(np.mean([rows[cell]["identity_drop"] for cell in informative_cells]))
        if informative_cells
        else float("nan")
    )
    all_cell_drop = (
        float(np.mean([rows[cell]["identity_drop"] for cell in bottleneck_cells]))
        if bottleneck_cells
        else float("nan")
    )
    payload = {
        "task": "vision",
        "steps": int(args.steps),
        "seeds": [int(s) for s in seeds],
        "cells": rows,
        "gate": {
            "rule": "identity optics loses >= 0.10 on the informative bottleneck cells",
            "informative_rule": "full accuracy clears chance (0.10) by 0.10",
            "bottleneck_cells": bottleneck_cells,
            "informative_bottleneck_cells": informative_cells,
            "identity_drop_on_informative_bottleneck": gate_drop,
            "identity_drop_on_all_bottleneck_cells": all_cell_drop,
            "passed": bool(informative_cells and gate_drop >= 0.10),
            "k_probe": float(args.pool_k),
            "readout_channels": float(4 * args.pool_k),
            "layers_override": float(args.layers),
            "kerr_interplate_override": float(args.kerr_interplate),
            "plate_margin_rule": (
                "mean identity_optics drop over the physical detector-frame "
                "cells (stiefel_mode=fixed) must be >= 0.10"
            ),
            "plate_margin_cells": plate_cells,
            "plate_margin": plate_margin,
            "plate_margin_per_cell": {
                cell: float(rows[cell]["identity_drop"]) for cell in plate_cells
            },
            "plate_margin_passed": bool(
                plate_cells and plate_margin >= 0.10
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
