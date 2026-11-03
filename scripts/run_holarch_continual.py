"""Continuous-learning probe: sensory shift, stationarity, rule context.

Evidence label: OBSERVED. Three probes share one protocol.

A. Sensory shift. The digits are pixel-permuted and the checkpoint is
   adapted by the fast path (and by mixed replay). Retention is the
   original-domain accuracy after adaptation.

B. Stationarity control. The same adaptation runs on the ORIGINAL domain.
   A correct surprise gate writes nothing there, so accuracy must not
   move. Decision 4 gates the fast learning rate on the Sagnac/prototype
   residual: eta_eff = eta * 1[surprise > tau OR s_max < tau_familiar].

C. Rule shift with slot 3. The labels rotate by three on identical
   pixels. Without a context variable the joint problem is a logical
   contradiction and Fano's inequality caps the combined accuracy at
   0.50. Decision 4 binds a deterministic rule phasor into slot 3, so the
   same pixels under two rules are two different input waves.

    python scripts/run_holarch_continual.py --steps 150
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from henri.holarch.config import HolarchConfig
from henri.holarch.pipeline import HenriHolarch
from henri.holarch.tasks import build_tasks
from henri.holarch.train import Trainer, TrainConfig


def accuracy(model, task, permutation=None, limit=300, rule=None, conditioned=False) -> float:
    correct = 0
    total = 0
    for batch in task.epoch(64, "val"):
        x = batch.x
        y = batch.y
        if permutation is not None:
            x = x.reshape(x.shape[0], -1)[:, permutation].reshape(x.shape)
        if rule is not None:
            y = (y + int(rule)) % 10
        output = model(
            "vision",
            x,
            training=False,
            use_memory=False,
            rule_id=(int(rule) if conditioned else None),
        )
        correct += int((output.predicted() == y).sum())
        total += int(y.numel())
        if total >= limit:
            break
    return correct / max(total, 1)


def fast_adapt(model, stream, lr=0.2, gate=True) -> float:
    """Delta-rule adaptation. Returns the mean plasticity gate rate."""
    rates: List[float] = []
    for x, y in stream:
        output = model("vision", x, training=False, use_memory=False)
        rates.append(float(model.plasticity_gate(output).mean()) if model.cfg.fast_gate else 1.0)
        model.fast_learn(output, y, lr=lr, gate=gate)
    return float(np.mean(rates)) if rates else 0.0


def replay_adapt(model, stream, old_x, old_y, lr=1e-3, gate=True) -> float:
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    generator = torch.Generator().manual_seed(7)
    rates: List[float] = []
    for x, y in stream:
        index = torch.randint(0, old_x.shape[0], (x.shape[0],), generator=generator)
        mix_x = torch.cat([x, old_x[index]])
        mix_y = torch.cat([y, old_y[index]])
        output = model("vision", mix_x, training=True, use_memory=False)
        loss = F.cross_entropy(output.task_logits, mix_y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        fresh = model("vision", x, training=False, use_memory=False)
        if model.cfg.fast_gate:
            rates.append(float(model.plasticity_gate(fresh).mean()))
        model.fast_learn(fresh, y, lr=0.2, gate=gate)
    return float(np.mean(rates)) if rates else 1.0


def train_checkpoint(cfg, task, steps: int, seed: int, engram_write: bool = False):
    torch.manual_seed(cfg.seed)
    path = Path(tempfile.gettempdir()) / f"henri_continual_{seed}_{steps}.sqlite3"
    if path.exists():
        path.unlink()
    model = HenriHolarch(cfg, backend="sqlite", memory_path=str(path))
    trainer = Trainer(
        model, {"vision": task}, TrainConfig(steps_per_task=int(steps), batch_size=32, seed=seed)
    )
    trainer.flags["engram_write"] = engram_write
    trainer.fit(steps_per_task=int(steps), order=["vision"])
    model.eval()
    return model


def clone_model(model, cfg, tag: str):
    """A parameter copy with its own Zone C handle.

    copy.deepcopy cannot be used: the memory carries a threading lock.
    """
    path = Path(tempfile.gettempdir()) / f"henri_clone_{tag}.sqlite3"
    if path.exists():
        path.unlink()
    clone = HenriHolarch(cfg, backend="sqlite", memory_path=str(path))
    clone.load_state_dict(model.state_dict())
    return clone


def train_rule_checkpoint(cfg, task, steps: int, rules, conditioned: bool, seed: int):
    """Train on a mixture of label rotations, with or without context.

    Every batch holds both rules at once, so the model cannot separate them
    by step order. With context it must condition on the slot-3 phasor
    within the batch; without context the two rules are a contradiction on
    identical pixels and the best it can do is split the difference.
    """
    torch.manual_seed(cfg.seed)
    path = Path(tempfile.gettempdir()) / f"henri_rule_{seed}_{steps}_{int(conditioned)}.sqlite3"
    if path.exists():
        path.unlink()
    model = HenriHolarch(cfg, backend="sqlite", memory_path=str(path))
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=2e-3, weight_decay=1e-4
    )
    per_rule = max(8, int(32 / max(1, len(rules))))
    for _ in range(int(steps)):
        parts = [task.sample(per_rule, "train") for _ in rules]
        x = torch.cat([part.x for part in parts])
        labels = torch.cat([(part.y + rule) % 10 for part, rule in zip(parts, rules)])
        ids = [int(rule) for part, rule in zip(parts, rules) for _ in range(part.x.shape[0])]
        output = model(
            "vision",
            x,
            training=True,
            use_memory=False,
            rule_id=(ids if conditioned else None),
        )
        loss = F.cross_entropy(output.task_logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval()
    return model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--batches", type=int, default=24)
    parser.add_argument("--rules", default="0,3")
    parser.add_argument("--out", default="receipts/holarch/continual_vision.json")
    args = parser.parse_args()

    cfg = HolarchConfig.small(fast_gate=True)
    tasks = build_tasks(cfg, seed=cfg.seed)
    task = tasks["vision"]
    model = train_checkpoint(cfg, task, int(args.steps), cfg.seed)
    # Decision 4 needs a familiarity reference: the clean prototype bank.
    model.calibrate_prototypes("vision", [task.sample(64, "train") for _ in range(4)])
    gate_calibration = model.calibrate_gate(
        "vision", [task.sample(32, "val").x for _ in range(4)]
    )
    model.eval()

    anchors = {name: value.detach().clone() for name, value in model.state_dict().items()}
    permutation = torch.randperm(64, generator=torch.Generator().manual_seed(99))
    original_before = accuracy(model, task)
    shifted_before = accuracy(model, task, permutation)

    def shifted_batches():
        for _ in range(int(args.batches)):
            batch = task.sample(32, "train")
            x = batch.x.reshape(batch.x.shape[0], -1)[:, permutation].reshape(batch.x.shape)
            yield x, batch.y

    def familiar_batches():
        for _ in range(int(args.batches)):
            batch = task.sample(32, "train")
            yield batch.x, batch.y

    old_features: List[torch.Tensor] = []
    old_labels: List[torch.Tensor] = []
    for batch in task.epoch(64, "train"):
        old_features.append(batch.x)
        old_labels.append(batch.y)
        if sum(len(part) for part in old_labels) >= 512:
            break
    old_x = torch.cat(old_features)
    old_y = torch.cat(old_labels)

    rows: Dict[str, Dict[str, float]] = {
        "no_adaptation": {
            "shifted": shifted_before,
            "original": original_before,
            "gate_rate": 0.0,
        }
    }

    model.load_state_dict(anchors)
    rate = fast_adapt(model, shifted_batches(), gate=False)
    rows["fast_ungated"] = {
        "shifted": accuracy(model, task, permutation),
        "original": accuracy(model, task),
        "gate_rate": rate,
    }

    model.load_state_dict(anchors)
    rate = fast_adapt(model, shifted_batches(), gate=True)
    rows["fast_gated"] = {
        "shifted": accuracy(model, task, permutation),
        "original": accuracy(model, task),
        "gate_rate": rate,
    }

    model.load_state_dict(anchors)
    rate = replay_adapt(model, shifted_batches(), old_x, old_y, gate=False)
    rows["replay_ungated"] = {
        "shifted": accuracy(model, task, permutation),
        "original": accuracy(model, task),
        "gate_rate": rate,
    }

    model.load_state_dict(anchors)
    rate = replay_adapt(model, shifted_batches(), old_x, old_y, gate=True)
    rows["replay_gated"] = {
        "shifted": accuracy(model, task, permutation),
        "original": accuracy(model, task),
        "gate_rate": rate,
    }

    baseline = rows["no_adaptation"]["shifted"]
    for name, row in rows.items():
        row["shifted_gain"] = row["shifted"] - baseline
        row["retention"] = row["original"] / max(original_before, 1e-9)

    # B. Stationarity control: the same plastic path on familiar data.
    stationary: Dict[str, Dict[str, float]] = {}
    model.load_state_dict(anchors)
    rate = fast_adapt(model, familiar_batches(), gate=False)
    after = accuracy(model, task)
    stationary["fast_ungated"] = {
        "before": original_before,
        "after": after,
        "drift": after - original_before,
        "gate_rate": rate,
    }
    model.load_state_dict(anchors)
    rate = fast_adapt(model, familiar_batches(), gate=True)
    after = accuracy(model, task)
    stationary["fast_gated"] = {
        "before": original_before,
        "after": after,
        "drift": after - original_before,
        "gate_rate": rate,
    }

    # C. Rule shift with slot-3 context.
    rules = [int(part) for part in str(args.rules).split(",") if part.strip()]
    # The rule arm needs enough steps per rule to be a fair context test:
    # the base checkpoint gets args.steps for ONE rule, so the mixture gets
    # args.steps * len(rules).
    rule_steps = int(args.steps) * max(1, len(rules))
    rule_model = train_rule_checkpoint(cfg, task, rule_steps, rules, True, cfg.seed + 1)
    plain_model = train_rule_checkpoint(cfg, task, rule_steps, rules, False, cfg.seed + 1)
    context = {}
    for name, trained, conditioned in (
        ("context_bound", rule_model, True),
        ("unconditioned", plain_model, False),
    ):
        per_rule = {}
        for rule in rules:
            per_rule[f"rule_{rule}"] = accuracy(
                trained, task, rule=rule, conditioned=conditioned
            )
        per_rule["combined"] = float(np.mean(list(per_rule.values())))
        context[name] = per_rule

    conditioned_before = {
        name: accuracy(rule_model, task, rule=int(name.split("_")[1]), conditioned=True)
        for name in context["context_bound"] if name.startswith("rule_")
    }
    rule_stream = [
        (batch.x, (batch.y + 3) % 10)
        for batch in (task.sample(32, "train") for _ in range(int(args.batches)))
    ]

    adapted: Dict[str, Dict[str, float]] = {"no_adaptation": dict(conditioned_before)}
    for arm_name in ("fast", "replay"):
        trial = clone_model(rule_model, cfg, f"rule_{arm_name}")
        if arm_name == "fast":
            rate = fast_adapt(trial, rule_stream, gate=False)
        else:
            rate = replay_adapt(trial, rule_stream, old_x, old_y, gate=False)
        row = {name: accuracy(trial, task, rule=int(name.split("_")[1]), conditioned=True)
               for name in conditioned_before}
        row["fast_gate_rate"] = rate
        adapted[arm_name] = row
    for name, row in adapted.items():
        values = [value for key, value in row.items() if key.startswith("rule_")]
        row["combined"] = float(np.mean(values)) if values else 0.0

    payload = {
        "task": "vision-pixel-permutation",
        "steps": int(args.steps),
        "adaptation_batches": int(args.batches),
        "permutation_seed": 99,
        "original_before_shift": original_before,
        "gate_calibration": gate_calibration,
        "variants": rows,
        "stationary": stationary,
        "rule_shift": {
            "rules": rules,
            "context": context,
            "adaptation": adapted,
            "fano_bound": 0.5,
        },
        "gate": {
            "gated_stationary_drift_at_most_0_005": bool(
                abs(stationary["fast_gated"]["drift"]) <= 0.005
            ),
            # Decision 4 cites the +0.513 gain of the fast + replay arm,
            # so that is the arm whose gain the gate must preserve.
            "gated_keeps_replay_gain": bool(
                rows["replay_gated"]["shifted_gain"]
                >= 0.9 * max(rows["replay_ungated"]["shifted_gain"], 0.0)
            ),
            "context_combined_above_conditioned_baseline": bool(
                adapted["no_adaptation"]["combined"]
                > context["unconditioned"]["combined"]
            ),
            "context_bound_combined_above_fano": bool(
                context["context_bound"]["combined"] > 0.5
            ),
            "context_beats_unconditioned": bool(
                context["context_bound"]["combined"] > context["unconditioned"]["combined"]
            ),
        },
        "evidence": "OBSERVED",
    }
    payload["gate"]["passed"] = bool(
        payload["gate"]["gated_stationary_drift_at_most_0_005"]
        and payload["gate"]["gated_keeps_replay_gain"]
        and payload["gate"]["context_bound_combined_above_fano"]
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    (out.parent / (out.name + ".sha256")).write_text(
        hashlib.sha256(out.read_bytes()).hexdigest() + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["gate"], indent=2, default=float))
    model.memory.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
