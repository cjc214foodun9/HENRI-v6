"""Training, evaluation, and receipts for the HENRI holarch.

Evidence label: OBSERVED for every number this module returns, because
each one is computed from a forward pass on a named split. No score is
estimated from a formula.

The loop is deliberately plain. Four learning paths run at different
rates, and the receipt records all four so an ablation can switch any of
them off and attribute the difference:

    gradient      task cross entropy through the photonic simulation
    fast          Zone A delta-rule write, every batch
    replay        slow-weight consolidation on a bounded buffer
    engram        Zone C append of the observed wave
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from henri.holarch.config import HolarchConfig, TASK_IDS, TASK_NAMES
from henri.holarch.model import evaluate_closed_loop_action, save_checkpoint
from henri.holarch.pipeline import HenriHolarch, HolarchOutput
from henri.holarch.zone_a import ReplayBuffer


@dataclass
class TrainConfig:
    """Optimisation and memory-write schedule."""

    steps_per_task: int = 150
    relational_multiplier: float = 2.0
    relational_isolated_bonus: bool = True
    relational_bonus_coupling: float = 0.5
    """Share of the bonus relational gradient that reaches shared modules.

    0.0 is full isolation (Decision B1 as specified). 1.0 is the retired
    fully coupled bonus pass of Decision A4. The measured trade-off curve
    is recorded in section 18.1 of docs/HOLARCH.md: the default is the
    smallest coupling that still lets the shared optical stack adapt to
    the relational task without rotating away from action.
    """
    """Decision B1: the bonus relational passes are gradient-isolated.

    The first relational step of a round is fully coupled. Each extra step
    detaches the shared post-photonic wave, so only the relational basis,
    the relational phase mask and the relational head update. The shared
    Zone B plates therefore see one relational gradient per round, the same
    as every other task, and the action interference of Decision A4 ends.
    """
    """Decision A4: relational steps per base step.

    Circular-convolution binding needs 600 steps to lock its phase, while
    the dense heads converge by 300. The default therefore trains the
    relational task at twice the base budget, so train --steps 300 runs
    the canonical 600 relational steps that Gate D4 measured.
    """
    batch_size: int = 32
    lr: float = 2e-3
    weight_decay: float = 1e-4
    fast_lr: float = 0.05
    grad_clip: float = 1.0
    aux_weight: float = 0.01
    memory_every: int = 4
    max_writes: int = 4
    replay_capacity: int = 3072
    consolidate_every: int = 40
    consolidate_steps: int = 1
    ood_weight: float = 3.0
    ood_margin: float = 0.0
    prototype_refresh: int = 50
    """Steps between in-training prototype rebuilds. 0 disables.

    Decision 6 couples the Zone C prototype logits to the task readout, so
    the head needs the bank to exist while it trains. A rebuild every N
    steps per task gives it one without leaking labels: the bank is the
    class mean of the model's own clean waves.
    """
    eval_every: int = 0
    eval_limit: int = 600
    seed: int = 20261102


@dataclass
class History:
    """Per-step trace plus the final evaluation."""

    steps: List[Dict[str, float]] = field(default_factory=list)
    eval: Dict[str, float] = field(default_factory=dict)
    wall_seconds: float = 0.0

    def summary(self) -> Dict[str, float]:
        if not self.steps:
            return {}
        last = self.steps[-1]
        return {
            "steps": float(len(self.steps)),
            "final_loss": float(last.get("loss", float("nan"))),
            "final_aux": float(last.get("aux", float("nan"))),
            "mean_loss": float(np.mean([s["loss"] for s in self.steps])),
            "wall_seconds": float(self.wall_seconds),
            **self.eval,
        }


class Trainer:
    """Runs the four learning paths against the three tasks."""

    def __init__(self, model: HenriHolarch, tasks: Dict[str, object], config: TrainConfig) -> None:
        self.model = model
        self.tasks = tasks
        self.config = config
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=float(config.lr),
            weight_decay=float(config.weight_decay),
        )
        self.replay = {
            name: ReplayBuffer(capacity=int(config.replay_capacity), seed=int(config.seed) + index)
            for index, name in enumerate(TASK_NAMES)
        }
        self.history = History()
        self._step = 0
        self._task_steps = {name: 0 for name in TASK_NAMES}
        self._relational_steps = 0
        self.flags: Dict[str, bool] = {
            "use_photonic": True,
            "use_swarm": True,
            "use_memory": True,
            "use_optical_features": True,
            "fast_learn": True,
            "engram_write": True,
        }

    def step(self, task_name: str, isolate: bool = False) -> Dict[str, float]:
        """One optimiser step for one task.

        Args:
            task_name: task key.
            isolate: Decision B1. Detach the shared post-photonic wave on the
                relational branch and drop every loss term that would train a
                shared module (the load-balance auxiliary), so this step can
                only update relational parameters.
        """
        task = self.tasks[task_name]
        batch = task.sample(int(self.config.batch_size), "train")
        # Decision B1: a positive coupling keeps the shared graph live and
        # scales its gradient, so the detach path is only used at coupling
        # exactly zero.
        shared_coupling = float(self.config.relational_bonus_coupling) if isolate else 1.0
        hard_isolate = bool(isolate) and shared_coupling <= 0.0
        output = self.model(
            task_name,
            batch.x,
            training=True,
            use_photonic=self.flags["use_photonic"],
            use_swarm=self.flags["use_swarm"],
            use_memory=self.flags["use_memory"],
            use_optical_features=self.flags["use_optical_features"],
            isolate_branch=hard_isolate,
        )
        loss = F.cross_entropy(output.task_logits, batch.y.long())
        if isolate:
            # A detached pass must not reach the shared router either: the
            # auxiliary load-balance term is a gradient path into it.
            total = loss
        else:
            total = loss + float(self.config.aux_weight) * output.swarm.load_balance
        if task_name == "relational":
            # Decision 3 right-sizes this penalty. The retired 0.05 was
            # measured to hold training accuracy at 0.696 on the standard
            # split while the same head reached 0.915 on the low-data split.
            l2 = (
                self.model.relational_proj.weight.pow(2).sum()
                + self.model.relational_gate.pow(2).sum()
            )
            if self.model.trilinear_head is not None:
                l2 = l2 + self.model.trilinear_head.weight.pow(2).sum()
            if getattr(self.model, "relational_expert", None) is not None:
                for parameter in self.model.relational_expert.parameters():
                    l2 = l2 + parameter.pow(2).sum()
            total = total + float(self.model.cfg.relational_l2) * l2
        if (
            not isolate
            and float(self.config.ood_weight) > 0.0
            and task_name in {"vision", "action"}
        ):
            # Outlier exposure: push the wave of a uniform input away from the
            # wave of a real input. This is what makes the raw Gram statistic a
            # usable OOD signal instead of a near-constant.
            clean = self.model.encode(task_name, batch.x)
            noise_input = torch.rand_like(batch.x)
            noise = self.model.encode(task_name, noise_input)
            coherence = (clean.conj() * noise).sum(dim=-1).real.abs()
            total = total + float(self.config.ood_weight) * F.relu(
                coherence - float(self.config.ood_margin)
            ).mean()
        self.optimizer.zero_grad(set_to_none=True)
        handles = []
        if isolate and 0.0 < shared_coupling < 1.0:
            for name, parameter in self.model.named_parameters():
                if parameter.requires_grad and not _is_relational_parameter(name):
                    handles.append(
                        parameter.register_hook(
                            lambda grad, scale=shared_coupling: grad * scale
                        )
                    )
        total.backward()
        for handle in handles:
            handle.remove()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config.grad_clip))
        self.optimizer.step()

        if self.flags["fast_learn"] and not isolate:
            fast = self.model.fast_learn(output, batch.y, lr=float(self.config.fast_lr))
        else:
            fast = {}
        self.replay[task_name].add(output.features.detach(), batch.y, int(_task_id(task_name)))

        if (
            not isolate
            and self.flags["engram_write"]
            and self._step % int(self.config.memory_every) == 0
        ):
            self.model.observe(task_name, batch.x, batch.y, max_writes=int(self.config.max_writes))
        if (
            not isolate
            and self._step % int(self.config.consolidate_every) == 0
            and len(self.replay[task_name]) > 0
        ):
            features, labels = self.replay[task_name].sample(
                int(_task_id(task_name)), int(self.config.batch_size)
            )
            self.model.zone_a.consolidate(
                features, labels, int(_task_id(task_name)), self.optimizer,
                steps=int(self.config.consolidate_steps),
            )
        self._step += 1
        self._task_steps[task_name] = self._task_steps.get(task_name, 0) + 1
        if task_name == "relational":
            self._relational_steps += 1
        refresh = int(self.config.prototype_refresh)
        if (
            refresh > 0
            and self._task_steps[task_name] % refresh == 0
            and self._task_steps[task_name] <= 4 * refresh
        ):
            self.model.calibrate_prototypes(
                task_name, [task.sample(64, "train") for _ in range(2)]
            )
        record = {
            "step": float(self._step),
            "task": float(_task_id(task_name)),
            "loss": float(loss.detach()),
            "aux": float(output.swarm.load_balance.detach()),
            "veto": float(output.veto.float().mean()),
            "delta": float(output.photonic.delta_memory.mean().detach()),
            "fast_loss": float(np.mean([v for k, v in fast.items() if k.startswith("expert_")] or [0.0])),
            "isolated": float(bool(isolate)),
        }
        self.history.steps.append(record)
        return record

    def dagger_round(self, episodes: int = 8, steps: int = 2, lr: Optional[float] = None) -> float:
        """On-policy correction for the action task.

        Roll the current policy in the grid, label every visited state
        with the BFS optimum, and take a few gradient steps on that set.
        The loop is the standard DAgger fix for compounding error; it is
        also the clearest continuous-learning demonstration in the build.
        """
        task = self.tasks["action"]
        if not hasattr(task, "rollout_pairs"):
            return 0.0
        pairs = task.rollout_pairs(lambda observation: self.model.act(observation), episodes=episodes)
        if len(pairs) < 8:
            return 0.0
        x = torch.from_numpy(np.stack([pair[0] for pair in pairs]))
        y = torch.tensor([pair[1] for pair in pairs], dtype=torch.long)
        total = 0.0
        for _ in range(max(1, int(steps))):
            output = self.model("action", x, training=True)
            loss = F.cross_entropy(output.task_logits, y)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config.grad_clip))
            self.optimizer.step()
            total += float(loss.detach())
        return total / max(1, int(steps))

    def fit(
        self,
        steps_per_task: Optional[int] = None,
        order: Sequence[str] = TASK_NAMES,
        relational_multiplier: Optional[float] = None,
    ) -> History:
        """Run the per-task schedule.

        Decision A4: the relational task runs at relational_multiplier
        times the base step count (2x by default, so 600 steps at
        --steps 300). The extra passes happen inside the same round, so
        the relational head sees its phase-locking budget without changing
        the budget of the other three tasks.
        """
        steps = int(self.config.steps_per_task if steps_per_task is None else steps_per_task)
        multiplier = float(
            self.config.relational_multiplier
            if relational_multiplier is None
            else relational_multiplier
        )
        extra = max(0, int(round(multiplier)) - 1)
        start = time.perf_counter()
        for round_index in range(steps):
            for task_name in order:
                # Decision A4: the extra relational passes run just BEFORE
                # the action step when both tasks are scheduled. The step
                # budget is identical, but the action head is no longer
                # trained two relational gradients behind the shared
                # stack. The first measured form of the schedule cost
                # 0.09 action accuracy to recency alone.
                if (
                    extra
                    and task_name == "action"
                    and "relational" in order
                ):
                    for _ in range(extra):
                        self.step(
                            "relational",
                            isolate=bool(self.config.relational_isolated_bonus),
                        )
                self.step(task_name)
                if (
                    extra
                    and task_name == "relational"
                    and "action" not in order
                ):
                    for _ in range(extra):
                        self.step(
                            task_name,
                            isolate=bool(self.config.relational_isolated_bonus),
                        )
        self.history.wall_seconds = time.perf_counter() - start
        return self.history


def _task_id(name: str) -> int:
    return int(TASK_IDS[name])


def _is_relational_parameter(name: str) -> bool:
    """True for parameters that belong to the relational task alone."""
    return name.startswith(("relational_", "encoders.relational", "projector.relational"))


@torch.no_grad()
def evaluate(
    model: HenriHolarch,
    tasks: Dict[str, object],
    split: str = "val",
    limit: int = 600,
    batch_size: int = 64,
    flags: Optional[Dict[str, bool]] = None,
) -> Dict[str, float]:
    """Measure accuracy, selective accuracy, memory recall, and latency."""
    model.eval()
    metrics: Dict[str, float] = {}
    for name in TASK_NAMES:
        task = tasks[name]
        correct = 0
        total = 0
        vetoed = 0
        selective_correct = 0
        covered = 0
        loss_sum = 0.0
        batches = 0
        started = time.perf_counter()
        for batch in task.epoch(int(batch_size), split):
            output = model(
                name,
                batch.x,
                training=False,
                use_photonic=(flags or {}).get("use_photonic", True),
                use_swarm=(flags or {}).get("use_swarm", True),
                use_memory=(flags or {}).get("use_memory", True),
                use_optical_features=(flags or {}).get("use_optical_features", True),
            )
            predicted = output.predicted()
            mask = output.veto
            hit = predicted == batch.y
            correct += int(hit.sum())
            total += int(batch.y.numel())
            vetoed += int(mask.sum())
            keep = ~mask
            if bool(keep.any()):
                covered += int(keep.sum())
                selective_correct += int((hit & keep).sum())
            loss_sum += float(F.cross_entropy(output.task_logits, batch.y))
            batches += 1
            if total >= int(limit):
                break
        elapsed = time.perf_counter() - started
        metrics[f"{name}_accuracy"] = correct / max(total, 1)
        metrics[f"{name}_loss"] = loss_sum / max(batches, 1)
        metrics[f"{name}_perplexity"] = math.exp(min(loss_sum / max(batches, 1), 20.0))
        metrics[f"{name}_veto_rate"] = vetoed / max(total, 1)
        metrics[f"{name}_coverage"] = covered / max(total, 1)
        metrics[f"{name}_selective_accuracy"] = selective_correct / max(covered, 1)
        metrics[f"{name}_ms_per_sample"] = (elapsed * 1e3) / max(total, 1)
        metrics[f"{name}_n"] = float(total)
    return metrics


@torch.no_grad()
def memory_recall(
    model: HenriHolarch,
    task,
    split: str = "val",
    limit: int = 200,
) -> Dict[str, float]:
    """Top-1 label recall of the Zone C index on held-out waves."""
    if getattr(model, "memory", None) is None:
        return {"recall_at_1": float("nan"), "n": 0.0}
    batch = task.sample(int(limit), split)
    waves = model.encode(task.name, batch.x).numpy()
    hits = 0
    used = 0
    from henri.holarch.zone_c import global_label

    offset = global_label(task.name, 0)
    for index, wave in enumerate(waves):
        _, read = model.memory.context_tensor(wave, k=1, task=task.name)
        if not read.hits:
            continue
        used += 1
        if int(read.hits[0].label) - offset == int(batch.y[index]):
            hits += 1
    return {"recall_at_1": hits / max(used, 1), "n": float(used)}


def action_success(model: HenriHolarch, task, episodes: int = 48) -> float:
    """Closed-loop success rate of the model as a grid-world policy."""
    return float(task.policy_success(lambda observation: model.act(observation), episodes=episodes))


def git_head() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_experiment(
    cfg: HolarchConfig,
    output_dir: str | Path = "receipts/holarch",
    name: str = "holarch",
    steps_per_task: int = 150,
    backend: str = "auto",
    memory_path: Optional[str] = None,
    tasks: Optional[Dict[str, object]] = None,
    train_config: Optional[TrainConfig] = None,
    seed: Optional[int] = None,
    calibrate: bool = True,
    evaluate_limits: Optional[Dict[str, int]] = None,
    dagger_rounds: int = 8,
    reset_memory: bool = False,
) -> Dict[str, float]:
    """Train, evaluate, and write one reproducible receipt."""
    from henri.holarch.budget import budget_report
    from henri.holarch.tasks import build_tasks

    receipt_name = str(name)
    train_config = train_config or TrainConfig(
        steps_per_task=int(steps_per_task), seed=int(cfg.seed if seed is None else seed)
    )
    tasks = tasks or build_tasks(cfg, seed=cfg.seed if seed is None else seed)
    torch.manual_seed(int(cfg.seed))
    model = HenriHolarch(cfg, backend=backend, memory_path=memory_path)
    preexisting = int(model.memory.stats()["count"]) if getattr(model, "memory", None) else 0
    if reset_memory and getattr(model, "memory", None) is not None:
        model.memory.reset()
        preexisting = 0
    start = time.perf_counter()
    trainer = Trainer(model, tasks, train_config)
    history = trainer.fit(steps_per_task=train_config.steps_per_task)
    dagger_losses = [
        trainer.dagger_round(episodes=10, steps=2) for _ in range(int(dagger_rounds))
    ]

    if calibrate and getattr(model, "memory", None) is not None:
        calibration = {}
        for name in TASK_NAMES:
            # Decision A3: 32 batches (1024 waves), not 8. A 256-wave
            # sample cannot resolve a 0.5 percent tail: its own expected
            # count there is 1.3 events, and the measured run-to-run
            # false-veto spread was 0.000 to 0.013. At 1024 waves the 0.5
            # percent tail holds five events and the empirical/Cornish-
            # Fisher cap has something to estimate.
            batches = [tasks[name].sample(32, "val").x for _ in range(32)]
            # Decision A3: both conditions are fitted on the same paired
            # sample, so the returned report carries the measured union
            # false-veto rate of the rule that is actually deployed.
            calibration[name] = model.calibrate_two_condition(
                name, batches, ceiling=0.98, target_false_veto=0.005
            )
        # Decision 2: build the clean post-photonic attractor bank from the
        # training split, then let relaxation read it. This runs BEFORE the
        # evaluation so the deployed loop is the one being measured.
        prototypes = {}
        for name in TASK_NAMES:
            clean = [tasks[name].sample(64, "train") for _ in range(4)]
            prototypes[name] = model.calibrate_prototypes(name, clean)
            # The surprise threshold needs the bank, so it is calibrated
            # after the prototypes and before any evaluation.
            calibration[name].update(
                model.calibrate_gate(name, [tasks[name].sample(32, "val").x for _ in range(4)])
            )
    else:
        calibration = {
            name: {
                "epsilon": float(model.veto_epsilon(name)),
                "tau_chance": float(model.chance_epsilon(name)),
                "calibrated": 0.0,
            }
            for name in TASK_NAMES
        }
        prototypes = {name: model.prototypes[name].report() for name in TASK_NAMES}

    limits = evaluate_limits or {}
    metrics = evaluate(model, tasks, "val", limit=int(limits.get("val", 600)))
    test_metrics = evaluate(model, tasks, "test", limit=int(limits.get("test", 600)))
    recall = memory_recall(model, tasks["vision"], split="val", limit=200)
    # Cycle E E1: the deployed closed-loop policy is the 1-step latent wave
    # lookahead. The reactive arm is measured on the same 48 layouts and
    # recorded beside it, so the delta is attributable.
    reactive_success = action_success(model, tasks["action"], episodes=48)
    lookahead_eval = evaluate_closed_loop_action(
        model, tasks["action"], episodes=48, limit=20
    )
    success = float(lookahead_eval["success_rate"])
    success_unmasked = tasks["action"].policy_success(
        lambda observation: model.act(observation, mask_illegal=False), episodes=48
    )
    # Relaxation attribution: same checkpoint, loop off, same episodes.
    model.cfg = cfg.replace(relax_steps=0)
    success_no_relax = action_success(model, tasks["action"], episodes=48)
    model.cfg = cfg

    # Directive 2 gate: uniform pixel noise against the vision veto.
    generator = torch.Generator().manual_seed(int(cfg.seed) + 99)
    ood_rates = []
    noise_residuals = []
    for _ in range(8):
        noise = torch.rand(32, 1, 8, 8, generator=generator)
        with torch.no_grad():
            ood = model("vision", noise, training=False)
        ood_rates.append(float(ood.veto.float().mean()))
        noise_residuals.extend(float(v) for v in ood.delta_sagnac.cpu())
    ood_veto = float(np.mean(ood_rates))
    # Decision D1 diagnostics: the clean and noise residual ranges that the
    # veto separates, so the receipt records the margin instead of only the
    # final rate.
    clean_residuals = []
    with torch.no_grad():
        for _ in range(4):
            clean_batch = tasks["vision"].sample(32, "val")
            clean = model("vision", clean_batch.x, training=False, relax=False)
            mask = clean.memory_mask
            if bool(mask.any()):
                clean_residuals.extend(float(v) for v in clean.delta_sagnac[mask].cpu())
    veto_diagnostics = {
        "clean_residual_max": float(np.max(clean_residuals)) if clean_residuals else 0.0,
        "clean_residual_mean": float(np.mean(clean_residuals)) if clean_residuals else 0.0,
        "noise_residual_min": float(np.min(noise_residuals)) if noise_residuals else 0.0,
        "noise_residual_mean": float(np.mean(noise_residuals)) if noise_residuals else 0.0,
        "tau_residual": float(model.veto_epsilon("vision")),
        "clean_waves": float(len(clean_residuals)),
        "noise_waves": float(len(noise_residuals)),
    }
    in_dist_veto = float(metrics.get("vision_veto_rate", float("nan")))
    # Persist the attractor bank as axiomatic Zone C engrams after the
    # recall and success metrics, so the memory numbers stay comparable to
    # the previous revision and the store is still the home of the bank.
    prototype_writes = model.store_prototypes() if hasattr(model, "store_prototypes") else {}
    # Cycle E: the converged weights and the calibrated prototype bank are
    # written beside the receipt, so the sidecar can mount exactly the model
    # this receipt measured instead of bootstrapping a new one.
    checkpoint = save_checkpoint(
        model,
        Path(output_dir) / f"{receipt_name}.pt",
        cfg=cfg,
        name=receipt_name,
        train_config=asdict(train_config),
        metrics={**metrics, **test_metrics, "action_success": success},
        seed=int(cfg.seed if seed is None else seed),
    )
    budget = budget_report(model, train_config.batch_size, cfg)
    elapsed = time.perf_counter() - start

    cuda_available = bool(torch.cuda.is_available())
    environment = {
        "cuda": {
            "available": cuda_available,
            "device_count": int(torch.cuda.device_count()) if cuda_available else 0,
            "device_name": torch.cuda.get_device_name(0) if cuda_available else "",
            "evidence": "OBSERVED" if cuda_available else "BLOCKED",
            "reason": ""
            if cuda_available
            else "no CUDA device is visible to torch in this checkout",
        },
        "storage": {
            "backend": str(getattr(model.memory, "backend", "unknown")),
            "backend_reason": str(getattr(model.memory, "backend_reason", "")),
            "timescale_dsn_env_set": bool(os.environ.get("HENRI_TSDB_DSN")),
            "evidence": "OBSERVED",
        },
    }
    receipt = {
        "name": receipt_name,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": git_head(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "platform": platform.platform(),
        "device": cfg.resolved_device(),
        "environment": environment,
        "config": {key: getattr(cfg, key) for key in cfg.__dataclass_fields__},
        "config_digest": cfg.digest(),
        "train_config": asdict(train_config),
        "parameters": model.parameter_count(),
        "seed": int(cfg.seed if seed is None else seed),
        "calibration": calibration,
        "prototypes": prototypes,
        "prototype_writes": prototype_writes,
        "checkpoint": checkpoint,
        "proto_alpha": float(model.proto_alpha.detach()),
        "relax_mode": str(cfg.relax_mode),
        "relational_binding": str(cfg.relational_binding),
        "readout": str(cfg.readout),
        "vision_encoder": str(cfg.vision_encoder),
        "vision_size": int(cfg.vision_size),
        "fast_gate": bool(cfg.fast_gate),
        "metrics": metrics,
        "test": test_metrics,
        "memory": recall,
        "action_success": success,
        "action_success_reactive": reactive_success,
        "closed_loop_lookahead": lookahead_eval,
        "action_success_unmasked": success_unmasked,
        "action_success_no_relax": success_no_relax,
        "ood_veto_rate": ood_veto,
        "in_distribution_veto_rate": in_dist_veto,
        "veto_diagnostics": veto_diagnostics,
        "gate_d2": {
            "ood_veto_at_least_0_95": bool(ood_veto >= 0.95),
            "false_veto_at_most_0_02": bool(in_dist_veto <= 0.02),
            "false_veto_at_most_0_01": bool(in_dist_veto <= 0.01),
            "passed": bool(ood_veto >= 0.95 and in_dist_veto <= 0.02),
            "a3_passed": bool(ood_veto >= 0.99 and in_dist_veto <= 0.01),
        },
        "relational_steps": int(trainer._relational_steps),
        "relational_multiplier": float(train_config.relational_multiplier),
        "dagger_rounds": int(dagger_rounds),
        "dagger_loss_mean": float(np.mean(dagger_losses)) if dagger_losses else 0.0,
        "memory_stats": model.memory_report(),
        "memory_preexisting": preexisting,
        "budget": budget,
        "history": history.summary(),
        "wall_seconds": elapsed,
        "evidence": "OBSERVED",
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{receipt_name}.json"
    path.write_text(json.dumps(receipt, indent=2, default=float), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (out / f"{receipt_name}.json.sha256").write_text(digest + "\n", encoding="utf-8")
    receipt["receipt_path"] = str(path)
    receipt["receipt_sha256"] = digest
    model.eval()
    return receipt
