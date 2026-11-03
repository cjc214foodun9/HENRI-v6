"""The HENRI holarch: one wired inference and learning pipeline.

Evidence label: DERIVED for the wiring; OBSERVED for the metrics a run
records. This module composes the three zones and nothing else.

Data path
---------
    tokens / pixels / sensors
        -> modality encoder            (real latent)
        -> wave projector              (complex wave on S^{D-1})
        -> Zone C read                 (holographic context, bounded tensor)
        -> wave fusion                 (psi_in + gate * context)
        -> Zone B                      (BTO D2NN, Sagnac loop, heaters)
        -> Sagnac veto                 (calibrated delta against context)
        -> pooled photonic features    -> Zone A swarm -> task logits

Learning paths, kept separate on purpose
----------------------------------------
1. gradient: task cross entropy back through the photonic simulation.
2. fast: the swarm's delta-rule write, no backprop, after every batch.
3. optical: Zone B phase-conjugate imprint, no backprop, on request.
4. memory: an append-only engram write per observed batch.

Ablations switch each path off independently so attribution is measured,
not asserted.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from henri.holarch.config import TASK_IDS, TASK_NAMES, HolarchConfig
from henri.holarch.zone_a import LeanSwarm, ReplayBuffer, SwarmOutput
from henri.holarch.zone_b import PhotonicCore, PhotonicState, _unit
from henri.holarch.zone_c import MemoryRead, ZoneCMemory

_EPS = 1e-8


class WaveProjector(nn.Module):
    """Map a real latent to a complex wave on the unit hypersphere."""

    def __init__(self, d_latent: int, dim: int, seed: int = 0) -> None:
        super().__init__()
        self.amplitude = nn.Linear(int(d_latent), int(dim))
        self.phase = nn.Linear(int(d_latent), int(dim))
        generator = torch.Generator().manual_seed(int(seed))
        with torch.no_grad():
            self.amplitude.weight.normal_(0.0, 1.0 / math.sqrt(d_latent), generator=generator)
            self.amplitude.bias.fill_(0.5)
            # The phase bias is a fixed random carrier. Without it every input
            # maps to nearly the same wave and the Sagnac statistic collapses
            # to a constant (measured: chance delta 1e-4 before this line).
            self.phase.weight.normal_(0.0, 2.0 / math.sqrt(d_latent), generator=generator)
            self.phase.bias.uniform_(-math.pi, math.pi, generator=generator)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        # Decision 3: zero-mean, unit-variance layer normalisation before the
        # phase map. A raw uniform image has a DC mean of about 0.5, and the
        # first linear layer turns that constant into a phase cluster. Removing
        # the DC and fixing the scale separates uniform noise without the
        # outlier-exposure accuracy tax.
        scale = math.sqrt(float(latent.shape[-1]))
        centred = latent - latent.mean(dim=-1, keepdim=True)
        scaled = F.normalize(centred, dim=-1) * scale
        amplitude = F.softplus(self.amplitude(scaled) * 0.5) + 1e-4
        phase = self.phase(scaled)
        wave = torch.polar(amplitude, phase)
        return _unit(wave)


class VisionEncoder(nn.Module):
    """Small convolutional encoder for 8x8 digit images."""

    def __init__(self, d_latent: int, seed: int = 0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.fc = nn.Linear(32 * 4 * 4, int(d_latent))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        h = F.gelu(self.conv1(x))
        h = F.gelu(self.conv2(h))
        h = F.adaptive_avg_pool2d(h, (4, 4)).flatten(1)
        return F.gelu(self.fc(h))


class LanguageEncoder(nn.Module):
    """Dilated convolutional character encoder."""

    def __init__(self, vocab_size: int, d_latent: int, width: int = 96, max_len: int = 64) -> None:
        super().__init__()
        self.embed = nn.Embedding(int(vocab_size), 64)
        self.conv = nn.Conv1d(64, int(width), 3, padding=2, dilation=2)
        self.fc = nn.Linear(int(width), int(d_latent))
        position = torch.zeros(max_len, 64)
        for index in range(max_len):
            for column in range(0, 64, 2):
                position[index, column] = math.sin(index / (10000 ** (column / 64)))
                position[index, column + 1] = math.cos(index / (10000 ** (column / 64)))
        self.register_buffer("position", position.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError("language input must be (batch, sequence)")
        length = min(x.shape[1], self.position.shape[1])
        embedded = self.embed(x[:, :length]) + self.position[:, :length]
        hidden = F.gelu(self.conv(embedded.transpose(1, 2)))
        return F.gelu(self.fc(hidden[:, :, -1]))


class ActionEncoder(nn.Module):
    """MLP over the flattened local grid observation."""

    def __init__(self, obs_dim: int, d_latent: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(int(obs_dim), 128)
        self.fc2 = nn.Linear(128, int(d_latent))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.fc2(F.gelu(self.fc1(x))))


class RelationalEncoder(nn.Module):
    """Three role-specific embeddings for a compositional triple."""

    def __init__(self, entities: int, actions: int, targets: int, d_latent: int) -> None:
        super().__init__()
        self.entity = nn.Embedding(int(entities), int(d_latent))
        self.action = nn.Embedding(int(actions), int(d_latent))
        self.target = nn.Embedding(int(targets), int(d_latent))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.shape[1] != 3:
            raise ValueError("relational input must be (batch, 3)")
        return torch.stack(
            [
                self.entity(x[:, 0].long()),
                self.action(x[:, 1].long()),
                self.target(x[:, 2].long()),
            ],
            dim=1,
        )


class SlotWaveProjector(nn.Module):
    """Map each role latent into its own wave slot."""

    def __init__(self, d_slot: int, d_latent: int, seed: int = 0) -> None:
        super().__init__()
        self.amplitude = nn.Linear(int(d_latent), int(d_slot))
        self.phase = nn.Linear(int(d_latent), int(d_slot))
        generator = torch.Generator().manual_seed(int(seed))
        with torch.no_grad():
            self.amplitude.weight.normal_(0.0, 1.0 / math.sqrt(d_latent), generator=generator)
            self.amplitude.bias.fill_(0.5)
            self.phase.weight.normal_(0.0, 2.0 / math.sqrt(d_latent), generator=generator)
            self.phase.bias.uniform_(-math.pi, math.pi, generator=generator)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        scale = math.sqrt(float(latent.shape[-1]))
        centred = latent - latent.mean(dim=-1, keepdim=True)
        scaled = F.normalize(centred, dim=-1) * scale
        amplitude = F.softplus(self.amplitude(scaled) * 0.5) + 1e-4
        phase = self.phase(scaled)
        return torch.polar(amplitude, phase)


@dataclass
class HolarchOutput:
    """One forward pass with its full audit trail."""

    task: str
    task_logits: torch.Tensor
    logits: torch.Tensor
    psi_in: torch.Tensor
    psi_fused: torch.Tensor
    photonic: PhotonicState
    swarm: SwarmOutput
    features: torch.Tensor
    context: torch.Tensor
    memory: Optional[MemoryRead]
    memory_mask: torch.Tensor
    veto: torch.Tensor
    margin: torch.Tensor
    s_max: torch.Tensor
    delta_sagnac: torch.Tensor

    @property
    def probs(self) -> torch.Tensor:
        return torch.softmax(self.task_logits, dim=-1)

    def predicted(self) -> torch.Tensor:
        return self.task_logits.argmax(dim=-1)

    def selective(self) -> torch.Tensor:
        """Predictions with vetoed rows replaced by -1 (abstain)."""
        labels = self.predicted()
        return torch.where(self.veto, torch.full_like(labels, -1), labels)

    def summary(self) -> Dict[str, float]:
        return {
            "veto_rate": float(self.veto.float().mean()),
            "margin": float(self.margin.mean()),
            "s_max": float(self.s_max.mean()),
            "delta_sagnac": float(self.delta_sagnac.mean().detach()),
            **self.photonic.summary(),
        }


class HenriHolarch(nn.Module):
    """The three-zone system as one trainable module."""

    def __init__(
        self,
        cfg: HolarchConfig,
        memory: Optional[ZoneCMemory] = None,
        backend: str = "auto",
        memory_path: Optional[str] = None,
        dsn: Optional[str] = None,
        table: str = "zone_c_engram_timeseries",
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.dim = int(cfg.dim)
        self.d_latent = int(cfg.d_latent)
        self.d_feat = int(cfg.d_feat)

        self.encoders = nn.ModuleDict(
            {
                "vision": VisionEncoder(self.d_latent),
                "language": LanguageEncoder(int(cfg.vocab_size), self.d_latent),
                "action": ActionEncoder(int(cfg.obs_dim), self.d_latent),
                "relational": RelationalEncoder(
                    int(cfg.relational_entities),
                    int(cfg.relational_actions),
                    int(cfg.relational_targets),
                    self.d_latent,
                ),
            }
        )
        self.projector = nn.ModuleDict(
            {
                name: WaveProjector(self.d_latent, self.dim, seed=int(cfg.seed) + 300 + index)
                for index, name in enumerate(("vision", "language", "action"))
            }
        )
        self.projector["relational"] = SlotWaveProjector(
            int(cfg.d_slot), self.d_latent, seed=int(cfg.seed) + 350
        )
        self.zone_b = PhotonicCore(cfg)
        self.feature_proj = nn.Linear(self.zone_b.raw_width, self.d_feat)
        self.relational_proj = nn.Linear(
            self.zone_b.relational_width, self.d_feat, bias=False
        )
        self.memory_proj = nn.Linear(
            int(cfg.context_tokens) * int(cfg.context_dim), self.d_feat, bias=False
        )
        # Skip path from the sensor embedding to the swarm. The photonic stage
        # is a reasoning operator, not a lossy codec: Zone A is allowed to see
        # the raw latent as well as the optical readout.
        self.latent_skip = nn.Linear(self.d_latent, self.d_feat)
        with torch.no_grad():
            self.memory_proj.weight.mul_(0.05)
        self.zone_a = LeanSwarm(cfg)
        self.trilinear_head = nn.Linear(int(cfg.pool_k), int(cfg.relational_classes))
        self.memory_gate = nn.Parameter(torch.tensor(-1.0))
        self.skip_alpha = float(cfg.skip_alpha)
        for name in TASK_NAMES:
            self.register_buffer(f"epsilon_{name}", torch.tensor(0.35))
            self.register_buffer(f"chance_{name}", torch.tensor(0.0))
        self.memory = memory if memory is not None else ZoneCMemory(
            cfg, backend=backend, path=memory_path, dsn=dsn, table=table
        )
        self.register_buffer("abstain", torch.tensor(0.0))

    # ------------------------------------------------------------- encoding
    def encode(self, task: str, x: torch.Tensor) -> torch.Tensor:
        """Encode raw modality data to a unit complex wave."""
        if task not in self.encoders:
            raise ValueError(f"unknown task {task!r}")
        latent = self.encoders[task](x)
        if task == "relational":
            return self._relational_wave(latent)
        return self.projector[task](latent)

    def _relational_wave(self, latent: torch.Tensor) -> torch.Tensor:
        """Bind the three role slots by pointwise phasor product.

        The product places the trilinear composition in slot 3, because
        multiplication of unit phasors adds their phases. That is VSA
        binding, and it is parameter free.
        """
        waves = self.projector["relational"](latent)
        bound = _unit((waves[:, 0] * waves[:, 1] * waves[:, 2]).unsqueeze(1))
        slots = torch.cat([waves, bound], dim=1)
        return _unit(slots.reshape(slots.shape[0], -1))

    def encode_observation(self, observation) -> torch.Tensor:
        """Encode one grid-world observation (numpy or tensor, 75 values)."""
        tensor = torch.as_tensor(np.asarray(observation, dtype=np.float32)).reshape(1, -1)
        return self.encode("action", tensor)

    # ------------------------------------------------------------ forward
    def forward(
        self,
        task: str,
        x: torch.Tensor,
        *,
        use_memory: bool = True,
        use_photonic: bool = True,
        use_swarm: bool = True,
        use_optical_features: bool = True,
        training: bool = False,
        memory_k: int = 4,
        skip_alpha: Optional[float] = None,
        relax: bool = True,
    ) -> HolarchOutput:
        """Run one batch through the holarch.

        Args:
            task: 'vision', 'language', or 'action'.
            x: raw modality tensor.
            use_memory: read Zone C and fuse its context wave.
            use_photonic: run Zone B. When False the wave passes through,
                which is the parameter-matched ablation of the photonic
                path.
            use_swarm: route through Zone A. When False one expert is
                used, which is the routing ablation.
            training: enables novelty-biased routing.
            memory_k: engrams fused per query.
        """
        if task not in TASK_IDS:
            raise ValueError(f"unknown task {task!r}")
        if task == "relational":
            latent = self.encoders[task](x)
            psi_in = self._relational_wave(latent)
            skip_vector = latent.mean(dim=1)
        else:
            latent = self.encoders[task](x)
            psi_in = self.projector[task](latent)
            skip_vector = latent
        batch = psi_in.shape[0]
        alpha = self.skip_alpha if skip_alpha is None else float(skip_alpha)

        context_wave = torch.zeros_like(psi_in)
        top_axiom = torch.zeros_like(psi_in)
        prefix = torch.zeros(
            batch, int(self.cfg.context_tokens) * int(self.cfg.context_dim)
        )
        reads: List[MemoryRead] = []
        has_hits = torch.zeros(batch, dtype=torch.bool, device=psi_in.device)
        s_max_values = [0.0] * batch
        read: Optional[MemoryRead] = None
        if use_memory and getattr(self, "memory", None) is not None:
            context_rows: List[torch.Tensor] = []
            axiom_rows: List[torch.Tensor] = []
            prefix_rows: List[np.ndarray] = []
            for index, row in enumerate(psi_in.detach().cpu().numpy()):
                prefix_np, row_read = self.memory.context_tensor(row, k=memory_k)
                prefix_rows.append(prefix_np.reshape(-1))
                reads.append(row_read)
                if row_read.hits:
                    context_rows.append(torch.from_numpy(row_read.context))
                    top = row_read.top_wave
                    if top is None:
                        top = row_read.context
                    axiom_rows.append(torch.from_numpy(np.asarray(top, dtype=np.complex64)))
                    has_hits[index] = True
                    s_max_values[index] = float(row_read.s_max)
                else:
                    context_rows.append(torch.zeros(self.dim, dtype=torch.complex64))
                    axiom_rows.append(torch.zeros(self.dim, dtype=torch.complex64))
            prefix = torch.from_numpy(np.stack(prefix_rows)).to(torch.float32)
            context_wave = _unit(torch.stack(context_rows).to(psi_in.dtype))
            top_axiom = _unit(torch.stack(axiom_rows).to(psi_in.dtype))
            read = reads[0] if reads else None

        gate = torch.sigmoid(self.memory_gate)
        fused = _unit(psi_in + gate * context_wave) if use_memory else psi_in
        s_max_tensor = torch.tensor(s_max_values, dtype=torch.float32, device=psi_in.device)

        axiom = top_axiom if (use_memory and bool(has_hits.any())) else None
        photonic = self.zone_b(fused, axiom=axiom, bypass=not use_photonic)
        if axiom is not None:
            photonic.delta_memory = torch.where(
                has_hits, photonic.delta_memory, torch.zeros_like(photonic.delta_memory)
            )
            mask_slots = has_hits.unsqueeze(-1)
            photonic.delta_slots = torch.where(
                mask_slots, photonic.delta_slots, torch.zeros_like(photonic.delta_slots)
            )

        tau_residual = self.veto_epsilon(task)
        delta_sagnac = photonic.delta_memory
        if (
            relax
            and not training
            and use_photonic
            and axiom is not None
            and int(self.cfg.relax_steps) > 0
            and bool((delta_sagnac > tau_residual).any())
        ):
            # Relax every row. Rows without a memory hit use their own wave as
            # the reference, so their residual is zero and the loop is a no-op
            # for them. Gating on has_hits.all() was a bug: one unmatched row
            # disabled relaxation for the whole batch.
            relax_axiom = torch.where(has_hits.unsqueeze(-1), axiom, fused)
            relaxed, _trace = self.zone_b.relax(
                fused, relax_axiom, tolerance=float(tau_residual)
            )
            relaxed.delta_memory = torch.where(
                has_hits, relaxed.delta_memory, torch.zeros_like(relaxed.delta_memory)
            )
            photonic = relaxed
            delta_sagnac = photonic.delta_memory

        if use_optical_features:
            raw = self.feature_proj(photonic.features[:, : self.zone_b.raw_width])
            if task == "relational":
                raw = raw + self.relational_proj(
                    photonic.features[:, self.zone_b.raw_width :]
                )
            optical = raw
        else:
            optical = 0.0
        features = optical + self.memory_proj(prefix) + alpha * self.latent_skip(skip_vector)
        features = F.gelu(features)

        swarm = self.zone_a(features, TASK_IDS[task], training=training)
        if not use_swarm:
            # Routing ablation: one fixed expert, parameter matched by
            # construction. The other experts still exist and are counted.
            single = swarm.expert_logits[0]
            swarm.logits = single
            swarm.expert_logits = single.unsqueeze(0)
            swarm.selected = torch.zeros(batch, 1, dtype=torch.long)
            swarm.weights = torch.ones(batch, 1)

        start, end = self.zone_a.task_slice(task)
        task_logits = swarm.logits[:, start:end]
        if task == "relational":
            task_logits = task_logits + self.trilinear_head(photonic.trilinear)
        margin = task_logits.topk(2, dim=-1).values
        margin = (margin[:, 0] - margin[:, 1]) if task_logits.shape[-1] > 1 else margin[:, 0]

        chance = self.chance_epsilon(task)
        chance_veto = has_hits & (s_max_tensor < chance)
        veto = (delta_sagnac > tau_residual) | chance_veto
        if not use_photonic or not use_memory:
            veto = torch.zeros(batch, dtype=torch.bool)
        return HolarchOutput(
            task=task,
            task_logits=task_logits,
            logits=swarm.logits,
            psi_in=psi_in,
            psi_fused=fused,
            photonic=photonic,
            swarm=swarm,
            features=features,
            context=context_wave,
            memory=read,
            memory_mask=has_hits,
            veto=veto,
            margin=margin,
            s_max=s_max_tensor,
            delta_sagnac=delta_sagnac,
        )

    # ------------------------------------------------------------- learning
    @torch.no_grad()
    def observe(
        self,
        task: str,
        x: torch.Tensor,
        labels: torch.Tensor,
        *,
        max_writes: int = 4,
        tier: int = 1,
    ) -> int:
        """Write detached input waves to Zone C. Returns the write count."""
        if getattr(self, "memory", None) is None:
            return 0
        waves = self.encode(task, x).detach().cpu().numpy()
        written = 0
        for index in range(min(int(max_writes), waves.shape[0])):
            self.memory.write(waves[index], label=int(labels[index]), tier=int(tier))
            written += 1
        return written

    def fast_learn(self, output: HolarchOutput, labels: torch.Tensor, lr: float = 0.05) -> Dict[str, float]:
        """Zone A delta-rule write for the batch that produced output."""
        start, end = self.zone_a.task_slice(output.task)
        return self.zone_a.online_update(
            output.features.detach(), labels.long(), TASK_IDS[output.task], lr=float(lr)
        )

    @torch.no_grad()
    def imprint(self, task: str, x: torch.Tensor, labels: torch.Tensor, rate: float = 0.5) -> float:
        """Zone B optical phase-conjugate write toward one class prototype."""
        waves = self.encode(task, x)
        error = 0.0
        classes = torch.unique(labels)
        for label in classes:
            mask = labels == label
            target = _unit(waves[mask].mean(dim=0, keepdim=True))
            error += float(self.zone_b.imprint(waves[mask][:1], target, rate=rate))
        return error / max(1, int(classes.numel()))

    # --------------------------------------------------------- calibration
    @torch.no_grad()
    def calibrate(self, task: str, x: torch.Tensor, n_pairs: int = 256) -> Dict[str, float]:
        """Calibrate the Sagnac veto on this task's encoded wave family."""
        waves = self.encode(task, x)
        take = min(int(n_pairs), waves.shape[0])
        return self.zone_b.calibrate([waves[i] for i in range(take)])

    @torch.no_grad()
    def calibrate_veto(
        self,
        task: str,
        batches: Sequence[torch.Tensor],
        sigma: float = 3.0,
        ceiling: float = 0.35,
        quantile: float = 0.995,
        margin: float = 0.01,
    ) -> Dict[str, float]:
        """Calibrate the residual threshold on in-distribution queries.

        A quantile is used instead of mean + k sigma because the residual
        distribution is bounded and skewed; the Gaussian form under-covers
        and produced a measured 23.7 percent false-veto rate. The quantile
        target is explicit: 99.5 percent of legitimate retrievals pass.
        """
        deltas: List[float] = []
        for batch in batches:
            # Calibrate on the RAW residual. Relaxation is monotone, so the
            # deployed path can only lower delta and cannot exceed this
            # threshold more often than the calibration tail.
            output = self.forward(task, batch, training=False, relax=False)
            mask = output.memory_mask
            if bool(mask.any()):
                deltas.extend(float(v) for v in output.photonic.delta_memory[mask].cpu())
        if not deltas:
            return {"epsilon": float(self.zone_b.epsilon), "n": 0.0, "calibrated": 0.0}
        values = torch.tensor(deltas, dtype=torch.float32)
        mean = float(values.mean())
        std = float(values.std(unbiased=False))
        quantile_value = float(torch.quantile(values, float(quantile)))
        raw = quantile_value + float(margin)
        epsilon = float(min(raw, float(ceiling)))
        with torch.no_grad():
            self.zone_b.epsilon.fill_(epsilon)
            getattr(self, f"epsilon_{task}").fill_(epsilon)
        return {
            "epsilon": epsilon,
            "delta_mean": mean,
            "delta_std": std,
            "delta_quantile": quantile_value,
            "quantile": float(quantile),
            "n": float(values.numel()),
            "capped": float(raw > ceiling),
            "calibrated": 1.0,
        }

    def veto_epsilon(self, task: str) -> torch.Tensor:
        """Per-task calibrated residual threshold."""
        return getattr(self, f"epsilon_{task}")

    def chance_epsilon(self, task: str) -> torch.Tensor:
        """Per-task OOD threshold on the raw top-1 Gram coherence."""
        return getattr(self, f"chance_{task}")

    @torch.no_grad()
    def calibrate_chance(
        self,
        task: str,
        batches: Sequence[torch.Tensor],
        percentile: float = 1.0,
        margin: float = 0.05,
    ) -> Dict[str, float]:
        """Set the OOD threshold from in-distribution raw Gram coherence.

        The threshold is the low percentile of the in-distribution s_max
        minus a margin. A query below it is outside the stored manifold,
        which is exactly what the softmax context cannot express.
        """
        values: List[float] = []
        for batch in batches:
            output = self.forward(task, batch, training=False, relax=False)
            mask = output.memory_mask
            if bool(mask.any()):
                values.extend(float(v) for v in output.s_max[mask].cpu())
        if not values:
            return {"tau_chance": 0.0, "n": 0.0, "calibrated": 0.0}
        tau = max(0.0, float(np.percentile(values, float(percentile))) - float(margin))
        with torch.no_grad():
            getattr(self, f"chance_{task}").fill_(tau)
        return {
            "tau_chance": tau,
            "s_max_mean": float(np.mean(values)),
            "s_max_p01": float(np.percentile(values, 1.0)),
            "n": float(len(values)),
            "calibrated": 1.0,
        }

    # ------------------------------------------------------------- serving
    def predict(self, task: str, x: torch.Tensor, selective: bool = False) -> torch.Tensor:
        output = self.forward(task, x, training=False)
        return output.selective() if selective else output.predicted()

    def act(self, observation, mask_illegal: bool = True) -> int:
        """Single-step policy for the grid world. Returns an action index.

        The legality mask removes actions that drive into a wall the sensor
        already reports. It is an inference-time safety filter, not a
        learned component, and the receipt reports success with and
        without it.
        """
        tensor = torch.as_tensor(np.asarray(observation, dtype=np.float32)).reshape(1, -1)
        with torch.no_grad():
            output = self.forward("action", tensor, training=False, use_memory=True)
            logits = output.task_logits[0].clone()
        if mask_illegal and tensor.shape[-1] >= 75:
            view = tensor[0, :75].reshape(5, 5, 3).numpy()
            for (row, col), action in (((1, 2), 0), ((3, 2), 1), ((2, 1), 2), ((2, 3), 3)):
                if view[row, col, 0] > 0.5:
                    logits[action] = -float("inf")
            if not bool(torch.isfinite(logits).any()):
                return 0
        return int(logits.argmax().item())

    # -------------------------------------------------------------- budget
    def parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad))

    def memory_report(self) -> Dict[str, float]:
        if getattr(self, "memory", None) is None:
            return {"count": 0.0}
        return self.memory.stats()
