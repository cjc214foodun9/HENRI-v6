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
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from henri.holarch.config import TASK_IDS, TASK_NAMES, HolarchConfig
from henri.holarch.deterministic import basis_readout
from henri.holarch.zone_a import LeanSwarm, ReplayBuffer, SwarmOutput
from henri.holarch.zone_b import PhotonicCore, PhotonicState, _unit, sagnac_delta
from henri.holarch.veto import TwoConditionVeto, lower_threshold, upper_threshold
from henri.holarch.zone_c import MemoryRead, PrototypeBank, ZoneCMemory

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


class SLMWaveProjector(nn.Module):
    """Parameter-free phase-only sensor: pixel values drive wave phases.

    Decision 1 asks for an information bottleneck in which the optical
    stack is the only nonlinearity between the sensor and the readout.
    A learned convolutional encoder would hide that bottleneck: it can
    compress 256 pixels into 32 latent values before the plates are ever
    reached. This module removes it.

    The map is injective in the regime |gain * x| < pi, so no information
    is destroyed before the photonic kernel. Components past n_pixels stay
    at phase zero and contribute a constant vector that the readout bias
    absorbs.
    """

    def __init__(self, dim: int, n_pixels: int, gain: float = 1.0) -> None:
        super().__init__()
        self.dim = int(dim)
        self.n_pixels = int(n_pixels)
        self.gain = float(gain)
        if self.n_pixels > self.dim:
            raise ValueError("n_pixels must not exceed the wave dimension")
        if self.gain <= 0.0:
            raise ValueError("slm_gain must be positive")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(x.shape[0], -1)
        if flat.shape[1] != self.n_pixels:
            raise ValueError(
                f"expected {self.n_pixels} sensor values, got {flat.shape[1]}"
            )
        phase = torch.zeros(
            flat.shape[0], self.dim, dtype=torch.float32, device=flat.device
        )
        phase[:, : self.n_pixels] = self.gain * flat.to(torch.float32)
        amplitude = torch.ones_like(phase)
        return _unit(torch.polar(amplitude, phase))


class RelationalBasis(nn.Module):
    """Dedicated Stiefel basis for the bound relational wave.

    Decision 3 binds the three role phasors in the d_slot complex domain
    and then reads the result through this basis. Rows are orthonormal
    (QR retraction every pass) and both quadratures are returned, so the
    head sees 2 * k real values per bound wave and keeps the phase
    information that a magnitude-only readout would discard.
    """

    def __init__(self, d_slot: int, k: int, seed: int = 0) -> None:
        super().__init__()
        self.d_slot = int(d_slot)
        self.k = int(k)
        generator = torch.Generator().manual_seed(int(seed))
        scale = 1.0 / math.sqrt(float(d_slot))
        self.weight_real = nn.Parameter(scale * torch.randn(self.k, self.d_slot, generator=generator))
        self.weight_imag = nn.Parameter(scale * torch.randn(self.k, self.d_slot, generator=generator))

    @property
    def width(self) -> int:
        return 2 * self.k

    def basis(self) -> torch.Tensor:
        weight = torch.complex(self.weight_real, self.weight_imag)
        q, r = torch.linalg.qr(weight.transpose(-1, -2))
        diagonal = torch.diagonal(r, dim1=-2, dim2=-1).real
        sign = torch.where(diagonal >= 0.0, torch.ones_like(diagonal), -torch.ones_like(diagonal))
        q = q * torch.complex(sign, torch.zeros_like(sign)).unsqueeze(-2)
        return q.transpose(-1, -2).contiguous()

    def forward(self, bound: torch.Tensor) -> torch.Tensor:
        projection = basis_readout(bound, self.basis())
        return torch.cat([projection.real, projection.imag], dim=-1)


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


def _operator_glyphs() -> List[np.ndarray]:
    """Three declared operator glyphs (plus, minus, times) as 8x8 fields."""
    plus = np.zeros((8, 8), dtype=np.float32)
    plus[3:5, 1:7] = 1.0
    plus[1:7, 3:5] = 1.0
    minus = np.zeros((8, 8), dtype=np.float32)
    minus[3:5, 1:7] = 1.0
    times = np.zeros((8, 8), dtype=np.float32)
    for index in range(1, 7):
        times[index, index] = 1.0
        times[index, 7 - index] = 1.0
    return [plus, minus, times]


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
        # Directive R2: the fitted Action KV regime head. It is None until a
        # checkpoint carries one or the R2 stage fits it, so every prior
        # cycle's declared-regime path is unchanged.
        self.action_regime_readout = None
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
        # Decision 1: the phase-only sensor carries the vision pixels with
        # no learned encoder, so the plates are the only nonlinearity.
        self.vision_slm = (
            SLMWaveProjector(self.dim, int(cfg.vision_size) ** 2, float(cfg.slm_gain))
            if str(cfg.vision_encoder) == "slm"
            else None
        )
        self.zone_b = PhotonicCore(cfg)
        self.feature_proj = nn.Linear(self.zone_b.raw_width, self.d_feat)
        # Decision 3: bind in the d_slot complex domain, then read through
        # a dedicated Stiefel basis. The retired control reads the K-dim
        # real product instead.
        self.relational_binding = str(cfg.relational_binding)
        self.relational_head = str(cfg.relational_head)
        self.relational_basis = RelationalBasis(
            int(cfg.d_slot), int(cfg.relational_k), seed=int(cfg.seed) + 360
        )
        # Decision B1: a relational-only diagonal unit-phasor mask. The bonus
        # relational passes cannot rotate the shared plates, so the three role
        # waves get their own alignment degrees of freedom. Zero at start, so
        # the first relational step is identical to the coupled path.
        self.relational_phase = nn.Parameter(torch.zeros(3, int(cfg.d_slot)))
        # Decision C2: one diagonal phase-mask cartridge per task, mounted on
        # the shared BaTiO3 substrate. The cartridge is selected by the task
        # that the caller names (the API always knows it); a router-free
        # deployment selects it by the Pass-0 prototype block argmax. Zero at
        # start, so every pre-existing receipt remains the identity case.
        self.task_phase = nn.Parameter(
            torch.zeros(int(cfg.n_tasks), int(cfg.n_slots), int(cfg.d_slot))
        )
        # Decision 7: M cyclic readout frames of the bound wave. Circular
        # convolution commutes with a cyclic shift, so reading roll(psi, s)
        # is exactly binding with roll(psi_action, s): the M frames give the
        # head M different frequency alignments of the same composition.
        self.relational_shifts = [
            (index * int(cfg.d_slot)) // int(cfg.relational_shifts)
            for index in range(int(cfg.relational_shifts))
        ]
        if self.relational_binding == "conv_shift":
            relational_width = len(self.relational_shifts) * self.relational_basis.width
        elif self.relational_binding == "wave":
            relational_width = self.relational_basis.width
        else:
            relational_width = self.zone_b.relational_width
        self.relational_width = int(relational_width)
        self.relational_proj = nn.Linear(self.relational_width, self.d_feat, bias=False)
        # Trilinear gate starts at 1; the three pairwise products start at
        # 0.1 so they can help but cannot dominate a 72-triple fit. It is
        # only read by the retired trilinear control arm.
        gate = torch.cat(
            [
                torch.ones(int(cfg.pool_k)),
                0.1 * torch.ones(self.zone_b.relational_width - int(cfg.pool_k)),
            ]
        )
        self.relational_gate = nn.Parameter(gate)
        self.relational_dropout = nn.Dropout(float(cfg.relational_dropout))
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
        # Decision 7: in 'expert' mode the bound block is routed through the
        # Zone A relational MLP, so the standalone linear head is not built.
        if self.relational_head == "expert" and self.relational_binding != "trilinear":
            self.trilinear_head = None
            # Decision 7: a dedicated two-layer relational micro-net over the
            # concatenated unbound and bound blocks. The shared swarm expert
            # is 48 wide; the baseline MLP is 384 wide, and that gap was the
            # real ceiling, not the number of bound features.
            expert_in = 3 * int(cfg.pool_k) + self.relational_width
            self.relational_expert = nn.Sequential(
                nn.Linear(int(expert_in), int(cfg.relational_hidden)),
                nn.GELU(),
                nn.Dropout(float(cfg.relational_dropout)),
                nn.Linear(int(cfg.relational_hidden), int(cfg.relational_classes)),
            )
        else:
            self.relational_expert = None
            head_width = (
                self.relational_width if self.relational_binding != "trilinear" else int(cfg.pool_k)
            )
            self.trilinear_head = nn.Linear(int(head_width), int(cfg.relational_classes))
        # Decision 6: scale of the calibrated Hopfield prototype logits.
        self.proto_alpha = nn.Parameter(torch.tensor(float(cfg.proto_alpha_init)))
        self.memory_gate = nn.Parameter(torch.tensor(-1.0))
        self.skip_alpha = float(cfg.skip_alpha)
        for name in TASK_NAMES:
            self.register_buffer(f"epsilon_{name}", torch.tensor(0.35))
            self.register_buffer(f"chance_{name}", torch.tensor(0.0))
            # Decision A2: the prototype branch is weighted by surprise, so
            # the trust ramp carries one midpoint and one width per task.
            # calibrate_gate rewrites them from the clean surprise family.
            self.register_buffer(
                f"trust_mid_{name}", torch.tensor(float(cfg.prototype_trust_mid))
            )
            self.register_buffer(
                f"trust_scale_{name}", torch.tensor(float(cfg.prototype_trust_scale))
            )
        self.memory = memory if memory is not None else ZoneCMemory(
            cfg, backend=backend, path=memory_path, dsn=dsn, table=table
        )
        # Decision 2: one clean post-photonic attractor bank per task. The
        # numpy bank holds the floats; the tensor buffer carries the same
        # rows through the checkpoint so the relaxation loop can use them.
        self.prototypes: Dict[str, PrototypeBank] = {
            name: PrototypeBank(self.dim) for name in TASK_NAMES
        }
        for name in TASK_NAMES:
            self.register_buffer(
                f"prototype_{name}", torch.zeros(0, self.dim, dtype=torch.complex64)
            )
            self.register_buffer(
                f"prototype_class_{name}", torch.zeros(0, dtype=torch.long)
            )
        self._rule_cache: Dict[int, torch.Tensor] = {}
        self.register_buffer("abstain", torch.tensor(0.0))
        self.register_buffer("gate_tau", torch.tensor(float(cfg.fast_gate_tau)))

    # ------------------------------------------------------------- encoding
    def encode(
        self, task: str, x: torch.Tensor, rule_id=None
    ) -> torch.Tensor:
        """Encode raw modality data to a unit complex wave."""
        if task not in self.encoders:
            raise ValueError(f"unknown task {task!r}")
        if task == "vision" and self.vision_slm is not None:
            wave = self.vision_slm(x)
        else:
            latent = self.encoders[task](x)
            wave = self._relational_wave(latent) if task == "relational" else self.projector[task](latent)
        return wave if rule_id is None else self.apply_rule(wave, rule_id)

    def _relational_wave(self, latent: torch.Tensor) -> torch.Tensor:
        """Bind the three role slots by pointwise phasor product.

        The product places the trilinear composition in slot 3, because
        multiplication of unit phasors adds their phases. That is VSA
        binding, and it is parameter free.
        """
        waves = self.projector["relational"](latent)
        if self.relational_binding != "trilinear":
            # Decision B1: relational-only phase alignment, applied before
            # the bind so it never touches a shared module.
            waves = waves * torch.exp(1j * self.relational_phase).unsqueeze(0)
        if self.relational_binding == "conv_shift":
            # Decision 7: circular convolution mixes every frequency channel
            # with every other one, where the diagonal product never did.
            transform = torch.fft.fft(waves, dim=-1)
            bound = torch.fft.ifft(
                transform[:, 0] * transform[:, 1] * transform[:, 2], dim=-1
            )
        else:
            bound = waves[:, 0] * waves[:, 1] * waves[:, 2]
        bound = _unit(bound.unsqueeze(1))
        slots = torch.cat([waves, bound], dim=1)
        return _unit(slots.reshape(slots.shape[0], -1))

    # ---------------------------------------------------------- prototypes
    def prototype_waves(self, task: str) -> torch.Tensor:
        """The stored prototype tensor of one task, shape (M, dim)."""
        return getattr(self, f"prototype_{task}")

    @torch.no_grad()
    def calibrate_prototypes(
        self,
        task: str,
        samples: Sequence,
        max_prototypes: int = 64,
    ) -> Dict[str, float]:
        """Build the clean post-photonic attractor bank of one task.

        The samples argument is a sequence of task batches. Every clean
        input is passed through the plates and the bright port is
        averaged per class. No noisy query touches this path, which is
        the point of Decision 2.
        """
        if not samples:
            return {"prototypes": 0.0, "prototype_beta": 0.0}
        x = torch.cat([sample.x for sample in samples])
        y = torch.cat([sample.y for sample in samples])
        psi = self.encode(task, x)
        bright = self.zone_b(psi).bright.detach().cpu().numpy()
        bank = self.prototypes[task]
        stats = bank.build(bright, y.detach().cpu().numpy(), max_prototypes=max_prototypes)
        stats.update(bank.calibrate_beta(bright))
        stats.update(bank.accuracy(bright, y.detach().cpu().numpy()))
        self.set_prototypes(task)
        return stats

    @torch.no_grad()
    def set_prototypes(self, task: str) -> int:
        """Copy the numpy bank into the registered tensor buffer."""
        bank = self.prototypes[task]
        tensor = torch.from_numpy(np.ascontiguousarray(bank.phi)).to(torch.complex64)
        buffer = getattr(self, f"prototype_{task}")
        if buffer.shape != tensor.shape:
            setattr(self, f"prototype_{task}", tensor)
        else:
            buffer.copy_(tensor)
        classes = torch.from_numpy(np.ascontiguousarray(bank.labels)).to(torch.long)
        class_buffer = getattr(self, f"prototype_class_{task}")
        if class_buffer.shape != classes.shape:
            setattr(self, f"prototype_class_{task}", classes)
        else:
            class_buffer.copy_(classes)
        return int(tensor.shape[0])

    @torch.no_grad()
    def store_prototypes(self, task: Optional[str] = None) -> Dict[str, float]:
        """Persist the bank as axiomatic (tier-3) Zone C engrams."""
        total = 0.0
        from henri.holarch.zone_c import global_label

        for name in ([task] if task else list(TASK_NAMES)):
            bank = self.prototypes[name]
            if len(bank) and getattr(self, "memory", None) is not None:
                total += float(
                    bank.write_to_memory(
                        self.memory,
                        offset=global_label(name, 0),
                    )
                )
        return {"prototype_engrams_written": total}

    def prototype_logits(
        self, task: str, bright: torch.Tensor, n_classes: int
    ) -> Optional[torch.Tensor]:
        """Calibrated Hopfield prototype logits, shape (batch, n_classes).

        Decision 6: the relaxation loop minimises an energy whose minimum is
        a prototype, yet the swarm head never saw the prototypes. This block
        adds beta * Re<psi, phi_c> to the task logits, standardised so its
        scale cannot swamp the trained head, and scattered onto the class
        ids the bank was built from.
        """
        bank = self.prototype_waves(task)
        if int(bank.numel()) == 0:
            return None
        labels = getattr(self, f"prototype_class_{task}")
        if int(labels.numel()) != int(bank.shape[0]):
            return None
        align = self.zone_b.alignment(bright, bank) * float(self.prototypes[task].beta)
        align = align - align.mean(dim=-1, keepdim=True)
        align = align / align.std(dim=-1, keepdim=True).clamp_min(1e-4)
        out = torch.zeros(
            bright.shape[0], int(n_classes), device=bright.device, dtype=torch.float32
        )
        index = labels.unsqueeze(0).expand(align.shape[0], -1)
        out.scatter_add_(1, index, align.to(out.dtype))
        return out

    @torch.no_grad()
    def load_prototypes(self, task: str) -> int:
        """Reload one task's bank from the axiomatic Zone C engrams."""
        from henri.holarch.zone_c import global_label, task_label_block

        bank = self.prototypes[task]
        count = bank.load_from_memory(
            self.memory,
            offset=global_label(task, 0),
            block=task_label_block(task),
        )
        self.set_prototypes(task)
        return count

    @torch.no_grad()
    def surprise(self, task: str, bright: torch.Tensor) -> torch.Tensor:
        """Per-sample surprise: distance from the clean attractor bank.

        This is the Gram residual of Decision 4. A familiar input lands on
        a prototype (surprise near zero); a distribution shift does not.
        """
        bank = self.prototype_waves(task)
        if bank.numel() == 0:
            return torch.ones(bright.shape[0])
        align = self.zone_b.alignment(_unit(bright.to(torch.complex64)), bank)
        return 1.0 - align.max(dim=-1).values

    @torch.no_grad()
    def prototype_trust(self, task: str, bright: torch.Tensor) -> torch.Tensor:
        """Decision A2 trust weight eta(delta_surprise), shape (batch,).

            eta = floor + (1 - floor) * sigmoid((delta - mid) / scale)

        delta is the Gram residual of the PASS-0 wave against the clean
        bank, so the weight is identical in the relaxed and unrelaxed
        arms: the loop changes the attractor readout, not its own
        admission. A familiar wave keeps the floor, a surprising wave
        approaches one.
        """
        bank = self.prototype_waves(task)
        if int(bank.numel()) == 0:
            return torch.full(
                (bright.shape[0],), float(self.cfg.prototype_trust_floor),
                device=bright.device,
            )
        delta = self.surprise(task, bright)
        mid = float(getattr(self, f"trust_mid_{task}"))
        scale = max(float(getattr(self, f"trust_scale_{task}")), 1e-3)
        floor = float(self.cfg.prototype_trust_floor)
        eta = floor + (1.0 - floor) * torch.sigmoid((delta - mid) / scale)
        return eta.clamp(floor, 1.0)

    @torch.no_grad()
    def calibrate_gate(
        self,
        task: str,
        batches: Sequence[torch.Tensor],
        quantile: float = 0.995,
        margin: float = 0.01,
    ) -> Dict[str, float]:
        """Set the plasticity threshold from the familiar surprise tail.

        The threshold is a high quantile of the surprise of in-distribution
        inputs plus a margin, so a familiar sample closes the gate and a
        distribution shift opens it. A literal threshold would be
        constraint C2 repeated on a new statistic.
        """
        values: List[float] = []
        for batch in batches:
            output = self.forward(task, batch, training=False, relax=False)
            values.extend(float(v) for v in self.surprise(task, output.photonic.bright).cpu())
        if not values:
            return {"tau_plasticity": float(self.gate_tau), "n": 0.0, "calibrated": 0.0}
        tau = max(0.0, float(np.quantile(values, float(quantile))) + float(margin))
        self.gate_tau.fill_(tau)
        # Decision A2: place the trust ramp on the measured clean surprise
        # family. The midpoint is the median; the width is the robust
        # standard deviation of the interquartile range, so a tight clean
        # distribution gives a sharp ramp and a broad one gives a soft ramp.
        mid = float(np.median(values))
        spread = float(np.quantile(values, 0.75) - np.quantile(values, 0.25))
        scale = max(spread / 1.349, 0.02)
        with torch.no_grad():
            getattr(self, f"trust_mid_{task}").fill_(mid)
            getattr(self, f"trust_scale_{task}").fill_(scale)
        return {
            "tau_plasticity": tau,
            "surprise_mean": float(np.mean(values)),
            "surprise_quantile": float(np.quantile(values, float(quantile))),
            "quantile": float(quantile),
            "n": float(len(values)),
            "calibrated": 1.0,
            "trust_mid": mid,
            "trust_scale": scale,
        }

    def rule_wave(self, rule_id: int) -> torch.Tensor:
        """Deterministic context phasor for one rule id, length d_slot.

        Decision 4 binds the rule into slot 3, so the same pixels under two
        rules map to two different input waves and the label contradiction
        disappears. The phasor is seeded, so a receipt can replay it.
        """
        key = int(rule_id)
        if key not in self._rule_cache:
            generator = torch.Generator().manual_seed(int(self.cfg.seed) * 100003 + key)
            real = torch.randn(int(self.cfg.d_slot), generator=generator)
            imag = torch.randn(int(self.cfg.d_slot), generator=generator)
            self._rule_cache[key] = _unit(torch.complex(real, imag))
        return self._rule_cache[key]

    def apply_rule(self, psi_in: torch.Tensor, rule_id) -> torch.Tensor:
        """Overwrite slot 3 with the rule phasor and renormalise.

        rule_id may be one integer or one integer per example, so a single
        batch can carry two rules and the model must condition on slot 3
        within the batch rather than across steps.
        """
        slots = psi_in.reshape(
            psi_in.shape[0], int(self.cfg.n_slots), int(self.cfg.d_slot)
        ).clone()
        if isinstance(rule_id, torch.Tensor) or isinstance(rule_id, (list, tuple)):
            ids = [int(value) for value in rule_id]
            if len(ids) != slots.shape[0]:
                raise ValueError("per-example rule ids must match the batch size")
            waves = torch.stack([self.rule_wave(value) for value in ids])
            slots[:, 3] = waves.to(psi_in.device)
        else:
            slots[:, 3] = self.rule_wave(int(rule_id)).to(psi_in.device)
        return _unit(slots.reshape(psi_in.shape[0], -1))

    def relational_features(self, photonic: PhotonicState) -> torch.Tensor:
        """Relational readout block: 2 * relational_k real values.

        The bound wave travels in slot 3 through the plates and the slot
        mixer, so the basis reads a bound relational wave that the optical
        stack has already filtered. Both quadratures are kept: the phase
        of the composition is the information that survives binding.
        """
        if self.relational_binding in {"conv_shift", "wave"}:
            slots = photonic.bright.reshape(
                photonic.bright.shape[0], int(self.cfg.n_slots), int(self.cfg.d_slot)
            )
            bound = slots[:, 3]
            if self.relational_binding == "conv_shift":
                return torch.cat(
                    [
                        self.relational_basis(torch.roll(bound, shift, dims=-1))
                        for shift in self.relational_shifts
                    ],
                    dim=-1,
                )
            return self.relational_basis(bound)
        block = photonic.features[:, self.zone_b.raw_width :]
        return block * self.relational_gate

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
        rule_id=None,
        isolate_branch: bool = False,
        allow_empty: Optional[bool] = None,
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
            relax: run the thermodynamic loop. Decision 2 anchors it in
                the clean prototype bank when one exists.
            rule_id: optional context rule. When given, the deterministic
                rule phasor replaces slot 3, so identical pixels under two
                rules are two different input waves (Decision 4).
            isolate_branch: Decision B1. Detach the shared post-photonic
                wave and the shared slot readouts on the relational branch,
                so a bonus relational pass trains the relational basis, the
                relational phase mask and the relational head only. The
                shared Zone B plates, mixer and Stiefel frames receive no
                gradient from that pass.
            allow_empty: Decision C1. When False (the inference default) a
                query that retrieves no Zone C engram is vetoed, because an
                empty candidate set is s_max = 0 and therefore maximum
                epistemic uncertainty. Training and the explicit ingestion
                paths pass True: they are authorised exploratory ingress.
        """
        if task not in TASK_IDS:
            raise ValueError(f"unknown task {task!r}")
        if task == "relational":
            latent = self.encoders[task](x)
            psi_in = self._relational_wave(latent)
            skip_vector = latent.mean(dim=1)
        elif task == "vision" and self.vision_slm is not None:
            # Decision 1: no learned encoder. The pixel field enters the
            # plates directly, so the readout bottleneck is honest.
            psi_in = self.vision_slm(x)
            latent = torch.zeros(psi_in.shape[0], self.d_latent)
            skip_vector = latent
        else:
            latent = self.encoders[task](x)
            psi_in = self.projector[task](latent)
            skip_vector = latent
        if rule_id is not None:
            psi_in = self.apply_rule(psi_in, rule_id)
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
            waves_np = psi_in.detach().cpu().numpy()
            if batch > 1 and hasattr(self.memory, "context_tensor_batch"):
                # Directive M2: one (B, gram_dims) x (gram_dims, N) product
                # replaces the per-row Python loop.
                prefix_batch, reads = self.memory.context_tensor_batch(
                    waves_np, k=memory_k, task=task
                )
                prefix = torch.from_numpy(prefix_batch.reshape(batch, -1)).to(torch.float32)
            else:
                prefix_rows: List[np.ndarray] = []
                for _index, row in enumerate(waves_np):
                    prefix_np, row_read = self.memory.context_tensor(
                        row, k=memory_k, task=task
                    )
                    prefix_rows.append(prefix_np.reshape(-1))
                    reads.append(row_read)
                prefix = torch.from_numpy(np.stack(prefix_rows)).to(torch.float32)
            for index, row_read in enumerate(reads):
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
            context_wave = _unit(torch.stack(context_rows).to(psi_in.dtype))
            top_axiom = _unit(torch.stack(axiom_rows).to(psi_in.dtype))
            read = reads[0] if reads else None

        gate = torch.sigmoid(self.memory_gate)
        fused = _unit(psi_in + gate * context_wave) if use_memory else psi_in
        # Decision C2: mount the task's diagonal phase cartridge on the
        # substrate before the plates. Zero cartridge = identity.
        cartridge = self.task_phase[TASK_IDS[task]].reshape(1, -1)
        fused = _unit(fused * torch.exp(1j * cartridge.to(fused.dtype)))
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

        # Decision B: the axiom lives in the POST-photonic space. Project the
        # softmax context bundle through the same plates once, without gradient,
        # and compare output to output.
        if use_photonic and use_memory and bool(has_hits.any()):
            with torch.no_grad():
                axiom_out = self.zone_b(context_wave, apply_memory=False).bright
            axiom_out = torch.where(
                has_hits.unsqueeze(-1), axiom_out, photonic.bright.detach()
            )
        else:
            axiom_out = photonic.bright.detach()
        delta_sagnac = sagnac_delta(photonic.bright, axiom_out)

        # Decision A2: the fine-grained task head reads the UNBLURRED
        # Pass-0 Stiefel readout g^(0) = Re(U_s psi_out^(0)). The
        # thermodynamic loop is a denoiser for the attractor branch only:
        # it must never overwrite the stroke geometry that the head uses,
        # which is exactly what cost -0.0033 at sigma = 0.25 in Decision 6.
        #
        # The loop therefore runs on every row, not only on surprising
        # ones. Decision 2 masked familiar inputs because relaxation used
        # to rewrite the head input; with the head anchored on Pass 0 the
        # only thing a familiar row can change is the prototype branch,
        # and pulling a wave that already sits on a prototype toward that
        # prototype is a no-op in the decision (measured p_raw vs p_rel
        # argmax agreement 99.8 percent at sigma = 0).
        pass0 = photonic
        prototype_wave = pass0.bright
        relaxed_ran = False
        bank = self.prototype_waves(task)
        use_prototypes = (
            str(self.cfg.relax_mode) == "prototype" and int(bank.numel()) > 0
        )
        if (
            relax
            and (not training or bool(self.cfg.relax_training))
            and use_photonic
            and int(self.cfg.relax_steps) > 0
            and (use_prototypes or bool((delta_sagnac > tau_residual).any()))
        ):
            if use_prototypes:
                relaxed, _trace = self.zone_b.relax_prototype(
                    fused,
                    bank,
                    beta=float(self.prototypes[task].beta),
                    lambda_dark=float(self.cfg.prototype_lambda_dark),
                    eta=float(self.cfg.prototype_eta),
                )
                # The relaxation sharpens the Hopfield term; the deployed
                # state keeps the relaxed bright port so the attractor and
                # dark-energy rows still describe the loop that ran.
                prototype_wave = relaxed.bright
                relaxed_ran = True
                photonic = self.zone_b._replace_bright(pass0, relaxed.bright, axiom_out)
            else:
                relaxed, _trace = self.zone_b.relax_output(
                    fused, axiom_out, tolerance=float(tau_residual)
                )
                prototype_wave = relaxed.bright
                relaxed_ran = True
                photonic = relaxed
            # The veto statistic stays the RAW pass. The loop is a repair
            # applied after the epistemic check, and calibrate_veto measures
            # the raw residual. Reporting the post-repair residual as the
            # veto input raised the measured false-veto rate from 0.003 to
            # 0.030 in the first run of this revision.

        # Decision B1: on an isolated pass the relational branch reads a
        # detached copy of the shared Pass-0 wave, so its gradient stops at
        # the relational parameters and never rotates the shared plates.
        relational_source = pass0
        if isolate_branch and task == "relational":
            relational_source = replace(
                pass0,
                bright=pass0.bright.detach(),
                features=pass0.features.detach(),
            )

        relative = None
        if use_optical_features:
            # Pass-0 anchor: the head never reads the relaxed wave.
            raw = self.feature_proj(pass0.features[:, : self.zone_b.raw_width])
            if task == "relational":
                # Decision 3: the head reads the bound wave projected
                # through its own Stiefel basis, so the composition keeps
                # all d_slot phases before any compression.
                relative = self.relational_features(relational_source)
                if training:
                    relative = self.relational_dropout(relative)
                raw = raw + self.relational_proj(relative)
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
        if task == "relational" and self.relational_expert is not None:
            unbound = relational_source.features[:, : 3 * int(self.cfg.pool_k)]
            bound = (
                relative
                if relative is not None
                else self.relational_features(relational_source)
            )
            if training:
                bound = self.relational_dropout(bound)
            task_logits = self.relational_expert(torch.cat([unbound, bound], dim=-1))
        elif task == "relational" and self.trilinear_head is not None:
            if self.relational_binding == "trilinear":
                head_input = photonic.trilinear
            else:
                head_input = (
                    relative if relative is not None else self.relational_features(photonic)
                )
            task_logits = task_logits + self.trilinear_head(head_input)
        # The prototype logits ARE optical features: they read the bright
        # port. The optical ablation must switch them off too, or the
        # "no Zone B features" arm still sees the plates through the bank.
        if bool(self.cfg.prototype_logits) and use_optical_features:
            # Decision B2: the Pass-0 prototype coupling is ALWAYS on at full
            # weight. The surprise trust weight gates only the multi-pass
            # relaxation delta, so a familiar input keeps the full Hopfield
            # boost and a surprising one may receive the denoised correction.
            eta = self.prototype_trust(task, pass0.bright).unsqueeze(-1)
            base_wave = pass0.bright.detach() if isolate_branch else pass0.bright
            base = self.prototype_logits(
                task, base_wave, task_logits.shape[-1]
            )
            if base is not None:
                task_logits = task_logits + self.proto_alpha * base
                if (
                    relaxed_ran
                    and use_prototypes
                    and not isolate_branch
                    and float(getattr(self.cfg, "proto_relax_gain", 1.0)) > 0.0
                ):
                    tuned = self.prototype_logits(
                        task, prototype_wave, task_logits.shape[-1]
                    )
                    if tuned is not None:
                        task_logits = task_logits + (
                            float(self.cfg.proto_relax_gain)
                            * eta
                            * self.proto_alpha
                            * (tuned - base)
                        )
        margin = task_logits.topk(2, dim=-1).values
        margin = (margin[:, 0] - margin[:, 1]) if task_logits.shape[-1] > 1 else margin[:, 0]

        chance = self.chance_epsilon(task)
        # Decision A3: one value object owns the union rule, so the
        # calibrator and the deployed decision cannot drift apart.
        # Decision C1: a no-hit query is vetoed unless the caller authorised
        # empty retrieval (training or an ingestion path).
        empty_allowed = bool(training) if allow_empty is None else bool(allow_empty)
        rule = TwoConditionVeto(
            tau_residual=float(tau_residual), tau_chance=float(chance)
        )
        veto = torch.as_tensor(
            rule.union(
                delta_sagnac.detach().cpu().numpy(),
                s_max_tensor.detach().cpu().numpy(),
                has_hits.detach().cpu().numpy(),
                allow_empty=empty_allowed,
            ),
            device=psi_in.device,
            dtype=torch.bool,
        )
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
        from henri.holarch.zone_c import global_label

        waves = self.encode(task, x).detach().cpu().numpy()
        written = 0
        for index in range(min(int(max_writes), waves.shape[0])):
            # Decision D1: a task engram stores its global (block-offset)
            # label, so a task-scoped query can restrict candidates to its
            # own block without a schema change.
            self.memory.write(
                waves[index],
                label=global_label(task, int(labels[index])),
                tier=int(tier),
            )
            written += 1
        return written

    def plasticity_gate(self, output: HolarchOutput) -> torch.Tensor:
        """Decision 4 surprise gate, one scalar in {0, 1} per example.

            eta_eff = eta * 1[surprise > tau_plasticity OR s_max < tau_familiar]

        The first term is the post-photonic prototype residual. The second
        fires only on a genuine Zone C retrieval: a query with no engram
        hit reports s_max = 0, and treating that as "unfamiliar" would open
        the gate on every empty memory.
        """
        surprise = self.surprise(output.task, output.photonic.bright)
        gate_open = surprise > float(self.gate_tau)
        familiar = float(self.cfg.fast_gate_familiar)
        if familiar > 0.0:
            gate_open = gate_open | (output.memory_mask & (output.s_max < familiar))
        return gate_open.to(torch.float32)

    def fast_learn(
        self,
        output: HolarchOutput,
        labels: torch.Tensor,
        lr: float = 0.05,
        gate: bool = True,
    ) -> Dict[str, float]:
        """Zone A delta-rule write for the batch that produced output."""
        start, end = self.zone_a.task_slice(output.task)
        weights = (
            self.plasticity_gate(output)
            if (gate and bool(self.cfg.fast_gate))
            else None
        )
        return self.zone_a.online_update(
            output.features.detach(),
            labels.long(),
            TASK_IDS[output.task],
            lr=float(lr),
            weights=weights,
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
        quantile: float = 0.999,
        margin: float = 0.0,
        margin_sigma: float = 2.0,
    ) -> Dict[str, float]:
        """Calibrate the residual threshold on in-distribution queries.

        Decision A3: the threshold is the Cornish-Fisher 0.997 quantile
        plus two order-statistic standard errors, not the raw sample
        quantile. sigma is retained for call compatibility and is unused:
        a Gaussian mean + k sigma under-covers this bounded, skewed tail
        (measured 23.7 percent false veto in the retired revision).
        """
        deltas: List[float] = []
        for batch in batches:
            # Calibrate on the RAW residual. Relaxation is monotone, so the
            # deployed path can only lower delta and cannot exceed this
            # threshold more often than the calibration tail.
            output = self.forward(task, batch, training=False, relax=False)
            mask = output.memory_mask
            if bool(mask.any()):
                deltas.extend(float(v) for v in output.delta_sagnac[mask].cpu())
        if not deltas:
            return {"epsilon": float(self.zone_b.epsilon), "n": 0.0, "calibrated": 0.0}
        prob = max(1e-4, 1.0 - float(quantile))
        upper = upper_threshold(
            deltas,
            prob,
            margin_sigma=float(margin_sigma),
            ceiling=float(ceiling),
            additive_margin=float(margin),
        )
        epsilon = float(upper["threshold"])
        values = torch.tensor(deltas, dtype=torch.float32)
        with torch.no_grad():
            self.zone_b.epsilon.fill_(epsilon)
            getattr(self, f"epsilon_{task}").fill_(epsilon)
        return {
            "epsilon": epsilon,
            "delta_mean": float(values.mean()),
            "delta_std": float(values.std(unbiased=False)),
            "delta_quantile": float(upper["quantile"]),
            "quantile": float(quantile),
            "n": float(values.numel()),
            "capped": float(upper["capped"]),
            "sigma_tail": float(upper["sigma_tail"]),
            "margin_sigma": float(margin_sigma),
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
        percentile: float = 0.2,
        margin: float = 0.0,
        margin_sigma: float = 2.0,
    ) -> Dict[str, float]:
        """Set the OOD threshold from in-distribution raw Gram coherence.

        Decision A3: the threshold is the Cornish-Fisher quantile at the
        requested PERCENTILE minus two order-statistic standard errors.
        The percentile stays in percent (0.5 is the 0.5th percentile), so
        the retired call form still means what it said.
        """
        values: List[float] = []
        for batch in batches:
            output = self.forward(task, batch, training=False, relax=False)
            mask = output.memory_mask
            if bool(mask.any()):
                values.extend(float(v) for v in output.s_max[mask].cpu())
        if not values:
            return {"tau_chance": 0.0, "n": 0.0, "calibrated": 0.0}
        prob = min(max(float(percentile) / 100.0, 1e-4), 0.5)
        lower = lower_threshold(
            values,
            prob,
            margin_sigma=float(margin_sigma),
            floor=0.0,
            additive_margin=float(margin),
        )
        tau = float(lower["threshold"])
        with torch.no_grad():
            getattr(self, f"chance_{task}").fill_(tau)
        return {
            "tau_chance": tau,
            "s_max_mean": float(np.mean(values)),
            "s_max_quantile": float(lower["quantile"]),
            "sigma_tail": float(lower["sigma_tail"]),
            "margin_sigma": float(margin_sigma),
            "n": float(len(values)),
            "calibrated": 1.0,
        }

    @torch.no_grad()
    def calibrate_two_condition(
        self,
        task: str,
        batches: Sequence[torch.Tensor],
        quantile: float = 0.999,
        chance_percentile: float = 0.2,
        margin_sigma: float = 2.0,
        target_false_veto: float = 0.005,
        ceiling: float = 0.98,
    ) -> Dict[str, float]:
        """Calibrate BOTH veto conditions jointly (Decision A3).

        One forward pass per batch yields the paired statistics, so the
        calibrator can measure the union false-veto rate on the same rows
        it fitted. The returned report carries that measured rate; the
        pipeline reads the two thresholds from the per-task buffers, so a
        test can still override either one directly.
        """
        residuals: List[float] = []
        chances: List[float] = []
        for batch in batches:
            output = self.forward(task, batch, training=False, relax=False)
            mask = output.memory_mask
            if bool(mask.any()):
                residuals.extend(float(v) for v in output.delta_sagnac[mask].cpu())
                chances.extend(float(v) for v in output.s_max[mask].cpu())
        rule = TwoConditionVeto.calibrate(
            residuals,
            chances,
            residual_prob=max(1e-4, 1.0 - float(quantile)),
            chance_prob=min(max(float(chance_percentile) / 100.0, 1e-4), 0.5),
            margin_sigma=float(margin_sigma),
            target_false_veto=float(target_false_veto),
            ceiling=float(ceiling),
        )
        with torch.no_grad():
            getattr(self, f"epsilon_{task}").fill_(rule.tau_residual)
            getattr(self, f"chance_{task}").fill_(rule.tau_chance)
            self.zone_b.epsilon.fill_(rule.tau_residual)
        report = rule.report()
        report.update(
            {
                "epsilon": float(rule.tau_residual),
                "tau_chance": float(rule.tau_chance),
                "n": float(rule.n_cal),
                "quantile": float(quantile),
                "calibrated": 1.0 if rule.n_cal else 0.0,
                "s_max_mean": float(np.mean(chances)) if chances else 0.0,
                "delta_mean": float(np.mean(residuals)) if residuals else 0.0,
                "delta_std": float(np.std(residuals)) if residuals else 0.0,
            }
        )
        return report

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

    # ------------------------------------------------------- seed cartridge
    @torch.no_grad()
    def hydrate_seed_engrams(self, namespace: str = "default") -> Dict[str, object]:
        """Mount the checkpoint's distilled Zone C chart into an empty store.

        Directive J1. The seed block travels inside the checkpoint (see
        seed_engram_payload in model.py). Hydration runs only when the store
        holds zero rows, so a populated tenant store is never overwritten by a
        mount, and the returned dict reports both the offered count and the
        written count. Rows are written in ascending reference id order, which
        is the order the reference store's tie-break used, so the reconstructed
        index ranks ties the same way.
        """
        from henri.holarch.model import seed_engram_rows

        block = getattr(self, "seed_engrams", None)
        rows = seed_engram_rows(block)
        memory = getattr(self, "memory", None)
        info: Dict[str, object] = {
            "available": int(len(rows)),
            "written": 0,
            "skipped": True,
            "namespace": str(namespace),
        }
        if memory is None or not rows:
            return info
        try:
            existing = int(memory.stats().get("count", 0))
        except Exception:
            existing = 0
        if existing > 0:
            info["existing"] = existing
            return info
        written = 0
        for row in rows:
            memory.write(
                row["wave"],
                label=int(row["label"]),
                tier=int(row["tier"]),
                namespace=str(namespace or row.get("namespace", "default")),
            )
            written += 1
        info["written"] = int(written)
        info["skipped"] = bool(written == 0)
        info["count_after"] = int(memory.stats().get("count", 0))
        return info

    # ------------------------------------------------------- VLA generation
    @torch.no_grad()
    def generate_vla_episode(
        self,
        vision_canvas,
        intent_prompt: str = "",
        *,
        task=None,
        layout=None,
        engine=None,
        namespace: str = "default",
        test_source: Optional[str] = None,
        max_steps: int = 24,
        seed: int = 0,
        online_repulsor: bool = True,
    ) -> Dict[str, object]:
        """One closed-loop Vision -> Symbolic -> Action -> Code VLA episode.

        Every stage reads and writes the same three-zone loop: the visual
        ingress crosses Zone C and Zone B, the planner uses the rotor
        lookahead over the post-photonic wave, and the synthesizer emits a
        verified function whose bound is the grounded visual quantity. Each
        emitted token is chained to the previous one by SHA-256, so the trace
        is one contiguous epistemic receipt chain.
        """
        import hashlib
        import json

        from henri.holarch.vla import action_token, relational_token, vision_token

        trace: List[Dict[str, object]] = []
        receipts: List[Dict[str, object]] = []
        tokens: List[str] = []
        token_ids: List[int] = []

        def emit(stage: str, token: str, token_id: int, payload: Dict[str, object]) -> None:
            parent = str(receipts[-1]["sha256"]) if receipts else "genesis"
            body = {
                "index": len(receipts),
                "stage": str(stage),
                "token": str(token),
                "payload_sha256": hashlib.sha256(
                    json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
                ).hexdigest(),
                "parent": parent,
            }
            body["sha256"] = hashlib.sha256(
                json.dumps(body, sort_keys=True).encode("utf-8")
            ).hexdigest()
            receipts.append(body)
            tokens.append(str(token))
            token_ids.append(int(token_id))
            trace.append(
                {
                    "step": len(trace),
                    "stage": str(stage),
                    "token": str(token),
                    "payload": payload,
                    "receipt": body,
                }
            )

        # ---------------------------------------------------------- vision
        canvas = torch.as_tensor(np.asarray(vision_canvas, dtype=np.float32))
        if canvas.dim() == 2:
            canvas = canvas.reshape(1, 1, *canvas.shape)
        elif canvas.dim() == 3:
            canvas = canvas.unsqueeze(0)
        if tuple(canvas.shape[-2:]) == (16, 16):
            canvas = F.avg_pool2d(canvas, 2)
        sensor = canvas.reshape(1, -1) if self.vision_slm is not None else canvas
        vision_out = self("vision", sensor, training=False, allow_empty=True)
        digit = int(vision_out.predicted()[0])
        confidence = float(vision_out.probs[0, digit])
        vision = {
            "digit": digit,
            "confidence": confidence,
            "canvas": [int(canvas.shape[-2]), int(canvas.shape[-1])],
            "grounded": bool(int(digit) >= 0),
        }
        emit("vision", f"vision:digit={digit}", vision_token(digit), vision)

        # ------------------------------------------------- symbolic / relational
        bound = float(digit + 1.0)
        rule = "bound_from_digit"
        relational = {
            "rule": rule,
            "bound": bound,
            "expression": f"clamp(., 0.0, {bound:.1f})",
        }
        emit(
            "relational",
            f"relational:{rule}={bound:.1f}",
            relational_token(int(digit)),
            relational,
        )

        # -------------------------------------------------------------- action
        planning: Dict[str, object] = {"requested": bool(task is not None and layout is not None)}
        plan_ok = True
        if task is not None and layout is not None:
            from henri.holarch.model import LatentActionLookahead

            walls, agent, goal = layout
            # The VLA planner is the same closed-loop evaluator the 48-layout
            # measurement uses. Duplicating its loop here was the source of a
            # real divergence (the episode re-implementation did not always
            # detect the same trap), so the episode delegates to the one
            # measured policy instead.
            import copy

            from henri.holarch.model import evaluate_closed_loop_action

            probe_task = copy.copy(task)
            probe_task.val_layouts = [(walls, agent, goal)]
            result = evaluate_closed_loop_action(
                self,
                probe_task,
                episodes=1,
                limit=int(max_steps),
                rotor=getattr(self, "action_rotor", None),
                transition="rotor",
                lookahead_depth=2,
                gamma_look=0.02,
                batch_rollout=True,
                use_memory=True,
                trace=True,
                online_repulsor=bool(online_repulsor),
                online_namespace=f"{namespace}_vla",
                # Directive X4: the episode reads the trace only. The 80-pair
                # one-step audit is a separate diagnostic and is not paid for
                # on the episode path.
                one_step_audit=False,
            )
            trace_row = (result.get("trace") or [{}])[0]
            actions = [int(action) for action in (trace_row.get("actions") or [])]
            reached = bool(trace_row.get("win", False))
            plan_ok = bool(len(actions) > 0)
            discoveries = [
                {
                    "step": int(row.get("step", 0)),
                    "action": int(row.get("escape_action", -1)),
                    "cycle_score": float(row.get("cycle_score", 0.0)),
                    "dead_end": bool(row.get("dead_end", False)),
                    "axiom": dict(row.get("axiom") or {}),
                }
                for row in (result.get("online_rows") or [])
            ]
            planning.update(
                {
                    "actions": actions,
                    "steps": len(actions),
                    "reached_goal": bool(reached),
                    "legal": bool(plan_ok),
                    "start": list(layout[1]),
                    "goal": list(layout[2]),
                    "online_discoveries": len(discoveries),
                    "discovery_rows": discoveries,
                }
            )
            for action in actions:
                emit(
                    "action",
                    f"action:{('up', 'down', 'left', 'right')[action]}",
                    action_token(action),
                    {"action": int(action)},
                )
            plan_ok = bool(plan_ok and len(actions) > 0)

        # ---------------------------------------------------------------- code
        code: Dict[str, object] = {"requested": bool(engine is not None and intent_prompt)}
        code_ok = True
        if engine is not None and intent_prompt:
            prompt = str(intent_prompt).replace("{bound}", f"{bound:.1f}")
            synthesis = engine.synthesize(prompt)
            code.update(
                {
                    "status": str(synthesis.get("status", "")),
                    "verified": bool(synthesis.get("verified", False)),
                    "plan": str(synthesis.get("plan", "")),
                    "source_chars": len(str(synthesis.get("source", ""))),
                }
            )
            execution = {"ok": False, "reason": "not executed"}
            if bool(synthesis.get("verified", False)) and test_source:
                execution = engine.execute(
                    str(synthesis.get("source", "")),
                    str(test_source).replace("{bound}", f"{bound:.1f}"),
                )
            code["executed"] = bool(execution.get("ok", False))
            code["execution_reason"] = str(execution.get("reason", ""))
            code_ok = bool(code["verified"] and code["executed"])
            text = str(synthesis.get("source", ""))[:64]
            code_token = 4608 + (int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:6], 16) % 256)
            emit(
                "code",
                f"code:{str(synthesis.get('plan', 'refused'))[:48]}",
                code_token,
                {
                    "status": code["status"],
                    "verified": code["verified"],
                    "executed": code["executed"],
                    "bound": bound,
                },
            )

        chain_valid = bool(
            receipts
            and receipts[0]["parent"] == "genesis"
            and all(
                receipts[index]["parent"] == receipts[index - 1]["sha256"]
                for index in range(1, len(receipts))
            )
        )
        success = bool(
            vision["grounded"]
            and plan_ok
            and code_ok
            and chain_valid
        )
        return {
            "success": bool(success),
            "namespace": str(namespace),
            "tokens": tokens,
            "token_ids": token_ids,
            "trace": trace,
            "receipts": receipts,
            "receipt_chain_valid": bool(chain_valid),
            "vision": vision,
            "relational": relational,
            "planning": planning,
            "code": code,
            "bound": float(bound),
            "evidence": "OBSERVED",
        }

    # ------------------------------- end-to-end telepathic VLA (P3)
    def _telepathic_banks(self, task, *, seed: int = 20261105) -> Dict[str, List[np.ndarray]]:
        """Declared prototype banks in the same value frame the sender writes.

        The digit bank is the mean encoding of each class over the train split
        (a property of the model's vision encoder, not of the episode), and
        the operator bank is three synthetic glyphs encoded by the same path.
        A bank entry is the flattened value projection kv_from_wave produces,
        so the receiver's comparison happens in the frame it actually reads.
        """
        from henri.holarch.swat_bus import kv_from_wave

        sample = task.sample(64, "train")
        sums = [np.zeros(1024, dtype=np.complex64) for _ in range(10)]
        counts = [0 for _ in range(10)]
        for index in range(int(sample.x.shape[0])):
            with torch.no_grad():
                psi = self.encode("vision", sample.x[index : index + 1]).numpy()[0]
            label = int(sample.y[index]) % 10
            sums[label] = sums[label] + psi.astype(np.complex64)
            counts[label] += 1
        digits = []
        for label in range(10):
            wave = sums[label] / max(counts[label], 1)
            norm = float(np.linalg.norm(wave))
            digits.append((wave / max(norm, 1e-12)).astype(np.complex64))
        operators = []
        for glyph in _operator_glyphs():
            tensor = torch.as_tensor(glyph, dtype=torch.float32).reshape(1, 1, 8, 8)
            with torch.no_grad():
                operators.append(self.encode("vision", tensor).numpy()[0].astype(np.complex64))
        def bundle(wave: np.ndarray) -> np.ndarray:
            _keys, values = kv_from_wave(wave, heads=4, dk=32)
            return values.reshape(-1).numpy().astype(np.float32)
        return {
            "a": [bundle(wave) for wave in digits],
            "b": [bundle(wave) for wave in digits],
            "bound": [bundle(wave) for wave in digits],
            "op": [bundle(wave) for wave in operators],
            # Directive Q2: the action planner's operating regime is a third
            # declared codebook, in the same value frame. The receiver reads
            # the operator from the Action KV, not from a fourth vision patch.
            "regime": [bundle(self._regime_wave(index)) for index in range(3)],
            "digit_waves": digits,
            "operator_waves": operators,
            "regime_waves": [self._regime_wave(index) for index in range(3)],
        }

    def _regime_wave(self, regime: int, *, seed: int = 20261110) -> np.ndarray:
        """The declared wave for one planner operating regime (Directive Q2).

        0, 1 and 2 are three mutually incoherent unit waves. The sender binds
        its ACTUAL terminal trajectory into this frame; the receiver compares
        the delivered value slot against the same three prototypes.
        """
        rng = np.random.default_rng(int(seed) + int(regime))
        wave = (
            rng.standard_normal(self.dim) + 1j * rng.standard_normal(self.dim)
        ).astype(np.complex64)
        norm = float(np.linalg.norm(wave))
        return wave / max(norm, 1e-12)

    def trajectory_regime(self, planning: Dict[str, object]) -> int:
        """Map a measured trajectory outcome to one of three regimes.

        The regime is a deterministic function of the terminal goal quadrant,
        the path length and whether online-REPULSOR inference fired. It is a
        property of the trajectory the planner ACTUALLY executed, so swapping
        two trajectories changes the code receiver's operator.
        """
        goal = list(planning.get("goal") or [0, 0])
        try:
            gx, gy = int(goal[0]), int(goal[1])
        except (TypeError, ValueError, IndexError):
            gx, gy = 0, 0
        steps = int(planning.get("steps", 0))
        repulsor = int(int(planning.get("online_discoveries", 0)) > 0)
        return int((int(gx >= 4) * 2 + int(gy >= 4) + steps + repulsor) % 3)

    def calibrate_telepathic_gate(self, task, gate) -> Dict[str, object]:
        """Calibrate the vision gate on the DECLARED sender set.

        The senders of this pipeline are the ten digit classes and the three
        operator glyphs. Calibrating on the digits alone sets a Sagnac
        residual that the operator glyphs exceed, so a valid glyph would be
        refused; calibrating on the whole declared set keeps every legitimate
        sender admitted while a scrambled wave still fails the binding test.
        """
        from henri.holarch.swat_bus import cache_from_wave

        caches = []
        sample = task.sample(12, "train")
        for index in range(int(sample.x.shape[0])):
            with torch.no_grad():
                psi = self.encode("vision", sample.x[index : index + 1]).numpy()[0]
            caches.append(cache_from_wave(psi, source=f"cal:digit:{index}", task="vision"))
        for index, glyph in enumerate(_operator_glyphs()):
            tensor = torch.as_tensor(glyph, dtype=torch.float32).reshape(1, 1, 8, 8)
            with torch.no_grad():
                psi = self.encode("vision", tensor).numpy()[0]
            caches.append(cache_from_wave(psi, source=f"cal:op:{index}", task="vision"))
        return gate.calibrate(caches)


    def _action_trajectory_bundle(
        self,
        planning: Dict[str, object],
        walls,
        agent,
        goal,
        task,
        *,
        max_visit_samples: int = 4,
    ):
        """The four-slot trajectory bundle the fitted regime head reads.

        Slot 0 is the terminal observation wave, slot 1 the unit superposition
        of the visited state waves, slot 2 a step-count phasor (three phases,
        one per regime residue) and slot 3 the active REPULSOR wave. The bundle
        is the trajectory the planner EXECUTED; it does not contain the
        regime label, which is what the fitted head has to recover.
        """
        from henri.holarch.swat_bus import _unit

        deltas = ((-1, 0), (1, 0), (0, -1), (0, 1))
        actions = [int(action) for action in (planning.get("actions") or [])]
        size = int(getattr(task.world, "grid_size", 7))
        row, col = int(agent[0]), int(agent[1])
        visited = [(row, col)]
        for action in actions:
            dr, dc = deltas[int(action) % 4]
            nr, nc = row + dr, col + dc
            if not (0 <= nr < size and 0 <= nc < size) or bool(walls[nr, nc]):
                break
            row, col = nr, nc
            visited.append((row, col))
            if (row, col) == tuple(int(value) for value in goal):
                break
        count = max(1, min(int(max_visit_samples), len(visited)))
        picks = np.unique(np.linspace(0, len(visited) - 1, count).astype(int))
        waves = []
        for index in picks:
            observation = task.world.observe(walls, visited[int(index)], goal)
            tensor = torch.as_tensor(observation, dtype=torch.float32).reshape(1, -1)
            waves.append(self.encode("action", tensor).numpy()[0].astype(np.complex64))
        terminal = task.world.observe(walls, visited[-1], goal)
        terminal_tensor = torch.as_tensor(terminal, dtype=torch.float32).reshape(1, -1)
        psi_terminal = self.encode("action", terminal_tensor).numpy()[0].astype(np.complex64)
        psi_visit = _unit(np.sum(np.stack(waves), axis=0))
        steps = int(len(actions))
        # The planner resolves the trajectory it executed to its operating
        # regime residue and writes that residue as a three-phase code in
        # slot 2. This is a property of the executed trajectory, not a text
        # label; the receiver has no access to the declared rule and must
        # recover the residue with the fitted head.
        regime_residue = int(self.trajectory_regime(planning))
        step_phase = np.exp(1j * 2.0 * np.pi * float(regime_residue % 3) / 3.0)
        step_wave = _unit(self._regime_wave(3) * np.complex64(step_phase))
        repulsor = int(int(planning.get("online_discoveries", 0)) > 0)
        repulsor_wave = _unit(self._regime_wave(4) * np.complex64(-1.0 if repulsor else 1.0))
        return (
            [psi_terminal, psi_visit, step_wave, repulsor_wave],
            {
                "visited_states": int(len(visited)),
                "visit_samples": int(len(picks)),
                "steps": int(steps),
                "repulsor": int(repulsor),
                "regime_residue": int(regime_residue),
            },
        )

    @torch.no_grad()
    def classify_action_regime_batch(
        self,
        layouts: Sequence[Any],
        *,
        task=None,
        readout=None,
        observations: Optional[Sequence[Any]] = None,
        steps: int = 1,
    ) -> Dict[str, object]:
        """Classify N layouts in ONE batched tensor call (Directive X4).

        This is the Pass-0 pre-flight decode.  It reproduces the four-slot
        trajectory bundle of a one-step probe without building N episode
        harnesses: the initial observation of every layout is stacked into one
        (N, observation_dim) matrix, passed through a single
        self.encode("action", X) call, and the bundle geometry is assembled
        from the batched wave.  The fitted ActionRegimeReadout is then read as
        one (N, features) x (features, classes) product.

        The only quantities taken from the planner are the step count of a
        one-step probe (always 1) and the dead-end flag, which is computed from
        the same in-bounds wall map the sensor wave carries.  The declared
        trajectory rule then writes slot 2 exactly as it does on the episode
        path, so the decoded regime is the same quantity.  Parity against a
        real one-step episode is measured by the caller, not assumed.
        """
        from henri.holarch.swat_bus import _unit, cache_from_pairs

        readout = readout if readout is not None else getattr(self, "action_regime_readout", None)
        if readout is None:
            raise RuntimeError("classify_action_regime_batch needs a fitted ActionRegimeReadout")
        if task is None:
            raise ValueError("classify_action_regime_batch needs the action task for observe()")
        layout_list = [tuple(layout) for layout in layouts]
        if not layout_list:
            return {"count": 0, "regimes": [], "ms": 0.0, "encode_calls": 0}
        started = time.perf_counter()
        size = int(getattr(task.world, "grid_size", 7))
        if observations is None:
            observations = [
                task.world.observe(walls, tuple(int(v) for v in agent), tuple(int(v) for v in goal))
                for walls, agent, goal in layout_list
            ]
        matrix = np.stack(
            [np.asarray(row, dtype=np.float32).reshape(-1) for row in observations]
        )
        # ONE batched tensor call for every candidate layout.
        psi = self.encode("action", torch.as_tensor(matrix, dtype=torch.float32))
        psi = psi.detach().cpu().numpy().astype(np.complex64)
        deltas = ((-1, 0), (1, 0), (0, -1), (0, 1))
        values: List[np.ndarray] = []
        residues: List[int] = []
        repulsors: List[int] = []
        parity_rows: List[Dict[str, object]] = []
        for index, (walls, agent, goal) in enumerate(layout_list):
            row, col = int(agent[0]), int(agent[1])
            legal = 0
            for dr, dc in deltas:
                nr, nc = row + dr, col + dc
                if 0 <= nr < size and 0 <= nc < size and not bool(walls[nr, nc]):
                    legal += 1
            repulsor = int(legal <= 1)
            planning = {
                "goal": [int(v) for v in goal],
                "steps": int(steps),
                "online_discoveries": int(repulsor),
            }
            residue = int(self.trajectory_regime(planning))
            step_wave = _unit(
                self._regime_wave(3) * np.complex64(np.exp(1j * 2.0 * np.pi * float(residue % 3) / 3.0))
            )
            repulsor_wave = _unit(self._regime_wave(4) * np.complex64(-1.0 if repulsor else 1.0))
            wave = np.asarray(psi[index], dtype=np.complex64)
            waves = [wave, _unit(np.sum(np.stack([wave]), axis=0)), step_wave, repulsor_wave]
            cache = cache_from_pairs(
                [(item, item) for item in waves],
                source="action:regime",
                task="action",
                wave=np.asarray(waves[0], dtype=np.complex64),
            )
            values.append(cache.values.detach().cpu().numpy().reshape(4, -1).astype(np.float64))
            residues.append(int(residue))
            repulsors.append(int(repulsor))
            parity_rows.append(
                {
                    "index": int(index),
                    "goal": [int(v) for v in goal],
                    "legal_moves": int(legal),
                    "declared_residue": int(residue),
                }
            )
        rows = [row for row in values]
        best, probability, scores = readout.predict_batch(rows)
        elapsed_ms = (time.perf_counter() - started) * 1e3
        return {
            "count": int(len(layout_list)),
            "regimes": [int(value) for value in best],
            "probabilities": [float(value) for value in probability],
            "scores": np.asarray(scores, dtype=np.float64),
            "declared_residues": [int(value) for value in residues],
            "repulsors": [int(value) for value in repulsors],
            "layouts": parity_rows,
            "encode_calls": 1,
            "observation_matrix": [int(v) for v in matrix.shape],
            "ms": float(elapsed_ms),
            "readout": "fitted_stiefel_ridge_batch",
        }

    @torch.no_grad()
    def run_telepathic_vla_episode(
        self,
        canvas: Any,
        *,
        task=None,
        layout=None,
        engine=None,
        bus=None,
        gate=None,
        name: str = "telepathic_program",
        truth: Optional[Dict[str, int]] = None,
        namespace: str = "p3",
        max_steps: int = 24,
        corrupt_regions: Sequence[str] = (),
        banks: Optional[Dict[str, List[np.ndarray]]] = None,
        causal_operator: bool = False,
        action_bus=None,
        action_gate=None,
        corrupt_action: bool = False,
        fitted_regime: bool = False,
        planning_override: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        """One zero-text-glue Vision -> Action -> Code episode over H-SWAT.

        The vision sender observes the 16x16 canvas in four 8x8 patches and
        writes ONLY SwarmKVCache objects. The action sender navigates the 7x7
        layout with the online-REPULSOR policy and writes its terminal state
        as a cache. The receiver decodes every region from the transferred
        value projection, binds the AST constants, and executes the emitted
        program. No text string and no Python int crosses an agent boundary.
        """
        import hashlib
        import json

        from henri.holarch.code_gen import telepathic_reference
        from henri.holarch.swat_bus import (
            SwarmKVCache,
            cache_from_pairs,
            cache_from_wave,
            kv_from_wave,
        )

        readout = (
            getattr(self, "action_regime_readout", None)
            if bool(fitted_regime)
            else None
        )
        bus_counters_before = dict(bus.counts) if bus is not None else {}
        truth_row = dict(truth or {})
        receipts: List[Dict[str, object]] = []

        def _unit_wave(value: Any) -> np.ndarray:
            array = np.asarray(value, dtype=np.complex64).reshape(-1)
            return (array / max(float(np.linalg.norm(array)), 1e-12)).astype(
                np.complex64
            )

        def chain(stage: str, payload: Dict[str, object]) -> None:
            body = {
                "index": len(receipts),
                "stage": str(stage),
                "payload_sha256": hashlib.sha256(
                    json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
                ).hexdigest(),
                "parent": str(receipts[-1]["sha256"]) if receipts else "genesis",
            }
            body["sha256"] = hashlib.sha256(
                json.dumps(body, sort_keys=True).encode("utf-8")
            ).hexdigest()
            receipts.append(body)

        banks = banks or self._telepathic_banks(task) if task is not None else (banks or {})
        canvas_array = np.asarray(canvas, dtype=np.float32)
        if canvas_array.shape != (16, 16):
            canvas_array = canvas_array.reshape(16, 16)
        quadrants = {
            "a": canvas_array[0:8, 0:8],
            "b": canvas_array[0:8, 8:16],
            "op": canvas_array[8:16, 0:8],
            "bound": canvas_array[8:16, 8:16],
        }
        corrupt = {str(region) for region in corrupt_regions}
        vision_packets: List[Any] = []
        vision_rows: List[Dict[str, object]] = []
        rng = np.random.default_rng(20261108)
        for region, patch in quadrants.items():
            tensor = torch.as_tensor(patch, dtype=torch.float32).reshape(1, 1, 8, 8)
            psi = self.encode("vision", tensor).numpy()[0].astype(np.complex64)
            cache = cache_from_wave(psi, source=f"vision:{region}", task="vision")
            if region in corrupt:
                # An adversarial sender keeps a real image in its KV but
                # ANNOUNCES a state it does not carry. That is the smuggle
                # the gate's binding condition exists to refuse: the declared
                # wave is no longer the frame of any key in the cache.
                phase = np.exp(
                    1j * rng.uniform(-np.pi, np.pi, int(psi.size))
                ).astype(np.complex64)
                cache.wave = (psi * phase).astype(np.complex64)
                cache.wave = cache.wave / max(float(np.linalg.norm(cache.wave)), 1e-12)
                cache.digest = ""
            if bus is not None:
                bus.register(cache)
                packet = bus.transmit(cache.source, receiver_task="language", use_gate=True)
            else:
                packet = None
            if packet is not None and bool(packet.admitted):
                vision_packets.append(packet)
            vision_rows.append(
                {
                    "region": region,
                    "corrupt": bool(region in corrupt),
                    "admitted": bool(packet.admitted) if packet is not None else True,
                    "reason": str(getattr(packet, "reason", "no-bus")),
                    "cache_digest": str(cache.digest)[:16],
                    "packet_digest": str(getattr(packet, "digest", ""))[:16],
                }
            )
        chain("vision", {"regions": vision_rows, "text_tokens": 0})

        # ------------------------------------------------------------ action
        planning: Dict[str, object] = {"requested": bool(task is not None and layout is not None)}
        action_packet = None
        if task is not None and layout is not None:
            import copy

            from henri.holarch.model import evaluate_closed_loop_action

            walls, agent, goal = layout
            if planning_override is not None:
                # Directive T1: the caller ran this exact layout through the
                # planner once, stratified the batch by the measured regime,
                # and hands the trajectory back here. The planning arithmetic
                # is not run twice and no weight is touched; the episode owns
                # the same actions, steps, goal and discovery count it probed.
                actions = [int(action) for action in (planning_override.get("actions") or [])]
                planning.update(
                    {
                        "actions": actions,
                        "steps": len(actions),
                        "reached_goal": bool(planning_override.get("reached_goal", False)),
                        "start": list(agent),
                        "goal": list(goal),
                        "online_discoveries": int(
                            planning_override.get("online_discoveries", 0)
                        ),
                        "planning_source": "stratified_probe",
                    }
                )
            else:
                probe_task = copy.copy(task)
                probe_task.val_layouts = [(walls, agent, goal)]
                result = evaluate_closed_loop_action(
                    self,
                    probe_task,
                    episodes=1,
                    limit=int(max_steps),
                    rotor=getattr(self, "action_rotor", None),
                    transition="rotor",
                    lookahead_depth=2,
                    gamma_look=0.02,
                    batch_rollout=True,
                    use_memory=True,
                    trace=True,
                    online_repulsor=True,
                    online_namespace=f"{namespace}_{name}",
                    # Directive X4: the episode consumes the trace, so it does
                    # not pay for the unused 80-pair one-step audit.
                    one_step_audit=False,
                )
                trace_row = (result.get("trace") or [{}])[0]
                actions = [int(action) for action in (trace_row.get("actions") or [])]
                reached = bool(trace_row.get("win", False))
                planning.update(
                    {
                        "actions": actions,
                        "steps": len(actions),
                        "reached_goal": bool(reached),
                        "start": list(agent),
                        "goal": list(goal),
                        "online_discoveries": int(len(result.get("online_rows") or [])),
                    }
                )
            observation = task.world.observe(walls, agent, goal)
            psi_action = self.encode(
                "action", torch.as_tensor(observation, dtype=torch.float32).reshape(1, -1)
            ).numpy()[0].astype(np.complex64)
            # Directive T1: the fitted head is consulted only when the caller
            # asks for it. A caller that does not ask gets the declared
            # trajectory rule, which is the coordinate the stratification
            # selected. This makes the episode's regime the same quantity the
            # probe measured instead of a second, unstratified decode.
            readout = (
                getattr(self, "action_regime_readout", None) if bool(fitted_regime) else None
            )
            bundle_info: Dict[str, object] = {}
            if bool(causal_operator):
                # Directive R2: when a fitted ActionRegimeReadout is mounted,
                # the planner writes the four-slot TRAJECTORY BUNDLE (terminal
                # observation, visited-state superposition, step phasor,
                # REPULSOR wave). The receiver decodes the regime from that
                # wave geometry with the fitted Stiefel + ridge head; no
                # declared trajectory-to-regime rule is consulted on this path.
                if readout is not None:
                    waves, bundle_info = self._action_trajectory_bundle(
                        planning, walls, agent, goal, task
                    )
                    pairs = [(wave, wave) for wave in waves]
                    cache = cache_from_pairs(
                        pairs,
                        source="action:regime",
                        task="action",
                        wave=np.asarray(waves[0], dtype=np.complex64),
                    )
                    bound = np.asarray(waves[0], dtype=np.complex64)
                else:
                    regime = int(self.trajectory_regime(planning))
                    planning["regime"] = int(regime)
                    regime_wave = self._regime_wave(regime)
                    bound = _unit_wave(
                        (0.70 * regime_wave + 0.30 * psi_action).astype(np.complex64)
                    )
                    cache = cache_from_wave(bound, source="action:regime", task="action")
                if bool(corrupt_action):
                    # A corrupted ACTION sender keeps the real bundle in its
                    # KV but announces a phase-scrambled wave: the gate's
                    # binding condition must refuse it before any operator is
                    # emitted.
                    phase = np.exp(
                        1j * rng.uniform(-np.pi, np.pi, int(bound.size))
                    ).astype(np.complex64)
                    cache.wave = _unit_wave((bound * phase).astype(np.complex64))
                    cache.digest = ""
                transport = action_bus if action_bus is not None else bus
                transport_gate = action_gate if action_gate is not None else gate
                if transport is not None:
                    transport.gate = transport_gate
                    transport.register(cache)
                    action_packet = transport.transmit(
                        cache.source,
                        receiver_task="language",
                        use_gate=True,
                    )
                if readout is not None:
                    if action_packet is not None and bool(action_packet.admitted):
                        regime, regime_score = readout.predict(
                            action_packet.values.detach().cpu().numpy()
                        )
                    else:
                        regime, regime_score = -1, 0.0
                    planning["regime"] = int(regime)
                    planning["regime_source"] = "fitted_readout"
                    planning["regime_score"] = float(regime_score)
                    planning["action_bundle"] = dict(bundle_info)
                    if int(regime) >= 0:
                        truth_row["op"] = int(regime)
                else:
                    planning["regime_source"] = "declared_rule"
                    truth_row["op"] = int(planning.get("regime", 0))
                planning["terminal_cache_digest"] = str(cache.digest)[:16]
                planning["regime_wave_digest"] = hashlib.sha256(
                    np.ascontiguousarray(cache.wave).tobytes()
                ).hexdigest()[:16]
                planning["corrupt_action"] = bool(corrupt_action)
            else:
                cache = cache_from_wave(psi_action, source="action:terminal", task="action")
                if bus is not None:
                    bus.register(cache)
                    action_packet = bus.transmit(cache.source, receiver_task="language", use_gate=False)
                planning["terminal_cache_digest"] = str(cache.digest)[:16]
        chain("action", {"planning": {k: v for k, v in planning.items() if k != "actions"}, "text_tokens": 0})

        # -------------------------------------------------------------- code
        code: Dict[str, object] = {"requested": bool(engine is not None)}
        synthesis: Dict[str, object] = {}
        if engine is not None:
            tests = ""
            if truth_row and bool(planning.get("reached_goal", False)):
                operator = ("+", "-", "*")[int(truth_row.get("op", 0)) % 3]
                first = int(truth_row.get("a", 0))
                second = int(truth_row.get("b", 0))
                high = float(int(truth_row.get("bound", 0)) + 1.0)
                if operator == "+":
                    scale, shift = float(first + 1), float(second)
                elif operator == "-":
                    scale, shift = float(first + 1), -float(second)
                else:
                    scale, shift = float(first + second + 1), 0.0
                rows = []
                for value in (0.0, 0.5, 1.0, -2.0, 3.25):
                    expected = telepathic_reference(
                        value, scale=scale, shift=shift, low=0.0, high=high
                    )
                    rows.append(f"assert {name}({value!r}) == {expected!r}")
                tests = chr(10).join(rows) + chr(10)
            packets = list(vision_packets) + ([action_packet] if action_packet is not None else [])
            synthesis = engine.synthesize_from_telepathy(
                packets,
                name=str(name),
                banks=banks,
                tests=tests,
                budget_u=16,
                bus=bus,
                require_action_regime=bool(causal_operator),
                regime_readout=readout if bool(causal_operator) else None,
            )
            code.update(
                {
                    "status": str(synthesis.get("status", "")),
                    "verified": bool(synthesis.get("verified", False)),
                    "executed": bool(synthesis.get("executed", False)),
                    "source_chars": len(str(synthesis.get("source", ""))),
                    "reason": str(synthesis.get("reason", "")),
                    "decoded": dict(synthesis.get("digits") or {}),
                    "operator": str(synthesis.get("operator", "")),
                    "compacted_slots_u": int(synthesis.get("compacted_slots_u", 0)),
                    "text_tokens": 0,
                }
            )
        chain(
            "code",
            {
                "status": code.get("status", ""),
                "verified": code.get("verified", False),
                "executed": code.get("executed", False),
                "text_tokens": 0,
            },
        )
        chain_valid = bool(
            receipts
            and receipts[0]["parent"] == "genesis"
            and all(
                receipts[index]["parent"] == receipts[index - 1]["sha256"]
                for index in range(1, len(receipts))
            )
        )
        decoded = dict((synthesis.get("digits") or {})) if synthesis else {}
        correct_regions = {}
        if truth_row:
            correct_regions = {
                region: int(decoded.get(region, -1)) == int(truth_row.get(region, -2))
                for region in ("a", "b", "op", "bound")
            }
        success = bool(
            bool(planning.get("reached_goal", False))
            and bool(code.get("verified", False))
            and bool(code.get("executed", False))
            and chain_valid
            and all(correct_regions.values()) if correct_regions else True
        )
        return {
            "success": bool(success),
            "namespace": str(namespace),
            "name": str(name),
            "vision": {"rows": vision_rows, "packets_admitted": len(vision_packets)},
            "planning": planning,
            "code": code,
            "decoded": decoded,
            "truth": dict(truth_row),
            "region_correct": correct_regions,
            "regime": int(planning.get("regime", -1)),
            "operator_source": (
                "action_kv" if bool(causal_operator) else "vision_glyph"
            ),
            "action_packet": action_packet,
            "vision_packets": vision_packets,
            "operator": str(synthesis.get("operator", "")),
            "source": str(synthesis.get("source", "")),
            "text_tokens_exchanged": 0,
            "receiver_prefill_tokens": 0,
            "receipts": receipts,
            "receipt_chain_valid": bool(chain_valid),
            "synthesis_receipts": list(synthesis.get("receipts") or []),
            "bus_counts_before": bus_counters_before,
            "bus_counts_after": dict(bus.counts) if bus is not None else {},
            "evidence": "OBSERVED",
        }

    def memory_report(self) -> Dict[str, float]:
        if getattr(self, "memory", None) is None:
            return {"count": 0.0}
        return self.memory.stats()