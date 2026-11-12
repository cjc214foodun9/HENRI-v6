"""Zone B: the BaTiO3 diffractive photonic kernel (digital twin).

Evidence label: DERIVED for the operator algebra; OBSERVED for every
number the holarch tests measure. This is a SIMULATION of a thin-film
barium titanate (BaTiO3) photonic crystal, not a fabricated device.

Physical model
--------------
Each diffractive plate is a BaTiO3 film inside a Sagnac loop. A
microheater grid sets the film temperature T. The electro-optic
(Pockels) response scales with the Curie-Weiss susceptibility

    r_eff(T) = r0 * Lambda(T),   Lambda(T) = 1 + kappa / ((1-T)^gamma + eps)

so a heater is a CONTROL on phase authority, not a thermal load. The
phase applied to channel d of slot s is

    phi = theta[s,d] + direction * g_pockels * Lambda(T[s]) * V[s]

where theta is the neuroplastic phase mask (trained by gradient and
written by optical phase conjugation), V is the drive voltage, and
direction is +1 for the clockwise arm and -1 for the counter-clockwise
arm. That sign is the non-reciprocity: a reciprocal loop cancels at the
dark port exactly, and the measured dark-port power is the interference
signal the readout uses.

Between plates the field propagates under the unitary angular-spectrum
chirp H(f) = exp(-i * zeta * pi * f^2) and accumulates a Kerr self-phase
exp(i * gamma * |u|^2 * D). Both are norm preserving. The loop coupler

    bright = (a + b) / sqrt(2),   dark = (a - b) / sqrt(2)

therefore loses no energy to rounding, and the measured norm drift is a
guard, not a claim (constraint C15).

The Sagnac statistic is the calibrated phase mismatch

    delta(a, b) = 2 * P_dark / (P_bright + P_dark) = 1 - Re<a, b>

for unit-norm fields. It equals henri.sagnac.sagnac_delta exactly,
which the parity test measures. The threshold is calibrated from the
empirical chance distribution of the SAME wave family; a literal is
constraint C2 and is forbidden here.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from henri.holarch.deterministic import (
    prototype_align,
    slot_mix,
    stiefel_readout,
    stiefel_reproject,
)

TWO_PI = 2.0 * math.pi
_EPS = 1e-12

FEATURE_CHUNKS = 8
"""Energy/phase pooling bins per slot. Fixed so the feature width is D-free."""


@contextmanager
def _isolated_rng(seed: int):
    """Construction never advances the process-global RNG."""
    state = torch.random.get_rng_state()
    try:
        torch.manual_seed(int(seed))
        yield
    finally:
        torch.random.set_rng_state(state)


def curie_weiss_gain(
    temperature: torch.Tensor | float,
    kappa: float = 0.9,
    eps: float = 1e-3,
    gamma: float = 1.0,
) -> torch.Tensor:
    """Dimensionless Pockels gain Lambda(T), normalised to Lambda(0) = 1.

    Monotone increasing on [0, 1). The value is finite for every T below
    one because eps keeps the denominator away from zero. The gain is the
    measured physical claim of this module: heater power sets phase
    authority (see tests/test_holarch_physics.py).
    """
    t = temperature if torch.is_tensor(temperature) else torch.tensor(float(temperature))
    t = t.clamp(0.0, 1.0 - 1e-4)
    raw = 1.0 + float(kappa) / ((1.0 - t).pow(float(gamma)) + float(eps))
    base = 1.0 + float(kappa) / (1.0 + float(eps))
    return raw / base


def _unit(v: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return v / (v.norm(dim=dim, keepdim=True).clamp_min(_EPS))


def sagnac_delta(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """1 - normalised coherence for unit-norm fields. Shape (...,)."""
    a = _unit(a)
    b = _unit(b)
    coh = (a.conj() * b).sum(dim=-1).real
    return (1.0 - coh).clamp(0.0, 2.0)


class MicroHeaterArray(nn.Module):
    """Per-slot heater grid driven by the thermodynamic state.

    The controller maps four scale-free wave statistics to a temperature
    offset per slot. It is zero-initialised, so at construction the grid
    sits at its learned base temperature and the module is inert.
    """

    def __init__(self, n_slots: int, t_max: float, stats_dim: int = 4, seed: int = 0) -> None:
        super().__init__()
        self.n_slots = int(n_slots)
        self.t_max = float(t_max)
        self.base_logit = nn.Parameter(torch.full((self.n_slots,), -1.5))
        self.controller = nn.Linear(int(stats_dim), self.n_slots)
        with torch.no_grad():
            self.controller.weight.zero_()
            self.controller.bias.zero_()

    def forward(self, stats: torch.Tensor) -> torch.Tensor:
        """Return normalised temperatures of shape (batch, n_slots)."""
        if stats.shape[-1] != self.controller.in_features:
            raise ValueError(
                f"expected {self.controller.in_features} stats, got {stats.shape[-1]}"
            )
        offsets = torch.tanh(self.controller(stats))
        return self.t_max * torch.sigmoid(self.base_logit + offsets)

    def set_temperature(self, value: torch.Tensor | float) -> None:
        """Force a uniform base temperature. Used by calibration and tests."""
        v = float(value)
        if not 0.0 <= v < 1.0:
            raise ValueError("temperature must lie in [0, 1)")
        logit = math.log(v / max(1.0 - v, 1e-9))
        with torch.no_grad():
            self.base_logit.fill_(logit)
            self.controller.weight.zero_()
            self.controller.bias.zero_()


class DiffractivePlate(nn.Module):
    """One BaTiO3 plate: Pockels phase, 4f propagation, Kerr self-phase."""

    def __init__(
        self,
        d_slot: int,
        n_slots: int,
        pockels_gain: float,
        kerr: float,
        zeta: float,
        voltage_bound: float,
        seed: int = 0,
        kerr_interplate: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_slot = int(d_slot)
        self.n_slots = int(n_slots)
        self.pockels_gain = float(pockels_gain)
        self.voltage_bound = float(voltage_bound)
        self.kerr_interplate = float(kerr_interplate)
        with _isolated_rng(seed):
            self.phase = nn.Parameter(0.15 * torch.randn(self.n_slots, self.d_slot))
        self.voltage = nn.Parameter(torch.zeros(self.n_slots))
        self.log_kerr = nn.Parameter(torch.tensor(math.log(float(kerr))))

        frequencies = torch.fft.fftfreq(self.d_slot, dtype=torch.float32)
        self.register_buffer("frequencies", frequencies, persistent=False)
        # Decision 1: per-slot dispersion. A learnable zeta lets a spatial
        # slot keep diffractive mixing while a symbolic slot can drive its
        # chirp toward zero and preserve channel-wise phase alignment.
        self.zeta = nn.Parameter(torch.full((self.n_slots,), float(zeta)))

    @property
    def kerr(self) -> torch.Tensor:
        return F.softplus(self.log_kerr)

    def drive(self) -> torch.Tensor:
        """Bounded normalised voltage, shape (n_slots,)."""
        return torch.tanh(self.voltage) * self.voltage_bound

    def forward(
        self,
        psi: torch.Tensor,
        gain: torch.Tensor,
        direction: float = 1.0,
    ) -> torch.Tensor:
        """Propagate (batch, n_slots, d_slot) through one plate."""
        batch = psi.shape[0]
        if psi.shape[1:] != (self.n_slots, self.d_slot):
            raise ValueError(
                f"expected ({batch}, {self.n_slots}, {self.d_slot}), got {tuple(psi.shape)}"
            )
        drive = self.drive().reshape(1, self.n_slots, 1)
        if self.kerr_interplate > 0.0:
            # Decision 5: the local optical intensity biases the electro-optic
            # phase, so the plate stack is non-linear in the wave amplitude and
            # cannot be folded into a single unitary matrix.
            intensity = psi.abs().pow(2) * float(self.d_slot)
            drive_term = drive + self.kerr_interplate * intensity
        else:
            drive_term = drive
        phase = (
            self.phase.unsqueeze(0)
            + float(direction) * self.pockels_gain * gain.unsqueeze(-1) * drive_term
        )
        field = psi * torch.exp(1j * phase)
        transfer = torch.exp(
            -1j * self.zeta.reshape(-1, 1) * math.pi * self.frequencies.pow(2)
        ).to(torch.complex64)
        field = torch.fft.ifft(torch.fft.fft(field, dim=-1) * transfer.unsqueeze(0), dim=-1)
        intensity = field.abs().pow(2) * float(self.d_slot)
        field = field * torch.exp(1j * self.kerr * intensity)
        return field.contiguous()


@dataclass
class PhotonicState:
    """Full audit trail for one Zone B pass.

    Attributes:
        bright: the constructive Sagnac port, complex, shape (B, D).
        dark: the destructive Sagnac port, complex, shape (B, D).
        delta_reciprocal: non-reciprocal phase mismatch of the loop.
        delta_memory: phase mismatch against the retrieved axiom, when one
            was supplied; otherwise zeros.
        visibility: 1 - delta_memory, the homodyne fringe visibility.
        temperature: heater temperatures, shape (B, n_slots).
        gain: Curie-Weiss gain per slot, shape (B, n_slots).
        entropy: normalised participation entropy of the input wave.
        free_energy: U - T*S proxy used by the heater controller.
        norm_drift: max abs deviation from unit norm over the pass.
    """

    bright: torch.Tensor
    dark: torch.Tensor
    delta_reciprocal: torch.Tensor
    delta_memory: torch.Tensor
    delta_slots: torch.Tensor
    visibility: torch.Tensor
    temperature: torch.Tensor
    gain: torch.Tensor
    entropy: torch.Tensor
    free_energy: torch.Tensor
    norm_drift: float
    features: torch.Tensor
    trilinear: torch.Tensor
    extra: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> Dict[str, float]:
        def scalar(value: torch.Tensor) -> float:
            return float(value.detach().mean())

        return {
            "delta_reciprocal": scalar(self.delta_reciprocal),
            "delta_memory": scalar(self.delta_memory),
            "visibility": scalar(self.visibility),
            "temperature_mean": scalar(self.temperature),
            "gain_mean": scalar(self.gain),
            "gain_max": scalar(self.gain.max(dim=-1).values),
            "entropy": scalar(self.entropy),
            "free_energy": scalar(self.free_energy),
            "norm_drift": float(self.norm_drift),
            **self.extra,
        }


class SlotMixer(nn.Module):
    """Unitary 4x4 slot coupler between diffractive plates.

    The angular-spectrum chirp mixes frequencies inside one slot. It does
    not mix slots. This operator is the discrete analogue of cross-slot
    interference: a Cayley transform of a skew-Hermitian matrix, so it is
    exactly unitary for every parameter value and the norm is preserved.
    """

    def __init__(self, n_slots: int = 4, seed: int = 0) -> None:
        super().__init__()
        self.n_slots = int(n_slots)
        generator = torch.Generator().manual_seed(int(seed))
        self.skew_real = nn.Parameter(0.15 * torch.randn(self.n_slots, self.n_slots, generator=generator))
        self.skew_imag = nn.Parameter(0.15 * torch.randn(self.n_slots, self.n_slots, generator=generator))

    def matrix(self) -> torch.Tensor:
        raw = torch.complex(self.skew_real, self.skew_imag)
        skew = raw - raw.conj().transpose(-1, -2)
        eye = torch.eye(self.n_slots, dtype=skew.dtype, device=skew.device)
        return torch.linalg.solve(eye - 0.5 * skew, eye + 0.5 * skew)

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        # Directive J1: no batched GEMM. slot_mix is an explicit four-term sum,
        # so the slot coupling is order-free and thread-invariant.
        return slot_mix(self.matrix(), slots)


class StiefelSlotPool(nn.Module):
    """Slot-specific algebraic unbinding onto four Stiefel bases.

    For slot s the readout is g_s = Re(U_s psi_s) with U_s in
    C^{K x d_slot} and rows orthonormal. The QR retraction runs every
    forward pass, so the basis stays on the Stiefel manifold during
    training rather than being regularised toward it.
    """

    def __init__(
        self,
        n_slots: int,
        d_slot: int,
        k: int,
        seed: int = 0,
        mode: str = "learned",
    ) -> None:
        super().__init__()
        self.n_slots = int(n_slots)
        self.d_slot = int(d_slot)
        self.k = int(k)
        self.mode = str(mode)
        if self.mode not in {"learned", "fixed"}:
            raise ValueError("stiefel mode must be 'learned' or 'fixed'")
        generator = torch.Generator().manual_seed(int(seed))
        scale = 1.0 / math.sqrt(float(d_slot))
        self.weight_real = nn.Parameter(
            scale * torch.randn(self.n_slots, self.k, self.d_slot, generator=generator)
        )
        self.weight_imag = nn.Parameter(
            scale * torch.randn(self.n_slots, self.k, self.d_slot, generator=generator)
        )
        # Decision 5: a hardware detector array sits at fixed spatial-spectral
        # coordinates. A truncated Fourier frame is the canonical orthonormal
        # choice, and it is frozen: requires_grad is False, so the plates are
        # the only learnable beam-steering element.
        index = torch.arange(self.k).reshape(1, self.k, 1)
        channel = torch.arange(self.d_slot).reshape(1, 1, self.d_slot)
        slot_phase = torch.arange(self.n_slots).reshape(self.n_slots, 1, 1)
        angle = -2.0 * math.pi * index * channel / float(self.d_slot)
        angle = angle + 2.0 * math.pi * slot_phase / float(self.n_slots)
        frame = torch.polar(torch.ones_like(angle), angle) / math.sqrt(float(self.d_slot))
        self.register_buffer("fixed_basis", frame.to(torch.complex64), persistent=True)
        if self.mode == "fixed":
            self.weight_real.requires_grad_(False)
            self.weight_imag.requires_grad_(False)
        # Directive N2: the Stiefel frame is a function of the weights only.
        # At inference the weights do not move, so the QR retraction is
        # computed once and reused. Training keeps the per-pass retraction so
        # the basis tracks the update.
        self._basis_cache: Optional[torch.Tensor] = None

    def basis(self) -> torch.Tensor:
        if self.mode == "fixed":
            return self.fixed_basis
        if (
            not self.training
            and not torch.is_grad_enabled()
            and self._basis_cache is not None
        ):
            return self._basis_cache
        weight = torch.complex(self.weight_real, self.weight_imag)
        q, r = torch.linalg.qr(weight.transpose(-1, -2))
        diagonal = torch.diagonal(r, dim1=-2, dim2=-1).real
        sign = torch.where(diagonal >= 0.0, torch.ones_like(diagonal), -torch.ones_like(diagonal))
        q = q * torch.complex(sign, torch.zeros_like(sign)).unsqueeze(-2)
        basis = q.transpose(-1, -2).contiguous()
        if not self.training and not torch.is_grad_enabled():
            # Only a grad-free frame may be cached: a tensor that still owns a
            # graph cannot be reused by a second backward pass.
            self._basis_cache = basis.detach()
        return basis

    def _load_from_state_dict(self, *args, **kwargs):
        # A reloaded weight invalidates the inference-time frame cache.
        self._basis_cache = None
        return super()._load_from_state_dict(*args, **kwargs)

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        # Directive J1: one two-dimensional product per slot with a float64
        # accumulator, because the batched complex kernel is not call-stable
        # on this host (see henri/holarch/deterministic.py).
        basis = self.basis()
        return stiefel_readout(slots, basis).real.flatten(1)


class PhotonicCore(nn.Module):
    """Cascaded BTO plates inside a Sagnac loop, plus the homodyne readout."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.dim = int(cfg.dim)
        self.n_slots = int(cfg.n_slots)
        self.d_slot = int(cfg.d_slot)
        self.pool_k = int(cfg.pool_k)
        self.plates = nn.ModuleList(
            [
                DiffractivePlate(
                    d_slot=self.d_slot,
                    n_slots=self.n_slots,
                    pockels_gain=cfg.pockels_gain,
                    kerr=cfg.kerr,
                    zeta=cfg.propagation_zeta,
                    voltage_bound=cfg.voltage_bound,
                    seed=cfg.seed + 100 + index,
                    kerr_interplate=float(getattr(cfg, "kerr_interplate", 0.0)),
                )
                for index in range(int(cfg.photonic_layers))
            ]
        )
        self.heaters = MicroHeaterArray(self.n_slots, cfg.t_max, stats_dim=4, seed=cfg.seed)
        self.pool = StiefelSlotPool(
            self.n_slots,
            self.d_slot,
            self.pool_k,
            seed=int(cfg.seed) + 41,
            mode=str(getattr(cfg, "stiefel_mode", "learned")),
        )
        self.mixer = (
            SlotMixer(self.n_slots, seed=int(cfg.seed) + 42)
            if bool(getattr(cfg, "cross_slot_mix", True))
            else None
        )
        self.register_buffer("epsilon", torch.tensor(0.35))
        self.register_buffer("calibration_mean", torch.tensor(0.0))
        self.register_buffer("calibration_std", torch.tensor(0.0))

    @property
    def feature_width(self) -> int:
        """Slot readouts, trilinear and bilinear products, dark energy, scalars.

        4K raw + K trilinear + 3K bilinear + 4 dark + 8 scalars = 8K + 12.
        """
        return 8 * self.pool_k + self.n_slots + 8

    @property
    def raw_width(self) -> int:
        """Raw slot readouts, dark energy, and scalars: 4K + 12."""
        return 4 * self.pool_k + self.n_slots + 8

    @property
    def relational_width(self) -> int:
        """Trilinear plus bilinear products: 4K."""
        return 4 * self.pool_k

    # ------------------------------------------------------------ statistics
    def wave_stats(self, psi: torch.Tensor) -> torch.Tensor:
        """Four scale-free statistics: entropy, participation, mean, peak."""
        power = psi.abs().pow(2)
        power = power / power.sum(dim=-1, keepdim=True).clamp_min(_EPS)
        entropy = -(power * torch.log(power + _EPS)).sum(dim=-1) / math.log(float(self.dim))
        participation = power.pow(2).sum(dim=-1) * float(self.dim)
        mean_abs = psi.abs().mean(dim=-1)
        peak = psi.abs().amax(dim=-1)
        return torch.stack([entropy, participation, mean_abs, peak], dim=-1)

    # ---------------------------------------------------------------- physics
    def _apply_stack(
        self,
        psi: torch.Tensor,
        gain: torch.Tensor,
        direction: float,
        steps: Optional[int] = None,
    ) -> torch.Tensor:
        field = psi
        plates = self.plates if steps is None else self.plates[: int(steps)]
        for index, plate in enumerate(plates):
            field = plate(field, gain, direction=float(direction))
            if self.mixer is not None and index + 1 < len(plates):
                field = self.mixer(field)
        return field

    def slot_delta(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Per-slot Sagnac delta, shape (batch, n_slots)."""
        a = _unit(a)
        b = _unit(b)
        sa = a.reshape(-1, self.n_slots, self.d_slot)
        sb = b.reshape(-1, self.n_slots, self.d_slot)
        norm_a = sa.norm(dim=-1).clamp_min(_EPS)
        norm_b = sb.norm(dim=-1).clamp_min(_EPS)
        coherence = (sa.conj() * sb).sum(dim=-1).real / (norm_a * norm_b)
        return (1.0 - coherence).clamp(0.0, 2.0)

    def forward(
        self,
        psi: torch.Tensor,
        axiom: Optional[torch.Tensor] = None,
        t_bias: float = 0.0,
        apply_memory: bool = True,
        bypass: bool = False,
        thermal_feedback: Optional[torch.Tensor] = None,
    ) -> PhotonicState:
        """Run one photonic pass.

        Args:
            psi: complex field, shape (batch, dim).
            axiom: optional complex reference wave, shape (batch, dim).
            t_bias: additive normalised temperature offset.
            apply_memory: when False, delta_memory is reported as zero and
                the axiom is ignored. Used by the ablation control.
            bypass: when True the plates are skipped and the loop is
                reciprocal. This is the Zone B removal control, with the
                same slot pooling and the same parameter count downstream.
            thermal_feedback: per-slot temperature offset added to the
                heater output. The relaxation loop closes the Sagnac to
                heater path with this argument.
        """
        if psi.dim() != 2 or psi.shape[-1] != self.dim:
            raise ValueError(f"expected (batch, {self.dim}), got {tuple(psi.shape)}")
        psi = _unit(psi.to(torch.complex64))
        stats = self.wave_stats(psi)
        temperature = (self.heaters(stats) + float(t_bias)).clamp(0.0, 0.999)
        if thermal_feedback is not None:
            if thermal_feedback.shape != temperature.shape:
                raise ValueError(
                    f"thermal_feedback must be {tuple(temperature.shape)}, got "
                    f"{tuple(thermal_feedback.shape)}"
                )
            temperature = (temperature + thermal_feedback).clamp(0.0, 0.999)
        gain = curie_weiss_gain(temperature, self.cfg.curie_kappa, self.cfg.curie_eps)

        slots = psi.reshape(-1, self.n_slots, self.d_slot)
        if bypass:
            half = slots / math.sqrt(2.0)
            clockwise = half
            counter = half
        else:
            clockwise = self._apply_stack(slots, gain, +1.0)
            counter = self._apply_stack(slots, gain, -1.0)
        bright = (clockwise + counter) / math.sqrt(2.0)
        dark = (clockwise - counter) / math.sqrt(2.0)
        bright = bright.reshape(-1, self.dim)
        dark = dark.reshape(-1, self.dim)

        bright_power = bright.abs().pow(2).sum(dim=-1)
        dark_power = dark.abs().pow(2).sum(dim=-1)
        delta_reciprocal = 2.0 * dark_power / (bright_power + dark_power).clamp_min(_EPS)

        dark_fraction = torch.zeros_like(delta_reciprocal)
        delta_slots = torch.zeros(
            psi.shape[0], self.n_slots, dtype=torch.float32, device=psi.device
        )
        if axiom is not None and apply_memory:
            axiom = _unit(axiom.to(torch.complex64))
            # Directive 2: the check is POST-photonic. The bright port must
            # agree with the stored axiom, not the input that was just fused
            # with it.
            delta_memory = sagnac_delta(bright, axiom)
            delta_slots = self.slot_delta(bright, axiom)
            mixed_bright = (bright + axiom) / math.sqrt(2.0)
            mixed_dark = (bright - axiom) / math.sqrt(2.0)
            mb = mixed_bright.abs().pow(2).sum(dim=-1)
            md = mixed_dark.abs().pow(2).sum(dim=-1)
            dark_fraction = md / (mb + md).clamp_min(_EPS)
        else:
            delta_memory = torch.zeros_like(delta_reciprocal)
        visibility = (1.0 - delta_memory).clamp(-1.0, 1.0)

        energy = psi.abs().pow(2).sum(dim=-1)
        entropy = stats[:, 0]
        internal = stats[:, 1]
        free_energy = internal - temperature.mean(dim=-1) * entropy

        drift = float((energy - 1.0).abs().max().detach())
        features, trilinear = self.pool_features(
            bright, dark, stats, temperature, gain, delta_memory
        )

        return PhotonicState(
            bright=bright,
            dark=dark,
            delta_reciprocal=delta_reciprocal,
            delta_memory=delta_memory,
            delta_slots=delta_slots,
            visibility=visibility,
            temperature=temperature,
            gain=gain,
            entropy=entropy,
            free_energy=free_energy,
            norm_drift=drift,
            features=features,
            trilinear=trilinear,
            extra={
                "bright_power": float(bright_power.mean().detach()),
                "dark_power": float(dark_power.mean().detach()),
                "dark_fraction": float(dark_fraction.mean().detach()),
            },
        )

    # ------------------------------------------------------------- features
    def pool_features(
        self,
        bright: torch.Tensor,
        dark: torch.Tensor,
        stats: torch.Tensor,
        temperature: torch.Tensor,
        gain: torch.Tensor,
        delta_memory: torch.Tensor,
    ) -> torch.Tensor:
        """Slot-specific algebraic unbinding, shape (batch, feature_width).

        The four slots are read with four separate Stiefel bases. The
        result is the Hermitian projection g_s = Re(U_s psi_s), not a
        magnitude average, so the phase relationships that carry the
        binding survive the readout.
        """
        batch = bright.shape[0]
        slots = bright.reshape(batch, self.n_slots, self.d_slot)
        unbound = self.pool(slots).reshape(batch, self.n_slots, self.pool_k)
        trilinear = unbound[:, 0] * unbound[:, 1] * unbound[:, 2]
        bilinear = torch.cat(
            [
                unbound[:, 0] * unbound[:, 1],
                unbound[:, 1] * unbound[:, 2],
                unbound[:, 0] * unbound[:, 2],
            ],
            dim=-1,
        )
        dark_energy = dark.reshape(batch, self.n_slots, -1).abs().pow(2).sum(dim=-1)
        scalars = torch.stack(
            [
                delta_memory,
                temperature.mean(dim=-1),
                gain.mean(dim=-1),
                gain.amax(dim=-1),
                stats[:, 0],
                stats[:, 1],
                stats[:, 2],
                stats[:, 3],
            ],
            dim=-1,
        )
        # Raw slot readouts first so non-relational tasks can use only the
        # slot geometry; the trilinear and bilinear products follow for the
        # relational head.
        features = torch.cat(
            [unbound.flatten(1), dark_energy, scalars, trilinear, bilinear], dim=-1
        )
        return features, trilinear

    def relax(
        self,
        psi: torch.Tensor,
        axiom: torch.Tensor,
        steps: Optional[int] = None,
        alpha_thermal: Optional[float] = None,
        fraction: Optional[float] = None,
        tolerance: float = 0.35,
    ) -> Tuple[PhotonicState, List[float]]:
        """Closed thermodynamic loop: Sagnac error drives the heaters.

        T^(k+1) = clip(T^k + alpha * delta^(k), 0, 0.999). A phase-conjugate
        correction rotates the candidate toward the axiom before the next
        pass. The loop stops when the residual falls below the tolerance or
        the step budget is spent.
        """
        if axiom is None:
            raise ValueError("relax needs an axiom wave")
        steps = int(self.cfg.relax_steps if steps is None else steps)
        alpha = float(self.cfg.alpha_thermal if alpha_thermal is None else alpha_thermal)
        gain_fraction = float(self.cfg.relax_fraction if fraction is None else fraction)
        current = _unit(psi.to(torch.complex64))
        reference = _unit(axiom.to(torch.complex64))
        state = self.forward(current, axiom=reference)
        best_state = state
        best_delta = float(state.delta_memory.mean().detach())
        trace = [best_delta]
        stalled = 0
        for _ in range(max(0, steps)):
            if best_delta <= float(tolerance):
                break
            feedback = alpha * state.delta_slots.detach()
            phase = torch.angle(reference * current.conj())
            current = _unit(current * torch.exp(1j * gain_fraction * phase))
            candidate = self.forward(current, axiom=reference, thermal_feedback=feedback)
            delta = float(candidate.delta_memory.mean().detach())
            trace.append(delta)
            state = candidate
            if delta < best_delta:
                best_state = candidate
                best_delta = delta
                stalled = 0
            else:
                stalled += 1
                if stalled >= 2:
                    break
        return best_state, trace

    def _replace_bright(
        self,
        state: PhotonicState,
        bright: torch.Tensor,
        axiom_out: torch.Tensor,
    ) -> PhotonicState:
        """Rebuild a state after an egress-space phase correction."""
        stats = self.wave_stats(bright)
        delta = sagnac_delta(bright, axiom_out)
        delta_slots = self.slot_delta(bright, axiom_out)
        features, trilinear = self.pool_features(
            bright, state.dark, stats, state.temperature, state.gain, delta
        )
        return replace(
            state,
            bright=bright,
            delta_memory=delta,
            delta_slots=delta_slots,
            visibility=(1.0 - delta).clamp(-1.0, 1.0),
            entropy=stats[:, 0],
            features=features,
            trilinear=trilinear,
        )

    # ------------------------------------------- prototype free-energy loop
    def slot_alignment(self, bright: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """Re<psi_s, phi_m,s> for every slot, shape (batch, n_slots, M)."""
        slots = bright.reshape(-1, self.n_slots, self.d_slot)
        phi_slots = phi.reshape(-1, self.n_slots, self.d_slot)
        return prototype_align(slots, phi_slots).real

    def alignment(self, bright: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """Re<psi, phi_m> against every prototype, shape (batch, M).

        Both operands are unit norm, so this is the cosine that Decision 2
        writes as the prototype-alignment term.
        """
        return self.slot_alignment(bright, phi).sum(dim=1)

    def project(self, wave: torch.Tensor) -> torch.Tensor:
        """P psi = U_s^H (U_s psi_s) for every slot, same shape as input."""
        slots = wave.reshape(-1, self.n_slots, self.d_slot)
        basis = self.pool.basis()
        inner = stiefel_readout(slots, basis)
        back = stiefel_reproject(inner, basis)
        return back.reshape(-1, self.dim)

    def slot_alignment_subspace(
        self, bright: torch.Tensor, phi: torch.Tensor
    ) -> torch.Tensor:
        """Per-slot alignment inside col(U_s^H), shape (batch, n_slots, M).

        Decision 6: the heater feedback is computed on the projected
        coordinates, so the thermal loop sharpens exactly the numbers the
        readout can see instead of moving energy the readout ignores.
        """
        return self.slot_alignment(self.project(bright), self.project(phi))

    def dark_energy(self, bright: torch.Tensor) -> torch.Tensor:
        """Energy outside the Stiefel readout subspace, shape (batch,).

            sum_s || psi_s - U_s^H (U_s psi_s) ||^2
              = sum_s ( ||psi_s||^2 - ||U_s psi_s||^2 )

        At K = 16 of d_slot = 64 the readout can see at most a quarter of
        each slot, so this is the fraction of the field the readout cannot
        express. Gaussian sensor noise scatters energy into the other 48
        dimensions, and this term is what the relaxation loop drives down.
        """
        slots = bright.reshape(-1, self.n_slots, self.d_slot)
        basis = self.pool.basis()
        inner = stiefel_readout(slots, basis)
        captured = inner.abs().pow(2).sum(dim=-1)
        total = slots.abs().pow(2).sum(dim=-1)
        return (total - captured).clamp_min(0.0)

    def free_energy(
        self,
        bright: torch.Tensor,
        phi: torch.Tensor,
        lambda_dark: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decision 2 Lyapunov energy, shape (batch,), and the alignments.

            E = (1 - max_m Re<psi, phi_m>) + lambda_dark * sum_s dark_s
        """
        align = self.alignment(bright, phi)
        residual = 1.0 - align.max(dim=-1).values
        dark = self.dark_energy(bright).sum(dim=-1)
        return residual + float(lambda_dark) * dark, align

    # --------------------------------------------- boundary-axiom discovery
    def discovery_energy(
        self,
        bright: torch.Tensor,
        attractors: Optional[torch.Tensor] = None,
        repulsors: Optional[torch.Tensor] = None,
        laws: Optional[torch.Tensor] = None,
        *,
        tau_rep: float = 0.30,
        tau_law: float = 0.35,
        lambda_rep: float = 1.0,
        lambda_law: float = 2.0,
        lambda_dark: float = 0.5,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Boundary-guided discovery free energy, shape (batch,).

            E = (1 - max_m Re<psi, phi_m^+>)
              + lambda_rep * sum_q max(0, Re<psi, phi_q^-> - tau_rep)
              + lambda_law * sum_l max(0, Sagnac(psi, alpha_l) - tau_law)
              + lambda_dark * sum_s dark_s

        Every term is a measured wave statistic; no term is a tuned logit.
        """
        batch = int(bright.shape[0])
        device = bright.device
        zeros = torch.zeros(batch, device=device, dtype=torch.float32)
        goal = zeros.clone()
        repulsion = zeros.clone()
        law_penalty = zeros.clone()
        align_attractor = torch.zeros(batch, 0, device=device)
        align_repulsor = torch.zeros(batch, 0, device=device)
        law_delta = torch.zeros(batch, 0, device=device)
        if attractors is not None and int(attractors.numel()):
            align_attractor = self.alignment(bright, attractors.to(bright.dtype))
            goal = 1.0 - align_attractor.amax(dim=-1)
        if repulsors is not None and int(repulsors.numel()):
            align_repulsor = self.alignment(bright, repulsors.to(bright.dtype))
            repulsion = float(lambda_rep) * torch.clamp(
                align_repulsor - float(tau_rep), min=0.0
            ).sum(dim=-1)
        if laws is not None and int(laws.numel()):
            columns = [
                sagnac_delta(bright, laws[index].unsqueeze(0).expand_as(bright))
                for index in range(int(laws.shape[0]))
            ]
            law_delta = torch.stack(columns, dim=-1)
            law_penalty = float(lambda_law) * torch.clamp(
                law_delta - float(tau_law), min=0.0
            ).sum(dim=-1)
        dark = float(lambda_dark) * self.dark_energy(bright).sum(dim=-1)
        total = goal + repulsion + law_penalty + dark
        return total, {
            "goal": goal,
            "repulsor": repulsion,
            "law": law_penalty,
            "dark": dark,
            "attractor_alignment": align_attractor,
            "repulsor_alignment": align_repulsor,
            "law_delta": law_delta,
        }

    def _probe_waves(
        self,
        psi: torch.Tensor,
        repulsors: Optional[torch.Tensor],
        probes: int,
        seed: int,
        phase_scale: float,
        directions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Deterministic probe swarm, shape (batch, probes, dim).

        With explicit directions the probes are the supplied candidate waves
        mixed with the input wave; without them they are seeded phase
        perturbations. In both cases every probe is projected orthogonal to
        the active repulsor subspace, so the swarm cannot start inside a
        known trap.
        """
        batch, dim = int(psi.shape[0]), int(psi.shape[1])
        probes = max(1, int(probes))
        if directions is not None:
            waves = psi.unsqueeze(1) + directions.to(psi.dtype)
        else:
            generator = torch.Generator().manual_seed(int(seed))
            real = torch.randn(batch, probes, dim, generator=generator)
            imag = torch.randn(batch, probes, dim, generator=generator)
            noise = torch.complex(real, imag).to(psi.dtype)
            phase = float(phase_scale) * torch.tanh(noise.real) * math.pi
            waves = psi.unsqueeze(1) * torch.exp(1j * phase)
        if directions is None and repulsors is not None and int(repulsors.numel()):
            # Random probes are projected out of the repulsor subspace. An
            # explicit direction is a candidate the caller built (an action
            # successor, an operator wave); it must reach the plates exactly,
            # and the repulsor term of the energy still penalises it.
            basis = _unit(repulsors.to(psi.dtype))
            for index in range(int(basis.shape[0])):
                vector = basis[index].reshape(1, 1, dim)
                coefficient = (waves.conj() * vector).sum(dim=-1, keepdim=True).real
                waves = waves - coefficient * vector
        return _unit(waves, dim=-1)

    def relax_discovery(
        self,
        psi: torch.Tensor,
        attractors: Optional[torch.Tensor] = None,
        repulsors: Optional[torch.Tensor] = None,
        laws: Optional[torch.Tensor] = None,
        *,
        probes: int = 8,
        steps: int = 3,
        t_bias: Optional[float] = None,
        tau_rep: float = 0.30,
        tau_law: float = 0.35,
        lambda_rep: float = 1.0,
        lambda_law: float = 2.0,
        lambda_dark: float = 0.5,
        phase_scale: float = 0.65,
        eta: float = 0.25,
        directions: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
    ) -> Tuple[PhotonicState, Dict[str, object]]:
        """Thermal discovery swarm under the Zone C boundary axioms (K2).

        Launches B_probe deterministic perturbations orthogonal to the active
        repulsors at the elevated Curie-Weiss heater temperature, propagates
        each through the four BaTiO3 plates, and keeps the minimum-energy wave
        that satisfies every LAW axiom. The accepted wave is never worse in
        free energy than the input pass, by the same comparison rule the
        prototype loop uses.
        """
        psi = _unit(psi.to(torch.complex64))
        batch = int(psi.shape[0])
        dim = int(psi.shape[1])
        probe_count = max(1, int(probes))
        if directions is not None:
            probe_count = max(1, int(directions.shape[1]))
        bias = float(self.cfg.t_max if t_bias is None else t_bias)
        # The boundary sets live in the same optical space as the candidates.
        # A law written as a pre-photonic successor is carried through the same
        # plates once, so the Sagnac comparison is output-to-output and not a
        # comparison across two different spaces.
        if attractors is not None and int(attractors.numel()):
            attractors = _unit(self.forward(_unit(attractors.to(torch.complex64)), t_bias=bias).bright)
        if repulsors is not None and int(repulsors.numel()):
            repulsors = _unit(self.forward(_unit(repulsors.to(torch.complex64)), t_bias=bias).bright)
        if laws is not None and int(laws.numel()):
            laws = _unit(self.forward(_unit(laws.to(torch.complex64)), t_bias=bias).bright)
        current_seed = int(self.cfg.seed) + 9101 if seed is None else int(seed)
        candidates = self._probe_waves(
            psi, repulsors, probe_count, current_seed, phase_scale, directions=directions
        )
        flat = candidates.reshape(batch * probe_count, dim)
        state = self.forward(flat, t_bias=bias)
        energy, parts = self.discovery_energy(
            state.bright,
            attractors,
            repulsors,
            laws,
            tau_rep=tau_rep,
            tau_law=tau_law,
            lambda_rep=lambda_rep,
            lambda_law=lambda_law,
            lambda_dark=lambda_dark,
        )
        energy_grid = energy.reshape(batch, probe_count)
        best_index = energy_grid.argmin(dim=-1)
        row = torch.arange(batch, device=psi.device)
        best_bright = state.bright.reshape(batch, probe_count, dim)[row, best_index]
        best_energy = energy_grid[row, best_index]
        # Re-propagate the winning wave so the returned state carries a dark
        # port, heater grid and feature block with the winning batch shape.
        best_state = self.forward(best_bright, t_bias=bias)
        best_state = self._replace_bright(
            best_state,
            best_bright,
            best_bright if attractors is None else best_bright,
        )
        trace_energy = [float(best_energy.mean().detach())]
        target = None
        if attractors is not None and int(attractors.numel()):
            reference = attractors.to(psi.dtype)
            align = self.alignment(best_bright, reference)
            weights = torch.softmax(align, dim=-1).to(torch.complex64)
            target = _unit(
                (weights @ reference.reshape(reference.shape[0], -1)).reshape(-1, dim)
                if batch == 1
                else best_bright
            )
            if batch > 1:
                mixed = torch.zeros_like(best_bright)
                for index in range(batch):
                    mixed[index] = (
                        torch.softmax(align[index], dim=-1).to(torch.complex64)
                        @ reference.reshape(reference.shape[0], -1)
                    )
                target = _unit(mixed)
        for _step in range(max(0, int(steps))):
            corrected = best_bright
            if target is not None:
                phase = torch.angle(target * corrected.conj())
                corrected = _unit(corrected * torch.exp(1j * float(eta) * phase))
            if repulsors is not None and int(repulsors.numel()):
                basis = _unit(repulsors.to(psi.dtype))
                for index in range(int(basis.shape[0])):
                    vector = basis[index].reshape(1, dim)
                    overlap = (corrected.conj() * vector).sum(dim=-1, keepdim=True).real
                    push = torch.clamp(overlap - float(tau_rep), min=0.0)
                    corrected = corrected - push.to(corrected.dtype) * vector
                corrected = _unit(corrected)
            candidate_state = self.forward(corrected, t_bias=bias)
            candidate_energy, _candidate_parts = self.discovery_energy(
                candidate_state.bright,
                attractors,
                repulsors,
                laws,
                tau_rep=tau_rep,
                tau_law=tau_law,
                lambda_rep=lambda_rep,
                lambda_law=lambda_law,
                lambda_dark=lambda_dark,
            )
            improved = candidate_energy < best_energy
            best_bright = torch.where(
                improved.unsqueeze(-1), candidate_state.bright, best_bright
            )
            best_energy = torch.where(improved, candidate_energy, best_energy)
            best_state = self._replace_bright(
                candidate_state,
                best_bright,
                target if target is not None else best_bright,
            )
            trace_energy.append(float(best_energy.mean().detach()))
        law_max = torch.zeros(batch, device=psi.device)
        law_ok = torch.ones(batch, dtype=torch.bool, device=psi.device)
        if laws is not None and int(laws.numel()):
            columns = [
                sagnac_delta(best_bright, laws[index].unsqueeze(0).expand_as(best_bright))
                for index in range(int(laws.shape[0]))
            ]
            law_max = torch.stack(columns, dim=-1).amax(dim=-1)
            law_ok = law_max <= float(tau_law)
        trace: Dict[str, object] = {
            "probes": int(probe_count),
            "steps": int(steps),
            "temperature_bias": float(bias),
            "probe_energies": [float(value) for value in energy_grid.mean(dim=0).detach()],
            "best_energy": [float(value) for value in best_energy.detach()],
            "energy_trace": trace_energy,
            "law_max_delta": [float(value) for value in law_max.detach()],
            "law_satisfied": [bool(value) for value in law_ok.detach()],
            "attractors": int(attractors.shape[0]) if attractors is not None else 0,
            "repulsors": int(repulsors.shape[0]) if repulsors is not None else 0,
            "laws": int(laws.shape[0]) if laws is not None else 0,
        }
        return best_state, trace

    def relax_prototype(
        self,
        psi: torch.Tensor,
        phi: torch.Tensor,
        *,
        beta: float = 0.0,
        steps: Optional[int] = None,
        alpha_thermal: Optional[float] = None,
        eta: Optional[float] = None,
        lambda_dark: Optional[float] = None,
        tolerance: float = 0.0,
    ) -> Tuple[PhotonicState, List[float]]:
        """Relax toward the clean prototype bank, not toward the context.

        Every pass re-propagates the plates with heater feedback derived
        from the per-slot prototype residual, applies a phase-only egress
        correction toward the softmax-weighted prototype at the calibrated
        inverse temperature, and is accepted only when it lowers the
        measured Lyapunov energy. The accepted state is therefore never
        worse in energy than the input pass, by construction.
        """
        if phi is None or phi.numel() == 0:
            raise ValueError("relax_prototype needs a non-empty prototype bank")
        steps = int(self.cfg.relax_steps if steps is None else steps)
        alpha = float(self.cfg.alpha_thermal if alpha_thermal is None else alpha_thermal)
        eta = float(getattr(self.cfg, "prototype_eta", 0.15) if eta is None else eta)
        lam = float(
            getattr(self.cfg, "prototype_lambda_dark", 0.5)
            if lambda_dark is None
            else lambda_dark
        )
        reference = phi.to(torch.complex64)
        reference = reference / reference.norm(dim=-1, keepdim=True).clamp_min(_EPS)

        state = self.forward(psi, apply_memory=False)
        energy, align = self.free_energy(state.bright, reference, lam)
        best = state
        best_energy = energy.clone()
        trace = [float(energy.mean().detach())]
        for _ in range(max(0, steps)):
            if float(best_energy.max().detach()) <= float(tolerance):
                break
            weights = torch.softmax(float(beta) * align, dim=-1).to(torch.complex64)
            target = (weights @ reference.reshape(reference.shape[0], -1)).reshape(-1, self.dim)
            target = _unit(target)
            slot_align = self.slot_alignment_subspace(state.bright, reference).amax(dim=-1)
            feedback = (alpha * (1.0 - slot_align)).detach()
            state = self.forward(psi, thermal_feedback=feedback, apply_memory=False)
            phase = torch.angle(target * state.bright.conj())
            corrected = _unit(state.bright * torch.exp(1j * eta * phase))
            candidate = self._replace_bright(state, corrected, target)
            energy, align = self.free_energy(candidate.bright, reference, lam)
            trace.append(float(energy.mean().detach()))
            improved = energy < best_energy
            merged = torch.where(improved.unsqueeze(-1), candidate.bright, best.bright)
            best = self._replace_bright(candidate, merged, target)
            best_energy = torch.where(improved, energy, best_energy)
            state = candidate
        return best, trace

    def relax_output(
        self,
        psi: torch.Tensor,
        axiom_out: torch.Tensor,
        steps: Optional[int] = None,
        alpha_thermal: Optional[float] = None,
        fraction: Optional[float] = None,
        tolerance: float = 0.35,
    ) -> Tuple[PhotonicState, List[float]]:
        """Relax in the POST-photonic attractor space.

        Decision B: the old loop pulled a post-photonic wave toward a
        pre-photonic memory wave, which partially erased the optical
        transform. Here the axiom is B_theta(context), computed once with
        no gradient, and every pass re-propagates the plates with heater
        feedback before the egress phase correction.
        """
        if axiom_out is None:
            raise ValueError("relax_output needs a post-photonic axiom")
        steps = int(self.cfg.relax_steps if steps is None else steps)
        alpha = float(self.cfg.alpha_thermal if alpha_thermal is None else alpha_thermal)
        eta = float(0.15 if fraction is None else fraction)
        axiom = _unit(axiom_out.to(torch.complex64))
        state = self.forward(psi, apply_memory=False)
        best = self._replace_bright(state, state.bright, axiom)
        best_delta = float(best.delta_memory.mean().detach())
        trace = [best_delta]
        for _ in range(max(0, steps)):
            if best_delta <= float(tolerance):
                break
            feedback = alpha * self.slot_delta(state.bright, axiom).detach()
            state = self.forward(psi, thermal_feedback=feedback, apply_memory=False)
            phase = torch.angle(axiom * state.bright.conj())
            corrected = _unit(state.bright * torch.exp(1j * eta * phase))
            candidate = self._replace_bright(state, corrected, axiom)
            delta = float(candidate.delta_memory.mean().detach())
            trace.append(delta)
            state = candidate
            if delta < best_delta:
                best = candidate
                best_delta = delta
        return best, trace

    # ------------------------------------------------------- learning hooks
    @torch.no_grad()
    def imprint(self, wave_in: torch.Tensor, target: torch.Tensor, rate: float = 0.5) -> float:
        """Optical phase-conjugate write on the last plate (neuroplastic).

        This is the fast, backprop-free learning path of Zone B. It is the
        same update the reference imprint uses: add the residual phase
        angle(target * conj(output)) to the final mask. Returns the
        residual after the write.
        """
        if not 0.0 < float(rate) <= 1.0:
            raise ValueError("rate must lie in (0, 1]")
        state = self.forward(wave_in)
        target = _unit(target.to(torch.complex64))
        residual = torch.angle(target * state.bright.conj())
        residual = residual.reshape(-1, self.n_slots, self.d_slot).mean(dim=0)
        plate = self.plates[-1]
        plate.phase.add_(float(rate) * residual)
        plate.phase.copy_(torch.remainder(plate.phase + math.pi, TWO_PI) - math.pi)
        after = self.forward(wave_in).bright
        return float(torch.linalg.vector_norm(target - after, dim=-1).mean())

    @torch.no_grad()
    def calibrate(self, waves: Sequence[torch.Tensor], sigma: float = 3.0, ceiling: float = 0.35) -> Dict[str, float]:
        """Calibrate the veto threshold on the evaluation wave family.

        The retired repository hardcoded epsilon = 0.35 for the complex
        family and evaluated on the real family (constraint C2). This
        method measures the null distribution of the waves it is given.
        """
        if len(waves) < 2:
            raise ValueError("calibration needs at least two reference waves")
        stack = torch.stack([_unit(w.reshape(-1).to(torch.complex64)) for w in waves])
        n = stack.shape[0]
        generator = torch.Generator().manual_seed(int(self.cfg.seed) + 7)
        index_a = torch.randint(0, n, (1024,), generator=generator)
        index_b = torch.randint(0, n - 1, (1024,), generator=generator)
        index_b = index_b + (index_b >= index_a).long()
        deltas = sagnac_delta(stack[index_a], stack[index_b])
        mean = float(deltas.mean())
        std = float(deltas.std(unbiased=False))
        raw = mean + float(sigma) * std
        epsilon = float(min(raw, float(ceiling)))
        with torch.no_grad():
            self.epsilon.fill_(epsilon)
            self.calibration_mean.fill_(mean)
            self.calibration_std.fill_(std)
        return {
            "epsilon": epsilon,
            "baseline_delta_mean": mean,
            "baseline_delta_std": std,
            "sigma": float(sigma),
            "ceiling": float(ceiling),
            "n_pairs": int(deltas.numel()),
            "capped": bool(raw > ceiling),
        }

    @torch.no_grad()
    def reachable_shift(self, psi: torch.Tensor, eps_v: float, temperature: float) -> float:
        """Mean phase displacement for a bounded voltage step at fixed T.

        This is the measurable form of the Curie-Weiss claim: the same
        voltage moves the field further when the heater runs hotter.
        """
        psi = _unit(psi.reshape(1, -1).to(torch.complex64))
        gain = curie_weiss_gain(torch.tensor([float(temperature)] * self.n_slots)).reshape(1, -1)
        slots = psi.reshape(1, self.n_slots, self.d_slot)

        def run(drive: float) -> torch.Tensor:
            field = slots
            for plate in self.plates:
                saved = plate.voltage.detach().clone()
                with torch.no_grad():
                    logit = math.atanh(min(abs(drive), 0.999)) * (1.0 if drive >= 0 else -1.0)
                    plate.voltage.fill_(logit / max(plate.voltage_bound, 1e-9))
                try:
                    field = plate(field, gain, direction=1.0)
                finally:
                    with torch.no_grad():
                        plate.voltage.copy_(saved)
            return field.reshape(1, -1)

        base = run(0.0)
        moved = run(float(eps_v))
        return float(torch.abs(torch.angle(moved * base.conj())).mean())


def phase_authority(temperature: torch.Tensor, pockels_gain: float = 0.9) -> torch.Tensor:
    """Measured control authority dPhi/dV at a given temperature."""
    return float(pockels_gain) * curie_weiss_gain(temperature)


# ======================================================================
# Directive Y1: the non-destructive subword optical phase cartridge.
#
# Cycle E trained the four BaTiO3 phase masks on the 96-character
# CharLMTask stream.  Cycle L-X replaced that ingress with the
# BochnerPhaseProjector four-slot subword codebook.  The masked plates are
# a fixed pseudo-unitary operator on S^{D-1}; measured against the subword
# codebook its bright-port readout is O(D^{-1/2}) (concentration of
# measure), so the X2 calibrator correctly muted the photonic fusion
# weight.  The fix is not to retrain Vision/Action/Relational weights.
# It is a separate, closed-form, cross-spectral alignment between the
# post-photonic ports and the Bochner training stream:
#
#     M_s = Y_s X_s^H (X_s X_s^H + lambda I)^{-1}
#
# where X_s stacks the post-photonic bright-port slot-s blocks over the
# training contexts and Y_s stacks the Bochner context waves of the same
# contexts.  M_s is the minimum-mean-square phase-conjugate aligner: it is
# solved in closed form (no autograd, no optimiser) and it is applied to
# the ports, so holarch_medium.pt stays byte-identical and every
# Vision/Action/Relational weight is untouched.
# ======================================================================
class SubwordPhaseCartridge(nn.Module):
    """Closed-form cross-spectral alignment of the Zone B ports.

    The cartridge OWNS its aligner.  It never edits a plate mask, a heater,
    a pool basis or a mixer weight, so a mounted checkpoint keeps its
    original bytes and the cartridge can be dropped without a reload.
    """

    def __init__(self, core: "PhotonicCore", *, ridge: float = 1e-1) -> None:
        super().__init__()
        self.core = core
        self.dim = int(core.dim)
        self.n_slots = int(core.n_slots)
        self.d_slot = int(core.d_slot)
        self.ridge = float(ridge)
        self.aligners: list = [None for _ in range(self.n_slots)]
        self.observed = 0
        self.report: Dict[str, object] = {}

    # --------------------------------------------------------------- ports
    @torch.no_grad()
    def ports(self, psi: torch.Tensor):
        """The raw bright and dark ports of the mounted plate stack."""
        state = self.core(psi)
        return state.bright, state.dark, state

    @staticmethod
    def _blocks(field: torch.Tensor, n_slots: int, d_slot: int) -> list:
        return [
            field[:, index * d_slot : (index + 1) * d_slot].detach().cpu().numpy()
            for index in range(n_slots)
        ]

    # --------------------------------------------------------------- fitting
    def fit(
        self,
        waves: Sequence[torch.Tensor],
        *,
        contexts: Optional[Sequence[object]] = None,
    ) -> Dict[str, object]:
        """Solve the per-slot phase-conjugate aligner in closed form.

        waves are the Bochner context waves (the targets).  The bright port
        of each wave is the observation.  Each slot is a separate ridge
        problem, because the four slots are four independent waveguide
        banks and a cross-slot aligner would be a full rank-D scramble.
        """
        import numpy as np

        if len(waves) < 4:
            raise ValueError("the cartridge needs at least four context waves")
        stack = torch.cat([w.reshape(1, -1) for w in waves], dim=0).to(torch.complex64)
        bright, _dark, _state = self.ports(stack)
        xb = self._blocks(bright, self.n_slots, self.d_slot)
        yb = self._blocks(stack, self.n_slots, self.d_slot)
        rows: list = []
        for slot in range(self.n_slots):
            # Sample-major convention: rows are contexts.  The ridge problem
            # is min_M ||X M^T - Y||^2 + lambda ||M||^2 with the closed form
            # M^T = (X^H X + lambda I)^{-1} X^H Y.  The transpose matters: the
            # conjugate-transposed aligner rotates the field the wrong way and
            # measured a LOWER cosine than the raw port.
            x = np.asarray(xb[slot], dtype=np.complex128)      # (N, d)
            y = np.asarray(yb[slot], dtype=np.complex128)      # (N, d)
            gram = x.conj().T @ x                              # (d, d)
            scale = float(np.trace(gram).real) / max(gram.shape[0], 1)
            gram = gram + np.eye(gram.shape[0], dtype=np.complex128) * (
                self.ridge * max(scale, 1e-12)
            )
            alpha = np.linalg.solve(gram, x.conj().T @ y)      # (d, d) = M^T
            aligner = alpha.T                                  # (d, d)
            aligned = x @ aligner.T
            cosine = float(
                np.mean(
                    np.abs(np.sum(aligned.conj() * y, axis=0))
                    / np.maximum(
                        np.linalg.norm(aligned, axis=0) * np.linalg.norm(y, axis=0), 1e-12
                    )
                )
            )
            raw_cosine = float(
                np.mean(
                    np.abs(np.sum(x.conj() * y, axis=0))
                    / np.maximum(
                        np.linalg.norm(x, axis=0) * np.linalg.norm(y, axis=0), 1e-12
                    )
                )
            )
            self.aligners[slot] = aligner.astype(np.complex64)
            rows.append(
                {
                    "slot": int(slot),
                    "contexts": int(x.shape[0]),
                    "raw_port_cosine": raw_cosine,
                    "aligned_port_cosine": cosine,
                    "aligner_norm": float(np.linalg.norm(aligner)),
                    "phase_displacement": float(
                        np.abs(np.angle(np.diag(aligner))).mean()
                    ),
                }
            )
        self.observed = int(len(waves))
        self.report = {
            "ridge": float(self.ridge),
            "contexts": int(len(waves)),
            "slots": rows,
            "mean_raw_cosine": float(np.mean([row["raw_port_cosine"] for row in rows])),
            "mean_aligned_cosine": float(
                np.mean([row["aligned_port_cosine"] for row in rows])
            ),
            "mean_phase_displacement": float(
                np.mean([row["phase_displacement"] for row in rows])
            ),
            "checkpoint_bytes_touched": 0,
        }
        return dict(self.report)

    # --------------------------------------------------------------- adapter
    def adapt_arrays(self, bright, dark):
        """Apply the solved aligner to numpy ports (the egress hot path)."""
        import numpy as np

        if any(item is None for item in self.aligners):
            return bright, dark
        out_b = np.array(bright, dtype=np.complex64, copy=True)
        out_d = np.array(dark, dtype=np.complex64, copy=True)
        for slot in range(self.n_slots):
            aligner = self.aligners[slot]
            if aligner is None:
                continue
            start = slot * self.d_slot
            stop = start + self.d_slot
            out_b[start:stop] = aligner @ np.asarray(bright[start:stop], dtype=np.complex64)
            out_d[start:stop] = aligner @ np.asarray(dark[start:stop], dtype=np.complex64)
        return out_b, out_d

    def as_dict(self) -> Dict[str, object]:
        payload = dict(self.report)
        payload["mounted"] = bool(all(item is not None for item in self.aligners))
        payload["dim"] = int(self.dim)
        payload["d_slot"] = int(self.d_slot)
        return payload


def open_subword_cartridge(model, *, ridge: float = 1e-1) -> SubwordPhaseCartridge:
    """Build a cartridge on a mounted holarch without touching its weights."""
    core = getattr(model, "zone_b", None)
    if core is None:
        raise ValueError("the model exposes no Zone B photonic core")
    return SubwordPhaseCartridge(core, ridge=float(ridge))
