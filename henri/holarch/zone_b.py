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
        return torch.einsum("ij,bjd->bid", self.matrix(), slots)


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

    def basis(self) -> torch.Tensor:
        if self.mode == "fixed":
            return self.fixed_basis
        weight = torch.complex(self.weight_real, self.weight_imag)
        q, r = torch.linalg.qr(weight.transpose(-1, -2))
        diagonal = torch.diagonal(r, dim1=-2, dim2=-1).real
        sign = torch.where(diagonal >= 0.0, torch.ones_like(diagonal), -torch.ones_like(diagonal))
        q = q * torch.complex(sign, torch.zeros_like(sign)).unsqueeze(-2)
        return q.transpose(-1, -2).contiguous()

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        basis = self.basis()
        return torch.einsum("bsd,skd->bsk", slots, basis.conj()).real.flatten(1)


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
        return torch.einsum("bsd,msd->bsm", slots.conj(), phi_slots).real

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
        inner = torch.einsum("bsd,skd->bsk", slots, basis.conj())
        back = torch.einsum("bsk,skd->bsd", inner, basis)
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
        inner = torch.einsum("bsd,skd->bsk", slots, basis.conj())
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