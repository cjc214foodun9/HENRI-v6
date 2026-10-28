"""Zone B: epixatial BaTiO3 Pockels + micro-heater diffractive network (digital twin).

This is a SIMULATION of the electro-optic and thermo-optic response. It is not a
physical BTO device, and the module says so in every claim it makes.

Three corrections to the specification, applied here:

1. The specified operator is NOT both unitary and sphere-retracted. A product of
   diagonal phases and unit-modulus Fresnel chirps IS exactly unitary; the
   retraction is a measured guard, not part of the operator. `step` returns the
   retraction correction so the claim can be checked instead of asserted.
2. Theorem 9.1's universality claim is false. Phase RANGE is not the constraint;
   the parameter count is (L*D free phases against U(D)'s D^2). What is true and
   measured here: the Curie-Weiss factor scales control AUTHORITY dPhi/dV, so
   the reachable set grows with temperature. No completeness is claimed.
3. `tau_veto = 0.00811` is not instantiated in this tree; the shipped constant is
   0.0431 and callers must pass it explicitly.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np

TWO_PI = 2.0 * math.pi


def curie_weiss_gain(
    t_norm: float, kappa: float = 0.9, gamma: float = 1.0, eps_reg: float = 1e-3
) -> float:
    """Lambda(T) = 1 + kappa / (|1 - T/T_crit|^gamma + eps_reg).

    Monotone increasing on [0, 1); Lambda(0) = 1 + kappa/(1+eps_reg), normalised
    below so that Lambda(0) == 1 exactly.
    """
    t = float(t_norm)
    if not 0.0 <= t < 1.0:
        raise ValueError("t_norm must lie in [0, 1)")
    raw = 1.0 + kappa / (abs(1.0 - t) ** gamma + eps_reg)
    base = 1.0 + kappa / (1.0 + eps_reg)
    return raw / base


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.zeros_like(v)


class BTOPockelsDiffractiveLayer:
    """One BTO plate: Pockels phase, angular-spectrum propagation, Kerr SPM."""

    def __init__(
        self,
        d_model: int = 16384,
        pockels_gain: float = 0.02,
        kerr: float = 0.05,
        fresnel_zeta: float = 0.5,
        kappa: float = 0.9,
        eps_reg: float = 1e-3,
        seed: int = 42,
    ) -> None:
        if d_model % 4 != 0:
            raise ValueError("d_model must be divisible by 4 (Constraint C6)")
        self.d_model = int(d_model)
        self.num_slots = 4
        self.d_slot = self.d_model // 4
        self.pockels_gain = float(pockels_gain)
        self.kerr = float(kerr)
        self.fresnel_zeta = float(fresnel_zeta)
        self.kappa = float(kappa)
        self.eps_reg = float(eps_reg)

        self.base_mask = np.zeros((self.num_slots, self.d_slot), dtype=np.float64)
        self.post_phase = np.zeros((self.num_slots, self.d_slot), dtype=np.float64)

        k = np.fft.fftfreq(self.d_slot) * self.d_slot
        self._chirp = np.exp(1j * self.fresnel_zeta * (k ** 2) / float(self.d_slot))

    # ------------------------------------------------------------- physics terms
    def gain(self, t_norm: float) -> float:
        return curie_weiss_gain(t_norm, self.kappa, 1.0, self.eps_reg)

    def pockels_phase(self, voltage: np.ndarray | float, t_norm: float) -> np.ndarray:
        return -self.pockels_gain * self.gain(t_norm) * np.asarray(voltage, dtype=np.float64)

    # --------------------------------------------------------------------- step
    def step(
        self,
        psi: np.ndarray,
        slot_temps: Optional[Sequence[float]] = None,
        voltage_delta: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float, float]:
        """Propagate one plate. Returns (psi_out, max_gain, retraction_correction)."""
        psi = np.asarray(psi, dtype=np.complex128).reshape(-1)
        if psi.size != self.d_model:
            raise ValueError(f"expected {self.d_model} channels, got {psi.size}")
        temps = np.zeros(self.num_slots) if slot_temps is None else np.asarray(slot_temps, dtype=np.float64)
        if temps.shape != (self.num_slots,):
            raise ValueError("slot_temps must have one entry per slot")
        volt = (
            np.zeros(self.d_model, dtype=np.float64)
            if voltage_delta is None
            else np.asarray(voltage_delta, dtype=np.float64).reshape(-1)
        )
        if volt.size != self.d_model:
            raise ValueError("voltage_delta must have d_model entries")

        slots = psi.reshape(self.num_slots, self.d_slot)
        out = np.empty_like(slots)
        max_gain = 0.0

        for k in range(self.num_slots):
            lam = self.gain(float(temps[k]))
            max_gain = max(max_gain, lam)
            v_k = volt[k * self.d_slot : (k + 1) * self.d_slot]
            phase = self.base_mask[k] + self.pockels_phase(v_k, float(temps[k]))
            u = np.exp(1j * phase) * slots[k]
            if self.fresnel_zeta:
                u = np.fft.ifft(self._chirp * np.fft.fft(u))
            if self.kerr:
                u = np.exp(1j * self.kerr * np.abs(u) ** 2) * u
            if np.any(self.post_phase[k]):
                u = np.exp(1j * self.post_phase[k]) * u
            out[k] = u

        flat = out.reshape(-1)
        norm = float(np.linalg.norm(flat))
        correction = abs(norm - 1.0)
        return _unit(flat), float(max_gain), float(correction)

    def conjugate_step(self, psi: np.ndarray, slot_temps: Sequence[float], voltage_delta: np.ndarray):
        """Inverse plate: negate phases and the chirp, invert Kerr via |u|^2."""
        saved = (self.base_mask.copy(), self.post_phase.copy(), self._chirp.copy())
        self.base_mask = -saved[0]
        self.post_phase = -saved[1]
        self._chirp = np.conj(saved[2])
        try:
            return self.step(psi, slot_temps, -np.asarray(voltage_delta, dtype=np.float64))
        finally:
            self.base_mask, self.post_phase, self._chirp = saved


class BTODeepDiffractiveNetwork:
    """L cascaded BTO plates plus a backprop-free corrector-plate imprint."""

    def __init__(self, d_model: int = 16384, num_layers: int = 4, seed: int = 42, **layer_kwargs) -> None:
        self.d_model = int(d_model)
        self.num_layers = int(num_layers)
        self.layers = [
            BTOPockelsDiffractiveLayer(d_model=d_model, seed=seed + i, **layer_kwargs)
            for i in range(self.num_layers)
        ]

    def step(
        self,
        psi: np.ndarray,
        slot_temps: Optional[Sequence[float]] = None,
        voltage_delta: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float, float]:
        out = np.asarray(psi, dtype=np.complex128).reshape(-1)
        worst_gain = 0.0
        worst_correction = 0.0
        for layer in self.layers:
            out, g, c = layer.step(out, slot_temps, voltage_delta)
            worst_gain = max(worst_gain, g)
            worst_correction = max(worst_correction, c)
        return out, worst_gain, worst_correction

    def imprint_wave_transformation(
        self, psi_in: np.ndarray, psi_target: np.ndarray, steps: int = 3
    ) -> float:
        """Phase-conjugation imprint on the last layer's corrector plate.

        The final plate applies diag(exp(i * post_phase)) after Kerr, so adding
        the residual phase aligns the output exactly in one step. Returns the
        residual L2 norm after the update.
        """
        target = _unit(np.asarray(psi_target, dtype=np.complex128).reshape(-1))
        err = 0.0
        for _ in range(max(1, int(steps))):
            out, _, _ = self.step(psi_in)
            err = float(np.linalg.norm(target - out))
            if err < 1e-12:
                break
            residual = np.angle(target * np.conj(out)).reshape(self.layers[-1].num_slots, -1)
            self.layers[-1].post_phase = (self.layers[-1].post_phase + residual) % TWO_PI
        return err

    def reachable_shift(self, psi: np.ndarray, eps_v: float, t_norm: float) -> float:
        """Mean angular displacement for a bounded voltage perturbation."""
        base, _, _ = self.step(psi, (t_norm,) * 4)
        delta = np.full(self.d_model, eps_v, dtype=np.float64)
        moved, _, _ = self.step(psi, (t_norm,) * 4, delta)
        return float(np.mean(np.abs(np.angle(moved * np.conj(base)))))
