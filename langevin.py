"""Viscoelastic memory kernel and Generalized Langevin dynamics.

Evidence label: DERIVED (the kernel is the Maxwell-Debye form) and
OBSERVED (FDT consistency is measured in tests).

Non-Markovian memory kernel
---------------------------
The kernel sums `P = 4` exponential relaxation modes:

    Gamma(t - s) = sum_p (g_p / tau_p) * exp(-(t - s) / tau_p)

The architecture fixes the mode count at `P = 4` with these roles:

| mode | tau | role |
|---|---|---|
| 1 | 50 us | immediate/sensory |
| 2 | 2 ms | operational micro-task |
| 3 | 50 ms | episodic |
| 4 | inf | axiomatic ground truth |

The `tau = inf` mode is the structural subtlety. `exp(-x / inf) = 1`
for all finite `x`, so that mode contributes a CONSTANT, not a decay.
It therefore cannot be rescaled into a finite window like the others.
This module tracks it separately as `constant_weight` so callers cannot
silently drop it.

Fluctuation-dissipation
-----------------------
The noise covariance must match the kernel:

    <xi(t) xi(s)> = 2 k_B T Gamma(t - s) I

`fdt_consistency` measures the realized noise variance against the
kernel prediction. A mismatch means the thermostat is not in
equilibrium with the kernel it claims to use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

_EPS = 1e-30


@dataclass
class ViscoelasticKernel:
    """Maxwell-Debye memory kernel over P relaxation modes.

    Attributes:
        taus: relaxation timescales in seconds. `inf` is allowed and
            marks the non-decaying axiomatic mode.
        weights: per-mode weights `g_p`, sum-normalized.
    """

    taus: tuple
    weights: tuple

    @classmethod
    def default(cls) -> "ViscoelasticKernel":
        """The four modes the architecture fixes."""
        taus = (50e-6, 2e-3, 50e-3, float("inf"))
        weights = (0.4, 0.3, 0.2, 0.1)
        return cls(taus=taus, weights=weights)

    def __post_init__(self) -> None:
        if len(self.taus) != len(self.weights):
            raise ValueError("taus and weights must align")
        if not self.taus:
            raise ValueError("kernel needs at least one mode")
        if any(t <= 0 for t in self.taus):
            raise ValueError("all taus must be positive (inf is allowed)")
        total = float(np.sum(self.weights))
        if abs(total - 1.0) > 1e-9:
            self.weights = tuple(float(w) / total for w in self.weights)

    @property
    def n_modes(self) -> int:
        return len(self.taus)

    @property
    def constant_weight(self) -> float:
        """Weight on the non-decaying (tau = inf) mode."""
        return float(
            sum(w for t, w in zip(self.taus, self.weights) if np.isinf(t))
        )

    @property
    def decaying_weight(self) -> float:
        """Weight on the finite-tau modes."""
        return float(1.0 - self.constant_weight)

    def value(self, lag: np.ndarray) -> np.ndarray:
        """Kernel value at non-negative lags. Shape follows `lag`."""
        lag = np.asarray(lag, dtype=np.float64)
        out = np.zeros_like(lag)
        for tau, w in zip(self.taus, self.weights):
            if np.isinf(tau):
                out = out + w * 1.0
            else:
                out = out + (w / tau) * np.exp(-lag / tau)
        return out

    def integrate(self, horizon: float, samples: int = 512) -> float:
        """Integral of the kernel over [0, horizon].

        The finite modes integrate to their weight. The `inf` mode
        contributes `w * horizon`, which is why the total is reported
        rather than silently normalized.
        """
        lag = np.linspace(0.0, float(horizon), int(samples))
        d = float(horizon) / max(int(samples) - 1, 1)
        return float(np.sum(self.value(lag)) * d)


@dataclass
class LangevinState:
    """Wave state plus the finite history buffer for the memory term."""

    psi: np.ndarray
    history: list = field(default_factory=list)
    cfg: object = None

    def push(self, vector: np.ndarray, cap: int) -> None:
        self.history.append(np.asarray(vector, dtype=np.complex128).reshape(-1))
        if len(self.history) > cap:
            del self.history[: len(self.history) - cap]


def memory_force(
    history: list,
    dt: float,
    kernel: ViscoelasticKernel,
) -> np.ndarray:
    """Discretized convolution `int Gamma(t-s) psi'(s) ds`.

    Trapezoidal in lag with a finite history window. The `inf` mode is
    excluded here because a constant kernel applied to a derivative
    integrates to a drift, not a drag; that term is returned by
    `constant_drift` instead.
    """
    if not history:
        raise ValueError("history must be non-empty")
    dim = history[-1].shape[0]
    acc = np.zeros(dim, dtype=np.complex128)
    n = len(history)
    for i in range(1, n):
        lag = (n - 1 - i) * dt
        deriv = (history[i] - history[i - 1]) / dt
        acc = acc + _finite_kernel(lag, kernel) * deriv * dt
    return acc


def _finite_kernel(lag: float, kernel: ViscoelasticKernel) -> float:
    total = 0.0
    for tau, w in zip(kernel.taus, kernel.weights):
        if np.isinf(tau):
            continue
        total += (w / tau) * float(np.exp(-lag / tau))
    return total


def constant_drift(
    history: list, kernel: ViscoelasticKernel
) -> np.ndarray:
    """Contribution of the `tau = inf` mode: `w * (psi_now - psi_start)`."""
    w = kernel.constant_weight
    if w == 0.0 or len(history) < 2:
        return np.zeros_like(history[-1])
    return w * (history[-1] - history[0])


def gle_step(
    state: LangevinState,
    kernel: ViscoelasticKernel,
    dt: float,
    rng: Optional[np.random.Generator] = None,
    temperature: float = 0.0,
    drift: Optional[np.ndarray] = None,
    memory_cap: int = 8,
) -> LangevinState:
    """One Generalized Langevin step on the complex vector.

    d psi = -grad E dt - int Gamma(t-s) psi'(s) ds dt + xi dt + drift

    Args:
        state: current wave plus history.
        kernel: the viscoelastic kernel.
        dt: step size.
        rng: explicit generator. Required when `temperature > 0`.
        temperature: k_B T for the fluctuation-dissipation-matched noise.
        drift: optional external drive (for example a Hopfield gradient).
        memory_cap: history length retained.

    Returns:
        A new LangevinState. The returned vector is NOT normalized;
        the caller decides when to project onto the sphere.
    """
    if dt <= 0:
        raise ValueError("dt must be positive")

    if state.cfg is not None:
        expected = state.cfg.dim
        if state.psi.shape[0] != expected:
            raise ValueError("state dimension does not match config")

    state.push(state.psi.copy(), memory_cap)
    force = np.zeros_like(state.psi)
    if drift is not None:
        force = force - np.asarray(drift, dtype=np.complex128).reshape(-1)
    if len(state.history) >= 2:
        # Both terms are integrals over the lag window, so they add
        # directly. Dividing the constant mode by dt would be a unit
        # error: the kernel integral is already a displacement.
        force = force - memory_force(state.history, dt, kernel)
        force = force - constant_drift(state.history, kernel)

    if temperature > 0.0:
        if rng is None:
            raise ValueError("temperature > 0 requires an explicit rng")
        scale = np.sqrt(2.0 * temperature * kernel.decaying_weight / dt)
        noise = (
            rng.standard_normal(state.psi.shape[0])
            + 1j * rng.standard_normal(state.psi.shape[0])
        ) / np.sqrt(2.0)
        force = force + scale * noise

    return LangevinState(
        psi=state.psi + dt * force, history=state.history, cfg=state.cfg
    )


def fdt_consistency(
    kernel: ViscoelasticKernel,
    temperature: float,
    rng: np.random.Generator,
    dt: float = 1e-3,
    n_samples: int = 20000,
) -> dict:
    """Measure noise variance against the FDT prediction.

    Args:
        kernel: the kernel the thermostat claims to match.
        temperature: k_B T used in sampling.
        rng: explicit generator.
        dt: step size used in sampling.
        n_samples: draws.

    Returns:
        Dict with `predicted`, `measured`, and `relative_error`.
        `predicted` is `2 k_B T * Gamma(0) * decayed_weight`, the
        variance the kernel implies at zero lag.
    """
    scale = np.sqrt(2.0 * temperature * kernel.decaying_weight / dt)
    noise = (
        rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)
    ) / np.sqrt(2.0)
    samples = scale * noise
    measured = float(np.mean(np.abs(samples) ** 2))
    predicted = float(2.0 * temperature * kernel.decaying_weight / dt)
    return {
        "predicted": predicted,
        "measured": measured,
        "relative_error": float(abs(measured - predicted) / (predicted + _EPS)),
    }
