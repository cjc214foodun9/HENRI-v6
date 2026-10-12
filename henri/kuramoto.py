"""Kuramoto phase locking and the order parameter.

Evidence label: DERIVED (the equation is the standard Kuramoto model)
and OBSERVED (the locking threshold 0.880 is stated as a target, and
the tests report the value actually reached).

Order parameter:

    r e^{i psi} = (1 / D) * sum_d exp(i * theta_d)

`r = 1` means full phase coherence, `r ~ 0` means uniform phases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from henri.config import WaveConfig
from henri.state import WaveState

LOCK_THRESHOLD = 0.880
"""Order parameter at or above which the field counts as locked."""


@dataclass
class KuramotoState:
    """Phase field and its natural frequencies.

    Attributes:
        phases: phase angles, shape (D,).
        omega: natural frequencies, shape (D,).
    """

    phases: np.ndarray
    omega: np.ndarray

    @classmethod
    def from_wave(
        cls,
        state: WaveState,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
        spread: float = 1.0,
    ) -> "KuramotoState":
        """Take phases from a wave and draw natural frequencies."""
        gen = rng if rng is not None else np.random.default_rng(seed)
        phases = np.angle(state.psi).copy()
        omega = gen.standard_normal(phases.shape[0]) * spread
        return cls(phases=phases, omega=omega)

    @property
    def dim(self) -> int:
        return int(self.phases.shape[0])


def order_parameter(phases: np.ndarray) -> tuple:
    """Return (r, psi). `r` is coherence in [0, 1]."""
    arr = np.asarray(phases, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("phases must be non-empty")
    z = np.mean(np.exp(1j * arr))
    return float(np.abs(z)), float(np.angle(z))


def kuramoto_step(
    state: KuramotoState,
    coupling: float,
    dt: float,
    rng: Optional[np.random.Generator] = None,
    noise: float = 0.0,
) -> KuramotoState:
    """Advance the phase field by one explicit Euler step.

    d theta_d / dt = omega_d + (K/D) * sum_j sin(theta_j - theta_d) + noise

    The mean-field form is used, so the coupling term costs O(D) rather
    than O(D^2):

        sum_j sin(theta_j - theta_d) = D * r * sin(psi - theta_d)

    Args:
        state: current phase field.
        coupling: K, the coupling strength. Must be non-negative.
        dt: step size.
        rng: explicit generator. Required when `noise > 0`.
        noise: thermal phase noise scale.

    Returns:
        A new KuramotoState with wrapped phases.
    """
    if coupling < 0:
        raise ValueError("coupling must be non-negative")
    if dt <= 0:
        raise ValueError("dt must be positive")

    r, psi = order_parameter(state.phases)
    drift = state.omega + coupling * r * np.sin(psi - state.phases)

    if noise > 0.0:
        if rng is None:
            raise ValueError("noise > 0 requires an explicit rng (determinism)")
        drift = drift + rng.standard_normal(state.phases.shape[0]) * noise / np.sqrt(dt)

    phases = state.phases + dt * drift
    phases = np.angle(np.exp(1j * phases))  # wrap to (-pi, pi]
    return KuramotoState(phases=phases, omega=state.omega)


def lock(
    state: KuramotoState,
    coupling: float,
    dt: float,
    steps: int,
    threshold: float = LOCK_THRESHOLD,
) -> tuple:
    """Iterate to synchrony. Returns (state, r trace)."""
    trace = np.empty(int(steps) + 1, dtype=np.float64)
    trace[0] = order_parameter(state.phases)[0]
    current = state
    for i in range(int(steps)):
        current = kuramoto_step(current, coupling, dt)
        trace[i + 1] = order_parameter(current.phases)[0]
    return current, trace


def is_locked(r: float, threshold: float = LOCK_THRESHOLD) -> bool:
    """True when coherence meets the stated target."""
    return float(r) >= float(threshold)


def to_wave(state: KuramotoState, cfg: WaveConfig) -> WaveState:
    """Lift a phase field back to a unit-norm wave."""
    return WaveState.from_complex(np.exp(1j * state.phases), cfg)
