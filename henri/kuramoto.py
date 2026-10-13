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
        """Take phases from a wave and draw natural frequencies.

        Determinism is not optional: pass `rng` or `seed`.

        DEFECT FOUND AND FIXED (recorded, not hidden). An earlier version
        fell through to `np.random.default_rng(None)`, which seeds from
        OS entropy. Measured on the same input wave: phases identical,
        `omega` differing by `4.840156` between two calls. That made
        `test_locking_raises_order_parameter` non-reproducible, and it
        failed once inside a full-suite run while passing in isolation.

        This guard makes the whole class of defect impossible, instead of
        repairing the call sites one at a time. It mirrors the existing
        fail-closed rule in `kuramoto_step`, which already refuses
        `noise > 0` without an explicit `rng`.
        """
        if rng is None and seed is None:
            raise ValueError(
                "from_wave requires an explicit rng or seed (determinism): "
                "default_rng(None) would seed from OS entropy"
            )
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


def lock_steps_for_dim(
    dim: int,
    coupling: float,
    dt: float = 1e-3,
    factor: float = 1.0,
) -> int:
    """Steps needed for a random phase field to reach order ~1.

    The order parameter of a uniformly random phase field is
    `r0 ~ 1/sqrt(D)`. Near `r = 0` the mean-field dynamics give
    `dr/dt ~ (K/2) r`, so `r` grows exponentially and the time to lock
    scales as:

        t_lock ~ 2 * ln(D) / K        steps = ceil(factor * t_lock / dt)

    This is a property of the DYNAMICS, not a harness constant. A step
    count calibrated at small D is too short at production D.

    Measured at D=65536, K=8, dt=5e-3, t=0.3: r moved 0.0039 -> 0.0119,
    matching `exp(4 * 0.3) = 3.3`; the predicted lock time is
    `2*ln(65536)/8 ~ 2.8`.

    `dt` is a parameter on purpose. An earlier version hardcoded a
    `* 1000` factor, which silently assumed `dt = 1e-3`. The caller used
    `dt = 5e-3`, so it returned ~11,000 steps where ~554 suffice -- a
    20x overkill that costs real wall-clock at production dimension.

    The `ln(D)` form is a coarse upper bound: it ignores the target
    order, so `factor = 1.0` still leaves roughly 2x margin.
    """
    if dim < 2:
        raise ValueError("dim must be >= 2")
    if coupling <= 0:
        raise ValueError("coupling must be positive")
    if dt <= 0:
        raise ValueError("dt must be positive")
    if factor <= 0:
        raise ValueError("factor must be positive")
    t_lock = 2.0 * float(np.log(dim)) / float(coupling)
    return int(np.ceil(factor * t_lock / float(dt)))


def is_locked(r: float, threshold: float = LOCK_THRESHOLD) -> bool:
    """True when coherence meets the stated target."""
    return float(r) >= float(threshold)


def to_wave(state: KuramotoState, cfg: WaveConfig) -> WaveState:
    """Lift a phase field back to a unit-norm wave."""
    return WaveState.from_complex(np.exp(1j * state.phases), cfg)
