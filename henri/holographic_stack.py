"""Reversible unitary pushdown call-stack inside Slot 3 (H_Context).

Omega Amendment 2a. A bounded-depth stack carried in one fixed-width wave.

Fixes applied to the reference sketch:
  - `pop` uses the FULL complex projection. The reference kept only the real
    part, which leaks the imaginary component of the top frame into the parent.
  - Empty-stack normalization is guarded instead of dividing by 1e-12.

Known limit (stated, not hidden): this is a bounded-depth stack, NOT a
Turing-complete machine. Capacity decays geometrically with depth and is
measured in the verification suite.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.zeros_like(v)
    return v / n


class HolographicCallStack:
    """Unit-modulus phase shift-register with push / peek / pop."""

    def __init__(
        self,
        d_slot: int = 16384,
        decay: float = 0.78,
        shift_stride: int = 127,
        seed: int = 108,
    ) -> None:
        if not 0.0 < decay <= 1.0:
            raise ValueError("decay must be in (0, 1]")
        self.d_slot = int(d_slot)
        self.decay = float(decay)
        self.shift_stride = int(shift_stride) % int(d_slot)
        rng = np.random.default_rng(seed)
        angles = rng.uniform(0.0, 2.0 * math.pi, self.d_slot)
        self.push_phasor = np.exp(1j * angles)

    # ------------------------------------------------------------------ helpers
    def depth_bound(self, tol: float = 1e-3) -> int:
        """Largest L with decay**L >= tol (the honest capacity statement)."""
        if self.decay >= 1.0:
            return self.d_slot
        return int(math.floor(math.log(tol) / math.log(self.decay)))

    # -------------------------------------------------------------------- stack
    def push(self, stack_wave: np.ndarray, subgoal_wave: np.ndarray) -> np.ndarray:
        goal = _unit(np.asarray(subgoal_wave, dtype=np.complex128))
        stack = np.asarray(stack_wave, dtype=np.complex128)
        if float(np.linalg.norm(stack)) < 1e-9:
            return goal
        shifted = np.roll(stack, self.shift_stride) * self.push_phasor
        return _unit(goal + self.decay * shifted)

    def peek(
        self, stack_wave: np.ndarray, codebook: np.ndarray
    ) -> Tuple[int, float]:
        stack = _unit(np.asarray(stack_wave, dtype=np.complex128))
        sims = np.real(np.asarray(codebook, dtype=np.complex128) @ np.conj(stack))
        idx = int(np.argmax(sims))
        return idx, float(sims[idx])

    def pop(
        self, stack_wave: np.ndarray, top_goal_wave: np.ndarray, empty_tol: float = 0.10
    ) -> np.ndarray:
        """Remove the top frame. Returns zeros when the stack is effectively empty.

        `empty_tol` is a relative amplitude, placed between the codebook
        cross-talk floor (~1/sqrt(D) = 0.016 at D=4096) and one decay step
        (0.78). Without it, normalising a ~2% residual amplifies pure noise
        back to unit norm and the stack never reports empty.
        """
        stack = np.asarray(stack_wave, dtype=np.complex128)
        goal = _unit(np.asarray(top_goal_wave, dtype=np.complex128))
        proj = np.vdot(goal, stack)  # full complex coefficient
        residual = stack - proj * goal
        scale = float(np.linalg.norm(stack))
        if scale < 1e-12 or float(np.linalg.norm(residual)) / scale < empty_tol:
            return np.zeros_like(stack)
        restored = np.roll(residual * np.conj(self.push_phasor), -self.shift_stride)
        return _unit(restored)

    def frame_weights(self, depth: int) -> np.ndarray:
        """Normalized amplitude of each frame after `depth` pushes (model)."""
        raw = np.array([self.decay ** k for k in range(depth)], dtype=np.float64)
        return raw / math.sqrt(float(np.sum(raw ** 2)))
