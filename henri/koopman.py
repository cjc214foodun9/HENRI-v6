"""Koopman transition operator on the hypersphere.

Evidence label: DERIVED (the operator is linear) and FALSIFIED for the
spec's unitarity claim.

Spectrum correction (recorded, not hidden)
------------------------------------------
`henri_wave_mechanical_model_scaffolding_specification.md` section 6.2
asserts:

    K = diag(m) + A S B^dagger,  S = -S^dagger
    => ||K psi||_2 = ||psi||_2 = 1.0

That implication is false. `diag(m)` is unitary when |m_d| = 1, but the
low-rank term `A S B^dagger` is a general complex matrix. The sum has
eigenvalues near `1 + i*lambda`, whose modulus is not 1. Skew-Hermitian
`S` makes the interaction energy-conserving in the interaction picture;
it does not make the composite operator unitary.

This module therefore does two honest things:
1. Exposes `apply_raw` and measures the actual norm drift.
2. Exposes `step`, which applies the operator and projects back onto
   the sphere. The projection is explicit, not hidden.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from henri.config import WaveConfig
from henri.state import WaveState


@dataclass
class KoopmanOperator:
    """Low-rank Koopman transition.

    Attributes:
        rotation: unit-modulus phase carriers `m`, shape (D,).
        left: adapter `A`, shape (D, r).
        right: adapter `B`, shape (D, r).
        skew: interaction kernel `S`, shape (r, r), skew-Hermitian.
    """

    rotation: np.ndarray
    left: np.ndarray
    right: np.ndarray
    skew: np.ndarray

    @classmethod
    def build(
        cls,
        cfg: WaveConfig,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
    ) -> "KoopmanOperator":
        """Construct an operator with unit-modulus carriers.

        `S` is built as `X - X^H`, which is skew-Hermitian by
        construction, so `S = -S^H` holds exactly rather than
        approximately.
        """
        gen = rng if rng is not None else np.random.default_rng(seed)
        d, r = cfg.dim, cfg.r_koopman

        phases = gen.uniform(-np.pi, np.pi, size=d)
        rotation = np.exp(1j * phases)

        scale = 1.0 / np.sqrt(d)
        left = (gen.standard_normal((d, r)) + 1j * gen.standard_normal((d, r))) * scale
        right = (gen.standard_normal((d, r)) + 1j * gen.standard_normal((d, r))) * scale

        raw = gen.standard_normal((r, r)) + 1j * gen.standard_normal((r, r))
        skew = raw - raw.conj().T

        return cls(rotation=rotation, left=left, right=right, skew=skew)

    # -- properties ----------------------------------------------------

    @property
    def dim(self) -> int:
        return int(self.rotation.shape[0])

    @property
    def rank(self) -> int:
        return int(self.left.shape[1])

    def is_skew_hermitian(self, tol: float = 1e-12) -> bool:
        """True when S = -S^H within tolerance."""
        return bool(np.allclose(self.skew, -self.skew.conj().T, atol=tol))

    def is_unitary(self, tol: float = 1e-9, probes: int = 8, seed: int = 0) -> bool:
        """True when K preserves norms.

        Tested by PROBE, not by materializing K. Calling `as_matrix`
        here would trip the C7 guard at production dimension (D = 65536
        is a ~68 GB dense complex matrix), so the dense path is
        restricted to the small dimensions where it is a legitimate
        oracle.

        A unitary operator preserves the norm of EVERY vector, so a set
        of random probes is a sufficient negative test: one violated
        probe refutes unitarity.
        """
        # Small dimensions: use the exact dense oracle.
        if self.dim <= 512:
            mat = self.as_matrix()
            return bool(
                np.allclose(mat.conj().T @ mat, np.eye(self.dim), atol=tol)
            )

        # Production dimensions: probe norms through the low-rank path.
        rng = np.random.default_rng(seed)
        for _ in range(int(probes)):
            v = rng.standard_normal(self.dim) + 1j * rng.standard_normal(self.dim)
            v = v / np.linalg.norm(v)
            out = (
                self.rotation * v
                + self.left @ (self.skew @ (self.right.conj().T @ v))
            )
            if abs(np.linalg.norm(out) - 1.0) > tol:
                return False
        return True

    # -- operators -----------------------------------------------------

    def as_matrix(self) -> np.ndarray:
        """Dense matrix form. TEST USE ONLY.

        Guarded against production use: at D = 65536 this allocates
        roughly 68 GB. Constraint C7.
        """
        if self.dim > 4096:
            raise MemoryError(
                f"refusing to materialize a {self.dim}x{self.dim} dense matrix "
                "(constraint C7: zero dense D x D allocations). "
                "Use apply_raw/step for production paths."
            )
        return (
            np.diag(self.rotation)
            + self.left @ self.skew @ self.right.conj().T
        )

    def apply_raw(self, state: WaveState) -> np.ndarray:
        """Apply K without normalizing. Returns a raw complex vector.

        Complexity is O(D r) via two rank-r products, never D^2.
        """
        if state.cfg.dim != self.dim:
            raise ValueError(f"dimension mismatch: {self.dim} vs {state.cfg.dim}")
        main = self.rotation * state.psi
        coeff = self.right.conj().T @ state.psi        # (r,)
        return main + self.left @ (self.skew @ coeff)

    def norm_drift(self, state: WaveState) -> float:
        """| ||K psi|| - 1 |. Nonzero drift is the falsified claim."""
        out = self.apply_raw(state)
        return float(abs(np.linalg.norm(out) - 1.0))

    def step(self, state: WaveState) -> WaveState:
        """Apply K and project back onto the sphere.

        The projection is the explicit correction for the non-unitary
        low-rank term. Returns a settled unit-norm state.
        """
        return WaveState.from_complex(self.apply_raw(state), state.cfg)

    def rollout(self, state: WaveState, steps: int) -> list:
        """Iterate `step` and collect the trajectory. Includes t=0."""
        if steps < 0:
            raise ValueError("steps must be non-negative")
        traj = [state.copy()]
        current = state
        for _ in range(int(steps)):
            current = self.step(current)
            traj.append(current)
        return traj

    def drift_trace(self, state: WaveState, steps: int) -> np.ndarray:
        """Norm drift per step. Used by the G-U3 falsification test."""
        out = np.empty(int(steps), dtype=np.float64)
        current = state
        for i in range(int(steps)):
            out[i] = self.norm_drift(current)
            current = self.step(current)
        return out
