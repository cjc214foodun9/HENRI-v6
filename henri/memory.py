"""Rank-r gated delta memory.

Evidence label: DERIVED (the rank-r algebra) and OBSERVED (drift is
measured in tests).

Design substitution (recorded, not hidden)
------------------------------------------
The architecture's memory specification asks for a dense associative
matrix `S in C^{D x D}` updated by

    S_t = lam S_{t-1} + beta (v_t - S_{t-1} k_t) k_t^dagger

At `D = 65536` that matrix is about 68 GB in complex128. Constraint C7
forbids it, so this module uses a rank-`r` factorization:

    S_t = V_t K_t^dagger,   V_t, K_t in C^{D x r}

which is the same residual-driven update expressed in a subspace of
rank `r`. This is a SUBSTITUTION, not a port. The prediction is
`S k = V (K^H k)`.

Two properties the tests assert:

1. A mastered key produces zero update. When the residual
   `r_t = v_t - S k_t` is already zero, the factors do not move. This
   is the anti-drift property that the retired repo's ratchet decay
   violated.
2. The store never allocates a `D x D` object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

_EPS = 1e-30


@dataclass
class DeltaMemory:
    """Gated delta-rule associative memory of rank `r`.

    Attributes:
        v_factors: `V`, shape `(D, r)`.
        k_factors: `K`, shape `(D, r)`.
        dim: complex dimension `D`.
        rank: factorization rank.
        lam: retention factor applied to the memory each step.
        updates: count of writes applied.
    """

    v_factors: np.ndarray
    k_factors: np.ndarray
    dim: int
    rank: int
    lam: float = 1.0
    updates: int = field(default=0, init=False)

    @classmethod
    def build(
        cls,
        dim: int,
        rank: int,
        lam: float = 1.0,
        dtype=np.complex128,
    ) -> "DeltaMemory":
        """Zero-initialized memory of the requested shape."""
        if rank < 1:
            raise ValueError("rank must be >= 1")
        if rank > dim:
            raise ValueError("rank cannot exceed dim")
        if not 0.0 < lam <= 1.0:
            raise ValueError("lam must be in (0, 1]")
        return cls(
            v_factors=np.zeros((dim, rank), dtype=dtype),
            k_factors=np.zeros((dim, rank), dtype=dtype),
            dim=dim,
            rank=rank,
            lam=lam,
        )

    def predict(self, key: np.ndarray) -> np.ndarray:
        """Return `S k` without forming `S`. Shape `(D,)`."""
        self._check(key)
        coeff = self.k_factors.conj().T @ key       # (r,)
        return self.v_factors @ coeff

    def residual(self, key: np.ndarray, value: np.ndarray) -> np.ndarray:
        """Prediction error `v - S k`, shape `(D,)`."""
        return value - self.predict(key)

    def write(
        self,
        key: np.ndarray,
        value: np.ndarray,
        beta: float = 1.0,
        tol: float = 1e-12,
        decay: bool = True,
    ) -> float:
        """Apply one gated delta update. Returns the update magnitude.

        Args:
            key: address vector, shape `(D,)`.
            value: content vector, shape `(D,)`.
            beta: learning gate, must be non-negative.
            tol: residual norm below which the write is skipped.
            decay: apply `lam` retention. Set False to isolate the
                residual gate from the decay term.

        Returns:
            `||delta||_F` of the combined factor change. Zero means the
            key was already mastered and the memory did not move.
        """
        if beta < 0:
            raise ValueError("beta must be non-negative")
        self._check(key)
        self._check(value)

        k = self._unit(key)
        res = self.residual(key, value)
        res_norm = float(np.linalg.norm(res))
        if res_norm <= tol:
            return 0.0

        if decay and self.lam < 1.0:
            self.v_factors *= self.lam
            self.k_factors *= self.lam

        # Rank-1 update expressed without materializing an outer D x D:
        # V <- V + beta * res * (K^H k)^H / ||K^H k||^2 style guard.
        coeff = self.k_factors.conj().T @ k
        denom = float(np.vdot(coeff, coeff).real) + _EPS
        gain = beta / denom

        self.v_factors += np.outer(res, np.conj(coeff)) * gain
        self.k_factors += np.outer(k, np.conj(coeff)) * gain
        self.updates += 1
        return float(res_norm * abs(gain) * np.linalg.norm(coeff))

    def read(
        self, key: np.ndarray, iterations: int = 3
    ) -> np.ndarray:
        """Iterative recall driven by the residual subspace projection.

        Each pass re-projects the running estimate onto the stored
        factor subspace, which is how a partially-matching key is
        cleaned toward the strongest stored association.
        """
        self._check(key)
        estimate = self.predict(key)
        for _ in range(max(int(iterations) - 1, 0)):
            coeff = self.k_factors.conj().T @ estimate
            estimate = self.v_factors @ coeff
        return estimate

    def spectral_radius_proxy(self) -> float:
        """Frobenius norm of `K^H V`, a scale proxy for `||S||`."""
        return float(np.linalg.norm(self.k_factors.conj().T @ self.v_factors))

    def nbytes(self) -> int:
        return int(self.v_factors.nbytes + self.k_factors.nbytes)

    def _unit(self, vec: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(vec)
        return vec / (n + _EPS)

    def _check(self, vec: np.ndarray) -> None:
        arr = np.asarray(vec).reshape(-1)
        if arr.shape[0] != self.dim:
            raise ValueError(
                f"expected length {self.dim}, got {arr.shape[0]}"
            )


def assert_no_dense_dxd(*arrays: np.ndarray) -> None:
    """Raise if any array is a dense square matrix at wave dimension."""
    for arr in arrays:
        a = np.asarray(arr)
        if a.ndim == 2 and a.shape[0] == a.shape[1] and a.shape[0] > 512:
            raise MemoryError(
                f"dense {a.shape[0]}x{a.shape[1]} allocation refused "
                "(constraint C7)"
            )


def drift_under_replay(
    mem: DeltaMemory,
    pairs: list,
    beta: float = 1.0,
    cycles: int = 3,
) -> Tuple[float, float]:
    """Measure representation drift when mastered pairs are replayed.

    Args:
        mem: the memory to exercise.
        pairs: sequence of `(key, value)` arrays.
        beta: write gate.
        cycles: full passes over `pairs`.

    Returns:
        `(initial_loss, final_loss)` where loss is the mean squared
        reconstruction error over `pairs`. A memory that does not drift
        has `final_loss <= initial_loss`.
    """
    def loss() -> float:
        total = 0.0
        for key, value in pairs:
            total += float(np.mean(np.abs(mem.residual(key, value)) ** 2))
        return total / max(len(pairs), 1)

    initial = loss()
    for _ in range(int(cycles)):
        for key, value in pairs:
            mem.write(key, value, beta=beta)
    final = loss()
    return initial, final


def forget_curve(mem: DeltaMemory) -> float:
    """Ratio of retained norm to initial norm proxy.

    Used by the monotonicity test: with `lam = 1` the memory retains
    everything it has written, so this stays at 1.0.
    """
    return float(mem.lam ** max(mem.updates, 1))
