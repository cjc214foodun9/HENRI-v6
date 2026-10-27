"""Blind K-way factorization of bound torus waves.

Omega Amendment 5.

Two ARMS, deliberately separate, because the specification conflates them:

  * `factorize_exact`  - deterministic joint search over the (K-1) codebooks.
    Establishes SOLVABILITY at a stated scale. Cost is O(M^(K-1) * D).
  * `factorize_iterative` - the adiabatic non-Hermitian Kuramoto-Langevin flow
    described in the amendment. This is the mechanism under test.

The verification suite runs both. A pass on the exact arm does NOT certify the
iterative dynamics; the suite reports each separately. Blind factorization is
non-convex, and Audit Flag 5 already concedes limit cycles are possible.
"""
from __future__ import annotations

import itertools
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np


def _to_unit_phasor(z: np.ndarray) -> np.ndarray:
    return np.exp(1j * np.angle(z))


class AdiabaticResonator:
    def __init__(
        self,
        d_slot: int = 16384,
        beta_min: float = 2.0,
        beta_max: float = 28.0,
        max_steps: int = 60,
        chiral_omega: float = 0.18,
        noise_sigma: float = 0.45,
        tau_lock: float = 0.0431,
        seed: int = 2026,
    ) -> None:
        self.d_slot = int(d_slot)
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.max_steps = int(max_steps)
        self.chiral_omega = float(chiral_omega)
        self.noise_sigma = float(noise_sigma)
        self.tau_lock = float(tau_lock)
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------- exact
    def factorize_exact(
        self, composite: np.ndarray, codebooks: Sequence[np.ndarray]
    ) -> Tuple[List[int], float, int]:
        comp = np.asarray(composite, dtype=np.complex128).reshape(-1)
        cbs = [np.asarray(c, dtype=np.complex128) for c in codebooks]
        K = len(cbs)
        inv_d = 1.0 / float(self.d_slot)
        if K == 1:
            sims = np.real(cbs[0] @ np.conj(comp)) * inv_d
            idx = int(np.argmax(sims))
            return [idx], float(1.0 - sims[idx]), 1
        if K == 2:
            best = (-2.0, [0, 0])
            for i in range(cbs[0].shape[0]):
                prod = cbs[0][i] * cbs[1]
                sims = np.real(prod @ np.conj(comp)) * inv_d
                j = int(np.argmax(sims))
                if sims[j] > best[0]:
                    best = (float(sims[j]), [i, j])
            return best[1], float(1.0 - best[0]), 1
        if K != 3:
            raise ValueError("exact arm supports K in {1, 2, 3}")

        best_sim = -2.0
        best_idx = [0, 0, 0]
        for i in range(cbs[0].shape[0]):
            for j in range(cbs[1].shape[0]):
                prefix = cbs[0][i] * cbs[1][j]
                unbound = comp * np.conj(prefix)
                sims = np.real(cbs[2] @ np.conj(unbound)) * inv_d
                k = int(np.argmax(sims))
                sim = float(np.real(np.vdot(prefix * cbs[2][k], comp)) * inv_d)
                if sim > best_sim:
                    best_sim = sim
                    best_idx = [i, j, k]
        return best_idx, float(max(0.0, 1.0 - best_sim)), 1

    # --------------------------------------------------------------- iterative
    def factorize_iterative(
        self, composite: np.ndarray, codebooks: Sequence[np.ndarray], num_restarts: int = 3
    ) -> Tuple[List[int], float, int]:
        comp = np.asarray(composite, dtype=np.complex128).reshape(-1)
        cbs = [np.asarray(c, dtype=np.complex128) for c in codebooks]
        K = len(cbs)
        inv_d = 1.0 / float(self.d_slot)

        best_indices = [0] * K
        best_error = 1.0

        for restart in range(int(num_restarts)):
            estimates: List[np.ndarray] = []
            for cb in cbs:
                if restart == 0:
                    estimates.append(_to_unit_phasor(cb.sum(axis=0)))
                else:
                    w = self.rng.random((cb.shape[0], 1)) + 0.5
                    estimates.append(_to_unit_phasor((cb * w).sum(axis=0)))

            for step in range(self.max_steps):
                progress = step / float(max(1, self.max_steps - 1))
                beta_t = self.beta_min + (self.beta_max - self.beta_min) * (progress ** 1.5)

                snapped: List[int] = []
                prod = np.ones_like(comp)
                for k in range(K):
                    sims = np.real(cbs[k] @ np.conj(estimates[k])) * inv_d
                    idx = int(np.argmax(sims))
                    snapped.append(idx)
                    prod = prod * cbs[k][idx]
                snap_err = 1.0 - float(np.real(np.vdot(prod, comp)) * inv_d)
                if snap_err < best_error:
                    best_error = snap_err
                    best_indices = snapped
                if snap_err <= self.tau_lock:
                    return snapped, snap_err, step + 1

                for k in range(K):
                    other = np.ones_like(comp)
                    for j in range(K):
                        if j != k:
                            other = other * np.conj(estimates[j])
                    unbound = comp * other
                    sims = np.real(cbs[k] @ np.conj(unbound)) * inv_d
                    logits = beta_t * (sims - sims.max())
                    probs = np.exp(logits)
                    probs = probs / probs.sum()
                    clean = probs.astype(np.complex128) @ cbs[k]

                    chiral = np.exp(1j * self.chiral_omega * snap_err * (k + 1))
                    thermal_angle = (
                        self.rng.standard_normal(self.d_slot)
                        * self.noise_sigma
                        * math.sqrt(max(0.0, snap_err))
                        * (1.0 - 0.8 * progress)
                    )
                    thermal = np.exp(1j * thermal_angle)
                    updated = 0.25 * estimates[k] * chiral + 0.75 * clean * thermal
                    estimates[k] = _to_unit_phasor(updated)

        return best_indices, best_error, self.max_steps

    # ---------------------------------------------------------------- dispatch
    def factorize(
        self, composite: np.ndarray, codebooks: Sequence[np.ndarray], num_restarts: int = 3
    ) -> Tuple[List[int], float, int]:
        """Backward-compatible entry point: the specified iterative dynamics."""
        return self.factorize_iterative(composite, codebooks, num_restarts=num_restarts)

    def search_budget(self, codebooks: Sequence[np.ndarray]) -> int:
        return int(np.prod([c.shape[0] for c in codebooks[:-1]]))
