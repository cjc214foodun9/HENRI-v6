"""Phase-gated Koopman sheaf with exact O(r^2) online Dual-EDMD updates.

Omega Amendment 2b.

Fixes applied to the reference sketch:
  - `diag_m` initializes to the IDENTITY phase, not random unit phasors.
    Random phasors make the prediction orthogonal to every stored engram.
  - `online_edmd_update` symmetrizes the inverse-covariance and reports the
    measured post-update residual, so the caller can test error reduction.
  - A dedicated `prediction_error` exposes the quantity a falsifiable gate needs.

Honest expressive limit: the operator class is
    K = diag(m) + A S B^H    with A, B in St(r, d_model)
so the learnable part moves energy only inside an r-dimensional subspace.
For r << d_model the sheaf models NEAR-IDENTITY dynamics plus a rank-r
correction. The verification suite tests exactly that, with a negative control
whose correction subspace is orthogonal to the target.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np


def retract_to_sphere(psi: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(psi))
    if n < 1e-12:
        return np.zeros_like(psi)
    return psi / n


class PhaseGatedKoopmanSheaf:
    """C conditional Stiefel charts gated by Slot-3 context resonance."""

    def __init__(
        self,
        d_model: int = 65536,
        d_slot: int = 16384,
        rank: int = 64,
        num_charts: int = 8,
        gate_beta: float = 18.0,
        forgetting_factor: float = 0.995,
        ridge_lambda: float = 1e-4,
        seed: int = 42,
    ) -> None:
        if d_model % 4 != 0:
            raise ValueError("d_model must be divisible by 4 (Constraint C6)")
        if not 0.0 < forgetting_factor <= 1.0:
            raise ValueError("forgetting_factor must be in (0, 1]")
        self.d_model = int(d_model)
        self.d_slot = int(d_slot)
        self.rank = int(rank)
        self.num_charts = int(num_charts)
        self.gate_beta = float(gate_beta)
        self.gamma = float(forgetting_factor)
        self.ridge_lambda = float(ridge_lambda)

        rng = np.random.default_rng(seed)

        raw_keys = rng.standard_normal((self.d_slot, self.num_charts)) + 1j * rng.standard_normal(
            (self.d_slot, self.num_charts)
        )
        q_keys, _ = np.linalg.qr(raw_keys)
        self.chart_keys = q_keys.T.conj()  # (C, d_slot), orthonormal rows

        # FIX: identity phase. Random phasors destroy the signal.
        self.diag_m = np.ones((self.num_charts, self.d_model), dtype=np.complex128)

        a_raw = rng.standard_normal((self.d_model, self.rank)) + 1j * rng.standard_normal(
            (self.d_model, self.rank)
        )
        self.stiefel_A, _ = np.linalg.qr(a_raw)
        b_raw = rng.standard_normal((self.d_model, self.rank)) + 1j * rng.standard_normal(
            (self.d_model, self.rank)
        )
        self.stiefel_B, _ = np.linalg.qr(b_raw)

        self.core_S = np.zeros((self.num_charts, self.rank, self.rank), dtype=np.complex128)
        eye = np.eye(self.rank, dtype=np.complex128)
        self.cov_P = np.tile(eye / self.ridge_lambda, (self.num_charts, 1, 1))
    def compute_chart_weights(self, psi: np.ndarray) -> np.ndarray:
        psi = np.asarray(psi, dtype=np.complex128).reshape(-1)
        slot3 = psi[3 * self.d_slot : 4 * self.d_slot]
        n = float(np.linalg.norm(slot3))
        weights = np.zeros(self.num_charts, dtype=np.float64)
        if n < 1e-6:
            weights[0] = 1.0
            return weights
        unit = slot3 / n
        logits = self.gate_beta * np.real(self.chart_keys @ np.conj(unit))
        logits -= logits.max()
        e = np.exp(logits)
        return e / e.sum()

    # -------------------------------------------------------------------- step
    def step(
        self, psi: np.ndarray, chart_idx: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        psi = np.asarray(psi, dtype=np.complex128).reshape(-1)
        if chart_idx is not None:
            weights = np.zeros(self.num_charts, dtype=np.float64)
            weights[int(chart_idx)] = 1.0
        else:
            weights = self.compute_chart_weights(psi)

        w = weights.astype(np.complex128)
        diag_term = (w @ self.diag_m) * psi
        z = self.stiefel_B.conj().T @ psi
        nz = float(np.linalg.norm(z))
        u = z / nz if nz > 1e-12 else z
        s_eff = np.einsum("c,cij->ij", w, self.core_S)
        low_rank = self.stiefel_A @ ((s_eff @ u) * nz)
        return retract_to_sphere(diag_term + low_rank), weights

    # ------------------------------------------------------------------ update
    def online_edmd_update(
        self, psi_t: np.ndarray, psi_next_target: np.ndarray, chart_idx: int = 0
    ) -> float:
        c = int(chart_idx)
        psi_t = np.asarray(psi_t, dtype=np.complex128).reshape(-1)
        target = np.asarray(psi_next_target, dtype=np.complex128).reshape(-1)

        # Normalized regressor. ||B^H psi||^2 is about rank/d_model (2e-3 here),
        # so an unnormalized regressor makes the ridge dominate the data term:
        # the first-step gain collapses to ||z||^2/(||z||^2 + lambda).
        z = self.stiefel_B.conj().T @ psi_t
        nz = float(np.linalg.norm(z))
        if nz < 1e-12:
            psi_pred, _ = self.step(psi_t, chart_idx=c)
            return float(np.linalg.norm(target - psi_pred))
        u = z / nz

        residual = target - self.diag_m[c] * psi_t
        z_target = self.stiefel_A.conj().T @ residual
        y = z_target / nz

        P = self.cov_P[c]
        S = self.core_S[c]

        Pu = P @ u
        a = float(np.real(np.vdot(u, Pu)))
        denom = self.gamma + max(a, 1e-12)
        P_new = (P - np.outer(Pu, np.conj(Pu)) / denom) / self.gamma
        P_new = 0.5 * (P_new + P_new.conj().T)

        pred_err = y - S @ u
        S_new = S + np.outer(pred_err, np.conj(u) @ P_new)

        self.cov_P[c] = P_new
        self.core_S[c] = S_new

        psi_pred, _ = self.step(psi_t, chart_idx=c)
        return float(np.linalg.norm(target - psi_pred))

    # -------------------------------------------------------------- diagnostics
    def prediction_error(
        self, psi_t: np.ndarray, target: np.ndarray, chart_idx: int = 0
    ) -> Tuple[float, float]:
        """Return (error_before_training_free_step, error_of_baseline_identity)."""
        psi_t = np.asarray(psi_t, dtype=np.complex128).reshape(-1)
        target = np.asarray(target, dtype=np.complex128).reshape(-1)
        pred, _ = self.step(psi_t, chart_idx=chart_idx)
        return float(np.linalg.norm(target - pred)), float(
            np.linalg.norm(target - self.diag_m[int(chart_idx)] * psi_t)
        )
