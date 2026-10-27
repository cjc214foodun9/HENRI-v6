"""Torch mirrors of the Omega hot paths.

Scope: four paths, not the legacy spine. The spine is 32/32 NumPy and stays
NumPy; porting it is a separate, larger approval.

NUMERICS. Torch runs in complex64. The NumPy modules run in complex128. A
complex64 vector of length D accumulates roughly sqrt(D) * eps error, so at
D = 16384 the expected agreement floor is about 1.6e-5. The parity suite uses
1e-4 for vector agreement and states that derivation; it is not a tuned bar.

CUDA. Nothing here claims a CUDA run. `verify_torch_parity.py` prints the
device it actually used and refuses to report latency for a device it did not
measure.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

DTYPE = torch.complex64


def retract_to_sphere_t(psi: torch.Tensor) -> torch.Tensor:
    return psi / torch.linalg.vector_norm(psi).clamp(min=1e-12)


def hopfield_readout_t(
    query: torch.Tensor, codebook: torch.Tensor, beta: float = 26.10
) -> Tuple[torch.Tensor, float, float]:
    """Mirror of the continuous modern-Hopfield cleanup."""
    q = retract_to_sphere_t(query)
    sims = torch.real(codebook @ q.conj())
    probs = torch.softmax(beta * sims, dim=-1)
    clean = retract_to_sphere_t(probs.to(DTYPE) @ codebook)
    top2 = torch.topk(sims, k=2).values
    margin = float((top2[0] - top2[1]).item())
    m = int(sims.shape[0])
    entropy = float(-(probs * torch.log(probs.clamp(min=1e-12))).sum().item())
    return clean, margin, entropy / math.log(float(m)) if m > 1 else 0.0


def stiefel_retract_t(X: torch.Tensor) -> float:
    """X_ortho = (X X^H)^(-1/2) X. Returns post-retraction c_max."""
    m = int(X.shape[0])
    if m < 2:
        return 0.0
    X64 = X.to(torch.complex128)
    G = X64 @ X64.conj().T
    G = 0.5 * (G + G.conj().T) + 1e-10 * torch.eye(m, dtype=torch.complex128, device=X.device)
    evals, evecs = torch.linalg.eigh(G)
    G_inv_sqrt = (evecs * torch.rsqrt(evals.clamp(min=1e-12))) @ evecs.conj().T
    Xo = (G_inv_sqrt @ X64).to(DTYPE)
    Xo = Xo / torch.linalg.vector_norm(Xo, dim=-1, keepdim=True).clamp(min=1e-12)
    G_post = torch.abs(Xo @ Xo.conj().T)
    G_post.fill_diagonal_(0.0)
    return float(G_post.max().item()) if G_post.numel() else 0.0


class KoopmanSheafT:
    """Mirror of PhaseGatedKoopmanSheaf: identity phase plus rank-r correction."""

    def __init__(
        self,
        d_model: int = 16384,
        rank: int = 32,
        num_charts: int = 4,
        ridge_lambda: float = 1e-4,
        forgetting_factor: float = 0.995,
        seed: int = 42,
        device: Optional[torch.device] = None,
    ) -> None:
        self.d_model = int(d_model)
        self.rank = int(rank)
        self.num_charts = int(num_charts)
        self.gamma = float(forgetting_factor)
        self.device = device or torch.device("cpu")
        gen = torch.Generator(device="cpu").manual_seed(seed)
        a = torch.randn(d_model, rank, generator=gen) + 1j * torch.randn(
            d_model, rank, generator=gen
        )
        self.A = torch.linalg.qr(a.to(DTYPE))[0].to(self.device)
        b = torch.randn(d_model, rank, generator=gen) + 1j * torch.randn(
            d_model, rank, generator=gen
        )
        self.B = torch.linalg.qr(b.to(DTYPE))[0].to(self.device)
        # Accumulators in complex128. The RLS covariance starts at 1/lambda = 1e4
        # and the rank-1 downdate cancels it almost exactly; in complex64 that
        # cancellation leaves ~1e-3 error (measured rel_err 9.6e-4 on the step).
        # Vectors stay complex64; only the small (r x r) recursion is promoted.
        self.core_S = torch.zeros(num_charts, rank, rank, dtype=torch.complex128, device=self.device)
        eye = torch.eye(rank, dtype=torch.complex128, device=self.device)
        self.cov_P = (eye / ridge_lambda).repeat(num_charts, 1, 1)

    def step(self, psi: torch.Tensor, chart: int = 0) -> torch.Tensor:
        z = self.B.conj().T @ psi
        nz = torch.linalg.vector_norm(z).clamp(min=1e-12)
        S = self.core_S[int(chart)].to(DTYPE)
        low_rank = self.A @ ((S @ (z / nz)) * nz)
        return retract_to_sphere_t(psi + low_rank)

    @torch.no_grad()
    def online_edmd_update(self, psi_t: torch.Tensor, target: torch.Tensor, chart: int = 0) -> float:
        c = int(chart)
        z = self.B.conj().T @ psi_t
        nz = torch.linalg.vector_norm(z)
        if float(nz.item()) < 1e-12:
            return float(torch.linalg.vector_norm(target - self.step(psi_t, c)).item())
        u = (z / nz).to(torch.complex128).unsqueeze(-1)
        y = ((self.A.conj().T @ (target - psi_t)) / nz).to(torch.complex128).unsqueeze(-1)

        P = self.cov_P[c]
        S = self.core_S[c]
        Pu = P @ u
        denom = self.gamma + torch.real(u.conj().T @ Pu).squeeze().clamp(min=1e-12)
        P_new = (P - (Pu @ Pu.conj().T) / denom) / self.gamma
        P_new = 0.5 * (P_new + P_new.conj().T)
        S_new = S + (y - S @ u) @ (u.conj().T @ P_new)
        self.cov_P[c] = P_new
        self.core_S[c] = S_new
        return float(torch.linalg.vector_norm(target - self.step(psi_t, c)).item())


def sagnac_slot_errors_t(
    psi_a: torch.Tensor, psi_b: torch.Tensor, num_slots: int = 4
) -> torch.Tensor:
    """Magnitude-weighted per-slot homodyne discrepancy (mirror of the gate)."""
    d_slot = int(psi_a.shape[-1]) // int(num_slots)
    a = psi_a.reshape(num_slots, d_slot)
    b = psi_b.reshape(num_slots, d_slot)
    na = torch.linalg.vector_norm(a, dim=-1)
    nb = torch.linalg.vector_norm(b, dim=-1)
    active = (na > 1e-6) & (nb > 1e-6)
    au = a / na.clamp(min=1e-12).unsqueeze(-1)
    bu = b / nb.clamp(min=1e-12).unsqueeze(-1)
    cos = torch.real((au * bu.conj()).sum(-1)).clamp(-1.0, 1.0)
    raw = 1.0 - cos
    both_min = torch.minimum(na, nb)
    weight = both_min / torch.clamp(both_min.max(), min=1.0)
    return torch.where(active, raw * weight, torch.zeros_like(raw))
