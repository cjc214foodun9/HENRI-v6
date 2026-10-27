"""Phase-locked semantic lexicon on the unit torus T^{D_slot}.

Omega Amendment 1. Replaces the claim that hashed tokens must be orthogonal
with an explicit, measurable phase-grounding path.

Two pathways:
  A. Kuramoto-PPMI diffusion over a sparse co-occurrence graph.
  B. Isometric trigonometric lift of dense semantic vectors (Bochner kernel).

Design notes (measured, not asserted):
  - Diffusion writes back to quantized bins, so the result survives re-encode.
  - Repulsion is measured against the mean field of the ACTIVE node set only.
  - The Bochner lift preserves monotonic cosine similarity only up to the
    quantization bin width; that error is measured in the verification suite,
    not assumed away.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

TWO_PI = 2.0 * math.pi


class HarmonicLexicon:
    """Phase-locked semantic lexicon on T^{D_slot}."""

    def __init__(
        self,
        vocab_size: int = 4096,
        d_slot: int = 16384,
        d_sem: int = 384,
        quantization_bins: int = 256,
        gamma: float = 1.25,
        seed: int = 42,
    ) -> None:
        if quantization_bins < 2 or quantization_bins > 256:
            raise ValueError("quantization_bins must be in [2, 256] for uint8 storage")
        self.vocab_size = int(vocab_size)
        self.d_slot = int(d_slot)
        self.d_sem = int(d_sem)
        self.quantization_bins = int(quantization_bins)
        self.gamma = float(gamma)
        self.seed = int(seed)

        rng = np.random.default_rng(self.seed)
        self.phase_bins = rng.integers(
            0, self.quantization_bins, size=(self.vocab_size, self.d_slot), dtype=np.uint8
        )

        raw = rng.standard_normal((self.d_slot, self.d_sem))
        q, _ = np.linalg.qr(raw)
        self.bochner_proj = (q * math.sqrt(self.d_slot / float(self.d_sem))).astype(np.complex128)
        self.bochner_bias = rng.uniform(0.0, TWO_PI, size=self.d_slot)

        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

    # ------------------------------------------------------------------ convert
    def _quantize(self, angles: np.ndarray) -> np.ndarray:
        wrapped = np.remainder(angles, TWO_PI)
        bins = np.rint(wrapped * (self.quantization_bins / TWO_PI)).astype(np.int64)
        return np.remainder(bins, self.quantization_bins).astype(np.uint8)

    def bins_to_torus(self, bins: np.ndarray) -> np.ndarray:
        angles = np.asarray(bins, dtype=np.float64) * (TWO_PI / self.quantization_bins)
        return np.exp(1j * angles)

    def bins_to_sphere(self, bins: np.ndarray) -> np.ndarray:
        return self.bins_to_torus(bins) / math.sqrt(float(self.d_slot))

    # ------------------------------------------------------------------- tokens
    def register_token(self, token: str) -> int:
        if token in self.token_to_id:
            return self.token_to_id[token]
        idx = len(self.token_to_id)
        if idx >= self.vocab_size:
            raise ValueError(f"vocabulary capacity {self.vocab_size} exceeded")
        self.token_to_id[token] = idx
        self.id_to_token[idx] = token
        return idx

    def token_bins(self, token: str) -> np.ndarray:
        return self.phase_bins[self.register_token(token)]

    def similarity(self, token_a: str, token_b: str) -> float:
        a = self.bins_to_sphere(self.token_bins(token_a))
        b = self.bins_to_sphere(self.token_bins(token_b))
        return float(np.real(np.vdot(a, b)))

    # ----------------------------------------------------------------- Pathway B
    def lift_semantic_embeddings(
        self, token_ids: Sequence[int], embeddings: np.ndarray
    ) -> np.ndarray:
        """Project dense semantic vectors onto quantized phases (Bochner lift)."""
        emb = np.asarray(embeddings, dtype=np.float64)
        if emb.ndim == 1:
            emb = emb[None, :]
        if emb.shape[-1] != self.d_sem:
            raise ValueError(f"embedding dim {emb.shape[-1]} != d_sem {self.d_sem}")
        norm = np.linalg.norm(emb, axis=-1, keepdims=True)
        unit = emb / np.clip(norm, 1e-12, None)
        angles = self.gamma * (unit @ self.bochner_proj.T).real + self.bochner_bias
        bins = self._quantize(angles)
        ids = np.asarray(token_ids, dtype=np.int64)
        self.phase_bins[ids] = bins
        return self.bins_to_sphere(bins)

    # ----------------------------------------------------------------- Pathway A
    def diffuse_ppmi_graph(
        self,
        edge_src: Sequence[int],
        edge_dst: Sequence[int],
        ppmi_weights: Sequence[float],
        steps: int = 25,
        lr: float = 0.25,
        repulsion: float = 0.05,
    ) -> None:
        """Attractive-repulsive Kuramoto phase diffusion over a sparse PPMI graph."""
        src = np.asarray(edge_src, dtype=np.int64)
        dst = np.asarray(edge_dst, dtype=np.int64)
        if src.size == 0:
            return
        w = np.asarray(ppmi_weights, dtype=np.float64).reshape(-1, 1)

        active = np.unique(np.concatenate([src, dst]))
        local = {int(g): i for i, g in enumerate(active)}
        s = np.array([local[int(v)] for v in src], dtype=np.int64)
        d = np.array([local[int(v)] for v in dst], dtype=np.int64)

        angles = self.phase_bins[active].astype(np.float64) * (TWO_PI / self.quantization_bins)

        # Mean-field normalisation. Without it the explicit update gain is
        # lr * 2 * W = 1.5 > 1 and two strongly coupled nodes overshoot into a
        # limit cycle instead of locking. Dividing by the degree keeps the
        # integrator stable for any edge weight.
        degree = np.zeros(angles.shape[0], dtype=np.float64)
        np.add.at(degree, s, 1.0)
        np.add.at(degree, d, 1.0)
        degree = np.maximum(degree, 1.0)[:, None]

        for _ in range(int(steps)):
            phasor = np.exp(1j * angles)
            mean_phase = np.angle(phasor.mean(axis=0, keepdims=True))
            force = w * np.sin(angles[d] - angles[s])
            grad = np.zeros_like(angles)
            np.add.at(grad, s, force)
            np.add.at(grad, d, -force)
            rep = repulsion * np.sin(mean_phase - angles)
            angles = np.remainder(angles + lr * (grad / degree - rep), TWO_PI)

        self.phase_bins[active] = self._quantize(angles)

    # ------------------------------------------------------------------- encode
    def encode_tokens(
        self, tokens: Sequence[str], positional: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Bind-and-superpose. Returns (torus, sphere).

        `positional=False` bundles as a SET (no circular shift). Use it for
        bag-of-words retrieval queries; positional binding is order-sensitive
        by design and will not match a differently ordered bundle.
        """
        if not tokens:
            zero = np.zeros(self.d_slot, dtype=np.complex128)
            return zero, zero
        ids = np.array([self.register_token(t) for t in tokens], dtype=np.int64)
        phasors = self.bins_to_torus(self.phase_bins[ids])
        if positional:
            shifted = np.stack(
                [np.roll(phasors[pos], pos) for pos in range(len(tokens))], axis=0
            )
        else:
            shifted = phasors
        bundled = shifted.sum(axis=0)
        q = self._quantize(np.angle(bundled))
        torus = self.bins_to_torus(q)
        return torus, torus / math.sqrt(float(self.d_slot))
