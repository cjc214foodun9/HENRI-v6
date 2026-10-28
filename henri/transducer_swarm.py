"""Zone A: minimalist first-principle transducer swarm.

Four slot-aligned transducers feed one 65,536-channel wave packet. This is the
INTERFACE the revised architecture specifies, not the neural SLM it describes:
the 10M-1.7B on-chip SLM is BLOCKED (no pinned local model on this host), so
`project_wave_to_slm_prefix` returns the fixed (16, d_tiny) tensor that such a
model would consume. No language model is faked here.

Reuse is deliberate: text goes through `HarmonicLexicon`, which already carries
the measured Bochner lift and the Kuramoto-PPMI diffuser.

Gates that can fail:
  - slot disjointness after routing is exact, not approximate
  - grid encode/decode round-trip must recover every cell
  - the prefix shape is invariant in N_ctx (the measurable form of "O(1)")
"""
from __future__ import annotations

import hashlib
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from henri.harmonic_lexicon import HarmonicLexicon

TWO_PI = 2.0 * math.pi


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.zeros_like(v)


class ZoneATransducerSwarm:
    """Slot 0-3 transducers plus the JIT KV-prefix bridge."""

    def __init__(
        self,
        d_model: int = 16384,
        d_tiny: int = 64,
        prefix_tokens: int = 16,
        vocab_size: int = 512,
        max_cells: int = 64,
        n_values: int = 4,
        seed: int = 42,
    ) -> None:
        if d_model % 4 != 0:
            raise ValueError("d_model must be divisible by 4 (Constraint C6)")
        self.d_model = int(d_model)
        self.d_slot = self.d_model // 4
        self.d_tiny = int(d_tiny)
        self.prefix_tokens = int(prefix_tokens)
        self.n_values = int(n_values)
        self.max_cells = int(max_cells)
        self.seed = int(seed)

        self.lexicon = HarmonicLexicon(
            vocab_size=vocab_size, d_slot=self.d_slot, d_sem=64, seed=seed
        )

        rng = np.random.default_rng(seed + 1)
        self.cell_base = rng.uniform(0.0, TWO_PI, (self.max_cells, self.d_slot))
        # Chunked over d_model, not d_slot: the prefix spans the whole wave.
        chunk = self.d_model // self.prefix_tokens
        self.prefix_proj = rng.standard_normal((chunk, self.d_tiny)) / math.sqrt(float(chunk))

    # ------------------------------------------------------------------ text
    def encode_text(self, tokens: Sequence[str]) -> np.ndarray:
        _, sphere = self.lexicon.encode_tokens(tokens, positional=False)
        return sphere

    def encode_operator(self, tokens: Sequence[str]) -> np.ndarray:
        # Same algebraic transducer; slot separation is applied at routing.
        return self.encode_text(tokens)

    # ------------------------------------------------------------------ grid
    def encode_grid(self, grid: np.ndarray) -> np.ndarray:
        g = np.asarray(grid, dtype=np.int64)
        if g.size > self.max_cells:
            raise ValueError(f"grid has {g.size} cells; capacity {self.max_cells}")
        if g.min() < 0 or g.max() >= self.n_values:
            raise ValueError(f"grid values must lie in [0, {self.n_values - 1}]")
        codes = np.exp(
            1j * (self.cell_base[: g.size] + TWO_PI * g.reshape(-1, 1) / self.n_values)
        )
        return _unit(codes.sum(axis=0))

    def decode_grid(self, wave: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
        n = int(np.prod(shape))
        w = np.asarray(wave, dtype=np.complex128).reshape(-1)
        out = np.zeros(n, dtype=np.int64)
        for cell in range(n):
            candidates = np.exp(
                1j * (self.cell_base[cell] + TWO_PI * np.arange(self.n_values, dtype=np.float64)[:, None] / self.n_values)
            )
            out[cell] = int(np.argmax(np.real(candidates @ np.conj(w))))
        return out.reshape(shape)

    # ---------------------------------------------------------------- sensor
    def encode_sensor(self, values: Sequence[float]) -> np.ndarray:
        v = np.asarray(values, dtype=np.float64).reshape(-1)
        pad = np.zeros(self.lexicon.d_sem, dtype=np.float64)
        pad[: min(v.size, pad.size)] = v[: pad.size]
        angles = self.lexicon.bochner_proj @ pad + self.lexicon.bochner_bias
        bins = self.lexicon._quantize(angles)
        return self.lexicon.bins_to_sphere(bins)

    # ---------------------------------------------------------------- routing
    def route(
        self,
        slot0_entity: Optional[np.ndarray] = None,
        slot1_action: Optional[np.ndarray] = None,
        slot2_target: Optional[np.ndarray] = None,
        slot3_context: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        psi = np.zeros(self.d_model, dtype=np.complex128)
        for k, wave in enumerate((slot0_entity, slot1_action, slot2_target, slot3_context)):
            if wave is None:
                continue
            flat = np.asarray(wave, dtype=np.complex128).reshape(-1)
            n = float(np.linalg.norm(flat))
            if n > 1e-9:
                psi[k * self.d_slot : (k + 1) * self.d_slot] = flat / n
        return _unit(psi)

    # ------------------------------------------------------------- prefix bridge
    def project_wave_to_slm_prefix(self, wave: np.ndarray) -> np.ndarray:
        """Fixed (prefix_tokens, d_tiny) tensor. Shape is invariant in N_ctx.

        This is what a pinned on-chip SLM would consume. The SLM itself is
        BLOCKED on this host; no model is simulated.
        """
        w = np.asarray(wave, dtype=np.complex128).reshape(-1)
        if w.size != self.d_model:
            raise ValueError(f"expected {self.d_model} channels, got {w.size}")
        chunks = w.reshape(self.prefix_tokens, -1)
        real = np.real(chunks) @ self.prefix_proj
        imag = np.imag(chunks) @ self.prefix_proj
        return (real + imag) / math.sqrt(2.0)

    # -------------------------------------------------------------- diagnostics
    def slot_energies(self, wave: np.ndarray) -> np.ndarray:
        return np.array(
            [
                float(np.linalg.norm(np.asarray(wave).reshape(4, self.d_slot)[k]))
                for k in range(4)
            ]
        )
