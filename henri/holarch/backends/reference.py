"""Reference wave kernels: the exact CPU/NumPy semantics every backend must match.

The planar layout is the single contract of the portable layer:

    psi in R^{2 x B x 4 x d_slot},   axis 0 = (real, imaginary)

Every routine below is deterministic: fixed reduction order, float64
accumulation on the reference path, and an integer 64-bit tie-break for the
Zone C ranking (Constraint C11).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


def planar_from_complex(wave: np.ndarray) -> np.ndarray:
    value = np.asarray(wave, dtype=np.complex64)
    return np.stack([value.real, value.imag], axis=0).astype(np.float32)


def complex_from_planar(planar: np.ndarray) -> np.ndarray:
    value = np.asarray(planar, dtype=np.float32)
    return (value[0] + 1j * value[1]).astype(np.complex64)


def planar_ingress(
    token_ids: np.ndarray,
    token_phases: np.ndarray,
    *,
    dim: int,
    role_phases: Optional[np.ndarray] = None,
    role_ids: Optional[np.ndarray] = None,
) -> np.ndarray:
    """TK-1: build the four-slot planar Bochner wave for a batch of windows.

    token_ids: (B, L) integer ids.  Each position contributes
    exp(i * phase[token]) / sqrt(L) to its slot; a role table adds a second,
    weaker conjugate copy so a role-abstracted prefix is present in the same
    wave.  The result is R^{2 x B x 4 x d_slot}.
    """
    ids = np.asarray(token_ids, dtype=np.int64)
    phases = np.asarray(token_phases, dtype=np.float64)
    batch, length = int(ids.shape[0]), int(ids.shape[1])
    d_slot = int(dim) // 4
    planar = np.zeros((2, batch, 4, d_slot), dtype=np.float32)
    if length == 0:
        return planar
    scale = 1.0 / math.sqrt(float(length))
    for index in range(batch):
        for position in range(length):
            token = int(ids[index, position])
            slot = token % 4
            phase = phases[token]
            planar[0, index, slot] += scale * math.cos(phase)
            planar[1, index, slot] += scale * math.sin(phase)
            if role_phases is not None and role_ids is not None:
                role = int(role_ids[token])
                role_phase = float(role_phases[role % role_phases.shape[0]])
                planar[0, index, (slot + 1) % 4] += 0.5 * scale * math.cos(role_phase)
                planar[1, index, (slot + 1) % 4] += 0.5 * scale * math.sin(role_phase)
    return planar


def sagnac_plate(
    planar: np.ndarray,
    mixers: np.ndarray,
    chirps: np.ndarray,
    *,
    kerr: float = 0.0,
    zeta: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """TK-2 reference: four BaTiO3 plates, CW/CCW arms, Sagnac coupler.

    mixers: (L, 4, 4) complex slot mixers (already unitary).
    chirps: (L, d_slot) real Fresnel chirp phase per plate.
    The intra-slot propagation is a 1D FFT, the chirp phase, and the inverse
    FFT; the inter-slot mixer is a 4x4 complex product at each channel.
    """
    wave = complex_from_planar(planar)
    clockwise = wave.copy()
    counter = wave.copy()
    for layer in range(int(mixers.shape[0])):
        chirp = np.asarray(chirps[layer], dtype=np.float64)
        for arm, sign in ((clockwise, 1.0), (counter, -1.0)):
            spectrum = np.fft.fft(arm, axis=-1)
            spectrum = spectrum * np.exp(-1j * sign * float(zeta) * chirp)[None, None, :]
            arm[:] = np.fft.ifft(spectrum, axis=-1)
            if kerr:
                arm[:] = arm * np.exp(1j * float(kerr) * np.abs(arm) ** 2)
        mixer = np.asarray(mixers[layer], dtype=np.complex128)
        clockwise = np.einsum("s t,b t d->b s d", mixer, clockwise)
        counter = np.einsum("s t,b t d->b s d", mixer, counter)
    bright = (clockwise + counter) / math.sqrt(2.0)
    dark = (clockwise - counter) / math.sqrt(2.0)
    return planar_from_complex(bright), planar_from_complex(dark)


def stiefel_readout(planar: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """TK-2 epilogue: G = Re(Psi U^dagger) as two real tensor-core products."""
    psi = np.asarray(planar, dtype=np.float32)
    frame = np.asarray(basis, dtype=np.complex64)
    real = psi[0] @ frame.real.T
    imag = psi[1] @ frame.imag.T
    return (real + imag).astype(np.float32)


def chart_retrieval(
    queries: np.ndarray, charts: np.ndarray, *, top_k: int = 2
) -> np.ndarray:
    """TK-3 reference: top-k chart ids with the 64-bit C11 total order.

    The score is the negative chordal distance, packed as
    (sortable_float32(score) << 32) | (~uint32(engram_id)) so a tie always
    resolves to the lower id, exactly as the GPU warp sort does.
    """
    query = np.asarray(queries, dtype=np.float32)
    bank = np.asarray(charts, dtype=np.float32)
    scores = -(np.sum((query[:, None, :] - bank[None, :, :]) ** 2, axis=-1))
    keys = np.zeros(scores.shape, dtype=np.uint64)
    sortable = scores.astype(np.float32).view(np.uint32).astype(np.uint64)
    ids = np.arange(scores.shape[1], dtype=np.uint64)
    keys = (sortable << np.uint64(32)) | (np.uint64(0xFFFFFFFF) - ids)[None, :]
    order = np.argsort(keys, axis=1, kind="stable")[:, : int(top_k)]
    return order.astype(np.int64)


def born_contextual_phase(
    phi: np.ndarray,
    rows: np.ndarray,
    steering: np.ndarray,
    values: np.ndarray,
    *,
    rank: int = 8,
) -> np.ndarray:
    """TK-5 reference: Theta = Phi + pi tanh((u . v) / sqrt(R)) and Born |A|^2."""
    theta = np.asarray(phi, dtype=np.float64) + math.pi * np.tanh(
        np.einsum("nmr,vr->nmv", steering, values) / math.sqrt(float(rank))
    )
    amplitude = np.einsum("nmv,nmv->nv", np.exp(1j * theta), rows)
    power = np.abs(amplitude) ** 2
    total = np.maximum(power.sum(axis=1, keepdims=True), 1e-30)
    return (power / total).astype(np.float32)


def swarm_topk(
    x: np.ndarray,
    routing: np.ndarray,
    expert_weights: np.ndarray,
    expert_biases: np.ndarray,
    *,
    top_k: int = 2,
) -> np.ndarray:
    """TK-4 reference: static top-k routing, then only the routed experts run."""
    value = np.asarray(x, dtype=np.float32)
    gates = value @ np.asarray(routing, dtype=np.float32).T
    order = np.argsort(-gates, axis=1, kind="stable")[:, : int(top_k)]
    out = np.zeros_like(value)
    for row in range(value.shape[0]):
        chosen = order[row]
        mass = float(np.sum(np.maximum(gates[row, chosen], 0.0))) + 1e-9
        for expert in chosen:
            weight = float(np.maximum(gates[row, int(expert)], 0.0)) / mass
            out[row] += weight * (
                value[row] @ np.asarray(expert_weights[int(expert)], dtype=np.float32)
                + np.asarray(expert_biases[int(expert)], dtype=np.float32)
            )
    return out.astype(np.float32)


class ReferenceWaveBackend:
    """The default backend: portable NumPy, exact on CPU and on Windows."""

    name = "reference"
    device = "cpu"
    triton = False

    def ingress(self, *args: Any, **kwargs: Any) -> np.ndarray:
        return planar_ingress(*args, **kwargs)

    def sagnac(self, *args: Any, **kwargs: Any):
        return sagnac_plate(*args, **kwargs)

    def readout(self, *args: Any, **kwargs: Any) -> np.ndarray:
        return stiefel_readout(*args, **kwargs)

    def charts(self, *args: Any, **kwargs: Any) -> np.ndarray:
        return chart_retrieval(*args, **kwargs)

    def born(self, *args: Any, **kwargs: Any) -> np.ndarray:
        return born_contextual_phase(*args, **kwargs)

    def swarm(self, *args: Any, **kwargs: Any) -> np.ndarray:
        return swarm_topk(*args, **kwargs)


__all__ = [
    "ReferenceWaveBackend",
    "planar_from_complex",
    "complex_from_planar",
    "planar_ingress",
    "sagnac_plate",
    "stiefel_readout",
    "chart_retrieval",
    "born_contextual_phase",
    "swarm_topk",
]
