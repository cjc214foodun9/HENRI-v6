"""Continuous Modern Hopfield cleanup (CCCP).

Evidence label: DERIVED (the energy and update follow Ramsauer et al.
2020) and OBSERVED (the monotonicity check is measured in tests).

Energy:

    E(x) = -tau * logsumexp( Re<X_m, x> / tau ) + 0.5 * ||x||^2

One CCCP step replaces x by the softmax-weighted codebook mean:

    p = softmax( Re(X x^H) / tau )
    x <- p @ X

Design note on monotonicity
---------------------------
The Lyapunov decrease guarantee applies to the UNNORMALIZED iterate.
Normalizing inside the loop changes the 0.5*||x||^2 term and can break
monotonicity. So this module iterates on the unnormalized variable and
normalizes only the returned state. The energy trace is what the gate
inspects, and the test asserts non-increase rather than assuming it.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from henri.state import WaveState

_EPS = 1e-30


def hopfield_energy(
    x: np.ndarray, codebook: np.ndarray, tau: float
) -> float:
    """Lyapunov energy of a raw vector against a codebook.

    Args:
        x: complex vector of shape (D,).
        codebook: complex array of shape (M, D).
        tau: temperature, must be positive.
    """
    if tau <= 0:
        raise ValueError("tau must be positive")
    logits = np.real(codebook @ np.conj(x)) / tau
    m = float(np.max(logits))
    lse = m + np.log(np.sum(np.exp(logits - m)) + _EPS)
    return float(-tau * lse + 0.5 * np.sum(np.abs(x) ** 2))


def cccp_step(x: np.ndarray, codebook: np.ndarray, tau: float) -> np.ndarray:
    """One Concave-Convex Procedure update on a raw vector."""
    logits = np.real(codebook @ np.conj(x)) / tau
    logits = logits - np.max(logits)
    weights = np.exp(logits)
    weights = weights / (np.sum(weights) + _EPS)
    return weights @ codebook


def cleanup(
    state: WaveState,
    codebook: np.ndarray,
    tau: Optional[float] = None,
    iters: Optional[int] = None,
    tol: float = 1e-9,
) -> Tuple[WaveState, np.ndarray]:
    """Relax a state into the nearest codebook attractor.

    Args:
        state: the query wave.
        codebook: complex array of shape (M, D). Rows are patterns.
        tau: temperature. Defaults to the config value.
        iters: iteration cap. Defaults to the config value.
        tol: early-stop when the energy change is below this value.

    Returns:
        (settled state, energy trace). The trace has one entry per
        iteration and is non-increasing under the stated conditions.
    """
    cfg = state.cfg
    tau = cfg.tau_hopfield if tau is None else tau
    iters = cfg.hopfield_iters if iters is None else iters

    book = np.asarray(codebook, dtype=np.complex128)
    if book.ndim != 2 or book.shape[1] != cfg.dim:
        raise ValueError(
            f"codebook must be (M, {cfg.dim}); got shape {book.shape}"
        )
    if book.shape[0] < 1:
        raise ValueError("codebook needs at least one pattern")

    x = state.psi.copy()
    trace = [hopfield_energy(x, book, tau)]
    for _ in range(int(iters)):
        x = cccp_step(x, book, tau)
        energy = hopfield_energy(x, book, tau)
        trace.append(energy)
        if abs(trace[-1] - trace[-2]) < tol:
            break

    settled = WaveState.from_complex(x, cfg)
    settled.cleaned = True
    return settled, np.asarray(trace, dtype=np.float64)


def snap_index(state: WaveState, codebook: np.ndarray) -> int:
    """Index of the codebook row with the largest real coherence.

    This is a raw argmax. It is NOT a ranking primitive: on a tie
    plateau it arbitrates rather than ranks (constraint C11). Use
    `max_group` from henri.readout for selection under ties.
    """
    book = np.asarray(codebook, dtype=np.complex128)
    scores = np.real(book @ np.conj(state.psi))
    return int(np.argmax(scores))


def max_group(state: WaveState, codebook: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Indices whose score equals the maximum, within `tol`.

    Tie-robust: returns the whole tied set instead of one index.
    """
    book = np.asarray(codebook, dtype=np.complex128)
    scores = np.real(book @ np.conj(state.psi))
    top = float(np.max(scores))
    return np.flatnonzero(scores >= top - tol)


def crosstalk(codebook: np.ndarray) -> float:
    """Mean squared off-diagonal coherence of a codebook (G-U2 metric).

    Each row is normalized first, so the value is scale-free.
    """
    book = np.asarray(codebook, dtype=np.complex128)
    norms = np.linalg.norm(book, axis=1, keepdims=True)
    norms[norms < _EPS] = 1.0
    unit = book / norms
    gram = unit @ np.conj(unit.T)
    k = gram.shape[0]
    if k < 2:
        return 0.0
    mag2 = np.abs(gram) ** 2
    off = float(np.sum(mag2) - np.sum(np.diag(mag2)))
    return off / (k * (k - 1))
