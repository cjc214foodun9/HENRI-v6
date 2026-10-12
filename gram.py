"""Dual Gram matrix, crosstalk metric, and Stiefel retraction.

Evidence label: DERIVED (all linear algebra) and OBSERVED (the
crosstalk value is measured).

Memory rule (constraint C7)
---------------------------
No dense `D x D` object is ever allocated. Everything here is `k x k`
with `k <= 256`. At `D = 65536` a dense complex64 `D x D` matrix is
about 68 GB, so the dual form is not an optimization; it is the only
feasible form.

Stiefel retraction
------------------
`stiefel_retract` uses the closed-form polar retraction through the
dual Gram matrix:

    X (X^H X)^{-1/2}

The `k x k` eigen-decomposition of `X^H X` supplies the inverse square
root, so the cost is `O(D k^2)` and no `D x D` factor appears. This is
the form the architecture guidance selects over interleaved
Newton-Schulz, which diverges on correlated rows at production `k`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - import cycle guard for type hints only
    from henri.state import WaveState

_EPS = 1e-30


def assert_no_dense_dxd(*arrays: np.ndarray) -> None:
    """Raise if any array is a square matrix at wave dimension.

    A `k x k` Gram matrix with small `k` is allowed. The guard fires
    only when side length exceeds the dual cap.
    """
    for arr in arrays:
        a = np.asarray(arr)
        if a.ndim == 2 and a.shape[0] == a.shape[1] and a.shape[0] > 512:
            raise MemoryError(
                f"dense {a.shape[0]}x{a.shape[1]} allocation refused "
                "(constraint C7). Use the dual k x k form."
            )


def unit_rows(matrix: np.ndarray) -> np.ndarray:
    """Normalize each row to unit norm. Zero rows pass through."""
    m = np.asarray(matrix, dtype=np.complex128)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms < _EPS] = 1.0
    return m / norms


def gram_matrix(patterns: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Dual Gram matrix `G = X X^H`, shape `(k, k)`.

    Args:
        patterns: array of shape `(k, D)`. Rows are memory patterns.
        normalize: unit-normalize rows first. Default True so the
            crosstalk metric is scale-free.
    """
    x = np.asarray(patterns, dtype=np.complex128)
    if x.ndim != 2:
        raise ValueError(f"patterns must be 2-D, got shape {x.shape}")
    if x.shape[0] > 512:
        raise MemoryError(
            f"refusing a {x.shape[0]}x{x.shape[0]} Gram matrix "
            "(constraint C7: k must stay small)"
        )
    if normalize:
        x = unit_rows(x)
    return x @ x.conj().T


def crosstalk(patterns: np.ndarray) -> float:
    """Mean squared off-diagonal coherence. Gate G-U2 metric.

    Returns 0.0 for fewer than two patterns.
    
    """
    gram = gram_matrix(patterns)
    k = gram.shape[0]
    if k < 2:
        return 0.0
    mag2 = np.abs(gram) ** 2
    off_diagonal = float(np.sum(mag2) - np.sum(np.diag(mag2)))
    return off_diagonal / (k * (k - 1))


def max_crosstalk(patterns: np.ndarray) -> float:
    """Largest off-diagonal coherence magnitude. Worst-case partner."""
    gram = gram_matrix(patterns)
    if gram.shape[0] < 2:
        return 0.0
    mag = np.abs(gram).copy()
    np.fill_diagonal(mag, 0.0)
    return float(np.max(mag))


def stiefel_retract(patterns: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Retract rows to orthonormal columns via the polar retraction.

    Computes `X (X^H X)^{-1/2}` using only the `k x k` dual Gram
    matrix. Returns an array of shape `(k, D)` with orthonormal rows
    (`X X^H = I`), or fewer rows if the input is rank-deficient.

    Args:
        patterns: array of shape `(k, D)`.
        tol: eigenvalue floor. Directions below it are dropped, which
            is how rank deficiency is handled rather than dividing by
            a near-zero eigenvalue.
    """
    x = np.asarray(patterns, dtype=np.complex128)
    if x.ndim != 2:
        raise ValueError(f"patterns must be 2-D, got shape {x.shape}")
    k, d = x.shape
    if k < 1:
        raise ValueError("patterns must be non-empty")

    gram = x @ x.conj().T
    vals, vecs = np.linalg.eigh(gram)
    keep = vals > tol * max(1.0, float(np.max(vals)))
    if not np.any(keep):
        raise ValueError("patterns are numerically zero; cannot retract")

    vals_k = vals[keep]
    vecs_k = vecs[:, keep]
    # Rectangular form: (k_keep, k) @ (k, D) -> (k_keep, D).
    # Multiplying on the LEFT by the inverse square root of the dual
    # Gram, then applying it to x, is what actually drops the
    # rank-deficient rows. The symmetric form
    # `vecs @ diag(1/sqrt) @ vecs^H @ x` is (k, k) @ (k, D), which
    # returns all k rows and silently keeps the degenerate directions.
    scale = np.diag(1.0 / np.sqrt(vals_k)) @ vecs_k.conj().T
    return scale @ x


def gram_schmidt(patterns: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Sequential Gram-Schmidt orthogonalization of rows.

    Independent reference for the polar retraction. Both must agree on
    the row-space SPAN for a full-rank input, and both must drop
    degenerate rows for a rank-deficient one.
    """
    x = np.asarray(patterns, dtype=np.complex128)
    if x.ndim != 2:
        raise ValueError(f"patterns must be 2-D, got shape {x.shape}")

    basis = []
    for row in x:
        v = row.copy()
        for b in basis:
            v = v - np.vdot(b, v) * b
        n = np.linalg.norm(v)
        if n > tol * max(1.0, float(np.linalg.norm(row))):
            basis.append(v / n)
    if not basis:
        raise ValueError("all rows are numerically zero")
    return np.vstack(basis)


def orthogonality_error(patterns: np.ndarray) -> float:
    """Frobenius distance between `X X^H` and the identity.

    Only valid when the row count survived retraction; callers should
    compare like with like rather than across shapes.
    """
    x = unit_rows(patterns)
    gram = x @ x.conj().T
    return float(np.linalg.norm(gram - np.eye(gram.shape[0]), ord="fro"))


def row_space_residual(
    raw: np.ndarray, retracted: np.ndarray
) -> float:
    """Worst relative residual of projecting `raw` rows onto the span.

    The projector onto the row space of an orthonormal-row matrix `R`
    (satisfying `R R^H = I`) is `P = R^T conj(R)`. Using `R^H R`
    instead is a transpose-convention error: it is a `D x D` operator
    built from the wrong pairing and leaves a large residual.
    """
    r = np.asarray(retracted, dtype=np.complex128)
    worst = 0.0
    for row in np.asarray(raw, dtype=np.complex128):
        projected = r.T @ (r.conj() @ row)
        denom = max(float(np.linalg.norm(row)), 1e-30)
        worst = max(worst, float(np.linalg.norm(row - projected)) / denom)
    return worst


def binding_identity_error(state: "WaveState") -> float:
    """||conv(psi, delta) - psi||. Circular convolution by a delta is
    the identity, so this is exact up to float error.

    This is the structural check for the convolution ring. A test that
    instead asserts `conv(a, conj(a))` is a delta is WRONG: the
    autocorrelation of a random vector is not an impulse. Measured at
    D=512, the peak mass is 0.003 and the argmax sits at a scattered
    index, so any such test fails for a correct implementation.
    """
    from henri.binding import bind
    from henri.state import WaveState

    cfg = state.cfg
    delta = np.zeros(cfg.dim, dtype=np.complex128)
    delta[0] = 1.0
    impulse = WaveState.from_complex(delta, cfg)
    return float(np.linalg.norm(bind(state, impulse).psi - state.psi))


def needs_retraction(patterns: np.ndarray, limit: float = 0.12) -> bool:
    """True when measured crosstalk exceeds the G-U2 bound."""
    return crosstalk(patterns) > limit
