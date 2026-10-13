"""qFHRR: unit-modulus phase-domain binding.

Evidence label: DERIVED for the algebra; OBSERVED for the fidelity
numbers measured by `tests/test_qfhrr.py`.

Directive 2 of the 2026-10-12 owner review: semantic tokens intended
for deep associative unbinding use unit-modulus phase encoding rather
than unconstrained complex Gaussian variables.

Defect found and fixed in this module (recorded, not hidden)
------------------------------------------------------------
The first version of `phasor_vector` divided every component by
`sqrt(D)`, giving `|z_d| = 1/sqrt(D)` and unit norm. That looks like
the owner's stated magnitude, but binding then SQUARES the component
scale: `bind(a, b)_d = a_d * b_d` has magnitude `1/D`, and the bound
vector norm collapses to `1/sqrt(D)`. Measured failure at D=2048:
component `0.00048828125` = `1/2048` against an expected
`0.0220970869` = `1/sqrt(2048)`. Every additional binding loses another
factor of `sqrt(D)`, so deep binding collapses -- the exact property
directive 2 requires.

Correct formulation
-------------------
The algebra lives on the unit-MODULUS ring:

    z_d = exp(i * theta_d),        |z_d| = 1  for every d,  ||z|| = sqrt(D)

Binding is the pointwise product and preserves `|z_d| = 1` EXACTLY, at
any depth. Unbinding is the conjugate product divided by `|key_d|^2`,
which is `1` on the ring, so recovery is exact.

The owner's `|Psi_d| = 1/sqrt(D)` is the SAME object after the single
division by `sqrt(D)` that places it on `S^{D-1}`. `to_sphere` performs
that division. So there is one encoding and two views: an exact algebra
view and a unit-norm sphere view.

Constraint C7 holds: every operation is O(D) or O(K*D). No dense
`D x D` object is formed.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-30


def phasor_vector(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Unit-modulus phasor. `|z_d| = 1` exactly; norm is `sqrt(dim)`.

    This is the algebra-domain encoding. Divide by `sqrt(dim)` with
    `to_sphere` for the unit-norm sphere view.
    """
    if dim < 1:
        raise ValueError("dim must be positive")
    phases = rng.uniform(-np.pi, np.pi, dim)
    return np.exp(1j * phases)


def to_sphere(v: np.ndarray) -> np.ndarray:
    """Project onto `S^{D-1}`. Result has `|Psi_d| = 1/sqrt(D)`.

    This is the owner's stated magnitude, reached by the single
    division that normalization applies.
    """
    arr = np.asarray(v, dtype=np.complex128).reshape(-1)
    n = float(np.linalg.norm(arr))
    if not np.isfinite(n) or n < _EPS:
        raise ValueError("cannot normalize a zero or non-finite vector")
    return arr / n


def pbind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pointwise phasor binding. Preserves `|z_d| = 1` exactly.

    Closure holds at any binding depth, because a product of unit
    moduli is a unit modulus in every component.
    """
    a = np.asarray(a, dtype=np.complex128).reshape(-1)
    b = np.asarray(b, dtype=np.complex128).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return a * b


def punbind(key: np.ndarray, bound: np.ndarray) -> np.ndarray:
    """Pointwise phasor unbinding.

    Divides by `|key_d|^2` so the operation is exact for any nonzero
    key, and reduces to the conjugate product on the unit-modulus ring.
    """
    key = np.asarray(key, dtype=np.complex128).reshape(-1)
    bound = np.asarray(bound, dtype=np.complex128).reshape(-1)
    if key.shape != bound.shape:
        raise ValueError(f"shape mismatch: {key.shape} vs {bound.shape}")
    denom = np.abs(key) ** 2
    denom = np.where(denom < _EPS, _EPS, denom)
    return bound * np.conj(key) / denom


def bundle(vectors: list) -> np.ndarray:
    """Superpose phasors and renormalize to unit modulus.

    Renormalization keeps the result in the algebra domain, so it can
    be bound again without scale drift.
    """
    if not vectors:
        raise ValueError("bundle needs at least one vector")
    total = np.zeros_like(np.asarray(vectors[0], dtype=np.complex128).reshape(-1))
    for v in vectors:
        total = total + np.asarray(v, dtype=np.complex128).reshape(-1)
    n = float(np.linalg.norm(total))
    if not np.isfinite(n) or n < _EPS:
        raise ValueError("degenerate bundle: zero norm")
    # Renormalize components back to unit modulus, preserving the ring.
    return total / np.abs(total)


def coherence(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized real coherence of two complex vectors. Scale-free."""
    a = np.asarray(a, dtype=np.complex128).reshape(-1)
    b = np.asarray(b, dtype=np.complex128).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + _EPS
    return float(np.real(np.vdot(a, b)) / denom)


def unbinding_residual(key: np.ndarray, bound: np.ndarray, value: np.ndarray) -> float:
    """`||unbind(key, bound) - value||_2`. Zero up to float error."""
    recovered = punbind(key, bound)
    return float(
        np.linalg.norm(recovered - np.asarray(value, dtype=np.complex128).reshape(-1))
    )
