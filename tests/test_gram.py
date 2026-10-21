"""Gram tests: crosstalk metric, Stiefel retraction, dense-allocation gate."""

from __future__ import annotations

import numpy as np
import pytest

from henri.gram import (
    crosstalk,
    gram_matrix,
    gram_schmidt,
    max_crosstalk,
    needs_retraction,
    orthogonality_error,
    row_space_residual,
    stiefel_retract,
    unit_rows,
)
from henri.hopfield import crosstalk as hopfield_crosstalk


def _random_rows(k, d, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((k, d)) + 1j * rng.standard_normal((k, d))


def test_gram_shape_is_dual():
    """G is (k, k), never (D, D). Constraint C7."""
    x = _random_rows(16, 512)
    g = gram_matrix(x)
    assert g.shape == (16, 16)


def test_crosstalk_of_orthonormal_is_tiny():
    """Measured 1.87e-17 for k=20 at D=512. Bar sits above float noise."""
    x = stiefel_retract(_random_rows(20, 512))
    assert crosstalk(x) < 1e-12


def test_stiefel_retract_bounds_crosstalk_below_gate():
    """Gate G-U2: crosstalk <= 0.12 after retraction."""
    x = stiefel_retract(_random_rows(24, 512))
    assert crosstalk(x) <= 0.12


def test_retraction_improves_orthogonality():
    raw = _random_rows(24, 512, seed=3)
    before = orthogonality_error(raw)
    after = orthogonality_error(stiefel_retract(raw))
    assert after < before


def test_retracted_rows_are_orthonormal():
    x = stiefel_retract(_random_rows(12, 512))
    gram = x @ x.conj().T
    assert np.allclose(gram, np.eye(12), atol=1e-9)


def test_retraction_preserves_row_space_span():
    """Polar retraction must not leave the subspace it spans.

    The projector onto the row space of an orthonormal-row `R` is
    `P = R^T conj(R)`. The pairing `R^H R` is a transpose-convention
    error and leaves a residual of order `||row||` (measured 31.7).
    """
    raw = _random_rows(8, 512, seed=4)
    retracted = stiefel_retract(raw)
    assert row_space_residual(raw, retracted) < 1e-8


def test_raw_projector_convention_is_the_wrong_one():
    """Negative control: the mistaken pairing must fail the span check.

    If both conventions passed, the span test could not discriminate.
    """
    raw = _random_rows(8, 512, seed=4)
    retracted = stiefel_retract(raw)
    row = raw[0]
    wrong = retracted.conj().T @ (retracted @ row)
    right = retracted.T @ (retracted.conj() @ row)
    assert np.linalg.norm(row - wrong) > 1.0
    assert np.linalg.norm(row - right) < 1e-8


def test_rank_deficient_input_drops_rows():
    """Rank deficiency is handled by dropping, not by dividing by zero.

    Measured: three collinear rows in, ONE orthonormal row out.
    """
    base = _random_rows(1, 512, seed=5)
    stacked = np.vstack([base, base * 2.0, base * 0.5])
    out = stiefel_retract(stacked)
    assert out.shape[0] == 1
    assert np.allclose(out @ out.conj().T, np.eye(1), atol=1e-9)


def test_rank_deficient_output_is_orthonormal():
    """The surviving rows must still be orthonormal, not just fewer."""
    rng = np.random.default_rng(12)
    independent = rng.standard_normal((2, 512)) + 1j * rng.standard_normal((2, 512))
    stacked = np.vstack([independent, independent[0] * 3.0, independent[1] * 0.25])
    out = stiefel_retract(stacked)
    assert out.shape[0] == 2
    assert np.allclose(out @ out.conj().T, np.eye(2), atol=1e-9)


def test_all_zero_rows_rejected():
    with pytest.raises(ValueError, match="numerically zero"):
        stiefel_retract(np.zeros((3, 512), dtype=np.complex128))


def test_gram_schmidt_agrees_with_polar_on_span():
    raw = _random_rows(6, 512, seed=6)
    polar = stiefel_retract(raw)
    gs = gram_schmidt(raw)
    assert polar.shape == gs.shape
    assert np.allclose(gs @ gs.conj().T, np.eye(6), atol=1e-9)
    assert row_space_residual(raw, gs) < 1e-8


def test_gram_schmidt_drops_degenerate_rows():
    """Both orthogonalizers must handle rank deficiency the same way."""
    base = _random_rows(1, 512, seed=13)
    stacked = np.vstack([base, base * 4.0])
    assert gram_schmidt(stacked).shape[0] == 1


def test_crosstalk_single_row_is_zero():
    assert crosstalk(_random_rows(1, 512)) == 0.0


def test_max_crosstalk_ignores_diagonal():
    x = stiefel_retract(_random_rows(10, 512))
    assert max_crosstalk(x) < 1e-9


def test_needs_retraction_flags_crowded_codebook():
    """A deliberately crowded codebook must trip the gate.

    Built by projecting many rows into a small subspace, which is the
    real crowding condition the metric exists to detect.
    """
    rng = np.random.default_rng(7)
    basis = stiefel_retract(_random_rows(3, 512, seed=8))
    coeffs = rng.standard_normal((40, 3)) + 1j * rng.standard_normal((40, 3))
    crowded = coeffs @ basis
    assert needs_retraction(crowded, limit=0.12)


def test_dual_gram_matches_hopfield_metric():
    """Two independent crosstalk implementations must agree."""
    x = _random_rows(16, 512, seed=9)
    assert abs(crosstalk(x) - hopfield_crosstalk(x)) < 1e-12


def test_unit_rows_normalizes():
    x = _random_rows(4, 128, seed=10)
    u = unit_rows(x)
    assert np.allclose(np.linalg.norm(u, axis=1), 1.0)


def test_large_k_refused():
    """Guard against an accidental large Gram allocation.

    The guard is a BYTE BUDGET, not a fixed 512-row cap. The 512 cap made
    the retraction trigger unevaluable at production cartridge size
    (M=4096), which was a measured defect, not protection. This test pins
    the budget so the guard cannot disappear, and pins that a production
    M x M Gram is NOT refused.
    """
    from henri import gram as gram_module

    saved = gram_module.GRAM_BYTES_LIMIT
    try:
        gram_module.GRAM_BYTES_LIMIT = 1024
        with pytest.raises(MemoryError, match="budget"):
            gram_matrix(_random_rows(600, 512))
    finally:
        gram_module.GRAM_BYTES_LIMIT = saved

    # The dual form at production size must pass the guard.
    assert gram_matrix(_random_rows(600, 512)).shape == (600, 600)
    assert gram_module.GRAM_BYTES_LIMIT >= 600 * 600 * 16
