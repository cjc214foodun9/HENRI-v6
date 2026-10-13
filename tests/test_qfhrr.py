"""qFHRR tests: exact phasor algebra and the F2 contrast.

Directive 2 asserts that unit-modulus phase encoding recovers what
Gaussian circular convolution saturates at `1/sqrt(2)`. These tests
measure that assertion rather than assume it.

They also pin the defect found while writing the first version: baking
`1/sqrt(D)` into each component makes binding collapse by a factor of
`sqrt(D)` per operation.
"""

from __future__ import annotations

import numpy as np
import pytest

from henri.qfhrr import (
    bundle,
    coherence,
    pbind,
    phasor_vector,
    punbind,
    to_sphere,
    unbinding_residual,
)

DIMS = (256, 4096)


def test_phasor_vector_is_unit_modulus():
    """`|z_d| = 1` exactly; norm is `sqrt(D)`, not 1."""
    rng = np.random.default_rng(3)
    for dim in DIMS:
        v = phasor_vector(dim, rng)
        assert v.shape == (dim,)
        np.testing.assert_allclose(np.abs(v), 1.0, rtol=0, atol=1e-15)
        assert abs(np.linalg.norm(v) - np.sqrt(dim)) < 1e-9


def test_to_sphere_matches_owner_stated_magnitude():
    """The owner's `|Psi_d| = 1/sqrt(D)` is the normalized view."""
    rng = np.random.default_rng(31)
    for dim in DIMS:
        v = to_sphere(phasor_vector(dim, rng))
        np.testing.assert_allclose(np.abs(v), 1 / np.sqrt(dim), rtol=1e-12)
        assert abs(np.linalg.norm(v) - 1.0) < 1e-12


def test_phasor_vector_rejects_bad_dim():
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError):
        phasor_vector(0, rng)
    with pytest.raises(ValueError):
        phasor_vector(-4, rng)


def test_pbind_preserves_unit_modulus():
    """Closure on the ring: the product is again unit modulus.

    This is the assertion the first version failed, at `1/D` instead of
    `1`. Pinned at exact tolerance so a scale regression cannot pass.
    """
    rng = np.random.default_rng(4)
    for dim in DIMS:
        a = phasor_vector(dim, rng)
        b = phasor_vector(dim, rng)
        ab = pbind(a, b)
        np.testing.assert_allclose(np.abs(ab), 1.0, rtol=0, atol=1e-15)
        assert abs(np.linalg.norm(ab) - np.sqrt(dim)) < 1e-9


def test_deep_binding_does_not_collapse():
    """No scale drift after 6 successive bindings.

    The retired formulation lost a factor of `sqrt(D)` per binding; at
    6 bindings and D=65536 that is a `2**24` collapse. This is the
    property directive 2 needs for deep associative unbinding.
    """
    rng = np.random.default_rng(41)
    dim = 65536
    acc = phasor_vector(dim, rng)
    for _ in range(6):
        acc = pbind(acc, phasor_vector(dim, rng))
        np.testing.assert_allclose(np.abs(acc), 1.0, rtol=0, atol=1e-12)
    assert abs(np.linalg.norm(acc) - np.sqrt(dim)) < 1e-6


def test_unbinding_is_exact():
    """`unbind(a, bind(a, b)) == b` to float precision.

    This is the directive-2 claim, measured. Gaussian circular
    convolution saturates at coherence `1/sqrt(2)`; the phasor product
    has no such saturation.
    """
    rng = np.random.default_rng(5)
    for dim in DIMS:
        a = phasor_vector(dim, rng)
        b = phasor_vector(dim, rng)
        residual = unbinding_residual(a, pbind(a, b), b)
        assert residual < 1e-9, f"dim={dim} residual={residual}"
        assert coherence(punbind(a, pbind(a, b)), b) > 1.0 - 1e-12


def test_unbinding_exact_after_deep_binding():
    """Recovery survives 5 nested bindings, unbound innermost-first."""
    rng = np.random.default_rng(51)
    dim = 4096
    keys = [phasor_vector(dim, rng) for _ in range(5)]
    value = phasor_vector(dim, rng)
    acc = value.copy()
    for k in keys:
        acc = pbind(acc, k)
    rec = acc.copy()
    for k in reversed(keys):
        rec = punbind(k, rec)
    assert float(np.linalg.norm(rec - value)) < 1e-9


def test_normalization_commutes_with_binding_up_to_one_scale():
    """Normalizing commutes with binding up to a global magnitude.

    `to_sphere(pbind(a, b)) == to_sphere(pbind(to_sphere(a), to_sphere(b)))`.

    The naive claim `to_sphere(pbind(a, b)) == pbind(to_sphere(a),
    to_sphere(b))` is FALSE and was asserted here before the defect was
    measured. Pointwise multiplication SQUARES the component scale, so
    the right side carries `1/D` components while the left carries
    `1/sqrt(D)`. They differ by exactly `sqrt(D)`. The two sides agree
    only after the right side is renormalized as well.
    """
    rng = np.random.default_rng(61)
    dim = 2048
    a = phasor_vector(dim, rng)
    b = phasor_vector(dim, rng)
    lhs = to_sphere(pbind(a, b))
    rhs = to_sphere(pbind(to_sphere(a), to_sphere(b)))
    np.testing.assert_allclose(lhs, rhs, rtol=1e-12, atol=1e-15)
    # Pin the defect itself: the un-renormalized right side is off by
    # exactly sqrt(D), never by 1.
    raw_rhs = pbind(to_sphere(a), to_sphere(b))
    ratio = float(np.linalg.norm(lhs) / np.linalg.norm(raw_rhs))
    assert abs(ratio - np.sqrt(dim)) < 1e-6


def test_bundle_rejects_empty_and_zero():
    with pytest.raises(ValueError):
        bundle([])
    zero = np.zeros(16, dtype=np.complex128)
    with pytest.raises(ValueError):
        bundle([zero])


def test_bundle_stays_on_the_ring():
    rng = np.random.default_rng(71)
    v = bundle([phasor_vector(1024, rng) for _ in range(3)])
    np.testing.assert_allclose(np.abs(v), 1.0, rtol=0, atol=1e-12)


def test_shape_mismatch_raises():
    rng = np.random.default_rng(7)
    a = phasor_vector(64, rng)
    b = phasor_vector(128, rng)
    with pytest.raises(ValueError):
        pbind(a, b)
    with pytest.raises(ValueError):
        punbind(a, b)
    with pytest.raises(ValueError):
        coherence(a, b)
