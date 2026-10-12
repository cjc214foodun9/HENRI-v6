"""Koopman tests. Includes the spec's unitarity claim falsification."""

from __future__ import annotations

import numpy as np
import pytest

from henri.koopman import KoopmanOperator


def test_skew_hermitian_by_construction(operator):
    """S = X - X^H is skew-Hermitian exactly, not approximately."""
    assert operator.is_skew_hermitian()


def test_spec_unitarity_claim_is_false(operator):
    """FALSIFIED: the scaffolding spec asserts ||K psi|| = 1 for this family.

    K = diag(m) + A S B^H is not unitary just because S is
    skew-Hermitian. The dense matrix check pins the refutation.
    """
    assert not operator.is_unitary()


def test_raw_norm_drift_is_nonzero(wave, operator):
    """Direct measurement of the nonzero drift the spec denies."""
    drift = operator.norm_drift(wave)
    assert drift > 0.0, (
        "expected nonzero drift from the low-rank term; a zero value would "
        "mean the operator degenerated to the unitary carrier alone"
    )


def test_step_preserves_norm(wave, operator):
    """The explicit projection restores the sphere exactly."""
    out = operator.step(wave)
    assert out.unit_norm_error() < 1e-12


def test_rollout_stays_on_sphere(wave, operator):
    traj = operator.rollout(wave, steps=5)
    assert len(traj) == 6
    for state in traj:
        assert state.unit_norm_error() < 1e-12


def test_rollout_zero_steps_returns_input(wave, operator):
    traj = operator.rollout(wave, steps=0)
    assert len(traj) == 1
    assert np.allclose(traj[0].psi, wave.psi)


def test_negative_steps_rejected(wave, operator):
    with pytest.raises(ValueError, match="non-negative"):
        operator.rollout(wave, steps=-1)


def test_drift_trace_shape(wave, operator):
    trace = operator.drift_trace(wave, steps=4)
    assert trace.shape == (4,)
    assert np.all(trace >= 0.0)


def test_dense_matrix_refused_at_high_dim():
    """Constraint C7: no dense D x D materialization."""
    cfg = type("C", (), {})()
    from henri.config import WaveConfig
    big = WaveConfig(dim=8192)
    op = KoopmanOperator.build(big, seed=1)
    with pytest.raises(MemoryError, match="dense"):
        op.as_matrix()


def test_small_dense_matrix_matches_raw_apply(wave, operator):
    """The dense form is a test oracle; it must agree with the fast path."""
    mat = operator.as_matrix()
    expected = mat @ wave.psi
    assert np.allclose(operator.apply_raw(wave), expected, atol=1e-10)


def test_dimension_mismatch_rejected(operator):
    from henri.config import WaveConfig
    from henri.state import WaveState
    other = WaveConfig.small(dim=256)
    with pytest.raises(ValueError, match="dimension mismatch"):
        operator.apply_raw(WaveState.random(other, seed=1))


def test_seed_reproduces_operator(cfg):
    a = KoopmanOperator.build(cfg, seed=5)
    b = KoopmanOperator.build(cfg, seed=5)
    assert np.allclose(a.rotation, b.rotation)
    assert np.allclose(a.skew, b.skew)


def test_carriers_have_unit_modulus(cfg):
    op = KoopmanOperator.build(cfg, seed=3)
    assert np.allclose(np.abs(op.rotation), 1.0)
