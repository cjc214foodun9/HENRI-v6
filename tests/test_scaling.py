"""Dimension-scaling laws.

These pin the two D-fragility defects found by running at production
dimension, so they cannot silently return. See F7 in
`docs/EVIDENCE_REGISTER.md`.
"""

from __future__ import annotations

import numpy as np
import pytest

from henri.config import WaveConfig
from henri.kuramoto import (
    LOCK_THRESHOLD,
    KuramotoState,
    lock,
    lock_steps_for_dim,
    order_parameter,
)
from henri.sagnac import perturb_with_coherence
from henri.state import WaveState


@pytest.mark.parametrize("dim", [64, 512, 4096])
def test_perturbation_is_dimension_invariant(dim):
    """Measured coherence matches the target at every dimension.

    The retired defect: a fixed 0.002 per-component scale gives a
    relative perturbation of `0.002*sqrt(2D)` -- 0.064 at D=512 but
    0.72 at D=65536. A dimension-invariant construction holds the
    coherence constant instead.
    """
    cfg = WaveConfig(dim=dim)
    rng = np.random.default_rng(20261012)
    base = WaveState.random(cfg, rng=rng)
    for target in (0.5, 0.9, 0.999):
        got = base.coherence(perturb_with_coherence(base, target, rng))
        assert abs(got - target) < 1e-6, (dim, target, got)


def test_perturbation_delta_norm_is_sqrt_2_1_minus_target():
    """After normalization, ||delta|| = sqrt(2 * (1 - target)).

    Not `eps`. `WaveState.from_complex` normalizes, so the stored delta
    is `pert / ||pert|| - v`, whose norm works out to:

        ||delta||^2 = (t - 1)^2 + eps^2 * t^2
                    = (1 - t)[(1 - t) + (1 + t)]
                    = 2 * (1 - t)

    For target 0.5 this is 1.0, while `eps` would be sqrt(3) = 1.732.
    Asserting `||delta|| == eps` is the error this test now prevents.
    """
    cfg = WaveConfig(dim=1024)
    rng = np.random.default_rng(7)
    base = WaveState.random(cfg, rng=rng)
    for target in (0.5, 0.9, 0.999):
        got = perturb_with_coherence(base, target, rng)
        delta = got.psi - base.psi
        assert abs(np.linalg.norm(delta) - np.sqrt(2.0 * (1.0 - target))) < 1e-9


def test_perturbation_along_base_component_is_target_minus_one():
    """<v, delta> = target - 1 exactly.

    The orthogonal projection makes the PRE-normalization perturbation
    orthogonal to `v`, but normalization rescales the whole vector, so
    the STORED delta acquires a component along `v` of `t - 1`. A test
    that expects `vdot(v, delta) == 0` fails for a correct
    implementation.
    """
    cfg = WaveConfig(dim=512)
    rng = np.random.default_rng(3)
    base = WaveState.random(cfg, rng=rng)
    for target in (0.5, 0.8, 0.99):
        got = perturb_with_coherence(base, target, rng)
        delta = got.psi - base.psi
        along = np.vdot(base.psi, delta)
        assert abs(along - (target - 1.0)) < 1e-9, (target, along)
        assert abs(along) > 1e-3  # NOT zero, by design


def test_perturbation_validates_target():
    cfg = WaveConfig(dim=64)
    rng = np.random.default_rng(0)
    base = WaveState.random(cfg, rng=rng)
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            perturb_with_coherence(base, bad, rng)


def test_lock_steps_scale_with_log_dim():
    """t_lock ~ 2 ln(D) / K, so steps = ceil(t_lock / dt)."""
    for dim in (1024, 4096, 65536):
        steps = lock_steps_for_dim(dim, coupling=8.0, dt=5e-3)
        expected = np.ceil(2.0 * np.log(dim) / 8.0 / 5e-3)
        assert steps == int(expected)


def test_lock_steps_respects_dt():
    """An earlier version hardcoded *1000, silently assuming dt=1e-3.

    The caller used dt=5e-3, so it returned ~20x too many steps at
    production dimension -- real wall-clock burned in stage 5.

    Compared without the ceil rounding: `ceil` adds at most one step to
    each side, so an exact 5.0 ratio is not available from the returned
    integers.
    """
    fine = lock_steps_for_dim(65536, 8.0, dt=1e-3)
    coarse = lock_steps_for_dim(65536, 8.0, dt=5e-3)
    assert fine > coarse
    raw = 2.0 * np.log(65536) / 8.0
    assert abs(fine - np.ceil(raw / 1e-3)) <= 1
    assert abs(coarse - np.ceil(raw / 5e-3)) <= 1
    assert abs(fine / 5.0 - coarse) <= 1.0


def test_lock_steps_rejects_bad_args():
    with pytest.raises(ValueError):
        lock_steps_for_dim(1, 8.0)
    with pytest.raises(ValueError):
        lock_steps_for_dim(1024, 0.0)
    with pytest.raises(ValueError):
        lock_steps_for_dim(1024, 8.0, dt=0.0)
    with pytest.raises(ValueError):
        lock_steps_for_dim(1024, 8.0, factor=0.0)


def test_predicted_lock_steps_are_sufficient():
    """The prediction must be SUFFICIENT, not merely monotone.

    Runs the real integrator for the predicted step count and checks the
    field crosses the lock threshold. A prediction that is too small
    fails here, which is the only property that matters.
    """
    cfg = WaveConfig(dim=4096)
    rng = np.random.default_rng(11)
    wave = WaveState.random(cfg, rng=rng)
    state = KuramotoState.from_wave(wave, rng=rng)
    steps = lock_steps_for_dim(cfg.dim, 8.0, dt=5e-3)
    locked, _ = lock(state, coupling=8.0, dt=5e-3, steps=steps)
    r = order_parameter(locked.phases)[0]
    assert r >= LOCK_THRESHOLD, (steps, r)
