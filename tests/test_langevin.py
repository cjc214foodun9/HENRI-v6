"""Viscoelastic kernel and GLE tests, including the inf-tau subtlety."""

from __future__ import annotations

import numpy as np
import pytest

from henri.langevin import (
    LangevinState,
    ViscoelasticKernel,
    constant_drift,
    fdt_consistency,
    gle_step,
    memory_force,
)
from henri.state import WaveState


def test_default_kernel_has_four_modes():
    k = ViscoelasticKernel.default()
    assert k.n_modes == 4
    assert np.isinf(k.taus[-1])


def test_weights_are_normalized():
    k = ViscoelasticKernel(taus=(1e-3, 2e-3), weights=(3.0, 1.0))
    assert abs(sum(k.weights) - 1.0) < 1e-12
    assert abs(k.weights[0] - 0.75) < 1e-12


def test_inf_mode_tracked_separately():
    """exp(-x/inf) = 1, so the axiomatic mode is constant, not decaying."""
    k = ViscoelasticKernel.default()
    assert abs(k.constant_weight - 0.1) < 1e-12
    assert abs(k.decaying_weight - 0.9) < 1e-12
    assert k.constant_weight + k.decaying_weight == pytest.approx(1.0)


def test_kernel_decays_for_finite_tau():
    k = ViscoelasticKernel(taus=(1e-3, float("inf")), weights=(0.5, 0.5))
    early = k.value(np.array([0.0]))[0]
    late = k.value(np.array([1.0]))[0]
    assert late < early
    # asymptote is exactly the constant mode's weight
    assert late == pytest.approx(0.5, abs=1e-12)


def test_mismatched_lengths_rejected():
    with pytest.raises(ValueError, match="align"):
        ViscoelasticKernel(taus=(1e-3,), weights=(0.5, 0.5))


def test_nonpositive_tau_rejected():
    with pytest.raises(ValueError, match="positive"):
        ViscoelasticKernel(taus=(0.0,), weights=(1.0,))


def test_kernel_integral_includes_inf_growth():
    k = ViscoelasticKernel.default()
    short = k.integrate(1e-3)
    long = k.integrate(1.0)
    assert long > short


def test_gle_step_requires_positive_dt(cfg):
    state = LangevinState(psi=WaveState.random(cfg, seed=1).psi, cfg=cfg)
    with pytest.raises(ValueError, match="positive"):
        gle_step(state, ViscoelasticKernel.default(), dt=0.0)


def test_temperature_requires_rng(cfg):
    state = LangevinState(psi=WaveState.random(cfg, seed=1).psi, cfg=cfg)
    with pytest.raises(ValueError, match="explicit rng"):
        gle_step(state, ViscoelasticKernel.default(), dt=1e-3, temperature=1.0)


def test_gle_step_preserves_shape(cfg):
    wave = WaveState.random(cfg, seed=1)
    state = LangevinState(psi=wave.psi, cfg=cfg)
    k = ViscoelasticKernel.default()
    out = gle_step(state, k, dt=1e-4, rng=np.random.default_rng(0))
    assert out.psi.shape == (cfg.dim,)


def test_history_grows_and_caps(cfg):
    wave = WaveState.random(cfg, seed=1)
    state = LangevinState(psi=wave.psi, cfg=cfg)
    k = ViscoelasticKernel(taus=(1e-3,), weights=(1.0,))
    for _ in range(20):
        state = gle_step(state, k, dt=1e-4, memory_cap=5, rng=np.random.default_rng(0))
    assert len(state.history) <= 6


def test_memory_force_zero_for_static_history(cfg):
    """A constant trajectory has zero derivative, so zero memory drag."""
    k = ViscoelasticKernel(taus=(1e-3,), weights=(1.0,))
    v = WaveState.random(cfg, seed=2).psi
    force = memory_force([v.copy(), v.copy()], dt=1e-4, kernel=k)
    assert np.allclose(force, 0.0)


def test_memory_force_nonzero_for_changing_history(cfg):
    k = ViscoelasticKernel(taus=(1e-3,), weights=(1.0,))
    a = WaveState.random(cfg, seed=3).psi
    b = WaveState.random(cfg, seed=4).psi
    force = memory_force([a, b], dt=1e-4, kernel=k)
    assert np.linalg.norm(force) > 0.0


def test_constant_drift_zero_when_unchanged(cfg):
    k = ViscoelasticKernel.default()
    v = WaveState.random(cfg, seed=5).psi
    drift = constant_drift([v.copy(), v.copy()], k)
    assert np.allclose(drift, 0.0)


def test_constant_drift_scales_with_weight(cfg):
    v0 = WaveState.random(cfg, seed=6).psi
    v1 = WaveState.random(cfg, seed=7).psi
    k_full = ViscoelasticKernel(taus=(float("inf"),), weights=(1.0,))
    k_none = ViscoelasticKernel(taus=(1e-3,), weights=(1.0,))
    full = np.linalg.norm(constant_drift([v0, v1], k_full))
    none = np.linalg.norm(constant_drift([v0, v1], k_none))
    assert full > 0.0
    assert none == 0.0


def test_fdt_consistency_matches_prediction(cfg):
    """The thermostat must match the kernel it claims (fluctuation-dissipation)."""
    k = ViscoelasticKernel.default()
    report = fdt_consistency(k, temperature=0.3, rng=np.random.default_rng(0), dt=1e-3)
    assert report["relative_error"] < 0.05, report


def test_fdt_is_reproducible():
    k = ViscoelasticKernel.default()
    a = fdt_consistency(k, 0.2, np.random.default_rng(7), dt=1e-3)
    b = fdt_consistency(k, 0.2, np.random.default_rng(7), dt=1e-3)
    assert a["measured"] == b["measured"]


def test_gle_noise_is_deterministic_with_seed(cfg):
    wave = WaveState.random(cfg, seed=8)
    k = ViscoelasticKernel.default()
    a = gle_step(
        LangevinState(psi=wave.psi, cfg=cfg), k, 1e-4,
        rng=np.random.default_rng(3), temperature=0.5,
    )
    b = gle_step(
        LangevinState(psi=wave.psi, cfg=cfg), k, 1e-4,
        rng=np.random.default_rng(3), temperature=0.5,
    )
    assert np.allclose(a.psi, b.psi)


def test_empty_history_memory_force_rejected():
    k = ViscoelasticKernel.default()
    with pytest.raises(ValueError, match="non-empty"):
        memory_force([], dt=1e-4, kernel=k)
