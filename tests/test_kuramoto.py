"""Kuramoto tests: order parameter and locking behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from henri.kuramoto import (
    LOCK_THRESHOLD,
    KuramotoState,
    is_locked,
    kuramoto_step,
    lock,
    order_parameter,
    to_wave,
)
from henri.state import WaveState


def test_order_parameter_full_coherence():
    phases = np.zeros(64)
    r, _ = order_parameter(phases)
    assert abs(r - 1.0) < 1e-12


def test_order_parameter_uniform_phases_is_near_zero():
    phases = np.linspace(-np.pi, np.pi, 4096, endpoint=False)
    r, _ = order_parameter(phases)
    assert r < 0.01


def test_order_parameter_bounds():
    rng = np.random.default_rng(0)
    phases = rng.uniform(-np.pi, np.pi, 512)
    r, _ = order_parameter(phases)
    assert 0.0 <= r <= 1.0


def test_empty_phases_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        order_parameter(np.array([]))


def test_negative_coupling_rejected(cfg):
    state = KuramotoState.from_wave(WaveState.random(cfg, seed=1))
    with pytest.raises(ValueError, match="non-negative"):
        kuramoto_step(state, coupling=-1.0, dt=1e-3)


def test_nonpositive_dt_rejected(cfg):
    state = KuramotoState.from_wave(WaveState.random(cfg, seed=1))
    with pytest.raises(ValueError, match="positive"):
        kuramoto_step(state, coupling=1.0, dt=0.0)


def test_noise_requires_explicit_rng(cfg):
    state = KuramotoState.from_wave(WaveState.random(cfg, seed=1))
    with pytest.raises(ValueError, match="explicit rng"):
        kuramoto_step(state, coupling=1.0, dt=1e-3, noise=0.1)


def test_locking_raises_order_parameter(cfg):
    """Strong coupling must increase coherence. Reported, not assumed."""
    state = KuramotoState.from_wave(WaveState.random(cfg, seed=2))
    r_before, _ = order_parameter(state.phases)
    _, trace = lock(state, coupling=8.0, dt=1e-3, steps=400)
    assert trace[-1] > r_before


def test_zero_coupling_does_not_lock(cfg):
    state = KuramotoState.from_wave(WaveState.random(cfg, seed=2))
    _, trace = lock(state, coupling=0.0, dt=1e-3, steps=200)
    assert not is_locked(trace[-1])


def test_phases_stay_wrapped(cfg):
    state = KuramotoState.from_wave(WaveState.random(cfg, seed=2))
    out = kuramoto_step(state, coupling=10.0, dt=1e-2)
    assert np.all(out.phases >= -np.pi - 1e-9)
    assert np.all(out.phases <= np.pi + 1e-9)


def test_omega_is_preserved(cfg):
    state = KuramotoState.from_wave(WaveState.random(cfg, seed=2))
    out = kuramoto_step(state, coupling=1.0, dt=1e-3)
    assert np.allclose(state.omega, out.omega)


def test_to_wave_is_unit_norm(cfg):
    state = KuramotoState.from_wave(WaveState.random(cfg, seed=2))
    wave = to_wave(state, cfg)
    assert wave.unit_norm_error() < 1e-12


def test_lock_threshold_constant():
    assert LOCK_THRESHOLD == 0.880


def test_trace_shape_matches_steps(cfg):
    state = KuramotoState.from_wave(WaveState.random(cfg, seed=2))
    _, trace = lock(state, coupling=4.0, dt=1e-3, steps=17)
    assert trace.shape == (18,)


def test_noise_keeps_determinism_with_seed(cfg):
    state = KuramotoState.from_wave(WaveState.random(cfg, seed=2))
    a = kuramoto_step(state, 1.0, 1e-3, rng=np.random.default_rng(9), noise=0.5)
    b = kuramoto_step(state, 1.0, 1e-3, rng=np.random.default_rng(9), noise=0.5)
    assert np.allclose(a.phases, b.phases)
