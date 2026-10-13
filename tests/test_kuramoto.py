"""Kuramoto tests: order parameter and locking behaviour.

DETERMINISM CONTRACT. `KuramotoState.from_wave` requires an explicit
`rng` or `seed`. `from_wave` with neither raises, because
`default_rng(None)` seeds from OS entropy. Every construction below
passes a seed.

DEFECT FOUND AND FIXED (recorded, not hidden). The locking test
originally built its state without a seed AND used only 400 steps. The
module's own `lock_steps_for_dim(512, 8.0, 1e-3)` predicts 1560 steps,
so the fixture ran at 0.26x budget. Combined with the unseeded omega,
the assertion `trace[-1] > r_before` became a coin flip that failed
once in a full-suite run and passed in isolation. Both causes are fixed
here; the assertion itself is UNCHANGED.
"""

from __future__ import annotations

import numpy as np
import pytest

from henri.kuramoto import (
    LOCK_THRESHOLD,
    KuramotoState,
    is_locked,
    kuramoto_step,
    lock,
    lock_steps_for_dim,
    order_parameter,
    to_wave,
)
from henri.state import WaveState

SEED = 2
"""Explicit omega seed for every construction in this module."""


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


# -- determinism contract ---------------------------------------------


def test_from_wave_requires_explicit_rng_or_seed(cfg):
    """Fail closed. `default_rng(None)` seeds from OS entropy."""
    with pytest.raises(ValueError, match="determinism"):
        KuramotoState.from_wave(WaveState.random(cfg, seed=SEED))


def test_from_wave_same_seed_gives_identical_omega(cfg):
    """Two calls with the same seed must agree bit for bit."""
    wave = WaveState.random(cfg, seed=SEED)
    a = KuramotoState.from_wave(wave, seed=SEED)
    b = KuramotoState.from_wave(wave, seed=SEED)
    assert np.array_equal(a.omega, b.omega)
    assert np.array_equal(a.phases, b.phases)


def test_from_wave_rng_and_seed_paths_agree(cfg):
    wave = WaveState.random(cfg, seed=SEED)
    a = KuramotoState.from_wave(wave, seed=7)
    b = KuramotoState.from_wave(wave, rng=np.random.default_rng(7))
    assert np.array_equal(a.omega, b.omega)


# -- validation --------------------------------------------------------


def test_negative_coupling_rejected(cfg):
    state = KuramotoState.from_wave(WaveState.random(cfg, seed=1), seed=SEED)
    with pytest.raises(ValueError, match="non-negative"):
        kuramoto_step(state, coupling=-1.0, dt=1e-3)


def test_nonpositive_dt_rejected(cfg):
    state = KuramotoState.from_wave(WaveState.random(cfg, seed=1), seed=SEED)
    with pytest.raises(ValueError, match="positive"):
        kuramoto_step(state, coupling=1.0, dt=0.0)


def test_noise_requires_explicit_rng(cfg):
    state = KuramotoState.from_wave(WaveState.random(cfg, seed=1), seed=SEED)
    with pytest.raises(ValueError, match="explicit rng"):
        kuramoto_step(state, coupling=1.0, dt=1e-3, noise=0.1)


# -- locking -----------------------------------------------------------


def test_locking_raises_order_parameter(cfg):
    """Strong coupling must increase coherence. Reported, not assumed.

    The step budget comes from the module's own scaling law, not a
    hand-picked constant. `lock_steps_for_dim` predicts 1560 steps at
    `dim=512, K=8, dt=1e-3`; the earlier fixture used 400 and could fail.
    """
    state = KuramotoState.from_wave(WaveState.random(cfg, seed=SEED), seed=SEED)
    r_before, _ = order_parameter(state.phases)
    steps = lock_steps_for_dim(cfg.dim, coupling=8.0, dt=1e-3, factor=1.5)
    _, trace = lock(state, coupling=8.0, dt=1e-3, steps=steps)
    assert trace[-1] > r_before, (
        f"r did not rise: r_before={r_before:.6f} r_after={trace[-1]:.6f} "
        f"steps={steps}"
    )


def test_locking_is_reproducible_across_seeds(cfg):
    """Every seed in a small sweep must show the rise. Reported."""
    wave = WaveState.random(cfg, seed=SEED)
    steps = lock_steps_for_dim(cfg.dim, coupling=8.0, dt=1e-3, factor=1.5)
    for s in (0, 1, 2, 3, 7, 11, 42):
        state = KuramotoState.from_wave(wave, seed=s)
        r_before, _ = order_parameter(state.phases)
        _, trace = lock(state, coupling=8.0, dt=1e-3, steps=steps)
        assert trace[-1] > r_before, f"seed {s}: {r_before:.6f} -> {trace[-1]:.6f}"


def test_zero_coupling_does_not_lock(cfg):
    state = KuramotoState.from_wave(WaveState.random(cfg, seed=SEED), seed=SEED)
    _, trace = lock(state, coupling=0.0, dt=1e-3, steps=200)
    assert not is_locked(trace[-1])


def test_phases_stay_wrapped(cfg):
    state = KuramotoState.from_wave(WaveState.random(cfg, seed=SEED), seed=SEED)
    out = kuramoto_step(state, coupling=10.0, dt=1e-2)
    assert np.all(out.phases >= -np.pi - 1e-9)
    assert np.all(out.phases <= np.pi + 1e-9)


def test_omega_is_preserved(cfg):
    state = KuramotoState.from_wave(WaveState.random(cfg, seed=SEED), seed=SEED)
    out = kuramoto_step(state, coupling=1.0, dt=1e-3)
    assert np.allclose(state.omega, out.omega)


def test_to_wave_is_unit_norm(cfg):
    state = KuramotoState.from_wave(WaveState.random(cfg, seed=SEED), seed=SEED)
    wave = to_wave(state, cfg)
    assert wave.unit_norm_error() < 1e-12


def test_lock_threshold_constant():
    assert LOCK_THRESHOLD == 0.880


def test_trace_shape_matches_steps(cfg):
    state = KuramotoState.from_wave(WaveState.random(cfg, seed=SEED), seed=SEED)
    _, trace = lock(state, coupling=4.0, dt=1e-3, steps=17)
    assert trace.shape == (18,)


def test_noise_keeps_determinism_with_seed(cfg):
    state = KuramotoState.from_wave(WaveState.random(cfg, seed=SEED), seed=SEED)
    a = kuramoto_step(state, 1.0, 1e-3, rng=np.random.default_rng(9), noise=0.5)
    b = kuramoto_step(state, 1.0, 1e-3, rng=np.random.default_rng(9), noise=0.5)
    assert np.allclose(a.phases, b.phases)


# -- scaling law -------------------------------------------------------


def test_lock_steps_scales_with_log_dim():
    """t_lock ~ 2 ln(D)/K. Doubling D adds a fixed increment."""
    a = lock_steps_for_dim(1024, coupling=8.0, dt=1e-3)
    b = lock_steps_for_dim(2048, coupling=8.0, dt=1e-3)
    gain = 2.0 * np.log(2.0) / 8.0 / 1e-3
    assert abs((b - a) - gain) <= 1.0


def test_lock_steps_shrinks_with_coupling():
    weak = lock_steps_for_dim(512, coupling=4.0, dt=1e-3, factor=1.5)
    strong = lock_steps_for_dim(512, coupling=16.0, dt=1e-3, factor=1.5)
    assert strong < weak
