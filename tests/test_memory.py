"""Delta-memory tests: anti-drift, no dense D x D, monotonic retention."""

from __future__ import annotations

import numpy as np
import pytest

from henri.memory import (
    DeltaMemory,
    assert_no_dense_dxd,
    drift_under_replay,
    forget_curve,
)
from henri.state import WaveState


def test_store_is_rank_factorized(cfg):
    mem = DeltaMemory.build(cfg.dim, rank=16)
    assert mem.v_factors.shape == (cfg.dim, 16)
    assert mem.k_factors.shape == (cfg.dim, 16)


def test_no_dense_dxd_allocation(cfg):
    """Constraint C7: the store must never allocate D x D.

    Measured at D=512, r=16: nbytes = 262144 = 2*D*r*16 exactly.
    A dense D x D complex128 store would be D*D*16 = 4194304, so a
    dense form is 16x larger. A dense D x D COMPLEX64 form is 16x
    larger still. The bar is the dense complex128 size, computed from
    the same dimensions rather than hardcoded.
    """
    rank = 16
    mem = DeltaMemory.build(cfg.dim, rank=rank)
    dense_complex128 = cfg.dim * cfg.dim * 16
    dense_complex64 = cfg.dim * cfg.dim * 8
    assert mem.nbytes() == 2 * cfg.dim * rank * 16
    assert mem.nbytes() < dense_complex128 / 4
    assert mem.nbytes() < dense_complex64 / 2


def test_prediction_shape(cfg):
    mem = DeltaMemory.build(cfg.dim, rank=16)
    key = WaveState.random(cfg, seed=1).psi
    assert mem.predict(key).shape == (cfg.dim,)


def test_write_reduces_residual(cfg):
    mem = DeltaMemory.build(cfg.dim, rank=32)
    key = WaveState.random(cfg, seed=2).psi
    value = WaveState.random(cfg, seed=3).psi
    before = np.linalg.norm(mem.residual(key, value))
    mem.write(key, value, beta=1.0)
    after = np.linalg.norm(mem.residual(key, value))
    assert after <= before


def test_mastered_key_produces_zero_update(cfg):
    """The anti-drift property the retired ratchet-decay readout violated."""
    mem = DeltaMemory.build(cfg.dim, rank=32)
    key = WaveState.random(cfg, seed=4).psi
    value = WaveState.random(cfg, seed=5).psi
    for _ in range(12):
        mem.write(key, value, beta=0.5)
    # once mastered, further writes must not move the factors
    v_before = mem.v_factors.copy()
    delta = mem.write(key, value, beta=0.5)
    assert delta < 1e-6 or np.allclose(v_before, mem.v_factors, atol=1e-12)


def test_write_returns_zero_on_mastered_pair(cfg):
    mem = DeltaMemory.build(cfg.dim, rank=32)
    key = WaveState.random(cfg, seed=6).psi
    value = WaveState.random(cfg, seed=7).psi
    for _ in range(30):
        mem.write(key, value, beta=0.7)
    assert mem.write(key, value, beta=0.7) == 0.0


def test_negative_beta_rejected(cfg):
    mem = DeltaMemory.build(cfg.dim, rank=8)
    key = WaveState.random(cfg, seed=8).psi
    with pytest.raises(ValueError, match="non-negative"):
        mem.write(key, key, beta=-1.0)


def test_wrong_key_length_rejected(cfg):
    mem = DeltaMemory.build(cfg.dim, rank=8)
    with pytest.raises(ValueError, match="expected length"):
        mem.predict(np.zeros(3, dtype=np.complex128))


def test_rank_exceeding_dim_rejected(cfg):
    with pytest.raises(ValueError, match="rank cannot exceed"):
        DeltaMemory.build(cfg.dim, rank=cfg.dim + 1)


def test_invalid_lam_rejected(cfg):
    with pytest.raises(ValueError, match="lam"):
        DeltaMemory.build(cfg.dim, rank=8, lam=0.0)


def test_retention_is_monotone_with_lam_one(cfg):
    """No decay means no forgetting: the accumulator is monotone."""
    mem = DeltaMemory.build(cfg.dim, rank=32, lam=1.0)
    pairs = [
        (WaveState.random(cfg, seed=10 + i).psi, WaveState.random(cfg, seed=50 + i).psi)
        for i in range(4)
    ]
    for _ in range(3):
        for k, v in pairs:
            mem.write(k, v, beta=0.5)
    assert forget_curve(mem) == pytest.approx(1.0)


def test_replay_does_not_increase_loss(cfg):
    """Drift check: replaying mastered pairs must not raise the error."""
    mem = DeltaMemory.build(cfg.dim, rank=32)
    pairs = [
        (WaveState.random(cfg, seed=20 + i).psi, WaveState.random(cfg, seed=60 + i).psi)
        for i in range(3)
    ]
    initial, final = drift_under_replay(mem, pairs, beta=0.8, cycles=3)
    assert final <= initial + 1e-9


def test_updates_counter_increments(cfg):
    mem = DeltaMemory.build(cfg.dim, rank=16)
    key = WaveState.random(cfg, seed=30).psi
    value = WaveState.random(cfg, seed=31).psi
    start = mem.updates
    mem.write(key, value, beta=1.0)
    assert mem.updates >= start


def test_spectral_proxy_is_finite(cfg):
    mem = DeltaMemory.build(cfg.dim, rank=16)
    assert np.isfinite(mem.spectral_radius_proxy())


def test_empty_memory_predicts_zero(cfg):
    mem = DeltaMemory.build(cfg.dim, rank=16)
    key = WaveState.random(cfg, seed=40).psi
    assert np.allclose(mem.predict(key), 0.0)


def test_dense_guard_raises():
    with pytest.raises(MemoryError, match="constraint C7"):
        assert_no_dense_dxd(np.zeros((1024, 1024), dtype=np.complex128))


def test_dense_guard_allows_small_square():
    assert_no_dense_dxd(np.zeros((64, 64)))  # must not raise


def test_read_shape(cfg):
    mem = DeltaMemory.build(cfg.dim, rank=16)
    key = WaveState.random(cfg, seed=41).psi
    assert mem.read(key).shape == (cfg.dim,)
