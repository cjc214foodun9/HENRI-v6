"""State tests: norm invariant, slot rule, and the zero-pad constraint."""

from __future__ import annotations

import numpy as np
import pytest

from henri.config import SLOT_NAMES, WaveConfig
from henri.state import WaveState, slot_table


def test_random_state_is_unit_norm(wave):
    assert wave.settled
    assert wave.unit_norm_error() < 1e-12


def test_seed_reproduces_across_calls(cfg):
    a = WaveState.random(cfg, seed=1234)
    b = WaveState.random(cfg, seed=1234)
    assert np.allclose(a.psi, b.psi)
    c = WaveState.random(cfg, seed=99)
    assert not np.allclose(a.psi, c.psi)


def test_real_family_seed_reproduces(cfg):
    a = WaveState.real(cfg, seed=5)
    b = WaveState.real(cfg, seed=5)
    assert np.allclose(a.psi, b.psi)
    assert np.allclose(a.psi.imag, 0.0)


def test_normalize_is_idempotent(wave):
    first = wave.normalize().norm
    second = wave.normalize().norm
    assert abs(first - second) < 1e-15
    assert abs(second - 1.0) < 1e-12


def test_zero_vector_cannot_be_normalized(cfg):
    with pytest.raises(ValueError, match="zero or non-finite"):
        WaveState.from_complex(np.zeros(cfg.dim, dtype=np.complex128), cfg)


def test_dimension_mismatch_rejected(cfg):
    with pytest.raises(ValueError, match="length"):
        WaveState(np.zeros(7, dtype=np.complex128), cfg)


def test_slot_energy_sums_to_one(wave):
    assert abs(float(np.sum(wave.slot_energy())) - 1.0) < 1e-12


def test_zero_slot_removes_energy_exactly(cfg):
    """Constraint C6: unused slots are ZERO, never noise-filled."""
    state = WaveState.random(cfg, rng=np.random.default_rng(3))
    state.zero_slot(2)
    sl = cfg.slot_slice(2)
    assert np.all(state.psi[sl] == 0.0)
    energy = state.slot_energy()
    assert energy[2] < 1e-30


def test_put_slot_wrong_length_rejected(cfg):
    state = WaveState.zeros(cfg)
    with pytest.raises(ValueError, match="expects"):
        state.put_slot(0, np.zeros(3, dtype=np.complex128))


def test_slot_bleed_zero_for_single_slot(cfg):
    """A state living in one slot has zero bleed into the others."""
    state = WaveState.zeros(cfg)
    rng = np.random.default_rng(4)
    sl = cfg.slot_slice(1)
    span = sl.stop - sl.start
    state.put_slot(1, rng.standard_normal(span) + 1j * rng.standard_normal(span))
    state.normalize()
    assert state.slot_bleed([1]) < 1e-24


def test_coherence_self_is_one(wave):
    assert abs(wave.coherence(wave) - 1.0) < 1e-12


def test_coherence_is_symmetric(wave, rng):
    other = WaveState.random(wave.cfg, rng=rng)
    assert abs(wave.coherence(other) - other.coherence(wave)) < 1e-12


def test_copy_is_independent(wave):
    clone = wave.copy()
    clone.psi[0] += 1.0
    assert not np.allclose(wave.psi, clone.psi)


def test_real_roundtrip(cfg):
    state = WaveState.random(cfg, seed=8)
    restored = WaveState.from_real_blocks(state.to_real_blocks(), cfg)
    assert np.allclose(state.psi, restored.psi)


def test_real_blocks_wrong_length(cfg):
    with pytest.raises(ValueError, match="real scalars"):
        WaveState.from_real_blocks(np.zeros(5), cfg)


def test_slot_table_reports_four_rows(cfg):
    table = slot_table(cfg)
    assert len(table) == 4
    assert table[0]["role"] == SLOT_NAMES[0]
    assert table[-1]["stop"] == cfg.dim


def test_phases_shape(wave):
    assert wave.phases().shape == (wave.cfg.dim,)
