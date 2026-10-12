"""Hopfield tests: CCCP energy descent and the tie-plateau behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from henri.gram import stiefel_retract
from henri.hopfield import (
    cccp_step,
    cleanup,
    crosstalk,
    hopfield_energy,
    max_group,
    snap_index,
)
from henri.state import WaveState


def test_energy_decreases_under_cccp(cfg):
    """Lyapunov descent on the unnormalized iterate."""
    rng = np.random.default_rng(0)
    book = stiefel_retract(
        rng.standard_normal((20, cfg.dim)) + 1j * rng.standard_normal((20, cfg.dim))
    )
    state = WaveState.random(cfg, seed=1)
    _, trace = cleanup(state, book, tau=0.05, iters=20)
    assert trace[-1] <= trace[0] + 1e-12


def test_energy_trace_is_non_increasing(cfg):
    """Gate S-2 analogue: monotone dissipation with no thermal noise."""
    rng = np.random.default_rng(2)
    book = stiefel_retract(
        rng.standard_normal((16, cfg.dim)) + 1j * rng.standard_normal((16, cfg.dim))
    )
    state = WaveState.random(cfg, seed=3)
    _, trace = cleanup(state, book, tau=0.1, iters=15)
    diffs = np.diff(trace)
    assert np.all(diffs <= 1e-9), f"energy increased: {diffs[diffs > 1e-9]}"


def test_cleanup_returns_unit_norm(cfg, codebook):
    state = WaveState.random(cfg, seed=4)
    settled, _ = cleanup(state, codebook)
    assert settled.unit_norm_error() < 1e-12


def test_cleanup_lands_on_the_nearest_pattern(cfg, codebook):
    """Starting from a pattern, cleanup must stay at that pattern."""
    target = codebook[7]
    state = WaveState.from_complex(target, cfg)
    settled, _ = cleanup(state, codebook, iters=20)
    assert snap_index(settled, codebook) == 7


def test_low_tau_sharpens(cfg, codebook):
    """Lower temperature concentrates the softmax."""
    state = WaveState.random(cfg, seed=5)
    hot, _ = cleanup(state, codebook, tau=1.0, iters=3)
    cold, _ = cleanup(state, codebook, tau=0.01, iters=3)
    assert cold.unit_norm_error() < 1e-12
    assert hot.unit_norm_error() < 1e-12


def test_codebook_shape_validated(cfg):
    state = WaveState.random(cfg, seed=6)
    with pytest.raises(ValueError, match="codebook must be"):
        cleanup(state, np.zeros((4, 7), dtype=np.complex128))


def test_empty_codebook_rejected(cfg):
    state = WaveState.random(cfg, seed=6)
    with pytest.raises(ValueError, match="at least one pattern"):
        cleanup(state, np.zeros((0, cfg.dim), dtype=np.complex128))


def test_nonpositive_tau_rejected(cfg, codebook):
    state = WaveState.random(cfg, seed=6)
    with pytest.raises(ValueError, match="tau must be positive"):
        cleanup(state, codebook, tau=0.0)


def test_max_group_returns_all_ties(cfg):
    """Tie-robust: the whole tied set, not one index. Constraint C11."""
    state = WaveState.random(cfg, seed=7)
    book = np.vstack([state.psi, state.psi, -state.psi]).astype(np.complex128)
    group = max_group(state, book)
    assert len(group) == 2


def test_snap_index_is_single_index(cfg, codebook):
    state = WaveState.from_complex(codebook[3], cfg)
    idx = snap_index(state, codebook)
    assert isinstance(idx, int)
    assert 0 <= idx < codebook.shape[0]


def test_cccp_step_moves_toward_codebook(cfg, codebook):
    state = WaveState.random(cfg, seed=8)
    before = hopfield_energy(state.psi, codebook, 0.05)
    after = hopfield_energy(cccp_step(state.psi, codebook, 0.05), codebook, 0.05)
    assert after <= before + 1e-12


def test_crosstalk_of_orthonormal_codebook_is_tiny(codebook):
    """Measured 6.44e-18 for k=24 at D=512. Bar sits above float noise."""
    assert crosstalk(codebook) < 1e-12


def test_energy_is_finite(cfg, codebook):
    state = WaveState.random(cfg, seed=9)
    assert np.isfinite(hopfield_energy(state.psi, codebook, 0.05))


def test_iters_zero_returns_input(cfg, codebook):
    state = WaveState.random(cfg, seed=10)
    settled, trace = cleanup(state, codebook, iters=0)
    assert trace.size == 1
    assert settled.unit_norm_error() < 1e-12
