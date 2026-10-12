"""Slow full-dimension smoke test. Not part of the default gate sweep."""

from __future__ import annotations

import numpy as np
import pytest

from henri.config import WaveConfig
from henri.koopman import KoopmanOperator
from henri.state import WaveState

pytestmark = pytest.mark.slow


def test_full_dimension_state_is_unit_norm():
    cfg = WaveConfig()
    assert cfg.dim == 65536
    state = WaveState.random(cfg, seed=1)
    assert state.unit_norm_error() < 1e-12


def test_full_dimension_slot_rule():
    cfg = WaveConfig()
    state = WaveState.random(cfg, seed=2)
    state.zero_slot(3)
    assert np.all(state.psi[cfg.slot_slice(3)] == 0.0)
    assert state.slot_bleed([0, 1, 2]) < 1e-30


def test_full_dimension_koopman_apply_completes():
    cfg = WaveConfig()
    op = KoopmanOperator.build(cfg, seed=3)
    state = WaveState.random(cfg, seed=4)
    out = op.step(state)
    assert out.unit_norm_error() < 1e-12
    assert out.psi.shape == (65536,)


def test_full_dimension_koopman_drift_is_measured():
    cfg = WaveConfig()
    op = KoopmanOperator.build(cfg, seed=5)
    state = WaveState.random(cfg, seed=6)
    drift = op.norm_drift(state)
    assert np.isfinite(drift)
