"""Directive 1 pin: the Koopman transition is two-step and explicit.

Owner directive 2026-10-12: do not attempt an intrinsically unitary
`K = diag(m) + A S B^H`. Retain the two-step architecture -- compute the
low-rank interaction in the tangent space, then execute explicit polar
normalization back onto `S^{D-1}`.

These tests pin that the projection is present, explicit, and actually
load-bearing (i.e. the raw step is measurably non-unitary).
"""

from __future__ import annotations

import numpy as np
import pytest

from henri.config import WaveConfig
from henri.koopman import KoopmanOperator
from henri.state import WaveState


@pytest.fixture(scope="module")
def setup():
    cfg = WaveConfig.small(dim=512)
    op = KoopmanOperator.build(cfg, seed=20261012)
    st = WaveState.random(cfg, seed=20261012)
    return cfg, op, st


def test_step_always_returns_unit_norm(setup):
    """Every transition lands back on the sphere, over many steps."""
    cfg, op, st = setup
    current = st
    for _ in range(25):
        current = op.step(current)
        assert current.unit_norm_error() < cfg.norm_tol


def test_raw_transition_is_measurably_non_unitary(setup):
    """If the raw step were unitary the projection would be a no-op.

    A projection that never changes anything cannot be load-bearing.
    """
    cfg, op, st = setup
    drift = op.norm_drift(st)
    assert drift > 1e-6, f"raw drift {drift} too small to need projection"


def test_projection_is_exactly_the_normalization(setup):
    """`step` output equals normalizing the raw output. No hidden step."""
    cfg, op, st = setup
    raw = op.apply_raw(st)
    expected = raw / float(np.linalg.norm(raw))
    stepped = op.step(st)
    np.testing.assert_allclose(stepped.psi, expected, rtol=1e-12, atol=1e-12)


def test_drift_trace_shows_correction_every_step(setup):
    """Per-step raw drift is nonzero, so the projection acts each step."""
    cfg, op, st = setup
    trace = op.drift_trace(st, 12)
    assert trace.shape == (12,)
    assert np.all(trace > 0.0)
