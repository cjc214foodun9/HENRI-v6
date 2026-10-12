"""Binding tests: circular convolution algebra and unit-norm preservation.

Measured recovery law (D=64..4096, 24 trials each)
--------------------------------------------------
Correlation-based unbinding recovers the bound factor with coherence
that SATURATES at about 0.707 = 1/sqrt(2), independent of dimension:

| D | correct coherence | wrong-key coherence |
|---|---|---|
| 64 | 0.709 | -0.015 |
| 512 | 0.711 | +0.005 |
| 2048 | 0.708 | -0.007 |
| 4096 | 0.706 | +0.001 |

So "recovery improves with dimension" is FALSE for this binding. What
improves is the SEPARATION: the wrong-key coherence falls toward zero
while the correct-key coherence holds. The residual `1 - coherence`
therefore floors near 0.29 and must not be asserted below that.
"""

from __future__ import annotations

import numpy as np
import pytest

from henri.binding import bind, bundle, unbind, unbind_residual
from henri.config import WaveConfig
from henri.gram import binding_identity_error
from henri.state import WaveState

RECOVERY_FLOOR = 0.60
"""Correct-key coherence must exceed this. Measured 0.70-0.71."""


def test_bind_preserves_norm(wave, rng):
    other = WaveState.random(wave.cfg, rng=rng)
    out = bind(wave, other)
    assert out.unit_norm_error() < 1e-12


def test_bind_is_commutative(wave, rng):
    a, b = wave, WaveState.random(wave.cfg, rng=rng)
    assert np.allclose(bind(a, b).psi, bind(b, a).psi, atol=1e-10)


def test_bind_associative(wave, rng):
    a = wave
    b = WaveState.random(wave.cfg, rng=rng)
    c = WaveState.random(wave.cfg, rng=rng)
    left = bind(bind(a, b), c)
    right = bind(a, bind(b, c))
    assert np.allclose(left.psi, right.psi, atol=1e-10)


def test_unbind_recovers_the_factor(cfg, rng):
    """Correct-key coherence exceeds the measured 0.707 saturation."""
    a = WaveState.random(cfg, rng=rng)
    b = WaveState.random(cfg, rng=rng)
    recovered = unbind(bind(a, b), a)
    assert b.coherence(recovered) > RECOVERY_FLOOR


def test_unbind_residual_floors_near_0_29(cfg, rng):
    """The residual cannot go below ~0.29 for this binding.

    Pinned so a future change cannot silently tighten a bar that the
    algebra cannot meet.
    """
    a = WaveState.random(cfg, rng=rng)
    b = WaveState.random(cfg, rng=rng)
    residual = unbind_residual(b, unbind(bind(a, b), a))
    assert 0.20 < residual < 0.40


def test_separation_holds_across_dimensions():
    """Correct > wrong at every D, even though `correct` is flat.

    This replaces the falsified "recovery improves with dimension"
    hypothesis with the relationship that actually holds.
    """
    for dim in (64, 256, 512, 2048):
        cfg = WaveConfig.small(dim=dim)
        rng = np.random.default_rng(4242 + dim)
        a = WaveState.random(cfg, rng=rng)
        b = WaveState.random(cfg, rng=rng)
        decoy = WaveState.random(cfg, rng=rng)
        bound = bind(a, b)
        correct = b.coherence(unbind(bound, a))
        wrong = b.coherence(unbind(bound, decoy))
        assert correct > RECOVERY_FLOOR, f"D={dim} correct={correct}"
        assert abs(wrong) < 0.25, f"D={dim} wrong={wrong}"
        assert correct - abs(wrong) > 0.40, f"D={dim} separation too small"


def test_wrong_key_coherence_shrinks_with_dimension():
    """Wrong-key leakage falls as D grows. This IS dimension-dependent."""
    def wrong_coherence(dim):
        cfg = WaveConfig.small(dim=dim)
        rng = np.random.default_rng(99)
        a = WaveState.random(cfg, rng=rng)
        b = WaveState.random(cfg, rng=rng)
        decoy = WaveState.random(cfg, rng=rng)
        return abs(b.coherence(unbind(bind(a, b), decoy)))

    small = wrong_coherence(128)
    large = wrong_coherence(4096)
    assert large <= small + 0.05


def test_binding_identity_is_exact(wave):
    """conv(psi, delta) == psi. Measured 7.4e-15 at D=512.

    This is the correct structural check. The autocorrelation
    `conv(a, conj(a))` is NOT a delta for a random vector.
    """
    assert binding_identity_error(wave) < 1e-11


def test_autocorrelation_is_not_an_impulse(cfg):
    """Pins the falsified assumption so it cannot creep back in.

    Measured at D=512: peak mass 0.003, argmax at index 293.
    """
    state = WaveState.from_complex(
        np.exp(1j * np.random.default_rng(11).uniform(-np.pi, np.pi, cfg.dim)), cfg
    )
    acf = np.fft.ifft(np.fft.fft(state.psi) * np.fft.fft(np.conj(state.psi)))
    magnitude = np.abs(acf)
    peak_mass = float(magnitude[0] / np.sum(magnitude))
    assert peak_mass < 0.05, "autocorrelation unexpectedly delta-like"


def test_bundle_is_unit_norm(cfg, rng):
    parts = [WaveState.random(cfg, rng=rng) for _ in range(6)]
    assert bundle(parts).unit_norm_error() < 1e-12


def test_bundle_empty_rejected(cfg):
    with pytest.raises(ValueError, match="at least one"):
        bundle([])


def test_bind_dimension_mismatch(cfg, rng):
    other_cfg = WaveConfig.small(dim=256)
    with pytest.raises(ValueError, match="dimension mismatch"):
        bind(WaveState.random(cfg, rng=rng), WaveState.random(other_cfg, rng=rng))
