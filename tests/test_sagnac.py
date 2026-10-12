"""Sagnac veto tests. The C2 calibration trap is the centrepiece."""

from __future__ import annotations

import numpy as np
import pytest

from henri.sagnac import (
    calibrate_epsilon,
    confidence,
    sagnac_delta,
    separation,
    veto,
)
from henri.state import WaveState


def test_delta_identical_is_zero(wave):
    assert abs(sagnac_delta(wave, wave)) < 1e-12


def test_delta_orthogonal_is_one(cfg):
    a = WaveState.from_complex(
        np.concatenate([np.ones(256), np.zeros(256)]).astype(np.complex128), cfg
    )
    b = WaveState.from_complex(
        np.concatenate([np.zeros(256), np.ones(256)]).astype(np.complex128), cfg
    )
    assert abs(sagnac_delta(a, b) - 1.0) < 1e-12


def test_delta_is_antisymmetric_in_range(wave, rng):
    other = WaveState.random(wave.cfg, rng=rng)
    d1 = sagnac_delta(wave, other)
    d2 = sagnac_delta(other, wave)
    assert abs(d1 - d2) < 1e-12
    assert 0.0 <= d1 <= 2.0


def test_delta_out_of_phase_approaches_two(cfg):
    state = WaveState.random(cfg, seed=1)
    flipped = WaveState.from_complex(-state.psi, cfg)
    # Re<psi, -psi> = -1 => delta = 2
    assert abs(sagnac_delta(flipped, state) - 2.0) < 1e-12


def test_dimension_mismatch_rejected(cfg, rng):
    from henri.config import WaveConfig
    other = WaveConfig.small(dim=256)
    with pytest.raises(ValueError, match="dimension mismatch"):
        sagnac_delta(WaveState.random(cfg, rng=rng), WaveState.random(other, rng=rng))


def test_calibration_needs_two_waves(cfg):
    with pytest.raises(ValueError, match="at least two"):
        calibrate_epsilon([WaveState.random(cfg, seed=1)])


def test_calibration_sigma_must_be_positive(cfg, rng):
    refs = [WaveState.random(cfg, rng=rng) for _ in range(4)]
    with pytest.raises(ValueError, match="sigma"):
        calibrate_epsilon(refs, sigma=0.0)


def test_epsilon_is_never_a_literal(wave_real, rng):
    """Constraint C2: epsilon comes from measurement, not from 0.35."""
    refs = [WaveState.random(wave_real.cfg, rng=rng) for _ in range(16)]
    cal = calibrate_epsilon(refs, sigma=3.0, ceiling=0.35, rng=rng)
    assert cal.epsilon != 0.35 or cal.capped
    assert cal.n_pairs > 0
    assert np.isfinite(cal.baseline_mean)


def test_real_family_has_higher_chance_spread(cfg, rng):
    """The family gap that broke the fixed threshold (constraint C2).

    NOTE on the statistic. The retired repo quotes chance coherence
    0.009792 (complex) and 0.120168 (real), a 12.3x gap. Those appear
    to be ABSOLUTE phase-overlap figures. This build's `baseline_mean`
    is the SIGNED cosine, whose mean is ~0 for both families
    (measured -0.003 complex, +0.001 real) -- so the signed mean does
    NOT separate them and must not be asserted as if it did.

    What actually separates the families here is the SPREAD and the
    resulting threshold:

        complex  std 0.0299  epsilon 0.0870
        real     std 0.0426  epsilon 0.1252

    The assertion uses those measured quantities.
    """
    complex_refs = [WaveState.random(cfg, rng=rng) for _ in range(24)]
    real_refs = [WaveState.real(cfg, rng=rng) for _ in range(24)]
    cal_c = calibrate_epsilon(complex_refs, rng=rng, n_pairs=6000)
    cal_r = calibrate_epsilon(real_refs, rng=rng, n_pairs=6000)

    assert cal_r.baseline_std > cal_c.baseline_std
    assert cal_r.epsilon > cal_c.epsilon
    # Signed means are both near zero; pin that so it is not
    # mistaken for a separating statistic later.
    assert abs(cal_c.baseline_mean) < 0.05
    assert abs(cal_r.baseline_mean) < 0.05


def test_calibration_reports_family(wave_real, rng):
    refs = [WaveState.real(wave_real.cfg, rng=rng) for _ in range(8)]
    cal = calibrate_epsilon(refs, rng=rng)
    assert cal.meta["is_complex_family"] is False


def test_ceiling_caps_epsilon(cfg, rng):
    refs = [WaveState.random(cfg, rng=rng) for _ in range(8)]
    cal = calibrate_epsilon(refs, sigma=50.0, ceiling=0.35, rng=rng)
    assert cal.epsilon <= 0.35
    assert cal.capped


def test_calibration_is_reproducible(cfg):
    refs = [WaveState.random(cfg, seed=i) for i in range(12)]
    a = calibrate_epsilon(refs, rng=np.random.default_rng(1), n_pairs=500)
    b = calibrate_epsilon(refs, rng=np.random.default_rng(1), n_pairs=500)
    assert a.epsilon == b.epsilon


def test_veto_separation_on_clean_fixture(cfg, rng):
    """Directive 3 bar: separation >= 0.70, false veto <= 0.05.

    Gold pairs are an axiom perturbed by small phase noise. Bad pairs
    are independent random draws from the same family.
    """
    axioms = [WaveState.random(cfg, rng=rng) for _ in range(20)]
    cal = calibrate_epsilon(axioms, sigma=3.0, ceiling=0.35, rng=rng, n_pairs=4000)

    noise = 0.002
    gold = []
    for ax in axioms:
        pert = ax.psi + noise * (
            rng.standard_normal(cfg.dim) + 1j * rng.standard_normal(cfg.dim)
        )
        gold.append((WaveState.from_complex(pert, cfg), ax))

    bad = []
    for ax in axioms:
        bad.append((WaveState.random(cfg, rng=rng), ax))

    stats = separation(gold, bad, cal)
    assert stats["false_veto"] <= 0.05, stats
    assert stats["true_veto"] >= 0.90, stats
    assert stats["separation"] >= 0.70, stats


def test_veto_fires_on_phase_inversion(cfg, rng):
    axiom = WaveState.random(cfg, rng=rng)
    flipped = WaveState.from_complex(-axiom.psi, cfg)
    cal = calibrate_epsilon(
        [WaveState.random(cfg, rng=rng) for _ in range(12)], rng=rng
    )
    assert veto(flipped, axiom, cal)


def test_confidence_bounds(wave, rng):
    other = WaveState.random(wave.cfg, rng=rng)
    c = confidence(wave, other)
    assert 0.0 <= c <= 1.0
    assert abs(confidence(wave, wave) - 1.0) < 1e-12


def test_nan_on_empty_pair_sets(cfg, rng):
    cal = calibrate_epsilon([WaveState.random(cfg, rng=rng) for _ in range(6)], rng=rng)
    stats = separation([], [], cal)
    assert np.isnan(stats["true_veto"])
    assert np.isnan(stats["separation"])
