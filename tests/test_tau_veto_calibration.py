"""Pin the tau_veto transfer-curve crossing to its closed form.

This test converts a measurement into an invariant. `probe_tau_veto_transfer_curve.py`
measured, at D = 65536, that EVERY family crosses the spec threshold at
`eps = 0.2967`. That number is not fitted; it is analytic.

DERIVATION. Corrupt the axiom by a pure phase perturbation:

    candidate_d = axiom_d * exp(i * eps * z_d),   mean_d z_d^2 = 1

The per-dimension phase difference is `angle(candidate_d * conj(axiom_d))
= eps * z_d`, independent of `|axiom_d|`. So

    delta(eps) = 1 - mean_d cos(eps * z_d)

and for small eps, `cos(x) ~= 1 - x^2/2`, giving

    delta(eps) ~= eps^2 * mean_d(z_d^2) / 2 = eps^2 / 2

Setting `delta(eps) == tau`:

    eps_at_tau = sqrt(2 * tau) = sqrt(0.0862) = 0.2936

MEASURED at D=65536: 0.2967. Deviation 0.0031, which is the expected
small-eps truncation error, not a discrepancy.

WHY THIS TEST MATTERS. It pins three things at once:
  1. The threshold is FAMILY-INDEPENDENT: one tau covers every family.
  2. The crossing is set by the NOISE NORMALIZATION, not by the family.
     Normalizing total L2 norm instead of per-dimension power moves the
     crossing to sqrt(2*tau*D) and breaks this identity. That was a real
     defect in the first draft of the probe, and this test would have
     caught it.
  3. form1 and form2 agree EXACTLY on unit-modulus rows, which is why the
     family matters only off the sphere.

The negative control at the end asserts the forms DIVERGE off unit
modulus. Without it, the identity above would pass on a statistic that
ignored magnitudes entirely.

No GPU required. Runs in the default gate sweep.
"""

from __future__ import annotations

import numpy as np
import pytest

from henri.config import WaveConfig

TAU_SPEC = 0.0431
ANALYTIC_CROSSING = float(np.sqrt(2.0 * TAU_SPEC))  # 0.2936
MEASURED_CROSSING_D65536 = 0.2967


def _unit_modulus_rows(dim: int, k: int, seed: int) -> np.ndarray:
    """qFHRR-style rows: |z_d| = 1/sqrt(D) exactly."""
    rng = np.random.default_rng(seed)
    q = rng.integers(0, 256, size=(k, dim))
    return np.stack([
        np.exp(1j * 2 * np.pi * q[i] / 256.0) / np.sqrt(dim)
        for i in range(k)
    ])


def _form1(cand: np.ndarray, axiom: np.ndarray) -> float:
    """Spec form: 1 - mean_d cos(angle(cand_d * conj(axiom_d)))."""
    return float(1.0 - np.mean(np.cos(np.angle(cand * np.conj(axiom)))))


def _form2(cand: np.ndarray, axiom: np.ndarray) -> float:
    """Normalized-inner-product form: 1 - Re<c,a> / (|c| |a|)."""
    return float(1.0 - np.real(np.vdot(cand, axiom))
                 / (np.linalg.norm(cand) * np.linalg.norm(axiom)))


def _per_dim_phase_noise(shape, rng) -> np.ndarray:
    """Real noise with mean_d z_d^2 == 1 (unit per-dimension power)."""
    z = rng.standard_normal(shape)
    return z * np.sqrt(z.size) / np.linalg.norm(z)


def _crossing(axiom, rng, tau=TAU_SPEC, n_trials=24):
    """Interpolated eps where the mean delta first exceeds tau."""
    prev = None
    for eps in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.275, 0.30, 0.325,
                0.35, 0.40, 0.50, 0.75, 1.00):
        vals = [_form1(axiom * np.exp(1j * eps * _per_dim_phase_noise(
            axiom.shape, rng)), axiom) for _ in range(n_trials)]
        cur = float(np.mean(vals))
        if cur > tau:
            if prev is None:
                return float(eps)
            d0, d1 = prev[1], cur
            e0, e1 = prev[0], float(eps)
            if d1 == d0:
                return e1
            frac = (tau - d0) / (d1 - d0)
            return float(e0 + frac * (e1 - e0))
        prev = (float(eps), cur)
    return None


def test_crossing_matches_closed_form():
    """eps_at_tau == sqrt(2*tau), independent of family and of D."""
    cfg = WaveConfig(dim=512)
    dim = int(cfg.slot_span)
    assert dim > 8, "slot must be large enough for the noise normalization"

    rows = _unit_modulus_rows(dim, 3, seed=20261012)
    for j in range(rows.shape[0]):
        rng = np.random.default_rng(1000 + j)
        got = _crossing(rows[j], rng)
        assert got is not None, (
            "row %d never crossed tau within eps <= 1.0; the transfer "
            "curve is not tied to sqrt(2*tau)" % j)
        assert abs(got - ANALYTIC_CROSSING) < 0.01, (
            "row %d: crossing %.4f deviates from sqrt(2*tau)=%.4f by %.4f"
            % (j, got, ANALYTIC_CROSSING, abs(got - ANALYTIC_CROSSING)))

    # Our production-D measurement must also sit inside the same tolerance,
    # so the CI-scale test and the production receipt cannot drift apart.
    assert abs(MEASURED_CROSSING_D65536 - ANALYTIC_CROSSING) < 0.01, (
        "recorded D=65536 crossing %.4f no longer agrees with the closed "
        "form %.4f" % (MEASURED_CROSSING_D65536, ANALYTIC_CROSSING))


def test_noise_normalization_is_the_thing_under_test():
    """Total-L2 normalization moves the crossing by sqrt(D) -- pin it.

    The first draft of the probe normalized `||z||_2 = 1`, leaving
    per-dimension power 1/D and crossing sqrt(2*tau*D). Every family then
    read NEVER at production D. This asserts the two normalizations give
    measurably different answers, so the identity above cannot be an
    artifact of the noise draw.
    """
    cfg = WaveConfig(dim=512)
    dim = int(cfg.slot_span)
    axiom = _unit_modulus_rows(dim, 1, seed=7)[0]
    rng_a = np.random.default_rng(99)
    rng_b = np.random.default_rng(99)

    def noise_total_norm(shape, rng):
        z = rng.standard_normal(shape) + 0j
        return z / np.linalg.norm(z)

    def noise_per_dim(shape, rng):
        z = rng.standard_normal(shape)
        return z * np.sqrt(z.size) / np.linalg.norm(z)

    eps_fixed = 0.30
    d_perdim = float(np.mean([
        _form1(axiom * np.exp(1j * eps_fixed * noise_per_dim(axiom.shape, rng_a)),
               axiom) for _ in range(12)]))
    d_totalnorm = float(np.mean([
        _form1(axiom * np.exp(1j * eps_fixed * noise_total_norm(axiom.shape, rng_b)),
               axiom) for _ in range(12)]))

    assert d_perdim > TAU_SPEC, (
        "per-dimension noise at eps=%.2f gave delta %.6f, below tau %.4f"
        % (eps_fixed, d_perdim, TAU_SPEC))
    assert d_totalnorm < TAU_SPEC, (
        "total-norm noise at eps=%.2f gave delta %.6f, which should sit far "
        "below tau because per-dimension corruption is eps/sqrt(D)"
        % (eps_fixed, d_totalnorm))
    assert d_perdim > 10.0 * d_totalnorm, (
        "the two normalizations must differ sharply; got %.6f vs %.6f"
        % (d_perdim, d_totalnorm))


def test_forms_agree_exactly_on_unit_modulus_and_diverge_off_it():
    """The scoping claim the calibration rests on, asserted both ways."""
    cfg = WaveConfig(dim=512)
    dim = int(cfg.slot_span)
    rows = _unit_modulus_rows(dim, 4, seed=31)

    # POSITIVE: unit modulus -> the two forms are the same statistic.
    worst = 0.0
    for i in range(rows.shape[0]):
        for j in range(rows.shape[0]):
            if i == j:
                continue
            worst = max(worst, abs(_form1(rows[i], rows[j])
                                   - _form2(rows[i], rows[j])))
    assert worst < 1e-12, (
        "unit-modulus forms must agree to 1e-12; worst deviation %.3e"
        % worst)

    # NEGATIVE CONTROL: off the sphere the forms must diverge. If this
    # ever stops holding, the scoping caveat is wrong and the calibration
    # guidance must be retracted.
    # A uniform positive rescale CANNOT separate the forms. form1 reads
    # only the per-dimension PHASE, and form2 normalizes by ||c|| ||a||,
    # so both are invariant to multiplying every row by the same constant.
    # My first control did exactly that (x1.6 everywhere) and measured
    # divergence 0.000e+00 -- a vacuous control that could never fire.
    # What DOES separate them is a non-uniform PER-DIMENSION magnitude
    # profile: form2 weights dimension d by |c_d||a_d|, while form1
    # weights every dimension equally.
    taper = np.linspace(0.2, 2.0, dim)
    off = rows * taper[None, :]
    worst_off = 0.0
    for i in range(off.shape[0]):
        for j in range(off.shape[0]):
            if i == j:
                continue
            worst_off = max(worst_off, abs(_form1(off[i], off[j])
                                           - _form2(off[i], off[j])))
    assert worst_off > 1e-6, (
        "off-sphere rows must show a measurable form divergence; got %.3e. "
        "The scoping claim would be vacuous." % worst_off)
