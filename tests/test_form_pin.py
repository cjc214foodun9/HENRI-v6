"""Pin the Sagnac statistic FORM, and pin the sparsity defect it exposes.

`henri/form_pin.py` claims a structural fact: `form_phase` divides real
phase stress by D while summing it over active dimensions only, because a
zero dimension reads `cos(angle(0)) == 1`. Constraint C6 mandates exactly
zeroed unused slots, so the defect is reachable in production.

This file asserts the mechanism, the closed form, the gate report, and a
positive control that must FAIL to fire on a dense state.
"""

from __future__ import annotations

import numpy as np
import pytest

from henri import form_pin
from henri.config import WaveConfig
from henri.sagnac_homodyne_gate import SagnacHomodyneGate, PassReceipt, VetoReceipt
from henri.state import WaveState

TAU = form_pin.TAU_SPEC


def _phasors(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q = rng.integers(0, 256, size=n)
    return np.exp(2j * np.pi * q / 256.0)


def c6_state(cfg: WaveConfig, eps: float, seed: int, active_slots: int = 1):
    """A C6-style state: `active_slots` populated, the rest EXACTLY zero."""
    span = int(cfg.slot_span)
    base = _phasors(span, seed) / np.sqrt(span)
    z = np.random.default_rng(seed + 1).standard_normal(span)
    z = z * np.sqrt(span) / np.linalg.norm(z)
    corrupted = base * np.exp(1j * eps * z)
    psi = np.zeros(cfg.dim, dtype=np.complex128)
    for k in range(active_slots):
        psi[cfg.slot_slice(k)] = corrupted if k == 0 else base
    axiom = np.zeros(cfg.dim, dtype=np.complex128)
    for k in range(active_slots):
        axiom[cfg.slot_slice(k)] = base
    return (WaveState.from_complex(psi, cfg),
            WaveState.from_complex(axiom, cfg),
            corrupted, base)


def mean_forms(cfg, eps, trials=48, active_slots=1):
    f1, f2, ratio = [], [], []
    for i in range(trials):
        cand, axiom, _, _ = c6_state(cfg, eps, seed=9000 + i,
                                     active_slots=active_slots)
        a = form_pin.form_phase(cand.psi, axiom.psi)
        b = form_pin.form_inner_product(cand.psi, axiom.psi)
        f1.append(a); f2.append(b)
        ratio.append(b / a if a > 0 else float("inf"))
    return float(np.mean(f1)), float(np.mean(f2)), float(np.mean(ratio))


# --------------------------------------------------------------- registry
def test_measured_table_is_pinned_to_the_receipt():
    """The seven families and their measured values cannot drift silently."""
    assert set(form_pin.MEASURED) == {
        "unit_modulus", "qfhrr_token", "qfhrr_family",
        "stiefel", "learned", "sensor", "gaussian"}
    assert form_pin.MEASURED["sensor"] == pytest.approx(1.231e-01, abs=1e-4)
    assert form_pin.MEASURED["unit_modulus"] < 1e-12
    # sensor is the only family whose curve maximum exceeds tau
    decisive = [f for f in form_pin.MEASURED if form_pin.is_decision_decisive(f)]
    assert decisive == ["sensor"], decisive
    assert form_pin.MEASURED["sensor"] / TAU > 2.8


def test_unknown_family_fails_closed():
    with pytest.raises(form_pin.UnknownFamilyError):
        form_pin.measured_divergence("no_such_family")
    with pytest.raises(form_pin.UnknownFamilyError):
        form_pin.require_form(None, None, "no_such_family")


def test_phase_form_is_refused_on_a_sparse_state():
    cfg = WaveConfig(dim=512)
    cand, _, _, _ = c6_state(cfg, 0.0, seed=3, active_slots=1)
    assert form_pin.dilution_factor(cand.psi) == pytest.approx(4.0, rel=1e-12)
    with pytest.raises(form_pin.UnpinnedFormError):
        form_pin.require_form(form_pin.DENSE_FORM, cand.psi)
    assert form_pin.require_form(form_pin.INVARIANT_FORM, cand.psi) \
        == form_pin.INVARIANT_FORM
    # a dense state passes the same gate
    dense = _phasors(cfg.dim, 5) / np.sqrt(cfg.dim)
    assert form_pin.require_form(form_pin.DENSE_FORM, dense) == form_pin.DENSE_FORM


# --------------------------------------------------------------- mechanism
def test_dilution_factor_is_exactly_four_for_one_active_slot():
    cfg = WaveConfig(dim=512)
    for eps in (0.15, 0.30, 0.45, 0.60):
        f1, f2, ratio = mean_forms(cfg, eps, active_slots=1)
        assert f1 > 0.0 and f2 > 0.0, (eps, f1, f2)
        assert ratio == pytest.approx(4.0, rel=0.05), (
            f"eps={eps}: measured f2/f1={ratio:.4f}, expected 4.0 for a "
            f"one-active-slot state; f1={f1:.6f} f2={f2:.6f}")


def test_crossings_match_the_closed_form_and_move_with_sparsity():
    """The band is closed-form: sqrt(2*tau*D/n) versus sqrt(2*tau)."""
    band = form_pin.decision_flip_band(np.zeros(65536), 16384)
    assert band["dilution"] == pytest.approx(4.0)
    assert band["eps_cross_inner"] == pytest.approx(0.2936, abs=1e-4)
    assert band["eps_cross_phase"] == pytest.approx(0.5872, abs=1e-4)
    assert band["flip_band_width"] == pytest.approx(0.2936, abs=1e-4)


def test_decision_flips_inside_the_predicted_band():
    """Mid-band the two forms must disagree about the veto, both ways."""
    cfg = WaveConfig(dim=512)
    inside = 0.40                      # strictly inside [0.2936, 0.5872]
    f1, f2, _ = mean_forms(cfg, inside, active_slots=1)
    assert f1 <= TAU < f2, (
        f"at eps={inside} the phase form reads {f1:.6f} and the invariant "
        f"form reads {f2:.6f}; the phase form must PASS and the invariant "
        f"form must VETO")

    # Below the inner crossing BOTH pass; above the phase crossing BOTH veto.
    lo1, lo2, _ = mean_forms(cfg, 0.20, active_slots=1)
    assert lo1 < TAU and lo2 < TAU, (lo1, lo2)
    hi1, hi2, _ = mean_forms(cfg, 0.75, active_slots=1)
    assert hi1 > TAU and hi2 > TAU, (hi1, hi2)


def test_control_dense_state_shows_no_dilution():
    """POSITIVE CONTROL. If this fires, the claim is not about sparsity.

    A dense state has every dimension non-zero, so the two forms must
    agree and the dilution factor must be exactly 1. A mechanism that
    also moved dense states would be a different (and wrong) claim.
    """
    cfg = WaveConfig(dim=512)
    base = _phasors(cfg.dim, 17) / np.sqrt(cfg.dim)
    z = np.random.default_rng(18).standard_normal(cfg.dim)
    z = z * np.sqrt(cfg.dim) / np.linalg.norm(z)
    cand = base * np.exp(1j * 0.30 * z)
    assert form_pin.dilution_factor(cand) == 1.0
    assert form_pin.form_phase(cand, base) == pytest.approx(
        form_pin.form_inner_product(cand, base), abs=1e-12)


# --------------------------------------------------------------- the gate
def test_gate_default_is_unchanged_on_dense_states():
    cfg = WaveConfig(dim=512)
    base = _phasors(cfg.dim, 21) / np.sqrt(cfg.dim)
    cand = WaveState.from_complex(base, cfg)
    axiom = WaveState.from_complex(_phasors(cfg.dim, 22) / np.sqrt(cfg.dim), cfg)
    gate = SagnacHomodyneGate()
    r = gate.evaluate_trajectory(cand, axiom)
    assert isinstance(r, (PassReceipt, VetoReceipt))
    assert gate.statistic == form_pin.DENSE_FORM
    d = r.as_dict()
    assert "dilution" not in d, "dense receipts must stay byte-identical"
    assert "statistic" not in d
    assert d["delta_sagnac"] == pytest.approx(
        gate.compute_phase_divergence(cand, axiom)[0], abs=1e-15)


def test_gate_reports_dilution_on_a_c6_state():
    cfg = WaveConfig(dim=512)
    cand, axiom, _, _ = c6_state(cfg, 0.0, seed=77, active_slots=1)
    gate = SagnacHomodyneGate()
    d = gate.evaluate_trajectory(cand, axiom).as_dict()
    assert d["dilution"] == pytest.approx(4.0, rel=1e-12)
    assert d["statistic"] == form_pin.DENSE_FORM


def test_gate_statistic_selection_changes_the_measured_value():
    cfg = WaveConfig(dim=512)
    cand, axiom, _, _ = c6_state(cfg, 0.40, seed=91, active_slots=1)
    dense_gate = SagnacHomodyneGate(statistic=form_pin.DENSE_FORM)
    inv_gate = SagnacHomodyneGate(statistic=form_pin.INVARIANT_FORM)
    a = dense_gate.evaluate_trajectory(cand, axiom).delta_sagnac
    b = inv_gate.evaluate_trajectory(cand, axiom).delta_sagnac
    assert b == pytest.approx(4.0 * a, rel=1e-9)
    assert inv_gate.statistic == form_pin.INVARIANT_FORM


def test_gate_rejects_an_unknown_statistic_name():
    with pytest.raises(form_pin.UnpinnedFormError):
        SagnacHomodyneGate(statistic="euclidean")
