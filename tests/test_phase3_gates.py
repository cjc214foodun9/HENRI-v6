"""Phase 3 gate tests: G-ACT1, G-ACT2, G-ACT3, plus the scalar-rotor refusal.

Every gate keeps the SPEC's bar. Where the literal implementation failed
the bar, the test asserts BOTH the failing literal arm and the passing
repaired arm, so the difference stays measurable.

Not claimed here: G-ACT4 (50 microsecond cycle latency). That gate needs
the Blackwell target hardware. Local CPU measurement showed the veto
statistic ALONE costs 1739.8 microseconds at D=65536, so G-ACT4 is
BLOCKED, not passed.
"""

from __future__ import annotations

import numpy as np
import pytest

from henri.active_inference_loop import (
    ActiveInferenceController,
    ScalarRotorRejectedError,
    circular_correlation_delta,
    dimensional_phase_ramp,
    gauge_invariance_error,
    recover_score_delta,
    scalar_rotor,
)
from henri.anisotropic_thermostat import (
    AnisotropicLangevinThermostat,
    coherent_space_apply,
    null_space_apply,
)
from henri.config import WaveConfig
from henri.gram import stiefel_retract
from henri.hopfield import cleanup
from henri.sagnac import sagnac_delta
from henri.sagnac_homodyne_gate import (
    CONSTRUCTIVE,
    DESTRUCTIVE,
    TAU_VETO_CEILING,
    RecoilBuffer,
    SagnacHomodyneGate,
)
from henri.state import WaveState

SEPARATION_BAR = 0.7000
FALSE_VETO_BAR = 0.0500


def _unit_rows(cfg, k, seed):
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((k, cfg.dim)) + 1j * rng.standard_normal((k, cfg.dim))
    return raw / np.linalg.norm(raw, axis=1, keepdims=True)


# ---------------------------------------------------------------- G-ACT1 --


def _phasor_rows(cfg, k, seed):
    """Unit-MODULUS rows: |z_d| = 1/sqrt(D) exactly. The qFHRR family."""
    rng = np.random.default_rng(seed)
    q = rng.integers(0, 256, size=(k, cfg.dim))
    return np.exp(2j * np.pi * q / 256.0) / np.sqrt(cfg.dim)


def test_homodyne_statistic_equals_sagnac_delta_on_unit_modulus(cfg):
    """The spec's statistic is NOT new: it equals `sagnac_delta` EXACTLY
    on the unit-modulus family.

    Identity: `Re<c,a> = sum_d |c_d||a_d| cos(dtheta_d)`.
    With `|c_d| = |a_d| = 1/sqrt(D)` this is `(1/D) sum_d cos(dtheta_d)`,
    which is exactly the spec's mean-cosine form.

    Measured max difference 0.000e+00. Pinning it prevents a second
    phase-mismatch convention from drifting in.
    """
    rows = _phasor_rows(cfg, 6, seed=900)
    gate = SagnacHomodyneGate()
    worst = 0.0
    for i in range(len(rows)):
        for j in range(len(rows)):
            if i == j:
                continue
            a = WaveState.from_complex(rows[i], cfg)
            b = WaveState.from_complex(rows[j], cfg)
            phase_form, _ = gate.compute_phase_divergence(a, b)
            worst = max(worst, abs(phase_form - sagnac_delta(a, b)))
    assert worst < 1e-12, f"forms differ by {worst:.3e}; convention drift"


def test_equivalence_does_not_hold_off_unit_modulus(cfg):
    """SCOPE LIMIT, measured. The two statistics diverge by MORE than
    the veto threshold on the repo's own Stiefel codebook.

        unit_modulus   max|diff| = 0.000000e+00
        gaussian_norm  max|diff| = 3.423e-02
        stiefel        max|diff| = 5.610e-02
        tau_veto                  = 4.310e-02

    The phase form discards magnitude; the scalar form weights by
    `|c_d||a_d|`. Off the unit-modulus family the choice of statistic
    can therefore FLIP a veto decision. Recorded, not reconciled: the
    Phase 3 spec mandates the phase form, and this is the measured price.
    """
    gate = SagnacHomodyneGate()
    rng = np.random.default_rng(950)

    def gauss_pair():
        v = rng.standard_normal((2, cfg.dim)) + 1j * rng.standard_normal((2, cfg.dim))
        v = v / np.linalg.norm(v, axis=1, keepdims=True)
        return (WaveState.from_complex(v[0], cfg), WaveState.from_complex(v[1], cfg))

    raw = rng.standard_normal((6, cfg.dim)) + 1j * rng.standard_normal((6, cfg.dim))
    st = stiefel_retract(raw)

    def worst_for(pairs):
        w = 0.0
        for a, b in pairs:
            pf, _ = gate.compute_phase_divergence(a, b)
            w = max(w, abs(pf - sagnac_delta(a, b)))
        return w

    g_pairs = [gauss_pair() for _ in range(12)]
    s_pairs = [
        (WaveState.from_complex(st[i], cfg), WaveState.from_complex(st[j], cfg))
        for i in range(len(st))
        for j in range(len(st))
        if i != j
    ]
    w_gauss = worst_for(g_pairs)
    w_stiefel = worst_for(s_pairs)

    print(
        f"\nSCOPE: gaussian {w_gauss:.6e}  stiefel {w_stiefel:.6e}  "
        f"tau_veto {TAU_VETO_CEILING:.4f}"
    )
    # Both must be nonzero: the equivalence is a unit-modulus property.
    assert w_gauss > 1e-3
    assert w_stiefel > 1e-3
    # And the Stiefel divergence is comparable to the threshold itself.
    assert w_stiefel > TAU_VETO_CEILING * 0.5, (
        "Stiefel divergence unexpectedly small; the scope limit changed"
    )


def test_spec_factor_of_two_inconsistency_is_pinned(cfg):
    """The spec declares two expressions equal that differ by exactly 2.

    Section 3.1 writes both
        1/(2D) sum_d (1 - cos(dtheta_d))   and   1 - mean_d cos(dtheta_d)
    The first is HALF the second. Pinned rather than silently reconciled.
    """
    rows = _unit_rows(cfg, 2, seed=901)
    a = WaveState.from_complex(rows[0], cfg)
    b = WaveState.from_complex(rows[1], cfg)
    m = SagnacHomodyneGate.phase_mismatch(a, b)
    left = float(np.mean(1.0 - np.cos(m))) / 2.0
    right = 1.0 - float(np.mean(np.cos(m)))
    assert left > 0
    assert abs(right / left - 2.0) < 1e-9


def test_gate_accepts_identical_and_refuses_unrelated(cfg):
    rows = _unit_rows(cfg, 2, seed=902)
    a = WaveState.from_complex(rows[0], cfg)
    b = WaveState.from_complex(rows[1], cfg)
    gate = SagnacHomodyneGate()
    ok = gate.evaluate_trajectory(a, a, RecoilBuffer())
    assert ok.status == CONSTRUCTIVE
    assert ok.delta_sagnac == pytest.approx(0.0, abs=1e-12)
    bad = gate.evaluate_trajectory(a, b, RecoilBuffer())
    assert bad.status == DESTRUCTIVE
    assert bad.delta_sagnac > TAU_VETO_CEILING


def test_dark_port_energy_is_recorded_not_discarded(cfg):
    """A veto must leave its signature in the recoil buffer."""
    rows = _unit_rows(cfg, 2, seed=903)
    a = WaveState.from_complex(rows[0], cfg)
    b = WaveState.from_complex(rows[1], cfg)
    buf = RecoilBuffer()
    assert len(buf) == 0
    SagnacHomodyneGate().evaluate_trajectory(a, b, buf)
    assert len(buf) == 1
    assert buf.total_energy > 0.0
    assert buf.latest_comb() is not None
    assert buf.latest_comb().shape == (cfg.dim,)


def test_recoil_buffer_is_circular_at_capacity():
    buf = RecoilBuffer(capacity=4)
    for i in range(10):
        buf.push(float(i), np.ones(3))
    assert len(buf) == 4
    assert buf.dropped == 6
    assert buf.intensities == [6.0, 7.0, 8.0, 9.0]


def test_veto_refuses_post_cleanup_candidate(cfg):
    """The vacuous-gate guard, measured on the real cleanup operator."""
    rows = stiefel_retract(_unit_rows(cfg, 8, seed=904))
    cand = WaveState.from_complex(rows[0], cfg)
    assert cand.cleaned is False, "raw candidate must not be marked cleaned"

    settled, _ = cleanup(cand, rows)
    assert settled.cleaned is True, "cleanup must mark its output"

    with pytest.raises(ValueError, match="pre-cleanup|PRE-cleanup"):
        SagnacHomodyneGate().compute_phase_divergence(settled, cand)


def test_gact1_separation_and_false_veto_rate(cfg):
    """G-ACT1 as specified: separation >= 0.70, false veto <= 0.05.

    Valid trajectory  : the axiom with a SMALL relative perturbation, so
                        the candidate genuinely conforms to the invariant.
    Violating         : an unrelated draw, so it genuinely does not.

    If the spec's 0.0431 threshold cannot separate these, the gate FAILS
    and the measured numbers are reported rather than the bar moved.
    """
    rng = np.random.default_rng(20261012)
    gate = SagnacHomodyneGate()
    n = 400
    scale = 0.05

    fv = 0
    for _ in range(n):
        rows = _unit_rows(cfg, 1, seed=int(rng.integers(0, 10**6)))
        axiom = WaveState.from_complex(rows[0], cfg)
        noise = rng.standard_normal(cfg.dim) + 1j * rng.standard_normal(cfg.dim)
        cand = WaveState.from_complex(axiom.psi + scale * noise / np.sqrt(cfg.dim), cfg)
        if gate.veto_decision(cand, axiom):
            fv += 1

    hit = 0
    for _ in range(n):
        pair = _unit_rows(cfg, 2, seed=int(rng.integers(0, 10**6)))
        axiom = WaveState.from_complex(pair[0], cfg)
        viol = WaveState.from_complex(pair[1], cfg)
        if gate.veto_decision(viol, axiom):
            hit += 1

    p_veto_violating = hit / n
    p_veto_valid = fv / n
    separation = p_veto_violating - p_veto_valid

    print(
        f"\nG-ACT1  P(veto|violating)={p_veto_violating:.4f}  "
        f"P(veto|valid)={p_veto_valid:.4f}  separation={separation:.4f}"
    )
    assert separation >= SEPARATION_BAR, (
        f"separation {separation:.4f} below bar {SEPARATION_BAR}"
    )
    assert p_veto_valid <= FALSE_VETO_BAR, (
        f"false veto rate {p_veto_valid:.4f} above bar {FALSE_VETO_BAR}"
    )


# ---------------------------------------------------------------- G-ACT2 --


def test_null_space_apply_never_forms_dense_dxd(cfg):
    """Constraint C7: the projector stays dual."""
    v = _unit_rows(cfg, 4, seed=910).conj().T          # (D, k)
    x = np.ones((cfg.dim, 3), dtype=np.complex128)
    out = null_space_apply(v, x)
    assert out.shape == x.shape


def test_null_space_apply_is_idempotent_on_its_own_output(cfg):
    v = _unit_rows(cfg, 3, seed=911).conj().T
    x = np.ones((cfg.dim, 2), dtype=np.complex128)
    once = null_space_apply(v, x)
    twice = null_space_apply(v, once)
    assert np.allclose(once, twice, atol=1e-9)


def test_coherent_projection_kills_null_component(cfg):
    """`V^H P_null = 0` is the identity that makes G-ACT2 clause 1 work."""
    v = _unit_rows(cfg, 3, seed=912).conj().T
    x = np.ones((cfg.dim, 2), dtype=np.complex128)
    pn = null_space_apply(v, x)
    assert np.allclose(v.conj().T @ pn, 0.0, atol=1e-9)


def test_projection_is_exact_before_clamping(cfg):
    """`P_coherent @ (P_null T) == 0` exactly, and T_raw is positive.

    The projection itself is correct. The leak reported by the thermostat
    comes from the CLAMP, tested next.
    """
    v = _unit_rows(cfg, 3, seed=914).conj().T
    comb = np.abs(np.random.default_rng(1).standard_normal(cfg.dim))
    t_raw = 0.01 + 1.0 * (1.0 - np.exp(-comb))
    assert t_raw.min() >= 0.0, "unprojected temperature must be positive"

    pn = null_space_apply(v, t_raw)
    leak = float(np.linalg.norm(coherent_space_apply(v, pn)))
    assert leak < 1e-9, f"projection itself leaks {leak:.3e}"
    # And the projected vector does go negative, which is WHY a clamp exists.
    assert float(np.min(np.real(pn))) < 0.0


def test_clamp_reintroduces_coherent_leak(cfg):
    """The clamp is NONLINEAR, so it breaks the exact null-space property.

    `T_active = clip(P_null T, 0, inf)`. Clipping is not a linear map, so
    `P_coherent(T_active) != 0` even though `P_coherent(P_null T) == 0`.
    Measured leak 0.3622 with a clamped fraction of 0.0117.

    The clamp is REQUIRED: `sqrt(2 T dt)` of a negative T is NaN. This
    test pins the trade-off instead of hiding it.
    """
    thermo = AnisotropicLangevinThermostat(cfg=cfg, seed=913)
    v = _unit_rows(cfg, 3, seed=914).conj().T
    comb = np.abs(np.random.default_rng(1).standard_normal(cfg.dim))
    t, rep = thermo.compute_spectral_temperatures(comb, v)

    assert np.all(t >= 0.0), "clamped temperatures must be non-negative"
    assert 0.0 < rep.clamped_fraction < 1.0, "expected some clamped coordinates"
    assert rep.raw_min < 0.0, "projection must produce negatives here"
    assert rep.coherent_leak > 1e-3, (
        "leak unexpectedly zero; the clamp is no longer breaking the "
        "null-space property, so this finding needs re-measurement"
    )
    print(
        f"\nCLAMP: raw_min={rep.raw_min:.4f} clamped={rep.clamped_fraction:.4f} "
        f"leak={rep.coherent_leak:.4f}"
    )


def test_gact2_projected_form_passes_both_clauses(cfg):
    """G-ACT2 with the WHOLE update projected through P_null.

    Clause 1: ||P_coherent dW||_F < 1e-5
    Clause 2: ||P_null     dW||_F >= 0.15
    """
    v = _unit_rows(cfg, 4, seed=915).conj().T
    rng = np.random.default_rng(916)
    w0 = (
        rng.standard_normal((cfg.dim, 2)) + 1j * rng.standard_normal((cfg.dim, 2))
    ) / np.sqrt(cfg.dim)
    comb = np.abs(rng.standard_normal(cfg.dim))
    thermo = AnisotropicLangevinThermostat(cfg=cfg, seed=917, T_base=50.0, kappa=200.0)
    t_active, _ = thermo.compute_spectral_temperatures(comb, v)

    def grad(w):
        return 0.01 * w

    w1 = thermo.apply_viscoelastic_creep(
        w0, grad(w0), t_active, basis_coherent=v, project_update=True, retract=False
    )
    dw = w1 - w0
    coherent = float(np.linalg.norm(coherent_space_apply(v, dw)))
    null = float(np.linalg.norm(null_space_apply(v, dw)))
    print(f"\nG-ACT2 projected: coherent={coherent:.3e}  null={null:.6e}")
    assert coherent < 1e-5, f"coherent leak {coherent:.3e}"
    assert null >= 0.15, f"null relaxation {null:.6e} below 0.15"


def test_gact2_literal_form_fails_clause_one(cfg):
    """The spec's literal update (no projection) FAILS its own clause 1.

    Retained as a discriminating negative control. If this ever starts
    passing, the projection is no longer doing the work.
    """
    v = _unit_rows(cfg, 4, seed=918).conj().T
    rng = np.random.default_rng(919)
    w0 = (
        rng.standard_normal((cfg.dim, 2)) + 1j * rng.standard_normal((cfg.dim, 2))
    ) / np.sqrt(cfg.dim)
    comb = np.abs(rng.standard_normal(cfg.dim))
    thermo = AnisotropicLangevinThermostat(cfg=cfg, seed=920, T_base=50.0, kappa=200.0)
    t_active, _ = thermo.compute_spectral_temperatures(comb, v)

    def grad(w):
        return np.ones_like(w)

    w1 = thermo.apply_viscoelastic_creep(
        w0, grad(w0), t_active, basis_coherent=v, project_update=False, retract=False
    )
    dw = w1 - w0
    coherent = float(np.linalg.norm(coherent_space_apply(v, dw)))
    assert coherent > 1e-5, (
        f"literal form coherent component {coherent:.3e} unexpectedly small; "
        "the negative control stopped discriminating"
    )


def test_negative_temperature_rejected(cfg):
    thermo = AnisotropicLangevinThermostat(cfg=cfg, seed=921)
    w = np.zeros((cfg.dim, 2), dtype=np.complex128)
    t = np.full(cfg.dim, -1.0)
    with pytest.raises(ValueError, match="non-negative"):
        thermo.apply_viscoelastic_creep(w, w, t)


# ---------------------------------------------------------------- G-ACT3 --


def test_scalar_rotor_is_refused(cfg):
    rows = _unit_rows(cfg, 1, seed=930)
    a = WaveState.from_complex(rows[0], cfg)
    with pytest.raises(ScalarRotorRejectedError):
        scalar_rotor(a, 0.3)


def test_gauge_invariance_justifies_the_refusal(cfg):
    """Measured proof that a scalar rotor carries no information."""
    rows = _unit_rows(cfg, 2, seed=931)
    a = WaveState.from_complex(rows[0], cfg)
    b = WaveState.from_complex(rows[1], cfg)
    for phi in (0.1, 1.0, 3.0):
        assert gauge_invariance_error(a, b, phi) < 1e-12


def test_dimensional_ramp_preserves_norm_and_magnitudes(cfg):
    rows = _unit_rows(cfg, 1, seed=932)
    a = WaveState.from_complex(rows[0], cfg)
    b = dimensional_phase_ramp(a, 0.75)
    assert b.unit_norm_error() < 1e-12
    assert np.allclose(np.abs(a.psi), np.abs(b.psi), atol=1e-14)


def test_zero_score_delta_is_refused(cfg):
    rows = _unit_rows(cfg, 1, seed=933)
    a = WaveState.from_complex(rows[0], cfg)
    with pytest.raises(ValueError, match="nonzero"):
        dimensional_phase_ramp(a, 0.0)


def test_gact3_phase_differencing_recovers_the_delta(cfg):
    """G-ACT3 via the method that WORKS. Bar 1e-6."""
    rows = _unit_rows(cfg, 1, seed=934)
    a = WaveState.from_complex(rows[0], cfg)
    worst = 0.0
    for dS in (0.1, 0.25, 0.5, 1.0, -0.25, -1.5):
        b = dimensional_phase_ramp(a, dS)
        rec = recover_score_delta(a, b)
        worst = max(worst, abs(rec - dS))
        assert abs(rec - dS) < 1e-6, f"dS={dS} recovered {rec}"
    print(f"\nG-ACT3 phase-differencing max error: {worst:.3e}")


def test_gact3_circular_correlation_fails_as_written(cfg):
    """The spec's stated recovery method cannot meet its own bar.

    Retained as a discriminating negative control. Measured: |corr| = 0
    at dS = 1.0, so recovery is impossible, and the max error is 1.5e3
    against a 1e-6 bar.
    """
    rows = _unit_rows(cfg, 1, seed=935)
    a = WaveState.from_complex(rows[0], cfg)
    errs = []
    for dS in (0.25, 0.5, 0.75, 1.0):
        b = dimensional_phase_ramp(a, dS)
        est = circular_correlation_delta(a, b)
        if np.isfinite(est):
            errs.append(abs(est - dS))
    assert errs, "no finite estimates; the method collapsed entirely"
    assert max(errs) > 1e-6, (
        "circular correlation unexpectedly met the bar; the negative "
        "control stopped discriminating"
    )


# ---------------------------------------------------------------- loop ----


def test_step_cycle_veto_then_pass(cfg):
    """One vetoed step, then a constructive step, with real bookkeeping."""
    rows = _unit_rows(cfg, 3, seed=940)
    axiom = WaveState.from_complex(rows[0], cfg)
    unrelated = WaveState.from_complex(rows[1], cfg)
    ctrl = ActiveInferenceController(cfg=cfg)

    r1 = ctrl.step_active_cycle(unrelated, axiom)
    assert r1.vetoed is True
    assert r1.status == DESTRUCTIVE
    assert r1.buffer_size == 1

    r2 = ctrl.step_active_cycle(axiom, axiom)
    assert r2.vetoed is False
    assert r2.status == CONSTRUCTIVE
    assert len(ctrl.history) == 2


def test_cycle_ingests_environment_delta_and_recovers_it(cfg):
    rows = _unit_rows(cfg, 1, seed=941)
    axiom = WaveState.from_complex(rows[0], cfg)

    class Env:
        def act(self, wave):
            return 0.5

    ctrl = ActiveInferenceController(cfg=cfg)
    r = ctrl.step_active_cycle(axiom, axiom, environment=Env())
    assert r.vetoed is False
    assert r.delta_score == pytest.approx(0.5)
    assert r.recovery_error < 1e-6, f"recovery error {r.recovery_error}"


def test_cycle_audit_trail_is_complete(cfg):
    rows = _unit_rows(cfg, 1, seed=942)
    axiom = WaveState.from_complex(rows[0], cfg)
    ctrl = ActiveInferenceController(cfg=cfg)
    ctrl.step_active_cycle(axiom, axiom)
    d = ctrl.history[0].as_dict()
    for key in ("step", "status", "delta_sagnac", "vetoed", "buffer_size"):
        assert key in d
    assert np.isfinite(d["delta_sagnac"])


def test_controller_uses_thermostat_recoil_buffer(cfg):
    rows = _unit_rows(cfg, 2, seed=943)
    axiom = WaveState.from_complex(rows[0], cfg)
    unrelated = WaveState.from_complex(rows[1], cfg)
    thermo = AnisotropicLangevinThermostat(cfg=cfg, seed=944)
    ctrl = ActiveInferenceController(cfg=cfg, thermostat=thermo)
    ctrl.step_active_cycle(unrelated, axiom)
    assert len(thermo.recoil) == 1, "veto did not reach the thermostat buffer"


def test_veto_recoil_drives_anisotropic_temperatures(cfg):
    """Dark-port energy must feed the thermostat, not vanish."""
    rows = _unit_rows(cfg, 2, seed=945)
    axiom = WaveState.from_complex(rows[0], cfg)
    unrelated = WaveState.from_complex(rows[1], cfg)
    thermo = AnisotropicLangevinThermostat(cfg=cfg, seed=946)
    gate = SagnacHomodyneGate()
    gate.evaluate_trajectory(unrelated, axiom, thermo.recoil)

    comb = thermo.recoil.latest_comb()
    assert comb is not None and comb.max() > 0.0
    v = _unit_rows(cfg, 2, seed=947).conj().T
    t, rep = thermo.compute_spectral_temperatures(comb, v)
    assert t.max() > 0.0
    assert rep.active_max > 0.0
