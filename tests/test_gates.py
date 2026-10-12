"""Verification-matrix gates from the attached specifications.

Each test names the gate it ports and states whether it is measurable
on this host. Hardware gates are marked `hardware` and skipped; they
are NEVER softened into a pass.

Source: `henri_wave_mechanical_model_scaffolding_specification.md`
section 9, and `Project HENRI.md` section 7.
"""

from __future__ import annotations

import numpy as np
import pytest

from henri.gram import crosstalk, stiefel_retract
from henri.hopfield import cleanup
from henri.readout import normalized_entropy, select
from henri.sagnac import calibrate_epsilon, separation
from henri.state import WaveState


# ---------------------------------------------------------------- G-U1

@pytest.mark.gate
def test_g_u1_slot_bleed_within_bound(cfg):
    """G-U1: cross-slot energy bleed <= 0.0015.

    A state built in one slot must not leak into the others. This is
    the zero-pad rule (constraint C6) measured as energy.
    """
    span = cfg.slot_span
    rng = np.random.default_rng(1)
    state = WaveState.zeros(cfg)
    state.put_slot(0, rng.standard_normal(span) + 1j * rng.standard_normal(span))
    state.normalize()
    assert state.slot_bleed([0]) <= 0.0015


@pytest.mark.gate
def test_g_u1_noise_fill_is_absent(cfg):
    """G-U1: unused slots are exactly zero, not pseudorandom hash fill.

    The retired repo documented a 99.85% noise-fill defect.
    """
    span = cfg.slot_span
    state = WaveState.zeros(cfg)
    state.put_slot(0, np.ones(span, dtype=np.complex128))
    state.normalize()
    for slot in (1, 2, 3):
        assert np.all(state.psi[cfg.slot_slice(slot)] == 0.0)


# ---------------------------------------------------------------- G-U2

@pytest.mark.gate
def test_g_u2_crosstalk_bound_after_retraction():
    """G-U2: Gram crosstalk <= 0.12 across an active codebook."""
    rng = np.random.default_rng(2)
    raw = (
        rng.standard_normal((64, 512)) + 1j * rng.standard_normal((64, 512))
    )
    assert crosstalk(stiefel_retract(raw)) <= 0.12


@pytest.mark.gate
def test_g_u2_retraction_is_required_not_assumed():
    """A crowded codebook must actually violate the bound.

    This is the negative control: if raw crowding also passed, the
    gate could not fail and would be vacuous.
    """
    rng = np.random.default_rng(3)
    basis = stiefel_retract(
        rng.standard_normal((3, 512)) + 1j * rng.standard_normal((3, 512))
    )
    coeffs = rng.standard_normal((64, 3)) + 1j * rng.standard_normal((64, 3))
    assert crosstalk(coeffs @ basis) > 0.12


# ---------------------------------------------------------------- G-U3

@pytest.mark.gate
def test_g_u3_energy_monotone_with_noise_disabled(cfg):
    """G-U3 (swarm monotonicity analogue): dE/dt <= 0 with noise off.

    Gate S-2 from the swarm specification. Measured, not assumed.
    """
    rng = np.random.default_rng(4)
    book = stiefel_retract(
        rng.standard_normal((20, cfg.dim)) + 1j * rng.standard_normal((20, cfg.dim))
    )
    state = WaveState.random(cfg, seed=5)
    _, trace = cleanup(state, book, tau=0.05, iters=25)
    assert np.all(np.diff(trace) <= 1e-9)


@pytest.mark.hardware
def test_g_u3_step_latency_blocked_on_cpu():
    """G-U3 latency: T_step <= 50 us at B=256 on RTX 5090.

    BLOCKED. This host reports `cuda_available = False` and no GPU.
    The gate is not softened into a pass; it is not measured.
    """
    pytest.skip("BLOCKED: requires RTX 5090 CUDA; host is CPU-only")


# ---------------------------------------------------------------- G-U4

@pytest.mark.gate
def test_g_u4_readout_reports_decisiveness_honestly(cfg):
    """G-U4: report margin and entropy, never a bare accuracy.

    The retired repo measured live waves LESS decisive than noise
    (margin 0.023 vs 0.050). This gate records the statistic and does
    NOT assert superiority, because that would be a fabricated result.
    """
    rng = np.random.default_rng(6)
    book = stiefel_retract(
        rng.standard_normal((32, cfg.dim)) + 1j * rng.standard_normal((32, cfg.dim))
    )
    live = WaveState.random(cfg, seed=7)
    random_wave = WaveState.random(cfg, seed=8)

    live_result = select(live, book)
    rand_result = select(random_wave, book)

    assert np.isfinite(live_result.margin)
    assert np.isfinite(rand_result.margin)
    assert 0.0 <= live_result.entropy <= 1.0
    assert 0.0 <= rand_result.entropy <= 1.0


@pytest.mark.gate
def test_g_u4_no_floor_smuggled_from_random_baseline(cfg):
    """Constraint C12: no readout floor is set from a random baseline."""
    rng = np.random.default_rng(9)
    book = stiefel_retract(
        rng.standard_normal((32, cfg.dim)) + 1j * rng.standard_normal((32, cfg.dim))
    )
    result = select(WaveState.random(cfg, seed=10), book)
    assert result.meta["floor"] is None
    assert result.abstained is False


@pytest.mark.gate
def test_g_u4_mutual_information_not_claimed(cfg, codebook):
    """G-U4 info retention: I(H;Psi)/H(Psi) >= 0.95 is NOT measured here.

    Measuring it needs a KSG estimator over trajectories that this
    build does not generate. Recorded as a gap, not a pass.
    """
    pytest.skip("BLOCKED: KSG mutual-information estimator not implemented in v6")


# ---------------------------------------------------------------- G-U5

@pytest.mark.gate
def test_g_u5_resonator_family_is_not_shipped(cfg):
    """G-U5 / Directive 2: the Tripartite Resonator is REFUTED.

    The retired repo measured it WORSE than the incumbent
    (0/16 ARC, +0.144 vs +0.395). v6 must not contain it. This test
    pins its absence so a future commit cannot reintroduce it silently.
    """
    import importlib

    for forbidden in ("henri.resonator", "henri.tripartite", "henri.functor_flow"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(forbidden)


# ---------------------------------------------------------------- G-U6

@pytest.mark.gate
def test_g_u6_veto_separation_meets_directive_3(cfg):
    """G-U6 / Directive 3: separation >= 0.70, false veto <= 0.05.

    Uses the CALIBRATED epsilon. The retired repo's fixed 0.35
    saturated to AUC 0.4982 (chance).
    """
    rng = np.random.default_rng(11)
    axioms = [WaveState.random(cfg, rng=rng) for _ in range(24)]
    cal = calibrate_epsilon(axioms, sigma=3.0, ceiling=0.35, rng=rng, n_pairs=6000)

    gold = []
    for ax in axioms:
        noisy = ax.psi + 0.002 * (
            rng.standard_normal(cfg.dim) + 1j * rng.standard_normal(cfg.dim)
        )
        gold.append((WaveState.from_complex(noisy, cfg), ax))
    bad = [(WaveState.random(cfg, rng=rng), ax) for ax in axioms]

    stats = separation(gold, bad, cal)
    assert stats["false_veto"] <= 0.05, stats
    assert stats["separation"] >= 0.70, stats


@pytest.mark.gate
def test_g_u6_phase_inversion_rejected(cfg):
    """G-U6: a pi phase inversion must fire the dark port."""
    from henri.sagnac import veto

    rng = np.random.default_rng(12)
    axiom = WaveState.random(cfg, rng=rng)
    flipped = WaveState.from_complex(-axiom.psi, cfg)
    cal = calibrate_epsilon(
        [WaveState.random(cfg, rng=rng) for _ in range(12)], rng=rng, n_pairs=2000
    )
    assert veto(flipped, axiom, cal)


# ---------------------------------------------------------------- G-U7

@pytest.mark.gate
def test_g_u7_no_dense_dxd_materialized():
    """G-U7 / G-M5: exactly zero dense D x D instantiations."""
    from henri.memory import assert_no_dense_dxd

    with pytest.raises(MemoryError, match="constraint C7"):
        assert_no_dense_dxd(np.zeros((65536, 65536), dtype=np.complex128))


@pytest.mark.gate
def test_g_u7_koopman_apply_is_low_rank():
    """The production path must cost O(D r), never D^2."""
    from henri.config import WaveConfig
    from henri.koopman import KoopmanOperator

    big = WaveConfig(dim=65536, r_koopman=64)
    op = KoopmanOperator.build(big, seed=13)
    state = WaveState.random(big, seed=14)
    out = op.apply_raw(state)
    assert out.shape == (65536,)
    with pytest.raises(MemoryError, match="dense"):
        op.as_matrix()


@pytest.mark.hardware
def test_g_u7_vram_ceiling_blocked_on_cpu():
    """G-U7 VRAM: peak <= 2.5 GiB on RTX 5090. BLOCKED."""
    pytest.skip("BLOCKED: requires RTX 5090 CUDA; host is CPU-only")


# ---------------------------------------------------------------- others

@pytest.mark.gate
def test_g_m3_prefetch_hit_rate_blocked(cfg):
    """G-M3 prefetch hit rate: needs held-out trajectory rollouts."""
    pytest.skip("BLOCKED: requires trajectory bank not built in v6")


@pytest.mark.gate
def test_g_m6_viscoelastic_pruning_blocked(cfg):
    """G-M6: needs a 30-day simulated aging test."""
    pytest.skip("BLOCKED: 30-day longitudinal aging simulation not run")
