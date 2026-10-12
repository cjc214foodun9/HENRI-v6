"""Pipeline end-to-end tests. Exercises the real integrated path."""

from __future__ import annotations

import numpy as np
import pytest

from henri.config import WaveConfig
from henri.koopman import KoopmanOperator
from henri.pipeline import build_ingress, make_codebook, run_pipeline
from henri.sagnac import calibrate_epsilon
from henri.state import WaveState


def _ingress(cfg, seed=0):
    rng = np.random.default_rng(seed)
    span = cfg.slot_span
    slots = []
    for _ in range(4):
        slots.append(rng.standard_normal(span) + 1j * rng.standard_normal(span))
    return build_ingress(cfg, slots)


def test_ingress_is_unit_norm(cfg):
    state = _ingress(cfg)
    assert state.unit_norm_error() < 1e-12


def test_ingress_zeroes_none_slots(cfg):
    """Constraint C6: a None slot is explicitly zero, never noise."""
    span = cfg.slot_span
    vals = [np.ones(span, dtype=np.complex128), None, None, None]
    state = build_ingress(cfg, vals)
    assert np.all(state.psi[cfg.slot_slice(1)] == 0.0)
    assert np.all(state.psi[cfg.slot_slice(2)] == 0.0)
    assert np.all(state.psi[cfg.slot_slice(3)] == 0.0)


def test_ingress_requires_four_entries(cfg):
    with pytest.raises(ValueError, match="4 slot entries"):
        build_ingress(cfg, [None, None])


def test_pipeline_runs_end_to_end(cfg):
    book = make_codebook(cfg, 16, seed=1)
    axiom = WaveState.from_complex(book[0], cfg)
    refs = [WaveState.from_complex(book[i], cfg) for i in range(16)]
    cal = calibrate_epsilon(refs, sigma=3.0, ceiling=0.35, seed=2, n_pairs=2000)

    result = run_pipeline(
        cfg=cfg,
        ingress=_ingress(cfg, seed=3),
        codebook=book,
        axiom=axiom,
        calibration=cal,
        seed=4,
        relax_steps=2,
        lock_steps=50,
    )
    assert result is not None
    assert result.index >= -1
    assert result.envelope is not None
    assert np.isfinite(result.order_parameter)


def test_pipeline_wave_stages_stay_on_sphere(cfg):
    book = make_codebook(cfg, 12, seed=5)
    axiom = WaveState.from_complex(book[0], cfg)
    cal = calibrate_epsilon(
        [WaveState.from_complex(book[i], cfg) for i in range(12)], seed=6, n_pairs=1000
    )
    result = run_pipeline(
        cfg=cfg,
        ingress=_ingress(cfg, seed=7),
        codebook=book,
        axiom=axiom,
        calibration=cal,
        seed=8,
        relax_steps=2,
        lock_steps=30,
    )
    for state in (result.ingress, result.relaxed, result.cleaned):
        assert state.unit_norm_error() < 1e-12


def test_pipeline_reports_full_audit_trail(cfg):
    book = make_codebook(cfg, 10, seed=9)
    axiom = WaveState.from_complex(book[0], cfg)
    cal = calibrate_epsilon(
        [WaveState.from_complex(book[i], cfg) for i in range(10)], seed=10, n_pairs=1000
    )
    result = run_pipeline(
        cfg=cfg,
        ingress=_ingress(cfg, seed=11),
        codebook=book,
        axiom=axiom,
        calibration=cal,
        seed=12,
        relax_steps=1,
        lock_steps=20,
    )
    d = result.as_dict()
    for key in ("norm_drift", "epsilon", "margin", "group_size", "entropy"):
        assert key in d["stages"]
        assert np.isfinite(d["stages"][key])


def test_pipeline_is_reproducible(cfg):
    book = make_codebook(cfg, 12, seed=13)
    axiom = WaveState.from_complex(book[0], cfg)
    cal = calibrate_epsilon(
        [WaveState.from_complex(book[i], cfg) for i in range(12)], seed=14, n_pairs=1000
    )

    def once():
        return run_pipeline(
            cfg=cfg,
            ingress=_ingress(cfg, seed=15),
            codebook=book,
            axiom=axiom,
            calibration=cal,
            seed=16,
            relax_steps=2,
            lock_steps=25,
        )

    a, b = once(), once()
    assert a.index == b.index
    assert abs(a.order_parameter - b.order_parameter) < 1e-12
    assert np.allclose(a.cleaned.psi, b.cleaned.psi)


def test_veto_path_emits_abstain(cfg):
    """An axiom far from the candidate must trigger the abstain envelope."""
    book = make_codebook(cfg, 8, seed=17)
    axiom = WaveState.random(cfg, seed=18)
    cal = calibrate_epsilon(
        [WaveState.random(cfg, seed=100 + i) for i in range(10)], seed=19, n_pairs=2000
    )
    result = run_pipeline(
        cfg=cfg,
        ingress=_ingress(cfg, seed=20),
        codebook=book,
        axiom=axiom,
        calibration=cal,
        seed=21,
        relax_steps=1,
        lock_steps=20,
    )
    if result.veto_fired:
        assert result.envelope.kind == "abstain"
        assert result.envelope.verified is False
        assert result.index == -1
    else:
        assert result.envelope.kind == "write_memory"


def test_make_codebook_is_deterministic(cfg):
    a = make_codebook(cfg, 8, seed=22)
    b = make_codebook(cfg, 8, seed=22)
    assert np.allclose(a, b)


def test_make_codebook_orthogonalized_has_low_crosstalk(cfg):
    from henri.gram import crosstalk

    book = make_codebook(cfg, 16, seed=23, orthogonalize=True)
    assert crosstalk(book) < 0.12


def test_make_codebook_count_validation(cfg):
    with pytest.raises(ValueError, match="count must be"):
        make_codebook(cfg, 0)


def test_pipeline_accepts_external_operator(cfg):
    book = make_codebook(cfg, 8, seed=24)
    axiom = WaveState.from_complex(book[0], cfg)
    cal = calibrate_epsilon(
        [WaveState.from_complex(book[i], cfg) for i in range(8)], seed=25, n_pairs=800
    )
    op = KoopmanOperator.build(cfg, seed=26)
    result = run_pipeline(
        cfg=cfg,
        ingress=_ingress(cfg, seed=27),
        codebook=book,
        axiom=axiom,
        calibration=cal,
        operator=op,
        relax_steps=1,
        lock_steps=10,
    )
    assert result is not None
    assert np.isfinite(result.stages["norm_drift"])
