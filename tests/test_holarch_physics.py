"""Physics invariants of the Zone B photonic kernel."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from henri.config import WaveConfig
from henri.holarch.config import HolarchConfig
from henri.holarch.zone_b import (
    PhotonicCore,
    curie_weiss_gain,
    phase_authority,
    sagnac_delta,
)
from henri.sagnac import sagnac_delta as reference_delta
from henri.state import WaveState


def _waves(cfg, count=6, seed=7):
    generator = torch.Generator().manual_seed(seed)
    wave = torch.randn(count, cfg.dim, generator=generator, dtype=torch.complex64)
    return wave / wave.norm(dim=-1, keepdim=True)


def test_norm_is_conserved_through_the_loop():
    cfg = HolarchConfig.small()
    core = PhotonicCore(cfg)
    wave = _waves(cfg)
    state = core(wave, axiom=torch.roll(wave, 2, dims=-1))
    assert state.norm_drift < 1e-4
    assert float(state.bright.abs().pow(2).sum(-1).mean().detach()) > 0.9
    assert float(state.dark.abs().pow(2).sum(-1).mean().detach()) >= 0.0


def test_reciprocal_loop_is_dark():
    cfg = HolarchConfig.small()
    core = PhotonicCore(cfg)
    state = core(_waves(cfg))
    # Zero drive -> both arms identical -> destructive port is exactly zero.
    assert float(state.delta_reciprocal.max().detach()) < 1e-5


def test_curie_weiss_gain_is_monotone_and_finite():
    temperatures = torch.linspace(0.0, 0.999, 32)
    gains = curie_weiss_gain(temperatures)
    assert float(gains[0]) == pytest.approx(1.0, abs=1e-6)
    assert bool((gains[1:] >= gains[:-1]).all())
    assert torch.isfinite(gains).all()
    assert float(gains[-1]) < 1e4


def test_heater_power_sets_phase_authority():
    cfg = HolarchConfig.small()
    core = PhotonicCore(cfg)
    wave = _waves(cfg, count=1)[0]
    cold = core.reachable_shift(wave, 0.1, 0.05)
    hot = core.reachable_shift(wave, 0.1, cfg.t_max - 1e-3)
    assert hot > cold * 2.0
    assert phase_authority(torch.tensor([cfg.t_max - 1e-3])) > phase_authority(torch.tensor([0.0]))


def test_sagnac_delta_matches_the_v6_reference():
    cfg = HolarchConfig.small()
    wave = _waves(cfg, count=4)
    torch_cfg = WaveConfig.small(dim=cfg.dim)
    a = WaveState.from_complex(wave[0].numpy().astype(np.complex128), torch_cfg)
    b = WaveState.from_complex(wave[1].numpy().astype(np.complex128), torch_cfg)
    expected = reference_delta(a, b)
    got = float(sagnac_delta(wave[0], wave[1]))
    assert got == pytest.approx(expected, abs=1e-5)


def test_chance_calibration_respects_the_ceiling():
    cfg = HolarchConfig.small()
    core = PhotonicCore(cfg)
    report = core.calibrate([_waves(cfg, count=8)[i] for i in range(8)])
    assert 0.0 < report["epsilon"] <= 0.35
    assert report["n_pairs"] == 1024


def test_phase_conjugate_imprint_reduces_residual():
    cfg = HolarchConfig.small()
    core = PhotonicCore(cfg)
    wave = _waves(cfg, count=1)
    target = _waves(cfg, count=1, seed=11)
    with torch.no_grad():
        before = float(torch.linalg.vector_norm(target - core(wave).bright))
    for _ in range(4):
        core.imprint(wave, target, rate=0.8)
    with torch.no_grad():
        after = float(torch.linalg.vector_norm(target - core(wave).bright))
    assert after < before


def test_gradients_reach_every_photonic_parameter():
    cfg = HolarchConfig.small()
    core = PhotonicCore(cfg)
    wave = _waves(cfg, count=3)
    loss = core(wave).features.pow(2).sum()
    loss.backward()
    for name, parameter in core.named_parameters():
        if "transfer" in name:
            continue
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
