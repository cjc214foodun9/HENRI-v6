"""Directive-level invariants: skip severance, Stiefel pooling, veto, relax."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from henri.holarch.config import HolarchConfig
from henri.holarch.pipeline import HenriHolarch
from henri.holarch.tasks import build_tasks
from henri.holarch.zone_b import PhotonicCore, StiefelSlotPool


def _wave(cfg, count=4, seed=3):
    generator = torch.Generator().manual_seed(seed)
    wave = torch.randn(count, cfg.dim, generator=generator, dtype=torch.complex64)
    return wave / wave.norm(dim=-1, keepdim=True)


def test_stiefel_basis_rows_are_orthonormal():
    cfg = HolarchConfig.small()
    pool = StiefelSlotPool(cfg.n_slots, cfg.d_slot, cfg.pool_k)
    basis = pool.basis()
    gram = basis @ basis.conj().transpose(-1, -2)
    eye = torch.eye(cfg.pool_k, dtype=gram.dtype).expand_as(gram)
    assert float((gram - eye).abs().max().detach()) < 1e-4


def test_slot_mixer_is_unitary():
    cfg = HolarchConfig.small()
    core = PhotonicCore(cfg)
    assert core.mixer is not None
    matrix = core.mixer.matrix()
    product = matrix @ matrix.conj().transpose(-1, -2)
    eye = torch.eye(cfg.n_slots, dtype=product.dtype)
    assert float((product - eye).abs().max().detach()) < 1e-5


def test_bypass_is_reciprocal_and_keeps_the_feature_width():
    cfg = HolarchConfig.small()
    core = PhotonicCore(cfg)
    wave = _wave(cfg)
    state = core(wave, bypass=True)
    assert float(state.delta_reciprocal.max().detach()) < 1e-5
    assert state.features.shape == (4, core.feature_width)


def test_relax_reduces_the_residual_and_never_increases_the_best():
    cfg = HolarchConfig.small()
    core = PhotonicCore(cfg)
    wave = _wave(cfg, 2, seed=5)
    axiom = _wave(cfg, 2, seed=6)
    state, trace = core.relax(wave, axiom, steps=4)
    assert min(trace) <= trace[0] + 1e-6
    assert float(state.delta_memory.mean().detach()) <= trace[0] + 1e-6


def test_thermal_feedback_raises_the_gain():
    cfg = HolarchConfig.small()
    core = PhotonicCore(cfg)
    wave = _wave(cfg, 1)
    cool = core(wave)
    hot = core(wave, thermal_feedback=torch.full((1, cfg.n_slots), 0.5))
    assert float(hot.gain.mean().detach()) > float(cool.gain.mean().detach())


def test_skip_alpha_zero_blocks_the_latent_skip_gradient(tmp_path):
    cfg = HolarchConfig.small()
    model = HenriHolarch(cfg, backend="sqlite", memory_path=str(tmp_path / "s.sqlite3"))
    batch = build_tasks(cfg)["vision"].sample(4, "train")
    output = model("vision", batch.x, training=False)
    loss = output.task_logits.sum()
    loss.backward()
    gradient = model.latent_skip.weight.grad
    assert gradient is None or float(gradient.abs().max()) < 1e-8
    model.memory.close()


def test_low_gram_coherence_triggers_the_ood_veto(tmp_path):
    cfg = HolarchConfig.small()
    model = HenriHolarch(cfg, backend="sqlite", memory_path=str(tmp_path / "v.sqlite3"))
    tasks = build_tasks(cfg)
    batch = tasks["vision"].sample(16, "train")
    model.observe("vision", batch.x, batch.y, max_writes=16)
    batches = [tasks["vision"].sample(16, "val").x for _ in range(3)]
    model.calibrate_chance("vision", batches, percentile=1.0, margin=0.02)
    model.calibrate_veto("vision", batches, ceiling=0.98)
    # The rule must fire when the threshold is above the query statistic.
    # The statistic's OOD separation after training is measured by the
    # canonical receipt (1.000 veto at 0.010 false veto), not by an
    # untrained fixture.
    model.chance_epsilon("vision").fill_(1.1)
    output = model("vision", batch.x[:4], training=False)
    assert bool(output.veto.all())
    model.memory.close()


def test_relational_holdout_keeps_every_primitive_and_class(tmp_path):
    cfg = HolarchConfig.small()
    task = build_tasks(cfg)["relational"]
    train = {tuple(row) for row in task.x_train.tolist()}
    test = {tuple(row) for row in task.x_test.tolist()}
    assert not (train & test)
    for column, size in ((0, cfg.relational_entities), (1, cfg.relational_actions), (2, cfg.relational_targets)):
        assert {row[column] for row in train} == set(range(size))
    assert set(task.y_train.tolist()) == set(range(cfg.relational_classes))
    assert set(task.y_test.tolist()) == set(range(cfg.relational_classes))


def test_relational_forward_binds_into_slot_three(tmp_path):
    cfg = HolarchConfig.small()
    model = HenriHolarch(cfg, backend="sqlite", memory_path=str(tmp_path / "r.sqlite3"))
    task = build_tasks(cfg)["relational"]
    batch = task.sample(3, "train")
    output = model("relational", batch.x, training=False)
    assert output.psi_in.shape == (3, cfg.dim)
    slots = output.psi_in.reshape(3, cfg.n_slots, cfg.d_slot)
    assert float(slots[:, 3].abs().sum().detach()) > 0.0
    model.memory.close()
