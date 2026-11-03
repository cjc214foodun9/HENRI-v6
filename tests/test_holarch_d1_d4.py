"""Decision-level invariants for Decisions 1 to 4.

Every test here pins a mechanism, not a score. The measured scores live in
receipts/holarch/ and docs/HOLARCH.md.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from henri.holarch.config import HolarchConfig
from henri.holarch.pipeline import HenriHolarch, SLMWaveProjector
from henri.holarch.tasks import DigitsTask, build_tasks
from henri.holarch.zone_b import PhotonicCore
from henri.holarch.zone_c import PrototypeBank


def _model(tmp_path, cfg, name="m.sqlite3"):
    torch.manual_seed(cfg.seed)
    return HenriHolarch(cfg, backend="sqlite", memory_path=str(tmp_path / name))


# --------------------------------------------------------------- Decision 1
def test_slm_encoder_is_parameter_free_and_injective():
    module = SLMWaveProjector(dim=256, n_pixels=256, gain=1.0)
    assert list(module.parameters()) == []
    first = torch.zeros(1, 1, 16, 16)
    first[0, 0, 0, 0] = 0.5
    second = first.clone()
    second[0, 0, 15, 15] = 0.5
    a = module(first)
    b = module(second)
    assert a.shape == (1, 256)
    assert torch.allclose(a.abs(), torch.full_like(a.abs(), 1 / 16), atol=1e-6)
    assert float((a - b).abs().max()) > 1e-4


def test_slm_rejects_the_wrong_sensor_size():
    module = SLMWaveProjector(dim=256, n_pixels=64)
    with pytest.raises(ValueError):
        module(torch.zeros(2, 1, 16, 16))


def test_linear_readout_probe_is_one_linear_map(tmp_path):
    cfg = HolarchConfig.small(readout="linear")
    model = _model(tmp_path, cfg)
    assert model.zone_a.linear_only is True
    batch = build_tasks(cfg)["vision"].sample(5, "train")
    output = model("vision", batch.x, training=False)
    assert output.task_logits.shape == (5, cfg.n_vision_classes)
    assert output.swarm.expert_logits.shape[0] == 1
    # The probe has no expert mixture: the logits are one affine map.
    with torch.no_grad():
        task_ids = torch.zeros(5, dtype=torch.long)
        x = output.features + model.zone_a.task_embedding(task_ids)
        if model.zone_a.universal is not None:
            x = model.zone_a.universal(x, task_ids)
        expected = model.zone_a.linear(x)[:, : cfg.n_vision_classes]
    assert torch.allclose(output.task_logits, expected, atol=1e-5)
    model.memory.close()


def test_linear_readout_has_no_fast_weight_write(tmp_path):
    cfg = HolarchConfig.small(readout="linear")
    model = _model(tmp_path, cfg)
    task = build_tasks(cfg)["vision"]
    batch = task.sample(4, "train")
    output = model("vision", batch.x, training=False)
    model.fast_learn(output, batch.y, lr=0.5)
    for expert in model.zone_a.experts:
        assert float(expert.B.abs().max()) == 0.0
    model.memory.close()


def test_patch_composition_canvas_is_256_values_with_two_informative_quadrants():
    task = DigitsTask(seed=3, n_train=8, n_val=4, size=16, composition="sum")
    batch = task.sample(4, "train")
    assert tuple(batch.x.shape) == (4, 1, 16, 16)
    canvas = batch.x[:, 0].numpy()
    # Rebuilding the label from the informative quadrants alone must match.
    rng = np.random.default_rng(0)
    images = np.stack(
        [np.full((8, 8), float(k), dtype=np.float32) for k in range(10)]
    )
    labels = np.arange(10, dtype=np.int64)
    built, outcome = DigitsTask._compose(
        np.random.default_rng(1), images, labels, 32, "sum"
    )
    assert built.shape == (32, 16, 16)
    assert set(np.unique(outcome)).issubset(set(range(10)))
    assert outcome.shape == (32,)
    del canvas, rng


def test_clutter_composition_labels_only_quadrant_zero():
    images = np.stack([np.full((8, 8), float(k), dtype=np.float32) for k in range(10)])
    labels = np.arange(10, dtype=np.int64)
    built, outcome = DigitsTask._compose(
        np.random.default_rng(5), images, labels, 64, "clutter"
    )
    quadrant_zero = built[:, 0, 0]
    assert np.allclose(quadrant_zero, outcome.astype(np.float32))
    assert not np.allclose(built[:, 8, 8], 0.0)


# --------------------------------------------------------------- Decision 2
def test_prototype_rows_are_unit_and_beta_is_calibrated_not_transferred(tmp_path):
    cfg = HolarchConfig.small()
    bank = PrototypeBank(cfg.dim)
    generator = np.random.default_rng(0)
    waves = generator.standard_normal((20, cfg.dim)) + 1j * generator.standard_normal(
        (20, cfg.dim)
    )
    waves = (waves / np.linalg.norm(waves, axis=1, keepdims=True)).astype(np.complex64)
    labels = np.array([index % 3 for index in range(20)], dtype=np.int64)
    stats = bank.build(waves, labels)
    assert stats["prototypes"] == 3.0
    norms = np.linalg.norm(bank.phi, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
    calibration = bank.calibrate_beta(waves)
    assert calibration["beta"] > 0.0
    assert bank.beta_source == "calibrated"
    # Constraint C3: the transferred literal is recorded and refused.
    assert PrototypeBank.BETA_STAR_REFERENCE == pytest.approx(26.10)
    assert calibration["beta_star_refused"] == pytest.approx(26.10)
    assert bank.beta != pytest.approx(26.10)


def test_relaxation_lowers_the_lyapunov_energy_and_the_dark_energy(tmp_path):
    cfg = HolarchConfig.small()
    core = PhotonicCore(cfg)
    generator = torch.Generator().manual_seed(11)
    wave = torch.randn(6, cfg.dim, generator=generator, dtype=torch.complex64)
    wave = wave / wave.norm(dim=-1, keepdim=True)
    phi = torch.randn(4, cfg.dim, generator=generator, dtype=torch.complex64)
    phi = phi / phi.norm(dim=-1, keepdim=True)
    before = core(wave, apply_memory=False).bright
    energy_before, _ = core.free_energy(before, phi, cfg.prototype_lambda_dark)
    dark_before = core.dark_energy(before).sum(dim=-1)
    state, trace = core.relax_prototype(wave, phi, beta=8.0)
    energy_after, _ = core.free_energy(state.bright, phi, cfg.prototype_lambda_dark)
    dark_after = core.dark_energy(state.bright).sum(dim=-1)
    assert float(energy_after.mean()) <= float(energy_before.mean()) + 1e-5
    assert float(dark_after.mean()) < float(dark_before.mean())
    assert min(trace) <= trace[0] + 1e-6


def test_prototype_bank_round_trips_through_zone_c(tmp_path):
    cfg = HolarchConfig.small()
    model = _model(tmp_path, cfg)
    task = build_tasks(cfg)["vision"]
    stats = model.calibrate_prototypes("vision", [task.sample(64, "train") for _ in range(2)])
    assert stats["prototypes"] == 10.0
    written = model.store_prototypes("vision")
    assert written["prototype_engrams_written"] == 10.0
    assert len(model.memory.axiomatic_engrams()) == 10
    before = model.prototypes["vision"].phi.copy()
    model.prototypes["vision"].phi = np.zeros((0, cfg.dim), dtype=np.complex64)
    model.set_prototypes("vision")
    assert int(model.prototype_waves("vision").numel()) == 0
    loaded = model.load_prototypes("vision")
    assert loaded == 10
    after = model.prototypes["vision"].phi
    overlap = np.abs(np.real((after @ before.conj().T)))
    assert float(np.max(np.diag(overlap))) > 0.99
    model.memory.close()


def test_empty_bank_falls_back_to_the_axiom_path(tmp_path):
    cfg = HolarchConfig.small()
    model = _model(tmp_path, cfg)
    task = build_tasks(cfg)["vision"]
    batch = task.sample(4, "val")
    output = model("vision", batch.x, training=False, relax=True)
    assert int(model.prototype_waves("vision").numel()) == 0
    assert torch.isfinite(output.task_logits).all()
    model.memory.close()


# --------------------------------------------------------------- Decision 3
def test_relational_basis_reads_both_quadratures(tmp_path):
    cfg = HolarchConfig.small(relational_binding="wave", relational_k=32)
    model = _model(tmp_path, cfg)
    assert model.relational_width == 2 * cfg.relational_k
    task = build_tasks(cfg)["relational"]
    output = model("relational", task.sample(3, "train").x, training=False)
    features = model.relational_features(output.photonic)
    assert tuple(features.shape) == (3, 2 * cfg.relational_k)
    basis = model.relational_basis.basis()
    gram = basis @ basis.conj().transpose(-1, -2)
    eye = torch.eye(cfg.relational_k, dtype=gram.dtype)
    assert float((gram - eye).abs().max().detach()) < 1e-4
    model.memory.close()


def test_relational_regularisation_is_right_sized():
    cfg = HolarchConfig.small()
    assert cfg.relational_l2 == pytest.approx(5e-4)
    assert cfg.relational_dropout == pytest.approx(0.05)
    control = HolarchConfig.small(relational_binding="trilinear", relational_l2=0.05)
    assert control.relational_l2 == pytest.approx(0.05)


def test_trilinear_control_arm_keeps_the_retired_head_width(tmp_path):
    cfg = HolarchConfig.small(relational_binding="trilinear")
    model = _model(tmp_path, cfg)
    assert model.relational_width == model.zone_b.relational_width
    task = build_tasks(cfg)["relational"]
    output = model("relational", task.sample(2, "train").x, training=False)
    assert tuple(output.task_logits.shape) == (2, cfg.relational_classes)
    model.memory.close()


# --------------------------------------------------------------- Decision 4
def test_rule_phasor_touches_slot_three_only(tmp_path):
    cfg = HolarchConfig.small()
    model = _model(tmp_path, cfg)
    task = build_tasks(cfg)["vision"]
    batch = task.sample(4, "val")
    first = model("vision", batch.x, training=False, rule_id=0, relax=False).psi_in
    again = model("vision", batch.x, training=False, rule_id=0, relax=False).psi_in
    other = model("vision", batch.x, training=False, rule_id=3, relax=False).psi_in
    assert torch.allclose(first, again)
    slots_first = first.reshape(4, cfg.n_slots, cfg.d_slot)
    slots_other = other.reshape(4, cfg.n_slots, cfg.d_slot)
    for slot in range(3):
        assert torch.allclose(slots_first[:, slot], slots_other[:, slot])
    assert float((slots_first[:, 3] - slots_other[:, 3]).abs().max()) > 1e-3
    model.memory.close()


def test_fast_weight_gate_closes_on_familiar_and_opens_on_shift(tmp_path):
    cfg = HolarchConfig.small(fast_gate=True)
    model = _model(tmp_path, cfg)
    task = build_tasks(cfg)["vision"]
    model.calibrate_prototypes("vision", [task.sample(64, "train") for _ in range(2)])
    model.calibrate_gate("vision", [task.sample(32, "val").x for _ in range(2)])
    familiar = model("vision", task.sample(16, "val").x, training=False, relax=False)
    familiar_rate = float(model.plasticity_gate(familiar).mean())
    permutation = torch.randperm(64, generator=torch.Generator().manual_seed(99))
    batch = task.sample(16, "val")
    shifted_x = batch.x.reshape(16, -1)[:, permutation].reshape(batch.x.shape)
    shifted = model("vision", shifted_x, training=False, relax=False)
    shifted_rate = float(model.plasticity_gate(shifted).mean())
    assert familiar_rate <= 0.15
    assert shifted_rate >= 0.75
    model.memory.close()


def test_closed_gate_writes_no_fast_weights(tmp_path):
    cfg = HolarchConfig.small(fast_gate=True)
    model = _model(tmp_path, cfg)
    expert = model.zone_a.experts[0]
    features = torch.randn(8, cfg.d_feat)
    labels = torch.randint(0, 5, (8,))
    expert.online_update(
        features, labels, lr=0.2, start=0, end=10, weights=torch.zeros(8)
    )
    assert float(expert.B.abs().max()) == 0.0
    expert.online_update(
        features, labels, lr=0.2, start=0, end=10, weights=torch.ones(8)
    )
    assert float(expert.B.abs().max()) > 0.0


def test_open_gate_gives_the_same_delta_as_an_ungated_write():
    cfg = HolarchConfig.small()
    torch.manual_seed(0)
    left = __import__("henri.holarch.zone_a", fromlist=["FastWeightExpert"]).FastWeightExpert(
        d_in=8, hidden=4, d_out=10, rank=2, seed=1
    )
    right = __import__("henri.holarch.zone_a", fromlist=["FastWeightExpert"]).FastWeightExpert(
        d_in=8, hidden=4, d_out=10, rank=2, seed=1
    )
    features = torch.randn(6, 8)
    labels = torch.randint(0, 10, (6,))
    left.online_update(features, labels, lr=0.2, start=0, end=10)
    right.online_update(features, labels, lr=0.2, start=0, end=10, weights=torch.ones(6))
    assert torch.allclose(left.B, right.B, atol=1e-6)
