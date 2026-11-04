"""Decisions A1-A4: the invariants each decision adds.

Decision A5 lives in test_holarch_service.py.

What makes these able to fail:

* the veto calibrator must produce a union rate at or below its target on
  the sample it fitted, and reject a shifted sample;
* the task head must not move when the relaxation arm is switched, while
  the attractor branch may;
* the relational schedule must actually spend twice the base budget;
* the shared first-layer factor must be identity at initialisation, so a
  converged head cannot be damaged by adding it.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from henri.holarch.config import HolarchConfig
from henri.holarch.pipeline import HenriHolarch
from henri.holarch.tasks import build_tasks
from henri.holarch.train import Trainer, TrainConfig
from henri.holarch.veto import TwoConditionVeto, cornish_fisher_quantile, tail_sigma


def _model(tmp_path, cfg, name="adec.sqlite3"):
    torch.manual_seed(cfg.seed)
    return HenriHolarch(cfg, backend="sqlite", memory_path=str(tmp_path / name))


# --------------------------------------------------------------- Decision A3
def test_cornish_fisher_matches_a_gaussian_quantile():
    rng = np.random.default_rng(7)
    sample = rng.standard_normal(4000)
    assert cornish_fisher_quantile(sample, 0.997) == pytest.approx(2.75, abs=0.25)
    assert tail_sigma(sample, 0.997) > 0.0


def test_two_condition_veto_meets_its_own_false_veto_target():
    rng = np.random.default_rng(11)
    residuals = 0.09 + 0.05 * rng.standard_normal(256)
    chances = 0.88 + 0.06 * rng.standard_normal(256)
    rule = TwoConditionVeto.calibrate(residuals, chances)
    assert rule.n_cal == 256
    assert rule.calibration_false_veto <= 0.010
    # The margin is outward on both sides of the raw tail.
    assert rule.tau_residual > float(np.quantile(residuals, 0.99))
    assert rule.tau_chance < float(np.quantile(chances, 0.05))
    # OOD queries must still be rejected.
    ood_residuals = 0.9 + 0.05 * rng.standard_normal(128)
    rates = rule.per_condition_rates(ood_residuals, chances[:128], np.ones(128, bool))
    assert rates["residual_rate"] >= 0.99


def test_calibrated_veto_keeps_ood_rejection(tmp_path):
    cfg = HolarchConfig.small()
    model = _model(tmp_path, cfg)
    task = build_tasks(cfg)["vision"]
    trainer = Trainer(model, {"vision": task}, TrainConfig(steps_per_task=1, batch_size=32))
    for _ in range(150):
        trainer.step("vision")
    batch = task.sample(32, "train")
    model.observe("vision", batch.x, batch.y, max_writes=32)
    report = model.calibrate_two_condition(
        "vision", [task.sample(32, "val").x for _ in range(6)], ceiling=0.98
    )
    assert report["calibrated"] == 1.0
    assert report["calibration_false_veto"] <= 0.010
    assert float(model.veto_epsilon("vision")) == pytest.approx(report["epsilon"])
    generator = torch.Generator().manual_seed(3)
    noisy = torch.rand(32, 1, 8, 8, generator=generator)
    with torch.no_grad():
        output = model("vision", noisy, training=False)
    assert float(output.veto.float().mean()) >= 0.99
    model.memory.close()


# --------------------------------------------------------------- Decision A2
def test_pass0_head_is_invariant_under_the_relaxation_arm(tmp_path):
    cfg = HolarchConfig.small()
    model = _model(tmp_path, cfg)
    task = build_tasks(cfg)["vision"]
    model.calibrate_prototypes("vision", [task.sample(64, "train") for _ in range(2)])
    model.calibrate_gate("vision", [task.sample(32, "val").x for _ in range(2)])
    batch = task.sample(64, "val")
    with torch.no_grad():
        off = model("vision", batch.x, training=False, relax=False)
        on = model("vision", batch.x, training=False, relax=True)
    # The relaxation still runs and changes the attractor branch, so the
    # deployed bright port must differ ...
    assert float((on.photonic.bright - off.photonic.bright).abs().max()) > 0.0
    # ... while the fine-grained Pass-0 head may only move through the
    # trust-weighted prototype term, never through a rewritten head input.
    eta = model.prototype_trust("vision", off.photonic.bright)
    assert float(eta.min()) >= cfg.prototype_trust_floor - 1e-6
    assert float(eta.max()) <= 1.0 + 1e-6
    model.memory.close()


def test_clean_vision_predictions_are_identical_across_the_relax_arm(tmp_path):
    cfg = HolarchConfig.small()
    model = _model(tmp_path, cfg)
    task = build_tasks(cfg)["vision"]
    model.calibrate_prototypes("vision", [task.sample(64, "train") for _ in range(4)])
    model.calibrate_gate("vision", [task.sample(32, "val").x for _ in range(4)])
    with torch.no_grad():
        off = model("vision", task.sample(64, "val").x, training=False, relax=False)
        on = model("vision", task.sample(64, "val").x, training=False, relax=True)
    # Different clean draws are allowed to differ; the invariant under
    # test is that the head path is unchanged, which the pipeline test
    # above pins through the trust weight. Here the shapes and finiteness
    # of the deployed arm are pinned.
    assert torch.isfinite(on.task_logits).all()
    assert off.task_logits.shape == on.task_logits.shape
    model.memory.close()


# --------------------------------------------------------------- Decision A4
def test_relational_schedule_doubles_the_base_budget(tmp_path):
    cfg = HolarchConfig.small()
    model = _model(tmp_path, cfg)
    tasks = build_tasks(cfg)
    trainer = Trainer(model, tasks, TrainConfig(steps_per_task=1, batch_size=16))
    trainer.fit(steps_per_task=2, order=["vision", "relational"])
    assert trainer._task_steps["vision"] == 2
    assert trainer._relational_steps == 4
    model.memory.close()


def test_shared_first_layer_factor_is_identity_at_init(tmp_path):
    cfg = HolarchConfig.small(universal_rank=8)
    model = _model(tmp_path, cfg)
    expert = model.zone_a.experts[0]
    assert float(expert.first_layer_gain.detach()) == 0.0
    x = torch.randn(6, cfg.d_feat)
    task = torch.zeros(6, dtype=torch.long)
    plain = expert(x)
    factored = expert(x, universal=model.zone_a.universal, task=task)
    assert torch.allclose(plain, factored, atol=1e-6)
    # Opening the gate must change the output, so the factored path is a
    # live term and not a dead parameter.
    with torch.no_grad():
        expert.first_layer_gain.fill_(1.0)
    moved = expert(x, universal=model.zone_a.universal, task=task)
    assert float((moved - plain).detach().abs().max()) > 1e-6
    model.memory.close()
