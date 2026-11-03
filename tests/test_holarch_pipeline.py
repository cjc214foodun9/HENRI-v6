"""End-to-end wiring tests for the three-zone holarch."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from henri.holarch.budget import budget_report
from henri.holarch.config import HolarchConfig
from henri.holarch.pipeline import HenriHolarch
from henri.holarch.tasks import build_tasks
from henri.holarch.train import Trainer, TrainConfig, evaluate


@pytest.fixture(scope="module")
def tasks():
    return build_tasks(HolarchConfig.small())


def _model(tmp_path, name="model.sqlite3", cfg=None):
    cfg = cfg or HolarchConfig.small()
    torch.manual_seed(cfg.seed)
    return HenriHolarch(cfg, backend="sqlite", memory_path=str(tmp_path / name)), cfg


def test_forward_shapes_and_audit_fields(tmp_path, tasks):
    model, cfg = _model(tmp_path)
    batch = tasks["vision"].sample(5, "train")
    output = model("vision", batch.x, training=False)
    assert output.task_logits.shape == (5, cfg.n_vision_classes)
    assert output.features.shape == (5, cfg.d_feat)
    assert output.psi_in.shape == (5, cfg.dim)
    assert output.veto.shape == (5,)
    assert torch.isfinite(output.task_logits).all()
    assert output.photonic.norm_drift < 1e-4
    model.memory.close()


def test_training_improves_vision_above_chance(tmp_path, tasks):
    model, cfg = _model(tmp_path)
    trainer = Trainer(model, tasks, TrainConfig(steps_per_task=1, batch_size=32, lr=2e-3))
    before = evaluate(model, tasks, "val", limit=200)["vision_accuracy"]
    for _ in range(90):
        trainer.step("vision")
    after = evaluate(model, tasks, "val", limit=200)["vision_accuracy"]
    assert after > before + 0.2
    assert after > 0.4
    model.memory.close()


def test_two_runs_with_the_same_seed_agree(tmp_path):
    from henri.holarch.tasks import DigitsTask

    losses = []
    for run in range(2):
        cfg = HolarchConfig.small()
        model, _ = _model(tmp_path, name=f"det{run}.sqlite3", cfg=cfg)
        torch.manual_seed(1234)
        # A fresh sampler is part of the seed contract. Reusing one advances
        # its generator, so two runs would draw different batches.
        run_tasks = {"vision": DigitsTask(seed=cfg.seed)}
        trainer = Trainer(model, run_tasks, TrainConfig(steps_per_task=1, batch_size=16, lr=1e-3))
        for _ in range(12):
            record = trainer.step("vision")
        losses.append(record["loss"])
        model.memory.close()
    assert abs(losses[0] - losses[1]) < 1e-5


def test_ablation_paths_run_without_the_photonic_or_swarm_stage(tmp_path, tasks):
    model, cfg = _model(tmp_path)
    batch = tasks["vision"].sample(4, "train")
    plain = model("vision", batch.x, use_photonic=False, use_swarm=False, training=False)
    assert torch.isfinite(plain.task_logits).all()
    assert float(plain.veto.float().sum()) == 0.0
    full = model("vision", batch.x, use_photonic=True, use_swarm=True, training=False)
    assert torch.isfinite(full.task_logits).all()
    model.memory.close()


def test_memory_observe_and_read_back(tmp_path, tasks):
    model, cfg = _model(tmp_path)
    batch = tasks["vision"].sample(8, "train")
    written = model.observe("vision", batch.x, batch.y, max_writes=8)
    assert written == 8
    assert model.memory.stats()["count"] == 8
    waves = model.encode("vision", batch.x).detach().numpy()
    _, read = model.memory.context_tensor(waves[3], k=1)
    assert read.hits and read.hits[0].engram_id == 4
    model.memory.close()


def test_veto_calibration_is_bounded(tmp_path, tasks):
    model, cfg = _model(tmp_path)
    batch = tasks["vision"].sample(16, "train")
    model.observe("vision", batch.x, batch.y, max_writes=16)
    report = model.calibrate_veto(
        "vision", [tasks["vision"].sample(16, "val").x for _ in range(3)], ceiling=0.98
    )
    assert 0.0 < report["epsilon"] <= 0.98
    assert float(model.veto_epsilon("vision")) == pytest.approx(report["epsilon"])
    model.memory.close()


def test_budget_fits_the_blackwell_target(tmp_path):
    model, cfg = _model(tmp_path)
    report = budget_report(model, batch_size=4, cfg=cfg)
    assert report["fits"] == 1.0
    assert 0.0 < report["training_gib"] < 32.0
    assert report["activation_bytes_training"] > 0
    model.memory.close()


def test_action_policy_returns_a_legal_action(tmp_path, tasks):
    model, cfg = _model(tmp_path)
    observation = tasks["action"].sample(1, "val").x[0].numpy()
    action = model.act(observation)
    assert 0 <= action < cfg.n_actions
    model.memory.close()
