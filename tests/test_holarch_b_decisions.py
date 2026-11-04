"""Decisions B1, B2 and B4: isolated bonus passes, always-on Pass-0 boost.

Decision B3 lives in test_holarch_service.py.

What makes these able to fail:

* the detached pass must leave every shared optical parameter without a
  gradient while the relational basis still trains;
* the Pass-0 prototype boost must stay at full weight, so switching the
  relaxation delta off must make the two relax arms identical;
* the production plate depth must be the value the strict gate measured.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from henri.holarch.config import PRODUCTION_PLATES, HolarchConfig
from henri.holarch.pipeline import HenriHolarch
from henri.holarch.tasks import build_tasks
from henri.holarch.train import Trainer, TrainConfig


def _model(tmp_path, cfg, name="bdec.sqlite3"):
    torch.manual_seed(cfg.seed)
    return HenriHolarch(cfg, backend="sqlite", memory_path=str(tmp_path / name))


# --------------------------------------------------------------- Decision B1
def test_bonus_relational_pass_is_gradient_isolated(tmp_path):
    cfg = HolarchConfig.small()
    model = _model(tmp_path, cfg)
    task = build_tasks(cfg)["relational"]
    batch = task.sample(16, "train")

    # The prototype branch is always on, so the isolation must hold WITH a
    # calibrated bank as well: otherwise the base logits re-open a gradient
    # path into the shared plates.
    model.calibrate_prototypes(
        "relational", [task.sample(64, "train") for _ in range(2)]
    )
    output = model("relational", batch.x, training=True, isolate_branch=True)
    F.cross_entropy(output.task_logits, batch.y).backward()
    assert model.zone_b.plates[0].phase.grad is None
    assert model.zone_b.pool.weight_real.grad is None
    assert model.projector["relational"].phase.weight.grad is None
    assert model.relational_basis.weight_real.grad is not None
    assert float(model.relational_basis.weight_real.grad.abs().max()) > 0.0
    assert model.relational_expert[0].weight.grad is not None

    model.zero_grad(set_to_none=True)
    coupled = model("relational", batch.x, training=True, isolate_branch=False)
    F.cross_entropy(coupled.task_logits, batch.y).backward()
    assert model.zone_b.plates[0].phase.grad is not None
    assert float(model.zone_b.plates[0].phase.grad.abs().max()) > 0.0
    model.memory.close()


def test_relational_only_phase_mask_starts_at_identity(tmp_path):
    cfg = HolarchConfig.small()
    model = _model(tmp_path, cfg)
    task = build_tasks(cfg)["relational"]
    batch = task.sample(8, "train")
    assert tuple(model.relational_phase.shape) == (3, cfg.d_slot)
    assert float(model.relational_phase.detach().abs().max()) == 0.0
    latent = model.encoders["relational"](batch.x)
    with torch.no_grad():
        wave = model._relational_wave(latent)
    with torch.no_grad():
        model.relational_phase.fill_(0.3)
        shifted = model._relational_wave(latent)
    assert float((wave - shifted).abs().max()) > 1e-6
    model.memory.close()


def test_isolated_bonus_schedule_doubles_relational_without_touching_shared(tmp_path):
    cfg = HolarchConfig.small()
    model = _model(tmp_path, cfg)
    tasks = build_tasks(cfg)
    trainer = Trainer(model, tasks, TrainConfig(steps_per_task=1, batch_size=16))
    trainer.fit(steps_per_task=2, order=["vision", "relational"])
    assert trainer._relational_steps == 4
    isolated = [row["isolated"] for row in trainer.history.steps if row["task"] == 3.0]
    assert isolated.count(1.0) == 2
    assert isolated.count(0.0) == 2
    model.memory.close()


# --------------------------------------------------------------- Decision B2
def test_pass0_prototype_boost_stays_on_when_the_delta_is_off(tmp_path):
    cfg = HolarchConfig.small()
    model = _model(tmp_path, cfg)
    task = build_tasks(cfg)["vision"]
    model.calibrate_prototypes("vision", [task.sample(64, "train") for _ in range(2)])
    model.calibrate_gate("vision", [task.sample(32, "val").x for _ in range(2)])
    batch = task.sample(64, "val")

    # With the relaxation delta disabled, B2 says the two arms are the same
    # decision function: the base term is always present regardless of relax.
    model.cfg = cfg.replace(proto_relax_gain=0.0)
    with torch.no_grad():
        off = model("vision", batch.x, training=False, relax=False)
        on = model("vision", batch.x, training=False, relax=True)
    assert torch.allclose(on.task_logits, off.task_logits, atol=1e-6)

    # And the base term is not zero: removing the prototype logits moves it.
    model.cfg = cfg.replace(proto_relax_gain=0.0, prototype_logits=False)
    with torch.no_grad():
        bare = model("vision", batch.x, training=False, relax=False)
    assert float((bare.task_logits - off.task_logits).abs().max()) > 1e-6
    model.cfg = cfg

    # With the delta enabled, the loop must be able to move the decision.
    with torch.no_grad():
        moved = model("vision", batch.x, training=False, relax=True)
    assert float((moved.task_logits - off.task_logits).abs().max()) > 0.0
    model.memory.close()


# --------------------------------------------------------------- Decision B4
def test_production_plate_depth_is_pinned():
    assert PRODUCTION_PLATES == 4
    assert HolarchConfig.production().photonic_layers == PRODUCTION_PLATES
    assert HolarchConfig().photonic_layers == PRODUCTION_PLATES
