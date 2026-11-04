"""Decisions C1, C2 and C4: fail-closed no-hit veto, cartridges, composition.

What makes these able to fail:

* an empty Zone C retrieval must veto a query by default and pass only when
  the caller authorises exploratory ingress;
* the task cartridge must be the identity at initialisation and must move
  only its own task when written;
* the medium profile must actually scale d_slot, not only the wrapper;
* the composite atlas must return the exact composed symbol and refuse a
  key it never stored.
"""

from __future__ import annotations

import numpy as np
import torch

from henri.holarch.config import PRODUCTION_PLATES, HolarchConfig
from henri.holarch.pipeline import HenriHolarch
from henri.holarch.symbolic import CompositeSolver, SymbolCodebook
from henri.holarch.tasks import build_tasks
from henri.holarch.veto import TwoConditionVeto
from henri.holarch.zone_c import ZoneCMemory


def _model(tmp_path, cfg, name="cdec.sqlite3"):
    torch.manual_seed(cfg.seed)
    return HenriHolarch(cfg, backend="sqlite", memory_path=str(tmp_path / name))


# --------------------------------------------------------------- Decision C1
def test_empty_retrieval_is_vetoed_unless_authorized():
    rule = TwoConditionVeto(tau_residual=0.5, tau_chance=0.5)
    assert bool(rule.union([0.0], [1.0], [False])[0])
    assert not bool(rule.union([0.0], [1.0], [False], allow_empty=True)[0])
    assert not bool(rule.union([0.0], [1.0], [True])[0])
    rates = rule.per_condition_rates([0.0], [1.0], [False])
    assert rates["empty_rate"] == 1.0 and rates["union_rate"] == 1.0


def test_pipeline_vetoes_a_query_with_no_engram(tmp_path):
    cfg = HolarchConfig.small()
    model = _model(tmp_path, cfg)
    task = build_tasks(cfg)["vision"]
    batch = task.sample(8, "val")
    with torch.no_grad():
        query = model("vision", batch.x, training=False, relax=False)
        exploratory = model("vision", batch.x, training=True, relax=False)
    # The store is empty, so every inference query is an empty retrieval.
    assert bool(query.veto.all())
    assert not bool(exploratory.veto.all())
    model.memory.close()


# --------------------------------------------------------------- Decision C2
def test_medium_profile_scales_the_wave_dimension():
    cfg = HolarchConfig.medium()
    assert cfg.dim == 1024
    assert cfg.d_slot == 256
    assert cfg.photonic_layers == PRODUCTION_PLATES


def test_task_cartridge_is_identity_then_moves_only_its_task(tmp_path):
    cfg = HolarchConfig.small()
    model = _model(tmp_path, cfg)
    task = build_tasks(cfg)["vision"]
    batch = task.sample(8, "val")
    assert float(model.task_phase.detach().abs().max()) == 0.0
    with torch.no_grad():
        base = model("vision", batch.x, training=False, relax=False).task_logits
        model.task_phase[0].fill_(0.25)
        moved = model("vision", batch.x, training=False, relax=False).task_logits
        other = model("relational", torch.zeros(8, 3, dtype=torch.long), training=False, relax=False)
    assert float((moved - base).abs().max()) > 1e-6
    # The cartridge for task 0 must not leak into the relational task.
    assert float(model.task_phase[3].detach().abs().max()) == 0.0
    assert torch.isfinite(other.task_logits).all()
    model.memory.close()


# --------------------------------------------------------------- Decision C4
def test_composite_solver_reads_an_exact_composition(tmp_path):
    cfg = HolarchConfig.small()
    memory = ZoneCMemory(cfg, backend="sqlite", path=str(tmp_path / "atlas.sqlite3"))
    codebook = SymbolCodebook(cfg.dim)
    solver = CompositeSolver(memory, codebook)
    rows = [
        ((f"d{a}", "add", f"d{b}"), f"d{(a + b) % 10}")
        for a in range(10)
        for b in range(10)
    ]
    assert solver.ingest(rows) == len(rows)
    report = solver.calibrate(rows)
    assert report["coherence_median"] > 0.9
    for a, b in ((3, 4), (9, 8), (0, 0), (7, 2)):
        result = solver.step((f"d{a}", "add", f"d{b}"))
        assert result["answer"] == f"d{(a + b) % 10}"
        assert not result["vetoed"]
    # A key that was never stored must be refused, not guessed.
    missing = solver.step(("d1", "mul", "d1"))
    assert missing["vetoed"] and missing["answer"] is None
    memory.close()
