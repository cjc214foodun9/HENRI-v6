"""Cycle M unit gates: prototype-residual cartridges, guard-clause
normalization, online REPULSOR induction, batched Zone C reads, and the
reader thread guard.

The heavy end-to-end measurements live in scripts/run_holarch_m_cycle.py;
these tests pin the units those measurements rest on.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest
import torch

from henri.holarch.code_gen import (
    _composable_body,
    _compiled_plan,
    _function_definition,
    compose_plan_dag,
    run_diverse_dag_battery,
)
from henri.holarch.config import HolarchConfig
from henri.holarch.model import (
    LatentActionLookahead,
    evaluate_closed_loop_action,
    seed_engram_payload,
    seed_engram_rows,
)
from henri.holarch.pipeline import HenriHolarch
from henri.holarch.server import SidecarService
from henri.holarch.tasks import build_tasks
from henri.holarch.zone_c import ZoneCMemory


@pytest.fixture(scope="module")
def cfg():
    return HolarchConfig.small()


@pytest.fixture()
def memory(cfg, tmp_path):
    store = ZoneCMemory(cfg, backend="sqlite", path=str(tmp_path / "zone_c.sqlite3"))
    yield store
    store.close()


def _wave(dim: int, seed: int) -> np.ndarray:
    generator = np.random.default_rng(int(seed))
    raw = generator.standard_normal(dim) + 1j * generator.standard_normal(dim)
    return (raw / np.linalg.norm(raw)).astype(np.complex64)


# ------------------------------------------------------- prototype residual
def test_residual_action_rows_round_trip_above_0995():
    rows = [
        {"label": 106, "tier": 3, "wave": _wave(512, 1)},
        {"label": 106, "tier": 1, "wave": _wave(512, 2)},
        {"label": 107, "tier": 3, "wave": _wave(512, 3)},
        {"label": 107, "tier": 1, "wave": _wave(512, 4)},
    ]
    block = seed_engram_payload(rows)
    decoded = seed_engram_rows(block)
    for source, target in zip(rows, decoded):
        assert abs(np.vdot(source["wave"], target["wave"])) >= 0.995


def test_action_prototype_phases_travel_in_the_block():
    rows = [
        {"label": 106, "tier": 3, "wave": _wave(256, 5)},
        {"label": 106, "tier": 1, "wave": _wave(256, 6)},
        {"label": 0, "tier": 1, "wave": _wave(256, 7)},
    ]
    block = seed_engram_payload(rows)
    assert block["format"] == "henri-seed-engrams-v3"
    assert int(block["action_rows"]) == 2
    assert int(block["action_proto_count"]) == 1
    assert int(block["support_rows"]) == 1
    assert int(block["bytes"]) <= 100_000


def test_support_rows_keep_high_coherence():
    rows = [{"label": 0, "tier": 1, "wave": _wave(512, 8)}]
    block = seed_engram_payload(rows)
    decoded = seed_engram_rows(block)
    assert abs(np.vdot(rows[0]["wave"], decoded[0]["wave"])) >= 0.9999


# ----------------------------------------------------------- guard clauses
def _guard_stage():
    return _compiled_plan(
        "safe_mean",
        "mean",
        "def safe_mean(values: list) -> float:\n"
        "    if not values:\n"
        "        return 0.0\n"
        "    total = 0.0\n"
        "    for value in values:\n"
        "        total += value\n"
        "    return total / len(values)\n",
    )


def _bound_stage():
    return _compiled_plan(
        "lower_bound",
        "bound",
        "def lower_bound(value: float, floor: float) -> float:\n"
        "    if value < floor:\n"
        "        return floor\n"
        "    return value\n",
    )


def test_guard_clause_pipeline_executes():
    verdict = compose_plan_dag(
        [_guard_stage(), _bound_stage()],
        {"func": "guard_pipeline", "args": ["values", "floor"], "numbers": []},
    )
    assert verdict["ok"] is True
    scope = {"__builtins__": __builtins__}
    exec(verdict["source"], scope)
    assert scope["guard_pipeline"]([], 1.0) == 1.0
    assert scope["guard_pipeline"]([2.0, 4.0], 1.0) == 3.0


def test_two_leading_guards_are_normalized():
    stage = _compiled_plan(
        "two_guards",
        "two",
        "def two_guards(value: float, high: float) -> float:\n"
        "    if value < 0.0:\n"
        "        return 0.0\n"
        "    if value > high:\n"
        "        return high\n"
        "    return value\n",
    )
    verdict = compose_plan_dag(
        [stage],
        {"func": "two_guards_probe", "args": ["value", "high"], "numbers": []},
    )
    assert verdict["ok"] is True
    scope = {"__builtins__": __builtins__}
    exec(verdict["source"], scope)
    assert scope["two_guards_probe"](-1.0, 2.0) == 0.0
    assert scope["two_guards_probe"](3.0, 2.0) == 2.0
    assert scope["two_guards_probe"](1.0, 2.0) == 1.0


def test_mid_loop_return_is_refused():
    definition = _function_definition(
        "def mid(values: list) -> float:\n"
        "    for value in values:\n"
        "        if value > 0:\n"
        "            return value\n"
        "    return 0.0\n"
    )
    assert _composable_body(definition) is None


def test_guard_after_a_statement_is_refused():
    definition = _function_definition(
        "def late(values: list) -> float:\n"
        "    total = 0.0\n"
        "    if not values:\n"
        "        return 0.0\n"
        "    return total\n"
    )
    assert _composable_body(definition) is None


def test_guard_with_else_is_refused():
    definition = _function_definition(
        "def guarded(values: list) -> float:\n"
        "    if values:\n"
        "        return 1.0\n"
        "    else:\n"
        "        return 0.0\n"
    )
    assert _composable_body(definition) is None


def test_84_task_guard_suite_passes_with_zero_drift():
    battery = run_diverse_dag_battery(seed=20261105, extended=True, guard_pipelines=True)
    assert battery["total"]["passed"] == battery["total"]["total"] == 84
    assert battery["guards"]["passed"] == battery["guards"]["total"] == 6
    assert battery["traps3"]["refused"] == battery["traps3"]["total"] == 6
    assert abs(float(battery["drift"])) < 1e-12


# ------------------------------------------------------------ online repulsor
def test_discover_escape_registers_a_repulsor_axiom(cfg, memory):
    model = HenriHolarch(cfg, memory=memory)
    task = build_tasks(cfg, seed=int(cfg.seed))["action"]
    lookahead = LatentActionLookahead(
        model, task, transition="shift", lookahead_depth=1
    )
    observation = task.val_pairs[0][0]
    psi = lookahead.codec.encode(observation)
    escape = lookahead.discover_escape(
        psi, [], np.ones(4, dtype=bool), None, memory, namespace="unit_m1"
    )
    assert escape["axiom"]["registered"] is True
    axioms = memory.boundary_axioms("unit_m1", "REPULSOR")
    assert len(axioms) == 1
    assert 0 <= int(escape["action"]) < 4
    assert bool(escape["candidate_mask"][int(escape["action"])])


def test_cycle_score_detects_a_repeated_wave(cfg, memory):
    model = HenriHolarch(cfg, memory=memory)
    lookahead = LatentActionLookahead(model, None, transition="shift", lookahead_depth=1)
    wave = torch.from_numpy(_wave(int(cfg.dim), 11))
    assert lookahead.cycle_score(wave, [wave, wave]) > 0.99


def test_discover_escape_respects_the_legal_mask(cfg, memory):
    model = HenriHolarch(cfg, memory=memory)
    task = build_tasks(cfg, seed=int(cfg.seed))["action"]
    lookahead = LatentActionLookahead(
        model, task, transition="shift", lookahead_depth=1
    )
    psi = lookahead.codec.encode(task.val_pairs[0][0])
    legal = np.array([True, True, False, False])
    escape = lookahead.discover_escape(
        psi, [], legal, None, memory, namespace="unit_m1_legal"
    )
    assert int(escape["action"]) in (0, 1)


# ------------------------------------------------------------- batched reads
def test_context_tensor_batch_matches_per_row(cfg, memory):
    for index in range(6):
        memory.write(_wave(int(cfg.dim), 20 + index), label=index % 10, tier=1)
    queries = np.stack([_wave(int(cfg.dim), 30 + index) for index in range(4)])
    prefix, reads = memory.context_tensor_batch(queries, k=3, task="vision")
    assert prefix.shape == (4, int(cfg.context_tokens), int(cfg.context_dim))
    for index in range(4):
        single, _ = memory.context_tensor(queries[index], k=3, task="vision")
        assert np.allclose(prefix[index].reshape(-1), single.reshape(-1), atol=1e-5)


def test_reader_scope_never_mutates_the_intra_op_pool(cfg, tmp_path):
    """Directive N2: a reader must not touch the process-global OpenMP pool.

    Cycle M pinned torch.set_num_threads(1) inside the read scope. That call
    resets a process-global C++ pool and starved the J4 planners, so the read
    scope now leaves the pool exactly as the deployment configured it.
    """
    service = SidecarService(
        cfg=cfg, backend="sqlite", memory_path=str(tmp_path / "guard.sqlite3"),
        autoload=False, rate_limit_rps=1000.0, rate_limit_burst=1000.0,
    )
    observed = []
    barrier = threading.Barrier(2)

    def reader() -> None:
        with service._read_scope():
            barrier.wait(timeout=10)
            observed.append(int(torch.get_num_threads()))

    original = int(torch.get_num_threads())
    try:
        threads = [threading.Thread(target=reader) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert observed and all(value == original for value in observed)
        assert int(torch.get_num_threads()) == original
    finally:
        service.close()
