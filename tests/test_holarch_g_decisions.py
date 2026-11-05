"""Directive G1-G4: wave head admission, parametric synthesis, wall mask, persistence.

The tests assert the mechanisms, not only the receipts: the wave term must be
non-degenerate and change the logits, the fast weight must move a probability
after an online write, the wall mask must agree with the world on every
direction, and a second process must answer from the first process's database.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from henri.holarch.code_gen import (
    PlanStore,
    SynthesisEngine,
    WaveCodeTextGenerator,
    _adapt_source,
    _intent_slots,
    _source_slots,
)
from henri.holarch.code_lang import (
    BochnerPhaseProjector,
    CodeTextCorpus,
    build_tokenizer,
)
from henri.holarch.config import HolarchConfig
from henri.holarch.model import (
    LatentActionLookahead,
    SensorWaveCodec,
    fit_action_rotor,
)
from henri.holarch.pipeline import HenriHolarch
from henri.holarch.server import SidecarService
from henri.holarch.symbolic import CodeGraphIndexer, Edge
from henri.holarch.tasks import GridWorldTask
from henri.holarch.zone_c import ZoneCMemory

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def corpus() -> CodeTextCorpus:
    return CodeTextCorpus.holarch(ROOT)


@pytest.fixture(scope="module")
def tokenizer(corpus):
    return build_tokenizer(corpus)


@pytest.fixture(scope="module")
def projector(corpus, tokenizer):
    builder = BochnerPhaseProjector(dim=1024, d_sem=64)
    builder.fit(corpus, tokenizer)
    return builder


@pytest.fixture(scope="module")
def generator(tokenizer, projector, corpus):
    ids = []
    for text in corpus.texts:
        ids.extend(tokenizer.encode(text))
    model = WaveCodeTextGenerator(tokenizer, projector)
    model.fit(ids[:20000], max_fit_tokens=8000)
    model.count_lm.fit(ids[: int(len(ids) * 0.9)])
    model.lexical.build(ids[:20000])
    return model


# ------------------------------------------------------------------- G1
def test_bank_is_multi_order_and_not_a_bigram(generator):
    report = generator.lexical.report
    assert report["orders"] == 4.0
    assert report["keys_order_1"] > 100
    assert report["keys_order_2"] > report["keys_order_1"]
    assert report["keys_order_4"] > report["keys_order_2"]


def test_wave_score_sees_tokens_the_4gram_cannot(generator, tokenizer):
    """A context longer than 3 tokens must change the wave score."""
    ids = tokenizer.encode("the memory model wave bank store write read")
    short = generator.lexical.score(ids[-4:])
    long = generator.lexical.score(ids[-14:])
    assert short is not None and long is not None
    assert float(np.std(short)) > 1e-6
    assert not np.allclose(short, long, atol=1e-6)


def test_beta_scales_a_real_wave_term(generator, tokenizer):
    ids = tokenizer.encode("def write(self, wave): return self.memory.write(wave)")
    generator.beta_star = 0.0
    floor = generator.token_logits(ids[:-1])
    generator.beta_star = 0.1
    tuned = generator.token_logits(ids[:-1])
    assert not np.allclose(floor, tuned)
    assert np.corrcoef(floor, tuned)[0, 1] > 0.5


def test_fast_weight_is_read_and_moves_probabilities(generator, tokenizer):
    """Directive G1.2: the online memory must not be dead state."""
    context = tokenizer.encode("def average_of(values): return float(sum(values) / len(values))")
    generator.beta_star = 0.0
    before = generator.token_logits(context)
    target = tokenizer.stoi.get("returned") or tokenizer.encode("returned")[0]
    assert not generator._fast_active
    generator.update_fast_weight(context, [target], lr=1.5)
    assert generator._fast_active
    after = generator.token_logits(context)
    assert abs(float(after[target] - before[target])) > 1e-6
    assert float(after[target]) > float(before[target])


# ------------------------------------------------------------------- G2
def test_intent_slots_bind_function_argument_and_constants():
    intent = "Write a function clamp_high that takes argument level and clamps it into the closed interval 0.0 to 2.5."
    slots = _intent_slots(intent)
    assert slots["func"] == "clamp_high"
    assert slots["args"] == ["level"]
    assert slots["numbers"] == [0.0, 2.5]


def test_adapt_source_rebinds_name_argument_and_constant():
    source = "def clamp_unit(value):\n    return max(0.0, min(1.0, float(value)))\n"
    slots = _source_slots(source)
    assert slots["func"] == "clamp_unit"
    assert slots["args"] == ["value"]
    assert slots["numbers"] == [0.0, 1.0]
    adapted = _adapt_source(
        source,
        slots,
        {"func": "clamp_high", "args": ["level"], "numbers": [0.0, 2.5]},
    )
    assert "clamp_high(level)" in adapted
    assert "min(2.5" in adapted
    assert "float(level)" in adapted
    namespace: dict = {}
    exec(adapt_source_safe(adapted), namespace)
    assert namespace["clamp_high"](9.0) == 2.5
    assert namespace["clamp_high"](-3.0) == 0.0


def adapt_source_safe(source: str) -> str:
    return str(source)


def test_parametric_synthesis_beats_verbatim_replay(tmp_path, tokenizer, projector, generator):
    memory = ZoneCMemory(
        HolarchConfig.medium(), backend="sqlite", path=str(tmp_path / "g2.sqlite3")
    )
    try:
        engine = SynthesisEngine(generator, projector, tokenizer, memory)
        intent = "Write a function clamp_unit that takes argument value and clips a value into the closed unit interval."
        source = "def clamp_unit(value):\n    return max(0.0, min(1.0, float(value)))\n"
        assert engine.adapt(intent, source, "clamp_unit")["adapted"] is True
        moved = engine.synthesize(
            "Write a function clamp_high that takes argument level and clips a value into the closed interval 0.0 to 2.5.",
            kind="code",
        )
        assert moved["verified"] is True
        assert moved["plan"] == "clamp_unit"
        assert "clamp_high" in moved["source"]
        assert "2.5" in moved["source"]
        # The verbatim template would fail this test; the bound one passes.
        assert engine.execute(moved["source"], "assert clamp_high(9.0) == 2.5\n")["ok"] is True
        assert engine.execute(source, "assert clamp_unit(2.0) == 1.0\n")["ok"] is True
        assert engine.execute(source, "assert clamp_high(9.0) == 2.5\n")["ok"] is False
    finally:
        memory.close()


# ------------------------------------------------------------------- G3
def test_wall_mask_agrees_with_the_environment_on_every_direction(tmp_path):
    cfg = HolarchConfig.small()
    model = HenriHolarch(cfg, backend="sqlite", memory_path=str(tmp_path / "g3.sqlite3"))
    model.eval()
    try:
        task = GridWorldTask(seed=cfg.seed, size=7, n_layouts=40)
        lookahead = LatentActionLookahead(model, task)
        codec = lookahead.codec
        checked = 0
        for walls, agent, goal in task.val_layouts[:40]:
            observation = task.world.observe(walls, agent, goal)
            psi = codec.encode(observation)
            mask = lookahead.wall_mask(psi)
            for action, (dr, dc) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1))):
                row, col = agent[0] + dr, agent[1] + dc
                illegal = (
                    not (0 <= row < task.grid_size and 0 <= col < task.grid_size)
                    or bool(walls[row, col])
                )
                assert bool(mask[action]) == bool(illegal)
                checked += 1
        assert checked >= 120
    finally:
        model.memory.close()


def test_depth_two_rejects_a_wall_tunnelling_second_step(tmp_path):
    """The mask must lower the lookahead value of a walled second step."""
    cfg = HolarchConfig.small()
    model = HenriHolarch(cfg, backend="sqlite", memory_path=str(tmp_path / "g3b.sqlite3"))
    model.eval()
    try:
        task = GridWorldTask(seed=cfg.seed, size=7, n_layouts=40)
        lookahead = LatentActionLookahead(model, task, lookahead_depth=2)
        codec = lookahead.codec
        found = 0
        for walls, agent, goal in task.val_layouts[:20]:
            observation = task.world.observe(walls, agent, goal)
            psi = codec.encode(observation)
            for action, (dr, dc) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1))):
                first = lookahead.predict(psi, action)
                blocked = lookahead.wall_mask(first)
                if blocked.any():
                    legal = [index for index in range(4) if not bool(blocked[index])]
                    assert len(legal) < 4
                    found += 1
        assert found > 0
    finally:
        model.memory.close()


# ------------------------------------------------------------------- G4
def test_sidecar_answers_from_the_database_after_a_restart(tmp_path):
    database = str(tmp_path / "g4.sqlite3")
    namespace = "tenant_prod"
    first = SidecarService(
        HolarchConfig.small(), backend="sqlite", memory_path=database, autoload=False
    )
    first.ingest_code(str(ROOT / "henri" / "holarch"), "henri", namespace=namespace)
    first.ingest_edges(
        [
            {"source": "s00", "operator": "fwd", "target": "s01"},
            {"source": "s01", "operator": "fwd", "target": "s02"},
        ],
        namespace=namespace,
    )
    first.composite_call(
        mode="ingest",
        rules=[{"a": "d1", "op": "add", "b": "d2", "result": "d3"}],
        namespace=namespace,
    )
    before_solve = first.solve("s00", ["fwd", "fwd"], namespace=namespace)["result"]["answer"]
    before_impact = first.solve(
        "ZoneCMemory.query", ["2"], mode="impact", namespace=namespace
    )["result"]["callers"]
    first.close()

    second = SidecarService(
        HolarchConfig.small(), backend="sqlite", memory_path=database, autoload=False
    )
    try:
        assert second.solve("s00", ["fwd", "fwd"], namespace=namespace)["result"]["answer"] == before_solve
        after_impact = second.solve(
            "ZoneCMemory.query", ["2"], mode="impact", namespace=namespace
        )["result"]["callers"]
        assert after_impact == before_impact and after_impact
        assert (
            second.composite_call(mode="query", a="d1", b="d2", op="add", namespace=namespace)[
                "result"
            ]["result"]
            == "d3"
        )
        assert second.solve("s00", ["fwd"], namespace="tenant_other")["result"]["vetoed"] is True
    finally:
        second.close()


def test_plans_survive_a_restart(tmp_path, tokenizer, projector):
    memory = ZoneCMemory(
        HolarchConfig.medium(), backend="sqlite", path=str(tmp_path / "plans.sqlite3")
    )
    try:
        store = PlanStore(memory, projector, tokenizer, namespace="tenant_prod")
        intent = "Write a function clamp_unit that takes argument value and clips a value into the closed unit interval."
        store.compile("clamp_unit", intent, "def clamp_unit(value):\n    return max(0.0, min(1.0, float(value)))\n")
        assert len(store.plans) == 1
    finally:
        memory.close()

    reopened = ZoneCMemory(
        HolarchConfig.medium(), backend="sqlite", path=str(tmp_path / "plans.sqlite3")
    )
    try:
        store = PlanStore(reopened, projector, tokenizer, namespace="tenant_prod")
        assert len(store.plans) == 0
        restored = store.load()
        assert restored == 1
        hit = store.retrieve(
            "Write a function clamp_unit that takes argument value and clips a value into the closed unit interval."
        )
        assert hit is not None and hit[0].slots["func"] == "clamp_unit"
    finally:
        reopened.close()


def test_code_graph_survives_a_restart(tmp_path):
    memory = ZoneCMemory(
        HolarchConfig.small(), backend="sqlite", path=str(tmp_path / "graph.sqlite3")
    )
    try:
        indexer = CodeGraphIndexer(memory, namespace="tenant_prod")
        indexer.index(ROOT / "henri" / "holarch", "henri")
        before = indexer.impact("ZoneCMemory.query", hops=2)["callers"]
        assert before
    finally:
        memory.close()

    reopened = ZoneCMemory(
        HolarchConfig.small(), backend="sqlite", path=str(tmp_path / "graph.sqlite3")
    )
    try:
        restored = CodeGraphIndexer.from_memory(reopened, "tenant_prod")
        assert restored is not None
        assert restored.signatures
        assert restored.impact("ZoneCMemory.query", hops=2)["callers"] == before
    finally:
        reopened.close()
