"""Cycle I decisions: immutable vocabulary lock, typed N-stage DAG synthesis,
Cul-de-Sac horizon traps, multi-package indexing and the unified agent turn.

These are unit-level checks of the mechanisms. The measured gates live in
receipts/holarch/i1_vocab_lock.json through i4_agent_scale.json.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from henri.holarch.code_gen import (
    Plan,
    PlanStore,
    TypeCompatibilityGate,
    _compose_plans_many,
)
from henri.holarch.code_lang import (
    CodeTextCorpus,
    VocabularyLedger,
    build_tokenizer,
)
from henri.holarch.config import HolarchConfig
from henri.holarch.model import SensorWaveCodec, build_cul_de_sac_traps
from henri.holarch.server import SidecarService, verify_receipt
from henri.holarch.symbolic import CodeGraphIndexer

CORE04 = 'def receipt_digest(receipt):\n    return str(receipt["sha256"])\n'
CORE08 = "def caller_names(edges):\n    return sorted({edge.source for edge in edges})\n"
CORE10 = "def mean_of(values):\n    return float(sum(values) / max(len(values), 1))\n"
CORE11 = "def clamp_unit(value):\n    return max(0.0, min(1.0, float(value)))\n"
SHIFT11 = "def sequence_length(tokens):\n    return int(len(tokens))\n"


def _corpus(texts) -> CodeTextCorpus:
    return CodeTextCorpus(texts=list(texts), names=[f"{index}.py" for index in range(len(texts))])


def _plan(name: str, source: str) -> Plan:
    return Plan(name=name, intent="", source=source)


def test_vocabulary_lock_pins_ids_and_appends():
    before = _corpus(
        [
            "def alpha(value):\n    return value + 1\n",
            "def beta(rows):\n    return len(rows)\n",
        ]
    )
    base = build_tokenizer(before, vocab_size=1150)
    assert len(base.identifiers) >= 2
    lock = base.ledger.truncated(1, vocab_size=1150)
    after = _corpus(
        [
            "def alpha(value):\n    return value + 1\n",
            "def beta(rows):\n    return len(rows)\n",
            "def gamma(values):\n    return zeta_probe(values)\n",
        ]
    )
    pinned = build_tokenizer(before, vocab_size=1150, vocab_lock=lock)
    grown = build_tokenizer(after, vocab_size=1150, vocab_lock=lock)
    for name in lock.identifiers:
        assert grown.stoi[name] == pinned.stoi[name]
    assert len(grown.identifiers) > len(lock.identifiers)
    tail = grown.identifiers[len(lock.identifiers) :]
    first_tail = lock.identifier_base + len(lock.identifiers)
    assert [grown.stoi[name] for name in tail] == list(range(first_tail, first_tail + len(tail)))
    for text in after.texts:
        assert grown.decode(grown.encode(text)) == text


def test_vocabulary_ledger_save_load_and_tamper(tmp_path):
    corpus = _corpus(["def alpha(value):\n    return value + 1\n"])
    ledger = build_tokenizer(corpus, vocab_size=1150).ledger
    saved = ledger.save(tmp_path / "lock.json")
    assert saved["sha256"]
    loaded = VocabularyLedger.load(tmp_path / "lock.json")
    assert loaded.digest() == ledger.digest()
    assert loaded.identifiers == ledger.identifiers
    payload = json.loads((tmp_path / "lock.json").read_text(encoding="utf-8"))
    payload["identifiers"] = list(payload["identifiers"])[:-1]
    (tmp_path / "lock.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        VocabularyLedger.load(tmp_path / "lock.json")


def test_type_gate_vetoes_incompatible_stages():
    gate = TypeCompatibilityGate()
    receipt = _plan("receipt_digest", CORE04)
    mean = _plan("mean_of", CORE10)
    clamp = _plan("clamp_unit", CORE11)
    sequence = _plan("caller_names", CORE08)
    length = _plan("sequence_length", SHIFT11)
    assert gate.check(receipt, mean)["ok"] is False
    assert gate.check(receipt, clamp)["ok"] is False
    assert gate.check(mean, receipt)["ok"] is False
    assert gate.check(sequence, receipt)["ok"] is False
    assert gate.check(sequence, mean)["ok"] is True
    # A string is a sized sequence, so digest -> length is legal.
    assert gate.check(receipt, length)["ok"] is True
    verdict = gate.check_chain([receipt, mean])
    assert verdict["ok"] is False
    assert verdict["reason"].startswith("type mismatch")


def test_three_stage_composition_splices_and_executes():
    plans = [
        _plan("core08", CORE08),
        _plan("shift11", SHIFT11),
        _plan("core11", CORE11),
    ]
    source = _compose_plans_many(
        plans, {"func": "tri_test", "args": ["links"], "numbers": [0.0, 2.5]}
    )
    assert source is not None
    assert "step_1" in source and "step_2" in source
    scope: dict = {}
    exec(source, scope)
    assert callable(scope["tri_test"])

    class _E:
        def __init__(self, source_name):
            self.source = source_name

    assert scope["tri_test"]([_E("b"), _E("a"), _E("b")]) == 2.0


def test_cul_de_sac_traps_are_geometrically_valid():
    from henri.holarch.tasks import _Grid

    world = _Grid(size=7, seed=0)
    traps = build_cul_de_sac_traps()
    assert len(traps) == 8
    for trap in traps:
        assert all(bool(value) for value in trap["invariants"].values())
        walls = trap["walls"]
        assert not walls[trap["pocket"]]
        path = world._bfs(walls, trap["agent"], trap["goal"])
        assert path is not None
        assert int(path[0]) == int(trap["bypass_action"])
        assert int(trap["bypass_action"]) != int(trap["trap_action"])


def test_batched_sensor_shift_matches_the_single_shift():
    codec = SensorWaveCodec(1024, 4)
    generator = np.random.default_rng(3)
    raw = generator.standard_normal(1024) + 1j * generator.standard_normal(1024)
    wave = torch.from_numpy(raw / np.linalg.norm(raw)).reshape(1, -1).to(torch.complex64)
    singles = torch.cat([codec.shift(wave, action) for action in range(4)], dim=0)
    batch = codec.shift_batch(wave, [0, 1, 2, 3])
    assert torch.allclose(singles, batch, atol=1e-6)


def test_split_stages_orders_the_discourse():
    stages = PlanStore.split_stages(
        "first returns a list of names then returns the number of tokens and then clips it"
    )
    assert len(stages) == 3
    assert stages[0].startswith("returns a list")
    assert "clips" in stages[2]


def test_agent_step_uniform_response_and_receipt(tmp_path):
    service = SidecarService(
        cfg=HolarchConfig.small(),
        backend="sqlite",
        memory_path=str(tmp_path / "agent.sqlite3"),
        autoload=False,
    )
    try:
        response = service.agent_step(
            "verify", {"claim": "def identity(value):\n    return value\n", "kind": "code"}
        )
        assert response["status"] == "supported"
        assert response["abstained"] is False
        assert set(response).issuperset(
            {"status", "abstained", "output", "epistemic_receipt"}
        )
        receipt = response["epistemic_receipt"]
        assert verify_receipt(receipt)
        assert receipt["parent"] == "0" * 64
        assert "checkpoint_sha256" in receipt
        with pytest.raises(ValueError):
            service.agent_step("warp", {})
    finally:
        service.close()


def test_multi_package_index_and_delta_parity(tmp_path):
    (tmp_path / "pkg_a").mkdir()
    (tmp_path / "pkg_b").mkdir()
    (tmp_path / "pkg_a" / "mod.py").write_text(
        "from pkg_b.mod import beta\n\ndef alpha(value):\n    return beta(value) + 1\n",
        encoding="utf-8",
    )
    (tmp_path / "pkg_b" / "mod.py").write_text(
        "def beta(value):\n    return value - 1\n", encoding="utf-8"
    )
    indexer = CodeGraphIndexer(None)
    report = indexer.index_paths(tmp_path, ["pkg_a", "pkg_b"], package="workspace")
    assert report.files == 2
    assert any(edge.target.endswith("mod:beta") for edge in indexer.graph.edges)
    reference = CodeGraphIndexer(None)
    reference.index_paths(tmp_path, ["pkg_a", "pkg_b"], package="workspace")
    assert [
        (edge.source, edge.operator, edge.target) for edge in indexer.graph.edges
    ] == [(edge.source, edge.operator, edge.target) for edge in reference.graph.edges]
    edited = "from pkg_b.mod import beta\n\ndef alpha(value):\n    return beta(value) * 2\n"
    indexer.update_file(tmp_path, tmp_path / "pkg_a" / "mod.py", source_text=edited)
    again = CodeGraphIndexer(None)
    again.index_paths(tmp_path, ["pkg_a", "pkg_b"], package="workspace")
    assert [
        (edge.source, edge.operator, edge.target) for edge in indexer.graph.edges
    ] == [(edge.source, edge.operator, edge.target) for edge in again.graph.edges]

def test_agent_step_over_rest_and_mcp(tmp_path):
    import asyncio

    from fastapi.testclient import TestClient

    from henri.holarch import mcp_server
    from henri.holarch.server import build_app

    service = SidecarService(
        cfg=HolarchConfig.small(),
        backend="sqlite",
        memory_path=str(tmp_path / "transports.sqlite3"),
        autoload=False,
    )
    payload = {"claim": "def identity(value):\n    return value\n", "kind": "code"}
    try:
        client = TestClient(build_app(service))
        body = client.post(
            "/v1/holarch/agent_step",
            json={"action": "verify", "payload": payload, "namespace": "test"},
        ).json()
        assert body["status"] == "supported"
        assert verify_receipt(body["epistemic_receipt"])
        mcp_server.set_service(service)
        try:
            response = asyncio.run(
                mcp_server.mcp._tool_manager.call_tool(
                    "holarch_agent_step",
                    {"action": "verify", "payload": payload, "namespace": "test"},
                )
            )
        finally:
            mcp_server.set_service(None)
        assert response["status"] == "supported"
        assert verify_receipt(response["epistemic_receipt"])
        assert response["epistemic_receipt"]["parent"] == body["epistemic_receipt"]["sha256"]
    finally:
        service.close()

