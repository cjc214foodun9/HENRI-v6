"""Cycle P: beyond-context telepathy, the telepathic AST binder, the gate's
binding condition, and the discovery rollback contract.

The tests are unit-level: they assert the invariants the Cycle P receipts
depend on, not a benchmark number.
"""

from __future__ import annotations

import numpy as np
import pytest

from henri.holarch.code_gen import (
    build_telepathic_program,
    telepathic_reference,
)
from henri.holarch.pipeline import _operator_glyphs
from henri.holarch.swat_bus import (
    SagnacKVGate,
    SwarmTelepathyBus,
    SwarmTelepathyMapper,
    cache_from_pairs,
    kv_from_wave,
)


def _wave(dim: int = 1024, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    value = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    return (value / np.linalg.norm(value)).astype(np.complex64)


def _bus(budget: int = 32) -> SwarmTelepathyBus:
    mapper = SwarmTelepathyMapper(dim=1024)
    mapper.set_shared_frame()
    return SwarmTelepathyBus(mapper, gate=None, budget_u=budget, dim=1024)


def _chunk(pairs, source="chunk", wave=None):
    return cache_from_pairs(
        [(np.asarray(a), np.asarray(b)) for a, b in pairs],
        source=source,
        task="language",
        wave=wave,
    )


def test_cache_from_pairs_topic_wave_is_the_first_key():
    first, second = _wave(seed=1), _wave(seed=2)
    cache = _chunk([(first, second)])
    assert cache.keys.shape == (1, 4, 32)
    assert np.allclose(cache.wave, first / np.linalg.norm(first), atol=1e-6)


def test_compact_long_horizon_resolves_three_hop_chain():
    bus = _bus()
    k, m1, m2, v = (_wave(seed=10 + index) for index in range(4))
    chunks = [
        _chunk([(k, m1)], source="c0"),
        _chunk([(m1, m2)], source="c1"),
        _chunk([(m2, v)], source="c2"),
    ]
    values = {"m1": m1, "m2": m2, "v": v}

    def resolver(context):
        best = (-2.0, "")
        for name, wave in values.items():
            probe = kv_from_wave(wave)[1][0].reshape(-1).numpy()
            cosine = float(
                np.dot(context, probe)
                / (np.linalg.norm(context) * np.linalg.norm(probe) + 1e-12)
            )
            if cosine > best[0]:
                best = (cosine, name)
        return None if not best[1] else (values[best[1]], best[1])

    report = bus.compact_long_horizon(
        chunks, k, per_chunk_u=8, total_budget_u=32, hops=3, resolver=resolver,
        session_id="s3",
    )
    assert report["admitted_chunks"] == 3
    assert [row["resolved"] for row in report["chain"]] == ["m1", "m2", "v"]
    assert report["compacted_slots_u"] <= 32
    assert report["receiver_prefill_tokens"] == 0
    assert report["swat_kv_digest"] == report["cache"].digest


def test_compact_long_horizon_vetoes_a_scrambled_chunk():
    bus = _bus()
    k, m1, v = (_wave(seed=30 + index) for index in range(3))
    good = _chunk([(k, m1), (m1, v)], source="good")
    bad = _chunk([(k, v)], source="bad", wave=(good.wave * np.exp(1j * 0.7)))
    bus.gate = _AdmitUnlessScrambled()
    report = bus.compact_long_horizon(
        [good, bad], k, per_chunk_u=8, total_budget_u=32, hops=1, session_id="s4"
    )
    assert report["vetoed_chunks"] == ["bad"]
    assert report["admitted_chunks"] == 1


class _AdmitUnlessScrambled:
    """A stand-in for the gate's binding condition, without a model."""

    def check(self, cache):
        gate = SagnacKVGate(None, None)
        coherence = gate.binding(cache)
        admitted = coherence >= 0.85
        return {
            "admitted": bool(admitted),
            "reason": "ok" if admitted else "binding",
            "s_max": 1.0,
            "delta_sagnac": 0.0,
            "binding_coherence": float(coherence),
        }


def test_gate_binding_is_one_for_a_consistent_cache():
    gate = SagnacKVGate(None, None)
    wave = _wave(seed=7)
    cache = _chunk([(wave, _wave(seed=8))])
    assert gate.binding(cache) > 0.99


def test_gate_binding_falls_for_a_substituted_wave():
    gate = SagnacKVGate(None, None)
    wave = _wave(seed=9)
    cache = _chunk([(wave, _wave(seed=10))])
    scrambled = _chunk(
        [(wave, _wave(seed=10))],
        wave=wave * np.exp(1j * np.linspace(0.0, 3.0, wave.size)).astype(np.complex64),
    )
    assert gate.binding(scrambled) < 0.85
    assert gate.binding(cache) > gate.binding(scrambled)


def test_telepathic_program_is_guard_normalized_and_matches_reference():
    source = build_telepathic_program("tele_probe", scale=4.0, shift=2.0, low=0.0, high=3.0)
    assert "if value is None:" in source
    scope: dict = {}
    exec(source, scope)
    function = scope["tele_probe"]
    for value in (0.0, 0.5, 1.0, -2.0, 3.25, None):
        assert function(value) == telepathic_reference(
            value, scale=4.0, shift=2.0, low=0.0, high=3.0
        )


def test_telepathic_program_clamps_into_the_bound():
    source = build_telepathic_program("tele_clamp", scale=10.0, shift=100.0, low=0.0, high=2.5)
    scope: dict = {}
    exec(source, scope)
    assert scope["tele_clamp"](0.0) == 2.5
    assert scope["tele_clamp"](-5.0) == 2.5


def test_operator_glyphs_are_three_distinct_8x8_fields():
    glyphs = _operator_glyphs()
    assert len(glyphs) == 3
    assert all(np.asarray(glyph).shape == (8, 8) for glyph in glyphs)
    assert not np.array_equal(glyphs[0], glyphs[1])
    assert not np.array_equal(glyphs[1], glyphs[2])


def test_p2_dataset_places_each_hop_in_its_own_chunk():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "pcycle", str(__import__("pathlib").Path(__file__).resolve().parents[1] / "scripts" / "run_holarch_p_cycle.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    chunks, queries = module._p2_dataset(queries=2, facts_per_chunk=8)
    assert len(chunks) == 8
    first = queries[0]
    doc0 = dict(chunks[0]["facts"])
    doc1 = dict(chunks[2]["facts"])
    doc3 = dict(chunks[6]["facts"])
    assert doc0[first["key"]] == first["mid1"]
    assert doc1[first["mid1"]] == first["mid2"]
    assert doc3[first["mid2"]] == first["answer"]


def test_p2_bm25_baseline_cannot_follow_a_cross_chunk_chain():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "pcycle2", str(__import__("pathlib").Path(__file__).resolve().parents[1] / "scripts" / "run_holarch_p_cycle.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    chunks, queries = module._p2_dataset(queries=2, facts_per_chunk=32)
    answer = module._bm25_answer(chunks, queries[0]["key"], budget_tokens=32)
    assert answer != queries[0]["answer"]


def test_holon_plugin_declares_four_seeds_and_four_traps():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "henri_holon.py"
    spec = importlib.util.spec_from_file_location("henri_holon_test", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert len(module.HOLON_SEEDS) == 4
    assert len(module.HOLON_TRAPS) == 4
    assert all(task["kind"] == "seed" for task in module.HOLON_SEEDS)
    assert all(task["kind"] == "trap" for task in module.HOLON_TRAPS)


def test_telepathy_request_model_accepts_both_modes():
    from henri.holarch.server import TelepathyRequest

    ingest = TelepathyRequest(
        mode="ingest_chunks",
        session_id="s",
        chunks=[{"source": "c", "facts": [["a", "b"]]}],
        query="a",
    )
    assert ingest.mode == "ingest_chunks"
    assert ingest.budget_u == 32 and ingest.per_chunk_u == 8
    query = TelepathyRequest(mode="query", session_id="s", query="a")
    assert query.mode == "query"


def test_discover_request_model_defaults():
    from henri.holarch.server import DiscoverRequest

    request = DiscoverRequest(intent="write a function")
    assert request.tests == ""
    assert request.request == {}
    assert request.namespace == "default"

def test_cache_from_pairs_keeps_one_slot_per_fact():
    first, second, third = (_wave(seed=40 + index) for index in range(3))
    cache = _chunk([(first, second), (second, third)])
    assert cache.keys.shape[0] == 2
    assert cache.spans.shape == (2, 2)
    assert cache.digest == cache.digest


def test_long_horizon_report_carries_the_audit_fields():
    bus = _bus()
    wave = _wave(seed=50)
    chunk = _chunk([(wave, _wave(seed=51))], source="single")
    report = bus.compact_long_horizon(
        [chunk], wave, per_chunk_u=4, total_budget_u=8, hops=1, session_id="audit"
    )
    for field in (
        "session_id",
        "admitted_chunks",
        "vetoed_chunks",
        "compacted_slots_u",
        "swat_kv_digest",
        "receiver_prefill_tokens",
        "chunks",
        "chain",
    ):
        assert field in report
    assert report["receiver_prefill_tokens"] == 0
    assert bus.chain_valid()


def test_telepathy_sessions_are_isolated_by_id():
    bus = _bus()
    first = _chunk([(_wave(seed=60), _wave(seed=61))], source="a")
    second = _chunk([(_wave(seed=62), _wave(seed=63))], source="b")
    bus.compact_long_horizon([first], _wave(seed=60), session_id="s-a")
    bus.compact_long_horizon([second], _wave(seed=62), session_id="s-b")
    assert set(bus.entries) >= {"s-a", "s-b"}
    assert bus.entries["s-a"].digest != bus.entries["s-b"].digest

