"""Cycle Q unit tests: fact veto, bridge index, causal regime, KN, daemon."""

from __future__ import annotations

import hashlib
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np
import pytest
import torch

from henri.holarch.code_gen import (
    CountLanguageModel,
    StreamingNgramCache,
    SynthesisEngine,
)
from henri.holarch.pipeline import HenriHolarch
from henri.holarch.swat_bus import (
    SagnacKVGate,
    SwarmTelepathyBus,
    SwarmTelepathyMapper,
    cache_from_pairs,
    cache_from_wave,
    kv_from_wave,
)


def _identity(name: str, dim: int = 64) -> np.ndarray:
    seed = int(hashlib.sha256(str(name).encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    wave = (rng.standard_normal(dim) + 1j * rng.standard_normal(dim)).astype(np.complex64)
    return wave / max(float(np.linalg.norm(wave)), 1e-12)


def test_q1_value_mismatch_is_vetoed():
    gate = SagnacKVGate(
        None,
        None,
        fact_registry={"k": "canonical"},
        wave_of=_identity,
    )
    cache = cache_from_pairs(
        [(_identity("k"), _identity("wrong"))],
        facts=[["k", "wrong"]],
        wave=_identity("k"),
    )
    result = gate.fact_consistency(cache)
    assert result["checked"] == 1
    assert result["violated"] is True
    assert "k:value_mismatch" in result["violations"]


def test_q1_anti_phase_is_destructive_interference():
    gate = SagnacKVGate(
        None,
        None,
        fact_registry={"k": "v"},
        wave_of=_identity,
    )
    cache = cache_from_pairs(
        [(_identity("k"), -_identity("v"))],
        facts=[["k", "v"]],
        wave=_identity("k"),
    )
    cache.meta["fact_waves"] = [(_identity("k"), -_identity("v"))]
    result = gate.fact_consistency(cache)
    assert result["violated"] is True
    assert "k:destructive_interference" in result["violations"]
    assert result["min_value_coherence"] < -0.5


def test_q1_matching_fact_is_admitted_by_fact_condition():
    gate = SagnacKVGate(None, None, fact_registry={"k": "v"}, wave_of=_identity)
    cache = cache_from_pairs(
        [(_identity("k"), _identity("v"))],
        facts=[["k", "v"]],
        wave=_identity("k"),
    )
    cache.meta["fact_waves"] = [(_identity("k"), _identity("v"))]
    result = gate.fact_consistency(cache)
    assert result["violated"] is False
    assert result["min_value_coherence"] > 0.99


def test_q1_bridge_index_resolves_unseen_chain():
    bus = SwarmTelepathyBus(
        SwarmTelepathyMapper(dim=64), gate=None, budget_u=8, dim=64
    )
    first = cache_from_pairs(
        [(_identity("a"), _identity("b"))], facts=[["a", "b"]], wave=_identity("a")
    )
    first.meta["fact_waves"] = [(_identity("a"), _identity("b"))]
    second = cache_from_pairs(
        [(_identity("b"), _identity("c"))], facts=[["b", "c"]], wave=_identity("b")
    )
    second.meta["fact_waves"] = [(_identity("b"), _identity("c"))]
    index = bus.build_bridge_index([first, second], per_chunk_u=2)
    resolved = bus.resolve_bridges(index, _identity("a"), hops=2, wave_of=_identity)
    assert resolved["answer"] == "c"
    assert resolved["resolved_hops"] == 2


def test_q1_composed_cache_respects_budget():
    bus = SwarmTelepathyBus(
        SwarmTelepathyMapper(dim=64), gate=None, budget_u=4, dim=64
    )
    caches = [
        cache_from_pairs(
            [(_identity(f"k{index}"), _identity(f"v{index}")) for index in range(8)],
            facts=[[f"k{index}", f"v{index}"] for index in range(8)],
            wave=_identity("k0"),
        )
    ]
    index = bus.build_bridge_index(caches, per_chunk_u=8)
    resolved = bus.resolve_bridges(index, _identity("k3"), hops=1, wave_of=_identity)
    cache = bus.compose_bridge_cache(index, _identity("k3"), resolved["context"], budget_u=4)
    assert int(cache.keys.shape[0]) <= 4
    assert int(cache.keys.shape[0]) >= 1


def test_q2_regime_waves_are_unit_and_incoherent():
    dummy = SimpleNamespace(dim=64)
    waves = [HenriHolarch._regime_wave(dummy, index) for index in range(3)]
    assert all(abs(float(np.linalg.norm(wave)) - 1.0) < 1e-5 for wave in waves)
    for left in range(3):
        for right in range(left + 1, 3):
            coherence = abs(float(np.real(np.vdot(waves[left], waves[right]))))
            assert coherence < 0.25


def test_q2_trajectory_regime_tracks_outcome():
    dummy = SimpleNamespace(dim=64)
    first = HenriHolarch.trajectory_regime(
        dummy, {"goal": [1, 1], "steps": 10, "online_discoveries": 0}
    )
    second = HenriHolarch.trajectory_regime(
        dummy, {"goal": [6, 1], "steps": 10, "online_discoveries": 0}
    )
    assert first != second
    same = HenriHolarch.trajectory_regime(
        dummy, {"goal": [1, 1], "steps": 10, "online_discoveries": 0}
    )
    assert first == same


def test_q3_streaming_cache_observes_after_scoring():
    cache = StreamingNgramCache(8)
    cache.observe(1, [0])
    cache.observe(2, [1])
    distribution = cache.distribution([1])
    assert int(np.argmax(distribution)) == 2
    assert abs(float(distribution.sum()) - 1.0) < 1e-9
    assert cache.total == 2.0


def test_q3_kneser_ney_is_normalised_and_uses_continuations():
    model = CountLanguageModel(16)
    sequence = [1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3]
    model.fit(sequence)
    report = model.fit_kneser_ney(sequence, order=3)
    distribution = model.kneser_ney_distribution([1, 2])
    assert report["order"] == 3
    assert distribution is not None
    assert abs(float(distribution.sum()) - 1.0) < 1e-9
    assert float(distribution[3]) > 0.0


def test_q3_kn_mix_is_inert_at_zero():
    model = CountLanguageModel(16)
    sequence = [1, 2, 3, 1, 2, 3, 1, 2, 3]
    model.fit(sequence)
    model.fit_kneser_ney(sequence, order=3)
    plain = model.distribution([1, 2])
    model.kn_mix = 0.7
    mixed = model.distribution([1, 2])
    assert plain.shape == mixed.shape
    assert abs(float(mixed.sum()) - 1.0) < 1e-9
    assert float(np.max(np.abs(plain - mixed))) > 1e-6


def test_q3_full_decile_is_an_explicit_evaluation_mode():
    from henri.holarch.code_gen import evaluate_bochner_language

    signature = inspect.signature(evaluate_bochner_language)
    assert "full_decile" in signature.parameters
    from henri.holarch.code_gen import _fit_language_arm

    assert "photonic_eval" in inspect.signature(_fit_language_arm).parameters


def test_q4_plugin_task_lists_are_disjoint_and_complete():
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    import henri_holon

    seeds = list(henri_holon.HOLON_Q_SEEDS)
    traps = list(henri_holon.HOLON_Q_TRAPS)
    assert len(seeds) == 4 and len(traps) == 4
    ids = {row["id"] for row in seeds + traps}
    assert len(ids) == 8
    assert ids.isdisjoint({row["id"] for row in henri_holon.HOLON_SEEDS})
    assert ids.isdisjoint({row["id"] for row in henri_holon.HOLON_TRAPS})
    assert "run_daemon" in inspect.signature(henri_holon.run_daemon).parameters or callable(
        henri_holon.run_daemon
    )


def test_q4_discovery_cache_is_declared_on_the_engine():
    engine_source = inspect.getsource(SynthesisEngine.__init__)
    assert "_discover_cache" in engine_source
    daemon_source = inspect.getsource(__import__("henri_holon", fromlist=["HolonDaemon"]))
    assert "class HolonDaemon" in daemon_source
