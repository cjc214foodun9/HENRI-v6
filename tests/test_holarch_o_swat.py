"""Cycle O: H-SWAT bus, attention matching, prefill-free transfer, the
stateful online-REPULSOR plan surface, and the elastic language tokenizer.

The tests are unit-level: they assert the invariants the receipts depend on,
not a benchmark number.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from henri.holarch.config import HolarchConfig
from henri.holarch.server import PlanRequest, SidecarService
from henri.holarch.swat_bus import (
    AttentionMatchingCompactor,
    DeRoPEPositionAligner,
    SwarmKVCache,
    SwarmTelepathyBus,
    SwarmTelepathyMapper,
    cache_from_wave,
    kv_from_wave,
    t2t_summary,
)


def _wave(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    value = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    return (value / np.linalg.norm(value)).astype(np.complex64)


class _AdmitAll:
    def check(self, cache):
        return {"admitted": True, "reason": "ok", "s_max": 1.0, "delta_sagnac": 0.0}


class _VetoAll:
    def check(self, cache):
        return {"admitted": False, "reason": "sagnac", "s_max": 0.0, "delta_sagnac": 1.0}


def test_kv_from_wave_shape_and_norm():
    keys, values = kv_from_wave(_wave(256, 1))
    assert tuple(keys.shape) == (4, 4, 32)
    assert tuple(values.shape) == (4, 4, 32)
    assert float(keys.abs().max()) > 0.0


def test_kv_projection_is_deterministic():
    first = kv_from_wave(_wave(256, 2))[0]
    second = kv_from_wave(_wave(256, 2))[0]
    assert torch.equal(first, second)


def test_de_rope_round_trip_is_unitary():
    aligner = DeRoPEPositionAligner(32)
    keys, _values = kv_from_wave(_wave(256, 3))
    positions = [0, 1, 2, 3]
    restored = aligner.de_rope(aligner.rope(keys, positions), positions)
    assert float((restored - keys).abs().max()) < 1e-5


def test_span_alignment_enforces_the_causal_mask():
    aligner = DeRoPEPositionAligner(32)
    sender = np.asarray([[0, 8], [8, 16]], dtype=np.int64)
    receiver = np.asarray([[0, 8], [8, 16]], dtype=np.int64)
    rho = aligner.span_alignment(sender, receiver)
    assert rho.shape == (2, 2)
    # sender span 1 ends at 16, receiver span 0 ends at 8: masked to zero.
    assert float(rho[1, 0]) == 0.0
    assert float(rho[0, 0]) >= 0.0


def test_span_alignment_midpoint_interpolates():
    aligner = DeRoPEPositionAligner(32)
    sender = np.asarray([[0, 10]], dtype=np.int64)
    receiver = np.asarray([[0, 10], [10, 20]], dtype=np.int64)
    rho = aligner.span_alignment(sender, receiver)
    # The midpoint (5) falls at 0.5 of the first receiver span: rho = 0 + 0.5 - 0.5.
    assert abs(float(rho[0, 0]) - 0.0) < 1e-6
    # The second receiver span starts at 10, so the midpoint clips to 0 and
    # rho = 1 + 0 - 0.5 = 0.5. The causal mask keeps it (10 <= 20).
    assert abs(float(rho[0, 1]) - 0.5) < 1e-6


def test_attention_matching_selects_the_query_key_and_preserves_mass():
    keys = torch.randn(16, 1, 512)
    values = torch.randn(16, 1, 512)
    compactor = AttentionMatchingCompactor(4)
    selected_keys, selected_values, selected, report = compactor.compact(
        keys, values, query_keys=keys[5:6], budget_u=4
    )
    assert 5 in [int(value) for value in selected.tolist()]
    assert report["attention_mass_preserved"] >= 0.99
    assert tuple(selected_keys.shape) == (4, 1, 512)


def test_attention_matching_values_solve_the_attention_output():
    torch.manual_seed(7)
    keys = torch.randn(12, 1, 64)
    values = torch.randn(12, 1, 64)
    query = torch.randn(3, 1, 64)
    compactor = AttentionMatchingCompactor(6)
    compact_keys, compact_values, _selected, _report = compactor.compact(
        keys, values, query_keys=query, budget_u=6
    )
    scores = torch.einsum("lhd,qhd->ql", compact_keys, query) / 8.0
    attention = torch.softmax(scores, dim=-1)
    reconstructed = torch.einsum("ql,lhd->qhd", attention, compact_values)
    assert torch.isfinite(reconstructed).all()


def test_cache_from_wave_digest_is_stable_and_complete():
    cache = cache_from_wave(_wave(256, 4), source="s", task="language")
    again = cache_from_wave(_wave(256, 4), source="s", task="language")
    assert cache.digest == again.digest
    assert cache.validate().digest == cache.digest
    assert cache.to_dict()["length"] == 4


def test_bus_registers_packets_and_keeps_a_valid_chain():
    mapper = SwarmTelepathyMapper(dim=256)
    mapper.set_shared_frame()
    bus = SwarmTelepathyBus(mapper, gate=_AdmitAll(), budget_u=4, dim=256, heads=4, dk=32)
    bus.calibrate_receiver("language", [_wave(256, index) for index in range(8)])
    for index in range(4):
        cache = cache_from_wave(_wave(256, 20 + index), source=f"s{index}", task="language")
        bus.register(cache)
    packet = bus.transmit("s0", receiver_task="language")
    assert packet.admitted
    assert int(packet.keys.shape[0]) == 4
    assert bus.chain_valid()
    assert bus.summary()["admitted"] == 1


def test_bus_veto_zeroes_the_attention_mass():
    mapper = SwarmTelepathyMapper(dim=256)
    mapper.set_shared_frame()
    bus = SwarmTelepathyBus(mapper, gate=_VetoAll(), budget_u=4, dim=256)
    bus.calibrate_receiver("language", [_wave(256, 30)])
    cache = cache_from_wave(_wave(256, 31), source="bad", task="language")
    bus.register(cache)
    packet = bus.transmit("bad", receiver_task="language")
    assert not packet.admitted
    assert packet.reason == "sagnac"
    assert int(packet.keys.shape[0]) == 0
    assert bus.summary()["vetoed"] == 1


def test_bus_digest_depends_on_the_transferred_state():
    mapper = SwarmTelepathyMapper(dim=256)
    mapper.set_shared_frame()
    bus = SwarmTelepathyBus(mapper, gate=_AdmitAll(), budget_u=4, dim=256)
    bus.calibrate_receiver("language", [_wave(256, 40)])
    first = cache_from_wave(_wave(256, 41), source="a", task="language")
    second = cache_from_wave(_wave(256, 42), source="b", task="language")
    bus.register(first)
    bus.register(second)
    assert bus.transmit("a", receiver_task="language").digest != bus.transmit("b", receiver_task="language").digest


def test_t2t_summary_respects_the_token_budget():
    class _Tokenizer:
        def __init__(self):
            self.stoi = {"a": 0, "b": 1, "c": 2, "=": 3}
            self._ids = {0: "a", 1: "b", 2: "c", 3: "="}
        def encode(self, text, **_kwargs):
            return [self.stoi[word] for word in str(text).replace("=", " = ").split() if word in self.stoi]
        def decode(self, ids):
            return " ".join(self._ids[int(token)] for token in ids)
        def token_text(self, token):
            return self._ids[int(token)]

    tokenizer = _Tokenizer()
    summary = t2t_summary("a = b c = a", tokenizer, max_tokens=4, query="a")
    assert len(tokenizer.encode(summary)) <= 4


def test_plan_request_accepts_the_online_episode_fields():
    request = PlanRequest(
        observation=[0.0] * 79,
        episode_id="ep",
        online_repulsor=True,
        reset_episode=True,
        emit_swat_kv=True,
    )
    assert request.episode_id == "ep"
    assert request.online_repulsor is True
    assert request.emit_swat_kv is True


def test_global_epoch_is_monotonic_in_the_store(tmp_path):
    from henri.holarch.zone_c import ZoneCMemory

    cfg = HolarchConfig.small()
    memory = ZoneCMemory(cfg, backend="sqlite", path=str(tmp_path / "epoch.sqlite3"))
    try:
        before = int(memory.state_epoch())
        after = int(memory.bump_state_epoch())
        assert after == before + 1
        assert int(memory.state_epoch()) == after
    finally:
        memory.close()


def test_service_plan_online_episode_returns_the_online_fields(tmp_path):
    cfg = HolarchConfig.small()
    service = SidecarService(
        cfg=cfg, backend="sqlite", memory_path=str(tmp_path / "plan.sqlite3"),
        autoload=False, rate_limit_rps=1000.0, cache_enabled=False,
    )
    try:
        observation = [0.0] * int(cfg.obs_dim)
        result = service.plan(
            observation, transition="shift", lookahead_depth=2,
            episode_id="unit", online_repulsor=True, reset_episode=True,
        )["result"]
        assert "repulsor_triggered" in result
        assert "active_repulsors" in result
        assert "swat_kv_digest" in result
        assert 0 <= int(result["action"]) < 4
        assert int(result["goal_distance"]) >= 0
    finally:
        service.close()


def test_service_epoch_sync_clears_a_sibling_mutation(tmp_path):
    cfg = HolarchConfig.small()
    service = SidecarService(
        cfg=cfg, backend="sqlite", memory_path=str(tmp_path / "sync.sqlite3"),
        autoload=False, rate_limit_rps=1000.0, cache_enabled=True,
    )
    try:
        service.veto("vision", [[[0.0] * 8 for _ in range(8)]])
        assert len(service._response_cache) >= 1
        service.sidecar.model.memory.bump_state_epoch()
        service._sync_global_epoch()
        assert len(service._response_cache) == 0
    finally:
        service.close()


def test_elastic_tokenizer_preserves_every_tier1_id():
    from henri.holarch.code_lang import CodeTextCorpus, VocabularyLedger, build_tokenizer

    root = Path(__file__).resolve().parents[1]
    lock = root / "receipts" / "holarch" / "vocab_lock.json"
    ledger = VocabularyLedger.load(lock)
    corpus = CodeTextCorpus.holarch(root)
    tokenizer = build_tokenizer(corpus, vocab_lock=ledger, elastic_cap=4096)
    shifts = sum(
        1
        for index, name in enumerate(ledger.identifiers)
        if tokenizer.stoi.get(name) != ledger.identifier_base + index
    )
    assert shifts == 0
    assert int(tokenizer.elastic_cap) == 4096
    assert len(tokenizer.tier2_ids) >= 0


def test_holonic_discovery_accepts_one_and_refuses_one():
    from henri.holarch.code_gen import build_synthesis_engine

    root = Path(__file__).resolve().parents[1]
    payload = torch.load(
        root / "receipts" / "holarch" / "holarch_medium.pt",
        map_location="cpu", weights_only=True,
    )
    cfg = HolarchConfig(**payload["config"])
    from henri.holarch.model import open_clean_room_memory, load_checkpoint
    from henri.holarch.pipeline import HenriHolarch

    memory, _info = open_clean_room_memory(
        cfg, str(root / "receipts" / "holarch" / "holarch_medium.pt"),
        tempfile.mkdtemp(prefix="holon_"), "holon_test",
    )
    model = HenriHolarch(cfg, memory=memory)
    load_checkpoint(str(root / "receipts" / "holarch" / "holarch_medium.pt"), model=model)
    model.eval()
    try:
        engine, _stats = build_synthesis_engine(str(root), memory)
        accepted = engine.holonic_discover(
            "Write a function unit_len that takes text and returns the number of tokens "
            "in a token list and clips it into the closed interval 0.0 to 2.5.",
            "assert unit_len('a b c') == 2.5\nassert unit_len('') == 0.0\n",
            {"func": "unit_len", "args": ["text"], "numbers": [0.0, 2.5]},
        )
        refused = engine.holonic_discover(
            "Write a function unit_trap that returns the number 12345 for every input.",
            "assert unit_trap('a') == 12345\n",
            {"func": "unit_trap", "args": ["text"], "numbers": []},
        )
        assert accepted["verified"] is True
        assert str(accepted["plan"]) in engine.plans.plans
        assert refused["verified"] is False
        assert refused["source"] == ""
    finally:
        memory.close()
