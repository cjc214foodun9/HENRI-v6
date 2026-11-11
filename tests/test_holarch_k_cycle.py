"""Cycle K unit gates: boundary axioms, discovery relaxation, the
multi-statement splicer, the rebased tenant vocabulary, the K1 cartridge
codec, and the unified VLA token space.

Every test is deterministic and touches no network. The heavy end-to-end
gates live in scripts/run_holarch_k_cycle.py; these tests pin the units those
gates rest on.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from henri.holarch.code_gen import (
    _composable_body,
    _compiled_plan,
    _function_definition,
    compose_plan_dag,
)
from henri.holarch.code_lang import (
    HolarchCodeTextTokenizer,
    VocabularyLedger,
    rebase_tenant_vocabulary,
)
from henri.holarch.config import HolarchConfig
from henri.holarch.model import (
    SEED_ENGRAM_BYTE_BUDGET,
    seed_engram_payload,
    seed_engram_rows,
    seed_engram_waves,
)
from henri.holarch.ontology_guard import OntologyVerifier
from henri.holarch.pipeline import HenriHolarch
from henri.holarch.vla import (
    VLA_RANGES,
    action_token,
    relational_token,
    token_modality,
    verify_chain,
    vision_token,
)
from henri.holarch.zone_b import PhotonicCore, _unit, sagnac_delta
from henri.holarch.zone_c import (
    BOUNDARY_POLARITIES,
    BoundaryAxiom,
    ZoneCMemory,
)

ROOT = Path(__file__).resolve().parents[1]


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


# ------------------------------------------------------------- zone C axioms
def test_boundary_axiom_is_persisted_as_a_tier3_engram(memory, cfg):
    wave = _wave(int(cfg.dim), 1)
    axiom = memory.register_axiom(
        wave,
        name="no_reverse",
        polarity="LAW",
        source="human_heuristic",
        slot_mask=(False, False, False, True),
        namespace="tenant",
    )
    assert isinstance(axiom, BoundaryAxiom)
    assert axiom.polarity == "LAW"
    assert axiom.slot_mask == (False, False, False, True)
    fetched = memory.boundary_axioms("tenant")
    assert len(fetched) == 1
    assert fetched[0].engram_id == axiom.engram_id
    rows = memory.store.metadata(cfg.dim, namespace="tenant")
    assert any(int(row[0]) == int(axiom.engram_id) and int(row[4]) == 3 for row in rows)


def test_boundary_axiom_masks_the_ungoverned_slots(memory, cfg):
    wave = _wave(int(cfg.dim), 2)
    axiom = memory.register_axiom(
        wave, name="slot3", polarity="ATTRACTOR", source="library_prototype",
        slot_mask=(False, False, False, True), namespace="tenant",
    )
    governed = axiom.governed()
    slots = governed.reshape(4, -1)
    assert float(np.linalg.norm(slots[:3])) <= 1e-6
    assert float(np.linalg.norm(slots[3])) > 0.99


def test_boundary_axiom_refuses_unknown_polarity_and_source(memory, cfg):
    wave = _wave(int(cfg.dim), 3)
    with pytest.raises(ValueError):
        memory.register_axiom(wave, "x", "MAGIC")
    with pytest.raises(ValueError):
        memory.register_axiom(wave, "x", "LAW", source="rumour")


def test_boundary_axiom_polarity_filter(memory, cfg):
    for index, polarity in enumerate(BOUNDARY_POLARITIES):
        memory.register_axiom(
            _wave(int(cfg.dim), 10 + index), f"axiom_{polarity}", polarity,
            namespace="filter",
        )
    assert len(memory.boundary_axioms("filter")) == 3
    assert len(memory.boundary_axioms("filter", "REPULSOR")) == 1
    waves = memory.axiom_waves("filter", "LAW")
    assert waves.shape[0] == 1


# ---------------------------------------------------------------- zone B loop
def test_discovery_energy_is_zero_law_penalty_when_the_candidate_is_the_law(cfg):
    core = PhotonicCore(cfg)
    bright = _unit(torch.from_numpy(_wave(int(cfg.dim), 21)).reshape(1, -1))
    energy, parts = core.discovery_energy(
        bright, attractors=bright, laws=bright, repulsors=None
    )
    assert float(parts["law"].max()) <= 1e-4
    assert float(parts["goal"].max()) <= 1e-4


def test_relax_discovery_selects_a_law_satisfying_wave(cfg):
    core = PhotonicCore(cfg)
    psi = _unit(torch.from_numpy(_wave(int(cfg.dim), 31)).reshape(1, -1))
    target = _unit(torch.from_numpy(_wave(int(cfg.dim), 32)).reshape(1, -1))
    trap = _unit(torch.from_numpy(_wave(int(cfg.dim), 33)).reshape(1, -1))
    directions = torch.stack([target - psi, trap - psi] * 4, dim=0).unsqueeze(0)
    state, trace = core.relax_discovery(
        psi,
        attractors=target,
        repulsors=trap,
        laws=target,
        probes=8,
        steps=2,
        tau_law=0.01,
        lambda_law=50.0,
        directions=directions,
        seed=7,
    )
    assert trace["probes"] == 8
    assert bool(trace["law_satisfied"][0])
    assert float(trace["energy_trace"][-1]) <= float(trace["energy_trace"][0]) + 1e-4
    law_out = core.forward(target, t_bias=float(cfg.t_max)).bright
    assert float(sagnac_delta(state.bright.detach(), law_out).mean()) <= 0.05


def test_relax_discovery_probes_are_deterministic(cfg):
    core = PhotonicCore(cfg)
    psi = _unit(torch.from_numpy(_wave(int(cfg.dim), 41)).reshape(1, -1))
    first, _ = core.relax_discovery(psi, probes=8, steps=1, seed=99)
    second, _ = core.relax_discovery(psi, probes=8, steps=1, seed=99)
    assert torch.equal(first.bright, second.bright)


# ------------------------------------------------------------ DAG splicing
def _filter_and_total_plans():
    first = _compiled_plan(
        "filter_above",
        "filter",
        "def filter_above(items: list, threshold: float) -> list:\n"
        "    kept = []\n"
        "    for item in items:\n"
        "        if item > threshold:\n"
        "            kept.append(item)\n"
        "    return kept\n",
    )
    second = _compiled_plan(
        "total",
        "sum",
        "def total(values: list) -> float:\n"
        "    accumulator = 0.0\n"
        "    for value in values:\n"
        "        accumulator += value\n"
        "    return accumulator\n",
    )
    return first, second


def test_multistatement_multiargument_pipeline_executes():
    first, second = _filter_and_total_plans()
    verdict = compose_plan_dag(
        [first, second],
        {"func": "pipeline", "args": ["items", "threshold"], "numbers": []},
    )
    assert verdict["ok"] is True
    scope = {"__builtins__": __builtins__}
    exec(verdict["source"], scope)
    assert scope["pipeline"]([1.0, 2.0, 3.0, 4.0], 2.0) == 7.0


def test_multistatement_pipeline_alpha_renames_each_stage():
    first, second = _filter_and_total_plans()
    verdict = compose_plan_dag(
        [first, second],
        {"func": "pipeline", "args": ["items", "threshold"], "numbers": []},
    )
    source = verdict["source"]
    assert "_s0_kept" in source
    assert "_s1_accumulator" in source
    assert "step_1" in source
    ast.parse(source)


def test_multistatement_extra_argument_binds_by_name():
    first, second = _filter_and_total_plans()
    third = _compiled_plan(
        "scale",
        "scale",
        "def scale(values: list, factor: float) -> list:\n"
        "    scaled = []\n"
        "    for value in values:\n"
        "        scaled.append(value * factor)\n"
        "    return scaled\n",
    )
    verdict = compose_plan_dag(
        [first, third, second],
        {"func": "pipeline", "args": ["items", "threshold", "factor"], "numbers": []},
    )
    assert verdict["ok"] is True
    scope = {"__builtins__": __builtins__}
    exec(verdict["source"], scope)
    assert scope["pipeline"]([1.0, 2.0, 3.0, 4.0], 2.0, 2.0) == 14.0


def test_early_return_stage_is_refused():
    # Directive M4 accepts a LEADING guard clause. This function opens with a
    # non-guard statement, so the return inside the following branch is an
    # early exit of the body and the stage is still refused.
    definition = _function_definition(
        "def early(items: list) -> list:\n"
        "    seen = list(items)\n"
        "    if seen:\n"
        "        return seen\n"
        "    return []\n"
    )
    assert _composable_body(definition) is None


def test_missing_binding_is_refused():
    first, _ = _filter_and_total_plans()
    third = _compiled_plan(
        "scale",
        "scale",
        "def scale(values: list, factor: float) -> list:\n"
        "    return [value * factor for value in values]\n",
    )
    verdict = compose_plan_dag(
        [first, third],
        {"func": "pipeline", "args": ["items", "threshold"], "numbers": []},
    )
    assert verdict["ok"] is False
    assert verdict["source"] == ""


# ----------------------------------------------------------------- rebase
def test_cross_tenant_rebase_has_zero_tier1_shifts_and_is_lossless():
    lock_path = ROOT / "receipts" / "holarch" / "vocab_lock.json"
    lock = VocabularyLedger.load(lock_path) if lock_path.exists() else None
    tenant_a = HolarchCodeTextTokenizer(vocab_lock=lock, elastic_cap=8192)
    tenant_b = HolarchCodeTextTokenizer(vocab_lock=lock, elastic_cap=8192)
    names_a = [f"unit_alpha_{index:03d}_value" for index in range(40)]
    names_b = [f"unit_beta_{index:03d}_value" for index in range(40)]
    tenant_a.add_identifiers(names_a)
    tenant_b.add_identifiers(names_b)
    rebase = rebase_tenant_vocabulary(tenant_a, tenant_b)
    assert rebase.tier1_shifts == 0
    merged = rebase.tokenizer
    for name in names_b[:10]:
        text = f"call {name}(value)"
        ids_b = tenant_b.encode(text)
        assert merged.decode(rebase.translate(ids_b, "b")) == tenant_b.decode(ids_b)
    for name in names_a[:10]:
        assert merged.stoi.get(name) == tenant_a.stoi.get(name)


def test_rebase_union_contains_every_identifier_once():
    tenant_a = HolarchCodeTextTokenizer(elastic_cap=8192)
    tenant_b = HolarchCodeTextTokenizer(elastic_cap=8192)
    tenant_a.add_identifiers([f"shared_unit_{index}" for index in range(20)])
    tenant_b.add_identifiers([f"shared_unit_{index}" for index in range(20)])
    rebase = rebase_tenant_vocabulary(tenant_a, tenant_b)
    assert len(rebase.union_identifiers) == len(set(rebase.union_identifiers))
    assert int(rebase.union_size) >= 20
    assert int(rebase.tier1_size) == 2048


# --------------------------------------------------------------- cartridge
def test_v3_cartridge_round_trip_preserves_the_wave_direction():
    rows = [
        {"label": 106, "tier": 3, "wave": _wave(1024, 90)},
        {"label": 106, "tier": 1, "wave": _wave(1024, 91)},
        {"label": 107, "tier": 3, "wave": _wave(1024, 92)},
        {"label": 107, "tier": 1, "wave": _wave(1024, 93)},
        {"label": 0, "tier": 1, "wave": _wave(1024, 100)},
        {"label": 1, "tier": 1, "wave": _wave(1024, 101)},
        {"label": 110, "tier": 1, "wave": _wave(1024, 102)},
        {"label": 10, "tier": 1, "wave": _wave(1024, 103)},
    ]
    block = seed_engram_payload(rows)
    assert block["format"] == "henri-seed-engrams-v3"
    assert block["count"] == 8
    assert int(block["bytes"]) <= SEED_ENGRAM_BYTE_BUDGET
    assert int(block["action_rows"]) == 4
    assert int(block["support_rows"]) == 4
    assert int(block["action_proto_count"]) == 2
    decoded = seed_engram_rows(block)
    assert len(decoded) == 8
    for source, target in zip(rows, decoded):
        coherence = abs(np.vdot(source["wave"], target["wave"]))
        if 106 <= int(source["label"]) < 110:
            assert coherence >= 0.995
        else:
            assert coherence >= 0.9999
    waves = seed_engram_waves(block)
    assert waves is not None and waves.shape == (8, 1024)


# --------------------------------------------------------------- VLA space
def test_vla_ranges_are_disjoint_and_cover_the_modalities():
    spans = sorted(VLA_RANGES.values())
    for left, right in zip(spans, spans[1:]):
        assert left[1] <= right[0]
    assert token_modality(vision_token(3)) == "vision"
    assert token_modality(action_token(2)) == "action"
    assert token_modality(relational_token(5)) == "relational"


def test_vla_receipt_chain_detects_a_broken_parent():
    receipts = [
        {"index": 0, "parent": "genesis", "sha256": "a"},
        {"index": 1, "parent": "a", "sha256": "b"},
    ]
    assert verify_chain(receipts)
    receipts[1]["parent"] = "c"
    assert not verify_chain(receipts)


def test_generate_vla_episode_completes_vision_and_relational_stages(cfg, memory):
    model = HenriHolarch(cfg, memory=memory)
    model.eval()
    canvas = np.linspace(0.0, 1.0, 256, dtype=np.float32).reshape(16, 16)
    episode = model.generate_vla_episode(canvas)
    assert episode["vision"]["grounded"] is True
    assert episode["receipt_chain_valid"] is True
    assert len(episode["receipts"]) == 2
    assert [token.split(":")[0] for token in episode["tokens"]] == ["vision", "relational"]


# ------------------------------------------------------------------- BM25
def test_vectorised_bm25_matches_the_reference_loop():
    verifier = OntologyVerifier(root=ROOT)
    verifier.passages = [object(), object(), object()]
    verifier.terms = [["alpha", "beta"], ["beta", "gamma"], ["gamma", "delta"]]
    verifier.numbers = [[], [], []]
    verifier.idf = {"alpha": 1.2, "beta": 0.4, "gamma": 0.7, "delta": 1.1}
    verifier.mean_length = 2.0
    verifier.doc_freq_total = 3
    verifier._build_bm25_matrices()
    for claim in ("alpha beta", "gamma", "delta alpha"):
        matrix = [index for _score, index in verifier.bm25(claim, top_k=3)]
        python = [index for _score, index in verifier._bm25_python(claim, top_k=3)]
        assert matrix == python
