"""Decision-level invariants for Decisions 5 to 8."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from henri.holarch.config import HolarchConfig
from henri.holarch.pipeline import HenriHolarch
from henri.holarch.symbolic import (
    Edge,
    SymbolCodebook,
    SymbolicDecoder,
    SymbolicTopologyTask,
    TopologySolver,
    index_code,
)
from henri.holarch.tasks import build_tasks
from henri.holarch.zone_b import PhotonicCore, StiefelSlotPool
from henri.holarch.zone_c import ZoneCMemory


def _model(tmp_path, cfg, name="d5.sqlite3"):
    torch.manual_seed(cfg.seed)
    return HenriHolarch(cfg, backend="sqlite", memory_path=str(tmp_path / name))


# --------------------------------------------------------------- Decision 5
def test_fixed_detector_frame_is_frozen_and_orthonormal():
    cfg = HolarchConfig.small(stiefel_mode="fixed")
    pool = StiefelSlotPool(cfg.n_slots, cfg.d_slot, cfg.pool_k, mode="fixed")
    assert pool.weight_real.requires_grad is False
    basis = pool.basis()
    gram = basis @ basis.conj().transpose(-1, -2)
    eye = torch.eye(cfg.pool_k, dtype=gram.dtype).expand_as(gram)
    assert float((gram - eye).abs().max()) < 1e-5
    assert float((basis - pool.fixed_basis).abs().max()) == 0.0


def test_fixed_frame_differs_from_a_learned_frame():
    cfg = HolarchConfig.small()
    generator = torch.Generator().manual_seed(3)
    wave = torch.randn(2, cfg.dim, generator=generator, dtype=torch.complex64)
    wave = wave / wave.norm(dim=-1, keepdim=True)
    fixed = PhotonicCore(cfg.replace(stiefel_mode="fixed"))(wave)
    learned = PhotonicCore(cfg.replace(stiefel_mode="learned"))(wave)
    assert float((fixed.features - learned.features).abs().max()) > 1e-4


def test_interplate_kerr_makes_the_stack_amplitude_dependent():
    cfg = HolarchConfig.small(kerr_interplate=0.75)
    core = PhotonicCore(cfg)
    plain = PhotonicCore(cfg.replace(kerr_interplate=0.0))
    generator = torch.Generator().manual_seed(7)
    wave = torch.randn(2, cfg.dim, generator=generator, dtype=torch.complex64)
    wave = wave / wave.norm(dim=-1, keepdim=True)
    a = core(wave).bright
    b = plain(wave).bright
    assert float((a - b).abs().max()) > 1e-3
    # A pure phase change cannot be produced by a fixed linear map: two
    # different input amplitudes must rotate the plate phase differently.
    scaled = wave.clone()
    scaled[:, : cfg.d_slot] = scaled[:, : cfg.d_slot] * 0.5
    scaled = scaled / scaled.norm(dim=-1, keepdim=True)
    c = core(scaled).bright
    d = plain(scaled).bright
    assert float((c - d).abs().max()) > 1e-3


# --------------------------------------------------------------- Decision 6
def test_prototype_logits_enter_the_decision(tmp_path):
    cfg = HolarchConfig.small()
    model = _model(tmp_path, cfg)
    task = build_tasks(cfg)["vision"]
    batch = task.sample(8, "val")
    before = model("vision", batch.x, training=False, relax=False).task_logits
    model.calibrate_prototypes("vision", [task.sample(64, "train") for _ in range(2)])
    model.calibrate_gate("vision", [task.sample(32, "val").x for _ in range(2)])
    output = model("vision", batch.x, training=False, relax=False)
    proto = model.prototype_logits("vision", output.photonic.bright, cfg.n_vision_classes)
    assert proto is not None
    assert tuple(proto.shape) == (8, cfg.n_vision_classes)
    assert float(proto.std()) > 1e-3
    with torch.no_grad():
        delta = output.task_logits - before
    assert float(delta.abs().max()) > 1e-6
    model.memory.close()


def test_heater_feedback_uses_the_projected_subspace(tmp_path):
    cfg = HolarchConfig.small()
    core = PhotonicCore(cfg)
    generator = torch.Generator().manual_seed(13)
    wave = torch.randn(4, cfg.dim, generator=generator, dtype=torch.complex64)
    wave = wave / wave.norm(dim=-1, keepdim=True)
    phi = torch.randn(3, cfg.dim, generator=generator, dtype=torch.complex64)
    phi = phi / phi.norm(dim=-1, keepdim=True)
    full = core.slot_alignment(wave, phi)
    projected = core.slot_alignment_subspace(wave, phi)
    assert tuple(projected.shape) == tuple(full.shape)
    assert float((projected - full).abs().max()) > 1e-4


# --------------------------------------------------------------- Decision 7
def test_conv_shift_binding_widens_the_relational_block(tmp_path):
    cfg = HolarchConfig.small(relational_binding="conv_shift", relational_shifts=4, relational_k=16)
    model = _model(tmp_path, cfg)
    assert model.relational_width == 4 * 2 * 16
    task = build_tasks(cfg)["relational"]
    output = model("relational", task.sample(3, "train").x, training=False)
    assert tuple(model.relational_features(output.photonic).shape) == (3, 128)
    assert model.relational_expert is not None
    assert tuple(output.task_logits.shape) == (3, cfg.relational_classes)
    model.memory.close()


def test_circular_convolution_spreads_a_single_channel():
    dim = 64
    delta = torch.zeros(dim, dtype=torch.complex64)
    delta[0] = 1.0
    flat = torch.ones(dim, dtype=torch.complex64) / np.sqrt(dim)
    spectral = torch.fft.fft(torch.stack([delta, flat, flat], dim=0), dim=-1)
    convolved = torch.fft.ifft(spectral[0] * spectral[1] * spectral[2], dim=-1)
    hadamard = delta * flat * flat
    # Convolution spreads the delta across every channel; the diagonal
    # product keeps it in channel zero. That is the cross-frequency mixing
    # Decision 7 asks for.
    assert float(convolved[1:].abs().max()) > 1e-3
    assert float(hadamard[1:].abs().max()) < 1e-6


def test_trilinear_control_keeps_the_retired_arm(tmp_path):
    cfg = HolarchConfig.small(relational_binding="trilinear", relational_head="linear")
    model = _model(tmp_path, cfg)
    assert model.relational_expert is None
    assert model.trilinear_head is not None
    model.memory.close()


# --------------------------------------------------------------- Decision 8
def test_codebook_chords_separate_negation():
    codebook = SymbolCodebook(256)
    affirmative = codebook.compose("alpha", "guard", role=3, negative=False)
    negative = codebook.compose("alpha", "guard", role=3, negative=True)
    assert abs(np.linalg.norm(affirmative) - 1.0) < 1e-5
    coherence = float(np.real(np.vdot(affirmative, negative)))
    assert coherence < -0.5


def test_topology_solver_answers_and_abstains(tmp_path):
    cfg = HolarchConfig.small()
    memory = ZoneCMemory(cfg, backend="sqlite", path=str(tmp_path / "topo.sqlite3"))
    task = SymbolicTopologyTask(seed=5)
    edges = task.graph(0)
    codebook = SymbolCodebook(cfg.dim)
    solver = TopologySolver(memory, codebook, mode="indexed")
    solver.ingest(edges)
    solver.calibrate(edges)
    operators, answer = task.query(edges, "s03", 2, salt=1)
    result = solver.solve("s03", operators)
    assert result.answer == answer
    source, missing = task.draw_unreachable(edges, salt=2)
    refused = solver.solve(source, missing)
    assert refused.vetoed and refused.answer is None
    memory.close()


def test_bundled_resonator_saturates_and_the_index_does_not(tmp_path):
    cfg = HolarchConfig.small()
    task = SymbolicTopologyTask(seed=5)
    edges = task.graph(0)[:32]
    targets = {(edge.source, edge.operator): edge.target for edge in edges}
    memory = ZoneCMemory(cfg, backend="sqlite", path=str(tmp_path / "cap.sqlite3"))
    codebook = SymbolCodebook(cfg.dim)
    resonator = TopologySolver(memory, codebook, mode="resonator")
    resonator.ingest(edges)
    indexed = TopologySolver(memory, codebook, mode="indexed")
    indexed.ingest(edges)
    hits_r = sum(
        int(resonator.step(s, o)["answer"] == t) for (s, o), t in targets.items()
    )
    hits_i = sum(int(indexed.step(s, o)["answer"] == t) for (s, o), t in targets.items())
    assert hits_i == len(targets)
    assert hits_r < len(targets)
    memory.close()


def test_code_indexer_finds_a_call_edge(tmp_path):
    module = tmp_path / "pkg"
    module.mkdir()
    (module / "a.py").write_text(
        "def helper(x):\n    return x + 1\n\n\ndef main(y):\n    return helper(y)\n",
        encoding="utf-8",
    )
    graph = index_code(tmp_path)
    assert graph.files == 1
    assert any(edge.source.endswith("main") and edge.target.endswith("helper") for edge in graph.edges)


def test_symbolic_decoder_renders_an_exact_abstention():
    from henri.holarch.symbolic import SolveResult

    decoder = SymbolicDecoder()
    assert "ABSTAIN" in decoder.render_path(SolveResult(None, [], [], True, "no transition"))
    rendered = decoder.render_path(
        SolveResult("s2", [("s0", "fwd"), ("s1", "map")], [0.9, 0.8], False, "")
    )
    assert "SOLVED s2" in rendered and "--fwd-->" in rendered


def test_universal_subspace_is_orthonormal_and_shared(tmp_path):
    cfg = HolarchConfig.small(universal_rank=8)
    model = _model(tmp_path, cfg)
    basis = model.zone_a.universal.basis()
    gram = basis @ basis.transpose(-1, -2)
    eye = torch.eye(cfg.universal_rank)
    assert float((gram - eye).abs().max()) < 1e-4
    task = build_tasks(cfg)["vision"]
    with torch.no_grad():
        output = model("vision", task.sample(4, "val").x, training=False)
    assert torch.isfinite(output.task_logits).all()
    model.memory.close()
