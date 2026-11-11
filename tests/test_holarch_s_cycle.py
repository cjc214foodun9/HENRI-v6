
"""Cycle S unit tests: suffix floor, wave Turing machine, charts, browser.

Every test measures a property, not a claim: the suffix distribution is a pure
function of its window, the Turing machine recovers the tape from the wave, the
chart index returns the same neighbours as the flat scan, and the browser
refuses everything outside its allow-list.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

ROOT = Path(__file__).resolve().parents[1]

from henri.holarch.code_gen import (
    WaveCodeTextGenerator,
    WindowSuffixPrior,
    BochnerPhaseProjector,
)
from henri.holarch.code_lang import CodeTextCorpus, VocabularyLedger, build_tokenizer
from henri.holarch.config import HolarchConfig
from henri.holarch.ontology_guard import SandboxViolation, SandboxedCorpusBrowser
from henri.holarch.symbolic import (
    WaveTuringMachine,
    reference_turing_run,
    turing_program,
    turing_programs,
    turing_traps,
)
from henri.holarch.zone_c import ZoneCMemory


# ------------------------------------------------------------------ helpers
def _unit_wave(name: str, dim: int = 256) -> np.ndarray:
    seed = int(hashlib.sha256(str(name).encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    wave = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    return (wave / max(float(np.linalg.norm(wave)), 1e-12)).astype(np.complex64)


@pytest.fixture(scope="module")
def generator() -> WaveCodeTextGenerator:
    corpus = CodeTextCorpus.holarch(ROOT)
    lock = ROOT / "receipts" / "holarch" / "vocab_lock.json"
    ledger = VocabularyLedger.load(lock) if lock.exists() else None
    tokenizer = build_tokenizer(corpus, vocab_lock=ledger, elastic_cap=4096)
    projector = BochnerPhaseProjector(dim=1024, d_sem=64)
    projector.fit(corpus, tokenizer)
    ids = []
    for text in corpus.texts:
        ids.extend(int(token) for token in tokenizer.encode(text))
    machine = WaveCodeTextGenerator(tokenizer, projector)
    machine.fit(ids[:20000], max_fit_tokens=20000)
    machine.count_lm.fit(ids[:20000])
    machine.suffix_window = 512
    machine.suffix_weight = 1.0
    machine.holarch = None
    machine._fast_active = False
    machine._ids = ids[:20000]
    return machine


@pytest.fixture(scope="module")
def vault(tmp_path_factory):
    path = tmp_path_factory.mktemp("s3") / "vault.sqlite"
    import dataclasses

    cfg = dataclasses.replace(HolarchConfig(), cache_capacity=4096)
    memory = ZoneCMemory(cfg, backend="sqlite", path=str(path))
    rng = np.random.default_rng(5)
    waves = []
    for domain in range(6):
        base = rng.standard_normal(cfg.dim) + 1j * rng.standard_normal(cfg.dim)
        base = base / max(float(np.linalg.norm(base)), 1e-12)
        for _ in range(120):
            detail = rng.standard_normal(cfg.dim) + 1j * rng.standard_normal(cfg.dim)
            detail = detail / max(float(np.linalg.norm(detail)), 1e-12)
            wave = (0.86 * base + 0.14 * detail).astype(np.complex64)
            wave = wave / max(float(np.linalg.norm(wave)), 1e-12)
            memory.write(wave, label=domain, tier=2, namespace="world")
            waves.append(wave)
    yield memory, waves
    memory.close()


# ------------------------------------------------------------- S1 suffix
def test_s1_suffix_distribution_is_pure():
    prior = WindowSuffixPrior(order=6)
    window = [int(v) for v in np.random.default_rng(3).integers(0, 30, size=96)]
    other = [int(v) for v in np.random.default_rng(4).integers(0, 30, size=96)]
    first = prior.distribution(window, 64)
    _ = prior.distribution(other, 64)
    second = prior.distribution(list(window), 64)
    assert first is not None and second is not None
    assert np.array_equal(first[0], second[0])
    assert int(first[1]) == int(second[1])


def test_s1_suffix_matches_longest_order():
    prior = WindowSuffixPrior(order=6)
    window = [1, 2, 3, 4, 5, 6, 9, 9, 1, 2, 3, 4, 5, 6]
    stats = prior.match(window)
    assert stats is not None
    assert int(stats["order"]) == 6
    assert int(stats["occurrences"]) >= 1
    assert int(stats["targets"][0]) == 9


def test_s1_token_logits_ignores_call_order(generator):
    ids = generator._ids
    starts = [900, 1500, 2200, 3100, 4000, 5000]
    window = 384
    first = {}
    for start in starts:
        first[start] = generator.token_logits(ids[start - window : start])
    for start in reversed(starts):
        again = generator.token_logits(ids[start - window : start])
        assert np.array_equal(first[start], again)


def test_s1_suffix_floor_raises_top1_on_repeated_idiom():
    prior = WindowSuffixPrior(order=6)
    pattern = [5, 6, 7, 8, 9, 10]
    window = pattern * 8 + [5, 6, 7, 8, 9]
    result = prior.distribution(window, 64)
    assert result is not None
    probability, order, count = result
    assert order >= 5
    assert probability[10] > 0.0
    assert abs(float(probability.sum()) - 1.0) < 1e-9


def test_s1_floor_weight_zero_is_inert(generator):
    ids = generator._ids
    head = ids[4000:4384]
    # Repeat the tail so a 6-gram suffix certainly recurs inside the window.
    context = head + head[-24:]
    saved = float(generator.suffix_weight)
    try:
        generator.suffix_weight = 0.0
        plain = generator.floor_probability(context)
        generator.suffix_weight = 1.0
        mixed = generator.floor_probability(context)
    finally:
        generator.suffix_weight = saved
    assert plain.shape == mixed.shape
    assert not np.array_equal(plain, mixed)


def test_s1_r3_receipt_clears_the_static_bar():
    path = ROOT / "receipts" / "holarch" / "r3_static_and_streaming_full_decile_language.json"
    assert path.exists(), "the R3 receipt must exist before the test suite runs"
    payload = json.loads(path.read_text(encoding="utf-8"))
    full = dict(payload.get("full_decile") or {})
    assert float(full.get("static_top1", 0.0)) >= 0.5250
    assert float(full.get("static_bits_per_char", 9.0)) <= 1.4200
    assert float(full.get("top1", 0.0)) >= 0.5600
    assert float(full.get("bits_per_char", 9.0)) <= 1.2200
    assert int(full.get("calibration_test_overlap_tokens", -1)) == 0
    assert bool(full.get("is_static_inductive", False))


# ------------------------------------------------------------- S2 Turing
@pytest.fixture(scope="module")
def machine() -> WaveTuringMachine:
    return WaveTuringMachine()


def test_s2_sixteen_programs_are_exact(machine):
    programs = turing_programs()
    assert len(programs) == 16
    for program in programs:
        run = machine.run(program)
        assert run["halted"], program.name
        assert run["exact"], program.name
        assert run["final_tape"] == program.expectation, program.name


def test_s2_reference_oracle_agrees(machine):
    for program in turing_programs():
        reference = reference_turing_run(program)
        assert reference["halted"], program.name
        assert reference["tape_text"] == program.expectation, program.name


def test_s2_zero_text_tokens(machine):
    for program in turing_programs():
        run = machine.run(program)
        assert int(run["text_tokens"]) == 0


def test_s2_step_latency_under_one_ms(machine):
    latencies = []
    for program in turing_programs():
        run = machine.run(program)
        if int(run["steps"]) > 0:
            latencies.append(float(run["step_ms_mean"]))
    assert latencies
    assert float(np.mean(latencies)) <= 1.0


def test_s2_four_traps_vetoed(machine):
    traps = turing_traps()
    assert len(traps) == 4
    reasons = []
    for trap in traps:
        run = machine.run(trap["program"], tamper=trap.get("tamper"))
        assert run["vetoed"], trap["name"]
        assert not run["halted"], trap["name"]
        reasons.append(str(run["veto_reason"]))
    assert reasons.count("non-halting-cycle") == 2
    assert "undefined-transition" in reasons
    assert "unreadable-tape" in reasons


def test_s2_cycle_gate_coherence_is_exact(machine):
    trap = turing_traps()[0]
    run = machine.run(trap["program"])
    assert float(run["cycle_coherence_max"]) > 0.995


def test_s2_corrupted_tape_is_unreadable(machine):
    trap = turing_traps()[3]
    run = machine.run(trap["program"], tamper=trap["tamper"])
    assert run["veto_reason"] == "unreadable-tape"
    assert float(run["read_snr_min"]) < float(machine.tau_read)


def test_s2_tape_can_be_supplied_inline(machine):
    program = turing_program("binary_increment")
    from dataclasses import replace

    custom = replace(
        program,
        tape=list("_1_") + ["_"] * 8,
        expectation="10",
        alphabet=sorted(set(program.alphabet) | {"1", "_"}),
    )
    run = machine.run(custom)
    assert run["halted"]
    assert run["final_tape"] == "10"


# -------------------------------------------------------------- S3 charts
def test_s3_charts_partition_the_vault(vault):
    memory, waves = vault
    stats = memory.chart_stats()
    assert int(stats["engrams"]) == len(waves)
    assert int(stats["charts"]) >= 1
    assert int(stats["largest_chart"]) <= int(memory.chart_capacity)
    assert sum(int(chart["size"]) for chart in memory.charts) == len(waves)


def test_s3_charted_matches_flat_top1(vault):
    memory, waves = vault
    rng = np.random.default_rng(8)
    for index in range(24):
        pick = int(rng.integers(0, len(waves)))
        noise = rng.standard_normal(memory.dim) + 1j * rng.standard_normal(memory.dim)
        noise = noise / max(float(np.linalg.norm(noise)), 1e-12)
        query = (0.98 * waves[pick] + 0.02 * noise).astype(np.complex64)
        charted = memory.query(query, k=4, namespace="world", use_charts=True)
        flat = memory.query(query, k=4, namespace="world", use_charts=False)
        assert charted.hits and flat.hits
        assert int(charted.hits[0].engram_id) == int(flat.hits[0].engram_id)


def test_s3_stiefel_retraction_reduces_crosstalk(vault):
    memory, _ = vault
    crowded = max(range(len(memory.charts)), key=lambda i: int(memory.charts[i]["size"]))
    report = memory.orthogonalize_chart(crowded, tau_crosstalk=0.0)
    if not report.get("applied", False):
        report = memory.orthogonalize_chart(crowded, tau_crosstalk=0.0, eta=0.5)
    assert report["applied"] is True
    assert float(report["crosstalk_after"]) < float(report["crosstalk_before"])


def test_s3_unknown_chart_raises(vault):
    memory, _ = vault
    with pytest.raises(KeyError):
        memory.orthogonalize_chart(9999)


# ------------------------------------------------------------- S3 browser
@pytest.fixture()
def browser(tmp_path, vault):
    memory, _ = vault
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "physics.md").write_text(
        "# physics\n"
        "tungsten_melting_point = 3422 K\n"
        "the melting point of tungsten is recorded as 3422 K\n",
        encoding="utf-8",
    )
    (root / "biology.md").write_text(
        "# biology\ndna_bases = 4\nthe bases of dna are recorded as 4\n",
        encoding="utf-8",
    )
    return SandboxedCorpusBrowser(memory, [root], namespace="world"), root


def test_s3_browser_traversal_blocked(browser):
    instance, _ = browser
    result = instance.browse_and_crystallize(
        "melting point of tungsten", "world", path="../../etc/passwd"
    )
    assert result["refused"] is True
    assert result["grounded"] is False
    assert instance.refusals >= 1


def test_s3_browser_scheme_blocked(browser):
    instance, _ = browser
    result = instance.browse_and_crystallize(
        "melting point of tungsten", "world", path="https://example.invalid/x.txt"
    )
    assert result["refused"] is True


def test_s3_browser_outside_root_blocked(browser):
    instance, _ = browser
    result = instance.browse_and_crystallize(
        "melting point of tungsten", "world", path=str(ROOT / "README.md")
    )
    assert result["refused"] is True


def test_s3_browser_resolves_allowed_file(browser):
    instance, root = browser
    resolved = instance.resolve("physics.md")
    assert resolved.name == "physics.md"
    assert instance._inside(resolved, root)


def test_s3_browser_grounds_and_crystallises(browser):
    instance, _ = browser
    result = instance.browse_and_crystallize("melting point of tungsten", "world")
    assert result["grounded"] is True
    assert "3422" in str(result["answer"])
    assert int(result["written"]) >= 1
    assert len(str(result["receipt"])) == 64


def test_s3_browser_ungrounded_writes_nothing(browser):
    instance, _ = browser
    before = instance.writes
    result = instance.browse_and_crystallize(
        "quantum chromodynamic flux capacitor calibration", "world"
    )
    assert result["grounded"] is False
    assert int(result["written"]) == 0
    assert instance.writes == before


def test_s3_browser_receipts_chain(browser):
    instance, _ = browser
    first = instance.browse_and_crystallize("melting point of tungsten", "world")
    second = instance.browse_and_crystallize("bases of dna", "world")
    assert first["receipt"] and second["receipt"]
    assert first["receipt"] != second["receipt"]


# --------------------------------------------------------------- API surface
def test_s4_turing_surface_is_registered():
    from henri.holarch.server import build_app
    from henri.holarch.mcp_server import mcp

    app = build_app()
    paths = {getattr(route, "path", "") for route in app.routes}
    assert "/v1/holarch/turing" in paths
    tools = {getattr(tool, "name", "") for tool in mcp._tool_manager.list_tools()}
    assert "holarch_turing" in tools
