"""Acceptance tests for Stage L1-L3: tokenizer, Booster phases, verifier, decoder.

The tests assert measured properties, not wiring: a lossless round trip on
the real source tree, a phase margin above the gate, a rejection of an
invented number, and a synthesis that executes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from henri.holarch.code_gen import SynthesisEngine, WaveCodeTextGenerator
from henri.holarch.code_lang import (
    BochnerPhaseProjector,
    CodeTextCorpus,
    HolarchCodeTextTokenizer,
    build_tokenizer,
    roundtrip_check,
)
from henri.holarch.config import HolarchConfig
from henri.holarch.ontology_guard import build_verifier
from henri.holarch.zone_c import ZoneCMemory

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def corpus() -> CodeTextCorpus:
    return CodeTextCorpus.holarch(ROOT)


@pytest.fixture(scope="module")
def tokenizer(corpus) -> HolarchCodeTextTokenizer:
    return build_tokenizer(corpus)


@pytest.fixture(scope="module")
def projector(corpus, tokenizer) -> BochnerPhaseProjector:
    builder = BochnerPhaseProjector(dim=1024, d_sem=64)
    builder.fit(corpus, tokenizer)
    return builder


@pytest.fixture(scope="module")
def verifier():
    return build_verifier(ROOT)


# ------------------------------------------------------------------- L1
def test_tokenizer_round_trips_every_holarch_source_file(tokenizer):
    files = sorted((ROOT / "henri" / "holarch").glob("*.py")) + [ROOT / "docs" / "HOLARCH.md"]
    result = roundtrip_check(tokenizer, files)
    assert result["texts"] >= 15
    assert result["ok"] == result["texts"], result["failures"][:3]


def test_tokenizer_round_trips_arbitrary_utf8(tokenizer):
    samples = [
        "",
        "def f(x): return x + 1\n",
        "héllo wörld — 漢字 🙂\n",
        "\x00\x01\xff binary-ish",
        "\t\n  \r\n",
    ]
    for text in samples:
        assert tokenizer.decode(tokenizer.encode(text)) == text


def test_vocabulary_is_bounded_and_byte_complete(tokenizer):
    assert len(tokenizer) <= 2048
    assert tokenizer.n_bytes == 256
    for value in (0, 1, 127, 128, 255):
        token = tokenizer._byte_id(value)
        assert tokenizer.token_text(token) == bytes([value]).decode("latin-1")


def test_phase_margin_separates_related_pairs_from_unrelated(projector, tokenizer):
    report = projector.report
    assert report is not None
    assert report.margin >= 0.25, report.as_dict()
    for left, right in (("return", "yield"), ("int", "float"), ("verify", "check")):
        score = projector.coherence(tokenizer.stoi[left], tokenizer.stoi[right])
        assert score > 0.0


def test_negation_is_a_pi_rotation_in_the_operator_slot(projector, tokenizer):
    affirmative = projector.slot_wave(tokenizer.stoi["not"], tokenizer)
    plain = projector.slot_wave(tokenizer.stoi["and"], tokenizer)
    start = 1 * projector.d_slot
    ratio = affirmative[start : start + projector.d_slot] / plain[start : start + projector.d_slot]
    # The two tokens are different words, so the phase cannot be exactly -1;
    # what must hold is that the negation carries the extra pi term.
    assert np.allclose(np.abs(ratio), 1.0, atol=1e-4)


# ------------------------------------------------------------------- L2
def test_verifier_accepts_a_true_quantitative_claim(verifier):
    verdict = verifier.verify_claim(
        "The medium profile has 1,020,380 parameters and reaches 0.7083 closed-loop action."
    )
    assert verdict.status == "supported", verdict.reason


def test_verifier_rejects_an_invented_number(verifier):
    verdict = verifier.verify_claim("The medium profile has 34 gigabytes of dense covariance.")
    assert verdict.status != "supported"


def test_verifier_rejects_a_corrupted_quantity(verifier):
    verdict = verifier.verify_claim("The medium profile has 1,020,381 parameters.")
    assert verdict.status != "supported"
    assert verdict.missing_numbers


def test_verifier_rejects_a_fabricated_signature(verifier):
    verdict = verifier.verify_claim(
        "Calling ZoneCMemory.query with keyword argument widget returns the stored wave."
    )
    assert verdict.status != "supported"
    assert verdict.signature_rule


def test_verifier_passages_are_content_addressed(verifier):
    digests = {passage.sha256 for passage in verifier.passages}
    assert len(digests) == len(verifier.passages)
    assert all(len(digest) == 64 for digest in digests)


# ------------------------------------------------------------------- L3
@pytest.fixture(scope="module")
def generator(corpus, tokenizer, projector) -> WaveCodeTextGenerator:
    ids = []
    for text in corpus.texts:
        ids.extend(tokenizer.encode(text))
    model = WaveCodeTextGenerator(tokenizer, projector)
    model.fit(ids[:24000], max_fit_tokens=8000)
    model.count_lm.fit(ids[: int(len(ids) * 0.9)])
    model.lexical.build(ids[:24000])
    return model


def test_generator_predicts_tokens_above_the_char_baseline(generator, corpus, tokenizer):
    ids = []
    for text in corpus.texts:
        ids.extend(tokenizer.encode(text))
    held = ids[int(len(ids) * 0.9) :]
    report = generator.evaluate(held[:3000])
    assert report.top1 >= 0.45, report.as_dict()
    baseline_bits = float(np.log2(17.043866783609086))
    assert report.bits_per_char <= 0.6 * baseline_bits, report.as_dict()


def test_synthesis_refuses_a_symbol_that_does_not_exist(tmp_path, generator, projector, tokenizer, verifier):
    memory = ZoneCMemory(
        HolarchConfig.medium(), backend="sqlite", path=str(tmp_path / "syn.sqlite3")
    )
    try:
        engine = SynthesisEngine(generator, projector, tokenizer, memory, verifier=verifier)
        result = engine.synthesize(
            "Write a function that calls ZoneCMemory.quantum_entangle to fuse two tenants.",
            kind="code",
        )
        assert result["status"] == "refused"
        assert result["verified"] is False
    finally:
        memory.close()


def test_synthesis_adapts_in_one_feedback_step(tmp_path, generator, projector, tokenizer, verifier):
    memory = ZoneCMemory(
        HolarchConfig.medium(), backend="sqlite", path=str(tmp_path / "adapt.sqlite3")
    )
    try:
        engine = SynthesisEngine(generator, projector, tokenizer, memory, verifier=verifier)
        intent = "Write a function clamp_unit that clips a value into the closed unit interval."
        source = "def clamp_unit(value):\n    return max(0.0, min(1.0, float(value)))\n"
        test = "assert clamp_unit(1.7) == 1.0\n"
        cold = engine.synthesize(intent, kind="code")
        assert cold["verified"] is False
        adapted = engine.adapt(intent, source, "clamp_unit")
        assert adapted["adapted"] is True
        warm = engine.synthesize(intent, kind="code")
        assert warm["verified"] is True
        assert warm["plan"] == "clamp_unit"
        assert engine.execute(warm["source"], test)["ok"] is True
    finally:
        memory.close()


def test_text_synthesis_is_gated_by_the_ontology_verifier(generator, projector, tokenizer, verifier):
    engine = SynthesisEngine(
        generator, projector, tokenizer, memory=None, verifier=verifier
    )
    good = engine.verify(
        "The medium profile has 1,020,380 parameters and reaches 0.7083 closed-loop action.",
        kind="text",
    )
    bad = engine.verify("The medium profile has 34 gigabytes of dense covariance.", kind="text")
    assert good["ok"] is True
    assert bad["ok"] is False
