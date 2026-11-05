"""Directive J2: the Tier-2 elastic tenant vocabulary.

The tests pin the two properties that make the extension safe: the locked
Tier-1 ids cannot move, and a restarted tenant can rebuild its exact Tier-2
address map from Zone C metadata without re-reading a corpus.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from henri.holarch.code_lang import (
    VOCAB_LEDGER_KIND,
    BochnerPhaseProjector,
    CodeTextCorpus,
    HolarchCodeTextTokenizer,
    VocabularyLedger,
)
from henri.holarch.config import HolarchConfig
from henri.holarch.zone_c import ZoneCMemory

ROOT = Path(__file__).resolve().parents[1]
VOCAB_LOCK = ROOT / "receipts" / "holarch" / "vocab_lock.json"
TENANT_FUNCTIONS = 600


def _mutation_corpus() -> CodeTextCorpus:
    lines = []
    for index in range(TENANT_FUNCTIONS):
        name = f"live_probe_{index}"
        lines.append(f"def {name}(value):")
        lines.append(f"    return {name}(live_probe_{(index + 1) % TENANT_FUNCTIONS}(value))")
        lines.append("")
    return CodeTextCorpus(
        texts=[chr(10).join(lines)], names=["tenant_mutation.py"], digest="tenant"
    )


@pytest.fixture(scope="module")
def lock() -> VocabularyLedger:
    return VocabularyLedger.load(VOCAB_LOCK)


def test_default_cap_reproduces_the_cycle_i_vocabulary(lock):
    tokenizer = HolarchCodeTextTokenizer(vocab_lock=lock)
    assert tokenizer.vocab_size == 2048
    assert tokenizer.elastic_cap == 2048
    assert len(tokenizer) == 2048
    assert tokenizer.tier2_ids == ()
    assert tokenizer.tier_of(2047) == 1


def test_elastic_cap_must_cover_the_locked_tier(lock):
    with pytest.raises(ValueError):
        HolarchCodeTextTokenizer(vocab_lock=lock, elastic_cap=1024)


def test_tier2_admits_novel_identifiers_without_moving_a_tier1_id(lock):
    base = HolarchCodeTextTokenizer(vocab_lock=lock)
    elastic = HolarchCodeTextTokenizer(vocab_lock=lock, elastic_cap=4096)
    before = dict(base.stoi)
    report = elastic.sync_corpus(_mutation_corpus(), namespace="tenant-test")
    assert report["added"] == TENANT_FUNCTIONS
    assert report["tier2_ids"] == TENANT_FUNCTIONS
    assert report["elastic_cap"] == 4096
    assert all(elastic.stoi.get(name) == token for name, token in before.items())
    tier2 = elastic.tier2_ids
    assert tier2[0] == 2048
    assert max(tier2) < 4096
    assert [elastic.token_text(token) for token in tier2[:1]] == ["live_probe_0"]
    call = "live_probe_0(value)"
    assert len(elastic.encode(call)) < len(base.encode(call))


def test_tier2_encoding_is_lossless(lock):
    elastic = HolarchCodeTextTokenizer(vocab_lock=lock, elastic_cap=4096)
    elastic.sync_corpus(_mutation_corpus(), namespace="tenant-test")
    text = "live_probe_7(live_probe_11(value))"
    assert elastic.decode(elastic.encode(text)) == text


def test_ledger_restores_the_exact_tier2_map_from_zone_c(tmp_path, lock):
    memory = ZoneCMemory(HolarchConfig.small(), backend="sqlite", path=str(tmp_path / "j2.sqlite3"))
    try:
        elastic = HolarchCodeTextTokenizer(vocab_lock=lock, elastic_cap=4096)
        elastic.sync_corpus(_mutation_corpus(), memory=memory, namespace="tenant-test")
        stored = memory.get_meta("tenant-test", VOCAB_LEDGER_KIND, "vocab_ledger")
        tier2 = memory.get_meta("tenant-test", VOCAB_LEDGER_KIND, "vocab_ledger_tier2")
        assert stored and tier2
        assert int(tier2[-1][1]["count"]) == TENANT_FUNCTIONS
        restored = HolarchCodeTextTokenizer(vocab_lock=lock, elastic_cap=4096)
        before = len(restored)
        added = restored.restore_ledger(dict(stored[-1][1]))
        # The restore replays the recorded identifier order; it does not parse
        # a corpus. Every Tier-2 name is re-appended at its pinned id.
        assert added == TENANT_FUNCTIONS
        assert before == 2048
        assert len(restored) == len(elastic)
        assert restored.stoi == elastic.stoi
        assert restored.tier2_ids == elastic.tier2_ids
    finally:
        memory.close()


def test_tier2_identifiers_get_distinct_phase_waves(lock):
    elastic = HolarchCodeTextTokenizer(vocab_lock=lock, elastic_cap=4096)
    elastic.sync_corpus(_mutation_corpus(), namespace="tenant-test")
    projector = BochnerPhaseProjector(dim=256, n_slots=4, d_sem=64, seed=20261104)
    projector.tokenizer = elastic
    fitted = 1500
    projector.n_tokens = fitted
    generator = np.random.default_rng(20261104)
    projector.embeddings = generator.standard_normal((fitted, projector.d_sem)).astype(np.float32)
    projector.class_anchors = {}
    tokens = list(elastic.tier2_ids[:24])
    waves = [projector.slot_wave(token) for token in tokens]
    fingerprints = {tuple(np.round(wave.real, 6)) for wave in waves}
    assert len(fingerprints) == len(tokens)
    assert all(abs(np.linalg.norm(wave) - 1.0) < 1e-5 for wave in waves)
    first = projector.slot_wave(tokens[0])
    second = projector.slot_wave(tokens[0])
    assert np.array_equal(first, second)
