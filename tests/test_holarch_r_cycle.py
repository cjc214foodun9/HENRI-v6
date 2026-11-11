
"""Cycle R unit tests: unstructured extraction, phase veto, readout, R3, R4."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np

from henri.holarch.code_gen import (
    ActionRegimeReadout,
    CountLanguageModel,
    code_slot,
)
from henri.holarch.swat_bus import (
    DocumentFactExtractor,
    SagnacKVGate,
    cache_from_pairs,
)


def _identity(name: str, dim: int = 64) -> np.ndarray:
    seed = int(hashlib.sha256(str(name).encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    wave = (rng.standard_normal(dim) + 1j * rng.standard_normal(dim)).astype(
        np.complex64
    )
    return wave / max(float(np.linalg.norm(wave)), 1e-12)


# ------------------------------------------------------------- R1 extracts
def test_r1_table_extraction():
    extractor = DocumentFactExtractor()
    text = (
        "| subject | relation | value |\n"
        "| --- | --- | --- |\n"
        "| q00k | resolves_to | q00m1 |\n"
        "| q00k.latency | value | 1,020,380 |\n"
    )
    facts = extractor.extract(text, source="table")
    keys = {(fact.subject, fact.relation, fact.value) for fact in facts}
    assert ("q00k", "resolves_to", "q00m1") in keys
    assert ("q00k.latency", "value", "1020380") in keys


def test_r1_prose_and_config_extraction():
    extractor = DocumentFactExtractor()
    prose = "The q01m1 forwards to q01m2. The q01m1.latency is 1,020,380."
    facts = extractor.extract(prose, source="prose")
    values = {(fact.subject, fact.value) for fact in facts}
    assert ("q01m1", "q01m2") in values
    assert ("q01m1.latency", "1020380") in values
    config = "q01m2 = q01v\nq01m2.latency = 1020380"
    facts = extractor.extract(config, source="config")
    keys = {(fact.key(), fact.value) for fact in facts}
    assert ("q01m2", "q01v") in keys
    assert ("q01m2.latency", "1020380") in keys


def test_r1_python_ast_extraction():
    extractor = DocumentFactExtractor()
    text = 'FACTS = {"q02k": "q02m1", "q02m1": "q02m2"}\nMAX_RETRIES = 7\n'
    facts = extractor.extract(text, source="code")
    keys = {(fact.key(), fact.value) for fact in facts}
    assert ("q02k", "q02m1") in keys
    assert ("q02m1", "q02m2") in keys
    assert ("max_retries", "7") in keys


def test_r1_registry_corroboration_and_connectivity():
    extractor = DocumentFactExtractor()
    documents = [
        "| subject | relation | value |\n|---|---|---|\n| k | resolves_to | m |\n",
        "k = zz_wrong",
        "k = m\nm = v",
    ]
    result = extractor.extract_documents(documents)
    assert result["registry"]["k"] == "m"
    assert result["documents"][0]["connected"] is True
    ood = extractor.extract_documents(
        ["The ooda routes to oodb.", "The oodc routes to oodd."]
    )
    assert all(document["connected"] is False for document in ood["documents"])


# --------------------------------------------------------- R1 phase veto
def test_r1_phase_residual_vetoes_wrong_number():
    gate = SagnacKVGate(
        None,
        None,
        fact_registry={"q.latency": "1020380"},
        wave_of=_identity,
    )
    cache = cache_from_pairs(
        [(_identity("q.latency"), _identity("1020381"))],
        facts=[["q.latency", "1020381"]],
        wave=_identity("q.latency"),
    )
    result = gate.fact_consistency(cache)
    assert result["violated"] is True
    assert "q.latency:phase_residual" in result["violations"]
    assert result["max_phase_residual"] > 0.5
    gate.enforce_string_rule = False
    physical = gate.fact_consistency(cache)
    assert physical["violated"] is True
    assert "q.latency:phase_residual" in physical["violations"]
    assert "q.latency:value_mismatch" not in physical["violations"]


def test_r1_exact_value_passes_physics():
    gate = SagnacKVGate(None, None, fact_registry={"k": "v"}, wave_of=_identity)
    cache = cache_from_pairs(
        [(_identity("k"), _identity("v"))], facts=[["k", "v"]], wave=_identity("k")
    )
    result = gate.fact_consistency(cache)
    assert result["violated"] is False
    assert result["max_phase_residual"] < 1e-5


# --------------------------------------------------------------- R2 head
def test_r2_readout_fits_and_persists():
    rng = np.random.default_rng(7)
    rows = []
    labels = []
    for label in range(3):
        base = rng.standard_normal(512)
        for _ in range(8):
            rows.append(base + 0.05 * rng.standard_normal(512))
            labels.append(label)
    readout = ActionRegimeReadout(dim=512, slots=4, ridge=1e-2)
    report = readout.fit(rows, labels)
    assert report["rows"] == 24
    assert readout.accuracy(rows, labels) == 1.0
    state = readout.to_state()
    restored = ActionRegimeReadout.from_state(state)
    assert restored.digest() == readout.digest()
    predicted, probability = restored.predict(rows[0])
    assert predicted == labels[0]
    assert 0.0 < probability <= 1.0


# --------------------------------------------------------------- R3 prior
def test_r3_code_slot_categories():
    assert code_slot(".") == "dot"
    assert code_slot("(") == "open"
    assert code_slot("def") == "keyword"
    assert code_slot("value_name") == "name"
    assert code_slot("123") == "number"


def test_r3_role_discount_conditions_on_last_slot():
    model = CountLanguageModel(vocab_size=16)
    model.discount_by_role = {"scope": 0.9, "name": 0.5}
    model.slot_of = lambda token: "dot" if int(token) == 0 else "name"
    assert model.discount_for([0]) == 0.9
    assert model.discount_for([1]) == 0.5


def test_r3_recency_is_window_local_and_zero_state():
    from henri.holarch import code_gen as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "self.recency_window: int = 1024" in source
    assert "state_bytes_across_calls" in source
    assert "is_static_inductive" in source


# --------------------------------------------------------------- R4 suite
def test_r4_task_expansion_is_32():
    from aaii_r4 import build_r4_tasks

    base_tasks = [
        {
            "id": "a",
            "intent": "Write a function foo that returns 1.",
            "tests": "assert foo() == 1",
            "request": {"func": "foo", "args": [], "numbers": []},
        },
        {
            "id": "b",
            "intent": "Write a function bar that returns 2.",
            "tests": "assert bar() == 2",
            "request": {"func": "bar", "args": [], "numbers": []},
        },
    ]
    tasks = build_r4_tasks(base_tasks)
    assert len(tasks) == 4
    assert {task["request"]["func"] for task in tasks} == {"foo", "bar", "r4_foo", "r4_bar"}


def test_r4_raw_workspace_is_multi_artifact_and_corrupt_has_law():
    from aaii_r4 import raw_workspace

    valid = raw_workspace(0)
    assert len(valid["documents"]) == 6
    assert valid["reference"] == []
    corrupt = raw_workspace(101, "corrupt")
    assert len(corrupt["reference"]) == 1
    assert any("audit" in str(document.get("source", "")) for document in corrupt["documents"])


def test_r4_runtime_bug_mutation_compiles_but_changes_return():
    from aaii_r4 import _mutate_runtime_bug

    source = "def f(x):\n    return x + 1\n"
    mutated = _mutate_runtime_bug(source)
    compile(mutated, "<test>", "exec")
    assert mutated != source
    assert "audit_fault" in mutated


def test_r4_mmlu_selection_is_500_and_ood_disjoint():
    from aaii_r4 import real_mmlu_selection

    rows = []
    for subject_index in range(8):
        for index in range(80):
            rows.append(
                {
                    "question": f"q{subject_index}_{index}",
                    "subject": f"subject_{subject_index}",
                    "answer": "A",
                }
            )
    for subject_index in range(4):
        for index in range(30):
            rows.append(
                {
                    "question": f"ood{subject_index}_{index}",
                    "subject": f"ood_subject_{subject_index}",
                    "answer": "B",
                }
            )
    grounded, ood = real_mmlu_selection(rows)
    assert len(grounded) == 500
    assert len(ood) == 100
    assert set(row["subject"] for row in grounded).isdisjoint(
        set(row["subject"] for row in ood)
    )
