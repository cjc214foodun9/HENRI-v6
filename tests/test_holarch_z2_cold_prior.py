"""Cycle Z2 contracts: the structural prior and the branch audit.

These tests are self-contained: every prompt is written here, so a failure
names a general reading rule, not a benchmark task.  The audit test guards
the hard rule that no task-name, task-id or dataset-specific branch may exist
in the cold synthesizer.
"""

from __future__ import annotations

from pathlib import Path

from henri.holarch.cold_synth import (
    ColdSynthesizer,
    audit_task_specific_branches,
    build_candidates,
    parse_problem,
)

ROOT = Path(__file__).resolve().parents[1]

PROMPT = (
    "def every_other(values: list) -> list:\n"
    '    """Return every other value of values, starting with the first.\n\n'
    "    assert every_other([1, 2, 3, 4]) == [1, 3]\n"
    "    assert every_other([5, 6, 7, 8]) == [5, 7]\n"
    '    """\n'
)


def test_structural_prior_reads_the_public_examples():
    spec = parse_problem(PROMPT)
    assert spec.name == "every_other"
    assert spec.arg_kinds == {"values": "list"}
    assert spec.return_kind == "list"
    # The output is shorter than the input, so this is a stride reading, not
    # an elementwise map: the two facts are deliberately different.
    assert spec.shape["elementwise"] is False
    assert spec.shape["halving"] is True
    assert spec.shape["subset"] is True
    assert spec.shape["permutation"] is False
    assert not spec.faults
    bodies = [row["body"] for row in build_candidates(spec)]
    assert "return values[::2]" in bodies


def test_schema_family_ranks_the_stride_sketch_high():
    spec = parse_problem(PROMPT)
    ranked, v_eff = ColdSynthesizer(budget=32).rank(spec)
    assert ranked[0]["body"] == "return values[::2]"
    assert max(v_eff.values()) <= 8


def test_no_public_example_is_a_fail_closed_refusal():
    prompt = (
        "def opaque(value: int) -> int:\n"
        '    """Return the decoded value."""\n'
    )
    spec = parse_problem(prompt)
    assert not spec.assertions
    outcome = ColdSynthesizer(budget=8).synthesize(prompt)
    assert outcome["status"] == "ABSTAIN"
    assert outcome["reason"] == "no_public_examples"
    assert outcome["rollouts"] == 0


def test_arity_contradiction_is_a_spec_fault():
    prompt = (
        "def pair_sum(a: int, b: int) -> int:\n"
        '    """Return the sum of a and b.\n\n'
        "    assert pair_sum(1, 2) == 3\n"
        "    assert pair_sum(1, 2, 3) == 6\n"
        '    """\n'
    )
    spec = parse_problem(prompt)
    assert any(str(fault).startswith("arity_mismatch") for fault in spec.faults)
    outcome = ColdSynthesizer(budget=8).synthesize(prompt)
    assert outcome["status"] == "ABSTAIN"
    assert outcome["reason"].startswith("spec_fault:")


def test_branch_audit_flags_a_task_name_comparison():
    source = (
        "def drive(spec):\n"
        '    if spec.name == "every_other":\n'
        "        return 1\n"
        "    return 0\n"
    )
    report = audit_task_specific_branches(source, identifiers=["every_other"])
    assert report["ok"] is False
    assert report["task_identity_comparisons"] >= 1
    assert "every_other" in report["compared_identifier_literals"]


def test_cold_synth_has_no_task_specific_branch():
    source = (ROOT / "henri" / "holarch" / "cold_synth.py").read_text(encoding="utf-8")
    report = audit_task_specific_branches(
        source, identifiers=["count_occurance", "is_perfect_square", "58", "803"]
    )
    assert report["ok"] is True
    assert report["task_identity_comparisons"] == 0
    assert report["compared_identifier_literal_count"] == 0
    assert report["unexplained_string_comparisons"] == 0
