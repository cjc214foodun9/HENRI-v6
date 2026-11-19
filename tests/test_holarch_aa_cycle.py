"""Cycle AA regression tests: guard, prose extraction, stateful schemas, pin.

Every test here is fast and deterministic.  The generated-code tests run
inside the same sandbox the synthesizer uses, so a candidate that passes a
test is a candidate that ran.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from henri.holarch.budget import probe_pnp_display_state
from henri.holarch.cold_synth import ColdSynthesizer, build_candidates, parse_problem
from henri.holarch.loop_schemas import stateful_candidates
from henri.holarch.pinned_split import load_snapshot, snapshot_train_ids
from henri.holarch.prose_examples import extract
from henri.holarch.sandbox import SubprocessSandboxExecutor
from henri.holarch.spend_guard import SpendGuard, SpendGuardError

ROOT = Path(__file__).resolve().parents[1]
PIN = "8f270f34e8f35ef84e415d076bea20a7bc15edff337f89ed72ad607447ba8554"


# ------------------------------------------------------------------ guard
def test_spend_guard_releases_and_counts(tmp_path: Path) -> None:
    lock = tmp_path / "runner.lock"
    guard = SpendGuard(lock, budget=2)
    guard.acquire()
    assert lock.exists()
    assert guard.spend() == 1
    assert guard.spend() == 2
    with pytest.raises(SpendGuardError):
        guard.spend()
    guard.release()
    assert not lock.exists()


def test_spend_guard_refuses_a_second_runner(tmp_path: Path) -> None:
    lock = tmp_path / "runner.lock"
    first = SpendGuard(lock, budget=4)
    first.acquire()
    second = SpendGuard(lock, budget=4)
    with pytest.raises(SpendGuardError):
        second.acquire()
    assert second.calls == 0
    first.release()


def test_spend_guard_reclaims_a_stale_lock(tmp_path: Path) -> None:
    lock = tmp_path / "runner.lock"
    lock.write_text(json.dumps({"pid": 999999, "started": 0.0}), encoding="utf-8")
    guard = SpendGuard(lock, budget=1)
    guard.acquire()
    assert guard.reclaimed_stale is True
    guard.release()


def test_spend_guard_requires_the_lock(tmp_path: Path) -> None:
    guard = SpendGuard(tmp_path / "runner.lock", budget=1)
    with pytest.raises(SpendGuardError):
        guard.spend()


# ------------------------------------------------------------------ prose
def test_prose_extractor_reads_arrow_and_equals() -> None:
    result = extract(
        "Example: f(1, 2) == 3\nAlso f(4, 5) -> 9\n",
        "f",
        ["a", "b"],
    )
    assert "assert (f(1, 2)) == (3)" in result["assertions"]
    assert "assert (f(4, 5)) == (9)" in result["assertions"]
    assert result["formats"]["call_equals"] >= 1
    assert result["formats"]["call_arrow"] >= 1


def test_prose_extractor_reads_input_output_blocks() -> None:
    prompt = (
        "def g(nums):\n"
        '    """\n'
        "    Example:\n"
        "    Input: nums = [1, 2, 3]\n"
        "    Output: 6\n"
        '    """\n'
    )
    result = extract(prompt, "g", ["nums"])
    assert result["assertions"] == ["assert (g(nums=[1, 2, 3])) == (6)"]


def test_prose_extractor_drops_non_literals_and_reports_empty() -> None:
    result = extract("f(x) == some_variable", "f", ["x"])
    assert result["assertions"] == []
    assert result["reason"] == "no_verifiable_io_examples"
    assert result["dropped"] >= 1


def test_prose_extractor_does_not_claim_assert_statements() -> None:
    result = extract("assert f(1) == 2", "f", ["a"])
    assert result["assertions"] == []


# ------------------------------------------------------- stateful schemas
def test_stateful_schemas_are_multi_statement_and_generic() -> None:
    prompt = (
        "def total_of_positive(values: list) -> int:\n"
        '    """Return the sum of the positive values.\n\n'
        "    >>> total_of_positive([1, -2, 3])\n"
        "    4\n"
        '    """\n'
    )
    spec = parse_problem(prompt)
    schemas = stateful_candidates(spec)
    assert schemas
    bodies = [row["body"] for row in schemas]
    assert any("\n" in body for body in bodies)
    assert all("total_of_positive" not in body for body in bodies)
    assert all("HumanEval" not in body for body in bodies)


def test_loop_schema_solves_a_public_example_task(tmp_path: Path) -> None:
    prompt = (
        "def count_positive(values: list) -> int:\n"
        '    """Count the positive values.\n\n'
        "    >>> count_positive([1, -2, 3])\n"
        "    2\n"
        "    >>> count_positive([-1, -2])\n"
        "    0\n"
        '    """\n'
    )
    sandbox = SubprocessSandboxExecutor("pure_algorithmic", timeout_s=8.0)
    synthesizer = ColdSynthesizer(sandbox=sandbox, budget=64, plan_store=None)
    outcome = synthesizer.synthesize(prompt)
    assert outcome["status"] == "SOLVED"
    assert "count_positive" in outcome["source"]
    assert "for " in outcome["source"]


# -------------------------------------------------------------------- pin
def test_frozen_pinned_snapshot_is_the_z_digest() -> None:
    snapshot = load_snapshot(ROOT)
    assert snapshot["_digest_recomputed"] == PIN
    assert int(snapshot["train_tokens"]) == 758656
    assert int(snapshot["val_tokens"]) == 84086
    assert len(snapshot["files"]) == 38
    assert all("HOLARCH.md" not in row["name"] for row in snapshot["files"])
    ids = snapshot_train_ids(snapshot)
    assert len(ids) == 758656


# --------------------------------------------------------------- hardware
def test_hardware_probe_separates_ghost_from_pcie() -> None:
    report = probe_pnp_display_state()
    provenance = report.get("hardware_provenance")
    if report.get("queried") is False:
        pytest.skip("PnP query is Windows-only")
    assert provenance is not None
    assert "ghost_registry_count" in provenance
    assert "active_pcie_count" in provenance
    if provenance["nvidia_pnp_count"] == 0:
        assert provenance["cuda_status"] == "BLOCKED_NO_PCIE_NVIDIA_DEVICE"
