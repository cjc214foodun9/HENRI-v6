"""Cycle AD contract tests: the repository patch engine and the science solver."""

from __future__ import annotations

import tempfile
from pathlib import Path

from henri.holarch.repo_patch import RepoPatchEngine
from henri.holarch.repo_patch_suite import build_cases, build_traps
from henri.holarch.science_solver import ScienceSolver
from henri.holarch.science_suite import build_tasks


def test_repository_engine_repairs_a_two_module_workspace():
    case = [row for row in build_cases() if row.name == "numeric_sign"][0]
    engine = RepoPatchEngine(rollouts=48)
    with tempfile.TemporaryDirectory(prefix="ad1_test_") as directory:
        outcome = engine.solve(case, workspace=Path(directory))
    assert outcome["status"] == "emitted"
    assert outcome["patch"]["qualname"] == "tolerance:upper_bound"
    assert "mean + delta" in outcome["patched_files"][outcome["patch"]["path"]]
    assert all(
        float(row.get("sheaf_energy", 0.0)) == 0.0
        for row in outcome["rollouts"]
        if not row.get("blocked")
    )


def test_repository_engine_refuses_a_contradictory_specification():
    case = [row for row in build_traps() if row.name == "trap_conflicting_assertions"][0]
    engine = RepoPatchEngine(rollouts=48)
    with tempfile.TemporaryDirectory(prefix="ad1_trap_") as directory:
        outcome = engine.solve(case, workspace=Path(directory))
    assert outcome["status"] == "abstained"
    assert outcome["reason"] == "contradictory_public_specification"
    assert outcome["patched_files"] == {}


def test_science_solver_computes_and_refuses():
    solver = ScienceSolver(".")
    tasks = {row["id"]: row for row in build_tasks()}
    with tempfile.TemporaryDirectory(prefix="ad2_test_") as directory:
        solved = solver.solve(tasks["sci03"], Path(directory))
        refused = solver.solve(tasks["sci19"], Path(directory))
    assert solved["status"] == "SOLVED"
    assert abs(float(solved["answer"]) - 3600.0) < 1e-9
    assert solved["dimensional_ok"] is True
    assert solved["residual"] <= 1e-9
    assert refused["status"] == "ABSTAIN"
    assert refused["reason"] == "no_dimensionally_consistent_relation"
