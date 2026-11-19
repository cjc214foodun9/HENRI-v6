"""Service adapters for Cycle AD deterministic solvers (Arm0).

Exposes three verified HENRI-v6 Cycle AD engines with fail-closed trap detection:
  1. `RepoPatchEngine` (`henri/holarch/repo_patch.py`) — multi-file AST fault
     localization, patch synthesis, sandbox verification, and Zone C repulsors.
  2. `ScienceSolver` (`henri/holarch/science_solver.py`) — dimensional analysis,
     inverse verification, sandbox parity, and ontology grounding.
  3. `ObjectLooInducer` (`henri/holarch/arc_objects.py`) — strict Leave-One-Out
     relational object program induction for ARC-AGI grids.

All three adapters run at zero LLM calls and return normalized `abstained` and
`trap_detected` flags so the Arm0->Arm4 `CascadeRouter` never escalates a trap.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from henri.holarch.arc_objects import (
    ObjectLooInducer,
    apply_chain,
    as_grid,
    propose_operators,
)
from henri.holarch.repo_patch import RepoCase, RepoPatchEngine
from henri.holarch.science_solver import ScienceSolver

REPO_ROOT = Path(__file__).resolve().parents[2]

REPO_PATCH_TRAP_REASONS = frozenset(
    {
        "contradictory_public_specification",
        "issue_conflict_public_tests_pass",
    }
)

SCIENCE_TRAP_PREFIXES = (
    "contradictory_quantity_",
    "nonphysical_",
    "unknown_unit_",
    "unknown_target_unit_",
    "malformed_quantities",
)


def run_repo_patch(
    *,
    name: str,
    issue: str,
    files: Dict[str, str],
    public: Dict[str, str],
    test_path: str = "tests/test_public.py",
    rollouts: int = 24,
    timeout_s: float = 30.0,
    memory: Any = None,
    namespace: str = "default",
) -> Dict[str, Any]:
    """Run Cycle AD1 `RepoPatchEngine` inside an isolated temporary workspace."""
    case = RepoCase(
        name=str(name),
        issue=str(issue),
        files={str(k): str(v) for k, v in files.items()},
        public={str(k): str(v) for k, v in public.items()},
        hidden={},
        test_path=str(test_path),
    )
    workdir = Path(tempfile.mkdtemp(prefix="henri_gw_repo_patch_"))
    try:
        engine = RepoPatchEngine(
            rollouts=max(1, min(int(rollouts), 64)),
            timeout_s=max(1.0, min(float(timeout_s), 90.0)),
            memory=memory,
            namespace=str(namespace),
            workdir=workdir,
        )
        raw = engine.solve(case, workspace=workdir / "repo")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    status = str(raw.get("status", "abstained"))
    reason = str(raw.get("reason", ""))
    abstained = status.lower() not in {"emitted", "patched", "solved"} and not bool(
        raw.get("emitted", False)
    )
    trap_detected = bool(
        (raw.get("public_conflict") or {}).get("detected")
        or reason in REPO_PATCH_TRAP_REASONS
    )
    return {
        **raw,
        "abstained": abstained,
        "trap_detected": trap_detected,
        "trap_reason": reason if trap_detected else "",
        "evidence_label": "OBSERVED",
    }


def run_science_solve(
    problem: Dict[str, Any],
    *,
    ontology_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run Cycle AD2 `ScienceSolver` with dimensional, inverse, and ontology checks."""
    solver = ScienceSolver(ontology_root=ontology_root or REPO_ROOT)
    workdir = Path(tempfile.mkdtemp(prefix="henri_gw_science_"))
    try:
        raw = solver.solve(dict(problem), workspace=workdir)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    status = str(raw.get("status", "ABSTAIN")).upper()
    reason = str(raw.get("reason", ""))
    abstained = status != "SOLVED"
    trap_detected = any(reason.startswith(prefix) for prefix in SCIENCE_TRAP_PREFIXES)
    return {
        **raw,
        "abstained": abstained,
        "trap_detected": trap_detected,
        "trap_reason": reason if trap_detected else "",
        "evidence_label": "OBSERVED",
    }


def run_arc_solve(
    *,
    train: Sequence[Dict[str, Any]],
    test: Sequence[Dict[str, Any]],
    node_budget: int = 60_000,
    depth: int = 2,
    time_budget: float = 10.0,
) -> Dict[str, Any]:
    """Run Cycle AC3/AD2 `ObjectLooInducer` with strict Leave-One-Out gating.

    Supports both scored evaluation (when test items include `output`) and
    live inference (when test items contain only `input`), returning exact
    predicted grids when a program passes strict Leave-One-Out verification
    or fail-closed `ABSTAIN` otherwise.
    """
    train_pairs: List[Tuple[np.ndarray, np.ndarray]] = [
        (as_grid(pair["input"]), as_grid(pair["output"])) for pair in train
    ]
    test_inputs: List[np.ndarray] = [as_grid(item["input"]) for item in test]
    has_expected = bool(test) and all("output" in item and item["output"] is not None for item in test)
    scored_tests: List[Tuple[np.ndarray, np.ndarray]] = (
        [(as_grid(item["input"]), as_grid(item["output"])) for item in test]
        if has_expected
        else []
    )

    inducer = ObjectLooInducer(
        node_budget=max(1000, min(int(node_budget), 500_000)),
        depth=max(1, min(int(depth), 3)),
        time_budget=max(0.5, min(float(time_budget), 30.0)),
    )
    raw = inducer.induce(train_pairs, scored_tests)

    predictions: List[Optional[List[List[int]]]] = []
    if raw.get("emitted") and raw.get("chain") and test_inputs:
        chain_labels = list(raw["chain"])
        chosen_ops = _reconstruct_chain(train_pairs, chain_labels, inducer)
        if chosen_ops is not None:
            for grid_in in test_inputs:
                pred = apply_chain(chosen_ops, grid_in)
                predictions.append(None if pred is None else pred.astype(int).tolist())

    abstained = not bool(raw.get("emitted", False))
    return {
        **raw,
        "predictions": predictions,
        "abstained": abstained,
        "trap_detected": False,
        "trap_reason": "",
        "llm_calls": 0,
        "evidence_label": "OBSERVED",
    }


def _reconstruct_chain(
    train_pairs: Sequence[Tuple[np.ndarray, np.ndarray]],
    chain_labels: Sequence[str],
    inducer: ObjectLooInducer,
):
    """Re-materialize the exact admitted ArcOp chain for live test prediction."""
    if len(chain_labels) == 1:
        target_label = chain_labels[0]
        for op in propose_operators(train_pairs):
            if op.label() == target_label and inducer._matches(op, train_pairs):
                return [op]
        return None
    if len(chain_labels) == 2:
        state = {"nodes": 0, "exhausted": 0}
        for head, second in inducer._depth_two(train_pairs, state):
            if [head.label(), second.label()] == list(chain_labels):
                return [head, second]
        return None
    if len(chain_labels) == 3:
        state = {"nodes": 0, "exhausted": 0}
        fams = tuple(str(lbl).split("(")[0] for lbl in chain_labels)
        for op1, op2, op3 in inducer._depth_three(train_pairs, state, families=fams):
            if [op1.label(), op2.label(), op3.label()] == list(chain_labels):
                return [op1, op2, op3]
    return None
