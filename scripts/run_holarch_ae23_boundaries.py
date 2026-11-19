"""Cycle AE2/AE3 boundary receipts: record the exact unexecuted scope.

    python scripts/run_holarch_ae23_boundaries.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_holarch_p_cycle as pc

RECEIPTS = ROOT / "receipts" / "holarch"


def main() -> int:
    started = time.perf_counter()
    ae2 = {
        "stage": "AE2",
        "status": "EXTERNAL_SLICE_NOT_VENDORED_NOT_EXECUTED",
        "targets": {
            "external_repositories": 12,
            "external_science_problems": 24,
            "repo_traps": 4,
            "science_traps": 6,
        },
        "executed": {"repositories": 0, "science_problems": 0, "traps": 0, "llm_calls": 0},
        "reason": (
            "The pinned external slices were not vendored in this session. No "
            "external pass@1, precision or trap number is claimed. AD1 16/16 and "
            "AD2 18/18 remain internal and unchanged."
        ),
        "gate": pc._finalise({
            "ae2_external_repos_vendored": False,
            "ae2_external_repo_pass_ge_8_of_12": False,
            "ae2_external_repo_precision_ge_095": False,
            "ae2_external_repo_traps_4_of_4": False,
            "ae2_external_science_vendored": False,
            "ae2_external_science_pass_ge_14_of_24": False,
            "ae2_external_science_traps_6_of_6": False,
            "ae2_zero_llm_calls": True,
            "ae2_pass": False,
        }),
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
        "passed": False,
    }
    ae3 = {
        "stage": "AE3",
        "status": "COMPOSITIONAL_DEPTH3_NOT_EXECUTED",
        "target": {"external_solved": 30, "external_total": 48, "precision": 1.0},
        "baseline": {"external_solved": 24, "external_total": 48, "precision": 1.0},
        "executed": {"depth3_search": False, "external_solved": 24},
        "reason": (
            "A bounded depth-three composition experiment was run outside the "
            "sealed runner and did not produce a new LOO-admitted task family "
            "within the session budget. The sealed AD2 24/48 result is unchanged "
            "and no new operator was added."
        ),
        "gate": pc._finalise({
            "ae3_depth2_depth3_implemented": False,
            "ae3_external_pass_ge_30_of_48": False,
            "ae3_precision_eq_10000": True,
            "ae3_zero_task_name_branches": True,
            "ae3_zero_llm_calls": True,
            "ae3_ad2_24_retained": True,
            "ae3_pass": False,
        }),
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
        "passed": False,
    }
    for name, payload in (
        ("ae2_external_repo_patch_and_science.json", ae2),
        ("ae3_compositional_external_arc.json", ae3),
    ):
        written = pc._write(RECEIPTS, name, payload)
        print("[ae23] %s passed=%s -> %s" % (name, payload["passed"], written["path"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
