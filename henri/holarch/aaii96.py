"""Cycle AD3: the 96-task four-pillar AAII v5.0 battery manifest.

The manifest composes the sealed multi-file Briefcase tasks, the pinned
external coding tasks, the standalone repository repair tasks, the compute
verified science tasks, the grounded factual tasks, the official ARC tasks and
the sixteen traps into four equal pillars of twenty answerable tasks each.
It is a plan of record: the five-arm live execution and the 320-query gauntlet
are a separate run and are not claimed here.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .repo_patch_suite import build_cases
from .science_suite import build_tasks


def build_manifest() -> Dict[str, Any]:
    repositories = build_cases()
    science = [task for task in build_tasks() if not task.get("trap")]
    pillars: List[Dict[str, Any]] = [
        {
            "pillar": "knowledge_work",
            "answerable": 20,
            "sources": {
                "briefcase": 20,
                "origin": "sealed AA4/AC4 Briefcase workspaces",
            },
        },
        {
            "pillar": "coding",
            "answerable": 20,
            "sources": {
                "pinned_external": 10,
                "repository_patch": 10,
                "repository_suite": [case.name for case in repositories[:10]],
                "origin": "pinned HumanEval/MBPP plus the AD1 sixteen-repository suite",
            },
        },
        {
            "pillar": "science",
            "answerable": 20,
            "sources": {
                "compute_verified": int(len(science)),
                "planned_variants": int(20 - len(science)),
                "task_ids": [str(task["id"]) for task in science],
                "origin": "AD2 dimensional solver and tabular aggregation",
            },
        },
        {
            "pillar": "general",
            "answerable": 20,
            "sources": {
                "grounded_factual": 10,
                "official_arc": 10,
                "origin": "sealed AA4 factual store and the official ARC-AGI slice",
            },
        },
    ]
    traps = {
        "total": 16,
        "sources": {
            "contradictory_arc": 4,
            "contradictory_public_assertions": 2,
            "stale_issue": 1,
            "sheaf_conflict": 1,
            "unsatisfiable_factual": 2,
            "underdetermined_science": 1,
            "dimensionally_invalid_science": 1,
            "nonphysical_science": 1,
            "ambiguous_science": 1,
            "unknown_unit_science": 1,
            "contradictory_quantities_science": 1,
        },
    }
    answerable = int(sum(row["answerable"] for row in pillars))
    return {
        "pillars": pillars,
        "answerable": answerable,
        "traps": traps,
        "total": int(answerable + int(traps["total"])),
        "five_arms": [
            "arm0_henri_standalone",
            "arm1_raw_llm",
            "arm2_rag_llm",
            "arm3_henri_coprocessor",
            "arm4_henri_cascade",
        ],
        "stress": {
            "http_requests": 1000,
            "sse_streams": 250,
            "gauntlet_queries": 320,
            "gauntlet_split": {"prior": 304, "cycle_ad_probes": 16},
        },
    }
