"""Cycle AD3: record the exact execution boundary of the 96-task battery.

    python scripts/run_holarch_ad3_deliverable_and_suite.py

The four-pillar manifest is complete and the standalone stages that already
measured their capability are cited.  The fresh five-arm live execution, the
deliverable composer run and the 1,000-request plus 320-query campaign were
NOT executed in this session.  This receipt records that boundary instead of
claiming the gates.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_holarch_p_cycle as pc

from henri.holarch.aaii96 import build_manifest

RECEIPTS = ROOT / "receipts" / "holarch"
PIN_SNAPSHOT = RECEIPTS / "pinned_corpus_snapshot.json"
PIN_SHA = "d675f3816bbf472493b25f885bbb56c62c5ec88a81ab31e31a9b87b3a0c23703"
PIN_DIGEST = "8f270f34e8f35ef84e415d076bea20a7bc15edff337f89ed72ad607447ba8554"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Cycle AD3 boundary receipt")
    parser.add_argument("--receipts", default=str(RECEIPTS))
    args = parser.parse_args(argv)
    directory = Path(args.receipts).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    manifest = build_manifest()
    pinned_before = _sha(PIN_SNAPSHOT) if PIN_SNAPSHOT.exists() else ""
    pinned_digest = (
        str(json.loads(PIN_SNAPSHOT.read_text(encoding="utf-8")).get("digest", ""))
        if PIN_SNAPSHOT.exists()
        else ""
    )
    gates = {
        "ad3_total_tasks_eq_96": bool(
            int(manifest["total"]) == 96
            and int(manifest["answerable"]) == 80
            and int(manifest["traps"]["total"]) == 16
            and all(int(row["answerable"]) == 20 for row in manifest["pillars"])
        ),
        "ad3_arm0_standalone_ge_66_of_80": False,
        "ad3_arm4_beats_arm3_composite": False,
        "ad3_http_stress_and_gauntlet_320": False,
        "ad3_deliverable_composer_executed": False,
        "ad3_pinned_snapshot_unchanged": bool(
            pinned_before
            and pinned_before == pinned_after
            and pinned_digest == PIN_DIGEST
        ) if False else False,
    }
    pinned_after = _sha(PIN_SNAPSHOT) if PIN_SNAPSHOT.exists() else ""
    gates["ad3_pinned_snapshot_unchanged"] = bool(
        pinned_before == pinned_after == PIN_SHA and pinned_digest == PIN_DIGEST
    )
    receipt = {
        "stage": "AD3",
        "status": "MANIFEST_COMPLETE_LIVE_EXECUTION_NOT_RUN",
        "manifest": manifest,
        "executed": {
            "arms": [],
            "arm0_tasks": 0,
            "http_requests": 0,
            "sse_streams": 0,
            "gauntlet_queries": 0,
            "deliverables": 0,
        },
        "reason": (
            "The 96-task manifest is complete. The fresh five-arm live run, the "
            "structured deliverable composer pass and the 1,000-request plus "
            "320-query campaign were not executed in this session, so no gate "
            "that depends on them is claimed."
        ),
        "reference_evidence": {
            "cycle_ac_five_arm_suite": "receipts/holarch/ac4_cascade_and_aaii80.json",
            "cycle_ad_repository_arm0": "receipts/holarch/ad1_standalone_repo_patch.json",
            "cycle_ad_science_arm0": "receipts/holarch/ad2_relational_arc_and_science.json",
        },
        "pinned_snapshot": {
            "path": str(PIN_SNAPSHOT.relative_to(ROOT)),
            "sha256": pinned_after,
            "sha256_before": pinned_before,
            "expected_sha256": PIN_SHA,
            "expected_digest": PIN_DIGEST,
        },
        "gate": pc._finalise(gates),
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
        "passed": bool(all(gates.values())),
    }
    written = pc._write(directory, "ad3_deliverable_and_aaii96.json", receipt)
    print(
        "[ad3] manifest total=%d answerable=%d traps=%d | live execution not run"
        % (manifest["total"], manifest["answerable"], manifest["traps"]["total"])
    )
    failed = [name for name, value in receipt["gate"].items() if not bool(value)]
    print("[ad3] passed=%s failed=%s receipt=%s" % (receipt["passed"], failed, written["path"]))
    return 0 if bool(receipt["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
