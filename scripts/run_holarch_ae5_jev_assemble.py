"""Cycle AE5: assemble the AE JEV receipt from the measured proposals and verdicts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_holarch_p_cycle as pc

RECEIPTS = ROOT / "receipts" / "holarch"
WORK = RECEIPTS / "_work_ae"


def main() -> int:
    proposals = json.loads((WORK / "ae_jev_proposals.json").read_text(encoding="utf-8"))
    challenges = {}
    accepted = []
    scores = []
    failed_union = []
    for key, verdict_name in (("ae3", "ae3_jev_verdict.json"), ("ae4", "ae4_jev_verdict.json")):
        proposal = proposals[key]
        verdict = json.loads((WORK / verdict_name).read_text(encoding="utf-8"))
        challenges[proposal["proposal_id"]] = {"proposal": proposal, "verdict": verdict}
        accepted.append(bool(verdict.get("accepted")))
        scores.append(float(verdict.get("score") or 0.0))
        for name in verdict.get("failed") or []:
            if str(name) not in failed_union:
                failed_union.append(str(name))
    gates = {
        "ae_jev_two_of_two_accepted": bool(len(challenges) == 2 and all(accepted) and not failed_union),
        "ae_jev_scores_ge_094": bool(scores and min(scores) >= 0.94),
        "ae_jev_expected_challenges": bool(
            "AE3-COMPOSITIONAL-TOROIDAL-OPERATOR-LOO-GENERALIZATION" in challenges
            and "AE4-RIEMANNIAN-CHART-SUBLINEAR-SCALING-AND-DRIFT-INVARIANCE" in challenges
        ),
    }
    receipt = {
        "stage": "AE5",
        "kind": "jev_mathematical_challenges",
        "challenges": challenges,
        "accepted": bool(len(challenges) == 2 and all(accepted) and not failed_union),
        "scores": scores,
        "failed_primitives": failed_union,
        "tool": "henri_jev (Tier-1 ontological gate)",
        "gate": pc._finalise(gates),
        "evidence": "OBSERVED",
        "passed": bool(all(gates.values())),
    }
    written = pc._write(RECEIPTS, "ae_jev_mathematical_challenges.json", receipt)
    print("[ae5-jev] accepted=%s scores=%s receipt=%s" % (receipt["accepted"], scores, written["path"]))
    failed = [name for name, value in receipt["gate"].items() if not bool(value)]
    print("[ae5-jev] passed=%s failed=%s" % (receipt["passed"], failed))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
