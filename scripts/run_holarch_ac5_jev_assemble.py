"""Assemble the Cycle AC JEV receipt from the measured proposals and the plugin verdicts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_holarch_p_cycle as pc

RECEIPTS = ROOT / "receipts" / "holarch"
WORK = RECEIPTS / "_work_ac"


def main() -> int:
    proposals = json.loads((WORK / "ac_jev_proposals.json").read_text(encoding="utf-8"))
    challenges = {}
    for key, verdict_name in (("ac1", "ac1_jev_verdict.json"), ("ac3", "ac3_jev_verdict.json")):
        proposal = proposals[key]
        verdict = json.loads((WORK / verdict_name).read_text(encoding="utf-8"))
        challenges[proposal["proposal_id"]] = {
            "proposal": proposal,
            "verdict": verdict,
        }
    accepted = all(bool(row["verdict"]["accepted"]) for row in challenges.values())
    gates = {}
    for name, row in challenges.items():
        verdict = row["verdict"]
        gates["%s_accepted" % name.split("-")[0].lower()] = bool(verdict["accepted"])
        for primitive in verdict.get("primitives") or []:
            if primitive.get("kind") == "Boolean":
                gates[
                    "%s_%s" % (name.split("-")[0].lower(), primitive["name"])
                ] = bool(primitive["passed"])
    gates["ac_jev_two_of_two_accepted"] = bool(len(challenges) >= 2 and accepted)
    receipt = {
        "stage": "AC5",
        "kind": "jev_mathematical_challenges",
        "challenges": challenges,
        "accepted": bool(accepted),
        "gate": pc._finalise(gates),
        "tool": "henri_jev (Tier-1 ontological gate)",
        "note": (
            "Both proposals were gated by the live henri_jev Tier-1 tool. "
            "The energy and entropy series are measured in "
            "scripts/run_holarch_ac5_jev_proposals.py: the codebook audit for AC1 "
            "and the grid-codec round trip plus the object-mass entropy for AC3."
        ),
        "evidence": "OBSERVED",
        "passed": bool(all(gates.values())),
    }
    written = pc._write(RECEIPTS, "ac_jev_mathematical_challenges.json", receipt)
    print(
        "[ac5-jev] accepted=%s scores=%s receipt=%s"
        % (
            accepted,
            [round(float(row["verdict"]["score"]), 4) for row in challenges.values()],
            written["path"],
        )
    )
    failed = [name for name, value in gates.items() if not value]
    print("[ac5-jev] passed=%s failed=%s" % (receipt["passed"], failed))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
