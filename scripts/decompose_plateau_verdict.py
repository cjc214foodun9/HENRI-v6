"""Decompose the v1 plateau verdict. Additive; edits no receipt.

WHY THIS EXISTS
---------------
`sripts/probe_readout_plateau_resolution.py` returned verdict
STILL_UNRESOLVED at D=1024 and D=65536. That verdict is CORRECT under
its own frozen rule and stands as filed. This script does not change it.

The v1 rule required three conditions simultaneously:

    C := all(legacy_entropy >= 0.99 on EVERY unrelated cell)
         AND all(entropy_beta <= 0.05 on every decisive cell)

Condition C is ILL-POSED. It conflates two independent claims and its
first clause contradicts v1's own mechanism M4, which states that a
STRUCTURED codebook pulls an unrelated query off the global mean, so the
legacy entropy dips below 0.99 on the structured family BY DESIGN.
v1 declared M4 and then wrote a condition that M4 must violate.

That is a defect in the RULE, not in the measurement. Governance says
bifurcate, never move a bar. So this script bifurcates: it recomputes
each claim separately from the SAME v1 receipt bytes and reports which
mechanisms hold and which do not.

EVIDENTIARY STATUS, STATED PLAINLY
----------------------------------
The conditions below were written AFTER seeing the v1 result. They
therefore have WEAKER status than a pre-registered rule. They are
DERIVED analysis over OBSERVED v1 cells, not a new experiment. The v1
verdict STILL_UNRESOLVED remains on the record and is not superseded by
a coincident-success reading; it is decomposed.

DECOMPOSED CLAIMS
-----------------
C1 plateau reproduces (isotropic only). Mechanism M1/M4 predicts the
   legacy plateau on the two ISOTROPIC families and NOT on the
   structured family. Test: every unrelated cell on gauss and phasor
   has legacy_entropy >= 0.99 and tie_group_size == k.
C2 mean-attractor. The cleaned wave of an isotropic unrelated query
   converges to the codebook mean: cos >= 0.99. On the structured
   family it must NOT (M4). Test both directions.
C3 decisiveness. For t >= 0.90, raw arm, every family:
   argmax == target_row AND margin >= 0.10.
C4 metric separation. The legacy metric is scale-blind; the
   beta-scaled metric separates. Test: max(entropy_beta) over decisive
   cells <= 0.05 while min(legacy_entropy) over the same cells >= 0.99.

Reported verdict is the conjunction. Any FALSE is reported as FALSE.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

V1 = Path("receipts/readout_plateau_resolution.json")
OUT = Path("receipts/readout_plateau_resolution_v2_decomposed.json")


def main() -> int:
    if not V1.exists():
        print(f"missing {V1}", file=sys.stderr)
        return 2
    v1_bytes = V1.read_bytes()
    v1 = json.loads(v1_bytes)
    v1_digest = hashlib.sha256(v1_bytes).hexdigest()
    cells = v1["cells"]
    k = v1["params"]["k"]

    unrelated = [c for c in cells if c["test"] == "A_mean_attractor"]
    iso_unrel = [c for c in unrelated if c["family"].startswith("isotropic")]
    struct_unrel = [c for c in unrelated if c["family"] == "clustered"]
    decisive = [
        c for c in cells
        if c["test"] == "B_decisive" and c.get("target_coherence", 0.0) >= 0.90
    ]

    def show(tag: str, rows: list) -> None:
        for c in rows:
            print(
                "  %-6s %-17s %-8s legacy=%.6f beta=%.6f margin=%+.6f ties=%d cosmean=%.6f"
                % (tag, c["family"], c["arm"], c["legacy_entropy"], c["entropy_beta"],
                   c["margin"], c["tie_group_size"], c["cos_with_codebook_mean"]),
                file=sys.stderr,
            )

    print("ISOTROPIC unrelated (C1/C2 expect plateau AND mean-attractor):", file=sys.stderr)
    show("iso", iso_unrel)
    print("STRUCTURED unrelated (M4 expects the plateau to BREAK here):", file=sys.stderr)
    show("struct", struct_unrel)
    print("DECISIVE t>=0.90 raw (C3/C4 expect hits AND beta collapse):", file=sys.stderr)
    show("dec", decisive)

    c1 = bool(iso_unrel) and all(
        c["legacy_entropy"] >= 0.99 and c["tie_group_size"] == k for c in iso_unrel
    )
    c2_iso = bool(iso_unrel) and all(c["cos_with_codebook_mean"] >= 0.99 for c in iso_unrel)
    c2_struct = bool(struct_unrel) and all(c["cos_with_codebook_mean"] < 0.99 for c in struct_unrel)
    c2 = c2_iso and c2_struct
    c3 = bool(decisive) and all(
        c["argmax"] == c.get("target_row") and c["margin"] >= 0.10 for c in decisive
    )
    c4 = bool(decisive) and (
        max(c["entropy_beta"] for c in decisive) <= 0.05
        and min(c["legacy_entropy"] for c in decisive) >= 0.99
    )

    conds = {
        "C1_plateau_reproduces_isotropic": c1,
        "C2_mean_attractor_including_M4_break": c2,
        "C3_decisive_on_informative": c3,
        "C4_metric_separation": c4,
    }
    verdict = "PLATEAU_EXPLAINED_GEOMETRY" if all(conds.values()) else "PARTIAL"
    if not c3:
        verdict = "READOUT_DEFECT"

    out = {
        "record": "readout_plateau_resolution_v2_decomposed",
        "kind": "DERIVED analysis over OBSERVED v1 cells",
        "derived_from": str(V1).replace("\\", "/"),
        "derived_from_sha256": v1_digest,
        "v1_verdict_preserved": v1["verdict"],
        "rule_revision_disclosure": (
            "Conditions C1-C4 were written AFTER the v1 result and have "
            "weaker status than a pre-registered rule. v1's condition C "
            "was ill-posed: its first clause contradicted v1's own "
            "mechanism M4. v1's STILL_UNRESOLVED verdict is not edited."
        ),
        "conditions": conds,
        "excluded_from_C1_by_design": {
            "family": "clustered",
            "reason": "M4 predicts the structured family breaks the plateau",
            "measured_cosmean": [c["cos_with_codebook_mean"] for c in struct_unrel],
            "measured_legacy_entropy": [c["legacy_entropy"] for c in struct_unrel],
        },
        "decisive_cell_margins": [c["margin"] for c in decisive],
        "decisive_cell_beta_entropy": [c["entropy_beta"] for c in decisive],
        "decisive_cell_legacy_entropy": [c["legacy_entropy"] for c in decisive],
        "verdict": verdict,
        "scope_limits": [
            "Synthetic geometry; no learned engram.",
            "No task attempted or solved. No capability claim.",
            "Post-hoc rule revision; weaker than pre-registration.",
        ],
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(out, indent=2, sort_keys=True).encode("utf-8")
    dig = hashlib.sha256(body).hexdigest()
    OUT.write_bytes(body)
    OUT.with_suffix(OUT.suffix + ".sha256").write_bytes((dig + "\n").encode())
    rb = hashlib.sha256(OUT.read_bytes()).hexdigest()

    print("conditions: " + json.dumps(conds, sort_keys=True), file=sys.stderr)
    print(f"verdict={verdict} sha256={dig} readback={rb == dig}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
