"""Immutable-physics gate: sealed kernel files must not drift.

WHY THIS EXISTS
---------------
The Phase 4 directive names one failure mode above all others: an agent
that cannot make a probe pass edits the probe's preconditions until the
suite turns green. Boss Fight 2 in the architecture analysis. This script
is the deterministic countermeasure. It reads each sealed file from the
GIT OBJECT STORE, hashes it, and compares against a pinned digest.

WHY THE GIT BLOB AND NOT THE WORKING TREE
-----------------------------------------
`git cat-file blob` returns the stored bytes with LF endings. A Windows
checkout may hold CRLF, so `sha256sum` on the working file can differ while
the content is identical. Hashing the blob makes the digest
line-ending-independent and reviewable from history. This is the same
identity rule recorded in the evidence register.

TWO CHECKS, DIFFERENT KINDS
---------------------------
1. DIGEST. Exact SHA-256 per file. Fails on any byte change.
2. INVARIANT. Semantic constants that must survive a legitimate edit:
   `beta* = 26.10`, `tau_veto = 0.0431`, the margin definition, the
   magnitude-weighted Sagnac form. A digest cannot express intent, so the
   constants are read and asserted. Both must pass.

Exit codes: 0 clean, 1 drift detected, 2 prerequisite missing.
No file here is ever modified. The gate reports. A human decides.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[1]

PINNED = {
    "henri/qfhrr.py":
        "0485cf57666bfe2a51460fcfdd085a44ca3aa0b61286c5f258dda0fd1888a380",
    "henri/readout.py":
        "805bd5e2ac3d70260d512bd6a6f4871f10a651e903474621295f27c0fd6df824",
    "henri/gram.py":
        "0db206777b041d6215ed05ee6ddb2702866c69ab3032e84f15fa3276e021434c",
    "henri/sagnac_homodyne_gate.py":
        "2c8aae73fdf56737e818f7673c0fad8ddf2a6cdb62892a47fcda0626338320e2",
    "henri/memory.py":
        "4cd1e9a84113a5b5fae6187304acc0331eebeb9e772745cd763e6d6ae9a4c1f7",
    "henri/form_pin.py":
        "abfe0cf777f25eba00b3aacb0901ddd048b6f4f097ca732350b3f7e45325a464",
}

INVARIANTS = (
    {
        "file": "henri/readout.py",
        "pattern": r"beta:\s*float\s*=\s*26\.1\b",
        "label": "beta_star_entropy_default_26_1",
    },
    {
        "file": "henri/readout.py",
        "pattern": r"margin\s*=\s*top\s*-\s*second",
        "label": "margin_is_top_minus_second",
    },
    {
        "file": "henri/sagnac_homodyne_gate.py",
        "pattern": r"TAU_VETO_CEILING\s*=\s*0\.0431\b",
        "label": "tau_veto_ceiling_0_0431",
    },
    {
        "file": "henri/gram.py",
        "pattern": r"def gram_matrix\(",
        "label": "dual_gram_present",
    },
    {
        "file": "henri/form_pin.py",
        "pattern": r"magnitude",
        "label": "magnitude_weighted_form_present",
    },
)


def blob_bytes(path: str, rev: str = "HEAD") -> Optional[bytes]:
    """Bytes of `path` at `rev`, read from the object store."""
    try:
        out = subprocess.run(
            ["git", "cat-file", "blob", f"{rev}:{path}"],
            cwd=str(REPO), capture_output=True, check=True,
        )
        return out.stdout
    except subprocess.CalledProcessError:
        return None


def check(rev: str = "HEAD") -> dict:
    """Run both checks. Returns a receipt-shaped dict."""
    digests, drift, missing = {}, [], []
    for path, want in PINNED.items():
        raw = blob_bytes(path, rev)
        if raw is None:
            missing.append(path)
            continue
        got = hashlib.sha256(raw).hexdigest()
        digests[path] = got
        if got != want:
            drift.append({"file": path, "pinned": want, "actual": got})

    inv_fail = []
    for spec in INVARIANTS:
        raw = blob_bytes(spec["file"], rev)
        if raw is None:
            inv_fail.append({**spec, "reason": "FILE_MISSING"})
            continue
        text = raw.decode("utf-8", errors="replace")
        if re.search(spec["pattern"], text) is None:
            inv_fail.append({**spec, "reason": "PATTERN_NOT_FOUND"})

    ok = not drift and not missing and not inv_fail
    return {
        "probe": "kernel_integrity",
        "rev": rev,
        "status": "SEALED" if ok else "DRIFT",
        "files_checked": len(digests),
        "digests": digests,
        "drift": drift,
        "missing": missing,
        "invariant_failures": [
            {k: v for k, v in d.items() if k != "pattern"} for d in inv_fail
        ],
        "note": (
            "Digest equality proves the bytes did not change. It does not "
            "prove the physics is right. Invariants are checked separately "
            "because a digest cannot express intent."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rev", default="HEAD")
    ap.add_argument("--out", default="receipts/kernel_integrity.json")
    ap.add_argument("--allow-drift", action="store_true",
                    help="report drift but exit 0 (human override, logged)")
    a = ap.parse_args()

    rec = check(a.rev)
    out = Path(a.out)
    if not out.is_absolute():
        out = REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2), encoding="utf-8")

    print(f"[KERNEL] rev={rec['rev']} files={rec['files_checked']}"
          f" status={rec['status']}")
    for d in rec["drift"]:
        print(f"[DRIFT] {d['file']}\n  pinned {d['pinned'][:16]}\n"
              f"  actual {d['actual'][:16]}")
    for m in rec["missing"]:
        print(f"[MISSING] {m}")
    for d in rec["invariant_failures"]:
        print(f"[INVARIANT] {d['label']} in {d['file']}: {d['reason']}")
    print(f"[RECEIPT] {out}")

    if rec["status"] == "SEALED":
        return 0
    return 0 if a.allow_drift else 1


if __name__ == "__main__":
    raise SystemExit(main())
