"""Promotion gate: the single verdict that authorizes a merge to main.

WHY A SCRIPT AND NOT A CHECKLIST
--------------------------------
Owner approval for promotion is conditional: "given a required-test gate".
A gate that exists as prose is not a gate. This one runs, returns one
verdict, and writes a receipt naming the commit it judged.

WHAT IT CHECKS, AND WHY EACH CHECK CAN FAIL
-------------------------------------------
1. WORKING TREE CLEAN. A promotion from a dirty tree ships bytes that no
   commit contains. Fails on any untracked or modified path.
2. KERNEL SEALED. `verify_kernel_integrity.check()` must return SEALED.
   Fails if a sealed file drifts, or if an invariant vanishes.
3. SUITE GREEN. Full pytest run: zero failures, zero errors, and a test
   floor. The floor catches a collection error that reports "0 failed".
4. HARDWARE SKIPS ALLOWED, NEVER A PASS. A skipped hardware test is
   reported as skipped. It never counts toward a pass.

NOT CHECKED, STATED PLAINLY
---------------------------
No benchmark score is a promotion input. This repository has no measured
above-chance task result, so making accuracy a gate would either block
forever or invite a number invented to clear it. Promotion here means the
kernel is sealed and the suite is green. It does NOT mean the model is
intelligent, and no receipt from this script claims that.

Exit codes: 0 promote, 1 block, 2 prerequisite missing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

TEST_FLOOR = 370
"""Minimum collected tests. Guards against a silent collection failure.

377 collected at the time of writing. The floor sits 7 below that so an
intentional test removal is possible but a collapse is not.
"""


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(REPO),
                          capture_output=True, text=True)


def tree_clean() -> tuple:
    """True when nothing is modified and nothing is untracked."""
    out = git("status", "--porcelain")
    if out.returncode != 0:
        return False, "git status failed"
    lines = [ln for ln in out.stdout.splitlines() if ln.strip()]
    if lines:
        return False, f"{len(lines)} dirty paths, first: {lines[0].strip()}"
    return True, "clean"


def run_suite(junit: Path) -> dict:
    """Run the full suite and parse the junit XML."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", f"--junitxml={junit}"],
        cwd=str(REPO), capture_output=True, text=True,
    )
    if not junit.exists():
        return {"ran": False, "reason": "no junit written",
                "rc": proc.returncode, "tail": proc.stdout[-800:]}
    import xml.etree.ElementTree as ET

    root = ET.parse(junit).getroot()
    suite = root if root.tag == "testsuite" else root[0]
    def _int(key: str) -> int:
        return int(suite.get(key) or 0)
    return {
        "ran": True,
        "rc": proc.returncode,
        "tests": _int("tests"),
        "failures": _int("failures"),
        "errors": _int("errors"),
        "skipped": _int("skipped"),
        "time_s": float(suite.get("time") or 0.0),
    }


def gate(rev: str) -> dict:
    clean, clean_note = tree_clean()

    import verify_kernel_integrity as vk
    seal = vk.check(rev)

    junit = REPO / "receipts" / "junit_promotion.xml"
    junit.parent.mkdir(parents=True, exist_ok=True)
    suite = run_suite(junit)

    reasons = []
    if not clean:
        reasons.append(f"TREE_DIRTY: {clean_note}")
    if seal["status"] != "SEALED":
        reasons.append(f"KERNEL_DRIFT: {len(seal['drift'])} files, "
                       f"{len(seal['invariant_failures'])} invariants")
    if not suite.get("ran"):
        reasons.append("SUITE_DID_NOT_RUN")
    else:
        if suite["failures"]:
            reasons.append(f"FAILURES: {suite['failures']}")
        if suite["errors"]:
            reasons.append(f"ERRORS: {suite['errors']}")
        if suite["tests"] < TEST_FLOOR:
            reasons.append(f"TEST_FLOOR: {suite['tests']} < {TEST_FLOOR}")

    ok = not reasons
    head = git("rev-parse", "HEAD").stdout.strip()
    rec = {
        "probe": "promotion_gate",
        "verdict": "PROMOTE" if ok else "BLOCK",
        "reasons": reasons,
        "rev": rev,
        "head": head,
        "head_short": head[:7],
        "tree_clean": clean,
        "tree_note": clean_note,
        "kernel_status": seal["status"],
        "suite": suite,
        "test_floor": TEST_FLOOR,
        "scope": (
            "Authorizes a merge of verified kernel state. Makes no claim "
            "about model intelligence, benchmark accuracy, or API readiness."
        ),
    }
    # Self-digest over the verdict-bearing fields only, so a reader can
    # check the receipt was not edited after the run.
    core = json.dumps({k: rec[k] for k in
                       ("verdict", "reasons", "head", "kernel_status",
                        "suite")}, sort_keys=True)
    rec["receipt_sha256"] = hashlib.sha256(core.encode()).hexdigest()
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rev", default="HEAD")
    ap.add_argument("--out", default="receipts/promotion_gate.json")
    a = ap.parse_args()
    rec = gate(a.rev)
    out = Path(a.out)
    if not out.is_absolute():
        out = REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2), encoding="utf-8")

    print(f"[GATE] verdict={rec['verdict']} rev={rec['head_short']}")
    print(f"[TREE] {rec['tree_note']}")
    print(f"[KERNEL] {rec['kernel_status']}")
    s = rec["suite"]
    if s.get("ran"):
        print(f"[SUITE] tests={s['tests']} failures={s['failures']}"
              f" errors={s['errors']} skipped={s['skipped']}"
              f" time={s['time_s']:.2f}s floor={TEST_FLOOR}")
    for r in rec["reasons"]:
        print(f"[BLOCK] {r}")
    print(f"[DIGEST] {rec['receipt_sha256'][:32]}")
    print(f"[RECEIPT] {out}")
    return 0 if rec["verdict"] == "PROMOTE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
