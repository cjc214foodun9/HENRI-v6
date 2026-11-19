"""Cycle AD master seal: run the hygiene battery and seal the dual-cycle ledger.

    python scripts/run_holarch_ad_cycle.py

The seal is passed: true only when every AD stage receipt passes, the previous
t/u/v/w/x and AC cycles still pass, pytest exits zero with no failure or
error, the consistency verifier accepts Sections 1..45 at 1.0000 sidecar
parity, the R3 language bars hold and the frozen snapshot is byte-identical.
No bar is moved here.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_holarch_p_cycle as pc

RECEIPTS = ROOT / "receipts" / "holarch"
DOCS = ROOT / "docs" / "HOLARCH.md"
PIN_SNAPSHOT = RECEIPTS / "pinned_corpus_snapshot.json"
PIN_SHA = "d675f3816bbf472493b25f885bbb56c62c5ec88a81ab31e31a9b87b3a0c23703"
PIN_DIGEST = "8f270f34e8f35ef84e415d076bea20a7bc15edff337f89ed72ad607447ba8554"
STAGES = (
    "ad1_standalone_repo_patch.json",
    "ad2_relational_arc_and_science.json",
    "ad3_deliverable_and_aaii96.json",
    "ad4_blackwell_cuda_and_jev.json",
)
PRIOR = ("t_cycle.json", "u_cycle.json", "v_cycle.json", "w_cycle.json",
         "x_cycle.json", "ac_cycle.json")


def _read(name: str) -> Dict[str, Any]:
    path = RECEIPTS / name
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - a broken receipt is a failed gate
        return {}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sidecar_parity() -> Dict[str, Any]:
    receipts = sorted(RECEIPTS.glob("*.json"))
    missing: List[str] = []
    invalid: List[str] = []
    for path in receipts:
        sidecar = path.with_name(path.name + ".sha256")
        if not sidecar.exists():
            missing.append(path.name)
            continue
        if sidecar.read_text(encoding="utf-8").strip() != _sha(path):
            invalid.append(path.name)
    return {
        "receipts": int(len(receipts)),
        "missing_sidecars": missing,
        "invalid_sidecars": invalid,
        "parity": float((len(receipts) - len(missing) - len(invalid)) / max(len(receipts), 1)),
    }


def secret_scan() -> Dict[str, Any]:
    patterns = ("sk-or-v1", "sk-proj", "OPENROUTER_API_KEY=", "VAST_API_KEY=", "Bearer sk-")
    hits: List[Dict[str, str]] = []
    for path in sorted(RECEIPTS.glob("*.json")):
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in patterns:
            if pattern in text:
                hits.append({"file": path.name, "pattern": pattern})
    if DOCS.exists():
        text = DOCS.read_text(encoding="utf-8", errors="replace")
        for pattern in patterns:
            if pattern in text:
                hits.append({"file": DOCS.name, "pattern": pattern})
    return {"hits": hits, "zero_credentials": bool(not hits)}


def run_pytest(timeout: int = 2400) -> Dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    text = (completed.stdout or "") + "\n" + (completed.stderr or "")
    raw = RECEIPTS / "_work_ad" / "pytest_output.txt"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(text, encoding="utf-8")
    counts: Dict[str, int] = {}
    for match in re.finditer(r"(\d+)\s+(passed|failed|error|errors|skipped|xfailed|xpassed)", text):
        counts[match.group(2)] = int(match.group(1))
    if "passed" not in counts:
        markers = "".join(
            line.strip() for line in text.splitlines()
            if re.match(r"^[.sFExX]+\s*(\[|$)", line.strip())
        )
        if markers:
            counts["passed"] = markers.count(".")
            counts["failed"] = markers.count("F")
            counts["error"] = markers.count("E")
            counts["skipped"] = markers.count("s")
            counts["collected"] = sum(markers.count(char) for char in ".sFExX")
    tail = [line for line in text.strip().splitlines() if line.strip()][-12:]
    return {
        "exit_code": int(completed.returncode),
        "counts": counts,
        "wall_seconds": float(time.perf_counter() - started),
        "tail": tail,
    }


def run_consistency() -> Dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "verify_holarch_consistency.py")],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=900,
    )
    try:
        payload = json.loads(completed.stdout)
    except Exception:  # noqa: BLE001 - a non-JSON report is a failed gate
        payload = {"raw": (completed.stdout or "")[:2000]}
    payload["exit_code"] = int(completed.returncode)
    return payload


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Cycle AD master seal")
    parser.add_argument("--receipts", default=str(RECEIPTS))
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args(argv)
    directory = Path(args.receipts).resolve()
    started = time.perf_counter()
    stages = {name: _read(name) for name in STAGES}
    prior = {name: _read(name) for name in PRIOR}
    r3 = _read("r3_static_and_streaming_full_decile_language.json")
    decile = dict(r3.get("full_decile") or {})
    streaming_bpc = float(decile.get("bits_per_char", 9.9))
    static_bpc = float(decile.get("static_bits_per_char", 9.9))
    snapshot_sha = _sha(PIN_SNAPSHOT) if PIN_SNAPSHOT.exists() else ""
    snapshot_digest = (
        str(json.loads(PIN_SNAPSHOT.read_text(encoding="utf-8")).get("digest", ""))
        if PIN_SNAPSHOT.exists()
        else ""
    )
    parity = sidecar_parity()
    secrets = secret_scan()
    consistency = run_consistency()
    tests: Dict[str, Any] = {"exit_code": 1, "counts": {}, "wall_seconds": 0.0, "tail": ["skipped"]}
    if not args.skip_tests:
        tests = run_pytest()
    docs_numbers = [
        int(value) for value in re.findall(r"^## (\d+)\.", DOCS.read_text(encoding="utf-8"), flags=re.M)
    ]
    ad4 = stages.get("ad4_blackwell_cuda_and_jev.json", {})
    jev_ok = bool((ad4.get("jev") or {}).get("accepted"))
    gates = {
        "ad_stage_receipts_present": bool(all(bool(stages.get(name)) for name in STAGES)),
        "ad0_ac_cycle_passed": bool((prior.get("ac_cycle.json") or {}).get("passed")),
        "ad1_passed": bool((stages.get("ad1_standalone_repo_patch.json") or {}).get("passed")),
        "ad2_passed": bool((stages.get("ad2_relational_arc_and_science.json") or {}).get("passed")),
        "ad3_passed": bool((stages.get("ad3_deliverable_and_aaii96.json") or {}).get("passed")),
        "ad4_passed": bool((stages.get("ad4_blackwell_cuda_and_jev.json") or {}).get("passed")),
        "ad_all_stages_passed": bool(
            all(bool((stages.get(name) or {}).get("passed")) for name in STAGES)
        ),
        "ad_previous_cycles_passed": bool(
            all(bool((prior.get(name) or {}).get("passed")) for name in PRIOR)
        ),
        "ad_jev_two_challenges_accepted": bool(jev_ok),
        "ad_pinned_snapshot_digest_unchanged": bool(snapshot_digest == PIN_DIGEST),
        "ad_pinned_snapshot_sha_unchanged": bool(snapshot_sha == PIN_SHA),
        "ad_sidecar_parity_one": bool(float(parity["parity"]) == 1.0),
        "ad_zero_credentials": bool(secrets["zero_credentials"]),
        "ad_consistency_valid": bool(consistency.get("exit_code") == 0),
        "ad_docs_sections_1_to_45": bool(docs_numbers == list(range(1, 46))),
        "ad_r3_static_bpc_le_14200": bool(static_bpc <= 1.4200),
        "ad_r3_streaming_bpc_le_12200": bool(streaming_bpc <= 1.2200),
        "ad_pytest_exit_zero": bool(int(tests["exit_code"]) == 0),
        "ad_pytest_zero_failures": bool(int(tests["counts"].get("failed", 0)) == 0),
        "ad_pytest_zero_errors": bool(
            int(tests["counts"].get("error", 0)) + int(tests["counts"].get("errors", 0)) == 0
        ),
    }
    receipt = {
        "stage": "AD",
        "cycle": "AD",
        "headline": {
            "ad0_ac_seal": bool((prior.get("ac_cycle.json") or {}).get("passed")),
            "ad1_repo_solved": int(
                ((stages.get("ad1_standalone_repo_patch.json") or {}).get("aggregate") or {}).get(
                    "hidden_solved", 0
                )
            ),
            "ad2_external_arc": int(
                ((stages.get("ad2_relational_arc_and_science.json") or {}).get("relational_arc") or {}).get(
                    "solved", 0
                )
            ),
            "ad2_science": int(
                ((stages.get("ad2_relational_arc_and_science.json") or {}).get("science_solver") or {}).get(
                    "answerable_solved", 0
                )
            ),
            "ad3_status": str((stages.get("ad3_deliverable_and_aaii96.json") or {}).get("status", "")),
            "ad4_jev_scores": list((ad4.get("jev") or {}).get("scores") or []),
            "ad4_cuda_status": str(((ad4.get("hardware") or {}).get("cuda_status") or "")),
        },
        "stages": {
            name: {
                "passed": bool(stages[name].get("passed", False)),
                "gate": dict(stages[name].get("gate") or {}),
                "wall_seconds": float(stages[name].get("wall_seconds") or 0.0),
            }
            for name in STAGES
        },
        "prior_cycles": {
            name: bool((prior[name] or {}).get("passed")) for name in PRIOR
        },
        "pytest": tests,
        "consistency": consistency,
        "sidecar_parity": parity,
        "secret_scan": secrets,
        "language_hygiene": {
            "r3_static_bits_per_char": static_bpc,
            "r3_streaming_bits_per_char": streaming_bpc,
            "static_bar": 1.4200,
            "streaming_bar": 1.2200,
        },
        "docs": {"sections": docs_numbers, "sections_1_to_45": bool(docs_numbers == list(range(1, 46)))},
        "pinned": {"snapshot_sha256": snapshot_sha, "digest": snapshot_digest},
        "gate": pc._finalise(gates),
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
        "passed": bool(all(gates.values())),
    }
    written = pc._write(directory, "ad_cycle.json", receipt)
    print(
        "[ad] AD0=%s AD1=%s AD2=%s AD3=%s AD4=%s pytest=%s"
        % (
            receipt["headline"]["ad0_ac_seal"],
            receipt["stages"]["ad1_standalone_repo_patch.json"]["passed"],
            receipt["stages"]["ad2_relational_arc_and_science.json"]["passed"],
            receipt["stages"]["ad3_deliverable_and_aaii96.json"]["passed"],
            receipt["stages"]["ad4_blackwell_cuda_and_jev.json"]["passed"],
            tests["counts"],
        )
    )
    failed = [name for name, value in receipt["gate"].items() if not bool(value)]
    print("[ad] passed=%s failed=%s receipt=%s" % (receipt["passed"], failed, written["path"]))
    return 0 if bool(receipt["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
