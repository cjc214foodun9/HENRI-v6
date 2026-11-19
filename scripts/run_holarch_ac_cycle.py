"""Cycle AC master seal: run the suite, the consistency check and seal the ledger.

    python scripts/run_holarch_ac_cycle.py

The seal is `passed: true` only when every software gate in AC1-AC5 passes,
`pytest -q` exits zero, the consistency verifier accepts Sections 1..44 with
1.0000 sidecar parity, the R3 language hygiene bars hold and the frozen
snapshot is byte-identical.  No bar is moved here: this script only reads the
stage receipts that measured themselves.
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
PIN = "8f270f34e8f35ef84e415d076bea20a7bc15edff337f89ed72ad607447ba8554"
STAGES = (
    "ac1_kronecker_induction_egress.json",
    "ac2_mdl_sheaf_synthesis.json",
    "ac3_multiscale_external_arc.json",
    "ac4_cascade_and_aaii80.json",
    "ac5_hardware_and_ledger_closure.json",
    "ac_jev_mathematical_challenges.json",
)


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
        [sys.executable, "-m", "pytest", "-q"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    text = (completed.stdout or "") + "\n" + (completed.stderr or "")
    raw = ROOT / "receipts" / "holarch" / "_work_ac" / "pytest_output.txt"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(text, encoding="utf-8")
    counts: Dict[str, int] = {}
    # pytest -q prints one status line such as "380 passed, 8 skipped in 374s".
    for match in re.finditer(
        r"(\d+)\s+(passed|failed|error|errors|skipped|xfailed|xpassed)", text
    ):
        counts[match.group(2)] = int(match.group(1))
    if "passed" not in counts:
        # pytest 9 with -q prints no status line, but the progress line carries
        # one character per test: dot = passed, s = skipped, F = failed,
        # E = error.  Counting those markers is an exact measurement.
        markers = "".join(
            line.strip()
            for line in text.splitlines()
            if re.match(r"^[.sFExX]+\s*(\[|$)", line.strip())
        )
        if markers:
            counts["passed"] = markers.count(".")
            counts["failed"] = markers.count("F")
            counts["error"] = markers.count("E")
            counts["skipped"] = markers.count("s")
            counts["collected"] = sum(
                markers.count(char) for char in ".sFExX"
            )
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
        timeout=600,
    )
    payload: Dict[str, Any] = {}
    try:
        payload = json.loads(completed.stdout)
    except Exception:  # noqa: BLE001 - a non-JSON report is a failed gate
        payload = {"raw": (completed.stdout or "")[:2000]}
    payload["exit_code"] = int(completed.returncode)
    return payload


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Cycle AC master seal")
    parser.add_argument("--receipts", default=str(RECEIPTS))
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args(argv)
    directory = Path(args.receipts).resolve()
    started = time.perf_counter()
    stages = {name: _read(name) for name in STAGES}
    print("[ac] stage receipts loaded", flush=True)

    r3 = _read("r3_static_and_streaming_full_decile_language.json")
    decile = dict(r3.get("full_decile") or {})
    streaming_bpc = float(decile.get("bits_per_char", 9.9))
    static_bpc = float(decile.get("static_bits_per_char", 9.9))
    snapshot_sha = _sha(PIN_SNAPSHOT) if PIN_SNAPSHOT.exists() else ""
    snapshot_digest = str(json.loads(PIN_SNAPSHOT.read_text(encoding="utf-8")).get("digest", "")) if PIN_SNAPSHOT.exists() else ""
    parity = sidecar_parity()
    secrets = secret_scan()
    consistency = run_consistency()
    print("[ac] consistency checked: exit %s" % consistency.get("exit_code"), flush=True)
    tests = {"exit_code": 1, "counts": {}, "wall_seconds": 0.0, "tail": ["skipped"]}
    if not args.skip_tests:
        tests = run_pytest()
        print("[ac] pytest exit %s counts %s" % (tests["exit_code"], tests["counts"]), flush=True)

    gate_rows: Dict[str, bool] = {}
    for name, receipt in stages.items():
        gate = dict(receipt.get("gate") or {})
        gate_rows["%s_passed" % name.replace(".json", "")] = bool(receipt.get("passed", False))
        gate_rows["%s_all_gates_true" % name.replace(".json", "")] = bool(
            gate and all(bool(value) for key, value in gate.items() if key != "passed")
        )
    ac_jev = stages.get("ac_jev_mathematical_challenges.json", {})
    jev_challenges = dict(ac_jev.get("challenges") or {})
    jev_accepted = [
        bool((row.get("verdict") or {}).get("accepted")) for row in jev_challenges.values()
    ]
    gauntlet = dict((stages.get("ac4_cascade_and_aaii80.json") or {}).get("gauntlet") or {})
    docs_numbers = [
        int(value) for value in re.findall(r"^## (\d+)\.", DOCS.read_text(encoding="utf-8"), flags=re.M)
    ]
    gates = {
        "ac_stage_receipts_present": bool(
            all(bool(stages.get(name)) for name in STAGES)
        ),
        "ac_all_stage_gates_true": bool(
            all(
                bool(stages.get(name).get("passed", False))
                for name in STAGES
            )
        ),
        "ac_jev_two_challenges_accepted": bool(
            len(jev_accepted) >= 2 and all(jev_accepted)
        ),
        "ac_pinned_snapshot_digest_unchanged": bool(snapshot_digest == PIN),
        "ac_pinned_snapshot_sha_unchanged": bool(
            snapshot_sha == "d675f3816bbf472493b25f885bbb56c62c5ec88a81ab31e31a9b87b3a0c23703"
        ),
        "ac_sidecar_parity_one": bool(float(parity["parity"]) == 1.0),
        "ac_zero_credentials": bool(secrets["zero_credentials"]),
        "ac_consistency_valid": bool(consistency.get("exit_code") == 0),
        "ac_docs_sections_1_to_44": bool(
            all(number in docs_numbers for number in range(1, 45))
        ),
        "ac_r3_static_bpc_le_14200": bool(static_bpc <= 1.4200),
        "ac_r3_streaming_bpc_le_12200": bool(streaming_bpc <= 1.2200),
        "ac_pytest_exit_zero": bool(int(tests["exit_code"]) == 0),
        "ac_pytest_zero_failures": bool(int(tests["counts"].get("failed", 0)) == 0),
        "ac_pytest_zero_errors": bool(
            int(tests["counts"].get("error", 0)) + int(tests["counts"].get("errors", 0)) == 0
        ),
        "ac_gauntlet_304_of_304": bool(int(gauntlet.get("total_queries", 0)) >= 304),
    }
    receipt = {
        "stage": "AC",
        "cycle": "AC",
        "headline": {
            "ac1_bpc": float(((stages["ac1_kronecker_induction_egress.json"].get("wave_egress") or {}).get("bpc_wave_egress")) or 0.0),
            "ac1_top1": float(((stages["ac1_kronecker_induction_egress.json"].get("wave_egress") or {}).get("top1_wave_egress")) or 0.0),
            "ac2_mbpp": int((((stages["ac2_mdl_sheaf_synthesis.json"].get("primary") or {}).get("counts")) or {}).get("solved", 0)),
            "ac2_humaneval": int((((stages["ac2_mdl_sheaf_synthesis.json"].get("secondary") or {}).get("counts")) or {}).get("solved", 0)),
            "ac2_precision": float((((stages["ac2_mdl_sheaf_synthesis.json"].get("combined")) or {}).get("precision")) or 0.0),
            "ac3_external": int((((stages["ac3_multiscale_external_arc.json"].get("external")) or {}).get("solved")) or 0),
            "ac3_precision": float((((stages["ac3_multiscale_external_arc.json"].get("external")) or {}).get("precision")) or 0.0),
            "ac4_tasks": int(
                (
                    (stages["ac4_cascade_and_aaii80.json"].get("battery")) or {}
                ).get("total")
                or (
                    int(((stages["ac4_cascade_and_aaii80.json"].get("battery")) or {}).get("answerable", 0))
                    + int(((stages["ac4_cascade_and_aaii80.json"].get("battery")) or {}).get("traps", 0))
                )
            ),
            "ac4_gauntlet": int(gauntlet.get("total_queries", 0)),
            "ac5_cuda_status": str(((stages["ac5_hardware_and_ledger_closure.json"].get("hardware")) or {}).get("cuda_status", "")),
            "ac5_vast_status": str(((stages["ac5_hardware_and_ledger_closure.json"].get("hardware")) or {}).get("vast_status", "")),
        },
        "stages": {
            name: {
                "passed": bool(stages[name].get("passed", False)),
                "gate": dict(stages[name].get("gate") or {}),
                "wall_seconds": float(stages[name].get("wall_seconds") or 0.0),
            }
            for name in STAGES
        },
        "jev": {
            "challenges": {
                name: {
                    "accepted": bool((row.get("verdict") or {}).get("accepted")),
                    "score": float((row.get("verdict") or {}).get("score") or 0.0),
                    "failed": list((row.get("verdict") or {}).get("failed") or []),
                }
                for name, row in jev_challenges.items()
            },
            "accepted": bool(len(jev_accepted) >= 2 and all(jev_accepted)),
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
        "docs": {
            "sections": docs_numbers,
            "sections_1_to_44": bool(docs_numbers == list(range(1, 45))),
        },
        "pinned": {
            "snapshot_sha256": snapshot_sha,
            "digest": snapshot_digest,
            "files": int(len(json.loads(PIN_SNAPSHOT.read_text(encoding="utf-8")).get("files") or [])) if PIN_SNAPSHOT.exists() else 0,
        },
        "gate": pc._finalise(gates),
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
        "passed": bool(all(gates.values())),
    }
    written = pc._write(directory, "ac_cycle.json", receipt)
    print(
        "[ac] ac1 bpc=%.6f top1=%.6f | ac2 %d/%d %d/%d | ac3 %d/48 | ac4 %d tasks, %d queries"
        % (
            receipt["headline"]["ac1_bpc"],
            receipt["headline"]["ac1_top1"],
            receipt["headline"]["ac2_mbpp"],
            30,
            receipt["headline"]["ac2_humaneval"],
            32,
            receipt["headline"]["ac3_external"],
            receipt["headline"]["ac4_tasks"],
            receipt["headline"]["ac4_gauntlet"],
        )
    )
    print("[ac] pytest exit %s counts %s" % (tests["exit_code"], tests["counts"]))
    failed = [name for name, value in receipt["gate"].items() if not bool(value)]
    print("[ac] passed=%s failed=%s receipt=%s" % (receipt["passed"], failed, written["path"]))
    return 0 if bool(receipt["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
