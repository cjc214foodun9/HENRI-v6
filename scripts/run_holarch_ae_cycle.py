"""Cycle AE master seal: hygiene battery and the Cycle AE ledger.

    python scripts/run_holarch_ae_cycle.py
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
    "ae0_ac1_subslice_asymmetry.json",
    "ae1_blackwell_cuda_and_d4096.json",
    "ae2_external_repo_patch_and_science.json",
    "ae3_compositional_external_arc.json",
    "ae4_api_and_10k_atlas.json",
    "ae_jev_mathematical_challenges.json",
)
PRIOR = ("t_cycle.json", "u_cycle.json", "v_cycle.json", "w_cycle.json",
         "x_cycle.json", "ac_cycle.json", "ad_cycle.json")


def _read(name: str) -> Dict[str, Any]:
    path = RECEIPTS / name
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
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
    raw = RECEIPTS / "_work_ae" / "pytest_output.txt"
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
    except Exception:  # noqa: BLE001
        payload = {"raw": (completed.stdout or "")[:2000]}
    payload["exit_code"] = int(completed.returncode)
    return payload


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Cycle AE master seal")
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
    jev = stages.get("ae_jev_mathematical_challenges.json", {})
    jev_ok = bool(jev.get("accepted"))
    gates = {
        "ae_stage_receipts_present": bool(all(bool(stages.get(name)) for name in STAGES)),
        "ae0_ac1_diagnostic_passed": bool((stages.get("ae0_ac1_subslice_asymmetry.json") or {}).get("passed")),
        "ae0_ad3_closed_and_ad_cycle_passed": bool((prior.get("ad_cycle.json") or {}).get("passed")),
        "ae1_passed": bool((stages.get("ae1_blackwell_cuda_and_d4096.json") or {}).get("passed")),
        "ae2_passed": bool((stages.get("ae2_external_repo_patch_and_science.json") or {}).get("passed")),
        "ae3_passed": bool((stages.get("ae3_compositional_external_arc.json") or {}).get("passed")),
        "ae4_passed": bool((stages.get("ae4_api_and_10k_atlas.json") or {}).get("passed")),
        "ae_all_stages_passed": bool(all(bool((stages.get(name) or {}).get("passed")) for name in STAGES)),
        "ae_jev_two_challenges_accepted": bool(jev_ok),
        "ae_previous_cycles_passed": bool(
            all(bool((prior.get(name) or {}).get("passed")) for name in PRIOR)
        ),
        "ae_pinned_snapshot_digest_unchanged": bool(snapshot_digest == PIN_DIGEST),
        "ae_pinned_snapshot_sha_unchanged": bool(snapshot_sha == PIN_SHA),
        "ae_sidecar_parity_one": bool(float(parity["parity"]) == 1.0),
        "ae_zero_credentials": bool(secrets["zero_credentials"]),
        "ae_consistency_valid": bool(consistency.get("exit_code") == 0),
        "ae_docs_sections_1_to_46": bool(docs_numbers == list(range(1, 47))),
        "ae_r3_static_bpc_le_14200": bool(static_bpc <= 1.4200),
        "ae_r3_streaming_bpc_le_12200": bool(streaming_bpc <= 1.2200),
        "ae_pytest_exit_zero": bool(int(tests["exit_code"]) == 0),
        "ae_pytest_zero_failures": bool(int(tests["counts"].get("failed", 0)) == 0),
        "ae_pytest_zero_errors": bool(
            int(tests["counts"].get("error", 0)) + int(tests["counts"].get("errors", 0)) == 0
        ),
    }
    receipt = {
        "stage": "AE",
        "cycle": "AE",
        "headline": {
            "ae0_verdict": str(((stages.get("ae0_ac1_subslice_asymmetry.json") or {}).get("verdict") or {}).get("decision", "")),
            "ae0_ad_cycle_passed": bool((prior.get("ad_cycle.json") or {}).get("passed")),
            "ae1_offers": int((((stages.get("ae1_blackwell_cuda_and_d4096.json") or {}).get("offers") or {}).get("count", 0))),
            "ae1_rented": bool(((stages.get("ae1_blackwell_cuda_and_d4096.json") or {}).get("hardware") or {}).get("rented", False)),
            "ae2_status": str((stages.get("ae2_external_repo_patch_and_science.json") or {}).get("status", "")),
            "ae3_status": str((stages.get("ae3_compositional_external_arc.json") or {}).get("status", "")),
            "ae4_status": str((stages.get("ae4_api_and_10k_atlas.json") or {}).get("api") and "ATLAS_MEASURED_API_NOT_RUN" or ""),
            "ae4_parity": float(((stages.get("ae4_api_and_10k_atlas.json") or {}).get("atlas") or {}).get("parity", 0.0)),
            "ae_jev_scores": list(jev.get("scores") or []),
        },
        "stages": {
            name: {
                "passed": bool(stages[name].get("passed", False)),
                "gate": dict(stages[name].get("gate") or {}),
                "wall_seconds": float(stages[name].get("wall_seconds") or 0.0),
            }
            for name in STAGES
        },
        "prior_cycles": {name: bool((prior[name] or {}).get("passed")) for name in PRIOR},
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
        "docs": {"sections": docs_numbers, "sections_1_to_46": bool(docs_numbers == list(range(1, 47)))},
        "pinned": {"snapshot_sha256": snapshot_sha, "digest": snapshot_digest},
        "gate": pc._finalise(gates),
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
        "passed": bool(all(gates.values())),
    }
    written = pc._write(directory, "ae_cycle.json", receipt)
    print(
        "[ae] verdict=%s pytest=%s parity=%.4f"
        % (receipt["headline"]["ae0_verdict"], tests["counts"], parity["parity"])
    )
    failed = [name for name, value in receipt["gate"].items() if not bool(value)]
    print("[ae] passed=%s failed=%s receipt=%s" % (receipt["passed"], failed, written["path"]))
    return 0 if bool(receipt["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
