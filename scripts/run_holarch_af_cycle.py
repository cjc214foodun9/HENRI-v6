"""Cycle AF master seal: full pytest, consistency check, sidecar parity, R3
language hygiene, credential scan, and receipts/holarch/af_cycle.json ledger.

    .venv/bin/python scripts/run_holarch_af_cycle.py
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

AF_STAGES = (
    "af0_gateway_seal.json",
    "af1_fast_riemannian_atlas.json",
    "af2_ad3_and_gauntlet_336.json",
    "af3_external_audits_and_compositional_arc.json",
    "af4_blackwell_and_jev.json",
    "af_jev_mathematical_challenges.json",
)
CLOSED_AE_RECEIPTS = (
    "ae2_external_repo_patch_and_science.json",
    "ae3_compositional_external_arc.json",
    "ae4_api_and_10k_atlas.json",
    "ad3_deliverable_and_aaii96.json",
)


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
    exact_matches = 0
    crlf_equiv_matches = 0
    for path in receipts:
        sidecar = path.with_name(path.name + ".sha256")
        if not sidecar.exists():
            missing.append(path.name)
            continue
        expected = sidecar.read_text(encoding="utf-8").strip()
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() == expected:
            exact_matches += 1
        elif b"\r\n" not in raw and hashlib.sha256(raw.replace(b"\n", b"\r\n")).hexdigest() == expected:
            crlf_equiv_matches += 1
        elif b"\r\n" in raw and hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() == expected:
            crlf_equiv_matches += 1
        else:
            invalid.append(path.name)
    return {
        "receipts": int(len(receipts)),
        "exact_matches": int(exact_matches),
        "crlf_equiv_matches": int(crlf_equiv_matches),
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
                hits.append({"file": str(path.relative_to(ROOT)), "pattern": pattern})
    if DOCS.exists():
        text = DOCS.read_text(encoding="utf-8", errors="replace")
        for pattern in patterns:
            if pattern in text:
                hits.append({"file": str(DOCS.relative_to(ROOT)), "pattern": pattern})
    for path in sorted((ROOT / "henri").rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in patterns:
            if pattern in text:
                # Ignore explicit redactor/pattern test constants in coprocessor.py
                if path.name == "coprocessor.py":
                    continue
                hits.append({"file": str(path.relative_to(ROOT)), "pattern": pattern})
    return {"hits": hits, "zero_credentials": bool(not hits)}


def run_pytest(timeout: int = 2400) -> Dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-o", "addopts=", "-q", "-rs", "-p", "no:cacheprovider"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    text = (completed.stdout or "") + "\n" + (completed.stderr or "")
    raw = RECEIPTS / "_work_af" / "pytest_output.txt"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(text, encoding="utf-8")
    lines = [line for line in text.strip().splitlines() if line.strip()]
    summary_line = lines[-1] if lines else ""
    counts: Dict[str, int] = {}
    for match in re.finditer(r"(\d+)\s+(passed|failed|error|errors|skipped|xfailed|xpassed)", summary_line):
        key = "error" if match.group(2) == "errors" else match.group(2)
        counts[key] = int(match.group(1))
    # Cross-check explicit FAILED / ERROR lines so counts can never silently read 0.
    explicit_failed = sum(1 for line in lines if line.startswith("FAILED "))
    explicit_error = sum(1 for line in lines if line.startswith("ERROR "))
    counts["failed"] = max(int(counts.get("failed", 0)), explicit_failed)
    counts["error"] = max(int(counts.get("error", 0)), explicit_error)
    skip_reasons: Dict[str, int] = {}
    for line in lines:
        if line.startswith("SKIPPED "):
            match = re.match(r"SKIPPED \[(\d+)\]\s+[^:]+:\s*(.+)$", line)
            if match:
                n_skip = int(match.group(1))
                reason = match.group(2).strip()
                skip_reasons[reason] = skip_reasons.get(reason, 0) + n_skip
    tail = lines[-15:]
    return {
        "exit_code": int(completed.returncode),
        "counts": counts,
        "skip_reasons": skip_reasons,
        "summary_line": summary_line,
        "wall_seconds": float(time.perf_counter() - started),
        "tail": tail,
    }


def run_consistency() -> Dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "verify_holarch_consistency.py")],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    try:
        payload = json.loads(completed.stdout)
    except Exception:  # noqa: BLE001
        payload = {"raw": (completed.stdout or "")[:2000]}
    payload["exit_code"] = int(completed.returncode)
    return payload


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Cycle AF master seal")
    parser.add_argument("--receipts", default=str(RECEIPTS))
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args(argv)
    directory = Path(args.receipts).resolve()
    started = time.perf_counter()

    stages = {name: _read(name) for name in AF_STAGES}
    ae_closed = {name: _read(name) for name in CLOSED_AE_RECEIPTS}

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

    tests: Dict[str, Any] = {"exit_code": 1, "counts": {}, "wall_seconds": 0.0, "tail": ["skipped"]}
    if not args.skip_tests:
        tests = run_pytest()

    docs_numbers = [
        int(value)
        for value in re.findall(r"^## (\d+)\.", DOCS.read_text(encoding="utf-8"), flags=re.M)
    ]

    # Write preliminary af_cycle.json so consistency + sidecar_parity include af_cycle.json itself
    def _build_receipt(consistency: Dict[str, Any], parity: Dict[str, Any], secrets: Dict[str, Any]) -> Dict[str, Any]:
        af0 = stages["af0_gateway_seal.json"]
        af1 = stages["af1_fast_riemannian_atlas.json"]
        af2 = stages["af2_ad3_and_gauntlet_336.json"]
        af3 = stages["af3_external_audits_and_compositional_arc.json"]
        af4 = stages["af4_blackwell_and_jev.json"]
        af_jev = stages["af_jev_mathematical_challenges.json"]

        ae2 = ae_closed["ae2_external_repo_patch_and_science.json"]
        ae3 = ae_closed["ae3_compositional_external_arc.json"]
        ae4 = ae_closed["ae4_api_and_10k_atlas.json"]
        ad3 = ae_closed["ad3_deliverable_and_aaii96.json"]

        gates = {
            "af_stage_receipts_present": bool(all(bool(stages.get(name)) for name in AF_STAGES)),
            "af0_gateway_sealed_passed": bool(af0.get("passed")),
            "af1_fast_atlas_passed": bool(af1.get("passed")),
            "af2_local_observed_gates_passed": bool(af2.get("local_observed_gates_passed")),
            "af2_ad3_live_llm_arms_closed": bool(af2.get("passed") and ad3.get("passed")),
            "af3_external_audits_and_arc_passed": bool(af3.get("passed")),
            "ae2_closed_passed": bool(ae2.get("passed")),
            "ae3_closed_passed": bool(ae3.get("passed")),
            "ae4_closed_passed": bool(ae4.get("passed")),
            "af4_policy_and_proposals_passed": bool(af4.get("local_policy_and_proposal_gates_passed")),
            "af4_jev_plugin_accepted": bool(af_jev.get("accepted")),
            "af_pinned_snapshot_digest_unchanged": bool(snapshot_digest == PIN_DIGEST),
            "af_pinned_snapshot_sha_unchanged": bool(snapshot_sha == PIN_SHA),
            "af_sidecar_parity_one": bool(float(parity["parity"]) == 1.0),
            "af_zero_credentials": bool(secrets["zero_credentials"]),
            "af_consistency_valid": bool(consistency.get("exit_code") == 0 and consistency.get("valid")),
            "af_docs_sections_1_to_47": bool(docs_numbers == list(range(1, 48))),
            "af_r3_static_bpc_le_14200": bool(static_bpc <= 1.4200),
            "af_r3_streaming_bpc_le_12200": bool(streaming_bpc <= 1.2200),
            "af_pytest_exit_zero": bool(int(tests["exit_code"]) == 0),
            "af_pytest_zero_failures": bool(int(tests["counts"].get("failed", 0)) == 0),
            "af_pytest_zero_errors": bool(
                int(tests["counts"].get("error", 0)) + int(tests["counts"].get("errors", 0)) == 0
            ),
        }
        blocked_external_keys = {"af2_ad3_live_llm_arms_closed", "af4_jev_plugin_accepted"}
        local_observed_all_passed = bool(
            all(v for k, v in gates.items() if k not in blocked_external_keys)
        )
        return {
            "stage": "AF",
            "cycle": "AF",
            "spec": "HENRI-SPEC-CYCLE-AF-GATEWAY-SEAL-AD3-CLOSURE-FAST-ATLAS-AND-EXTERNAL-DECOUPLING-V1",
            "evidence_label": "OBSERVED",
            "compute_device": "cpu",
            "headline": {
                "af0_gateway_passed": bool(af0.get("passed")),
                "af1_speedup_10k": float((af1.get("atlas") or {}).get("speedup_10k_vs_flat_complex", 0.0)),
                "af1_speedup_12k": float((af1.get("atlas") or {}).get("speedup_12k_vs_flat_complex", 0.0)),
                "af1_online_recall": int((af1.get("atlas") or {}).get("online_recall", 0)),
                "af2_arm0_answerable": int((((af2.get("aaii96_battery") or {}).get("arm0_henri_standalone") or {}).get("answerable_solved", 0))),
                "af2_arm0_traps_refused": int((((af2.get("aaii96_battery") or {}).get("arm0_henri_standalone") or {}).get("traps_refused", 0))),
                "af2_stress_requests": int((af2.get("gateway_stress_1000") or {}).get("requests_total", 0)),
                "af2_gauntlet_correct": int((af2.get("gauntlet_336") or {}).get("queries_correct", 0)),
                "af2_live_llm_status": str(((af2.get("aaii96_battery") or {}).get("live_llm_arms") or {}).get("status", "")),
                "ae2_passed": bool(ae2.get("passed")),
                "ae2_repo_pass": f"{int((ae2.get('executed') or {}).get('repositories_passed_hidden', 0))}/{int((ae2.get('executed') or {}).get('repositories', 0))}",
                "ae2_science_pass": f"{int((ae2.get('executed') or {}).get('science_solved', 0))}/{int((ae2.get('executed') or {}).get('science_problems', 0))}",
                "ae3_passed": bool(ae3.get("passed")),
                "ae3_arc_pass": f"{int((ae3.get('executed') or {}).get('external_solved', 0))}/{int((ae3.get('executed') or {}).get('external_total', 0))}",
                "ae4_passed": bool(ae4.get("passed")),
                "af4_cuda_status": str(af4.get("cuda_status", "")),
                "af4_jev_status": str(af_jev.get("status", "")),
                "local_observed_all_passed": local_observed_all_passed,
            },
            "stages": {
                name: {
                    "passed": bool(stages[name].get("passed", False)),
                    "gate": dict(stages[name].get("gate") or {}),
                    "wall_seconds": float(stages[name].get("wall_seconds") or 0.0),
                }
                for name in AF_STAGES
            },
            "closed_prior_receipts": {
                name: {
                    "passed": bool(ae_closed[name].get("passed", False)),
                    "status": str(ae_closed[name].get("status", "")),
                }
                for name in CLOSED_AE_RECEIPTS
            },
            "blocked_external_boundaries": {
                "live_llm_arms_ad3": "BLOCKED_MISSING_LLM_CREDENTIALS",
                "blackwell_sm120_vast": "CUDA: BLOCKED_BY_POLICY",
                "henri_jev_plugin": "BLOCKED_MISSING_RND_AGENT_PLUGIN",
                "gitignored_mmlu_csv": "BLOCKED_MISSING_GITIGNORED_MMLU_CSV",
                "sentence_transformers_offline_model": "BLOCKED_MISSING_SENTENCE_TRANSFORMERS_MODEL",
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
                "sections_1_to_47": bool(docs_numbers == list(range(1, 48))),
            },
            "pinned": {"snapshot_sha256": snapshot_sha, "digest": snapshot_digest},
            "gate": pc._finalise(gates),
            "local_observed_all_passed": local_observed_all_passed,
            "wall_seconds": float(time.perf_counter() - started),
            "evidence": "OBSERVED",
            "passed": bool(all(gates.values())),
        }

    # First write establishes af_cycle.json + .sha256 on disk; then re-run parity & consistency
    # so the sealed af_cycle.json records the exact final receipt count including itself.
    pc._write(
        directory,
        "af_cycle.json",
        _build_receipt({"exit_code": 0, "valid": True}, {"parity": 1.0, "receipts": 0}, {"hits": [], "zero_credentials": True}),
    )
    parity = sidecar_parity()
    secrets = secret_scan()
    consistency = run_consistency()
    final_receipt = _build_receipt(consistency, parity, secrets)
    written = pc._write(directory, "af_cycle.json", final_receipt)

    # Re-verify consistency after final write so sidecar parity is guaranteed 1.0000
    post_check = run_consistency()
    failed = [k for k, v in final_receipt["gate"].items() if k != "passed" and not bool(v)]
    print(
        "[af] local_observed_all_passed=%s passed=%s failed=%s pytest=%s consistency_valid=%s receipts=%s -> %s"
        % (
            final_receipt["local_observed_all_passed"],
            final_receipt["passed"],
            failed,
            tests["counts"],
            post_check.get("valid"),
            post_check.get("receipts"),
            written["path"],
        ),
        flush=True,
    )
    return 0 if (final_receipt["local_observed_all_passed"] and post_check.get("valid")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
