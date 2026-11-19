"""Cycle AG0: Plugin-Host Closure of Pending AF4 JEV Challenges, Live 5-Arm
AAII-96 Cascade Probe, and Side-by-Side Neural vs. Gateway Gauntlet Record.

    .venv/bin/python scripts/run_holarch_ag0_host_closure.py

Strict scientific honesty:
* Invokes mounted @RND AGENT ``henri_jev`` plugin on the two sealed AF4 formal
  challenges whenever the plugin executable is present; when absent, stamps
  ``BLOCKED_MISSING_RND_AGENT_PLUGIN`` with zero fabricated scores and leaves
  ``ag0_jev_af4_2_of_2_accepted = false``.
* Executes AAII-96 Arm0 standalone LIVE in this process (80 answerable + 16
  traps, 0 external calls) and checks ``resolve_llm_credentials()`` under
  SpendGuard for arms 1..4; when credentials are absent, stamps
  ``BLOCKED_MISSING_LLM_CREDENTIALS`` with 0 fabricated calls and leaves
  ``ag0_arm4_live_beats_arm3 = false`` and ``ad3_arm4_beats_arm3_composite = false``.
* Checks ``data/official_benchmarks/canonical/mmlu/mmlu.csv`` for re-running
  legacy neural ``scripts/run_holarch_agent_eval.py``; when absent, stamps
  ``BLOCKED_MISSING_GITIGNORED_MMLU_CSV`` side by side with the freshly
  observed 336-query clean-room gateway gauntlet with explicit provenance labels.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_holarch_af2_ad3_and_gauntlet as af2
import run_holarch_p_cycle as pc
from henri.holarch.coprocessor import resolve_llm_credentials
from henri.holarch.spend_guard import SpendGuard

RECEIPTS = ROOT / "receipts" / "holarch"
PIN_SNAPSHOT = RECEIPTS / "pinned_corpus_snapshot.json"
PIN_SHA = "d675f3816bbf472493b25f885bbb56c62c5ec88a81ab31e31a9b87b3a0c23703"
PIN_DIGEST = "8f270f34e8f35ef84e415d076bea20a7bc15edff337f89ed72ad607447ba8554"
SPEC_ID = (
    "HENRI-SPEC-CYCLE-AG-BLIND-HOLDOUT-RAW-PROSE-SCIENCE-"
    "LIVE-CASCADE-100K-ATLAS-AND-BLACKWELL-SCALE-V1"
)
AF4_CHALLENGE_IDS = (
    "AF1-PACKED-RIEMANNIAN-CONTIGUOUS-BLAS-SUBLINEAR-BOUND",
    "AF3-DEPTH3-COMPOSITIONAL-LOO-FALSE-CONSENSUS-BOUND",
)
SECRET_PATTERNS = (
    "sk-or-v1",
    "sk-proj",
    "OPENROUTER_API_KEY=",
    "VAST_API_KEY=",
    "Bearer sk-",
)


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _secret_scan(payload: Any) -> List[str]:
    blob = json.dumps(payload, sort_keys=True)
    return [pat for pat in SECRET_PATTERNS if pat in blob]


def _probe_rnd_agent_jev() -> Dict[str, Any]:
    """Check whether the external @RND AGENT henri_jev plugin binary is mounted."""
    candidates = ("henri_jev", "rnd-agent-jev", "cordis-jev")
    found = {name: shutil.which(name) for name in candidates}
    mounted_bin = next((path for path in found.values() if path), None)
    return {
        "plugin_present": bool(mounted_bin),
        "mounted_binary": mounted_bin,
        "searched_binaries": candidates,
        "note": (
            "Local henri.cli cmd_jev only verifies local ontology/sidecar integrity "
            "(EVIDENCE HOLDS); external @RND AGENT henri_jev Tier-1 mathematical "
            "proposal scorer is absent on this clean-room VM."
            if not mounted_bin
            else "Mounted external henri_jev binary detected."
        ),
    }


def _reconcile_af4_jev(probe: Dict[str, Any]) -> Dict[str, Any]:
    """Submit AF4 challenges if henri_jev is mounted; else preserve honest block."""
    jev_path = RECEIPTS / "af_jev_mathematical_challenges.json"
    existing = json.loads(jev_path.read_text(encoding="utf-8"))
    challenges = dict(existing.get("challenges") or {})
    present_ids = [cid for cid in AF4_CHALLENGE_IDS if cid in challenges]
    proposals_intact = len(present_ids) == len(AF4_CHALLENGE_IDS)
    for cid in present_ids:
        prop = challenges[cid].get("proposal") or {}
        if not prop.get("equations") or not prop.get("energy_series"):
            proposals_intact = False

    accepted_count = 0
    scores: Dict[str, Any] = {}
    failed_primitives: Dict[str, Any] = {}
    if probe["plugin_present"] and proposals_intact:
        # Execute mounted plugin cleanly without fabricating anything.
        for cid in AF4_CHALLENGE_IDS:
            proc = subprocess.run(
                [str(probe["mounted_binary"]), "--proposal-id", cid],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=120,
            )
            if proc.returncode == 0:
                try:
                    verdict = json.loads(proc.stdout)
                    sc = float(verdict.get("score", 0.0))
                    fp = int(verdict.get("failed_primitives", 1))
                    scores[cid] = sc
                    failed_primitives[cid] = fp
                    challenges[cid]["verdict"] = verdict
                    if sc >= 0.94 and fp == 0:
                        accepted_count += 1
                except Exception as exc:
                    challenges[cid]["verdict"] = {
                        "status": "PLUGIN_PARSE_ERROR",
                        "error": str(exc),
                    }
        status = (
            "ACCEPTED_OBSERVED"
            if accepted_count == len(AF4_CHALLENGE_IDS)
            else "PARTIAL_OR_REJECTED_OBSERVED"
        )
        evidence_label = "OBSERVED"
    else:
        status = "BLOCKED_MISSING_RND_AGENT_PLUGIN"
        evidence_label = "BLOCKED"
        for cid in present_ids:
            challenges[cid]["verdict"] = {
                "status": "BLOCKED_MISSING_RND_AGENT_PLUGIN",
                "evidence_label": "BLOCKED",
                "score": None,
                "failed_primitives": None,
                "reason": (
                    "External @RND AGENT henri_jev plugin is not mounted in this "
                    "host environment; zero scores were fabricated."
                ),
            }
            scores[cid] = None
            failed_primitives[cid] = None

    existing["status"] = status
    existing["evidence_label"] = evidence_label
    existing["plugin_present"] = bool(probe["plugin_present"])
    existing["updated_by_cycle"] = "AG0"
    existing["ag0_probe"] = probe
    existing["challenges"] = challenges
    existing["accepted_count"] = int(accepted_count)
    existing["total_challenges"] = int(len(AF4_CHALLENGE_IDS))
    existing["passed"] = bool(
        probe["plugin_present"]
        and proposals_intact
        and accepted_count == len(AF4_CHALLENGE_IDS)
    )
    written = pc._write(RECEIPTS, "af_jev_mathematical_challenges.json", existing)
    print(
        "[ag0 jev] plugin_present=%s proposals_intact=%s accepted=%d/%d status=%s -> %s"
        % (
            probe["plugin_present"],
            proposals_intact,
            accepted_count,
            len(AF4_CHALLENGE_IDS),
            status,
            written["path"],
        ),
        flush=True,
    )
    return {
        "receipt": "receipts/holarch/af_jev_mathematical_challenges.json",
        "receipt_sha256": written["sha256"],
        "plugin_present": bool(probe["plugin_present"]),
        "proposals_intact": bool(proposals_intact),
        "challenge_ids": list(AF4_CHALLENGE_IDS),
        "accepted_count": int(accepted_count),
        "total_challenges": int(len(AF4_CHALLENGE_IDS)),
        "scores": scores,
        "failed_primitives": failed_primitives,
        "status": status,
        "evidence_label": evidence_label,
        "probe": probe,
    }


def _probe_neural_agent_eval() -> Dict[str, Any]:
    """Re-run scripts/run_holarch_agent_eval.py if mmlu.csv exists; else stamp honest block."""
    mmlu_path = ROOT / "data" / "official_benchmarks" / "canonical" / "mmlu" / "mmlu.csv"
    hist_path = RECEIPTS / "agent_eval.json"
    hist_sidecar = RECEIPTS / "agent_eval.json.sha256"
    hist_valid = bool(
        hist_path.exists()
        and hist_sidecar.exists()
        and _sha_file(hist_path) == hist_sidecar.read_text(encoding="utf-8").strip()
    )
    hist_summary: Dict[str, Any] = {}
    if hist_path.exists():
        try:
            hd = json.loads(hist_path.read_text(encoding="utf-8"))
            hist_summary = {
                "path": "receipts/holarch/agent_eval.json",
                "sidecar_valid": hist_valid,
                "sha256": _sha_file(hist_path),
                "total_queries": int(hd.get("total_queries", hd.get("queries_total", 304))),
                "passed": bool(hd.get("passed", True)),
            }
        except Exception:
            hist_summary = {"path": "receipts/holarch/agent_eval.json", "sidecar_valid": hist_valid}

    if not mmlu_path.exists():
        return {
            "provenance_tier": "legacy_neural_agent_eval_304",
            "script": "scripts/run_holarch_agent_eval.py",
            "mmlu_csv_path": str(mmlu_path.relative_to(ROOT)),
            "mmlu_csv_present": False,
            "status": "BLOCKED_MISSING_GITIGNORED_MMLU_CSV",
            "evidence_label": "BLOCKED",
            "executed_live_in_ag0": False,
            "reason": (
                "Gitignored data/official_benchmarks/canonical/mmlu/mmlu.csv is not "
                "mounted on this host; legacy neural agent eval was not fabricated."
            ),
            "historical_sealed_receipt": hist_summary,
        }

    started = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_holarch_agent_eval.py")],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=600,
    )
    return {
        "provenance_tier": "legacy_neural_agent_eval_304",
        "script": "scripts/run_holarch_agent_eval.py",
        "mmlu_csv_path": str(mmlu_path.relative_to(ROOT)),
        "mmlu_csv_present": True,
        "status": "OBSERVED" if proc.returncode == 0 else "FAILED_RUNTIME",
        "evidence_label": "OBSERVED",
        "executed_live_in_ag0": True,
        "exit_code": int(proc.returncode),
        "seconds": float(time.perf_counter() - started),
        "historical_sealed_receipt": hist_summary,
    }


def main() -> int:
    started = time.perf_counter()
    pinned_before = _sha_file(PIN_SNAPSHOT) if PIN_SNAPSHOT.exists() else ""
    pinned_digest = (
        str(json.loads(PIN_SNAPSHOT.read_text(encoding="utf-8")).get("digest", ""))
        if PIN_SNAPSHOT.exists()
        else ""
    )

    # 1. AF4 JEV challenges reconciliation
    jev_probe = _probe_rnd_agent_jev()
    jev_result = _reconcile_af4_jev(jev_probe)

    # 2. AAII-96 5-arm battery (Arm0 LIVE + SpendGuard probe for Arm1..Arm4)
    with tempfile.TemporaryDirectory(prefix="henri_ag0_") as dtmp:
        tmp_path = Path(dtmp)
        deliv_res = af2.run_deliverable_composer_audit(tmp_path / "deliv")
        battery_res = af2.run_aaii96_battery(tmp_path / "battery")

    # SpendGuard lockfile check explicitly under _work_ag
    ag_lock = RECEIPTS / "_work_ag" / ".ag0_live_cascade.lock"
    credential = resolve_llm_credentials()
    with SpendGuard(ag_lock, budget=120, label="ag0_aaii96_cascade") as guard:
        spend_report = guard.report()

    arm0 = battery_res["arm0_henri_standalone"]
    live_arms = battery_res["live_llm_arms"]
    live_creds_present = bool(credential is not None)
    arm4_beats_arm3 = False
    arm4_call_ratio = None

    # 3. Side-by-side gauntlet: legacy neural eval vs gateway 336-query clean-room gauntlet
    neural_gauntlet = _probe_neural_agent_eval()
    gateway_gauntlet_336 = af2.run_cleanroom_gauntlet_336()

    # 4. Update ad3_deliverable_and_aaii96.json preserving all existing AF2 fields
    ad3_path = RECEIPTS / "ad3_deliverable_and_aaii96.json"
    ad3_payload = json.loads(ad3_path.read_text(encoding="utf-8"))
    ad3_payload["updated_by_cycle"] = "AG0"
    ad3_payload["deliverable_composer"] = deliv_res
    ad3_payload["arm0_henri_standalone"] = arm0
    ad3_payload["live_llm_arms"] = live_arms
    ad3_payload["gauntlet_336"] = gateway_gauntlet_336
    ad3_payload["neural_agent_eval_304"] = neural_gauntlet
    ad3_payload["executed"]["arm0_answerable_solved"] = int(arm0["answerable_solved"])
    ad3_payload["executed"]["arm0_traps_refused"] = int(arm0["traps_refused"])
    ad3_payload["executed"]["arm0_hallucinations"] = int(arm0["hallucinations"])
    ad3_payload["executed"]["arm0_external_api_calls"] = int(arm0["external_api_calls"])
    ad3_payload["executed"]["gauntlet_queries"] = int(gateway_gauntlet_336["queries_total"])
    ad3_payload["executed"]["deliverables"] = int(
        deliv_res["answerable_total"] + deliv_res["traps_total"]
    )
    ad3_gate = dict(ad3_payload.get("gate") or {})
    ad3_gate["ad3_arm0_standalone_ge_66_of_80"] = bool(
        arm0["answerable_solved"] >= 66
        and arm0["traps_refused"] == 16
        and arm0["hallucinations"] == 0
        and arm0["external_api_calls"] == 0
    )
    ad3_gate["ad3_arm4_beats_arm3_composite"] = bool(arm4_beats_arm3)
    ad3_gate["passed"] = bool(all(v for k, v in ad3_gate.items() if k != "passed"))
    ad3_payload["gate"] = ad3_gate
    ad3_payload["passed"] = bool(ad3_gate["passed"])
    ad3_payload["status"] = (
        "CLOSED_OBSERVED"
        if ad3_payload["passed"]
        else "PARTIAL_OBSERVED_LIVE_LLM_ARMS_BLOCKED_MISSING_CREDENTIALS"
    )
    ad3_written = pc._write(RECEIPTS, "ad3_deliverable_and_aaii96.json", ad3_payload)

    pinned_after = _sha_file(PIN_SNAPSHOT) if PIN_SNAPSHOT.exists() else ""
    pinned_ok = bool(pinned_before == pinned_after == PIN_SHA and pinned_digest == PIN_DIGEST)

    checks = {
        "ag0_arm0_ge_75_of_80_and_16_traps_zero_hall_zero_llm": bool(
            arm0["answerable_solved"] >= 75
            and arm0["answerable_total"] == 80
            and arm0["traps_refused"] == 16
            and arm0["traps_total"] == 16
            and arm0["hallucinations"] == 0
            and arm0["external_api_calls"] == 0
        ),
        "ag0_arm4_live_beats_arm3_at_le_15pct_calls": bool(arm4_beats_arm3),
        "ag0_jev_af4_2_of_2_accepted_ge_094": bool(
            jev_result["plugin_present"]
            and jev_result["accepted_count"] == 2
        ),
        "ag0_af4_jev_proposals_intact_honest_provenance": bool(
            jev_result["proposals_intact"]
            and (
                jev_result["status"] == "ACCEPTED_OBSERVED"
                or jev_result["status"] == "BLOCKED_MISSING_RND_AGENT_PLUGIN"
            )
        ),
        "ag0_gateway_gauntlet_336_observed_perfect": bool(
            gateway_gauntlet_336["queries_total"] == 336
            and gateway_gauntlet_336["queries_correct"] == 336
            and gateway_gauntlet_336["reachable_accuracy"] == 1.0
            and gateway_gauntlet_336["hallucination_rate"] == 0.0
            and gateway_gauntlet_336["query_file_reads"] == 0
            and gateway_gauntlet_336["receipt_chain_valid"]
        ),
        "ag0_neural_gauntlet_rerun_observed": bool(
            neural_gauntlet["executed_live_in_ag0"]
            and neural_gauntlet["status"] == "OBSERVED"
        ),
        "ag0_pinned_snapshot_unchanged": pinned_ok,
    }

    wall = float(time.perf_counter() - started)
    payload: Dict[str, Any] = {
        "stage": "AG0",
        "spec": SPEC_ID,
        "evidence_label": "OBSERVED_WITH_HONEST_HOST_BLOCKS",
        "compute_device": "cpu",
        "status": (
            "CLOSED_OBSERVED"
            if all(checks.values())
            else "PARTIAL_OBSERVED_HOST_EXTERNAL_DEPS_BLOCKED"
        ),
        "af4_jev_closure": jev_result,
        "aaii96_battery": {
            "total_tasks": int(battery_res["total_tasks"]),
            "answerable_total": int(battery_res["answerable_total"]),
            "traps_total": int(battery_res["traps_total"]),
            "arm0_henri_standalone": arm0,
            "live_llm_arms": live_arms,
            "credentials_present": live_creds_present,
            "arm4_call_ratio_vs_arm3": arm4_call_ratio,
            "spend_guard": spend_report,
            "updated_ad3_receipt": {
                "path": "receipts/holarch/ad3_deliverable_and_aaii96.json",
                "sha256": ad3_written["sha256"],
            },
        },
        "gauntlet_comparison": {
            "gateway_cleanroom_gauntlet_336": {
                "provenance_tier": "gateway_cleanroom_gauntlet_336",
                "evidence_label": "OBSERVED",
                "executed_live_in_ag0": True,
                "queries_total": int(gateway_gauntlet_336["queries_total"]),
                "queries_correct": int(gateway_gauntlet_336["queries_correct"]),
                "reachable_correct": int(gateway_gauntlet_336["reachable_correct"]),
                "reachable_total": int(gateway_gauntlet_336["reachable_total"]),
                "traps_refused": int(gateway_gauntlet_336["traps_refused"]),
                "traps_total": int(gateway_gauntlet_336["traps_total"]),
                "hallucination_rate": float(gateway_gauntlet_336["hallucination_rate"]),
                "query_file_reads": int(gateway_gauntlet_336["query_file_reads"]),
                "receipt_chain_valid": bool(gateway_gauntlet_336["receipt_chain_valid"]),
                "per_category": gateway_gauntlet_336["per_category"],
            },
            "legacy_neural_agent_eval_304": neural_gauntlet,
        },
        "host_blocks": [
            {
                "component": "henri_jev_plugin",
                "code": "BLOCKED_MISSING_RND_AGENT_PLUGIN",
                "present": bool(jev_probe["plugin_present"]),
            },
            {
                "component": "aaii96_live_llm_arms_1_to_4",
                "code": "BLOCKED_MISSING_LLM_CREDENTIALS",
                "present": live_creds_present,
            },
            {
                "component": "legacy_neural_agent_eval_mmlu",
                "code": "BLOCKED_MISSING_GITIGNORED_MMLU_CSV",
                "present": bool(neural_gauntlet["mmlu_csv_present"]),
            },
        ],
        "pinned_snapshot": {
            "path": str(PIN_SNAPSHOT.relative_to(ROOT)),
            "sha256": pinned_after,
            "sha256_before": pinned_before,
            "expected_sha256": PIN_SHA,
            "expected_digest": PIN_DIGEST,
        },
        "wall_seconds": wall,
    }

    leaks = _secret_scan(payload) + _secret_scan(ad3_payload)
    checks["ag0_zero_credential_leaks"] = bool(not leaks)
    gate = pc._finalise(checks)
    payload["credential_scan"] = {"leaks": leaks, "zero_leaks": bool(not leaks)}
    payload["gate"] = gate
    payload["passed"] = bool(gate["passed"])
    payload["local_observed_gates_passed"] = bool(
        checks["ag0_arm0_ge_75_of_80_and_16_traps_zero_hall_zero_llm"]
        and checks["ag0_af4_jev_proposals_intact_honest_provenance"]
        and checks["ag0_gateway_gauntlet_336_observed_perfect"]
        and checks["ag0_pinned_snapshot_unchanged"]
        and checks["ag0_zero_credential_leaks"]
    )
    payload["evidence"] = "OBSERVED"

    written = pc._write(RECEIPTS, "ag0_live_cascade_and_jev_closure.json", payload)
    failed = [k for k, v in gate.items() if k != "passed" and not bool(v)]
    print(
        "[ag0] SEALED arm0=%d/80 traps=%d/16 hall=%d gauntlet336=%d/336 "
        "local_observed=%s full_passed=%s blocked_gates=%s -> %s"
        % (
            arm0["answerable_solved"],
            arm0["traps_refused"],
            arm0["hallucinations"],
            gateway_gauntlet_336["queries_correct"],
            payload["local_observed_gates_passed"],
            payload["passed"],
            failed,
            written["path"],
        ),
        flush=True,
    )
    return 0 if payload["local_observed_gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
