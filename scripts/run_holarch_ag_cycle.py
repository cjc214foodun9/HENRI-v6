"""Cycle AG master seal: full pytest, consistency check, sidecar parity, R3
language hygiene, credential scan, core/frozen immutability audit, and
receipts/holarch/ag_cycle.json ledger.

    python scripts/run_holarch_ag_cycle.py
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

import run_holarch_af_cycle as af_cycle
import run_holarch_p_cycle as pc

RECEIPTS = ROOT / "receipts" / "holarch"
DOCS = ROOT / "docs" / "HOLARCH.md"
PIN_SNAPSHOT = RECEIPTS / "pinned_corpus_snapshot.json"
PIN_SHA = "d675f3816bbf472493b25f885bbb56c62c5ec88a81ab31e31a9b87b3a0c23703"
PIN_DIGEST = "8f270f34e8f35ef84e415d076bea20a7bc15edff337f89ed72ad607447ba8554"

AG_STAGES = (
    "ag0_live_cascade_and_jev_closure.json",
    "ag1_blind_arc_100_audit.json",
    "ag2_raw_prose_science_audit.json",
    "ag3_multifile_repo_and_100k_atlas.json",
    "ag4_blackwell_and_jev.json",
    "ag_jev_mathematical_challenges.json",
)

IMMUTABLE_CORE_PHYSICS = {
    "henri/holarch/model.py": "18da661b81cf4f2caf0e272b5124512e5ed6a1ca7720df4750008d823accece5",
    "henri/holarch/zone_a.py": "e6a0fa1cf4f7cfdb77fc3d5386b2312dfe2acec0f87784b8ce3a645794f8527e",
    "henri/holarch/zone_b.py": "38ee1740605e2c5a39d6f71fc21fec3894251045f59705b12c89e4fb11a88228",
    "henri/holarch/backends/__init__.py": "57c45d99782d915824ab82ebc93e6681fa41b4074c8f21a4c7e641974145b9b8",
    "henri/holarch/backends/blackwell.py": "f457a3ecfbaf65a599d15f4038dd6afd5d1120bfc392cbac03bee669bf7647c1",
    "henri/holarch/backends/reference.py": "b92d978082226bb56edc28485e9058c484bf8bed9b212cf9e45aa6db400dc08d",
}

FROZEN_HOLDOUT_MODULES = {
    "henri/holarch/arc_objects.py": "94d34f3266c7c27aee4492743b27fad8e7d2507fd92b74cacd8f65d2556491a8",
    "henri/holarch/science_parser.py": "03ba04c6cc866c603dd3e97bfc331ee4917cc7283e662c5a1f8763565a390a08",
    "henri/holarch/science_solver.py": "2cc3937906a35782e06a35f515ae9843fed5ab42f65ae48b17d9151e79c4c553",
}


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


def audit_digests(expected_map: Dict[str, str]) -> Dict[str, Any]:
    files: Dict[str, Dict[str, Any]] = {}
    all_ok = True
    for rel, exp in expected_map.items():
        actual = _sha(ROOT / rel)
        ok = bool(actual == exp)
        if not ok:
            all_ok = False
        files[rel] = {"expected_sha256": exp, "actual_sha256": actual, "unchanged": ok}
    return {"all_unchanged": all_ok, "files": files}


def run_pytest_ag(timeout: int = 2400) -> Dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-o", "addopts=", "-q", "-rs", "-p", "no:cacheprovider"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    text = (completed.stdout or "") + "\n" + (completed.stderr or "")
    lines = [line for line in text.strip().splitlines() if line.strip()]
    summary_line = lines[-1] if lines else ""
    counts: Dict[str, int] = {}
    for match in re.finditer(
        r"(\d+)\s+(passed|failed|error|errors|skipped|xfailed|xpassed)", summary_line
    ):
        key = "error" if match.group(2) == "errors" else match.group(2)
        counts[key] = int(match.group(1))
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
    return {
        "exit_code": int(completed.returncode),
        "counts": counts,
        "skip_reasons": skip_reasons,
        "summary_line": summary_line,
        "wall_seconds": float(time.perf_counter() - started),
        "tail": lines[-15:],
    }


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Cycle AG master seal")
    parser.add_argument("--receipts", default=str(RECEIPTS))
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args(argv)
    directory = Path(args.receipts).resolve()
    started = time.perf_counter()

    stages = {name: _read(name) for name in AG_STAGES}
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

    core_audit = audit_digests(IMMUTABLE_CORE_PHYSICS)
    frozen_audit = audit_digests(FROZEN_HOLDOUT_MODULES)

    tests: Dict[str, Any] = {
        "exit_code": 1,
        "counts": {},
        "skip_reasons": {},
        "summary_line": "skipped",
        "wall_seconds": 0.0,
        "tail": ["skipped"],
    }
    if not args.skip_tests:
        print("[ag] Running full pytest suite...", flush=True)
        tests = run_pytest_ag()
        print(
            "[ag] pytest finished: rc=%d counts=%s wall=%.1fs"
            % (tests["exit_code"], tests["counts"], tests["wall_seconds"]),
            flush=True,
        )

    docs_numbers = [
        int(value)
        for value in re.findall(r"^## (\d+)\.", DOCS.read_text(encoding="utf-8"), flags=re.M)
    ]

    def _build_receipt(
        consistency: Dict[str, Any], parity: Dict[str, Any], secrets: Dict[str, Any]
    ) -> Dict[str, Any]:
        ag0 = stages["ag0_live_cascade_and_jev_closure.json"]
        ag1 = stages["ag1_blind_arc_100_audit.json"]
        ag2 = stages["ag2_raw_prose_science_audit.json"]
        ag3 = stages["ag3_multifile_repo_and_100k_atlas.json"]
        ag4 = stages["ag4_blackwell_and_jev.json"]
        ag_jev = stages["ag_jev_mathematical_challenges.json"]

        ag3_mf = ((ag3.get("part_a_repo_patch") or {}).get("multifile") or {})
        ag3_qb = ((ag3.get("part_a_repo_patch") or {}).get("quixbugs_non_regression") or {})
        ag3_b = (ag3.get("part_b_100k_persistent_atlas") or {})
        ag3_100 = (ag3_b.get("world_100k") or {})
        ag3_105 = (ag3_b.get("online_105k") or {})
        ag3_cold = ((ag3_b.get("persistence_and_cold_restart") or {}).get("cold_gateway_subprocess") or {})
        ag4_d4096 = (ag4.get("d4096_cpu_thermodynamic_contract") or {})

        gates = {
            "ag_stage_receipts_present": bool(all(bool(stages.get(name)) for name in AG_STAGES)),
            "ag0_local_observed_gates_passed": bool(ag0.get("local_observed_gates_passed")),
            "ag0_live_llm_arms_closed": bool(
                (ag0.get("gate") or {}).get("ag0_arm4_live_beats_arm3_at_le_15pct_calls")
            ),
            "ag0_af4_jev_plugin_accepted": bool(
                (ag0.get("gate") or {}).get("ag0_jev_af4_2_of_2_accepted_ge_094")
            ),
            "ag0_neural_gauntlet_rerun_observed": bool(
                (ag0.get("gate") or {}).get("ag0_neural_gauntlet_rerun_observed")
            ),
            "ag1_blind_arc_100_passed": bool(ag1.get("passed")),
            "ag2_raw_prose_science_passed": bool(ag2.get("passed")),
            "ag3_multifile_and_100k_atlas_passed": bool(ag3.get("passed")),
            "ag4_policy_d4096_and_proposals_passed": bool(
                ag4.get("local_policy_and_proposal_gates_passed")
            ),
            "ag4_jev_plugin_accepted": bool(ag_jev.get("accepted")),
            "ag_core_physics_zero_bytes_mutated": bool(core_audit["all_unchanged"]),
            "ag_frozen_holdout_modules_unchanged": bool(frozen_audit["all_unchanged"]),
            "ag_pinned_snapshot_digest_unchanged": bool(snapshot_digest == PIN_DIGEST),
            "ag_pinned_snapshot_sha_unchanged": bool(snapshot_sha == PIN_SHA),
            "ag_sidecar_parity_one": bool(float(parity["parity"]) == 1.0),
            "ag_zero_credentials": bool(secrets["zero_credentials"]),
            "ag_consistency_valid": bool(
                consistency.get("exit_code") == 0 and consistency.get("valid")
            ),
            "ag_docs_sections_1_to_48": bool(docs_numbers == list(range(1, 49))),
            "ag_r3_static_bpc_le_14200": bool(static_bpc <= 1.4200),
            "ag_r3_streaming_bpc_le_12200": bool(streaming_bpc <= 1.2200),
            "ag_pytest_exit_zero": bool(int(tests["exit_code"]) == 0),
            "ag_pytest_zero_failures": bool(int(tests["counts"].get("failed", 0)) == 0),
            "ag_pytest_zero_errors": bool(
                int(tests["counts"].get("error", 0)) + int(tests["counts"].get("errors", 0)) == 0
            ),
        }
        blocked_external_keys = {
            "ag0_live_llm_arms_closed",
            "ag0_af4_jev_plugin_accepted",
            "ag0_neural_gauntlet_rerun_observed",
            "ag4_jev_plugin_accepted",
        }
        local_observed_all_passed = bool(
            all(v for k, v in gates.items() if k not in blocked_external_keys)
        )
        return {
            "stage": "AG",
            "cycle": "AG",
            "spec": "HENRI-SPEC-CYCLE-AG-BLIND-HOLDOUT-RAW-PROSE-SCIENCE-LIVE-CASCADE-100K-ATLAS-AND-BLACKWELL-SCALE-V1",
            "evidence_label": "OBSERVED",
            "compute_device": "cpu",
            "headline": {
                "ag0_arm0_answerable": int(
                    (((ag0.get("aaii96_battery") or {}).get("arm0_henri_standalone") or {}).get("answerable_solved", 0))
                ),
                "ag0_arm0_traps_refused": int(
                    (((ag0.get("aaii96_battery") or {}).get("arm0_henri_standalone") or {}).get("traps_refused", 0))
                ),
                "ag0_gauntlet336_correct": int(
                    (
                        ((ag0.get("gauntlet_comparison") or {}).get("gateway_cleanroom_gauntlet_336") or {})
                        .get("queries_correct", 0)
                    )
                ),
                "ag1_blind_arc_100_solved": int((ag1.get("results") or {}).get("solved", 0)),
                "ag1_blind_arc_100_precision": float(
                    (ag1.get("results") or {}).get("emitted_precision", 0.0)
                ),
                "ag2_dev24_raw_prose_si": int(
                    (ag2.get("tier1_dev24_raw_prose") or {}).get("solved_si_equivalent", 0)
                ),
                "ag2_dev24_raw_prose_scalar": int(
                    (ag2.get("tier1_dev24_raw_prose") or {}).get("solved_raw_scalar", 0)
                ),
                "ag2_blind24_emitted": int(
                    (ag2.get("tier2_blind24_scibench_holdout") or {}).get("emitted", 0)
                ),
                "ag2_blind24_hallucinations": int(
                    max(
                        0,
                        int((ag2.get("tier2_blind24_scibench_holdout") or {}).get("emitted", 0))
                        - int(
                            (ag2.get("tier2_blind24_scibench_holdout") or {}).get(
                                "solved_si_equivalent", 0
                            )
                        ),
                    )
                ),
                "ag3_multifile_pass": f"{int(ag3_mf.get('passed_hidden', 0))}/{int(ag3_mf.get('total_cases', 12))}",
                "ag3_multifile_precision": float(ag3_mf.get("emitted_precision", 0.0)),
                "ag3_multifile_traps": f"{int(ag3_mf.get('traps_refused', 0))}/{int(ag3_mf.get('traps_total', 4))}",
                "ag3_quixbugs_pass": f"{int(ag3_qb.get('passed_hidden', 0))}/{int(ag3_qb.get('total_tasks', 14))}",
                "ag3_100k_speedup_complex": float(ag3_100.get("speedup_vs_flat_complex_blas", 0.0)),
                "ag3_100k_speedup_real": float(ag3_100.get("speedup_vs_flat_real_blas", 0.0)),
                "ag3_105k_speedup_complex": float(ag3_105.get("speedup_vs_flat_complex_blas", 0.0)),
                "ag3_105k_speedup_real": float(ag3_105.get("speedup_vs_flat_real_blas", 0.0)),
                "ag3_105k_online_recall": int(ag3_105.get("online_recall_hits", 0)),
                "ag3_cold_mount_seconds": float(ag3_cold.get("cold_mount_seconds", 0.0)),
                "ag3_cold_rebuilt_charts": int(ag3_cold.get("rebuilt_charts_total", -1)),
                "ag4_cuda_status": str(ag4.get("cuda_status", "")),
                "ag4_d4096_params": int(ag4_d4096.get("parameter_count", 0)),
                "ag4_d4096_cpu_contract_ok": bool(ag4_d4096.get("contract_ok", False)),
                "ag4_jev_status": str(ag_jev.get("status", "")),
                "local_observed_all_passed": local_observed_all_passed,
            },
            "stages": {
                name: {
                    "passed": bool(stages[name].get("passed", False)),
                    "local_ok": bool(
                        stages[name].get(
                            "local_observed_gates_passed",
                            stages[name].get(
                                "local_policy_and_proposal_gates_passed",
                                stages[name].get("passed", False),
                            ),
                        )
                    ),
                }
                for name in AG_STAGES
            },
            "blocked_external_boundaries": {
                "live_llm_arms_aaii96": "BLOCKED_MISSING_LLM_CREDENTIALS",
                "blackwell_sm120_vast": "CUDA: BLOCKED_BY_POLICY",
                "henri_jev_plugin": "BLOCKED_MISSING_RND_AGENT_PLUGIN",
                "gitignored_mmlu_csv": "BLOCKED_MISSING_GITIGNORED_MMLU_CSV",
                "sentence_transformers_offline_model": "BLOCKED_MISSING_SENTENCE_TRANSFORMERS_MODEL",
            },
            "immutability": {
                "core_physics": core_audit,
                "frozen_holdout_modules": frozen_audit,
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
                "sections_1_to_48": bool(docs_numbers == list(range(1, 49))),
            },
            "pinned": {"snapshot_sha256": snapshot_sha, "digest": snapshot_digest},
            "gate": pc._finalise(gates),
            "local_observed_all_passed": local_observed_all_passed,
            "wall_seconds": float(time.perf_counter() - started),
            "evidence": "OBSERVED",
            "passed": bool(all(gates.values())),
        }

    # First write registers ag_cycle.json + .sha256 so consistency & parity count it.
    pc._write(
        directory,
        "ag_cycle.json",
        _build_receipt(
            {"exit_code": 0, "valid": True},
            {"parity": 1.0, "receipts": 0},
            {"hits": [], "zero_credentials": True},
        ),
    )
    parity = af_cycle.sidecar_parity()
    secrets = af_cycle.secret_scan()
    consistency = af_cycle.run_consistency()
    final_receipt = _build_receipt(consistency, parity, secrets)
    written = pc._write(directory, "ag_cycle.json", final_receipt)

    post_check = af_cycle.run_consistency()
    failed = [k for k, v in final_receipt["gate"].items() if k != "passed" and not bool(v)]
    print(
        "[ag] local_observed_all_passed=%s passed=%s failed=%s pytest=%s consistency_valid=%s receipts=%s -> %s"
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
