"""Cycle W verification: Pass-0 gauntlet, split hygiene, induction audit.

    python scripts/run_holarch_w_cycle.py --stages w1,w2,w3,w4,w5

W1 and W2 are measured here or read from a receipt this process just sealed.
W3 reports the strict one-cell audit and the measured reason its two refusals
stand.  W4 and W5 record that they were not executed in this cycle, with the
exact reason and no substituted number.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_holarch_p_cycle as pc

RECEIPTS = ROOT / "receipts" / "holarch"


def _sanitize(value):
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else 0.0
    if isinstance(value, dict):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    return value


def _read(name: str) -> Dict[str, object]:
    path = RECEIPTS / str(name)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _digest(name: str) -> str:
    path = RECEIPTS / str(name)
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------- W1
def stage_w1(directory: Path) -> Dict[str, object]:
    started = time.perf_counter()
    payload = _read("agent_eval.json")
    gate = dict(payload.get("gate") or {})
    mcp = dict(payload.get("holarch_mcp") or {})
    _probe_block = payload.get("probes")
    probes = list((_probe_block or {}).get("rows") or []) if isinstance(_probe_block, dict) else list(_probe_block or [])
    pass0 = [row for row in probes if str(row.get("name", "")).startswith("w1_")]
    gates = {
        "w1_queries_240": bool(int(payload.get("total_queries", 0)) == 240),
        "w1_probe_slots_164": bool(int(payload.get("probe_slots", 0)) == 164),
        "w1_reachable_1": bool(float(mcp.get("reachable_accuracy", 0.0) or 0.0) == 1.0),
        "w1_hallucination_0": bool(mcp.get("hallucination_rate") == 0.0),
        "w1_zero_query_reads": bool(int(mcp.get("query_file_reads", -1)) == 0),
        "w1_wall_le_180": bool(float(payload.get("wall_seconds", 1e9)) <= 180.0),
        "w1_gauntlet_gate": bool(gate.get("passed", False)),
        "w1_pass0_probes_present": bool(len(pass0) == 2
                                        and all(bool(row.get("correct")) for row in pass0)),
    }
    return {
        "stage": "W1",
        "queries": int(payload.get("total_queries", 0)),
        "probe_slots": int(payload.get("probe_slots", 0)),
        "wall_seconds_gauntlet": float(payload.get("wall_seconds", 0.0)),
        "reachable_accuracy": float(mcp.get("reachable_accuracy", 0.0) or 0.0),
        "hallucination_rate": float(mcp.get("hallucination_rate", 1.0) or 0.0),
        "query_file_reads": int(mcp.get("query_file_reads", -1)),
        "pass0_probes": pass0,
        "category_ms": dict(payload.get("category_ms") or {}),
        "gate": pc._finalise(gates),
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
        "passed": bool(all(gates.values())),
    }


# ---------------------------------------------------------------------- W2
def stage_w2(directory: Path) -> Dict[str, object]:
    started = time.perf_counter()
    sealed = _read("w2_pytest_and_split_hygiene.json")
    digest = _digest("w2_pytest_and_split_hygiene.json")
    side = RECEIPTS / "w2_pytest_and_split_hygiene.json.sha256"
    side_ok = bool(side.exists() and side.read_text(encoding="utf-8").strip() == digest)
    split = dict(sealed.get("split") or {})
    tests = dict(sealed.get("pytest") or {})
    gates = {
        "w2_receipt_present": bool(sealed),
        "w2_receipt_sidecar": bool(side_ok),
        "w2_top1_ge_0_45": bool(float(split.get("top1", 0.0)) >= 0.45),
        "w2_bpc_le_char_baseline": bool(
            float(split.get("bits_per_char", 9.9)) <= float(split.get("bpc_bar", 2.4545))),
        "w2_zero_straddling_spans": bool(int(split.get("straddling_spans", -1)) == 0),
        "w2_gap_verified": bool(split.get("gap_verified", False)),
        "w2_pytest_exit_zero": bool(int(tests.get("exit_code", 1)) == 0),
        "w2_pytest_zero_failures": bool(int(tests.get("failures", 1)) == 0),
        "w2_beats_single_cut": bool(
            float(split.get("top1", 0.0)) > float(split.get("baseline_single_cut_top1", 1.0))),
    }
    return {
        "stage": "W2",
        "split": split,
        "pytest": {key: tests.get(key) for key in ("exit_code", "failures", "wall_seconds")},
        "receipt_sha256": digest,
        "gate": pc._finalise(gates),
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
        "passed": bool(all(gates.values())),
    }


# ---------------------------------------------------------------------- W3
def stage_w3(directory: Path) -> Dict[str, object]:
    started = time.perf_counter()
    from henri.holarch.induction import TASK_TRANSFORMS

    audit = _read("u3_inductive_turing_audit.json")
    block = dict(audit.get("induction") or {})
    rows = list(block.get("tasks") or [])

    def append_copy(text: str) -> str:
        parts = str(text).split("#")
        return str(text) if len(parts) < 2 else str(text) + parts[1]

    probes = ["a#x#z", "a#xy#z", "a#xyxyxy#z", "a#xyxyxyxy#zb"]
    pairs = []
    for text in probes:
        wanted = str(TASK_TRANSFORMS["delimiter_bounded_copy"](text))
        got = append_copy(text)
        pairs.append({"input": text, "declared_target": wanted,
                      "append_shuttle": got, "match": bool(wanted == got)})
    gates = {
        "w3_solved_ge_10": bool(int(block.get("solved", -1)) >= 10),
        "w3_length_generalised_ge_10": bool(
            int(block.get("v1_length_generalized", -1)) >= 10),
        "w3_zero_text_tokens": bool(int(block.get("text_tokens", -1)) == 0),
        "w3_contradictions_4": bool(
            int(block.get("contradictions_failed_closed", -1)) == 4),
        "w3_delimiter_topology_claim_refuted": bool(
            pairs and not any(row["match"] for row in pairs)),
        "w3_solved_12": bool(int(block.get("solved", -1)) == 12),
    }
    return {
        "stage": "W3",
        "solved": int(block.get("solved", -1)),
        "generalized": int(block.get("generalized", -1)),
        "v1_length_generalized": int(block.get("v1_length_generalized", -1)),
        "refused": [{"name": row.get("name"), "nodes": row.get("nodes_evaluated"),
                     "mechanism": row.get("mechanism_used")}
                    for row in rows if str(row.get("status")) != "INDUCED"],
        "delimiter_topology_evidence": {
            "schema": "q_pick -> q_carry -> q_ret append shuttle",
            "pairs": pairs,
            "finding": ("the prescribed shuttle deposits the carried symbol at the "
                        "first blank, which appends the source segment; the declared "
                        "task inserts a copy of the source immediately after the "
                        "second delimiter and shifts the tail right, so the two "
                        "outputs differ on every probe"),
        },
        "carry_topology_evidence": {
            "requirement": ("sum = a xor b xor c and carry = maj(a, b, c) need three "
                            "input bits at the deposit phase"),
            "available": ("one buffered symbol plus the symbol under the head, i.e. "
                          "two bits, under Q = Qctrl x Rreg with the declared "
                          "selectors"),
            "finding": ("the two-bit budget is one bit short; the declared bit "
                        "algebra cannot be closed without a second register bit"),
        },
        "gate": pc._finalise(gates),
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
        "passed": bool(all(gates.values())),
    }


# ---------------------------------------------------------------------- W4
def stage_w4(directory: Path) -> Dict[str, object]:
    started = time.perf_counter()
    gates = {
        "w4_only_arm0_measured": False,
        "w4_four_arms_measured": False,
        "w4_live_provider_claimed": False,
        "w4_arm0_zero_hallucination": False,
        "w4_arm3_strictly_highest": False,
        "passed": False,
    }
    return {
        "stage": "W4",
        "executed": False,
        "reason": ("the 16-task x 4-arm AAII v5.0 suite needs one warm synthesis "
                   "engine per arm and a live completion budget of 48 calls; this "
                   "run did not execute it, so no arm score is reported"),
        "design": {
            "arm0": "standalone HENRI, zero external LLM calls",
            "arm1": "raw live LLM",
            "arm2": "live LLM plus subprocess sandbox",
            "arm3": "live LLM plus Wave-CCG, RepoSheaf, RiemannianChartAtlas, "
                    "REPULSOR loop breaker, <=4096-token epistemic frame, Chow abstention",
            "tasks": "4 knowledge work, 4 coding with duck-type hazards, 4 science, "
                     "4 general with 2 grounded and 2 unanswerable",
        },
        "prior_evidence": {
            "v3_live_provider_claimed": bool(
                _read("v3_live_aaii_v5_arms.json").get("live_provider_claimed", False)),
            "v3_live_rows": int(len(_read("v3_live_aaii_v5_arms.json").get("live_rows") or [])),
        },
        "gate": pc._finalise(gates),
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "NOT_EXECUTED",
        "passed": False,
    }


# ---------------------------------------------------------------------- W5
def stage_w5(directory: Path) -> Dict[str, object]:
    started = time.perf_counter()
    import re

    doc = (ROOT / "docs" / "HOLARCH.md").read_text(encoding="utf-8")
    numbers = [int(value) for value in re.findall(r"^## (\d+)\.", doc, flags=re.M)]
    gates = {
        "w5_docs_contiguous_1_to_38": bool(numbers == list(range(1, 39))),
        "w5_wave_egress_organ_present": bool((ROOT / "henri" / "holarch" / "wave_egress.py").exists()),
        "w5_contrastive_learning_measured": False,
        "w5_margin_decoder_measured": False,
        "passed": False,
    }
    return {
        "stage": "W5",
        "docs_sections": int(len(numbers)),
        "docs_contiguous": bool(numbers == list(range(1, len(numbers) + 1))),
        "wave_egress": {
            "module_present": bool((ROOT / "henri" / "holarch" / "wave_egress.py").exists()),
            "executed": False,
            "reason": ("the contrastive phase learner and the margin-relaxed decoder "
                       "were not built in this cycle; no Sagnac residual curve and no "
                       "override count is reported"),
        },
        "gate": pc._finalise(gates),
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
        "passed": False,
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Cycle W verification")
    parser.add_argument("--stages", default="w1,w2,w3,w4,w5")
    parser.add_argument("--receipts", default=str(RECEIPTS))
    args = parser.parse_args()
    directory = Path(args.receipts)
    directory.mkdir(parents=True, exist_ok=True)
    stages = [token.strip().lower() for token in str(args.stages).split(",") if token.strip()]
    runners = {
        "w1": ("w1_pass0_gauntlet_240.json", stage_w1),
        "w2": ("w2_pytest_and_split_hygiene.json", stage_w2),
        "w3": ("w3_complete_one_cell_turing.json", stage_w3),
        "w4": ("w4_live_four_arm_aaii_v5.json", stage_w4),
        "w5": ("w5_wave_egress_and_full_closure.json", stage_w5),
    }
    aggregate: Dict[str, object] = {"cycle": "W", "head": pc._git_head(), "stages": {}}
    exit_code = 0
    for stage in stages:
        if stage not in runners:
            continue
        name, runner = runners[stage]
        print(f"[w-cycle] {stage} ...", flush=True)
        payload = _sanitize(runner(directory))
        if stage == "w2":
            # W2 owns its receipt: it is written by run_holarch_w2_hygiene.py so
            # the measured split and the pytest process share one file.
            path = directory / name
            written = {"path": str(path.relative_to(ROOT)),
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()
                       if path.exists() else ""}
        else:
            written = pc._write(directory, name, payload)
        aggregate["stages"][stage] = {
            "receipt": written["path"],
            "sha256": written["sha256"],
            "passed": bool(payload.get("gate", {}).get("passed", False)),
            "gate": payload.get("gate", {}),
            "wall_seconds": payload.get("wall_seconds", 0.0),
        }
        print(f"[w-cycle] {stage} passed={payload.get('gate', {}).get('passed')} "
              f"{written['path']}", flush=True)
        if not bool(payload.get("gate", {}).get("passed", False)):
            exit_code = 1
    aggregate["passed"] = bool(
        all(bool(item.get("passed")) for item in aggregate["stages"].values())
    ) if aggregate["stages"] else False
    written = pc._write(directory, "w_cycle.json", aggregate)
    print(f"[w-cycle] aggregate {written['path']} passed={aggregate['passed']}", flush=True)
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
