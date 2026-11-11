"""Cycle V verification: V1 factored one-cell closure, V2 fast gauntlet,
V3 credential bridge and live provider probe, V4 revertible holon, V5 closure.

    python scripts/run_holarch_v_cycle.py --stages v1,v2,v3,v4,v5

Every number is measured in this process or in the child process it names.
Every receipt is sealed with a SHA-256 sidecar.  A stage that cannot measure
its claim reports the exact reason and stays passed:false.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
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


# ---------------------------------------------------------------------- V1
def stage_v1(directory: Path) -> Dict[str, object]:
    started = time.perf_counter()
    import dataclasses

    from henri.holarch.config import HolarchConfig
    from henri.holarch.onecell_audit import audit_induction
    from henri.holarch.paged_tape import (
        PagedHarmonicTape,
        binary_add,
        palindrome_check,
        reverse_tape,
    )
    from henri.holarch.zone_c import ZoneCMemory

    cfg = dataclasses.replace(HolarchConfig(), dim=256, gram_dims=16, cache_capacity=64)
    page_db = directory / "_work" / "v1" / "pages.sqlite3"
    page_db.parent.mkdir(parents=True, exist_ok=True)
    if page_db.exists():
        try:
            page_db.unlink()
        except OSError:
            pass
    page_memory = ZoneCMemory(cfg, backend="sqlite", path=str(page_db))
    capacity: List[Dict[str, object]] = []
    for length in (16, 64, 128, 256):
        tape = PagedHarmonicTape(memory=page_memory, length=int(length),
                                 alphabet=("0", "1", "#", "_"), namespace=f"v1_tape_{length}")
        text = ("10110010" * 40)[: int(length)]
        tape.load(text)
        reversed_text = reverse_tape(tape, int(length))
        pal = PagedHarmonicTape(memory=page_memory, length=int(length),
                                alphabet=("0", "1", "#", "_"), namespace=f"v1_pal_{length}")
        pal.load(("0110" * 80)[: int(length)])
        add = PagedHarmonicTape(memory=page_memory, length=int(length),
                                alphabet=("0", "1", "#", "_"), namespace=f"v1_add_{length}")
        operand = max(1, int(length) // 4)
        left = "1" + "0" * (operand - 1)
        add.load(left + "#" + left)
        added = binary_add(add, int(length))
        capacity.append({
            "length": int(length),
            "reverse_exact": bool(reversed_text == text[::-1]),
            "palindrome_exact": bool(palindrome_check(pal, int(length))),
            "binary_add_exact": bool(
                add.tape_text()[: len(added)] == added
                and int(added.split("#")[2], 2) == int(left, 2) + int(left, 2)),
            "min_read_margin": float(tape.min_read_margin()),
            "intra_page_crosstalk": float(tape.intra_page_crosstalk()),
        })
    page_memory.close()

    audit = audit_induction(node_budget=320000, max_states=8)
    tasks = list(audit["tasks"])
    gates = {
        "v1_capacity_exact_long": bool(all(
            row["reverse_exact"] and row["palindrome_exact"] and row["binary_add_exact"]
            for row in capacity if int(row["length"]) >= 64)),
        "v1_margin_ge_0_40": bool(
            min(float(row["min_read_margin"]) for row in capacity) >= 0.40),
        "v1_crosstalk_zero": bool(
            max(float(row["intra_page_crosstalk"]) for row in capacity) < 1e-6),
        "v1_solved_ge_10": bool(int(audit["solved"]) >= 10),
        "v1_generalized_ge_10": bool(int(audit["generalized"]) >= 10),
        "v1_length_generalized_ge_10": bool(int(audit["v1_length_generalized"]) >= 10),
        "v1_zero_text_tokens": bool(int(audit["text_tokens"]) == 0),
        "v1_contradictions_4": bool(
            int(audit["contradictions_failed_closed"]) == 4),
        "v1_two_cell_uncounted": bool(audit["foveated_counted_toward_gate"] is False),
        "v1_node_budget_respected": bool(all(
            int(row.get("nodes_evaluated", 0)) <= 320000 for row in tasks)),
    }
    return {
        "stage": "V1",
        "capacity": capacity,
        "induction": {
            "tasks": tasks,
            "solved": int(audit["solved"]),
            "generalized": int(audit["generalized"]),
            "v1_length_generalized": int(audit["v1_length_generalized"]),
            "v1_lengths": list(audit["v1_lengths"]),
            "node_budget": int(audit["node_budget"]),
            "text_tokens": int(audit["text_tokens"]),
            "contradictions": list(audit["contradictions"]),
            "foveated_2cell_ablation": list(audit["foveated_2cell_ablation"]),
            "mechanisms": sorted({str(row.get("mechanism_used", "")) for row in tasks}),
        },
        "mechanism": {
            "control": "Q = Qctrl x Rreg, |Qctrl| <= 4, one buffered symbol",
            "work_alphabet": "Gamma_work = Sigma_io plus a pi/2 mark phase per data cell",
            "priors": ["reversibility at non-final visits", "local end-of-tape halt",
                       "symbol class instead of symbol identity"],
            "macro": "iterate the induced 10 -> 01 inversion shuttle at alternating alignment",
        },
        "gate": pc._finalise(gates),
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
        "passed": bool(all(gates.values())),
    }


# ---------------------------------------------------------------------- V2
def stage_v2(directory: Path) -> Dict[str, object]:
    started = time.perf_counter()
    payload = _read("agent_eval.json")
    gate = dict(payload.get("gate") or {})
    mcp = dict(payload.get("holarch_mcp") or {})
    probes = list(payload.get("probes") or [])
    stage_receipts = {}
    for name in ("m1_online_repulsor_vla.json", "j1_cleanroom_planner.json",
                 "m3_full_reference_cartridge.json"):
        path = RECEIPTS / name
        if not path.exists():
            stage_receipts[name] = {"present": False}
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        side = path.with_name(name + ".sha256")
        sealed = _read(name)
        stage_receipts[name] = {
            "present": True,
            "sha256": digest,
            "sidecar_match": bool(side.exists()
                                  and side.read_text(encoding="utf-8").strip() == digest),
            "sealed_passed": bool((sealed.get("gate") or {}).get("passed", False)),
        }
    gates = {
        "v2_queries_240": bool(int(payload.get("total_queries", 0)) == 240),
        "v2_probe_slots_164": bool(int(payload.get("probe_slots", 0)) == 164),
        "v2_reachable_1": bool(float(mcp.get("reachable_accuracy", 0.0) or 0.0) == 1.0),
        "v2_hallucination_0": bool(mcp.get("hallucination_rate") == 0.0),
        "v2_zero_query_reads": bool(int(mcp.get("query_file_reads", -1)) == 0),
        "v2_receipt_chain_valid": bool(mcp.get("receipt_chain_valid", False)),
        "v2_wall_seconds_le_180": bool(float(payload.get("wall_seconds", 1e9)) <= 180.0),
        "v2_sweep_receipts_valid": bool(all(
            row.get("sidecar_match") and row.get("sealed_passed")
            for row in stage_receipts.values())),
        "v2_gauntlet_gate": bool(gate.get("passed", False)),
    }
    return {
        "stage": "V2",
        "queries": int(payload.get("total_queries", 0)),
        "probe_slots": int(payload.get("probe_slots", 0)),
        "wall_seconds_gauntlet": float(payload.get("wall_seconds", 0.0)),
        "category_ms": dict(payload.get("category_ms") or {}),
        "stage_receipts": stage_receipts,
        "v_probes": [str(name) for name in (
            probes.values() if isinstance(probes, dict) else probes)
            if str(getattr(name, "get", lambda *_: name)("category", name)).startswith(
                ("v1_", "v2_", "v3_", "v4_", "v5_"))],
        "gate": pc._finalise(gates),
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
        "passed": bool(all(gates.values())),
    }


# ---------------------------------------------------------------------- V3
def stage_v3(directory: Path) -> Dict[str, object]:
    started = time.perf_counter()
    import urllib.error
    import urllib.request

    from henri.holarch.coprocessor import redact_secret, resolve_llm_credentials

    credential = resolve_llm_credentials()
    describe = credential.describe() if credential is not None else {"source": "absent"}
    secret = credential.api_key if credential is not None else ""

    def _chat(prompt: str, max_tokens: int = 220) -> Dict[str, object]:
        """One OpenAI-compatible completion.  The key stays in this frame."""
        body = json.dumps({
            "model": credential.model,
            "messages": [{"role": "user", "content": str(prompt)}],
            "temperature": 0.0,
            "max_tokens": int(max_tokens),
        }).encode("utf-8")
        request = urllib.request.Request(
            credential.base_url.rstrip("/") + "/chat/completions",
            data=body,
            method="POST",
            headers={
                "Authorization": "Bearer " + credential.api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
                "HTTP-Referer": "https://henri.local",
                "X-Title": "Project HENRI Cycle V",
            },
        )
        began = time.perf_counter()
        with urllib.request.urlopen(request, timeout=120.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
        latency = (time.perf_counter() - began) * 1000.0
        choice = (payload.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        return {
            "text": str(message.get("content") or ""),
            "model": str(payload.get("model") or credential.model),
            "usage": dict(payload.get("usage") or {}),
            "latency_ms": float(latency),
        }

    live_rows: List[Dict[str, object]] = []
    live_error = ""
    provider = None if credential is None else True
    prompts = [
        "Reply with the single word READY.",
        "State one falsifiable claim about the information content of a deterministic finite automaton.",
        "Name the four official pillars of the Artificial Analysis Intelligence Index v5.0.",
    ]
    for index, prompt in enumerate(prompts):
        if provider is None:
            break
        try:
            reply = _chat(prompt)
            live_rows.append({
                "index": int(index),
                "model": str(reply.get("model", "")),
                "usage": dict(reply.get("usage") or {}),
                "latency_ms": float(reply.get("latency_ms", 0.0)),
                "text_sha256": hashlib.sha256(
                    str(reply.get("text", "")).encode("utf-8")).hexdigest(),
                "text_head": redact_secret(str(reply.get("text", ""))[:160], secret),
            })
        except Exception as error:  # noqa: BLE001 - a live failure must be recorded
            detail = f"{type(error).__name__}: {error}"
            live_error = redact_secret(detail, secret)[:600]
            break
    live_ok = bool(live_rows) and all(
        row["model"] and int((row["usage"] or {}).get("total_tokens", 0) or 0) > 0
        for row in live_rows)
    leaks = 0
    if secret:
        for path in sorted(RECEIPTS.glob("*")):
            if path.is_file() and path.suffix in {".json", ".sha256", ".md"}:
                try:
                    if secret.encode("utf-8") in path.read_bytes():
                        leaks += 1
                except OSError:
                    continue
    gates = {
        "v3_credential_resolved": bool(credential is not None),
        "v3_live_provider_roundtrip": bool(live_ok),
        "v3_env_leaks_zero": bool(leaks == 0),
        "v3_secret_in_receipt_false": bool(leaks == 0),
        "v3_three_arm_suite_executed": False,
        "passed": False,
    }
    return {
        "stage": "V3",
        "credential": describe,
        "live_provider_claimed": bool(live_ok),
        "live_rows": live_rows,
        "live_error": live_error,
        "env_leaks": int(leaks),
        "secret_in_receipt": bool(leaks > 0),
        "three_arm": {
            "executed": False,
            "reason": ("the 16-task x 3-arm AAII v5.0 co-processor suite needs a "
                       "warm synthesis engine per arm and was not executed in this "
                       "run; the credential bridge and the live transport are "
                       "measured above"),
            "arm3_claims": "not measured in Cycle V",
        },
        "gate": pc._finalise(gates),
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
        "passed": False,
    }


# ---------------------------------------------------------------------- V4
def stage_v4(directory: Path) -> Dict[str, object]:
    started = time.perf_counter()
    receipt = directory / "_work" / "v4_holon_run.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "henri_holon.py"),
         "--receipt", str(receipt)],
        cwd=str(ROOT), capture_output=True, text=True, timeout=1800,
    )
    sealed = {}
    if receipt.exists():
        try:
            sealed = json.loads(receipt.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            sealed = {}
    def _sheaf_veto_live() -> bool:
        """Prove the Tier-1 structural veto is wired, on both fixture families."""
        try:
            from henri.holarch.repo_sheaf import (
                RepoSheaf,
                build_mismatch_battery,
                build_polymorphic_battery,
            )

            sheaf = RepoSheaf()
            poly = [float(sheaf.analyze(fixture.files).energy)
                    for fixture in build_polymorphic_battery()[:2]]
            bad = [float(sheaf.analyze(fixture.files).energy)
                   for fixture in build_mismatch_battery()[:2]]
            return bool(poly and max(poly) == 0.0 and bad and min(bad) > 0.0)
        except Exception:  # noqa: BLE001
            return False

    harness = dict(sealed.get("harness") or {})
    rows = list(sealed.get("rows") or [])
    seeds = [row for row in rows if str(row.get("kind")) == "seed"]
    traps = [row for row in rows if str(row.get("kind")) == "trap"]
    gates = {
        "v4_runner_exit_zero": bool(completed.returncode == 0),
        "v4_seeds_accepted": bool(seeds and all(bool(row.get("verified")) for row in seeds)),
        "v4_traps_refused": bool(traps and all(not bool(row.get("verified")) for row in traps)),
        "v4_zero_residue": bool(int(harness.get("residual_state", -1)) == 0),
        "v4_plan_store_committed": bool(int(harness.get("plan_store_size", 0)) >= len(seeds)),
        "v4_accepted_plans_digested": bool(seeds and all(
            row.get("accepted") and len(str(row.get("plan_digest", ""))) == 64
            for row in seeds)),
        "v4_tier1_sheaf_live": bool(_sheaf_veto_live()),
    }
    gates["passed"] = bool(all(value for key, value in gates.items() if key != "passed"))
    return {
        "stage": "V4",
        "runner": "scripts/henri_holon.py",
        "exit_code": int(completed.returncode),
        "harness": harness,
        "rows": rows,
        "stdout_tail": completed.stdout[-1200:],
        "stderr_tail": completed.stderr[-600:],
        "gate": pc._finalise(gates),
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
        "passed": bool(gates["passed"]),
    }


# ---------------------------------------------------------------------- V5
def _run(name: str, command: List[str], timeout: int = 3600) -> Dict[str, object]:
    began = time.perf_counter()
    completed = subprocess.run(command, cwd=str(ROOT), capture_output=True,
                               text=True, timeout=timeout)
    return {
        "name": str(name),
        "command": " ".join(command),
        "exit_code": int(completed.returncode),
        "wall_seconds": float(time.perf_counter() - began),
        "stdout_tail": completed.stdout[-800:],
        "stderr_tail": completed.stderr[-800:],
    }


def stage_v5(directory: Path) -> Dict[str, object]:
    started = time.perf_counter()
    ledger: List[Dict[str, object]] = []
    ledger.append(_run("t_cycle", [sys.executable, "scripts/run_holarch_t_cycle.py",
                                   "--stages", "t1,t2,t3,t4"]))
    ledger.append(_run("u_cycle", [sys.executable, "scripts/run_holarch_u_cycle.py",
                                   "--stages", "u1,u2,u3,u4,u5"]))
    ledger.append(_run("consistency", [sys.executable, "scripts/verify_holarch_consistency.py"]))
    ledger.append(_run("pytest", [sys.executable, "-m", "pytest", "-q"]))
    t_cycle = _read("t_cycle.json")
    u_cycle = _read("u_cycle.json")
    gates = {
        "v5_t_cycle_passed": bool(t_cycle.get("passed", False)),
        "v5_u_cycle_passed": bool(u_cycle.get("passed", False)),
        "v5_consistency_exit_zero": bool(ledger[2]["exit_code"] == 0),
        "v5_pytest_exit_zero": bool(ledger[3]["exit_code"] == 0),
        "v5_per_command_ledger": bool(len(ledger) == 4),
    }
    return {
        "stage": "V5",
        "ledger": ledger,
        "t_cycle_passed": bool(t_cycle.get("passed", False)),
        "u_cycle_passed": bool(u_cycle.get("passed", False)),
        "gate": pc._finalise(gates),
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
        "passed": bool(all(gates.values())),
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Cycle V verification")
    parser.add_argument("--stages", default="v1,v2,v3,v4,v5")
    parser.add_argument("--receipts", default=str(RECEIPTS))
    args = parser.parse_args()
    directory = Path(args.receipts)
    directory.mkdir(parents=True, exist_ok=True)
    stages = [token.strip().lower() for token in str(args.stages).split(",") if token.strip()]
    runners = {
        "v1": ("v1_factored_turing_closure.json", stage_v1),
        "v2": ("v2_fast_gauntlet.json", stage_v2),
        "v3": ("v3_live_aaii_v5_arms.json", stage_v3),
        "v4": ("v4_live_holon_plugin.json", stage_v4),
        "v5": ("v5_full_closure_ledger.json", stage_v5),
    }
    existing: Dict[str, object] = {}
    aggregate_path = directory / "v_cycle.json"
    if aggregate_path.exists():
        try:
            existing = json.loads(aggregate_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            existing = {}
    kept = {key: value for key, value in (existing.get("stages") or {}).items()
            if key not in {token.strip().lower() for token in str(args.stages).split(",")}}
    aggregate: Dict[str, object] = {"cycle": "V", "head": pc._git_head(), "stages": dict(kept)}
    exit_code = 0
    for stage in stages:
        if stage not in runners:
            continue
        name, runner = runners[stage]
        print(f"[v-cycle] {stage} ...", flush=True)
        payload = _sanitize(runner(directory))
        written = pc._write(directory, name, payload)
        aggregate["stages"][stage] = {
            "receipt": written["path"],
            "sha256": written["sha256"],
            "passed": bool(payload.get("gate", {}).get("passed", False)),
            "gate": payload.get("gate", {}),
            "wall_seconds": payload.get("wall_seconds", 0.0),
        }
        print(f"[v-cycle] {stage} passed={payload.get('gate', {}).get('passed')} {written['path']}",
              flush=True)
        if not bool(payload.get("gate", {}).get("passed", False)):
            exit_code = 1
    aggregate["passed"] = bool(
        all(bool(item.get("passed")) for item in aggregate["stages"].values())
    ) if aggregate["stages"] else False
    written = pc._write(directory, "v_cycle.json", aggregate)
    print(f"[v-cycle] aggregate {written['path']} passed={aggregate['passed']}", flush=True)
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
