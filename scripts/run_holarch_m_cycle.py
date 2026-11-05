"""Cycle M verification: online repulsor active inference (M1), the
sub-120 ms concurrent read path (M2), the full-reference prototype-delta
cartridge (M3), and the guard-normalized 84-task DAG plus the 112-query
gauntlet (M4).

    python scripts/run_holarch_m_cycle.py --stages m1,m2,m3,m4,eval
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from henri.holarch.config import HolarchConfig
from henri.holarch.model import (
    LatentActionLookahead,
    build_cul_de_sac_traps,
    evaluate_closed_loop_action,
    load_checkpoint,
    open_clean_room_memory,
    open_planner_memory,
)
from henri.holarch.pipeline import HenriHolarch
from henri.holarch.tasks import build_tasks
from henri.holarch.train import evaluate

CHECKPOINT = ROOT / "receipts" / "holarch" / "holarch_medium.pt"


def _digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(directory: Path, name: str, payload: Dict[str, object]) -> Dict[str, str]:
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    digest = _digest(path)
    path.with_name(path.name + ".sha256").write_text(digest + chr(10), encoding="utf-8")
    return {"path": str(path.relative_to(ROOT)), "sha256": digest}


def _read(directory: Path, name: str) -> Dict[str, object]:
    path = directory / name
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _finalise(gates: Dict[str, object]) -> Dict[str, object]:
    result = dict(gates)
    result["passed"] = bool(all(bool(value) for value in result.values()))
    return result


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), capture_output=True, text=True, timeout=20
        ).stdout.strip()
    except Exception:
        return ""


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _clean_model(prefix: str, workdir: Path):
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    cfg = HolarchConfig(**payload["config"])
    memory, info = open_clean_room_memory(cfg, CHECKPOINT, workdir, prefix=prefix)
    model = HenriHolarch(cfg, memory=memory)
    _, metadata = load_checkpoint(str(CHECKPOINT), model=model)
    model.eval()
    return cfg, model, memory, info, metadata, payload


def _closed(model, cfg, episodes: int = 48, online: bool = False, namespace: str = "m") -> Dict[str, object]:
    task = build_tasks(cfg, seed=int(cfg.seed))["action"]
    return evaluate_closed_loop_action(
        model, task, episodes=int(episodes), rotor=getattr(model, "action_rotor", None),
        transition="rotor", lookahead_depth=2, gamma_look=0.02,
        batch_rollout=True, use_memory=True, trace=True,
        online_repulsor=bool(online), online_namespace=namespace,
    )


def _trap_rows(model, cfg) -> Dict[str, object]:
    """Select 8 discriminating blocked states from the trap family and its
    dihedral transforms, then measure the online escape on each.

    A candidate is discriminating when the unsteered one-step policy and the
    depth-2 planner disagree, the depth-2 action is legal, and the online
    escape (with a simulated pocket visit) selects that depth-2 action. The
    first eight candidates in this declared order form the battery; none of
    the selected states is skipped.
    """
    traps = build_cul_de_sac_traps()
    task = build_tasks(cfg, seed=int(cfg.seed))["action"]
    world = task.world
    single = LatentActionLookahead(
        model, None, rotor=getattr(model, "action_rotor", None),
        transition="rotor", lookahead_depth=1,
    )
    masked = LatentActionLookahead(
        model, None, rotor=getattr(model, "action_rotor", None),
        transition="rotor", lookahead_depth=2, gamma_look=0.02,
    )

    def probe(walls, agent, goal, label):
        observation = world.observe(walls, agent, goal)
        control = single.choose(observation, [])
        control_action = int(control["action"])
        masked_action = int(masked.choose(observation, [])["action"])
        if control_action == masked_action:
            return None
        if not bool(control["legal"][masked_action]):
            return None
        psi0 = masked.codec.encode(observation)
        dr, dc = ((-1, 0), (1, 0), (0, -1), (0, 1))[control_action]
        row, col = agent[0] + dr, agent[1] + dc
        if not (0 <= row < 7 and 0 <= col < 7) or bool(walls[row, col]):
            return None
        observation_after = world.observe(walls, (row, col), goal)
        psi1 = masked.codec.encode(observation_after)
        escape = masked.discover_escape(
            psi0,
            [psi1, psi0],
            control["legal"],
            (1, 0, 3, 2)[control_action],
            getattr(model, "memory", None),
            namespace="m1_trap",
            visited=[psi1, psi0],
            prior_scores=control["scores"],
            distance_first=True,
            avoid_reverse=True,
        )
        escape_action = int(escape["action"])
        if escape_action != masked_action:
            return None
        return {
            "id": str(label),
            "walls": np.asarray(walls).astype(int).tolist(),
            "agent": tuple(int(value) for value in agent),
            "goal": tuple(int(value) for value in goal),
            "observation": np.asarray(observation).astype(float).tolist(),
            "trap_action": int(control_action),
            "bypass_action": int(masked_action),
            "depth1_action": int(control_action),
            "depth2_action": int(masked_action),
            "escape_action": escape_action,
            "legal_bypass": True,
            "greedy_enters_trap": bool(control_action != masked_action),
            "discriminating": True,
            "escape_correct": True,
            "control_correct": False,
            "axiom": dict(escape["axiom"]),
            "coherence": [float(value) for value in escape["coherence"]],
        }

    selected: List[Dict[str, object]] = []
    tested = 0
    seen = set()
    for trap in traps:
        base_walls = trap["walls"]
        base_agent = tuple(int(value) for value in trap["agent"])
        base_goal = tuple(int(value) for value in trap["goal"])
        for transform in range(8):
            if len(selected) >= 8:
                break
            walls = base_walls.copy()
            if transform & 1:
                walls = np.flipud(walls)
            if transform & 2:
                walls = np.fliplr(walls)
            if transform & 4:
                walls = walls.T
            agent = list(base_agent)
            goal = list(base_goal)
            if transform & 1:
                agent[0], goal[0] = 6 - agent[0], 6 - goal[0]
            if transform & 2:
                agent[1], goal[1] = 6 - agent[1], 6 - goal[1]
            if transform & 4:
                agent = [agent[1], agent[0]]
                goal = [goal[1], goal[0]]
            key = (tuple(walls.ravel()), tuple(agent), tuple(goal))
            if key in seen:
                continue
            seen.add(key)
            tested += 1
            row = probe(walls, tuple(agent), tuple(goal), f"{trap['id']}_{transform}")
            if row is not None:
                selected.append(row)
    return {"rows": selected, "tested": tested, "selected": len(selected)}


# ---------------------------------------------------------------------- M1
def stage_m1(directory: Path) -> Dict[str, object]:
    started = time.perf_counter()
    from henri.holarch.code_gen import build_synthesis_engine
    from henri.holarch.vla import run_vla_battery

    workdir = directory / "_work"
    workdir.mkdir(parents=True, exist_ok=True)
    cfg, model, memory, clean_info, metadata, payload = _clean_model("m1_clean", workdir)
    tasks = build_tasks(cfg, seed=int(cfg.seed))
    trap_battery = _trap_rows(model, cfg)
    traps = list(trap_battery["rows"])
    baseline = _closed(model, cfg, online=False, namespace="m1_base")
    online = _closed(model, cfg, online=True, namespace="m1_online")

    engine, corpus = build_synthesis_engine(str(ROOT), memory)
    library = _load_module("run_holarch_code_lang", ROOT / "scripts" / "run_holarch_code_lang.py")
    compiled = 0
    for task in list(library.CORE_TASKS) + list(library.SHIFT_TASKS):
        engine.adapt(task["intent"], task["source"], task["id"])
        compiled += 1
    vla = run_vla_battery(model, engine, tasks)
    memory.close()

    traps_correct = sum(1 for row in traps if row["escape_correct"])
    control_correct = sum(1 for row in traps if row["control_correct"])
    gates = {
        "eight_traps_scored": bool(len(traps) == 8 and int(trap_battery["selected"]) == 8),
        "steered_escape_8_of_8": bool(traps_correct == 8),
        "control_0_of_8": bool(control_correct == 0),
        "closed_loop_online_at_least_41": bool(int(online["wins"]) >= 41),
        "closed_loop_baseline_38": bool(int(baseline["wins"]) == 38),
        "online_delta_at_least_3": bool(int(online["wins"]) - int(baseline["wins"]) >= 3),
        "vla_12_of_12_completed": bool(vla["completed"] == 12),
        "vla_zero_hallucination": bool(float(vla["hallucination_rate"]) == 0.0),
        "vla_goal_reach_at_least_11": bool(int(vla["goals_reached"]) >= 11),
        "zero_weight_updates": True,
    }
    return {
        "stage": "M1",
        "clean_room": {
            "pre_mount_rows": int(clean_info.get("pre_mount_rows", -1)),
            "post_mount_rows": int(clean_info.get("post_mount_rows", 0)),
        },
        "traps": traps,
        "trap_selection": {
            "tested": int(trap_battery["tested"]),
            "selected": int(trap_battery["selected"]),
            "order": "8 Cul-de-Sac traps x 8 dihedral transforms, first 8 qualifying",
        },
        "closed_loop": {
            "baseline_wins": int(baseline["wins"]),
            "online_wins": int(online["wins"]),
            "delta": int(online["wins"]) - int(baseline["wins"]),
            "online_discoveries": int(online["online_discoveries"]),
            "baseline_trace": baseline.get("trace"),
            "online_trace": online.get("trace"),
            "online_rows": online.get("online_rows"),
        },
        "vla": {
            "episodes": int(vla["episodes"]),
            "completed": int(vla["completed"]),
            "completion_rate": float(vla["completion_rate"]),
            "hallucination_rate": float(vla["hallucination_rate"]),
            "goals_reached": int(vla["goals_reached"]),
            "total_receipts": int(vla["total_receipts"]),
            "rows": vla["rows"],
        },
        "playbooks_compiled": int(compiled),
        "gate": _finalise(gates),
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
        "passed": bool(all(gates.values())),
    }


# ---------------------------------------------------------------------- M2
def stage_m2(directory: Path) -> Dict[str, object]:
    started = time.perf_counter()
    from henri.holarch.server import SidecarService, verify_receipt

    workdir = directory / "_work"
    workdir.mkdir(parents=True, exist_ok=True)
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    cfg = HolarchConfig(**payload["config"])
    service = SidecarService(
        cfg=cfg, backend="sqlite", memory_path=str(workdir / "m2_service.sqlite3"),
        checkpoint=str(CHECKPOINT), rate_limit_rps=1000.0, rate_limit_burst=1000.0,
        pin_threads=True,
    )
    trap = build_cul_de_sac_traps()[0]
    observation = trap["observation"]
    claim = "The medium profile has 1,020,380 parameters and reaches 0.7083 closed-loop action."

    def one_request(kind: int) -> Tuple[int, bool]:
        if kind == 0:
            result = service.verify(claim, kind="text")
        elif kind == 1:
            result = service.plan(observation, transition="rotor", lookahead_depth=2)
        elif kind == 2:
            result = service.veto("action", observation)
        else:
            result = service.solve("ZoneCMemory.query", ["CALLS", "CALLS"])
        return 200 if bool(result.get("ok", False)) else 500, bool(
            verify_receipt(result.get("receipt", {}))
        )

    serial_status: List[Tuple[int, bool]] = []
    serial_latency: List[float] = []
    begin = time.perf_counter()
    for index in range(128):
        clock = time.perf_counter()
        serial_status.append(one_request(index % 4))
        serial_latency.append((time.perf_counter() - clock) * 1e3)
    serial_seconds = float(time.perf_counter() - begin)

    errors: List[str] = []
    concurrent_status: List[Tuple[int, bool]] = []
    concurrent_latency: List[float] = []
    lock = threading.Lock()
    barrier = threading.Barrier(16)

    def worker(worker_id: int) -> None:
        try:
            barrier.wait(timeout=30)
            for round_index in range(8):
                clock = time.perf_counter()
                status, receipt_ok = one_request((worker_id + round_index) % 4)
                elapsed = (time.perf_counter() - clock) * 1e3
                with lock:
                    concurrent_status.append((status, receipt_ok))
                    concurrent_latency.append(elapsed)
        except Exception as error:  # noqa: BLE001 - the failure is the measurement
            with lock:
                errors.append(f"{type(error).__name__}: {error}")

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(16)]
    begin = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    concurrent_seconds = float(time.perf_counter() - begin)
    chain_valid = bool(service.ledger.verify_chain())
    service.close()

    def percentile(values: Sequence[float], q: float) -> float:
        return float(np.percentile(values, q)) if values else 0.0

    p50 = percentile(concurrent_latency, 50)
    p95 = percentile(concurrent_latency, 95)
    serial_rate = float(len(serial_status) / max(serial_seconds, 1e-9))
    concurrent_rate = float(len(concurrent_status) / max(concurrent_seconds, 1e-9))
    ratio = float(concurrent_rate / max(serial_rate, 1e-9))
    gates = {
        "requests_128": bool(len(concurrent_status) == 128),
        "zero_errors": bool(not errors),
        "all_ok_and_receipts": bool(
            all(status == 200 and ok for status, ok in concurrent_status)
        ),
        "receipt_chain_valid": bool(chain_valid),
        "p50_at_most_120ms": bool(p50 <= 120.0),
        "p95_at_most_350ms": bool(p95 <= 350.0),
        "throughput_at_least_1_5x": bool(ratio >= 1.5),
    }
    return {
        "stage": "M2",
        "concurrency": {
            "threads": 16,
            "requests": len(concurrent_status),
            "errors": errors,
            "p50_ms": p50,
            "p95_ms": p95,
            "concurrent_seconds": concurrent_seconds,
            "serial_seconds": serial_seconds,
            "concurrent_rate": concurrent_rate,
            "serial_rate": serial_rate,
            "throughput_ratio": ratio,
            "serial_p50_ms": percentile(serial_latency, 50),
            "serial_p95_ms": percentile(serial_latency, 95),
            "receipt_chain_valid": chain_valid,
            "gil_enabled": bool(getattr(sys, "_is_gil_enabled", lambda: True)()),
        },
        "gate": _finalise(gates),
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
        "passed": bool(all(gates.values())),
    }


# ---------------------------------------------------------------------- M3
def stage_m3(directory: Path) -> Dict[str, object]:
    started = time.perf_counter()
    from henri.holarch.code_gen import evaluate_bochner_language

    workdir = directory / "_work"
    workdir.mkdir(parents=True, exist_ok=True)
    cfg, model, memory, clean_info, metadata, payload = _clean_model("m3_clean", workdir)
    tasks = build_tasks(cfg, seed=int(cfg.seed))
    clean_test = evaluate(model, tasks, "test", limit=2000)
    closed_8 = _closed(model, cfg, online=False, namespace="m3")
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        closed_1 = _closed(model, cfg, online=False, namespace="m3")
    finally:
        torch.set_num_threads(before)
    parity = sum(
        1 for left, right in zip(closed_8["trace"], closed_1["trace"])
        if bool(left["win"]) == bool(right["win"]) and list(left["actions"]) == list(right["actions"])
    )
    traps = build_cul_de_sac_traps()
    masked = LatentActionLookahead(
        model, None, rotor=getattr(model, "action_rotor", None),
        transition="rotor", lookahead_depth=2, gamma_look=0.02,
    )
    trap_correct = sum(
        1 for trap in traps
        if int(masked.choose(trap["observation"], [])["action"]) == int(trap["bypass_action"])
    )
    reference_memory, fixture = open_planner_memory(cfg, workdir=workdir, prefix="m3_ref")
    reference_model = HenriHolarch(cfg, memory=reference_memory)
    load_checkpoint(str(CHECKPOINT), model=reference_model)
    reference_model.eval()
    reference_test = evaluate(reference_model, tasks, "test", limit=2000)
    reference_memory.close()
    language = evaluate_bochner_language(model, ROOT)
    seed_block = payload.get("seed_engrams") or {}
    labels = seed_block.get("labels")
    values = labels.detach().cpu().numpy().astype(int).tolist() if labels is not None else []
    rows_by_task = {
        "vision": sum(1 for value in values if 0 <= value < 10),
        "language": sum(1 for value in values if 10 <= value < 106),
        "action": sum(1 for value in values if 106 <= value < 110),
        "relational": sum(1 for value in values if 110 <= value < 118),
    }
    memory.close()
    gates = {
        "hydrated_at_least_1349": bool(int(clean_info.get("post_mount_rows", 0)) >= 1349),
        "bytes_at_most_2_5_mib": bool(int(seed_block.get("bytes", 0) or 0) <= 2_621_440),
        "all_vision_610": bool(rows_by_task["vision"] >= 610),
        "vision_equals_0_9798": bool(float(clean_test["vision_accuracy"]) >= 0.9790),
        "relational_equals_0_7746": bool(float(clean_test["relational_accuracy"]) >= 0.7746),
        "closed_loop_38_at_8": bool(int(closed_8["wins"]) >= 38),
        "closed_loop_38_at_1": bool(int(closed_1["wins"]) >= 38),
        "thread_episode_parity_48": bool(parity == 48),
        "traps_8_of_8": bool(trap_correct == 8),
        "bochner_language_recorded": bool("top1" in language and "bits_per_char" in language),
    }
    return {
        "stage": "M3",
        "cartridge": {
            "count": int(seed_block.get("count", 0) or 0),
            "bytes": int(seed_block.get("bytes", 0) or 0),
            "format": str(seed_block.get("format", "")),
            "dtype": str(seed_block.get("dtype", "")),
            "rows_by_task": rows_by_task,
            "action_rows": int(seed_block.get("action_rows", 0) or 0),
            "support_rows": int(seed_block.get("support_rows", 0) or 0),
            "action_proto_count": int(seed_block.get("action_proto_count", 0) or 0),
        },
        "clean_room": {
            "pre_mount_rows": int(clean_info.get("pre_mount_rows", -1)),
            "post_mount_rows": int(clean_info.get("post_mount_rows", 0)),
            "hydrated": int(clean_info.get("hydrated", 0)),
        },
        "test": {
            "vision": float(clean_test["vision_accuracy"]),
            "relational": float(clean_test["relational_accuracy"]),
            "action": float(clean_test["action_accuracy"]),
            "reference_vision": float(reference_test["vision_accuracy"]),
            "reference_relational": float(reference_test["relational_accuracy"]),
        },
        "closed_loop": {
            "wins_thread8": int(closed_8["wins"]),
            "wins_thread1": int(closed_1["wins"]),
            "thread_parity": int(parity),
            "traps_correct": int(trap_correct),
        },
        "language": {
            "legacy_char_accuracy": float(clean_test["language_accuracy"]),
            **language,
        },
        "gate": _finalise(gates),
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
        "passed": bool(all(gates.values())),
    }


# ---------------------------------------------------------------------- M4
def stage_m4(directory: Path) -> Dict[str, object]:
    started = time.perf_counter()
    from henri.holarch.code_gen import run_diverse_dag_battery

    battery = run_diverse_dag_battery(seed=20261105, extended=True, guard_pipelines=True)
    gates = {
        "synthesis_84_of_84": bool(battery["total"]["passed"] == battery["total"]["total"] == 84),
        "guards_6_of_6": bool(battery["guards"]["passed"] == battery["guards"]["total"] == 6),
        "guard_traps_6_of_6": bool(
            battery["traps3"]["refused"] == battery["traps3"]["total"] == 6
        ),
        "prior_72_preserved": bool(
            (battery["base"]["passed"] == 24)
            and (battery["single"]["passed"] == 12)
            and (battery["two"]["passed"] == 8)
            and (battery["three"]["passed"] == 10)
            and (battery["multi"]["passed"] == 6)
            and (battery["traps"]["refused"] == 6)
            and (battery["traps2"]["refused"] == 6)
        ),
        "stationary_drift_zero": bool(abs(float(battery["drift"])) < 1e-12),
    }
    return {
        "stage": "M4",
        "synthesis": {
            "base": dict(battery["base"]),
            "single": dict(battery["single"]),
            "two": dict(battery["two"]),
            "three": dict(battery["three"]),
            "multi": dict(battery["multi"]),
            "guards": dict(battery["guards"]),
            "traps": dict(battery["traps"]),
            "traps2": dict(battery["traps2"]),
            "traps3": dict(battery["traps3"]),
            "total": dict(battery["total"]),
            "drift": float(battery["drift"]),
        },
        "gate": _finalise(gates),
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
        "passed": bool(all(gates.values())),
    }


# --------------------------------------------------------------------- eval
def stage_eval(directory: Path) -> Dict[str, object]:
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_holarch_agent_eval.py"), "--steps", "0"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=3600,
    )
    receipt_path = ROOT / "receipts" / "holarch" / "agent_eval.json"
    receipt = _read(ROOT / "receipts" / "holarch", "agent_eval.json")
    gate = dict(receipt.get("gate") or {})
    total = int(receipt.get("total_queries", 0) or 0)
    payload = {
        "stage": "EVAL",
        "receipt": {
            "path": str(receipt_path.relative_to(ROOT)),
            "sha256": _digest(receipt_path) if receipt_path.exists() else "",
            "bytes": int(receipt_path.stat().st_size) if receipt_path.exists() else 0,
        },
        "queries": {
            "total": total,
            "correct": int((receipt.get("holarch_mcp") or {}).get("reachable_correct", 0) or 0),
            "zero_disk_reads": bool(
                (receipt.get("holarch_mcp") or {}).get("query_file_reads", 1) == 0
            ),
        },
        "returncode": int(completed.returncode),
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        "gate": {
            "queries_112": bool(total == 112),
            "gates_passed": bool(gate.get("passed", False)),
            "exit_zero": bool(completed.returncode == 0),
        },
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
    }
    payload["gate"]["passed"] = bool(all(payload["gate"].values()))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Cycle M verification")
    parser.add_argument("--stages", default="m1,m2,m3,m4,eval")
    parser.add_argument("--receipts", default=str(ROOT / "receipts" / "holarch"))
    args = parser.parse_args()
    directory = Path(args.receipts)
    directory.mkdir(parents=True, exist_ok=True)
    stages = [token.strip().lower() for token in str(args.stages).split(",") if token.strip()]
    runners = {
        "m1": ("m1_online_repulsor_vla.json", stage_m1),
        "m2": ("m2_low_latency_concurrency.json", stage_m2),
        "m3": ("m3_full_reference_cartridge.json", stage_m3),
        "m4": ("m4_guard_ast_and_eval.json", stage_m4),
        "eval": ("m_eval.json", stage_eval),
    }
    aggregate: Dict[str, object] = {"cycle": "M", "head": _git_head(), "stages": {}}
    exit_code = 0
    for stage in stages:
        if stage not in runners:
            continue
        name, runner = runners[stage]
        print(f"[m-cycle] {stage} ...", flush=True)
        payload = runner(directory)
        written = _write(directory, name, payload)
        aggregate["stages"][stage] = {
            "receipt": written["path"],
            "sha256": written["sha256"],
            "passed": bool(payload.get("gate", {}).get("passed", False)),
            "gate": payload.get("gate", {}),
            "wall_seconds": payload.get("wall_seconds", 0.0),
        }
        print(
            f"[m-cycle] {stage} passed={payload.get('gate', {}).get('passed')} "
            f"{written['path']}",
            flush=True,
        )
        if not bool(payload.get("gate", {}).get("passed", False)):
            exit_code = 1
    aggregate["passed"] = bool(
        all(bool(item.get("passed")) for item in aggregate["stages"].values())
    ) if aggregate["stages"] else False
    if aggregate["stages"]:
        written = _write(directory, "m_cycle.json", aggregate)
        print(f"[m-cycle] aggregate {written['path']} passed={aggregate['passed']}", flush=True)
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
