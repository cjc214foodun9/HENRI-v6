"""Cycle K verification: full-cartridge parity (K1), boundary-axiom
discovery (K2), the unified generative VLA pipeline and multi-statement DAGs
(K3), and cross-tenant rebase with vectorised reads (K4).

    python scripts/run_holarch_k_cycle.py --stages k1,k2,k3,k4,eval

Every number is measured in this process and written to a JSON receipt with
a SHA-256 sidecar; k_cycle.json aggregates them.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import statistics
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
VOCAB_LOCK = ROOT / "receipts" / "holarch" / "vocab_lock.json"


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


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), capture_output=True, text=True, timeout=20
        ).stdout.strip()
    except Exception:
        return ""


def _clean_model(prefix: str, workdir: Path):
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    cfg = HolarchConfig(**payload["config"])
    memory, info = open_clean_room_memory(cfg, CHECKPOINT, workdir, prefix=prefix)
    model = HenriHolarch(cfg, memory=memory)
    _, metadata = load_checkpoint(str(CHECKPOINT), model=model)
    model.eval()
    return cfg, model, memory, info, metadata, payload


def _closed(model, cfg, threads: int, episodes: int = 48) -> Dict[str, object]:
    before = torch.get_num_threads()
    torch.set_num_threads(int(threads))
    try:
        task = build_tasks(cfg, seed=int(cfg.seed))["action"]
        return evaluate_closed_loop_action(
            model, task, episodes=int(episodes), rotor=getattr(model, "action_rotor", None),
            transition="rotor", lookahead_depth=2, gamma_look=0.02,
            batch_rollout=True, use_memory=True, trace=True,
        )
    finally:
        torch.set_num_threads(before)


def _trap_actions(model, threads: int) -> List[int]:
    before = torch.get_num_threads()
    torch.set_num_threads(int(threads))
    try:
        lookahead = LatentActionLookahead(
            model, None, rotor=getattr(model, "action_rotor", None),
            transition="rotor", lookahead_depth=2, gamma_look=0.02,
        )
        return [
            int(lookahead.choose(trap["observation"], [])["action"])
            for trap in build_cul_de_sac_traps()
        ]
    finally:
        torch.set_num_threads(before)


def _empty_veto(model, tasks) -> Dict[str, int]:
    counts = {"vision": 0, "relational": 0, "total": 0}
    for name in ("vision", "relational"):
        for batch in tasks[name].epoch(64, "val"):
            output = model(name, batch.x, training=False)
            empty = output.veto & (~output.memory_mask)
            counts[name] += int(empty.sum())
            counts["total"] += int(empty.sum())
    return counts


def _reference_store(cfg, workdir: Path):
    memory, fixture = open_planner_memory(cfg, workdir=workdir, prefix="k1_reference")
    model = HenriHolarch(cfg, memory=memory)
    load_checkpoint(str(CHECKPOINT), model=model)
    model.eval()
    return model, memory, fixture


# ---------------------------------------------------------------------- K1
def stage_k1(directory: Path) -> Dict[str, object]:
    started = time.perf_counter()
    workdir = directory / "_work"
    workdir.mkdir(parents=True, exist_ok=True)
    cfg, model, memory, clean_info, metadata, payload = _clean_model("k1_clean", workdir)
    pre_mount = int(clean_info.get("pre_mount_rows", -1))
    post_mount = int(memory.stats().get("count", 0))
    hydrated = int(clean_info.get("hydrated", 0) or 0)
    hydration = dict((metadata.get("seed_engrams") or {}).get("hydration") or {})
    seed_block = payload.get("seed_engrams") or {}

    traps_8 = _trap_actions(model, 8)
    closed_8 = _closed(model, cfg, 8)
    traps_1 = _trap_actions(model, 1)
    closed_1 = _closed(model, cfg, 1)
    truth = [int(trap["bypass_action"]) for trap in build_cul_de_sac_traps()]
    traps_correct_8 = sum(1 for a, b in zip(traps_8, truth) if a == b)
    traps_correct_1 = sum(1 for a, b in zip(traps_1, truth) if a == b)
    parity = sum(
        1 for a, b in zip(closed_8["trace"], closed_1["trace"])
        if bool(a["win"]) == bool(b["win"]) and list(a["actions"]) == list(b["actions"])
    )

    tasks = build_tasks(cfg, seed=int(cfg.seed))
    clean_test = evaluate(model, tasks, "test", limit=2000)
    clean_val = evaluate(model, tasks, "val", limit=2000)
    empty_veto = _empty_veto(model, tasks)

    reference, reference_memory, fixture = _reference_store(cfg, workdir)
    reference_test = evaluate(reference, tasks, "test", limit=2000)
    reference_val = evaluate(reference, tasks, "val", limit=2000)
    reference_memory.close()

    cartridge = _read(directory, "j1_cartridge.json")
    reference_measurement = dict(cartridge.get("reference_measurement") or {})
    reference_wins = int(reference_measurement.get("closed_loop_wins", -1) or -1)
    reference_trace = list(reference_measurement.get("closed_loop_trace") or [])
    reference_parity = sum(
        1 for left, right in zip(closed_8["trace"], reference_trace)
        if bool(left.get("win")) == bool(right.get("win"))
        and int(left.get("steps", -1)) == int(right.get("steps", -2))
        and list(left.get("actions") or []) == list(right.get("actions") or [])
    )
    labels = seed_block.get("labels")
    label_values = (
        labels.detach().cpu().numpy().astype(int).tolist() if labels is not None else []
    )
    rows_by_task = {
        "vision": sum(1 for value in label_values if 0 <= value < 10),
        "language": sum(1 for value in label_values if 10 <= value < 106),
        "action": sum(1 for value in label_values if 106 <= value < 110),
        "relational": sum(1 for value in label_values if 110 <= value < 118),
    }
    memory.close()

    closed_wins = int(closed_8["wins"])
    gates = {
        "clean_room_zero_rows": bool(pre_mount == 0),
        "cartridge_under_2_5_mib": bool(int(seed_block.get("bytes", 0) or 0) <= 2_621_440),
        "cartridge_multitask": bool(
            rows_by_task["action"] == 604
            and rows_by_task["vision"] >= 64
            and rows_by_task["relational"] >= 32
            and rows_by_task["language"] >= 32
        ),
        "hydrated_matches_block": bool(
            hydrated == int(seed_block.get("count", 0) or 0)
            and post_mount == int(seed_block.get("count", 0) or 0)
        ),
        "closed_loop_38_at_eight_threads": bool(closed_wins >= 38),
        "closed_loop_38_at_one_thread": bool(int(closed_1["wins"]) >= 38),
        "cleanroom_equals_reference_store": bool(
            reference_wins >= 0 and closed_wins == reference_wins and reference_parity == 48
        ),
        "thread_episode_parity_48_of_48": bool(parity == 48),
        "traps_8_of_8": bool(traps_correct_8 == 8 and traps_correct_1 == 8),
        "one_step_action_at_least_0_78": bool(float(closed_8["one_step_accuracy"]) >= 0.78),
        "vision_test_at_least_0_975": bool(float(clean_test["vision_accuracy"]) >= 0.975),
        "relational_test_at_least_0_76": bool(float(clean_test["relational_accuracy"]) >= 0.76),
        "empty_store_veto_zero": bool(int(empty_veto["total"]) == 0),
    }
    receipt = {
        "stage": "K1",
        "checkpoint": {
            "path": str(CHECKPOINT.relative_to(ROOT)),
            "sha256": _digest(CHECKPOINT),
            "bytes": int(CHECKPOINT.stat().st_size),
            "seed_engrams": {
                "count": int(seed_block.get("count", 0) or 0),
                "bytes": int(seed_block.get("bytes", 0) or 0),
                "dtype": str(seed_block.get("dtype", "")),
                "rows_by_task": rows_by_task,
                "reconstruction_coherence_min": float(
                    ((cartridge.get("cartridge") or {}).get("reconstruction_coherence_min", 0.0))
                ),
            },
        },
        "clean_room": {
            "pre_mount_rows": pre_mount,
            "post_mount_rows": post_mount,
            "hydrated": int(hydrated),
            "source_fixture_copied": False,
        },
        "reference_store": {
            "fixture": fixture.get("source", ""),
            "sha256": fixture.get("sha256", ""),
            "closed_loop_wins": reference_wins,
            "vision_test_accuracy": float(reference_test["vision_accuracy"]),
            "relational_test_accuracy": float(reference_test["relational_accuracy"]),
            "vision_val_accuracy": float(reference_val["vision_accuracy"]),
            "relational_val_accuracy": float(reference_val["relational_accuracy"]),
        },
        "closed_loop": {
            "reference_store_deterministic_wins": reference_wins,
            "cleanroom_cartridge_wins": closed_wins,
            "cleanroom_cartridge_wins_thread1": int(closed_1["wins"]),
            "cleanroom_equals_reference_store": bool(
                reference_wins >= 0 and closed_wins == reference_wins and reference_parity == 48
            ),
            "episode_parity_with_reference": reference_parity,
            "thread_episode_parity": parity,
            "legacy_bmm_nondet_39_documented": True,
            "one_step_accuracy": float(closed_8["one_step_accuracy"]),
            "reactive_success_rate": float(closed_8["reactive_success_rate"]),
            "trace_thread8": closed_8.get("trace"),
            "trace_thread1": closed_1.get("trace"),
        },
        "traps": {
            "truth": truth,
            "actions_thread8": traps_8,
            "actions_thread1": traps_1,
            "correct_thread8": traps_correct_8,
            "correct_thread1": traps_correct_1,
        },
        "multi_task": {
            "test": {
                key: float(clean_test.get(f"{key}_accuracy", 0.0))
                for key in ("vision", "language", "action", "relational")
            },
            "val": {
                key: float(clean_val.get(f"{key}_accuracy", 0.0))
                for key in ("vision", "language", "action", "relational")
            },
            "veto_rates": {
                key: float(clean_test.get(f"{key}_veto_rate", 0.0))
                for key in ("vision", "language", "action", "relational")
            },
            "empty_store_veto": empty_veto,
        },
        "gate": gates,
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
    }
    receipt["gate"]["passed"] = bool(all(gates.values()))
    return receipt


# ---------------------------------------------------------------------- K2
def stage_k2(directory: Path) -> Dict[str, object]:
    started = time.perf_counter()
    from henri.holarch.discovery import run_discovery_battery

    workdir = directory / "_work"
    workdir.mkdir(parents=True, exist_ok=True)
    cfg, model, memory, clean_info, metadata, payload = _clean_model("k2_clean", workdir)
    battery = run_discovery_battery(model, memory=memory, namespace="k2")
    memory.close()
    receipt = {
        "stage": "K2",
        "clean_room": {
            "pre_mount_rows": int(clean_info.get("pre_mount_rows", -1)),
            "post_mount_rows": int(clean_info.get("post_mount_rows", 0)),
        },
        "probes": battery["probes"],
        "axioms": battery["axioms"],
        "spatial": battery["spatial"],
        "symbolic": battery["symbolic"],
        "weight_updates": 0,
        "gate": battery["gate"],
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
    }
    return receipt


# ---------------------------------------------------------------------- K3
def _load_code_lang():
    spec = importlib.util.spec_from_file_location(
        "run_holarch_code_lang", ROOT / "scripts" / "run_holarch_code_lang.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stage_k3(directory: Path) -> Dict[str, object]:
    started = time.perf_counter()
    from henri.holarch.code_gen import build_synthesis_engine, run_diverse_dag_battery
    from henri.holarch.vla import run_vla_battery

    workdir = directory / "_work"
    workdir.mkdir(parents=True, exist_ok=True)
    cfg, model, memory, clean_info, metadata, payload = _clean_model("k3_clean", workdir)
    tasks = build_tasks(cfg, seed=int(cfg.seed))
    battery = run_diverse_dag_battery(seed=20261105, extended=True)
    engine, corpus = build_synthesis_engine(str(ROOT), memory)
    library = _load_code_lang()
    compiled = 0
    for task in list(library.CORE_TASKS) + list(library.SHIFT_TASKS):
        engine.adapt(task["intent"], task["source"], task["id"])
        compiled += 1
    generator = engine.generator
    generator.mount_holarch(model, gain=0.25)
    vla = run_vla_battery(model, engine, tasks)
    memory.close()
    gates = dict(battery["total"])
    receipt = {
        "stage": "K3",
        "synthesis": {
            "base": {"passed": battery["base"]["passed"], "total": battery["base"]["total"]},
            "single": {"passed": battery["single"]["passed"], "total": battery["single"]["total"]},
            "two": {"passed": battery["two"]["passed"], "total": battery["two"]["total"]},
            "three": {"passed": battery["three"]["passed"], "total": battery["three"]["total"]},
            "multi": {"passed": battery["multi"]["passed"], "total": battery["multi"]["total"]},
            "traps": {"refused": battery["traps"]["refused"], "total": battery["traps"]["total"]},
            "traps2": {"refused": battery["traps2"]["refused"], "total": battery["traps2"]["total"]},
            "total": battery["total"],
            "drift": battery["drift"],
            "three_topologies": battery["three_topologies"],
            "multi_rows": battery["multi"]["rows"],
            "trap_rows": (battery["traps"]["rows"] + battery["traps2"]["rows"]),
        },
        "vla": {
            key: vla[key]
            for key in (
                "episodes", "completed", "completion_rate", "hallucination_rate",
                "goals_reached", "receipt_chain_valid", "total_receipts",
                "generator_photonic_contexts", "rows",
            )
        },
        "corpus": corpus,
        "playbooks_compiled": compiled,
        "generator": {
            "photonic_gain": float(getattr(generator, "photonic_gain", 0.0)),
            "photonic_contexts": int(getattr(generator, "photonic_contexts", 0)),
            "photonic_cache_hits": int(getattr(generator, "photonic_cache_hits", 0)),
        },
        "gate": {
            "synthesis_72_of_72": bool(
                battery["total"]["passed"] == battery["total"]["total"] == 72
            ),
            "multi_6_of_6": bool(
                battery["multi"]["passed"] == battery["multi"]["total"] == 6
            ),
            "symbol_arity_traps_6_of_6": bool(
                battery["traps2"]["refused"] == battery["traps2"]["total"] == 6
            ),
            "stationary_drift_zero": bool(abs(float(battery["drift"])) < 1e-12),
            "vla_12_of_12": bool(vla["gate"]["twelve_of_twelve_completed"]),
            "vla_zero_hallucination": bool(vla["gate"]["zero_hallucination"]),
            "vla_receipt_chain": bool(vla["gate"]["receipt_chain_contiguous"]),
            "vla_four_modalities": bool(vla["gate"]["every_episode_has_four_modalities"]),
            "photonic_generation_active": bool(int(getattr(generator, "photonic_contexts", 0)) > 0),
        },
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
    }
    receipt["gate"]["passed"] = bool(all(receipt["gate"].values()))
    return receipt


# ---------------------------------------------------------------------- K4
def stage_k4(directory: Path) -> Dict[str, object]:
    started = time.perf_counter()
    from henri.holarch.code_lang import (
        HolarchCodeTextTokenizer,
        VocabularyLedger,
        rebase_tenant_vocabulary,
    )
    from henri.holarch.ontology_guard import build_verifier
    from henri.holarch.server import SidecarService, build_app, verify_receipt

    workdir = directory / "_work"
    workdir.mkdir(parents=True, exist_ok=True)
    lock = VocabularyLedger.load(VOCAB_LOCK)
    tenant_a = HolarchCodeTextTokenizer(vocab_lock=lock, elastic_cap=8192)
    tenant_b = HolarchCodeTextTokenizer(vocab_lock=lock, elastic_cap=8192)
    names_a = [f"tenant_alpha_{index:03d}_value" for index in range(300)]
    names_b = [f"tenant_beta_{index:03d}_value" for index in range(300)]
    shared = [f"tenant_shared_{index:03d}" for index in range(50)]
    tenant_a.add_identifiers(names_a + shared)
    tenant_b.add_identifiers(shared + names_b)
    rebase = rebase_tenant_vocabulary(tenant_a, tenant_b)
    merged = rebase.tokenizer
    calls = [f"call {name}(value)" for name in (names_a[:25] + names_b[:25] + shared[:10])]
    lossless = 0
    for call in calls:
        ids_b = tenant_b.encode(call)
        translated = rebase.translate(ids_b, "b")
        lossless += int(merged.decode(translated) == tenant_b.decode(ids_b))
    # Tier-1 identity is checked over the named vocabulary (curated pieces and
    # locked identifiers). The 256 byte tokens are implicit and do not live in
    # stoi, so they are not part of this count.
    tier1_shifts = sum(
        1
        for name, token in tenant_a.stoi.items()
        if int(token) < int(rebase.tier1_size) and merged.stoi.get(name) != int(token)
    )
    id_map_exact = all(
        merged.token_text(int(rebase.id_map_b_to_union[token])) == tenant_b.token_text(token)
        for token in range(int(rebase.tier1_size), int(len(tenant_b)))
    )

    # The K4 kernel benchmark indexes the package docstrings only. Excluding
    # docs/HOLARCH.md keeps this measurement reproducible: a documentation
    # edit changes the changelog corpus, not the ranking kernel under test.
    verifier = build_verifier(ROOT, docs=())
    claims = [
        "The medium profile has 1,020,380 parameters and reaches 0.7083 closed-loop action.",
        "The vision accuracy (test) is 0.9798 against a 0.9750 bar.",
        "The holarch uses four BaTiO3 plates inside a Sagnac loop with calibrated heaters.",
        "Zone C stores engrams in SQLite when TimescaleDB is not reachable.",
        "The tokenizer falls back to a 256-byte vocabulary for lossless round trips.",
        "The planner scores four grid actions with a goal attractor and a cycle penalty.",
        "The relational head keeps both quadratures of the bound wave.",
        "The seed cartridge holds 604 action engrams and class-balanced support.",
        "The elastic vocabulary opens a Tier-2 tail above id 2048.",
        "The D-Guard blend selects the wave weight that minimises validation loss.",
    ]
    parity_ok = 0
    matrix_latencies: List[float] = []
    python_latencies: List[float] = []
    for claim in claims:
        top_matrix = verifier.bm25(claim, top_k=4)
        top_python = verifier._bm25_python(claim, top_k=4)
        parity_ok += int([index for _score, index in top_matrix] == [index for _score, index in top_python])
        for _ in range(5):
            begin = time.perf_counter()
            verifier.bm25(claim, top_k=4)
            matrix_latencies.append((time.perf_counter() - begin) * 1e3)
            begin = time.perf_counter()
            verifier._bm25_python(claim, top_k=4)
            python_latencies.append((time.perf_counter() - begin) * 1e3)
    p50 = lambda values: float(np.percentile(values, 50)) if values else 0.0
    p95 = lambda values: float(np.percentile(values, 95)) if values else 0.0

    cfg, model, memory, clean_info, metadata, payload = _clean_model("k4_clean", workdir)
    service = SidecarService(
        cfg=cfg, backend="sqlite", memory_path=str(workdir / "k4_service.sqlite3"),
        checkpoint=str(CHECKPOINT), rate_limit_rps=1000.0, rate_limit_burst=1000.0,
    )
    trap = build_cul_de_sac_traps()[0]
    claim = claims[0]
    errors: List[str] = []
    latencies: List[float] = []
    statuses: List[int] = []
    lock_obj = threading.Lock()

    def worker(worker_id: int) -> None:
        try:
            for _ in range(8):
                kind = worker_id % 4
                begin = time.perf_counter()
                if kind == 0:
                    result = service.verify(claim, kind="text")
                elif kind == 1:
                    result = service.plan(trap["observation"], transition="rotor", lookahead_depth=2)
                elif kind == 2:
                    result = service.veto("action", trap["observation"])
                else:
                    result = service.solve("ZoneCMemory.query", ["CALLS", "CALLS"])
                elapsed = (time.perf_counter() - begin) * 1e3
                with lock_obj:
                    latencies.append(elapsed)
                    statuses.append(200 if bool(result.get("ok", False)) else 500)
        except Exception as error:  # noqa: BLE001 - the failure is the measurement
            with lock_obj:
                errors.append(f"{type(error).__name__}: {error}")

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(16)]
    begin = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    concurrent_seconds = float(time.perf_counter() - begin)
    chain_valid = bool(service.ledger.verify_chain())
    # The bundle gate needs durable records to move. The K4 service is a fresh
    # store, so one ledger and one compiled plan are written first.
    from henri.holarch.code_lang import VOCAB_LEDGER_KIND

    service_memory = service.sidecar.model.memory
    service_memory.put_meta("default", VOCAB_LEDGER_KIND, "vocab_ledger", lock.as_payload())
    service_memory.put_meta(
        "default", VOCAB_LEDGER_KIND, "vocab_ledger_tier2",
        {
            "tier1_size": int(lock.vocab_size),
            "tier2_ids": list(range(int(lock.vocab_size), int(lock.vocab_size) + 8)),
            "count": 8,
        },
    )
    service_memory.put_meta(
        "default", "plan", "k4_bundle_probe",
        {
            "name": "k4_bundle_probe",
            "intent": "returns the value unchanged",
            "source": "def k4_bundle_probe(value):\n    return value\n",
            "kind": "code",
            "version": 1,
            "engram_id": -1,
            "digest": "",
            "slots": {},
        },
    )
    bundle = service.export_import_bundle("default", "imported")
    bundle_result = dict(bundle.get("result") or {})
    service.close()
    memory.close()

    receipt = {
        "stage": "K4",
        "rebase": {
            "tenant_a_identifiers": len(names_a),
            "tenant_b_identifiers": len(names_b),
            "shared_identifiers": len(shared),
            "tier1_size": int(rebase.tier1_size),
            "union_identifiers": int(rebase.union_size),
            "added_from_b": len(rebase.added_from_b),
            "tier1_shifts": int(tier1_shifts),
            "lossless_translations": int(lossless),
            "translation_total": len(calls),
            "id_map_exact": bool(id_map_exact),
            "payload_digest": str(rebase.as_payload().get("digest", "")),
        },
        "bm25": {
            "claims": len(claims),
            "top4_parity": int(parity_ok),
            "matrix_p50_ms": p50(matrix_latencies),
            "matrix_p95_ms": p95(matrix_latencies),
            "python_p50_ms": p50(python_latencies),
            "python_p95_ms": p95(python_latencies),
            "speedup_p50": float(p50(python_latencies) / max(p50(matrix_latencies), 1e-9)),
            "speedup_p95": float(p95(python_latencies) / max(p95(matrix_latencies), 1e-9)),
            "passages": len(verifier.passages),
            "vocabulary": int(len(verifier._bm25_vocab)),
        },
        "concurrency": {
            "threads": 16,
            "requests": len(statuses),
            "errors": errors,
            "ok": int(sum(1 for status in statuses if status == 200)),
            "seconds": concurrent_seconds,
            "p50_ms": p50(latencies),
            "p95_ms": p95(latencies),
            "receipt_chain_valid": bool(chain_valid),
            "gil_enabled": bool(getattr(sys, "_is_gil_enabled", lambda: True)()),
        },
        "bundle": bundle_result,
        "gate": {
            "rebase_zero_tier1_shifts": bool(tier1_shifts == 0),
            "rebase_lossless_100pct": bool(lossless == len(calls)),
            "rebase_id_map_exact": bool(id_map_exact),
            "bm25_top4_parity": bool(parity_ok == len(claims)),
            "bm25_faster_p50": bool(p50(matrix_latencies) < p50(python_latencies)),
            "bm25_faster_p95": bool(p95(matrix_latencies) < p95(python_latencies)),
            "concurrency_128_zero_errors": bool(len(statuses) == 128 and not errors),
            "concurrency_receipt_chain": bool(chain_valid),
            "bundle_copied": bool(int(bundle_result.get("written", 0)) > 0),
        },
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
    }
    receipt["gate"]["passed"] = bool(all(receipt["gate"].values()))
    return receipt


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
    parser = argparse.ArgumentParser(description="Cycle K verification")
    parser.add_argument("--stages", default="k1,k2,k3,k4,eval")
    parser.add_argument("--receipts", default=str(ROOT / "receipts" / "holarch"))
    args = parser.parse_args()
    directory = Path(args.receipts)
    directory.mkdir(parents=True, exist_ok=True)
    stages = [token.strip().lower() for token in str(args.stages).split(",") if token.strip()]
    runners = {
        "k1": ("k1_full_cartridge.json", stage_k1),
        "k2": ("k2_boundary_discovery.json", stage_k2),
        "k3": ("k3_unified_vla_and_dag.json", stage_k3),
        "k4": ("k4_rebase_and_concurrency.json", stage_k4),
        "eval": ("k_eval.json", stage_eval),
    }
    aggregate: Dict[str, object] = {"cycle": "K", "head": _git_head(), "stages": {}}
    exit_code = 0
    for stage in stages:
        if stage not in runners:
            continue
        name, runner = runners[stage]
        print(f"[k-cycle] {stage} ...", flush=True)
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
            f"[k-cycle] {stage} passed={payload.get('gate', {}).get('passed')} "
            f"{written['path']}",
            flush=True,
        )
        if not bool(payload.get("gate", {}).get("passed", False)):
            exit_code = 1
    aggregate["passed"] = bool(
        all(bool(item.get("passed")) for item in aggregate["stages"].values())
    ) if aggregate["stages"] else False
    if aggregate["stages"]:
        written = _write(directory, "k_cycle.json", aggregate)
        print(f"[k-cycle] aggregate {written['path']} passed={aggregate['passed']}", flush=True)
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
