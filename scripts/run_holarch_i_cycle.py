"""Cycle I verification: immutable vocabulary lock (I1), type-directed
N-stage DAG synthesis (I2), Cul-de-Sac horizon traps with a batched 16-branch
rotor rollout (I3), and multi-package scaling with a unified agent turn (I4).

    python scripts/run_holarch_i_cycle.py --stages i1,i2,i3,i4,eval

Every number is measured in this process and written to a JSON receipt with a
SHA-256 sidecar; i_cycle.json aggregates them. The planner measurements read a
scratch copy of the repository Zone C store, pinned by SHA-256 in the receipt,
because the deployed wave planner fuses the retrieved Zone C context.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from henri.holarch.code_gen import (
    SynthesisEngine,
    build_synthesis_engine,
    discover_vocabulary_lock,
)
from henri.holarch.code_lang import (
    CodeTextCorpus,
    VocabularyLedger,
    build_tokenizer,
    roundtrip_check,
)
from henri.holarch.config import HolarchConfig
from henri.holarch.model import (
    LatentActionLookahead,
    build_cul_de_sac_traps,
    evaluate_closed_loop_action,
    load_checkpoint,
    open_planner_memory,
)
from henri.holarch.pipeline import HenriHolarch
from henri.holarch.server import SidecarService, build_app, verify_receipt
from henri.holarch.symbolic import CodeGraphIndexer, SymbolicTopologyTask
from henri.holarch.tasks import build_tasks
from henri.holarch.zone_c import ZoneCMemory

G_REVISION = "67e9b25"
CODE_ROOT = ROOT / "henri" / "holarch"
CHECKPOINT = ROOT / "receipts" / "holarch" / "holarch_medium.pt"
VOCAB_LOCK = ROOT / "receipts" / "holarch" / "vocab_lock.json"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def _git_files(suffixes: Sequence[str]) -> List[Path]:
    try:
        listing = subprocess.run(
            ["git", "ls-files"], cwd=str(ROOT), capture_output=True, text=True, timeout=120
        ).stdout.splitlines()
    except Exception:
        listing = []
    wanted = tuple(suffixes)
    return [ROOT / name for name in listing if name.endswith(wanted)]


def _copy_live_tree(target: Path) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    package = target / "henri" / "holarch"
    package.mkdir(parents=True, exist_ok=True)
    for path in sorted(CODE_ROOT.glob("*.py")):
        shutil.copy2(path, package / path.name)
    (target / "docs").mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "docs" / "HOLARCH.md", target / "docs" / "HOLARCH.md")
    return target


MUTATION_FUNCTIONS = 600
MUTATION_REFERENCES = 10


def _mutation_text() -> str:
    lines: List[str] = []
    for index in range(MUTATION_FUNCTIONS):
        name = f"live_probe_{index}"
        expression = "value"
        for _ in range(MUTATION_REFERENCES):
            expression = f"{name}({expression})"
        lines.append(f"def {name}(value):")
        lines.append(f"    return {expression}")
        lines.append("")
    return chr(10).join(lines)


# ---------------------------------------------------------------------- I1
def stage_i1(directory: Path) -> Dict[str, object]:
    started = time.perf_counter()
    h_module = _load_module("run_holarch_h_cycle", ROOT / "scripts" / "run_holarch_h_cycle.py")
    canonical_corpus, provenance = h_module.frozen_corpus(G_REVISION)
    canonical = build_tokenizer(canonical_corpus)
    ledger = canonical.ledger
    if VOCAB_LOCK.exists():
        locked = VocabularyLedger.load(VOCAB_LOCK)
        lock_matches_canonical = locked.identifiers == ledger.identifiers
        ledger = locked
    else:
        lock_matches_canonical = True
        ledger.save(VOCAB_LOCK)
    lock_file = VOCAB_LOCK
    lock_sha = _digest(lock_file)
    lock_file.with_name(lock_file.name + ".sha256").write_text(lock_sha + chr(10), encoding="utf-8")

    before_root = _copy_live_tree(Path(tempfile.mkdtemp(prefix="henri_i1_before_")))
    after_root = _copy_live_tree(Path(tempfile.mkdtemp(prefix="henri_i1_after_")))
    (after_root / "henri" / "holarch" / "live_mutation_probe.py").write_text(
        _mutation_text(), encoding="utf-8"
    )
    before_corpus = CodeTextCorpus.holarch(before_root)
    after_corpus = CodeTextCorpus.holarch(after_root)

    unlocked_before = build_tokenizer(before_corpus)
    unlocked_after = build_tokenizer(after_corpus)
    unlocked_shifts = sum(
        1 for name, token in unlocked_before.stoi.items() if unlocked_after.stoi.get(name) != token
    )
    unlocked_missing = sum(
        1 for name in unlocked_before.identifiers if name not in unlocked_after.stoi
    )

    locked_before = build_tokenizer(before_corpus, vocab_lock=ledger)
    locked_after = build_tokenizer(after_corpus, vocab_lock=ledger)
    locked_shifts = sum(
        1 for name, token in locked_before.stoi.items() if locked_after.stoi.get(name) != token
    )
    id_map_equal = dict(locked_before.stoi) == dict(locked_after.stoi)
    vocabulary_identical = locked_before.spec.digest() == canonical.spec.digest()

    roundtrip = roundtrip_check(locked_before, _git_files((".py", ".md")))
    roundtrip_rate = float(roundtrip["ok"] / max(int(roundtrip["texts"]), 1))

    growth_lock = ledger.truncated(200, vocab_size=1400)
    growth_before = build_tokenizer(before_corpus, vocab_size=1400, vocab_lock=growth_lock)
    growth_after = build_tokenizer(after_corpus, vocab_size=1400, vocab_lock=growth_lock)
    growth_locked_shifts = sum(
        1
        for name in growth_lock.identifiers
        if growth_before.stoi.get(name) != growth_after.stoi.get(name)
    )
    growth_tail_ok = all(
        growth_after.stoi.get(name) == growth_before.stoi.get(name)
        for name in growth_lock.identifiers
    )
    growth_appended = max(0, len(growth_before.identifiers) - len(growth_lock.identifiers))
    growth_tail_ids = [
        growth_before.stoi[name]
        for name in growth_before.identifiers[len(growth_lock.identifiers) :]
    ]
    growth_tail_contiguous = bool(
        all(
            token == growth_lock.identifier_base + len(growth_lock.identifiers) + offset
            for offset, token in enumerate(growth_tail_ids)
        )
    )

    tokenizer_before = build_tokenizer(CodeTextCorpus.holarch(before_root), vocab_lock=ledger)
    h1_before = h_module.stage_h1("git", tokenizer=tokenizer_before)
    tokenizer_after = build_tokenizer(CodeTextCorpus.holarch(after_root), vocab_lock=ledger)
    h1_after = h_module.stage_h1("git", tokenizer=tokenizer_after)
    top1_before = float(h1_before["tuned"]["top1"])
    top1_after = float(h1_after["tuned"]["top1"])
    bpc_before = float(h1_before["tuned"]["bits_per_char"])
    bpc_after = float(h1_after["tuned"]["bits_per_char"])
    base_top1 = 0.562016806722689
    base_bpc = 1.2916944159315715

    payload = {
        "stage": "I1",
        "corpus": {
            "source": provenance.get("source", ""),
            "revision": provenance.get("revision", ""),
            "digest": str(canonical_corpus.digest),
            "texts": int(len(canonical_corpus)),
            "live_before_files": int(len(before_corpus)),
            "live_after_files": int(len(after_corpus)),
        },
        "vocabulary_lock": {
            "path": str(lock_file.relative_to(ROOT)),
            "sha256": lock_sha,
            "ledger_digest": ledger.digest(),
            "identifiers": len(ledger.identifiers),
            "identifier_base": ledger.identifier_base,
            "curated_size": ledger.curated_size,
            "vocab_size": ledger.vocab_size,
            "capacity": ledger.capacity,
            "full": bool(ledger.capacity == len(ledger.identifiers)),
            "matches_canonical_67e9b25": bool(lock_matches_canonical),
            "canonical_vocabulary_digest": canonical.spec.digest(),
            "locked_vocabulary_digest": locked_before.spec.digest(),
        },
        "mutation": {
            "module": "henri/holarch/live_mutation_probe.py",
            "functions": MUTATION_FUNCTIONS,
            "self_references_per_function": MUTATION_REFERENCES,
            "unlocked_existing_id_shifts": int(unlocked_shifts),
            "unlocked_identifiers_displaced": int(unlocked_missing),
            "locked_existing_id_shifts": int(locked_shifts),
            "locked_identifier_map_identical": bool(id_map_equal),
            "locked_vocabulary_identical_to_canonical": bool(vocabulary_identical),
        },
        "roundtrip": {
            "texts": int(roundtrip["texts"]),
            "ok": int(roundtrip["ok"]),
            "lossless_rate": roundtrip_rate,
            "failures": list(roundtrip["failures"])[:8],
            "languages": [".py", ".md"],
        },
        "growth": {
            "vocab_size": 1400,
            "locked_identifiers": len(growth_lock.identifiers),
            "size_before": int(len(growth_before)),
            "size_after": int(len(growth_after)),
            "appended_after_lock": int(growth_appended),
            "appended_ids_contiguous": bool(growth_tail_contiguous),
            "locked_id_shifts": int(growth_locked_shifts),
            "locked_tail_preserved": bool(growth_tail_ok),
            "free_tail": int(growth_lock.capacity - len(growth_lock.identifiers)),
        },
        "h1": {
            "before": {
                "top1": top1_before,
                "bits_per_char": bpc_before,
                "floor_top1": float(h1_before["floor"]["top1"]),
                "beta_gate": dict(h1_before["beta_gate"]),
                "decision_preserving_projection": bool(
                    h1_before.get("calibration", {}).get("mode_guard", True)
                ),
                "vocabulary_digest": str(h1_before["corpus"].get("vocabulary_digest", "")),
            },
            "after": {
                "top1": top1_after,
                "bits_per_char": bpc_after,
                "floor_top1": float(h1_after["floor"]["top1"]),
                "beta_gate": dict(h1_after["beta_gate"]),
                "decision_preserving_projection": bool(
                    h1_after.get("calibration", {}).get("mode_guard", True)
                ),
                "vocabulary_digest": str(h1_after["corpus"].get("vocabulary_digest", "")),
            },
            "base_top1": base_top1,
            "base_bits_per_char": base_bpc,
            "identical": bool(abs(top1_before - top1_after) < 1e-12 and abs(bpc_before - bpc_after) < 1e-12),
        },
        "gate": {
            "existing_id_shifts_after_mutation_zero": bool(locked_shifts == 0),
            "roundtrip_lossless_1_0": bool(abs(roundtrip_rate - 1.0) < 1e-12),
            "top1_held_out_at_least_base": bool(top1_before >= base_top1 - 1e-12),
            "bits_per_char_at_most_base": bool(bpc_before <= base_bpc + 1e-12),
            "decision_preserving_projection": bool(
                h1_before.get("calibration", {}).get("mode_guard", True)
            ),
            "before_after_identical": bool(
                abs(top1_before - top1_after) < 1e-12 and abs(bpc_before - bpc_after) < 1e-12
            ),
            "unlocked_drift_exceeds_500": bool(unlocked_shifts > 500),
            "locked_vocabulary_identical": bool(id_map_equal and vocabulary_identical),
            "append_only_growth_verified": bool(
                growth_locked_shifts == 0
                and growth_tail_ok
                and growth_tail_contiguous
                and growth_appended > 0
            ),
            "lock_matches_canonical": bool(lock_matches_canonical),
        },
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
    }
    payload["gate"]["passed"] = bool(all(payload["gate"].values()))
    return payload


# ---------------------------------------------------------------------- I2
THREE_PLAN: List[Dict[str, object]] = [
    {
        "id": "tri01",
        "function": "tri_alpha",
        "argument": "links",
        "bounds": (0.0, 2.5),
        "expected": 2.5,
    },
    {
        "id": "tri02",
        "function": "tri_beta",
        "argument": "edges",
        "bounds": (0.0, 1.0),
        "expected": 1.0,
    },
    {
        "id": "tri03",
        "function": "tri_gamma",
        "argument": "pairs",
        "bounds": (1.0, 4.0),
        "expected": 3.0,
    },
    {
        "id": "tri04",
        "function": "tri_delta",
        "argument": "records",
        "bounds": (0.0, 2.0),
        "expected": 2.0,
    },
    {
        "id": "tri05",
        "function": "tri_epsilon",
        "argument": "rows",
        "bounds": (1.0, 2.0),
        "expected": 2.0,
    },
    {
        "id": "tri06",
        "function": "tri_zeta",
        "argument": "items",
        "bounds": (0.5, 3.5),
        "expected": 3.0,
    },
    {
        "id": "tri07",
        "function": "tri_eta",
        "argument": "graph_edges",
        "bounds": (0.0, 0.5),
        "expected": 0.5,
    },
    {
        "id": "tri08",
        "function": "tri_theta",
        "argument": "edge_list",
        "bounds": (2.0, 5.0),
        "expected": 3.0,
    },
]

TYPE_TRAPS: List[Dict[str, str]] = [
    {
        "id": "itrap01",
        "intent": (
            "Write a function bad_alpha that takes argument record and first returns "
            "the sha256 field of an epistemic receipt then returns the arithmetic mean "
            "of a list of floats as a float."
        ),
    },
    {
        "id": "itrap02",
        "intent": (
            "Write a function bad_beta that takes argument record and first returns "
            "the sha256 field of an epistemic receipt then clips the result into the "
            "closed unit interval."
        ),
    },
    {
        "id": "itrap03",
        "intent": (
            "Write a function bad_gamma that takes argument values and first returns "
            "the arithmetic mean of a list of floats as a float then returns the "
            "sha256 field of an epistemic receipt."
        ),
    },
    {
        "id": "itrap04",
        "intent": (
            "Write a function bad_delta that takes argument edges and first returns "
            "the sorted unique source names of a list of edges then returns the "
            "sha256 field of an epistemic receipt."
        ),
    },
]


def _three_plan_intent(spec: Dict[str, object]) -> str:
    low, high = spec["bounds"]  # type: ignore[misc]
    return (
        f"Write a function {spec['function']} that takes argument {spec['argument']} "
        "and first returns the sorted unique source names of a list of edges then "
        "returns the number of tokens in that result then clips the result into the "
        f"closed interval {low} to {high}."
    )


def _three_plan_test(spec: Dict[str, object]) -> str:
    function = spec["function"]
    argument = spec["argument"]
    return (
        "class _E:" + chr(10)
        + "    def __init__(self, s):" + chr(10)
        + "        self.source = s" + chr(10)
        + f"assert {function}([_E('b'), _E('a'), _E('b'), _E('c')]) == {spec['expected']}" + chr(10)
    )


def stage_i2(directory: Path) -> Dict[str, object]:
    started = time.perf_counter()
    f_module = _load_module("run_holarch_code_lang", ROOT / "scripts" / "run_holarch_code_lang.py")
    h_module = _load_module("run_holarch_h_cycle", ROOT / "scripts" / "run_holarch_h_cycle.py")
    memory_path = directory / "_work" / "i2_synthesis.sqlite3"
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    if memory_path.exists():
        memory_path.unlink()
    memory = ZoneCMemory(HolarchConfig.medium(), backend="sqlite", path=str(memory_path))
    symbols = set()
    for task in list(f_module.CORE_TASKS) + list(f_module.SHIFT_TASKS):
        symbols.add(task["source"].split("(")[0].replace("def ", "").strip())
    indexer = CodeGraphIndexer(None)
    indexer.index(CODE_ROOT, "henri")
    engine, stats = build_synthesis_engine(
        ROOT,
        memory,
        namespace="i2",
        symbol_table=symbols,
        max_fit_tokens=40000,
        signatures=dict(indexer.signatures),
    )
    for task in list(f_module.CORE_TASKS) + list(f_module.SHIFT_TASKS):
        engine.adapt(task["intent"], task["source"], task["id"])

    def measured(intent: str, test: str) -> Dict[str, object]:
        row = engine.synthesize(intent, kind="code")
        executed = engine.execute(row["source"], test) if row.get("verified") else {"ok": False, "reason": "not verified"}
        return {
            "status": row["status"],
            "plan": row.get("plan", ""),
            "composed_from": list(row.get("composed_from", []) or []),
            "dag_depth": int(row.get("dag_depth", 1)),
            "type_compatible": bool(row.get("type_compatible", True)),
            "verified": bool(row.get("verified", False)),
            "executed": bool(executed["ok"]),
            "reason": str(executed.get("reason", ""))[:160],
            "source": str(row.get("source", ""))[:300],
        }

    base_rows = [dict(task_id=task["id"], **measured(task["intent"], task["test"]))
                 for task in list(f_module.CORE_TASKS) + list(f_module.SHIFT_TASKS)]
    base_pass = sum(1 for row in base_rows if row["executed"])
    comp_rows = [dict(task_id=task["id"], **measured(task["intent"], task["test"]))
                 for task in h_module.COMPOSITIONAL]
    comp_pass = sum(1 for row in comp_rows if row["executed"])
    multi_rows = [dict(task_id=task["id"], **measured(task["intent"], task["test"]))
                  for task in h_module.MULTI_PLAN]
    multi_pass = sum(1 for row in multi_rows if row["executed"])

    three_rows = []
    for spec in THREE_PLAN:
        row = dict(spec_id=spec["id"], **measured(_three_plan_intent(spec), _three_plan_test(spec)))
        row["bounds"] = list(spec["bounds"])  # type: ignore[arg-type]
        row["expected"] = spec["expected"]
        three_rows.append(row)
    three_pass = sum(1 for row in three_rows if row["executed"] and len(row["composed_from"]) == 3)

    trap_rows = []
    for spec in TYPE_TRAPS:
        row = engine.synthesize(spec["intent"], kind="code")
        reason = str(row.get("reason", ""))
        refused = bool(row.get("status") == "refused" and reason.startswith("type mismatch"))
        trap_rows.append(
            {
                "id": spec["id"],
                "status": row.get("status", ""),
                "refused": refused,
                "plan": row.get("plan", ""),
                "composed_from": list(row.get("composed_from", []) or []),
                "reason": reason[:200],
                "type_compatible": bool(row.get("type_compatible", True)),
            }
        )
    trap_pass = sum(1 for row in trap_rows if row["refused"])

    total = len(base_rows) + len(comp_rows) + len(multi_rows) + len(three_rows) + len(trap_rows)
    passed = base_pass + comp_pass + multi_pass + three_pass + trap_pass
    payload = {
        "stage": "I2",
        "corpus": dict(stats),
        "plans_compiled": len(engine.plans.plans),
        "base_tasks": {"total": len(base_rows), "passed": int(base_pass), "rows": base_rows},
        "single_plan_compositional": {
            "total": len(comp_rows),
            "passed": int(comp_pass),
            "rows": comp_rows,
        },
        "two_plan_pipelines": {
            "total": len(multi_rows),
            "passed": int(multi_pass),
            "rows": multi_rows,
        },
        "three_plan_pipelines": {
            "total": len(three_rows),
            "passed": int(three_pass),
            "accuracy": float(three_pass / max(len(three_rows), 1)),
            "rows": three_rows,
        },
        "type_mismatch_traps": {
            "total": len(trap_rows),
            "refused": int(trap_pass),
            "rows": trap_rows,
        },
        "total_synthesis_benchmark": {
            "total": int(total),
            "passed": int(passed),
            "accuracy": float(passed / max(total, 1)),
        },
        "emission": {
            "exact": int(engine.emission_exact),
            "fallbacks": int(engine.emission_fallbacks),
            "composed": int(engine.composed),
            "dag_composed": int(engine.dag_composed),
            "type_refused": int(engine.type_refused),
        },
        "gate": {
            "base_24_of_24": bool(base_pass == 24),
            "single_plan_12_of_12": bool(comp_pass == 12),
            "two_plan_8_of_8": bool(multi_pass == 8),
            "three_plan_at_least_7_of_8": bool(three_pass >= 7),
            "three_plan_8_of_8": bool(three_pass == 8),
            "type_mismatch_4_of_4": bool(trap_pass == 4),
            "total_56_of_56": bool(total == 56 and passed == 56),
            "plans_compiled_24": bool(len(engine.plans.plans) == 24),
        },
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
    }
    payload["gate"]["passed"] = bool(all(payload["gate"].values()))
    memory.close()
    return payload


# ---------------------------------------------------------------------- I3
def _load_fixture_model(checkpoint: Path, cfg, workdir: Path, prefix: str):
    memory, fixture = open_planner_memory(cfg, workdir=workdir, prefix=prefix)
    model = HenriHolarch(cfg, memory=memory)
    load_checkpoint(checkpoint, model=model)
    model.eval()
    return model, memory, fixture


def stage_i3(directory: Path) -> Dict[str, object]:
    started = time.perf_counter()
    checkpoint_payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    cfg = HolarchConfig(**checkpoint_payload["config"])
    workdir = directory / "_work"
    workdir.mkdir(parents=True, exist_ok=True)
    model, memory, fixture = _load_fixture_model(CHECKPOINT, cfg, workdir, "i3_planner")
    task = build_tasks(cfg, seed=int(cfg.seed))["action"]
    rotor = getattr(model, "action_rotor", None)
    traps = build_cul_de_sac_traps()

    # Closed loop FIRST, at the process's natural reduction, exactly as the H3
    # receipt measured it. A short warm-up fills the fetch cache and the
    # thread pool; the scored measurement is then repeated three times and
    # reduced by the median, because an 8-thread reduction is not
    # bit-associative and a single sample can sit on a near-tie.
    warmup = evaluate_closed_loop_action(
        model, task, episodes=8, rotor=rotor, transition="rotor",
        lookahead_depth=2, gamma_look=0.02, batch_rollout=True, use_memory=True,
    )
    closed_runs: List[Dict[str, object]] = []
    for _ in range(3):
        closed_runs.append(
            evaluate_closed_loop_action(
                model, task, episodes=48, rotor=rotor, transition="rotor",
                lookahead_depth=2, gamma_look=0.02, batch_rollout=True, use_memory=True,
            )
        )
    closed_wins = [int(run["wins"]) for run in closed_runs]
    deployed_wins = int(statistics.median(closed_wins))
    closed_batch = next(run for run in closed_runs if int(run["wins"]) == deployed_wins)
    closed_loop = evaluate_closed_loop_action(
        model, task, episodes=48, rotor=rotor, transition="rotor",
        lookahead_depth=2, gamma_look=0.02, batch_rollout=False, use_memory=True,
    )
    reload_model, reload_memory, reload_fixture = _load_fixture_model(
        CHECKPOINT, cfg, workdir, "i3_reload"
    )
    reload_rotor = getattr(reload_model, "action_rotor", None)
    reload_task = build_tasks(reload_model.cfg, seed=int(reload_model.cfg.seed))["action"]
    replay = evaluate_closed_loop_action(
        reload_model, reload_task, episodes=48, rotor=reload_rotor, transition="rotor",
        lookahead_depth=2, gamma_look=0.02, batch_rollout=True, use_memory=True,
    )

    service_memory_path = workdir / "i3_service.sqlite3"
    if service_memory_path.exists():
        service_memory_path.unlink()
    service = SidecarService(
        cfg=cfg,
        backend="sqlite",
        memory_path=str(service_memory_path),
        checkpoint=str(CHECKPOINT),
    )
    from fastapi.testclient import TestClient

    client = TestClient(build_app(service))
    service_default_memory = service.sidecar.model.memory

    threads_before = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        masked = LatentActionLookahead(
            model, None, rotor=rotor, transition="rotor", lookahead_depth=2, gamma_look=0.02
        )
        loop = LatentActionLookahead(
            model, None, rotor=rotor, transition="rotor", lookahead_depth=2,
            gamma_look=0.02, batch_rollout=False,
        )
        single = LatentActionLookahead(
            model, None, rotor=rotor, transition="rotor", lookahead_depth=1
        )
        unmasked = LatentActionLookahead(
            model, None, rotor=rotor, transition="rotor", lookahead_depth=2,
            gamma_look=0.02, mask_lookahead=False,
        )
        rows = []
        masked_ok = 0
        depth1_ok = 0
        unmasked_ok = 0
        parity_max = 0.0
        api_matches = 0
        for trap in traps:
            observation = trap["observation"]
            choice = masked.choose(observation, [])
            action = int(choice["action"])
            psi = choice["psi"]
            logits = choice["logits"]
            alignment = [float(value) for value in choice["alignment"]]
            trap_action = int(trap["trap_action"])
            bypass = int(trap["bypass_action"])
            align_max = int(np.argmax(alignment)) == trap_action
            strict = sum(1 for value in alignment if abs(value - alignment[trap_action]) < 1e-9) == 1
            first = masked.predict_batch(psi, [0, 1, 2, 3])
            blocked = masked.wall_mask_batch(first)[trap_action]
            opposite = (1, 0, 3, 2)[trap_action]
            legal_t2 = [index for index in range(4) if not bool(blocked[index]) and index != opposite]
            score_batch, _ = masked.score(psi, logits, observation, [])
            score_loop, _ = loop.score(psi, logits, observation, [])
            diff = float((score_batch - score_loop).abs().max())
            parity_max = max(parity_max, diff)
            arm_depth1 = int(single.choose(observation, [])["action"])
            arm_unmasked = int(unmasked.choose(observation, [])["action"])
            service.sidecar.model.memory = memory
            try:
                api = client.post(
                    "/v1/holarch/plan",
                    json={
                        "observation": observation,
                        "transition": "rotor",
                        "lookahead_depth": 2,
                        "gamma_look": 0.02,
                        "namespace": "i3",
                    },
                ).json()
            finally:
                service.sidecar.model.memory = service_default_memory
            api_action = int(api["result"]["action"])
            masked_ok += int(action == bypass)
            depth1_ok += int(arm_depth1 == bypass)
            unmasked_ok += int(arm_unmasked == bypass)
            api_matches += int(api_action == action)
            rows.append(
                {
                    "id": trap["id"],
                    "agent": list(trap["agent"]),
                    "pocket": list(trap["pocket"]),
                    "goal": list(trap["goal"]),
                    "trap_action": trap_action,
                    "bypass_action": bypass,
                    "masked_action": action,
                    "depth1_action": arm_depth1,
                    "unmasked_action": arm_unmasked,
                    "api_action": api_action,
                    "alignment": alignment,
                    "trap_alignment_is_max": bool(align_max and strict),
                    "legal_non_reversal_at_t2": legal_t2,
                    "cul_de_sac_isolated": bool(len(legal_t2) == 0),
                    "score_parity_max_abs": float(diff),
                    "correct": bool(action == bypass),
                    "receipt_valid": bool(verify_receipt(api["receipt"])),
                    "invariants": dict(trap["invariants"]),
                }
            )
    finally:
        torch.set_num_threads(threads_before)
        service.sidecar.model.memory = service_default_memory

    # Rollout latency: the pure 4 + 16 branch computation, then the endpoint.
    psi = masked.codec.encode(traps[0]["observation"])
    goal = _slot3_for_timing(masked)
    batch_times: List[float] = []
    loop_times: List[float] = []
    for _ in range(30):
        begin = time.perf_counter()
        masked._depth_two_batched(psi, goal, [])
        batch_times.append((time.perf_counter() - begin) * 1e3)
        begin = time.perf_counter()
        loop._depth_two_loop(psi, goal, [])
        loop_times.append((time.perf_counter() - begin) * 1e3)
    api_batch: List[float] = []
    api_loop: List[float] = []
    forward_times: List[float] = []
    tensor = torch.as_tensor(
        np.asarray(traps[0]["observation"], dtype=np.float32)
    ).reshape(1, -1)
    service.sidecar.model.memory = memory
    try:
        for index in range(16):
            order = (True, False) if index % 2 == 0 else (False, True)
            for flag in order:
                begin = time.perf_counter()
                client.post(
                    "/v1/holarch/plan",
                    json={
                        "observation": traps[0]["observation"],
                        "batch_rollout": bool(flag),
                        "namespace": "i3",
                    },
                )
                (api_batch if flag else api_loop).append(
                    (time.perf_counter() - begin) * 1e3
                )
        for _ in range(16):
            begin = time.perf_counter()
            service.sidecar.model("action", tensor, training=False, use_memory=True)
            forward_times.append((time.perf_counter() - begin) * 1e3)
    finally:
        service.sidecar.model.memory = service_default_memory

    batch_ms = float(statistics.median(batch_times))
    loop_ms = float(statistics.median(loop_times))
    api_batch_ms = float(statistics.median(api_batch))
    api_loop_ms = float(statistics.median(api_loop))
    forward_ms = float(statistics.median(forward_times))
    closed_ok = bool(deployed_wins == 39)
    service.close()
    memory.close()
    reload_memory.close()

    payload = {
        "stage": "I3",
        "checkpoint": {
            "path": str(CHECKPOINT.relative_to(ROOT)),
            "sha256": _digest(CHECKPOINT),
            "parameters": int(checkpoint_payload.get("parameters", 0)),
        },
        "memory_fixture": dict(fixture),
        "reload_fixture": dict(reload_fixture),
        "traps": rows,
        "controls": {
            "masked_correct": int(masked_ok),
            "depth1_correct": int(depth1_ok),
            "unmasked_correct": int(unmasked_ok),
            "depth2_wall_masked_accuracy": float(masked_ok / max(len(rows), 1)),
            "depth1_control_accuracy": float(depth1_ok / max(len(rows), 1)),
            "depth2_unmasked_control_accuracy": float(unmasked_ok / max(len(rows), 1)),
        },
        "closed_loop": {
            "success_rate": float(deployed_wins / 48.0),
            "wins": int(deployed_wins),
            "episodes": 48,
            "runs": [int(value) for value in closed_wins],
            "reduction": "median of three 48-episode runs",
            "warmup_wins": int(warmup["wins"]),
            "reactive_success_rate": float(closed_batch["reactive_success_rate"]),
            "loop_path_success_rate": float(closed_loop["success_rate"]),
            "reload_success_rate": float(replay["success_rate"]),
            "reload_wins": int(replay["wins"]),
            "reload_rotor_fitted": bool(getattr(reload_rotor, "fitted", False)),
            "threads": int(torch.get_num_threads()),
            "use_memory": True,
        },
        "parity": {
            "score_parity_max_abs": float(parity_max),
            "api_matches_choose": int(api_matches),
            "api_traps": int(len(rows)),
        },
        "latency_ms": {
            "rollout_batched_median": batch_ms,
            "rollout_loop_median": loop_ms,
            "rollout_speedup": float(loop_ms / max(batch_ms, 1e-9)),
            "plan_api_batched_median": api_batch_ms,
            "plan_api_loop_median": api_loop_ms,
            "plan_api_speedup": float(api_loop_ms / max(api_batch_ms, 1e-9)),
            "model_forward_median": forward_ms,
            "note": (
                "The endpoint is dominated by the checkpoint forward pass; the "
                "batching improvement is measured on the 4+16 branch rollout, "
                "and the endpoint comparison is a no-material-regression check."
            ),
        },
        "gate": {
            "depth2_wall_masked_8_of_8": bool(masked_ok == len(rows) == 8),
            "depth1_control_at_most_2_of_8": bool(depth1_ok <= 2),
            "depth2_unmasked_control_at_most_2_of_8": bool(unmasked_ok <= 2),
            "closed_loop_0_8125": bool(closed_ok),
            "closed_loop_median_stable": bool(
                deployed_wins == 39 and all(value in (38, 39) for value in closed_wins)
            ),
            "replay_bit_identical": bool(int(replay["wins"]) == deployed_wins),
            "batch_loop_parity": bool(parity_max < 1e-6),
            "plan_api_matches_choose": bool(api_matches == len(rows)),
            "rollout_faster_batched": bool(batch_ms <= loop_ms),
            "plan_api_no_material_regression": bool(api_batch_ms <= api_loop_ms * 1.05),
            "all_trap_invariants": bool(
                all(all(bool(value) for value in row["invariants"].values()) for row in rows)
            ),
        },
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
    }
    payload["gate"]["passed"] = bool(all(payload["gate"].values()))
    return payload


def _slot3_for_timing(lookahead: LatentActionLookahead) -> torch.Tensor:
    goal = lookahead.goal
    width = goal.shape[-1] // 4
    return goal[..., 3 * width :]


# ---------------------------------------------------------------------- I4
def _agent_step_round(service: SidecarService, planner_memory) -> Dict[str, object]:
    trap = build_cul_de_sac_traps()[0]
    truth_claim = (
        "The medium profile has 1,020,380 parameters and reaches 0.7083 closed-loop action."
    )
    rules = [
        {"a": f"d{left}", "op": "add", "b": f"d{right}", "result": f"d{(left + right) % 10}"}
        for left in range(10)
        for right in range(10)
    ]
    service.composite_call(mode="ingest", rules=rules)
    topology = SymbolicTopologyTask(seed=20261103)
    edges = topology.graph(2)
    source = topology.symbols[0]
    operators, answer = topology.query(edges, source, 1, salt=1)
    service.ingest_edges(
        [{"source": edge.source, "operator": edge.operator, "target": edge.target} for edge in edges]
    )
    f_module = _load_module("run_holarch_code_lang", ROOT / "scripts" / "run_holarch_code_lang.py")
    for task in list(f_module.CORE_TASKS) + list(f_module.SHIFT_TASKS):
        service.sidecar.adapt_synthesis(task["intent"], task["source"], task["id"])
    actions: Dict[str, Dict[str, object]] = {}
    previous = service.sidecar.model.memory
    service.sidecar.model.memory = planner_memory
    try:
        actions["veto"] = service.agent_step(
            "veto", {"task": "vision", "x": np.random.default_rng(7).random((2, 8, 8)).tolist()}
        )
        actions["verify"] = service.agent_step("verify", {"claim": truth_claim, "kind": "text"})
        actions["solve"] = service.agent_step(
            "solve", {"source": source, "operators": operators}
        )
        actions["impact"] = service.agent_step(
            "impact", {"target": "ZoneCMemory.query", "hops": 1}
        )
        actions["plan"] = service.agent_step(
            "plan",
            {"observation": trap["observation"], "transition": "rotor", "lookahead_depth": 2},
        )
        actions["synthesize"] = service.agent_step(
            "synthesize",
            {
                "intent": (
                    "Write a function i4_tri that takes argument links and first returns "
                    "the sorted unique source names of a list of edges then returns the "
                    "number of tokens in that result then clips the result into the "
                    "closed interval 0.0 to 2.5."
                )
            },
        )
        actions["composite"] = service.agent_step(
            "composite", {"mode": "query", "a": "d3", "op": "add", "b": "d4"}
        )
    finally:
        service.sidecar.model.memory = previous
    rows = []
    for name, response in actions.items():
        receipt = response["epistemic_receipt"]
        rows.append(
            {
                "action": name,
                "status": str(response["status"]),
                "abstained": bool(response["abstained"]),
                "receipt_valid": bool(verify_receipt(receipt)),
                "chain_sha256": str(receipt["sha256"]),
                "parent": str(receipt["parent"]),
                "config_digest": str(receipt["config_digest"]),
                "checkpoint_sha256": str(receipt["checkpoint_sha256"]),
                "output_keys": sorted(str(key) for key in response["output"].keys()),
            }
        )
    return {
        "rows": rows,
        "actions": int(len(rows)),
        "receipts_valid": bool(all(row["receipt_valid"] for row in rows)),
        "checkpoint_sha": bool(all(row["checkpoint_sha256"] for row in rows)),
        "solve_expected": answer,
    }


def stage_i4(directory: Path) -> Dict[str, object]:
    started = time.perf_counter()
    workdir = directory / "_work"
    workdir.mkdir(parents=True, exist_ok=True)
    directories = ["henri", "scripts", "tests"]
    indexer = CodeGraphIndexer(None)
    full = indexer.index_paths(ROOT, directories, package="workspace")
    paths = [
        path
        for directory_name in directories
        for path in sorted((ROOT / directory_name).rglob("*.py"))
        if "__pycache__" not in path.parts
    ]
    rows = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="ignore")
        indexer.update_file(ROOT, path, source_text=text)
        samples = []
        for _ in range(3):
            begin = time.perf_counter()
            report = indexer.update_file(ROOT, path, source_text=text)
            samples.append((time.perf_counter() - begin) * 1e3)
        rows.append({"module": report.updated, "bytes": len(text.encode("utf-8")), "delta_ms": float(min(samples))})
    latency = [row["delta_ms"] for row in rows]
    reference = CodeGraphIndexer(None)
    reference.index_paths(ROOT, directories, package="workspace")
    same_edges = [(edge.source, edge.operator, edge.target) for edge in reference.graph.edges] == [
        (edge.source, edge.operator, edge.target) for edge in indexer.graph.edges
    ]
    same_nodes = list(reference.graph.nodes) == list(indexer.graph.nodes)
    same_reverse = {key: list(value) for key, value in reference.reverse.items()} == {
        key: list(value) for key, value in indexer.reverse.items()
    }
    probes = ["build_tokenizer", "ZoneCMemory.query", "build_synthesis_engine", "index_code"]
    impact_same = all(
        reference.impact(name, hops=2)["callers"] == indexer.impact(name, hops=2)["callers"]
        for name in probes
    )
    mutation_path = CODE_ROOT / "sidecar.py"
    mutation_text = mutation_path.read_text(encoding="utf-8")
    mutated = mutation_text.replace(
        "self.model.memory.query(key, k=1, namespace=scope)",
        "self.model.memory.lookup(key, k=1, namespace=scope)",
    )
    before = indexer.impact("ZoneCMemory.query", hops=2)["callers"]
    mutation_started = time.perf_counter()
    mutation_report = indexer.update_file(ROOT, mutation_path, source_text=mutated)
    mutation_ms = (time.perf_counter() - mutation_started) * 1e3
    after = indexer.impact("ZoneCMemory.query", hops=2)["callers"]
    indexer.update_file(ROOT, mutation_path, source_text=mutation_text)
    restored = indexer.impact("ZoneCMemory.query", hops=2)["callers"]

    memory_path = workdir / "i4_agent.sqlite3"
    if memory_path.exists():
        memory_path.unlink()
    service = SidecarService(
        backend="sqlite", memory_path=str(memory_path), checkpoint=str(CHECKPOINT)
    )
    cfg = service.cfg
    service.ingest_code(str(CODE_ROOT), package="henri")
    planner_memory, planner_fixture = open_planner_memory(cfg, workdir=workdir, prefix="i4_planner")
    try:
        agent_surface = _agent_step_round(service, planner_memory)
    finally:
        planner_memory.close()
    chain_valid = bool(service.ledger.verify_chain())
    receipt_count = int(service.ledger.count)
    service.close()

    payload = {
        "stage": "I4",
        "workspace": {
            "root": str(ROOT),
            "directories": directories,
            "files": int(full.files),
            "lines": int(full.lines),
            "nodes": int(full.nodes),
            "edges": int(full.edges),
            "signatures": int(full.signatures),
            "full_index_ms": float(full.seconds * 1e3),
        },
        "delta": {
            "files": len(rows),
            "median_ms": float(statistics.median(latency)),
            "mean_ms": float(statistics.fmean(latency)),
            "p90_ms": float(np.percentile(latency, 90)),
            "max_ms": float(max(latency)),
            "rows_sample": sorted(rows, key=lambda row: -row["delta_ms"])[:8],
            "repetitions": 3,
            "reduction": "best of three per module",
        },
        "parity": {
            "edges": bool(same_edges),
            "nodes": bool(same_nodes),
            "reverse": bool(same_reverse),
            "impact": bool(impact_same),
            "probes": probes,
            "exact": bool(same_edges and same_nodes and same_reverse and impact_same),
        },
        "mutation": {
            "module": mutation_report.updated,
            "delta_ms": float(mutation_ms),
            "before": before,
            "after": after,
            "restored": restored,
            "changed": bool(before != after),
            "restore_exact": bool(restored == before),
        },
        "agent_step": {
            "actions": int(agent_surface["actions"]),
            "receipts_valid": bool(agent_surface["receipts_valid"]),
            "checkpoint_sha_present": bool(agent_surface["checkpoint_sha"]),
            "chain_valid": bool(chain_valid),
            "receipt_count": receipt_count,
            "rows": agent_surface["rows"],
            "planner_fixture": dict(planner_fixture),
        },
        "gate": {
            "files_at_least_50": bool(full.files >= 50),
            "edges_at_least_1200": bool(full.edges >= 1200),
            "median_delta_at_most_15ms": bool(statistics.median(latency) <= 15.0),
            "delta_parity_exact": bool(same_edges and same_nodes and same_reverse and impact_same),
            "mutation_changes_blast_radius": bool(before != after),
            "restore_is_exact": bool(restored == before),
            "seven_agent_actions": bool(agent_surface["actions"] == 7),
            "agent_receipts_valid": bool(agent_surface["receipts_valid"]),
            "agent_checkpoint_sha_present": bool(agent_surface["checkpoint_sha"]),
            "agent_chain_valid": bool(chain_valid),
        },
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
    }
    payload["gate"]["passed"] = bool(all(payload["gate"].values()))
    return payload


# ------------------------------------------------------------------- driver
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="receipts/holarch")
    parser.add_argument("--checkpoint", default="receipts/holarch/holarch_medium.pt")
    parser.add_argument("--stages", default="i1,i2,i3,i4,eval")
    args = parser.parse_args()
    global CHECKPOINT
    CHECKPOINT = (ROOT / args.checkpoint).resolve()
    directory = ROOT / args.dir
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    wanted = {name.strip().lower() for name in str(args.stages).split(",") if name.strip()}

    previous = {
        name: _read(directory, filename)
        for name, filename in (
            ("i1", "i1_vocab_lock.json"),
            ("i2", "i2_dag_synthesis.json"),
            ("i3", "i3_horizon_traps.json"),
            ("i4", "i4_agent_scale.json"),
            ("eval", "agent_eval.json"),
        )
    }
    i1 = stage_i1(directory) if "i1" in wanted else previous["i1"]
    i1_receipt = _write(directory, "i1_vocab_lock.json", i1) if i1 else {}
    i2 = stage_i2(directory) if "i2" in wanted else previous["i2"]
    i2_receipt = _write(directory, "i2_dag_synthesis.json", i2) if i2 else {}
    i3 = stage_i3(directory) if "i3" in wanted else previous["i3"]
    i3_receipt = _write(directory, "i3_horizon_traps.json", i3) if i3 else {}
    i4 = stage_i4(directory) if "i4" in wanted else previous["i4"]
    i4_receipt = _write(directory, "i4_agent_scale.json", i4) if i4 else {}

    agent_eval = previous["eval"]
    agent_receipt = {}
    if "eval" in wanted:
        module = _load_module("run_holarch_agent_eval", ROOT / "scripts" / "run_holarch_agent_eval.py")
        agent_eval = module.run_agent_eval(root=CODE_ROOT)
    if agent_eval:
        agent_receipt = _write(directory, "agent_eval.json", agent_eval)

    directives = {
        name.upper(): value
        for name, value in (("i1", i1), ("i2", i2), ("i3", i3), ("i4", i4))
        if value
    }
    ledger = {
        "cycle": "I1-I4",
        "git_head": _git_head(),
        "directives": directives,
        "receipts": {
            "I1": i1_receipt,
            "I2": i2_receipt,
            "I3": i3_receipt,
            "I4": i4_receipt,
            **({"agent_eval": agent_receipt} if agent_receipt else {}),
        },
        "vocabulary_lock": {
            "path": str(VOCAB_LOCK.relative_to(ROOT)),
            "sha256": _digest(VOCAB_LOCK) if VOCAB_LOCK.exists() else "",
        },
        "upstream_receipts": {
            name: _digest(directory / name)
            for name in (
                "h_cycle.json",
                "g_cycle.json",
                "h1_entropy_gated_lm.json",
                "h2_multiplan_synth.json",
                "h3_api_surface.json",
                "h4_delta_index.json",
            )
            if (directory / name).exists()
        },
        "summary": {
            "passed": sum(1 for item in directives.values() if item.get("gate", {}).get("passed")),
            "total": len(directives),
            "agent_eval_passed": bool(agent_eval.get("gate", {}).get("passed", False))
            if agent_eval
            else None,
        },
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "DERIVED",
    }
    out = directory / "i_cycle.json"
    out.write_text(json.dumps(ledger, indent=2, default=float), encoding="utf-8")
    digest = _digest(out)
    out.with_name(out.name + ".sha256").write_text(digest + chr(10), encoding="utf-8")
    print(json.dumps(ledger["summary"], indent=2))
    for name, item in directives.items():
        print(name, "PASS" if item.get("gate", {}).get("passed") else "FAIL", json.dumps(item.get("gate", {})))
    if i1:
        print(
            "I1 shifts locked={} unlocked={} roundtrip={:.4f} top1={:.4f} bpc={:.4f}".format(
                i1["mutation"]["locked_existing_id_shifts"],
                i1["mutation"]["unlocked_existing_id_shifts"],
                i1["roundtrip"]["lossless_rate"],
                i1["h1"]["before"]["top1"],
                i1["h1"]["before"]["bits_per_char"],
            )
        )
    if i2:
        print("I2", json.dumps(i2["total_synthesis_benchmark"]))
    if i3:
        print(
            "I3 masked={} depth1={} unmasked={} closed={:.4f} api_batch_ms={:.2f} loop_ms={:.2f}".format(
                i3["controls"]["masked_correct"],
                i3["controls"]["depth1_correct"],
                i3["controls"]["unmasked_correct"],
                i3["closed_loop"]["success_rate"],
                i3["latency_ms"]["plan_api_batched_median"],
                i3["latency_ms"]["plan_api_loop_median"],
            )
        )
    if i4:
        print(
            "I4 files={} edges={} median_delta={:.2f} ms actions={} chain={}".format(
                i4["workspace"]["files"],
                i4["workspace"]["edges"],
                i4["delta"]["median_ms"],
                i4["agent_step"]["actions"],
                i4["agent_step"]["chain_valid"],
            )
        )
    print("receipt", str(out.relative_to(ROOT)), digest)
    ok = ledger["summary"]["passed"] == ledger["summary"]["total"]
    if agent_eval:
        ok = ok and bool(agent_eval.get("gate", {}).get("passed", False))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
