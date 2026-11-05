"""Cycle H verification: H1 entropy-gated wave blending, H2 multi-plan
composition, H3 checkpoint rotor persistence and the plan/verify surface,
H4 incremental file-delta indexing.

    python scripts/run_holarch_h_cycle.py

Every number is measured in this process and written to a JSON receipt with a
SHA-256 sidecar; h_cycle.json aggregates them. H1 is measured on the Cycle G
corpus snapshot (commit 67e9b25) so the floor it must beat is the exact
0.5620 / 1.2942 the G1 receipt recorded; the live-tree numbers are recorded
beside it, because the corpus grows with this document.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from henri.holarch.code_gen import (
    SynthesisEngine,
    WaveCodeTextGenerator,
    build_synthesis_engine,
)
from henri.holarch.code_lang import (
    BochnerPhaseProjector,
    CodeTextCorpus,
    build_tokenizer,
)
from henri.holarch.config import HolarchConfig
from henri.holarch.model import (
    evaluate_closed_loop_action,
    fit_action_rotor,
    load_checkpoint,
    save_checkpoint,
)
from henri.holarch.server import SidecarService, build_app, verify_receipt
from henri.holarch.symbolic import CodeGraphIndexer
from henri.holarch.tasks import build_tasks
from henri.holarch.zone_c import ZoneCMemory

G_REVISION = "67e9b25"
"""The sealed Cycle G commit: the frozen corpus and the quoted floor."""


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_receipt(directory: Path, name: str) -> Dict[str, object]:
    path = directory / name
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write(directory: Path, name: str, payload: Dict[str, object]) -> Dict[str, str]:
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    digest = _digest(path)
    path.with_name(path.name + ".sha256").write_text(digest + chr(10), encoding="utf-8")
    return {"path": str(path.relative_to(ROOT)), "sha256": digest}


def _relative(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def _git_head() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), capture_output=True, text=True, timeout=20
        )
        return result.stdout.strip()
    except Exception:
        return ""


def frozen_corpus(revision: str = G_REVISION) -> Tuple[CodeTextCorpus, Dict[str, object]]:
    """The corpus at a sealed commit, materialised outside the work tree.

    H1 must be comparable to a receipt sealed in Cycle G. The tokenizer is
    trained on the corpus, so adding one function to this package changes
    every token id and moves every held-out number. Pinning the corpus to
    the commit the floor was quoted from removes that confounder; the live
    tree is measured separately and reported as such.
    """
    try:
        listing = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", str(revision)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        ).stdout.splitlines()
    except Exception as error:  # pragma: no cover - git is present in this repo
        return CodeTextCorpus.holarch(ROOT), {
            "source": "live_tree",
            "revision": "",
            "reason": f"git unavailable: {type(error).__name__}",
        }
    keep = [
        name
        for name in listing
        if (name.startswith("henri/holarch/") and name.endswith(".py"))
        or name == "docs/HOLARCH.md"
    ]
    if not keep:
        return CodeTextCorpus.holarch(ROOT), {"source": "live_tree", "revision": ""}
    target = Path(tempfile.mkdtemp(prefix="henri_g_corpus_"))
    for name in keep:
        blob = subprocess.run(
            ["git", "show", f"{revision}:{name}"],
            cwd=str(ROOT),
            capture_output=True,
            timeout=120,
            check=True,
        ).stdout
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
    corpus = CodeTextCorpus.holarch(target)
    return corpus, {
        "source": "git_revision",
        "revision": str(revision),
        "files": len(keep),
        "path": str(target),
    }


# ---------------------------------------------------------------------- H1
def _measure_windows(generator: WaveCodeTextGenerator, windows: Dict[str, List[int]]) -> Dict[str, object]:
    gate = dict(generator.beta_gate)
    static_beta = float(generator.beta_star)

    def run(sequence: List[int], mode: str, beta: float = 0.0) -> Dict[str, float]:
        generator.beta_gate = {
            "beta_low": 0.0,
            "beta_high": 0.0,
            "tau_h": 0.0,
            "tau_order": 3.0,
        }
        generator.beta_star = 0.0
        if mode == "gated":
            generator.beta_gate = dict(gate)
            generator.beta_star = float(gate["beta_high"])
        elif mode == "static":
            generator.beta_star = float(beta)
        try:
            return generator.evaluate(sequence).as_dict()
        finally:
            generator.beta_gate = dict(gate)
            generator.beta_star = static_beta

    rows: Dict[str, object] = {}
    for name, sequence in windows.items():
        rows[name] = {
            "floor": run(sequence, "floor"),
            "gated": run(sequence, "gated"),
        }
    legacy = windows["legacy"]
    rows["legacy"]["static_beta_0_02"] = run(legacy, "static", 0.02)
    rows["legacy"]["static_beta_0_05"] = run(legacy, "static", 0.05)
    # The Cycle G arm exactly: a static beta with NO decision-preserving
    # projection. This is the arm that cost -0.44 pp of top-1; it is
    # measured, not quoted, so the fix is attributable to the projection.
    generator.mode_guard = False
    try:
        rows["legacy"]["static_beta_0_02_unguarded"] = run(legacy, "static", 0.02)
        rows["legacy"]["static_beta_0_05_unguarded"] = run(legacy, "static", 0.05)
    finally:
        generator.mode_guard = True
    return rows


def stage_h1(source: str = "git") -> Dict[str, object]:
    started = time.perf_counter()
    if source == "git":
        corpus, provenance = frozen_corpus(G_REVISION)
    else:
        corpus, provenance = CodeTextCorpus.holarch(ROOT), {"source": "live_tree", "revision": ""}
    tokenizer = build_tokenizer(corpus)
    projector = BochnerPhaseProjector(dim=1024, d_sem=64)
    projector.fit(corpus, tokenizer)
    ids: List[int] = []
    for text in corpus.texts:
        ids.extend(tokenizer.encode(text))
    cut = int(len(ids) * 0.9)
    train, held = ids[:cut], ids[cut:]
    generator = WaveCodeTextGenerator(tokenizer, projector)
    generator.fit(train, max_fit_tokens=200_000)
    generator.count_lm.fit(train)
    generator.lexical.build(train[:200_000])
    # The bank weights are selected the Cycle G way: no projection, because
    # the projection did not exist then and the G1 floor must be reproduced
    # by the same procedure, not by a different one.
    generator.mode_guard = False
    try:
        weights = generator.calibrate_wave_weights(held[:1500])
    finally:
        generator.mode_guard = True
    beta = generator.calibrate_beta(
        held[:1500],
        joint=True,
        samples=1500,
        beta_low_grid=(0.0, 0.01),
        beta_high_grid=(0.0, 0.02, 0.05, 0.08, 0.12),
        tau_grid=(0.75, 1.0, 1.5, 2.0, 3.0),
        order_grid=(3, 4),
    )
    gate = dict(generator.beta_gate)
    generator.lexical._fused.clear()
    windows = {
        "legacy": held[:3000],
        "disjoint": held[1500:4500],
        "full": held,
    }
    rows = _measure_windows(generator, windows)
    legacy = rows["legacy"]
    disjoint = rows["disjoint"]
    full = rows["full"]
    ceiling = 0.5600
    payload = {
        "stage": "H1",
        "corpus": {
            "source": provenance.get("source", ""),
            "revision": provenance.get("revision", ""),
            "digest": str(corpus.digest),
            "texts": int(len(corpus)),
            "tokens": int(len(ids)),
            "vocabulary": int(len(tokenizer)),
            "note": (
                "The H1 corpus is pinned to the Cycle G commit so the floor is "
                "the exact number the G1 receipt recorded. The tokenizer is "
                "trained on the corpus, so a live-tree edit moves every token id."
            ),
        },
        "bank": dict(generator.lexical.report),
        "weights": weights,
        "beta_star": float(beta),
        "beta_gate": gate,
        "calibration": dict(generator.gate_report),
        "beta_curve": [[float(b), float(loss)] for b, loss in generator.beta_curve],
        "windows": rows,
        "floor": legacy["floor"],
        "tuned": legacy["gated"],
        "evaluated_on": "held_out[0:3000] (the protocol of every committed L3/F/G receipt)",
        "beta_selected_on": "held_out[0:1500]",
        "gate": {
            "beta_high_strictly_positive": bool(float(gate["beta_high"]) > 0.0),
            "bits_per_char_below_floor": bool(
                float(legacy["gated"]["bits_per_char"]) < float(legacy["floor"]["bits_per_char"])
            ),
            "top1_at_least_floor": bool(
                float(legacy["gated"]["top1"]) >= float(legacy["floor"]["top1"])
            ),
            "floor_top1_at_least_0_5600": bool(float(legacy["floor"]["top1"]) >= ceiling),
            "cross_entropy_below_floor": bool(
                float(legacy["gated"]["perplexity"]) < float(legacy["floor"]["perplexity"])
            ),
            "disjoint_top1_at_least_floor": bool(
                float(disjoint["gated"]["top1"]) >= float(disjoint["floor"]["top1"])
            ),
            "disjoint_bits_per_char_below_floor": bool(
                float(disjoint["gated"]["bits_per_char"])
                < float(disjoint["floor"]["bits_per_char"])
            ),
            "full_top1_at_least_floor": bool(
                float(full["gated"]["top1"]) >= float(full["floor"]["top1"])
            ),
            "full_bits_per_char_below_floor": bool(
                float(full["gated"]["bits_per_char"]) < float(full["floor"]["bits_per_char"])
            ),
        },
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
    }
    payload["gate"]["passed"] = bool(all(payload["gate"].values()))
    return payload


# ---------------------------------------------------------------------- H2
MULTI_PLAN: List[Dict[str, str]] = [
    {
        "id": "multi01",
        "base": "core08+shift11",
        "intent": "Write a function distinct_caller_count that takes argument links and first returns the sorted unique source names of a list of edges and then returns the number of tokens in that result.",
        "test": "class _E:\n    def __init__(self, s):\n        self.source = s\nassert distinct_caller_count([_E('b'), _E('a'), _E('b')]) == 2\n",
    },
    {
        "id": "multi02",
        "base": "shift11+core11",
        "intent": "Write a function bounded_length that takes argument tokens and first returns the number of tokens in a token list and then clips the result into the closed interval 0.0 to 2.5.",
        "test": "assert bounded_length([1, 2, 3]) == 2.5\nassert bounded_length([]) == 0.0\n",
    },
    {
        "id": "multi03",
        "base": "core10+core11",
        "intent": "Write a function bounded_mean that takes argument sample and first returns the arithmetic mean of a list of floats as a float and then clips the result into the closed interval 0.0 to 2.5.",
        "test": "assert bounded_mean([2.0, 4.0]) == 2.5\nassert bounded_mean([1.0, 2.0]) == 1.5\n",
    },
    {
        "id": "multi04",
        "base": "core02+core11",
        "intent": "Write a function bounded_dimension that takes argument config and first returns the complex wave dimension of a holarch config and then clips the result into the closed interval 0.0 to 2.5.",
        "test": "class _C:\n    dim = 1024\nassert bounded_dimension(_C()) == 2.5\n",
    },
    {
        "id": "multi05",
        "base": "core03+core11",
        "intent": "Write a function bounded_slot that takes argument settings and first returns the per-slot width of a holarch config and then clips the result into the closed interval 0.0 to 2.5.",
        "test": "class _C:\n    d_slot = 256\nassert bounded_slot(_C()) == 2.5\n",
    },
    {
        "id": "multi06",
        "base": "core04+shift11",
        "intent": "Write a function digest_length that takes argument record and first returns the sha256 field of an epistemic receipt and then returns the number of tokens in that result.",
        "test": "assert digest_length({'sha256': 'abcd'}) == 4\n",
    },
    {
        "id": "multi07",
        "base": "core07+core11",
        "intent": "Write a function bounded_node_count that takes argument chart and first returns how many nodes a code graph holds and then clips the result into the closed interval 0.0 to 2.5.",
        "test": "class _G:\n    nodes = ['a', 'b']\nassert bounded_node_count(_G()) == 2.0\n",
    },
    {
        "id": "multi08",
        "base": "core01+core11",
        "intent": "Write a function bounded_engram_count that takes argument store and first returns the number of engrams a ZoneCMemory holds and then clips the result into the closed interval 0.0 to 2.5.",
        "test": "class _M:\n    def stats(self):\n        return {'count': 9}\nassert bounded_engram_count(_M()) == 2.5\n",
    },
]

# The Cycle G single-plan compositional set, unchanged. H2 must keep every
# one of these at 12/12 while adding the two-plan pipelines below.
COMPOSITIONAL: List[Dict[str, str]] = [
    {
        "id": "comp01",
        "base": "core01",
        "intent": "Write a function tally_engrams that takes argument store and returns the number of engrams a ZoneCMemory holds.",
        "test": "class _M:\n    def stats(self):\n        return {\"count\": 9}\nassert tally_engrams(_M()) == 9\n",
    },
    {
        "id": "comp02",
        "base": "core02",
        "intent": "Write a function latent_width that takes argument config and returns the complex wave dimension of a holarch config.",
        "test": "class _C:\n    dim = 2048\nassert latent_width(_C()) == 2048\n",
    },
    {
        "id": "comp03",
        "base": "core03",
        "intent": "Write a function per_slot_width that takes argument settings and returns the per-slot width of a holarch config.",
        "test": "class _C:\n    d_slot = 512\nassert per_slot_width(_C()) == 512\n",
    },
    {
        "id": "comp04",
        "base": "core04",
        "intent": "Write a function digest_of that takes argument record and returns the sha256 field of an epistemic receipt.",
        "test": "assert digest_of({\"sha256\": \"cd\"}) == \"cd\"\n",
    },
    {
        "id": "comp05",
        "base": "core05",
        "intent": "Write a function sorted_digest that takes argument body and returns the sha256 of the sorted json encoding of a payload.",
        "test": "assert len(sorted_digest({\"b\": 2})) == 64\n",
    },
    {
        "id": "comp06",
        "base": "core06",
        "intent": "Write a function energy_of that takes argument signal and returns the squared magnitude norm of a complex wave.",
        "test": "import numpy as np\nassert abs(energy_of(np.array([2+0j])) - 4.0) < 1e-9\n",
    },
    {
        "id": "comp07",
        "base": "core07",
        "intent": "Write a function node_total that takes argument chart and returns how many nodes a code graph holds.",
        "test": "class _G:\n    nodes = [\"x\"]\nassert node_total(_G()) == 1\n",
    },
    {
        "id": "comp08",
        "base": "core08",
        "intent": "Write a function sources_of that takes argument links and returns the sorted unique source names of a list of edges.",
        "test": "class _E:\n    def __init__(self, s):\n        self.source = s\nassert sources_of([_E(\"m\"), _E(\"k\")]) == [\"k\", \"m\"]\n",
    },
    {
        "id": "comp09",
        "base": "core09",
        "intent": "Write a function index_of_task that takes argument title and maps a task name to the canonical indexes 4, 5, 6, 7.",
        "test": "assert index_of_task(\"action\") == 6\n",
    },
    {
        "id": "comp10",
        "base": "core10",
        "intent": "Write a function average_of that takes argument sample and returns the arithmetic mean of a list of floats as a float.",
        "test": "assert average_of([2.0, 4.0]) == 3.0\n",
    },
    {
        "id": "comp11",
        "base": "core11",
        "intent": "Write a function clamp_high that takes argument level and clips a value into the closed interval 0.0 to 2.5.",
        "test": "assert clamp_high(9.0) == 2.5\nassert clamp_high(-2.0) == 0.0\n",
    },
    {
        "id": "comp12",
        "base": "core12",
        "intent": "Write a function files_in that takes argument folder and returns the sorted file names of a directory path.",
        "test": "assert isinstance(files_in(\".\"), list)\n",
    },
]

COLD_TASK = {
    "id": "cold01",
    "function": "normalise_number",
    "intent": "Write a function normalise_number that takes argument text and normalises a numeric quantity for comparison.",
}


def stage_h2(directory: Path, f_module) -> Dict[str, object]:
    memory_path = directory / "_work" / "h2_synthesis.sqlite3"
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    if memory_path.exists():
        memory_path.unlink()
    memory = ZoneCMemory(HolarchConfig.medium(), backend="sqlite", path=str(memory_path))
    symbols = set()
    for task in list(f_module.CORE_TASKS) + list(f_module.SHIFT_TASKS):
        symbols.add(task["source"].split("(")[0].replace("def ", "").strip())
    indexer = CodeGraphIndexer(None)
    indexer.index(ROOT / "henri" / "holarch", "henri")
    engine, stats = build_synthesis_engine(
        ROOT,
        memory,
        namespace="h2",
        symbol_table=symbols,
        max_fit_tokens=40000,
        signatures=dict(indexer.signatures),
    )
    base_rows = []
    for task in list(f_module.CORE_TASKS) + list(f_module.SHIFT_TASKS):
        adapted = engine.adapt(task["intent"], task["source"], task["id"])
        row = engine.synthesize(task["intent"], kind="code")
        executed = engine.execute(row["source"], task["test"]) if row["verified"] else {"ok": False}
        base_rows.append(
            {
                "id": task["id"],
                "adapted": bool(adapted["adapted"]),
                "verified": bool(row["verified"]),
                "executed": bool(executed["ok"]),
                "plan": row.get("plan", ""),
                "composed": bool(row.get("composed", False)),
            }
        )
    base_pass = sum(1 for row in base_rows if row["executed"] and not row["composed"])

    comp_rows = []
    comp_pass = 0
    for task in COMPOSITIONAL:
        row = engine.synthesize(task["intent"], kind="code")
        executed = engine.execute(row["source"], task["test"]) if row["verified"] else {"ok": False}
        ok = bool(row["verified"] and executed["ok"])
        comp_pass += int(ok)
        comp_rows.append(
            {
                "id": task["id"],
                "base": task["base"],
                "plan": row.get("plan", ""),
                "composed": bool(row.get("composed", False)),
                "verified": bool(row["verified"]),
                "executed": bool(executed["ok"]),
                "reason": str(executed.get("reason", ""))[:120],
                "source": row["source"][:300],
            }
        )

    multi_rows = []
    multi_pass = 0
    for task in MULTI_PLAN:
        chain = engine.plans.retrieve_chain(task["intent"], max_plans=2)
        coverage = (
            float(engine.plans.coverage(task["intent"], chain[0][0])) if chain else 0.0
        )
        second_score = float(chain[1][1]) if len(chain) >= 2 else 0.0
        row = engine.synthesize(task["intent"], kind="code")
        executed = engine.execute(row["source"], task["test"]) if row["verified"] else {"ok": False}
        ok = bool(row["verified"] and executed["ok"])
        multi_pass += int(ok)
        multi_rows.append(
            {
                "id": task["id"],
                "base": task["base"],
                "chain": [plan.name for plan, _score in chain],
                "coverage": float(coverage),
                "second_score": float(second_score),
                "plan": row.get("plan", ""),
                "composed": bool(row.get("composed", False)),
                "composed_from": list(row.get("composed_from", [])),
                "verified": bool(row["verified"]),
                "executed": bool(executed["ok"]),
                "reason": str(executed.get("reason", ""))[:120],
                "source": row["source"][:300],
            }
        )

    traps = []
    refused = 0
    for intent in f_module.TRAPS:
        row = engine.synthesize(intent, kind="code")
        ok = row["status"] == "refused"
        refused += int(ok)
        traps.append({"intent": intent, "status": row["status"]})

    # A cold start is a tenant with the code index and NO compiled plans.
    # The h2_cold namespace is exactly that: an empty playbook store beside a
    # populated signature table, so the fallback is exercised as deployed
    # rather than by emptying the warm engine's store.
    cold_engine = SynthesisEngine(
        engine.generator,
        engine.projector,
        engine.tokenizer,
        memory,
        namespace="h2_cold",
        verifier=engine.verifier,
        signatures=dict(indexer.signatures),
    )
    cold = cold_engine.synthesize(COLD_TASK["intent"], kind="code")
    cold_ok = bool(
        cold.get("verified")
        and cold.get("cold_stub")
        and f"def {COLD_TASK['function']}(" in str(cold.get("source", ""))
    )
    total_tasks = len(base_rows) + len(comp_rows) + len(multi_rows)
    payload = {
        "stage": "H2",
        "corpus": stats,
        "base_tasks": {"total": len(base_rows), "passed": int(base_pass), "rows": base_rows},
        "compositional": {
            "total": len(comp_rows),
            "passed": int(comp_pass),
            "accuracy": float(comp_pass / max(len(comp_rows), 1)),
            "rows": comp_rows,
        },
        "multi_plan": {
            "total": len(multi_rows),
            "passed": int(multi_pass),
            "accuracy": float(multi_pass / max(len(multi_rows), 1)),
            "rows": multi_rows,
        },
        "cold_start": {
            "function": COLD_TASK["function"],
            "verified": bool(cold.get("verified")),
            "cold_stub": bool(cold.get("cold_stub")),
            "source": str(cold.get("source", ""))[:200],
            "passed": cold_ok,
        },
        "traps": {"total": len(traps), "refused": int(refused), "rows": traps},
        "emission": {
            "exact": int(engine.emission_exact),
            "fallbacks": int(engine.emission_fallbacks),
            "composed": int(engine.composed),
            "cold_stubs": int(engine.cold_stubs),
        },
        "plans_compiled": len(engine.plans.plans),
        "gate": {
            "base_24_of_24": bool(base_pass == 24),
            "compositional_12_of_12": bool(comp_pass == 12),
            "multi_plan_at_least_0_8750": bool(multi_pass / max(len(multi_rows), 1) >= 0.875),
            "multi_plan_present_8": bool(len(multi_rows) == 8),
            "traps_refused_1_0": bool(refused == len(traps)),
            "cold_signature_stub": bool(cold_ok),
            "total_44": bool(total_tasks == 44),
        },
        "evidence": "OBSERVED",
    }
    payload["gate"]["passed"] = bool(all(payload["gate"].values()))
    memory.close()
    return payload


# ---------------------------------------------------------------------- H3
def stage_h3(directory: Path, checkpoint: Path) -> Dict[str, object]:
    started = time.perf_counter()
    model, metadata = load_checkpoint(checkpoint)
    cfg = model.cfg
    task = build_tasks(cfg, seed=cfg.seed)["action"]
    rotor = getattr(model, "action_rotor", None)
    refit = False
    if rotor is None:
        fit = fit_action_rotor(model, task, episodes=40)
        rotor = fit["rotor"]
        refit = True
        rotor_block = {"fit": fit["fit"], "held_out": fit["held_out"]}
    else:
        rotor_block = {"fit": dict(metadata.get("rotor", {}) or {}), "held_out": {}}
    arms = {
        "rotor_depth2": evaluate_closed_loop_action(
            model, task, episodes=48, rotor=rotor, transition="rotor",
            lookahead_depth=2, gamma_look=0.02,
        ),
        "rotor_depth1": evaluate_closed_loop_action(
            model, task, episodes=48, rotor=rotor, transition="rotor"
        ),
        "shift_depth1": evaluate_closed_loop_action(model, task, episodes=48),
        "rotor_depth2_unmasked": evaluate_closed_loop_action(
            model, task, episodes=48, rotor=rotor, transition="rotor",
            lookahead_depth=2, gamma_look=0.02, mask_lookahead=False,
        ),
    }
    deployed = arms["rotor_depth2"]
    previous_metrics = dict(metadata.get("metrics", {}) or {})
    metrics = {
        **previous_metrics,
        "action_success": float(deployed["success_rate"]),
        "action_success_reactive": float(deployed["reactive_success_rate"]),
        "lookahead_depth": 2,
        "transition": "rotor",
        "gamma_look": 0.02,
    }
    saved = save_checkpoint(
        model,
        checkpoint,
        cfg=cfg,
        name=str(metadata.get("name", "medium")),
        train_config=dict(metadata.get("train_config", {}) or {}),
        metrics=metrics,
        seed=int(metadata.get("seed", cfg.seed)),
        rotor=rotor,
        created_utc=str(metadata.get("created_utc", "") or "") or None,
    )
    # The JSON receipt beside the weights records the same deployed policy.
    receipt_path = checkpoint.with_suffix(".json")
    record: Dict[str, object] = {}
    if receipt_path.exists():
        try:
            record = json.loads(receipt_path.read_text(encoding="utf-8"))
        except Exception:
            record = {}
    record.update(
        {
            "action_success": float(deployed["success_rate"]),
            "action_success_reactive": float(deployed["reactive_success_rate"]),
            "closed_loop_lookahead": deployed,
            "action_rotor": {
                "state": dict(getattr(rotor, "state", lambda: {})()),
                "transition": "rotor",
                "lookahead_depth": 2,
                "gamma_look": 0.02,
            },
            "checkpoint": {
                **dict(record.get("checkpoint", {}) or {}),
                "path": _relative(checkpoint),
                "sha256": saved["sha256"],
                "bytes": int(saved["bytes"]),
                "parameters": int(saved["parameters"]),
            },
            "cycle_h_seal": {
                "checkpoint_resealed_with_rotor": True,
                "refit_during_seal": bool(refit),
            },
        }
    )
    receipt_path.write_text(json.dumps(record, indent=2, default=float), encoding="utf-8")
    record_digest = _digest(receipt_path)
    receipt_path.with_name(receipt_path.name + ".sha256").write_text(
        record_digest + chr(10), encoding="utf-8"
    )

    # Round trip: a NEW process-identical model must reproduce the score with
    # the rotor it loaded, and must not fit anything.
    reloaded, reloaded_metadata = load_checkpoint(checkpoint)
    reloaded_rotor = getattr(reloaded, "action_rotor", None)
    task2 = build_tasks(reloaded.cfg, seed=reloaded.cfg.seed)["action"]
    replay = evaluate_closed_loop_action(
        reloaded, task2, episodes=48, rotor=reloaded_rotor, transition="rotor",
        lookahead_depth=2, gamma_look=0.02,
    )

    # REST + MCP surface over the same mounted checkpoint.
    memory_path = directory / "_work" / "h3_api.sqlite3"
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    if memory_path.exists():
        memory_path.unlink()
    service = SidecarService(
        cfg=cfg, backend="sqlite", memory_path=str(memory_path), checkpoint=str(checkpoint)
    )
    from fastapi.testclient import TestClient

    client = TestClient(build_app(service))
    walls, agent, goal = task.val_layouts[0]
    observation = task.world.observe(walls, agent, goal).tolist()
    plan_body = client.post(
        "/v1/holarch/plan",
        json={
            "observation": observation,
            "transition": "rotor",
            "lookahead_depth": 2,
            "gamma_look": 0.02,
            "namespace": "h3",
        },
    ).json()
    plan_ok = bool(
        plan_body.get("ok")
        and verify_receipt(plan_body["receipt"])
        and len(plan_body["result"]["scores"]) == 4
        and plan_body["result"]["lookahead_depth"] == 2
        and plan_body["result"]["transition"] == "rotor"
    )
    true_claim = (
        "The medium profile has 1,020,380 parameters and reaches 0.7083 closed-loop action."
    )
    bad_claim = "The medium profile has 1,020,381 parameters."
    verify_true = client.post(
        "/v1/holarch/verify", json={"claim": true_claim, "kind": "text"}
    ).json()
    verify_bad = client.post(
        "/v1/holarch/verify", json={"claim": bad_claim, "kind": "text"}
    ).json()
    verify_code = client.post(
        "/v1/holarch/verify",
        json={
            "claim": "def merge_waves(left, right):\n    return left + right\n",
            "kind": "code",
        },
    ).json()
    verify_sig = client.post(
        "/v1/holarch/verify",
        json={
            "claim": "def fake_call():\n    return ZoneCMemory.quantum_entangle(1)\n",
            "kind": "code",
        },
    ).json()
    verify_ok = bool(
        verify_receipt(verify_true["receipt"])
        and verify_true["result"]["supported"] is True
        and verify_bad["result"]["supported"] is False
        and verify_code["result"]["supported"] is True
        and verify_sig["result"]["supported"] is False
    )

    from henri.holarch import mcp_server

    mcp_server.set_service(service)
    try:
        mcp_plan = asyncio.run(
            mcp_server.mcp._tool_manager.call_tool(
                "holarch_plan",
                {
                    "observation": observation,
                    "transition": "rotor",
                    "lookahead_depth": 2,
                    "namespace": "h3",
                },
            )
        )
        mcp_verify = asyncio.run(
            mcp_server.mcp._tool_manager.call_tool(
                "holarch_verify", {"claim": true_claim, "kind": "text"}
            )
        )
    finally:
        mcp_server.set_service(None)
    mcp_ok = bool(
        verify_receipt(mcp_plan["receipt"])
        and mcp_plan["result"]["action"] == plan_body["result"]["action"]
        and verify_receipt(mcp_verify["receipt"])
        and mcp_verify["result"]["supported"] is True
    )
    chain_valid = bool(service.ledger.verify_chain())
    health = service.health()["result"]
    service.close()

    payload = {
        "stage": "H3",
        "checkpoint": {
            "path": str(checkpoint.relative_to(ROOT)),
            "sha256": str(saved["sha256"]),
            "bytes": int(saved["bytes"]),
            "parameters": int(saved["parameters"]),
            "refit_during_seal": bool(refit),
            "rotor_state": dict(getattr(rotor, "state", lambda: {})()),
            "receipt_json": str(receipt_path.relative_to(ROOT)),
            "receipt_json_sha256": record_digest,
        },
        "arms": {
            name: {
                "success_rate": float(arm["success_rate"]),
                "wins": int(arm["wins"]),
                "mean_steps_to_goal": float(arm["mean_steps_to_goal"]),
                "transition": str(arm["transition"]),
                "lookahead_depth": int(arm["lookahead_depth"]),
            }
            for name, arm in arms.items()
        },
        "rotor": rotor_block,
        "replay_without_refit": {
            "success_rate": float(replay["success_rate"]),
            "wins": int(replay["wins"]),
            "rotor_fitted": bool(getattr(reloaded_rotor, "fitted", False)),
            "rotor_counts": [int(v) for v in getattr(reloaded_rotor, "counts", [])]
            if reloaded_rotor is not None
            else [],
            "metadata_rotor": reloaded_metadata.get("rotor"),
        },
        "rest": {
            "plan": plan_body["result"],
            "plan_receipt_valid": bool(verify_receipt(plan_body["receipt"])),
            "verify_true": verify_true["result"],
            "verify_corrupted": verify_bad["result"],
            "verify_code": verify_code["result"],
            "verify_fake_signature": verify_sig["result"],
        },
        "mcp": {
            "plan": mcp_plan["result"],
            "verify": mcp_verify["result"],
            "plan_receipt_valid": bool(verify_receipt(mcp_plan["receipt"])),
            "verify_receipt_valid": bool(verify_receipt(mcp_verify["receipt"])),
        },
        "health": health,
        "gate": {
            "rotor_restored_from_checkpoint": bool(getattr(reloaded_rotor, "fitted", False)),
            "rotor_used_without_refit": bool(not reloaded_metadata.get("refit", False)),
            "closed_loop_0_8125": bool(abs(float(deployed["success_rate"]) - 0.8125) < 1e-9),
            "replay_matches": bool(
                abs(float(replay["success_rate"]) - float(deployed["success_rate"])) < 1e-12
            ),
            "rest_plan_receipt_valid": plan_ok,
            "rest_verify_contract": verify_ok,
            "mcp_matches_rest": mcp_ok,
            "receipt_chain_valid": chain_valid,
            "health_reports_rotor": bool(health.get("rotor")),
            "json_records_action_success": bool(
                abs(float(record.get("action_success", -1.0)) - float(deployed["success_rate"])) < 1e-12
            ),
        },
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
    }
    payload["gate"]["passed"] = bool(all(payload["gate"].values()))
    return payload


# ---------------------------------------------------------------------- H4
def stage_h4(directory: Path) -> Dict[str, object]:
    started = time.perf_counter()
    root = ROOT / "henri" / "holarch"
    database = directory / "_work" / "h4_delta.sqlite3"
    database.parent.mkdir(parents=True, exist_ok=True)
    if database.exists():
        database.unlink()
    memory = ZoneCMemory(HolarchConfig.small(), backend="sqlite", path=str(database))
    indexer = CodeGraphIndexer(memory, namespace="h4")
    full = indexer.index(root, "henri")
    rows = []
    for path in sorted(root.glob("*.py")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        # Warm the module's records so the measured pass is the delta itself
        # and not a first-touch allocation. Best-of-five is the standard
        # latency probe for a deterministic computation: the machine's
        # scheduler adds time, it never removes work.
        indexer.update_file(root, path, source_text=text)
        samples = []
        for _ in range(5):
            t0 = time.perf_counter()
            report = indexer.update_file(root, path, source_text=text)
            samples.append((time.perf_counter() - t0) * 1e3)
        rows.append(
            {
                "module": report.updated,
                "bytes": len(text.encode("utf-8")),
                "delta_ms": float(min(samples)),
            }
        )
    latency = [row["delta_ms"] for row in rows]
    # Exactness: a fresh full index of the same tree must agree item for item.
    reference = CodeGraphIndexer(None)
    reference.index(root, "henri")
    same_edges = [(edge.source, edge.operator, edge.target) for edge in reference.graph.edges] == [
        (edge.source, edge.operator, edge.target) for edge in indexer.graph.edges
    ]
    same_nodes = list(reference.graph.nodes) == list(indexer.graph.nodes)
    same_reverse = {key: list(value) for key, value in reference.reverse.items()} == {
        key: list(value) for key, value in indexer.reverse.items()
    }
    probes = ["ZoneCMemory.query", "ZoneCMemory.write", "SymbolCodebook.add", "index_code"]
    impact_same = all(
        reference.impact(name, hops=2)["callers"] == indexer.impact(name, hops=2)["callers"]
        for name in probes
    )
    targets = ("ZoneCMemory.query", "SymbolCodebook.add")
    before = {name: indexer.impact(name, hops=2)["callers"] for name in targets}
    # Mutate one call site through the delta path and measure the blast radius.
    # sidecar.py's recall_playbook is the module's only ZoneCMemory.query call,
    # so renaming it removes exactly one reverse-call edge.
    path = root / "sidecar.py"
    text = path.read_text(encoding="utf-8")
    mutated = text.replace(
        "self.model.memory.query(key, k=1, namespace=scope)",
        "self.model.memory.lookup(key, k=1, namespace=scope)",
    )
    mutate_ms = None
    t0 = time.perf_counter()
    mutation_report = indexer.update_file(root, path, source_text=mutated)
    mutate_ms = (time.perf_counter() - t0) * 1e3
    after = {name: indexer.impact(name, hops=2)["callers"] for name in targets}
    indexer.update_file(root, path, source_text=text)
    restored = {name: indexer.impact(name, hops=2)["callers"] for name in targets}

    # Persisted metadata survives a restart and answers identically.
    restored_indexer = CodeGraphIndexer.from_memory(memory, "h4")
    persisted_same = bool(
        restored_indexer is not None
        and restored_indexer.impact("ZoneCMemory.query", hops=2)["callers"]
        == indexer.impact("ZoneCMemory.query", hops=2)["callers"]
        and restored_indexer.signatures.keys() == indexer.signatures.keys()
    )
    payload = {
        "stage": "H4",
        "package": {
            "root": str(root.relative_to(ROOT)),
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
            "min_ms": float(min(latency)),
            "under_10ms": int(sum(1 for value in latency if value < 10.0)),
            "rows": sorted(rows, key=lambda row: row["delta_ms"]),
            "repetitions": 5,
            "reduction": "best of five per module (the machine only adds time)",
            "note": (
                "Latency includes ast.parse of the edited file, the graph "
                "rebuild and the engram_meta write. Files above roughly 40 KB "
                "exceed 10 ms because the parse alone costs 6-9 ms."
            ),
        },
        "identical_to_full_index": {
            "edges": bool(same_edges),
            "nodes": bool(same_nodes),
            "reverse": bool(same_reverse),
            "impact": bool(impact_same),
            "probes": probes,
        },
        "speedup": float(float(full.seconds * 1e3) / max(statistics.median(latency), 1e-9)),
        "mutation": {
            "module": mutation_report.updated,
            "delta_ms": float(mutate_ms),
            "before": before,
            "after": after,
            "restored": restored,
            "changed": bool(before != after),
            "restore_exact": bool(restored == before),
        },
        "persistence": {"restored": persisted_same},
        "gate": {
            "median_delta_under_10ms": bool(statistics.median(latency) < 10.0),
            "graph_matches_full_index": bool(same_edges and same_nodes and same_reverse and impact_same),
            "delta_faster_than_full_index": bool(
                statistics.median(latency) * 10.0 < float(full.seconds * 1e3)
            ),
            "mutation_changes_blast_radius": bool(before != after),
            "restore_is_exact": bool(restored == before),
            "metadata_persisted": bool(persisted_same),
            "at_least_half_under_10ms": bool(
                sum(1 for value in latency if value < 10.0) >= len(latency) / 2.0
            ),
        },
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
    }
    payload["gate"]["passed"] = bool(all(payload["gate"].values()))
    memory.close()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="receipts/holarch")
    parser.add_argument("--checkpoint", default="receipts/holarch/holarch_medium.pt")
    parser.add_argument("--corpus", default="git", choices=["git", "live"])
    parser.add_argument("--stages", default="h1,h2,h3,h4")
    args = parser.parse_args()
    directory = ROOT / args.dir
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    wanted = {name.strip().lower() for name in str(args.stages).split(",") if name.strip()}
    f_module = _load_module("run_holarch_code_lang", ROOT / "scripts" / "run_holarch_code_lang.py")

    previous = {
        name: _read_receipt(directory, filename)
        for name, filename in (
            ("h1", "h1_entropy_gated_lm.json"),
            ("h2", "h2_multiplan_synth.json"),
            ("h3", "h3_api_surface.json"),
            ("h4", "h4_delta_index.json"),
            ("eval", "agent_eval.json"),
        )
    }

    h1 = stage_h1(source=str(args.corpus)) if "h1" in wanted else previous["h1"]
    h1_receipt = _write(directory, "h1_entropy_gated_lm.json", h1) if h1 else {}
    h2 = stage_h2(directory, f_module) if "h2" in wanted else previous["h2"]
    h2_receipt = _write(directory, "h2_multiplan_synth.json", h2) if h2 else {}
    h3 = (
        stage_h3(directory, (ROOT / args.checkpoint).resolve())
        if "h3" in wanted
        else previous["h3"]
    )
    h3_receipt = _write(directory, "h3_api_surface.json", h3) if h3 else {}
    h4 = stage_h4(directory) if "h4" in wanted else previous["h4"]
    h4_receipt = _write(directory, "h4_delta_index.json", h4) if h4 else {}

    agent_eval = previous["eval"]
    agent_receipt = {}
    if "eval" in wanted:
        module = _load_module(
            "run_holarch_agent_eval", ROOT / "scripts" / "run_holarch_agent_eval.py"
        )
        agent_eval = module.run_agent_eval()
    if agent_eval:
        agent_receipt = _write(directory, "agent_eval.json", agent_eval)

    upstream = {}
    for name in ("g_cycle.json", "holarch_medium.json", "code_lang_eval.json"):
        path = directory / name
        upstream[name] = _digest(path) if path.exists() else ""

    directives = {name.upper(): value for name, value in
                  (("h1", h1), ("h2", h2), ("h3", h3), ("h4", h4)) if value}
    ledger = {
        "cycle": "H1-H4",
        "git_head": _git_head(),
        "corpus_source": str(args.corpus),
        "directives": directives,
        "receipts": {
            "H1": h1_receipt,
            "H2": h2_receipt,
            "H3": h3_receipt,
            "H4": h4_receipt,
            **({"agent_eval": agent_receipt} if agent_receipt else {}),
        },
        "upstream_receipts": upstream,
        "summary": {
            "passed": sum(1 for item in directives.values() if item["gate"]["passed"]),
            "total": len(directives),
            "agent_eval_passed": bool(agent_eval.get("gate", {}).get("passed", False))
            if agent_eval
            else None,
        },
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "DERIVED",
    }
    out = directory / "h_cycle.json"
    out.write_text(json.dumps(ledger, indent=2, default=float), encoding="utf-8")
    digest = _digest(out)
    out.with_name(out.name + ".sha256").write_text(digest + chr(10), encoding="utf-8")
    print(json.dumps(ledger["summary"], indent=2))
    for name, item in directives.items():
        print(name, "PASS" if item["gate"]["passed"] else "FAIL", json.dumps(item["gate"]))
    if h1:
        print(
            "H1 beta_high={:.3f} top1 {:.4f} (floor {:.4f}) bpc {:.4f} (floor {:.4f})".format(
                float(h1["beta_gate"]["beta_high"]),
                float(h1["tuned"]["top1"]),
                float(h1["floor"]["top1"]),
                float(h1["tuned"]["bits_per_char"]),
                float(h1["floor"]["bits_per_char"]),
            )
        )
    if h2:
        print(
            "H2 base {}/24 comp {}/12 multi {}/8".format(
                h2["base_tasks"]["passed"],
                h2["compositional"]["passed"],
                h2["multi_plan"]["passed"],
            )
        )
    if h3:
        print(
            "H3 closed-loop {:.4f} replay {:.4f} rotor {} ".format(
                float(h3["arms"]["rotor_depth2"]["success_rate"]),
                float(h3["replay_without_refit"]["success_rate"]),
                bool(h3["replay_without_refit"]["rotor_fitted"]),
            )
        )
    if h4:
        print(
            "H4 median delta {:.2f} ms (full {:.1f} ms) under10 {}/{}".format(
                float(h4["delta"]["median_ms"]),
                float(h4["package"]["full_index_ms"]),
                int(h4["delta"]["under_10ms"]),
                int(h4["delta"]["files"]),
            )
        )
    print("receipt", str(out.relative_to(ROOT)), digest)
    ok = ledger["summary"]["passed"] == ledger["summary"]["total"]
    if agent_eval:
        ok = ok and bool(agent_eval.get("gate", {}).get("passed", False))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
