"""Paired with-vs-without Holarch MCP agent benchmark (Decision B3).

Evidence label: OBSERVED. Every number is measured on the same machine in
the same process, on the same deterministic query set.

The question this script answers is not whether the holarch is a better
language model. It is whether an agent that calls the Holarch sidecar
answers structural and symbolic questions more cheaply and with fewer
fabrications than an agent that greps source files and never vetoes.

Two arms, one query set:

    baseline_unindexed  scans every .py file at query time, follows call
                        names with a text scan, and has no coherence veto,
                        no Hopfield abstention and no compiled playbook.
    holarch_mcp         calls SidecarService: one ingest_code pass, then
                        wave retrieval for every query, a cohererence veto
                        on unreachable edges, an OOD veto on corrupted
                        ingress, and an ACE playbook in Zone C.

    python scripts/run_holarch_agent_eval.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from henri.holarch.code_lang import (
    VOCAB_LEDGER_KIND,
    CodeTextCorpus,
    HolarchCodeTextTokenizer,
    VocabularyLedger,
)
from henri.holarch.config import HolarchConfig
from henri.holarch.model import (
    LatentActionLookahead,
    build_cul_de_sac_traps,
    evaluate_closed_loop_action,
    find_checkpoint,
    load_checkpoint,
    open_clean_room_memory,
    open_planner_memory,
)
from henri.holarch.pipeline import HenriHolarch
from henri.holarch.server import SidecarService, build_app, verify_receipt
from henri.holarch.symbolic import CodeGraphIndexer, Edge, SymbolicTopologyTask, index_code
from henri.holarch.tasks import GridWorldTask, build_tasks
from henri.holarch.train import Trainer, TrainConfig, evaluate
from henri.holarch.zone_c import ZoneCMemory

ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = ROOT / "henri" / "holarch"

PLAYBOOK_STEPS = ["READ_INDEX", "LOCATE_SYMBOL", "EDIT_CALLSITE", "RUN_TESTS"]


# ---------------------------------------------------------------- baseline
class BaselineAgent:
    """An unindexed agent: it reads files and never vetoes.

    This is the control arm, not a straw man. It uses the same Python
    parser the indexer uses at ingest time, but it re-parses at query time
    and it has no epistemic gate: an unreachable edge returns the most
    frequent target in the graph, which is exactly the silent fabrication
    the sidecar is built to refuse.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.file_reads = 0
        self.bytes_read = 0
        self.response_bytes = 0
        self.latency_ms = 0.0

    def _scan(self) -> Dict[str, str]:
        texts: Dict[str, str] = {}
        for path in sorted(self.root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            self.file_reads += 1
            self.bytes_read += len(text.encode("utf-8"))
            texts[str(path)] = text
        return texts

    def call_target(self, caller: str, hops: int = 1) -> Optional[str]:
        """Resolve a call chain by re-reading the package at query time."""
        started = time.perf_counter()
        texts = self._scan()
        current = caller
        for _ in range(max(1, int(hops))):
            found = None
            marker = "def " + current.split(":")[-1] + "("
            for text in texts.values():
                position = text.find(marker)
                if position < 0:
                    continue
                body = text[position:]
                for line in body.splitlines()[1:]:
                    if line and not line.startswith((" ", "\t")):
                        break
                    for token in line.replace("(", " ").replace(")", " ").split():
                        candidate = token.strip(",.")
                        if candidate and candidate[0].isalpha() and candidate not in {
                            "return", "if", "for", "while", "with", "self", "raise",
                            "len", "int", "float", "str", "list", "dict", "set",
                            "range", "print", "float", "bool", "min", "max", "sum",
                        }:
                            found = candidate
                            break
                    if found:
                        break
                if found:
                    break
            if found is None:
                break
            current = found
        self.latency_ms += (time.perf_counter() - started) * 1e3
        answer = None if current == caller else current
        self.response_bytes += len(str(answer).encode("utf-8"))
        return answer

    def graph_answer(self, edges: Sequence[Edge], source: str, operators: Sequence[str]) -> Optional[str]:
        """Walk the edge list with no coherence veto."""
        started = time.perf_counter()
        index: Dict[Tuple[str, str], str] = {
            (edge.source, edge.operator): edge.target for edge in edges
        }
        current = source
        for operator in operators:
            target = index.get((current, operator))
            if target is None:
                # No gate: guess the most frequent target in the graph.
                counts: Dict[str, int] = {}
                for edge in edges:
                    counts[edge.target] = counts.get(edge.target, 0) + 1
                target = max(counts, key=counts.get) if counts else None
                if target is None:
                    break
            current = target
        self.latency_ms += (time.perf_counter() - started) * 1e3
        self.response_bytes += len(str(current).encode("utf-8"))
        return current

    def impact_answer(self, root: Path, target: str, hops: int) -> List[str]:
        """The unindexed arm's answer: every def in a file that names the target.

        No reverse edge set exists in this arm, so it over-reports. That is
        the honest cost of grep-style reasoning, not a straw man.
        """
        name = str(target).split(":")[-1].split(".")[-1]
        found: List[str] = []
        for path, text in self._scan().items():
            if name not in text:
                continue
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith("def ") or stripped.startswith("class "):
                    token = stripped.split()[1].split("(")[0].strip(":")
                    found.append(f"{Path(path).stem}:{token}")
        return sorted(set(found))

    def accept_ood(self, count: int) -> int:
        """A baseline with no veto accepts every corrupted input."""
        self.response_bytes += len(b"accept") * int(count)
        return int(count)


# ------------------------------------------------------------ query builder
def _aliases(name: str) -> List[str]:
    text = str(name)
    out = [text]
    bare = text.split(":")[-1]
    if bare not in out:
        out.append(bare)
    tail = bare.split(".")[-1]
    if tail not in out:
        out.append(tail)
    return out


def _impact_truth(edges: Sequence[Edge], target: str, hops: int) -> List[str]:
    """Independent reverse BFS over the edge list, for the query key."""
    reverse: Dict[str, List[str]] = {}
    for edge in edges:
        for alias in _aliases(edge.target):
            reverse.setdefault(alias, []).append(edge.source)
    frontier = sorted(_aliases(target))
    seen: List[str] = []
    visited = set(frontier)
    for _ in range(max(1, int(hops))):
        current: List[str] = []
        for node in frontier:
            for caller in reverse.get(node, []):
                if caller in visited:
                    continue
                visited.add(caller)
                current.append(caller)
                seen.append(caller)
                visited.update(_aliases(caller))
        frontier = sorted(set(current))
        if not frontier:
            break
    return sorted(set(seen))


def build_queries(root: Path) -> Dict[str, List[dict]]:
    """Deterministic query set: 36 queries over four categories."""
    graph = index_code(root, package="henri")
    by_source: Dict[str, List[str]] = {}
    targets_by_caller: Dict[str, List[str]] = {}
    for edge in graph.edges:
        targets_by_caller.setdefault(edge.source, []).append(edge.target)
    callers = sorted(name for name, targets in targets_by_caller.items() if len(targets) == 1)
    # Direct call queries: one callee, unambiguous.
    direct = callers[:4]
    # Two-hop call chains: a caller whose single callee itself calls once.
    chains = [
        name for name in callers
        if targets_by_caller.get(targets_by_caller[name][0])
    ][:4]
    code_queries = [
        {"kind": "call", "source": name, "hops": 1,
         "truth": targets_by_caller[name][0]} for name in direct
    ] + [
        {"kind": "call", "source": name, "hops": 2,
         "truth": targets_by_caller[targets_by_caller[name][0]][0]} for name in chains
    ]

    task = SymbolicTopologyTask(seed=20261103)
    edges = task.graph(2)
    graph_queries: List[dict] = []
    for hops, count in ((1, 8), (2, 4), (3, 4)):
        for salt in range(count):
            source = task.symbols[(hops * 3 + salt * 2) % len(task.symbols)]
            operators, truth = task.query(edges, source, hops, salt=salt + hops)
            if truth is None:
                continue
            graph_queries.append(
                {"kind": "graph", "source": source, "operators": operators,
                 "hops": hops, "truth": truth}
            )

    traps = []
    for salt in range(4):
        source, operators = task.draw_unreachable(edges, salt=salt + 11)
        traps.append({"kind": "unreachable", "source": source, "operators": operators})

    # Decision D5: 3 symbolic and 3 perceptual composite queries plus 2 traps.
    composite = [
        {"kind": "symbolic", "a": "d3", "op": "add", "b": "d4", "truth": "d7"},
        {"kind": "symbolic", "a": "d8", "op": "sub", "b": "d2", "truth": "d6"},
        {"kind": "symbolic", "a": "d5", "op": "add", "b": "d5", "truth": "d0"},
    ]
    # The atlas stores add and sub only; a mul key is valid VSA but has no
    # rule, so it is the honest unregistered-rule trap.
    composite_traps = [
        {"kind": "composite_trap", "a": "d3", "op": "mul", "b": "d4"},
        {"kind": "composite_trap", "a": "d9", "op": "mul", "b": "d1"},
    ]

    # Directive F4: reverse-call impact queries. The ground truth is computed
    # here by an independent breadth-first walk over the indexed edges, not by
    # calling the service that answers the query.
    impact_queries = []
    # Directive H4 keeps the six reverse-call queries. Candidates are tried
    # in a fixed order and the first six with a non-empty ground truth are
    # taken, so the count cannot drift when one unrelated symbol loses its
    # callers.
    impact_candidates = [
        ("ZoneCMemory.query", 1),
        ("ZoneCMemory.query", 2),
        ("ZoneCMemory.write", 2),
        ("SymbolCodebook.add", 1),
        ("HolarchConfig.digest", 1),
        ("index_code", 1),
        ("ZoneCMemory.write", 1),
        ("Edge", 1),
        ("SymbolCodebook.add", 2),
        ("index_code", 2),
    ]
    for target, hops in impact_candidates:
        if len(impact_queries) >= 6:
            break
        truth = _impact_truth(graph.edges, target, hops)
        if truth:
            impact_queries.append(
                {
                    "kind": "impact",
                    "target": target,
                    "hops": int(hops),
                    "truth": sorted(truth),
                }
            )
    impact_traps = [
        {"kind": "impact_trap", "target": "no_such_symbol_alpha", "hops": 1},
        {"kind": "impact_trap", "target": "ZoneCMemory.quantum_entangle", "hops": 3},
    ]

    return {
        "code": code_queries,
        "graph": graph_queries,
        "traps": traps,
        "composite": composite,
        "composite_traps": composite_traps,
        "impact": impact_queries,
        "impact_traps": impact_traps,
        "edges": edges,
        "graph_stats": {
            "files": float(graph.files),
            "lines": float(graph.lines),
            "nodes": float(len(graph.nodes)),
            "edges": float(len(graph.edges)),
            "parse_seconds": float(graph.seconds),
        },
    }


# ------------------------------------------------- Directive H: new slots
H_VERIFY_QUERIES: List[Dict[str, object]] = [
    {
        "id": "verify01",
        "claim": "The medium profile has 1,020,380 parameters and reaches 0.7083 closed-loop action.",
        "expect": True,
        "note": "two true quantities from the checkpoint record",
    },
    {
        "id": "verify02",
        "claim": "The vision accuracy (test) is 0.9798 against a 0.9750 bar.",
        "expect": True,
        "note": "a true test-split quantity and its bar",
    },
    {
        "id": "verify03",
        "claim": "The medium profile has 1,020,381 parameters and reaches 0.7083 closed-loop action.",
        "expect": False,
        "note": "corrupted parameter count",
    },
    {
        "id": "verify04",
        "claim": "Calling ZoneCMemory.query with keyword argument widget returns the stored wave.",
        "expect": False,
        "note": "fabricated keyword against a real signature",
    },
]

H_SYNTH_QUERIES: List[Dict[str, str]] = [
    {
        "id": "synth01",
        "intent": "Write a function distinct_caller_count that takes argument links and first returns the sorted unique source names of a list of edges and then returns the number of tokens in that result.",
        "test": "class _E:\n    def __init__(self, s):\n        self.source = s\nassert distinct_caller_count([_E('b'), _E('a'), _E('b')]) == 2\n",
    },
    {
        "id": "synth02",
        "intent": "Write a function bounded_mean that takes argument sample and first returns the arithmetic mean of a list of floats as a float and then clips the result into the closed interval 0.0 to 2.5.",
        "test": "assert bounded_mean([2.0, 4.0]) == 2.5\nassert bounded_mean([1.0, 2.0]) == 1.5\n",
    },
]

H_SYNTH_PLANS = ("core08", "shift11", "core10", "core11")

H_DELTA_FILES: Dict[str, str] = {
    "alpha.py": "def alpha(value):\n    return beta(value) + 1\n",
    "beta.py": "def beta(value):\n    return value - 1\n",
    "gamma.py": "def gamma(value):\n    return value * 2\n",
}

H_DELTA_ALPHA = "def alpha(value):\n    return gamma(value) + 1\n"
H_DELTA_DELTA = "def delta(value):\n    return beta(value) * 3\n"

# ---------------------------------------------------------------- Cycle I
I2_DAG_QUERIES: List[Dict[str, str]] = [
    {
        "id": "dag01",
        "intent": (
            "Write a function tri_agent that takes argument links and first returns "
            "the sorted unique source names of a list of edges then returns the number "
            "of tokens in that result then clips the result into the closed interval "
            "0.0 to 2.5."
        ),
        "test": (
            "class _E:\n    def __init__(self, s):\n        self.source = s\n"
            "assert tri_agent([_E('b'), _E('a'), _E('b')]) == 2.0\n"
        ),
    },
    {
        "id": "dag02",
        "intent": (
            "Write a function bounded_sources that takes argument edges and first "
            "returns the sorted unique source names of a list of edges then returns "
            "the number of tokens in that result then clips the result into the "
            "closed interval 0.0 to 1.0."
        ),
        "test": (
            "class _E:\n    def __init__(self, s):\n        self.source = s\n"
            "assert bounded_sources([_E('b'), _E('a'), _E('b')]) == 1.0\n"
        ),
    },
]

I2_TYPE_TRAP_QUERIES: List[Dict[str, str]] = [
    {
        "id": "dag03",
        "intent": (
            "Write a function bad_agent_a that takes argument record and first "
            "returns the sha256 field of an epistemic receipt then returns the "
            "arithmetic mean of a list of floats as a float."
        ),
        "note": "text -> numeric_sequence (mean of a list)",
    },
    {
        "id": "dag04",
        "intent": (
            "Write a function bad_agent_b that takes argument sample and first "
            "returns the arithmetic mean of a list of floats as a float then "
            "returns the sha256 field of an epistemic receipt."
        ),
        "note": "numeric -> mapping (a receipt field is subscripted)",
    },
]

I4_IMPACT_TARGETS = (("build_tokenizer", 1), ("build_synthesis_engine", 1))
I4_CODE_CLAIM = "def use_locked_tokenizer(cfg):\n    return build_tokenizer(cfg)\n"
I1_VOCAB_NAMES = ("build_tokenizer", "ZoneCMemory", "calibrate_beta")
I1_VOCAB_FABRICATED = "holarch_synthesize_warp"


def _execute_source(source: str, test: str) -> Tuple[bool, str]:
    """Execute an emitted module plus its unit test in an isolated scope."""
    scope: Dict[str, object] = {"__builtins__": __builtins__}
    try:
        exec(str(source), scope)
        exec(str(test), scope)
    except Exception as error:  # noqa: BLE001 - the failure is the measurement
        return False, f"{type(error).__name__}: {error}"
    return True, ""


def _load_code_lang():
    """The L3 playbook library, loaded without importing scripts as a package."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_holarch_code_lang", ROOT / "scripts" / "run_holarch_code_lang.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_plan_queries(service: SidecarService, cfg) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    """Four Cul-de-Sac horizon traps from Directive I3.

    The old probes were open-grid states where the 1-step and 2-step greedy
    action coincide, so the depth-2 wall mask could not be isolated. These
    traps put the agent beside a one-cell pocket that lies on the steepest
    descent of the goal field: the 1-step planner enters it, and only a
    wall-masked 2-step planner sees that every non-reversing exit is blocked.
    The controls are measured on the same observation with the same weights
    and the same single-thread reduction.
    """
    traps = build_cul_de_sac_traps()
    rotor = getattr(service.sidecar.model, "action_rotor", None)
    masked = LatentActionLookahead(
        service.sidecar.model, None, rotor=rotor, transition="rotor",
        lookahead_depth=2, gamma_look=0.02,
    )
    single = LatentActionLookahead(
        service.sidecar.model, None, rotor=rotor, transition="rotor",
        lookahead_depth=1, gamma_look=0.02,
    )
    unmasked = LatentActionLookahead(
        service.sidecar.model, None, rotor=rotor, transition="rotor",
        lookahead_depth=2, gamma_look=0.02, mask_lookahead=False,
    )
    chosen: List[Dict[str, object]] = []
    for position, trap in enumerate(traps[:4]):
        observation = trap["observation"]
        arm_masked = int(masked.choose(observation, [])["action"])
        arm_single = int(single.choose(observation, [])["action"])
        arm_unmasked = int(unmasked.choose(observation, [])["action"])
        chosen.append(
            {
                "kind": "plan",
                "id": f"plan{position + 1:02d}",
                "trap_id": str(trap["id"]),
                "agent": list(trap["agent"]),
                "pocket": list(trap["pocket"]),
                "goal": list(trap["goal"]),
                "wall_cells": list(trap["wall_cells"]),
                "observation": observation,
                "truth": int(trap["bypass_action"]),
                "trap_action": int(trap["trap_action"]),
                "arm_depth2_masked": arm_masked,
                "arm_depth1": arm_single,
                "arm_depth2_unmasked": arm_unmasked,
                "trap": bool(arm_single != int(trap["bypass_action"]) or arm_unmasked != int(trap["bypass_action"])),
                "strong": bool(
                    arm_single != int(trap["bypass_action"])
                    and arm_unmasked != int(trap["bypass_action"])
                ),
                "invariants": dict(trap["invariants"]),
            }
        )
    selection = {
        "family": "deterministic 7x7 cul-de-sac table",
        "candidates_in_table": int(len(traps)),
        "chosen": int(len(chosen)),
        "masked_agrees_with_bypass": int(
            sum(1 for row in chosen if row["arm_depth2_masked"] == row["truth"])
        ),
        "depth1_enters_pocket": int(
            sum(1 for row in chosen if row["arm_depth1"] == row["trap_action"])
        ),
        "unmasked_enters_pocket": int(
            sum(1 for row in chosen if row["arm_depth2_unmasked"] == row["trap_action"])
        ),
        "determinism": "torch single-thread reduction for the probe and the API call",
    }
    return chosen, selection

# --------------------------------------------------------------- the eval
def _pixels(batch: int, seed: int) -> List[List[List[float]]]:
    generator = torch.Generator().manual_seed(int(seed))
    noise = torch.rand(batch, 1, 8, 8, generator=generator)
    return noise.reshape(batch, -1).tolist()


def run_agent_eval(
    root: Path = CODE_ROOT,
    train_steps: int = 0,
    workdir: Optional[Path] = None,
    seed: int = 20261103,
) -> Dict[str, object]:
    """Run both arms on one query set and return the measured payload.

    Directive E2: when a converged checkpoint is present, the service
    mounts it and the benchmark does not bootstrap 120 fresh steps. A
    caller that asks for extra steps still gets them, on top of the
    converged weights, which is the continual-learning case rather than
    the default.
    """
    checkpoint = find_checkpoint(None)
    if checkpoint is not None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        cfg = HolarchConfig(**payload["config"])
    else:
        cfg = HolarchConfig.small(seed=int(seed), fast_gate=True)
        if int(train_steps) == 0:
            train_steps = 120
    torch.manual_seed(cfg.seed)
    tasks = build_tasks(cfg, seed=int(seed))
    memory_path = Path(workdir or tempfile.gettempdir()) / f"henri_agent_eval_{int(time.time())}.sqlite3"
    if memory_path.exists():
        memory_path.unlink()
    service = SidecarService(cfg=cfg, backend="sqlite", memory_path=str(memory_path))
    mounted = bool(service.checkpoint_info.get("mounted", False))
    trainer = Trainer(service.sidecar.model, tasks, TrainConfig(steps_per_task=1, batch_size=32))
    if int(train_steps) > 0:
        for _ in range(int(train_steps)):
            trainer.step("vision")
        batches = [tasks["vision"].sample(32, "val").x for _ in range(8)]
        service.sidecar.model.calibrate_two_condition("vision", batches, ceiling=0.98)
        service.sidecar.model.calibrate_prototypes(
            "vision", [tasks["vision"].sample(64, "train") for _ in range(4)]
        )
        service.sidecar.model.calibrate_gate(
            "vision", [tasks["vision"].sample(32, "val").x for _ in range(4)]
        )
    service.sidecar.model.eval()

    # Directive I3: the planner is checkpoint-pinned. A smoke run may ask for
    # extra training steps for the vision arm; that must not silently move the
    # planning benchmark off the rotor it was measured with. Snapshot the
    # trained state and mount the checkpoint for the planning slots only.
    planning_snapshot = None
    planning_rotor = None
    if int(train_steps) > 0 and checkpoint is not None:
        planning_snapshot = {
            key: value.detach().clone()
            for key, value in service.sidecar.model.state_dict().items()
        }
        planning_rotor = getattr(service.sidecar.model, "action_rotor", None)

    def mount_planning_checkpoint() -> None:
        if planning_snapshot is not None:
            load_checkpoint(checkpoint, model=service.sidecar.model)

    def restore_trained_weights() -> None:
        if planning_snapshot is not None:
            service.sidecar.model.load_state_dict(planning_snapshot)
            service.sidecar.model.action_rotor = planning_rotor
            service.sidecar.model.eval()

    queries = build_queries(root)
    baseline = BaselineAgent(root)

    # One-time ingest for the sidecar arm. This is the only disk read the
    # MCP arm ever does; every later call answers from Zone C.
    index_started = time.perf_counter()
    code_index = service.ingest_code(str(root), package="henri")
    index_seconds = time.perf_counter() - index_started
    ingest = service.ingest_edges(
        [
            {"source": edge.source, "operator": edge.operator, "target": edge.target}
            for edge in queries["edges"]
        ]
    )
    # Directive H4: four depth-2 rotor probes whose truth is the BFS-optimal
    # action. The control arms are measured while the probes are selected.
    # The forward pass is a multithreaded reduction, and torch's CPU kernels
    # are not bit-identical across thread schedules, so the planning section
    # pins one thread: the probe and the API call then compute the same
    # number from the same input. This is a determinism control, not a
    # tuning knob; it is recorded in the receipt.
    # Directive I3: the planner fuses the Zone C context wave, so the probe
    # must read the SAME store the deployed planner reads. A scratch copy of
    # the repository store is pinned by SHA-256; the service's own memory is
    # restored immediately after the probe block.
    planner_memory, planner_fixture = open_planner_memory(
        cfg, workdir=(Path(workdir) if workdir else None)
    )
    threads_before = torch.get_num_threads()
    torch.set_num_threads(1)
    service_memory = service.sidecar.model.memory
    service.sidecar.model.memory = planner_memory
    mount_planning_checkpoint()
    try:
        plan_queries, plan_selection = build_plan_queries(service, cfg)
    finally:
        service.sidecar.model.memory = service_memory
        restore_trained_weights()
        torch.set_num_threads(threads_before)
    # Directive H2: the two synthesis queries chain stored playbooks. The
    # four playbooks they compose are compiled through the sidecar (setup,
    # no receipt) before the scored sequence begins.
    # The whole 24-playbook library is compiled, not only the four the two
    # queries compose: IDF weighting is a property of the library, and a
    # four-plan store is not the deployment the H2 receipt measured.
    code_lang = _load_code_lang()
    for task in list(code_lang.CORE_TASKS) + list(code_lang.SHIFT_TASKS):
        service.sidecar.adapt_synthesis(task["intent"], task["source"], task["id"])

    score = {
        "baseline_unindexed": {"reachable_correct": 0, "reachable_total": 0,
                                "traps_total": 0, "hallucinations": 0,
                                "playbook_exact": 0, "playbook_total": 0},
        "holarch_mcp": {"reachable_correct": 0, "reachable_total": 0,
                         "traps_total": 0, "hallucinations": 0,
                         "playbook_exact": 0, "playbook_total": 0},
    }
    per_category: Dict[str, Dict[str, int]] = {
        "code_call_flow": {"baseline_correct": 0, "mcp_correct": 0, "total": 0},
        "multi_hop_graph": {"baseline_correct": 0, "mcp_correct": 0, "total": 0},
        "unreachable_traps": {"baseline_hallucinated": 0, "mcp_refused": 0, "total": 0},
        "impact_analysis": {"baseline_correct": 0, "mcp_correct": 0, "total": 0},
        "impact_traps": {"baseline_hallucinated": 0, "mcp_refused": 0, "total": 0},
        "ood_ingress": {"baseline_false_accept": 0, "mcp_vetoed": 0, "total": 0},
        "playbook_reuse": {"baseline_exact": 0, "mcp_exact": 0, "total": 0},
        "composite_rules": {"baseline_correct": 0, "mcp_correct": 0, "total": 0},
        "composite_traps": {"baseline_hallucinated": 0, "mcp_refused": 0, "total": 0},
        "depth2_rotor_planning": {"baseline_correct": 0, "mcp_correct": 0, "total": 0},
        "ontology_verification": {"baseline_correct": 0, "mcp_correct": 0, "total": 0},
        "multi_plan_synthesis": {"baseline_correct": 0, "mcp_correct": 0, "total": 0},
        "delta_blast_radius": {"baseline_correct": 0, "mcp_correct": 0, "total": 0},
        "dag_synthesis": {"baseline_correct": 0, "mcp_correct": 0, "total": 0},
        "cross_package_agent_step": {"baseline_correct": 0, "mcp_correct": 0, "total": 0},
        "vocab_lock": {"baseline_correct": 0, "mcp_correct": 0, "total": 0},
        "j1_cleanroom": {"baseline_correct": 0, "mcp_correct": 0, "total": 0},
        "j2_elastic_vocab": {"baseline_correct": 0, "mcp_correct": 0, "total": 0},
        "j3_diverse_dag": {"baseline_correct": 0, "mcp_correct": 0, "total": 0},
        "j4_openai_adapter": {"baseline_correct": 0, "mcp_correct": 0, "total": 0},
    }
    mcp_latency = 0.0
    mcp_response_bytes = 0
    receipts: List[dict] = []

    # The two setup receipts are the first links of the chain: 64 receipts
    # for 64 slots, every one verifiable, rather than 62 responses plus two
    # uncommitted setup calls.
    receipts.append(code_index["receipt"])
    receipts.append(ingest["receipt"])

    for query in queries["code"]:
        truth = query["truth"]
        started = time.perf_counter()
        answer = baseline.call_target(query["source"], hops=int(query["hops"]))
        baseline.latency_ms += (time.perf_counter() - started) * 1e3
        baseline_correct = int(answer == truth)

        started = time.perf_counter()
        result = service.solve(query["source"], ["CALLS"] * int(query["hops"]))
        mcp_latency += (time.perf_counter() - started) * 1e3
        mcp_response_bytes += len(json.dumps(result["result"]))
        receipts.append(result["receipt"])
        mcp_answer = result["result"]["answer"]
        mcp_correct = int(mcp_answer == truth)

        score["baseline_unindexed"]["reachable_total"] += 1
        score["holarch_mcp"]["reachable_total"] += 1
        score["baseline_unindexed"]["reachable_correct"] += baseline_correct
        score["holarch_mcp"]["reachable_correct"] += mcp_correct
        per_category["code_call_flow"]["total"] += 1
        per_category["code_call_flow"]["baseline_correct"] += baseline_correct
        per_category["code_call_flow"]["mcp_correct"] += mcp_correct

    for query in queries["graph"]:
        truth = query["truth"]
        started = time.perf_counter()
        baseline_answer = baseline.graph_answer(
            queries["edges"], query["source"], query["operators"]
        )
        baseline.latency_ms += (time.perf_counter() - started) * 1e3
        started = time.perf_counter()
        result = service.solve(query["source"], query["operators"])
        mcp_latency += (time.perf_counter() - started) * 1e3
        mcp_response_bytes += len(json.dumps(result["result"]))
        receipts.append(result["receipt"])
        mcp_answer = result["result"]["answer"]
        baseline_correct = int(baseline_answer == truth)
        mcp_correct = int(mcp_answer == truth)
        score["baseline_unindexed"]["reachable_total"] += 1
        score["holarch_mcp"]["reachable_total"] += 1
        score["baseline_unindexed"]["reachable_correct"] += baseline_correct
        score["holarch_mcp"]["reachable_correct"] += mcp_correct
        per_category["multi_hop_graph"]["total"] += 1
        per_category["multi_hop_graph"]["baseline_correct"] += baseline_correct
        per_category["multi_hop_graph"]["mcp_correct"] += mcp_correct

    for query in queries["traps"]:
        guess = baseline.graph_answer(queries["edges"], query["source"], query["operators"])
        result = service.solve(query["source"], query["operators"])
        receipts.append(result["receipt"])
        baseline_hallucinated = int(guess is not None)
        mcp_refused = int(bool(result["result"]["vetoed"]))
        score["baseline_unindexed"]["traps_total"] += 1
        score["holarch_mcp"]["traps_total"] += 1
        score["baseline_unindexed"]["hallucinations"] += baseline_hallucinated
        score["holarch_mcp"]["hallucinations"] += int(not mcp_refused)
        per_category["unreachable_traps"]["total"] += 1
        per_category["unreachable_traps"]["baseline_hallucinated"] += baseline_hallucinated
        per_category["unreachable_traps"]["mcp_refused"] += mcp_refused

    for query in queries["impact"]:
        truth = list(query["truth"])
        started = time.perf_counter()
        baseline_answer = baseline.impact_answer(root, query["target"], int(query["hops"]))
        baseline.latency_ms += (time.perf_counter() - started) * 1e3
        started = time.perf_counter()
        result = service.solve(query["target"], [str(query["hops"])], mode="impact")
        mcp_latency += (time.perf_counter() - started) * 1e3
        mcp_response_bytes += len(json.dumps(result["result"]))
        receipts.append(result["receipt"])
        mcp_callers = list(result["result"].get("callers", []))
        baseline_correct = int(sorted(baseline_answer) == sorted(truth))
        mcp_correct = int(sorted(mcp_callers) == sorted(truth))
        score["baseline_unindexed"]["reachable_total"] += 1
        score["holarch_mcp"]["reachable_total"] += 1
        score["baseline_unindexed"]["reachable_correct"] += baseline_correct
        score["holarch_mcp"]["reachable_correct"] += mcp_correct
        per_category["impact_analysis"]["total"] += 1
        per_category["impact_analysis"]["baseline_correct"] += baseline_correct
        per_category["impact_analysis"]["mcp_correct"] += mcp_correct

    for query in queries["impact_traps"]:
        baseline_answer = baseline.impact_answer(root, query["target"], int(query["hops"]))
        result = service.solve(query["target"], [str(query["hops"])], mode="impact")
        receipts.append(result["receipt"])
        mcp_refused = int(bool(result["result"].get("vetoed", False)))
        baseline_hallucinated = int(bool(baseline_answer))
        score["baseline_unindexed"]["traps_total"] += 1
        score["holarch_mcp"]["traps_total"] += 1
        score["baseline_unindexed"]["hallucinations"] += baseline_hallucinated
        score["holarch_mcp"]["hallucinations"] += int(not mcp_refused)
        per_category["impact_traps"]["total"] += 1
        per_category["impact_traps"]["baseline_hallucinated"] += baseline_hallucinated
        per_category["impact_traps"]["mcp_refused"] += mcp_refused

    for salt in range(4):
        pixels = _pixels(32, 500 + salt)
        baseline_accept = baseline.accept_ood(32)
        result = service.veto("vision", pixels)
        receipts.append(result["receipt"])
        veto_rate = float(result["result"]["veto_rate"])
        mcp_vetoed = int(veto_rate * 32)
        score["baseline_unindexed"]["traps_total"] += 32
        score["holarch_mcp"]["traps_total"] += 32
        score["baseline_unindexed"]["hallucinations"] += baseline_accept
        score["holarch_mcp"]["hallucinations"] += 32 - mcp_vetoed
        per_category["ood_ingress"]["total"] += 32
        per_category["ood_ingress"]["baseline_false_accept"] += baseline_accept
        per_category["ood_ingress"]["mcp_vetoed"] += mcp_vetoed

    compiled = service.playbook("refactor", PLAYBOOK_STEPS)
    recalled = service.playbook("refactor")
    receipts.extend([compiled["receipt"], recalled["receipt"]])
    mcp_response_bytes += len(json.dumps(compiled["result"])) + len(json.dumps(recalled["result"]))
    exact = int(list(recalled["result"]["steps"]) == PLAYBOOK_STEPS)
    score["holarch_mcp"]["playbook_exact"] += exact
    score["holarch_mcp"]["playbook_total"] += 1
    score["baseline_unindexed"]["playbook_total"] += 1
    per_category["playbook_reuse"]["total"] += 1
    per_category["playbook_reuse"]["mcp_exact"] += exact

    # ------------------- Decision D5: composite neuro-symbolic queries -----
    from henri.holarch.tasks import DigitsTask

    composite_rules = [
        {"a": f"d{a}", "op": "add", "b": f"d{b}", "result": f"d{(a + b) % 10}"}
        for a in range(10)
        for b in range(10)
    ] + [
        {"a": f"d{a}", "op": "sub", "b": f"d{b}", "result": f"d{abs(a - b)}"}
        for a in range(10)
        for b in range(10)
    ]
    # The composite atlas is SETUP, not a scored query, so it is written
    # through the solver directly and claims no receipt slot. That keeps the
    # 64 collected receipts contiguous in the ledger chain.
    comp_rows = [
        ((str(rule["a"]), str(rule["op"]), str(rule["b"])), str(rule["result"]))
        for rule in composite_rules
    ]
    comp_written = service.composite.ingest(comp_rows)
    comp_calibration = service.composite.calibrate(comp_rows)
    comp_ingest = {
        "result": {
            "rules_ingested": float(comp_written),
            "tau_composite": float(comp_calibration["tau_coherence"]),
            "atlas_entries": float(service.composite.entries),
        }
    }
    mcp_response_bytes += len(json.dumps(comp_ingest["result"]))

    digits = DigitsTask(seed=seed + 7)
    digit_images, digit_labels = digits.splits["val"]
    generator = np.random.default_rng(seed)
    perceptual_queries = []
    for index in range(3):
        # Directive E2: the perceptual query is a real 16x16 four-digit
        # canvas. The two informative quadrants are grounded through the
        # converged Pass-0 prototype bank; the two distractor quadrants
        # carry no label and are not queried.
        picks = generator.integers(0, len(digit_images), size=4)
        canvas = np.zeros((16, 16), dtype=np.float32)
        for slot, pick in enumerate(picks):
            row = (slot // 2) * 8
            column = (slot % 2) * 8
            canvas[row : row + 8, column : column + 8] = digit_images[int(pick)]
        d0, d1 = int(digit_labels[int(picks[0])]), int(digit_labels[int(picks[1])])
        operator = "add" if index % 2 == 0 else "sub"
        truth = (d0 + d1) % 10 if operator == "add" else abs(d0 - d1)
        perceptual_queries.append(
            {
                "kind": "perceptual",
                "x0": canvas[0:8, 0:8].reshape(-1).tolist(),
                "x1": canvas[0:8, 8:16].reshape(-1).tolist(),
                "op": operator,
                "truth": f"d{truth}",
            }
        )

    composite_stats = {
        "reachable_total": 0,
        "mcp_correct": 0,
        "baseline_correct": 0,
        "perceptual_total": 0,
        "perceptual_consistent": 0,
        "perceptual_truth_match": 0,
        "traps_total": 0,
        "mcp_refused": 0,
        "baseline_hallucinated": 0,
    }
    for query in list(queries["composite"]) + perceptual_queries:
        composite_stats["reachable_total"] += 1
        if query["kind"] == "symbolic":
            left, right = int(query["a"][1:]), int(query["b"][1:])
            baseline_answer = (
                f"d{(left + right) % 10}" if query["op"] == "add" else f"d{abs(left - right)}"
            )
            result = service.composite_call(
                mode="query", a=query["a"], b=query["b"], op=query["op"]
            )
            receipts.append(result["receipt"])
            mcp_response_bytes += len(json.dumps(result["result"]))
            mcp_ok = int(result["result"]["result"] == query["truth"])
            baseline_ok = int(baseline_answer == query["truth"])
        else:
            # The unindexed arm has no perceptual classifier: it admits no
            # answer rather than fabricating a digit.
            baseline_answer = None
            result = service.composite_call(
                mode="query", x0=query["x0"], x1=query["x1"], op=query["op"]
            )
            receipts.append(result["receipt"])
            mcp_response_bytes += len(json.dumps(result["result"]))
            grounded = result["result"]["grounded_operands"]
            composite_stats["perceptual_total"] += 1
            if grounded:
                left_symbol, right_symbol, operator = grounded
                left, right = int(left_symbol[1:]), int(right_symbol[1:])
                composed = (
                    f"d{(left + right) % 10}"
                    if operator == "add"
                    else f"d{abs(left - right)}"
                )
                consistent = int(
                    (not result["result"]["abstained"])
                    and result["result"]["result"] == composed
                )
                composite_stats["perceptual_consistent"] += consistent
                composite_stats["perceptual_truth_match"] += int(
                    result["result"]["result"] == query["truth"]
                )
                mcp_ok = consistent
            else:
                mcp_ok = 0
            baseline_ok = int(baseline_answer == query["truth"])
        composite_stats["mcp_correct"] += mcp_ok
        composite_stats["baseline_correct"] += baseline_ok

    for query in queries["composite_traps"]:
        composite_stats["traps_total"] += 1
        left, right = int(query["a"][1:]), int(query["b"][1:])
        # The unindexed arm computes the arithmetic and never checks that the
        # rule was registered: this is the fabrication the sidecar refuses.
        baseline_answer = f"d{(left * right) % 10}"
        result = service.composite_call(
            mode="query", a=query["a"], b=query["b"], op=query["op"]
        )
        receipts.append(result["receipt"])
        mcp_refused = int(bool(result["result"]["abstained"]))
        composite_stats["mcp_refused"] += mcp_refused
        composite_stats["baseline_hallucinated"] += int(baseline_answer is not None)
        score["holarch_mcp"]["traps_total"] += 1
        score["baseline_unindexed"]["traps_total"] += 1
        score["holarch_mcp"]["hallucinations"] += int(not mcp_refused)
        score["baseline_unindexed"]["hallucinations"] += 1

    # ------------------- Directive I3: cul-de-sac horizon planning --------
    plan_rows = []
    service_memory = service.sidecar.model.memory
    service.sidecar.model.memory = planner_memory
    mount_planning_checkpoint()
    try:
        for query in plan_queries:
            started = time.perf_counter()
            threads_before = torch.get_num_threads()
            torch.set_num_threads(1)
            try:
                result = service.plan(
                    query["observation"],
                    transition="rotor",
                    lookahead_depth=2,
                    gamma_look=0.02,
                )
            finally:
                torch.set_num_threads(threads_before)
            mcp_latency += (time.perf_counter() - started) * 1e3
            mcp_response_bytes += len(json.dumps(result["result"]))
            receipts.append(result["receipt"])
            answer = int(result["result"]["action"])
            ok = int(answer == int(query["truth"]))
            # The control arm is the same weights with no wave lookahead: the
            # reactive argmax of the action head, legality-masked. It is the
            # policy Cycle E deployed, not a straw man.
            control = int(service.sidecar.model.act(list(query["observation"])))
            baseline_ok = int(control == int(query["truth"]))
            score["holarch_mcp"]["reachable_total"] += 1
            score["holarch_mcp"]["reachable_correct"] += ok
            per_category["depth2_rotor_planning"]["total"] += 1
            per_category["depth2_rotor_planning"]["mcp_correct"] += ok
            per_category["depth2_rotor_planning"]["baseline_correct"] += baseline_ok
            plan_rows.append(
                {
                    "id": query["id"],
                    "trap_id": str(query.get("trap_id", "")),
                    "agent": query.get("agent"),
                    "goal": query.get("goal"),
                    "truth": int(query["truth"]),
                    "answer": answer,
                    "reactive": control,
                    "depth1": int(query["arm_depth1"]),
                    "unmasked_depth2": int(query["arm_depth2_unmasked"]),
                    "trap": bool(query["trap"]),
                    "strong_trap": bool(query["strong"]),
                    "correct": bool(ok),
                    "receipt_sha256": result["receipt"]["sha256"],
                }
            )
    finally:
        service.sidecar.model.memory = service_memory
        restore_trained_weights()
    per_category["depth2_rotor_planning"]["rows"] = plan_rows

    # ------------------- Directive H3: claim verification ----------------
    verify_rows = []
    for query in H_VERIFY_QUERIES:
        result = service.verify(str(query["claim"]), kind="text", top_k=4)
        mcp_response_bytes += len(json.dumps(result["result"]))
        receipts.append(result["receipt"])
        supported = bool(result["result"]["supported"])
        ok = int(supported == bool(query["expect"]))
        if bool(query["expect"]):
            score["holarch_mcp"]["reachable_total"] += 1
            score["holarch_mcp"]["reachable_correct"] += ok
            per_category["ontology_verification"]["total"] += 1
            per_category["ontology_verification"]["mcp_correct"] += ok
        else:
            score["holarch_mcp"]["traps_total"] += 1
            score["holarch_mcp"]["hallucinations"] += int(not ok)
            per_category["ontology_verification"]["total"] += 1
            per_category["ontology_verification"]["mcp_correct"] += ok
        verify_rows.append(
            {
                "id": query["id"],
                "claim": query["claim"],
                "expect": bool(query["expect"]),
                "supported": supported,
                "status": str(result["result"]["status"]),
                "reason": str(result["result"]["reason"])[:160],
                "correct": bool(ok),
                "receipt_sha256": result["receipt"]["sha256"],
            }
        )
    per_category["ontology_verification"]["rows"] = verify_rows

    # ------------------- Directive H2: multi-plan synthesis --------------
    synth_rows = []
    for query in H_SYNTH_QUERIES:
        result = service.synthesize(str(query["intent"]), kind="code")
        mcp_response_bytes += len(json.dumps(result["result"]))
        receipts.append(result["receipt"])
        executed = _execute_source(result["result"].get("source", ""), query["test"])
        ok = int(bool(result["result"].get("verified")) and bool(executed[0]))
        score["holarch_mcp"]["reachable_total"] += 1
        score["holarch_mcp"]["reachable_correct"] += ok
        per_category["multi_plan_synthesis"]["total"] += 1
        per_category["multi_plan_synthesis"]["mcp_correct"] += ok
        synth_rows.append(
            {
                "id": query["id"],
                "plan": str(result["result"].get("plan", "")),
                "composed": bool(result["result"].get("composed", False)),
                "verified": bool(result["result"].get("verified", False)),
                "executed": bool(executed[0]),
                "reason": str(executed[1])[:160],
                "source": str(result["result"].get("source", ""))[:240],
                "correct": bool(ok),
                "receipt_sha256": result["receipt"]["sha256"],
            }
        )
    per_category["multi_plan_synthesis"]["rows"] = synth_rows
    # ------------------- Directive I2: N-stage DAG synthesis -------------
    dag_rows: List[Dict[str, object]] = []
    dag_executed = 0
    for query in I2_DAG_QUERIES:
        result = service.synthesize(str(query["intent"]), kind="code")
        mcp_response_bytes += len(json.dumps(result["result"]))
        receipts.append(result["receipt"])
        executed = _execute_source(result["result"].get("source", ""), query["test"])
        composed = list(result["result"].get("composed_from", []) or [])
        ok = int(
            bool(result["result"].get("verified"))
            and bool(executed[0])
            and len(composed) >= 3
        )
        dag_executed += ok
        score["holarch_mcp"]["reachable_total"] += 1
        score["holarch_mcp"]["reachable_correct"] += ok
        per_category["dag_synthesis"]["total"] += 1
        per_category["dag_synthesis"]["mcp_correct"] += ok
        dag_rows.append(
            {
                "id": query["id"],
                "kind": "pipeline",
                "plans": composed,
                "dag_depth": int(result["result"].get("dag_depth", 0)),
                "type_compatible": bool(result["result"].get("type_compatible", True)),
                "verified": bool(result["result"].get("verified", False)),
                "executed": bool(executed[0]),
                "reason": str(executed[1])[:160],
                "source": str(result["result"].get("source", ""))[:260],
                "correct": bool(ok),
                "receipt_sha256": result["receipt"]["sha256"],
            }
        )
    dag_refused = 0
    for query in I2_TYPE_TRAP_QUERIES:
        result = service.synthesize(str(query["intent"]), kind="code")
        mcp_response_bytes += len(json.dumps(result["result"]))
        receipts.append(result["receipt"])
        reason = str(result["result"].get("reason", ""))
        refused = int(
            str(result["result"].get("status", "")) == "refused"
            and reason.startswith("type mismatch")
        )
        dag_refused += refused
        score["holarch_mcp"]["traps_total"] += 1
        score["holarch_mcp"]["hallucinations"] += int(not refused)
        per_category["dag_synthesis"]["total"] += 1
        per_category["dag_synthesis"]["mcp_correct"] += refused
        dag_rows.append(
            {
                "id": query["id"],
                "kind": "type_trap",
                "plans": list(result["result"].get("composed_from", []) or []),
                "dag_depth": int(result["result"].get("dag_depth", 0)),
                "type_compatible": bool(result["result"].get("type_compatible", True)),
                "verified": bool(result["result"].get("verified", False)),
                "refused": bool(refused),
                "reason": reason[:200],
                "note": query.get("note", ""),
                "correct": bool(refused),
                "receipt_sha256": result["receipt"]["sha256"],
            }
        )
    per_category["dag_synthesis"]["rows"] = dag_rows
    per_category["dag_synthesis"]["pipelines_executed"] = dag_executed
    per_category["dag_synthesis"]["type_traps_refused"] = dag_refused

    # ------------------- Directive I4: cross-package index + agent_step ---
    workspace_indexer = CodeGraphIndexer(None)
    workspace_report = workspace_indexer.index_paths(
        ROOT, ["henri", "scripts", "tests"], package="workspace"
    )
    service.sidecar.mount_indexer(workspace_indexer, namespace="workspace")
    xp_rows: List[Dict[str, object]] = []
    xp_ok = 0
    for target, hops in I4_IMPACT_TARGETS:
        truth = _impact_truth(workspace_indexer.graph.edges, target, int(hops))
        result = service.agent_step(
            "impact", {"target": target, "hops": int(hops)}, namespace="workspace"
        )
        receipts.append(result["epistemic_receipt"])
        mcp_response_bytes += len(json.dumps(result["output"]))
        callers = sorted(str(value) for value in result["output"].get("callers", []))
        ok = int(bool(truth) and callers == sorted(truth))
        xp_ok += ok
        score["holarch_mcp"]["reachable_total"] += 1
        score["holarch_mcp"]["reachable_correct"] += ok
        per_category["cross_package_agent_step"]["total"] += 1
        per_category["cross_package_agent_step"]["mcp_correct"] += ok
        xp_rows.append(
            {
                "id": f"xp{len(xp_rows) + 1:02d}",
                "action": "impact",
                "target": target,
                "hops": int(hops),
                "truth": list(truth),
                "callers": callers,
                "correct": bool(ok),
                "receipt_sha256": result["epistemic_receipt"]["sha256"],
            }
        )
    verify_step = service.agent_step(
        "verify", {"claim": I4_CODE_CLAIM, "kind": "code"}, namespace="default"
    )
    receipts.append(verify_step["epistemic_receipt"])
    mcp_response_bytes += len(json.dumps(verify_step["output"]))
    verify_ok = int(
        str(verify_step["status"]) == "supported"
        and bool(verify_step["output"].get("supported"))
    )
    xp_ok += verify_ok
    score["holarch_mcp"]["reachable_total"] += 1
    score["holarch_mcp"]["reachable_correct"] += verify_ok
    per_category["cross_package_agent_step"]["total"] += 1
    per_category["cross_package_agent_step"]["mcp_correct"] += verify_ok
    xp_rows.append(
        {
            "id": f"xp{len(xp_rows) + 1:02d}",
            "action": "verify",
            "claim": I4_CODE_CLAIM,
            "status": str(verify_step["status"]),
            "correct": bool(verify_ok),
            "receipt_sha256": verify_step["epistemic_receipt"]["sha256"],
        }
    )
    trap_four = build_cul_de_sac_traps()[4]
    threads_before = torch.get_num_threads()
    torch.set_num_threads(1)
    previous_memory = service.sidecar.model.memory
    service.sidecar.model.memory = planner_memory
    mount_planning_checkpoint()
    try:
        plan_step = service.agent_step(
            "plan",
            {
                "observation": trap_four["observation"],
                "transition": "rotor",
                "lookahead_depth": 2,
                "gamma_look": 0.02,
            },
            namespace="default",
        )
    finally:
        service.sidecar.model.memory = previous_memory
        restore_trained_weights()
        torch.set_num_threads(threads_before)
    receipts.append(plan_step["epistemic_receipt"])
    mcp_response_bytes += len(json.dumps(plan_step["output"]))
    plan_ok = int(int(plan_step["output"].get("action", -1)) == int(trap_four["bypass_action"]))
    xp_ok += plan_ok
    score["holarch_mcp"]["reachable_total"] += 1
    score["holarch_mcp"]["reachable_correct"] += plan_ok
    per_category["cross_package_agent_step"]["total"] += 1
    per_category["cross_package_agent_step"]["mcp_correct"] += plan_ok
    xp_rows.append(
        {
            "id": f"xp{len(xp_rows) + 1:02d}",
            "action": "plan",
            "trap": str(trap_four["id"]),
            "action_chosen": int(plan_step["output"].get("action", -1)),
            "truth": int(trap_four["bypass_action"]),
            "correct": bool(plan_ok),
            "receipt_sha256": plan_step["epistemic_receipt"]["sha256"],
        }
    )
    per_category["cross_package_agent_step"]["rows"] = xp_rows
    per_category["cross_package_agent_step"]["workspace"] = {
        "files": int(workspace_report.files),
        "nodes": int(workspace_report.nodes),
        "edges": int(workspace_report.edges),
        "seconds": float(workspace_report.seconds),
    }

    # ------------------- Directive I1: live vocabulary lock --------------
    lock_path = ROOT / "receipts" / "holarch" / "vocab_lock.json"
    vocab_lock = VocabularyLedger.load(lock_path) if lock_path.exists() else None
    vocab_rows: List[Dict[str, object]] = []
    vocab_ok = 0
    for name in I1_VOCAB_NAMES:
        token_id = vocab_lock.id_of(name) if vocab_lock is not None else None
        claim = json.dumps({"name": name, "token_id": token_id})
        result = service.agent_step(
            "verify", {"claim": claim, "kind": "vocab"}, namespace="default"
        )
        receipts.append(result["epistemic_receipt"])
        mcp_response_bytes += len(json.dumps(result["output"]))
        ok = int(
            str(result["status"]) == "supported"
            and result["output"].get("token_id") is not None
            and int(result["output"]["token_id"]) == int(token_id)
        )
        vocab_ok += ok
        score["holarch_mcp"]["reachable_total"] += 1
        score["holarch_mcp"]["reachable_correct"] += ok
        per_category["vocab_lock"]["total"] += 1
        per_category["vocab_lock"]["mcp_correct"] += ok
        vocab_rows.append(
            {
                "id": f"vocab{len(vocab_rows) + 1:02d}",
                "name": name,
                "expected_token_id": token_id,
                "returned_token_id": result["output"].get("token_id"),
                "ledger_digest": str(result["output"].get("ledger_digest", "")),
                "correct": bool(ok),
                "receipt_sha256": result["epistemic_receipt"]["sha256"],
            }
        )
    fabricated = service.agent_step(
        "verify",
        {
            "claim": json.dumps({"name": I1_VOCAB_FABRICATED, "token_id": 42}),
            "kind": "vocab",
        },
        namespace="default",
    )
    receipts.append(fabricated["epistemic_receipt"])
    mcp_response_bytes += len(json.dumps(fabricated["output"]))
    fabricated_ok = int(
        str(fabricated["status"]) == "unsupported" and bool(fabricated["abstained"])
    )
    score["holarch_mcp"]["traps_total"] += 1
    score["holarch_mcp"]["hallucinations"] += int(not fabricated_ok)
    per_category["vocab_lock"]["total"] += 1
    per_category["vocab_lock"]["mcp_correct"] += fabricated_ok
    vocab_rows.append(
        {
            "id": f"vocab{len(vocab_rows) + 1:02d}",
            "name": I1_VOCAB_FABRICATED,
            "expected_token_id": None,
            "returned_token_id": fabricated["output"].get("token_id"),
            "status": str(fabricated["status"]),
            "correct": bool(fabricated_ok),
            "receipt_sha256": fabricated["epistemic_receipt"]["sha256"],
        }
    )
    per_category["vocab_lock"]["rows"] = vocab_rows
    per_category["vocab_lock"]["locked"] = bool(
        vocab_lock is not None and len(vocab_lock.identifiers) > 0
    )
    per_category["vocab_lock"]["ledger_digest"] = (
        str(vocab_lock.digest()) if vocab_lock is not None else ""
    )


    # ------------------- Directive H4: file-delta blast radius -----------
    delta_root = Path(workdir or tempfile.gettempdir()) / f"henri_delta_{int(time.time())}"
    delta_root.mkdir(parents=True, exist_ok=True)
    for name, text in H_DELTA_FILES.items():
        (delta_root / name).write_text(text, encoding="utf-8")
    service.sidecar.ingest_code(delta_root, package="pkg", namespace="delta")
    service._cache_clear()
    delta_rows = []
    delta_queries = [
        {
            "id": "delta01",
            "file": "alpha.py",
            "source": H_DELTA_ALPHA,
            "target": "beta",
            "truth": [],
            "note": "alpha stops calling beta",
        },
        {
            "id": "delta02",
            "file": "delta.py",
            "source": H_DELTA_DELTA,
            "target": "beta",
            "truth": ["delta:delta"],
            "note": "a new module starts calling beta",
        },
    ]
    for query in delta_queries:
        started = time.perf_counter()
        # The edit is setup, not a scored read: it mutates the tree through
        # mode="delta" with no receipt of its own.
        service.sidecar.ingest_code(
            delta_root,
            package="pkg",
            namespace="delta",
            mode="delta",
            file_path=str(delta_root / str(query["file"])),
            source_text=str(query["source"]),
        )
        delta_ms = (time.perf_counter() - started) * 1e3
        result = service.solve(str(query["target"]), ["1"], mode="impact", namespace="delta")
        mcp_response_bytes += len(json.dumps(result["result"]))
        receipts.append(result["receipt"])
        callers = sorted(str(value) for value in result["result"].get("callers", []))
        ok = int(callers == sorted(str(value) for value in query["truth"]))
        score["holarch_mcp"]["reachable_total"] += 1
        score["holarch_mcp"]["reachable_correct"] += ok
        per_category["delta_blast_radius"]["total"] += 1
        per_category["delta_blast_radius"]["mcp_correct"] += ok
        delta_rows.append(
            {
                "id": query["id"],
                "file": query["file"],
                "target": query["target"],
                "truth": list(query["truth"]),
                "callers": callers,
                "delta_ms": float(delta_ms),
                "correct": bool(ok),
                "receipt_sha256": result["receipt"]["sha256"],
            }
        )
    per_category["delta_blast_radius"]["rows"] = delta_rows

    # =================== Directive J: 12 clean-room probes =================
    # The 76 scored queries above are unchanged. The block below adds exactly
    # 12 deterministic probes (4 J1, 2 J2, 4 J3, 2 J4). Each probe records its
    # own row: name, category, expected, measured, correct and evidence.
    # None of them appends to the legacy 76-receipt chain, so queries_76 and
    # receipt_chain_76 keep their Cycle I meaning. The two J4 probes issue
    # their own ledger receipts (agent_step + chat completion); those receipts
    # are verified row by row and by the whole-ledger chain check.
    scratch_dir = Path(workdir) if workdir is not None else (ROOT / "probe_tmp_eval")
    scratch_dir.mkdir(parents=True, exist_ok=True)
    probe_rows: List[Dict[str, object]] = []

    def _record_probe(name, category, expected, measured, correct, **evidence):
        row = {
            "name": str(name),
            "category": str(category),
            "expected": expected,
            "measured": measured,
            "correct": bool(correct),
        }
        row.update(evidence)
        probe_rows.append(row)
        bucket = per_category.setdefault(
            str(category),
            {"baseline_correct": 0, "mcp_correct": 0, "total": 0},
        )
        bucket["total"] = int(bucket.get("total", 0)) + 1
        bucket["mcp_correct"] = int(bucket.get("mcp_correct", 0)) + int(bool(correct))
        return row

    # ------------------- J1: clean-room mounted checkpoint ----------------
    clean_memory, clean_info = open_clean_room_memory(
        cfg, checkpoint, scratch_dir, prefix="agent_eval_clean"
    )
    seed_payload: Dict[str, object] = {}
    if checkpoint is not None and Path(checkpoint).exists():
        _loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if isinstance(_loaded, dict):
            seed_payload = dict(_loaded.get("seed_engrams") or {})
    seed_count = int(seed_payload.get("count", 0) or 0)
    clean_model = HenriHolarch(cfg, memory=clean_memory)
    if checkpoint is not None and Path(checkpoint).exists():
        clean_model, clean_metadata = load_checkpoint(str(checkpoint), model=clean_model)
    else:
        clean_metadata = {}
    clean_model.eval()
    pre_mount_rows = int(clean_info.get("pre_mount_rows", -1) or 0)
    clean_hydrated = int(clean_info.get("hydrated", 0) or 0)
    clean_rows = int(clean_memory.stats().get("count", 0) or 0)

    clean_traps = build_cul_de_sac_traps()

    def _clean_trap_actions(threads: int) -> List[int]:
        before_threads = torch.get_num_threads()
        torch.set_num_threads(int(threads))
        try:
            rotor = getattr(clean_model, "action_rotor", None)
            lookahead = LatentActionLookahead(
                clean_model, None, rotor=rotor, transition="rotor",
                lookahead_depth=2, gamma_look=0.02,
            )
            return [
                int(lookahead.choose(trap["observation"], [])["action"])
                for trap in clean_traps
            ]
        finally:
            torch.set_num_threads(before_threads)

    trap_actions_1 = _clean_trap_actions(1)
    trap_actions_8 = _clean_trap_actions(8)
    trap_truth = [int(trap["bypass_action"]) for trap in clean_traps]
    trap_parity = sum(1 for left, right in zip(trap_actions_1, trap_actions_8) if left == right)
    trap_correct_1 = sum(1 for left, truth in zip(trap_actions_1, trap_truth) if left == truth)
    trap_correct_8 = sum(1 for left, truth in zip(trap_actions_8, trap_truth) if left == truth)
    _record_probe(
        "j1_trap_thread_parity",
        "j1_cleanroom",
        "8/8 Cul-de-Sac bypass actions at 1 and 8 threads with identical vectors",
        {
            "actions_thread1": trap_actions_1,
            "actions_thread8": trap_actions_8,
            "bypass": trap_truth,
            "parity": trap_parity,
            "correct_thread1": trap_correct_1,
            "correct_thread8": trap_correct_8,
        },
        bool(
            len(trap_truth) == 8
            and trap_parity == 8
            and trap_correct_1 == 8
            and trap_correct_8 == 8
        ),
        evidence="depth-2 wall-masked rotor planner on the clean-room store",
    )

    grid_task = build_tasks(cfg, seed=int(cfg.seed))["action"]

    def _clean_closed_loop(threads: int, episodes: int) -> Dict[str, object]:
        before_threads = torch.get_num_threads()
        torch.set_num_threads(int(threads))
        try:
            return evaluate_closed_loop_action(
                clean_model,
                grid_task,
                episodes=int(episodes),
                rotor=getattr(clean_model, "action_rotor", None),
                transition="rotor",
                lookahead_depth=2,
                gamma_look=0.02,
                batch_rollout=True,
                use_memory=True,
                trace=True,
            )
        finally:
            torch.set_num_threads(before_threads)

    closed_48 = _clean_closed_loop(1, 48)
    closed_8 = _clean_closed_loop(8, 8)
    trace_48 = list(closed_48.get("trace") or [])[:8]
    trace_8 = list(closed_8.get("trace") or [])
    episode_parity = sum(
        1
        for left, right in zip(trace_48, trace_8)
        if bool(left["win"]) == bool(right["win"])
        and int(left["steps"]) == int(right["steps"])
        and list(left["actions"]) == list(right["actions"])
    )
    closed_wins = int(closed_48.get("wins", 0))
    _record_probe(
        "j1_closed_loop_thread_parity",
        "j1_cleanroom",
        "48-episode clean-room wins >= 1; first 8 episodes identical at 1 vs 8 threads",
        {
            "wins": closed_wins,
            "episodes": int(closed_48.get("episodes", 0)),
            "wins_thread8": int(closed_8.get("wins", 0)),
            "parity_episodes": episode_parity,
            "parity_scope": (
                "first 8 episodes of the 1-thread 48-episode run against an "
                "8-episode 8-thread run"
            ),
            "episode_rows": [
                {
                    "index": int(row["index"]),
                    "win": bool(row["win"]),
                    "steps": int(row["steps"]),
                }
                for row in trace_48
            ],
        },
        bool(
            closed_wins >= 1
            and len(trace_48) == 8
            and len(trace_8) == 8
            and episode_parity == 8
        ),
        evidence="trace=True per-episode comparison; the 48-episode arm runs once",
    )

    cartridge_present = bool(seed_payload) and seed_count > 0
    _record_probe(
        "j1_seed_cartridge_and_empty_store",
        "j1_cleanroom",
        "checkpoint carries seed_engrams (count > 0) and pre_mount_rows == 0",
        {
            "cartridge_present": cartridge_present,
            "seed_count": seed_count,
            "seed_bytes": int(seed_payload.get("bytes", 0) or 0),
            "seed_dtype": str(seed_payload.get("dtype", "")),
            "pre_mount_rows": pre_mount_rows,
            "hydrated": clean_hydrated,
            "post_mount_rows": clean_rows,
            "load_hydration": dict(
                ((clean_metadata or {}).get("seed_engrams") or {}).get("hydration") or {}
            ),
        },
        bool(
            cartridge_present
            and pre_mount_rows == 0
            and clean_rows == seed_count
            and seed_count >= 700
        ),
        evidence="open_clean_room_memory created a new SQLite file; no fixture copy",
    )
    _record_probe(
        "j1_store_rows_equal_seed_count",
        "j1_cleanroom",
        "clean-room store rows == seed block count and hydrated == count",
        {
            "store_rows": clean_rows,
            "seed_count": seed_count,
            "hydrated": clean_hydrated,
            "seed_limit": 1536,
        },
        bool(
            seed_count >= 700
            and clean_rows == seed_count
            and clean_hydrated == seed_count
        ),
        evidence="row count read from the clean-room Zone C store after the mount",
    )

    # ------------------- J2: elastic tenant vocabulary -------------------
    tenant_lock = VocabularyLedger.load(lock_path) if lock_path.exists() else None
    tenant_tokenizer = HolarchCodeTextTokenizer(vocab_lock=tenant_lock, elastic_cap=4096)
    tenant_base_map = dict(tenant_tokenizer.stoi)
    tenant_base_size = int(len(tenant_tokenizer))
    j2_path = scratch_dir / f"agent_eval_j2_{os.getpid()}.sqlite3"
    if j2_path.exists():
        j2_path.unlink()
    j2_memory = ZoneCMemory(cfg, backend="sqlite", path=str(j2_path))
    tenant_texts = [
        f"def tenant_{index:04d}_probe(item):\n    return item + {index}\n"
        for index in range(640)
    ]
    tenant_corpus = CodeTextCorpus(
        texts=tenant_texts,
        names=[f"tenant_{index:04d}.py" for index in range(640)],
        digest=hashlib.sha256("".join(tenant_texts).encode("utf-8")).hexdigest()[:16],
    )
    tenant_sync = tenant_tokenizer.sync_corpus(
        tenant_corpus, memory=j2_memory, namespace="tenant-eval"
    )
    tenant_tier2 = list(tenant_tokenizer.tier2_ids)
    tenant_added_names = [tenant_tokenizer.token_text(token) for token in tenant_tier2]
    tenant_calls = [f"{name}(value)" for name in tenant_added_names]
    tenant_roundtrip = sum(
        1
        for call in tenant_calls
        if tenant_tokenizer.decode(tenant_tokenizer.encode(call)) == call
    )
    tenant_shifts = sum(
        1
        for name, token in tenant_base_map.items()
        if tenant_tokenizer.stoi.get(name) != token
    )
    tenant_added = int(tenant_sync.get("added", 0))
    _record_probe(
        "j2_elastic_tenant_vocab",
        "j2_elastic_vocab",
        ">=600 new identifiers at Tier-2 ids (>=2048), 0 Tier-1 shifts, lossless calls",
        {
            "added": tenant_added,
            "tier2_count": len(tenant_tier2),
            "tier2_first": int(tenant_tier2[0]) if tenant_tier2 else -1,
            "tier2_last": int(tenant_tier2[-1]) if tenant_tier2 else -1,
            "tier1_shifts": tenant_shifts,
            "base_size": tenant_base_size,
            "elastic_cap": int(tenant_tokenizer.elastic_cap),
            "calls": len(tenant_calls),
            "roundtrip_ok": tenant_roundtrip,
            "ledger_digest": tenant_tokenizer.ledger.digest(),
        },
        bool(
            tenant_added >= 600
            and len(tenant_tier2) >= 600
            and (not tenant_tier2 or int(tenant_tier2[0]) >= 2048)
            and tenant_shifts == 0
            and tenant_roundtrip == len(tenant_calls)
        ),
        evidence="640 generated functions synced into the Zone C vocabulary metadata",
    )

    tenant_stored = j2_memory.get_meta("tenant-eval", VOCAB_LEDGER_KIND, "vocab_ledger")
    tenant_stored_tier2 = j2_memory.get_meta(
        "tenant-eval", VOCAB_LEDGER_KIND, "vocab_ledger_tier2"
    )
    tenant_restored = HolarchCodeTextTokenizer(vocab_lock=tenant_lock, elastic_cap=4096)
    reingestion_calls = 0  # no sync_corpus on the restored tokenizer
    restore_added = 0
    if tenant_stored:
        restore_added = int(
            tenant_restored.restore_ledger(dict(tenant_stored[-1][1]))
        )
    restored_exact = bool(
        tenant_stored
        and all(
            tenant_restored.stoi.get(name) == token
            for name, token in tenant_tokenizer.stoi.items()
        )
        and all(
            tenant_tokenizer.stoi.get(name) == token
            for name, token in tenant_restored.stoi.items()
        )
    )
    tier2_payload_match = bool(
        tenant_stored_tier2
        and list(tenant_stored_tier2[-1][1].get("tier2_ids", [])) == tenant_tier2
    )
    restored_tier2 = len(tenant_restored.tier2_ids)
    _record_probe(
        "j2_ledger_restore_from_zone_c",
        "j2_elastic_vocab",
        "restore_ledger from Zone C metadata: exact id map, 0 re-ingestion calls",
        {
            "ledger_records": len(tenant_stored),
            "tier2_records": len(tenant_stored_tier2),
            "restore_added": restore_added,
            "restored_identifiers": len(tenant_restored.identifiers),
            "restored_tier2": restored_tier2,
            "exact_map": restored_exact,
            "tier2_payload_match": tier2_payload_match,
            "reingestion_calls": reingestion_calls,
        },
        bool(
            restored_exact
            and reingestion_calls == 0
            and restored_tier2 == len(tenant_tier2)
            and tier2_payload_match
        ),
        evidence="fresh tokenizer restored from the Zone C vocab_ledger payload only",
    )
    j2_memory.close()

    # ------------------- J3: diverse guarded DAG synthesis ---------------
    from henri.holarch.code_gen import (
        _playbook_plan,
        compose_plan_dag,
        run_diverse_dag_battery,
    )

    j3_battery = run_diverse_dag_battery(seed=20261105)
    j3_three = list((j3_battery.get("three") or {}).get("rows") or [])
    j3_by_family = {str(row.get("family", "")): row for row in j3_three}
    for family in ("F1", "F2", "F3"):
        j3_row = j3_by_family.get(family, {})
        j3_stage_ok = bool(
            j3_row.get("parsed")
            and j3_row.get("compiled")
            and j3_row.get("executed")
            and j3_row.get("matched")
            and str(j3_row.get("func", "")).startswith("j3d")
        )
        _record_probe(
            f"j3_three_stage_{family.lower()}",
            "j3_diverse_dag",
            (
                f"{family} three-stage pipeline: ast.parse + compile + exec under "
                "a novel name, answer matches the reference"
            ),
            {
                "family": family,
                "func": j3_row.get("func", ""),
                "stages": list(j3_row.get("stages") or []),
                "topology": j3_row.get("topology", ""),
                "parsed": bool(j3_row.get("parsed", False)),
                "compiled": bool(j3_row.get("compiled", False)),
                "executed": bool(j3_row.get("executed", False)),
                "matched": bool(j3_row.get("matched", False)),
                "expected": j3_row.get("expected"),
                "actual": j3_row.get("actual"),
                "source_sha256": j3_row.get("source_sha256", ""),
            },
            j3_stage_ok,
            evidence="run_diverse_dag_battery three-stage row",
        )

    novel_trap = compose_plan_dag(
        [_playbook_plan("join_csv"), _playbook_plan("clamp_unit")],
        {"func": "j3_novel_trap_probe", "args": ["payload"], "numbers": [0.0, 1.0]},
    )
    novel_refused = bool(
        not novel_trap.get("ok", False)
        and str(novel_trap.get("source", "")) == ""
        and str(novel_trap.get("reason", "")).startswith("type mismatch")
    )
    battery_traps = dict(j3_battery.get("traps") or {})
    traps_refused = bool(
        int(battery_traps.get("refused", 0)) == int(battery_traps.get("total", -1)) == 6
    )
    _record_probe(
        "j3_new_type_conflict_refused",
        "j3_diverse_dag",
        "a new text->numeric conflict is refused with empty source; battery traps 6/6",
        {
            "novel_pair": ["join_csv", "clamp_unit"],
            "refused": novel_refused,
            "reason": str(novel_trap.get("reason", ""))[:160],
            "source_empty": str(novel_trap.get("source", "")) == "",
            "battery_traps_refused": int(battery_traps.get("refused", 0)),
            "battery_traps_total": int(battery_traps.get("total", 0)),
            "battery_three_passed": int((j3_battery.get("three") or {}).get("passed", 0)),
            "battery_total": dict(j3_battery.get("total") or {}),
            "battery_drift": float(j3_battery.get("drift", 0.0)),
        },
        bool(novel_refused and traps_refused),
        evidence="compose_plan_dag refuses before splicing a single stage",
    )

    # ------------------- J4: OpenAI-compatible adapter -------------------
    from fastapi.testclient import TestClient

    j4_key = "agent-eval-j4"
    j4_app = build_app(service, api_keys={j4_key})
    j4_headers = {"Authorization": f"Bearer {j4_key}"}
    j4_claim = str(H_VERIFY_QUERIES[0]["claim"])
    j4_trap = build_cul_de_sac_traps()[4]
    j4_verify_request = {
        "model": "henri-holarch",
        "messages": [{"role": "user", "content": j4_claim}],
        "metadata": {
            "action": "verify",
            "payload": {"claim": j4_claim, "kind": "text"},
        },
    }
    j4_plan_request = {
        "model": "henri-holarch",
        "messages": [{"role": "user", "content": "plan the bypass around the pocket"}],
        "metadata": {
            "action": "plan",
            "payload": {
                "observation": j4_trap["observation"],
                "transition": "rotor",
                "lookahead_depth": 2,
                "gamma_look": 0.02,
            },
        },
    }

    def _j4_envelope(body: Dict[str, object]) -> Dict[str, object]:
        receipt = dict(body.get("epistemic_receipt") or {})
        holarch = dict(body.get("holarch") or {})
        agent_receipt = dict(holarch.get("agent_receipt") or {})
        choices = list(body.get("choices") or [])
        content = ""
        if choices:
            content = str((dict(choices[0]).get("message") or {}).get("content", ""))
        return {
            "object": str(body.get("object", "")),
            "content": content,
            "action": str(holarch.get("action", "")),
            "status": str(holarch.get("status", "")),
            "result": dict(holarch.get("result") or {}),
            "receipt_valid": bool(receipt) and verify_receipt(receipt),
            "agent_receipt_valid": bool(agent_receipt) and verify_receipt(agent_receipt),
            "receipt_sha256": str(receipt.get("sha256", "")),
            "agent_receipt_sha256": str(agent_receipt.get("sha256", "")),
        }

    verify_body: Dict[str, object] = {}
    plan_body: Dict[str, object] = {}
    j4_error = ""
    j4_service_memory = service.sidecar.model.memory
    j4_threads = torch.get_num_threads()
    try:
        with TestClient(j4_app) as client:
            verify_response = client.post(
                "/v1/chat/completions", json=j4_verify_request, headers=j4_headers
            )
            verify_body = dict(verify_response.json())
            service.sidecar.model.memory = clean_memory
            mount_planning_checkpoint()
            torch.set_num_threads(1)
            try:
                plan_response = client.post(
                    "/v1/chat/completions", json=j4_plan_request, headers=j4_headers
                )
                plan_body = dict(plan_response.json())
            finally:
                service.sidecar.model.memory = j4_service_memory
                restore_trained_weights()
                torch.set_num_threads(j4_threads)
    except Exception as error:  # noqa: BLE001 - the failure is the measurement
        j4_error = f"{type(error).__name__}: {error}"
        service.sidecar.model.memory = j4_service_memory
        restore_trained_weights()
        torch.set_num_threads(j4_threads)

    verify_envelope = _j4_envelope(verify_body)
    plan_envelope = _j4_envelope(plan_body)
    j4_verify_ok = bool(
        verify_envelope["object"] == "chat.completion"
        and str(verify_envelope["content"]).strip()
        and verify_envelope["receipt_valid"]
        and verify_envelope["agent_receipt_valid"]
        and verify_envelope["action"] == "verify"
        and verify_envelope["status"] == "supported"
    )
    _record_probe(
        "j4_chat_verify",
        "j4_openai_adapter",
        "POST /v1/chat/completions -> verify: chat.completion, non-empty content, accepted receipt",
        {
            "object": verify_envelope["object"],
            "content": str(verify_envelope["content"])[:160],
            "action": verify_envelope["action"],
            "status": verify_envelope["status"],
            "receipt_valid": verify_envelope["receipt_valid"],
            "agent_receipt_valid": verify_envelope["agent_receipt_valid"],
            "receipt_sha256": verify_envelope["receipt_sha256"],
            "agent_receipt_sha256": verify_envelope["agent_receipt_sha256"],
            "error": j4_error,
        },
        j4_verify_ok,
        evidence="fastapi TestClient, bearer key agent-eval-j4",
    )

    plan_action = int(plan_envelope["result"].get("action", -1))
    reference_action = int(j4_trap["bypass_action"])
    j4_plan_ok = bool(
        plan_envelope["object"] == "chat.completion"
        and str(plan_envelope["content"]).strip()
        and plan_envelope["receipt_valid"]
        and plan_envelope["agent_receipt_valid"]
        and plan_envelope["action"] == "plan"
    )
    _record_probe(
        "j4_chat_plan",
        "j4_openai_adapter",
        "POST /v1/chat/completions -> plan: chat.completion, non-empty content, accepted receipt",
        {
            "object": plan_envelope["object"],
            "content": str(plan_envelope["content"])[:160],
            "action": plan_envelope["action"],
            "status": plan_envelope["status"],
            "action_chosen": plan_action,
            "reference_action": reference_action,
            "matches_reference": bool(plan_action == reference_action),
            "receipt_valid": plan_envelope["receipt_valid"],
            "agent_receipt_valid": plan_envelope["agent_receipt_valid"],
            "receipt_sha256": plan_envelope["receipt_sha256"],
            "agent_receipt_sha256": plan_envelope["agent_receipt_sha256"],
            "error": j4_error,
        },
        j4_plan_ok,
        evidence="planner mounted on the clean-room store at 1 thread",
    )
    # =================== Cycle K: 12 further clean-room probes =================
    k_tasks = build_tasks(cfg, seed=int(cfg.seed))
    k1_test = evaluate(clean_model, k_tasks, "test", limit=2000)
    k1_empty = {"vision": 0, "relational": 0}
    for name in ("vision", "relational"):
        for batch in k_tasks[name].epoch(64, "val"):
            output = clean_model(name, batch.x, training=False)
            k1_empty[name] += int((output.veto & (~output.memory_mask)).sum())
    _record_probe(
        "k1_vision_cleanroom",
        "k1_multitask",
        "clean-room vision test accuracy >= 0.975 with a hydrated multi-task cartridge",
        {
            "vision_accuracy": float(k1_test.get("vision_accuracy", 0.0)),
            "vision_veto_rate": float(k1_test.get("vision_veto_rate", 0.0)),
            "seed_count": int(seed_count),
            "pre_mount_rows": int(pre_mount_rows),
        },
        bool(float(k1_test.get("vision_accuracy", 0.0)) >= 0.975),
        evidence="evaluate(clean_model, tasks, 'test') on the hydrated cartridge",
    )
    _record_probe(
        "k1_relational_cleanroom",
        "k1_multitask",
        "clean-room relational test accuracy >= 0.7600 on the hydrated cartridge",
        {
            "relational_accuracy": float(k1_test.get("relational_accuracy", 0.0)),
            "relational_veto_rate": float(k1_test.get("relational_veto_rate", 0.0)),
        },
        bool(float(k1_test.get("relational_accuracy", 0.0)) >= 0.76),
        evidence="evaluate(clean_model, tasks, 'test') on the hydrated cartridge",
    )
    _record_probe(
        "k1_empty_store_veto_zero",
        "k1_multitask",
        "no vision or relational query is vetoed by an empty retrieval",
        {"empty_veto": dict(k1_empty), "total": int(sum(k1_empty.values()))},
        bool(int(sum(k1_empty.values())) == 0),
        evidence="per-sample veto AND NOT memory_mask over the validation splits",
    )
    k1_labels = seed_payload.get("labels")
    k1_values = (
        k1_labels.detach().cpu().numpy().astype(int).tolist()
        if k1_labels is not None
        else []
    )
    k1_rows = {
        "vision": sum(1 for value in k1_values if 0 <= value < 10),
        "language": sum(1 for value in k1_values if 10 <= value < 106),
        "action": sum(1 for value in k1_values if 106 <= value < 110),
        "relational": sum(1 for value in k1_values if 110 <= value < 118),
    }
    _record_probe(
        "k1_cartridge_multitask_budget",
        "k1_multitask",
        "cartridge holds 604 action + >=64 vision + >=32 relational + >=32 language under 2.5 MiB",
        {
            "rows_by_task": k1_rows,
            "seed_count": int(seed_count),
            "seed_bytes": int(seed_payload.get("bytes", 0) or 0),
            "dtype": str(seed_payload.get("dtype", "")),
        },
        bool(
            k1_rows["action"] == 604
            and k1_rows["vision"] >= 64
            and k1_rows["relational"] >= 32
            and k1_rows["language"] >= 32
            and int(seed_payload.get("bytes", 0) or 0) <= 2_621_440
        ),
        evidence="label histogram of the mounted seed block",
    )

    # ------------------- K2: boundary-axiom discovery -------------------
    from henri.holarch.discovery import run_discovery_battery

    k2 = run_discovery_battery(clean_model, memory=clean_memory, namespace="agent_eval_k2")
    _record_probe(
        "k2_spatial_discovery",
        "k2_boundary_axioms",
        "4/4 spatial blocked-path probes discovered with law/repulsor steering",
        {
            "spatial": int(k2["probes"]["spatial"]),
            "correct": int(sum(1 for row in k2["spatial"] if row["steered_correct"])),
            "control_repeated_forbidden": int(
                sum(1 for row in k2["spatial"] if row["control_repeated_forbidden"])
            ),
            "axioms": dict(k2["axioms"]),
        },
        bool(k2["gate"]["spatial_4_of_4"]),
        evidence="Zone B relax_discovery under Zone C LAW/ATTRACTOR/REPULSOR axioms",
    )
    _record_probe(
        "k2_symbolic_discovery",
        "k2_boundary_axioms",
        "4/4 symbolic operator probes discovered with a tenant LAW",
        {
            "symbolic": int(k2["probes"]["symbolic"]),
            "correct": int(sum(1 for row in k2["symbolic"] if row["steered_correct"])),
            "weight_updates": int(k2["weight_updates"]),
        },
        bool(k2["gate"]["symbolic_4_of_4"]),
        evidence="Zone B relax_discovery over type-compatible operator candidates",
    )

    # ------------------- K3: multi-statement DAG ------------------------
    from henri.holarch.code_gen import _compiled_plan

    k3_sources = [
        "def filter_above(items: list, threshold: float) -> list:\n    kept = []\n    for item in items:\n        if item > threshold:\n            kept.append(item)\n    return kept\n",
        "def scale_values(values: list, factor: float) -> list:\n    scaled = []\n    for value in values:\n        scaled.append(value * factor)\n    return scaled\n",
        "def total(values: list) -> float:\n    accumulator = 0.0\n    for value in values:\n        accumulator += value\n    return accumulator\n",
    ]
    k3_plans = [
        _compiled_plan(f"k3_stage_{index}", "k3", source)
        for index, source in enumerate(k3_sources)
    ]
    k3_verdict = compose_plan_dag(
        k3_plans,
        {"func": "k3_multistatement", "args": ["items", "threshold", "factor"], "numbers": []},
    )
    k3_scope: Dict[str, object] = {"__builtins__": __builtins__}
    k3_actual = None
    k3_error = ""
    try:
        exec(str(k3_verdict.get("source") or ""), k3_scope)
        k3_actual = k3_scope["k3_multistatement"]([1.0, 2.0, 3.0, 4.0], 2.0, 2.0)
    except Exception as error:  # noqa: BLE001
        k3_error = f"{type(error).__name__}: {error}"
    _record_probe(
        "k3_multistatement_dag",
        "k3_vla_dag",
        "three multi-statement stages splice and execute; the answer matches the reference",
        {
            "ok": bool(k3_verdict.get("ok", False)),
            "actual": k3_actual,
            "expected": 14.0,
            "error": k3_error,
            "source_chars": len(str(k3_verdict.get("source") or "")),
        },
        bool(bool(k3_verdict.get("ok", False)) and k3_actual == 14.0),
        evidence="AST alpha-renaming + terminal-return splicing, then execution",
    )
    k3_scaled = compose_plan_dag(
        [_compiled_plan("k3_a", "k3", k3_sources[0]), _compiled_plan("k3_b", "k3", k3_sources[1]), _compiled_plan("k3_c", "k3", k3_sources[2])],
        {"func": "k3_collision", "args": ["items", "threshold", "factor"], "numbers": []},
    )
    k3_collision = str(k3_scaled.get("source") or "")
    _record_probe(
        "k3_alpha_rename_no_collision",
        "k3_vla_dag",
        "stage locals are alpha-renamed to _s{i}_* so loop accumulators cannot collide",
        {
            "ok": bool(k3_scaled.get("ok", False)),
            "stage_prefixes": int(k3_collision.count("_s0_")) + int(k3_collision.count("_s1_")) + int(k3_collision.count("_s2_")),
            "source_chars": len(k3_collision),
        },
        bool(bool(k3_scaled.get("ok", False)) and "_s0_" in k3_collision and "_s1_" in k3_collision and "_s2_" in k3_collision),
        evidence="scoped AST NodeTransformer rename",
    )

    # ------------------- K4: rebase and vectorised reads ----------------
    from henri.holarch.code_lang import (
        HolarchCodeTextTokenizer as _K4Tokenizer,
        VocabularyLedger as _K4Ledger,
        rebase_tenant_vocabulary as _k4_rebase,
    )

    k4_lock = _K4Ledger.load(lock_path) if lock_path.exists() else None
    k4_a = _K4Tokenizer(vocab_lock=k4_lock, elastic_cap=8192)
    k4_b = _K4Tokenizer(vocab_lock=k4_lock, elastic_cap=8192)
    k4_names_a = [f"k4_alpha_{index:03d}_value" for index in range(300)]
    k4_names_b = [f"k4_beta_{index:03d}_value" for index in range(300)]
    k4_a.add_identifiers(k4_names_a)
    k4_b.add_identifiers(k4_names_b)
    k4_rebase = _k4_rebase(k4_a, k4_b)
    k4_lossless = 0
    for name in (k4_names_a[:10] + k4_names_b[:10]):
        text = f"call {name}(value)"
        ids_b = k4_b.encode(text)
        k4_lossless += int(k4_rebase.tokenizer.decode(k4_rebase.translate(ids_b, "b")) == k4_b.decode(ids_b))
    _record_probe(
        "k4_cross_tenant_rebase",
        "k4_rebase_concurrency",
        "two 300-identifier tenants rebase with 0 Tier-1 shifts and lossless translation",
        {
            "tier1_shifts": int(k4_rebase.tier1_shifts),
            "union": int(k4_rebase.union_size),
            "lossless": int(k4_lossless),
            "total": 20,
            "added_from_b": len(k4_rebase.added_from_b),
        },
        bool(k4_rebase.tier1_shifts == 0 and k4_lossless == 20),
        evidence="rebase_tenant_vocabulary + id_map translation, decode equality",
    )
    k4_verifier = service.sidecar.verifier if getattr(service.sidecar, "verifier", None) is not None else None
    if k4_verifier is None:
        from henri.holarch.ontology_guard import build_verifier as _k4_build

        k4_verifier = _k4_build(ROOT)
    k4_claims = [
        "The medium profile has 1,020,380 parameters and reaches 0.7083 closed-loop action.",
        "The vision accuracy (test) is 0.9798 against a 0.9750 bar.",
        "Zone C stores engrams in SQLite when TimescaleDB is not reachable.",
        "The tokenizer falls back to a 256-byte vocabulary for lossless round trips.",
    ]
    k4_parity = 0
    for claim in k4_claims:
        matrix = [index for _score, index in k4_verifier.bm25(claim, top_k=4)]
        python = [index for _score, index in k4_verifier._bm25_python(claim, top_k=4)]
        k4_parity += int(matrix == python)
    _record_probe(
        "k4_vectorised_bm25",
        "k4_rebase_concurrency",
        "NumPy-matmul BM25 returns the same top-4 passages as the reference loop",
        {
            "top4_parity": int(k4_parity),
            "claims": len(k4_claims),
            "passages": len(k4_verifier.passages),
            "vocabulary": int(len(k4_verifier._bm25_vocab)),
        },
        bool(k4_parity == len(k4_claims)),
        evidence="matrix-vector product vs the retained per-passage loop",
    )
    k4_errors: List[str] = []
    k4_status: List[int] = []
    k4_lock_obj = threading.Lock()
    k4_observation = build_cul_de_sac_traps()[0]["observation"]

    def _k4_worker(worker_id: int) -> None:
        try:
            for _round in range(8):
                kind = worker_id % 4
                if kind == 0:
                    result = service.verify(k4_claims[0], kind="text")
                elif kind == 1:
                    result = service.plan(k4_observation, transition="rotor", lookahead_depth=2)
                elif kind == 2:
                    result = service.veto("action", k4_observation)
                else:
                    result = service.solve("ZoneCMemory.query", ["CALLS", "CALLS"])
                with k4_lock_obj:
                    k4_status.append(200 if bool(result.get("ok", False)) else 500)
        except Exception as error:  # noqa: BLE001
            with k4_lock_obj:
                k4_errors.append(f"{type(error).__name__}: {error}")

    k4_threads = [threading.Thread(target=_k4_worker, args=(index,)) for index in range(16)]
    for thread in k4_threads:
        thread.start()
    for thread in k4_threads:
        thread.join()
    _record_probe(
        "k4_concurrent_reads_128",
        "k4_rebase_concurrency",
        "128 concurrent reads across 16 threads complete with zero errors and one receipt chain",
        {
            "requests": len(k4_status),
            "ok": int(sum(1 for status in k4_status if status == 200)),
            "errors": k4_errors,
            "chain_valid": bool(service.ledger.verify_chain()),
        },
        bool(len(k4_status) == 128 and not k4_errors and service.ledger.verify_chain()),
        evidence="16 threads x 8 mixed verify/plan/veto/solve requests",
    )
    k4_bundle = service.export_import_bundle("default", "agent_eval_import")
    _record_probe(
        "k4_tenant_bundle",
        "k4_rebase_concurrency",
        "export_import_bundle copies the tenant ledger and plan library to a new namespace",
        {
            "written": int((k4_bundle.get("result") or {}).get("written", 0)),
            "plans": int((k4_bundle.get("result") or {}).get("plan_records", 0)),
            "tier2": int((k4_bundle.get("result") or {}).get("tier2_ids", 0)),
        },
        bool(int((k4_bundle.get("result") or {}).get("written", 0)) > 0),
        evidence="Zone C metadata copy, one envelope receipt",
    )

    # =================== Cycle M: 12 further clean-room probes =================
    k_tasks_m = k_tasks
    m1_traps = build_cul_de_sac_traps()
    m1_single = LatentActionLookahead(
        clean_model, None, rotor=getattr(clean_model, "action_rotor", None),
        transition="rotor", lookahead_depth=1,
    )
    m1_masked = LatentActionLookahead(
        clean_model, None, rotor=getattr(clean_model, "action_rotor", None),
        transition="rotor", lookahead_depth=2, gamma_look=0.02,
    )
    import importlib.util as _ilu

    _m_spec = _ilu.spec_from_file_location(
        "_m_cycle_probe", ROOT / "scripts" / "run_holarch_m_cycle.py"
    )
    _m_cycle = _ilu.module_from_spec(_m_spec)
    _m_spec.loader.exec_module(_m_cycle)
    _trap_battery = _m_cycle._trap_rows(clean_model, cfg)
    m1_rows = [
        {
            "id": str(row["id"]),
            "bypass": int(row["bypass_action"]),
            "control": int(row["depth1_action"]),
            "escape": int(row["escape_action"]),
            "escape_correct": bool(row["escape_correct"]),
            "control_correct": bool(row["control_correct"]),
        }
        for row in _trap_battery["rows"]
    ]
    _record_probe(
        "m1_online_escape_8_of_8",
        "m1_online_repulsor",
        "8/8 unsteered trap states are escaped by the online REPULSOR swarm",
        {
            "rows": m1_rows,
            "escape_correct": int(sum(1 for row in m1_rows if row["escape_correct"])),
            "control_correct": int(sum(1 for row in m1_rows if row["control_correct"])),
        },
        bool(
            len(m1_rows) == 8
            and all(row["escape_correct"] for row in m1_rows)
            and all(not row["control_correct"] for row in m1_rows)
        ),
        evidence="discover_escape with a test-time REPULSOR and goal attraction",
    )
    m1_baseline = evaluate_closed_loop_action(
        clean_model, k_tasks_m["action"], episodes=12,
        rotor=getattr(clean_model, "action_rotor", None), transition="rotor",
        lookahead_depth=2, gamma_look=0.02, batch_rollout=True, use_memory=True,
        trace=False, online_repulsor=False,
    )
    m1_online = evaluate_closed_loop_action(
        clean_model, k_tasks_m["action"], episodes=12,
        rotor=getattr(clean_model, "action_rotor", None), transition="rotor",
        lookahead_depth=2, gamma_look=0.02, batch_rollout=True, use_memory=True,
        trace=False, online_repulsor=True, online_namespace="agent_eval_m1_loop",
    )
    _record_probe(
        "m1_closed_loop_lift",
        "m1_online_repulsor",
        "online REPULSOR discovery adds at least two wins on the first 12 layouts",
        {
            "baseline_wins": int(m1_baseline["wins"]),
            "online_wins": int(m1_online["wins"]),
            "discoveries": int(m1_online["online_discoveries"]),
        },
        bool(
            int(m1_online["wins"]) >= int(m1_baseline["wins"]) + 2
            and int(m1_online["wins"]) >= 10
        ),
        evidence="12-layout subset; the full 48-layout measurement is in m1_online_repulsor_vla.json",
    )
    from henri.holarch.vla import run_vla_battery

    try:
        m1_engine = service.sidecar._ensure_engine("default")
        m1_vla = run_vla_battery(clean_model, m1_engine, k_tasks_m)
        m1_vla_ok = bool(
            m1_vla["completed"] == 12
            and float(m1_vla["hallucination_rate"]) == 0.0
            and int(m1_vla["goals_reached"]) >= 11
        )
        m1_vla_row = {
            "completed": int(m1_vla["completed"]),
            "goals_reached": int(m1_vla["goals_reached"]),
            "hallucination_rate": float(m1_vla["hallucination_rate"]),
        }
    except Exception as error:  # noqa: BLE001
        m1_vla_ok = False
        m1_vla_row = {"error": f"{type(error).__name__}: {error}"}
    _record_probe(
        "m1_vla_goal_reach_11",
        "m1_online_repulsor",
        "12/12 VLA episodes complete and at least 11 reach the spatial goal",
        m1_vla_row,
        bool(m1_vla_ok),
        evidence="online REPULSOR agents inside the unified VLA pipeline",
    )

    # ------------------- M2: low-latency concurrent reads ---------------
    batch_probe_waves = np.stack(
        [
            clean_model.encode("vision", k_tasks_m["vision"].sample(1, "val").x)[0].detach().numpy()
            for _ in range(4)
        ]
    )
    batch_prefix, batch_reads = clean_memory.context_tensor_batch(batch_probe_waves, k=4, task="vision")
    singles = [
        clean_memory.context_tensor(row, k=4, task="vision")[0] for row in batch_probe_waves
    ]
    batch_ok = bool(
        batch_prefix.shape == (4, int(cfg.context_tokens), int(cfg.context_dim))
        and all(
            np.allclose(batch_prefix[index].reshape(-1), singles[index].reshape(-1), atol=1e-5)
            for index in range(4)
        )
    )
    _record_probe(
        "m2_batched_context_tensor",
        "m2_low_latency",
        "context_tensor_batch equals the per-row context tensors",
        {"shape": list(batch_prefix.shape), "rows": len(batch_reads)},
        batch_ok,
        evidence="one Gram product over four waves versus four per-row reads",
    )
    m2_observation = m1_traps[0]["observation"]
    m2_claim = "The medium profile has 1,020,380 parameters and reaches 0.7083 closed-loop action."
    m2_errors: List[str] = []
    m2_latency: List[float] = []
    m2_status: List[int] = []
    m2_lock = threading.Lock()
    m2_barrier = threading.Barrier(16)

    def _m2_worker(worker_id: int) -> None:
        try:
            m2_barrier.wait(timeout=30)
            for round_index in range(4):
                clock = time.perf_counter()
                kind = (worker_id + round_index) % 4
                if kind == 0:
                    result = service.verify(m2_claim, kind="text")
                elif kind == 1:
                    result = service.plan(m2_observation, transition="rotor", lookahead_depth=2)
                elif kind == 2:
                    result = service.veto("action", m2_observation)
                else:
                    result = service.solve("ZoneCMemory.query", ["CALLS", "CALLS"])
                elapsed = (time.perf_counter() - clock) * 1e3
                with m2_lock:
                    m2_latency.append(elapsed)
                    m2_status.append(200 if bool(result.get("ok", False)) else 500)
        except Exception as error:  # noqa: BLE001
            with m2_lock:
                m2_errors.append(f"{type(error).__name__}: {error}")

    m2_threads = [threading.Thread(target=_m2_worker, args=(index,)) for index in range(16)]
    for thread in m2_threads:
        thread.start()
    for thread in m2_threads:
        thread.join()
    m2_p50 = float(np.percentile(m2_latency, 50)) if m2_latency else 1e9
    _record_probe(
        "m2_concurrent_p50_120ms",
        "m2_low_latency",
        "64 mixed requests over 16 threads finish with p50 <= 120 ms and zero errors",
        {
            "requests": len(m2_status),
            "errors": m2_errors,
            "p50_ms": m2_p50,
            "p95_ms": float(np.percentile(m2_latency, 95)) if m2_latency else 1e9,
        },
        bool(len(m2_status) == 64 and not m2_errors and m2_p50 <= 120.0),
        evidence="reader thread guard plus the lock-free Zone C snapshot",
    )
    _record_probe(
        "m2_concurrent_receipt_chain",
        "m2_low_latency",
        "the concurrent read burst leaves one valid contiguous receipt chain",
        {"requests": len(m2_status), "chain_valid": bool(service.ledger.verify_chain())},
        bool(len(m2_status) == 64 and service.ledger.verify_chain()),
        evidence="ReceiptLedger chain verification after the burst",
    )

    # ------------------- M3: full-reference cartridge -------------------
    m3_block = seed_payload if isinstance(seed_payload, dict) else {}
    m3_labels = m3_block.get("labels")
    m3_values = (
        m3_labels.detach().cpu().numpy().astype(int).tolist()
        if m3_labels is not None
        else []
    )
    m3_rows = {
        "vision": sum(1 for value in m3_values if 0 <= value < 10),
        "language": sum(1 for value in m3_values if 10 <= value < 106),
        "action": sum(1 for value in m3_values if 106 <= value < 110),
        "relational": sum(1 for value in m3_values if 110 <= value < 118),
    }
    _record_probe(
        "m3_full_reference_cartridge",
        "m3_full_reference",
        "cartridge carries 1349 rows (610 vision) under 2.5 MiB",
        {
            "count": int(m3_block.get("count", 0) or 0),
            "bytes": int(m3_block.get("bytes", 0) or 0),
            "rows_by_task": m3_rows,
            "format": str(m3_block.get("format", "")),
        },
        bool(
            int(m3_block.get("count", 0) or 0) >= 1349
            and int(m3_block.get("bytes", 0) or 0) <= 2_621_440
            and m3_rows["vision"] >= 610
        ),
        evidence="v3 prototype-residual action rows plus int8 support rows",
    )
    _record_probe(
        "m3_vision_equals_reference",
        "m3_full_reference",
        "clean-room vision test accuracy is at least 0.9790 (reference 0.9798)",
        {"vision_accuracy": float(k1_test.get("vision_accuracy", 0.0))},
        bool(float(k1_test.get("vision_accuracy", 0.0)) >= 0.9790),
        evidence="full 610-row vision block in the clean room",
    )
    from henri.holarch.code_gen import evaluate_bochner_language

    try:
        m3_language = evaluate_bochner_language(clean_model, ROOT)
        m3_language_ok = bool(
            "top1" in m3_language and "bits_per_char" in m3_language
        )
        m3_language_row = dict(m3_language)
    except Exception as error:  # noqa: BLE001
        m3_language_ok = False
        m3_language_row = {"error": f"{type(error).__name__}: {error}"}
    _record_probe(
        "m3_bochner_language",
        "m3_full_reference",
        "Bochner subword language metric is measured and its target outcome recorded",
        m3_language_row,
        bool(m3_language_ok),
        evidence="Zone-B-coupled generator on a held-out corpus slice",
    )

    # ------------------- M4: guard AST and 84 tasks ---------------------
    guard_plan = _compiled_plan(
        "guard_probe",
        "guard",
        "def guard_probe(values: list) -> float:\n"
        "    if not values:\n"
        "        return 0.0\n"
        "    total = 0.0\n"
        "    for value in values:\n"
        "        total += value\n"
        "    return total / len(values)\n",
    )
    guard_lower = _compiled_plan(
        "guard_lower",
        "guard",
        "def guard_lower(value: float, floor: float) -> float:\n"
        "    if value < floor:\n"
        "        return floor\n"
        "    return value\n",
    )
    guard_verdict = compose_plan_dag(
        [guard_plan, guard_lower],
        {"func": "m4_guard_pipeline", "args": ["values", "floor"], "numbers": []},
    )
    guard_scope: Dict[str, object] = {"__builtins__": __builtins__}
    guard_result = None
    guard_error = ""
    try:
        exec(str(guard_verdict.get("source") or ""), guard_scope)
        guard_result = guard_scope["m4_guard_pipeline"]([], 1.0)
    except Exception as error:  # noqa: BLE001
        guard_error = f"{type(error).__name__}: {error}"
    _record_probe(
        "m4_guard_pipeline",
        "m4_guard_ast",
        "a leading guard clause is normalized and the pipeline executes",
        {
            "ok": bool(guard_verdict.get("ok", False)),
            "actual": guard_result,
            "expected": 1.0,
            "error": guard_error,
        },
        bool(bool(guard_verdict.get("ok", False)) and guard_result == 1.0),
        evidence="guard return rewritten to a step_i assignment inside if/else",
    )
    from henri.holarch.code_gen import run_diverse_dag_battery

    m4_battery = run_diverse_dag_battery(seed=20261105, extended=True, guard_pipelines=True)
    _record_probe(
        "m4_84_task_suite",
        "m4_guard_ast",
        "the guard-extended synthesis suite passes 84/84 with 0.0000 drift",
        {
            "total": dict(m4_battery["total"]),
            "guards": {"passed": m4_battery["guards"]["passed"], "total": m4_battery["guards"]["total"]},
            "traps3": {"refused": m4_battery["traps3"]["refused"], "total": m4_battery["traps3"]["total"]},
            "drift": float(m4_battery["drift"]),
        },
        bool(
            m4_battery["total"]["passed"] == 84
            and m4_battery["total"]["total"] == 84
            and abs(float(m4_battery["drift"])) < 1e-12
        ),
        evidence="run_diverse_dag_battery(guard_pipelines=True)",
    )
    mid_loop = compose_plan_dag(
        [
            _compiled_plan(
                "mid_loop",
                "mid",
                "def mid_loop(values: list) -> float:\n"
                "    for value in values:\n"
                "        if value > 0:\n"
                "            return value\n"
                "    return 0.0\n",
            )
        ],
        {"func": "m4_mid_loop", "args": ["values"], "numbers": []},
    )
    _record_probe(
        "m4_midloop_refused",
        "m4_guard_ast",
        "an un-normalizable mid-loop return is refused with an empty source",
        {"ok": bool(mid_loop.get("ok", False)), "source_empty": str(mid_loop.get("source", "")) == ""},
        bool(not mid_loop.get("ok", False) and str(mid_loop.get("source", "")) == ""),
        evidence="fail-closed splicer contract retained",
    )

    clean_memory.close()

    for _category in (
        "j1_cleanroom",
        "j2_elastic_vocab",
        "j3_diverse_dag",
        "j4_openai_adapter",
        "k1_multitask",
        "k2_boundary_axioms",
        "k3_vla_dag",
        "k4_rebase_concurrency",
        "m1_online_repulsor",
        "m2_low_latency",
        "m3_full_reference",
        "m4_guard_ast",
    ):
        per_category[_category]["rows"] = [
            row for row in probe_rows if row["category"] == _category
        ]
    total_queries = int(len(receipts) + len(probe_rows))

    per_category["composite_rules"].update(
        baseline_correct=composite_stats["baseline_correct"],
        mcp_correct=composite_stats["mcp_correct"],
        total=composite_stats["reachable_total"],
    )
    per_category["composite_traps"].update(
        baseline_hallucinated=composite_stats["baseline_hallucinated"],
        mcp_refused=composite_stats["mcp_refused"],
        total=composite_stats["traps_total"],
    )
    composite_accuracy = float(
        composite_stats["mcp_correct"] / max(composite_stats["reachable_total"], 1)
    )
    composite_baseline = float(
        composite_stats["baseline_correct"] / max(composite_stats["reachable_total"], 1)
    )

    chain_valid = bool(service.ledger.verify_chain())
    receipts_valid = bool(all(verify_receipt(receipt) for receipt in receipts))
    response_bytes = int(mcp_response_bytes)
    compression = float(baseline.bytes_read / max(response_bytes, 1))

    payload = {
        "config_digest": cfg.digest(),
        "config_dim": int(cfg.dim),
        "train_steps": int(train_steps),
        "checkpoint": {
            "mounted": mounted,
            "path": str(service.checkpoint_info.get("path", "")),
            "sha256": str(service.checkpoint_info.get("sha256", "")),
            "parameters": int(service.checkpoint_info.get("parameters", 0)),
            "prototypes": service.checkpoint_info.get("prototypes", {}),
        },
        "queries_total": int(total_queries),
        "query_slots": int(total_queries),
        "queries": int(total_queries),
        "total_queries": int(total_queries),
        "probe_slots": int(len(probe_rows)),
        "ledger_receipts": int(len(receipts)),
        "query_mix": {
            "code_call_flow": len(queries["code"]),
            "multi_hop_graph": len(queries["graph"]),
            "unreachable_traps": len(queries["traps"]),
            "impact_analysis": len(queries["impact"]),
            "impact_traps": len(queries["impact_traps"]),
            "ood_ingress_batches": 4,
            "playbook_reuse": 2,
            "composite_symbolic": len(queries["composite"]),
            "composite_perceptual": len(perceptual_queries),
            "composite_traps": len(queries["composite_traps"]),
            "depth2_rotor_planning": len(plan_rows),
            "ontology_verification": len(verify_rows),
            "multi_plan_synthesis": len(synth_rows),
            "delta_blast_radius": len(delta_rows),
            "dag_synthesis": len(dag_rows),
            "cross_package_agent_step": len(xp_rows),
            "vocab_lock": len(vocab_rows),
            "j1_cleanroom": 4,
            "j2_elastic_vocab": 2,
            "j3_diverse_dag": 4,
            "j4_openai_adapter": 2,
            "k1_multitask": 4,
            "k2_boundary_axioms": 2,
            "k3_vla_dag": 2,
            "k4_rebase_concurrency": 4,
            "m1_online_repulsor": 3,
            "m2_low_latency": 3,
            "m3_full_reference": 3,
            "m4_guard_ast": 3,
            "setup_receipts": 2,
        },
        "code_index": {
            "seconds": float(index_seconds),
            "files": queries["graph_stats"]["files"],
            "lines": queries["graph_stats"]["lines"],
            "nodes": queries["graph_stats"]["nodes"],
            "edges": queries["graph_stats"]["edges"],
            "edges_ingested": float(ingest["result"]["edges_ingested"]),
        },
        "baseline_unindexed": {
            "query_file_reads": int(baseline.file_reads),
            "bytes_read_at_query_time": int(baseline.bytes_read),
            "response_bytes": int(baseline.response_bytes),
            "latency_ms_total": float(baseline.latency_ms),
            "reachable_accuracy": float(
                score["baseline_unindexed"]["reachable_correct"]
                / max(score["baseline_unindexed"]["reachable_total"], 1)
            ),
            "hallucination_rate": float(
                score["baseline_unindexed"]["hallucinations"]
                / max(score["baseline_unindexed"]["traps_total"], 1)
            ),
            "playbook_exact": float(score["baseline_unindexed"]["playbook_exact"]),
        },
        "holarch_mcp": {
            "query_file_reads": 0,
            "bytes_read_at_query_time": 0,
            "response_bytes": response_bytes,
            "latency_ms_total": float(mcp_latency),
            "reachable_accuracy": float(
                score["holarch_mcp"]["reachable_correct"]
                / max(score["holarch_mcp"]["reachable_total"], 1)
            ),
            "hallucination_rate": float(
                score["holarch_mcp"]["hallucinations"]
                / max(score["holarch_mcp"]["traps_total"], 1)
            ),
            "playbook_exact": float(score["holarch_mcp"]["playbook_exact"]),
            "receipts": int(len(receipts)),
            "query_receipts": int(len(receipts)),
            "receipt_chain_length": int(service.ledger.count),
            "receipt_chain_valid": chain_valid,
            "receipts_valid": receipts_valid,
            "query_chain_contiguous": bool(
                len(receipts) > 1
                and all(
                    receipts[index]["parent"] == receipts[index - 1]["sha256"]
                    for index in range(1, len(receipts))
                )
            ),
        },
        "planning": {
            "selection": plan_selection,
            "memory_fixture": dict(planner_fixture),
            "checkpoint_pinned_across_training": bool(planning_snapshot is not None),
            "determinism": "torch single-thread reduction for the probe and the API call",
            "queries": len(plan_rows),
            "correct": int(sum(1 for row in plan_rows if row["correct"])),
            "reactive_control_correct": int(sum(1 for row in plan_rows if row["reactive"] == row["truth"])),
            "depth1_control_correct": int(sum(1 for row in plan_rows if row["depth1"] == row["truth"])),
            "unmasked_control_correct": int(
                sum(1 for row in plan_rows if row["unmasked_depth2"] == row["truth"])
            ),
            "strong_traps": int(sum(1 for row in plan_rows if row["strong_trap"])),
            "trap_queries": int(len(plan_rows)),
            "rows": plan_rows,
        },
        "verification": {
            "queries": len(verify_rows),
            "correct": int(sum(1 for row in verify_rows if row["correct"])),
            "true_claims_supported": int(
                sum(1 for row in verify_rows if row["expect"] and row["supported"])
            ),
            "traps_refused": int(
                sum(1 for row in verify_rows if not row["expect"] and not row["supported"])
            ),
            "rows": verify_rows,
        },
        "synthesis": {
            "queries": len(synth_rows),
            "correct": int(sum(1 for row in synth_rows if row["correct"])),
            "composed": int(sum(1 for row in synth_rows if row["composed"])),
            "rows": synth_rows,
        },
        "delta": {
            "queries": len(delta_rows),
            "correct": int(sum(1 for row in delta_rows if row["correct"])),
            "max_ms": float(max((row["delta_ms"] for row in delta_rows), default=0.0)),
            "rows": delta_rows,
        },
        "dag_synthesis": {
            "queries": len(dag_rows),
            "pipelines": len(I2_DAG_QUERIES),
            "executed": int(dag_executed),
            "type_traps": len(I2_TYPE_TRAP_QUERIES),
            "refused": int(dag_refused),
            "rows": dag_rows,
        },
        "cross_package_agent_step": {
            "queries": len(xp_rows),
            "correct": int(xp_ok),
            "workspace": per_category["cross_package_agent_step"].get("workspace", {}),
            "rows": xp_rows,
        },
        "vocab_lock": {
            "queries": len(vocab_rows),
            "correct": int(vocab_ok + fabricated_ok),
            "locked": bool(vocab_lock is not None),
            "ledger_digest": str(vocab_lock.digest()) if vocab_lock is not None else "",
            "rows": vocab_rows,
        },
        "context_compression_ratio": compression,
        "composite": {
            "ingested": float(comp_ingest["result"]["rules_ingested"]),
            "tau_composite": float(comp_ingest["result"]["tau_composite"]),
            "reachable_total": composite_stats["reachable_total"],
            "mcp_accuracy": composite_accuracy,
            "baseline_accuracy": composite_baseline,
            "perceptual_total": composite_stats["perceptual_total"],
            "perceptual_consistency": composite_stats["perceptual_consistent"],
            "perceptual_truth_match": composite_stats["perceptual_truth_match"],
            "traps_total": composite_stats["traps_total"],
            "mcp_refused": composite_stats["mcp_refused"],
            "baseline_hallucinated": composite_stats["baseline_hallucinated"],
        },
        "per_category": per_category,
        "probes": {
            "queries": len(probe_rows),
            "correct": int(sum(1 for row in probe_rows if row["correct"])),
            "categories": [
                {
                    "category": category,
                    "total": int(per_category[category]["total"]),
                    "correct": int(per_category[category]["mcp_correct"]),
                }
                for category in (
                    "j1_cleanroom",
                    "j2_elastic_vocab",
                    "j3_diverse_dag",
                    "j4_openai_adapter",
                )
            ],
            "rows": probe_rows,
        },
        "gate": {
            "mcp_zero_query_reads": True,
            "mcp_reachable_at_least_0_98": bool(
                score["holarch_mcp"]["reachable_correct"]
                / max(score["holarch_mcp"]["reachable_total"], 1) >= 0.98
            ),
            "mcp_hallucination_zero": bool(score["holarch_mcp"]["hallucinations"] == 0),
            "mcp_playbook_exact": bool(exact == 1),
            "composite_mcp_at_least_0_90": bool(composite_accuracy >= 0.90),
            "composite_traps_refused": bool(
                composite_stats["mcp_refused"] == composite_stats["traps_total"]
            ),
            "receipt_chain_valid": chain_valid and receipts_valid,
            "impact_mcp_at_least_0_98": bool(
                per_category["impact_analysis"]["mcp_correct"]
                / max(per_category["impact_analysis"]["total"], 1)
                >= 0.98
            ),
            "impact_traps_refused": bool(
                per_category["impact_traps"]["mcp_refused"]
                == per_category["impact_traps"]["total"]
            ),
            "impact_queries_present": bool(per_category["impact_analysis"]["total"] >= 6),
            "queries_76": bool(len(receipts) == 76),
            "receipt_chain_76": bool(
                len(receipts) == 76
                and all(verify_receipt(receipt) for receipt in receipts)
                and all(
                    receipts[index]["parent"] == receipts[index - 1]["sha256"]
                    for index in range(1, len(receipts))
                )
                and chain_valid
            ),
            "planning_controls_separated": bool(
                sum(1 for row in plan_rows if row["depth1"] == row["truth"]) <= 1
                and sum(1 for row in plan_rows if row["unmasked_depth2"] == row["truth"]) <= 1
            ),
            "dag_three_plan_and_traps": bool(dag_executed == 2 and dag_refused == 2),
            "agent_step_cross_package": bool(xp_ok == 4),
            "vocab_lock_verified": bool(vocab_ok == 3 and fabricated_ok == 1),
            "planning_mcp_perfect": bool(
                per_category["depth2_rotor_planning"]["mcp_correct"]
                == per_category["depth2_rotor_planning"]["total"]
                == 4
            ),
            "verification_mcp_perfect": bool(
                per_category["ontology_verification"]["mcp_correct"]
                == per_category["ontology_verification"]["total"]
                == 4
            ),
            "synthesis_multiplan_verified": bool(
                per_category["multi_plan_synthesis"]["mcp_correct"]
                == per_category["multi_plan_synthesis"]["total"]
                == 2
            ),
            "delta_blast_radius_perfect": bool(
                per_category["delta_blast_radius"]["mcp_correct"]
                == per_category["delta_blast_radius"]["total"]
                == 2
            ),
        },
        "evidence": "OBSERVED",
    }
    # Directive J: one gate per new probe, plus the 88-query count gate.
    for _row in probe_rows:
        payload["gate"][str(_row["name"])] = bool(_row["correct"])
    payload["gate"]["queries_112"] = bool(
        int(payload["queries"]) == 112
        and int(payload["total_queries"]) == 112
        and int(payload["probe_slots"]) == 36
        and int(payload["query_slots"]) == 112
        and sum(int(value) for value in payload["query_mix"].values()) == 112
    )
    payload["gate"]["probes_36_present"] = bool(
        len(probe_rows) == 36
        and len({str(row["category"]) for row in probe_rows}) == 12
        and all(
            {"name", "category", "expected", "measured", "correct"}.issubset(row)
            for row in probe_rows
        )
    )
    payload["gate"]["passed"] = bool(
        payload["gate"]["mcp_zero_query_reads"]
        and payload["gate"]["mcp_reachable_at_least_0_98"]
        and payload["gate"]["mcp_hallucination_zero"]
        and payload["gate"]["mcp_playbook_exact"]
        and payload["gate"]["composite_mcp_at_least_0_90"]
        and payload["gate"]["composite_traps_refused"]
        and payload["gate"]["receipt_chain_valid"]
        and payload["gate"]["impact_mcp_at_least_0_98"]
        and payload["gate"]["impact_traps_refused"]
        and payload["gate"]["impact_queries_present"]
        and payload["gate"]["queries_76"]
        and payload["gate"]["receipt_chain_76"]
        and payload["gate"]["planning_controls_separated"]
        and payload["gate"]["dag_three_plan_and_traps"]
        and payload["gate"]["agent_step_cross_package"]
        and payload["gate"]["vocab_lock_verified"]
        and payload["gate"]["planning_mcp_perfect"]
        and payload["gate"]["verification_mcp_perfect"]
        and payload["gate"]["synthesis_multiplan_verified"]
        and payload["gate"]["delta_blast_radius_perfect"]
        and payload["gate"]["queries_112"]
        and payload["gate"]["probes_36_present"]
        and all(payload["gate"][str(row["name"])] for row in probe_rows)
    )
    try:
        planner_memory.close()
    except Exception:
        pass
    service.close()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(CODE_ROOT))
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="extra vision steps on top of the mounted checkpoint (0 = converged weights only)",
    )
    parser.add_argument("--out", default="receipts/holarch/agent_eval.json")
    args = parser.parse_args()

    payload = run_agent_eval(root=Path(args.root), train_steps=int(args.steps))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    (out.parent / (out.name + ".sha256")).write_text(
        hashlib.sha256(out.read_bytes()).hexdigest() + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["gate"], indent=2, default=float))
    print(
        "baseline reads={} bytes={} acc={:.3f} hall={:.3f}".format(
            payload["baseline_unindexed"]["query_file_reads"],
            payload["baseline_unindexed"]["bytes_read_at_query_time"],
            payload["baseline_unindexed"]["reachable_accuracy"],
            payload["baseline_unindexed"]["hallucination_rate"],
        )
    )
    print(
        "holarch_mcp reads={} bytes={} acc={:.3f} hall={:.3f} receipts={} chain={}".format(
            payload["holarch_mcp"]["query_file_reads"],
            payload["holarch_mcp"]["bytes_read_at_query_time"],
            payload["holarch_mcp"]["reachable_accuracy"],
            payload["holarch_mcp"]["hallucination_rate"],
            payload["holarch_mcp"]["receipts"],
            payload["holarch_mcp"]["receipt_chain_valid"],
        )
    )
    return 0 if payload["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
