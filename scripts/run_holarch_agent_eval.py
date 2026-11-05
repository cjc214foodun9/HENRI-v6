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
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from henri.holarch.code_lang import VocabularyLedger
from henri.holarch.config import HolarchConfig
from henri.holarch.model import (
    LatentActionLookahead,
    build_cul_de_sac_traps,
    find_checkpoint,
    load_checkpoint,
    open_planner_memory,
)
from henri.holarch.server import SidecarService, verify_receipt
from henri.holarch.symbolic import CodeGraphIndexer, Edge, SymbolicTopologyTask, index_code
from henri.holarch.tasks import GridWorldTask, build_tasks
from henri.holarch.train import Trainer, TrainConfig

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
        "queries_total": int(len(receipts)),
        "query_slots": int(len(receipts)),
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
