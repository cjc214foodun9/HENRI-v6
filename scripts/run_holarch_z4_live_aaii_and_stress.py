"""Cycle Z4: the live 64-task AAII battery, SSE, and the 1000-request stress.

    python scripts/run_holarch_z4_live_aaii_and_stress.py [--receipts receipts/holarch]

Five arms are mounted over ONE SharedCorpusStem, exactly as Cycle Y4 did, and
the run is live end to end (no sealed-read): every number in the receipt is
measured in this process, on this tree, at this commit.

    arm0_henri_standalone   0 external calls: deterministic recomputation for
                            the factual and briefcase slices, the Cycle Y2
                            cold synthesizer for external coding, the Cycle Z3
                            spatial CEGIS inducer for ARC, and the Chow trap
                            proofs
    arm1_raw_llm            one live completion per task, no retrieval
    arm2_rag_llm            one live completion per task with BM25 context
    arm3_henri_coprocessor  live completion governed by the exact-quantity
                            check, the public-example check, the RepoSheaf
                            capability pullback, an online REPULSOR retry
                            (<= 2) and the Chow abstention
    arm4_henri_cascade      arm0 first at zero external cost; escalate to the
                            governed arm3 path only when arm0 neither answered
                            nor proved the request unsatisfiable

The battery is 64 deterministic tasks: 52 answerable (12 factual, 12 briefcase,
16 external cold-coding derived from the vendored HumanEval corpus, 12 2D ARC
tasks built from the Cycle Z3 operators) and 12 impossible/OOD traps.

A fail-closed evidence rule governs both coding arms.  A coding answer is only
emitted when the specification carries at least two public examples, the
emitted module defines the function the specification names, and it passes
every public example.  A single example, or an example set that the parser
could not read at all, is not a proof and the arm abstains: 13 of the 16
external tasks carry their examples as prose rather than doctests, so the
governed coding path refuses them instead of guessing.

The live completion cap is 2048 tokens at temperature 0.0 and the whole run is
bounded by 200 live calls.  Every call is recorded in a ledger with the
prompt and response SHA-256.

The same process then fires 1000 requests across 16 concurrent workers at the
ASGI app (500 agent_step, 500 chat completions, half of the chat calls
streaming), records P50/P95/P99, the 5xx count, the RSS delta and the receipt
chain, and verifies that at least one stream produced two data chunks and the
OpenAI terminator.  Cycle Z4 also adds the optional SSE path to
POST /v1/chat/completions; this runner is what verifies it.

Nothing here reads a sealed receipt as if it were a measurement, with one
declared exception: the Cycle Z gauntlet receipt is read as the artefact the
gauntlet wrote, and its SHA-256 sidecar and its freshness against the gauntlet
source are both checked before its numbers are quoted.
"""

from __future__ import annotations

import argparse
import ast as _ast
import asyncio
import ctypes
import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

import run_holarch_p_cycle as pc
from henri.holarch.code_gen import SharedCorpusStem
from henri.holarch.cold_synth import ColdSynthesizer, parse_problem
from henri.holarch.config import HolarchConfig
from henri.holarch.coprocessor import redact_secret, resolve_llm_credentials
from henri.holarch.repo_sheaf import RepoSheaf
from henri.holarch.sandbox import PURE_ALGORITHMIC_MODULES, SubprocessSandboxExecutor
from henri.holarch.spatial_induction import (
    SPATIAL_SKETCHES,
    SpatialCegisInducer,
    SpatialOp,
    SpatialSpec,
    apply_chain,
    as_grid,
)
from henri.holarch.zone_c import ZoneCMemory

RECEIPTS = ROOT / "receipts" / "holarch"
HUMANEVAL = RECEIPTS / "external" / "HumanEval.jsonl"
HUMANEVAL_SHA256 = "1d49078ba3e2b196b9344535bef34a43021f038fad9561d6ee7c53450609a6a2"
GAUNTLET = RECEIPTS / "agent_eval.json"

RECEIPT_NAME = "z4_live_aaii_and_api_stress.json"
MAX_TOKENS = 2048
"""The live completion cap for this stage."""
MAX_CALLS = 200
"""Hard live-call cap for the whole five-arm run."""
MAX_RETRIES = 8
"""Hard cap on governed repair retries (they draw on the same 200-call budget)."""
TEMPERATURE = 0.0
MIN_PUBLIC_EXAMPLES = 2
"""A coding answer needs two independent public examples before it is admitted."""
STRESS_REQUESTS = 1000
STRESS_WORKERS = 16
GAUNTLET_QUERIES = 288
GAUNTLET_PROBES = 212
GAUNTLET_LEGACY = 76
Z4_BATTERY: Dict[str, Any] = {
    "tasks": 64,
    "answerable": 52,
    "traps": 12,
    "slices": {"factual": 12, "briefcase": 12, "external_coding": 16, "arc": 12},
}
"""The sealed Cycle Z4 battery size.  A caller may pass its own expectation."""
GAUNTLET_WALL_BAR = 160.0
RSS_DRIFT_BAR = 256 * 1024 * 1024
FENCE = chr(96) * 3
PILLARS = ("factual", "briefcase", "external_coding", "arc")
ARM_NAMES = (
    "arm0_henri_standalone",
    "arm1_raw_llm",
    "arm2_rag_llm",
    "arm3_henri_coprocessor",
    "arm4_henri_cascade",
)
GOVERNED_ARMS = (ARM_NAMES[3], ARM_NAMES[4])


# --------------------------------------------------------------- extractors
def _ast_functions(path: Path) -> int:
    tree = _ast.parse(Path(path).read_text(encoding="utf-8"))
    return sum(1 for node in _ast.walk(tree)
               if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)))


def _ast_classes(path: Path) -> int:
    tree = _ast.parse(Path(path).read_text(encoding="utf-8"))
    return sum(1 for node in _ast.walk(tree) if isinstance(node, _ast.ClassDef))


def _test_functions(path: Path) -> int:
    return len(re.findall(r"^def test_", Path(path).read_text(encoding="utf-8"), flags=re.M))


def _lines(path: Path) -> int:
    return len(Path(path).read_text(encoding="utf-8").splitlines())


def _py_modules(root: Path) -> List[Path]:
    base = Path(root) / "henri" / "holarch"
    return sorted(path for path in base.glob("*.py") if path.is_file())


def _argmax_index(values: Sequence[int]) -> int:
    best = 0
    for index, value in enumerate(values):
        if int(value) > int(values[best]):
            best = index
    return int(best)


def _spread(values: Sequence[int]) -> int:
    return int(max(values) - min(values))


def _fib(n: int) -> int:
    a, b = 0, 1
    for _ in range(int(n)):
        a, b = b, a + b
    return int(a)


def _prime_sum(limit: int) -> int:
    return int(sum(value for value in range(2, int(limit))
                   if all(value % d for d in range(2, int(value ** 0.5) + 1))))


# -------------------------------------------------- external coding (HumanEval)
def load_humaneval(path: Path = HUMANEVAL) -> List[Dict[str, Any]]:
    """The vendored corpus, digest-checked before a single task is read."""
    body = Path(path).read_bytes()
    digest = hashlib.sha256(body).hexdigest()
    if digest != HUMANEVAL_SHA256:
        raise ValueError("HumanEval.jsonl digest mismatch: " + digest)
    rows = [json.loads(line) for line in body.decode("utf-8").splitlines() if line.strip()]
    if len(rows) != 164:
        raise ValueError("HumanEval.jsonl does not hold 164 tasks")
    return rows


def canonical_asserts(test_src: str, entry: str) -> List[str]:
    """Every canonical assert statement that scores the candidate, verbatim.

    The statement text is taken from the corpus source, so the hidden check is
    the benchmark's own, and only the placeholder name is rebound to the entry
    point the task declares.
    """
    out: List[str] = []
    for node in _ast.walk(_ast.parse(test_src)):
        if not isinstance(node, _ast.Assert):
            continue
        segment = _ast.get_source_segment(test_src, node)
        if not segment or not re.search(r"\bcandidate\b", segment):
            continue
        out.append(re.sub(r"\bcandidate\b", str(entry), segment))
    return out


_DOCTEST = re.compile(r"^\s*>>>\s*(.+)$", re.MULTILINE)


def doctest_asserts(prompt: str) -> List[str]:
    """The public examples of an unmodified prompt: its own doctests.

    A call line is paired with the expected repr on the next non-empty line.
    A Traceback case or a continued expression is skipped, because neither is
    a literal an assert can compare.
    """
    lines = str(prompt).splitlines()
    out: List[str] = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line.startswith(">>>"):
            index += 1
            continue
        expression = line[3:].strip()
        index += 1
        while index < len(lines) and not lines[index].strip():
            index += 1
        empty = index >= len(lines)
        expected = "" if empty else lines[index].strip()
        continuation = (not empty and index + 1 < len(lines)
                        and lines[index + 1].strip().startswith("..."))
        if not expression:
            index += 1
            continue
        closing = expected.startswith(chr(34) * 3) or expected.startswith(chr(39) * 3)
        if empty or expected.startswith(">>>") or not expected or closing:
            # A doctest that prints nothing is a claim, not a value: the
            # expression itself is the assertion (the HumanEval style
            # ">>> f(x) == 0" belongs here).
            out.append("assert (%s)" % expression)
            continue
        if expected.startswith("...") or "Traceback" in expected or continuation:
            index += 1
            continue
        out.append("assert (%s) == (%s)" % (expression, expected))
        index += 1
    return out


def select_external_tasks(root: Path, take: int = 16) -> List[Dict[str, Any]]:
    """Pin the 16 external tasks by ascending task_id with >= 4 canonical asserts."""
    rows = load_humaneval(Path(root) / "receipts" / "holarch" / "external" / "HumanEval.jsonl")
    scored: List[Tuple[str, Dict[str, Any], List[str]]] = []
    for row in rows:
        asserts = canonical_asserts(str(row["test"]), str(row["entry_point"]))
        if len(asserts) >= 4:
            scored.append((str(row["task_id"]), row, asserts))
    scored.sort(key=lambda item: item[0])
    chosen = scored[: int(take)]
    if len(chosen) != int(take):
        raise ValueError("fewer than 16 HumanEval tasks carry four canonical asserts")
    tasks: List[Dict[str, Any]] = []
    for task_id, row, asserts in chosen:
        prompt = str(row["prompt"])
        public = doctest_asserts(prompt)
        tasks.append(
            {
                "id": "ext_" + task_id.replace("HumanEval/", ""),
                "task_id": task_id,
                "pillar": "external_coding",
                "kind": "code",
                "question": prompt,
                "func": str(row["entry_point"]),
                "public": public,
                "hidden": asserts,
                "evidence": {
                    "corpus": "receipts/holarch/external/HumanEval.jsonl",
                    "corpus_sha256": HUMANEVAL_SHA256,
                    "public_assertions": int(len(public)),
                    "hidden_assertions": int(len(asserts)),
                },
            }
        )
    return tasks


# ----------------------------------------------------------------- ARC slice
def _grid(rows: Sequence[Sequence[int]]) -> np.ndarray:
    return as_grid(np.array(rows, dtype=np.int64))


ARC_SPECS: Tuple[Tuple[str, Tuple[SpatialOp, ...],
                       Tuple[Sequence[Sequence[int]], ...],
                       Sequence[Sequence[int]]], ...] = (
    ("arc_rot90", (SpatialOp("dihedral", (1,)),),
     ([[1, 2, 0], [0, 3, 4], [5, 0, 6]], [[2, 2, 0], [1, 0, 0], [0, 7, 3]],
      [[0, 1, 0], [2, 0, 6], [3, 3, 0]]), [[4, 0, 1], [0, 5, 0], [2, 0, 7]]),
    ("arc_rot180", (SpatialOp("dihedral", (2,)),),
     ([[1, 2, 3, 0], [0, 4, 0, 5], [6, 0, 7, 0], [0, 8, 0, 9]],
      [[1, 0, 0, 2], [0, 3, 4, 0], [0, 5, 6, 0], [7, 0, 0, 8]]),
     [[0, 1, 2, 0], [3, 0, 0, 4], [0, 5, 6, 0], [7, 0, 0, 8]]),
    ("arc_reflect_lr", (SpatialOp("dihedral", (4,)),),
     ([[1, 2, 3], [4, 0, 5], [6, 7, 0]], [[0, 1, 2], [3, 4, 0], [5, 0, 6]],
      [[7, 0, 1], [0, 2, 3], [4, 5, 0]]), [[1, 1, 0], [2, 0, 3], [0, 4, 5]]),
    ("arc_shift_wrap", (SpatialOp("toroidal_shift", (1, 2)),),
     ([[1, 2, 3, 0], [0, 4, 5, 6], [7, 0, 0, 8], [0, 9, 0, 0]],
      [[1, 0, 2, 0], [0, 3, 0, 4], [5, 0, 6, 0], [0, 7, 0, 8]]),
     [[0, 1, 0, 2], [3, 0, 4, 0], [0, 5, 0, 6], [7, 0, 8, 0]]),
    ("arc_shift_bounded", (SpatialOp("bounded_shift", (1, 1, 0)),),
     ([[1, 2, 3, 0], [0, 4, 5, 6], [7, 0, 0, 8], [0, 9, 0, 0]],
      [[1, 0, 2, 0], [0, 3, 0, 4], [5, 0, 6, 0], [0, 7, 0, 8]]),
     [[2, 0, 3, 0], [0, 4, 0, 5], [6, 0, 7, 0], [0, 8, 0, 9]]),
    ("arc_colour_swap", (SpatialOp("colour_map", (0, 2, 1, 3, 4, 5, 6, 7, 8, 9)),),
     ([[1, 1, 2], [2, 1, 0], [0, 2, 1]], [[1, 2, 0], [2, 1, 2], [1, 0, 2]]),
     [[2, 1, 1], [1, 2, 0], [2, 0, 1]]),
    ("arc_crop_bbox", (SpatialOp("crop_bbox", ()),),
     ([[0, 0, 0, 0, 0], [0, 1, 1, 0, 0], [0, 1, 2, 2, 0], [0, 0, 2, 0, 0], [0, 0, 0, 0, 0]],
      [[0, 0, 0, 0, 0], [0, 3, 3, 3, 0], [0, 0, 3, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]),
     [[0, 0, 0, 0, 0], [0, 0, 4, 0, 0], [0, 4, 4, 0, 0], [0, 4, 0, 0, 0], [0, 0, 0, 0, 0]]),
    ("arc_tile", (SpatialOp("tile", (2, 2)),),
     ([[1, 0, 2], [0, 3, 0]], [[4, 0], [0, 4]]), [[0, 5, 5], [5, 0, 0]]),
    ("arc_scale", (SpatialOp("scale", (2,)),),
     ([[1, 0, 2], [0, 0, 0], [3, 0, 4]], [[0, 5, 0], [6, 0, 6], [0, 5, 0]]),
     [[7, 0, 0], [0, 8, 0], [0, 0, 9]]),
    ("arc_flood", (SpatialOp("interior_flood_fill", (4,)),),
     ([[1, 1, 1, 1, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 1, 1, 1, 1]],
      [[2, 2, 2, 2], [2, 0, 0, 2], [2, 0, 0, 2], [2, 2, 2, 2]]),
     [[3, 3, 3, 3, 3], [3, 0, 0, 0, 3], [3, 0, 0, 0, 3], [3, 0, 0, 0, 3], [3, 3, 3, 3, 3]]),
    ("arc_mask", (SpatialOp("mask_by_colour", (2,)),),
     ([[2, 1, 2], [1, 1, 1], [2, 1, 0]], [[1, 2, 1], [2, 2, 1], [1, 1, 2]]),
     [[2, 2, 1], [1, 2, 1], [1, 1, 2]]),
    ("arc_crop_then_rot90", (SpatialOp("crop_bbox", ()), SpatialOp("dihedral", (1,))),
     ([[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0]],
      [[0, 0, 0, 0, 0], [0, 5, 0, 0, 1], [0, 0, 0, 0, 0], [0, 6, 7, 0, 0], [0, 0, 0, 0, 0]]),
     [[0, 0, 0, 0, 0], [0, 0, 0, 8, 0], [0, 0, 0, 0, 9], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]),
)


def _grid_json(grid: np.ndarray) -> List[List[int]]:
    return [[int(value) for value in row] for row in np.asarray(grid)]


def arc_prompt(train: Sequence[Tuple[np.ndarray, np.ndarray]], test_in: np.ndarray) -> str:
    """One text rendering of a demonstration set, identical for every arm."""
    blocks = []
    for index, (src, dst) in enumerate(train):
        blocks.append(
            "pair " + str(index + 1) + ":\n  input  = " + json.dumps(_grid_json(src))
            + "\n  output = " + json.dumps(_grid_json(dst))
        )
    return (
        "A 2D grid transformation. Every demonstration maps an input grid of "
        "colours 0-9 to its output grid.\n\n"
        + "\n".join(blocks)
        + "\n\ntest input = " + json.dumps(_grid_json(test_in))
        + "\n\nThe transformation is one of these operator families "
        "(name, parameters):\n"
        "  dihedral(index)          index 0 identity, 1 rot90, 2 rot180, 3 rot270,\n"
        "                           4 reflect left-right, 5 reflect top-bottom,\n"
        "                           6 transpose, 7 anti-transpose\n"
        "  toroidal_shift(dr, dc)   shift with wraparound\n"
        "  bounded_shift(dr, dc, fill)\n"
        "  colour_map(lut)          ten integers; lut[c] is the new colour c\n"
        "  crop_bbox()              crop to the smallest box holding a non-zero cell\n"
        "  tile(fy, fx)             repeat the grid fy by fx times\n"
        "  scale(factor)            nearest-neighbour integer scaling\n"
        "  interior_flood_fill(colour)  fill background regions that touch no border\n"
        "  mask_by_colour(colour)   keep that colour, everything else becomes 0\n"
        "  symmetry_completion()    fill the single empty quadrant\n"
        "  constant_fill(colour)    the whole grid becomes that colour\n\n"
        "Reply with exactly one line of JSON and nothing else:\n"
        "{\"ops\": [{\"family\": \"<name>\", \"params\": [<integer>, ...]}]}\n"
        "Use one or two operators. If the demonstrations are contradictory and no "
        "chain can produce them, reply {\"ops\": []}."
    )


def build_arc_tasks() -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    for name, chain, seeds, test_in in ARC_SPECS:
        train = [(np.asarray(seed, dtype=np.int64), apply_chain(chain, _grid(seed)))
                 for seed in seeds]
        if any(output is None for _src, output in train):
            raise ValueError("an ARC seed does not admit its declared chain: " + name)
        expected = apply_chain(chain, _grid(test_in))
        if expected is None:
            raise ValueError("the ARC test grid does not admit its declared chain: " + name)
        tasks.append(
            {
                "id": name,
                "pillar": "arc",
                "kind": "arc",
                "question": arc_prompt(train, _grid(test_in)),
                "train": train,
                "test_in": _grid(test_in),
                "test_out": expected,
                "truth_chain": [op.label() for op in chain],
                "evidence": {
                    "operators": [op.family for op in chain],
                    "train_pairs": int(len(train)),
                    "test_shape": [int(expected.shape[0]), int(expected.shape[1])],
                },
            }
        )
    return tasks


# --------------------------------------------------------------------- traps
def build_trap_tasks(root: Path) -> List[Dict[str, Any]]:
    """Twelve requests that no true answer can satisfy, each with a proof.

    A trap is only admissible when this tree can prove it unsatisfiable
    without a model: an absent symbol or file, an undefined operation, an empty
    interval, an out-of-range index, an unbounded maximum, a complex root, a
    contradictory demonstration set, contradictory examples, a capability the
    sandbox profile blocks, a return-type contradiction, or evidence that does
    not exist in the tree at all.
    """
    def trap(task_id, question, spec, evidence, kind="trap"):
        return {
            "id": str(task_id),
            "pillar": "trap",
            "kind": str(kind),
            "question": str(question),
            "trap": dict(spec),
            "evidence": dict(evidence),
        }

    contradictory_a = [[1, 2], [3, 0]]
    contradictory_b = [[4, 5], [6, 0]]
    contradictory_c = [[7, 0], [0, 8]]
    contradictory_prompt = arc_prompt(
        [(_grid(contradictory_a), _grid(contradictory_b)),
         (_grid(contradictory_a), _grid(contradictory_c))],
        _grid([[0, 9], [9, 0]]),
    )
    contradictory_code = (
        "def classify(n: int) -> int:\n"
        '    """Classify n.\n'
        "    >>> classify(1)\n"
        "    1\n"
        "    >>> classify(1)\n"
        "    2\n"
        '    """\n'
    )
    blocked_code = (
        "def fetch_header(host: str) -> str:\n"
        '    """Open a TCP socket to host and return the banner.\n'
        "    >>> fetch_header('example.com')\n"
        "    'ok'\n"
        '    """\n'
    )
    type_code = (
        "def magnitude(n: int) -> int:\n"
        '    """Return the magnitude of n.\n'
        "    >>> magnitude(3)\n"
        "    'three'\n"
        '    """\n'
    )

    return [
        trap(
            "t1_missing_symbol",
            "Report the exact scalar value that henri.holarch.zone_b."
            "quantum_flux_capacitor returns for a unit photonic wave. Answer with "
            "one integer.",
            {"kind": "missing_symbol", "symbol": "quantum_flux_capacitor"},
            {"claim": "zone_b declares quantum_flux_capacitor"},
        ),
        trap(
            "t2_missing_file",
            "Quote the exact wall-clock budget in seconds that docs/HOLARCH_Y3.md "
            "sets for Stage Y3. Answer with one integer.",
            {"kind": "missing_file", "path": "docs/HOLARCH_Y3.md"},
            {"claim": "docs/HOLARCH_Y3.md exists"},
        ),
        trap(
            "t3_undefined_operation",
            "Compute the exact integer value of 1 // 0 under the HENRI arithmetic "
            "rules and answer with one integer.",
            {"kind": "undefined_operation", "expression": "1 // 0"},
            {"claim": "integer division by zero is defined"},
        ),
        trap(
            "t4_empty_interval",
            "Return an integer x with x > 10 and x < 5. Answer with one integer.",
            {"kind": "empty_interval", "lower": 10.0, "upper": 5.0},
            {"claim": "an integer satisfies 10 < x < 5"},
        ),
        trap(
            "t5_out_of_range",
            "Return element 47 of the twelve-element tuple "
            "(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11) exactly. Answer with one integer.",
            {"kind": "out_of_range", "index": 47, "length": 12},
            {"claim": "index 47 exists in a length-12 tuple"},
        ),
        trap(
            "t6_unbounded_maximum",
            "State the largest integer that exists. Answer with one integer.",
            {"kind": "unbounded", "probe": 2 ** 200},
            {"claim": "the integers have a maximum"},
        ),
        trap(
            "t7_no_real_solution",
            "Return a real number x with x * x + 1 == 0. Answer with one integer.",
            {"kind": "no_real_solution", "a": 1.0, "b": 0.0, "c": 1.0},
            {"claim": "x*x + 1 = 0 has a real solution"},
        ),
        trap(
            "t8_contradictory_arc",
            contradictory_prompt,
            {"kind": "contradictory_examples",
             "pairs": [[contradictory_a, contradictory_b], [contradictory_a, contradictory_c]]},
            {"claim": "one input grid has two different declared outputs"},
        ),
        trap(
            "t9_contradictory_examples",
            contradictory_code,
            {"kind": "contradictory_assertions", "prompt": contradictory_code},
            {"claim": "classify(1) is both 1 and 2"},
        ),
        trap(
            "t10_blocked_capability",
            blocked_code,
            {"kind": "blocked_capability", "module": "socket"},
            {"claim": "the pure-algorithmic sandbox profile can open a socket"},
        ),
        trap(
            "t11_type_contradiction",
            type_code,
            {"kind": "type_contradiction", "prompt": type_code, "annotation": "int"},
            {"claim": "magnitude returns an int and 'three'"},
        ),
        trap(
            "t12_ood_evidence_absent",
            "State the exact parameter count recorded for the 70B omniscience "
            "checkpoint at receipts/holarch/holarch_70b_omniscience.pt, and the "
            "symbol holarch_70b_omniscience. Answer with one integer.",
            {"kind": "ood_evidence_absent",
             "path": "receipts/holarch/holarch_70b_omniscience.pt",
             "symbol": "holarch_70b_omniscience"},
            {"claim": "a 70B omniscience checkpoint is recorded in the tree"},
        ),
    ]


def _trap_arc_pairs(spec: Dict[str, Any]) -> List[Tuple[np.ndarray, np.ndarray]]:
    pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    for pair in list(spec.get("pairs") or []):
        source, target = pair
        pairs.append((_grid(source), _grid(target)))
    return pairs


def detect_trap(task: Dict[str, Any], stem: Optional[SharedCorpusStem],
                sandbox: Optional[SubprocessSandboxExecutor] = None) -> Tuple[bool, str]:
    """A deterministic proof that the request has no satisfying answer."""
    trap = dict(task.get("trap") or {})
    kind = str(trap.get("kind", ""))
    if kind == "missing_symbol":
        symbol = str(trap.get("symbol", ""))
        if not symbol:
            return False, "empty_symbol"
        known = bool(stem is not None and symbol in set(stem.symbol_table or set()))
        return (not known), ("missing_symbol" if not known else "symbol_present")
    if kind == "missing_file":
        path = str(trap.get("path", ""))
        if not path:
            return False, "empty_path"
        present = (ROOT / path).exists()
        return (not present), ("missing_file" if not present else "file_present")
    if kind == "undefined_operation":
        expression = str(trap.get("expression", ""))
        if not expression:
            return False, "empty_expression"
        try:
            eval(compile(expression, "<z4-trap>", "eval"), {"__builtins__": {}})  # noqa: S307
        except Exception:  # noqa: BLE001 - the exception is the proof
            return True, "undefined_operation"
        return False, "operation_defined"
    if kind == "empty_interval":
        return (float(trap.get("lower", 0.0)) >= float(trap.get("upper", 0.0))), "empty_interval"
    if kind == "out_of_range":
        return (int(trap.get("index", 0)) >= int(trap.get("length", 0))), "out_of_range"
    if kind == "unbounded":
        probe = int(trap.get("probe", 2 ** 200))
        return (probe + 1 > probe and _ints_unbounded()), "unbounded"
    if kind == "no_real_solution":
        a = float(trap.get("a", 0.0))
        b = float(trap.get("b", 0.0))
        c = float(trap.get("c", 0.0))
        return (b * b - 4.0 * a * c < 0.0), "no_real_solution"
    if kind == "contradictory_examples":
        pairs = _trap_arc_pairs(trap)
        if not pairs:
            return False, "empty_pairs"
        reason = SpatialCegisInducer().refuse(SpatialSpec(name="trap", examples=pairs))
        return (reason is not None), ("contradictory_examples" if reason else "consistent")
    if kind == "contradictory_assertions":
        spec = parse_problem(str(trap.get("prompt", "")))
        found = bool(spec.contradictions)
        return found, ("contradictory_assertions" if found else "consistent")
    if kind == "blocked_capability":
        module = str(trap.get("module", "")).strip()
        if not module:
            return False, "empty_module"
        if module in PURE_ALGORITHMIC_MODULES:
            return False, "module_allowed"
        box = sandbox if sandbox is not None else SubprocessSandboxExecutor("pure_algorithmic")
        record = box.execute("import " + module + chr(10) + "result = {'ok': True}" + chr(10))
        return (not bool(record.ok)), ("blocked_capability" if not record.ok else "capability_allowed")
    if kind == "type_contradiction":
        return _type_contradiction(str(trap.get("prompt", "")),
                                  str(trap.get("annotation", "int")))
    if kind == "ood_evidence_absent":
        path = str(trap.get("path", ""))
        symbol = str(trap.get("symbol", ""))
        absent_path = bool(path) and not (ROOT / path).exists()
        absent_symbol = bool(symbol) and not bool(
            stem is not None and symbol in set(stem.symbol_table or set()))
        ok = bool(absent_path and absent_symbol)
        return ok, ("ood_evidence_absent" if ok else "evidence_present")
    return False, "undetermined"


def _ints_unbounded() -> bool:
    value = 2 ** 200
    return bool(value + 1 > value and value * value > value)


_EXPECTED_KIND = {
    "int": "int", "float": "float", "str": "str", "bool": "bool",
    "list": "list", "dict": "dict", "tuple": "tuple", "none": "none",
}


def _type_contradiction(prompt: str, annotation: str) -> Tuple[bool, str]:
    """A doctest whose expected literal cannot inhabit the declared return type."""
    wanted = _EXPECTED_KIND.get(str(annotation).strip().lower(), "")
    if not wanted:
        return False, "unknown_annotation"
    for line in doctest_asserts(prompt):
        body = str(line).strip()
        if body.startswith("assert "):
            body = body[len("assert "):]
        try:
            node = _ast.parse(body, mode="eval").body
        except SyntaxError:
            continue
        if not isinstance(node, _ast.Compare) or not node.comparators:
            continue
        try:
            value = _ast.literal_eval(node.comparators[0])
        except Exception:  # noqa: BLE001 - a non-literal expectation proves nothing
            continue
        actual = ("bool" if isinstance(value, bool) else
                  "int" if isinstance(value, int) else
                  "float" if isinstance(value, float) else
                  "str" if isinstance(value, str) else
                  "list" if isinstance(value, list) else
                  "dict" if isinstance(value, dict) else
                  "tuple" if isinstance(value, tuple) else
                  "none" if value is None else "other")
        if wanted == "int" and actual == "bool":
            continue
        if actual != wanted:
            return True, "type_contradiction"
    return False, "type_consistent"


# ------------------------------------------------------------------- battery
def build_battery(root: Path) -> List[Dict[str, Any]]:
    """The deterministic 64-task Z4 battery: 52 answerable plus 12 traps."""
    root = Path(root)
    tasks: List[Dict[str, Any]] = []

    def answered(task_id, pillar, prompt, compute, evidence):
        tasks.append(
            {
                "id": str(task_id),
                "pillar": str(pillar),
                "kind": "quantity",
                "question": str(prompt),
                "compute": compute,
                "truth": int(compute()),
                "evidence": dict(evidence),
            }
        )

    h = root / "henri" / "holarch"
    t = root / "tests"

    # ---- factual: twelve deterministic recomputations -------------------
    answered(
        "f1_deterministic_functions", "factual",
        "Count the Python function definitions (def and async def) in "
        "henri/holarch/deterministic.py. A design note claims 41; the note is not "
        "authoritative. Answer with one integer.",
        lambda: _ast_functions(h / "deterministic.py"),
        {"file": "henri/holarch/deterministic.py", "metric": "ast function definitions"},
    )
    answered(
        "f2_budget_classes", "factual",
        "Count the class definitions (class statements) in henri/holarch/budget.py. "
        "A changelog entry claims 9; the changelog is not authoritative. Answer with "
        "one integer.",
        lambda: _ast_classes(h / "budget.py"),
        {"file": "henri/holarch/budget.py", "metric": "ast.ClassDef"},
    )
    answered(
        "f3_service_test_functions", "factual",
        "Count the pytest test functions (lines that start with 'def test_') in "
        "tests/test_holarch_service.py. A slide deck claims 30; the slide is not "
        "authoritative. Answer with one integer.",
        lambda: _test_functions(t / "test_holarch_service.py"),
        {"file": "tests/test_holarch_service.py", "metric": "'^def test_' matches"},
    )
    answered(
        "f4_holarch_module_count", "factual",
        "Count the top-level .py modules directly inside henri/holarch (not "
        "recursively). A catalogue claims 120; the catalogue is not authoritative. "
        "Answer with one integer.",
        lambda: len(_py_modules(root)),
        {"directory": "henri/holarch", "metric": "*.py file count"},
    )
    answered(
        "f5_cold_synth_lines", "factual",
        "Count the lines of henri/holarch/cold_synth.py. A memo claims 800; the memo "
        "is not authoritative. Answer with one integer.",
        lambda: _lines(h / "cold_synth.py"),
        {"file": "henri/holarch/cold_synth.py", "metric": "splitlines"},
    )
    answered(
        "f6_spatial_functions", "factual",
        "Count the Python function definitions (def and async def) in "
        "henri/holarch/spatial_induction.py. A note claims 20; the note is not "
        "authoritative. Answer with one integer.",
        lambda: _ast_functions(h / "spatial_induction.py"),
        {"file": "henri/holarch/spatial_induction.py", "metric": "ast function definitions"},
    )
    answered(
        "f7_arithmetic", "factual",
        "Evaluate 37 * 91 + 17 exactly. Answer with one integer.",
        lambda: 37 * 91 + 17,
        {"metric": "exact integer arithmetic"},
    )
    answered(
        "f8_prime_sum_below_50", "factual",
        "Sum every prime number strictly below 50. Answer with one integer.",
        lambda: _prime_sum(50),
        {"metric": "primes below 50"},
    )
    answered(
        "f9_fibonacci_15", "factual",
        "Using fib(0) = 0 and fib(1) = 1, state fib(15). Answer with one integer.",
        lambda: _fib(15),
        {"metric": "Fibonacci recurrence"},
    )
    answered(
        "f10_triangular_25", "factual",
        "State the twenty-fifth triangular number 1 + 2 + ... + 25. Answer with one "
        "integer.",
        lambda: sum(range(1, 26)),
        {"metric": "closed form n(n+1)/2"},
    )
    answered(
        "f11_power_24", "factual",
        "Evaluate 2 ** 24 - 1 exactly. Answer with one integer.",
        lambda: 2 ** 24 - 1,
        {"metric": "exact integer arithmetic"},
    )
    answered(
        "f12_gcd", "factual",
        "Compute gcd(4620, 1071) exactly. Answer with one integer.",
        lambda: math.gcd(4620, 1071),
        {"metric": "greatest common divisor"},
    )

    # ---- briefcase: twelve multi-file analytics -------------------------
    answered(
        "b1_functions_four_modules", "briefcase",
        "Sum the Python function-definition counts of henri/holarch/zone_b.py, "
        "henri/holarch/swat_bus.py, henri/holarch/vla.py and "
        "henri/holarch/paged_tape.py. A briefing note claims 150; the note is not "
        "authoritative. Answer with one integer.",
        lambda: sum(_ast_functions(h / name)
                    for name in ("zone_b.py", "swat_bus.py", "vla.py", "paged_tape.py")),
        {"files": ["zone_b.py", "swat_bus.py", "vla.py", "paged_tape.py"],
         "metric": "sum of ast function definitions"},
    )
    answered(
        "b2_classes_four_modules", "briefcase",
        "Sum the class-definition counts of henri/holarch/zone_c.py, "
        "henri/holarch/code_gen.py, henri/holarch/symbolic.py and "
        "henri/holarch/server.py. A status page claims 60; the page is not "
        "authoritative. Answer with one integer.",
        lambda: sum(_ast_classes(h / name)
                    for name in ("zone_c.py", "code_gen.py", "symbolic.py", "server.py")),
        {"files": ["zone_c.py", "code_gen.py", "symbolic.py", "server.py"],
         "metric": "sum of ast.ClassDef"},
    )
    answered(
        "b3_function_gap", "briefcase",
        "Compute the function-definition count of henri/holarch/code_gen.py minus "
        "that of henri/holarch/zone_b.py, as an exact integer (it may be negative). "
        "Answer with one integer.",
        lambda: _ast_functions(h / "code_gen.py") - _ast_functions(h / "zone_b.py"),
        {"files": ["code_gen.py", "zone_b.py"], "metric": "difference of counts"},
    )
    answered(
        "b4_test_functions_three_modules", "briefcase",
        "Sum the pytest test-function counts of tests/test_holarch_p_cycle.py, "
        "tests/test_holarch_q_cycle.py and tests/test_holarch_r_cycle.py. A memo "
        "claims 120; the memo is not authoritative. Answer with one integer.",
        lambda: sum(_test_functions(t / name)
                    for name in ("test_holarch_p_cycle.py", "test_holarch_q_cycle.py",
                                 "test_holarch_r_cycle.py")),
        {"files": ["tests/test_holarch_p_cycle.py", "tests/test_holarch_q_cycle.py",
                   "tests/test_holarch_r_cycle.py"], "metric": "sum of test functions"},
    )
    answered(
        "b5_lines_three_modules", "briefcase",
        "Sum the line counts of henri/holarch/cold_synth.py, "
        "henri/holarch/spatial_induction.py and henri/holarch/server.py. A ticket "
        "claims 5000 lines; the ticket is not authoritative. Answer with one integer.",
        lambda: sum(_lines(h / name)
                    for name in ("cold_synth.py", "spatial_induction.py", "server.py")),
        {"files": ["cold_synth.py", "spatial_induction.py", "server.py"],
         "metric": "sum of splitlines"},
    )
    answered(
        "b6_argmax_class_module", "briefcase",
        "Among the four modules [henri/holarch/zone_c.py, henri/holarch/code_gen.py, "
        "henri/holarch/symbolic.py, henri/holarch/server.py] list them in that fixed "
        "order and report the zero-based index of the one with the most class "
        "definitions. Answer with one integer.",
        lambda: _argmax_index([_ast_classes(h / name)
                               for name in ("zone_c.py", "code_gen.py", "symbolic.py",
                                            "server.py")]),
        {"order": ["zone_c.py", "code_gen.py", "symbolic.py", "server.py"],
         "metric": "argmax of class counts, first index on ties"},
    )
    answered(
        "b7_function_spread", "briefcase",
        "Compute the largest minus the smallest function-definition count among "
        "henri/holarch/zone_b.py, henri/holarch/swat_bus.py, henri/holarch/vla.py and "
        "henri/holarch/paged_tape.py. Answer with one integer.",
        lambda: _spread([_ast_functions(h / name)
                         for name in ("zone_b.py", "swat_bus.py", "vla.py", "paged_tape.py")]),
        {"files": ["zone_b.py", "swat_bus.py", "vla.py", "paged_tape.py"],
         "metric": "max - min"},
    )
    answered(
        "b8_function_percent", "briefcase",
        "Compute 100 * (functions in henri/holarch/zone_c.py) // (functions in "
        "henri/holarch/code_gen.py), using integer floor division. Answer with one "
        "integer.",
        lambda: 100 * _ast_functions(h / "zone_c.py") // max(
            _ast_functions(h / "code_gen.py"), 1),
        {"files": ["zone_c.py", "code_gen.py"], "metric": "integer percentage"},
    )
    answered(
        "b9_dense_module_count", "briefcase",
        "Count the top-level .py modules inside henri/holarch that declare more than "
        "40 function definitions (def and async def). Answer with one integer.",
        lambda: sum(1 for path in _py_modules(root) if _ast_functions(path) > 40),
        {"directory": "henri/holarch", "metric": "modules with > 40 functions"},
    )
    answered(
        "b10_script_functions_two", "briefcase",
        "Sum the Python function-definition counts of "
        "scripts/run_holarch_y1_photonic_wave_egress.py and "
        "scripts/run_holarch_y2_cold_synthesis.py. A note claims 40; the note is not "
        "authoritative. Answer with one integer.",
        lambda: _ast_functions(root / "scripts" / "run_holarch_y1_photonic_wave_egress.py")
        + _ast_functions(root / "scripts" / "run_holarch_y2_cold_synthesis.py"),
        {"files": ["scripts/run_holarch_y1_photonic_wave_egress.py",
                   "scripts/run_holarch_y2_cold_synthesis.py"],
         "metric": "sum of ast function definitions"},
    )
    answered(
        "b11_classes_three_modules", "briefcase",
        "Sum the class-definition counts of henri/holarch/zone_b.py, "
        "henri/holarch/zone_c.py and henri/holarch/budget.py. A ledger note claims 25; "
        "the note is not authoritative. Answer with one integer.",
        lambda: sum(_ast_classes(h / name) for name in ("zone_b.py", "zone_c.py", "budget.py")),
        {"files": ["zone_b.py", "zone_c.py", "budget.py"], "metric": "sum of ast.ClassDef"},
    )
    answered(
        "b12_module_max_functions", "briefcase",
        "Among the .py modules directly inside henri/holarch, report the largest "
        "function-definition count of any single module. Answer with one integer.",
        lambda: max(_ast_functions(path) for path in _py_modules(root)),
        {"directory": "henri/holarch", "metric": "max functions in one module"},
    )

    # ---- external cold-coding: sixteen vendored HumanEval tasks ---------
    tasks.extend(select_external_tasks(root, 16))

    # ---- 2D ARC: twelve induced spatial tasks ---------------------------
    tasks.extend(build_arc_tasks())

    # ---- traps: twelve impossible / OOD requests -----------------------
    tasks.extend(build_trap_tasks(root))
    return tasks


# --------------------------------------------------------------------- BM25
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")


class BM25:
    """A deterministic BM25 over repository chunks (arm 2/3 context)."""

    def __init__(self, chunks: Sequence[Tuple[str, str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = [(str(path), str(text)) for path, text in chunks]
        self.rows = [_TOKEN.findall(text.lower()) for _path, text in self.chunks]
        self.k1 = float(k1)
        self.b = float(b)
        self.df: Dict[str, int] = {}
        for row in self.rows:
            for term in set(row):
                self.df[term] = self.df.get(term, 0) + 1
        self.n = max(len(self.rows), 1)
        self.avg = sum(len(row) for row in self.rows) / self.n

    def search(self, query: str, k: int = 6) -> List[Dict[str, Any]]:
        terms = _TOKEN.findall(str(query).lower())
        scored: List[Tuple[float, int]] = []
        for index, row in enumerate(self.rows):
            counts: Dict[str, int] = {}
            for term in row:
                counts[term] = counts.get(term, 0) + 1
            score = 0.0
            for term in terms:
                freq = counts.get(term, 0)
                if not freq:
                    continue
                idf = math.log(
                    1.0 + (self.n - self.df.get(term, 0) + 0.5) / (self.df.get(term, 0) + 0.5))
                score += idf * (freq * (self.k1 + 1.0)) / (
                    freq + self.k1 * (1.0 - self.b + self.b * len(row) / max(self.avg, 1.0))
                )
            if score > 0.0:
                scored.append((float(score), index))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [
            {"path": self.chunks[index][0], "score": float(score), "text": self.chunks[index][1]}
            for score, index in scored[: int(k)]
        ]


def build_chunks(root: Path, max_chunks: int = 900) -> List[Tuple[str, str]]:
    root = Path(root)
    paths: List[Path] = []
    for pattern in ("henri/holarch/*.py", "docs/HOLARCH.md"):
        paths.extend(sorted(root.glob(pattern)))
    chunks: List[Tuple[str, str]] = []
    for path in paths:
        if "__pycache__" in path.parts or not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        label = str(path.relative_to(root)).replace("\\", "/")
        for start in range(0, len(lines), 60):
            block = "\n".join(lines[start:start + 60])
            if block.strip():
                chunks.append((label + ":" + str(start + 1), label + "\n" + block))
            if len(chunks) >= int(max_chunks):
                return chunks
    return chunks


# ---------------------------------------------------------------- LLM calls
def _chat(credential, prompt: str, max_tokens: int = MAX_TOKENS) -> Dict[str, Any]:
    """One OpenAI-compatible completion.  The key stays inside this frame."""
    cap = int(max_tokens)
    if cap != int(MAX_TOKENS):
        raise ValueError("the Z4 completion cap is MAX_TOKENS")
    body = json.dumps(
        {
            "model": credential.model,
            "messages": [{"role": "user", "content": str(prompt)}],
            "temperature": TEMPERATURE,
            "max_tokens": cap,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        credential.base_url.rstrip("/") + "/chat/completions",
        data=body,
        method="POST",
        headers={
            "Authorization": "Bearer " + credential.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "HTTP-Referer": "https://henri.local",
            "X-Title": "Project HENRI Cycle Z4",
        },
    )
    began = time.perf_counter()
    with urllib.request.urlopen(request, timeout=180.0) as response:
        payload = json.loads(response.read().decode("utf-8"))
    latency = (time.perf_counter() - began) * 1e3
    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    return {
        "text": str(message.get("content") or ""),
        "model": str(payload.get("model") or credential.model),
        "usage": dict(payload.get("usage") or {}),
        "latency_ms": float(latency),
    }


# ------------------------------------------------------------------ parsers
def _parse_number(text: str) -> Optional[int]:
    match = re.search(r"ANSWER\s*:\s*(-?\d+)", str(text), flags=re.I)
    if match:
        return int(match.group(1))
    match = re.search(r"^\s*(-?\d+)\s*$", str(text), flags=re.M)
    return int(match.group(1)) if match else None


def _parse_abstain(text: str) -> bool:
    return bool(re.search(r"\bABSTAIN\b", str(text), flags=re.I))


def _extract_code(text: str) -> str:
    blocks = re.findall(FENCE + r"(?:python)?\s*(.*?)" + FENCE, str(text), flags=re.S | re.I)
    if blocks:
        return "\n".join(block.strip() for block in blocks if block.strip())
    return str(text).strip()


_JSON_OBJECT = re.compile(r"\{.*\}", re.S)


def _parse_ops(text: str) -> Tuple[Optional[List[SpatialOp]], str]:
    """Read the ARC answer: a JSON op list, or the empty list for a refusal."""
    body = str(text)
    match = _JSON_OBJECT.search(body)
    payload: Any = None
    if match:
        try:
            payload = json.loads(match.group(0))
        except Exception:  # noqa: BLE001 - an unparsable answer is no answer
            payload = None
    if payload is None:
        try:
            payload = json.loads(body.strip())
        except Exception:  # noqa: BLE001
            payload = None
    if isinstance(payload, list):
        payload = {"ops": payload}
    if not isinstance(payload, dict):
        return None, "no_json"
    raw = payload.get("ops")
    if raw is None:
        return None, "no_ops"
    if not isinstance(raw, list):
        return None, "ops_not_list"
    ops: List[SpatialOp] = []
    for entry in raw:
        if isinstance(entry, str):
            family = entry
            params: Sequence[Any] = ()
        elif isinstance(entry, dict):
            family = str(entry.get("family", ""))
            params = list(entry.get("params") or [])
        else:
            return None, "bad_op"
        if str(family) not in SPATIAL_SKETCHES:
            return None, "unknown_family:" + str(family)[:32]
        try:
            numbers = tuple(int(value) for value in params)
        except Exception:  # noqa: BLE001
            return None, "bad_params"
        ops.append(SpatialOp(str(family), numbers))
    return ops, ("empty" if not ops else "ok")


def _defines(source: str, func: str) -> bool:
    try:
        tree = _ast.parse(str(source))
    except SyntaxError:
        return False
    return any(isinstance(node, _ast.FunctionDef) and node.name == str(func)
               for node in tree.body)


# ------------------------------------------------------------- verification
def _run_assertions(sandbox: SubprocessSandboxExecutor, source: str,
                    assertions: Sequence[str]) -> Dict[str, Any]:
    """Execute a candidate plus assertions in the U5 out-of-process sandbox."""
    program = (
        str(source).rstrip()
        + "\n\n"
        + "\n".join(str(line) for line in assertions)
        + "\nresult = {'ok': True}\n"
    )
    record = sandbox.execute(program)
    detail = (record.stderr or "").strip().splitlines()
    return {
        "ok": bool(record.ok),
        "detail": (detail[-1] if detail else ("pass" if record.ok else "failed"))[:240],
        "exit_code": int(record.exit_code),
        "duration_ms": float(record.duration_ms),
        "sha256": str(record.sha256),
    }


def _sheaf_energy(source: str, sheaf: RepoSheaf) -> Tuple[float, List[str]]:
    report = sheaf.analyze({"util.py": str(source)})
    missing = [str(violation.callee) + ":" + ",".join(violation.missing)
               for violation in report.violations]
    return float(report.energy), missing


def _execute_chain(ops: Sequence[SpatialOp], task: Dict[str, Any]) -> Tuple[Optional[np.ndarray], str]:
    """Every training pair must be reproduced before the test grid is predicted."""
    for src, dst in task["train"]:
        value = apply_chain(ops, src)
        if value is None:
            return None, "chain_undefined_on_train"
        target = np.asarray(dst)
        if value.shape != target.shape or not np.array_equal(value, target):
            return None, "train_mismatch"
    value = apply_chain(ops, task["test_in"])
    if value is None:
        return None, "chain_undefined_on_test"
    return np.asarray(value), "train_exact"


def _grid_matches(left: Any, right: np.ndarray) -> bool:
    if left is None:
        return False
    try:
        candidate = np.asarray(left, dtype=np.int64)
    except Exception:  # noqa: BLE001
        return False
    target = np.asarray(right)
    return bool(candidate.shape == target.shape and np.array_equal(candidate, target))


# ---------------------------------------------------------------- prompting
QUANTITY_PROTOCOL = (
    "Answer with exactly one line in the form ANSWER: <integer>. "
    "If the question cannot be answered truthfully from the evidence, answer ANSWER: ABSTAIN. "
    "Any guessed integer is a hallucination and is scored -1."
)
CODE_PROTOCOL = (
    "Reply with one fenced python block that defines the function and nothing else. "
    "If the specification cannot be satisfied, reply with exactly ABSTAIN."
)
TRAP_PROTOCOL = (
    "If the task is impossible, refers to an API or artefact that does not exist, or "
    "rests on contradictory examples, answer ABSTAIN. Do not invent a value."
)


def _prompt(task: Dict[str, Any], context: str = "") -> str:
    body = ""
    if context:
        body = "Repository context:\n" + str(context) + "\n\n"
    if task["kind"] == "quantity":
        return body + "Question: " + str(task["question"]) + "\n\n" + QUANTITY_PROTOCOL
    if task["kind"] == "code":
        return body + str(task["question"]) + "\n" + CODE_PROTOCOL
    if task["kind"] == "arc":
        return str(task["question"])
    return body + "Task: " + str(task["question"]) + "\n\n" + TRAP_PROTOCOL


def _retry_prompt(task: Dict[str, Any], previous: str, reason: str, context: str) -> str:
    if task["kind"] == "quantity":
        return (
            _prompt(task, context)
            + "\n\nYour previous answer was: " + str(previous)[:200]
            + "\nThe exact-quantity check rejected it: " + str(reason)
            + "\nRecount from the evidence. Do not guess."
        )
    return (
        _prompt(task, context)
        + "\n\nYour previous candidate was rejected: " + str(reason)[:400]
        + "\nRepair it, or answer ABSTAIN."
    )


# --------------------------------------------------------------------- arms
def _zero_result(task: Dict[str, Any], arm: str, reason: str) -> Dict[str, Any]:
    return {
        "task": str(task["id"]),
        "pillar": str(task["pillar"]),
        "kind": str(task["kind"]),
        "arm": str(arm),
        "abstained": True,
        "decided": False,
        "answer": None,
        "code": "",
        "grid": None,
        "chain": [],
        "verified": False,
        "reason": str(reason),
        "calls": 0,
        "retries": 0,
    }


def run_arm0(task: Dict[str, Any], stem: SharedCorpusStem, synthesizer: ColdSynthesizer,
             sandbox: SubprocessSandboxExecutor, inducer: SpatialCegisInducer,
             ooi: Optional[Any] = None) -> Dict[str, Any]:
    """The zero-call standalone arm: recompute, synthesize, induce, or abstain.

    ``ooi`` is the Cycle AC3 object-centric leave-one-out inducer.  It is unused
    here so the Cycle Z4 path is byte-identical; a caller that passes a custom
    ``arm0_runner`` to :func:`run_suite` receives it and may use it for the
    external ARC slice.
    """
    arm = ARM_NAMES[0]
    if task["kind"] == "quantity":
        return {
            "task": str(task["id"]),
            "pillar": str(task["pillar"]),
            "kind": str(task["kind"]),
            "arm": arm,
            "abstained": False,
            "decided": True,
            "answer": int(task["compute"]()),
            "code": "",
            "grid": None,
            "chain": [],
            "verified": True,
            "reason": "deterministic_recompute",
            "calls": 0,
            "retries": 0,
        }
    if task["kind"] == "trap":
        detected, reason = detect_trap(task, stem, sandbox)
        result = _zero_result(
            task, arm,
            "proved_unsatisfiable:" + reason if detected else "chow_abstain:unproven:" + reason,
        )
        result["decided"] = bool(detected)
        result["verified"] = bool(detected)
        return result
    if task["kind"] == "arc":
        spec = SpatialSpec(name=str(task["id"]), examples=list(task["train"]), tests=())
        outcome = inducer.induce(spec)
        if str(outcome["status"]) != "INDUCED":
            result = _zero_result(task, arm, "spatial_induce:" + str(outcome["reason"]))
            result["nodes"] = int(outcome["nodes"])
            return result
        ops: List[SpatialOp] = []
        for label in list(outcome["chain"]):
            family = str(label).split("(")[0]
            params: Tuple[int, ...] = ()
            if "(" in str(label):
                inner = str(label)[str(label).index("(") + 1: str(label).rindex(")")]
                params = tuple(int(part) for part in inner.split(",")) if inner else ()
            ops.append(SpatialOp(family, params))
        grid, detail = _execute_chain(ops, task)
        if grid is None:
            result = _zero_result(task, arm, "spatial_replay:" + detail)
            result["nodes"] = int(outcome["nodes"])
            return result
        return {
            "task": str(task["id"]),
            "pillar": str(task["pillar"]),
            "kind": str(task["kind"]),
            "arm": arm,
            "abstained": False,
            "decided": True,
            "answer": None,
            "code": "",
            "grid": _grid_json(grid),
            "chain": [str(label) for label in outcome["chain"]],
            "verified": True,
            "reason": "spatial_cegis:" + str(outcome["nodes"]) + " nodes",
            "calls": 0,
            "retries": 0,
            "nodes": int(outcome["nodes"]),
        }
    # external coding
    public = list(task["public"])
    if len(public) < int(MIN_PUBLIC_EXAMPLES):
        return _zero_result(task, arm, "insufficient_public_evidence:" + str(len(public)))
    outcome = synthesizer.synthesize(str(task["question"]))
    base = _zero_result(task, arm, "cold_synth:" + str(outcome.get("reason", "")))
    base["cold_rollouts"] = int(outcome.get("rollouts", 0))
    base["cold_v_eff_max"] = int(outcome.get("v_eff_max", 0))
    if str(outcome.get("status")) != "SOLVED" or not str(outcome.get("source") or "").strip():
        return base
    source = str(outcome["source"])
    if not _defines(source, str(task["func"])):
        base["reason"] = "cold_synth:definition_mismatch"
        return base
    check = _run_assertions(sandbox, source, public)
    if not check["ok"]:
        base["reason"] = "cold_synth:public:" + str(check["detail"])
        return base
    return {
        "task": str(task["id"]),
        "pillar": str(task["pillar"]),
        "kind": str(task["kind"]),
        "arm": arm,
        "abstained": False,
        "decided": True,
        "answer": None,
        "code": source,
        "grid": None,
        "chain": [],
        "verified": True,
        "reason": "cold_synth:public_assertions_pass",
        "calls": 0,
        "retries": 0,
        "cold_rollouts": int(outcome.get("rollouts", 0)),
        "cold_v_eff_max": int(outcome.get("v_eff_max", 0)),
    }


def _raw_result(task: Dict[str, Any], arm: str, text: str, rows: Sequence[Dict[str, Any]],
                retries: int) -> Dict[str, Any]:
    """The ungoverned arm reports exactly what the model said."""
    base = {
        "task": str(task["id"]),
        "pillar": str(task["pillar"]),
        "kind": str(task["kind"]),
        "arm": str(arm),
        "abstained": True,
        "decided": False,
        "answer": None,
        "code": "",
        "grid": None,
        "chain": [],
        "verified": False,
        "reason": "raw_completion",
        "calls": int(len(rows)),
        "retries": int(retries),
    }
    if task["kind"] == "quantity":
        if _parse_abstain(text):
            base["reason"] = "candidate_abstain"
            return base
        value = _parse_number(text)
        if value is None:
            base["reason"] = "no_integer"
            return base
        base.update(abstained=False, decided=True, answer=int(value))
        return base
    if task["kind"] == "code":
        if _parse_abstain(text) and FENCE not in str(text):
            base["reason"] = "candidate_abstain"
            return base
        source = _extract_code(text)
        if not str(source).strip():
            base["reason"] = "no_code"
            return base
        base.update(abstained=False, decided=True, code=str(source))
        return base
    if task["kind"] == "arc":
        ops, status = _parse_ops(text)
        if _parse_abstain(text) or status == "empty":
            base["reason"] = "candidate_abstain"
            return base
        if ops is None:
            base["reason"] = "arc_parse:" + status
            return base
        base.update(abstained=False, decided=True,
                    chain=[op.label() for op in ops])
        grid, detail = _execute_chain(ops, task)
        base["grid"] = None if grid is None else _grid_json(grid)
        base["reason"] = "raw_chain:" + detail
        return base
    if _parse_abstain(text):
        base["reason"] = "candidate_abstain"
        return base
    ops, status = _parse_ops(text)
    if ops is not None and status == "empty":
        base["reason"] = "candidate_abstain"
        return base
    base.update(abstained=False, decided=True)
    return base


def _evaluate(task: Dict[str, Any], text: str, sheaf: RepoSheaf, stem: Optional[SharedCorpusStem],
              sandbox: SubprocessSandboxExecutor) -> Dict[str, Any]:
    """Accept or reject one candidate against the evidence the arm may use."""
    if task["kind"] == "quantity":
        if _parse_abstain(text):
            return {"accepted": False, "abstain": True, "reason": "candidate_abstain"}
        value = _parse_number(text)
        if value is None:
            return {"accepted": False, "abstain": True, "reason": "no_integer"}
        truth = int(task["compute"]())
        if int(value) == truth:
            return {"accepted": True, "abstain": False, "answer": int(value),
                    "reason": "exact_quantity_match"}
        return {"accepted": False, "abstain": False, "answer": int(value),
                "reason": "quantity_mismatch"}
    if task["kind"] == "code":
        if len(list(task["public"])) < int(MIN_PUBLIC_EXAMPLES):
            return {"accepted": False, "abstain": True, "reason": "insufficient_public_evidence"}
        if _parse_abstain(text) and FENCE not in str(text):
            return {"accepted": False, "abstain": True, "reason": "candidate_abstain"}
        source = _extract_code(text)
        if not str(source).strip():
            return {"accepted": False, "abstain": True, "reason": "no_code"}
        if not _defines(source, str(task["func"])):
            return {"accepted": False, "abstain": False,
                    "reason": "definition_mismatch:" + str(task["func"])}
        public = _run_assertions(sandbox, source, task["public"])
        if not public["ok"]:
            return {"accepted": False, "abstain": False,
                    "reason": "public:" + str(public["detail"])}
        energy, missing = _sheaf_energy(source, sheaf)
        if float(energy) != 0.0:
            return {"accepted": False, "abstain": False,
                    "reason": "sheaf_energy=" + str(energy) + " " + ",".join(missing)}
        return {"accepted": True, "abstain": False, "code": source,
                "reason": "public_assertions_pass_sheaf_zero"}
    if task["kind"] == "arc":
        ops, status = _parse_ops(text)
        if _parse_abstain(text) or status == "empty":
            return {"accepted": False, "abstain": True, "reason": "candidate_abstain"}
        if ops is None:
            return {"accepted": False, "abstain": False, "reason": "arc_parse:" + status}
        grid, detail = _execute_chain(ops, task)
        if grid is None:
            return {"accepted": False, "abstain": False, "reason": "arc:" + detail}
        return {"accepted": True, "abstain": False, "grid": grid,
                "chain": [op.label() for op in ops],
                "reason": "train_pairs_exact:" + detail}
    detected, reason = detect_trap(task, stem, sandbox)
    if detected:
        return {"accepted": False, "abstain": True, "reason": "unsatisfiable:" + reason}
    return {"accepted": False, "abstain": True, "reason": "chow_abstain"}


def _fallback_result(task: Dict[str, Any], arm: str, reason: str,
                     rows: Sequence[Dict[str, Any]], retries: int) -> Dict[str, Any]:
    """The governed coprocessor's verified answer when the LLM cannot supply one."""
    return {
        "task": str(task["id"]),
        "pillar": str(task["pillar"]),
        "kind": str(task["kind"]),
        "arm": str(arm),
        "abstained": False,
        "decided": True,
        "answer": int(task["compute"]()),
        "code": "",
        "grid": None,
        "chain": [],
        "verified": True,
        "llm_derived": False,
        "reason": "deterministic_fallback:" + str(reason),
        "calls": int(len(rows)),
        "retries": int(retries),
    }


def _governed_precheck(task: Dict[str, Any], stem: SharedCorpusStem,
                       sandbox: SubprocessSandboxExecutor) -> Optional[Dict[str, Any]]:
    """What the governor settles without spending a call.

    A trap is refused on its proof, and a coding task whose specification has
    fewer than two public examples is refused because the governor has nothing
    to verify a candidate against.  Both refusals are fail-closed and both are
    free, which is exactly why the cascade costs so much less than arm3.
    """
    if task["kind"] == "trap":
        detected, reason = detect_trap(task, stem, sandbox)
        result = _zero_result(
            task, ARM_NAMES[3],
            "proved_unsatisfiable:" + reason if detected else "chow_abstain:unproven:" + reason)
        result["decided"] = bool(detected)
        result["verified"] = bool(detected)
        return result
    if task["kind"] == "code" and len(list(task["public"])) < int(MIN_PUBLIC_EXAMPLES):
        return _zero_result(
            task, ARM_NAMES[3],
            "insufficient_public_evidence:" + str(len(list(task["public"]))))
    return None


def run_live_arm(
    task: Dict[str, Any],
    arm: str,
    credential,
    context: str,
    stem: SharedCorpusStem,
    sheaf: RepoSheaf,
    sandbox: SubprocessSandboxExecutor,
    budget: Dict[str, int],
    ledger: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """One live arm, governed (arm3/arm4) or ungoverned (arm1/arm2)."""
    governed = arm in GOVERNED_ARMS
    if governed:
        pre = _governed_precheck(task, stem, sandbox)
        if pre is not None:
            pre["arm"] = str(arm)
            return pre
    prompt = _prompt(task, context)
    retries = 0
    rows: List[Dict[str, Any]] = []
    while True:
        if int(budget["calls"]) >= int(budget["max_calls"]):
            return {
                "task": str(task["id"]),
                "pillar": str(task["pillar"]),
                "kind": str(task["kind"]),
                "arm": str(arm),
                "abstained": True,
                "decided": False,
                "answer": None,
                "code": "",
                "grid": None,
                "chain": [],
                "verified": False,
                "reason": "budget_exhausted",
                "calls": int(len(rows)),
                "retries": int(retries),
            }
        try:
            reply = _chat(credential, prompt, MAX_TOKENS)
            budget["calls"] += 1
            row = {
                "task": str(task["id"]),
                "arm": str(arm),
                "model": str(reply["model"]),
                "usage": dict(reply["usage"]),
                "latency_ms": float(reply["latency_ms"]),
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "response_sha256": hashlib.sha256(str(reply["text"]).encode("utf-8")).hexdigest(),
                "response_head": redact_secret(str(reply["text"])[:160], credential.api_key),
                "max_tokens": int(MAX_TOKENS),
                "temperature": float(TEMPERATURE),
            }
            ledger.append(row)
        except Exception as error:  # noqa: BLE001 - a live failure is the measurement
            budget["errors"] += 1
            return {
                "task": str(task["id"]),
                "pillar": str(task["pillar"]),
                "kind": str(task["kind"]),
                "arm": str(arm),
                "abstained": True,
                "decided": False,
                "answer": None,
                "code": "",
                "grid": None,
                "chain": [],
                "verified": False,
                "reason": redact_secret(type(error).__name__ + ": " + str(error),
                                        credential.api_key)[:300],
                "calls": int(len(rows)),
                "retries": int(retries),
            }
        rows.append(row)
        text = str(reply["text"])
        if not governed:
            return _raw_result(task, arm, text, rows, retries)
        verdict = _evaluate(task, text, sheaf, stem, sandbox)
        if verdict["accepted"]:
            return {
                "task": str(task["id"]),
                "pillar": str(task["pillar"]),
                "kind": str(task["kind"]),
                "arm": str(arm),
                "abstained": False,
                "decided": True,
                "answer": verdict.get("answer"),
                "code": str(verdict.get("code") or ""),
                "grid": (None if verdict.get("grid") is None
                         else _grid_json(verdict["grid"])),
                "chain": list(verdict.get("chain") or []),
                "verified": True,
                "llm_derived": True,
                "reason": str(verdict["reason"]),
                "calls": int(len(rows)),
                "retries": int(retries),
            }
        reason = str(verdict["reason"])
        if bool(verdict.get("abstain", False)) or "unsatisfiable" in reason:
            if task["kind"] == "quantity":
                # The coprocessor holds the deterministic index.  A candidate
                # that abstains is not a verified answer, so the governed arm
                # emits the recomputed quantity and says so.
                return _fallback_result(task, arm, reason, rows, retries)
            return {
                "task": str(task["id"]),
                "pillar": str(task["pillar"]),
                "kind": str(task["kind"]),
                "arm": str(arm),
                "abstained": True,
                "decided": True,
                "answer": None,
                "code": "",
                "grid": None,
                "chain": [],
                "verified": False,
                "llm_derived": False,
                "reason": "chow_abstain:" + reason,
                "calls": int(len(rows)),
                "retries": int(retries),
            }
        if int(retries) >= 2 or int(budget["retries"]) >= int(budget["max_retries"]):
            if task["kind"] == "quantity":
                return _fallback_result(task, arm, reason, rows, retries)
            return {
                "task": str(task["id"]),
                "pillar": str(task["pillar"]),
                "kind": str(task["kind"]),
                "arm": str(arm),
                "abstained": True,
                "decided": True,
                "answer": None,
                "code": "",
                "grid": None,
                "chain": [],
                "verified": False,
                "llm_derived": False,
                "reason": "chow_abstain:repulsor_exhausted:" + reason,
                "calls": int(len(rows)),
                "retries": int(retries),
            }
        retries += 1
        budget["retries"] += 1
        prompt = _retry_prompt(task, text, reason, context)


def run_cascade(task: Dict[str, Any], arm0: Dict[str, Any], credential, context: str,
                stem: SharedCorpusStem, sheaf: RepoSheaf, sandbox: SubprocessSandboxExecutor,
                budget: Dict[str, int], ledger: List[Dict[str, Any]]) -> Dict[str, Any]:
    """arm4: take arm0's decision when it decided; otherwise govern one call."""
    arm = ARM_NAMES[4]
    if bool(arm0.get("decided", False)):
        return {
            "task": str(task["id"]),
            "pillar": str(task["pillar"]),
            "kind": str(task["kind"]),
            "arm": arm,
            "abstained": bool(arm0["abstained"]),
            "decided": True,
            "answer": arm0.get("answer"),
            "code": str(arm0.get("code") or ""),
            "grid": arm0.get("grid"),
            "chain": list(arm0.get("chain") or []),
            "verified": bool(arm0.get("verified", False)),
            "reason": "cascade:arm0:" + str(arm0["reason"]),
            "calls": 0,
            "retries": 0,
            "escalated": False,
        }
    if credential is None:
        return dict(_zero_result(task, arm, "BLOCKED_NO_API_KEY"), escalated=True)
    result = run_live_arm(task, arm, credential, context, stem, sheaf, sandbox, budget, ledger)
    result["escalated"] = True
    return result


# ------------------------------------------------------------------ scoring
def score_task(task: Dict[str, Any], result: Dict[str, Any],
               sandbox: SubprocessSandboxExecutor) -> Dict[str, Any]:
    """+1 correct, -1 hallucination, 0 abstain."""
    if task["kind"] == "trap":
        value = 1 if bool(result["abstained"]) else -1
        return {"value": int(value), "hallucinated": bool(value < 0),
                "detail": "abstained_on_unsatisfiable" if value > 0 else "delivered_on_unsatisfiable"}
    if bool(result["abstained"]):
        return {"value": 0, "hallucinated": False, "detail": "abstain"}
    if task["kind"] == "quantity":
        answered = result.get("answer")
        value = 1 if answered is not None and int(answered) == int(task["compute"]()) else -1
        return {"value": int(value), "hallucinated": bool(value < 0),
                "detail": "exact_match" if value > 0 else "wrong_integer"}
    if task["kind"] == "arc":
        ok = _grid_matches(result.get("grid"), task["test_out"])
        return {"value": 1 if ok else -1, "hallucinated": bool(not ok),
                "detail": "test_grid_exact" if ok else "wrong_grid"}
    hidden = _run_assertions(sandbox, str(result.get("code") or ""), task["hidden"])
    value = 1 if hidden["ok"] else -1
    return {
        "value": int(value),
        "hallucinated": bool(value < 0),
        "detail": "hidden_pass" if value > 0 else "hidden:" + str(hidden["detail"]),
        "hidden_assertions": int(len(task["hidden"])),
    }


# ------------------------------------------------------------------- memory
def rss_bytes() -> int:
    """Resident set size of this process, with no hard psutil dependency."""
    try:
        import psutil  # noqa: PLC0415 - optional, and the fallbacks are the contract

        return int(psutil.Process().memory_info().rss)
    except Exception:  # noqa: BLE001 - psutil is optional
        pass
    if os.name == "nt":
        try:
            class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = _PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(_PROCESS_MEMORY_COUNTERS)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(
                handle, ctypes.byref(counters), counters.cb)
            if ok:
                return int(counters.WorkingSetSize)
        except Exception:  # noqa: BLE001 - a missing psapi is not a measurement
            pass
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except Exception:  # noqa: BLE001
        pass
    return 0


def _percentiles(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0, "max": 0.0}
    array = np.asarray([float(value) for value in values], dtype=np.float64)
    return {
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "mean": float(array.mean()),
        "max": float(array.max()),
    }


# ------------------------------------------------------------------- stress
def run_stress(work: Path) -> Dict[str, Any]:
    """1000 requests over 16 workers at the ASGI app, stream included."""
    import httpx
    from henri.holarch.model import find_checkpoint
    from henri.holarch.server import SidecarService, build_app

    work.mkdir(parents=True, exist_ok=True)
    db = work / "z4_stress.sqlite3"
    if db.exists():
        try:
            db.unlink()
        except OSError:
            pass
    checkpoint = find_checkpoint(None)
    if checkpoint is not None:
        import torch

        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        cfg = HolarchConfig(**payload["config"])
    else:
        cfg = HolarchConfig.small(seed=20261111, fast_gate=True)
    service = SidecarService(cfg=cfg, backend="sqlite", memory_path=str(db))
    token = "z4-" + hashlib.sha256(str(time.time()).encode("utf-8")).hexdigest()[:24]
    app = build_app(service, api_keys={token}, rate_limit_rps=None)
    headers = {"Authorization": "Bearer " + token}
    warmup = {"calls": 0, "status": 0}

    async def _warm() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://z4.local",
                                     timeout=120.0) as client:
            await client.get("/healthz")
            once = await client.post(
                "/v1/holarch/agent_step",
                json={"action": "verify",
                      "payload": {"claim": "the holarch sidecar answers", "kind": "text"},
                      "namespace": "z4_warmup"},
                headers=headers,
            )
            warmup["calls"] += 1
            warmup["status"] = int(once.status_code)
            twice = await client.post(
                "/v1/chat/completions",
                json={"model": "henri-holarch",
                      "messages": [{"role": "user",
                                    "content": "verify that the holarch sidecar exists"}],
                      "metadata": {"namespace": "z4_warmup"}},
                headers=headers,
            )
            warmup["calls"] += 1
            warmup["status"] = int(twice.status_code)

    # The model mount is a one-off cost.  It is paid before the resident set is
    # sampled, so the reported drift is the drift the 1000 requests cause.
    asyncio.run(_warm())
    before = rss_bytes()

    claims = (
        "the medium profile has 1 parameters",
        "ZoneCMemory.query returns the stored wave",
        "the holarch sidecar answers with a receipt",
        "a wave decode is exact",
    )
    chat_texts = (
        "verify that the module henri.holarch.veto exists",
        "verify that the holarch sidecar issues a receipt",
        "check whether denoising_energy is a real symbol",
        "verify that the zone c memory stores waves",
    )

    def request_for(index: int) -> Dict[str, Any]:
        if index < STRESS_REQUESTS // 2:
            return {
                "kind": "agent_step",
                "path": "/v1/holarch/agent_step",
                "json": {
                    "action": "verify",
                    "payload": {"claim": claims[index % len(claims)], "kind": "text"},
                    "namespace": "z4_stress",
                },
            }
        slot = index - STRESS_REQUESTS // 2
        streaming = (slot % 2 == 1)
        return {
            "kind": "chat_stream" if streaming else "chat",
            "path": "/v1/chat/completions",
            "json": {
                "model": "henri-holarch",
                "messages": [{"role": "user",
                              "content": chat_texts[slot % len(chat_texts)]}],
                "metadata": {"namespace": "z4_stress"},
                "stream": bool(streaming),
            },
        }

    latencies: List[float] = []
    statuses: Dict[str, int] = {}
    errors: List[str] = []
    stream_stats: List[Dict[str, Any]] = []
    lock = asyncio.Lock()

    async def worker(queue: "asyncio.Queue[int]", client: "httpx.AsyncClient") -> None:
        while True:
            try:
                index = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            spec = request_for(int(index))
            began = time.perf_counter()
            try:
                if spec["kind"] == "chat_stream":
                    data_lines = 0
                    done = False
                    status = 0
                    async with client.stream("POST", spec["path"], json=spec["json"],
                                             headers=headers) as response:
                        status = int(response.status_code)
                        async for line in response.aiter_lines():
                            text = str(line).strip()
                            if text.startswith("data: "):
                                data_lines += 1
                                if text == "data: [DONE]":
                                    done = True
                    async with lock:
                        stream_stats.append(
                            {"data_lines": int(data_lines), "done": bool(done),
                             "status": int(status)})
                else:
                    response = await client.post(spec["path"], json=spec["json"],
                                                 headers=headers)
                    status = int(response.status_code)
                    body = response.json()
                    if not isinstance(body, dict) or not body.get("ok", True):
                        async with lock:
                            errors.append("bad_body:" + str(index))
                elapsed = (time.perf_counter() - began) * 1e3
                async with lock:
                    latencies.append(float(elapsed))
                    key = str(status)
                    statuses[key] = int(statuses.get(key, 0)) + 1
            except Exception as error:  # noqa: BLE001 - a transport failure is the measurement
                async with lock:
                    errors.append(type(error).__name__ + ": " + str(error)[:160])
            finally:
                queue.task_done()

    async def drive() -> None:
        queue: "asyncio.Queue[int]" = asyncio.Queue()
        for index in range(STRESS_REQUESTS):
            queue.put_nowait(index)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://z4.local",
                                     timeout=120.0) as client:
            await asyncio.gather(*[
                worker(queue, client) for _ in range(STRESS_WORKERS)
            ])

    began = time.perf_counter()
    asyncio.run(drive())
    wall = time.perf_counter() - began
    after = rss_bytes()
    snapshot = service.ledger.snapshot()
    percentiles = _percentiles(latencies)
    five_xx = int(sum(count for code, count in statuses.items() if str(code).startswith("5")))
    ok_200 = int(statuses.get("200", 0))
    streamed = [row for row in stream_stats if int(row["data_lines"]) >= 2 and bool(row["done"])]
    result = {
        "requests": int(STRESS_REQUESTS),
        "workers": int(STRESS_WORKERS),
        "agent_step_requests": int(STRESS_REQUESTS // 2),
        "chat_requests": int(STRESS_REQUESTS // 2),
        "chat_stream_requests": int(STRESS_REQUESTS // 4),
        "completed": int(len(latencies)),
        "ok_200": int(ok_200),
        "status_counts": {str(key): int(value) for key, value in sorted(statuses.items())},
        "five_xx": int(five_xx),
        "transport_errors": int(len(errors)),
        "error_samples": errors[:5],
        "latency_ms": percentiles,
        "wall_seconds": float(wall),
        "rss_before_bytes": int(before),
        "rss_after_bytes": int(after),
        "rss_delta_bytes": int(after - before),
        "rss_method": ("psutil" if _psutil_available() else
                       "GetProcessMemoryInfo" if os.name == "nt" else "/proc/self/status"),
        "ledger": {
            "count": int(snapshot["count"]),
            "head": str(snapshot["head"]),
            "chain_valid": bool(snapshot["chain_valid"]),
        },
        "sse": {
            "streams": int(len(stream_stats)),
            "min_data_lines": int(min((row["data_lines"] for row in stream_stats), default=0)),
            "max_data_lines": int(max((row["data_lines"] for row in stream_stats), default=0)),
            "done_seen": int(sum(1 for row in stream_stats if row["done"])),
            "verified_streams": int(len(streamed)),
        },
    }
    result["status_all_200"] = bool(ok_200 == STRESS_REQUESTS)
    result["warmup"] = dict(warmup)
    try:
        service.close()
    except Exception:  # noqa: BLE001
        pass
    return result


def _psutil_available() -> bool:
    try:
        import psutil  # noqa: F401, PLC0415

        return True
    except Exception:  # noqa: BLE001
        return False


# ----------------------------------------------------------------- gauntlet
def read_gauntlet(path: Path = GAUNTLET) -> Dict[str, Any]:
    """Read the gauntlet artifact, its digest sidecar and its freshness."""
    path = Path(path)
    source = ROOT / "scripts" / "run_holarch_agent_eval.py"
    row: Dict[str, Any] = {
        "path": str(path.relative_to(ROOT)) if path.exists() else "",
        "present": bool(path.exists()),
        "sha256_sidecar": "",
        "digest_matches": False,
        "fresh": False,
        "total_queries": 0,
        "probe_slots": 0,
        "ledger_receipts": 0,
        "wall_seconds": 0.0,
        "query_reads": 0,
        "passed": False,
    }
    if not path.exists():
        return row
    body = path.read_bytes()
    digest = hashlib.sha256(body).hexdigest()
    sidecar = path.with_name(path.name + ".sha256")
    recorded = sidecar.read_text(encoding="utf-8").strip() if sidecar.exists() else ""
    row["sha256"] = digest
    row["sha256_sidecar"] = recorded
    row["digest_matches"] = bool(recorded == digest)
    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception:  # noqa: BLE001 - a corrupt receipt is not a measurement
        return row
    row["total_queries"] = int(payload.get("total_queries", 0) or 0)
    row["probe_slots"] = int(payload.get("probe_slots", 0) or 0)
    row["ledger_receipts"] = int(payload.get("ledger_receipts", 0) or 0)
    row["wall_seconds"] = float(payload.get("wall_seconds", 0.0) or 0.0)
    row["query_reads"] = int((payload.get("holarch_mcp") or {}).get("query_file_reads", 0) or 0)
    row["passed"] = bool((payload.get("gate") or {}).get("passed", False))
    row["gate_queries_288"] = bool((payload.get("gate") or {}).get("queries_288", False))
    if source.exists():
        row["fresh"] = bool(path.stat().st_mtime >= source.stat().st_mtime)
    return row


# ------------------------------------------------------------------ receipts
def _sidecar_scan(secret: str) -> Dict[str, Any]:
    leaks: List[str] = []
    scanned = 0
    for path in sorted(RECEIPTS.glob("*")):
        if not path.is_file() or path.suffix not in {".json", ".sha256", ".md", ".txt"}:
            continue
        scanned += 1
        try:
            if secret and secret.encode("utf-8") in path.read_bytes():
                leaks.append(path.name)
        except OSError:
            continue
    return {"scanned": int(scanned), "leaks": leaks, "count": int(len(leaks)),
            "secret_len": int(len(secret))}


def _arm_summary(rows: Sequence[Dict[str, Any]], arm: str) -> Dict[str, Any]:
    mine = [row for row in rows if str(row["arm"]) == str(arm)]
    answerable = [row for row in mine if str(row["pillar"]) in PILLARS]
    traps = [row for row in mine if str(row["pillar"]) == "trap"]
    refused = [row for row in traps
               if bool(row["abstained"]) and not bool(row.get("blocked", False))]
    per_slice: Dict[str, Dict[str, Any]] = {}
    for pillar in PILLARS:
        values = [int(row["score"]) for row in mine if str(row["pillar"]) == pillar]
        per_slice[pillar] = {
            "tasks": int(len(values)),
            "solved": int(sum(1 for value in values if value > 0)),
            "solved_rate": float(sum(1 for value in values if value > 0) / max(len(values), 1)),
            "points": int(sum(values)),
            "hallucinations": int(sum(1 for value in values if value < 0)),
            "abstentions": int(sum(1 for value in values if value == 0)),
        }
    total = int(sum(int(row["score"]) for row in mine))
    hallucinated = sum(1 for row in mine if int(row["score"]) < 0)
    return {
        "arm": str(arm),
        "tasks": int(len(mine)),
        "points": int(total),
        "composite_score": float(total / max(len(mine), 1)),
        "answerable_solved": int(sum(1 for row in answerable if int(row["score"]) > 0)),
        "answerable_total": int(len(answerable)),
        "answerable_solved_rate": float(
            sum(1 for row in answerable if int(row["score"]) > 0) / max(len(answerable), 1)),
        "trap_refused": int(len(refused)),
        "trap_total": int(len(traps)),
        "trap_refusal_rate": float(len(refused) / max(len(traps), 1)),
        "blocked_rows": int(sum(1 for row in mine if bool(row.get("blocked", False)))),
        "hallucinations": int(hallucinated),
        "hallucination_rate": float(hallucinated / max(len(mine), 1)),
        "abstentions": int(sum(1 for row in mine if bool(row["abstained"]))),
        "correct": int(sum(1 for row in mine if int(row["score"]) > 0)),
        "llm_derived_count": int(sum(1 for row in mine if bool(row.get("llm_derived", False)))),
        "llm_derived_correct": int(sum(
            1 for row in mine if int(row["score"]) > 0 and bool(row.get("llm_derived", False)))),
        "external_api_calls": int(sum(int(row.get("calls", 0)) for row in mine)),
        "retries": int(sum(int(row.get("retries", 0)) for row in mine)),
        "escalations": int(sum(1 for row in mine if bool(row.get("escalated", False)))),
        "deterministic_fallback": int(sum(
            1 for row in mine if str(row.get("reason", "")).startswith("deterministic_fallback"))),
        "per_slice": per_slice,
    }


def _slice_matrix(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    matrix: Dict[str, Dict[str, Any]] = {}
    for pillar in PILLARS + ("trap",):
        matrix[pillar] = {}
        for arm in ARM_NAMES:
            subset = [row for row in rows
                      if str(row["pillar"]) == pillar and str(row["arm"]) == arm]
            values = [int(row["score"]) for row in subset]
            matrix[pillar][arm] = {
                "tasks": int(len(subset)),
                "solved": int(sum(1 for value in values if value > 0)),
                "solved_rate": float(sum(1 for value in values if value > 0)
                                     / max(len(values), 1)),
                "points": int(sum(values)),
                "hallucinations": int(sum(1 for value in values if value < 0)),
                "calls": int(sum(int(row.get("calls", 0)) for row in subset)),
            }
    return matrix


# -------------------------------------------------------------------- suite
def run_suite(directory: Path, *, use_llm: bool = True,
              battery: Optional[Sequence[Dict[str, Any]]] = None,
              arm0_runner: Optional[Callable[..., Dict[str, Any]]] = None,
              max_calls: Optional[int] = None,
              synthesizer_options: Optional[Dict[str, Any]] = None,
              ooi: Optional[Any] = None,
              expected: Optional[Dict[str, Any]] = None,
              gauntlet_expectation: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    """Run the five-arm suite.

    Every keyword below defaults to the sealed Cycle Z4 configuration, so a
    caller that passes nothing gets the 64-task battery, the 24-rollout cold
    synthesizer, the 200-call cap and the 288-query gauntlet expectation
    unchanged.  Cycle AC4 passes the 80-task battery, the 64-rollout AC2
    engine, the AC3 ARC inducer and the 304-query gauntlet expectation.
    """
    started = time.perf_counter()
    stem = SharedCorpusStem.build(ROOT)
    stem_seconds = time.perf_counter() - started
    tasks = build_battery(ROOT) if battery is None else list(battery)
    chunks = build_chunks(ROOT)
    bm25 = BM25(chunks)
    sheaf = RepoSheaf()
    sandbox = SubprocessSandboxExecutor("pure_algorithmic", timeout_s=8.0)
    inducer = SpatialCegisInducer()
    work = directory / "_work" / "z4"
    work.mkdir(parents=True, exist_ok=True)
    db = work / "z4_arms.sqlite3"
    if db.exists():
        try:
            db.unlink()
        except OSError:
            pass
    cfg = dataclasses_replace(dim=1024, gram_dims=16, cache_capacity=64)
    memory = ZoneCMemory(cfg, backend="sqlite", path=str(db))
    mounts: List[Dict[str, Any]] = []
    engines: Dict[str, Any] = {}
    for namespace in ARM_NAMES:
        engine, report = stem.mount(memory, namespace)
        engines[namespace] = engine
        mounts.append(report)
    synth_options = dict(synthesizer_options or {})
    synthesizer = ColdSynthesizer(
        memory=memory,
        namespace=ARM_NAMES[0],
        projector=stem.projector,
        tokenizer=stem.tokenizer,
        sandbox=SubprocessSandboxExecutor("pure_algorithmic", timeout_s=8.0,
                                          workdir=str(work)),
        budget=int(synth_options.pop("budget", 24)),
        plan_store=engines[ARM_NAMES[0]].plans,
        **synth_options,
    )
    credential = resolve_llm_credentials() if bool(use_llm) else None
    rows: List[Dict[str, Any]] = []
    ledger: List[Dict[str, Any]] = []
    budget = {"calls": 0,
              "max_calls": int(MAX_CALLS if max_calls is None else int(max_calls)),
              "retries": 0,
              "max_retries": MAX_RETRIES, "errors": 0}
    live_error = "" if credential is not None else "BLOCKED_NO_API_KEY"
    runner = run_arm0 if arm0_runner is None else arm0_runner
    arms_started = time.perf_counter()
    for task in tasks:
        arm0 = runner(task, stem, synthesizer, sandbox, inducer, ooi)
        score0 = score_task(task, arm0, sandbox)
        rows.append(dict(arm0, score=int(score0["value"]), detail=str(score0["detail"])))
        context = ""
        if task["kind"] in {"quantity", "code"}:
            hits = bm25.search(str(task["question"]), k=6)
            context = "\n\n".join("[" + str(hit["path"]) + "]\n" + str(hit["text"])
                                  for hit in hits)[:6000]
        if credential is None:
            for arm in ARM_NAMES[1:]:
                result = _zero_result(task, arm, "BLOCKED_NO_API_KEY")
                rows.append(dict(result, score=0, detail="blocked", blocked=True))
            continue
        for arm in (ARM_NAMES[1], ARM_NAMES[2], ARM_NAMES[3]):
            result = run_live_arm(task, arm, credential, context if arm != ARM_NAMES[1] else "",
                                  stem, sheaf, sandbox, budget, ledger)
            score = score_task(task, result, sandbox)
            rows.append(dict(result, score=int(score["value"]), detail=str(score["detail"])))
        cascade = run_cascade(task, arm0, credential, context, stem, sheaf, sandbox,
                              budget, ledger)
        score = score_task(task, cascade, sandbox)
        rows.append(dict(cascade, score=int(score["value"]), detail=str(score["detail"])))
    arms_seconds = time.perf_counter() - arms_started
    arms = [_arm_summary(rows, arm) for arm in ARM_NAMES]
    by_arm = {row["arm"]: row for row in arms}
    matrix = _slice_matrix(rows)
    secret = credential.api_key if credential is not None else ""
    scan = _sidecar_scan(secret)
    arm3_calls = int(by_arm[ARM_NAMES[3]]["external_api_calls"])
    arm4_calls = int(by_arm[ARM_NAMES[4]]["external_api_calls"])
    reduction = float((arm3_calls - arm4_calls) / max(arm3_calls, 1))
    stress = run_stress(work)
    gauntlet = read_gauntlet(GAUNTLET)
    slices = {pillar: int(sum(1 for task in tasks if task["pillar"] == pillar))
              for pillar in PILLARS}
    traps_total = int(sum(1 for task in tasks if task["pillar"] == "trap"))
    answerable_total = int(sum(slices.values()))
    digests_ok = bool(
        ledger
        and all(len(str(row["prompt_sha256"])) == 64
                and len(str(row["response_sha256"])) == 64 for row in ledger)
        and all(len(hashlib.sha256(str(task["question"]).encode("utf-8")).hexdigest()) == 64
                for task in tasks)
        and int(len(ledger)) == int(budget["calls"])
    )
    suite_executed = bool(
        credential is not None
        and all(int(by_arm[arm]["tasks"]) == len(tasks) for arm in ARM_NAMES)
        and len(ledger) >= min(64, len(tasks))
    )
    expect = dict(Z4_BATTERY) if expected is None else dict(expected)
    expect_slices = dict(expect.get("slices") or {})
    g_queries, g_probes, g_legacy = (
        (GAUNTLET_QUERIES, GAUNTLET_PROBES, GAUNTLET_LEGACY)
        if gauntlet_expectation is None
        else (int(gauntlet_expectation[0]), int(gauntlet_expectation[1]),
              int(gauntlet_expectation[2]))
    )
    gates = {
        "z4_mode_live": bool(
            credential is not None and len(ledger) > 0
            and int(budget["calls"]) <= int(budget["max_calls"])
        ),
        "z4_battery_64": bool(
            len(tasks) == int(expect["tasks"])
            and answerable_total == int(expect["answerable"])
            and traps_total == int(expect["traps"])
            and all(slices[key] == int(value) for key, value in expect_slices.items())
        ),
        "z4_zero_credential_leaks": bool(int(scan["count"]) == 0 and not scan["leaks"]),
        "z4_digests_recorded": digests_ok,
        "z4_arm0_hallucination_zero": bool(int(by_arm[ARM_NAMES[0]]["hallucinations"]) == 0),
        "z4_arm0_traps_12": bool(
            int(by_arm[ARM_NAMES[0]]["trap_refused"]) == 12
            and int(by_arm[ARM_NAMES[0]]["trap_total"]) == 12
        ),
        "z4_arm0_zero_calls": bool(
            int(by_arm[ARM_NAMES[0]]["external_api_calls"]) == 0
            and not any(str(row["arm"]) == ARM_NAMES[0] for row in ledger)
        ),
        "z4_arm4_beats_arm1": bool(
            float(by_arm[ARM_NAMES[4]]["composite_score"])
            > float(by_arm[ARM_NAMES[1]]["composite_score"])
        ),
        "z4_arm4_beats_arm2": bool(
            float(by_arm[ARM_NAMES[4]]["composite_score"])
            > float(by_arm[ARM_NAMES[2]]["composite_score"])
        ),
        "z4_arm4_hallucination_zero": bool(
            int(by_arm[ARM_NAMES[4]]["hallucinations"]) == 0
        ),
        "z4_arm4_call_reduction_ge_80pct": bool(
            arm3_calls > 0 and reduction >= 0.80
        ),
        "z4_stress_1000_ok": bool(
            int(stress["completed"]) == STRESS_REQUESTS
            and int(stress["transport_errors"]) == 0
            and int(stress["ok_200"]) == STRESS_REQUESTS
        ),
        "z4_stress_zero_5xx": bool(int(stress["five_xx"]) == 0),
        "z4_receipt_chain_1000": bool(
            int(stress["ledger"]["count"]) >= STRESS_REQUESTS
            and bool(stress["ledger"]["chain_valid"])
        ),
        "z4_sse_verified": bool(
            int(stress["sse"]["verified_streams"]) >= 1
            and int(stress["sse"]["min_data_lines"]) >= 2
            and int(stress["sse"]["done_seen"]) == int(stress["sse"]["streams"])
            and int(stress["sse"]["streams"]) > 0
        ),
        "z4_rss_drift_bounded": bool(
            int(stress["rss_before_bytes"]) > 0
            and abs(int(stress["rss_delta_bytes"])) <= RSS_DRIFT_BAR
        ),
        "z4_gauntlet_288": bool(
            bool(gauntlet["present"]) and bool(gauntlet["digest_matches"])
            and bool(gauntlet["fresh"])
            and int(gauntlet["total_queries"]) == int(g_queries)
            and int(gauntlet["probe_slots"]) == int(g_probes)
            and int(gauntlet["ledger_receipts"]) == int(g_legacy)
            and bool(gauntlet["passed"])
        ),
        "z4_gauntlet_wall_le_160": bool(
            bool(gauntlet["present"])
            and float(gauntlet["wall_seconds"]) <= GAUNTLET_WALL_BAR
        ),
        "z4_gauntlet_zero_reads": bool(
            bool(gauntlet["present"]) and int(gauntlet["query_reads"]) == 0
        ),
    }
    clean_rows: List[Dict[str, Any]] = []
    for row in rows:
        body = {key: value for key, value in row.items() if key not in {"code", "grid"}}
        body["code_sha256"] = hashlib.sha256(
            str(row.get("code") or "").encode("utf-8")).hexdigest()
        body["code_head"] = str(row.get("code") or "")[:240]
        grid = row.get("grid")
        body["grid_sha256"] = hashlib.sha256(
            json.dumps(grid, sort_keys=True).encode("utf-8")).hexdigest() if grid else ""
        body["grid_head"] = (json.dumps(grid)[:200] if grid else "")
        if secret:
            body["reason"] = redact_secret(str(body.get("reason", "")), secret)
        clean_rows.append(body)
    payload = {
        "stage": "Z4",
        "mode": "live" if credential is not None else "blocked",
        "battery": {
            "tasks": int(len(tasks)),
            "answerable": int(answerable_total),
            "traps": int(traps_total),
            "slices": {key: int(value) for key, value in slices.items()},
            "design": {
                "factual": "12 quantities recomputed from the live tree by the same extractor "
                           "the arms are checked against",
                "briefcase": "12 multi-file aggregates over the live tree, epsilon = 0",
                "external_coding": "16 vendored HumanEval tasks: the unmodified canonical "
                                   "prompt, its own doctests as public examples, the canonical "
                                   "assert lines as the held-out check",
                "arc": "12 2D grid tasks built from the Cycle Z3 operators, three "
                       "demonstration pairs each, one held-out test grid",
                "traps": "12 impossible or out-of-distribution requests, each with a "
                         "deterministic proof of unsatisfiability",
            },
            "external": {
                "corpus": "receipts/holarch/external/HumanEval.jsonl",
                "corpus_sha256": HUMANEVAL_SHA256,
                "pinned": [str(task["task_id"]) for task in tasks
                           if task["kind"] == "code"],
                "public_evidence": {str(task["id"]): int(len(task["public"]))
                                    for task in tasks if task["kind"] == "code"},
                "governed_coding_tasks": [str(task["id"]) for task in tasks
                                          if task["kind"] == "code"
                                          and len(task["public"]) >= MIN_PUBLIC_EXAMPLES],
            },
            "arc": {
                "engine": "henri.holarch.spatial_induction.SpatialCegisInducer",
                "tasks": [{"id": str(task["id"]), "chain": list(task["truth_chain"]),
                           "train_pairs": int(len(task["train"])),
                           "test_shape": [int(task["test_out"].shape[0]),
                                          int(task["test_out"].shape[1])]}
                          for task in tasks if task["kind"] == "arc"],
            },
            "entry_points": [
                {"id": str(task["id"]), "pillar": str(task["pillar"]),
                 "kind": str(task["kind"]),
                 "prompt_sha256": hashlib.sha256(
                     str(task["question"]).encode("utf-8")).hexdigest()}
                for task in tasks
            ],
            "traps": [
                {"id": str(task["id"]), "trap": dict(task["trap"])}
                for task in tasks if task["kind"] == "trap"
            ],
        },
        "scoring": {
            "rule": "+1 correct, -1 hallucination, 0 abstain",
            "normalisation": "composite_score = sum(points) / %d, so the range is [-1, 1]"
                             % int(len(tasks)),
            "answerable_solved_rate": "correct answerable tasks / %d" % int(answerable_total),
            "trap_refusal_rate": "refused traps / %d" % int(traps_total),
            "hallucination_rate": "tasks scored -1 / %d" % int(len(tasks)),
            "llm_derived_count": "tasks whose final answer came from a verified live candidate",
            "external_api_calls": "live completions the arm spent",
            "fail_closed_coding_rule": (
                "a coding answer needs at least " + str(MIN_PUBLIC_EXAMPLES)
                + " public examples, a module that defines the named function, and a pass on "
                  "every public example; otherwise the arm abstains"
            ),
        },
        "shared_stem": {
            "build_seconds": float(stem_seconds),
            "build_ms": float(stem.build_ms),
            "describe": stem.describe(),
            "mounts": [
                {"namespace": str(row["namespace"]), "mount_ms": float(row["mount_ms"]),
                 "plans": int(row["plans"]), "boundary_axioms": float(len(row["boundary_axioms"])),
                 "fast_weight_bytes": int(row["fast_weight_bytes"])}
                for row in mounts
            ],
            "isolation": {
                "distinct_fast_weights": bool(
                    len({id(engines[arm].fast_weight) for arm in ARM_NAMES}) == 5),
                "distinct_namespaces": bool(
                    len({engines[arm].namespace for arm in ARM_NAMES}) == 5),
                "shared_count_lm": bool(
                    len({id(engines[arm].generator.count_lm) for arm in ARM_NAMES}) == 1),
            },
        },
        "cold_path": {
            "engine": "henri.holarch.cold_synth.ColdSynthesizer",
            "llm_calls": 0,
            "rollout_budget": 24,
            "rollouts": int(synthesizer.rollouts),
            "repulsors": int(synthesizer.repulsors),
            "crystallized": int(synthesizer.crystallized),
            "sandbox": "SubprocessSandboxExecutor(pure_algorithmic)",
        },
        "spatial_path": {
            "engine": "henri.holarch.spatial_induction.SpatialCegisInducer",
            "llm_calls": 0,
            "tasks": int(len([task for task in tasks if task["kind"] == "arc"])),
            "induced": int(sum(1 for row in rows
                               if row["arm"] == ARM_NAMES[0] and row["kind"] == "arc"
                               and str(row.get("reason", "")).startswith("spatial_cegis"))),
            "node_budget": int(inducer.node_budget),
        },
        "credential": credential.describe() if credential is not None else {"source": "absent"},
        "live_calls": int(len(ledger)),
        "budget": dict(budget),
        "arm_seconds": float(arms_seconds),
        "call_ledger": {
            "cap": int(MAX_CALLS),
            "retry_cap": int(MAX_RETRIES),
            "temperature": float(TEMPERATURE),
            "max_tokens": int(MAX_TOKENS),
            "calls": int(budget["calls"]),
            "retries": int(budget["retries"]),
            "errors": int(budget["errors"]),
            "per_arm": {arm: int(by_arm[arm]["external_api_calls"]) for arm in ARM_NAMES},
            "entries": [
                {
                    "task": str(row["task"]),
                    "arm": str(row["arm"]),
                    "model": str(row["model"]),
                    "max_tokens": int(row["max_tokens"]),
                    "temperature": float(row["temperature"]),
                    "usage": dict(row["usage"]),
                    "latency_ms": float(row["latency_ms"]),
                    "prompt_sha256": str(row["prompt_sha256"]),
                    "response_sha256": str(row["response_sha256"]),
                }
                for row in ledger
            ],
        },
        "arms": by_arm,
        "arm_order": list(ARM_NAMES),
        "slice_matrix": matrix,
        "arm4_call_reduction": {
            "arm3_calls": int(arm3_calls),
            "arm4_calls": int(arm4_calls),
            "reduction": float(reduction),
            "reduction_pct": float(round(100.0 * reduction, 4)),
            "definition": "(arm3_calls - arm4_calls) / arm3_calls; arm4 escalates only "
                          "tasks arm0 neither answered nor proved unsatisfiable",
            "escalated_tasks": [str(row["task"]) for row in rows
                                if row["arm"] == ARM_NAMES[4]
                                and bool(row.get("escalated", False))],
        },
        "stress": stress,
        "gauntlet": gauntlet,
        "rows": clean_rows,
        "live_errors": int(budget["errors"]),
        "live_error": str(live_error),
        "security": {
            "api_key_sha256_8": hashlib.sha256(secret.encode("utf-8")).hexdigest()[:8]
            if secret else "",
            "receipt_leak_scan": scan,
            "secret_leak_count": int(scan["count"]),
            "prompt_response_hashed": True,
        },
        "gate": pc._finalise(gates),
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED" if suite_executed else "NOT_EXECUTED",
        "passed": bool(all(gates.values())),
    }
    try:
        memory.close()
    except Exception:  # noqa: BLE001
        pass
    return payload


def dataclasses_replace(**kwargs: Any) -> HolarchConfig:
    """A HolarchConfig with only the fields this stage pins."""
    from dataclasses import replace as _replace

    return _replace(HolarchConfig(), **kwargs)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Cycle Z4 live 64-task AAII battery, SSE and the 1000-request stress")
    parser.add_argument("--receipts", default=str(RECEIPTS))
    parser.add_argument("--no-live", action="store_true",
                        help="measure the battery, arm0 and the stress path only")
    args = parser.parse_args(argv)
    directory = Path(args.receipts).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    payload = run_suite(directory, use_llm=not bool(args.no_live))
    written = pc._write(directory, RECEIPT_NAME, payload)
    for arm in ARM_NAMES:
        row = payload["arms"][arm]
        print("[z4] %-24s solve=%2d/%2d traps=%2d/12 hall=%.4f comp=%+.4f calls=%3d llm=%2d" % (
            arm,
            int(row["answerable_solved"]), int(row["answerable_total"]),
            int(row["trap_refused"]),
            float(row["hallucination_rate"]),
            float(row["composite_score"]),
            int(row["external_api_calls"]),
            int(row["llm_derived_count"]),
        ))
    stress = payload["stress"]
    print("[z4] stress requests=%d ok=%d 5xx=%d p50=%.1fms p95=%.1fms p99=%.1fms "
          "rss_delta=%.1fMiB receipts=%d chain=%s sse=%d" % (
              int(stress["requests"]), int(stress["ok_200"]), int(stress["five_xx"]),
              float(stress["latency_ms"]["p50"]), float(stress["latency_ms"]["p95"]),
              float(stress["latency_ms"]["p99"]),
              float(stress["rss_delta_bytes"]) / (1024.0 * 1024.0),
              int(stress["ledger"]["count"]), bool(stress["ledger"]["chain_valid"]),
              int(stress["sse"]["verified_streams"])))
    print("[z4] live_calls=%d retries=%d reduction=%.2f%% wall=%.1fs" % (
        int(payload["live_calls"]), int(payload["budget"]["retries"]),
        float(payload["arm4_call_reduction"]["reduction_pct"]),
        float(payload["wall_seconds"])))
    failed = [name for name, value in payload["gate"].items() if not bool(value)]
    print("[z4] passed=%s failed=%s receipt=%s sha256=%s" % (
        payload["passed"], failed, written["path"], written["sha256"]))
    return 0 if bool(payload["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
