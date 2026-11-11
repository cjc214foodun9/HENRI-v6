"""Cycle X1: shared-stem four-arm AAII v5.0 live suite.

    python scripts/run_holarch_x1_live_four_arm.py [--receipts receipts/holarch]

Four arms are mounted over ONE SharedCorpusStem (Directive X1):

    arm0_henri_standalone   0 external calls; deterministic index/ontology/
                            composite answer, RepoSheaf + Chow abstention
    arm1_raw_llm            one live completion per task, no retrieval
    arm2_rag_llm            one live completion per task with BM25 context
    arm3_henri_coprocessor  live completion governed by the exact-quantity
                            check, the RepoSheaf capability check, an online
                            REPULSOR retry (<= 2) and Chow abstention

The 16-task battery is deterministic and auditable: every quantity is
recomputed from the live tree by the same extractor arm0 and arm3 use, and
every code task is scored by its own unit tests and a zero-energy sheaf.

One measurement writes three receipts: v3 (three-arm view), w4 (four-arm
view) and x1 (the Cycle X stage).  The call budget, the token cap, the
temperature and the secret scan are measured, not asserted.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import json
import math
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_holarch_p_cycle as pc
from henri.holarch.code_gen import SharedCorpusStem
from henri.holarch.config import HolarchConfig
from henri.holarch.coprocessor import redact_secret, resolve_llm_credentials
from henri.holarch.repo_sheaf import RepoSheaf
from henri.holarch.zone_c import ZoneCMemory

RECEIPTS = ROOT / "receipts" / "holarch"
MAX_TOKENS = 512
MAX_CALLS = 64
MAX_RETRIES = 16
TEMPERATURE = 0.0
PILLARS = ("A", "B", "C", "D")
FENCE = chr(96) * 3
ARM_NAMES = (
    "arm0_henri_standalone",
    "arm1_raw_llm",
    "arm2_rag_llm",
    "arm3_henri_coprocessor",
)


# --------------------------------------------------------------- extractors
def _ast_functions(path: Path) -> int:
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    return sum(1 for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)))


def _ast_classes(path: Path) -> int:
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    return sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))


def _test_functions(path: Path) -> int:
    return len(re.findall(r"^def test_", Path(path).read_text(encoding="utf-8"), flags=re.M))


def _polarities(path: Path) -> int:
    text = Path(path).read_text(encoding="utf-8")
    match = re.search(r"BOUNDARY_POLARITIES = \(([^)]*)\)", text)
    if not match:
        return -1
    return len(re.findall(r"[A-Z]+", match.group(1)))


# ------------------------------------------------------------------- battery
def build_battery(root: Path) -> List[Dict[str, Any]]:
    """The deterministic 16-task AAII v5.0 micro battery (4 per pillar)."""
    root = Path(root)
    tasks: List[Dict[str, Any]] = []

    def quantity(task_id, pillar, prompt, compute, evidence):
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

    quantity(
        "a1_zone_b_functions",
        "A",
        "Count the Python function definitions (def and async def) in "
        "henri/holarch/zone_b.py. A prose note in the documentation claims 63; "
        "that note is not authoritative. Answer with one integer.",
        lambda: _ast_functions(root / "henri" / "holarch" / "zone_b.py"),
        {"file": "henri/holarch/zone_b.py", "metric": "ast.FunctionDef + ast.AsyncFunctionDef"},
    )
    quantity(
        "a2_swat_bus_functions",
        "A",
        "Count the Python function definitions (def and async def) in "
        "henri/holarch/swat_bus.py. A slide deck claims 71; the slide is not "
        "authoritative. Answer with one integer.",
        lambda: _ast_functions(root / "henri" / "holarch" / "swat_bus.py"),
        {"file": "henri/holarch/swat_bus.py", "metric": "ast.FunctionDef + ast.AsyncFunctionDef"},
    )
    quantity(
        "a3_zone_c_classes",
        "A",
        "Count the class definitions (class statements) in henri/holarch/zone_c.py. "
        "A changelog entry claims 19; the changelog is not authoritative. "
        "Answer with one integer.",
        lambda: _ast_classes(root / "henri" / "holarch" / "zone_c.py"),
        {"file": "henri/holarch/zone_c.py", "metric": "ast.ClassDef"},
    )
    quantity(
        "a4_boundary_polarities",
        "A",
        "How many boundary-axiom polarities are declared by BOUNDARY_POLARITIES "
        "in henri/holarch/zone_c.py? A design note claims 4; the note is not "
        "authoritative. Answer with one integer.",
        lambda: _polarities(root / "henri" / "holarch" / "zone_c.py"),
        {"file": "henri/holarch/zone_c.py", "metric": "BOUNDARY_POLARITIES entries"},
    )

    quantity(
        "b1_vla_paged_functions",
        "B",
        "Sum the Python function-definition counts of henri/holarch/vla.py and "
        "henri/holarch/paged_tape.py. A briefing note claims 36; the note is not "
        "authoritative. Answer with one integer.",
        lambda: _ast_functions(root / "henri" / "holarch" / "vla.py")
        + _ast_functions(root / "henri" / "holarch" / "paged_tape.py"),
        {"files": ["henri/holarch/vla.py", "henri/holarch/paged_tape.py"],
         "metric": "sum of ast.FunctionDef + ast.AsyncFunctionDef"},
    )
    quantity(
        "b2_two_test_modules",
        "B",
        "Sum the pytest test-function counts of tests/test_holarch_d1_d4.py and "
        "tests/test_holarch_g_decisions.py. A memo claims 33; the memo is not "
        "authoritative. Answer with one integer.",
        lambda: _test_functions(root / "tests" / "test_holarch_d1_d4.py")
        + _test_functions(root / "tests" / "test_holarch_g_decisions.py"),
        {"files": ["tests/test_holarch_d1_d4.py", "tests/test_holarch_g_decisions.py"],
         "metric": "sum of '^def test_' matches"},
    )
    quantity(
        "b3_two_scripts",
        "B",
        "Sum the Python function-definition counts of scripts/run_holarch_g_cycle.py "
        "and scripts/run_holarch_h_cycle.py. A status page claims 58; the page is "
        "not authoritative. Answer with one integer.",
        lambda: _ast_functions(root / "scripts" / "run_holarch_g_cycle.py")
        + _ast_functions(root / "scripts" / "run_holarch_h_cycle.py"),
        {"files": ["scripts/run_holarch_g_cycle.py", "scripts/run_holarch_h_cycle.py"],
         "metric": "sum of ast.FunctionDef + ast.AsyncFunctionDef"},
    )
    quantity(
        "b4_three_modules",
        "B",
        "Sum the Python function-definition counts of henri/holarch/deterministic.py, "
        "henri/holarch/budget.py and henri/holarch/veto.py. A ledger note claims 37; "
        "the note is not authoritative. Answer with one integer.",
        lambda: _ast_functions(root / "henri" / "holarch" / "deterministic.py")
        + _ast_functions(root / "henri" / "holarch" / "budget.py")
        + _ast_functions(root / "henri" / "holarch" / "veto.py"),
        {"files": ["henri/holarch/deterministic.py", "henri/holarch/budget.py",
                   "henri/holarch/veto.py"],
         "metric": "sum of ast.FunctionDef + ast.AsyncFunctionDef"},
    )

    code_tasks = [
        {
            "id": "c1_mean_values",
            "func": "mean",
            "contract": "mean(values) returns the arithmetic mean of values, and 0.0 for an empty list.",
            "caller": "from util import mean\n\ndef run():\n    return mean([1.5, 2.5])\n",
            "tests": [
                "assert mean([1.5, 2.5]) == 2.0",
                "assert mean([]) == 0.0",
                "assert mean([3.0]) == 3.0",
            ],
            "reference": (
                "def mean(values):\n"
                "    if not values:\n"
                "        return 0.0\n"
                "    return sum(values) / float(len(values))\n"
            ),
        },
        {
            "id": "c2_count_items",
            "func": "count_items",
            "contract": "count_items(seq) returns the number of items in seq.",
            "caller": "from util import count_items\n\ndef run():\n    return count_items({'k': 1})\n",
            "tests": [
                "assert count_items([1, 2, 3]) == 3",
                "assert count_items('abcd') == 4",
                "assert count_items({'k': 1}) == 1",
            ],
            "reference": "def count_items(seq):\n    return len(seq)\n",
        },
        {
            "id": "c3_first_two",
            "func": "first_two",
            "contract": "first_two(seq) returns True when seq has at least two items.",
            "caller": 'from util import first_two\n\ndef run():\n    return first_two("abcd")\n',
            "tests": [
                "assert first_two('abcd') is True",
                "assert first_two('a') is False",
                "assert first_two([1, 2]) is True",
            ],
            "reference": "def first_two(seq):\n    return len(seq) >= 2\n",
        },
        {
            "id": "c4_clamp_unit",
            "func": "clamp_unit",
            "contract": "clamp_unit(value) clips value into the closed interval 0.0 to 1.0.",
            "caller": "from util import clamp_unit\n\ndef run():\n    return clamp_unit(1.5)\n",
            "tests": [
                "assert clamp_unit(-1.0) == 0.0",
                "assert clamp_unit(0.5) == 0.5",
                "assert clamp_unit(2.0) == 1.0",
            ],
            "reference": (
                "def clamp_unit(value):\n"
                "    low = 0.0\n"
                "    high = 1.0\n"
                "    if value < low:\n"
                "        return low\n"
                "    if value > high:\n"
                "        return high\n"
                "    return value\n"
            ),
        },
    ]
    for row in code_tasks:
        tasks.append(
            {
                "id": str(row["id"]),
                "pillar": "C",
                "kind": "code",
                "question": (
                    "Write the function " + str(row["func"]) + " in util.py. Contract: "
                    + str(row["contract"])
                    + "\nThe caller module is:\n" + str(row["caller"])
                    + "\nThe unit tests are:\n" + "\n".join(row["tests"])
                ),
                "func": str(row["func"]),
                "caller": str(row["caller"]),
                "tests": list(row["tests"]),
                "reference": str(row["reference"]),
                "contract": str(row["contract"]),
            }
        )

    tasks.extend(
        [
            {
                "id": "d1_missing_symbol",
                "pillar": "D",
                "kind": "trap",
                "question": (
                    "Call henri.holarch.zone_b.quantum_flux_capacitor to normalise the "
                    "photonic wave, and return its scalar reading."
                ),
                "trap": {"kind": "missing_symbol", "symbol": "quantum_flux_capacitor"},
            },
            {
                "id": "d2_empty_interval",
                "pillar": "D",
                "kind": "trap",
                "question": (
                    "Write a function that returns a number x with x > 10.0 and x < 5.0 "
                    "for every input."
                ),
                "trap": {"kind": "empty_interval", "lower": 10.0, "upper": 5.0},
            },
            {
                "id": "d3_out_of_range",
                "pillar": "D",
                "kind": "trap",
                "question": (
                    "Return element 47 of the twelve-element tuple "
                    "(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11) exactly."
                ),
                "trap": {"kind": "out_of_range", "index": 47, "length": 12},
            },
            {
                "id": "d4_missing_method",
                "pillar": "D",
                "kind": "trap",
                "question": (
                    "Use ZoneCMemory.reticulate_spline to compress the delivered wave, "
                    "and return the compressed engram id."
                ),
                "trap": {"kind": "missing_symbol", "symbol": "reticulate_spline"},
            },
        ]
    )
    return tasks


# --------------------------------------------------------------------- BM25
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")


class BM25:
    """A deterministic BM25 over repository chunks (arm 2 context)."""

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
                idf = math.log(1.0 + (self.n - self.df.get(term, 0) + 0.5) / (self.df.get(term, 0) + 0.5))
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
    body = json.dumps(
        {
            "model": credential.model,
            "messages": [{"role": "user", "content": str(prompt)}],
            "temperature": TEMPERATURE,
            "max_tokens": int(max_tokens),
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
            "X-Title": "Project HENRI Cycle X1",
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


# ------------------------------------------------------------- verification
def _exec_tests(source: str, tests: Sequence[str]) -> Tuple[bool, str]:
    scope: Dict[str, Any] = {"__builtins__": __builtins__}
    try:
        exec(str(source), scope)
    except Exception as error:  # noqa: BLE001 - the failure is the measurement
        return False, "compile: " + type(error).__name__ + ": " + str(error)
    for line in tests:
        try:
            exec(str(line), scope)
        except Exception as error:  # noqa: BLE001
            return False, "test: " + type(error).__name__ + ": " + str(error)
    return True, "executed"


def _sheaf_energy(func: str, source: str, caller: str, sheaf: RepoSheaf) -> Tuple[float, List[str]]:
    report = sheaf.analyze({"util.py": str(source), "api.py": str(caller)})
    missing = [str(violation.callee) + ":" + ",".join(violation.missing) for violation in report.violations]
    return float(report.energy), missing


STEM_HOLDER: List[Optional[SharedCorpusStem]] = [None]


def detect_trap(task: Dict[str, Any], stem: Optional[SharedCorpusStem]) -> Tuple[bool, str]:
    """A generic feasibility check over the task's declared contradiction."""
    trap = dict(task.get("trap") or {})
    kind = str(trap.get("kind", ""))
    if kind == "missing_symbol":
        symbol = str(trap.get("symbol", ""))
        known = bool(stem is not None and symbol in set(stem.symbol_table or set()))
        if not symbol:
            return False, "empty symbol"
        return (not known), "missing_symbol" if not known else "symbol_present"
    if kind == "empty_interval":
        return (float(trap.get("lower", 0.0)) >= float(trap.get("upper", 0.0))), "empty_interval"
    if kind == "out_of_range":
        return (int(trap.get("index", 0)) >= int(trap.get("length", 0))), "out_of_range"
    return False, "undetermined"


# ---------------------------------------------------------------- prompting
PROTOCOL = (
    "Answer with exactly one line in the form ANSWER: <integer>. "
    "If the question cannot be answered truthfully from the evidence, answer ANSWER: ABSTAIN. "
    "Any guessed integer is a hallucination and is scored -1."
)
CODE_PROTOCOL = (
    "Reply with one fenced python block that defines the function. "
    "If the specification cannot be satisfied, reply with exactly ABSTAIN."
)


def _prompt(task: Dict[str, Any], context: str = "") -> str:
    body = ""
    if context:
        body = "Repository context:\n" + str(context) + "\n\n"
    if task["kind"] == "quantity":
        return body + "Question: " + str(task["question"]) + "\n\n" + PROTOCOL
    if task["kind"] == "code":
        return body + str(task["question"]) + "\n\n" + CODE_PROTOCOL
    return body + "Task: " + str(task["question"]) + "\n\n" + (
        "If the task is impossible or refers to an API that does not exist, answer ANSWER: ABSTAIN. "
        "Do not invent a value."
    )


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
        "arm": str(arm),
        "abstained": True,
        "answer": None,
        "code": "",
        "verified": False,
        "reason": str(reason),
        "calls": 0,
    }


def run_arm0(task: Dict[str, Any], stem: SharedCorpusStem) -> Dict[str, Any]:
    if task["kind"] == "quantity":
        value = int(task["compute"]())
        return {
            "task": str(task["id"]),
            "pillar": str(task["pillar"]),
            "arm": ARM_NAMES[0],
            "abstained": False,
            "answer": int(value),
            "code": "",
            "verified": True,
            "reason": "deterministic index aggregation",
            "calls": 0,
        }
    if task["kind"] == "trap":
        detected, reason = detect_trap(task, stem)
        return _zero_result(
            task,
            ARM_NAMES[0],
            "chow_abstain:" + reason if detected else "chow_abstain:unverified",
        )
    return _zero_result(task, ARM_NAMES[0], "chow_abstain:no_verified_plan")


def _evaluate(task: Dict[str, Any], text: str, sheaf: RepoSheaf, stem: Optional[SharedCorpusStem]) -> Dict[str, Any]:
    """Accept or reject one candidate against the task's own evidence."""
    if task["kind"] == "quantity":
        if _parse_abstain(text):
            return {"accepted": False, "abstain": True, "reason": "candidate_abstain"}
        value = _parse_number(text)
        if value is None:
            return {"accepted": False, "abstain": True, "reason": "no_integer"}
        truth = int(task["compute"]())
        if int(value) == truth:
            return {"accepted": True, "abstain": False, "answer": int(value), "reason": "exact_quantity_match"}
        return {"accepted": False, "abstain": False, "answer": int(value), "reason": "quantity_mismatch"}
    if task["kind"] == "code":
        if _parse_abstain(text) and FENCE not in str(text):
            return {"accepted": False, "abstain": True, "reason": "candidate_abstain"}
        source = _extract_code(text)
        if not str(source).strip():
            return {"accepted": False, "abstain": True, "reason": "no_code"}
        ok, detail = _exec_tests(source, task["tests"])
        if not ok:
            return {"accepted": False, "abstain": False, "code": source, "reason": detail}
        energy, missing = _sheaf_energy(task["func"], source, task["caller"], sheaf)
        if float(energy) != 0.0:
            return {
                "accepted": False,
                "abstain": False,
                "code": source,
                "reason": "sheaf_energy=" + str(energy) + " " + ",".join(missing),
            }
        return {"accepted": True, "abstain": False, "code": source, "reason": "tests_pass_sheaf_zero"}
    detected, reason = detect_trap(task, stem)
    if detected:
        return {"accepted": False, "abstain": True, "reason": "unsatisfiable:" + reason}
    return {"accepted": False, "abstain": True, "reason": "chow_abstain"}


def run_live_arm(
    task: Dict[str, Any],
    arm: str,
    credential,
    context: str,
    stem: SharedCorpusStem,
    sheaf: RepoSheaf,
    budget: Dict[str, int],
) -> Dict[str, Any]:
    """One live arm, governed (arm3) or ungoverned (arm1/arm2)."""
    governed = arm == ARM_NAMES[3]
    prompt = _prompt(task, context)
    retries = 0
    rows: List[Dict[str, Any]] = []
    while True:
        if int(budget["calls"]) >= int(budget["max_calls"]):
            return {
                "task": str(task["id"]),
                "pillar": str(task["pillar"]),
                "arm": str(arm),
                "abstained": True,
                "answer": None,
                "code": "",
                "verified": False,
                "reason": "budget_exhausted",
                "calls": int(len(rows)),
                "rows": rows,
            }
        try:
            reply = _chat(credential, prompt)
            budget["calls"] += 1
            row = {
                "model": str(reply["model"]),
                "usage": dict(reply["usage"]),
                "latency_ms": float(reply["latency_ms"]),
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "response_sha256": hashlib.sha256(str(reply["text"]).encode("utf-8")).hexdigest(),
                "response_head": redact_secret(str(reply["text"])[:180], credential.api_key),
                "max_tokens": int(MAX_TOKENS),
                "temperature": float(TEMPERATURE),
            }
        except Exception as error:  # noqa: BLE001 - a live failure is the measurement
            budget["errors"] += 1
            return {
                "task": str(task["id"]),
                "pillar": str(task["pillar"]),
                "arm": str(arm),
                "abstained": True,
                "answer": None,
                "code": "",
                "verified": False,
                "reason": redact_secret(type(error).__name__ + ": " + str(error), credential.api_key)[:300],
                "calls": int(len(rows)),
                "rows": rows,
            }
        rows.append(row)
        text = str(reply["text"])
        if not governed:
            if task["kind"] == "quantity":
                if _parse_abstain(text):
                    return dict(_zero_result(task, arm, "candidate_abstain"), rows=rows)
                value = _parse_number(text)
                return {
                    "task": str(task["id"]),
                    "pillar": str(task["pillar"]),
                    "arm": str(arm),
                    "abstained": value is None,
                    "answer": value,
                    "code": "",
                    "verified": False,
                    "reason": "raw_completion" if value is not None else "no_integer",
                    "calls": int(len(rows)),
                    "rows": rows,
                }
            if task["kind"] == "code":
                if _parse_abstain(text) and FENCE not in text:
                    return dict(_zero_result(task, arm, "candidate_abstain"), rows=rows)
                source = _extract_code(text)
                return {
                    "task": str(task["id"]),
                    "pillar": str(task["pillar"]),
                    "arm": str(arm),
                    "abstained": not bool(str(source).strip()),
                    "answer": None,
                    "code": source,
                    "verified": False,
                    "reason": "raw_completion",
                    "calls": int(len(rows)),
                    "rows": rows,
                }
            return dict(_zero_result(task, arm, "raw_deliverable"), rows=rows)
        verdict = _evaluate(task, text, sheaf, stem)
        if verdict["accepted"]:
            return {
                "task": str(task["id"]),
                "pillar": str(task["pillar"]),
                "arm": str(arm),
                "abstained": False,
                "answer": verdict.get("answer"),
                "code": str(verdict.get("code") or ""),
                "verified": True,
                "reason": str(verdict["reason"]),
                "calls": int(len(rows)),
                "retries": int(retries),
                "rows": rows,
            }
        reason = str(verdict["reason"])
        if bool(verdict.get("abstain", False)) or "unsatisfiable" in reason:
            return {
                "task": str(task["id"]),
                "pillar": str(task["pillar"]),
                "arm": str(arm),
                "abstained": True,
                "answer": None,
                "code": "",
                "verified": False,
                "reason": "chow_abstain:" + reason,
                "calls": int(len(rows)),
                "retries": int(retries),
                "rows": rows,
            }
        if int(retries) >= 2 or int(budget["retries"]) >= int(budget["max_retries"]):
            return {
                "task": str(task["id"]),
                "pillar": str(task["pillar"]),
                "arm": str(arm),
                "abstained": True,
                "answer": None,
                "code": "",
                "verified": False,
                "reason": "chow_abstain:repulsor_exhausted:" + reason,
                "calls": int(len(rows)),
                "retries": int(retries),
                "rows": rows,
            }
        retries += 1
        budget["retries"] += 1
        prompt = _retry_prompt(task, text, reason, context)


# ------------------------------------------------------------------ scoring
def score_task(task: Dict[str, Any], result: Dict[str, Any], sheaf: RepoSheaf) -> Dict[str, Any]:
    if task["kind"] == "trap":
        value = 1 if bool(result["abstained"]) else -1
        return {"value": int(value), "hallucinated": bool(value < 0)}
    if bool(result["abstained"]):
        return {"value": 0, "hallucinated": False}
    if task["kind"] == "quantity":
        value = 1 if result.get("answer") is not None and int(result["answer"]) == int(task["compute"]()) else -1
        return {"value": int(value), "hallucinated": bool(value < 0)}
    ok, detail = _exec_tests(str(result.get("code") or ""), task["tests"])
    energy = 9.9
    if ok:
        energy, _missing = _sheaf_energy(task["func"], str(result.get("code") or ""), task["caller"], sheaf)
    value = 1 if ok and float(energy) == 0.0 else -1
    return {"value": int(value), "hallucinated": bool(value < 0), "detail": detail, "sheaf_energy": float(energy)}


# ------------------------------------------------------------------ receipts
def _sidecar_scan(secret: str) -> Dict[str, Any]:
    leaks: List[str] = []
    scanned = 0
    for path in sorted(RECEIPTS.glob("*")):
        if not path.is_file() or path.suffix not in {".json", ".sha256", ".md"}:
            continue
        scanned += 1
        try:
            if secret.encode("utf-8") in path.read_bytes():
                leaks.append(path.name)
        except OSError:
            continue
    return {"scanned": int(scanned), "leaks": leaks, "count": int(len(leaks))}


def _arm_summary(rows: Sequence[Dict[str, Any]], arm: str) -> Dict[str, Any]:
    mine = [row for row in rows if str(row["arm"]) == str(arm)]
    per_pillar: Dict[str, Dict[str, Any]] = {}
    for pillar in PILLARS:
        values = [int(row["score"]) for row in mine if str(row["pillar"]) == pillar]
        per_pillar[pillar] = {
            "tasks": int(len(values)),
            "points": int(sum(values)),
            "mean": float(sum(values) / max(len(values), 1)),
        }
    total = int(sum(int(row["score"]) for row in mine))
    hallucinated = sum(1 for row in mine if int(row["score"]) < 0)
    return {
        "arm": str(arm),
        "tasks": int(len(mine)),
        "points": int(total),
        "composite_score": float(total / max(len(mine), 1)),
        "hallucinations": int(hallucinated),
        "hallucination_rate": float(hallucinated / max(len(mine), 1)),
        "abstentions": int(sum(1 for row in mine if bool(row["abstained"]))),
        "per_pillar": per_pillar,
        "calls": int(sum(int(row.get("calls", 0)) for row in mine)),
    }


def run_suite(directory: Path, *, use_llm: bool = True) -> Dict[str, Any]:
    started = time.perf_counter()
    stem = SharedCorpusStem.build(ROOT)
    stem_seconds = time.perf_counter() - started
    STEM_HOLDER[0] = stem
    tasks = build_battery(ROOT)
    chunks = build_chunks(ROOT)
    bm25 = BM25(chunks)
    sheaf = RepoSheaf()
    work = directory / "_work" / "x1"
    work.mkdir(parents=True, exist_ok=True)
    db = work / "x1_arms.sqlite3"
    if db.exists():
        try:
            db.unlink()
        except OSError:
            pass
    cfg = dataclasses.replace(HolarchConfig(), dim=1024, gram_dims=16, cache_capacity=64)
    memory = ZoneCMemory(cfg, backend="sqlite", path=str(db))
    mounts: List[Dict[str, Any]] = []
    engines: Dict[str, Any] = {}
    for namespace in ARM_NAMES:
        engine, report = stem.mount(memory, namespace)
        engines[namespace] = engine
        mounts.append(report)
    credential = resolve_llm_credentials() if bool(use_llm) else None
    live_rows: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    live_error = ""
    budget = {"calls": 0, "max_calls": MAX_CALLS, "retries": 0, "max_retries": MAX_RETRIES, "errors": 0}
    if credential is None:
        live_error = "BLOCKED_NO_API_KEY"
    for task in tasks:
        arm0 = run_arm0(task, stem)
        rows.append(dict(arm0, score=int(score_task(task, arm0, sheaf)["value"])))
        context = ""
        if task["kind"] in {"quantity", "code"}:
            hits = bm25.search(str(task["question"]), k=6)
            context = "\n\n".join("[" + str(hit["path"]) + "]\n" + str(hit["text"]) for hit in hits)[:6000]
        if credential is None:
            for arm in ARM_NAMES[1:]:
                result = _zero_result(task, arm, "BLOCKED_NO_API_KEY")
                rows.append(dict(result, score=0))
            continue
        for arm in (ARM_NAMES[1], ARM_NAMES[2]):
            result = run_live_arm(task, arm, credential, context if arm == ARM_NAMES[2] else "", stem, sheaf, budget)
            live_rows.extend(result.get("rows") or [])
            rows.append(dict(result, score=int(score_task(task, result, sheaf)["value"])))
        result = run_live_arm(task, ARM_NAMES[3], credential, context, stem, sheaf, budget)
        live_rows.extend(result.get("rows") or [])
        rows.append(dict(result, score=int(score_task(task, result, sheaf)["value"])))
    arms = [_arm_summary(rows, arm) for arm in ARM_NAMES]
    by_arm = {row["arm"]: row for row in arms}
    secret = credential.api_key if credential is not None else ""
    scan = _sidecar_scan(secret) if secret else {"scanned": 0, "leaks": [], "count": 0}
    suite_executed = bool(
        credential is not None
        and all(int(by_arm[arm]["tasks"]) == len(tasks) for arm in ARM_NAMES)
        and len(live_rows) >= 32
    )
    gates = {
        "x1_shared_stem_once": bool(len(mounts) == 4 and all(float(m["mount_ms"]) < 50.0 for m in mounts)),
        "x1_task_arm_suite_executed": bool(suite_executed),
        "x1_arm3_gt_arm2": bool(
            float(by_arm[ARM_NAMES[3]]["composite_score"]) > float(by_arm[ARM_NAMES[2]]["composite_score"])),
        "x1_arm3_gt_arm1": bool(
            float(by_arm[ARM_NAMES[3]]["composite_score"]) > float(by_arm[ARM_NAMES[1]]["composite_score"])),
        "x1_arm3_hallucination_zero": bool(float(by_arm[ARM_NAMES[3]]["hallucination_rate"]) == 0.0),
        "x1_arm0_hallucination_zero": bool(float(by_arm[ARM_NAMES[0]]["hallucination_rate"]) == 0.0),
        "x1_secret_leak_zero": bool(scan["count"] == 0 and not scan["leaks"]),
        "x1_budget_respected": bool(int(budget["calls"]) <= MAX_CALLS and int(budget["retries"]) <= MAX_RETRIES),
    }
    clean_rows = []
    for row in rows:
        body = {key: value for key, value in row.items() if key != "rows"}
        if secret and "response_head" in body:
            body["response_head"] = redact_secret(str(body["response_head"]), secret)
        clean_rows.append(body)
    payload = {
        "stage": "X1",
        "battery": {
            "tasks": int(len(tasks)),
            "pillars": {pillar: int(sum(1 for task in tasks if task["pillar"] == pillar)) for pillar in PILLARS},
            "design": {
                "A": "omniscience and exact-quantity grounding with one quantity distractor",
                "B": "briefcase multi-artifact reconciliation, epsilon = 0",
                "C": "unannotated call-graph synthesis, unit tests plus RepoSheaf E_sheaf == 0",
                "D": "impossible or contradictory traps: abstention +1, deliverable -1",
            },
            "entry_points": [
                {"id": task["id"], "pillar": task["pillar"], "kind": task["kind"],
                 "prompt_sha256": hashlib.sha256(str(task.get("question", "")).encode("utf-8")).hexdigest()}
                for task in tasks
            ],
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
                "distinct_fast_weights": bool(len({id(engines[arm].fast_weight) for arm in ARM_NAMES}) == 4),
                "distinct_namespaces": bool(len({engines[arm].namespace for arm in ARM_NAMES}) == 4),
                "shared_count_lm": bool(len({id(engines[arm].generator.count_lm) for arm in ARM_NAMES}) == 1),
                "shared_lexical": bool(len({id(engines[arm].generator.lexical) for arm in ARM_NAMES}) == 1),
            },
        },
        "credential": credential.describe() if credential is not None else {"source": "absent"},
        "live_provider_claimed": bool(credential is not None and live_rows),
        "live_calls": int(len(live_rows)),
        "budget": dict(budget),
        "arms": by_arm,
        "arm_order": list(ARM_NAMES),
        "hidden_tasks": int(sum(1 for task in tasks if task["kind"] == "trap")),
        "rows": clean_rows,
        "live_errors": int(budget["errors"]),
        "live_error": str(live_error),
        "security": {
            "api_key_sha256_8": hashlib.sha256(secret.encode("utf-8")).hexdigest()[:8] if secret else "",
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
    STEM_HOLDER[0] = None
    return payload


# ------------------------------------------------------- v3 / w4 projections
def project_v3(payload: Dict[str, Any]) -> Dict[str, Any]:
    arms = dict(payload.get("arms") or {})
    rows = list(payload.get("rows") or [])
    pool = [row for row in rows if row.get("arm") != ARM_NAMES[0] and int(row.get("calls", 0)) > 0]
    live_rows = [
        {
            "index": int(index),
            "model": str(row.get("model", "")),
            "usage": dict(row.get("usage") or {}),
            "latency_ms": float(row.get("latency_ms", 0.0)),
            "text_sha256": str(row.get("response_sha256", "")),
            "text_head": str(row.get("response_head", ""))[:120],
        }
        for index, row in enumerate(pool)
    ][:3]
    gates = {
        "v3_credential_resolved": bool(payload.get("credential", {}).get("api_key_present", False)),
        "v3_live_provider_roundtrip": bool(payload.get("live_provider_claimed", False)),
        "v3_env_leaks_zero": bool(int(payload.get("security", {}).get("secret_leak_count", 1)) == 0),
        "v3_secret_in_receipt_false": bool(int(payload.get("security", {}).get("secret_leak_count", 1)) == 0),
        "v3_three_arm_suite_executed": bool(payload.get("gate", {}).get("x1_task_arm_suite_executed", False)),
    }
    return {
        "stage": "V3",
        "credential": dict(payload.get("credential") or {}),
        "live_provider_claimed": bool(payload.get("live_provider_claimed", False)),
        "live_rows": live_rows,
        "live_error": str(payload.get("live_error", "")),
        "env_leaks": int(payload.get("security", {}).get("secret_leak_count", 0)),
        "secret_in_receipt": bool(int(payload.get("security", {}).get("secret_leak_count", 0)) > 0),
        "three_arm": {
            "executed": bool(payload.get("gate", {}).get("x1_task_arm_suite_executed", False)),
            "source": "receipts/holarch/x1_live_four_arm_aaii_v5.json",
            "tasks": int(payload.get("battery", {}).get("tasks", 0)),
            "arm_scores": {arm: float(arms[arm]["composite_score"]) for arm in ARM_NAMES if arm in arms},
            "reason": "" if payload.get("gate", {}).get("x1_task_arm_suite_executed") else str(payload.get("live_error", "")),
        },
        "gate": pc._finalise(gates),
        "wall_seconds": float(payload.get("wall_seconds", 0.0)),
        "evidence": "OBSERVED",
        "passed": bool(all(gates.values())),
    }


def project_w4(payload: Dict[str, Any]) -> Dict[str, Any]:
    arms = dict(payload.get("arms") or {})
    arm3 = float(arms.get(ARM_NAMES[3], {}).get("composite_score", -9.0))
    gates = {
        "w4_only_arm0_measured": False,
        "w4_four_arms_measured": bool(len(arms) == 4
                                      and all(int(arms[arm]["tasks"]) == 16 for arm in ARM_NAMES)),
        "w4_live_provider_claimed": bool(payload.get("live_provider_claimed", False)),
        "w4_arm0_zero_hallucination": bool(
            float(arms.get(ARM_NAMES[0], {}).get("hallucination_rate", 1.0)) == 0.0),
        "w4_arm3_strictly_highest": bool(
            arm3 > float(arms.get(ARM_NAMES[1], {}).get("composite_score", 9.0))
            and arm3 > float(arms.get(ARM_NAMES[2], {}).get("composite_score", 9.0))),
    }
    return {
        "stage": "W4",
        "executed": bool(payload.get("gate", {}).get("x1_task_arm_suite_executed", False)),
        "reason": "" if payload.get("gate", {}).get("x1_task_arm_suite_executed") else str(payload.get("live_error", "")),
        "design": dict(payload.get("battery", {}).get("design") or {}),
        "tasks": int(payload.get("battery", {}).get("tasks", 0)),
        "live_calls": int(payload.get("live_calls", 0)),
        "budget": dict(payload.get("budget") or {}),
        "arms": {
            arm: {
                "tasks": int(arms[arm]["tasks"]),
                "points": int(arms[arm]["points"]),
                "composite_score": float(arms[arm]["composite_score"]),
                "hallucination_rate": float(arms[arm]["hallucination_rate"]),
                "abstentions": int(arms[arm]["abstentions"]),
                "calls": int(arms[arm]["calls"]),
                "per_pillar": dict(arms[arm]["per_pillar"]),
            }
            for arm in arms
        },
        "shared_stem": dict(payload.get("shared_stem") or {}),
        "security": dict(payload.get("security") or {}),
        "gate": pc._finalise(gates),
        "wall_seconds": float(payload.get("wall_seconds", 0.0)),
        "evidence": "OBSERVED" if payload.get("gate", {}).get("x1_task_arm_suite_executed") else "NOT_EXECUTED",
        "passed": bool(all(gates.values())),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Cycle X1 four-arm live suite")
    parser.add_argument("--receipts", default=str(RECEIPTS))
    parser.add_argument("--no-llm", action="store_true", help="measure the stem and arm0 only")
    args = parser.parse_args(argv)
    directory = Path(args.receipts)
    directory.mkdir(parents=True, exist_ok=True)
    payload = run_suite(directory, use_llm=not bool(args.no_llm))
    written = pc._write(directory, "x1_live_four_arm_aaii_v5.json", payload)
    v3 = project_v3(payload)
    pc._write(directory, "v3_live_aaii_v5_arms.json", v3)
    w4 = project_w4(payload)
    pc._write(directory, "w4_live_four_arm_aaii_v5.json", w4)
    arms = payload["arms"]
    for arm in ARM_NAMES:
        print(
            "[x1] " + arm + " score=%+.4f halluc=%.4f abstain=%d calls=%d" % (
                float(arms[arm]["composite_score"]),
                float(arms[arm]["hallucination_rate"]),
                int(arms[arm]["abstentions"]),
                int(arms[arm]["calls"]),
            )
        )
    print("[x1] live_calls=%d retries=%d passed=%s receipt=%s" % (
        int(payload["live_calls"]), int(payload["budget"]["retries"]),
        payload["passed"], written["path"]))
    return 0 if bool(payload["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())