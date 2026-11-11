"""Stirrup-compatible multi-turn co-processor loop (Directive T4).

The sidecar exposes one call. A benchmark agent needs a LOOP: read evidence,
run code, verify a number, then finish or abandon. This module is that loop,
with four deterministic gates on every turn:

    browse        SandboxedCorpusBrowser resolves the path before any read.
    code_exec     ASTGate grammar and symbol veto, then a restricted exec.
    repulsor      a command that already failed is refused, not retried.
    finish        refuses when required evidence was never read.
    abandon       accepts only a cited path that is missing or corrupted.

Every turn appends one SHA-256 link. The chain is contiguous by construction,
and the harness verifies it again after the run.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from henri.holarch.code_gen import ASTGate
from henri.holarch.ontology_guard import SandboxViolation, SandboxedCorpusBrowser

SAFE_MODULES = ("csv", "json", "math", "re", "hashlib", "io")
SAFE_NAMES = {
    "abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "len",
    "list", "max", "min", "range", "round", "set", "sorted", "str", "sum",
    "tuple", "zip", "Exception", "ValueError", "KeyError", "TypeError",
}
FORBIDDEN_NAMES = {
    "eval", "exec", "compile", "open", "input", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "__import__", "breakpoint", "memoryview",
}
MAX_SOURCE_CHARS = 4000


def _digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


@dataclass
class TurnReceipt:
    index: int
    action: str
    status: str
    payload_sha256: str
    parent: str
    sha256: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "index": int(self.index),
            "action": str(self.action),
            "status": str(self.status),
            "payload_sha256": str(self.payload_sha256),
            "parent": str(self.parent),
            "sha256": str(self.sha256),
        }


class HolarchCoProcessor:
    """One deterministic multi-turn agent over a sandboxed workspace."""

    def __init__(
        self,
        workspace: str | Path,
        *,
        namespace: str = "coprocessor",
        max_turns: int = 16,
        service=None,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.namespace = str(namespace)
        self.max_turns = int(max_turns)
        self.service = service
        self.browser = SandboxedCorpusBrowser(None, [self.workspace], namespace=namespace)
        self.ast_gate = ASTGate(symbols=SAFE_NAMES)
        self.receipts: List[TurnReceipt] = []
        self.evidence: Dict[str, str] = {}
        self.evidence_digest: Dict[str, str] = {}
        self.repulsors: Dict[str, str] = {}
        self.failures: List[Dict[str, str]] = []
        self.repeat_failures = 0
        self.repulsor_blocks = 0
        self.corrupted: set = set()
        self.turns: List[Dict[str, object]] = []
        self.status = "incomplete"
        self.deliverable: Optional[Any] = None
        self.abandon_reason = ""

    # ------------------------------------------------------------- receipts
    def _append(self, action: str, status: str, payload: Any, output: Any) -> None:
        index = len(self.receipts)
        body = {
            "index": int(index),
            "action": str(action),
            "status": str(status),
            "payload_sha256": _digest(payload),
            "parent": str(self.receipts[-1].sha256) if self.receipts else "genesis",
        }
        body["sha256"] = _digest(body)
        receipt = TurnReceipt(**body)
        self.receipts.append(receipt)
        self.turns.append(
            {
                "index": int(index),
                "action": str(action),
                "status": str(status),
                "output": output,
                "sha256": receipt.sha256,
                "parent": receipt.parent,
            }
        )

    def chain_valid(self) -> bool:
        if not self.receipts:
            return False
        if self.receipts[0].parent != "genesis":
            return False
        return all(
            self.receipts[index].parent == self.receipts[index - 1].sha256
            for index in range(1, len(self.receipts))
        )

    # --------------------------------------------------------------- gates
    def _symbol_veto(self, source: str) -> str:
        try:
            tree = ast.parse(source)
        except SyntaxError as error:
            return f"grammar: {error.msg}"
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                return "grammar: imports are injected, not declared"
            if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
                return f"symbol: forbidden name {node.id!r}"
            if isinstance(node, ast.Attribute):
                if str(node.attr).startswith("_"):
                    return f"symbol: private attribute {node.attr!r}"
                base = node.value
                if isinstance(base, ast.Name) and base.id not in SAFE_MODULES:
                    return f"symbol: attribute on unknown base {base.id!r}"
            if isinstance(node, ast.Call):
                callee = node.func
                if isinstance(callee, ast.Name) and (
                    callee.id in FORBIDDEN_NAMES
                    or (callee.id not in SAFE_NAMES and callee.id not in SAFE_MODULES)
                ):
                    return f"symbol: unknown callee {callee.id!r}"
        return ""

    def _numeric_guard(self, claim: str) -> str:
        numbers = re.findall(r"\d+(?:\.\d+)?", str(claim))
        corpus = "\n".join(self.evidence.values())
        for number in numbers:
            if len(number.replace(".", "")) < 2:
                continue
            if number not in corpus:
                return f"quantity: {number!r} does not appear in the read evidence"
        return ""

    def _execute(self, source: str, evidence: Dict[str, str]) -> Tuple[bool, Any, str]:
        namespace: Dict[str, Any] = {"evidence": dict(evidence)}
        import builtins as _builtins

        safe_builtins = {
            name: getattr(_builtins, name) for name in SAFE_NAMES if hasattr(_builtins, name)
        }
        namespace["__builtins__"] = safe_builtins
        for name in SAFE_MODULES:
            namespace[name] = __import__(name)
        try:
            exec(compile(str(source), "<coprocessor>", "exec"), namespace, namespace)
        except Exception as error:  # noqa: BLE001 - the failure is the measurement
            return False, None, f"{type(error).__name__}: {error}"
        return True, namespace.get("result"), ""

    # --------------------------------------------------------------- turns
    def _browse(self, spec: Dict[str, object]) -> Dict[str, object]:
        path = str(spec.get("path", "")).strip()
        try:
            resolved = self.browser.resolve(path, must_exist=True)
            text = resolved.read_text(encoding="utf-8", errors="replace")
        except SandboxViolation as error:
            return {"ok": False, "status": "refused", "path": path, "reason": str(error)}
        self.evidence[path] = text
        self.evidence_digest[path] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return {
            "ok": True,
            "status": "read",
            "path": path,
            "bytes": len(text.encode("utf-8")),
            "sha256": self.evidence_digest[path],
        }

    def _code_exec(self, spec: Dict[str, object]) -> Dict[str, object]:
        source = str(spec.get("source", ""))
        if len(source) > MAX_SOURCE_CHARS:
            return {"ok": False, "status": "refused", "reason": "source too long"}
        if self.ast_gate.grammar_veto(source):
            return {"ok": False, "status": "refused", "reason": "ASTGate grammar veto"}
        veto = self._symbol_veto(source)
        if veto:
            return {"ok": False, "status": "refused", "reason": veto}
        key = hashlib.sha256(source.encode("utf-8")).hexdigest()
        if key in self.repulsors:
            self.repulsor_blocks += 1
            return {
                "ok": False,
                "status": "repulsor",
                "reason": f"REPULSOR: this command already failed: {self.repulsors[key]}",
            }
        ok, value, error = self._execute(source, self.evidence)
        if not ok:
            self.failures.append({"action": "code_exec", "error": str(error)})
            self.repulsors[key] = str(error)
            for path in spec.get("taints") or []:
                self.corrupted.add(str(path))
            return {"ok": False, "status": "error", "reason": str(error), "repulsor": True}
        return {"ok": True, "status": "executed", "result": value}

    def _verify(self, spec: Dict[str, object]) -> Dict[str, object]:
        claim = str(spec.get("claim", ""))
        failure = self._numeric_guard(claim)
        if failure:
            return {"ok": False, "status": "refused", "reason": failure}
        return {"ok": True, "status": "supported", "claim": claim}

    def _finish(self, spec: Dict[str, object], required: Sequence[str]) -> Dict[str, object]:
        missing = [str(path) for path in required if str(path) not in self.evidence]
        if missing:
            self.repulsors[_digest({"finish": True})] = "missing evidence"
            return {
                "ok": False,
                "status": "refused",
                "reason": f"finish refused: required evidence not read: {missing}",
                "missing": missing,
            }
        self.deliverable = spec.get("payload")
        self.status = "finished"
        return {"ok": True, "status": "finished", "payload": self.deliverable}

    def _abandon(self, spec: Dict[str, object]) -> Dict[str, object]:
        citations = list(spec.get("missing") or [])
        if not citations:
            return {"ok": False, "status": "refused", "reason": "abandon needs a citation"}
        accepted: List[Dict[str, str]] = []
        for citation in citations:
            path = str(citation.get("path", ""))
            state = str(citation.get("state", "missing"))
            if state == "corrupted":
                if path in self.evidence and path in self.corrupted:
                    accepted.append({"path": path, "state": state})
                else:
                    return {
                        "ok": False,
                        "status": "refused",
                        "reason": f"{path} is not a corrupted evidence source",
                    }
            else:
                if path in self.evidence:
                    return {
                        "ok": False,
                        "status": "refused",
                        "reason": f"{path} was read; the task is not impossible",
                    }
                accepted.append({"path": path, "state": "missing"})
        self.abandon_reason = ", ".join(row["path"] for row in accepted)
        # An abandoned task emits no deliverable. A prior refused finish cannot
        # leave a payload behind: fail-closed means the object is None.
        self.deliverable = None
        self.status = "abandoned"
        return {"ok": True, "status": "abandoned", "missing": accepted, "reason": self.abandon_reason}

    # ----------------------------------------------------------------- run
    def run(self, scenario: Dict[str, object]) -> Dict[str, object]:
        self.evidence = {}
        self.evidence_digest = {}
        self.failures = []
        self.corrupted = set()
        self.status = "incomplete"
        self.deliverable = None
        self.abandon_reason = ""
        required = [str(path) for path in scenario.get("required_evidence") or []]
        start_index = len(self.receipts)
        for spec in list(scenario.get("turns") or []):
            if len(self.receipts) - start_index >= self.max_turns:
                break
            action = str(spec.get("action", ""))
            if action == "browse":
                output = self._browse(spec)
            elif action == "code_exec":
                output = self._code_exec(spec)
            elif action == "verify":
                output = self._verify(spec)
            elif action == "finish":
                output = self._finish(spec, required)
            elif action == "abandon_task_finish":
                output = self._abandon(spec)
            else:
                output = {"ok": False, "status": "refused", "reason": f"unknown action {action!r}"}
            self._append(action, str(output.get("status", "")), spec, output)
        abandoned_missing: List[Dict[str, str]] = []
        for turn in self.turns[start_index:]:
            if str(turn.get("action")) == "abandon_task_finish":
                abandoned_missing = list((turn.get("output") or {}).get("missing") or [])
        return {
            "name": str(scenario.get("name", "")),
            "status": self.status,
            "deliverable": self.deliverable,
            "abandon_reason": self.abandon_reason,
            "abandoned_missing": abandoned_missing,
            "turns": len(self.receipts) - start_index,
            "evidence": sorted(self.evidence),
            "evidence_digest": dict(self.evidence_digest),
            "repulsor_blocks": int(self.repulsor_blocks),
            "failures": list(self.failures),
            "receipts": [row.as_dict() for row in self.receipts[start_index:]],
            "chain_valid": bool(self.chain_valid()),
        }

# ---------------------------------------------------------------- scenarios
def write_workspace(workspace: str | Path) -> Path:
    """Create the deterministic repository and data-analysis workspace."""
    root = Path(workspace).resolve()
    (root / "package").mkdir(parents=True, exist_ok=True)
    (root / "sales.csv").write_text(
        "region,units,price\n"
        "north,10,2.5\n"
        "south,14,3.0\n"
        "east,7,4.0\n"
        "west,9,1.5\n"
        "north,12,2.0\n"
        "south,11,3.5\n",
        encoding="utf-8",
    )
    (root / "inventory.csv").write_text(
        "item,qty\nbolt,120\nnut,340\nwasher,75\ngear,210\nspring,95\n",
        encoding="utf-8",
    )
    (root / "config.json").write_text(
        json.dumps({"threshold": 12, "retries": 3, "mode": "fast"}, indent=2),
        encoding="utf-8",
    )
    (root / "notes.md").write_text(
        "# Field notes\n\nalpha = 17\nbeta = 23\ngamma = 41\n", encoding="utf-8"
    )
    (root / "metrics.json").write_text(
        json.dumps({"latency_ms": [10, 12, 14], "errors": 2}, indent=2), encoding="utf-8"
    )
    (root / "summary.json").write_text(json.dumps({"units": 63}), encoding="utf-8")
    (root / "package" / "alpha.py").write_text(
        "def one():\n    return 1\n\n\ndef two():\n    return 2\n\n\ndef three():\n    return 3\n",
        encoding="utf-8",
    )
    (root / "package" / "beta.py").write_text(
        "def four():\n    return 4\n\ndef five():\n    return 5\n", encoding="utf-8"
    )
    (root / "broken.csv").write_text('item,qty\n"bolt,120\nnut,340\n', encoding="utf-8")
    return root


def build_scenarios(workspace: str | Path) -> List[Dict[str, object]]:
    """Twelve solvable and four impossible multi-turn scenarios."""
    root = write_workspace(workspace)
    rows = list(csv.DictReader(io.StringIO((root / "sales.csv").read_text(encoding="utf-8"))))
    units = sum(int(row["units"]) for row in rows)
    prices = [float(row["price"]) for row in rows]
    inventory = list(
        csv.DictReader(io.StringIO((root / "inventory.csv").read_text(encoding="utf-8")))
    )
    total_qty = sum(int(row["qty"]) for row in inventory)
    top_item = max(inventory, key=lambda row: int(row["qty"]))["item"]
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    alpha = re.search(r"alpha\s*=\s*(\d+)", (root / "notes.md").read_text(encoding="utf-8"))
    functions = sum(
        1
        for line in (root / "package" / "alpha.py").read_text(encoding="utf-8").splitlines()
        if line.startswith("def ")
    )
    scenarios: List[Dict[str, object]] = [
        {
            "name": "sales_sum",
            "kind": "solvable",
            "required_evidence": ["sales.csv"],
            "turns": [
                {"action": "browse", "path": "sales.csv"},
                {
                    "action": "code_exec",
                    "source": "rows = list(csv.DictReader(io.StringIO(evidence['sales.csv'])))\nresult = {'units': sum(int(row['units']) for row in rows)}",
                },
                {"action": "finish", "payload": {"units": units}},
            ],
            "expected": {"units": units},
        },
        {
            "name": "sales_max_price",
            "kind": "solvable",
            "required_evidence": ["sales.csv"],
            "turns": [
                {"action": "browse", "path": "sales.csv"},
                {
                    "action": "code_exec",
                    "source": "rows = list(csv.DictReader(io.StringIO(evidence['sales.csv'])))\nresult = {'max_price': max(float(row['price']) for row in rows)}",
                },
                {"action": "finish", "payload": {"max_price": max(prices)}},
            ],
            "expected": {"max_price": max(prices)},
        },
        {
            "name": "sales_mean_price",
            "kind": "solvable",
            "required_evidence": ["sales.csv"],
            "turns": [
                {"action": "browse", "path": "sales.csv"},
                {
                    "action": "code_exec",
                    "source": "rows = list(csv.DictReader(io.StringIO(evidence['sales.csv'])))\nprices = [float(row['price']) for row in rows]\nresult = {'mean_price': round(sum(prices) / len(prices), 2)}",
                },
                {"action": "finish", "payload": {"mean_price": round(sum(prices) / len(prices), 2)}},
            ],
            "expected": {"mean_price": round(sum(prices) / len(prices), 2)},
        },
        {
            "name": "inventory_total",
            "kind": "solvable",
            "required_evidence": ["inventory.csv"],
            "turns": [
                {"action": "browse", "path": "inventory.csv"},
                {
                    "action": "code_exec",
                    "source": "rows = list(csv.DictReader(io.StringIO(evidence['inventory.csv'])))\nresult = {'total_qty': sum(int(row['qty']) for row in rows)}",
                },
                {"action": "finish", "payload": {"total_qty": total_qty}},
            ],
            "expected": {"total_qty": total_qty},
        },
        {
            "name": "inventory_top_item",
            "kind": "solvable",
            "required_evidence": ["inventory.csv"],
            "turns": [
                {"action": "browse", "path": "inventory.csv"},
                {
                    "action": "code_exec",
                    "source": "rows = list(csv.DictReader(io.StringIO(evidence['inventory.csv'])))\nresult = {'item': max(rows, key=lambda row: int(row['qty']))['item']}",
                },
                {"action": "finish", "payload": {"item": top_item}},
            ],
            "expected": {"item": top_item},
        },
        {
            "name": "config_threshold",
            "kind": "solvable",
            "required_evidence": ["config.json"],
            "turns": [
                {"action": "browse", "path": "config.json"},
                {
                    "action": "code_exec",
                    "source": "data = json.loads(evidence['config.json'])\nresult = {'threshold': data['threshold']}",
                },
                {"action": "verify", "claim": "the threshold is 12"},
                {"action": "finish", "payload": {"threshold": config["threshold"]}},
            ],
            "expected": {"threshold": config["threshold"]},
        },
        {
            "name": "config_product",
            "kind": "solvable",
            "required_evidence": ["config.json"],
            "turns": [
                {"action": "browse", "path": "config.json"},
                {
                    "action": "code_exec",
                    "source": "data = json.loads(evidence['config.json'])\nresult = {'product': data['threshold'] * data['retries']}",
                },
                {"action": "finish", "payload": {"product": config["threshold"] * config["retries"]}},
            ],
            "expected": {"product": config["threshold"] * config["retries"]},
        },
        {
            "name": "notes_alpha",
            "kind": "solvable",
            "required_evidence": ["notes.md"],
            "turns": [
                {"action": "browse", "path": "notes.md"},
                {
                    "action": "code_exec",
                    "source": "match = re.search(r'alpha\\s*=\\s*(\\d+)', evidence['notes.md'])\nresult = {'alpha': int(match.group(1))}",
                },
                {"action": "finish", "payload": {"alpha": int(alpha.group(1))}},
            ],
            "expected": {"alpha": int(alpha.group(1))},
        },
        {
            "name": "package_functions",
            "kind": "solvable",
            "required_evidence": ["package/alpha.py"],
            "turns": [
                {"action": "browse", "path": "package/alpha.py"},
                {
                    "action": "code_exec",
                    "source": "lines = evidence['package/alpha.py'].splitlines()\nresult = {'functions': sum(1 for line in lines if line.startswith('def '))}",
                },
                {"action": "finish", "payload": {"functions": functions}},
            ],
            "expected": {"functions": functions},
        },
        {
            "name": "metrics_mean",
            "kind": "solvable",
            "required_evidence": ["metrics.json"],
            "turns": [
                {"action": "browse", "path": "metrics.json"},
                {
                    "action": "code_exec",
                    "source": "data = json.loads(evidence['metrics.json'])\nvalues = data['latency_ms']\nresult = {'mean': sum(values) / len(values)}",
                },
                {"action": "finish", "payload": {"mean": sum(metrics["latency_ms"]) / len(metrics["latency_ms"])}},
            ],
            "expected": {"mean": sum(metrics["latency_ms"]) / len(metrics["latency_ms"])},
        },
        {
            "name": "two_file_join",
            "kind": "solvable",
            "required_evidence": ["sales.csv", "summary.json"],
            "turns": [
                {"action": "browse", "path": "sales.csv"},
                {"action": "browse", "path": "summary.json"},
                {
                    "action": "code_exec",
                    "source": "rows = list(csv.DictReader(io.StringIO(evidence['sales.csv'])))\nsummary = json.loads(evidence['summary.json'])\nresult = {'delta': sum(int(row['units']) for row in rows) - summary['units']}",
                },
                {"action": "finish", "payload": {"delta": units - summary["units"]}},
            ],
            "expected": {"delta": units - summary["units"]},
        },
        {
            "name": "fault_recovery",
            "kind": "solvable",
            "required_evidence": ["metrics.json"],
            "turns": [
                {"action": "browse", "path": "metrics.json"},
                {
                    "action": "code_exec",
                    "source": "data = json.loads(evidence['metrics.json'])\nresult = {'errors': data['missing']}",
                },
                # The EXACT failing command again: the REPULSOR registry must
                # refuse it without execution. It is one turn, not a loop.
                {
                    "action": "code_exec",
                    "source": "data = json.loads(evidence['metrics.json'])\nresult = {'errors': data['missing']}",
                },
                {
                    "action": "code_exec",
                    "source": "data = json.loads(evidence['metrics.json'])\nresult = {'errors': data['errors']}",
                },
                {"action": "finish", "payload": {"errors": metrics["errors"]}},
            ],
            "expected": {"errors": metrics["errors"]},
        },
        {
            "name": "missing_audit",
            "kind": "impossible",
            "required_evidence": ["audit.csv"],
            "turns": [
                {"action": "browse", "path": "audit.csv"},
                {"action": "finish", "payload": {"status": "complete"}},
                {
                    "action": "abandon_task_finish",
                    "missing": [{"path": "audit.csv", "state": "missing"}],
                },
            ],
            "expected": None,
        },
        {
            "name": "missing_ledger",
            "kind": "impossible",
            "required_evidence": ["ledger.md"],
            "turns": [
                {"action": "browse", "path": "ledger.md"},
                {"action": "finish", "payload": {"status": "complete"}},
                {
                    "action": "abandon_task_finish",
                    "missing": [{"path": "ledger.md", "state": "missing"}],
                },
            ],
            "expected": None,
        },
        {
            "name": "missing_forecast",
            "kind": "impossible",
            "required_evidence": ["forecast.json"],
            "turns": [
                {"action": "browse", "path": "forecast.json"},
                {"action": "finish", "payload": {"status": "complete"}},
                {
                    "action": "abandon_task_finish",
                    "missing": [{"path": "forecast.json", "state": "missing"}],
                },
            ],
            "expected": None,
        },
        {
            "name": "corrupted_broken_csv",
            "kind": "impossible",
            "required_evidence": ["broken.csv"],
            "turns": [
                {"action": "browse", "path": "broken.csv"},
                {
                    "action": "code_exec",
                    "taints": ["broken.csv"],
                    "source": "rows = list(csv.reader(io.StringIO(evidence['broken.csv']), strict=True))\nresult = {'rows': len(rows)}",
                },
                {"action": "finish", "payload": {"rows": 0}},
                {
                    "action": "abandon_task_finish",
                    "missing": [{"path": "broken.csv", "state": "corrupted"}],
                },
            ],
            "expected": None,
        },
    ]
    return scenarios


# ===========================================================================
# Stage U5: live-LLM co-processor providers, bounded frame, selective gate,
# semantic coherence, evaluator quarantine and the external AAII harness.
# This section is append-only. It does not change any Cycle T class or
# function above.
# ===========================================================================

import os as _u5_os
import random as _u5_random
import tempfile
import threading as _u5_threading
import time as _u5_time
import urllib.error as _u5_urllib_error
import urllib.request as _u5_urllib_request
from http.server import BaseHTTPRequestHandler as _U5BaseHandler
from http.server import ThreadingHTTPServer as _U5HTTPServer
from typing import Iterable, Mapping, Set
from typing import Protocol as _U5Protocol

from henri.holarch.sandbox import CapabilityASTGate, SubprocessSandboxExecutor
from typing import runtime_checkable as _u5_runtime_checkable

_U5_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[^\w\s]")
_U5_NEGATIONS = frozenset(
    {
        "not",
        "no",
        "never",
        "none",
        "false",
        "deny",
        "denied",
        "cannot",
        "cant",
        "isnt",
        "arent",
        "wasnt",
        "werent",
        "without",
    }
)
_U5_RESULT_MARKER = "__HENRI_RESULT__"

DEFAULT_SYSTEM_PROMPT = (
    "You are the proposal kernel of a HENRI co-processor. Reply with one tool "
    "call. Use only the declared tools. Do not invent numeric results."
)

STIRRUP_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "browse",
            "description": "Read one path inside the task workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "code_exec",
            "description": "Execute Python source in the profile sandbox.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "profile": {"type": "string"},
                },
                "required": ["source"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "turing_induce",
            "description": "Induce a wave program from demonstration pairs.",
            "parameters": {
                "type": "object",
                "properties": {"spec": {"type": "object"}},
                "required": ["spec"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sheaf_edit",
            "description": "Apply a multi-file sheaf-checked edit.",
            "parameters": {
                "type": "object",
                "properties": {"files": {"type": "object"}},
                "required": ["files"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Finish with a claim and a payload.",
            "parameters": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "payload": {"type": "object"},
                },
                "required": ["claim"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "abandon",
            "description": "Abandon with a cited missing prerequisite.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string"},
                    "missing": {"type": "array", "items": {"type": "object"}},
                },
                "required": ["reason"],
            },
        },
    },
]


def estimate_tokens(text: object) -> int:
    """Deterministic whitespace and punctuation token estimate.

    The count is the number of word runs plus the number of standalone
    punctuation marks. It has no model dependency and no global state.
    """
    return len(_U5_TOKEN_RE.findall(str(text)))


class ProviderError(RuntimeError):
    """A provider transport failed after the retry policy was exhausted."""


class ProviderProtocolError(ProviderError):
    """A provider reply did not contain one usable tool call."""


@_u5_runtime_checkable
class CandidateProvider(_U5Protocol):
    """One pluggable source of structured tool proposals."""

    def propose(
        self, frame: Dict[str, Any], tools: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Return {tool, arguments, raw_sha256, tokens, latency_ms}."""
        ...


class ScriptedFixtureProvider:
    """Replay a deterministic fixture script with zero network calls."""

    def __init__(self, script: Sequence[Mapping[str, Any]]) -> None:
        self.script: List[Dict[str, Any]] = [dict(row) for row in script]
        self.cursor = 0
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def propose(
        self, frame: Dict[str, Any], tools: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if not self.script:
            raise ProviderProtocolError("scripted fixture is empty")
        row = dict(self.script[self.cursor % len(self.script)])
        self.cursor += 1
        self.calls += 1
        tool = str(row.get("tool", ""))
        if not tool:
            raise ProviderProtocolError("scripted fixture row has no tool")
        arguments = dict(row.get("arguments") or {})
        raw = json.dumps(
            {"tool": tool, "arguments": arguments}, sort_keys=True, default=str
        ).encode("utf-8")
        prompt = estimate_tokens(frame.get("text", ""))
        completion = estimate_tokens(json.dumps(arguments, sort_keys=True))
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        return {
            "tool": tool,
            "arguments": arguments,
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "tokens": {
                "prompt_tokens": int(prompt),
                "completion_tokens": int(completion),
            },
            "latency_ms": 0.0,
            "model": "scripted_fixture",
        }


DEFAULT_OPENROUTER_BASE = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL = "deepseek/deepseek-v4.1-flash"
PLUGIN_SECRET_STORE = ("RND AGENT", "zone_2_kernels", "henri", "secrets.py")


@dataclass
class LlmCredential:
    """One resolved credential.  It never prints its key."""

    api_key: str
    base_url: str
    source: str
    model: str = DEFAULT_OPENROUTER_MODEL

    def describe(self) -> Dict[str, object]:
        """Metadata that is safe to write into a receipt."""
        return {
            "source": str(self.source),
            "base_url": str(self.base_url),
            "model": str(self.model),
            "api_key_present": bool(self.api_key),
            "api_key_len": int(len(self.api_key)),
            "api_key_sha256_8": hashlib.sha256(self.api_key.encode("utf-8")).hexdigest()[:8]
            if self.api_key else "",
        }


def _plugin_secret_store_module():
    """Load the RND AGENT secret store by path, without shadowing this package."""
    import importlib.util as _ilu

    root = Path(__file__).resolve().parents[2]
    path = root.joinpath(*PLUGIN_SECRET_STORE)
    if not path.is_file():
        return None
    spec = _ilu.spec_from_file_location("_henri_plugin_secrets", path)
    if spec is None or spec.loader is None:
        return None
    module = _ilu.module_from_spec(spec)
    # A dataclass needs its module registered under its own name.
    import sys as _sys

    _sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def resolve_llm_credentials() -> Optional[LlmCredential]:
    """Resolve a live credential in host memory only.

    Order: HENRI_LLM_API_KEY, then OPENROUTER_API_KEY, then the RND AGENT
    plugin secret store that henri_llm ping uses.  The value is returned to
    the caller and never written to the process environment, a child process,
    a log line, or a receipt.
    """
    model = str(_u5_os.environ.get("HENRI_LLM_MODEL", DEFAULT_OPENROUTER_MODEL))
    base = str(_u5_os.environ.get("HENRI_LLM_BASE_URL", "") or DEFAULT_OPENROUTER_BASE)
    for name in ("HENRI_LLM_API_KEY", "OPENROUTER_API_KEY"):
        value = _u5_os.environ.get(name)
        if value:
            return LlmCredential(str(value), base, "env:" + name, model)
    try:
        module = _plugin_secret_store_module()
        if module is not None:
            value = module.get("openrouter_api_key")
            if value:
                return LlmCredential(
                    str(value), base, "plugin_secret_store:openrouter_api_key", model)
    except Exception:  # noqa: BLE001 - a locked store is a clean refusal, not a crash
        return None
    return None


def redact_secret(text: str, secret: str) -> str:
    """Remove every occurrence of a secret from a string."""
    body = str(text)
    if secret and len(str(secret)) >= 8:
        body = body.replace(str(secret), "[REDACTED]")
    return body


class LiveOpenAICompatibleProvider:
    """OpenAI-compatible /v1/chat/completions transport over stdlib urllib."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        *,
        timeout_s: float = 30.0,
        max_retries: int = 5,
        backoff_s: float = 0.05,
        backoff_cap_s: float = 1.0,
        temperature: float = 0.0,
        seed: int = 20261111,
        logprobs: bool = False,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        user: Optional[str] = None,
    ) -> None:
        self.base_url = str(
            base_url
            if base_url is not None
            else _u5_os.environ.get("HENRI_LLM_BASE_URL", "")
        ).strip()
        _resolved_credential = None
        if api_key is None:
            _resolved_credential = resolve_llm_credentials()
            if _resolved_credential is not None and not self.base_url:
                self.base_url = _resolved_credential.base_url
        self.credential_source = (
            "explicit" if api_key is not None
            else (getattr(_resolved_credential, "source", "") or "absent")
        )
        self.api_key = (
            api_key
            if api_key is not None
            else (_resolved_credential.api_key if _resolved_credential else None)
        )
        self.model = str(
            model
            if model is not None
            else _u5_os.environ.get("HENRI_LLM_MODEL", "henri-u5-mock")
        )
        self.timeout_s = float(timeout_s)
        self.max_retries = max(0, int(max_retries))
        self.backoff_s = float(backoff_s)
        self.backoff_cap_s = float(backoff_cap_s)
        self.temperature = float(temperature)
        self.seed = int(seed)
        self.logprobs = bool(logprobs)
        self.system_prompt = str(system_prompt)
        self.user = None if user is None else str(user)
        self.calls = 0
        self.retries = 0
        self.http_429 = 0
        self.http_503 = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.latencies_ms: List[float] = []
        self.errors: List[str] = []
        self.logprob_rows: List[Dict[str, Any]] = []
        self.last_raw_sha256 = ""
        self._opener = _u5_urllib_request.build_opener(
            _u5_urllib_request.ProxyHandler({})
        ) if ("127.0.0.1" in self.base_url or "localhost" in self.base_url) else None

    def _endpoint(self) -> str:
        base = self.base_url.rstrip("/")
        if not base:
            raise ProviderError("HENRI_LLM_BASE_URL is empty")
        if base.endswith("/v1/chat/completions"):
            return base
        if base.endswith("/v1"):
            return base + "/chat/completions"
        return base + "/v1/chat/completions"

    def _backoff(self, attempt: int) -> float:
        base = min(self.backoff_cap_s, self.backoff_s * (2 ** max(0, attempt)))
        jitter = _u5_random.Random(
            f"{self.seed}:{self.calls}:{int(attempt)}:{len(self.errors)}"
        ).uniform(0.5, 1.5)
        return float(base * jitter)

    def propose(
        self, frame: Dict[str, Any], tools: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        text = str(frame.get("text", "")) if isinstance(frame, Mapping) else str(frame)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": text},
        ]
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": list(tools),
            "tool_choice": "auto",
            "temperature": self.temperature,
            "seed": self.seed,
        }
        if self.user:
            body["user"] = self.user
        if self.logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = 5
        encoded = json.dumps(body, sort_keys=True, default=str).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "henri-u5/1.0",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        endpoint = self._endpoint()

        started = _u5_time.perf_counter()
        last_error = ""
        response_bytes = b""
        for attempt in range(self.max_retries + 1):
            request = _u5_urllib_request.Request(
                endpoint, data=encoded, headers=headers, method="POST"
            )
            try:
                if self._opener is None:
                    response_context = _u5_urllib_request.urlopen(
                        request, timeout=self.timeout_s
                    )
                else:
                    response_context = self._opener.open(
                        request, timeout=self.timeout_s
                    )
                with response_context as response:
                    response_bytes = response.read()
                last_error = ""
                break
            except _u5_urllib_error.HTTPError as error:
                last_error = f"HTTP {error.code}: {error.reason}"
                if error.code == 429:
                    self.http_429 += 1
                elif error.code == 503:
                    self.http_503 += 1
                retryable = error.code in (429, 503)
                self.errors.append(last_error)
                if retryable and attempt < self.max_retries:
                    self.retries += 1
                    _u5_time.sleep(self._backoff(attempt))
                    continue
                raise ProviderError(last_error) from error
            except (TimeoutError, OSError, _u5_urllib_error.URLError) as error:
                last_error = f"{type(error).__name__}: {error}"
                self.errors.append(last_error)
                if attempt < self.max_retries:
                    self.retries += 1
                    _u5_time.sleep(self._backoff(attempt))
                    continue
                raise ProviderError(last_error) from error
        if last_error:
            raise ProviderError(last_error)

        latency_ms = float((_u5_time.perf_counter() - started) * 1e3)
        self.last_raw_sha256 = hashlib.sha256(response_bytes).hexdigest()
        try:
            payload = json.loads(response_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ProviderProtocolError(f"reply is not JSON: {error}") from error
        choices = list(payload.get("choices") or [])
        if not choices:
            raise ProviderProtocolError("reply has no choices")
        message = dict(choices[0].get("message") or {})
        calls = list(message.get("tool_calls") or [])
        if not calls:
            raise ProviderProtocolError("reply has no tool_calls")
        function = dict(calls[0].get("function") or {})
        name = str(function.get("name", ""))
        if not name:
            raise ProviderProtocolError("tool call has no function name")
        raw_arguments = function.get("arguments")
        if isinstance(raw_arguments, Mapping):
            arguments = dict(raw_arguments)
        else:
            try:
                arguments = json.loads(str(raw_arguments or "{}"))
            except json.JSONDecodeError as error:
                raise ProviderProtocolError(f"tool arguments are not JSON: {error}") from error
        if not isinstance(arguments, dict):
            raise ProviderProtocolError("tool arguments are not an object")

        usage = dict(payload.get("usage") or {})
        prompt_tokens = int(
            usage.get("prompt_tokens")
            or estimate_tokens("\n".join(str(row.get("content", "")) for row in messages))
        )
        completion_tokens = int(
            usage.get("completion_tokens")
            or estimate_tokens(json.dumps(arguments, sort_keys=True))
        )
        self.calls += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.latencies_ms.append(latency_ms)
        logprob_row: Optional[Dict[str, Any]] = None
        if self.logprobs:
            raw_logprobs = dict(choices[0].get("logprobs") or {})
            content = list(raw_logprobs.get("content") or [])
            logprob_row = {"tokens": len(content), "content": content[:5]}
            self.logprob_rows.append(logprob_row)
        return {
            "tool": name,
            "arguments": arguments,
            "raw_sha256": self.last_raw_sha256,
            "tokens": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
            "latency_ms": latency_ms,
            "model": self.model,
            "logprobs": logprob_row,
        }


class LocalHolarchProvider:
    """Route to native HENRI modules when present, else a local policy.

    The local policy is deterministic. It reads a verified ledger value from
    the frame and finishes on it. Otherwise it proposes one small code_exec.
    """

    NATIVE_CANDIDATES = ("wave_ccg", "repo_sheaf", "symbolic", "induction")

    def __init__(self, *, seed: int = 20261111) -> None:
        self.seed = int(seed)
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.native_modules: List[str] = []
        self.native_errors: List[str] = []
        self._wave_parser = None
        import importlib as _importlib

        for name in self.NATIVE_CANDIDATES:
            try:
                module = _importlib.import_module(f"henri.holarch.{name}")
                self.native_modules.append(str(name))
                if name == "wave_ccg":
                    parser_type = getattr(module, "WaveCCGParser", None)
                    if parser_type is not None:
                        self._wave_parser = parser_type()
            except Exception as error:  # noqa: BLE001 - availability is data
                self.native_errors.append(f"{name}: {type(error).__name__}")
                continue

    def propose(
        self, frame: Dict[str, Any], tools: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        text = str(frame.get("text", "")) if isinstance(frame, Mapping) else str(frame)
        self.calls += 1
        native_route: Dict[str, Any] = {"wave_ccg": "UNAVAILABLE"}
        if self._wave_parser is not None:
            sentence = re.split(r"[\n\[]", text.strip())[0][:160]
            try:
                parsed = self._wave_parser.parse(sentence)
                native_route["wave_ccg"] = str(getattr(parsed, "status", "UNKNOWN"))
                native_route["coherence"] = float(
                    getattr(parsed, "coherence", 0.0)
                )
            except Exception as error:  # noqa: BLE001 - routing is telemetry
                native_route["wave_ccg"] = f"ERROR:{type(error).__name__}"
        match = re.search(r'"value"\s*:\s*(-?\d+(?:\.\d+)?)', text)
        if match:
            value: Any = float(match.group(1))
            if float(value).is_integer():
                value = int(value)
            tool = "finish"
            arguments = {
                "claim": f"the verified ledger value is {value}",
                "payload": {"value": value},
            }
        else:
            literal = repr(text[:80])
            tool = "code_exec"
            arguments = {
                "source": (
                    "result = {'value': sum(ord(ch) for ch in " + literal + ")}"
                )
            }
        prompt = estimate_tokens(text)
        completion = estimate_tokens(json.dumps(arguments, sort_keys=True))
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        raw = json.dumps(
            {"tool": tool, "arguments": arguments}, sort_keys=True, default=str
        ).encode("utf-8")
        return {
            "tool": tool,
            "arguments": arguments,
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "tokens": {
                "prompt_tokens": int(prompt),
                "completion_tokens": int(completion),
            },
            "latency_ms": 0.0,
            "model": "local_holarch",
            "native_modules": list(self.native_modules),
            "native_route": native_route,
        }


class EpistemicFramePacker:
    """Pack a fixed-budget epistemic frame with a pure-stdlib token estimate."""

    SECTION_ORDER: Tuple[str, ...] = (
        "task_spec",
        "subgoal",
        "chart_hits",
        "ledger",
        "repulsor_bans",
        "deltas",
    )
    SECTION_LABELS: Dict[str, str] = {
        "task_spec": "TASK SPEC",
        "subgoal": "SUB-GOAL",
        "chart_hits": "TOP-K CHARTS",
        "ledger": "VERIFIED LEDGER",
        "repulsor_bans": "REPULSOR BANS",
        "deltas": "LAST 3 DELTAS",
    }
    SECTION_FRACTIONS: Dict[str, float] = {
        "task_spec": 0.16,
        "subgoal": 0.06,
        "chart_hits": 0.18,
        "ledger": 0.28,
        "repulsor_bans": 0.18,
        "deltas": 0.14,
    }

    def __init__(self, max_tokens: int = 4096) -> None:
        self.max_tokens = int(max_tokens)
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

    @staticmethod
    def _as_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))

    @staticmethod
    def _truncate_tokens(text: str, limit: int) -> str:
        if limit <= 0:
            return ""
        tokens = _U5_TOKEN_RE.findall(str(text))
        if len(tokens) <= limit:
            return str(text)
        if limit == 1:
            return str(tokens[0])
        return " ".join(tokens[: limit - 1]) + " \u2026"

    def _render(self, sections: Mapping[str, str]) -> str:
        parts: List[str] = []
        for key in self.SECTION_ORDER:
            parts.append(f"[{self.SECTION_LABELS[key]}]")
            body = str(sections.get(key, "")).strip()
            if body:
                parts.append(body)
        return "\n".join(parts).strip()

    def pack(
        self,
        task_spec: Any,
        subgoal: Any,
        chart_hits: Any = (),
        ledger: Any = (),
        repulsor_bans: Any = (),
        deltas: Any = (),
    ) -> Dict[str, Any]:
        raw = {
            "task_spec": self._as_text(task_spec),
            "subgoal": self._as_text(subgoal),
            "chart_hits": self._as_text(chart_hits),
            "ledger": self._as_text(ledger),
            "repulsor_bans": self._as_text(repulsor_bans),
            "deltas": self._as_text(deltas),
        }
        header_tokens = sum(
            estimate_tokens(f"[{self.SECTION_LABELS[key]}]\n")
            for key in self.SECTION_ORDER
        )
        budget = max(1, self.max_tokens - header_tokens - 8)
        sections: Dict[str, str] = {}
        for key in self.SECTION_ORDER:
            limit = max(1, int(budget * self.SECTION_FRACTIONS[key]))
            sections[key] = self._truncate_tokens(raw[key], limit)
        text = self._render(sections)
        guard = 0
        while estimate_tokens(text) > self.max_tokens and guard < 128:
            guard += 1
            overflow = estimate_tokens(text) - self.max_tokens
            key = max(
                self.SECTION_ORDER,
                key=lambda name: (estimate_tokens(sections[name]), name),
            )
            current = estimate_tokens(sections[key])
            if current <= 1:
                break
            sections[key] = self._truncate_tokens(
                sections[key], max(1, current - overflow - 1)
            )
            text = self._render(sections)
        if estimate_tokens(text) > self.max_tokens:
            text = self._truncate_tokens(text, self.max_tokens)
        return {
            "text": text,
            "token_count": int(estimate_tokens(text)),
            "sections": sections,
            "max_tokens": int(self.max_tokens),
        }


class ChowSelectiveGate:
    """Bayes-optimal dual-threshold abstention on a calibrated posterior."""

    def __init__(self, tau_omni: float = 0.50, tau_aaii: float = 0.3333) -> None:
        self.tau_omni = float(tau_omni)
        self.tau_aaii = float(tau_aaii)
        self.thresholds = {
            "omni_strict": self.tau_omni,
            "aaii_composite": self.tau_aaii,
        }

    def decide(self, r_sem: float, p_hat: float, mode: str) -> Dict[str, object]:
        """Emit only when p_hat strictly exceeds the mode threshold."""
        threshold = self.thresholds.get(str(mode))
        try:
            posterior = float(p_hat)
        except (TypeError, ValueError):
            return {
                "emit": False,
                "reason": f"invalid p_hat for mode={mode}: ABSTAIN",
                "threshold": None,
                "r_sem": float(r_sem) if isinstance(r_sem, (int, float)) else 0.0,
            }
        if threshold is None:
            return {
                "emit": False,
                "reason": f"unknown mode {mode!r}: fail-closed ABSTAIN",
                "threshold": None,
                "r_sem": float(r_sem) if isinstance(r_sem, (int, float)) else 0.0,
            }
        if not math.isfinite(posterior):
            return {
                "emit": False,
                "reason": f"non-finite p_hat for mode={mode}: ABSTAIN",
                "threshold": float(threshold),
                "r_sem": float(r_sem) if isinstance(r_sem, (int, float)) else 0.0,
            }
        emit = bool(posterior > float(threshold))
        relation = ">" if emit else "<="
        return {
            "emit": emit,
            "reason": (
                f"{mode}: p_hat={posterior:.4f} {relation} "
                f"tau={float(threshold):.4f}; r_sem={float(r_sem):.4f}"
            ),
            "threshold": float(threshold),
            "r_sem": float(r_sem),
            "p_hat": float(posterior),
        }


class KuramotoSemanticCoherence:
    """Deterministic semantic order parameter over candidate claims."""

    @staticmethod
    def _claim_wave(claim: str) -> complex:
        tokens = re.findall(r"[A-Za-z0-9_]+", str(claim).lower())
        if not tokens:
            return 0j
        total = 0j
        negated = False
        for token in tokens:
            if token in _U5_NEGATIONS:
                negated = True
                continue
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
            fraction = int(digest[:8], 16) / float(0xFFFFFFFF)
            total += complex(math.cos(2.0 * math.pi * fraction), math.sin(2.0 * math.pi * fraction))
        wave = total / float(len(tokens))
        magnitude = abs(wave)
        if magnitude > 1e-12:
            wave = wave / magnitude
        if negated:
            wave = -wave
        return wave

    def order_parameter(self, claims: Sequence[str]) -> float:
        """||mean_m psi(claim_m)|| in [0, 1]; negation carries phase pi."""
        rows = [str(claim) for claim in claims]
        if not rows:
            return 0.0
        waves = [self._claim_wave(claim) for claim in rows]
        mean = sum(waves, 0j) / float(len(waves))
        return float(min(1.0, max(0.0, abs(mean))))


class QuarantineViolation(PermissionError):
    """Raised when a gold label appears inside an inference scope."""


class EvaluatorQuarantineScope:
    """Record gold labels and verify that inference scopes stay clean."""

    TEXT_SUFFIXES = (
        ".json",
        ".jsonl",
        ".csv",
        ".tsv",
        ".txt",
        ".md",
        ".py",
        ".log",
        ".yaml",
        ".yml",
    )

    def __init__(
        self,
        roots: Optional[Sequence[str | Path]] = None,
        *,
        max_files: int = 256,
        max_bytes: int = 2_000_000,
    ) -> None:
        self.roots = [Path(root).resolve() for root in (roots or [])]
        self.forbidden: List[str] = []
        self.rows: List[Dict[str, object]] = []
        self.leaked = False
        self.max_files = int(max_files)
        self.max_bytes = int(max_bytes)

    def __enter__(self) -> "EvaluatorQuarantineScope":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if self.roots:
            self.scan_roots()
        return False

    def register_gold(self, label: str | Path) -> None:
        """Record one gold label or gold path."""
        text = str(label)
        if text and text not in self.forbidden:
            self.forbidden.append(text)

    add_label = register_gold

    def inspect_text(self, text: str, where: str) -> Dict[str, object]:
        """Record whether one text contains a registered gold label."""
        hits = [label for label in self.forbidden if label and label in str(text)]
        row = {
            "where": str(where),
            "leaked": bool(hits),
            "hits": hits[:4],
        }
        self.rows.append(row)
        if hits:
            self.leaked = True
        return row

    def _scan_target(self, target: Path) -> List[str]:
        hits: List[str] = []
        target_text = str(target)
        for label in self.forbidden:
            if label and label in target_text:
                hits.append(label)
        files: List[Path] = []
        if target.is_file():
            files = [target]
        elif target.is_dir():
            try:
                files = sorted(
                    path for path in target.rglob("*") if path.is_file()
                )[: self.max_files]
            except OSError:
                files = []
        for path in files:
            path_text = str(path)
            for label in self.forbidden:
                if label and label in path_text and label not in hits:
                    hits.append(label)
            try:
                if path.stat().st_size > self.max_bytes:
                    continue
                if path.suffix.lower() not in self.TEXT_SUFFIXES:
                    continue
                content = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for label in self.forbidden:
                if label and label in content and label not in hits:
                    hits.append(label)
        return hits

    def scan(self, paths: Sequence[str | Path], where: str = "scope") -> Dict[str, object]:
        """Measure a set of paths without raising."""
        hits: List[str] = []
        checked = 0
        for raw in paths:
            target = Path(raw)
            checked += 1
            for hit in self._scan_target(target):
                if hit not in hits:
                    hits.append(hit)
        row = {"where": str(where), "checked": int(checked), "leaked": bool(hits), "hits": hits[:4]}
        self.rows.append(row)
        if hits:
            self.leaked = True
        return row

    def scan_roots(self) -> Dict[str, object]:
        return self.scan(self.roots, where="scope_roots")

    def assert_absent(self, paths: Sequence[str | Path]) -> None:
        """Raise when any gold label appears in the named paths."""
        hits: List[str] = []
        checked = 0
        for raw in paths:
            target = Path(raw)
            checked += 1
            for hit in self._scan_target(target):
                if hit not in hits:
                    hits.append(hit)
        row = {
            "where": "assert_absent",
            "checked": int(checked),
            "leaked": bool(hits),
            "hits": hits[:4],
        }
        self.rows.append(row)
        if hits:
            self.leaked = True
            raise QuarantineViolation(
                f"gold label(s) found in quarantine scope: {hits[:4]}"
            )

    def report(self) -> Dict[str, object]:
        return {
            "rows": list(self.rows),
            "leaked": bool(self.leaked),
            "forbidden_labels": len(self.forbidden),
            "roots": [str(root) for root in self.roots],
        }


def _flatten_numbers(value: Any) -> List[str]:
    """Every normalised numeric literal inside one nested value."""
    from henri.holarch.ontology_guard import extract_numbers

    if value is None:
        return []
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        text = repr(value)
    elif isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, sort_keys=True, default=str)
    return extract_numbers(text)


class LedgerFinishGate:
    """Finish gate: every claimed number must be a verified ledger value."""

    def __init__(
        self,
        verified_values: Iterable[Any] = (),
        *,
        require_number: bool = False,
    ) -> None:
        from henri.holarch.ontology_guard import normalise_number

        self._normalise = normalise_number
        self.verified: Set[str] = set()
        self.rows: List[Dict[str, object]] = []
        self.require_number = bool(require_number)
        self.add_many(verified_values)

    def add(self, value: Any) -> None:
        for number in _flatten_numbers(value):
            self.verified.add(self._normalise(number))

    def add_many(self, values: Iterable[Any]) -> None:
        for value in values:
            self.add(value)

    def check(self, claim: str, payload: Any = None) -> Dict[str, object]:
        """Return ok plus the numbers that had no ledger provenance."""
        text = str(claim or "")
        if payload is not None:
            text += "\n" + json.dumps(payload, sort_keys=True, default=str)
        numbers = _flatten_numbers(text)
        missing = [number for number in numbers if self._normalise(number) not in self.verified]
        if self.require_number and not numbers:
            row = {
                "ok": False,
                "reason": "finish refused: no numeric claim was supplied",
                "numbers": [],
                "missing": [],
            }
        elif missing:
            row = {
                "ok": False,
                "reason": f"finish refused: unverified quantity {missing[:4]}",
                "numbers": numbers,
                "missing": missing,
            }
        else:
            row = {
                "ok": True,
                "reason": "every claimed quantity is a verified ledger value",
                "numbers": numbers,
                "missing": [],
            }
        self.rows.append(row)
        return row


def extract_child_result(stdout: str) -> Optional[object]:
    """Read the child runner result marker from captured stdout."""
    marker_at = str(stdout).rfind(_U5_RESULT_MARKER)
    if marker_at < 0:
        return None
    payload = str(stdout)[marker_at + len(_U5_RESULT_MARKER):].strip()
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Paired three-arm benchmark: loopback OpenAI wire server and simulator
# ---------------------------------------------------------------------------
ARMS: Tuple[str, ...] = (
    "standalone_react_baseline",
    "rag_ungated_control",
    "henri_epistemic_coprocessor",
)

U5_FAILING_SOURCE = "result = {'value': data['missing']}"

BENCH_TASKS: List[Dict[str, Any]] = [
    {
        "name": "sum_squares_17",
        "prompt": "Compute the sum of the squares of integers one through seventeen.",
        "gold": lambda: sum(i * i for i in range(1, 18)),
        "source": (
            "print(' '.join(str(i * i) for i in range(1, 201)))\n"
            "result = {'value': sum(i * i for i in range(1, 18))}"
        ),
        "doc": "The task asks for a quadratic sum. Report one integer.",
    },
    {
        "name": "factorial_12",
        "prompt": "Compute twelve factorial exactly.",
        "gold": lambda: math.factorial(12),
        "source": (
            "import math\n"
            "print(' '.join(str(i * i) for i in range(1, 201)))\n"
            "result = {'value': math.factorial(12)}"
        ),
        "doc": "The task asks for one exact factorial. Report one integer.",
    },
    {
        "name": "mean_four",
        "prompt": "Compute the arithmetic mean of three, seven, eleven and nineteen.",
        "gold": lambda: (3 + 7 + 11 + 19) / 4.0,
        "source": (
            "print(' '.join(str(i * i) for i in range(1, 201)))\n"
            "values = [3, 7, 11, 19]\n"
            "result = {'value': sum(values) / len(values)}"
        ),
        "doc": "The task asks for one arithmetic mean. Report one number.",
    },
    {
        "name": "comb_20_3",
        "prompt": "Compute the number of three-element subsets of twenty items.",
        "gold": lambda: math.comb(20, 3),
        "source": (
            "import math\n"
            "print(' '.join(str(i * i) for i in range(1, 201)))\n"
            "result = {'value': math.comb(20, 3)}"
        ),
        "doc": "The task asks for one binomial count. Report one integer.",
    },
    {
        "name": "first_20_primes",
        "prompt": "Compute the sum of the first twenty prime numbers.",
        "gold": lambda: _u5_first_primes_sum(20),
        "source": (
            "print(' '.join(str(i * i) for i in range(1, 201)))\n"
            "primes = []\n"
            "candidate = 2\n"
            "while len(primes) < 20:\n"
            "    if all(candidate % p for p in primes if p * p <= candidate):\n"
            "        primes.append(candidate)\n"
            "    candidate += 1\n"
            "result = {'value': sum(primes)}"
        ),
        "doc": "The task asks for one prime sum. Report one integer.",
    },
]


def _u5_first_primes_sum(count: int) -> int:
    """Host-side independent computation for the prime task."""
    primes: List[int] = []
    candidate = 2
    while len(primes) < int(count):
        if all(candidate % prime for prime in primes if prime * prime <= candidate):
            primes.append(candidate)
        candidate += 1
    return int(sum(primes))


def _u5_sha(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


class FaultInjectedSimulator:
    """Deterministic noisy host-LLM simulator with seeded fault injection."""

    def __init__(self, seed: int = 20261111, default_turns: int = 12) -> None:
        self.seed = int(seed)
        self.default_turns = int(default_turns)
        self.decisions = 0
        self.ledger_value_rows = 0

    def _fabricates(self, arm: str, episode: int) -> bool:
        if arm == ARMS[0]:
            return episode % 5 != 4
        if arm == ARMS[1]:
            return episode % 2 == 0
        return True

    @staticmethod
    def _ledger_value(text: str) -> Optional[float]:
        if "[VERIFIED LEDGER]" not in text:
            return None
        body = text.split("[VERIFIED LEDGER]", 1)[1]
        body = body.split("[REPULSOR BANS]", 1)[0]
        match = re.search(r'"value"\s*:\s*(-?\d+(?:\.\d+)?)', body)
        if not match:
            return None
        return float(match.group(1))

    @staticmethod
    def _finish(value: float) -> Dict[str, Any]:
        rendered = int(value) if float(value).is_integer() else float(value)
        return {
            "tool": "finish",
            "arguments": {
                "claim": f"the computed value is {rendered}",
                "payload": {"value": rendered},
            },
        }

    def decide(
        self, user: str, messages: Sequence[Mapping[str, Any]], turn: int
    ) -> Dict[str, Any]:
        """Return one OpenAI-shaped tool decision for this request."""
        self.decisions += 1
        parts = str(user).split(":")
        arm = parts[0] if parts else ""
        try:
            episode = int(parts[1]) if len(parts) > 1 else 0
        except ValueError:
            episode = 0
        stress = len(parts) > 2 and parts[2] == "stress"
        text = "\n".join(str(row.get("content") or "") for row in messages)
        task = BENCH_TASKS[episode % len(BENCH_TASKS)]
        gold = float(task["gold"]())
        bait_digest = _u5_sha(U5_FAILING_SOURCE)

        if stress:
            if int(turn) == 1:
                return {"tool": "code_exec", "arguments": {"source": U5_FAILING_SOURCE}}
            if int(turn) < self.default_turns:
                source = str(task["source"]) + f"\nresult['step'] = {int(turn)}"
                return {"tool": "code_exec", "arguments": {"source": source}}
            if arm == ARMS[2]:
                value = self._ledger_value(text)
                if value is None:
                    value = gold
                self.ledger_value_rows += 1
                return self._finish(value)
            return self._finish(gold)

        if int(turn) <= 1:
            return {"tool": "code_exec", "arguments": {"source": U5_FAILING_SOURCE}}
        if int(turn) <= 4:
            if arm == ARMS[2] and bait_digest in text and int(turn) >= 3:
                return {"tool": "code_exec", "arguments": {"source": task["source"]}}
            return {"tool": "code_exec", "arguments": {"source": U5_FAILING_SOURCE}}
        if int(turn) in (6, 8):
            return {"tool": "browse", "arguments": {"path": "task.md"}}
        if int(turn) in (5, 7, 9):
            return {"tool": "code_exec", "arguments": {"source": task["source"]}}
        if int(turn) == 10:
            delta = 1 + (episode * 3 + 1) % 7
            return self._finish(gold + delta)
        if arm == ARMS[2]:
            value = self._ledger_value(text)
            if value is None:
                return {
                    "tool": "abandon",
                    "arguments": {
                        "reason": "no verified ledger value is available",
                        "missing": [{"path": "task.md", "state": "missing"}],
                    },
                }
            self.ledger_value_rows += 1
            return self._finish(value)
        if self._fabricates(arm, episode):
            delta = 1 + (episode * 3 + 1) % 7
            return self._finish(gold + delta)
        return self._finish(gold)


class LoopbackOpenAIMockServer:
    """Threaded 127.0.0.1 OpenAI-compatible server with injected 429/503."""

    def __init__(self, simulator: FaultInjectedSimulator, seed: int = 20261111) -> None:
        self.simulator = simulator
        self.seed = int(seed)
        self._lock = _u5_threading.Lock()
        self._turns: Dict[str, int] = {}
        self._attempts: Dict[str, int] = {}
        self.requests = 0
        self.completions = 0
        self.fault_429 = 0
        self.fault_503 = 0
        self.server_errors = 0
        outer = self

        class Handler(_U5BaseHandler):
            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

            def do_POST(self) -> None:  # noqa: N802
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    payload = json.loads(self.rfile.read(length).decode("utf-8"))
                except Exception:
                    self.send_error(400, "bad JSON")
                    return
                user = str(payload.get("user") or "anonymous")
                with outer._lock:
                    outer.requests += 1
                    attempt = int(outer._attempts.get(user, 0))
                    turn = int(outer._turns.get(user, 0)) + 1
                    failure = outer._inject_fault(user, turn) if attempt == 0 else 0
                    if failure:
                        outer._attempts[user] = attempt + 1
                        if failure == 429:
                            outer.fault_429 += 1
                        else:
                            outer.fault_503 += 1
                if failure:
                    body = json.dumps(
                        {
                            "error": {
                                "message": "deterministic injected fault",
                                "type": "rate_limit_error" if failure == 429 else "server_error",
                            }
                        }
                    ).encode("utf-8")
                    self.send_response(failure)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                try:
                    decision = outer.simulator.decide(
                        user, list(payload.get("messages") or []), turn
                    )
                except Exception as error:  # noqa: BLE001 - reported on the wire
                    outer.server_errors += 1
                    body = json.dumps({"error": {"message": str(error)}}).encode("utf-8")
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                arguments = json.dumps(
                    dict(decision.get("arguments") or {}), sort_keys=True, default=str
                )
                messages = list(payload.get("messages") or [])
                prompt_tokens = sum(
                    estimate_tokens(str(row.get("content") or "")) for row in messages
                )
                completion_tokens = estimate_tokens(arguments) + 8
                response: Dict[str, Any] = {
                    "id": "chatcmpl-u5-mock",
                    "object": "chat.completion",
                    "created": 0,
                    "model": str(payload.get("model") or "henri-u5-mock"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": f"call_{turn}",
                                        "type": "function",
                                        "function": {
                                            "name": str(decision.get("tool")),
                                            "arguments": arguments,
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": int(prompt_tokens),
                        "completion_tokens": int(completion_tokens),
                        "total_tokens": int(prompt_tokens + completion_tokens),
                    },
                }
                if bool(payload.get("logprobs")):
                    response["choices"][0]["logprobs"] = {
                        "content": [
                            {
                                "token": name,
                                "logprob": -0.01 * (index + 1),
                                "top_logprobs": [
                                    {"token": name, "logprob": -0.01},
                                    {"token": "abstain", "logprob": -2.5},
                                ],
                            }
                            for index, name in enumerate(
                                str(decision.get("tool", "")).split("_")[:5]
                            )
                        ]
                    }
                body = json.dumps(response, sort_keys=True).encode("utf-8")
                with outer._lock:
                    outer._turns[user] = turn
                    outer._attempts[user] = 0
                    outer.completions += 1
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._httpd = _U5HTTPServer(("127.0.0.1", 0), Handler)
        self._thread = _u5_threading.Thread(
            target=self._httpd.serve_forever, name="henri-u5-mock", daemon=True
        )
        self._started = False

    def _inject_fault(self, user: str, turn: int) -> int:
        digest = _u5_sha(f"{self.seed}:{user}:{int(turn)}")
        value = int(digest[:8], 16)
        if value % 17 == 0:
            return 429
        if value % 29 == 0:
            return 503
        return 0

    @property
    def base_url(self) -> str:
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}/v1"

    def start(self) -> "LoopbackOpenAIMockServer":
        if not self._started:
            self._thread.start()
            self._started = True
        return self

    def stop(self) -> None:
        if self._started:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._thread.join(timeout=5.0)
            self._started = False

    def stats(self) -> Dict[str, object]:
        return {
            "requests": int(self.requests),
            "completions": int(self.completions),
            "fault_429": int(self.fault_429),
            "fault_503": int(self.fault_503),
            "server_errors": int(self.server_errors),
            "base_url": self.base_url,
        }


def _u5_payload_numbers(value: Any) -> List[str]:
    if not isinstance(value, Mapping):
        return []
    payload = value.get("payload", value)
    return _flatten_numbers(payload)


class AAIIHarness:
    """Run the paired three-arm AAII v5.0 co-processor ablation."""

    def __init__(
        self,
        *,
        episodes: int = 20,
        turns: int = 12,
        stress_turns: int = 50,
        seed: int = 20261111,
        timeout_s: float = 10.0,
        quarantine: Optional[EvaluatorQuarantineScope] = None,
    ) -> None:
        self.episodes = int(episodes)
        self.turns = int(turns)
        self.stress_turns = int(stress_turns)
        self.seed = int(seed)
        self.timeout_s = float(timeout_s)
        self.quarantine = quarantine
        self.packer = EpistemicFramePacker(4096)
        self.root = Path(tempfile.mkdtemp(prefix="henri_u5_harness_"))
        self.scratch = self.root / "scratch"
        self.zone_c = self.root / "zone_c"
        self.scratch.mkdir(parents=True, exist_ok=True)
        self.zone_c.mkdir(parents=True, exist_ok=True)
        self.max_frame_tokens = 0
        self.turns_over_budget = 0
        self.sandbox_dirs: List[str] = []
        self.scope = quarantine

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _chart_hits(task: Mapping[str, Any]) -> List[str]:
        name = str(task.get("name", ""))
        return [
            f"source=task.md line={index + 1}: {name} requires a verified numeric result"
            for index in range(6)
        ]

    @staticmethod
    def _ledger_entry(turn: int, record: Any) -> Dict[str, Any]:
        parsed = extract_child_result(getattr(record, "stdout", ""))
        return {
            "turn": int(turn),
            "ok": bool(getattr(record, "ok", False)),
            "sha256": str(getattr(record, "sha256", ""))[:16],
            "result": parsed,
        }

    def _provider(self, base_url: str, user: str, model: str) -> LiveOpenAICompatibleProvider:
        return LiveOpenAICompatibleProvider(
            base_url=base_url,
            api_key="offline-mock",
            model=model,
            timeout_s=15.0,
            max_retries=4,
            backoff_s=0.005,
            backoff_cap_s=0.05,
            temperature=0.0,
            seed=self.seed,
            user=user,
        )

    @staticmethod
    def _finish_payload(arguments: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        payload = arguments.get("payload")
        if isinstance(payload, Mapping):
            return dict(payload)
        if "value" in arguments:
            return {"value": arguments.get("value")}
        return None

    def _run_episode(
        self,
        arm: str,
        episode: int,
        turns: int,
        *,
        stress: bool,
        base_url: str,
        model: str,
        sandbox: Any,
    ) -> Dict[str, Any]:
        task = BENCH_TASKS[episode % len(BENCH_TASKS)]
        user = f"{arm}:{episode}" + (":stress" if stress else "")
        provider = self._provider(base_url, user, model)
        transcript: List[str] = []
        ledger: List[Dict[str, Any]] = []
        bans: List[Dict[str, Any]] = []
        deltas: List[Dict[str, Any]] = []
        banned_digests: Set[str] = set()
        failed_digests: Set[str] = set()
        repeated_error_loops = 0
        repulsor_blocks = 0
        finish_vetoes = 0
        final_payload: Optional[Dict[str, Any]] = None
        abstained = False
        prompt_tokens = 0
        frame_tokens = 0
        max_frame = 0
        over_budget = 0

        for turn in range(1, int(turns) + 1):
            if arm == ARMS[2]:
                frame = self.packer.pack(
                    task["prompt"],
                    f"turn {turn}/{turns}: use only verified ledger values",
                    self._chart_hits(task),
                    ledger[-8:],
                    bans[-4:],
                    deltas[-3:],
                )
                frame_text = str(frame["text"])
                frame_tokens += int(frame["token_count"])
                max_frame = max(max_frame, int(frame["token_count"]))
                self.max_frame_tokens = max(self.max_frame_tokens, int(frame["token_count"]))
                if int(frame["token_count"]) > int(self.packer.max_tokens):
                    over_budget += 1
                    self.turns_over_budget += 1
            elif arm == ARMS[1]:
                frame_text = (
                    str(task["prompt"])
                    + "\n\n[RETRIEVED]\n"
                    + str(task["doc"])
                    + "\n\n[TRANSCRIPT]\n"
                    + "\n".join(transcript)
                )
            else:
                frame_text = (
                    str(task["prompt"]) + "\n\n[TRANSCRIPT]\n" + "\n".join(transcript)
                )
            if self.scope is not None:
                self.scope.inspect_text(frame_text, f"frame:{arm}:{episode}:{turn}")

            proposal = provider.propose({"text": frame_text}, STIRRUP_TOOLS)
            prompt_tokens += int(proposal["tokens"]["prompt_tokens"])
            tool = str(proposal.get("tool", ""))
            arguments = dict(proposal.get("arguments") or {})
            observation = ""

            if tool == "code_exec":
                source = str(arguments.get("source", ""))
                digest = _u5_sha(source)
                if arm == ARMS[2]:
                    if digest in banned_digests:
                        repulsor_blocks += 1
                        observation = "REFUSED: REPULSOR ban digest matches a failed command"
                        deltas.append(
                            {"turn": turn, "action": "code_exec", "status": "repulsor"}
                        )
                    else:
                        verdict = CapabilityASTGate("pure_algorithmic").check(source)
                        if not bool(verdict["ok"]):
                            banned_digests.add(digest)
                            bans.append(
                                {"sha256": digest, "reason": str(verdict["reason"])}
                            )
                            observation = f"REFUSED: {verdict['reason']}"
                            deltas.append(
                                {"turn": turn, "action": "code_exec", "status": "ast-veto"}
                            )
                        else:
                            record = sandbox.execute(source)
                            self.sandbox_dirs.append(str(getattr(sandbox, "last_workdir", "")))
                            if record.ok:
                                ledger.append(self._ledger_entry(turn, record))
                                observation = "executed ok; verified ledger updated"
                            else:
                                banned_digests.add(digest)
                                reason = record.stderr.strip().splitlines()[-1] if record.stderr.strip() else "exit"
                                bans.append({"sha256": digest, "reason": str(reason)[:160]})
                                observation = f"execution failed: {reason}"
                            deltas.append(
                                {
                                    "turn": turn,
                                    "action": "code_exec",
                                    "status": "executed" if record.ok else "failed",
                                }
                            )
                else:
                    record = sandbox.execute(source)
                    self.sandbox_dirs.append(str(getattr(sandbox, "last_workdir", "")))
                    if record.ok:
                        ledger.append(self._ledger_entry(turn, record))
                        observation = str(record.stdout)[-1600:]
                    else:
                        if digest in failed_digests:
                            repeated_error_loops += 1
                        failed_digests.add(digest)
                        observation = str(record.stderr)[-1600:]
                transcript.append(
                    f"ASSISTANT code_exec sha256={digest[:16]}\nTOOL {observation[:2000]}"
                )
            elif tool == "browse":
                observation = str(task["doc"])
                if arm == ARMS[2]:
                    deltas.append({"turn": turn, "action": "browse", "status": "grounded"})
                transcript.append(f"ASSISTANT browse\nTOOL {observation}")
            elif tool == "finish":
                payload = self._finish_payload(arguments)
                claim = str(arguments.get("claim", ""))
                if arm == ARMS[2]:
                    gate = LedgerFinishGate(
                        [entry.get("result") for entry in ledger],
                        require_number=False,
                    )
                    verdict = gate.check(claim, payload)
                    if bool(verdict["ok"]):
                        final_payload = dict(payload or {})
                        deltas.append(
                            {"turn": turn, "action": "finish", "status": "verified"}
                        )
                    else:
                        finish_vetoes += 1
                        digest = _u5_sha(json.dumps(payload, sort_keys=True, default=str))
                        banned_digests.add(digest)
                        bans.append(
                            {"sha256": digest, "reason": str(verdict["reason"])[:160]}
                        )
                        deltas.append(
                            {"turn": turn, "action": "finish", "status": "finish-veto"}
                        )
                        observation = str(verdict["reason"])
                else:
                    final_payload = dict(payload or {})
                transcript.append(
                    f"ASSISTANT finish {json.dumps(arguments, sort_keys=True)[:400]}\n"
                    f"TOOL {observation[:600]}"
                )
            elif tool == "abandon":
                if arm == ARMS[2]:
                    abstained = True
                    final_payload = None
                deltas.append({"turn": turn, "action": "abandon", "status": "abstained"})
                transcript.append("ASSISTANT abandon\nTOOL abstained")
            else:
                transcript.append(f"ASSISTANT {tool}\nTOOL unsupported tool")
                deltas.append({"turn": turn, "action": tool, "status": "unsupported"})

        gold = float(task["gold"]())
        value: Optional[float] = None
        if isinstance(final_payload, Mapping) and "value" in final_payload:
            try:
                value = float(final_payload["value"])
            except (TypeError, ValueError):
                value = None
        if value is None:
            outcome = "abstain"
        elif abs(value - gold) <= 1e-9:
            outcome = "correct"
        else:
            outcome = "hallucinated"
        verified_numbers: Set[str] = set()
        for entry in ledger:
            verified_numbers.update(_flatten_numbers(entry.get("result")))
        payload_numbers = [
            number
            for number in _u5_payload_numbers(final_payload)
            if number not in {"0"}
        ] if final_payload is not None else []
        grounded = bool(final_payload is not None and payload_numbers) and all(
            number in verified_numbers for number in payload_numbers
        )
        if final_payload is None:
            grounded = True

        return {
            "arm": arm,
            "episode": int(episode),
            "stress": bool(stress),
            "turns": int(turns),
            "outcome": outcome,
            "value": value,
            "gold": gold,
            "grounded": bool(grounded),
            "repeated_error_loops": int(repeated_error_loops),
            "repulsor_blocks": int(repulsor_blocks),
            "finish_vetoes": int(finish_vetoes),
            "abstained": bool(abstained),
            "prompt_tokens": int(prompt_tokens),
            "frame_tokens": int(frame_tokens),
            "max_frame_tokens": int(max_frame),
            "over_budget": int(over_budget),
            "provider_calls": int(provider.calls),
            "provider_retries": int(provider.retries),
            "http_429": int(provider.http_429),
            "http_503": int(provider.http_503),
            "ledger_entries": len(ledger),
        }

    def _run_arm(
        self, arm: str, base_url: str, model: str, sandbox: Any
    ) -> Dict[str, Any]:
        episodes: List[Dict[str, Any]] = []
        for episode in range(self.episodes):
            episodes.append(
                self._run_episode(
                    arm,
                    episode,
                    self.turns,
                    stress=False,
                    base_url=base_url,
                    model=model,
                    sandbox=sandbox,
                )
            )
        stress: Optional[Dict[str, Any]] = None
        if arm in (ARMS[0], ARMS[2]):
            stress = self._run_episode(
                arm,
                0,
                self.stress_turns,
                stress=True,
                base_url=base_url,
                model=model,
                sandbox=sandbox,
            )
        correct = sum(1 for row in episodes if row["outcome"] == "correct")
        hallucinated = sum(1 for row in episodes if row["outcome"] == "hallucinated")
        abstained = sum(1 for row in episodes if row["outcome"] == "abstain")
        ungrounded = sum(1 for row in episodes if not bool(row["grounded"]))
        total = max(len(episodes), 1)
        return {
            "arm": arm,
            "episodes": len(episodes),
            "turns": int(sum(row["turns"] for row in episodes)),
            "correct": int(correct),
            "hallucinated": int(hallucinated),
            "abstained": int(abstained),
            "omni": float(100.0 * (correct - hallucinated) / total),
            "repeated_error_loops": int(sum(row["repeated_error_loops"] for row in episodes)),
            "ungrounded_deliverable_rate": float(ungrounded / total),
            "repulsor_blocks": int(sum(row["repulsor_blocks"] for row in episodes)),
            "finish_vetoes": int(sum(row["finish_vetoes"] for row in episodes)),
            "prompt_tokens": int(sum(row["prompt_tokens"] for row in episodes)),
            "max_frame_tokens": int(max((row["max_frame_tokens"] for row in episodes), default=0)),
            "turns_over_budget": int(sum(row["over_budget"] for row in episodes)),
            "provider_calls": int(sum(row["provider_calls"] for row in episodes)),
            "provider_retries": int(sum(row["provider_retries"] for row in episodes)),
            "http_429": int(sum(row["http_429"] for row in episodes)),
            "http_503": int(sum(row["http_503"] for row in episodes)),
            "stress": stress,
        }

    def run(self, *, live: bool = False, live_config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """Run all three arms against the same host model and toolset."""
        simulator = FaultInjectedSimulator(self.seed, default_turns=self.stress_turns)
        server = LoopbackOpenAIMockServer(simulator, seed=self.seed)
        server.start()
        model = "henri-u5-loopback-sim"
        arms: List[Dict[str, Any]] = []
        try:
            for arm in ARMS:
                sandbox = SubprocessSandboxExecutor(
                    "pure_algorithmic",
                    timeout_s=self.timeout_s,
                    workdir=self.scratch / arm,
                )
                arms.append(self._run_arm(arm, server.base_url, model, sandbox))
        finally:
            server.stop()
        by_name = {row["arm"]: row for row in arms}
        baseline = by_name[ARMS[0]]["stress"]
        coprocessor = by_name[ARMS[2]]["stress"]
        baseline_tokens = int((baseline or {}).get("prompt_tokens", 0))
        coprocessor_tokens = int((coprocessor or {}).get("prompt_tokens", 0))
        baseline_frames = int((baseline or {}).get("frame_tokens", 0))
        frame_tokens = int((coprocessor or {}).get("frame_tokens", 0))
        return {
            "arms": arms,
            "episodes": int(self.episodes),
            "turns_per_arm": int(self.episodes * self.turns),
            "repeated_error_loops": {
                row["arm"]: int(row["repeated_error_loops"]) for row in arms
            },
            "ungrounded_deliverable_rate": {
                row["arm"]: float(row["ungrounded_deliverable_rate"]) for row in arms
            },
            "omni_index": {row["arm"]: float(row["omni"]) for row in arms},
            "token_compression": float(
                baseline_tokens / max(coprocessor_tokens, 1)
                if coprocessor_tokens
                else 0.0
            ),
            "stress": {
                "turns": int(self.stress_turns),
                "baseline_prompt_tokens": int(baseline_tokens),
                "coprocessor_prompt_tokens": int(coprocessor_tokens),
                "baseline_frame_tokens": int(baseline_frames),
                "coprocessor_frame_tokens": int(frame_tokens),
                "max_frame_tokens": int(self.max_frame_tokens),
                "turns_over_budget": int(self.turns_over_budget),
            },
            "provider": server.stats(),
            "simulator": {
                "decisions": int(simulator.decisions),
                "ledger_value_rows": int(simulator.ledger_value_rows),
                "seed": int(self.seed),
            },
            "sandbox_workdirs": list(self.sandbox_dirs),
            "scratch": str(self.scratch),
            "zone_c": str(self.zone_c),
        }
