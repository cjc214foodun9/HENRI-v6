"""The six-holon scientific graph.

Evidence label: DERIVED for the stage wiring, OBSERVED for the verdicts.

One holon is one stage of the workflow, with one job and one output. The
graph runs them in order and stops at the first failure, because a later
holon that inherits a broken fact produces a confident wrong answer.

    FRAME      restate the goal as a claim that can fail
    MEASURE    run the registered evidence commands
    FALSIFY    prove the verification machinery can fail
    ATTRIBUTE  tie every number to a receipt and its SHA-256
    RECORD     append immutable ontology entries and a ledger row
    GATE       decide promote or block

Every stage returns a HolonResult. Nothing is asserted without evidence,
and the falsify stage is what stops the graph from being a rubber stamp.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from henri.ontology import OntologyLedger, sha256_file

HOLONS: Tuple[str, ...] = ("FRAME", "MEASURE", "FALSIFY", "ATTRIBUTE", "RECORD", "GATE")


@dataclass
class HolonResult:
    name: str
    status: str
    summary: str
    facts: List[Tuple[str, str, str]] = field(default_factory=list)
    evidence: Dict[str, str] = field(default_factory=dict)
    detail: Dict[str, object] = field(default_factory=dict)
    seconds: float = 0.0

    def as_dict(self) -> Dict[str, object]:
        return {
            "holon": self.name,
            "status": self.status,
            "summary": self.summary,
            "facts": [list(fact) for fact in self.facts],
            "evidence": dict(self.evidence),
            "detail": self.detail,
            "seconds": round(self.seconds, 3),
        }


def _run(command: Sequence[str], cwd: Path, timeout: int) -> Dict[str, object]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            list(command),
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=int(timeout),
            check=False,
        )
        return {
            "command": " ".join(command),
            "exit": int(completed.returncode),
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-2000:],
            "seconds": round(time.perf_counter() - started, 3),
        }
    except subprocess.TimeoutExpired:
        return {
            "command": " ".join(command),
            "exit": 124,
            "stdout": "",
            "stderr": f"timeout after {timeout}s",
            "seconds": round(time.perf_counter() - started, 3),
        }
    except FileNotFoundError as exc:
        return {
            "command": " ".join(command),
            "exit": 127,
            "stdout": "",
            "stderr": str(exc),
            "seconds": round(time.perf_counter() - started, 3),
        }


class HolonGraph:
    """Runs the six holons over one goal."""

    def __init__(
        self,
        root: str | Path = ".",
        profile: str = "quick",
        timeout: int = 120,
        ontology_path: Optional[str | Path] = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.profile = str(profile)
        if self.profile not in {"quick", "full"}:
            raise ValueError("profile must be 'quick' or 'full'")
        self.timeout = int(timeout)
        self.ontology = OntologyLedger(
            ontology_path or (self.root / "data" / "holarch" / "ontology.jsonl")
        )
        self.receipts = self.root / "receipts" / "holarch"
        self.docs = self.root / "docs" / "HOLARCH.md"

    # ------------------------------------------------------------- FRAME
    def frame(self, goal: str) -> HolonResult:
        started = time.perf_counter()
        text = " ".join(str(goal or "").split())
        if not text:
            return HolonResult("FRAME", "FAIL", "empty goal", seconds=time.perf_counter() - started)
        claim = text if text.endswith(".") else text + "."
        falsifiable = bool(re.search(r"\b(measure|test|gate|receipt|improve|close|verify|prove|reduce|raise)\b", claim, re.I))
        result = HolonResult(
            "FRAME",
            "PASS" if falsifiable else "FAIL",
            "claim is testable" if falsifiable else "goal states no testable claim",
            facts=[("goal", "framed_as", claim)],
            detail={"claim": claim, "testable": falsifiable},
            seconds=time.perf_counter() - started,
        )
        return result

    # ----------------------------------------------------------- MEASURE
    def measure(self, context: Dict[str, object]) -> HolonResult:
        started = time.perf_counter()
        checks: Dict[str, Dict[str, object]] = {}
        integrity = self._receipt_integrity()
        checks["receipt_integrity"] = integrity
        docs = self._documentation_consistency()
        checks["documentation"] = docs
        runtime = self._runtime(profile=self.profile)
        checks["runtime"] = runtime
        failed = [name for name, row in checks.items() if not bool(row.get("ok"))]
        result = HolonResult(
            "MEASURE",
            "PASS" if not failed else "FAIL",
            "all measurements ran" if not failed else f"failed: {', '.join(failed)}",
            facts=[
                (
                    "receipts",
                    "verified",
                    f"{integrity['matched']}/{integrity['total']} sha256 match",
                ),
                (
                    "documentation",
                    "covers",
                    f"{docs['referenced']}/{docs['receipts']} receipts",
                ),
                ("runtime", "exit", str(runtime.get("exit"))),
            ],
            evidence={**integrity.get("evidence", {}), **docs.get("evidence", {})},
            detail={"checks": checks, "profile": self.profile},
            seconds=time.perf_counter() - started,
        )
        context["measure"] = checks
        return result

    # ----------------------------------------------------------- FALSIFY
    def falsify(self, context: Dict[str, object]) -> HolonResult:
        """Prove the integrity check can fail, on a corrupted copy.

        A gate that has never rejected anything is not a gate. This stage
        flips one character in a copy of a real sidecar and requires the
        checker to reject it.
        """
        started = time.perf_counter()
        sidecars = sorted(self.receipts.glob("*.sha256")) if self.receipts.exists() else []
        if not sidecars:
            return HolonResult(
                "FALSIFY", "SKIP", "no receipt sidecar to corrupt",
                seconds=time.perf_counter() - started,
            )
        source = sidecars[0]
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / source.name
            text = source.read_text(encoding="utf-8").strip()
            flipped = text[:-1] + ("0" if text[-1] != "0" else "1")
            target.write_text(flipped + "\n", encoding="utf-8")
            original = source.read_text(encoding="utf-8").strip()
            detected = original != flipped
        # The control is the corrupted copy; the check must see a mismatch.
        control = self._check_sidecar(source, expected_text=flipped)
        status = "PASS" if (detected and not control["ok"]) else "FAIL"
        result = HolonResult(
            "FALSIFY",
            status,
            "corrupted sidecar rejected" if status == "PASS" else "corruption went undetected",
            facts=[("integrity_check", "rejects", "corrupted sha256" if status == "PASS" else "nothing")],
            detail={"control": control, "sidecar": source.name},
            seconds=time.perf_counter() - started,
        )
        context["falsify"] = result.detail
        return result

    # --------------------------------------------------------- ATTRIBUTE
    def attribute(self, context: Dict[str, object]) -> HolonResult:
        started = time.perf_counter()
        integrity = self._receipt_integrity()
        rows = integrity.pop("rows", [])
        matched = [row for row in rows if row["ok"]]
        evidence = {row["receipt"]: row["digest"] for row in matched}
        result = HolonResult(
            "ATTRIBUTE",
            "PASS" if matched and len(matched) == len(rows) else "FAIL",
            f"{len(matched)} of {len(rows)} receipts carry a matching digest",
            facts=[(row["receipt"], "sha256", row["digest"]) for row in matched],
            evidence=evidence,
            detail={"unmatched": [row for row in rows if not row["ok"]]},
            seconds=time.perf_counter() - started,
        )
        context["attribute"] = evidence
        return result

    # ------------------------------------------------------------ RECORD
    def record(self, context: Dict[str, object], results: Sequence[HolonResult]) -> HolonResult:
        started = time.perf_counter()
        appended = 0
        for result in results:
            for subject, predicate, obj in result.facts:
                self.ontology.append(
                    kind="measurement" if result.name != "GATE" else "gate",
                    subject=subject,
                    predicate=predicate,
                    obj=obj,
                    evidence={"holon": result.name},
                )
                appended += 1
        verified = self.ontology.verify()
        status = "PASS" if verified.get("ok") else "FAIL"
        return HolonResult(
            "RECORD",
            status,
            f"{appended} ontology entries appended, chain length {verified.get('length')}",
            facts=[("ontology", "appended", str(appended)), ("ontology", "verified", str(bool(verified.get('ok'))))],
            detail={"appended": appended, "verify": verified},
            seconds=time.perf_counter() - started,
        )

    # -------------------------------------------------------------- GATE
    def gate(self, context: Dict[str, object], results: Sequence[HolonResult]) -> HolonResult:
        started = time.perf_counter()
        stages = {result.name: result.status for result in results}
        chain = self.ontology.verify()
        gates = {
            "all_holons_pass": all(stages.get(name) in {"PASS", "SKIP"} for name in HOLONS if name in stages),
            "evidence_integrity": bool(context.get("measure", {}).get("receipt_integrity", {}).get("ok")),
            "falsifier_rejects": stages.get("FALSIFY") == "PASS",
            "ontology_chain": bool(chain.get("ok")) and int(chain.get("length", 0)) > 0,
        }
        passed = all(gates.values())
        return HolonResult(
            "GATE",
            "PASS" if passed else "FAIL",
            "promote" if passed else "block",
            facts=[(name, "passed" if value else "failed", "gate") for name, value in gates.items()],
            detail={"gates": gates, "stages": stages, "ontology_head": chain.get("head")},
            seconds=time.perf_counter() - started,
        )

    # ------------------------------------------------------------ checks
    def _check_sidecar(self, sidecar: Path, expected_text: Optional[str] = None) -> Dict[str, object]:
        stem = sidecar.name[: -len(".sha256")]
        target = sidecar.parent / stem
        recorded = (expected_text if expected_text is not None else sidecar.read_text(encoding="utf-8")).strip()
        if not target.exists():
            return {"ok": False, "receipt": stem, "reason": "missing file"}
        actual = sha256_file(target)
        return {"ok": actual == recorded, "receipt": stem, "digest": actual, "recorded": recorded}

    def _receipt_integrity(self) -> Dict[str, object]:
        rows: List[Dict[str, object]] = []
        sidecars = sorted(self.receipts.glob("*.sha256")) if self.receipts.exists() else []
        for sidecar in sidecars:
            rows.append(self._check_sidecar(sidecar))
        matched = sum(1 for row in rows if row["ok"])
        return {
            "ok": bool(rows) and matched == len(rows),
            "total": len(rows),
            "matched": matched,
            "rows": rows,
            "evidence": {row["receipt"]: str(row.get("digest", "")) for row in rows if row["ok"]},
        }

    def _documentation_consistency(self) -> Dict[str, object]:
        if not self.docs.exists():
            return {"ok": False, "receipts": 0, "referenced": 0, "missing": []}
        text = self.docs.read_text(encoding="utf-8")
        receipts = sorted(self.receipts.glob("*.json")) if self.receipts.exists() else []
        missing = [path.name for path in receipts if path.name not in text]
        return {
            "ok": not missing and bool(receipts),
            "receipts": len(receipts),
            "referenced": len(receipts) - len(missing),
            "missing": missing,
            "evidence": {"docs": str(self.docs.relative_to(self.root))},
        }

    def _runtime(self, profile: str) -> Dict[str, object]:
        if profile == "full":
            row = _run(
                [sys.executable, "-m", "pytest", "tests/test_holarch_d5_d8.py", "tests/test_holarch_d1_d4.py", "-q"],
                self.root,
                self.timeout,
            )
        else:
            row = _run([sys.executable, "-m", "henri.holarch", "selftest", "--json"], self.root, self.timeout)
        row["ok"] = int(row.get("exit", 1)) == 0
        return row
