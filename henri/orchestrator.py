"""The super-agent orchestrator.

Evidence label: DERIVED for the policy, OBSERVED for every verdict.

The orchestrator is the part that decides, runs and records. It does not
answer questions from memory. It takes a goal, picks a plan, runs the six
holons, updates the immutable ontology layer, writes one ledger row, and
returns a verdict with the evidence attached.

Autonomy here is bounded on purpose:

* the plan chooses a profile and a timeout, never arbitrary code;
* the only external actions are registered commands;
* a failed holon stops the graph, so the verdict is always grounded;
* every run is recorded before the answer is returned.

That is the difference between an agent that orchestrates a scientific
workflow and an agent that narrates one.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from henri.holon import HOLONS, HolonGraph, HolonResult

PLANS: Dict[str, Dict[str, object]] = {
    "promotion": {
        "keywords": ("gate", "promote", "close", "d1", "d3", "d4", "release", "ship"),
        "profile": "full",
        "reason": "a promotion claim needs the test suite, not only the receipts",
    },
    "measurement": {
        "keywords": ("measure", "benchmark", "noise", "relational", "accuracy", "receipt"),
        "profile": "quick",
        "reason": "a measurement claim needs receipt integrity and documentation coverage",
    },
    "release": {
        "keywords": ("commit", "push", "publish", "version", "tag"),
        "profile": "quick",
        "reason": "a release claim needs a clean evidence chain before it leaves the host",
    },
    "survey": {
        "keywords": (),
        "profile": "quick",
        "reason": "default survey of the evidence chain",
    },
}


@dataclass
class Plan:
    name: str
    profile: str
    reason: str

    def as_dict(self) -> Dict[str, str]:
        return {"name": self.name, "profile": self.profile, "reason": self.reason}


def choose_plan(goal: str) -> Plan:
    text = str(goal or "").lower()
    for name, spec in PLANS.items():
        if name == "survey":
            continue
        if any(word in text for word in spec["keywords"]):
            return Plan(name=name, profile=str(spec["profile"]), reason=str(spec["reason"]))
    spec = PLANS["survey"]
    return Plan(name="survey", profile=str(spec["profile"]), reason=str(spec["reason"]))


class Orchestrator:
    """Goal in, grounded verdict out."""

    def __init__(
        self,
        root: str | Path = ".",
        db_path: Optional[str | Path] = None,
        ontology_path: Optional[str | Path] = None,
        timeout: int = 120,
    ) -> None:
        self.root = Path(root).resolve()
        self.db_path = Path(db_path or (self.root / "data" / "holarch" / "ledger.sqlite3"))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.ontology_path = ontology_path
        self.timeout = int(timeout)
        self._migrate()

    # ------------------------------------------------------------- ledger
    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.db_path))
        connection.row_factory = sqlite3.Row
        return connection

    def _migrate(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    ts REAL NOT NULL,
                    goal TEXT NOT NULL,
                    plan TEXT NOT NULL,
                    profile TEXT NOT NULL,
                    status TEXT NOT NULL,
                    gates TEXT NOT NULL,
                    ontology_head TEXT NOT NULL,
                    seconds REAL NOT NULL,
                    report TEXT NOT NULL
                )
                """
            )

    def history(self, limit: int = 10) -> List[Dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT run_id, ts, goal, plan, profile, status, seconds FROM runs ORDER BY ts DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [dict(row) for row in rows]

    def leaderboard(self) -> List[Dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT plan, status, COUNT(*) AS n FROM runs GROUP BY plan, status ORDER BY n DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    # --------------------------------------------------------------- run
    def run(self, goal: str, profile: Optional[str] = None) -> Dict[str, object]:
        plan = choose_plan(goal)
        if profile:
            plan.profile = str(profile)
        graph = HolonGraph(
            root=self.root,
            profile=plan.profile,
            timeout=self.timeout,
            ontology_path=self.ontology_path,
        )
        started = time.perf_counter()
        run_id = f"run-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        results: List[HolonResult] = []
        context: Dict[str, object] = {}

        for name in HOLONS:
            if name == "RECORD":
                result = graph.record(context, results)
            elif name == "GATE":
                result = graph.gate(context, results)
            elif name == "FRAME":
                result = graph.frame(goal)
            elif name == "MEASURE":
                result = graph.measure(context)
            elif name == "FALSIFY":
                result = graph.falsify(context)
            else:
                result = graph.attribute(context)
            results.append(result)
            if result.status == "FAIL" and name not in {"GATE", "RECORD"}:
                # Fail-closed: a broken fact must not reach the gate as if
                # it were measured. RECORD still runs so the failure is
                # written into the immutable layer.
                remaining = [holon for holon in HOLONS[HOLONS.index(name) + 1 :]]
                for holon in remaining:
                    if holon == "RECORD":
                        results.append(graph.record(context, results))
                    elif holon == "GATE":
                        results.append(graph.gate(context, results))
                    else:
                        results.append(
                            HolonResult(holon, "SKIP", f"skipped after {name} failed")
                        )
                break

        gate = next((row for row in results if row.name == "GATE"), None)
        status = "PROMOTE" if gate is not None and gate.status == "PASS" else "BLOCKED"
        seconds = time.perf_counter() - started
        chain = graph.ontology.verify()
        report = {
            "run_id": run_id,
            "goal": goal,
            "plan": plan.as_dict(),
            "status": status,
            "holons": [row.as_dict() for row in results],
            "gates": (gate.detail.get("gates") if gate else {}),
            "ontology": chain,
            "seconds": round(seconds, 3),
            "root": str(self.root),
        }
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO runs (run_id, ts, goal, plan, profile, status, gates, ontology_head, seconds, report)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    time.time(),
                    goal,
                    plan.name,
                    plan.profile,
                    status,
                    json.dumps(report["gates"], sort_keys=True),
                    str(chain.get("head", "")),
                    seconds,
                    json.dumps(report, default=str),
                ),
            )
        report["ledger"] = {"path": str(self.db_path), "run_id": run_id}
        return report
