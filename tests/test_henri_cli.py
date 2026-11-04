"""Tests for the HENRI v6 command line and the six-holon orchestrator."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from henri.cli import REPO_ROOT, envelope, lint_text, main
from henri.holon import HOLONS, HolonGraph
from henri.ontology import GENESIS, OntologyLedger
from henri.orchestrator import Orchestrator, choose_plan


# ------------------------------------------------------------------ ontology
def test_ontology_appends_and_verifies(tmp_path):
    ledger = OntologyLedger(tmp_path / "ontology.jsonl")
    first = ledger.append("measurement", "vision", "accuracy", "0.956", {"receipt": "x"})
    second = ledger.append("gate", "d4", "passed", "true")
    assert first.prev == GENESIS
    assert second.prev == first.digest
    assert ledger.verify()["ok"] is True
    assert ledger.verify()["length"] == 2


def test_ontology_detects_a_rewritten_row(tmp_path):
    path = tmp_path / "ontology.jsonl"
    ledger = OntologyLedger(path)
    ledger.append("measurement", "vision", "accuracy", "0.956")
    ledger.append("measurement", "action", "accuracy", "0.775")
    rows = path.read_text(encoding="utf-8").splitlines()
    row = json.loads(rows[0])
    row["object"] = "0.999"
    rows[0] = json.dumps(row, sort_keys=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    result = ledger.verify()
    assert result["ok"] is False
    assert result["broken_at"] == 0


def test_ontology_is_append_only(tmp_path):
    ledger = OntologyLedger(tmp_path / "ontology.jsonl")
    ledger.append("measurement", "a", "b", "c")
    before = ledger.entries()
    ledger.append("measurement", "d", "e", "f")
    after = ledger.entries()
    assert after[: len(before)] == before


# --------------------------------------------------------------------- ste100
def test_ste100_flags_the_planted_faults():
    report = lint_text("The valve has been utilized; it was tested by the team.")
    rules = {violation["rule"] for violation in report["violations"]}
    assert {"semicolon", "perfect_tense", "banned_word", "passive"} <= rules
    assert report["clean"] is False


def test_ste100_accepts_a_clean_procedural_sentence():
    report = lint_text("Open the valve. Record the pressure. Close the valve.")
    assert report["clean"] is True


# ---------------------------------------------------------------- CLI surface
def _run(capsys, argv):
    code = main(argv)
    captured = capsys.readouterr().out
    return code, json.loads(captured)


def test_doctor_returns_the_tool_envelope(capsys):
    code, payload = _run(capsys, ["doctor", "--json"])
    assert {"ok", "exitCode", "summary", "result"} <= set(payload)
    assert payload["result"]["checks"]["numpy"]["available"] is True
    assert payload["exitCode"] == code


def test_ste100_command_exits_one_on_a_violation(capsys):
    code, payload = _run(capsys, ["ste100", "--text", "The pump was started;", "--json"])
    assert code == 1
    assert payload["ok"] is False
    assert payload["result"]["violations"]


def test_zone_check_denies_an_uncovered_module(capsys):
    code, payload = _run(capsys, ["zone-check", "--path", "henri/cli.py", "--zone", "2", "--json"])
    assert code in (0, 1)
    assert payload["result"]["zone"] == 2


def test_vast_dry_run_rents_nothing(capsys):
    code, payload = _run(capsys, ["vast", "--json"])
    assert code == 0
    assert payload["result"]["rented"] is False


def test_vast_live_without_permission_is_blocked(capsys):
    code, payload = _run(capsys, ["vast", "--live", "--json"])
    assert code == 1
    assert payload["result"]["live"] is True
    assert payload["result"]["rented"] is False


# --------------------------------------------------------------------- holon
def test_plan_routing_selects_a_promotion_plan():
    assert choose_plan("close gate D4 and promote").name == "promotion"
    assert choose_plan("measure the noise probe").name == "measurement"
    assert choose_plan("hello").name == "survey"


def test_six_holon_graph_promotes_on_a_healthy_repo(tmp_path):
    graph = HolonGraph(root=REPO_ROOT, profile="quick", timeout=120, ontology_path=tmp_path / "o.jsonl")
    context = {}
    results = []
    results.append(graph.frame("close gate D4 and verify the evidence chain"))
    results.append(graph.measure(context))
    results.append(graph.falsify(context))
    results.append(graph.attribute(context))
    results.append(graph.record(context, results))
    results.append(graph.gate(context, results))
    assert [row.name for row in results] == list(HOLONS)
    assert results[1].status == "PASS", results[1].detail
    assert results[2].status == "PASS"
    assert results[-1].status == "PASS", results[-1].detail


def test_orchestrator_records_a_run(tmp_path):
    orchestrator = Orchestrator(
        root=REPO_ROOT,
        db_path=tmp_path / "ledger.sqlite3",
        ontology_path=tmp_path / "ontology.jsonl",
        timeout=120,
    )
    report = orchestrator.run("close gate D4", profile="quick")
    assert report["status"] in {"PROMOTE", "BLOCKED"}
    assert len(report["holons"]) == len(HOLONS)
    history = orchestrator.history()
    assert history and history[0]["run_id"] == report["run_id"]


def test_module_entry_point_exists():
    assert (REPO_ROOT / "henri" / "__main__.py").exists()
    completed = subprocess.run(
        [sys.executable, "-m", "henri", "vast", "--json"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0
    assert json.loads(completed.stdout)["result"]["rented"] is False
