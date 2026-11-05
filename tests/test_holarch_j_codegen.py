"""Directive J3: a first-class diverse playbook library and typed DAG battery.

These are unit-level checks of the library, the guarded splicer and the
fail-closed type gate. The battery itself executes every emitted source
(ast.parse + compile + call) and compares against an independent reference;
the tests below only assert its measured counts and the mechanisms behind them.
"""

from __future__ import annotations

import ast
import hashlib
import json

import pytest

from henri.holarch.code_gen import (
    PLAYBOOK_FAMILIES,
    PLAYBOOK_LIBRARY,
    Plan,
    TypeCompatibilityGate,
    _abstract_signature,
    compose_plan_dag,
    playbook_catalog,
    run_diverse_dag_battery,
)

NOVEL_NAMES = ("boundary_probe", "raw_score")


def _plan(stage: str) -> Plan:
    """A plan with no cached slots, so the gate infers from the source."""
    source = PLAYBOOK_LIBRARY[str(stage)]
    return Plan(
        name=str(stage),
        intent="",
        source=source,
        digest=hashlib.sha256(source.encode("utf-8")).hexdigest(),
    )


@pytest.fixture(scope="module")
def battery():
    return run_diverse_dag_battery(seed=0)


def test_playbook_library_has_five_distinct_typed_families():
    catalog = playbook_catalog()
    assert len(catalog["sources"]) == 15
    assert len(PLAYBOOK_LIBRARY) == 15
    assert len(catalog["families"]) >= 5
    assert set(catalog["families"]) == set(PLAYBOOK_FAMILIES)
    triples = set()
    for family, spec in PLAYBOOK_FAMILIES.items():
        stages = list(spec["stages"])
        assert len(stages) == 3
        assert len(set(stages)) == 3
        assert all(stage in PLAYBOOK_LIBRARY for stage in stages)
        entry = catalog["families"][family]
        assert [list(pair) for pair in entry["types"]] == [
            [catalog["signatures"][stage]["input"], catalog["signatures"][stage]["output"]]
            for stage in stages
        ]
        triples.add(entry["topology"])
    assert len(triples) >= 5
    assert sorted(triples) == catalog["topologies"]
    assert len(str(catalog["digest"])) == 64
    assert catalog["digest"] == playbook_catalog()["digest"]
    for source in catalog["sources"].values():
        compile(ast.parse(source), "<playbook>", "exec")


def test_every_family_chain_passes_the_type_gate():
    gate = TypeCompatibilityGate()
    for family, spec in PLAYBOOK_FAMILIES.items():
        verdict = gate.check_chain([_plan(stage) for stage in spec["stages"]])
        assert verdict["ok"] is True, (family, verdict["reason"])


def test_declared_signatures_match_the_stage_sources():
    catalog = playbook_catalog()
    for name, source in PLAYBOOK_LIBRARY.items():
        declared = catalog["signatures"][name]
        inferred = _abstract_signature(source)
        assert inferred["arity"] == 1
        assert [item["type"] for item in inferred["args"]] == [[declared["input"]]]
        assert inferred["returns"] == [declared["output"]]


def test_base_tasks_24_of_24(battery):
    assert battery["base"]["passed"] == 24
    assert battery["base"]["total"] == 24
    for row in battery["base"]["rows"]:
        assert row["matched"] is True, row
        assert row["parsed"] and row["compiled"] and row["executed"], row


def test_single_plan_novel_bindings_12_of_12(battery):
    assert battery["single"]["passed"] == 12
    assert battery["single"]["total"] == 12
    names = set()
    for row in battery["single"]["rows"]:
        assert row["matched"] is True, row
        assert row["novel_func"].startswith("j3s")
        assert row["novel_func"] not in PLAYBOOK_LIBRARY
        assert row["novel_arg"].startswith("payload_")
        names.add(row["novel_func"])
    assert len(names) == 12


def test_two_plan_pipelines_8_of_8(battery):
    assert battery["two"]["passed"] == 8
    assert battery["two"]["total"] == 8
    for row in battery["two"]["rows"]:
        assert row["matched"] is True, row
        assert len(row["stages"]) == 2


def test_three_plan_diverse_10_of_10_across_five_topologies(battery):
    assert battery["three"]["passed"] == 10
    assert battery["three"]["total"] == 10
    for row in battery["three"]["rows"]:
        assert row["matched"] is True, row
        assert len(row["stages"]) == 3
    topologies = battery["three_topologies"]
    assert topologies == sorted(set(topologies))
    assert len(topologies) >= 5
    assert topologies == playbook_catalog()["topologies"]
    assert {row["family"] for row in battery["three"]["rows"]} == set(PLAYBOOK_FAMILIES)


def test_guarded_templates_survive_splicing_and_rebinding():
    guarded = []
    for name, source in PLAYBOOK_LIBRARY.items():
        if any(isinstance(node, ast.IfExp) for node in ast.walk(ast.parse(source))):
            guarded.append(name)
    assert len(guarded) >= 2
    assert {"clamp_unit", "normalize_scores", "max_or_zero"}.issubset(set(guarded))

    verdict = compose_plan_dag(
        [_plan("caller_names"), _plan("sequence_length"), _plan("clamp_unit")],
        {"func": NOVEL_NAMES[0], "args": [NOVEL_NAMES[1]], "numbers": [2.0, 5.0]},
    )
    assert verdict["ok"] is True, verdict["reason"]
    source = str(verdict["source"])
    tree = ast.parse(source)
    assert any(isinstance(node, ast.IfExp) for node in ast.walk(tree))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    assert function.name == NOVEL_NAMES[0]
    assert [argument.arg for argument in function.args.args] == [NOVEL_NAMES[1]]
    assert "2.0" in source and "5.0" in source
    scope: dict = {}
    exec(source, scope)
    probe = scope[NOVEL_NAMES[0]]
    assert probe("a\nb\nc\n") == 3.0
    assert probe("a\nb\nc\nd\ne\nf\ng\n") == 5.0


def test_single_plan_binding_uses_novel_names_and_constants():
    verdict = compose_plan_dag(
        [_plan("clamp_unit")],
        {"func": NOVEL_NAMES[0], "args": [NOVEL_NAMES[1]], "numbers": [-2.0, 7.0]},
    )
    assert verdict["ok"] is True, verdict["reason"]
    scope: dict = {}
    exec(str(verdict["source"]), scope)
    probe = scope[NOVEL_NAMES[0]]
    assert probe(9.0) == 7.0
    assert probe(-9.0) == -2.0


def test_type_traps_refuse_before_any_source_is_emitted(battery):
    rows = battery["traps"]["rows"]
    assert battery["traps"]["refused"] == 6
    assert battery["traps"]["total"] == 6
    assert len(rows) == 6
    assert {row["conflict"] for row in rows} == {
        "boolean->numeric_sequence",
        "mapping->numeric",
        "text->numeric_sequence",
        "numeric_sequence->mapping",
        "sequence->mapping",
        "boolean->numeric",
    }
    for row in rows:
        assert row["refused"] is True, row
        assert row["source_empty"] is True, row
        assert row["reason"].startswith("type mismatch"), row
    gate = TypeCompatibilityGate()
    assert gate.check(_plan("exceeds_tau"), _plan("mean_of"))["ok"] is False
    assert gate.check(_plan("invert_index"), _plan("clamp_unit"))["ok"] is False


def test_total_60_of_60_and_zero_stationary_drift(battery):
    total = battery["total"]
    assert total["total"] == 60
    assert total["passed"] == 60
    section_total = sum(
        battery[name]["total"] for name in ("base", "single", "two", "three", "traps")
    )
    assert section_total == total["total"]
    assert battery["drift"] == 0.0
    assert isinstance(battery["drift"], float)


def test_battery_is_deterministic_for_the_same_seed():
    first = run_diverse_dag_battery(seed=11)
    second = run_diverse_dag_battery(seed=11)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["total"] == {"passed": 60, "total": 60}
    other = run_diverse_dag_battery(seed=12)
    assert other["total"] == first["total"]
    assert other["three_topologies"] == first["three_topologies"]
    assert json.dumps(first, sort_keys=True) != json.dumps(other, sort_keys=True)
