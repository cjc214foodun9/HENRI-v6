"""Cycle AF2 unit tests: deterministic OntologyVerifier-governed DeliverableComposer."""

from __future__ import annotations

import json

from henri.holarch.deliverable_composer import DeliverableComposer


def _seed_workspace(tmp_path):
    (tmp_path / "metrics.csv").write_text(
        "region,calls,cost_usd\n"
        "west,100,1.25\n"
        "east,250,3.75\n",
        encoding="utf-8",
    )
    (tmp_path / "summary.md").write_text(
        "# Summary\n- Owner: HolarchOps\n- Target: 99.9\n",
        encoding="utf-8",
    )
    (tmp_path / "events.jsonl").write_text(
        json.dumps({"level": "INFO", "component": "gateway", "latency": 12}) + "\n"
        + json.dumps({"level": "WARN", "component": "sandbox", "latency": 8}) + "\n",
        encoding="utf-8",
    )
    return tmp_path


def test_deliverable_composer_exact_zero_error_and_ontology_verified(tmp_path):
    ws = _seed_workspace(tmp_path)
    composer = DeliverableComposer(ws)
    res = composer.compose(
        title="Operations Audit",
        required_files=("metrics.csv", "summary.md", "events.jsonl"),
        required_metrics=("metrics.calls.sum", "metrics.cost_usd.sum", "events.latency.sum"),
        required_entities=("west", "holarchops", "gateway"),
    )
    assert res.status == "EMITTED"
    assert res.emitted is True
    assert res.max_numeric_error == 0.0
    assert res.all_claims_supported is True
    assert res.exact_metrics["metrics.calls.sum"] == "350"
    assert res.exact_metrics["metrics.cost_usd.sum"] == "5"
    assert res.exact_metrics["events.latency.sum"] == "20"
    assert len(res.sections) == 4
    assert all(sec.verified for sec in res.sections)
    assert res.deliverable_sha256


def test_deliverable_composer_abstains_fail_closed_on_ungrounded_prompts(tmp_path):
    ws = _seed_workspace(tmp_path)
    composer = DeliverableComposer(ws)

    # Missing file
    r1 = composer.compose(title="Missing", required_files=("metrics.csv", "ghost.csv"))
    assert r1.status == "ABSTAIN" and not r1.emitted and r1.markdown == ""

    # Unknown metric
    r2 = composer.compose(title="Unknown", required_metrics=("metrics.fake_col.sum",))
    assert r2.status == "ABSTAIN" and not r2.emitted and r2.markdown == ""

    # Ungrounded entity
    r3 = composer.compose(title="Entity", required_entities=("lunar-base-7",))
    assert r3.status == "ABSTAIN" and not r3.emitted and r3.markdown == ""

    # Contradicted quantitative claim
    r4 = composer.compose(
        title="Contradicted",
        claims_to_verify=("CSV metrics.csv column calls across 2 rows has sum 99999.",),
    )
    assert r4.status == "ABSTAIN" and not r4.emitted and r4.markdown == ""
