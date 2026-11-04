"""Decisions D1 and D4: task-scoped retrieval and the composite service.

What makes these able to fail:

* a task-scoped query must never return a label from another task block;
* an empty task block must produce an empty read, not a fallback to the
  global store;
* the composite endpoint must ingest, resolve, abstain on an unknown rule,
  reject an unknown operator, and chain a verifiable receipt;
* the MCP tool must return the same object as the REST route.
"""

from __future__ import annotations

import asyncio

import torch
from fastapi.testclient import TestClient

from henri.holarch import mcp_server
from henri.holarch.config import HolarchConfig
from henri.holarch.pipeline import HenriHolarch
from henri.holarch.server import SidecarService, build_app, verify_receipt
from henri.holarch.tasks import build_tasks
from henri.holarch.zone_c import global_label, task_label_block


def _model(tmp_path, cfg, name="ddec.sqlite3"):
    torch.manual_seed(cfg.seed)
    return HenriHolarch(cfg, backend="sqlite", memory_path=str(tmp_path / name))


def _service(tmp_path, name="ddec_service.sqlite3") -> SidecarService:
    return SidecarService(
        HolarchConfig.small(),
        backend="sqlite",
        memory_path=str(tmp_path / name),
    )


# --------------------------------------------------------------- Decision D1
def test_task_scoped_retrieval_returns_only_the_task_block(tmp_path):
    cfg = HolarchConfig.small()
    model = _model(tmp_path, cfg)
    tasks = build_tasks(cfg)
    vision = tasks["vision"].sample(32, "train")
    language = tasks["language"].sample(32, "train")
    model.observe("vision", vision.x, vision.y)
    model.observe("language", language.x, language.y)
    wave = model.encode("vision", vision.x[:1])[0].detach().numpy()

    _, scoped = model.memory.context_tensor(wave, k=4, task="vision")
    _, unscoped = model.memory.context_tensor(wave, k=4, task=None)
    assert scoped.hits, "the vision block is populated"
    low, high = task_label_block("vision")
    assert all(low <= int(hit.label) < high for hit in scoped.hits)
    # The global read can see the language block; the scoped read must not.
    assert any(int(hit.label) >= high for hit in unscoped.hits)
    model.memory.close()


def test_empty_task_block_is_an_empty_read(tmp_path):
    cfg = HolarchConfig.small()
    model = _model(tmp_path, cfg)
    tasks = build_tasks(cfg)
    vision = tasks["vision"].sample(16, "train")
    model.observe("vision", vision.x, vision.y)
    wave = model.encode("vision", vision.x[:1])[0].detach().numpy()
    _, read = model.memory.context_tensor(wave, k=4, task="relational")
    assert read.hits == []
    assert read.s_max == 0.0
    model.memory.close()


def test_global_label_round_trip():
    assert global_label("vision", 3) == 3
    assert global_label("language", 0) == 10
    assert global_label("action", 2) == 108
    assert global_label("relational", 7) == 117
    for task in ("vision", "language", "action", "relational"):
        low, high = task_label_block(task)
        assert low < high


# --------------------------------------------------------------- Decision D4
def _partial_rules() -> list:
    """Operands 0-4 only: (d9, add, d9) stays unregistered."""
    rules = [
        {"a": a, "op": "add", "b": b, "result": (a + b) % 10}
        for a in range(5)
        for b in range(5)
    ]
    rules += [
        {"a": a, "op": "sub", "b": b, "result": abs(a - b)}
        for a in range(5)
        for b in range(5)
    ]
    return rules


def _rules() -> list:
    rules = [
        {"a": a, "op": "add", "b": b, "result": (a + b) % 10}
        for a in range(10)
        for b in range(10)
    ]
    rules += [
        {"a": a, "op": "sub", "b": b, "result": abs(a - b)}
        for a in range(10)
        for b in range(10)
    ]
    return rules


def test_composite_rest_roundtrip_and_receipts(tmp_path):
    service = _service(tmp_path)
    client = TestClient(build_app(service))

    ingest = client.post(
        "/v1/holarch/composite", json={"mode": "ingest", "rules": _partial_rules()}
    )
    assert ingest.status_code == 200
    assert ingest.json()["result"]["rules_ingested"] == 50.0
    assert verify_receipt(ingest.json()["receipt"])

    query = client.post(
        "/v1/holarch/composite", json={"mode": "query", "a": "d3", "op": "add", "b": "d4"}
    )
    assert query.status_code == 200
    body = query.json()["result"]
    assert body["result"] == "d7" and body["abstained"] is False
    assert verify_receipt(query.json()["receipt"])

    # An operator that was never registered must be a schema error, not a guess.
    bad = client.post(
        "/v1/holarch/composite", json={"mode": "query", "a": "d3", "op": "div", "b": "d4"}
    )
    assert bad.status_code == 400

    # A composite key that was never ingested must abstain.
    trap = client.post(
        "/v1/holarch/composite",
        json={"mode": "query", "a": "d9", "op": "add", "b": "d9"},
    )
    assert trap.status_code == 200
    assert trap.json()["result"]["abstained"] is True
    assert trap.json()["result"]["result"].startswith("ABSTAIN")

    chain = client.get("/v1/holarch/receipt?limit=32").json()
    assert chain["chain_valid"] is True
    assert all(verify_receipt(item) for item in chain["receipts"])
    service.close()


def test_composite_perceptual_query_grounds_and_resolves(tmp_path):
    service = _service(tmp_path, name="ddec_perceptual.sqlite3")
    client = TestClient(build_app(service))
    client.post("/v1/holarch/composite", json={"mode": "ingest", "rules": _rules()})
    pixels = [[0.0] * 8 for _ in range(8)]
    response = client.post(
        "/v1/holarch/composite",
        json={"mode": "query", "x0": pixels, "x1": pixels, "op": "add"},
    )
    result = response.json()["result"]
    assert result["grounded_operands"] is not None
    first, second, operator = result["grounded_operands"]
    assert operator == "add"
    assert result["abstained"] is False
    expected = f"d{(int(first[1:]) + int(second[1:])) % 10}"
    assert result["result"] == expected
    service.close()


def test_composite_mcp_tool_matches_the_rest_route(tmp_path):
    service = _service(tmp_path, name="ddec_mcp.sqlite3")
    mcp_server.set_service(service)
    try:
        asyncio.run(
            mcp_server.mcp._tool_manager.call_tool(
                "holarch_composite", {"mode": "ingest", "rules": _rules()}
            )
        )
        payload = asyncio.run(
            mcp_server.mcp._tool_manager.call_tool(
                "holarch_composite",
                {"mode": "query", "a": "d3", "op": "add", "b": "d4"},
            )
        )
        assert payload["ok"] is True
        assert payload["result"]["result"] == "d7"
        assert verify_receipt(payload["receipt"])
    finally:
        mcp_server.set_service(None)
        service.close()
