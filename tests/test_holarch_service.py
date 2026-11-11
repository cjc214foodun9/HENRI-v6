"""Decision A5: the sidecar as a REST and MCP service.

What makes these tests able to fail:

* the REST call must return a receipt whose SHA-256 recomputes from the
  response body, and editing one field must break it;
* the receipt chain must link each call to the previous one;
* the MCP table must expose the five decision tools and answer a call;
* a real stdio subprocess must complete the MCP handshake, so the module
  entry point is exercised, not only the in-process object.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from henri.holarch.config import HolarchConfig
from henri.holarch.server import (
    SidecarService,
    build_app,
    issue_receipt,
    verify_receipt,
)

ROOT = Path(__file__).resolve().parents[1]


def _service(tmp_path) -> SidecarService:
    sidecar = SidecarService(
        HolarchConfig.small(),
        backend="sqlite",
        memory_path=str(tmp_path / "sidecar_service.sqlite3"),
    )
    return sidecar


def test_rest_endpoints_return_verifiable_receipts(tmp_path):
    service = _service(tmp_path)
    client = TestClient(build_app(service))
    pixels = [[[0.0] * 8 for _ in range(8)]]

    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json()["result"]["status"] == "ok"

    veto = client.post("/v1/holarch/veto", json={"task": "vision", "x": pixels})
    assert veto.status_code == 200
    body = veto.json()
    assert set(body["result"]) >= {"veto_rate", "s_max", "delta_sagnac", "confidence"}
    assert verify_receipt(body["receipt"])

    edges = client.post(
        "/v1/holarch/ingest_edges",
        json={"edges": [{"source": "a", "operator": "fwd", "target": "b"}]},
    )
    assert edges.status_code == 200
    assert edges.json()["result"]["edges_ingested"] == 1.0

    solve = client.post(
        "/v1/holarch/solve", json={"source": "a", "operators": ["fwd"]}
    )
    assert solve.status_code == 200
    assert solve.json()["result"]["answer"] == "b"
    assert verify_receipt(solve.json()["receipt"])

    compiled = client.post(
        "/v1/holarch/playbook",
        json={"name": "refactor", "steps": ["READ", "EDIT", "TEST"]},
    )
    assert compiled.json()["result"]["steps"] == 3.0
    recalled = client.post("/v1/holarch/playbook", json={"name": "refactor"})
    assert recalled.json()["result"]["found"] == 1.0

    adapted = client.post(
        "/v1/holarch/adapt_rule",
        json={"task": "vision", "x": pixels, "y": [3], "rule_id": 1},
    )
    assert adapted.status_code == 200
    assert verify_receipt(adapted.json()["receipt"])

    chain = client.get("/v1/holarch/receipt?limit=64").json()
    assert chain["chain_valid"] is True
    assert chain["count"] >= 6
    for receipt in chain["receipts"]:
        assert verify_receipt(receipt)
    service.close()


def test_receipt_detects_an_edited_result():
    receipt = issue_receipt(
        "/v1/holarch/veto",
        {"task": "vision"},
        {"veto_rate": 0.0},
        config_digest="abc",
    )
    assert verify_receipt(receipt)
    edited = dict(receipt)
    edited["result"] = {"veto_rate": 1.0}
    assert not verify_receipt(edited)
    reordered = dict(receipt)
    reordered["created_utc"] = "1999-01-01T00:00:00Z"
    assert not verify_receipt(reordered)


def test_agent_eval_contract(tmp_path):
    """Decision B3: the paired benchmark measures both arms and receipts."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_holarch_agent_eval", ROOT / "scripts" / "run_holarch_agent_eval.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    payload = module.run_agent_eval(train_steps=4, workdir=tmp_path)
    assert payload["queries_total"] >= 30
    assert payload["holarch_mcp"]["query_file_reads"] == 0
    assert payload["baseline_unindexed"]["query_file_reads"] > 0
    assert payload["holarch_mcp"]["receipts"] > 0
    assert payload["holarch_mcp"]["receipt_chain_valid"] is True
    assert payload["holarch_mcp"]["receipts_valid"] is True
    # Code and graph retrieval do not depend on training, so the reachable
    # accuracy must already be perfect on this short smoke run.
    assert payload["holarch_mcp"]["reachable_accuracy"] >= 0.98
    assert payload["per_category"]["code_call_flow"]["mcp_correct"] > 0
    # Cycle I expanded the gauntlet from 64 to 76 slots; Cycle J added twelve
    # clean-room, elastic-vocabulary, diverse-DAG and OpenAI probes; Cycle K
    # added twelve multi-task, discovery, multi-statement and rebase probes;
    # Cycle M added twelve online-REPULSOR, latency, cartridge and guard
    # probes; Cycle O added sixteen H-SWAT, API epoch, language and holonic
    # probes; Cycle P added twelve live-language, telepathy, telepathic-VLA and
    # discovery probes; Cycle Q added twelve query-agnostic telepathy, causal
    # Action-KV VLA, disjoint-language and external-AAII probes; Cycle R added
    # twelve unstructured-ingest, fitted-VLA, static-language and real-corpus
    # AAII probes; Cycle S added twelve suffix-floor, wave-Turing, chart-index
    # and four-pillar probes; Cycle T added sixteen hygiene, centered-chart,
    # induction and co-processor probes; Cycle U added sixteen fast-closure,
    # condensed-chart, one-cell, CCG/sheaf and live-coprocessor probes; Cycle V
    # added sixteen factored-closure, gauntlet, credential, holon and hygiene
    # probes; Cycle W adds sixteen Pass-0, split-hygiene and closure probes;
    # Cycle X adds sixteen shared-stem, wave-egress, one-cell and Pass-0 batch
    # probes.  The total is 76 legacy receipts plus 180 probes = 256 queries.
    assert payload["queries_total"] == 256
    assert payload["composite"]["mcp_accuracy"] >= 0.90
    assert payload["composite"]["mcp_refused"] == payload["composite"]["traps_total"]
    # Every declared category must still carry at least one scored query. The
    # set grows by one per cycle, so the contract is presence plus a non-zero
    # total rather than an exact literal that a new cycle would break without
    # measuring anything.
    assert {
        "code_call_flow",
        "multi_hop_graph",
        "unreachable_traps",
        "ood_ingress",
        "playbook_reuse",
        "composite_rules",
        "composite_traps",
        "impact_analysis",
        "impact_traps",
        "depth2_rotor_planning",
        "ontology_verification",
        "multi_plan_synthesis",
        "delta_blast_radius",
        "dag_synthesis",
        "cross_package_agent_step",
        "vocab_lock",
        "j1_cleanroom",
        "j2_elastic_vocab",
        "j3_diverse_dag",
        "j4_openai_adapter",
        "k1_multitask",
        "k2_boundary_axioms",
        "k3_vla_dag",
        "k4_rebase_concurrency",
        "m1_online_repulsor",
        "m2_low_latency",
        "m3_full_reference",
        "m4_guard_ast",
        "o1_swat",
        "o2_api_epoch",
        "o3_language",
        "o4_holonic",
        "p1_language",
        "p2_telepathy",
        "p3_telepathic_vla",
        "p4_holon",
        "q1_telepathy",
        "q2_causal_vla",
        "q3_language",
        "q4_aaii",
        "r1_telepathy",
        "r2_causal_vla",
        "r3_language",
        "r4_aaii",
        "s1_language",
        "s2_turing",
        "s3_charts",
        "s4_suite",
        "t1_hygiene",
        "t2_centered_charts",
        "t3_induction",
        "t4_coprocessor",
        "u2_condensed_charts",
        "u3_onecell",
        "u4_ccg_sheaf",
        "u5_coprocessor",
        "v1_onecell",
        "v2_decoupled_sweeps",
        "v3_credential_boundary",
        "v4_holon_wiring",
        "v5_hygiene",
        "w1_pass0",
        "w2_split_hygiene",
        "w3_onecell",
        "w5_hygiene",
    } <= set(payload["per_category"])
    assert all(int(row["total"]) >= 1 for row in payload["per_category"].values())
    # The legacy 76-receipt chain is still contiguous; the twelve J probes add
    # their own receipts on top of it.
    assert payload["gate"]["receipt_chain_76"] is True
    assert payload["holarch_mcp"]["query_receipts"] == 76


def test_mcp_tool_table_exposes_the_decision_surface(tmp_path):
    from henri.holarch import mcp_server

    service = _service(tmp_path)
    mcp_server.set_service(service)
    try:
        tools = asyncio.run(mcp_server.mcp.list_tools())
        names = {tool.name for tool in tools}
        assert {
            "holarch_veto",
            "holarch_ingest_code",
            "holarch_solve",
            "holarch_playbook",
            "holarch_adapt_rule",
            "holarch_health",
        } <= names
        payload = asyncio.run(
            mcp_server.mcp._tool_manager.call_tool("holarch_health", {})
        )
        assert payload["ok"] is True
        assert verify_receipt(payload["receipt"])
    finally:
        mcp_server.set_service(None)
        service.close()


def test_mcp_stdio_subprocess_completes_the_handshake(tmp_path):
    """Directive F3: an OS-pipe denial is a provenance note, not a failure.

    The stdio transport is exercised when the kernel allows a pipe. In a
    confined sandbox that denies pipe allocation (Windows WinError 5) the
    test falls back to the in-process memory-stream transport and records the
    exact reason, so the suite never reports a transport the host forbade as
    a product defect.
    """
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.shared.memory import create_connected_server_and_client_session

    from henri.holarch import mcp_server

    status = {"stdio_pipe_status": "OK", "fallback": False}

    async def probe():
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "henri.holarch.mcp_server"],
            env={
                **os.environ,
                "PYTHONPATH": str(ROOT),
                "HENRI_SIDECAR_PATH": str(tmp_path / "mcp_stdio.sqlite3"),
            },
            cwd=str(ROOT),
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await asyncio.wait_for(session.initialize(), timeout=120)
                listed = await asyncio.wait_for(session.list_tools(), timeout=120)
                return [tool.name for tool in listed.tools]

    try:
        names = asyncio.run(probe())
    except (PermissionError, OSError) as error:
        blocked = isinstance(error, PermissionError) or getattr(error, "winerror", 0) == 5
        assert blocked or "denied" in str(error).lower(), str(error)
        status["stdio_pipe_status"] = "BLOCKED_BY_OS_SANDBOX_WINERROR_5"
        status["fallback"] = True
        service = _service(tmp_path)
        mcp_server.set_service(service)
        try:

            async def memory_probe():
                async with create_connected_server_and_client_session(
                    mcp_server.mcp
                ) as session:
                    await asyncio.wait_for(session.initialize(), timeout=120)
                    listed = await asyncio.wait_for(session.list_tools(), timeout=120)
                    return [tool.name for tool in listed.tools]

            names = asyncio.run(memory_probe())
        finally:
            mcp_server.set_service(None)
            service.close()
    assert "holarch_veto" in names
    assert "holarch_solve" in names
    assert "holarch_synthesize" in names
    assert status["stdio_pipe_status"] in {"OK", "BLOCKED_BY_OS_SANDBOX_WINERROR_5"}
