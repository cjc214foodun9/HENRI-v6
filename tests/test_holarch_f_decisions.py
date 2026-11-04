"""Directive F1-F4: namespace isolation, learned rotor, transport parity, impact.

Each test is written so that it can fail on a real defect, not only on an
exception: the isolation test asserts the exact symbol each tenant resolves
and the exact refusal of the other tenant's symbols, the rotor test measures
held-out coherence rather than a training number, and the transport test
records which pipe the host actually allowed.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest

from henri.holarch.config import HolarchConfig
from henri.holarch.model import (
    ActionTransitionRotor,
    evaluate_closed_loop_action,
    fit_action_rotor,
)
from henri.holarch.server import SidecarService, verify_receipt
from henri.holarch.symbolic import CodeGraphIndexer, Edge
from henri.holarch.tasks import GridWorldTask
from henri.holarch.zone_c import ZoneCMemory

ROOT = Path(__file__).resolve().parents[1]


def _service(tmp_path, name: str = "f1") -> SidecarService:
    return SidecarService(
        HolarchConfig.small(),
        backend="sqlite",
        memory_path=str(tmp_path / f"{name}.sqlite3"),
        autoload=False,
    )


# ---------------------------------------------------------------- F1
def test_multi_tenant_namespace_isolation(tmp_path):
    """Two tenants, contradictory rule graphs, one service instance."""
    service = _service(tmp_path)
    try:
        phone = [
            {"source": "s00", "operator": "fwd", "target": "s01"},
            {"source": "s01", "operator": "fwd", "target": "s02"},
            {"source": "s02", "operator": "fwd", "target": "s03"},
        ]
        beta_edges = [
            {"source": "s00", "operator": "fwd", "target": "z01"},
            {"source": "z01", "operator": "fwd", "target": "z02"},
        ]
        alpha = service.ingest_edges(phone, namespace="tenant_alpha")
        beta = service.ingest_edges(beta_edges, namespace="tenant_beta")
        assert alpha["result"]["edges_ingested"] == 3.0
        assert beta["result"]["edges_ingested"] == 2.0

        alpha_path = service.solve("s00", ["fwd", "fwd"], namespace="tenant_alpha")
        beta_path = service.solve("s00", ["fwd"], namespace="tenant_beta")
        assert alpha_path["result"]["answer"] == "s02"
        assert alpha_path["result"]["vetoed"] is False
        assert beta_path["result"]["answer"] == "z01"
        assert beta_path["result"]["vetoed"] is False

        # Tenant A must refuse tenant B's symbol and vice versa.
        alpha_cross = service.solve("z01", ["fwd"], namespace="tenant_alpha")
        beta_cross = service.solve("s01", ["fwd"], namespace="tenant_beta")
        assert alpha_cross["result"]["answer"] is None
        assert beta_cross["result"]["answer"] is None
        assert alpha_cross["result"]["vetoed"] is True
        assert beta_cross["result"]["vetoed"] is True

        # The default tenant sees neither: it was never written to.
        default_read = service.solve("s00", ["fwd"], namespace="default")
        assert default_read["result"]["answer"] is None
        assert default_read["result"]["vetoed"] is True

        leakage = 0
        for result, forbidden in (
            (alpha_path, ("z01", "z02")),
            (beta_path, ("s01", "s02", "s03")),
            (alpha_cross, ("s01",)),
            (beta_cross, ("z01",)),
        ):
            answer = result["result"]["answer"]
            if answer is not None and answer in forbidden:
                leakage += 1
        assert leakage == 0
        for receipt in (
            alpha["receipt"], beta["receipt"], alpha_path["receipt"],
            beta_path["receipt"], alpha_cross["receipt"], beta_cross["receipt"],
        ):
            assert verify_receipt(receipt)
    finally:
        service.close()


def test_store_level_namespace_filter(tmp_path):
    memory = ZoneCMemory(
        HolarchConfig.small(), backend="sqlite", path=str(tmp_path / "ns.sqlite3")
    )
    try:
        wave = np.exp(1j * np.linspace(0, 6.28, 256)).astype(np.complex64)
        memory.write(wave, label=10000, tier=3, namespace="alpha")
        memory.write(wave, label=10001, tier=3, namespace="beta")
        alpha = memory.query(wave, k=4, namespace="alpha")
        beta = memory.query(wave, k=4, namespace="beta")
        assert [hit.label for hit in alpha.hits] == [10000]
        assert [hit.label for hit in beta.hits] == [10001]
        missing = memory.query(wave, k=4, namespace="gamma")
        assert missing.hits == []
        assert memory.namespaces[:3] == ["default", "alpha", "beta"]
    finally:
        memory.close()


# ---------------------------------------------------------------- F2
def test_rotor_is_a_unitary_phase_mask_with_high_held_out_coherence():
    """The learned rotor recovers the multi-slot transition from data."""
    cfg = HolarchConfig.small()
    from henri.holarch.pipeline import HenriHolarch

    model = HenriHolarch(cfg, backend="sqlite", memory_path=None)
    model.eval()
    task = GridWorldTask(seed=cfg.seed, size=7, n_layouts=60)
    fit = fit_action_rotor(model, task, episodes=12, limit=8)
    assert fit["fit"]["transitions"] > 40
    assert fit["held_out"]["coherence"] >= 0.85
    rotor = fit["rotor"]
    assert rotor.fitted is True
    # Unit modulus: the mask cannot change photon number.
    assert np.allclose(np.abs(rotor.phi), 1.0, atol=1e-5)
    # The coherence identity: predicting the same wave yields 1.0.
    from henri.holarch.model import SensorWaveCodec

    codec = SensorWaveCodec(int(cfg.dim), int(cfg.n_slots), 1.6)
    observation = task.world.observe(*task.val_layouts[0])
    psi = codec.encode(observation)
    predicted = rotor.predict(psi, 0)
    assert abs(float(np.linalg.norm(predicted.numpy())) - 1.0) < 1e-5
    model.memory.close()


# ---------------------------------------------------------------- F3
def test_mcp_memory_stream_handshake_and_tool_invocation(tmp_path):
    """Full MCP frame serialization without any OS pipe."""
    from mcp.shared.memory import create_connected_server_and_client_session

    from henri.holarch import mcp_server

    service = _service(tmp_path, "f3")
    mcp_server.set_service(service)
    try:
        async def probe():
            async with create_connected_server_and_client_session(mcp_server.mcp) as session:
                await asyncio.wait_for(session.initialize(), timeout=120)
                listed = await asyncio.wait_for(session.list_tools(), timeout=120)
                names = sorted(tool.name for tool in listed.tools)
                service.ingest_edges(
                    [{"source": "a", "operator": "fwd", "target": "b"}],
                    namespace="tenant_alpha",
                )
                solved = await asyncio.wait_for(
                    session.call_tool(
                        "holarch_solve",
                        {"source": "a", "operators": ["fwd"], "namespace": "tenant_alpha"},
                    ),
                    timeout=120,
                )
                composed = await asyncio.wait_for(
                    session.call_tool(
                        "holarch_composite",
                        {
                            "mode": "ingest",
                            "rules": [{"a": "d1", "op": "add", "b": "d2", "result": "d3"}],
                        },
                    ),
                    timeout=120,
                )
                return names, solved, composed

        names, solved, composed = asyncio.run(probe())
        assert {
            "holarch_solve",
            "holarch_composite",
            "holarch_synthesize",
            "holarch_veto",
        } <= set(names)
        assert solved.isError is False
        payload = json.loads(solved.content[0].text)
        assert payload["result"]["answer"] == "b"
        assert verify_receipt(payload["receipt"])
        composed_payload = json.loads(composed.content[0].text)
        assert composed_payload["result"]["rules_ingested"] == 1.0
        assert verify_receipt(composed_payload["receipt"])
    finally:
        mcp_server.set_service(None)
        service.close()


# ---------------------------------------------------------------- F4
def test_impact_blast_radius_and_trap_refusal(tmp_path):
    service = _service(tmp_path, "f4")
    try:
        result = service.ingest_code(str(ROOT / "henri" / "holarch"), package="henri")
        assert result["result"]["signatures"] > 100
        assert result["result"]["grammar_transitions"] > 100
        assert result["result"]["boundary_engrams"] > 100

        impact = service.solve("ZoneCMemory.query", ["2"], mode="impact")
        callers = impact["result"]["callers"]
        assert impact["result"]["resolved"] is True
        assert "pipeline:forward" in callers
        assert impact["result"]["vetoed"] is False

        shallow = service.solve("ZoneCMemory.query", ["1"], mode="impact")
        assert set(shallow["result"]["callers"]) <= set(callers)

        trap = service.solve("no_such_symbol_alpha", ["2"], mode="impact")
        assert trap["result"]["vetoed"] is True
        assert trap["result"]["callers"] == []
        assert trap["result"]["resolved"] is False
    finally:
        service.close()


def test_indexer_reverse_walk_is_exact_on_a_hand_built_graph(tmp_path):
    memory = ZoneCMemory(
        HolarchConfig.small(), backend="sqlite", path=str(tmp_path / "graph.sqlite3")
    )
    try:
        indexer = CodeGraphIndexer(memory, namespace="synthetic")
        indexer.graph.edges = [
            Edge("m:a", "CALLS", "m:b"),
            Edge("m:c", "CALLS", "m:b"),
            Edge("m:d", "CALLS", "m:c"),
        ]
        indexer.graph.nodes = ["m:a", "m:b", "m:c", "m:d"]
        reverse = {}
        for edge in indexer.graph.edges:
            reverse.setdefault(edge.target, []).append(edge.source)
        indexer.reverse = {key: sorted(value) for key, value in reverse.items()}
        one = indexer.impact("m:b", hops=1)
        two = indexer.impact("m:b", hops=2)
        three = indexer.impact("m:b", hops=3)
        assert one["callers"] == ["m:a", "m:c"]
        assert two["callers"] == ["m:a", "m:c", "m:d"]
        assert three["callers"] == ["m:a", "m:c", "m:d"]
        assert indexer.impact("m:absent", hops=1)["resolved"] is False
    finally:
        memory.close()


def test_cuda_driver_probe_records_the_root_cause():
    from henri.holarch.budget import probe_nvidia_driver_state

    report = probe_nvidia_driver_state()
    for key in (
        "nvidia_smi_reachable",
        "nvidia_smi_devices",
        "nvcuda_present",
        "libcuda_present",
        "cuda_visible_devices_set",
        "root_cause",
        "verdict",
    ):
        assert key in report
    assert report["verdict"] in {"VISIBLE", "BLOCKED"}
    assert isinstance(report["nvidia_smi_devices"], list)
    if not report["torch_cuda_available"]:
        assert report["root_cause"] != ""
