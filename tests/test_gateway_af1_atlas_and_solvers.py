"""Cycle AF0/AF1 tests: packed RiemannianChartAtlas + Gateway REST & MCP solver wiring."""

from __future__ import annotations

import asyncio
import numpy as np
from fastapi.testclient import TestClient

from henri.gateway.app import create_gateway_app
from henri.gateway.config import GatewayConfig
from henri.gateway.mcp_mount import build_mcp_server
from henri.gateway.service import UnifiedGatewayService
from henri.gateway.signing import verify_signed_receipt
from henri.holarch.riemannian_atlas import RiemannianChartAtlas


def _make_gateway(tmp_path):
    cfg = GatewayConfig(
        zone_c_backend="sqlite",
        sqlite_path=str(tmp_path / "gw.sqlite3"),
        zone_c_sqlite_path=str(tmp_path / "zonec.sqlite3"),
        signing_key_path=str(tmp_path / "ed25519.pem"),
        require_auth=True,
        holarch_autoload=False,
    )
    svc = UnifiedGatewayService(config=cfg)
    app = create_gateway_app(svc, mount_mcp=True)
    return svc, app


def test_packed_riemannian_atlas_parity_and_online_recall():
    rng = np.random.default_rng(77)
    dim, domains, n_world, n_online = 64, 16, 1600, 320
    centers = rng.standard_normal((domains, dim)) + 1j * rng.standard_normal((domains, dim))
    centers /= np.linalg.norm(centers, axis=-1, keepdims=True)

    def sample(n):
        lbl = rng.integers(0, domains, size=n)
        v = centers[lbl] + 0.30 * (rng.standard_normal((n, dim)) + 1j * rng.standard_normal((n, dim)))
        return v / np.linalg.norm(v, axis=-1, keepdims=True)

    world = sample(n_world)
    online = sample(n_online)
    atlas = RiemannianChartAtlas(rank=8, capacity=128, b_stage=32, min_cluster=6, top_charts=2, top_charts_max=4)
    atlas.build(world, list(range(n_world)), namespace="ns")

    for q_idx in rng.choice(n_world, size=40, replace=False):
        flat = int(np.argmax(np.real(world @ np.conj(world[q_idx]))))
        row_id, _score, _scored, _det = atlas.query_top1(world[q_idx], namespace="ns")
        assert row_id == flat

    for off, vec in enumerate(online):
        atlas.observe(vec, n_world + off, namespace="ns")
    atlas.finalize()

    hits = 0
    for off in range(60):
        row_id, _score, _scored, _det = atlas.query_top1(online[off], namespace="ns")
        if row_id == n_world + off:
            hits += 1
    assert hits == 60
    assert atlas.stats()["boundary_ratio"] <= 0.08


def test_gateway_atlas_rest_signed_receipts_and_tenant_isolation(tmp_path):
    svc, app = _make_gateway(tmp_path)
    try:
        key_a, _rec_a = svc.auth.create_key(tenant_id="tenant_a", allowed_namespaces=["alpha"])
        key_b, _rec_b = svc.auth.create_key(tenant_id="tenant_b", allowed_namespaces=["beta"])
        client = TestClient(app)

        rng = np.random.default_rng(42)
        vecs = rng.standard_normal((16, 8)).tolist()

        res_build = client.post(
            "/v1/holarch/atlas/build",
            headers={"Authorization": f"Bearer {key_a}"},
            json={"vectors": vecs, "namespace": "alpha", "capacity": 8},
        )
        assert res_build.status_code == 200
        body_b = res_build.json()
        assert body_b["result"]["evidence_label"] == "OBSERVED"
        assert body_b["result"]["compute_device"] == svc.compute_device
        assert verify_signed_receipt(body_b["receipt"], svc.signer.jwks())[0] is True

        # Cross-tenant access must fail closed with 403.
        res_forbidden = client.post(
            "/v1/holarch/atlas/query",
            headers={"Authorization": f"Bearer {key_b}"},
            json={"vector": vecs[0], "namespace": "alpha"},
        )
        assert res_forbidden.status_code == 403

        # Observe online write and finalize.
        online_vec = (rng.standard_normal(8) * 2.5).tolist()
        res_obs = client.post(
            "/v1/holarch/atlas/observe",
            headers={"Authorization": f"Bearer {key_a}"},
            json={"vector": online_vec, "row": 999, "finalize": True, "namespace": "alpha"},
        )
        assert res_obs.status_code == 200
        assert verify_signed_receipt(res_obs.json()["receipt"], svc.signer.jwks())[0] is True

        # Query online write back with exact top-1 match.
        res_q = client.post(
            "/v1/holarch/atlas/query",
            headers={"Authorization": f"Bearer {key_a}"},
            json={"vector": online_vec, "k": 1, "namespace": "alpha"},
        )
        assert res_q.status_code == 200
        q_body = res_q.json()
        assert q_body["result"]["hits"][0]["row"] == 999
        assert verify_signed_receipt(q_body["receipt"], svc.signer.jwks())[0] is True

        res_stats = client.get(
            "/v1/holarch/atlas/stats?namespace=alpha",
            headers={"Authorization": f"Bearer {key_a}"},
        )
        assert res_stats.status_code == 200
        assert res_stats.json()["result"]["built"] is True
        assert res_stats.json()["result"]["stats"]["rows"] == 17

        # OpenAPI 3.1.0 schema includes all new atlas routes + solver routes.
        schema = app.openapi()
        assert schema["openapi"].startswith("3.1")
        paths = set(schema["paths"].keys())
        assert {
            "/v1/holarch/atlas/build",
            "/v1/holarch/atlas/observe",
            "/v1/holarch/atlas/query",
            "/v1/holarch/atlas/stats",
            "/v1/holarch/repo_patch",
            "/v1/holarch/science",
            "/v1/holarch/arc",
            "/v1/chat/completions",
            "/.well-known/henri-jwks.json",
        } <= paths
    finally:
        svc.shutdown()


def test_gateway_mcp_exposes_solvers_and_atlas(tmp_path):
    svc, _app = _make_gateway(tmp_path)
    try:
        mcp = build_mcp_server(svc)
        tools = asyncio.run(mcp.list_tools())
        names = {t.name for t in tools}
        assert {
            "holarch_repo_patch",
            "holarch_science_solve",
            "holarch_arc_solve",
            "holarch_atlas_build",
            "holarch_atlas_observe",
            "holarch_atlas_query",
            "holarch_atlas_stats",
        } <= names

        # Execute atlas build + query directly through MCP tool manager.
        vecs = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
        built = asyncio.run(
            mcp._tool_manager.call_tool(
                "holarch_atlas_build", {"vectors": vecs, "rows": [10, 20, 30], "namespace": "default"}
            )
        )
        assert verify_signed_receipt(built["receipt"], svc.signer.jwks())[0] is True
        queried = asyncio.run(
            mcp._tool_manager.call_tool(
                "holarch_atlas_query", {"vector": [0.0, 1.0, 0.0, 0.0], "k": 1, "namespace": "default"}
            )
        )
        assert queried["result"]["hits"][0]["row"] == 20
        assert verify_signed_receipt(queried["receipt"], svc.signer.jwks())[0] is True
    finally:
        svc.shutdown()
