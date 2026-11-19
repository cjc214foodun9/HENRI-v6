"""End-to-end tests for Cycle AD routes, v1 backwards compatibility, persistence, and MCP parity."""

from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

from henri.api import DigitalTwinAPI
from henri.gateway import (
    GatewayConfig,
    UnifiedGatewayService,
    build_mcp_server,
    create_gateway_app,
    verify_signed_receipt,
)
from henri.holarch.config import HolarchConfig
from henri.holarch.server import SidecarService
from henri.config import WaveConfig
from henri.robotics_loop import RoboticsLoop


def _build_stack(tmp_path):
    cfg = GatewayConfig(
        sqlite_path=str(tmp_path / "gw.sqlite3"),
        zone_c_sqlite_path=str(tmp_path / "zone_c.sqlite3"),
        require_auth=False,
        digital_twin_dim=1024,
        digital_twin_rank=16,
    )
    holarch = SidecarService(
        cfg=HolarchConfig.small(),
        backend="sqlite",
        memory_path=str(tmp_path / "zone_c.sqlite3"),
        autoload=False,
    )
    dt = DigitalTwinAPI(dim=1024, rank=16)
    robotics = RoboticsLoop(cfg=WaveConfig(dim=256))
    svc = UnifiedGatewayService(
        config=cfg,
        holarch_service=holarch,
        digital_twin=dt,
        robotics_loop=robotics,
    )
    client = TestClient(create_gateway_app(svc, mount_mcp=False))
    return cfg, svc, client


def test_cycle_ad_science_route_solves_and_refuses_trap(tmp_path):
    _cfg, svc, client = _build_stack(tmp_path)
    try:
        jwks = client.get("/.well-known/henri-jwks.json").json()

        # Valid physics problem: F = m * a = 4 kg * 3 m/s^2 = 12 N
        r_ok = client.post(
            "/v1/holarch/science",
            json={
                "namespace": "sci",
                "problem": {
                    "id": "sci-force-1",
                    "domain": "mechanics",
                    "quantities": [
                        {"symbol": "mass", "value": 4.0, "unit": "kg"},
                        {"symbol": "acceleration", "value": 3.0, "unit": "m/s^2"},
                    ],
                    "target": {"symbol": "force", "unit": "N"},
                },
            },
        )
        assert r_ok.status_code == 200
        body = r_ok.json()
        assert body["ok"] is True
        assert body["result"]["status"] == "SOLVED"
        assert abs(float(body["result"]["answer"]) - 12.0) < 1e-6
        assert verify_signed_receipt(body["epistemic_receipt"], jwks)[0] is True

        # Nonphysical negative Kelvin temperature trap -> fail-closed ABSTAIN
        r_trap = client.post(
            "/v1/holarch/science",
            json={
                "namespace": "sci",
                "problem": {
                    "id": "sci-trap-1",
                    "domain": "thermo",
                    "quantities": [{"symbol": "temperature", "value": -20.0, "unit": "K"}],
                    "target": {"symbol": "pressure", "unit": "Pa"},
                },
            },
        )
        assert r_trap.status_code == 200
        trap_body = r_trap.json()
        assert trap_body["ok"] is False
        assert trap_body["result"]["trap_detected"] is True
        assert trap_body["result"]["status"] == "ABSTAIN"
    finally:
        svc.shutdown()


def test_cycle_ad_arc_route_induces_and_predicts_grids(tmp_path):
    _cfg, svc, client = _build_stack(tmp_path)
    try:
        # 90-degree rotation / colour swap: colour 1 -> 2 across 3 demonstration pairs
        r_arc = client.post(
            "/v1/holarch/arc",
            json={
                "namespace": "arc-ns",
                "depth": 1,
                "train": [
                    {"input": [[1, 0], [0, 1]], "output": [[2, 0], [0, 2]]},
                    {"input": [[1, 1], [0, 0]], "output": [[2, 2], [0, 0]]},
                    {"input": [[0, 1], [1, 0]], "output": [[0, 2], [2, 0]]},
                ],
                "test": [{"input": [[1, 0], [1, 1]]}],
            },
        )
        assert r_arc.status_code == 200
        data = r_arc.json()
        assert data["ok"] is True
        assert data["result"]["emitted"] is True
        assert data["result"]["loo_passed"] is True
        assert data["result"]["predictions"] == [[[2, 0], [2, 2]]]

        # Single pair -> must fail closed ABSTAIN
        r_abstain = client.post(
            "/v1/holarch/arc",
            json={
                "namespace": "arc-ns",
                "train": [{"input": [[1, 0]], "output": [[2, 0]]}],
                "test": [{"input": [[1, 0]]}],
            },
        )
        assert r_abstain.status_code == 200
        assert r_abstain.json()["ok"] is False
    finally:
        svc.shutdown()


def test_cycle_ad_repo_patch_route_fixes_bug_and_refuses_conflict(tmp_path):
    _cfg, svc, client = _build_stack(tmp_path)
    try:
        # Real buggy off-by-one operator in a small repository
        r_fix = client.post(
            "/v1/holarch/repo_patch",
            json={
                "namespace": "patch-ns",
                "name": "calc_add",
                "issue": "add(a, b) returns a - b instead of a + b",
                "files": {
                    "pkg/__init__.py": "",
                    "pkg/math_ops.py": "def add(a, b):\n    return a - b\n",
                },
                "public": {
                    "tests/test_public.py": (
                        "from pkg.math_ops import add\n\n"
                        "def test_add():\n"
                        "    assert add(2, 3) == 5\n"
                        "    assert add(10, 4) == 14\n"
                    )
                },
                "test_path": "tests/test_public.py",
                "rollouts": 16,
                "cascade_policy": "wave_only",
            },
        )
        assert r_fix.status_code == 200
        body = r_fix.json()
        assert body["ok"] is True
        assert body["result"]["llm_calls"] == 0

        # Conflicting public assertions -> fail-closed trap refusal
        r_trap = client.post(
            "/v1/holarch/repo_patch",
            json={
                "namespace": "patch-ns",
                "name": "conflict_trap",
                "issue": "impossible spec",
                "files": {"pkg/math_ops.py": "def f(x):\n    return x\n"},
                "public": {
                    "tests/test_public.py": (
                        "from pkg.math_ops import f\n"
                        "def test_conflict():\n"
                        "    assert f(1) == 2\n"
                        "    assert f(1) == 3\n"
                    )
                },
                "test_path": "tests/test_public.py",
            },
        )
        assert r_trap.status_code == 200
        assert r_trap.json()["ok"] is False
        assert r_trap.json()["result"]["trap_detected"] is True
    finally:
        svc.shutdown()


def test_v1_compat_cartridge_persistence_chat_sse_robotics_and_mcp(tmp_path):
    cfg, svc, client = _build_stack(tmp_path)
    try:
        # 1. Fail-closed mount before write -> 404
        assert client.post(
            "/v1/cartridges/mount", json={"tenant_id": "tenant-c"}
        ).status_code == 404

        # 2. Write cartridge & mount
        rw = client.post(
            "/v1/cartridges/write",
            json={
                "tenant_id": "tenant-c",
                "keys": ["gravity constant", "speed of light"],
                "values": ["9.80665 m/s^2", "299792458 m/s"],
                "subject": "physics",
            },
        )
        assert rw.status_code == 200
        assert rw.json()["persisted_sha256"]

        # 3. Rank options & robotics step & seal
        rr = client.post(
            "/v1/reason/rank",
            json={
                "question": "Which option mentions gravity?",
                "options": ["gravity constant", "banana pie"],
                "namespace": "tenant-c",
            },
        )
        assert rr.status_code == 200
        assert "epistemic_receipt" in rr.json()

        r_rob = client.post(
            "/v1/robotics/step",
            json={"sensor_values": [0.1, 0.2, 0.3, 0.4], "namespace": "tenant-c"},
        )
        assert r_rob.status_code == 200
        assert "epistemic_receipt" in r_rob.json()

        assert client.get("/v1/health/seal").status_code == 200
        assert client.get("/v1/health/executor").status_code == 200

        # 4. OpenAI-compatible chat completions (non-streaming and SSE)
        r_chat = client.post(
            "/v1/chat/completions",
            json={
                "model": "henri-holarch",
                "messages": [{"role": "user", "content": "verify claim"}],
                "metadata": {"namespace": "tenant-c", "cascade_policy": "wave_only"},
                "stream": False,
            },
        )
        assert r_chat.status_code == 200
        assert "epistemic_receipt" in r_chat.json()

        r_sse = client.post(
            "/v1/chat/completions",
            json={
                "model": "henri-holarch",
                "messages": [{"role": "user", "content": "verify claim"}],
                "metadata": {"namespace": "tenant-c", "cascade_policy": "wave_only"},
                "stream": True,
            },
        )
        assert r_sse.status_code == 200
        assert "text/event-stream" in r_sse.headers["content-type"]
        assert "data:" in r_sse.text

        # 5. MCP tool table includes Cycle AD and DigitalTwin tools
        mcp = build_mcp_server(svc)
        tools = asyncio.run(mcp.list_tools())
        names = {t.name for t in tools}
        for expected in (
            "holarch_health",
            "holarch_veto",
            "holarch_plan",
            "holarch_solve",
            "holarch_repo_patch",
            "holarch_science_solve",
            "holarch_arc_solve",
            "digital_twin_rank",
        ):
            assert expected in names
    finally:
        svc.shutdown()

    # 6. Verify cartridge bank hydrates cleanly from SQLite after restart!
    holarch2 = SidecarService(
        cfg=HolarchConfig.small(),
        backend="sqlite",
        memory_path=str(tmp_path / "zone_c.sqlite3"),
        autoload=False,
    )
    svc2 = UnifiedGatewayService(
        config=cfg,
        holarch_service=holarch2,
        digital_twin=DigitalTwinAPI(dim=1024, rank=16),
    )
    client2 = TestClient(create_gateway_app(svc2, mount_mcp=False))
    try:
        rm = client2.post("/v1/cartridges/mount", json={"tenant_id": "tenant-c"})
        assert rm.status_code == 200
        assert rm.json()["mounted"] is True
        assert rm.json()["cartridges"] == 1
        assert rm.json()["rows"] == 2
    finally:
        svc2.shutdown()
