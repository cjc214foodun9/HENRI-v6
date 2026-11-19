"""Tests for the hardened sandbox sidecar Unix-socket IPC protocol and /v1/code/solve routing."""

from __future__ import annotations

from fastapi.testclient import TestClient

from henri.api import DigitalTwinAPI
from henri.gateway import (
    GatewayConfig,
    SandboxClient,
    UnifiedGatewayService,
    create_gateway_app,
)
from henri.gateway.sandbox_worker import SandboxUnixServer
from henri.holarch.config import HolarchConfig
from henri.holarch.server import SidecarService


def test_sandbox_unix_ipc_exec_block_and_timeout(tmp_path):
    sock_path = str(tmp_path / "exec.sock")
    server = SandboxUnixServer(sock_path)
    server.start()
    try:
        client = SandboxClient(socket_path=sock_path, default_timeout_s=2.0)
        ping = client.ping()
        assert ping["ok"] is True
        assert ping["mode"] == "sidecar_ipc"

        # Valid pure algorithmic execution passes
        res_ok = client.execute(
            "def sq(n):\n    return n * n\n",
            tests=["assert sq(7) == 49"],
            profile="pure_algorithmic",
        )
        assert res_ok["ok"] is True
        assert res_ok["mode"] == "sidecar_ipc"
        assert res_ok["exit_code"] == 0
        assert len(res_ok["artifact_sha256"]) == 64

        # Disallowed import blocked at Stage 1 AST gate
        res_blocked = client.execute(
            "import os\nos.listdir('.')\n",
            profile="pure_algorithmic",
        )
        assert res_blocked["ok"] is False
        assert res_blocked["blocked"] is True
        assert res_blocked["stage"] == "ast_gate"

        # Infinite loop killed cleanly on timeout
        res_timeout = client.execute(
            "while True:\n    pass\n",
            profile="pure_algorithmic",
            timeout_s=0.6,
        )
        assert res_timeout["ok"] is False
        assert res_timeout["timed_out"] is True
    finally:
        server.stop()


def test_gateway_code_solve_routes_through_sandbox_sidecar(tmp_path):
    sock_path = str(tmp_path / "code.sock")
    server = SandboxUnixServer(sock_path)
    server.start()
    cfg = GatewayConfig(
        sqlite_path=str(tmp_path / "gw.sqlite3"),
        zone_c_sqlite_path=str(tmp_path / "zone_c.sqlite3"),
        sandbox_socket=sock_path,
        digital_twin_dim=1024,
        digital_twin_rank=16,
    )
    holarch = SidecarService(
        cfg=HolarchConfig.small(),
        backend="sqlite",
        memory_path=str(tmp_path / "zone_c.sqlite3"),
        autoload=False,
    )
    svc = UnifiedGatewayService(
        config=cfg,
        holarch_service=holarch,
        digital_twin=DigitalTwinAPI(dim=1024, rank=16),
    )
    client = TestClient(create_gateway_app(svc, mount_mcp=False))
    try:
        health = client.get("/v1/health/executor").json()
        assert health["sandbox"]["mode"] == "sidecar_ipc"
        assert health["sandbox"]["ok"] is True

        resp = client.post(
            "/v1/code/solve",
            json={
                "spec": "Return the double of x",
                "candidates": [
                    "def dbl(x):\n    return x + 1\n",
                    "def dbl(x):\n    return x * 2\n",
                ],
                "tests": ["assert dbl(5) == 10", "assert dbl(0) == 0"],
                "tenant_id": "default",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["verdict"] == "VERIFIED"
        assert body["selected_index"] == 1
        assert body["sandbox_mode"] == "sidecar_ipc"
        assert "epistemic_receipt" in body
    finally:
        svc.shutdown()
        server.stop()
