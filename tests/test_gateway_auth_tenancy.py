"""Tests for scoped API-key auth, namespace isolation, rate limits, and ops gating."""

from __future__ import annotations

from fastapi.testclient import TestClient

from henri.gateway import GatewayConfig, UnifiedGatewayService, create_gateway_app
from henri.holarch.config import HolarchConfig
from henri.holarch.server import SidecarService


def _make_gateway(tmp_path, *, clock=None, ops_token=None, legacy_keys=()):
    cfg = GatewayConfig(
        sqlite_path=str(tmp_path / "gw.sqlite3"),
        zone_c_sqlite_path=str(tmp_path / "zone_c.sqlite3"),
        require_auth=True,
        legacy_api_keys=list(legacy_keys),
        ops_token=ops_token,
        default_rpm=60,
        default_burst=5,
    )
    holarch = SidecarService(
        cfg=HolarchConfig.small(),
        backend="sqlite",
        memory_path=str(tmp_path / "zone_c.sqlite3"),
        autoload=False,
    )
    svc = UnifiedGatewayService(config=cfg, holarch_service=holarch, clock=clock)
    app = create_gateway_app(svc, mount_mcp=False)
    return svc, TestClient(app)


def test_scoped_api_key_enforces_401_and_403_namespace_isolation(tmp_path):
    svc, client = _make_gateway(tmp_path)
    try:
        tok_a, _ = svc.auth.create_key(
            tenant_id="acme", allowed_namespaces=["acme-prod"]
        )
        tok_b, _ = svc.auth.create_key(
            tenant_id="beta", allowed_namespaces=["beta-prod"]
        )

        # Public discovery endpoints remain accessible without auth
        assert client.get("/healthz").status_code == 200
        assert client.get("/.well-known/henri-jwks.json").status_code == 200

        # Missing or invalid token -> 401
        r_no_auth = client.post(
            "/v1/holarch/playbook",
            json={"name": "p1", "steps": ["step1"], "namespace": "acme-prod"},
        )
        assert r_no_auth.status_code == 401

        r_bad = client.post(
            "/v1/holarch/playbook",
            headers={"Authorization": "Bearer henri_live_nope.wrong"},
            json={"name": "p1", "steps": ["step1"], "namespace": "acme-prod"},
        )
        assert r_bad.status_code == 401

        # Authorized namespace succeeds
        r_ok = client.post(
            "/v1/holarch/playbook",
            headers={"Authorization": f"Bearer {tok_a}"},
            json={"name": "p1", "steps": ["step1"], "namespace": "acme-prod"},
        )
        assert r_ok.status_code == 200

        # Cross-tenant namespace access fails closed with 403
        r_cross = client.post(
            "/v1/holarch/playbook",
            headers={"Authorization": f"Bearer {tok_a}"},
            json={"name": "p1", "steps": ["step1"], "namespace": "beta-prod"},
        )
        assert r_cross.status_code == 403

        # Tenant B cannot read Tenant A's receipts
        r_rcpt_cross = client.get(
            "/v1/holarch/receipt?namespace=acme-prod",
            headers={"Authorization": f"Bearer {tok_b}"},
        )
        assert r_rcpt_cross.status_code == 403
    finally:
        svc.shutdown()


def test_per_tenant_rate_limit_isolates_tenants(tmp_path):
    now = [100.0]
    svc, client = _make_gateway(tmp_path, clock=lambda: now[0])
    try:
        tok_a, _ = svc.auth.create_key(
            tenant_id="acme", allowed_namespaces=["acme"], rpm_limit=60, burst_limit=2
        )
        tok_b, _ = svc.auth.create_key(
            tenant_id="beta", allowed_namespaces=["beta"], rpm_limit=60, burst_limit=2
        )

        # Exhaust tenant A's burst of 2
        assert client.get(
            "/v1/holarch/receipt?namespace=acme",
            headers={"Authorization": f"Bearer {tok_a}"},
        ).status_code == 200
        assert client.get(
            "/v1/holarch/receipt?namespace=acme",
            headers={"Authorization": f"Bearer {tok_a}"},
        ).status_code == 200

        r_429 = client.get(
            "/v1/holarch/receipt?namespace=acme",
            headers={"Authorization": f"Bearer {tok_a}"},
        )
        assert r_429.status_code == 429
        assert "Retry-After" in r_429.headers

        # Tenant B is completely unaffected
        assert client.get(
            "/v1/holarch/receipt?namespace=beta",
            headers={"Authorization": f"Bearer {tok_b}"},
        ).status_code == 200

        # Advance clock 2 seconds -> tenant A refills
        now[0] += 2.0
        assert client.get(
            "/v1/holarch/receipt?namespace=acme",
            headers={"Authorization": f"Bearer {tok_a}"},
        ).status_code == 200
    finally:
        svc.shutdown()


def test_legacy_api_key_binds_only_to_default_and_ops_token_gates_metrics(tmp_path):
    svc, client = _make_gateway(
        tmp_path, ops_token="ops-secret-123", legacy_keys=["legacy-raw-key"]
    )
    try:
        # Legacy key works on `default` namespace
        r_def = client.get(
            "/v1/holarch/receipt?namespace=default",
            headers={"Authorization": "Bearer legacy-raw-key"},
        )
        assert r_def.status_code == 200

        # Legacy key cannot access arbitrary tenant namespaces
        r_other = client.get(
            "/v1/holarch/receipt?namespace=other-tenant",
            headers={"Authorization": "Bearer legacy-raw-key"},
        )
        assert r_other.status_code == 403

        # Metrics gated by ops token
        assert client.get("/metrics").status_code == 401
        assert client.get(
            "/metrics", headers={"Authorization": "Bearer wrong-ops"}
        ).status_code == 403
        r_metrics = client.get(
            "/metrics", headers={"Authorization": "Bearer ops-secret-123"}
        )
        assert r_metrics.status_code == 200
        assert "henri_gateway_requests_total" in r_metrics.text
    finally:
        svc.shutdown()
