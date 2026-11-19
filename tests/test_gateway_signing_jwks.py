"""Tests for Ed25519 receipt signing, JWKS discovery, tamper detection, and restart persistence."""

from __future__ import annotations

import copy

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi.testclient import TestClient

from henri.gateway import (
    GatewayConfig,
    UnifiedGatewayService,
    create_gateway_app,
    verify_signed_receipt,
)
from henri.holarch.config import HolarchConfig
from henri.holarch.server import SidecarService


def _write_pem_key(path) -> str:
    priv = Ed25519PrivateKey.generate()
    pem = priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    path.write_bytes(pem)
    return str(path)


def test_ed25519_receipt_verification_and_tamper_rejection(tmp_path):
    key_path = _write_pem_key(tmp_path / "ed25519.pem")
    cfg = GatewayConfig(
        sqlite_path=str(tmp_path / "gw.sqlite3"),
        zone_c_sqlite_path=str(tmp_path / "zone_c.sqlite3"),
        signing_key_path=key_path,
        signing_key_id="henri-test-kid-1",
    )
    holarch = SidecarService(
        cfg=HolarchConfig.small(),
        backend="sqlite",
        memory_path=str(tmp_path / "zone_c.sqlite3"),
        autoload=False,
    )
    svc = UnifiedGatewayService(config=cfg, holarch_service=holarch)
    client = TestClient(create_gateway_app(svc, mount_mcp=False))
    try:
        jwks = client.get("/.well-known/henri-jwks.json").json()
        assert len(jwks["keys"]) == 1
        assert jwks["keys"][0]["kty"] == "OKP"
        assert jwks["keys"][0]["crv"] == "Ed25519"
        assert jwks["keys"][0]["kid"] == "henri-test-kid-1"

        # Issue two calls in namespace "alpha"
        r1 = client.post(
            "/v1/holarch/ingest_edges",
            json={
                "namespace": "alpha",
                "edges": [{"source": "A", "operator": "next", "target": "B"}],
            },
        )
        assert r1.status_code == 200
        rcpt1 = r1.json()["epistemic_receipt"]
        ok, reason = verify_signed_receipt(rcpt1, jwks)
        assert ok, reason
        assert rcpt1["seq"] == 1
        assert rcpt1["parent"] == "0" * 64
        assert rcpt1["evidence_label"] == "OBSERVED"
        assert rcpt1["provenance"] == "wave_only"
        assert rcpt1["signature"]["alg"] == "EdDSA"

        r2 = client.post(
            "/v1/holarch/solve",
            json={"source": "A", "operators": ["next"], "namespace": "alpha"},
        )
        assert r2.status_code == 200
        rcpt2 = r2.json()["epistemic_receipt"]
        ok2, reason2 = verify_signed_receipt(rcpt2, jwks)
        assert ok2, reason2
        assert rcpt2["seq"] == 2
        assert rcpt2["parent"] == rcpt1["sha256"]

        # Tampering with payload breaks verification
        tampered = copy.deepcopy(rcpt2)
        tampered["result"] = {"forged": True}
        ok_bad, why = verify_signed_receipt(tampered, jwks)
        assert not ok_bad
        assert why == "sha256_mismatch"

        # Tampering with signature breaks verification
        bad_sig = copy.deepcopy(rcpt2)
        bad_sig["signature"]["sig"] = "A" * len(bad_sig["signature"]["sig"])
        ok_sig, why_sig = verify_signed_receipt(bad_sig, jwks)
        assert not ok_sig
        assert "signature" in why_sig
    finally:
        svc.shutdown()

    # Restart a fresh gateway pointing at the same SQLite DB & signing key
    holarch2 = SidecarService(
        cfg=HolarchConfig.small(),
        backend="sqlite",
        memory_path=str(tmp_path / "zone_c.sqlite3"),
        autoload=False,
    )
    svc2 = UnifiedGatewayService(config=cfg, holarch_service=holarch2)
    client2 = TestClient(create_gateway_app(svc2, mount_mcp=False))
    try:
        # Chain persists across restart and seq continues cleanly at 3
        chain = client2.get("/v1/holarch/receipt?namespace=alpha").json()
        assert chain["chain_valid"] is True
        assert chain["count"] == 2

        r3 = client2.post(
            "/v1/holarch/solve",
            json={"source": "A", "operators": ["next"], "namespace": "alpha"},
        )
        rcpt3 = r3.json()["epistemic_receipt"]
        assert rcpt3["seq"] == 3
        assert rcpt3["parent"] == chain["head"]
        jwks2 = client2.get("/.well-known/henri-jwks.json").json()
        assert verify_signed_receipt(rcpt3, jwks2)[0] is True
    finally:
        svc2.shutdown()
