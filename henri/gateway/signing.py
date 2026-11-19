"""Ed25519 cryptographic signing and JWKS discovery for HENRI epistemic receipts.

Evidence label: OBSERVED for every signature produced or verified by this
module.

Unlike the legacy unsigned self-digest in `henri/holarch/server.py`, every
gateway receipt carries:
  1. A canonical SHA-256 digest chained to `parent` (tamper-evident order).
  2. An Ed25519 signature (`alg: "EdDSA"`) over the raw 32-byte SHA-256 digest
     using a server-held key whose public counterpart is published at
     `/.well-known/henri-jwks.json`.
  3. Explicit epistemic metadata: `evidence_label` (OBSERVED | DERIVED | BLOCKED),
     `provenance` (wave_only | llm_escalated | cached), and `compute_device`.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

GENESIS = "0" * 64
VALID_EVIDENCE_LABELS = frozenset(
    {"OBSERVED", "DERIVED", "INFERRED", "HYPOTHESIS", "FALSIFIED", "BLOCKED"}
)
VALID_PROVENANCE = frozenset({"wave_only", "llm_escalated", "cached"})


def _b64u_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64u_decode(text: str) -> bytes:
    pad = "=" * ((4 - (len(text) % 4)) % 4)
    return base64.urlsafe_b64decode(text + pad)


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_clean(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return _json_clean(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return _json_clean(value.detach().cpu().tolist())
    return str(value)


def canonical_json(payload: Any) -> str:
    return json.dumps(
        _json_clean(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def build_receipt_body(
    *,
    endpoint: str,
    namespace: str,
    seq: int,
    request: Any,
    result: Any,
    parent: str = GENESIS,
    created_utc: Optional[str] = None,
    config_digest: str = "",
    evidence_label: str = "OBSERVED",
    provenance: str = "wave_only",
    compute_device: str = "cpu",
) -> Dict[str, Any]:
    """Construct the exact canonical body covered by `sha256` and Ed25519."""
    ev = str(evidence_label).upper()
    if ev not in VALID_EVIDENCE_LABELS:
        raise ValueError(f"invalid evidence_label: {evidence_label!r}")
    prov = str(provenance)
    if prov not in VALID_PROVENANCE:
        raise ValueError(f"invalid provenance: {provenance!r}")
    return {
        "algorithm": "sha256",
        "canonical": "json-sorted-keys",
        "compute_device": str(compute_device),
        "config_digest": str(config_digest),
        "created_utc": created_utc or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "endpoint": str(endpoint),
        "evidence_label": ev,
        "namespace": str(namespace),
        "parent": str(parent),
        "provenance": prov,
        "request": _json_clean(request),
        "result": _json_clean(result),
        "seq": int(seq),
    }


def public_key_to_jwk(pubkey: Ed25519PublicKey, kid: str) -> Dict[str, str]:
    raw = pubkey.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return {
        "kty": "OKP",
        "crv": "Ed25519",
        "alg": "EdDSA",
        "use": "sig",
        "kid": str(kid),
        "x": _b64u_encode(raw),
    }


def jwk_to_public_key(jwk: Dict[str, Any]) -> Ed25519PublicKey:
    if jwk.get("kty") != "OKP" or jwk.get("crv") != "Ed25519":
        raise ValueError("JWK is not an Ed25519 OKP key")
    raw = _b64u_decode(str(jwk["x"]))
    return Ed25519PublicKey.from_public_bytes(raw)


def _derive_kid(pubkey: Ed25519PublicKey, prefix: str = "henri-ed25519") -> str:
    raw = pubkey.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    fp = hashlib.sha256(raw).hexdigest()[:12]
    return f"{prefix}-{fp}"


@dataclass
class SigningKeyInfo:
    kid: str
    ephemeral: bool
    public_jwk: Dict[str, str]


class Ed25519ReceiptSigner:
    """Signs canonical SHA-256 receipts with Ed25519 and serves JWKS."""

    def __init__(
        self,
        private_key: Optional[Ed25519PrivateKey] = None,
        *,
        key_path: Optional[str | Path] = None,
        kid: Optional[str] = None,
        rotated_public_keys: Optional[Dict[str, Ed25519PublicKey]] = None,
    ) -> None:
        self._ephemeral = False
        if private_key is not None:
            self._private = private_key
        elif key_path is not None and Path(key_path).exists():
            pem = Path(key_path).read_bytes()
            loaded = serialization.load_pem_private_key(pem, password=None)
            if not isinstance(loaded, Ed25519PrivateKey):
                raise ValueError(f"signing key at {key_path} is not Ed25519")
            self._private = loaded
        else:
            self._private = Ed25519PrivateKey.generate()
            self._ephemeral = True

        self._public = self._private.public_key()
        prefix = "dev-ephemeral-ed25519" if self._ephemeral else "henri-ed25519"
        self.kid = str(kid) if kid else _derive_kid(self._public, prefix=prefix)
        self._verify_keys: Dict[str, Ed25519PublicKey] = {self.kid: self._public}
        if rotated_public_keys:
            for r_kid, r_pub in rotated_public_keys.items():
                self._verify_keys[str(r_kid)] = r_pub

    @classmethod
    def from_config(
        cls,
        *,
        key_path: Optional[str] = None,
        kid: Optional[str] = None,
        rotated_public_keys_dir: Optional[str] = None,
    ) -> "Ed25519ReceiptSigner":
        rotated: Dict[str, Ed25519PublicKey] = {}
        if rotated_public_keys_dir:
            root = Path(rotated_public_keys_dir)
            if root.is_dir():
                for pem_file in sorted(root.glob("*.pem")):
                    loaded = serialization.load_pem_public_key(pem_file.read_bytes())
                    if isinstance(loaded, Ed25519PublicKey):
                        rotated[pem_file.stem] = loaded
        return cls(key_path=key_path, kid=kid, rotated_public_keys=rotated)

    @property
    def info(self) -> SigningKeyInfo:
        return SigningKeyInfo(
            kid=self.kid,
            ephemeral=self._ephemeral,
            public_jwk=public_key_to_jwk(self._public, self.kid),
        )

    def jwks(self) -> Dict[str, List[Dict[str, str]]]:
        """RFC 7517 / RFC 8037 JWK Set for `/.well-known/henri-jwks.json`."""
        return {
            "keys": [
                public_key_to_jwk(pub, k)
                for k, pub in sorted(self._verify_keys.items())
            ]
        }

    def issue_signed(
        self,
        *,
        endpoint: str,
        namespace: str,
        seq: int,
        request: Any,
        result: Any,
        parent: str = GENESIS,
        created_utc: Optional[str] = None,
        config_digest: str = "",
        evidence_label: str = "OBSERVED",
        provenance: str = "wave_only",
        compute_device: str = "cpu",
    ) -> Dict[str, Any]:
        body = build_receipt_body(
            endpoint=endpoint,
            namespace=namespace,
            seq=seq,
            request=request,
            result=result,
            parent=parent,
            created_utc=created_utc,
            config_digest=config_digest,
            evidence_label=evidence_label,
            provenance=provenance,
            compute_device=compute_device,
        )
        digest_bytes = hashlib.sha256(canonical_json(body).encode("utf-8")).digest()
        digest_hex = digest_bytes.hex()
        sig_bytes = self._private.sign(digest_bytes)
        return {
            **body,
            "sha256": digest_hex,
            "signature": {
                "alg": "EdDSA",
                "kid": self.kid,
                "sig": _b64u_encode(sig_bytes),
            },
        }

    def verify(self, receipt: Dict[str, Any]) -> Tuple[bool, str]:
        return verify_signed_receipt(receipt, self.jwks())


def verify_signed_receipt(
    receipt: Dict[str, Any], jwks: Dict[str, Any]
) -> Tuple[bool, str]:
    """Verify both the canonical SHA-256 body digest and Ed25519 signature."""
    try:
        body = build_receipt_body(
            endpoint=receipt["endpoint"],
            namespace=receipt["namespace"],
            seq=int(receipt["seq"]),
            request=receipt.get("request"),
            result=receipt.get("result"),
            parent=receipt.get("parent", GENESIS),
            created_utc=receipt.get("created_utc"),
            config_digest=receipt.get("config_digest", ""),
            evidence_label=receipt.get("evidence_label", "OBSERVED"),
            provenance=receipt.get("provenance", "wave_only"),
            compute_device=receipt.get("compute_device", "cpu"),
        )
    except Exception as exc:
        return False, f"malformed_body: {exc}"

    digest_bytes = hashlib.sha256(canonical_json(body).encode("utf-8")).digest()
    expected_hex = digest_bytes.hex()
    if receipt.get("sha256") != expected_hex:
        return False, "sha256_mismatch"

    sig_obj = receipt.get("signature")
    if not isinstance(sig_obj, dict):
        return False, "missing_signature"
    if sig_obj.get("alg") != "EdDSA":
        return False, f"unsupported_alg: {sig_obj.get('alg')}"
    kid = str(sig_obj.get("kid") or "")
    sig_b64 = str(sig_obj.get("sig") or "")
    if not kid or not sig_b64:
        return False, "incomplete_signature"

    keys = {str(k.get("kid")): k for k in (jwks.get("keys") or [])}
    jwk = keys.get(kid)
    if jwk is None:
        return False, f"unknown_kid: {kid}"

    try:
        pubkey = jwk_to_public_key(jwk)
        pubkey.verify(_b64u_decode(sig_b64), digest_bytes)
    except InvalidSignature:
        return False, "invalid_ed25519_signature"
    except Exception as exc:
        return False, f"signature_error: {exc}"
    return True, "ok"
