"""Scoped API-key authentication, namespace authorization, and per-tenant rate limiting.

Evidence label: OBSERVED for every auth and rate-limit decision enforced here.

Security invariants:
  * Fail-closed: when auth is enabled, missing/invalid Bearer tokens return 401
    and cross-namespace access returns 403.
  * Constant-time secret verification (`hmac.compare_digest`).
  * Structured key format `henri_live_<key_id>.<secret>` backed by
    `GatewayStore` (only SHA-256 secret hashes are persisted).
  * Legacy `HENRI_API_KEYS` compatibility keys bind explicitly to `default`
    namespace so existing clients keep working without leaking other tenants.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from fastapi import HTTPException

from henri.gateway.storage import GatewayStore, StoredApiKey


KEY_PREFIX = "henri_live_"


def hash_secret(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def parse_structured_token(token: str) -> Optional[Tuple[str, str]]:
    """Parse `henri_live_<key_id>.<secret>` into `(key_id, secret)`."""
    if not token.startswith(KEY_PREFIX):
        return None
    rest = token[len(KEY_PREFIX):]
    if "." not in rest:
        return None
    key_id, secret = rest.split(".", 1)
    if not key_id or not secret:
        return None
    return key_id, secret


@dataclass(frozen=True)
class TenantPolicy:
    key_id: str
    tenant_id: str
    allowed_namespaces: Tuple[str, ...]
    default_namespace: str
    rpm_limit: int = 120
    burst_limit: int = 30
    llm_call_budget: int = 50
    allow_cascade: bool = True
    allow_sandbox_exec: bool = True
    authenticated: bool = True

    def allows_namespace(self, namespace: str) -> bool:
        if "*" in self.allowed_namespaces:
            return True
        return str(namespace) in self.allowed_namespaces


class PerTenantTokenBucket:
    """Thread-safe per-tenant token bucket with injectable clock for tests."""

    def __init__(self, clock: Optional[Callable[[], float]] = None) -> None:
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._buckets: Dict[str, Tuple[float, float]] = {}

    def allow(
        self, tenant_id: str, *, rpm: int, burst: int, cost: float = 1.0
    ) -> Tuple[bool, float]:
        rate_per_sec = max(0.01, float(rpm) / 60.0)
        cap = max(1.0, float(burst))
        now = float(self._clock())
        with self._lock:
            tokens, last = self._buckets.get(tenant_id, (cap, now))
            elapsed = max(0.0, now - last)
            tokens = min(cap, tokens + elapsed * rate_per_sec)
            if tokens >= cost:
                self._buckets[tenant_id] = (tokens - cost, now)
                return True, 0.0
            retry_after = (cost - tokens) / rate_per_sec
            self._buckets[tenant_id] = (tokens, now)
            return False, retry_after

    def reset(self) -> None:
        with self._lock:
            self._buckets.clear()


class ScopedAuthManager:
    """Validates Bearer tokens, enforces namespace isolation, and rate limits."""

    def __init__(
        self,
        store: GatewayStore,
        *,
        require_auth: bool = False,
        legacy_api_keys: Sequence[str] = (),
        default_rpm: int = 120,
        default_burst: int = 30,
        default_llm_budget: int = 50,
        ops_token: Optional[str] = None,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self.store = store
        self.require_auth = bool(require_auth)
        self.default_rpm = int(default_rpm)
        self.default_burst = int(default_burst)
        self.default_llm_budget = int(default_llm_budget)
        self.ops_token = ops_token or None
        self.limiter = PerTenantTokenBucket(clock=clock)
        self._legacy_hashes: Dict[str, str] = {}
        for idx, raw_key in enumerate(legacy_api_keys):
            raw_key = str(raw_key).strip()
            if raw_key:
                self._legacy_hashes[f"legacy-{idx}"] = hash_secret(raw_key)

    def create_key(
        self,
        *,
        tenant_id: str,
        allowed_namespaces: Sequence[str],
        default_namespace: Optional[str] = None,
        key_id: Optional[str] = None,
        secret: Optional[str] = None,
        rpm_limit: Optional[int] = None,
        burst_limit: Optional[int] = None,
        llm_call_budget: Optional[int] = None,
        allow_cascade: bool = True,
        allow_sandbox_exec: bool = True,
    ) -> Tuple[str, StoredApiKey]:
        """Issue and persist a new scoped API key. Returns `(plaintext_token, record)`."""
        if not allowed_namespaces:
            raise ValueError("allowed_namespaces must not be empty")
        kid = key_id or secrets.token_hex(6)
        sec = secret or secrets.token_urlsafe(24)
        ns_tuple = tuple(str(n) for n in allowed_namespaces)
        default_ns = str(default_namespace or ns_tuple[0])
        if "*" not in ns_tuple and default_ns not in ns_tuple:
            raise ValueError("default_namespace must be inside allowed_namespaces")
        rec = StoredApiKey(
            key_id=kid,
            secret_hash=hash_secret(sec),
            tenant_id=str(tenant_id),
            allowed_namespaces=ns_tuple,
            default_namespace=default_ns,
            rpm_limit=int(rpm_limit if rpm_limit is not None else self.default_rpm),
            burst_limit=int(burst_limit if burst_limit is not None else self.default_burst),
            llm_call_budget=int(
                llm_call_budget if llm_call_budget is not None else self.default_llm_budget
            ),
            allow_cascade=bool(allow_cascade),
            allow_sandbox_exec=bool(allow_sandbox_exec),
            revoked=False,
        )
        self.store.upsert_api_key(rec)
        return f"{KEY_PREFIX}{kid}.{sec}", rec

    def bootstrap_from_json(self, raw_json: Optional[str]) -> int:
        """Load pre-provisioned tenant keys from `HENRI_GATEWAY_KEYS_JSON`."""
        if not raw_json:
            return 0
        rows = json.loads(raw_json)
        if not isinstance(rows, list):
            raise ValueError("HENRI_GATEWAY_KEYS_JSON must be a JSON list")
        loaded = 0
        for row in rows:
            token = str(row.get("token") or "")
            parsed = parse_structured_token(token)
            if parsed is None:
                raise ValueError("bootstrap token must use henri_live_<kid>.<secret> format")
            kid, sec = parsed
            namespaces = row.get("allowed_namespaces") or [row.get("tenant_id", "default")]
            self.create_key(
                tenant_id=str(row.get("tenant_id", "default")),
                allowed_namespaces=[str(n) for n in namespaces],
                default_namespace=row.get("default_namespace"),
                key_id=kid,
                secret=sec,
                rpm_limit=row.get("rpm_limit"),
                burst_limit=row.get("burst_limit"),
                llm_call_budget=row.get("llm_call_budget"),
                allow_cascade=bool(row.get("allow_cascade", True)),
                allow_sandbox_exec=bool(row.get("allow_sandbox_exec", True)),
            )
            loaded += 1
        return loaded

    def is_auth_enforced(self) -> bool:
        return (
            self.require_auth
            or bool(self._legacy_hashes)
            or self.store.count_api_keys() > 0
        )

    def authenticate(self, authorization: Optional[str]) -> TenantPolicy:
        """Resolve a Bearer header into a TenantPolicy or raise HTTP 401."""
        enforced = self.is_auth_enforced()
        if not authorization:
            if not enforced:
                return TenantPolicy(
                    key_id="anonymous-dev",
                    tenant_id="default",
                    allowed_namespaces=("*",),
                    default_namespace="default",
                    rpm_limit=self.default_rpm,
                    burst_limit=self.default_burst,
                    llm_call_budget=self.default_llm_budget,
                    allow_cascade=True,
                    allow_sandbox_exec=True,
                    authenticated=False,
                )
            raise HTTPException(status_code=401, detail="missing Authorization Bearer token")

        parts = authorization.strip().split(" ", 1)
        if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1].strip():
            raise HTTPException(status_code=401, detail="invalid Authorization header format")
        token = parts[1].strip()

        parsed = parse_structured_token(token)
        if parsed is not None:
            kid, secret = parsed
            rec = self.store.get_api_key(kid)
            if rec is None or rec.revoked:
                raise HTTPException(status_code=401, detail="invalid or revoked API key")
            if not hmac.compare_digest(hash_secret(secret), rec.secret_hash):
                raise HTTPException(status_code=401, detail="invalid API key")
            return TenantPolicy(
                key_id=rec.key_id,
                tenant_id=rec.tenant_id,
                allowed_namespaces=rec.allowed_namespaces,
                default_namespace=rec.default_namespace,
                rpm_limit=rec.rpm_limit,
                burst_limit=rec.burst_limit,
                llm_call_budget=rec.llm_call_budget,
                allow_cascade=rec.allow_cascade,
                allow_sandbox_exec=rec.allow_sandbox_exec,
                authenticated=True,
            )

        # Fallback: legacy HENRI_API_KEYS bound strictly to `default` namespace.
        token_hash = hash_secret(token)
        for leg_id, expected_hash in self._legacy_hashes.items():
            if hmac.compare_digest(token_hash, expected_hash):
                return TenantPolicy(
                    key_id=leg_id,
                    tenant_id="default",
                    allowed_namespaces=("default",),
                    default_namespace="default",
                    rpm_limit=self.default_rpm,
                    burst_limit=self.default_burst,
                    llm_call_budget=self.default_llm_budget,
                    allow_cascade=True,
                    allow_sandbox_exec=True,
                    authenticated=True,
                )
        raise HTTPException(status_code=401, detail="invalid API key")

    def authorize_namespace(
        self, policy: TenantPolicy, requested_namespace: Optional[str]
    ) -> str:
        """Validate namespace against the caller's TenantPolicy (fail-closed 403)."""
        ns = (requested_namespace or "").strip() or policy.default_namespace
        if not policy.allows_namespace(ns):
            raise HTTPException(
                status_code=403,
                detail=f"API key {policy.key_id!r} is not authorized for namespace {ns!r}",
            )
        return ns

    def check_rate_limit(self, policy: TenantPolicy) -> None:
        ok, retry_after = self.limiter.allow(
            policy.tenant_id, rpm=policy.rpm_limit, burst=policy.burst_limit
        )
        if not ok:
            raise HTTPException(
                status_code=429,
                detail=f"rate limit exceeded for tenant {policy.tenant_id!r}",
                headers={"Retry-After": f"{max(1, int(retry_after + 0.999))}"},
            )

    def authorize_ops(self, authorization: Optional[str]) -> None:
        """Gate internal operational endpoints like /metrics when ops_token is set."""
        if not self.ops_token:
            return
        if not authorization:
            raise HTTPException(status_code=401, detail="ops bearer token required")
        parts = authorization.strip().split(" ", 1)
        if len(parts) != 2 or parts[0].lower() != "bearer":
            raise HTTPException(status_code=401, detail="invalid ops authorization header")
        if not hmac.compare_digest(parts[1].strip(), self.ops_token):
            raise HTTPException(status_code=403, detail="forbidden ops token")
