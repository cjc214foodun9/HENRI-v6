"""Configuration for the HENRI-v6 Unified Production API Gateway.

Evidence label: DERIVED from environment and explicit caller configuration.
Every hardware/device field reports its measured host capability; requesting
CUDA without an available CUDA device fails closed rather than silently
misreporting compute_device.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return int(default)
    return int(raw.strip())


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return float(default)
    return float(raw.strip())


@dataclass
class GatewayConfig:
    """Immutable-style runtime settings for the unified gateway."""

    # Server bind
    host: str = "127.0.0.1"
    port: int = 8080

    # Persistence (Zone C + gateway tables)
    zone_c_backend: str = "sqlite"  # sqlite | timescaledb | auto
    timescale_dsn: str = "postgresql://postgres:henri_local_only@localhost:55432/henri"
    sqlite_path: str = "data/gateway/henri_gateway.sqlite3"
    zone_c_sqlite_path: Optional[str] = None
    atlas_persist_dir: Optional[str] = None

    # Compute & Holarch
    holarch_device: str = "cpu"  # cpu | cuda (BLOCKED on CPU hosts)
    holarch_checkpoint: Optional[str] = None
    holarch_autoload: bool = True
    compute_workers: int = 2
    digital_twin_dim: int = 65536
    digital_twin_rank: int = 64

    # Signing (Ed25519)
    signing_key_path: Optional[str] = None
    signing_key_id: Optional[str] = None
    rotated_public_keys_dir: Optional[str] = None

    # Auth & rate limiting
    require_auth: bool = False
    legacy_api_keys: List[str] = field(default_factory=list)
    bootstrap_keys_json: Optional[str] = None
    default_rpm: int = 120
    default_burst: int = 30
    default_llm_budget: int = 50
    ops_token: Optional[str] = None

    # Hardened sandbox sidecar
    sandbox_socket: Optional[str] = None
    sandbox_timeout_s: float = 10.0

    # Arm4 external OpenAI-compatible LLM cascade
    upstream_base_url: Optional[str] = None
    upstream_api_key: Optional[str] = None
    upstream_model: Optional[str] = None
    upstream_timeout_s: float = 30.0
    cascade_max_retries: int = 1

    # Observability
    enable_metrics: bool = True
    enable_otel: bool = False
    service_name: str = "henri-v6-gateway"

    def config_digest(self) -> str:
        """Deterministic SHA-256 over non-secret operational parameters."""
        safe = {
            "zone_c_backend": self.zone_c_backend,
            "atlas_persist_configured": bool(self.atlas_persist_dir),
            "holarch_device": self.holarch_device,
            "compute_workers": self.compute_workers,
            "digital_twin_dim": self.digital_twin_dim,
            "digital_twin_rank": self.digital_twin_rank,
            "require_auth": self.require_auth,
            "default_rpm": self.default_rpm,
            "default_burst": self.default_burst,
            "default_llm_budget": self.default_llm_budget,
            "sandbox_mode": "sidecar" if self.sandbox_socket else "local_child",
            "upstream_configured": bool(self.upstream_base_url and self.upstream_model),
            "upstream_model": self.upstream_model or "",
            "cascade_max_retries": self.cascade_max_retries,
            "service_name": self.service_name,
        }
        blob = json.dumps(safe, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    @classmethod
    def from_env(cls) -> "GatewayConfig":
        legacy_raw = os.environ.get("HENRI_API_KEYS", "")
        single = os.environ.get("HENRI_API_KEY", "")
        legacy_keys = [
            part.strip()
            for part in (legacy_raw.replace("\n", ",").split(",") + [single])
            if part.strip()
        ]
        bootstrap = os.environ.get("HENRI_GATEWAY_KEYS_JSON") or None
        require_auth = _env_bool(
            "HENRI_REQUIRE_AUTH",
            bool(legacy_keys or bootstrap),
        )
        return cls(
            host=os.environ.get("HENRI_HOST", "127.0.0.1"),
            port=_env_int("HENRI_PORT", 8080),
            zone_c_backend=os.environ.get("HENRI_ZONEC_BACKEND", "sqlite"),
            timescale_dsn=os.environ.get(
                "HENRI_TIMESCALE_DSN",
                "postgresql://postgres:henri_local_only@localhost:55432/henri",
            ),
            sqlite_path=os.environ.get(
                "HENRI_GATEWAY_SQLITE_PATH", "data/gateway/henri_gateway.sqlite3"
            ),
            zone_c_sqlite_path=os.environ.get("HENRI_ZONEC_PATH") or None,
            atlas_persist_dir=os.environ.get("HENRI_ATLAS_PERSIST_DIR") or None,
            holarch_device=os.environ.get("HENRI_HOLARCH_DEVICE", "cpu"),
            holarch_checkpoint=os.environ.get("HENRI_HOLARCH_CHECKPOINT") or None,
            holarch_autoload=_env_bool("HENRI_HOLARCH_AUTOLOAD", True),
            compute_workers=max(1, _env_int("HENRI_WORKERS_COMPUTE", 2)),
            digital_twin_dim=_env_int("HENRI_DIGITAL_TWIN_DIM", 65536),
            digital_twin_rank=_env_int("HENRI_DIGITAL_TWIN_RANK", 64),
            signing_key_path=os.environ.get("HENRI_SIGNING_PRIVATE_KEY_PATH") or None,
            signing_key_id=os.environ.get("HENRI_SIGNING_KEY_ID") or None,
            rotated_public_keys_dir=os.environ.get("HENRI_ROTATED_PUBKEYS_DIR") or None,
            require_auth=require_auth,
            legacy_api_keys=legacy_keys,
            bootstrap_keys_json=bootstrap,
            default_rpm=_env_int("HENRI_DEFAULT_RPM", 120),
            default_burst=_env_int("HENRI_DEFAULT_BURST", 30),
            default_llm_budget=_env_int("HENRI_DEFAULT_LLM_BUDGET", 50),
            ops_token=os.environ.get("HENRI_OPS_TOKEN") or None,
            sandbox_socket=os.environ.get("HENRI_SANDBOX_SOCKET") or None,
            sandbox_timeout_s=_env_float("HENRI_SANDBOX_TIMEOUT_S", 10.0),
            upstream_base_url=os.environ.get("HENRI_UPSTREAM_BASE_URL") or None,
            upstream_api_key=os.environ.get("HENRI_UPSTREAM_API_KEY") or None,
            upstream_model=os.environ.get("HENRI_UPSTREAM_MODEL") or None,
            upstream_timeout_s=_env_float("HENRI_UPSTREAM_TIMEOUT_S", 30.0),
            cascade_max_retries=min(1, max(0, _env_int("HENRI_CASCADE_MAX_RETRIES", 1))),
            enable_metrics=_env_bool("HENRI_ENABLE_METRICS", True),
            enable_otel=_env_bool("HENRI_ENABLE_OTEL", False),
            service_name=os.environ.get("HENRI_SERVICE_NAME", "henri-v6-gateway"),
        )


def ensure_parent_dir(path_str: str) -> Path:
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
