# HENRI-v6 Unified Production Gateway (`henri.gateway`)

Status: `OBSERVED` for software/API unit & integration verification on CPU; `BLOCKED` for live GPU/Blackwell timing or Docker daemon execution on hosts without CUDA/Docker.

## Overview

`henri/gateway/` unifies HENRI-v6's three surfaces behind one core service (`UnifiedGatewayService`):

1. **OpenAI-compatible Chat & SSE**: `POST /v1/chat/completions` (streaming & non-streaming) backed by the governed **Arm0 → Arm4** cascade.
2. **Typed REST `/v1/*`**: preserves 100% route compatibility with both `henri/holarch/server.py` and `henri/api_server.py`, and adds Cycle AD solver routes:
   - `POST /v1/holarch/repo_patch` (`RepoPatchEngine`)
   - `POST /v1/holarch/science` (`ScienceSolver`)
   - `POST /v1/holarch/arc` (`ObjectLooInducer`)
3. **Model Context Protocol (MCP)**: mounted at `/mcp` (SSE) and available over stdio (`python -m henri.gateway mcp --transport stdio`).

## Security & Epistemic Guarantees

- **Scoped Multi-Tenancy (`henri/gateway/auth.py`)**: API keys use `henri_live_<key_id>.<secret>`, stored as SHA-256 hashes in TimescaleDB (`gateway_api_keys`) or SQLite. Every call enforces `allowed_namespaces` (HTTP 403 on cross-tenant access) and per-tenant token-bucket rate limits (HTTP 429).
- **Ed25519 Signed Hash-Chained Receipts (`henri/gateway/signing.py`)**: Every endpoint appends a monotonically sequenced, parent-chained SHA-256 receipt signed with Ed25519 (`EdDSA`). Public keys are published at `GET /.well-known/henri-jwks.json`.
- **Governed Arm0 → Arm4 Cascade (`henri/gateway/cascade.py`)**: Arm0 deterministic wave/symbolic/Cycle AD solvers always run first at zero LLM cost. Detected traps never escalate. Eligible non-trap abstentions escalate to a configured OpenAI-compatible upstream under an atomic per-tenant SpendGuard budget, with at most 1 governed retry; if Sagnac/AST/ontology verification still vetoes or budget is exhausted, the gateway fails closed with `ABSTAIN`.
- **Two-Stage Code Sandbox (`henri/gateway/sandbox_worker.py` & `sandbox_client.py`)**: Stage 1 static `CapabilityASTGate` runs inside the gateway; Stage 2 runs over `AF_UNIX` IPC inside the hardened `henri-sandbox` container (`network_mode: none`, read-only rootfs, `cap_drop: ALL`, cgroup limits).

## Quickstart

### Local development (SQLite)

```bash
python -m henri.gateway keygen --tenant acme --namespaces acme-prod
HENRI_REQUIRE_AUTH=true python -m henri.gateway serve --host 127.0.0.1 --port 8080
```

### Production Docker Compose (Caddy + Gateway + Sandbox + TimescaleDB)

```bash
export POSTGRES_PASSWORD="<strong-secret>"
export HENRI_DOMAIN="api.example.com"
docker compose -f deploy/docker-compose.prod.yml up --build -d
```

### Observability

- `GET /healthz` — liveness, storage backend, sandbox mode, signing KID, compute device.
- `GET /.well-known/henri-jwks.json` — Ed25519 public key set for offline receipt verification.
- `GET /metrics` — Prometheus scrape endpoint (optionally gated by `HENRI_OPS_TOKEN`).
