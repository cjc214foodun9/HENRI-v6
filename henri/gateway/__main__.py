"""CLI entrypoint for the HENRI-v6 Unified Production Gateway.

Usage:
    python -m henri.gateway serve [--host 0.0.0.0] [--port 8080]
    python -m henri.gateway mcp [--transport stdio|sse]
    python -m henri.gateway keygen --tenant acme --namespaces acme-prod
"""

from __future__ import annotations

import argparse
import json
import sys

from henri.gateway.app import create_gateway_app
from henri.gateway.config import GatewayConfig
from henri.gateway.mcp_mount import build_mcp_server
from henri.gateway.service import UnifiedGatewayService


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="henri.gateway")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_serve = sub.add_parser("serve", help="Run the unified HTTP/SSE/MCP gateway with uvicorn")
    p_serve.add_argument("--host", default=None)
    p_serve.add_argument("--port", type=int, default=None)

    p_mcp = sub.add_parser("mcp", help="Run the unified MCP transport (stdio or sse)")
    p_mcp.add_argument("--transport", choices=("stdio", "sse"), default="stdio")

    p_key = sub.add_parser("keygen", help="Issue and persist a scoped API key")
    p_key.add_argument("--tenant", required=True)
    p_key.add_argument("--namespaces", required=True, help="Comma-separated allowed namespaces")
    p_key.add_argument("--rpm", type=int, default=120)
    p_key.add_argument("--burst", type=int, default=30)
    p_key.add_argument("--llm-budget", type=int, default=50)
    p_key.add_argument("--no-cascade", action="store_true")
    p_key.add_argument("--no-sandbox", action="store_true")

    args = parser.parse_args(argv)
    cfg = GatewayConfig.from_env()

    if args.cmd == "serve":
        import uvicorn

        host = args.host or cfg.host
        port = args.port or cfg.port
        app = create_gateway_app(config=cfg)
        uvicorn.run(app, host=host, port=port, log_level="info")
        return 0

    if args.cmd == "mcp":
        svc = UnifiedGatewayService(config=cfg)
        mcp = build_mcp_server(svc)
        mcp.run(transport=args.transport)
        return 0

    if args.cmd == "keygen":
        svc = UnifiedGatewayService(config=cfg)
        namespaces = [n.strip() for n in args.namespaces.split(",") if n.strip()]
        token, rec = svc.auth.create_key(
            tenant_id=args.tenant,
            allowed_namespaces=namespaces,
            rpm_limit=args.rpm,
            burst_limit=args.burst,
            llm_call_budget=args.llm_budget,
            allow_cascade=not args.no_cascade,
            allow_sandbox_exec=not args.no_sandbox,
        )
        print(
            json.dumps(
                {
                    "token": token,
                    "key_id": rec.key_id,
                    "tenant_id": rec.tenant_id,
                    "allowed_namespaces": list(rec.allowed_namespaces),
                    "default_namespace": rec.default_namespace,
                    "storage_backend": svc.store.backend,
                },
                indent=2,
            )
        )
        svc.shutdown()
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
