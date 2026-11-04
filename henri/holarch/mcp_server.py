"""Model Context Protocol server for the HENRI epistemic sidecar.

Evidence label: DERIVED for the transport wiring, OBSERVED for every
number a tool returns.

Decision A5 asks for the sidecar to be callable by an agent over MCP.
This module exposes the same SidecarService that the REST app uses, so a
tool call and an HTTP call return the same result and the same kind of
SHA-256 receipt.

Transports (MCP 1.x):

    python -m henri.holarch.mcp_server                 stdio (default)
    python -m henri.holarch.mcp_server --transport sse --port 8090

The stdio form is what an MCP client such as Claude Desktop or an IDE
plugin launches. The SSE form mounts the same tool table behind an HTTP
event stream. Both are read-modify-write safe through one process lock.

Tools:

    holarch_veto          is this input inside the manifold?
    holarch_ingest_code   index a package once, no weight change
    holarch_ingest_edges  write an unseen rule graph into Zone C
    holarch_solve         answer a symbolic query or return ABSTAIN
    holarch_playbook      compile a strategy, or recall its exact steps
    holarch_adapt_rule    one surprise-gated fast-weight write
    holarch_health        receipt chain state and parameter count
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from henri.holarch.config import HolarchConfig
from henri.holarch.server import SidecarService

mcp = FastMCP(
    "henri-holarch-sidecar",
    instructions=(
        "Fail-closed epistemic tools. Every answer carries a SHA-256 "
        "receipt; an unanswerable symbolic query returns vetoed=true and "
        "text starting with ABSTAIN instead of a guess."
    ),
)

_SERVICE: Optional[SidecarService] = None


def service() -> SidecarService:
    """Lazily build the one sidecar this process serves."""
    global _SERVICE
    if _SERVICE is None:
        backend = os.environ.get("HENRI_SIDECAR_BACKEND", "sqlite")
        path = os.environ.get("HENRI_SIDECAR_PATH") or None
        cfg = HolarchConfig.small()
        _SERVICE = SidecarService(cfg=cfg, backend=backend, memory_path=path)
    return _SERVICE


def set_service(value: Optional[SidecarService]) -> None:
    """Inject a service (tests, or a process that already holds one)."""
    global _SERVICE
    _SERVICE = value


@mcp.tool()
def holarch_veto(task: str, x: Any, rule_id: Optional[int] = None) -> Dict[str, Any]:
    """Return the veto rate, top-1 Gram coherence, and Sagnac residual.

    Args:
        task: one of vision, language, action, relational.
        x: nested pixel or observation payload for one or more examples.
        rule_id: optional slot-3 rule context.
    """
    return service().veto(task, x, rule_id)


@mcp.tool()
def holarch_ingest_code(root: str, package: str = "henri") -> Dict[str, Any]:
    """Parse a package once and index its symbols and call edges in Zone C.

    Query time performs no file reads. The graph is stored as wave engrams,
    so no model weight changes.
    """
    return service().ingest_code(root, package)


@mcp.tool()
def holarch_ingest_edges(edges: List[Dict[str, str]]) -> Dict[str, Any]:
    """Store an unseen rule graph. Each edge has source, operator, target."""
    return service().ingest_edges(edges)


@mcp.tool()
def holarch_solve(source: str, operators: Any) -> Dict[str, Any]:
    """Follow an operator sequence from source, or abstain fail-closed.

    operators may be a list or a comma-separated string. An unreachable
    path returns vetoed=true with text starting ABSTAIN and no invented
    symbol.
    """
    return service().solve(source, operators)


@mcp.tool()
def holarch_playbook(
    name: str,
    steps: Optional[List[str]] = None,
    merge: bool = True,
) -> Dict[str, Any]:
    """Compile steps into a versioned playbook, or recall the exact steps.

    With steps the call merges them (ACE delta form) and bumps the
    version. Without steps it recalls and decodes the stored plan.
    """
    return service().playbook(name, steps, merge)


@mcp.tool()
def holarch_adapt_rule(
    task: str,
    x: Any,
    y: List[int],
    rule_id: Optional[int] = None,
    lr: float = 0.2,
) -> Dict[str, Any]:
    """One surprise-gated fast-weight write, with the rule in slot 3."""
    return service().adapt_rule(task, x, y, rule_id, lr)


@mcp.tool()
def holarch_health() -> Dict[str, Any]:
    """Report parameter count, receipt count, and chain validity."""
    return service().health()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m henri.holarch.mcp_server")
    parser.add_argument("--transport", default="stdio", choices=["stdio", "sse", "streamable-http"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args(argv)
    mcp.settings.host = str(args.host)
    mcp.settings.port = int(args.port)
    mcp.run(transport=str(args.transport))
    return 0


if __name__ == "__main__":
    sys.exit(main())
