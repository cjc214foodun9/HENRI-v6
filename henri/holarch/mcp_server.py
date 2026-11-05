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
        # Directive E2: a service built without a pinned profile mounts the
        # converged checkpoint that matches the host, rather than forcing a
        # fresh 120-step small bootstrap.
        _SERVICE = SidecarService(backend=backend, memory_path=path)
    return _SERVICE


def set_service(value: Optional[SidecarService]) -> None:
    """Inject a service (tests, or a process that already holds one)."""
    global _SERVICE
    _SERVICE = value


@mcp.tool()
def holarch_veto(
    task: str, x: Any, rule_id: Optional[int] = None, namespace: str = "default"
) -> Dict[str, Any]:
    """Return the veto rate, top-1 Gram coherence, and Sagnac residual.

    Args:
        task: one of vision, language, action, relational.
        x: nested pixel or observation payload for one or more examples.
        rule_id: optional slot-3 rule context.
    """
    return service().veto(task, x, rule_id, namespace)


@mcp.tool()
def holarch_ingest_code(
    root: str,
    package: str = "henri",
    namespace: str = "default",
    mode: str = "edges",
    file_path: Optional[str] = None,
    source_text: Optional[str] = None,
) -> Dict[str, Any]:
    """Parse a package once and index its symbols and call edges in Zone C.

    Query time performs no file reads. The graph is stored as wave engrams,
    so no model weight changes. mode="delta" re-indexes exactly ONE file
    (file_path, optional source_text) without re-parsing the package.
    """
    return service().ingest_code(root, package, namespace, mode, file_path, source_text)


@mcp.tool()
def holarch_plan(
    observation: Any,
    history_waves: Optional[List[Any]] = None,
    transition: str = "rotor",
    lookahead_depth: int = 2,
    gamma_look: float = 0.02,
    namespace: str = "default",
) -> Dict[str, Any]:
    """Depth-2 wall-masked rotor planning on the mounted checkpoint.

    Expects the 79-value sensor observation (5x5x3 local view plus four
    relative coordinates). history_waves optionally carries recent waves
    for the cycle penalty. Returns the action, its name, all four scores,
    the goal alignment, the second-step lookahead values, the cycle
    penalty, the legal-move mask, and a SHA-256 receipt.
    """
    return service().plan(
        observation, history_waves, transition, lookahead_depth, gamma_look, namespace
    )


@mcp.tool()
def holarch_verify(
    claim: str, kind: str = "text", top_k: int = 4, namespace: str = "default"
) -> Dict[str, Any]:
    """Verify a prose claim against the corpus, or code against the AST.

    Text runs the ontology guard: every rare term and every quantity must
    appear in the retrieved evidence, otherwise the claim is contradicted
    or unsupported. Code is parsed, compiled, and every callee is checked
    against the code index, so a fabricated signature is refused.
    """
    return service().verify(claim, kind, top_k, namespace)


@mcp.tool()
def holarch_ingest_edges(
    edges: List[Dict[str, str]], namespace: str = "default"
) -> Dict[str, Any]:
    """Store an unseen rule graph. Each edge has source, operator, target."""
    return service().ingest_edges(edges, namespace)


@mcp.tool()
def holarch_solve(
    source: str, operators: Any, namespace: str = "default", mode: str = "path"
) -> Dict[str, Any]:
    """Follow an operator sequence from source, or abstain fail-closed.

    operators may be a list or a comma-separated string. An unreachable
    path returns vetoed=true with text starting ABSTAIN and no invented
    symbol.
    """
    return service().solve(source, operators, namespace, mode)


@mcp.tool()
def holarch_playbook(
    name: str,
    steps: Optional[List[str]] = None,
    merge: bool = True,
    namespace: str = "default",
) -> Dict[str, Any]:
    """Compile steps into a versioned playbook, or recall the exact steps.

    With steps the call merges them (ACE delta form) and bumps the
    version. Without steps it recalls and decodes the stored plan.
    """
    return service().playbook(name, steps, merge, namespace)


@mcp.tool()
def holarch_adapt_rule(
    task: str,
    x: Any,
    y: List[int],
    rule_id: Optional[int] = None,
    lr: float = 0.2,
    namespace: str = "default",
) -> Dict[str, Any]:
    """One surprise-gated fast-weight write, with the rule in slot 3."""
    return service().adapt_rule(task, x, y, rule_id, lr, namespace)


@mcp.tool()
def holarch_composite(
    mode: str = "query",
    rules: Optional[List[Dict[str, Any]]] = None,
    a: Any = None,
    b: Any = None,
    op: Any = None,
    x0: Any = None,
    x1: Any = None,
    namespace: str = "default",
) -> Dict[str, Any]:
    """Register or resolve a multi-arity composite rule (Decision D4).

    mode=ingest stores rows {a, op, b, result} at zero weight cost.
    mode=query resolves digits (a, op, b) or grounds two 8x8 quadrant
    payloads (x0, x1) through the Pass-0 prototype bank, then unbinds the
    composite qFHRR key in Zone C. A coherence below tau returns
    abstained=true with result 'ABSTAIN: coherence below tau'.
    """
    return service().composite_call(
        mode=mode, rules=rules, a=a, b=b, op=op, x0=x0, x1=x1, namespace=namespace
    )


@mcp.tool()
def holarch_synthesize(
    intent: str,
    kind: str = "code",
    namespace: str = "default",
    max_tokens: int = 320,
) -> Dict[str, Any]:
    """Emit verified code or grounded text through the wave decoder, or refuse.

    The emitted source must pass ast.parse and compile (code) or the ontology
    verifier (text). A request that names a symbol the code index does not
    contain is refused, never invented. Every call returns a SHA-256 receipt.
    """
    return service().synthesize(intent, kind, namespace, max_tokens)


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
