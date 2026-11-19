"""Unified Model Context Protocol (MCP) surface over UnifiedGatewayService.

Every MCP tool calls the exact same `UnifiedGatewayService` methods as REST
and `/v1/chat/completions`, guaranteeing zero behavioral divergence and full
Ed25519-signed receipt continuity across all three transports.

Supports both MCP 1.x (`FastMCP`) and MCP 2.x (`MCPServer`).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from henri.gateway.auth import TenantPolicy
from henri.gateway.service import UnifiedGatewayService


def _create_mcp_server(name: str, instructions: str) -> Any:
    try:
        from mcp.server.fastmcp import FastMCP

        return FastMCP(name, instructions=instructions)
    except ImportError:
        from mcp.server.mcpserver import MCPServer

        return MCPServer(name, instructions=instructions)


def build_mcp_server(
    service: UnifiedGatewayService,
    *,
    default_policy: Optional[TenantPolicy] = None,
) -> Any:
    """Build an MCP server bound directly to `UnifiedGatewayService`."""
    policy = default_policy or TenantPolicy(
        key_id="mcp-local",
        tenant_id="default",
        allowed_namespaces=("*",),
        default_namespace="default",
        rpm_limit=service.config.default_rpm,
        burst_limit=service.config.default_burst,
        llm_call_budget=service.config.default_llm_budget,
        allow_cascade=True,
        allow_sandbox_exec=True,
        authenticated=True,
    )

    mcp = _create_mcp_server(
        "henri-v6-gateway",
        instructions=(
            "Fail-closed HENRI-v6 wave-mechanical world model & epistemic gateway. "
            "Every tool returns an Ed25519-signed, SHA-256 hash-chained receipt with "
            "explicit OBSERVED/DERIVED/BLOCKED evidence labels."
        ),
    )

    @mcp.tool()
    def holarch_health() -> Dict[str, Any]:
        """Return gateway health, signing KID, sandbox status, and Holarch checkpoint info."""
        return service.healthz()

    @mcp.tool()
    def holarch_veto(
        task: str, x: Any, rule_id: Optional[int] = None, namespace: str = "default"
    ) -> Dict[str, Any]:
        """Return Sagnac residual, top-1 Gram coherence, and manifold veto status."""
        ns = service.auth.authorize_namespace(policy, namespace)
        return service.veto(task, x, rule_id, ns)

    @mcp.tool()
    def holarch_ingest_code(
        root: str,
        package: str = "henri",
        namespace: str = "default",
        mode: str = "edges",
        file_path: Optional[str] = None,
        source_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Index package symbols and call edges into Zone C at zero weight cost."""
        ns = service.auth.authorize_namespace(policy, namespace)
        return service.ingest_code(root, package, ns, mode, file_path, source_text)

    @mcp.tool()
    def holarch_plan(
        observation: Any,
        history_waves: Optional[List[Any]] = None,
        transition: str = "rotor",
        lookahead_depth: int = 2,
        gamma_look: float = 0.02,
        namespace: str = "default",
        episode_id: Optional[str] = None,
        online_repulsor: bool = False,
        reset_episode: bool = False,
        emit_swat_kv: bool = False,
    ) -> Dict[str, Any]:
        """Depth-2 wall-masked rotor planning with stateful online REPULSOR axioms."""
        ns = service.auth.authorize_namespace(policy, namespace)
        return service.plan(
            observation,
            history_waves,
            transition,
            lookahead_depth,
            gamma_look,
            ns,
            True,
            episode_id,
            online_repulsor,
            reset_episode,
            emit_swat_kv,
        )

    @mcp.tool()
    def holarch_verify(
        claim: str, kind: str = "text", top_k: int = 4, namespace: str = "default"
    ) -> Dict[str, Any]:
        """Verify prose via OntologyGuard or code via AST + code index."""
        ns = service.auth.authorize_namespace(policy, namespace)
        return service.verify(claim, kind, top_k, ns)

    @mcp.tool()
    def holarch_ingest_edges(
        edges: List[Dict[str, str]], namespace: str = "default"
    ) -> Dict[str, Any]:
        """Store an unseen symbolic rule graph in Zone C."""
        ns = service.auth.authorize_namespace(policy, namespace)
        return service.ingest_edges(edges, ns)

    @mcp.tool()
    def holarch_solve(
        source: str, operators: Any, namespace: str = "default", mode: str = "path"
    ) -> Dict[str, Any]:
        """Traverse an operator sequence in Zone C or abstain fail-closed."""
        ns = service.auth.authorize_namespace(policy, namespace)
        return service.solve(source, operators, ns, mode)

    @mcp.tool()
    def holarch_synthesize(
        intent: str, kind: str = "code", namespace: str = "default", max_tokens: int = 320
    ) -> Dict[str, Any]:
        """Emit AST/ontology-verified code or grounded text, or refuse."""
        ns = service.auth.authorize_namespace(policy, namespace)
        return service.synthesize(intent, kind, ns, max_tokens)

    @mcp.tool()
    def holarch_playbook(
        name: str,
        steps: Optional[List[str]] = None,
        merge: bool = True,
        namespace: str = "default",
    ) -> Dict[str, Any]:
        """Compile ACE delta playbook steps or recall exact stored plan."""
        ns = service.auth.authorize_namespace(policy, namespace)
        return service.playbook(name, steps, merge, ns)

    @mcp.tool()
    def holarch_adapt_rule(
        task: str,
        x: Any,
        y: List[int],
        rule_id: Optional[int] = None,
        lr: float = 0.2,
        namespace: str = "default",
    ) -> Dict[str, Any]:
        """Surprise-gated Zone A fast-weight write, isolated per tenant namespace."""
        ns = service.auth.authorize_namespace(policy, namespace)
        return service.adapt_rule(task, x, y, rule_id, lr, ns)

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
        """Register or resolve a multi-arity composite qFHRR rule."""
        ns = service.auth.authorize_namespace(policy, namespace)
        return service.composite_call(
            mode=mode, rules=rules, a=a, b=b, op=op, x0=x0, x1=x1, namespace=ns
        )

    @mcp.tool()
    def holarch_agent_step(
        action: str,
        payload: Optional[Dict[str, Any]] = None,
        namespace: str = "default",
    ) -> Dict[str, Any]:
        """Run one deterministic epistemic agent step."""
        ns = service.auth.authorize_namespace(policy, namespace)
        return service.agent_step(action, dict(payload or {}), ns)

    # -- Cycle AD solvers ----------------------------------------------------
    @mcp.tool()
    def holarch_repo_patch(
        name: str,
        issue: str,
        files: Dict[str, str],
        public: Dict[str, str],
        test_path: str = "tests/test_public.py",
        rollouts: int = 24,
        namespace: str = "default",
        cascade_policy: str = "cascade",
    ) -> Dict[str, Any]:
        """Localize faults, synthesize and sandbox-verify a multi-file Python patch."""
        ns = service.auth.authorize_namespace(policy, namespace)
        return service.repo_patch(
            policy=policy,
            name=name,
            issue=issue,
            files=files,
            public=public,
            test_path=test_path,
            rollouts=rollouts,
            timeout_s=30.0,
            namespace=ns,
            cascade_policy=cascade_policy,
        )

    @mcp.tool()
    def holarch_science_solve(
        problem: Dict[str, Any], namespace: str = "default"
    ) -> Dict[str, Any]:
        """Solve a quantitative science problem with dimensional, inverse, and ontology checks."""
        ns = service.auth.authorize_namespace(policy, namespace)
        return service.science_solve(
            policy=policy, problem=problem, namespace=ns, cascade_policy="wave_only"
        )

    @mcp.tool()
    def holarch_arc_solve(
        train: List[Dict[str, Any]],
        test: List[Dict[str, Any]],
        depth: int = 2,
        namespace: str = "default",
    ) -> Dict[str, Any]:
        """Induce an object-centric ARC program under strict Leave-One-Out gating."""
        ns = service.auth.authorize_namespace(policy, namespace)
        return service.arc_solve(
            train=train,
            test=test,
            node_budget=60_000,
            depth=depth,
            time_budget=10.0,
            namespace=ns,
        )

    @mcp.tool()
    def holarch_atlas_build(
        vectors: List[Any],
        rows: Optional[List[int]] = None,
        capacity: int = 192,
        namespace: str = "default",
    ) -> Dict[str, Any]:
        """Build a packed contiguous-BLAS RiemannianChartAtlas for a namespace."""
        ns = service.auth.authorize_namespace(policy, namespace)
        return service.atlas_build(
            vectors=vectors, rows=rows, capacity=capacity, namespace=ns
        )

    @mcp.tool()
    def holarch_atlas_observe(
        vector: List[Any],
        row: int,
        finalize: bool = False,
        namespace: str = "default",
    ) -> Dict[str, Any]:
        """Record an online vector write with coherent condensation and Voronoi dual-membership."""
        ns = service.auth.authorize_namespace(policy, namespace)
        return service.atlas_observe(
            vector=vector, row=row, finalize=finalize, namespace=ns
        )

    @mcp.tool()
    def holarch_atlas_query(
        vector: List[Any],
        k: int = 1,
        namespace: str = "default",
    ) -> Dict[str, Any]:
        """Query the packed RiemannianChartAtlas over contiguous chart blocks."""
        ns = service.auth.authorize_namespace(policy, namespace)
        return service.atlas_query(vector=vector, k=k, namespace=ns)

    @mcp.tool()
    def holarch_atlas_stats(namespace: str = "default") -> Dict[str, Any]:
        """Return packed RiemannianChartAtlas statistics for a namespace."""
        ns = service.auth.authorize_namespace(policy, namespace)
        return service.atlas_stats(namespace=ns)

    # -- DigitalTwin D=65536 -------------------------------------------------
    @mcp.tool()
    def digital_twin_rank(
        question: str,
        options: List[str],
        strip: bool = True,
        channel_weight: float = 0.0,
        namespace: str = "default",
    ) -> Dict[str, Any]:
        """Rank candidate options through the D=65536 wave kernel with Sagnac telemetry."""
        ns = service.auth.authorize_namespace(policy, namespace)
        return service.reason_rank(
            question=question,
            options=options,
            strip=strip,
            signing_key=None,
            channel_weight=channel_weight,
            namespace=ns,
        )

    return mcp


def create_mcp_sse_subapp(service: UnifiedGatewayService) -> Optional[Any]:
    """Return an ASGI sub-app for mounting MCP SSE under `/mcp` on FastAPI."""
    mcp = build_mcp_server(service)
    if hasattr(mcp, "sse_app"):
        return mcp.sse_app()
    if hasattr(mcp, "streamable_http_app"):
        return mcp.streamable_http_app()
    return None
