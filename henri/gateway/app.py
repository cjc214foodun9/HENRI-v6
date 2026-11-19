"""FastAPI application factory for the HENRI-v6 Unified Production Gateway.

Preserves 100% route compatibility with both:
  * `henri/holarch/server.py` (`/healthz`, `/v1/holarch/*`, `/v1/chat/completions`)
  * `henri/api_server.py` (`/v1/health/*`, `/v1/reason/rank`, `/v1/cartridges/*`,
    `/v1/code/*`, `/v1/robotics/step`)

Adds:
  * Cycle AD solver endpoints: `/v1/holarch/repo_patch`, `/v1/holarch/science`,
    `/v1/holarch/arc`
  * Ed25519 JWKS discovery: `GET /.well-known/henri-jwks.json`
  * Prometheus observability: `GET /metrics`
  * Mounted MCP SSE transport: `/mcp`
  * Scoped API-key auth, fail-closed namespace authorization, per-tenant token
    bucket rate limiting, and bounded compute-pool offloading.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from henri.api_server import (
    MAX_ITEMS,
    MAX_SENSOR_VALUES,
    MAX_TEXT_CHARS,
    CartridgeMountRequest,
    CartridgeWriteRequest,
    CodeBankIngestRequest,
    CodeSolveRequest,
    RankRequest,
    RoboticsStepRequest,
)
from henri.gateway.auth import TenantPolicy
from henri.gateway.config import GatewayConfig
from henri.gateway.mcp_mount import create_mcp_sse_subapp
from henri.gateway.service import UnifiedGatewayService
from henri.holarch.server import (
    AdaptRuleRequest,
    AgentStepRequest,
    ChatCompletionRequest,
    CompositeRequest,
    DiscoverRequest,
    IngestCodeRequest,
    IngestEdgesRequest,
    PlanRequest,
    PlaybookRequest,
    SolveRequest,
    SynthesizeRequest,
    TelepathyRequest,
    TuringRequest,
    VerifyRequest,
    VetoRequest,
)


# -- Additive request models for Cycle AD & optional namespace fields --------
class GatewayRankRequest(RankRequest):
    namespace: Optional[str] = None


class GatewayRoboticsStepRequest(RoboticsStepRequest):
    namespace: Optional[str] = None


class RepoPatchRequest(BaseModel):
    name: str = "case"
    issue: str
    files: Dict[str, str]
    public: Dict[str, str]
    test_path: str = "tests/test_public.py"
    rollouts: int = 24
    timeout_s: float = 30.0
    namespace: Optional[str] = None
    cascade_policy: str = "cascade"


class ScienceSolveRequest(BaseModel):
    problem: Optional[Dict[str, Any]] = None
    prompt: Optional[str] = None
    problem_text: Optional[str] = None
    target_unit: Optional[str] = None
    namespace: Optional[str] = None
    cascade_policy: str = "wave_only"


class ArcPair(BaseModel):
    input: List[List[int]]
    output: Optional[List[List[int]]] = None


class ArcSolveRequest(BaseModel):
    train: List[ArcPair]
    test: List[ArcPair] = Field(default_factory=list)
    node_budget: int = 60_000
    depth: int = 2
    time_budget: float = 10.0
    namespace: Optional[str] = None


class AtlasBuildRequest(BaseModel):
    vectors: List[Any]
    rows: Optional[List[int]] = None
    rank: int = 8
    capacity: int = 192
    top_charts: int = 2
    top_charts_max: int = 4
    namespace: Optional[str] = None


class AtlasObserveRequest(BaseModel):
    vector: List[Any]
    row: int
    finalize: bool = False
    namespace: Optional[str] = None


class AtlasQueryRequest(BaseModel):
    vector: List[Any]
    k: int = 1
    top_charts: Optional[int] = None
    namespace: Optional[str] = None


def create_gateway_app(
    service: Optional[UnifiedGatewayService] = None,
    *,
    config: Optional[GatewayConfig] = None,
    mount_mcp: bool = True,
) -> FastAPI:
    """Create the unified HENRI-v6 FastAPI application."""
    svc = service or UnifiedGatewayService(config=config)
    app = FastAPI(
        title="HENRI-v6 Unified Wave-Mechanical World Model Gateway",
        version="0.6.0",
        description=(
            "Unified production API combining Holarch (Zones A/B/C), Cycle AD "
            "deterministic solvers, and the D=65536 DigitalTwin wave kernel "
            "with governed Arm0->Arm4 LLM cascade and Ed25519-signed receipts."
        ),
    )
    app.state.gateway = svc

    # ------------------------------------------------- latency & metrics hook
    @app.middleware("http")
    async def observability_middleware(request: Request, call_next):
        started = time.perf_counter()
        endpoint = request.url.path
        status_code = 500
        with svc.telemetry.span(f"http {request.method} {endpoint}", {"http.route": endpoint}):
            try:
                response = await call_next(request)
                status_code = response.status_code
                return response
            finally:
                elapsed = time.perf_counter() - started
                svc.telemetry.requests_total.labels(
                    endpoint=endpoint, status=str(status_code)
                ).inc()
                svc.telemetry.request_duration_seconds.labels(endpoint=endpoint).observe(
                    elapsed
                )

    # ----------------------------------------------------- auth dependency
    def require_tenant(
        authorization: Optional[str] = Header(default=None, alias="Authorization")
    ) -> TenantPolicy:
        policy = svc.auth.authenticate(authorization)
        svc.auth.check_rate_limit(policy)
        return policy

    # ---------------------------------------------- public / ops endpoints
    @app.get("/healthz")
    async def healthz() -> Dict[str, Any]:
        return await svc.run_in_pool(svc.healthz)

    @app.get("/.well-known/henri-jwks.json")
    async def jwks() -> Dict[str, Any]:
        return svc.jwks()

    @app.get("/metrics")
    async def metrics(
        authorization: Optional[str] = Header(default=None, alias="Authorization")
    ) -> Response:
        if not svc.config.enable_metrics:
            raise HTTPException(status_code=404, detail="metrics disabled")
        svc.auth.authorize_ops(authorization)
        body, media_type = svc.telemetry.render_prometheus()
        return Response(content=body, media_type=media_type)

    # ---------------------------------------- Holarch world-model endpoints
    @app.post("/v1/holarch/veto")
    async def holarch_veto(
        req: VetoRequest, policy: TenantPolicy = Depends(require_tenant)
    ) -> Dict[str, Any]:
        ns = svc.auth.authorize_namespace(policy, req.namespace)
        return await svc.run_in_pool(svc.veto, req.task, req.x, req.rule_id, ns)

    @app.post("/v1/holarch/ingest_code")
    async def holarch_ingest_code(
        req: IngestCodeRequest, policy: TenantPolicy = Depends(require_tenant)
    ) -> Dict[str, Any]:
        ns = svc.auth.authorize_namespace(policy, req.namespace)
        try:
            return await svc.run_in_pool(
                svc.ingest_code,
                req.root,
                req.package,
                ns,
                req.mode,
                req.file_path,
                req.source_text,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/holarch/plan")
    async def holarch_plan(
        req: PlanRequest, policy: TenantPolicy = Depends(require_tenant)
    ) -> Dict[str, Any]:
        ns = svc.auth.authorize_namespace(policy, req.namespace)
        try:
            return await svc.run_in_pool(
                svc.plan,
                req.observation,
                req.history_waves,
                req.transition,
                req.lookahead_depth,
                req.gamma_look,
                ns,
                req.batch_rollout,
                req.episode_id,
                req.online_repulsor,
                req.reset_episode,
                req.emit_swat_kv,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/holarch/verify")
    async def holarch_verify(
        req: VerifyRequest, policy: TenantPolicy = Depends(require_tenant)
    ) -> Dict[str, Any]:
        ns = svc.auth.authorize_namespace(policy, req.namespace)
        return await svc.run_in_pool(svc.verify, req.claim, req.kind, req.top_k, ns)

    @app.post("/v1/holarch/ingest_edges")
    async def holarch_ingest_edges(
        req: IngestEdgesRequest, policy: TenantPolicy = Depends(require_tenant)
    ) -> Dict[str, Any]:
        ns = svc.auth.authorize_namespace(policy, req.namespace)
        return await svc.run_in_pool(svc.ingest_edges, req.edges, ns)

    @app.post("/v1/holarch/solve")
    async def holarch_solve(
        req: SolveRequest, policy: TenantPolicy = Depends(require_tenant)
    ) -> Dict[str, Any]:
        ns = svc.auth.authorize_namespace(policy, req.namespace)
        return await svc.run_in_pool(svc.solve, req.source, req.operators, ns, req.mode)

    @app.post("/v1/holarch/synthesize")
    async def holarch_synthesize(
        req: SynthesizeRequest, policy: TenantPolicy = Depends(require_tenant)
    ) -> Dict[str, Any]:
        ns = svc.auth.authorize_namespace(policy, req.namespace)
        return await svc.run_in_pool(
            svc.synthesize, req.intent, req.kind, ns, req.max_tokens
        )

    @app.post("/v1/holarch/telepathy")
    async def holarch_telepathy(
        req: TelepathyRequest, policy: TenantPolicy = Depends(require_tenant)
    ) -> Dict[str, Any]:
        ns = svc.auth.authorize_namespace(policy, req.namespace)
        try:
            return await svc.run_in_pool(
                svc.telepathy,
                mode=req.mode,
                session_id=req.session_id,
                namespace=ns,
                chunks=req.chunks,
                query=req.query,
                budget_u=req.budget_u,
                per_chunk_u=req.per_chunk_u,
                hops=req.hops,
                top_k=req.top_k,
                reset=req.reset,
                raw_documents=req.raw_documents,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/holarch/turing")
    async def holarch_turing(
        req: TuringRequest, policy: TenantPolicy = Depends(require_tenant)
    ) -> Dict[str, Any]:
        ns = svc.auth.authorize_namespace(policy, req.namespace)
        try:
            return await svc.run_in_pool(
                svc.turing,
                req.mode,
                req.program,
                req.tape,
                req.max_steps,
                ns,
                req.examples,
                req.alphabet,
                req.name,
                namespace=ns,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/holarch/discover")
    async def holarch_discover(
        req: DiscoverRequest, policy: TenantPolicy = Depends(require_tenant)
    ) -> Dict[str, Any]:
        ns = svc.auth.authorize_namespace(policy, req.namespace)
        if not policy.allow_sandbox_exec:
            raise HTTPException(status_code=403, detail="sandbox execution disabled for tenant")
        try:
            return await svc.run_in_pool(
                svc.discover, req.intent, req.tests, req.request, ns, req.library
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/holarch/playbook")
    async def holarch_playbook(
        req: PlaybookRequest, policy: TenantPolicy = Depends(require_tenant)
    ) -> Dict[str, Any]:
        ns = svc.auth.authorize_namespace(policy, req.namespace)
        return await svc.run_in_pool(svc.playbook, req.name, req.steps, req.merge, ns)

    @app.post("/v1/holarch/adapt_rule")
    async def holarch_adapt_rule(
        req: AdaptRuleRequest, policy: TenantPolicy = Depends(require_tenant)
    ) -> Dict[str, Any]:
        ns = svc.auth.authorize_namespace(policy, req.namespace)
        return await svc.run_in_pool(
            svc.adapt_rule, req.task, req.x, req.y, req.rule_id, req.lr, ns
        )

    @app.post("/v1/holarch/composite")
    async def holarch_composite(
        req: CompositeRequest, policy: TenantPolicy = Depends(require_tenant)
    ) -> Dict[str, Any]:
        ns = svc.auth.authorize_namespace(policy, req.namespace)
        rules = None if req.rules is None else [r.model_dump() for r in req.rules]
        try:
            return await svc.run_in_pool(
                svc.composite_call,
                mode=req.mode,
                rules=rules,
                a=req.a,
                b=req.b,
                op=req.op,
                x0=req.x0,
                x1=req.x1,
                namespace=ns,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/holarch/agent_step")
    async def holarch_agent_step(
        req: AgentStepRequest, policy: TenantPolicy = Depends(require_tenant)
    ) -> Dict[str, Any]:
        ns = svc.auth.authorize_namespace(policy, req.namespace)
        try:
            return await svc.run_in_pool(svc.agent_step, req.action, req.payload, ns)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/holarch/receipt")
    async def holarch_receipt(
        limit: int = 16,
        namespace: Optional[str] = None,
        policy: TenantPolicy = Depends(require_tenant),
    ) -> Dict[str, Any]:
        ns = svc.auth.authorize_namespace(policy, namespace)
        return await svc.run_in_pool(svc.receipts, ns, limit)

    # ---------------------------------------------- Cycle AD solver routes
    @app.post("/v1/holarch/repo_patch")
    async def holarch_repo_patch(
        req: RepoPatchRequest, policy: TenantPolicy = Depends(require_tenant)
    ) -> Dict[str, Any]:
        ns = svc.auth.authorize_namespace(policy, req.namespace)
        if not req.files:
            raise HTTPException(status_code=422, detail="files must not be empty")
        if not req.public:
            raise HTTPException(status_code=422, detail="public test suite is required")
        try:
            return await svc.run_in_pool(
                svc.repo_patch,
                policy=policy,
                name=req.name,
                issue=req.issue,
                files=req.files,
                public=req.public,
                test_path=req.test_path,
                rollouts=req.rollouts,
                timeout_s=req.timeout_s,
                namespace=ns,
                cascade_policy=req.cascade_policy,
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc

    @app.post("/v1/holarch/science")
    async def holarch_science(
        req: ScienceSolveRequest, policy: TenantPolicy = Depends(require_tenant)
    ) -> Dict[str, Any]:
        ns = svc.auth.authorize_namespace(policy, req.namespace)
        raw_prompt = req.prompt if req.prompt is not None else req.problem_text
        if req.problem is None and (raw_prompt is None or not str(raw_prompt).strip()):
            raise HTTPException(
                status_code=422,
                detail="either problem or prompt/problem_text is required",
            )
        return await svc.run_in_pool(
            svc.science_solve,
            policy=policy,
            problem=req.problem,
            prompt=raw_prompt,
            target_unit=req.target_unit,
            namespace=ns,
            cascade_policy=req.cascade_policy,
        )

    @app.post("/v1/holarch/arc")
    async def holarch_arc(
        req: ArcSolveRequest, policy: TenantPolicy = Depends(require_tenant)
    ) -> Dict[str, Any]:
        ns = svc.auth.authorize_namespace(policy, req.namespace)
        train_dicts = [p.model_dump() for p in req.train]
        if any(p.get("output") is None for p in train_dicts):
            raise HTTPException(status_code=422, detail="each train pair requires input and output")
        test_dicts = [p.model_dump() for p in req.test]
        return await svc.run_in_pool(
            svc.arc_solve,
            train=train_dicts,
            test=test_dicts,
            node_budget=req.node_budget,
            depth=req.depth,
            time_budget=req.time_budget,
            namespace=ns,
        )

    # ------------------------------------ Packed RiemannianChartAtlas (AF1)
    @app.post("/v1/holarch/atlas/build")
    async def holarch_atlas_build(
        req: AtlasBuildRequest, policy: TenantPolicy = Depends(require_tenant)
    ) -> Dict[str, Any]:
        ns = svc.auth.authorize_namespace(policy, req.namespace)
        try:
            return await svc.run_in_pool(
                svc.atlas_build,
                vectors=req.vectors,
                rows=req.rows,
                rank=req.rank,
                capacity=req.capacity,
                top_charts=req.top_charts,
                top_charts_max=req.top_charts_max,
                namespace=ns,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/holarch/atlas/observe")
    async def holarch_atlas_observe(
        req: AtlasObserveRequest, policy: TenantPolicy = Depends(require_tenant)
    ) -> Dict[str, Any]:
        ns = svc.auth.authorize_namespace(policy, req.namespace)
        try:
            return await svc.run_in_pool(
                svc.atlas_observe,
                vector=req.vector,
                row=req.row,
                finalize=req.finalize,
                namespace=ns,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/holarch/atlas/query")
    async def holarch_atlas_query(
        req: AtlasQueryRequest, policy: TenantPolicy = Depends(require_tenant)
    ) -> Dict[str, Any]:
        ns = svc.auth.authorize_namespace(policy, req.namespace)
        try:
            return await svc.run_in_pool(
                svc.atlas_query,
                vector=req.vector,
                k=req.k,
                top_charts=req.top_charts,
                namespace=ns,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/v1/holarch/atlas/stats")
    async def holarch_atlas_stats(
        namespace: Optional[str] = None, policy: TenantPolicy = Depends(require_tenant)
    ) -> Dict[str, Any]:
        ns = svc.auth.authorize_namespace(policy, namespace)
        return await svc.run_in_pool(svc.atlas_stats, namespace=ns)

    # ------------------------------------------ OpenAI-compatible chat SSE
    @app.post("/v1/chat/completions")
    async def chat_completions(
        req: ChatCompletionRequest, policy: TenantPolicy = Depends(require_tenant)
    ):
        meta = dict(req.metadata or {})
        ns = svc.auth.authorize_namespace(policy, meta.get("namespace"))
        payload = await svc.run_in_pool(
            svc.chat_completion,
            policy=policy,
            model=req.model,
            messages=[m.model_dump() for m in req.messages],
            tools=req.tools,
            metadata=meta,
            namespace=ns,
        )
        if not bool(req.stream):
            return payload

        lines = svc.stream_chat_lines(payload)

        async def _events():
            for line in lines:
                yield line + "\n\n"

        return StreamingResponse(
            _events(),
            media_type="text/event-stream",
            headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
        )

    # ---------------------------------------- DigitalTwin D=65536 endpoints
    @app.get("/v1/health/seal")
    async def health_seal(
        namespace: Optional[str] = None, policy: TenantPolicy = Depends(require_tenant)
    ) -> Dict[str, Any]:
        ns = svc.auth.authorize_namespace(policy, namespace)
        return await svc.run_in_pool(svc.health_seal, ns)

    @app.get("/v1/health/executor")
    async def health_executor(
        namespace: Optional[str] = None, policy: TenantPolicy = Depends(require_tenant)
    ) -> Dict[str, Any]:
        ns = svc.auth.authorize_namespace(policy, namespace)
        return await svc.run_in_pool(svc.health_executor, ns)

    @app.post("/v1/reason/rank")
    async def reason_rank(
        req: GatewayRankRequest, policy: TenantPolicy = Depends(require_tenant)
    ) -> Dict[str, Any]:
        if not req.question.strip():
            raise HTTPException(422, "question is required")
        if len(req.question) > MAX_TEXT_CHARS:
            raise HTTPException(422, f"question exceeds {MAX_TEXT_CHARS} characters")
        if len(req.options) < 2:
            raise HTTPException(422, "at least two options are required")
        if len(req.options) > MAX_ITEMS:
            raise HTTPException(422, f"at most {MAX_ITEMS} options are permitted")
        if any(len(opt) > MAX_TEXT_CHARS for opt in req.options):
            raise HTTPException(422, f"an option exceeds {MAX_TEXT_CHARS} characters")
        if not math.isfinite(float(req.channel_weight)):
            raise HTTPException(422, "channel_weight must be finite")
        ns = svc.auth.authorize_namespace(policy, req.namespace)
        return await svc.run_in_pool(
            svc.reason_rank,
            question=req.question,
            options=req.options,
            strip=req.strip,
            signing_key=req.signing_key,
            channel_weight=req.channel_weight,
            namespace=ns,
        )

    @app.post("/v1/cartridges/write")
    async def cartridges_write(
        req: CartridgeWriteRequest, policy: TenantPolicy = Depends(require_tenant)
    ) -> Dict[str, Any]:
        if not req.tenant_id.strip():
            raise HTTPException(422, "tenant_id is required")
        if len(req.keys) != len(req.values):
            raise HTTPException(422, "keys and values must have equal length")
        if not req.keys:
            raise HTTPException(422, "at least one pair is required")
        if len(req.keys) > MAX_ITEMS:
            raise HTTPException(422, f"at most {MAX_ITEMS} pairs are permitted")
        if any(len(k) > MAX_TEXT_CHARS for k in req.keys) or any(
            len(v) > MAX_TEXT_CHARS for v in req.values
        ):
            raise HTTPException(422, f"a key or value exceeds {MAX_TEXT_CHARS} characters")
        ns = svc.auth.authorize_namespace(policy, req.tenant_id)
        try:
            return await svc.run_in_pool(
                svc.cartridges_write,
                tenant_id=ns,
                keys=req.keys,
                values=req.values,
                subject=req.subject,
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/v1/cartridges/mount")
    async def cartridges_mount(
        req: CartridgeMountRequest, policy: TenantPolicy = Depends(require_tenant)
    ) -> Dict[str, Any]:
        ns = svc.auth.authorize_namespace(policy, req.tenant_id)
        res = await svc.run_in_pool(svc.cartridges_mount, tenant_id=ns)
        if res is None:
            raise HTTPException(404, f"namespace {ns!r} has no bank; write first")
        return res

    @app.post("/v1/code/solve")
    async def code_solve(
        req: CodeSolveRequest, policy: TenantPolicy = Depends(require_tenant)
    ) -> Dict[str, Any]:
        if not req.spec.strip():
            raise HTTPException(422, "spec is required")
        if len(req.spec) > MAX_TEXT_CHARS:
            raise HTTPException(422, f"spec exceeds {MAX_TEXT_CHARS} characters")
        if len(req.candidates) > MAX_ITEMS or len(req.tests) > MAX_ITEMS:
            raise HTTPException(422, f"at most {MAX_ITEMS} candidates and tests are permitted")
        if not req.candidates and not req.synthesize_from_bank:
            raise HTTPException(
                422, "at least one candidate is required unless synthesize_from_bank is set"
            )
        ns = svc.auth.authorize_namespace(policy, req.tenant_id)
        try:
            return await svc.run_in_pool(
                svc.code_solve,
                policy=policy,
                spec=req.spec,
                candidates=req.candidates,
                tests=req.tests,
                top_k=req.top_k,
                timeout=req.timeout,
                synthesize_from_bank=req.synthesize_from_bank,
                k_templates=req.k_templates,
                max_variants=req.max_variants,
                tenant_id=ns,
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc

    @app.post("/v1/code/bank/ingest")
    async def code_bank_ingest(
        req: CodeBankIngestRequest, policy: TenantPolicy = Depends(require_tenant)
    ) -> Dict[str, Any]:
        if req.limit <= 0:
            raise HTTPException(422, "limit must be positive")
        ns = svc.auth.authorize_namespace(policy, req.tenant_id)
        try:
            return await svc.run_in_pool(
                svc.code_bank_ingest, tenant_id=ns, limit=req.limit
            )
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/v1/robotics/step")
    async def robotics_step(
        req: GatewayRoboticsStepRequest, policy: TenantPolicy = Depends(require_tenant)
    ) -> Dict[str, Any]:
        if not req.sensor_values:
            raise HTTPException(422, "sensor_values must not be empty")
        if len(req.sensor_values) > MAX_SENSOR_VALUES:
            raise HTTPException(422, f"at most {MAX_SENSOR_VALUES} sensor values are permitted")
        arr = np.asarray(req.sensor_values, dtype=float)
        if not np.all(np.isfinite(arr)):
            raise HTTPException(422, "sensor_values must all be finite")
        if req.axiom_reference is not None:
            if len(req.axiom_reference) != len(req.sensor_values):
                raise HTTPException(
                    422,
                    f"axiom_reference has {len(req.axiom_reference)} entries, "
                    f"sensor_values has {len(req.sensor_values)}",
                )
            ax = np.asarray(req.axiom_reference, dtype=float)
            if not np.all(np.isfinite(ax)):
                raise HTTPException(422, "axiom_reference must all be finite")
        ns = svc.auth.authorize_namespace(policy, req.namespace)
        try:
            return await svc.run_in_pool(
                svc.robotics_step,
                sensor_values=req.sensor_values,
                axiom_reference=req.axiom_reference,
                adapt_on_veto=req.adapt_on_veto,
                namespace=ns,
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    # ---------------------------------------------------- MCP SSE mount
    if mount_mcp:
        try:
            mcp_subapp = create_mcp_sse_subapp(svc)
            if mcp_subapp is not None:
                app.mount("/mcp", mcp_subapp)
        except Exception:
            pass

    return app
