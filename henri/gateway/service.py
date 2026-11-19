"""Unified production gateway service over HENRI-v6.

Single source of truth shared across all three client surfaces:
  1. OpenAI-compatible `/v1/chat/completions` (SSE + non-streaming)
  2. Typed REST `/v1/*` (Holarch + Cycle AD + DigitalTwin)
  3. Model Context Protocol (`/mcp` HTTP/SSE + stdio CLI)

Composes:
  * `SidecarService` (Holarch Zones A/B/C) + Cycle AD solvers
    (`RepoPatchEngine`, `ScienceSolver`, `ObjectLooInducer`)
  * `DigitalTwinAPI` (`D=65536` wave kernel) + persistent `CartridgeBank`s
  * `GatewayStore` (TimescaleDB / SQLite)
  * `Ed25519ReceiptSigner` (Ed25519 over canonical SHA-256 hash chain)
  * `ScopedAuthManager` (scoped API keys + per-tenant rate limiter)
  * `SandboxClient` (hardened sidecar IPC or local child fallback)
  * `CascadeRouter` (governed Arm0 -> Arm4 external LLM escalation)
  * `GatewayTelemetry` (Prometheus + optional OpenTelemetry)
"""

from __future__ import annotations

import asyncio
import json
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from henri.api import DigitalTwinAPI
from henri.cartridge_bank import CartridgeBank
from henri.code_verify import executor_status, solve_verified_code
from henri.gateway.auth import ScopedAuthManager, TenantPolicy
from henri.gateway.cascade import (
    CascadeRouter,
    OpenAICompatibleUpstream,
    UpstreamCallable,
)
from henri.gateway.config import GatewayConfig
from henri.gateway.sandbox_client import SandboxClient
from henri.gateway.signing import Ed25519ReceiptSigner
from henri.gateway.solvers import run_arc_solve, run_repo_patch, run_science_solve
from henri.gateway.storage import GatewayStore
from henri.gateway.telemetry import GatewayTelemetry
from henri.holarch.server import RWLock, SidecarService, chat_stream_chunks


def _detect_compute_device(requested: str) -> Tuple[str, str]:
    """Verify host compute capability; never fabricate GPU claims."""
    req = str(requested or "cpu").lower()
    if req == "cpu":
        return "cpu", "OBSERVED"
    if req == "cuda":
        try:
            import torch

            if torch.cuda.is_available():
                cap = torch.cuda.get_device_capability(0)
                return f"cuda:sm_{cap[0]}{cap[1]}", "OBSERVED"
        except Exception:
            pass
        return "cpu", "BLOCKED"
    return "cpu", "OBSERVED"


class UnifiedGatewayService:
    """Thread-safe, multi-tenant unified engine backing REST, OpenAI, and MCP."""

    def __init__(
        self,
        config: Optional[GatewayConfig] = None,
        *,
        holarch_service: Optional[SidecarService] = None,
        digital_twin: Optional[DigitalTwinAPI] = None,
        robotics_loop: Any = None,
        store: Optional[GatewayStore] = None,
        signer: Optional[Ed25519ReceiptSigner] = None,
        sandbox: Optional[SandboxClient] = None,
        upstream_callable: Optional[UpstreamCallable] = None,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self.config = config or GatewayConfig.from_env()
        self.config_digest = self.config.config_digest()
        self.compute_device, self.device_evidence = _detect_compute_device(
            self.config.holarch_device
        )

        # Persistence
        self.store = store or GatewayStore(
            backend=self.config.zone_c_backend,
            sqlite_path=self.config.sqlite_path,
            timescale_dsn=self.config.timescale_dsn,
        )

        # Signing (Ed25519 + JWKS)
        self.signer = signer or Ed25519ReceiptSigner.from_config(
            key_path=self.config.signing_key_path,
            kid=self.config.signing_key_id,
            rotated_public_keys_dir=self.config.rotated_public_keys_dir,
        )

        # Telemetry
        self.telemetry = GatewayTelemetry(
            service_name=self.config.service_name,
            enable_otel=self.config.enable_otel,
        )

        # Auth & Rate Limiting
        self.auth = ScopedAuthManager(
            self.store,
            require_auth=self.config.require_auth,
            legacy_api_keys=self.config.legacy_api_keys,
            default_rpm=self.config.default_rpm,
            default_burst=self.config.default_burst,
            default_llm_budget=self.config.default_llm_budget,
            ops_token=self.config.ops_token,
            clock=clock,
        )
        if self.config.bootstrap_keys_json:
            self.auth.bootstrap_from_json(self.config.bootstrap_keys_json)

        # Sandbox
        self.sandbox = sandbox or SandboxClient(
            socket_path=self.config.sandbox_socket,
            default_timeout_s=self.config.sandbox_timeout_s,
        )

        # Cascade (Arm0 -> Arm4)
        upstream = None
        if self.config.upstream_base_url and self.config.upstream_model:
            upstream = OpenAICompatibleUpstream(
                base_url=self.config.upstream_base_url,
                api_key=self.config.upstream_api_key,
                model=self.config.upstream_model,
                timeout_s=self.config.upstream_timeout_s,
            )
        self.cascade = CascadeRouter(
            store=self.store,
            telemetry=self.telemetry,
            upstream=upstream,
            upstream_callable=upstream_callable,
            max_retries=self.config.cascade_max_retries,
        )

        # Engines (lazy init unless injected)
        self._init_lock = threading.Lock()
        self._holarch: Optional[SidecarService] = holarch_service
        self._digital_twin: Optional[DigitalTwinAPI] = digital_twin
        self._robotics: Any = robotics_loop

        # Hot caches + per-namespace locks
        self._ns_locks_guard = threading.Lock()
        self._ns_locks: Dict[str, RWLock] = {}
        self._receipt_locks: Dict[str, threading.Lock] = {}
        self._cartridge_cache: Dict[str, CartridgeBank] = {}
        self._code_banks: Dict[str, Any] = {}
        self._zone_a_snapshots: Dict[str, Dict[str, Any]] = {}

        # Bounded compute pool so heavy numpy/torch ops never stall SSE/MCP
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, int(self.config.compute_workers)),
            thread_name_prefix="henri-compute",
        )

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False)
        self.store.close()

    # --------------------------------------------------------- lazy engines
    def holarch(self) -> SidecarService:
        if self._holarch is None:
            with self._init_lock:
                if self._holarch is None:
                    self._holarch = SidecarService(
                        backend=self.config.zone_c_backend,
                        memory_path=self.config.zone_c_sqlite_path,
                        checkpoint=self.config.holarch_checkpoint,
                        autoload=self.config.holarch_autoload,
                    )
        return self._holarch

    def digital_twin(self) -> DigitalTwinAPI:
        if self._digital_twin is None:
            with self._init_lock:
                if self._digital_twin is None:
                    self._digital_twin = DigitalTwinAPI(
                        dim=self.config.digital_twin_dim,
                        rank=self.config.digital_twin_rank,
                    )
        return self._digital_twin

    def robotics(self) -> Any:
        if self._robotics is None:
            with self._init_lock:
                if self._robotics is None:
                    from henri.robotics_loop import RoboticsLoop

                    self._robotics = RoboticsLoop()
        return self._robotics

    def _ns_lock(self, namespace: str) -> RWLock:
        with self._ns_locks_guard:
            lock = self._ns_locks.get(namespace)
            if lock is None:
                lock = RWLock()
                self._ns_locks[namespace] = lock
            return lock

    def _receipt_lock(self, namespace: str) -> threading.Lock:
        with self._ns_locks_guard:
            lock = self._receipt_locks.get(namespace)
            if lock is None:
                lock = threading.Lock()
                self._receipt_locks[namespace] = lock
            return lock

    async def run_in_pool(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        loop = asyncio.get_running_loop()
        self.telemetry.active_compute_jobs.inc()
        try:
            return await loop.run_in_executor(self._pool, lambda: fn(*args, **kwargs))
        finally:
            self.telemetry.active_compute_jobs.dec()

    # --------------------------------------------------------- receipts
    def issue_signed_receipt(
        self,
        *,
        endpoint: str,
        namespace: str,
        request_payload: Any,
        result_payload: Any,
        evidence_label: str = "OBSERVED",
        provenance: str = "wave_only",
    ) -> Dict[str, Any]:
        """Atomically append an Ed25519-signed receipt to the tenant's hash chain."""
        with self._receipt_lock(namespace):
            next_seq, parent_sha = self.store.get_receipt_head(namespace)
            receipt = self.signer.issue_signed(
                endpoint=endpoint,
                namespace=namespace,
                seq=next_seq,
                request=request_payload,
                result=result_payload,
                parent=parent_sha,
                config_digest=self.config_digest,
                evidence_label=evidence_label,
                provenance=provenance,
                compute_device=self.compute_device,
            )
            self.store.append_receipt(receipt)
            return receipt

    def _wrap_holarch(
        self,
        *,
        endpoint: str,
        namespace: str,
        request_payload: Dict[str, Any],
        raw_envelope: Dict[str, Any],
        provenance: str = "wave_only",
        evidence_label: str = "OBSERVED",
    ) -> Dict[str, Any]:
        """Attach Ed25519-signed gateway receipt while preserving legacy envelope fields."""
        result_obj = raw_envelope.get("result", raw_envelope)
        signed = self.issue_signed_receipt(
            endpoint=endpoint,
            namespace=namespace,
            request_payload=request_payload,
            result_payload=result_obj,
            evidence_label=evidence_label,
            provenance=provenance,
        )
        out = dict(raw_envelope)
        out["receipt"] = signed
        out["epistemic_receipt"] = signed
        out["provenance"] = provenance
        out["evidence_label"] = evidence_label
        return out

    # ---------------------------------------------------- Zone A isolation
    def _save_zone_a_fast_weights(self, namespace: str) -> None:
        """Persist per-namespace Zone A fast-weight snapshot for tenant isolation."""
        svc = self.holarch()
        model = getattr(svc.sidecar, "model", None)
        zone_a = getattr(model, "zone_a", None)
        if zone_a is None:
            return
        state: Dict[str, Any] = {}
        for attr in ("fast_weights", "fast_u", "fast_v", "rule_deltas"):
            val = getattr(zone_a, attr, None)
            if val is not None and hasattr(val, "detach"):
                state[attr] = val.detach().cpu().tolist()
        if state:
            self._zone_a_snapshots[namespace] = state
            self.store.put_tenant_state(namespace, "zone_a_fast_weights", "latest", state)

    def _get_or_hydrate_bank(self, tenant_id: str) -> Optional[CartridgeBank]:
        bank = self._cartridge_cache.get(tenant_id)
        if bank is not None:
            return bank
        loaded = self.store.load_cartridge_bank(tenant_id)
        if loaded is not None:
            self._cartridge_cache[tenant_id] = loaded
        return loaded

    # --------------------------------------------------------- health / jwks
    def healthz(self) -> Dict[str, Any]:
        db_ok, db_detail = self.store.ping()
        sbx = self.sandbox.ping()
        holarch_health: Dict[str, Any] = {"initialized": self._holarch is not None}
        if self._holarch is not None:
            holarch_health = self._holarch.health()
        return {
            "ok": bool(db_ok),
            "service": self.config.service_name,
            "compute_device": self.compute_device,
            "device_evidence": self.device_evidence,
            "config_digest": self.config_digest,
            "storage": {
                "ok": db_ok,
                "backend": self.store.backend,
                "reason": self.store.backend_reason,
                "detail": db_detail,
            },
            "sandbox": sbx,
            "signing": {
                "kid": self.signer.info.kid,
                "ephemeral": self.signer.info.ephemeral,
                "alg": "EdDSA",
            },
            "cascade": {
                "upstream_configured": self.cascade.configured,
                "max_retries": self.cascade.max_retries,
            },
            "auth_enforced": self.auth.is_auth_enforced(),
            "holarch": holarch_health,
            "evidence_label": "OBSERVED",
        }

    def jwks(self) -> Dict[str, Any]:
        return self.signer.jwks()

    def receipts(self, namespace: str, limit: int = 16) -> Dict[str, Any]:
        valid, count, head = self.store.verify_receipt_chain(namespace)
        items = self.store.list_receipts(namespace, limit=limit)
        return {
            "namespace": namespace,
            "count": count,
            "head": head,
            "chain_valid": valid,
            "signing_kid": self.signer.kid,
            "receipts": items,
            "evidence_label": "OBSERVED",
        }

    # ------------------------------------------------------- Holarch methods
    def veto(
        self, task: str, x: Any, rule_id: Optional[int], namespace: str
    ) -> Dict[str, Any]:
        with self._ns_lock(namespace).read_locked():
            raw = self.holarch().veto(task, x, rule_id, namespace)
        vetoed = bool((raw.get("result") or {}).get("veto_rate", 0.0) > 0.0)
        self.telemetry.veto_total.labels(endpoint="/v1/holarch/veto", vetoed=str(vetoed).lower()).inc()
        return self._wrap_holarch(
            endpoint="/v1/holarch/veto",
            namespace=namespace,
            request_payload={"task": task, "rule_id": rule_id, "namespace": namespace},
            raw_envelope=raw,
        )

    def ingest_code(
        self,
        root: str,
        package: str,
        namespace: str,
        mode: str,
        file_path: Optional[str],
        source_text: Optional[str],
    ) -> Dict[str, Any]:
        with self._ns_lock(namespace).write_locked():
            raw = self.holarch().ingest_code(root, package, namespace, mode, file_path, source_text)
        return self._wrap_holarch(
            endpoint="/v1/holarch/ingest_code",
            namespace=namespace,
            request_payload={
                "root": root,
                "package": package,
                "namespace": namespace,
                "mode": mode,
                "file_path": file_path,
            },
            raw_envelope=raw,
        )

    def plan(
        self,
        observation: Any,
        history_waves: Optional[List[Any]],
        transition: str,
        lookahead_depth: int,
        gamma_look: float,
        namespace: str,
        batch_rollout: bool,
        episode_id: Optional[str],
        online_repulsor: bool,
        reset_episode: bool,
        emit_swat_kv: bool,
    ) -> Dict[str, Any]:
        lock_ctx = (
            self._ns_lock(namespace).write_locked()
            if (online_repulsor or reset_episode)
            else self._ns_lock(namespace).read_locked()
        )
        # Scope episode_id by namespace to prevent cross-tenant episode collision
        scoped_ep = f"{namespace}:{episode_id}" if episode_id else None
        with lock_ctx:
            raw = self.holarch().plan(
                observation,
                history_waves,
                transition,
                lookahead_depth,
                gamma_look,
                namespace,
                batch_rollout,
                scoped_ep,
                online_repulsor,
                reset_episode,
                emit_swat_kv,
            )
        return self._wrap_holarch(
            endpoint="/v1/holarch/plan",
            namespace=namespace,
            request_payload={
                "transition": transition,
                "lookahead_depth": lookahead_depth,
                "episode_id": episode_id,
                "online_repulsor": online_repulsor,
                "namespace": namespace,
            },
            raw_envelope=raw,
        )

    def verify(
        self, claim: str, kind: str, top_k: int, namespace: str
    ) -> Dict[str, Any]:
        with self._ns_lock(namespace).read_locked():
            raw = self.holarch().verify(claim, kind, top_k, namespace)
        res = raw.get("result") or {}
        vetoed = bool(res.get("vetoed") or not res.get("supported", True))
        self.telemetry.veto_total.labels(endpoint="/v1/holarch/verify", vetoed=str(vetoed).lower()).inc()
        return self._wrap_holarch(
            endpoint="/v1/holarch/verify",
            namespace=namespace,
            request_payload={"claim": claim, "kind": kind, "top_k": top_k, "namespace": namespace},
            raw_envelope=raw,
        )

    def ingest_edges(self, edges: List[Dict[str, str]], namespace: str) -> Dict[str, Any]:
        with self._ns_lock(namespace).write_locked():
            raw = self.holarch().ingest_edges(edges, namespace)
        return self._wrap_holarch(
            endpoint="/v1/holarch/ingest_edges",
            namespace=namespace,
            request_payload={"edge_count": len(edges), "namespace": namespace},
            raw_envelope=raw,
        )

    def solve(
        self, source: str, operators: Any, namespace: str, mode: str
    ) -> Dict[str, Any]:
        with self._ns_lock(namespace).read_locked():
            raw = self.holarch().solve(source, operators, namespace, mode)
        res = raw.get("result") or {}
        self.telemetry.veto_total.labels(
            endpoint="/v1/holarch/solve",
            vetoed=str(bool(res.get("vetoed", False))).lower(),
        ).inc()
        return self._wrap_holarch(
            endpoint="/v1/holarch/solve",
            namespace=namespace,
            request_payload={"source": source, "operators": operators, "mode": mode, "namespace": namespace},
            raw_envelope=raw,
        )

    def synthesize(
        self, intent: str, kind: str, namespace: str, max_tokens: int
    ) -> Dict[str, Any]:
        with self._ns_lock(namespace).read_locked():
            raw = self.holarch().synthesize(intent, kind, namespace, max_tokens)
        return self._wrap_holarch(
            endpoint="/v1/holarch/synthesize",
            namespace=namespace,
            request_payload={"intent": intent, "kind": kind, "max_tokens": max_tokens, "namespace": namespace},
            raw_envelope=raw,
        )

    def telepathy(self, **kwargs: Any) -> Dict[str, Any]:
        namespace = str(kwargs["namespace"])
        mode = str(kwargs.get("mode", "ingest_chunks"))
        lock_ctx = (
            self._ns_lock(namespace).write_locked()
            if mode in {"ingest_chunks", "reset"} or kwargs.get("reset")
            else self._ns_lock(namespace).read_locked()
        )
        with lock_ctx:
            raw = self.holarch().telepathy(**kwargs)
        return self._wrap_holarch(
            endpoint="/v1/holarch/telepathy",
            namespace=namespace,
            request_payload={"mode": mode, "session_id": kwargs.get("session_id"), "namespace": namespace},
            raw_envelope=raw,
        )

    def turing(self, *args: Any, namespace: str = "default", **kwargs: Any) -> Dict[str, Any]:
        with self._ns_lock(namespace).read_locked():
            raw = self.holarch().turing(*args, **kwargs)
        return self._wrap_holarch(
            endpoint="/v1/holarch/turing",
            namespace=namespace,
            request_payload={"namespace": namespace, "mode": args[0] if args else kwargs.get("mode")},
            raw_envelope=raw,
        )

    def discover(
        self,
        intent: str,
        tests: str,
        request: Dict[str, Any],
        namespace: str,
        library: Optional[Dict[str, str]],
    ) -> Dict[str, Any]:
        with self._ns_lock(namespace).read_locked():
            raw = self.holarch().discover(intent, tests, request, namespace, library)
        return self._wrap_holarch(
            endpoint="/v1/holarch/discover",
            namespace=namespace,
            request_payload={"intent": intent, "namespace": namespace},
            raw_envelope=raw,
        )

    def playbook(
        self, name: str, steps: Optional[List[str]], merge: bool, namespace: str
    ) -> Dict[str, Any]:
        lock_ctx = (
            self._ns_lock(namespace).write_locked()
            if steps is not None
            else self._ns_lock(namespace).read_locked()
        )
        with lock_ctx:
            raw = self.holarch().playbook(name, steps, merge, namespace)
        return self._wrap_holarch(
            endpoint="/v1/holarch/playbook",
            namespace=namespace,
            request_payload={"name": name, "has_steps": steps is not None, "merge": merge, "namespace": namespace},
            raw_envelope=raw,
        )

    def adapt_rule(
        self,
        task: str,
        x: Any,
        y: List[int],
        rule_id: Optional[int],
        lr: float,
        namespace: str,
    ) -> Dict[str, Any]:
        with self._ns_lock(namespace).write_locked():
            raw = self.holarch().adapt_rule(task, x, y, rule_id, lr, namespace)
            self._save_zone_a_fast_weights(namespace)
        return self._wrap_holarch(
            endpoint="/v1/holarch/adapt_rule",
            namespace=namespace,
            request_payload={"task": task, "rule_id": rule_id, "lr": lr, "namespace": namespace},
            raw_envelope=raw,
        )

    def composite_call(self, **kwargs: Any) -> Dict[str, Any]:
        namespace = str(kwargs.get("namespace", "default"))
        mode = str(kwargs.get("mode", "query"))
        lock_ctx = (
            self._ns_lock(namespace).write_locked()
            if mode == "ingest"
            else self._ns_lock(namespace).read_locked()
        )
        with lock_ctx:
            raw = self.holarch().composite_call(**kwargs)
        return self._wrap_holarch(
            endpoint="/v1/holarch/composite",
            namespace=namespace,
            request_payload={"mode": mode, "namespace": namespace},
            raw_envelope=raw,
        )

    def agent_step(
        self, action: str, payload: Dict[str, Any], namespace: str
    ) -> Dict[str, Any]:
        with self._ns_lock(namespace).read_locked():
            raw = self.holarch().agent_step(action, payload, namespace)
        return self._wrap_holarch(
            endpoint="/v1/holarch/agent_step",
            namespace=namespace,
            request_payload={"action": action, "namespace": namespace},
            raw_envelope=raw,
        )

    # ------------------------------------------ OpenAI chat + Arm0->Arm4
    def chat_completion(
        self,
        *,
        policy: TenantPolicy,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        metadata: Optional[Dict[str, Any]],
        namespace: str,
    ) -> Dict[str, Any]:
        meta = dict(metadata or {})
        meta["namespace"] = namespace
        cascade_policy = str(meta.get("cascade_policy") or "cascade").lower()

        with self._ns_lock(namespace).read_locked():
            arm0 = self.holarch().chat_completion(model, messages, tools, meta)

        henri_meta = dict(arm0.get("henri") or {})
        arm0_abstained = bool(henri_meta.get("abstained", False))
        arm0_status = str(henri_meta.get("status", ""))
        trap_detected = arm0_status in {"refused", "contradicted", "trap"}

        def _verify_candidate(candidate_text: str) -> Tuple[bool, Any, str]:
            text = candidate_text.strip()
            if not text:
                return False, None, "empty_candidate"
            verify_env = self.holarch().verify(text, "text", 4, namespace)
            vres = verify_env.get("result") or {}
            verdict = str(vres.get("verdict") or "").lower()
            supported = bool(vres.get("supported", False)) or verdict == "supported"
            if supported and not bool(vres.get("vetoed", False)):
                updated = dict(arm0)
                choices = [dict(c) for c in (updated.get("choices") or [])]
                if choices:
                    msg = dict(choices[0].get("message") or {})
                    msg["content"] = text
                    choices[0]["message"] = msg
                    updated["choices"] = choices
                updated_henri = dict(henri_meta)
                updated_henri["abstained"] = False
                updated_henri["status"] = "verified_arm4"
                updated_henri["verification"] = vres
                updated["henri"] = updated_henri
                return True, updated, "ontology_verified"
            return False, None, f"ontology_veto:{verdict or 'unsupported'}"

        prompt_msgs = [
            {"role": str(m.get("role", "user")), "content": str(m.get("content", ""))}
            for m in messages
        ]
        outcome = self.cascade.resolve(
            endpoint="/v1/chat/completions",
            policy=policy,
            cascade_policy=cascade_policy,
            arm0_output=arm0,
            arm0_abstained=arm0_abstained,
            trap_detected=trap_detected,
            prompt_messages=prompt_msgs,
            verify_fn=_verify_candidate,
        )

        final_payload = dict(outcome.output)
        final_henri = dict(final_payload.get("henri") or {})
        final_henri["cascade"] = {
            "verdict": outcome.verdict,
            "provenance": outcome.provenance,
            "llm_calls": outcome.llm_calls,
            **outcome.cascade_detail,
        }
        signed = self.issue_signed_receipt(
            endpoint="/v1/chat/completions",
            namespace=namespace,
            request_payload={"model": model, "namespace": namespace, "cascade_policy": cascade_policy},
            result_payload={
                "status": final_henri.get("status"),
                "abstained": outcome.abstained,
                "verdict": outcome.verdict,
                "llm_calls": outcome.llm_calls,
            },
            evidence_label="OBSERVED",
            provenance=outcome.provenance,
        )
        final_henri["epistemic_receipt"] = signed
        final_payload["henri"] = final_henri
        final_payload["epistemic_receipt"] = signed
        return final_payload

    @staticmethod
    def stream_chat_lines(payload: Dict[str, Any]) -> List[str]:
        return chat_stream_chunks(payload)

    # ---------------------------------------------------- Cycle AD endpoints
    def repo_patch(
        self,
        *,
        policy: TenantPolicy,
        name: str,
        issue: str,
        files: Dict[str, str],
        public: Dict[str, str],
        test_path: str,
        rollouts: int,
        timeout_s: float,
        namespace: str,
        cascade_policy: str = "cascade",
    ) -> Dict[str, Any]:
        if not policy.allow_sandbox_exec:
            raise PermissionError("tenant policy disables sandbox execution")
        mem = self.holarch().sidecar.model.memory
        with self._ns_lock(namespace).write_locked():
            arm0 = run_repo_patch(
                name=name,
                issue=issue,
                files=files,
                public=public,
                test_path=test_path,
                rollouts=rollouts,
                timeout_s=timeout_s,
                memory=mem,
                namespace=namespace,
            )

        def _verify_patch(candidate_text: str) -> Tuple[bool, Any, str]:
            # Candidate must be JSON object mapping file_path -> patched source
            try:
                patch_map = json.loads(candidate_text)
            except Exception as exc:
                return False, None, f"invalid_json_patch:{exc}"
            if not isinstance(patch_map, dict) or not patch_map:
                return False, None, "patch_must_be_nonempty_dict"
            merged = dict(files)
            for p, src in patch_map.items():
                if str(p) not in files:
                    return False, None, f"unknown_file:{p}"
                merged[str(p)] = str(src)
            # Re-verify via sandboxed RepoPatchEngine test run
            recheck = run_repo_patch(
                name=f"{name}_arm4",
                issue=issue,
                files=merged,
                public=public,
                test_path=test_path,
                rollouts=1,
                timeout_s=timeout_s,
                memory=None,
                namespace=namespace,
            )
            if (recheck.get("initial_tests") or {}).get("passed"):
                return (
                    True,
                    {
                        **arm0,
                        "status": "PATCHED",
                        "emitted": True,
                        "abstained": False,
                        "patch": patch_map,
                        "verified_by": "sandbox_tests",
                    },
                    "sandbox_tests_passed",
                )
            return False, None, "sandbox_tests_failed"

        outcome = self.cascade.resolve(
            endpoint="/v1/holarch/repo_patch",
            policy=policy,
            cascade_policy=cascade_policy,
            arm0_output=arm0,
            arm0_abstained=bool(arm0["abstained"]),
            trap_detected=bool(arm0["trap_detected"]),
            prompt_messages=[
                {
                    "role": "system",
                    "content": "Return ONLY a JSON object mapping relative file paths to patched Python source.",
                },
                {
                    "role": "user",
                    "content": json.dumps({"issue": issue, "files": files, "public_tests": public}),
                },
            ],
            verify_fn=_verify_patch,
        )
        result_body = dict(outcome.output)
        result_body["llm_calls"] = outcome.llm_calls
        result_body["cascade"] = {"verdict": outcome.verdict, **outcome.cascade_detail}
        signed = self.issue_signed_receipt(
            endpoint="/v1/holarch/repo_patch",
            namespace=namespace,
            request_payload={"name": name, "files": sorted(files), "test_path": test_path, "namespace": namespace},
            result_payload={
                "status": result_body.get("status"),
                "abstained": outcome.abstained,
                "trap_detected": bool(arm0["trap_detected"]),
                "llm_calls": outcome.llm_calls,
            },
            evidence_label="OBSERVED",
            provenance=outcome.provenance,
        )
        return {
            "ok": not outcome.abstained,
            "endpoint": "/v1/holarch/repo_patch",
            "result": result_body,
            "provenance": outcome.provenance,
            "evidence_label": "OBSERVED",
            "epistemic_receipt": signed,
        }

    def science_solve(
        self,
        *,
        policy: TenantPolicy,
        problem: Dict[str, Any],
        namespace: str,
        cascade_policy: str = "wave_only",
    ) -> Dict[str, Any]:
        arm0 = run_science_solve(problem)
        outcome = self.cascade.resolve(
            endpoint="/v1/holarch/science",
            policy=policy,
            cascade_policy=cascade_policy,
            arm0_output=arm0,
            arm0_abstained=bool(arm0["abstained"]),
            trap_detected=bool(arm0["trap_detected"]),
            prompt_messages=[
                {"role": "system", "content": "Provide a JSON science problem specification."},
                {"role": "user", "content": json.dumps(problem, default=str)},
            ],
            verify_fn=lambda _text: (False, None, "science_requires_dimensional_proof"),
        )
        result_body = dict(outcome.output)
        result_body["cascade"] = {"verdict": outcome.verdict, **outcome.cascade_detail}
        signed = self.issue_signed_receipt(
            endpoint="/v1/holarch/science",
            namespace=namespace,
            request_payload={"id": problem.get("id", ""), "domain": problem.get("domain", ""), "namespace": namespace},
            result_payload={
                "status": result_body.get("status"),
                "abstained": outcome.abstained,
                "trap_detected": bool(arm0["trap_detected"]),
                "reason": result_body.get("reason", ""),
            },
            evidence_label="OBSERVED",
            provenance=outcome.provenance,
        )
        return {
            "ok": not outcome.abstained,
            "endpoint": "/v1/holarch/science",
            "result": result_body,
            "provenance": outcome.provenance,
            "evidence_label": "OBSERVED",
            "epistemic_receipt": signed,
        }

    def arc_solve(
        self,
        *,
        train: Sequence[Dict[str, Any]],
        test: Sequence[Dict[str, Any]],
        node_budget: int,
        depth: int,
        time_budget: float,
        namespace: str,
    ) -> Dict[str, Any]:
        result_body = run_arc_solve(
            train=train,
            test=test,
            node_budget=node_budget,
            depth=depth,
            time_budget=time_budget,
        )
        signed = self.issue_signed_receipt(
            endpoint="/v1/holarch/arc",
            namespace=namespace,
            request_payload={"train_pairs": len(train), "test_items": len(test), "depth": depth, "namespace": namespace},
            result_payload={
                "status": result_body.get("status"),
                "emitted": bool(result_body.get("emitted", False)),
                "chain": result_body.get("chain", []),
                "loo_passed": bool(result_body.get("loo_passed", False)),
            },
            evidence_label="OBSERVED",
            provenance="wave_only",
        )
        return {
            "ok": not bool(result_body.get("abstained", True)),
            "endpoint": "/v1/holarch/arc",
            "result": result_body,
            "provenance": "wave_only",
            "evidence_label": "OBSERVED",
            "epistemic_receipt": signed,
        }

    # ---------------------------------------------- DigitalTwin D=65536
    def health_seal(self, namespace: str) -> Dict[str, Any]:
        raw = self.digital_twin().verify_seal()
        body = raw.to_dict() if hasattr(raw, "to_dict") else dict(raw)
        signed = self.issue_signed_receipt(
            endpoint="/v1/health/seal",
            namespace=namespace,
            request_payload={"namespace": namespace},
            result_payload={"seal_status": body.get("seal_status", body.get("status"))},
        )
        body["epistemic_receipt"] = signed
        return body

    def health_executor(self, namespace: str) -> Dict[str, Any]:
        base = executor_status()
        sbx = self.sandbox.ping()
        body = {**base, "sandbox": sbx, "evidence_label": "OBSERVED"}
        signed = self.issue_signed_receipt(
            endpoint="/v1/health/executor",
            namespace=namespace,
            request_payload={"namespace": namespace},
            result_payload={"sandbox_mode": sbx.get("mode"), "sandbox_ok": sbx.get("ok")},
        )
        body["epistemic_receipt"] = signed
        return body

    def reason_rank(
        self,
        *,
        question: str,
        options: List[str],
        strip: bool,
        signing_key: Optional[str],
        channel_weight: float,
        namespace: str,
    ) -> Dict[str, Any]:
        api = self.digital_twin()
        with self._ns_lock(namespace).read_locked():
            bank = self._get_or_hydrate_bank(namespace)
            # Attach tenant-scoped bank during this call safely under lock
            prev_bank = api.bank
            api.bank = bank
            try:
                key_bytes = signing_key.encode("utf-8") if signing_key else None
                rec = api.rank_options(
                    question,
                    options,
                    strip=strip,
                    signing_key=key_bytes,
                    channel_weight=channel_weight,
                )
            finally:
                api.bank = prev_bank
        body = rec.to_dict() if hasattr(rec, "to_dict") else dict(rec)
        signed = self.issue_signed_receipt(
            endpoint="/v1/reason/rank",
            namespace=namespace,
            request_payload={"n_options": len(options), "strip": strip, "namespace": namespace},
            result_payload={
                "selected_index": body.get("selected_index"),
                "margin": body.get("margin"),
                "sagnac_delta": body.get("sagnac_delta"),
                "cartridge_hit": body.get("cartridge_hit"),
            },
        )
        body["epistemic_receipt"] = signed
        return body

    def cartridges_write(
        self,
        *,
        tenant_id: str,
        keys: List[str],
        values: List[str],
        subject: str,
    ) -> Dict[str, Any]:
        api = self.digital_twin()
        with self._ns_lock(tenant_id).write_locked():
            k_arr = np.stack([api.backbone.encode(k).psi for k in keys], axis=0)
            v_arr = np.stack([api.backbone.encode(v).psi for v in values], axis=0)
            wave_dim = int(k_arr.shape[1])

            bank = self._get_or_hydrate_bank(tenant_id)
            if bank is None:
                bank = CartridgeBank(dim=wave_dim, rank=api.rank)
                self._cartridge_cache[tenant_id] = bank
            elif bank.dim != wave_dim:
                raise RuntimeError(f"bank dim {bank.dim} != wave dim {wave_dim}")

            cart = bank.add_cartridge(k_arr, v_arr, subject=subject)
            bank.finalize()
            cart_sha = self.store.save_cartridge(tenant_id, cart)

        body = {
            "endpoint": "cartridges_write",
            "tenant_id": tenant_id,
            "cartridge_index": int(cart.index),
            "rows": int(cart.rows),
            "cartridges": len(bank.cartridges),
            "subject": subject,
            "persisted_sha256": cart_sha,
            "storage_backend": self.store.backend,
            "bank": bank.stats(),
        }
        body["epistemic_receipt"] = self.issue_signed_receipt(
            endpoint="/v1/cartridges/write",
            namespace=tenant_id,
            request_payload={"tenant_id": tenant_id, "rows": len(keys), "subject": subject},
            result_payload={"cartridge_index": int(cart.index), "persisted_sha256": cart_sha},
        )
        return body

    def cartridges_mount(self, *, tenant_id: str) -> Optional[Dict[str, Any]]:
        with self._ns_lock(tenant_id).read_locked():
            bank = self._get_or_hydrate_bank(tenant_id)
            if bank is None:
                return None
            body = {
                "endpoint": "cartridges_mount",
                "tenant_id": tenant_id,
                "mounted": True,
                "cartridges": len(bank.cartridges),
                "rows": int(sum(c.rows for c in bank.cartridges)),
                "storage_backend": self.store.backend,
                "bank": bank.stats(),
            }
        body["epistemic_receipt"] = self.issue_signed_receipt(
            endpoint="/v1/cartridges/mount",
            namespace=tenant_id,
            request_payload={"tenant_id": tenant_id},
            result_payload={"mounted": True, "cartridges": body["cartridges"]},
        )
        return body

    def code_solve(
        self,
        *,
        policy: TenantPolicy,
        spec: str,
        candidates: List[str],
        tests: List[str],
        top_k: int,
        timeout: Optional[int],
        synthesize_from_bank: bool,
        k_templates: int,
        max_variants: int,
        tenant_id: str,
    ) -> Dict[str, Any]:
        if not policy.allow_sandbox_exec:
            raise PermissionError("tenant policy disables sandbox execution")
        api = self.digital_twin()
        bank = self._code_banks.get(tenant_id)
        if synthesize_from_bank and bank is None:
            body = {
                "endpoint": "code_solve",
                "verdict": "NO_CANDIDATES",
                "n_candidates": 0,
                "n_tests": len(tests),
                "executed": [],
                "synthesize_from_bank": True,
                "note": "no code-template bank is mounted, so nothing can be synthesized. Ingest templates first.",
            }
        elif self.sandbox.socket_path and candidates and not synthesize_from_bank:
            # Route direct candidate verification through the hardened sandbox sidecar
            body = self._code_solve_via_sidecar(
                api, spec, candidates, tests, top_k=top_k, timeout=timeout
            )
        else:
            body = solve_verified_code(
                api.backbone,
                spec,
                candidates,
                tests=tests,
                top_k=top_k,
                timeout=timeout,
                bank=bank,
                synthesize_from_bank=synthesize_from_bank,
                k_templates=k_templates,
                max_variants=max_variants,
            )
        body["epistemic_receipt"] = self.issue_signed_receipt(
            endpoint="/v1/code/solve",
            namespace=tenant_id,
            request_payload={"n_candidates": len(candidates), "n_tests": len(tests), "tenant_id": tenant_id},
            result_payload={"verdict": body.get("verdict"), "sandbox_mode": self.sandbox.mode},
        )
        return body

    def _code_solve_via_sidecar(
        self,
        api: DigitalTwinAPI,
        spec: str,
        candidates: List[str],
        tests: List[str],
        *,
        top_k: int,
        timeout: Optional[int],
    ) -> Dict[str, Any]:
        q = api.backbone.encode(spec).psi
        scored = []
        for idx, src in enumerate(candidates):
            sim = float(np.real(np.vdot(q, api.backbone.encode(src).psi)))
            scored.append((sim, idx, src))
        scored.sort(key=lambda row: (-row[0], row[1]))
        if not tests:
            return {
                "endpoint": "code_solve",
                "verdict": "NO_TESTS_UNVERIFIABLE",
                "n_candidates": len(candidates),
                "n_tests": 0,
                "executed": [],
                "sandbox_mode": self.sandbox.mode,
            }
        executed = []
        winner = None
        for sim, idx, src in scored[: max(1, int(top_k))]:
            run = self.sandbox.execute(
                src,
                tests=tests,
                profile="pure_algorithmic",
                timeout_s=float(timeout or self.config.sandbox_timeout_s),
            )
            self.telemetry.sandbox_exec_total.labels(
                mode=self.sandbox.mode, outcome="ok" if run["ok"] else "fail"
            ).inc()
            row = {"candidate_index": idx, "score": round(sim, 9), **run}
            executed.append(row)
            if run["ok"] and winner is None:
                winner = row
                break
        return {
            "endpoint": "code_solve",
            "verdict": "VERIFIED" if winner is not None else "FAILED",
            "selected_index": winner["candidate_index"] if winner is not None else None,
            "n_candidates": len(candidates),
            "n_tests": len(tests),
            "executed": executed,
            "sandbox_mode": self.sandbox.mode,
        }

    def code_bank_ingest(self, *, tenant_id: str, limit: int) -> Dict[str, Any]:
        from henri.code_synth import CodeTemplateBank, Template

        api = self.digital_twin()
        path = Path(__file__).resolve().parents[2] / "data" / "official_benchmarks" / "mbpp.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"corpus missing: {path}")
        rows = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
                if len(rows) >= int(limit):
                    break
        if not rows:
            raise ValueError("corpus has no rows")
        dim = int(api.backbone.encode(rows[0]["text"]).psi.shape[0])
        bank = CodeTemplateBank(dim=dim, rank=api.rank)
        for r in rows:
            text = str(r.get("text", "")).strip()
            code = str(r.get("code", "")).strip()
            if text and code:
                bank.add(api.backbone, Template(docstring=text, source=code, task_id=str(r.get("task_id", ""))))
        if not bank.templates:
            raise ValueError("corpus rows had no usable text/code pairs")
        bank.finalize()
        self._code_banks[tenant_id] = bank
        body = {
            "endpoint": "code_bank_ingest",
            "tenant_id": tenant_id,
            "corpus": path.name,
            "templates": bank.rows,
            "bank": bank.to_dict(),
        }
        body["epistemic_receipt"] = self.issue_signed_receipt(
            endpoint="/v1/code/bank/ingest",
            namespace=tenant_id,
            request_payload={"tenant_id": tenant_id, "limit": limit},
            result_payload={"templates": bank.rows},
        )
        return body

    def robotics_step(
        self,
        *,
        sensor_values: List[float],
        axiom_reference: Optional[List[float]],
        adapt_on_veto: bool,
        namespace: str,
    ) -> Dict[str, Any]:
        loop = self.robotics()
        with self._ns_lock(namespace).write_locked():
            body = dict(
                loop.step(
                    sensor_values,
                    axiom_reference=axiom_reference,
                    adapt_on_veto=adapt_on_veto,
                )
            )
        body["epistemic_receipt"] = self.issue_signed_receipt(
            endpoint="/v1/robotics/step",
            namespace=namespace,
            request_payload={"n_sensors": len(sensor_values), "has_axiom": axiom_reference is not None, "namespace": namespace},
            result_payload={"vetoed": body.get("vetoed"), "sagnac_delta": body.get("sagnac_delta")},
        )
        return body
