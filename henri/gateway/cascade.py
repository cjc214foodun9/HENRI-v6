"""Governed wave-first cascade router (Arm0 -> Arm4).

Evidence label: OBSERVED for every decision, escalation, verification, and
SpendGuard accounting event recorded here.

Governance invariants:
  1. Arm0 (HENRI-v6 deterministic wave/symbolic/Cycle AD core) ALWAYS executes
     first at zero LLM cost.
  2. Detected traps NEVER escalate to an external LLM.
  3. Eligible non-trap Arm0 abstentions escalate ONLY when:
       - tenant policy enables `allow_cascade`,
       - request policy is `"cascade"` (not `"wave_only"`),
       - an OpenAI-compatible upstream is configured (or injected in tests),
       - `GatewayStore.reserve_llm_call` atomically grants budget.
  4. Every Arm4 candidate MUST pass the native HENRI verifier before return.
  5. At most 1 governed retry with structured verifier feedback is allowed.
  6. If verification still fails (or budget is exhausted), the router returns
     strict fail-closed `ABSTAIN` — NEVER leaking unverified LLM text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx

from henri.gateway.auth import TenantPolicy
from henri.gateway.storage import GatewayStore
from henri.gateway.telemetry import GatewayTelemetry


@dataclass
class UpstreamCompletion:
    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw_id: str = ""


UpstreamCallable = Callable[[List[Dict[str, str]], str], UpstreamCompletion]
VerifierCallable = Callable[[str], Tuple[bool, Any, str]]


class OpenAICompatibleUpstream:
    """Provider-agnostic OpenAI-compatible `/chat/completions` client."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: Optional[str],
        model: str,
        timeout_s: float = 30.0,
    ) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.api_key = api_key or ""
        self.model = str(model)
        self.timeout_s = float(timeout_s)

    def complete(self, messages: List[Dict[str, str]], model: Optional[str] = None) -> UpstreamCompletion:
        url = (
            self.base_url
            if self.base_url.endswith("/chat/completions")
            else f"{self.base_url}/chat/completions"
        )
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": model or self.model,
            "messages": messages,
            "temperature": 0.0,
        }
        with httpx.Client(timeout=self.timeout_s) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        content = str(msg.get("content") or "")
        usage = data.get("usage") or {}
        return UpstreamCompletion(
            content=content,
            model=str(data.get("model") or model or self.model),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            raw_id=str(data.get("id") or ""),
        )


@dataclass
class CascadeOutcome:
    verified: bool
    abstained: bool
    provenance: str  # wave_only | llm_escalated | cached
    output: Any
    llm_calls: int
    verdict: str
    cascade_detail: Dict[str, Any]


class CascadeRouter:
    """Executes Arm0 -> optional governed Arm4 escalation with strict fail-closed veto."""

    def __init__(
        self,
        *,
        store: GatewayStore,
        telemetry: GatewayTelemetry,
        upstream: Optional[OpenAICompatibleUpstream] = None,
        upstream_callable: Optional[UpstreamCallable] = None,
        max_retries: int = 1,
    ) -> None:
        self.store = store
        self.telemetry = telemetry
        self.upstream = upstream
        self._callable = upstream_callable
        self.max_retries = min(1, max(0, int(max_retries)))

    @property
    def configured(self) -> bool:
        return self._callable is not None or self.upstream is not None

    def _invoke_upstream(self, messages: List[Dict[str, str]]) -> UpstreamCompletion:
        if self._callable is not None:
            model = self.upstream.model if self.upstream else "injected-upstream"
            return self._callable(messages, model)
        assert self.upstream is not None
        return self.upstream.complete(messages)

    def resolve(
        self,
        *,
        endpoint: str,
        policy: TenantPolicy,
        cascade_policy: str,
        arm0_output: Any,
        arm0_abstained: bool,
        trap_detected: bool,
        prompt_messages: List[Dict[str, str]],
        verify_fn: VerifierCallable,
    ) -> CascadeOutcome:
        """Apply Arm0->Arm4 decision policy around an already-computed Arm0 result."""
        # 1. Arm0 verified -> zero LLM cost
        if not arm0_abstained:
            self.telemetry.cascade_total.labels(endpoint=endpoint, outcome="arm0_verified").inc()
            return CascadeOutcome(
                verified=True,
                abstained=False,
                provenance="wave_only",
                output=arm0_output,
                llm_calls=0,
                verdict="arm0_verified",
                cascade_detail={"escalated": False, "reason": "arm0_verified"},
            )

        # 2. Trap detected -> never escalate
        if trap_detected:
            self.telemetry.cascade_total.labels(endpoint=endpoint, outcome="trap_refused").inc()
            return CascadeOutcome(
                verified=False,
                abstained=True,
                provenance="wave_only",
                output=arm0_output,
                llm_calls=0,
                verdict="trap_refused",
                cascade_detail={"escalated": False, "reason": "trap_never_escalates"},
            )

        # 3. Escalation disallowed by caller, tenant policy, or unconfigured upstream
        if cascade_policy == "wave_only" or not policy.allow_cascade or not self.configured:
            reason = (
                "caller_wave_only"
                if cascade_policy == "wave_only"
                else ("tenant_cascade_disabled" if not policy.allow_cascade else "upstream_not_configured")
            )
            self.telemetry.cascade_total.labels(endpoint=endpoint, outcome=reason).inc()
            return CascadeOutcome(
                verified=False,
                abstained=True,
                provenance="wave_only",
                output=arm0_output,
                llm_calls=0,
                verdict=f"abstained_{reason}",
                cascade_detail={"escalated": False, "reason": reason},
            )

        # 4. Arm4 escalation under atomic SpendGuard + verifier gate (max 1 retry)
        attempts = 1 + self.max_retries
        messages = list(prompt_messages)
        llm_calls = 0
        last_feedback = ""
        upstream_model = ""

        for attempt in range(1, attempts + 1):
            granted, used, limit = self.store.reserve_llm_call(
                policy.tenant_id, policy.llm_call_budget
            )
            if not granted:
                self.telemetry.spend_calls_total.labels(
                    tenant_id=policy.tenant_id, outcome="exhausted"
                ).inc()
                self.telemetry.cascade_total.labels(
                    endpoint=endpoint, outcome="budget_exhausted"
                ).inc()
                return CascadeOutcome(
                    verified=False,
                    abstained=True,
                    provenance="llm_escalated" if llm_calls > 0 else "wave_only",
                    output=arm0_output,
                    llm_calls=llm_calls,
                    verdict="abstained_budget_exhausted",
                    cascade_detail={
                        "escalated": llm_calls > 0,
                        "reason": "spend_guard_budget_exhausted",
                        "calls_used": used,
                        "budget_limit": limit,
                        "last_feedback": last_feedback,
                    },
                )

            self.telemetry.spend_calls_total.labels(
                tenant_id=policy.tenant_id, outcome="granted"
            ).inc()
            llm_calls += 1

            try:
                comp = self._invoke_upstream(messages)
            except Exception as exc:
                self.telemetry.cascade_total.labels(
                    endpoint=endpoint, outcome="upstream_error"
                ).inc()
                return CascadeOutcome(
                    verified=False,
                    abstained=True,
                    provenance="llm_escalated",
                    output=arm0_output,
                    llm_calls=llm_calls,
                    verdict="abstained_upstream_error",
                    cascade_detail={
                        "escalated": True,
                        "reason": f"upstream_error: {type(exc).__name__}: {exc}",
                        "attempts": attempt,
                    },
                )

            upstream_model = comp.model
            self.store.record_llm_tokens(
                policy.tenant_id, comp.prompt_tokens, comp.completion_tokens
            )
            ok, verified_payload, feedback = verify_fn(comp.content)
            if ok:
                self.telemetry.cascade_total.labels(
                    endpoint=endpoint, outcome="arm4_verified"
                ).inc()
                return CascadeOutcome(
                    verified=True,
                    abstained=False,
                    provenance="llm_escalated",
                    output=verified_payload,
                    llm_calls=llm_calls,
                    verdict="arm4_verified",
                    cascade_detail={
                        "escalated": True,
                        "attempts": attempt,
                        "upstream_model": upstream_model,
                        "verifier_feedback": feedback,
                    },
                )

            last_feedback = str(feedback)
            if attempt < attempts:
                messages = messages + [
                    {"role": "assistant", "content": comp.content},
                    {
                        "role": "user",
                        "content": (
                            "Your previous candidate failed HENRI fail-closed verification: "
                            f"{last_feedback}. Provide a corrected candidate that satisfies "
                            "the verifier."
                        ),
                    },
                ]

        # Strict fail-closed: NEVER return unverified LLM text.
        self.telemetry.cascade_total.labels(
            endpoint=endpoint, outcome="arm4_vetoed_abstain"
        ).inc()
        return CascadeOutcome(
            verified=False,
            abstained=True,
            provenance="llm_escalated",
            output=arm0_output,
            llm_calls=llm_calls,
            verdict="vetoed_after_retry",
            cascade_detail={
                "escalated": True,
                "attempts": llm_calls,
                "upstream_model": upstream_model,
                "verifier_feedback": last_feedback,
                "unverified_llm_suppressed": True,
            },
        )
