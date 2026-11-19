"""Tests for the governed Arm0 -> Arm4 wave-first cascade, retry cap, and SpendGuard."""

from __future__ import annotations

import json

from henri.gateway import (
    CascadeRouter,
    GatewayStore,
    TenantPolicy,
    UpstreamCompletion,
)
from henri.gateway.telemetry import GatewayTelemetry


def _policy(budget: int = 10, allow_cascade: bool = True) -> TenantPolicy:
    return TenantPolicy(
        key_id="k1",
        tenant_id="acme",
        allowed_namespaces=("acme",),
        default_namespace="acme",
        llm_call_budget=budget,
        allow_cascade=allow_cascade,
    )


def test_arm0_verified_makes_zero_upstream_calls(tmp_path):
    store = GatewayStore(backend="sqlite", sqlite_path=str(tmp_path / "gw.sqlite3"))
    tel = GatewayTelemetry()
    calls = []

    def fake_llm(messages, model):
        calls.append(messages)
        return UpstreamCompletion(content="should not run", model=model)

    router = CascadeRouter(store=store, telemetry=tel, upstream_callable=fake_llm)
    out = router.resolve(
        endpoint="/v1/holarch/repo_patch",
        policy=_policy(),
        cascade_policy="cascade",
        arm0_output={"status": "PATCHED", "answer": "ok"},
        arm0_abstained=False,
        trap_detected=False,
        prompt_messages=[{"role": "user", "content": "fix bug"}],
        verify_fn=lambda txt: (True, txt, "ok"),
    )
    assert out.verified is True
    assert out.abstained is False
    assert out.provenance == "wave_only"
    assert out.llm_calls == 0
    assert len(calls) == 0
    store.close()


def test_detected_trap_never_escalates_to_llm(tmp_path):
    store = GatewayStore(backend="sqlite", sqlite_path=str(tmp_path / "gw.sqlite3"))
    tel = GatewayTelemetry()
    calls = []

    def fake_llm(messages, model):
        calls.append(messages)
        return UpstreamCompletion(content="bad", model=model)

    router = CascadeRouter(store=store, telemetry=tel, upstream_callable=fake_llm)
    out = router.resolve(
        endpoint="/v1/holarch/science",
        policy=_policy(),
        cascade_policy="cascade",
        arm0_output={"status": "ABSTAIN", "reason": "nonphysical_negative_temperature"},
        arm0_abstained=True,
        trap_detected=True,
        prompt_messages=[{"role": "user", "content": "T = -50 K"}],
        verify_fn=lambda txt: (True, txt, "ok"),
    )
    assert out.abstained is True
    assert out.verdict == "trap_refused"
    assert out.provenance == "wave_only"
    assert out.llm_calls == 0
    assert len(calls) == 0
    store.close()


def test_arm4_governed_retry_passes_on_second_attempt(tmp_path):
    store = GatewayStore(backend="sqlite", sqlite_path=str(tmp_path / "gw.sqlite3"))
    tel = GatewayTelemetry()
    responses = [
        UpstreamCompletion(content="wrong_attempt", model="mock-llm", prompt_tokens=10, completion_tokens=5),
        UpstreamCompletion(content="verified_attempt", model="mock-llm", prompt_tokens=20, completion_tokens=8),
    ]
    seen_prompts = []

    def fake_llm(messages, model):
        seen_prompts.append(list(messages))
        return responses.pop(0)

    def verifier(text: str):
        if text == "verified_attempt":
            return True, {"answer": text}, "verified"
        return False, None, "failed_unit_tests"

    router = CascadeRouter(store=store, telemetry=tel, upstream_callable=fake_llm, max_retries=1)
    out = router.resolve(
        endpoint="/v1/chat/completions",
        policy=_policy(budget=5),
        cascade_policy="cascade",
        arm0_output={"status": "ABSTAIN"},
        arm0_abstained=True,
        trap_detected=False,
        prompt_messages=[{"role": "user", "content": "solve task"}],
        verify_fn=verifier,
    )
    assert out.verified is True
    assert out.abstained is False
    assert out.provenance == "llm_escalated"
    assert out.llm_calls == 2
    assert out.output == {"answer": "verified_attempt"}
    # Retry prompt carries verifier feedback
    assert "failed_unit_tests" in seen_prompts[1][-1]["content"]
    store.close()


def test_arm4_vetoed_after_retry_fails_closed_never_leaks_unverified_text(tmp_path):
    store = GatewayStore(backend="sqlite", sqlite_path=str(tmp_path / "gw.sqlite3"))
    tel = GatewayTelemetry()

    def fake_llm(messages, model):
        return UpstreamCompletion(content="UNVERIFIED_HALLUCINATION_SECRET", model=model)

    router = CascadeRouter(store=store, telemetry=tel, upstream_callable=fake_llm, max_retries=1)
    arm0 = {"status": "ABSTAIN", "reason": "not_in_manifold"}
    out = router.resolve(
        endpoint="/v1/chat/completions",
        policy=_policy(budget=5),
        cascade_policy="cascade",
        arm0_output=arm0,
        arm0_abstained=True,
        trap_detected=False,
        prompt_messages=[{"role": "user", "content": "hallucinate"}],
        verify_fn=lambda _txt: (False, None, "ontology_veto"),
    )
    assert out.verified is False
    assert out.abstained is True
    assert out.verdict == "vetoed_after_retry"
    assert out.llm_calls == 2
    assert out.cascade_detail["unverified_llm_suppressed"] is True
    assert "UNVERIFIED_HALLUCINATION_SECRET" not in json.dumps(out.output)
    store.close()


def test_spend_guard_budget_exhaustion_fails_closed(tmp_path):
    store = GatewayStore(backend="sqlite", sqlite_path=str(tmp_path / "gw.sqlite3"))
    tel = GatewayTelemetry()
    calls = []

    def fake_llm(messages, model):
        calls.append(1)
        return UpstreamCompletion(content="candidate", model=model)

    router = CascadeRouter(store=store, telemetry=tel, upstream_callable=fake_llm, max_retries=1)
    # Budget of 1 allows attempt 1, denies retry attempt 2
    out = router.resolve(
        endpoint="/v1/chat/completions",
        policy=_policy(budget=1),
        cascade_policy="cascade",
        arm0_output={"status": "ABSTAIN"},
        arm0_abstained=True,
        trap_detected=False,
        prompt_messages=[{"role": "user", "content": "q"}],
        verify_fn=lambda _txt: (False, None, "vetoed"),
    )
    assert out.abstained is True
    assert out.verdict == "abstained_budget_exhausted"
    assert out.llm_calls == 1
    assert len(calls) == 1

    # Subsequent request gets 0 calls immediately
    out2 = router.resolve(
        endpoint="/v1/chat/completions",
        policy=_policy(budget=1),
        cascade_policy="cascade",
        arm0_output={"status": "ABSTAIN"},
        arm0_abstained=True,
        trap_detected=False,
        prompt_messages=[{"role": "user", "content": "q2"}],
        verify_fn=lambda _txt: (True, "ok", "ok"),
    )
    assert out2.abstained is True
    assert out2.verdict == "abstained_budget_exhausted"
    assert out2.llm_calls == 0
    assert len(calls) == 1
    store.close()
