"""Directive J4: the API surface an external harness actually touches.

What makes these tests able to fail:

* a missing or wrong Bearer token must be refused before a model runs, and
  /healthz must stay public while /v1/* is closed;
* the token bucket must empty, refuse, and refill from a clock the test
  advances, so no assertion depends on sleeping;
* sixteen reader threads must produce only valid receipts and a contiguous
  chain, and their throughput is compared against the pre-J4 baseline that
  held one lock for every call;
* every chat action must come back as an OpenAI envelope whose receipt
  re-hashes, and a refusal must read exactly "ABSTAIN".
"""

from __future__ import annotations

import threading
import time

import pytest
from fastapi.testclient import TestClient

from henri.holarch.config import HolarchConfig
from henri.holarch.server import (
    RWLock,
    RateLimiter,
    SidecarService,
    build_app,
    select_chat_action,
    verify_receipt,
)

PIXELS = [[[float((row * col) % 5) / 5.0 for col in range(8)] for row in range(8)]]


class FakeClock:
    """The only clock the limiter sees, so refill is arithmetic, not sleep."""

    def __init__(self, start: float = 1_000.0) -> None:
        self.now = float(start)
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self.now

    def advance(self, seconds: float) -> None:
        with self._lock:
            self.now += float(seconds)


def _service(tmp_path, name: str = "j4.sqlite3", **kwargs) -> SidecarService:
    return SidecarService(
        cfg=HolarchConfig.small(),
        backend="sqlite",
        memory_path=str(tmp_path / name),
        autoload=False,
        **kwargs,
    )


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _chat_body(content: str, metadata=None, tools=None, model: str = "henri-holarch") -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": content}]}
    if metadata is not None:
        body["metadata"] = metadata
    if tools is not None:
        body["tools"] = tools
    return body


def _assert_chat_envelope(body: dict, action: str) -> None:
    assert body["id"].startswith("chatcmpl-")
    assert body["object"] == "chat.completion"
    assert isinstance(body["created"], int)
    assert body["model"]
    choice = body["choices"][0]
    assert choice["index"] == 0
    assert choice["message"]["role"] == "assistant"
    assert isinstance(choice["message"]["content"], str)
    assert choice["finish_reason"] == "stop"
    usage = body["usage"]
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    assert body["holarch"]["action"] == action
    assert verify_receipt(body["epistemic_receipt"])


# --------------------------------------------------------------- the bucket
def test_token_bucket_is_deterministic_with_a_fake_clock():
    clock = FakeClock()
    limiter = RateLimiter(rps=2.0, burst=3.0, clock=clock)
    assert [limiter.take("alpha") for _ in range(4)] == [True, True, True, False]
    clock.advance(0.25)  # half a token: still refused
    assert limiter.take("alpha") is False
    clock.advance(0.25)  # one whole token: allowed exactly once
    assert limiter.take("alpha") is True
    assert limiter.take("alpha") is False
    # A second namespace has its own full bucket.
    assert limiter.take("beta") is True
    assert limiter.stats()["refused"] == 3.0


def test_rate_limit_can_be_disabled():
    assert RateLimiter(rps=None).take("anything") is True
    assert RateLimiter(rps=None).enabled is False


# ----------------------------------------------------------------- the gate
def test_bearer_auth_valid_missing_invalid_and_healthz_public(tmp_path):
    service = _service(tmp_path, api_keys={"s3cret"})
    try:
        client = TestClient(build_app(service))
        assert client.get("/healthz").status_code == 200

        missing = client.post("/v1/holarch/veto", json={"task": "vision", "x": PIXELS})
        assert missing.status_code == 401
        assert missing.json() == {"ok": False, "error": "unauthorized"}

        invalid = client.post(
            "/v1/holarch/veto",
            json={"task": "vision", "x": PIXELS},
            headers=_auth("not-the-key"),
        )
        assert invalid.status_code == 401
        assert invalid.json() == {"ok": False, "error": "unauthorized"}

        wrong_scheme = client.post(
            "/v1/holarch/veto",
            json={"task": "vision", "x": PIXELS},
            headers={"Authorization": "Basic s3cret"},
        )
        assert wrong_scheme.status_code == 401

        valid = client.post(
            "/v1/holarch/veto",
            json={"task": "vision", "x": PIXELS},
            headers=_auth("s3cret"),
        )
        assert valid.status_code == 200
        assert verify_receipt(valid.json()["receipt"])

        chat = client.post("/v1/chat/completions", json=_chat_body("verify this"))
        assert chat.status_code == 401
        chat_ok = client.post(
            "/v1/chat/completions",
            json=_chat_body(
                "verify this claim",
                metadata={
                    "action": "verify",
                    "payload": {"claim": "def identity(value):\n    return value\n", "kind": "code"},
                },
            ),
            headers=_auth("s3cret"),
        )
        assert chat_ok.status_code == 200
        assert verify_receipt(chat_ok.json()["epistemic_receipt"])
    finally:
        service.close()


def test_env_api_keys_are_honoured_at_build_time(tmp_path, monkeypatch):
    monkeypatch.delenv("HENRI_API_KEYS", raising=False)
    monkeypatch.setenv("HENRI_API_KEY", "env-key")
    service = _service(tmp_path)
    try:
        client = TestClient(build_app(service))
        body = _chat_body(
            "verify this claim",
            metadata={
                "action": "verify",
                "payload": {"claim": "def identity(value):\n    return value\n", "kind": "code"},
            },
        )
        assert client.post("/v1/chat/completions", json=body).status_code == 401
        assert (
            client.post("/v1/chat/completions", json=body, headers=_auth("env-key")).status_code
            == 200
        )

        monkeypatch.delenv("HENRI_API_KEY")
        monkeypatch.setenv("HENRI_API_KEYS", "one, two")
        client = TestClient(build_app(service))
        assert client.post("/v1/chat/completions", json=body).status_code == 401
        assert client.post("/v1/chat/completions", json=body, headers=_auth("two")).status_code == 200
        assert client.post("/v1/chat/completions", json=body, headers=_auth("env-key")).status_code == 401
    finally:
        service.close()


def test_unconfigured_auth_passes_everything(tmp_path, monkeypatch):
    monkeypatch.delenv("HENRI_API_KEYS", raising=False)
    monkeypatch.delenv("HENRI_API_KEY", raising=False)
    service = _service(tmp_path)
    try:
        client = TestClient(build_app(service))
        veto = client.post("/v1/holarch/veto", json={"task": "vision", "x": PIXELS})
        assert veto.status_code == 200
        assert verify_receipt(veto.json()["receipt"])
        chat = client.post(
            "/v1/chat/completions",
            json=_chat_body(
                "verify this claim",
                metadata={
                    "action": "verify",
                    "payload": {"claim": "def identity(value):\n    return value\n", "kind": "code"},
                },
            ),
        )
        assert chat.status_code == 200
        assert verify_receipt(chat.json()["epistemic_receipt"])
    finally:
        service.close()


def test_rate_limit_burst_refill_and_namespace_isolation(tmp_path):
    clock = FakeClock()
    service = _service(tmp_path, rate_limit_rps=10.0, rate_limit_burst=5.0, clock=clock)
    try:
        client = TestClient(build_app(service))
        for _ in range(5):
            assert client.post("/v1/holarch/veto", json={"task": "vision", "x": PIXELS}).status_code == 200
        blocked = client.post("/v1/holarch/veto", json={"task": "vision", "x": PIXELS})
        assert blocked.status_code == 429
        assert blocked.json() == {"ok": False, "error": "rate_limited"}

        # Health stays public and is never charged to the bucket.
        for _ in range(20):
            assert client.get("/healthz").status_code == 200

        # A second namespace owns a second bucket.
        other = client.post(
            "/v1/holarch/veto",
            json={"task": "vision", "x": PIXELS, "namespace": "other"},
        )
        assert other.status_code == 200

        # 0.1 s of refill at 10 rps is exactly one token.
        clock.advance(0.1)
        assert client.post("/v1/holarch/veto", json={"task": "vision", "x": PIXELS}).status_code == 200
        assert client.post("/v1/holarch/veto", json={"task": "vision", "x": PIXELS}).status_code == 429
    finally:
        service.close()


def test_rate_limit_applies_to_chat_by_metadata_namespace(tmp_path):
    clock = FakeClock()
    service = _service(tmp_path, rate_limit_rps=1.0, rate_limit_burst=1.0, clock=clock)
    try:
        client = TestClient(build_app(service))
        body = _chat_body(
            "verify this claim",
            metadata={
                "action": "verify",
                "namespace": "tenant-j",
                "payload": {"claim": "def identity(value):\n    return value\n", "kind": "code"},
            },
        )
        assert client.post("/v1/chat/completions", json=body).status_code == 200
        second = client.post("/v1/chat/completions", json=body)
        assert second.status_code == 429
        assert second.json() == {"ok": False, "error": "rate_limited"}
        # The default namespace was never charged by the tenant-j call.
        assert service.limiter.available("default") == pytest.approx(1.0)
    finally:
        service.close()


def test_rate_limit_disabled_allows_a_long_burst(tmp_path):
    service = _service(tmp_path, rate_limit_rps=None)
    try:
        client = TestClient(build_app(service))
        for _ in range(120):
            assert client.post("/v1/holarch/veto", json={"task": "vision", "x": PIXELS}).status_code == 200
    finally:
        service.close()


# ------------------------------------------------------------------ the lock
def test_rwlock_readers_share_and_a_waiting_writer_is_preferred():
    lock = RWLock()
    barrier = threading.Barrier(8)
    release = threading.Event()
    peaks: list[int] = []

    def reader() -> None:
        with lock.read_locked():
            peaks.append(lock.stats()["readers"])
            barrier.wait(timeout=5)
            release.wait(timeout=5)

    readers = [threading.Thread(target=reader) for _ in range(8)]
    for thread in readers:
        thread.start()
    deadline = time.monotonic() + 5.0
    while len(peaks) < 8 and time.monotonic() < deadline:
        time.sleep(0.005)
    assert len(peaks) == 8, "eight readers did not enter the shared section together"
    assert max(peaks) >= 2
    release.set()
    for thread in readers:
        thread.join(timeout=5)
        assert not thread.is_alive()

    # Writer preference: a reader arriving behind a waiting writer waits.
    order: list[str] = []
    inside = RWLock()
    writer_holds = threading.Event()
    reader_attempted = threading.Event()
    reader_done = threading.Event()

    def writer() -> None:
        with inside.write_locked():
            order.append("writer")
            writer_holds.set()
            time.sleep(0.05)

    def late_reader() -> None:
        reader_attempted.set()
        with inside.read_locked():
            order.append("reader")
        reader_done.set()

    assert inside.acquire_read()
    try:
        writer_thread = threading.Thread(target=writer)
        writer_thread.start()
        deadline = time.monotonic() + 5.0
        while inside.stats()["waiting_writers"] < 1 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert inside.stats()["waiting_writers"] == 1
        reader_thread = threading.Thread(target=late_reader)
        reader_thread.start()
        assert reader_attempted.wait(timeout=5)
        time.sleep(0.02)
        assert inside.stats()["readers"] == 1, "a new reader jumped a waiting writer"
    finally:
        inside.release_read()
    writer_holds.wait(timeout=5)
    assert not reader_done.is_set(), "the late reader entered while the writer held the lock"
    writer_thread.join(timeout=5)
    reader_thread.join(timeout=5)
    assert order == ["writer", "reader"]


def _run_threads(fn, per_thread: int, threads: int):
    receipts: list[dict] = []
    errors: list[BaseException] = []
    barrier = threading.Barrier(threads)

    def worker() -> None:
        try:
            barrier.wait(timeout=10)
            for _ in range(per_thread):
                receipts.append(fn())
        except BaseException as exc:  # noqa: BLE001 - the test asserts the list
            errors.append(exc)

    workers = [threading.Thread(target=worker) for _ in range(threads)]
    started = time.perf_counter()
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join(timeout=120)
    return time.perf_counter() - started, receipts, errors


def test_sixteen_concurrent_reads_valid_receipts_and_throughput(tmp_path):
    # Directive N2: the response cache is off for this probe. A cached read
    # returns before the reader lock, so it cannot show reader overlap; the
    # benchmark measures the execution path, not the memo table.
    service = _service(tmp_path, cache_enabled=False)
    try:
        service.veto("vision", PIXELS)  # one warm call outside the measurement
        threads, per_thread = 16, 8

        def serialized_call():
            # The pre-J4 shape: every call queued on the one service lock.
            with service.lock:
                return service.veto("vision", PIXELS)

        def shared_call():
            return service.veto("vision", PIXELS)

        serial_seconds, serial_receipts, serial_errors = _run_threads(
            serialized_call, per_thread, threads
        )
        shared_seconds, shared_receipts, shared_errors = _run_threads(
            shared_call, per_thread, threads
        )
        assert serial_errors == []
        assert shared_errors == [], shared_errors
        assert len(shared_receipts) == threads * per_thread
        assert all(response["ok"] for response in shared_receipts)
        assert all(verify_receipt(response["receipt"]) for response in shared_receipts)
        assert service.ledger.verify_chain() is True

        serial_rate = len(serial_receipts) / max(serial_seconds, 1e-9)
        shared_rate = len(shared_receipts) / max(shared_seconds, 1e-9)
        ratio = shared_rate / max(serial_rate, 1e-9)
        # This host is a GIL build, so the model work itself does not overlap;
        # the shared lock removes the queue, it does not create cores. The
        # measured ratio is reported, and the floor is "no serialisation
        # collapse". True reader concurrency is proven by the barrier test
        # above, where eight readers hold the shared section at once.
        print(
            f"\nthroughput serialized={serial_rate:.1f}/s shared={shared_rate:.1f}/s "
            f"ratio={ratio:.2f} threads={threads} calls={len(shared_receipts)}"
        )
        assert shared_rate > serial_rate * 0.4
        assert shared_rate > 10.0
        assert service.rw.stats()["total_reads"] >= threads * per_thread
    finally:
        service.close()


# ------------------------------------------------------------------- the chat
def test_chat_action_selection_is_deterministic():
    cases = {
        "please verify this claim": "verify",
        "check whether alpha exists": "verify",
        "what is the impact of beta": "impact",
        "who calls beta": "impact",
        "plan the next action": "plan",
        "this is a cul-de-sac": "plan",
        "synthesize code for a clamp": "synthesize",
        "write a function that clamps a unit value": "synthesize",
        "solve this path": "solve",
        "hello there": "composite",
    }
    for text, expected in cases.items():
        assert select_chat_action([{"role": "user", "content": text}]) == expected, text

    tool = [{"type": "function", "function": {"name": "holarch_solve"}}]
    assert select_chat_action([{"role": "user", "content": "hello"}], tools=tool) == "solve"
    assert (
        select_chat_action(
            [{"role": "user", "content": "please verify this"}],
            metadata={"action": "holarch_plan"},
        )
        == "plan"
    )
    assert (
        select_chat_action([{"role": "user", "content": "action=impact please"}]) == "impact"
    )


def _package(tmp_path):
    root = tmp_path / "pkg"
    root.mkdir()
    (root / "alpha.py").write_text(
        "def alpha(value):\n    return beta(value) + 1\n", encoding="utf-8"
    )
    (root / "beta.py").write_text("def beta(value):\n    return value - 1\n", encoding="utf-8")
    return root


def test_chat_completions_verify_impact_plan_synthesize(tmp_path):
    service = _service(tmp_path)
    root = _package(tmp_path)
    try:
        service.ingest_code(str(root), package="pkg", namespace="chat")
        client = TestClient(build_app(service))

        verify = client.post(
            "/v1/chat/completions",
            json=_chat_body(
                "verify this claim",
                metadata={
                    "namespace": "chat",
                    "action": "verify",
                    "payload": {"claim": "def identity(value):\n    return value\n", "kind": "code"},
                },
            ),
        )
        assert verify.status_code == 200
        verify_body = verify.json()
        _assert_chat_envelope(verify_body, "verify")
        assert verify_body["holarch"]["status"] == "supported"
        assert verify_body["choices"][0]["message"]["content"]
        assert (
            verify_body["epistemic_receipt"]["parent"]
            == verify_body["holarch"]["agent_receipt"]["sha256"]
        )

        impact = client.post(
            "/v1/chat/completions",
            json=_chat_body("who calls beta", metadata={"namespace": "chat"}),
        )
        assert impact.status_code == 200
        impact_body = impact.json()
        _assert_chat_envelope(impact_body, "impact")
        assert "alpha:alpha" in impact_body["choices"][0]["message"]["content"]

        plan = client.post(
            "/v1/chat/completions",
            json=_chat_body(
                "plan the next action",
                metadata={
                    "namespace": "chat",
                    "payload": {"observation": [0.0] * 79, "transition": "shift"},
                },
            ),
        )
        assert plan.status_code == 200
        plan_body = plan.json()
        _assert_chat_envelope(plan_body, "plan")
        assert 0 <= plan_body["holarch"]["result"]["action"] < 4
        assert plan_body["choices"][0]["message"]["content"]

        synthesize = client.post(
            "/v1/chat/completions",
            json=_chat_body(
                "synthesize a function that clamps a unit value",
                metadata={"namespace": "chat"},
            ),
        )
        assert synthesize.status_code == 200
        synth_body = synthesize.json()
        _assert_chat_envelope(synth_body, "synthesize")
        assert verify_receipt(synth_body["epistemic_receipt"])
        if synth_body["holarch"]["abstained"]:
            assert synth_body["choices"][0]["message"]["content"] == "ABSTAIN"
        assert service.ledger.verify_chain() is True
    finally:
        service.close()


def test_chat_completions_abstains_on_refusal(tmp_path):
    service = _service(tmp_path)
    try:
        client = TestClient(build_app(service))
        refused = client.post(
            "/v1/chat/completions",
            json=_chat_body(
                "verify this claim",
                metadata={
                    "action": "verify",
                    "payload": {
                        "claim": "The medium profile has 1,020,381 parameters.",
                        "kind": "text",
                    },
                },
            ),
        )
        assert refused.status_code == 200
        body = refused.json()
        _assert_chat_envelope(body, "verify")
        assert body["holarch"]["abstained"] is True
        assert body["choices"][0]["message"]["content"] == "ABSTAIN"
        assert body["holarch"]["agent_receipt"] is not None

        # A missing code index is a refusal, not a 500.
        missing = client.post(
            "/v1/chat/completions",
            json=_chat_body(
                "who calls beta",
                metadata={"action": "impact", "namespace": "empty"},
            ),
        )
        assert missing.status_code == 200
        missing_body = missing.json()
        _assert_chat_envelope(missing_body, "impact")
        assert missing_body["choices"][0]["message"]["content"] == "ABSTAIN"
        assert missing_body["holarch"]["status"] == "error"

        # Composite without operands is refused before a model runs.
        composite = client.post(
            "/v1/chat/completions",
            json=_chat_body("please do something"),
        )
        assert composite.status_code == 200
        composite_body = composite.json()
        _assert_chat_envelope(composite_body, "composite")
        assert composite_body["choices"][0]["message"]["content"] == "ABSTAIN"
        assert composite_body["holarch"]["agent_receipt"] is None
    finally:
        service.close()
