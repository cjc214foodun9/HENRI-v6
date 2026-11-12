"""FastAPI and MCP transports for the HENRI agentic sidecar.

Evidence label: DERIVED for the transport wiring, OBSERVED for every
number an endpoint returns.

Decision A5 asks for one epistemic service, two transports. Both call the
same SidecarService object, so the REST endpoint and the MCP tool cannot
disagree about a veto, a solve, or a playbook.

Every call returns a SHA-256 epistemic receipt:

    receipt = sha256(canonical(endpoint, request, result, parent,
                               created_utc, config_digest))

The canonical form is JSON with sorted keys and no whitespace, so a
client can recompute the digest from the response body. Each receipt
names its parent, which makes the in-process sequence a hash chain: a
reordered or edited response breaks the chain at the next call. The
receipt is evidence of what was answered, not a signature; no key is
claimed.

The endpoints are:

    POST /v1/holarch/veto          is this input inside the manifold?
    POST /v1/holarch/ingest_code   index a package once, no weight change
    POST /v1/holarch/solve         answer a symbolic query or abstain
    POST /v1/holarch/playbook      compile or recall an ACE playbook
    POST /v1/holarch/adapt_rule    one surprise-gated fast-weight write
    POST /v1/chat/completions      the same service behind an OpenAI envelope

Directive J4 adds optional Bearer auth (api_keys or HENRI_API_KEYS /
HENRI_API_KEY at build time), a per-namespace token bucket with an
injected clock, and a writer-preferring RWLock. /healthz stays public and
is never rate limited; every /v1/* route is gated when keys are configured.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from henri.holarch.config import HolarchConfig
from henri.holarch.model import find_checkpoint, load_checkpoint
from henri.holarch.sidecar import EpistemicSidecar
from henri.holarch.symbolic import CompositeSolver

GENESIS = "0" * 64


def canonical_json(payload: Any) -> str:
    """Stable JSON: sorted keys, no whitespace, no NaN."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False)


def _clean(value: Any) -> Any:
    """Convert tensors and numpy scalars to JSON-safe Python values.

    A non-finite float becomes null: strict JSON has no NaN, and a
    consumer that needs the value must read the numeric row instead.
    """
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, torch.Tensor):
        return _clean(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return _clean(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    return str(value)


def receipt_body(
    endpoint: str,
    request: Any,
    result: Any,
    *,
    parent: str = GENESIS,
    created_utc: Optional[str] = None,
    config_digest: str = "",
) -> Dict[str, Any]:
    """The exact object the digest covers."""
    return {
        "algorithm": "sha256",
        "canonical": "json-sorted-keys",
        "config_digest": str(config_digest),
        "created_utc": created_utc or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "endpoint": str(endpoint),
        "parent": str(parent),
        "request": _clean(request),
        "result": _clean(result),
    }


def issue_receipt(
    endpoint: str,
    request: Any,
    result: Any,
    *,
    parent: str = GENESIS,
    created_utc: Optional[str] = None,
    config_digest: str = "",
) -> Dict[str, Any]:
    """Return the receipt for one call, chained to parent."""
    body = receipt_body(
        endpoint,
        request,
        result,
        parent=parent,
        created_utc=created_utc,
        config_digest=config_digest,
    )
    digest = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
    return {**body, "sha256": digest}


def verify_receipt(receipt: Dict[str, Any]) -> bool:
    """Recompute the digest of a receipt and compare it."""
    try:
        expected = hashlib.sha256(canonical_json(receipt_body(
            receipt["endpoint"],
            receipt.get("request"),
            receipt.get("result"),
            parent=receipt.get("parent", GENESIS),
            created_utc=receipt.get("created_utc"),
            config_digest=receipt.get("config_digest", ""),
        )).encode("utf-8")).hexdigest()
    except Exception:
        return False
    return expected == receipt.get("sha256")


class RWLock:
    """Writer-preferring readers-writer lock.

    Directive J4: the receipt chain needs one writer at a time, but a read
    call does not. Readers run together; a writer waits for every reader to
    leave. A new reader waits while any writer is waiting, so a continuous
    stream of reads cannot starve a write.

    Reentrancy: the thread that holds the write lock may acquire it again or
    take a read lock. A thread that already holds a read lock may take the
    read lock again even when a writer is queued, because that writer can
    only proceed after this reader leaves and waiting would deadlock the
    reader against itself.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition(threading.Lock())
        self._readers = 0
        self._reader_depth: Dict[int, int] = {}
        self._writer: Optional[int] = None
        self._write_depth = 0
        self._waiting_writers = 0
        self._total_reads = 0
        self._total_writes = 0
        self._peak_readers = 0

    # ------------------------------------------------------------- readers
    def acquire_read(self, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        me = threading.get_ident()
        with self._condition:
            deadline = None if timeout is None else time.monotonic() + float(timeout)
            while True:
                if self._writer == me or self._reader_depth.get(me, 0) > 0:
                    break
                if self._writer is None and self._waiting_writers == 0:
                    break
                if not blocking:
                    return False
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0:
                        return False
                    self._condition.wait(remaining)
                else:
                    self._condition.wait()
            self._reader_depth[me] = self._reader_depth.get(me, 0) + 1
            self._readers += 1
            self._total_reads += 1
            if self._readers > self._peak_readers:
                self._peak_readers = self._readers
            return True

    def release_read(self) -> None:
        me = threading.get_ident()
        with self._condition:
            depth = self._reader_depth.get(me, 0)
            if depth <= 0:
                raise RuntimeError("release_read called without acquire_read")
            if depth == 1:
                del self._reader_depth[me]
            else:
                self._reader_depth[me] = depth - 1
            self._readers -= 1
            if self._readers == 0:
                self._condition.notify_all()

    # -------------------------------------------------------------- writer
    def acquire_write(self, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        me = threading.get_ident()
        with self._condition:
            if self._writer == me:
                self._write_depth += 1
                self._total_writes += 1
                return True
            deadline = None if timeout is None else time.monotonic() + float(timeout)
            self._waiting_writers += 1
            try:
                while self._writer is not None or self._readers > 0:
                    if not blocking:
                        return False
                    if deadline is not None:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0.0:
                            return False
                        self._condition.wait(remaining)
                    else:
                        self._condition.wait()
                self._writer = me
                self._write_depth = 1
                self._total_writes += 1
                return True
            finally:
                self._waiting_writers -= 1

    def release_write(self) -> None:
        me = threading.get_ident()
        with self._condition:
            if self._writer != me:
                raise RuntimeError("release_write called without acquire_write")
            self._write_depth -= 1
            if self._write_depth == 0:
                self._writer = None
                self._condition.notify_all()

    # ------------------------------------------------------ context forms
    @contextmanager
    def read_locked(self) -> Iterator[None]:
        self.acquire_read()
        try:
            yield
        finally:
            self.release_read()

    @contextmanager
    def write_locked(self) -> Iterator[None]:
        self.acquire_write()
        try:
            yield
        finally:
            self.release_write()

    # -------------------------------------------------------------- survey
    def stats(self) -> Dict[str, int]:
        """A point-in-time view, for health output and for tests."""
        with self._condition:
            return {
                "readers": int(self._readers),
                "writer": 0 if self._writer is None else 1,
                "waiting_writers": int(self._waiting_writers),
                "peak_readers": int(self._peak_readers),
                "total_reads": int(self._total_reads),
                "total_writes": int(self._total_writes),
            }


class RateLimiter:
    """Deterministic in-memory token bucket, one bucket per namespace.

    The bucket starts full, refills at rate_limit_rps tokens per second, and
    never holds more than rate_limit_burst. The clock is injected so a test
    advances time instead of sleeping. rate_limit_rps=None disables the
    limiter: take() always returns True and nothing is stored.
    """

    def __init__(
        self,
        rps: Optional[float] = 50.0,
        burst: float = 100.0,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        if rps is not None and float(rps) < 0.0:
            raise ValueError("rate_limit_rps must be non-negative or None")
        if float(burst) <= 0.0:
            raise ValueError("rate_limit_burst must be positive")
        self.rps = None if rps is None else float(rps)
        self.burst = float(burst)
        self.clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._buckets: Dict[str, Tuple[float, float]] = {}
        self._allowed = 0
        self._refused = 0

    @property
    def enabled(self) -> bool:
        return self.rps is not None

    def take(self, namespace: str = "default", tokens: float = 1.0) -> bool:
        """Spend tokens from one bucket. True means the call may proceed."""
        if self.rps is None:
            return True
        key = str(namespace or "default")
        need = max(0.0, float(tokens))
        with self._lock:
            now = float(self.clock())
            available, stamp = self._buckets.get(key, (self.burst, now))
            available = min(self.burst, available + max(0.0, now - float(stamp)) * self.rps)
            if available + 1e-12 >= need:
                self._buckets[key] = (available - need, now)
                self._allowed += 1
                return True
            self._buckets[key] = (available, now)
            self._refused += 1
            return False

    def available(self, namespace: str = "default") -> float:
        """Tokens left in one bucket at the current clock reading."""
        if self.rps is None:
            return float("inf")
        key = str(namespace or "default")
        with self._lock:
            now = float(self.clock())
            available, stamp = self._buckets.get(key, (self.burst, now))
            return min(self.burst, available + max(0.0, now - float(stamp)) * self.rps)

    def reset(self) -> None:
        with self._lock:
            self._buckets.clear()
            self._allowed = 0
            self._refused = 0

    def stats(self) -> Dict[str, float]:
        with self._lock:
            return {
                "enabled": 1.0 if self.rps is not None else 0.0,
                "rps": -1.0 if self.rps is None else float(self.rps),
                "burst": float(self.burst),
                "namespaces": float(len(self._buckets)),
                "allowed": float(self._allowed),
                "refused": float(self._refused),
            }


@dataclass
class ReceiptLedger:
    """Append-only hash chain of the receipts this process issued.

    Directive J4: issuance is the one thing that must stay serial, because
    each receipt names its parent. The chain lock is short (hash + append)
    and is never held while a model runs, so readers do not queue on it.
    """

    parent: str = GENESIS
    count: int = 0
    receipts: List[Dict[str, Any]] = field(default_factory=list)
    max_retained: int = 256
    _lock: Any = field(default_factory=threading.RLock, repr=False, compare=False)

    def issue(self, endpoint: str, request: Any, result: Any, config_digest: str = "") -> Dict[str, Any]:
        with self._lock:
            receipt = issue_receipt(
                endpoint,
                request,
                result,
                parent=self.parent,
                config_digest=config_digest,
            )
            self.parent = receipt["sha256"]
            self.count += 1
            self.receipts.append(receipt)
            if len(self.receipts) > int(self.max_retained):
                del self.receipts[: len(self.receipts) - int(self.max_retained)]
            return receipt

    def head(self) -> str:
        with self._lock:
            return self.parent

    def snapshot(self) -> Dict[str, Any]:
        """count, head and chain validity read from one consistent state."""
        with self._lock:
            return {
                "count": int(self.count),
                "head": str(self.parent),
                "chain_valid": bool(self._verify_chain_locked()),
            }

    def verify_chain(self) -> bool:
        with self._lock:
            return self._verify_chain_locked()

    def _verify_chain_locked(self) -> bool:
        # Retention may have dropped the genesis receipt; the oldest retained
        # one still names the parent it was issued against.
        retained = list(self.receipts)
        parent = retained[0].get("parent", GENESIS) if retained else GENESIS
        for receipt in retained:
            if receipt.get("parent") != parent:
                return False
            if not verify_receipt(receipt):
                return False
            parent = receipt["sha256"]
        return True


# ------------------------------------------------------- chat adapter (J4)
CHAT_ACTIONS: Tuple[str, ...] = (
    "verify",
    "solve",
    "impact",
    "plan",
    "synthesize",
    "composite",
    "veto",
)

_CHAT_ACTION_ALIASES: Dict[str, str] = {
    "verify": "verify",
    "check": "verify",
    "claim": "verify",
    "veto": "veto",
    "solve": "solve",
    "answer": "solve",
    "path": "solve",
    "impact": "impact",
    "callers": "impact",
    "blast": "impact",
    "plan": "plan",
    "planner": "plan",
    "planning": "plan",
    "synthesize": "synthesize",
    "synthesise": "synthesize",
    "synthesis": "synthesize",
    "generate": "synthesize",
    "codegen": "synthesize",
    "composite": "composite",
    "compose": "composite",
}

_CHAT_REFUSAL_STATUSES = frozenset(
    {"refused", "rejected", "unsupported", "vetoed", "abstained", "error", "invalid_request"}
)

_CHAT_STOPWORDS = frozenset(
    {
        "the", "a", "an", "of", "for", "to", "in", "on", "is", "are", "was",
        "what", "which", "who", "whom", "whose", "does", "do", "did", "and",
        "or", "please", "impact", "verify", "check", "plan", "solve", "synthesize",
        "code", "function", "target", "callers", "calls", "blast", "radius",
        "next", "action", "cul", "de", "sac", "exists", "exist", "this", "that",
        "with", "from", "about", "tell", "me", "show", "write", "generate",
    }
)


def normalize_chat_action(value: Any) -> str:
    """Map a tool name, an alias or an action= token to one agent action."""
    text = re.sub(r"[^a-z0-9_]+", "_", str(value or "").strip().lower()).strip("_")
    if not text:
        return ""
    for prefix in ("holarch_", "henri_"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    return _CHAT_ACTION_ALIASES.get(text, "")


def _chat_message_field(message: Any, name: str, default: Any = None) -> Any:
    if isinstance(message, dict):
        return message.get(name, default)
    return getattr(message, name, default)


def chat_message_text(content: Any) -> str:
    """Flatten an OpenAI message content value to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        parts: List[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part)
    return str(content)


def chat_text(messages: Optional[Sequence[Any]]) -> str:
    """The user text: every user message in order, else every message."""
    message_list = list(messages or [])
    user = [
        chat_message_text(_chat_message_field(message, "content", ""))
        for message in message_list
        if str(_chat_message_field(message, "role", "user")) == "user"
    ]
    if not any(part.strip() for part in user):
        user = [
            chat_message_text(_chat_message_field(message, "content", ""))
            for message in message_list
        ]
    return "\n".join(part for part in user if part).strip()


def _tool_call_names(messages: Optional[Sequence[Any]]) -> List[str]:
    names: List[str] = []
    for message in list(messages or []):
        calls = _chat_message_field(message, "tool_calls", None) or []
        for call in calls:
            if isinstance(call, dict):
                function = call.get("function")
                name = function.get("name") if isinstance(function, dict) else call.get("name")
            else:
                function = getattr(call, "function", None)
                name = getattr(function, "name", None) if function is not None else None
            if name:
                names.append(str(name))
    return names


def select_chat_action(
    messages: Optional[Sequence[Any]] = None,
    tools: Optional[Sequence[Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Choose one agent action deterministically.

    Order: metadata.action, then a tool name in the request, then an
    action=<name> instruction, then the keyword rules over the user text,
    then "composite" as the explicit fallback.
    """
    meta = metadata if isinstance(metadata, dict) else {}
    chosen = normalize_chat_action(meta.get("action"))
    if chosen:
        return chosen
    for tool in list(tools or []):
        if isinstance(tool, dict):
            function = tool.get("function")
            name = function.get("name") if isinstance(function, dict) else tool.get("name")
        else:
            function = getattr(tool, "function", None)
            name = getattr(function, "name", None) if function is not None else None
        chosen = normalize_chat_action(name)
        if chosen:
            return chosen
    text = chat_text(messages)
    instructions = list(_tool_call_names(messages)) + [text]
    for source in instructions:
        match = re.search(r"action\s*[=:]\s*([A-Za-z_][A-Za-z0-9_\-]*)", str(source or ""))
        if match:
            chosen = normalize_chat_action(match.group(1))
            if chosen:
                return chosen
    lowered = text.lower()
    if re.search(r"\bverif(?:y|ies|ication)\b|\bcheck\b|\bdoes\b[^.?!]*\bexist", lowered):
        return "verify"
    if re.search(r"\bimpact\b|\bblast[\s_-]*radius\b|\bwho\s+calls\b|\bcallers?\b", lowered):
        return "impact"
    if re.search(r"\bplan\b|\bcul[\s_-]*de[\s_-]*sac\b|\bnext\s+action\b", lowered):
        return "plan"
    if re.search(r"\bsynthesi[sz]e\b|\bgenerate\s+code\b|\bwrite\s+a\s+function\b", lowered):
        return "synthesize"
    if re.search(r"\bsolve\b|\banswer\b", lowered):
        return "solve"
    return "composite"


def looks_like_code(text: str) -> bool:
    return bool(
        re.search(r"(^|\n)\s*(async\s+def|def|class|import|from)\s+[A-Za-z_]", str(text or ""))
    )


def extract_chat_target(text: str) -> str:
    """The symbol a plain-language impact or solve request is about."""
    body = str(text or "")
    for pattern in (
        r"who\s+calls\s+[`'\"]?([A-Za-z_][A-Za-z0-9_.:]*)",
        r"(?:impact|blast[\s_-]*radius)\s+(?:of|for|on)\s+[`'\"]?([A-Za-z_][A-Za-z0-9_.:]*)",
        r"(?:target|source)\s*[=:]\s*[`'\"]?([A-Za-z_][A-Za-z0-9_.:]*)",
    ):
        match = re.search(pattern, body, re.IGNORECASE)
        if match:
            return str(match.group(1))
    for token in re.findall(r"[A-Za-z_][A-Za-z0-9_.:]*", body):
        if len(token) >= 2 and token.lower() not in _CHAT_STOPWORDS:
            return token
    return ""


def chat_content(action: str, output: Any, refused: bool) -> str:
    """The assistant text for one turn: the verified output, or ABSTAIN."""
    if refused:
        return "ABSTAIN"
    if isinstance(output, dict):
        if action == "plan":
            name = str(output.get("action_name", "") or "")
            if name:
                return f"{name} (action {output.get('action')})"
        if action == "impact":
            callers = [str(item) for item in (output.get("callers") or [])]
            target = str(output.get("target") or "")
            if callers:
                return f"impact {target}: {len(callers)} caller(s): " + ", ".join(callers)
            return f"impact {target}: 0 caller(s)"
        for key in ("answer", "source", "text", "reason", "result"):
            value = output.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return canonical_json(output)
    if output is None:
        return ""
    return str(output)


def chat_stream_chunks(payload: Dict[str, Any]) -> List[str]:
    """OpenAI-style SSE lines for one completed chat turn.

    Directive Z4. The non-stream body is computed first and then replayed as
    delta chunks, so a streaming turn and a blocking turn carry the same
    content, the same usage and the same epistemic receipt. The final line is
    the OpenAI terminator "data: [DONE]".
    """
    completion = dict(payload or {})
    chunk_id = str(completion.get("id", "chatcmpl-henri"))
    created = int(completion.get("created", 0) or 0)
    model = str(completion.get("model", "henri-holarch"))
    choices = list(completion.get("choices") or [{}])
    first = dict(choices[0]) if choices else {}
    content = str((first.get("message") or {}).get("content") or "")
    finish = str(first.get("finish_reason") or "stop")
    usage = dict(completion.get("usage") or {})
    receipt = dict(completion.get("epistemic_receipt") or {})
    holarch = dict(completion.get("holarch") or {})

    def envelope(delta: Dict[str, Any], finish_reason: Any) -> Dict[str, Any]:
        return {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(created),
            "model": model,
            "choices": [
                {"index": 0, "delta": dict(delta), "finish_reason": finish_reason}
            ],
        }

    words = content.split(" ") if content else [""]
    pieces: List[str] = []
    for start in range(0, len(words)):
        text = " ".join(words[start:start + 1])
        if start + 1 < len(words):
            text = text + " "
        pieces.append(text)
    if not pieces:
        pieces = [""]
    events: List[Dict[str, Any]] = [envelope({"role": "assistant", "content": ""}, None)]
    for piece in pieces:
        events.append(envelope({"content": piece}, None))
    final = envelope({}, finish)
    final["usage"] = usage
    final["epistemic_receipt"] = receipt
    final["holarch"] = holarch
    events.append(final)
    lines = ["data: " + canonical_json(event) for event in events]
    lines.append("data: [DONE]")
    return lines


class SidecarService:
    """One sidecar, one receipt chain, one reader/writer lock.

    Directive J4: reads share the lock and writes take it alone, so a veto,
    a solve or a plan no longer queues behind another reader. The chain stays
    contiguous because issuance is serialised on the ledger's own short lock,
    not on the model lock. Every public method returns the uniform envelope

        {ok, endpoint, result, receipt}
    """

    def __init__(
        self,
        cfg: Optional[HolarchConfig] = None,
        *,
        backend: str = "sqlite",
        memory_path: Optional[str] = None,
        sidecar: Optional[EpistemicSidecar] = None,
        checkpoint: Optional[str] = None,
        autoload: bool = True,
        api_keys: Optional[Set[str]] = None,
        rate_limit_rps: Optional[float] = 50.0,
        rate_limit_burst: float = 100.0,
        clock: Optional[Callable[[], float]] = None,
        pin_threads: bool = False,
        cache_enabled: bool = True,
    ) -> None:
        # Directive E2: mount the converged checkpoint and its calibrated
        # prototype bank by default. A fresh bootstrap is the explicit
        # fallback, never the silent one: health() reports which weights
        # the service is answering with and their SHA-256.
        self.checkpoint_info: Dict[str, Any] = {"mounted": False}
        path = None
        if sidecar is None and (autoload or checkpoint):
            path = find_checkpoint(cfg, explicit=checkpoint)
        if path is not None:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if cfg is None:
                cfg = HolarchConfig(**payload["config"])
            elif int(payload.get("config", {}).get("dim", -1)) != int(cfg.dim):
                if checkpoint:
                    raise ValueError(
                        f"checkpoint {path} has dim "
                        f"{payload.get('config', {}).get('dim')} but cfg.dim is {cfg.dim}"
                    )
                path = None
        self.cfg = cfg or HolarchConfig.small()
        self.sidecar = sidecar or EpistemicSidecar(
            cfg=self.cfg, backend=backend, memory_path=memory_path
        )
        if path is not None:
            _, metadata = load_checkpoint(path, model=self.sidecar.model)
            self.checkpoint_info = {
                "mounted": True,
                "path": str(path),
                "sha256": str(metadata.get("sha256", "")),
                "name": str(metadata.get("name", "")),
                "parameters": int(metadata.get("parameters", 0)),
                "config_digest": str(metadata.get("config_digest", "")),
                "created_utc": str(metadata.get("created_utc", "")),
                "prototypes": metadata.get("prototypes", {}),
            }
        self.ledger = ReceiptLedger()
        # J4: reads share, writes are exclusive. self.lock stays for
        # compatibility with callers that already hold it, but no endpoint
        # serialises its readers on it any more.
        self.rw = RWLock()
        self.lock = threading.RLock()
        # Directive N2: a reader never touches the process-global OpenMP
        # pool. The Cycle M path pinned torch.set_num_threads(1) whenever a
        # second reader arrived, which reset a process-global C++ pool inside
        # a live request and starved the J4 planners. The pool is now fixed
        # once at construction (pin_threads) or left at the interpreter
        # default, and the read scope does not mutate it.
        self._reader_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._response_cache: Dict[str, Any] = {}
        # Directive N1: the epoch is part of every cache key. A mutating call
        # advances it and drops the response cache, so a delta index, a
        # playbook compile or an adaptation cannot be answered from a stale
        # pre-mutation entry. The sidecar carries its own epoch, so a direct
        # sidecar.ingest_code call (the gauntlet delta probe) invalidates the
        # service cache through the listener registered below.
        self._state_epoch: int = 0
        self._cache_enabled: bool = bool(cache_enabled)
        # Directive O2: the shared Zone C epoch and the per-episode planner
        # trajectory live on the service, not on the request.
        self._seen_global_epoch: int = -1
        self._episodes: Dict[Any, Dict[str, Any]] = {}
        self._swat_bus = None
        self._swat_gate = None
        # Directive P2: the long-horizon telepathy surface. One bus, one
        # calibrated language gate, and a session registry keyed by id.
        self._lh_bus = None
        self._lh_gate_calibration: Dict[str, Any] = {}
        self._telepathy_sessions: Dict[str, Any] = {}
        self._lh_wave_cache: Dict[str, Any] = {}
        self._active_readers = 0
        self._pinned_threads = False
        self._saved_threads = int(torch.get_num_threads())
        # Directive M2: a concurrent deployment pins the intra-op pool to one
        # thread per caller, so sixteen readers do not spawn 128 OpenMP
        # workers. The pin is opt-in for a dedicated concurrent service and is
        # restored by close(); the serial evaluation service keeps the default
        # pool.
        if bool(pin_threads):
            torch.set_num_threads(1)
            self._pinned_threads = True
        self.api_keys: Set[str] = {str(key) for key in (api_keys or ()) if str(key)}
        self.clock: Callable[[], float] = clock or time.monotonic
        self.limiter = RateLimiter(
            rps=rate_limit_rps, burst=rate_limit_burst, clock=self.clock
        )
        self.config_digest = str(self.cfg.digest())
        # Decision D4: the composite atlas is attached to the service, so a
        # REST or MCP caller can register and resolve multi-arity rules.
        self.composite = CompositeSolver(self.sidecar.model.memory, self.sidecar.codebook)
        self.sidecar.add_state_listener(self._invalidate_state)

    # ------------------------------------------------------------- plumbing
    @contextmanager
    def _read_scope(self) -> Iterator[None]:
        """Shared read lock for one endpoint call.

        Directive N2: no OpenMP thread mutation happens here. The old M2
        path called torch.set_num_threads(1) inside the request, which
        resets the process-global pool and starved the J4 planners. The
        intra-op pool is set once at construction, never per request.
        """
        with self._reader_lock:
            self._active_readers += 1
        try:
            with self.rw.read_locked():
                yield
        finally:
            with self._reader_lock:
                self._active_readers = max(0, self._active_readers - 1)

    def _cache_key(self, endpoint: str, request: Any) -> str:
        return hashlib.sha256(
            json.dumps(
                [
                    int(self._state_epoch),
                    int(getattr(self.sidecar, "state_epoch", 0)),
                    str(endpoint),
                    request,
                ],
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest()

    def _cache_get(self, endpoint: str, request: Any) -> Optional[Any]:
        # Directive O2: a sibling lane may have mutated the shared store since
        # this process last read. The check runs even when the cache is off,
        # because it also drops a stale code index before an impact read.
        self._sync_global_epoch()
        if not self._cache_enabled:
            return None
        with self._cache_lock:
            return self._response_cache.get(self._cache_key(endpoint, request))

    def _cache_put(self, endpoint: str, request: Any, result: Any) -> None:
        if not self._cache_enabled:
            return
        with self._cache_lock:
            if len(self._response_cache) > 512:
                self._response_cache.clear()
            self._response_cache[self._cache_key(endpoint, request)] = result

    def _cache_clear(self) -> None:
        with self._cache_lock:
            self._response_cache.clear()

    def _invalidate_state(self) -> None:
        """Advance the epoch and drop every memoised read (Directive N1)."""
        with self._cache_lock:
            self._state_epoch = int(self._state_epoch) + 1
            self._response_cache.clear()

    def _mark_state(self) -> None:
        """Public spelling used by service-level mutators."""
        memory = getattr(self.sidecar.model, "memory", None)
        bumper = getattr(memory, "bump_state_epoch", None)
        if bumper is not None:
            try:
                bumper()
            except Exception:
                pass
        self._invalidate_state()

    def _sync_global_epoch(self) -> None:
        """Drop this lane's cache and index when a sibling lane mutated.

        Directive O2. The epoch lives in engram_meta, so the comparison is a
        short SQLite read. A changed value clears the response cache and the
        per-process code index; the next impact read rebuilds the index from
        the persisted graph the writer lane left behind.
        """
        memory = getattr(self.sidecar.model, "memory", None)
        reader = getattr(memory, "state_epoch", None)
        if reader is None:
            return
        try:
            epoch = int(reader())
        except Exception:
            return
        if epoch != int(self._seen_global_epoch):
            self._seen_global_epoch = epoch
            self._invalidate_state()
            try:
                self.sidecar._indexers.clear()
                self.sidecar.indexer = None
            except Exception:
                pass

    def _envelope(self, endpoint: str, request: Any, result: Any) -> Dict[str, Any]:
        receipt = self.ledger.issue(
            endpoint, request, result, config_digest=self.config_digest
        )
        return {
            "ok": True,
            "endpoint": endpoint,
            "result": _clean(result),
            "receipt": receipt,
        }

    def _as_tensor(self, task: str, payload: Any) -> torch.Tensor:
        tensor = torch.as_tensor(payload, dtype=torch.float32)
        size = int(getattr(self.cfg, "vision_size", 8))
        if task == "vision":
            if tensor.dim() == 1:
                tensor = tensor.reshape(1, -1)
            if tensor.dim() == 2 and tensor.shape[-1] == size * size:
                tensor = tensor.reshape(-1, 1, size, size)
            elif tensor.dim() == 2 and tuple(tensor.shape) == (size, size):
                tensor = tensor.reshape(1, 1, size, size)
            elif tensor.dim() == 3 and tuple(tensor.shape[-2:]) == (size, size):
                tensor = tensor.unsqueeze(1)
        elif tensor.dim() == 1:
            tensor = tensor.reshape(1, -1)
        return tensor

    # ------------------------------------------------------------- endpoints
    def veto(
        self,
        task: str,
        x: Any,
        rule_id: Optional[int] = None,
        namespace: str = "default",
    ) -> Dict[str, Any]:
        request = {"task": task, "x": x, "rule_id": rule_id, "namespace": str(namespace)}
        cached = self._cache_get("/v1/holarch/veto", request)
        if cached is not None:
            return self._envelope("/v1/holarch/veto", request, cached)
        with self._read_scope():
            tensor = self._as_tensor(task, x)
            if rule_id is None:
                result = self.sidecar.veto_check(task, tensor)
            else:
                output = self.sidecar.model(task, tensor, training=False, rule_id=int(rule_id))
                result = {
                    "veto_rate": float(output.veto.float().mean()),
                    "s_max": float(output.s_max.mean()),
                    "delta_sagnac": float(output.delta_sagnac.mean()),
                    "confidence": float(output.margin.mean()),
                }
            self._cache_put("/v1/holarch/veto", request, result)
            return self._envelope("/v1/holarch/veto", request, result)

    def ingest_code(
        self,
        root: str,
        package: str = "henri",
        namespace: str = "default",
        mode: str = "edges",
        file_path: Optional[str] = None,
        source_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        request = {
            "root": str(root),
            "package": str(package),
            "namespace": str(namespace),
            "mode": str(mode),
            "file_path": None if file_path is None else str(file_path),
            "source_bytes": 0 if source_text is None else len(str(source_text).encode("utf-8")),
        }
        with self.rw.write_locked():
            result = self.sidecar.ingest_code(
                root,
                package=package,
                namespace=str(namespace),
                mode=str(mode),
                file_path=file_path,
                source_text=source_text,
            )
            result["mode"] = str(mode)
            self._mark_state()
            return self._envelope("/v1/holarch/ingest_code", request, result)

    # ------------------------------------- telepathy API + long horizon (P2)
    def _long_horizon_bus(self):
        """The H-SWAT bus used by the public telepathy surface (Directive P2).

        It is separate from the planner's bus: the planner hands over an
        action wave under the action gate, while a telepathy session carries
        document states under a language gate calibrated on real token waves.
        """
        if getattr(self, "_lh_bus", None) is None:
            from henri.holarch.swat_bus import (
                SagnacKVGate,
                SwarmTelepathyBus,
                SwarmTelepathyMapper,
                cache_from_wave,
            )

            mapper = SwarmTelepathyMapper(dim=int(self.cfg.dim))
            mapper.set_shared_frame()
            gate = SagnacKVGate(self.sidecar.model, self.sidecar.model.memory, task="language")
            engine = self.sidecar._ensure_engine("default")
            projector, tokenizer = engine.projector, engine.tokenizer
            calibration = []
            # Calibrate on the SAME encoding the session's names use. A name
            # is a short token sequence, not one vocabulary slot, so a
            # single-token calibration would set a chance floor above the
            # waves the session actually sends.
            for index in range(16):
                wave = self._identity_wave(f"ref{index:02d}")
                calibration.append(
                    cache_from_wave(wave, source=f"calib:{index}", task="language")
                )
            gate.calibrate(calibration)
            bus = SwarmTelepathyBus(mapper, gate=gate, budget_u=32, dim=int(self.cfg.dim))
            self._lh_bus = bus
            self._lh_gate_calibration = dict(gate.calibration)
        return self._lh_bus

    def _identity_wave(self, name: str) -> np.ndarray:
        """A deterministic, mutually incoherent identity code for one name.

        The projector's sequence wave is a SEMANTIC code: two short names can
        share most of their phase (measured: cos(pm1, pm2) = 0.82), which is
        right for language and wrong for a key-value registry, where two
        symbols must be distinguishable. The identity code is a unit vector in
        the same C^D space derived from a SHA-256 of the name, so every symbol
        has its own phase and the read is exact.
        """
        cache = getattr(self, "_lh_identity_cache", None)
        if cache is None:
            cache = {}
            self._lh_identity_cache = cache
        key = str(name)
        cached = cache.get(key)
        if cached is not None:
            return cached
        seed = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        dim = int(self.cfg.dim)
        wave = (
            rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
        ).astype(np.complex64)
        wave = wave / max(float(np.linalg.norm(wave)), 1e-12)
        if len(cache) < 16384:
            cache[key] = wave
        return wave

    def _name_wave(self, name: str, projector, tokenizer) -> np.ndarray:
        cache = getattr(self, "_lh_wave_cache", None)
        if cache is None:
            cache = {}
            self._lh_wave_cache = cache
        key = str(name)
        if key in cache:
            return cache[key]
        tokens = list(tokenizer.encode(str(name)))[:8]
        wave = (
            projector.sequence_wave(tokens, tokenizer)
            if tokens
            else projector.slot_wave(0, tokenizer)
        )
        if len(cache) < 8192:
            cache[key] = wave
        return wave

    def _value_vector(self, wave: np.ndarray) -> np.ndarray:
        """The frame a fact VALUE is written in: slot 0 of its key projection.

        cache_from_pairs writes each fact as (key slot 0, value slot 0), so
        the receiver's candidate table must live in exactly that frame.
        """
        from henri.holarch.swat_bus import kv_from_wave

        # cache_from_pairs writes a fact VALUE from the value half of
        # kv_from_wave, so the receiver's candidate table must live in exactly
        # that half. Reading the key half here compared two decorrelated
        # projections and made every answer a coin flip.
        _keys, values = kv_from_wave(wave, heads=4, dk=32)
        return values[0].reshape(-1).numpy().astype(np.float32)

    def telepathy(
        self,
        mode: str = "ingest_chunks",
        session_id: Optional[str] = None,
        namespace: str = "default",
        chunks: Optional[Sequence[Any]] = None,
        query: Optional[str] = None,
        budget_u: int = 32,
        per_chunk_u: int = 8,
        hops: int = 3,
        top_k: int = 1,
        reset: bool = False,
        reference_facts: Optional[Sequence[Sequence[str]]] = None,
        raw_documents: Optional[Sequence[Any]] = None,
        fact_string_rule: bool = True,
    ) -> Dict[str, Any]:
        """SWAT Appendix F over REST/MCP: ingest C chunks, then query them.

        Directive P2 accepted a query at ingest and walked the bridges inside
        the sender. Directive Q1 removes that restriction: when query is None
        the service gates and compacts every chunk ONCE against its own mean
        key, indexes the cross-chunk fact bindings, and stores the index. A
        later mode="query" call with an unseen query walks that index and
        composes the terminal binding into the receiver cache in under a
        millisecond, with zero receiver text prefill tokens.
        """
        import time as _time

        from henri.holarch.swat_bus import cache_from_pairs

        from henri.holarch.swat_bus import DocumentFactExtractor

        mode = str(mode or "ingest_chunks").strip().lower()
        bus = self._long_horizon_bus()
        projectors = self.sidecar._ensure_engine(namespace)
        projector, tokenizer = projectors.projector, projectors.tokenizer
        sessions = getattr(self, "_telepathy_sessions", None)
        if sessions is None:
            sessions = {}
            self._telepathy_sessions = sessions
        if mode in {"ingest", "ingest_chunks", "encode"}:
            raw_chunks = list(chunks or [])
            raw_documents = list(raw_documents or [])
            if not raw_chunks and not raw_documents:
                raise ValueError("ingest_chunks needs at least one chunk or raw document")
            # ------------------------------------------------ parse (Q1/R1)
            parsed: List[Dict[str, Any]] = []
            extraction: Dict[str, Any] = {}
            if raw_documents:
                # Directive R1: the documents arrive as raw text. The
                # deterministic extractor parses Markdown tables, config and
                # JSON lines, Python AST assignments and prose link sentences
                # into normalised triples. The reference law is built by
                # corroboration across the documents themselves: no caller
                # supplied fact_registry is consulted on this path.
                extractor = DocumentFactExtractor()
                extraction = extractor.extract_documents(raw_documents)
                for document in extraction["documents"]:
                    meta = dict(document.get("meta") or {})
                    facts = [
                        [str(fact.get("key") or fact["subject"]), str(fact["value"])]
                        for fact in document["facts"]
                    ]
                    parsed.append(
                        {
                            "index": int(document["index"]),
                            "facts": facts,
                            "source": str(document["source"]),
                            "corrupt": False,
                            "fabricated": bool(meta.get("fabricated", False)),
                            "fabricated_mode": str(meta.get("fabricated_mode", "mismatch")),
                            "ood": not bool(document["connected"]),
                            "raw": True,
                            "triples": list(document["facts"]),
                            "names": list(document.get("names") or ()),
                        }
                    )
            else:
                for index, spec in enumerate(raw_chunks):
                    if not isinstance(spec, dict):
                        spec = {"facts": spec}
                    facts = [list(pair) for pair in (spec.get("facts") or [])]
                    text = str(spec.get("text") or "")
                    if not facts and text:
                        for line in text.replace(";", chr(10)).splitlines():
                            if "=" in line:
                                left, _, right = line.partition("=")
                                if left.strip() and right.strip():
                                    facts.append([left.strip(), right.strip()])
                    parsed.append(
                        {
                            "index": int(index),
                            "facts": facts,
                            "source": str(spec.get("source") or spec.get("id") or f"chunk{index}"),
                            "corrupt": bool(spec.get("corrupt", False)),
                            "fabricated": bool(spec.get("fabricated", False)),
                            "fabricated_mode": str(spec.get("fabricated_mode", "mismatch")),
                            "ood": bool(spec.get("ood", False)),
                            "raw": False,
                        }
                    )
            # The reference registry is the session's ground truth. On the raw
            # path it is the corroborated law the extractor built; on the
            # legacy path it is every fact of an honest content chunk.
            reference: Dict[str, str] = {}
            if extraction:
                reference.update(
                    {str(key): str(value) for key, value in (extraction.get("registry") or {}).items()}
                )
            else:
                for row in parsed:
                    if row["corrupt"] or row["fabricated"]:
                        continue
                    for key, value in row["facts"]:
                        reference.setdefault(str(key), str(value))
            # A caller may declare the Tier-3 reference bindings explicitly
            # (the Briefcase audit does), so a fabricated chunk can be tested
            # against a law that is not itself one of the content chunks.
            for pair in list(reference_facts or ()):
                if len(pair) >= 2:
                    reference.setdefault(str(pair[0]), str(pair[1]))
            gate = getattr(bus, "gate", None)
            if gate is not None:
                gate.fact_registry = dict(reference)
                gate.wave_of = self._identity_wave
                gate.enforce_string_rule = bool(fact_string_rule)
            caches: List[Any] = []
            chunk_rows: List[Dict[str, Any]] = []
            candidates: Dict[str, np.ndarray] = {}
            for row in parsed:
                facts = row["facts"]
                if not facts:
                    continue
                fact_waves = None
                if bool(row["corrupt"]):
                    # A corrupted state is a REAL wave with its phase scrambled:
                    # the payload still looks like a KV state, but its retrieval
                    # coherence collapses, which is what the Sagnac gate tests.
                    rng = np.random.default_rng(20261108 + int(row["index"]))
                    pairs = [
                        (self._identity_wave(str(a)), self._identity_wave(str(b)))
                        for a, b in facts
                    ]
                    topic = np.asarray(pairs[0][0], dtype=np.complex64)
                    topic = topic / max(float(np.linalg.norm(topic)), 1e-12)
                    phase = np.exp(
                        1j * rng.uniform(-np.pi, np.pi, int(self.cfg.dim))
                    ).astype(np.complex64)
                    wave = topic * phase
                    fact_waves = [(a, b) for a, b in pairs]
                elif bool(row["fabricated"]):
                    # An ON-TOPIC fabricated chunk announces a valid topic
                    # wave (frame binding 1.0) but carries a contradicted
                    # value. Two modes are exercised: a substituted value and
                    # a phase-inverted value. The gate's fourth condition must
                    # refuse both.
                    pairs = []
                    fact_waves = []
                    for key, value in facts:
                        key = str(key)
                        expected = reference.get(key)
                        key_wave = self._identity_wave(key)
                        if row["fabricated_mode"] == "anti_phase" and expected is not None:
                            value_wave = -self._identity_wave(str(expected))
                        else:
                            value_wave = self._identity_wave(str(value))
                        pairs.append((key_wave, value_wave))
                        fact_waves.append((key_wave, value_wave))
                    topic = np.asarray(self._identity_wave(str(facts[0][0])), dtype=np.complex64)
                    wave = topic / max(float(np.linalg.norm(topic)), 1e-12)
                else:
                    pairs = [
                        (self._identity_wave(str(a)), self._identity_wave(str(b)))
                        for a, b in facts
                    ]
                    wave = np.asarray(pairs[0][0], dtype=np.complex64)
                    wave = wave / max(float(np.linalg.norm(wave)), 1e-12)
                    fact_waves = [(a, b) for a, b in pairs]
                if not pairs:
                    continue
                cache = cache_from_pairs(
                    pairs,
                    source=row["source"],
                    task="language",
                    wave=np.asarray(wave, dtype=np.complex64),
                    facts=[[str(a), str(b)] for a, b in facts],
                )
                if fact_waves is not None:
                    cache.meta["fact_waves"] = fact_waves
                if bool(row.get("ood", False)):
                    # Directive R1: a document that shares no entity with any
                    # other document is off-topic for this session. The
                    # connectivity test is measured in the identity code
                    # (shared name -> coherence 1.0), and the document is
                    # refused before it can contribute a bridge slot.
                    cache.meta["ood"] = True
                caches.append(cache)
                if not row["corrupt"] and not row["fabricated"] and not row.get("ood", False):
                    for key, value in facts:
                        candidates[str(value)] = self._value_vector(
                            self._identity_wave(str(value))
                        )
                        candidates.setdefault(
                            str(key),
                            self._value_vector(self._identity_wave(str(key))),
                        )
                chunk_rows.append(
                    {
                        "source": row["source"],
                        "facts": len(facts),
                        "digest": cache.digest,
                        "fabricated": bool(row["fabricated"]),
                        "corrupt": bool(row["corrupt"]),
                        "ood": bool(row.get("ood", False)),
                    }
                )
            query_agnostic = query is None
            sid = str(session_id or f"telepathy-{len(sessions) + 1}-{int(_time.time())}")
            if reset:
                sessions.pop(sid, None)
            if query_agnostic:
                # ------------------------- Q1: ingest once, no query in hand
                ingest_started = _time.perf_counter()
                gate = getattr(bus, "gate", None)
                gate_rows: List[Dict[str, Any]] = []
                admitted_caches: List[Any] = []
                vetoed: List[str] = []
                for cache in caches:
                    if bool(cache.meta.get("ood", False)):
                        row = {
                            "admitted": False,
                            "reason": "coherence",
                            "ood": True,
                            "topic_connectivity": 0.0,
                        }
                    else:
                        row = (
                            gate.check(cache)
                            if gate is not None
                            else {"admitted": True, "reason": "gate-disabled"}
                        )
                    gate_rows.append(
                        {"source": str(cache.source), "digest": str(cache.digest), **row}
                    )
                    if bool(row.get("admitted", True)):
                        admitted_caches.append(cache)
                    else:
                        vetoed.append(str(cache.source))
                        bus.counts["vetoed"] = int(bus.counts["vetoed"]) + 1
                index = bus.build_bridge_index(
                    admitted_caches, per_chunk_u=int(per_chunk_u)
                )
                compacted = int(
                    sum(int(row["keys"].shape[0]) for row in index["per_chunk"])
                )
                digest = hashlib.sha256(
                    ("q1:" + "|".join(str(row["digest"]) for row in chunk_rows)).encode(
                        "utf-8"
                    )
                ).hexdigest()
                sessions[sid] = {
                    "agnostic": True,
                    "query": "",
                    "index": index,
                    "reference_facts": int(len(reference)),
                    "candidates": candidates,
                    "chunks": chunk_rows,
                    "gate_rows": gate_rows,
                    "vetoed": list(vetoed),
                    "digest": digest,
                    "budget_u": int(budget_u),
                    "per_chunk_u": int(per_chunk_u),
                    "hops": int(hops),
                    "namespace": str(namespace),
                }
                ingest_ms = (_time.perf_counter() - ingest_started) * 1e3
                return {
                    "mode": "ingest_chunks",
                    "session_id": sid,
                    "namespace": str(namespace),
                    "query_agnostic": True,
                    "chunks": int(len(caches)),
                    "admitted_chunks": int(len(admitted_caches)),
                    "vetoed_chunks": list(vetoed),
                    "compacted_slots_u": int(compacted),
                    "per_chunk_u": int(per_chunk_u),
                    "total_budget_u": int(budget_u),
                    "hops": int(hops),
                    "bridge_slots": int(len(index["bridge_value_names"])),
                    "cross_chunk_bridges": int(len(index["bridge_value_names"])),
                    "reference_facts": int(len(reference)),
                    "receiver_prefill_tokens": 0,
                    "swat_kv_digest": str(digest),
                    "gate_calibration": dict(getattr(self, "_lh_gate_calibration", {})),
                    "ttft_ms": float(ingest_ms),
                    "chunk_rows": chunk_rows,
                    "gate_rows": gate_rows,
                }
            # ------------------------------- P2 path: the query is in hand
            def resolver(context: np.ndarray):
                best = (-2.0, "")
                norm = float(np.linalg.norm(context))
                ranked: List[Tuple[float, str]] = []
                for name, vector in candidates.items():
                    cosine = float(
                        np.dot(context, vector)
                        / max(norm * float(np.linalg.norm(vector)), 1e-12)
                    )
                    ranked.append((cosine, name))
                    if cosine > best[0]:
                        best = (cosine, name)
                ranked.sort(key=lambda pair: -pair[0])
                self._lh_resolver_trace = ranked[:5]
                if not best[1]:
                    return None
                return self._identity_wave(best[1]), best[1]

            query_wave = self._identity_wave(str(query)) if query else None
            if query_wave is None:
                query_wave = np.asarray(caches[0].wave, dtype=np.complex64)
            started = _time.perf_counter()
            report = bus.compact_long_horizon(
                caches,
                query_wave,
                per_chunk_u=int(per_chunk_u),
                total_budget_u=int(budget_u),
                hops=int(hops),
                resolver=resolver if query else None,
                session_id=sid,
                use_gate=True,
                compose=True,
            )
            ttft_ms = (_time.perf_counter() - started) * 1e3
            names = list(candidates.keys())
            matrix = (
                np.stack([candidates[name] for name in names]).astype(np.float32)
                if names
                else np.zeros((0, 4 * 32), dtype=np.float32)
            )
            sessions[sid] = {
                "agnostic": False,
                "query": str(query or ""),
                "query_wave": query_wave,
                "candidates": candidates,
                "candidate_names": names,
                "candidate_matrix": matrix,
                "candidate_norms": np.linalg.norm(matrix, axis=1) if names else np.zeros(0),
                "cache": report["cache"],
                "chunks": chunk_rows,
                "vetoed": list(report["vetoed_chunks"]),
                "chain": list(report["chain"]),
                "digest": str(report["swat_kv_digest"]),
                "budget_u": int(budget_u),
                "compacted_slots_u": int(report["compacted_slots_u"]),
                "namespace": str(namespace),
                "hops": int(hops),
            }
            return {
                "mode": "ingest_chunks",
                "session_id": sid,
                "namespace": str(namespace),
                "query_agnostic": False,
                "chunks": int(len(caches)),
                "admitted_chunks": int(report["admitted_chunks"]),
                "vetoed_chunks": list(report["vetoed_chunks"]),
                "compacted_slots_u": int(report["compacted_slots_u"]),
                "per_chunk_u": int(per_chunk_u),
                "total_budget_u": int(budget_u),
                "hops": int(hops),
                "chain": list(report["chain"]),
                "cross_chunk_bridges": int(report.get("cross_chunk_bridges", 0)),
                "answer_label": str(report.get("answer_label", "")),
                "receiver_prefill_tokens": 0,
                "swat_kv_digest": str(report["swat_kv_digest"]),
                "gate_calibration": dict(getattr(self, "_lh_gate_calibration", {})),
                "ttft_ms": float(ttft_ms),
                "chunk_rows": chunk_rows,
                "gate_rows": list(report.get("chunks") or []),
            }
        if mode == "query":
            sid = str(session_id or "")
            session = sessions.get(sid)
            if session is None:
                raise KeyError(f"telepathy session {sid!r} is not on this service")
            started = _time.perf_counter()
            if bool(session.get("agnostic", False)):
                # Directive Q1: an unseen query is resolved against the
                # ingested bridge index and composed into the receiver cache
                # here, at query time, with zero text prefill.
                wave = self._identity_wave(str(query or ""))
                resolve = bus.resolve_bridges(
                    session["index"],
                    wave,
                    hops=int(session.get("hops", 3)),
                    wave_of=self._identity_wave,
                )
                cache = bus.compose_bridge_cache(
                    session["index"],
                    wave,
                    resolve["context"],
                    budget_u=int(session.get("budget_u", 32)),
                    source=str(sid),
                )
                ttft_ms = (_time.perf_counter() - started) * 1e3
                answer = str(resolve.get("answer", ""))
                trace = list(resolve.get("trace") or [])
                # Fail closed: a terminal hop that does not actually match the
                # indexed key (peak coherence below the declared abstention
                # floor) is not an answer. A broken cross-chunk chain ends in
                # ABSTAIN, never in the nearest unrelated slot.
                abstained = bool(
                    not trace
                    or float(trace[-1].get("peak", 0.0)) < 0.50
                    or int(resolve.get("resolved_hops", 0)) < int(session.get("hops", 3))
                )
                if abstained:
                    answer = ""
                session["last"] = {
                    "query": str(query or ""),
                    "answer": answer,
                    "cache_digest": str(cache.digest),
                    "ttft_ms": float(ttft_ms),
                }
                return {
                    "mode": "query",
                    "session_id": sid,
                    "query_agnostic": True,
                    "query": str(query or ""),
                    "answer": answer,
                    "abstained": bool(abstained),
                    "abstain_reason": (
                        "terminal bridge coherence below 0.50"
                        if abstained
                        else ""
                    ),
                    "top_k": [
                        {"name": str(row["value"]), "score": float(row["peak"])}
                        for row in trace[: max(1, int(top_k))]
                    ],
                    "score": float(trace[-1]["peak"]) if trace else 0.0,
                    "compacted_slots_u": int(cache.keys.shape[0]),
                    "bridge_slots": int(len(session["index"]["bridge_value_names"])),
                    "cross_chunk_bridges": int(max(0, len(trace) - 1)),
                    "resolve_hops": int(resolve.get("resolved_hops", 0)),
                    "receiver_prefill_tokens": 0,
                    "swat_kv_digest": str(cache.digest),
                    "session_digest": str(session.get("digest", "")),
                    "ttft_ms": float(ttft_ms),
                    "trace": trace,
                }
            wave = self._identity_wave(str(query or session["query"]))
            query_key = bus._query_keys(wave)
            cache = session["cache"]
            context, weights = bus._read_values(cache.keys, cache.values, query_key)
            matrix = session["candidate_matrix"]
            names = session["candidate_names"]
            context_np = context.numpy().astype(np.float32)
            norm = float(np.linalg.norm(context_np))
            if matrix.shape[0]:
                scores = (matrix @ context_np) / np.maximum(
                    session["candidate_norms"] * norm, 1e-12
                )
                order = np.argsort(-scores)[: max(1, int(top_k))]
                ranked = [(float(scores[index]), str(names[index])) for index in order]
            else:
                ranked = []
            ttft_ms = (_time.perf_counter() - started) * 1e3
            answer = ranked[0][1] if ranked else ""
            return {
                "mode": "query",
                "session_id": sid,
                "query_agnostic": False,
                "query": str(query or session["query"]),
                "answer": str(answer),
                "top_k": [{"name": name, "score": float(score)} for score, name in ranked[: max(1, int(top_k))]],
                "score": float(ranked[0][0]) if ranked else 0.0,
                "compacted_slots_u": int(cache.keys.shape[0]),
                "receiver_prefill_tokens": 0,
                "swat_kv_digest": str(session["digest"]),
                "ttft_ms": float(ttft_ms),
                "chain": list(session["chain"]),
            }
        raise ValueError("mode must be one of ingest_chunks or query")

    # ------------------------------------------------ planner + guard (H3)
    def _telepathy_bus(self):
        """Lazy H-SWAT bus for telepathic handoff of a plan wave (O2)."""
        if self._swat_bus is None:
            from henri.holarch.swat_bus import (
                SagnacKVGate,
                SwarmTelepathyBus,
                SwarmTelepathyMapper,
            )

            mapper = SwarmTelepathyMapper(dim=int(self.cfg.dim))
            mapper.set_shared_frame()
            self._swat_gate = SagnacKVGate(
                self.sidecar.model, self.sidecar.model.memory, task="action"
            )
            self._swat_bus = SwarmTelepathyBus(
                mapper, gate=self._swat_gate, budget_u=8, dim=int(self.cfg.dim)
            )
        return self._swat_bus

    def _plan_episode(
        self,
        observation: Any,
        history_waves: Optional[Sequence[Any]],
        transition: str,
        lookahead_depth: int,
        gamma_look: float,
        namespace: str,
        batch_rollout: bool,
        episode_id: str,
        reset_episode: bool,
        emit_swat_kv: bool,
        online_repulsor: bool = True,
    ) -> Dict[str, Any]:
        """Stateful online-REPULSOR planning for one episode (Directive O2).

        The trigger is the offline evaluator's rule: a non-adjacent revisit
        (coherence > 0.98), a dead end, or a five-step goal-distance stall.
        A second signal is the depth-1 / depth-2 disagreement that defines a
        discriminating trap; on that signal the escape search runs with a
        model-predicted next wave. The escape itself is
        PhotonicCore.relax_discovery, so a repulsor axiom is written to Zone C
        and the settled action is returned with its receipt.
        """
        from henri.holarch.model import ACTION_NAMES, LatentActionLookahead

        key = (str(namespace), str(episode_id or "default"))
        if bool(reset_episode) or key not in self._episodes:
            self._episodes[key] = {
                "waves": [],
                "actions": [],
                "distances": [],
                "repulsors": 0,
            }
        state = self._episodes[key]
        # The lookahead's cycle penalty decays by position, so the newest
        # wave must be index 0 (the order evaluate_closed_loop_action uses).
        recent_tensors = list(reversed(state["waves"][-4:])) or None
        lookahead, choice, recent = self.sidecar.plan_choice(
            observation,
            history_waves=history_waves,
            transition=str(transition),
            lookahead_depth=int(lookahead_depth),
            gamma_look=float(gamma_look),
            batch_rollout=bool(batch_rollout),
            recent_waves=recent_tensors,
        )
        psi = choice["psi"].detach().reshape(-1)
        action = int(choice["action"])
        distance = int(lookahead.goal_distance(psi))
        triggered = False
        escape_action = None
        if bool(online_repulsor):
            stalled = bool(
                len(state["distances"]) >= 5
                and min(state["distances"][-4:]) >= state["distances"][-5]
            )
            trapped = bool(
                lookahead.cycle_score(psi, state["waves"][-12:]) > 0.98
                or lookahead.dead_end(psi)
                or stalled
            )
            single = LatentActionLookahead(
                self.sidecar.model,
                None,
                rotor=getattr(self.sidecar.model, "action_rotor", None),
                transition=str(transition),
                lookahead_depth=1,
            )
            control = single.choose(observation, [])
            disagreement = int(control["action"]) != action
        else:
            trapped = False
            disagreement = False
            control = None
        if bool(online_repulsor) and (trapped or disagreement):
            if disagreement and not trapped:
                predicted = lookahead.predict(psi, int(control["action"])).detach().reshape(-1)
                basis = [predicted, psi]
                prior = control["scores"]
                distance_first = True
                avoid_reverse = True
            else:
                basis = list(state["waves"][-4:]) or list(recent)
                prior = choice["scores"]
                distance_first = False
                avoid_reverse = False
            escape = lookahead.discover_escape(
                psi,
                basis,
                choice["legal"],
                int(state["actions"][-1]) if state["actions"] else None,
                self.sidecar.model.memory,
                namespace=f"api_{namespace}_{episode_id or 'default'}",
                visited=list(state["waves"][-12:]) or list(recent),
                prior_scores=prior,
                distance_first=distance_first,
                avoid_reverse=avoid_reverse,
            )
            escape_action = int(escape["action"])
            action = escape_action
            triggered = True
            state["repulsors"] = int(state["repulsors"]) + int(
                bool((escape.get("axiom") or {}).get("registered", False))
            )
        state["waves"].append(psi)
        state["waves"] = state["waves"][-24:]
        state["actions"].append(action)
        state["actions"] = state["actions"][-24:]
        state["distances"].append(distance)
        state["distances"] = state["distances"][-24:]
        swat_digest = ""
        if bool(emit_swat_kv):
            from henri.holarch.swat_bus import cache_from_wave

            bus = self._telepathy_bus()
            cache = cache_from_wave(
                psi.detach().cpu().numpy(),
                source=f"plan:{namespace}:{episode_id or 'default'}",
                task="action",
            )
            bus.register(cache)
            packet = bus.transmit(cache.source, receiver_task="language")
            swat_digest = str(packet.digest)
        scores = choice["scores"]
        result: Dict[str, Any] = {
            "action": int(action),
            "action_name": ACTION_NAMES[int(action) % len(ACTION_NAMES)],
            "scores": [float(value) for value in scores],
            "alignment": [float(value) for value in choice["alignment"]],
            "penalty": [float(value) for value in choice["penalty"]],
            "lookahead": (
                [float(value) for value in choice["lookahead"]]
                if int(lookahead_depth) == 2
                else []
            ),
            "legal_mask": [bool(value) for value in choice["legal"]],
            "transition": str(transition),
            "lookahead_depth": int(lookahead_depth),
            "gamma_look": float(gamma_look),
            "history_waves": int(len(recent)),
            "batch_rollout": bool(batch_rollout),
            "namespace": str(namespace),
            "episode_id": str(episode_id or ""),
            "online_repulsor": bool(online_repulsor),
            "repulsor_triggered": bool(triggered),
            "escape_action": None if escape_action is None else int(escape_action),
            "active_repulsors": int(state["repulsors"]),
            "goal_distance": int(distance),
            "swat_kv_digest": swat_digest,
        }
        return result

    def plan(
        self,
        observation: Any,
        history_waves: Optional[Sequence[Any]] = None,
        transition: str = "rotor",
        lookahead_depth: int = 2,
        gamma_look: float = 0.02,
        namespace: str = "default",
        batch_rollout: bool = True,
        episode_id: Optional[str] = None,
        online_repulsor: bool = False,
        reset_episode: bool = False,
        emit_swat_kv: bool = False,
    ) -> Dict[str, Any]:
        """Depth-2 wall-masked rotor planning over the mounted checkpoint.

        Directive O2 adds an optional stateful episode: with online_repulsor
        the service keeps the visit/goal-distance trajectory per
        (namespace, episode_id), registers a REPULSOR axiom when the policy
        repeats, dead-ends or stalls, and returns the escape action plus the
        telepathic KV digest. A stateless call is unchanged and cacheable.
        """
        request = {
            "observation": _clean(observation),
            "history_waves": 0 if not history_waves else len(list(history_waves)),
            "transition": str(transition),
            "lookahead_depth": int(lookahead_depth),
            "gamma_look": float(gamma_look),
            "batch_rollout": bool(batch_rollout),
            "namespace": str(namespace),
            "episode_id": None if episode_id is None else str(episode_id),
            "online_repulsor": bool(online_repulsor),
            "reset_episode": bool(reset_episode),
            "emit_swat_kv": bool(emit_swat_kv),
        }
        stateful = bool(episode_id)
        if not stateful:
            cached = self._cache_get("/v1/holarch/plan", request)
            if cached is not None:
                return self._envelope("/v1/holarch/plan", request, cached)
        with self._read_scope():
            if stateful:
                result = self._plan_episode(
                    observation,
                    history_waves,
                    transition,
                    lookahead_depth,
                    gamma_look,
                    namespace,
                    batch_rollout,
                    str(episode_id),
                    reset_episode,
                    emit_swat_kv,
                    bool(online_repulsor),
                )
            else:
                result = self.sidecar.plan(
                    observation,
                    history_waves=history_waves,
                    transition=str(transition),
                    lookahead_depth=int(lookahead_depth),
                    gamma_look=float(gamma_look),
                    namespace=str(namespace),
                    batch_rollout=bool(batch_rollout),
                )
                self._cache_put("/v1/holarch/plan", request, result)
            return self._envelope("/v1/holarch/plan", request, result)

    def verify(
        self,
        claim: str,
        kind: str = "text",
        top_k: int = 4,
        namespace: str = "default",
    ) -> Dict[str, Any]:
        """Ontology claim verification (text) or AST/signature checks (code)."""
        request = {
            "claim": str(claim),
            "kind": str(kind),
            "top_k": int(top_k),
            "namespace": str(namespace),
        }
        cached = self._cache_get("/v1/holarch/verify", request)
        if cached is not None:
            return self._envelope("/v1/holarch/verify", request, cached)
        with self._read_scope():
            result = self.sidecar.verify_claim(
                claim, kind=str(kind), top_k=int(top_k), namespace=str(namespace)
            )
            self._cache_put("/v1/holarch/verify", request, result)
            return self._envelope("/v1/holarch/verify", request, result)

    def ingest_edges(
        self, edges: Sequence[Dict[str, str]], namespace: str = "default"
    ) -> Dict[str, Any]:
        from henri.holarch.symbolic import Edge

        request = {"edges": list(edges), "namespace": str(namespace)}
        with self.rw.write_locked():
            parsed = [
                Edge(str(edge["source"]), str(edge["operator"]), str(edge["target"]))
                for edge in edges
            ]
            result = self.sidecar.ingest_edges(parsed, namespace=str(namespace))
            self._mark_state()
            return self._envelope("/v1/holarch/ingest_edges", request, result)

    def solve(
        self, source: str, operators: Any, namespace: str = "default", mode: str = "path"
    ) -> Dict[str, Any]:
        """Resolve a multi-hop path, or a reverse-call blast radius (F4)."""
        if isinstance(operators, str):
            operators = [item for item in operators.replace(",", " ").split() if item]
        operators = list(operators)
        request = {
            "source": str(source),
            "operators": operators,
            "namespace": str(namespace),
            "mode": str(mode),
        }
        cached = self._cache_get("/v1/holarch/solve", request)
        if cached is not None:
            return self._envelope("/v1/holarch/solve", request, cached)
        with self._read_scope():
            if str(mode) == "impact":
                hops = int(operators[0]) if operators and str(operators[0]).isdigit() else 1
                result = self.sidecar.impact(source, hops=hops, namespace=str(namespace))
                self._cache_put("/v1/holarch/solve", request, result)
                return self._envelope("/v1/holarch/solve", request, result)
            result = self.sidecar.solve(source, operators, namespace=str(namespace))
            self._cache_put("/v1/holarch/solve", request, result)
            return self._envelope("/v1/holarch/solve", request, result)

    def playbook(
        self,
        name: str,
        steps: Optional[Sequence[str]] = None,
        merge: bool = True,
        namespace: str = "default",
    ) -> Dict[str, Any]:
        request = {
            "name": str(name),
            "steps": None if steps is None else list(steps),
            "merge": bool(merge),
            "namespace": str(namespace),
        }
        if steps:
            with self.rw.write_locked():
                result = self.sidecar.compile_playbook(
                    name, list(steps), merge=bool(merge), namespace=str(namespace)
                )
                self._mark_state()
                action = "compile"
        else:
            with self._read_scope():
                result = self.sidecar.recall_playbook(name, namespace=str(namespace))
                action = "recall"
        return self._envelope("/v1/holarch/playbook", {**request, "action": action}, result)

    def adapt_rule(
        self,
        task: str,
        x: Any,
        y: Sequence[int],
        rule_id: Optional[int] = None,
        lr: float = 0.2,
        namespace: str = "default",
    ) -> Dict[str, Any]:
        request = {
            "task": task,
            "x": x,
            "y": list(y),
            "rule_id": rule_id,
            "lr": float(lr),
            "namespace": str(namespace),
        }
        with self.rw.write_locked():
            tensor = self._as_tensor(task, x)
            labels = torch.as_tensor(list(y), dtype=torch.long)
            result = self.sidecar.adapt_rule(
                task, tensor, labels, rule_id=rule_id, lr=float(lr)
            )
            self._mark_state()
            return self._envelope("/v1/holarch/adapt_rule", request, result)


    # ---------------------------------------------------- composite (D4)
    @staticmethod
    def _digit_symbol(value: Any) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("empty digit operand")
        if text.lower().startswith("d"):
            text = text[1:]
        if not text.isdigit() or not 0 <= int(text) <= 9:
            raise ValueError(f"digit operand must be d0..d9, got {value!r}")
        return f"d{int(text)}"

    @staticmethod
    def _operator(value: Any) -> str:
        text = str(value).strip().lower()
        aliases = {"+": "add", "add": "add", "sum": "add", "plus": "add",
                   "-": "sub", "sub": "sub", "diff": "sub", "minus": "sub",
                   "*": "mul", "mul": "mul", "times": "mul"}
        if text not in aliases:
            raise ValueError(f"operator must be add or sub, got {value!r}")
        return aliases[text]

    def _ground_operand(self, payload: Any) -> str:
        """Stage 1: ground one 8x8 quadrant through the Pass-0 prototype bank."""
        tensor = self._as_tensor("vision", payload)
        with torch.no_grad():
            psi = self.sidecar.model.encode("vision", tensor)
            bright = self.sidecar.model.zone_b(psi).bright
            bank = self.sidecar.model.prototype_waves("vision")
            if int(bank.numel()):
                labels = getattr(self.sidecar.model, "prototype_class_vision")
                align = self.sidecar.model.zone_b.alignment(bright, bank)
                index = int(align.argmax(dim=-1)[0])
                return f"d{int(labels[index])}"
            predicted = self.sidecar.model("vision", tensor, training=False).predicted()
            return f"d{int(predicted[0])}"

    def composite_call(
        self,
        mode: str = "query",
        rules: Optional[Sequence[Dict[str, Any]]] = None,
        a: Any = None,
        b: Any = None,
        op: Any = None,
        x0: Any = None,
        x1: Any = None,
        namespace: str = "default",
        _issue_receipt: bool = True,
    ) -> Dict[str, Any]:
        """Register or resolve a composite rule (Decision D4).

        mode="ingest" stores (a, op, b) -> result rows at zero weight cost.
        mode="query" accepts digit symbols or two 8x8 quadrant payloads.
        A coherence below the calibrated threshold returns an explicit
        ABSTAIN string, never a guess.
        """
        request = {
            "mode": str(mode),
            "rules": rules,
            "a": a,
            "b": b,
            "op": op,
            "x0": x0,
            "x1": x1,
            "namespace": str(namespace),
        }
        lock = self.rw.write_locked() if str(mode) == "ingest" else self.rw.read_locked()
        with lock:
            if str(mode) == "ingest":
                if not rules:
                    raise ValueError("ingest needs at least one rule")
                rows = []
                for rule in rules:
                    operand_a = self._digit_symbol(rule["a"])
                    operand_b = self._digit_symbol(rule["b"])
                    operator = self._operator(rule["op"])
                    result = self._digit_symbol(rule["result"])
                    rows.append(((operand_a, operator, operand_b), result))
                written = self.composite.ingest(rows, namespace=str(namespace))
                calibration = self.composite.calibrate(rows, namespace=str(namespace))
                self._mark_state()
                result = {
                    "mode": "ingest",
                    "rules_ingested": float(written),
                    "atlas_entries": float(self.composite.entries),
                    "tau_composite": float(calibration["tau_coherence"]),
                }
                if not _issue_receipt:
                    return {"ok": True, "endpoint": "/v1/holarch/composite", "result": _clean(result)}
                return self._envelope("/v1/holarch/composite", request, result)

            grounded = None
            if x0 is not None or x1 is not None:
                if x0 is None or x1 is None:
                    raise ValueError("perceptual query needs both x0 and x1")
                operand_a = self._ground_operand(x0)
                operand_b = self._ground_operand(x1)
                operator = self._operator(op)
                grounded = [operand_a, operand_b, operator]
            else:
                operand_a = self._digit_symbol(a)
                operand_b = self._digit_symbol(b)
                operator = self._operator(op)
            step = self.composite.step(
                (operand_a, operator, operand_b), namespace=str(namespace)
            )
            abstained = bool(step["vetoed"]) or step["answer"] is None
            answer = (
                "ABSTAIN: coherence below tau" if abstained else str(step["answer"])
            )
            result = {
                "mode": "query",
                "result": answer,
                "abstained": abstained,
                "coherence": float(step["coherence"]),
                "tau_composite": float(self.composite.tau_coherence),
                "grounded_operands": grounded,
                "key": [operand_a, operator, operand_b],
            }
            if not _issue_receipt:
                return {"ok": True, "endpoint": "/v1/holarch/composite", "result": _clean(result)}
            return self._envelope("/v1/holarch/composite", request, result)

    # ------------------------------------------------------- synthesize (L3)
    def synthesize(
        self,
        intent: str,
        kind: str = "code",
        namespace: str = "default",
        max_tokens: int = 320,
    ) -> Dict[str, Any]:
        """Emit verified code or text, or an explicit refusal.

        The response carries the same SHA-256 receipt as every other
        endpoint, so a caller can recompute the digest from the body.
        """
        request = {
            "intent": str(intent),
            "kind": str(kind),
            "namespace": str(namespace),
            "max_tokens": int(max_tokens),
        }
        with self._read_scope():
            result = self.sidecar.synthesize(
                intent, kind=str(kind), namespace=str(namespace), max_tokens=int(max_tokens)
            )
            result = {**result, "status": str(result.get("status", ""))}
            return self._envelope("/v1/holarch/synthesize", request, result)

    def discover(
        self,
        intent: str,
        tests: str = "",
        request: Optional[Dict[str, Any]] = None,
        namespace: str = "default",
        library: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Real holonic playbook discovery over the public surface (P4).

        One tick is Ideation -> Tier-1 JEV/AST/type gate -> Tier-2 sandbox
        execution critic -> Tier-3 PlanStore compilation. A refused tick
        compiles nothing, so the store is revertible by construction; the
        envelope reports the store sizes so the caller can verify that.
        """
        engine = self.sidecar._ensure_engine(namespace)
        engine.namespace = str(namespace)
        engine.plans.namespace = str(namespace)
        body = {
            "intent": str(intent),
            "tests": str(tests),
            "request": dict(request or {}),
            "namespace": str(namespace),
            "library": None if library is None else sorted(library),
        }
        started = time.perf_counter()
        report = engine.holonic_discover(
            str(intent), str(tests), dict(request or {}), library=library
        )
        elapsed = (time.perf_counter() - started) * 1e3
        result = {
            **report,
            "namespace": str(namespace),
            "elapsed_ms": float(elapsed),
            "plan_store_size": int(len(engine.plans.plans)),
            "integrity": bool(
                int(report.get("residual_state", 0)) == 0
                and int(report.get("plans_before", 0))
                + (1 if bool(report.get("verified")) else 0)
                == int(report.get("plans_after", 0))
            ),
        }
        payload = self._envelope("/v1/holarch/discover", body, result)
        return payload

    # ------------------------------------------------- wave Turing (S2)
    def turing(
        self,
        mode: str = "list",
        program: str = "binary_increment",
        tape: Optional[Sequence[Any]] = None,
        max_steps: int = 0,
        namespace: str = "default",
        examples: Optional[Sequence[Any]] = None,
        alphabet: Optional[Sequence[str]] = None,
        name: str = "induction",
    ) -> Dict[str, Any]:
        """Execute a deterministic program in the 4-slot wave manifold (S2).

        mode="list"   the sixteen programs and their expectations
        mode="run"    one program, tape recovered from the wave
        mode="suite"  all sixteen plus the four adversarial traps, scored
        mode="traps"  the four traps alone

        Every executed step reads the tape by unbinding slot 2 and writes it by
        a linear holographic delta. No step passes through the tokenizer, so the
        receipt records text_tokens = 0.
        """
        from .symbolic import (
            WaveTuringMachine,
            reference_turing_run,
            turing_program,
            turing_programs,
            turing_traps,
        )

        mode = str(mode or "list").strip().lower()
        machine = WaveTuringMachine()
        if mode == "list":
            result = {
                "mode": mode,
                "programs": [
                    {
                        "name": item.name,
                        "tape": "".join(item.tape).strip("_"),
                        "expectation": item.expectation,
                        "rules": int(len(item.rules)),
                    }
                    for item in turing_programs()
                ],
                "count": len(turing_programs()),
                "slots": int(machine.SLOTS),
                "dim": int(machine.dim),
            }
        elif mode == "run":
            item = turing_program(program)
            if tape is not None:
                cells = [str(symbol) for symbol in tape]
                item = type(item)(
                    name=item.name,
                    tape=cells + ["_"] * 8,
                    alphabet=sorted(set(item.alphabet) | set(cells)),
                    rules=dict(item.rules),
                    start=item.start,
                    halt=item.halt,
                    expectation="".join(cells).strip("_"),
                    description=item.description,
                    head=1,
                    max_steps=int(max_steps or item.max_steps),
                )
            result = machine.run(item)
            result["reference"] = reference_turing_run(item)
            result["mode"] = mode
        elif mode in {"suite", "traps"}:
            rows: List[Dict[str, Any]] = []
            programs = [] if mode == "traps" else list(turing_programs())
            for item in programs:
                run = machine.run(item)
                reference = reference_turing_run(item)
                rows.append(
                    {
                        "name": item.name,
                        "kind": "program",
                        "halted": bool(run["halted"]),
                        "exact": bool(run["exact"]),
                        "wave_tape": run["final_tape"],
                        "reference_tape": reference.get("tape_text", ""),
                        "expected": item.expectation,
                        "steps": int(run["steps"]),
                        "step_ms_mean": float(run["step_ms_mean"]),
                        "text_tokens": int(run["text_tokens"]),
                        "atlas_rules": int(run["atlas_rules"]),
                    }
                )
            traps: List[Dict[str, Any]] = []
            for trap in turing_traps():
                run = machine.run(trap["program"], tamper=trap.get("tamper"))
                traps.append(
                    {
                        "name": trap["name"],
                        "kind": trap["kind"],
                        "vetoed": bool(run["vetoed"]),
                        "veto_reason": str(run["veto_reason"]),
                        "steps": int(run["steps"]),
                        "cycle_coherence_max": float(run["cycle_coherence_max"]),
                        "read_score_min": float(run["read_snr_min"]),
                    }
                )
            solved = sum(1 for row in rows if row["exact"] and row["halted"])
            latencies = [float(row["step_ms_mean"]) for row in rows if row["steps"] > 0]
            result = {
                "mode": mode,
                "programs": rows,
                "traps": traps,
                "solved": int(solved),
                "program_count": int(len(rows)),
                "exact_rate": float(solved / max(len(rows), 1)),
                "traps_vetoed": int(sum(1 for row in traps if row["vetoed"])),
                "trap_count": int(len(traps)),
                "false_accepts": int(sum(1 for row in traps if not row["vetoed"])),
                "step_ms_mean": float(np.mean(latencies)) if latencies else 0.0,
                "text_tokens": 0,
                "slots": int(machine.SLOTS),
                "dim": int(machine.dim),
            }
        elif mode == "induce":
            from .induction import InductionSpec, WaveProgramInducer

            pairs = []
            for item in list(examples or []):
                if isinstance(item, dict):
                    pairs.append((str(item.get("tape_in", "")), str(item.get("tape_out", ""))))
                else:
                    values = list(item or [])
                    if len(values) < 2:
                        raise ValueError("each induction example needs tape_in and tape_out")
                    pairs.append((str(values[0]), str(values[1])))
            if not pairs:
                raise ValueError("mode=induce needs at least one example")
            symbols = {str(symbol) for symbol in (alphabet or [])}
            if not symbols:
                symbols = {
                    char
                    for tape_in, tape_out in pairs
                    for char in str(tape_in) + str(tape_out)
                    if char != "_"
                }
            spec = InductionSpec(
                name=str(name or "induction"),
                alphabet=sorted(symbols),
                examples=pairs,
                max_steps=int(max_steps or 96),
            )
            inducer = WaveProgramInducer(
                memory=self.sidecar.model.memory,
                namespace=f"induce_{namespace}",
            )
            outcome = inducer.induce(spec)
            result = {"mode": mode, "examples": int(len(pairs)), **outcome.as_dict()}
            result["evidence"] = "OBSERVED"
        else:
            raise ValueError("mode must be list, run, suite, traps or induce")
        request = {
            "mode": mode,
            "program": str(program),
            "tape": None if tape is None else [str(symbol) for symbol in tape],
            "namespace": str(namespace),
            "examples": None if examples is None else [list(pair) for pair in examples],
            "alphabet": None if alphabet is None else [str(symbol) for symbol in alphabet],
            "name": str(name),
        }
        return self._envelope("/v1/holarch/turing", request, _clean(result))

    def adapt_synthesis(
        self,
        intent: str,
        corrected_source: str,
        name: str,
        kind: str = "code",
        namespace: str = "default",
    ) -> Dict[str, Any]:
        request = {
            "intent": str(intent),
            "name": str(name),
            "kind": str(kind),
            "namespace": str(namespace),
        }
        with self.rw.write_locked():
            result = self.sidecar.adapt_synthesis(
                intent, corrected_source, name, kind=str(kind), namespace=str(namespace)
            )
            self._mark_state()
            return self._envelope("/v1/holarch/adapt_rule", request, result)

    # ------------------------------------------------- agent turn (I4)
    def agent_step(
        self,
        action: str,
        payload: Optional[Dict[str, Any]] = None,
        namespace: str = "default",
    ) -> Dict[str, Any]:
        """One unified agent turn, dispatched under the service lock (I4).

        The seven actions are the sidecar's public verbs. The response is
        uniform - status, abstained, output, epistemic_receipt - and carries
        exactly ONE receipt per turn, so an external harness can treat every
        action as the same tool call and still get a contiguous hash chain.
        """
        action = str(action or "").strip().lower()
        body = dict(payload or {})
        request = {"action": action, "payload": _clean(body), "namespace": str(namespace)}
        mutating = bool(
            (action == "composite" and str(body.get("mode", "query")) == "ingest")
            or action == "discover"
        )
        lock = self.rw.write_locked() if mutating else self.rw.read_locked()
        with lock:
            if action == "verify":
                output = self.sidecar.verify_claim(
                    str(body.get("claim", "")),
                    kind=str(body.get("kind", "text")),
                    top_k=int(body.get("top_k", 4)),
                    namespace=str(namespace),
                )
                status = str(output.get("status", ""))
                abstained = not bool(output.get("supported", False))
            elif action == "solve":
                if str(body.get("mode", "path")) == "impact":
                    output = self.sidecar.impact(
                        str(body.get("target") or body.get("source", "")),
                        hops=int(body.get("hops", 1)),
                        namespace=str(namespace),
                    )
                else:
                    output = self.sidecar.solve(
                        str(body.get("source", "")),
                        body.get("operators") or [],
                        namespace=str(namespace),
                    )
                abstained = bool(output.get("vetoed", False))
                status = "abstained" if abstained else "answered"
            elif action == "impact":
                output = self.sidecar.impact(
                    str(body.get("target") or body.get("source", "")),
                    hops=int(body.get("hops", 1)),
                    namespace=str(namespace),
                )
                abstained = not bool(output.get("resolved", True))
                status = "abstained" if abstained else "ok"
            elif action == "plan":
                # Call the planner directly: agent_step issues exactly one
                # receipt for the turn, so an inner envelope would break the
                # contiguous 76-query chain.
                if body.get("episode_id"):
                    output = self._plan_episode(
                        body.get("observation"),
                        body.get("history_waves"),
                        str(body.get("transition", "rotor")),
                        int(body.get("lookahead_depth", 2)),
                        float(body.get("gamma_look", 0.02)),
                        str(namespace),
                        True,
                        str(body.get("episode_id")),
                        bool(body.get("reset_episode", False)),
                        bool(body.get("emit_swat_kv", False)),
                        bool(body.get("online_repulsor", False)),
                    )
                else:
                    output = self.sidecar.plan(
                        body.get("observation"),
                        history_waves=body.get("history_waves"),
                        transition=str(body.get("transition", "rotor")),
                        lookahead_depth=int(body.get("lookahead_depth", 2)),
                        gamma_look=float(body.get("gamma_look", 0.02)),
                        namespace=str(namespace),
                    )
                abstained = False
                status = "ok"
            elif action == "synthesize":
                output = self.sidecar.synthesize(
                    str(body.get("intent", "")),
                    kind=str(body.get("kind", "code")),
                    namespace=str(namespace),
                    max_tokens=int(body.get("max_tokens", 320)),
                )
                status = str(output.get("status", ""))
                abstained = status == "refused"
            elif action == "composite":
                inner = self.composite_call(
                    mode=str(body.get("mode", "query")),
                    rules=body.get("rules"),
                    a=body.get("a"),
                    b=body.get("b"),
                    op=body.get("op"),
                    x0=body.get("x0"),
                    x1=body.get("x1"),
                    namespace=str(namespace),
                    _issue_receipt=False,
                )
                output = dict(inner["result"])
                abstained = bool(output.get("abstained", False))
                status = "abstained" if abstained else "ok"
            elif action == "veto":
                task = str(body.get("task", "vision"))
                tensor = self._as_tensor(task, body.get("x"))
                output = self.sidecar.veto_check(task, tensor)
                abstained = False
                status = "ok"
            elif action == "telepathy":
                # Directive P2: the same two modes the REST route serves.
                output = self.telepathy(
                    mode=str(body.get("mode", "ingest_chunks")),
                    session_id=body.get("session_id"),
                    namespace=str(namespace),
                    chunks=body.get("chunks"),
                    query=body.get("query"),
                    budget_u=int(body.get("budget_u", 32)),
                    per_chunk_u=int(body.get("per_chunk_u", 8)),
                    hops=int(body.get("hops", 3)),
                    top_k=int(body.get("top_k", 1)),
                    reset=bool(body.get("reset", False)),
                )
                if str(body.get("mode", "ingest_chunks")).startswith("query"):
                    abstained = not bool(output.get("answer"))
                    status = "answered" if not abstained else "abstained"
                else:
                    abstained = int(output.get("admitted_chunks", 0)) == 0
                    status = "ok" if not abstained else "abstained"
            elif action == "turing":
                # Directive S2: one wave-space program execution per turn. The
                # inner call is issued without its own receipt so the turn keeps
                # exactly one link in the chain.
                output = self.turing(
                    mode=str(body.get("mode", "run")),
                    program=str(body.get("program", "binary_increment")),
                    tape=body.get("tape"),
                    max_steps=int(body.get("max_steps", 0)),
                    namespace=str(namespace),
                    examples=body.get("examples"),
                    alphabet=body.get("alphabet"),
                    name=str(body.get("name", "induction")),
                )
                inner = dict(output.get("result") or {})
                output = inner
                abstained = bool(inner.get("vetoed", False)) or (
                    str(body.get("mode", "run")) == "run" and not inner.get("halted", False)
                )
                status = "abstained" if abstained else "ok"
            elif action == "discover":
                # Directive P4: a real discovery tick, no envelope inside the
                # turn (agent_step issues exactly one receipt).
                engine = self.sidecar._ensure_engine(str(namespace))
                engine.namespace = str(namespace)
                engine.plans.namespace = str(namespace)
                output = engine.holonic_discover(
                    str(body.get("intent", "")),
                    str(body.get("tests", "")),
                    dict(body.get("request") or {}),
                    library=body.get("library"),
                )
                abstained = not bool(output.get("verified", False))
                status = "accepted" if not abstained else "refused"
            else:
                raise ValueError(
                    "action must be one of verify, solve, impact, plan, "
                    "synthesize, composite, veto, telepathy, discover, turing"
                )
            result = {
                "status": status,
                "abstained": bool(abstained),
                "output": _clean(output),
            }
            receipt = self.ledger.issue(
                "/v1/holarch/agent_step", request, result, config_digest=self.config_digest
            )
            receipt["checkpoint_sha256"] = str(
                self.checkpoint_info.get("sha256", "") or ""
            )
            return {
                "ok": True,
                "endpoint": "/v1/holarch/agent_step",
                "status": status,
                "abstained": bool(abstained),
                "output": _clean(output),
                "epistemic_receipt": receipt,
            }

    # --------------------------------------------------- chat adapter (J4)
    def _chat_payload(
        self, action: str, text: str, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        raw = metadata.get("payload")
        payload: Dict[str, Any] = dict(raw) if isinstance(raw, dict) else {}
        if action == "verify":
            payload.setdefault("claim", text)
            if not payload.get("kind"):
                payload["kind"] = "code" if looks_like_code(text) else "text"
        elif action == "impact":
            payload.setdefault("target", extract_chat_target(text))
            payload.setdefault("hops", 1)
        elif action == "plan":
            payload.setdefault("observation", [0.0] * int(getattr(self.cfg, "obs_dim", 79)))
            payload.setdefault("transition", "shift")
        elif action == "solve":
            payload.setdefault("source", extract_chat_target(text))
            payload.setdefault("operators", [])
        elif action == "synthesize":
            payload.setdefault("intent", text)
            payload.setdefault("kind", "code")
        elif action == "composite":
            payload.setdefault("mode", "query")
        elif action == "veto":
            payload.setdefault("task", "vision")
        elif action == "turing":
            payload.setdefault("mode", "run")
            payload.setdefault("program", "binary_increment")
        return payload

    def chat_completion(
        self,
        model: str = "henri-holarch",
        messages: Optional[Sequence[Any]] = None,
        tools: Optional[Sequence[Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """One OpenAI-compatible turn over one epistemic agent_step.

        The action is chosen deterministically, routed through agent_step,
        and reported twice: once inside the agent envelope and once in the
        OpenAI-shaped body. The chat response carries its own receipt, whose
        parent is the agent receipt when an agent turn ran.
        """
        message_list = list(messages or [])
        meta = dict(metadata) if isinstance(metadata, dict) else {}
        namespace = str(meta.get("namespace") or "default")
        action = select_chat_action(message_list, tools, meta)
        text = chat_text(message_list)
        payload = self._chat_payload(action, text, meta)
        status = "ok"
        abstained = False
        output: Any = None
        agent_receipt: Dict[str, Any] = {}
        error = ""
        if action == "composite" and not any(
            payload.get(key) is not None for key in ("a", "b", "x0", "x1")
        ):
            error = "composite needs a, b and op in metadata.payload"
        else:
            try:
                envelope = self.agent_step(action, payload, namespace)
            except (ValueError, KeyError, TypeError) as exc:
                error = str(exc)
            else:
                status = str(envelope.get("status", ""))
                abstained = bool(envelope.get("abstained", False))
                output = envelope.get("output")
                agent_receipt = dict(envelope.get("epistemic_receipt") or {})
        if error:
            status, abstained = "error", True
            output = {"error": error, "action": action}
        refused = bool(abstained) or status in _CHAT_REFUSAL_STATUSES
        content = chat_content(action, output, refused)
        prompt_tokens = sum(
            len(chat_message_text(_chat_message_field(message, "content", "")).split())
            for message in message_list
        )
        completion_tokens = len(content.split())
        receipt = self.ledger.issue(
            "/v1/chat/completions",
            {
                "model": str(model),
                "action": action,
                "namespace": namespace,
                "messages": [
                    {
                        "role": str(_chat_message_field(message, "role", "user")),
                        "content": chat_message_text(
                            _chat_message_field(message, "content", "")
                        ),
                    }
                    for message in message_list
                ],
            },
            {
                "action": action,
                "content": content,
                "status": status,
                "abstained": bool(refused),
                "namespace": namespace,
                "agent_receipt": str(agent_receipt.get("sha256", "") or ""),
            },
            config_digest=self.config_digest,
        )
        return {
            "id": "chatcmpl-" + str(receipt["sha256"])[:24],
            "object": "chat.completion",
            "created": int(time.time()),
            "model": str(model),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "total_tokens": int(prompt_tokens + completion_tokens),
            },
            "epistemic_receipt": receipt,
            "holarch": {
                "action": action,
                "status": status,
                "abstained": bool(refused),
                "result": _clean(output),
                "agent_receipt": agent_receipt or None,
                "checkpoint_sha256": str(self.checkpoint_info.get("sha256", "") or ""),
            },
        }

    def export_import_bundle(
        self, src_ns: str = "default", dst_ns: str = "imported"
    ) -> Dict[str, Any]:
        """Copy one tenant's vocabulary ledger and plan library to another.

        Directive K4. The bundle carries the exact identifier order and the
        Tier-2 address map, so the destination namespace can answer with the
        source tenant's token ids without re-tokenizing a string. The
        rebase id map is produced by rebase_tenant_vocabulary when the caller
        holds two live tokenizers; this method moves the durable records.
        """
        from henri.holarch.code_lang import VOCAB_LEDGER_KIND

        memory = getattr(self.sidecar.model, "memory", None)
        if memory is None:
            raise ValueError("the service has no Zone C memory")
        source = str(src_ns or "default")
        dest = str(dst_ns or "imported")
        ledger = memory.get_meta(source, VOCAB_LEDGER_KIND, "vocab_ledger")
        tier2 = memory.get_meta(source, VOCAB_LEDGER_KIND, "vocab_ledger_tier2")
        plans = memory.get_meta(source, "plan")
        written = 0
        if ledger:
            memory.put_meta(dest, VOCAB_LEDGER_KIND, "vocab_ledger", dict(ledger[-1][1]))
            written += 1
        if tier2:
            memory.put_meta(dest, VOCAB_LEDGER_KIND, "vocab_ledger_tier2", dict(tier2[-1][1]))
            written += 1
        for key, payload in plans:
            memory.put_meta(dest, "plan", str(key), dict(payload))
            written += 1
        tier2_ids = list(dict(tier2[-1][1]).get("tier2_ids", [])) if tier2 else []
        result = {
            "source": source,
            "destination": dest,
            "ledger_records": len(ledger),
            "tier2_records": len(tier2),
            "plan_records": len(plans),
            "tier2_ids": len(tier2_ids),
            "written": int(written),
            "ledger_digest": str(
                dict(ledger[-1][1]).get("digest", "") if ledger else ""
            ),
            "plans": [str(key) for key, _payload in plans],
        }
        self._mark_state()
        return self._envelope("/v1/holarch/bundle", {"src": source, "dst": dest}, result)

    def health(self) -> Dict[str, Any]:
        with self._read_scope():
            rotor = getattr(self.sidecar.model, "action_rotor", None)
            ledger_state = self.ledger.snapshot()
            result = {
                "status": "ok",
                "config_digest": self.config_digest,
                "parameters": float(self.sidecar.model.parameter_count()),
                "receipts": float(ledger_state["count"]),
                "chain_head": ledger_state["head"],
                "chain_valid": bool(ledger_state["chain_valid"]),
                "checkpoint": dict(self.checkpoint_info),
                "rotor": dict(getattr(rotor, "state", lambda: {})())
                if rotor is not None
                else None,
                "lock": self.rw.stats(),
                "rate_limit": self.limiter.stats(),
            }
            return self._envelope("/v1/holarch/health", {}, result)

    def close(self) -> None:
        try:
            self.sidecar.close()
        finally:
            if self._pinned_threads:
                torch.set_num_threads(self._saved_threads)
                self._pinned_threads = False


# --------------------------------------------------------------------- REST
# The request models are module level on purpose: this module uses
# postponed annotations, and FastAPI cannot resolve a model that only
# exists in a function's local scope. A local model silently degrades to a
# query parameter, which is a 422 at the first call.
from pydantic import BaseModel, Field  # noqa: E402  (transport-only import)


class VetoRequest(BaseModel):
    task: str = "vision"
    x: Any = Field(..., description="nested pixel or observation payload")
    rule_id: Optional[int] = None
    namespace: str = "default"


class IngestCodeRequest(BaseModel):
    root: str
    package: str = "henri"
    namespace: str = "default"
    mode: str = "edges"
    file_path: Optional[str] = Field(None, description="delta mode: the one file to re-index")
    source_text: Optional[str] = Field(None, description="delta mode: the new source text")


class IngestEdgesRequest(BaseModel):
    edges: List[Dict[str, str]]
    namespace: str = "default"


class SolveRequest(BaseModel):
    source: str
    operators: Any
    namespace: str = "default"
    mode: str = "path"


class PlaybookRequest(BaseModel):
    name: str
    steps: Optional[List[str]] = None
    merge: bool = True
    namespace: str = "default"


class AdaptRuleRequest(BaseModel):
    task: str
    x: Any
    y: List[int]
    rule_id: Optional[int] = None
    lr: float = 0.2
    namespace: str = "default"


class SynthesizeRequest(BaseModel):
    intent: str
    kind: str = "code"
    namespace: str = "default"
    max_tokens: int = 320


class TelepathyRequest(BaseModel):
    mode: str = "ingest_chunks"
    session_id: Optional[str] = None
    namespace: str = "default"
    chunks: Optional[List[Any]] = None
    # Directive R1: raw unstructured documents (Markdown, prose, config,
    # Python) are parsed by DocumentFactExtractor when no fact list is given.
    raw_documents: Optional[List[Any]] = None
    query: Optional[str] = None
    budget_u: int = 32
    per_chunk_u: int = 8
    hops: int = 3
    top_k: int = 1
    reset: bool = False


class TuringRequest(BaseModel):
    mode: str = Field("list", description="list, run, suite, traps or induce")
    program: str = Field("binary_increment", description="the program name for mode=run")
    tape: Optional[List[Any]] = Field(None, description="optional initial tape cells")
    max_steps: int = 0
    namespace: str = "default"
    examples: Optional[List[Any]] = Field(
        None, description="mode=induce: input/output tape pairs, no rules"
    )
    alphabet: Optional[List[str]] = Field(None, description="mode=induce: optional symbols")
    name: str = Field("induction", description="mode=induce: task name")


class DiscoverRequest(BaseModel):
    intent: str
    tests: str = ""
    request: Dict[str, Any] = Field(default_factory=dict)
    namespace: str = "default"
    library: Optional[Dict[str, str]] = None


class PlanRequest(BaseModel):
    observation: Any = Field(..., description="the 79-value sensor observation")
    history_waves: Optional[List[Any]] = Field(
        None, description="optional recent waves, 2*D interleaved or D real values each"
    )
    transition: str = "rotor"
    lookahead_depth: int = 2
    gamma_look: float = 0.02
    batch_rollout: bool = True
    namespace: str = "default"
    # Directive O2: the stateful online-REPULSOR surface.
    episode_id: Optional[str] = Field(None, description="one trajectory; reset to start over")
    online_repulsor: bool = False
    reset_episode: bool = False
    emit_swat_kv: bool = False


class VerifyRequest(BaseModel):
    claim: str
    kind: str = Field("text", description="text or code")
    top_k: int = 4
    namespace: str = "default"


class CompositeRule(BaseModel):
    a: Any
    op: Any
    b: Any
    result: Any


class AgentStepRequest(BaseModel):
    action: str = Field(..., description="verify|solve|impact|plan|synthesize|composite|veto")
    payload: Dict[str, Any] = Field(default_factory=dict)
    namespace: str = "default"


class CompositeRequest(BaseModel):
    mode: str = Field("query", description="ingest or query")
    rules: Optional[List[CompositeRule]] = None
    a: Any = None
    b: Any = None
    op: Any = None
    x0: Any = None
    x1: Any = None
    namespace: str = "default"


class ChatMessage(BaseModel):
    model_config = {"extra": "allow"}

    role: str = "user"
    content: Any = ""


class ChatCompletionRequest(BaseModel):
    """The subset of the OpenAI chat request this sidecar can honour."""

    model_config = {"extra": "allow"}

    model: str = "henri-holarch"
    messages: List[ChatMessage] = Field(default_factory=list)
    tools: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None
    stream: bool = False
    """Directive Z4: true asks for text/event-stream delta chunks."""


_UNSET: Any = object()


def _authorized(header: Any, keys: Set[str]) -> bool:
    """True when the header carries one of the configured Bearer tokens."""
    if not keys:
        return True
    text = str(header or "").strip()
    if not text:
        return False
    scheme, _, token = text.partition(" ")
    if scheme.lower() != "bearer":
        return False
    token = token.strip()
    if not token:
        return False
    return any(hmac.compare_digest(token, str(key)) for key in keys)


def _env_api_keys() -> Set[str]:
    """HENRI_API_KEYS (comma separated) plus HENRI_API_KEY, read once."""
    keys: Set[str] = set()
    for part in str(os.environ.get("HENRI_API_KEYS", "")).replace("\n", ",").split(","):
        if part.strip():
            keys.add(part.strip())
    single = str(os.environ.get("HENRI_API_KEY", "")).strip()
    if single:
        keys.add(single)
    return keys


def _namespace_from_body(body: bytes) -> str:
    """The bucket key: the request namespace, else the chat metadata one."""
    if not body:
        return "default"
    try:
        payload = json.loads(body)
    except Exception:
        return "default"
    if not isinstance(payload, dict):
        return "default"
    namespace = payload.get("namespace")
    if namespace:
        return str(namespace)
    metadata = payload.get("metadata")
    if isinstance(metadata, dict) and metadata.get("namespace"):
        return str(metadata["namespace"])
    return "default"


async def _send_json(send: Any, status: int, payload: Dict[str, Any]) -> None:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": int(status),
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("latin-1")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body, "more_body": False})


class EpistemicGateMiddleware:
    """Bearer auth and the per-namespace token bucket, ahead of routing.

    /healthz is public and unlimited. Every /v1/* request is checked for a
    key (when any are configured) and then charged to its namespace bucket,
    so a refused call never reaches a model. The body is read once and
    replayed to the router, which is how the namespace is known before the
    endpoint's request model has parsed it.
    """

    def __init__(
        self,
        app: Any,
        *,
        api_keys: Optional[Set[str]] = None,
        limiter: Optional[RateLimiter] = None,
    ) -> None:
        self.app = app
        self.api_keys = {str(key) for key in (api_keys or ())}
        self.limiter = limiter if limiter is not None else RateLimiter(rps=None)

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        path = str(scope.get("path", "") or "")
        if not path.startswith("/v1/"):
            await self.app(scope, receive, send)
            return
        headers = {
            bytes(name).decode("latin-1").lower(): bytes(value).decode("latin-1")
            for name, value in (scope.get("headers") or [])
        }
        if self.api_keys and not _authorized(headers.get("authorization"), self.api_keys):
            await _send_json(send, 401, {"ok": False, "error": "unauthorized"})
            return
        body = b""
        more = True
        while more:
            message = await receive()
            if message.get("type") != "http.request":
                await _send_json(send, 400, {"ok": False, "error": "bad_request"})
                return
            body += message.get("body", b"") or b""
            more = bool(message.get("more_body", False))
        if self.limiter.enabled and not self.limiter.take(_namespace_from_body(body)):
            await _send_json(send, 429, {"ok": False, "error": "rate_limited"})
            return

        replayed = False

        async def replay() -> Dict[str, Any]:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            # Directive Z4: a streaming response keeps a listener on receive
            # and stops the stream when it sees http.disconnect. Delegating the
            # later reads to the real channel keeps that listener honest
            # instead of handing it a synthetic disconnect before the body has
            # been delivered. A blocking response never reads receive twice,
            # so the non-stream path is untouched.
            return await receive()

        await self.app(scope, replay, send)


def build_app(
    service: Optional[SidecarService] = None,
    *,
    api_keys: Optional[Set[str]] = None,
    rate_limit_rps: Any = _UNSET,
    rate_limit_burst: Any = _UNSET,
    clock: Optional[Callable[[], float]] = None,
):
    """Build the FastAPI app. Imported lazily so the core stays light.

    Auth and rate limiting are configuration, not new routes. Pass api_keys
    (or set HENRI_API_KEYS / HENRI_API_KEY) to require a Bearer token, and
    rate_limit_rps / rate_limit_burst to shape the per-namespace bucket;
    rate_limit_rps=None disables the bucket. _UNSET keeps the service's own
    values, so an already-configured service is never silently overridden.
    """
    from fastapi import FastAPI, HTTPException

    if service is None:
        service = SidecarService(
            api_keys=api_keys,
            rate_limit_rps=50.0 if rate_limit_rps is _UNSET else rate_limit_rps,
            rate_limit_burst=100.0 if rate_limit_burst is _UNSET else rate_limit_burst,
            clock=clock,
        )
    elif rate_limit_rps is not _UNSET or rate_limit_burst is not _UNSET or clock is not None:
        service.clock = clock or service.clock
        service.limiter = RateLimiter(
            rps=service.limiter.rps if rate_limit_rps is _UNSET else rate_limit_rps,
            burst=service.limiter.burst if rate_limit_burst is _UNSET else rate_limit_burst,
            clock=service.clock,
        )

    configured_keys: Set[str] = {str(key) for key in (api_keys or ()) if str(key)}
    if not configured_keys:
        configured_keys = set(service.api_keys)
    if not configured_keys:
        configured_keys = _env_api_keys()

    app = FastAPI(
        title="HENRI holarch epistemic sidecar",
        version="1.0",
        description="Fail-closed veto, symbolic solve, playbook, and rule adaptation, each with a SHA-256 receipt.",
    )
    app.add_middleware(
        EpistemicGateMiddleware, api_keys=configured_keys, limiter=service.limiter
    )
    app.state.epistemic_service = service
    app.state.api_keys = configured_keys
    app.state.rate_limiter = service.limiter

    def get_service() -> SidecarService:
        nonlocal service
        if service is None:
            service = SidecarService()
        return service

    @app.get("/healthz")
    def healthz():
        return get_service().health()

    @app.post("/v1/holarch/veto")
    def veto(request: VetoRequest):
        return get_service().veto(request.task, request.x, request.rule_id, request.namespace)

    @app.post("/v1/holarch/ingest_code")
    def ingest_code(request: IngestCodeRequest):
        try:
            return get_service().ingest_code(
                request.root,
                request.package,
                request.namespace,
                request.mode,
                request.file_path,
                request.source_text,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/v1/holarch/plan")
    def plan(request: PlanRequest):
        try:
            return get_service().plan(
                request.observation,
                request.history_waves,
                request.transition,
                request.lookahead_depth,
                request.gamma_look,
                request.namespace,
                request.batch_rollout,
                request.episode_id,
                request.online_repulsor,
                request.reset_episode,
                request.emit_swat_kv,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/v1/holarch/verify")
    def verify(request: VerifyRequest):
        return get_service().verify(
            request.claim, request.kind, request.top_k, request.namespace
        )

    @app.post("/v1/holarch/ingest_edges")
    def ingest_edges(request: IngestEdgesRequest):
        return get_service().ingest_edges(request.edges, request.namespace)

    @app.post("/v1/holarch/solve")
    def solve(request: SolveRequest):
        return get_service().solve(
            request.source, request.operators, request.namespace, request.mode
        )

    @app.post("/v1/holarch/synthesize")
    def synthesize(request: SynthesizeRequest):
        return get_service().synthesize(
            request.intent, request.kind, request.namespace, request.max_tokens
        )

    @app.post("/v1/holarch/telepathy")
    def telepathy(request: TelepathyRequest):
        try:
            return get_service().telepathy(
                mode=request.mode,
                session_id=request.session_id,
                namespace=request.namespace,
                chunks=request.chunks,
                query=request.query,
                budget_u=request.budget_u,
                per_chunk_u=request.per_chunk_u,
                hops=request.hops,
                top_k=request.top_k,
                reset=request.reset,
                raw_documents=request.raw_documents,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/v1/holarch/turing")
    def turing(request: TuringRequest):
        try:
            return get_service().turing(
                request.mode,
                request.program,
                request.tape,
                request.max_steps,
                request.namespace,
                request.examples,
                request.alphabet,
                request.name,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/v1/holarch/discover")
    def discover(request: DiscoverRequest):
        try:
            return get_service().discover(
                request.intent,
                request.tests,
                request.request,
                request.namespace,
                request.library,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/v1/holarch/playbook")
    def playbook(request: PlaybookRequest):
        return get_service().playbook(
            request.name, request.steps, request.merge, request.namespace
        )

    @app.post("/v1/holarch/adapt_rule")
    def adapt_rule(request: AdaptRuleRequest):
        return get_service().adapt_rule(
            request.task, request.x, request.y, request.rule_id, request.lr, request.namespace
        )

    @app.post("/v1/holarch/composite")
    def composite(request: CompositeRequest):
        rules = None if request.rules is None else [rule.model_dump() for rule in request.rules]
        try:
            return get_service().composite_call(
                mode=request.mode,
                rules=rules,
                a=request.a,
                b=request.b,
                op=request.op,
                x0=request.x0,
                x1=request.x1,
                namespace=request.namespace,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/v1/holarch/agent_step")
    def agent_step(request: AgentStepRequest):
        try:
            return get_service().agent_step(
                request.action, request.payload, request.namespace
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest):
        from fastapi.responses import StreamingResponse

        payload = get_service().chat_completion(
            request.model,
            [message.model_dump() for message in request.messages],
            request.tools,
            request.metadata,
        )
        if not bool(request.stream):
            return payload
        lines = chat_stream_chunks(payload)

        def events():
            for line in lines:
                yield line + "\n\n"

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
        )

    @app.get("/v1/holarch/receipt")
    def receipts(limit: int = 16):
        ledger = get_service().ledger
        state = ledger.snapshot()
        return {
            "count": state["count"],
            "head": state["head"],
            "chain_valid": state["chain_valid"],
            "receipts": ledger.receipts[-max(1, int(limit)):],
        }

    return app


def serve_rest(host: str = "127.0.0.1", port: int = 8080, **kwargs) -> None:
    """Run the REST service with uvicorn."""
    import uvicorn

    app = build_app(SidecarService(**kwargs))
    uvicorn.run(app, host=str(host), port=int(port), log_level="info")
