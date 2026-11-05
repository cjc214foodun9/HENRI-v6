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
        self.api_keys: Set[str] = {str(key) for key in (api_keys or ()) if str(key)}
        self.clock: Callable[[], float] = clock or time.monotonic
        self.limiter = RateLimiter(
            rps=rate_limit_rps, burst=rate_limit_burst, clock=self.clock
        )
        self.config_digest = str(self.cfg.digest())
        # Decision D4: the composite atlas is attached to the service, so a
        # REST or MCP caller can register and resolve multi-arity rules.
        self.composite = CompositeSolver(self.sidecar.model.memory, self.sidecar.codebook)

    # ------------------------------------------------------------- plumbing
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
        with self.rw.read_locked():
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
            return self._envelope("/v1/holarch/ingest_code", request, result)

    # ------------------------------------------------ planner + guard (H3)
    def plan(
        self,
        observation: Any,
        history_waves: Optional[Sequence[Any]] = None,
        transition: str = "rotor",
        lookahead_depth: int = 2,
        gamma_look: float = 0.02,
        namespace: str = "default",
        batch_rollout: bool = True,
    ) -> Dict[str, Any]:
        """Depth-2 wall-masked rotor planning over the mounted checkpoint."""
        request = {
            "observation": _clean(observation),
            "history_waves": 0 if not history_waves else len(list(history_waves)),
            "transition": str(transition),
            "lookahead_depth": int(lookahead_depth),
            "gamma_look": float(gamma_look),
            "batch_rollout": bool(batch_rollout),
            "namespace": str(namespace),
        }
        with self.rw.read_locked():
            result = self.sidecar.plan(
                observation,
                history_waves=history_waves,
                transition=str(transition),
                lookahead_depth=int(lookahead_depth),
                gamma_look=float(gamma_look),
                namespace=str(namespace),
                batch_rollout=bool(batch_rollout),
            )
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
        with self.rw.read_locked():
            result = self.sidecar.verify_claim(
                claim, kind=str(kind), top_k=int(top_k), namespace=str(namespace)
            )
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
        with self.rw.read_locked():
            if str(mode) == "impact":
                hops = int(operators[0]) if operators and str(operators[0]).isdigit() else 1
                result = self.sidecar.impact(source, hops=hops, namespace=str(namespace))
                return self._envelope("/v1/holarch/solve", request, result)
            result = self.sidecar.solve(source, operators, namespace=str(namespace))
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
                action = "compile"
        else:
            with self.rw.read_locked():
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
        with self.rw.read_locked():
            result = self.sidecar.synthesize(
                intent, kind=str(kind), namespace=str(namespace), max_tokens=int(max_tokens)
            )
            result = {**result, "status": str(result.get("status", ""))}
            return self._envelope("/v1/holarch/synthesize", request, result)

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
        mutating = action == "composite" and str(body.get("mode", "query")) == "ingest"
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
            else:
                raise ValueError(
                    "action must be one of verify, solve, impact, plan, "
                    "synthesize, composite, veto"
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
        return self._envelope("/v1/holarch/bundle", {"src": source, "dst": dest}, result)

    def health(self) -> Dict[str, Any]:
        with self.rw.read_locked():
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
        self.sidecar.close()


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
            return {"type": "http.disconnect"}

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
        return get_service().chat_completion(
            request.model,
            [message.model_dump() for message in request.messages],
            request.tools,
            request.metadata,
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
