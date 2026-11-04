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
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

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


@dataclass
class ReceiptLedger:
    """Append-only hash chain of the receipts this process issued."""

    parent: str = GENESIS
    count: int = 0
    receipts: List[Dict[str, Any]] = field(default_factory=list)
    max_retained: int = 256

    def issue(self, endpoint: str, request: Any, result: Any, config_digest: str = "") -> Dict[str, Any]:
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
        return self.parent

    def verify_chain(self) -> bool:
        # Retention may have dropped the genesis receipt; the oldest retained
        # one still names the parent it was issued against.
        parent = self.receipts[0].get("parent", GENESIS) if self.receipts else GENESIS
        for receipt in self.receipts:
            if receipt.get("parent") != parent:
                return False
            if not verify_receipt(receipt):
                return False
            parent = receipt["sha256"]
        return True


class SidecarService:
    """One sidecar, one lock, one receipt chain.

    The lock serialises calls because a torch module is not thread safe.
    Every public method returns the uniform envelope

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
        self.lock = threading.RLock()
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
        with self.lock:
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
        self, root: str, package: str = "henri", namespace: str = "default", mode: str = "edges"
    ) -> Dict[str, Any]:
        request = {
            "root": str(root),
            "package": str(package),
            "namespace": str(namespace),
            "mode": str(mode),
        }
        with self.lock:
            result = self.sidecar.ingest_code(root, package=package, namespace=str(namespace))
            result["mode"] = str(mode)
            return self._envelope("/v1/holarch/ingest_code", request, result)

    def ingest_edges(
        self, edges: Sequence[Dict[str, str]], namespace: str = "default"
    ) -> Dict[str, Any]:
        from henri.holarch.symbolic import Edge

        request = {"edges": list(edges), "namespace": str(namespace)}
        with self.lock:
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
        with self.lock:
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
        with self.lock:
            if steps:
                result = self.sidecar.compile_playbook(
                    name, list(steps), merge=bool(merge), namespace=str(namespace)
                )
                action = "compile"
            else:
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
        with self.lock:
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
        with self.lock:
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
        with self.lock:
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
        with self.lock:
            result = self.sidecar.adapt_synthesis(
                intent, corrected_source, name, kind=str(kind), namespace=str(namespace)
            )
            return self._envelope("/v1/holarch/adapt_rule", request, result)

    def health(self) -> Dict[str, Any]:
        with self.lock:
            result = {
                "status": "ok",
                "config_digest": self.config_digest,
                "parameters": float(self.sidecar.model.parameter_count()),
                "receipts": float(self.ledger.count),
                "chain_head": self.ledger.head(),
                "chain_valid": bool(self.ledger.verify_chain()),
                "checkpoint": dict(self.checkpoint_info),
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


class CompositeRule(BaseModel):
    a: Any
    op: Any
    b: Any
    result: Any


class CompositeRequest(BaseModel):
    mode: str = Field("query", description="ingest or query")
    rules: Optional[List[CompositeRule]] = None
    a: Any = None
    b: Any = None
    op: Any = None
    x0: Any = None
    x1: Any = None
    namespace: str = "default"


def build_app(service: Optional[SidecarService] = None):
    """Build the FastAPI app. Imported lazily so the core stays light."""
    from fastapi import FastAPI, HTTPException

    app = FastAPI(
        title="HENRI holarch epistemic sidecar",
        version="1.0",
        description="Fail-closed veto, symbolic solve, playbook, and rule adaptation, each with a SHA-256 receipt.",
    )

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
        return get_service().ingest_code(
            request.root, request.package, request.namespace, request.mode
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

    @app.get("/v1/holarch/receipt")
    def receipts(limit: int = 16):
        ledger = get_service().ledger
        return {
            "count": ledger.count,
            "head": ledger.head(),
            "chain_valid": ledger.verify_chain(),
            "receipts": ledger.receipts[-max(1, int(limit)):],
        }

    return app


def serve_rest(host: str = "127.0.0.1", port: int = 8080, **kwargs) -> None:
    """Run the REST service with uvicorn."""
    import uvicorn

    app = build_app(SidecarService(**kwargs))
    uvicorn.run(app, host=str(host), port=int(port), log_level="info")
