"""FastAPI daemon over the verified Digital Twin API.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
It wraps `henri.api.DigitalTwinAPI` in an async-capable FastAPI app with a
per-namespace cartridge registry.

It is NOT authenticated multi-tenancy. `tenant_id` is a NAMESPACE KEY, not an
verified identity. There is no auth, no rate limit, no TLS, and no isolation
between namespaces beyond separate bank objects in one process. Anyone who
can reach the port can write to any namespace. Do not expose it publicly
without a reverse proxy that supplies those controls.

Endpoints:
    GET  /v1/health/seal       -> kernel seal status
    POST /v1/reason/rank       -> rank_options receipt
    POST /v1/cartridges/write  -> write one cartridge into a namespace
    POST /v1/cartridges/mount  -> confirm a namespace is mounted (404 if absent)
    POST /v1/code/solve        -> rank + execution-verify candidate programs

Mounting is FAIL-CLOSED: a mount for a namespace that was never written is a
404, never a silent empty bank. A caller that asks for memory must not
receive an empty one dressed as a success.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from henri.api import DigitalTwinAPI
from henri.cartridge_bank import CartridgeBank
from henri.code_verify import executor_status, solve_verified_code

app = FastAPI(title="HENRI Digital Twin", version="0.1.0")

_API: Optional[DigitalTwinAPI] = None
_BANKS: dict[str, CartridgeBank] = {}
# Code-template banks, keyed by namespace. Separate from `_BANKS` on
# purpose: `CartridgeBank` stores waves only, and a wave cannot be decoded
# back into Python source. `CodeTemplateBank` holds the wave bank AND a text
# registry beside it. Conflating the two would imply the wave core decoded
# source, which it does not.
_CODE_BANKS: dict[str, object] = {}


def get_api() -> DigitalTwinAPI:
    """Lazy singleton. The backbone loads the encoder once."""
    global _API
    if _API is None:
        _API = DigitalTwinAPI()
    return _API


def set_api(api: DigitalTwinAPI) -> None:
    """Inject an API instance. Used by tests to reuse one loaded encoder."""
    global _API
    _API = api


def reset_banks() -> None:
    _BANKS.clear()
    _CODE_BANKS.clear()


def _body(obj) -> dict:
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return dict(obj)


# -- request models ------------------------------------------------------
# Validation is explicit rather than relying on pydantic version-specific
# Field keywords. A wrong keyword on `list[str]` raises at import time in one
# major version and warns in another, so the check lives in the handler.

class RankRequest(BaseModel):
    question: str
    options: list[str]
    strip: bool = True
    signing_key: Optional[str] = None
    channel_weight: float = 0.0


class CartridgeWriteRequest(BaseModel):
    tenant_id: str
    keys: list[str]
    values: list[str]
    subject: str = ""


class CartridgeMountRequest(BaseModel):
    tenant_id: str


class CodeSolveRequest(BaseModel):
    spec: str
    candidates: list[str] = []
    tests: list[str] = []
    top_k: int = 3
    timeout: Optional[int] = None
    synthesize_from_bank: bool = False
    k_templates: int = 8
    max_variants: int = 2
    tenant_id: str = "default"


class CodeBankIngestRequest(BaseModel):
    tenant_id: str
    limit: int = 200


class RoboticsStepRequest(BaseModel):
    sensor_values: list[float]
    axiom_reference: Optional[list[float]] = None
    adapt_on_veto: bool = True


# -- endpoints -----------------------------------------------------------

@app.get("/v1/health/seal")
def health_seal() -> dict:
    return _body(get_api().verify_seal())


@app.get("/v1/health/executor")
def health_executor() -> dict:
    return executor_status()


@app.post("/v1/reason/rank")
def reason_rank(req: RankRequest) -> dict:
    if len(req.options) < 2:
        raise HTTPException(422, "at least two options are required")
    key = req.signing_key.encode("utf-8") if req.signing_key else None
    return _body(get_api().rank_options(req.question, req.options,
                                        strip=req.strip, signing_key=key,
                                        channel_weight=req.channel_weight))


@app.post("/v1/cartridges/write")
def cartridges_write(req: CartridgeWriteRequest) -> dict:
    if not req.tenant_id.strip():
        raise HTTPException(422, "tenant_id is required")
    if len(req.keys) != len(req.values):
        raise HTTPException(422, "keys and values must have equal length")
    if not req.keys:
        raise HTTPException(422, "at least one pair is required")

    api = get_api()
    # MEASURED DEFECT, repaired. The first draft used
    # `CartridgeBank(dim=api.backbone.dim, ...)`, but `SemanticBackbone` has
    # no `.dim` attribute (measured `hasattr(bb, 'dim') == False`), so every
    # write would have raised AttributeError. The wave is authoritative:
    # encode first, then size the bank from the actual width.
    keys = np.stack([api.backbone.encode(k).psi for k in req.keys], axis=0)
    values = np.stack([api.backbone.encode(v).psi for v in req.values], axis=0)
    wave_dim = int(keys.shape[1])

    bank = _BANKS.get(req.tenant_id)
    if bank is None:
        bank = CartridgeBank(dim=wave_dim, rank=api.rank)
        _BANKS[req.tenant_id] = bank
    elif bank.dim != wave_dim:
        raise HTTPException(
            409, f"bank dim {bank.dim} != wave dim {wave_dim}")

    try:
        cart = bank.add_cartridge(keys, values, subject=req.subject)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    bank.finalize()
    return {
        "endpoint": "cartridges_write",
        "tenant_id": req.tenant_id,
        "cartridge_index": int(cart.index),
        "rows": int(cart.rows),
        "cartridges": len(bank.cartridges),
        "subject": req.subject,
        "bank": bank.stats(),
        "namespace_note": "tenant_id is a namespace key, not an identity",
    }


@app.post("/v1/cartridges/mount")
def cartridges_mount(req: CartridgeMountRequest) -> dict:
    bank = _BANKS.get(req.tenant_id)
    if bank is None:
        raise HTTPException(
            404, f"namespace {req.tenant_id!r} has no bank; write first")
    return {
        "endpoint": "cartridges_mount",
        "tenant_id": req.tenant_id,
        "mounted": True,
        "cartridges": len(bank.cartridges),
        "rows": int(sum(c.rows for c in bank.cartridges)),
        "bank": bank.stats(),
    }


@app.post("/v1/code/solve")
def code_solve(req: CodeSolveRequest) -> dict:
    if not req.spec.strip():
        raise HTTPException(422, "spec is required")
    api = get_api()

    bank = _CODE_BANKS.get(req.tenant_id)
    if not req.candidates and not req.synthesize_from_bank:
        raise HTTPException(
            422, "at least one candidate is required unless "
                 "synthesize_from_bank is set")
    if req.synthesize_from_bank and bank is None:
        # FAIL-CLOSED, and it says which reason. A missing bank is not an
        # empty successful answer; it is a request that cannot be served.
        return {
            "endpoint": "code_solve",
            "verdict": "NO_CANDIDATES",
            "n_candidates": 0,
            "n_tests": len(req.tests),
            "executed": [],
            "synthesize_from_bank": True,
            "note": ("no code-template bank is mounted, so nothing can be "
                     "synthesized. Ingest templates first."),
        }
    return solve_verified_code(api.backbone, req.spec, req.candidates,
                               tests=req.tests, top_k=req.top_k,
                               timeout=req.timeout,
                               bank=bank,
                               synthesize_from_bank=req.synthesize_from_bank,
                               k_templates=req.k_templates,
                               max_variants=req.max_variants)


@app.post("/v1/code/bank/ingest")
def code_bank_ingest(req: CodeBankIngestRequest) -> dict:
    """Ingest (docstring, source) pairs from the staged MBPP corpus.

    The corpus is read, never written. A missing corpus is 409, not an
    empty bank dressed as success.
    """
    import json as _json
    from pathlib import Path as _Path

    from henri.code_synth import CodeTemplateBank, Template

    if req.limit <= 0:
        raise HTTPException(422, "limit must be positive")
    api = get_api()
    path = _Path(__file__).resolve().parents[1] / "data" / \
        "official_benchmarks" / "mbpp.jsonl"
    if not path.exists():
        raise HTTPException(409, f"corpus missing: {path}")

    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(_json.loads(line))
            if len(rows) >= int(req.limit):
                break
    if not rows:
        raise HTTPException(409, "corpus has no rows")

    # SIZE THE BANK FROM THE WAVE, not from a hoped-for attribute.
    # `SemanticBackbone` has no `.dim`; Phase 7 measured AttributeError on
    # exactly that assumption.
    dim = int(api.backbone.encode(rows[0]["text"]).psi.shape[0])
    bank = CodeTemplateBank(dim=dim, rank=api.rank)
    for r in rows:
        text = str(r.get("text", "")).strip()
        code = str(r.get("code", "")).strip()
        if not text or not code:
            continue
        bank.add(api.backbone,
                 Template(docstring=text, source=code,
                          task_id=str(r.get("task_id", ""))))
    if not bank.templates:
        raise HTTPException(409, "corpus rows had no usable text/code pairs")
    bank.finalize()
    _CODE_BANKS[req.tenant_id] = bank
    return {
        "endpoint": "code_bank_ingest",
        "tenant_id": req.tenant_id,
        "corpus": path.name,
        "templates": bank.rows,
        "bank": bank.to_dict(),
        "note": ("waves are stored for retrieval; SOURCE TEXT is kept in a "
                 "registry beside them, because a wave cannot be decoded "
                 "back into Python source"),
    }


# -- robotics: closed-loop active inference -------------------------------
# The loop is built once and reused. At D=65536 construction costs ~0.17 s;
# rebuilding per request would charge that to every step and make the
# reported latency meaningless. `set_robotics` exists so tests can inject a
# reduced-dimension loop instead of paying full-D cost on every case.

_ROBOTICS = None


def get_robotics():
    global _ROBOTICS
    if _ROBOTICS is None:
        from henri.robotics_loop import RoboticsLoop
        _ROBOTICS = RoboticsLoop()
    return _ROBOTICS


def set_robotics(loop) -> None:
    global _ROBOTICS
    _ROBOTICS = loop


def reset_robotics() -> None:
    global _ROBOTICS
    _ROBOTICS = None


@app.post("/v1/robotics/step")
def robotics_step(req: RoboticsStepRequest) -> dict:
    """One closed-loop step: ingest, predict, veto, relax, emit.

    Fail-closed rules, all of them 422 rather than a silent pass:
      * empty `sensor_values`            -> nothing to encode
      * non-finite `sensor_values`       -> NaN propagates through the gate
      * `axiom_reference` length differs -> the comparison is ill-posed

    A MISSING `axiom_reference` is not an error. It returns
    `vacuous_axiom: true` and `axiom_source: "self"`, because the loop then
    compares the candidate against itself. That arm cannot detect a fault;
    the flag says so instead of pretending a veto was evaluated.
    """
    if not req.sensor_values:
        raise HTTPException(422, "sensor_values must not be empty")
    arr = np.asarray(req.sensor_values, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise HTTPException(422, "sensor_values must all be finite")
    if req.axiom_reference is not None:
        if len(req.axiom_reference) != len(req.sensor_values):
            raise HTTPException(
                422,
                f"axiom_reference has {len(req.axiom_reference)} entries, "
                f"sensor_values has {len(req.sensor_values)}")
        ax = np.asarray(req.axiom_reference, dtype=float)
        if not np.all(np.isfinite(ax)):
            raise HTTPException(422, "axiom_reference must all be finite")

    loop = get_robotics()
    try:
        return loop.step(req.sensor_values,
                         axiom_reference=req.axiom_reference,
                         adapt_on_veto=req.adapt_on_veto)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
