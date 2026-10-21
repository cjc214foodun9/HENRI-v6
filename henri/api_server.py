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


class CartridgeWriteRequest(BaseModel):
    tenant_id: str
    keys: list[str]
    values: list[str]
    subject: str = ""


class CartridgeMountRequest(BaseModel):
    tenant_id: str


class CodeSolveRequest(BaseModel):
    spec: str
    candidates: list[str]
    tests: list[str] = []
    top_k: int = 3
    timeout: Optional[int] = None


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
                                        strip=req.strip, signing_key=key))


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
    if not req.candidates:
        raise HTTPException(422, "at least one candidate is required")
    api = get_api()
    return solve_verified_code(api.backbone, req.spec, req.candidates,
                               tests=req.tests, top_k=req.top_k,
                               timeout=req.timeout)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
