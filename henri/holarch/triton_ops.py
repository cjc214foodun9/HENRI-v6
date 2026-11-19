"""Functional surface of the portable kernel layer (TK-1 .. TK-5).

Every entry point takes the reference semantics and an optional backend.  The
default is the auto-selected backend (ReferenceWaveBackend on this host), and
the return value always matches the reference contract bit for bit within
1e-5, which is what scripts/run_holarch_remote_cuda.py asserts.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from henri.holarch.backends import (
    ReferenceWaveBackend,
    born_contextual_phase,
    chart_retrieval,
    parity,
    planar_ingress,
    probe_device,
    sagnac_plate,
    select_backend,
    stiefel_readout,
    swarm_topk,
)

DEFAULT_BACKEND = None


def backend(name: str = "auto"):
    global DEFAULT_BACKEND
    if name == "auto" and DEFAULT_BACKEND is not None:
        return DEFAULT_BACKEND
    chosen = select_backend(name)
    if name == "auto":
        DEFAULT_BACKEND = chosen
    return chosen


def tk1_ingress(token_ids, token_phases, *, dim: int, backend_name: str = "auto"):
    return backend(backend_name).ingress(token_ids, token_phases, dim=dim)


def tk2_sagnac(planar, mixers, chirps, *, zeta: float = 0.0, kerr: float = 0.0, backend_name: str = "auto"):
    return backend(backend_name).sagnac(planar, mixers, chirps, zeta=zeta, kerr=kerr)


def tk3_charts(queries, charts, *, top_k: int = 2, backend_name: str = "auto"):
    return backend(backend_name).charts(queries, charts, top_k=top_k)


def tk4_swarm(x, routing, expert_weights, expert_biases, *, top_k: int = 2, backend_name: str = "auto"):
    chosen = backend(backend_name)
    if hasattr(chosen, "swarm"):
        return chosen.swarm(x, routing, expert_weights, expert_biases, top_k=top_k)
    return swarm_topk(x, routing, expert_weights, expert_biases, top_k=top_k)


def tk5_born(phi, rows, steering, values, *, rank: int = 8, backend_name: str = "auto"):
    chosen = backend(backend_name)
    if hasattr(chosen, "born"):
        return chosen.born(phi, rows, steering, values, rank=rank)
    return born_contextual_phase(phi, rows, steering, values, rank=rank)


def parity_report(tol: float = 1e-5) -> Dict[str, Any]:
    reference = ReferenceWaveBackend()
    profile = probe_device()
    chosen = select_backend("auto", profile)
    if isinstance(chosen, ReferenceWaveBackend):
        return {
            "backend": "reference",
            "profile": profile.as_dict(),
            "parity": {
                "max_abs_delta": 0.0,
                "tol": float(tol),
                "parity_ok": True,
                "note": "the reference backend is compared with itself on this host",
            },
        }
    return {
        "backend": str(chosen.name),
        "profile": profile.as_dict(),
        "parity": parity(reference, chosen, tol=tol),
    }


__all__ = [
    "backend",
    "tk1_ingress",
    "tk2_sagnac",
    "tk3_charts",
    "tk4_swarm",
    "tk5_born",
    "parity_report",
    "probe_device",
]
