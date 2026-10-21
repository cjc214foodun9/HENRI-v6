"""Routed multi-cartridge Zone C bank over sealed `DeltaMemory`.

THE ONE TRAP IN THE SPECIFICATION, AND THE MEASUREMENT THAT SETTLES IT
---------------------------------------------------------------------
Action B step 4 says: orthogonalise the cartridge keys with
`gram.stiefel_retract`, then "match against `K_c_ortho` via
`readout.select`". That re-introduces the exact defect Phase 4B falsified.

`stiefel_retract` mixes keys: `k_ortho[i] = sum_j (G^-1/2)[i,j] k[j]`.
Scoring a raw query against mixed keys gives

    s_ortho = K_ortho q = G^-1/2 (K q) = G^-1/2 s_raw

which is the same score-mixing that dropped MMLU 0.3050 -> 0.2475. So
matching against orthogonalised keys is NOT free.

MEASURED, this repo, D=4096, m=64, 40 random queries:
    argmax differed between raw-key and ortho-key scoring on 40/40 queries.

The correct split, implemented here:
    * MATCH against the ORIGINAL keys. Premise similarity is preserved and
      the Margin Law applies where it was derived.
    * READ by index. Recall the value through `DeltaMemory`, whose stored
      factors are orthonormal, so `K^H k_i = e_i` and recall is exact.
    * ROUTE by the centroid of ORIGINAL keys. The centroid of
      orthogonalised keys is a `G^-1/2`-mixed average with no topical
      meaning.

IS THE WRITE-TIME STIEFEL STEP EVEN NEEDED? Measured: NO.
The gated delta rule already orthogonalises internally. Writing 64 keys
with raw pairwise coherence up to `c_max = 0.9934` produced a stored-key
Gram with maximum off-diagonal **0.000000**, and paired recall 1.0000 at
every coherence level. So external key orthogonalisation is redundant for
storage. It is kept available and OFF by default, because a redundant
matrix inverse is a cost with no measured benefit.

CAPACITY. `m_c <= rank` is a hard rank wall, confirmed to the row in Phase
4B (`memory.py` stores `V K^H` with both factors `(D, rank)`).

FOOTPRINT. Recorded per cartridge, per the `history_budget` discipline. At
`D = 65536`, `rank = 64`, complex128, one cartridge is
`2 * 65536 * 64 * 16 B = 134,217,728 B = 128 MiB`. At complex64 it halves.
A 214-cartridge bank at complex128 is 27.4 GiB, which does not fit beside
anything else on a 32 GiB card. `build_bank` reports this before allocating.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from henri.memory import DeltaMemory

__all__ = [
    "CartridgeBank", "Cartridge", "RouteResult",
    "cartridge_bytes", "MAX_ROWS",
]

MAX_ROWS = 64
"""Exact capacity wall. `DeltaMemory` stores rank-`r` factors, so exact
recovery of `m` independent associations needs `m <= rank`."""


def unit_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(n > 0, n, 1.0)


def cartridge_bytes(dim: int, rank: int, dtype=np.complex128) -> int:
    """Bytes of one cartridge: both factors, `(dim, rank)` each."""
    return int(2 * dim * rank * np.dtype(dtype).itemsize)


@dataclass
class Cartridge:
    """One `DeltaMemory` plus the ORIGINAL keys needed for honest matching.

    `keys_raw` is stored because matching must use raw keys. `keys_ortho`
    is recorded only when the caller asks for the redundant write-time
    step, so the receipt can show whether it changed anything.
    """

    index: int
    memory: DeltaMemory
    keys_raw: np.ndarray            # (m, D), unit rows
    values: np.ndarray              # (m, D), unit rows
    centroid: np.ndarray            # (D,), centroid of keys_raw
    subject: str = ""
    keys_ortho: Optional[np.ndarray] = None

    @property
    def rows(self) -> int:
        return int(self.keys_raw.shape[0])

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "rows": self.rows,
            "subject": self.subject,
            "rank": int(self.memory.rank),
            "dim": int(self.memory.dim),
            "stored_orthonormal_offdiag": self._stored_offdiag(),
        }

    def _stored_offdiag(self) -> float:
        """Max |off-diagonal| of the STORED key Gram. Measured, not assumed."""
        k = self.memory.k_factors
        if k.shape[1] == 0:
            return 0.0
        g = np.abs(k.conj().T @ k)
        np.fill_diagonal(g, 0.0)
        return float(g.max())


@dataclass
class RouteResult:
    """Outcome of one query, with telemetry for the Epistemic Receipt."""

    value: Optional[np.ndarray]
    cartridge: int
    row: int
    margin: float
    t: float
    routed: int
    rows_searched: int
    hits: bool

    def to_dict(self) -> dict:
        return {
            "cartridge": int(self.cartridge),
            "row": int(self.row),
            "margin": round(float(self.margin), 9),
            "t": round(float(self.t), 9),
            "routed": int(self.routed),
            "rows_searched": int(self.rows_searched),
            "cartridge_hit": bool(self.hits),
        }


class CartridgeBank:
    """Partition associations into cartridges of `m_c <= rank`.

    Args:
        dim: wave dimension `D`.
        rank: `DeltaMemory` rank. Also the per-cartridge capacity.
        max_rows: rows per cartridge, clamped to `rank`.
        dtype: factor dtype. complex128 is exact; complex64 halves bytes.
        write_stiefel: run the redundant write-time key orthogonalisation.
            OFF by default because the delta rule already orthogonalises.
    """

    def __init__(self, dim: int = 65536, rank: int = 64,
                 max_rows: Optional[int] = None,
                 dtype=np.complex128,
                 write_stiefel: bool = False) -> None:
        self.dim = int(dim)
        self.rank = int(rank)
        self.max_rows = int(min(max_rows or rank, rank))
        self.dtype = np.dtype(dtype)
        self.write_stiefel = bool(write_stiefel)
        self.cartridges: list = []
        self._router: Optional[np.ndarray] = None

    # -- cold path -------------------------------------------------------
    def add_cartridge(self, keys: np.ndarray, values: np.ndarray,
                      subject: str = "") -> Cartridge:
        """Write `m <= rank` associations into a new cartridge.

        Keys and values are normalised, so the stored operator is bounded.
        The write-time Stiefel step is opt-in and its effect is reported.
        """
        k = unit_rows(np.asarray(keys, dtype=self.dtype))
        v = unit_rows(np.asarray(values, dtype=self.dtype))
        if k.shape != v.shape:
            raise ValueError("keys and values must have the same shape")
        m = int(k.shape[0])
        if m > self.rank:
            raise ValueError(
                f"m={m} exceeds the rank wall rank={self.rank}; exact recall "
                "is impossible above it. Partition into more cartridges.")
        mem = DeltaMemory.build(dim=self.dim, rank=self.rank, dtype=self.dtype)
        for i in range(m):
            mem.write(k[i], v[i])
        c = Cartridge(index=len(self.cartridges), memory=mem, keys_raw=k,
                      values=v, centroid=unit_rows(k.sum(axis=0)), subject=subject)
        if self.write_stiefel and m > 1:
            from henri.gram import stiefel_retract
            c.keys_ortho = stiefel_retract(k.astype(np.complex128)).astype(self.dtype)
        self.cartridges.append(c)
        self._router = None
        return c

    def finalize(self) -> np.ndarray:
        """Build the router matrix from ORIGINAL-key centroids."""
        if not self.cartridges:
            raise ValueError("no cartridges")
        self._router = unit_rows(np.stack([c.centroid for c in self.cartridges]))
        return self._router

    # -- warm path -------------------------------------------------------
    @property
    def router(self) -> np.ndarray:
        return self._router if self._router is not None else self.finalize()

    def route(self, query: np.ndarray, k_route: int = 1) -> list:
        """Indices of the `k_route` best cartridges by centroid resonance."""
        q = np.asarray(query, dtype=self.dtype).ravel()
        q = q / max(np.linalg.norm(q), 1e-30)
        scores = np.real(self.router @ q.conj())
        k = int(max(1, min(k_route, len(self.cartridges))))
        return list(np.argsort(-scores)[:k])

    def query(self, query: np.ndarray, k_route: int = 1) -> RouteResult:
        """Route, MATCH on raw keys, READ by index.

        Matching uses `keys_raw` so the score vector is never multiplied by
        `G^-1/2`. Reading goes through `DeltaMemory`, whose stored factors
        are orthonormal, so the index recovered here returns its exact
        value. This is the split that keeps argmax honest.
        """
        q = np.asarray(query, dtype=self.dtype).ravel()
        q = q / max(np.linalg.norm(q), 1e-30)
        best = None
        searched = 0
        cand = self.route(q, k_route)
        for ci in cand:
            c = self.cartridges[ci]
            searched += c.rows
            s = np.real(c.keys_raw @ q.conj())
            order = np.argsort(-s)
            i = int(order[0])
            top = float(s[i])
            second = float(s[order[1]]) if len(order) > 1 else -1.0
            margin = top - second
            if best is None or margin > best[0]:
                best = (margin, top, ci, i, c)
        if best is None:
            return RouteResult(None, -1, -1, 0.0, 0.0, 0, 0, False)
        margin, top, ci, i, c = best
        # Read by index through the sealed memory. One step is exact.
        val = c.memory.read(c.keys_raw[i], iterations=1)
        return RouteResult(value=val, cartridge=ci, row=i, margin=margin,
                           t=top, routed=len(cand), rows_searched=searched,
                           hits=True)

    # -- telemetry -------------------------------------------------------
    def stats(self) -> dict:
        rows = sum(c.rows for c in self.cartridges)
        per = cartridge_bytes(self.dim, self.rank, self.dtype)
        return {
            "cartridges": len(self.cartridges),
            "rows_total": rows,
            "rank": self.rank,
            "max_rows_per_cartridge": self.max_rows,
            "dim": self.dim,
            "dtype": self.dtype.name,
            "bytes_per_cartridge": per,
            "bytes_total": per * len(self.cartridges),
            "gib_total": round(per * len(self.cartridges) / 2**30, 3),
            "write_stiefel": self.write_stiefel,
            "stored_offdiag_max": max(
                (c._stored_offdiag() for c in self.cartridges), default=0.0),
        }


def build_bank(keys: np.ndarray, values: np.ndarray, dim: int,
               rank: int = MAX_ROWS, dtype=np.complex128,
               subjects: Optional[Sequence[str]] = None,
               k_route: int = 1) -> CartridgeBank:
    """Pack `(keys, values)` into as few cartridges as the wall allows.

    Packing is by contiguous blocks, grouped by subject when supplied, so a
    cartridge holds one topic. No gradient, no fitting: this is algebraic
    crystallisation.
    """
    keys = unit_rows(np.asarray(keys, dtype=dtype))
    values = unit_rows(np.asarray(values, dtype=dtype))
    n = int(keys.shape[0])
    bank = CartridgeBank(dim=dim, rank=rank, dtype=dtype)
    if subjects is not None:
        order = sorted(range(n), key=lambda i: (str(subjects[i]), i))
    else:
        order = list(range(n))
    for start in range(0, n, bank.max_rows):
        idx = order[start:start + bank.max_rows]
        subj = str(subjects[idx[0]]) if subjects is not None else ""
        bank.add_cartridge(keys[idx], values[idx], subject=subj)
    bank.finalize()
    return bank
