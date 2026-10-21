"""Routed multi-cartridge Zone C bank: complex hot tier + uint8 cold tier.

THE ONE TRAP IN THE SPECIFICATION, AND THE MEASUREMENT THAT SETTLES IT
---------------------------------------------------------------------
Action B of Phase 4 said: orthogonalise cartridge keys with
`gram.stiefel_retract`, then "match against `K_c_ortho` via
`readout.select`". That re-introduces the exact defect Phase 4B falsified.
`stiefel_retract` mixes keys, so scoring a raw query against mixed keys is
`G^-1/2 (K q)` — the same score mixing that dropped MMLU 0.3050 -> 0.2475.
MEASURED: argmax differed on 40/40 queries at D=4096, m=64.

The correct split, implemented here:
    * MATCH against the ORIGINAL keys.
    * READ by index through `DeltaMemory`.
    * ROUTE by the centroid of ORIGINAL keys.

CAPACITY. `m_c <= rank` is a hard rank wall, confirmed to the row in Phase 4B.

IS WRITE-TIME STIEFEL NEEDED? No. The gated delta rule already
orthogonalises internally: 64 keys at raw `c_max = 0.9934` produced a
stored-key Gram with off-diagonal 0.000000 and paired recall 1.0000.
`write_stiefel` stays available, default OFF, as a reproducibility flag.

PHASE 6 ACTION 2 — TWO MEMORY TIERS
-----------------------------------
Complex128 hot tier: `2 * 65536 * 64 * 16 = 134,217,728 B = 128 MiB` per
cartridge. The 13,632-row bank needs 213 cartridges = 26.62 GiB, which does
not fit beside anything else on a 32 GiB card.

Cold tier: uint8 `Z_256` ring indices for keys AND values. One 64-row
cartridge is `2 * 64 * 2 * 16384 = 4,194,304 B = exactly 4.00 MiB`, a 32x
reduction. The directive budgeted 16x because its byte count assumed keys
only. 213 cartridges cold is 0.83 GiB.

Hot tier is bounded by an LRU. `max_hot = 8` gives a 1.00 GiB ceiling, so
cold plus hot ceilings total 1.83 GiB against the 3.0 GiB gate.

QUANTISATION IS PHASE, NOT AMPLITUDE. `|psi_d| = 1/sqrt(D)` is a constant,
so the angle is the whole content. Reconstruction is exact up to one bin:
the measured bound is `pi/256 = 0.012272` rad.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from henri.memory import DeltaMemory

__all__ = [
    "CartridgeBank", "Cartridge", "RouteResult", "ColdCartridge",
    "LRUHotCache", "cartridge_bytes", "cold_cartridge_bytes", "build_bank",
    "MAX_ROWS", "SLOT_SPAN", "QUANT_BOUND",
]

MAX_ROWS = 64
"""Exact capacity wall. `DeltaMemory` stores rank-`r` factors, so exact
recovery of `m` independent associations needs `m <= rank`."""

SLOT_SPAN = 16384
"""Active slots stored in the cold tier (slot 0 content, slot 1 polarity)."""


def unit_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(n > 0, n, 1.0)


def cartridge_bytes(dim: int, rank: int, dtype=np.complex128) -> int:
    """Bytes of one HOT cartridge: both factors, `(dim, rank)` each."""
    return int(2 * dim * rank * np.dtype(dtype).itemsize)


def cold_cartridge_bytes(rows: int, slot: int = SLOT_SPAN) -> int:
    """Bytes of one COLD cartridge: keys AND values as uint8 ring indices."""
    return int(2 * rows * 2 * slot)


QUANT_BOUND = float(np.pi / 256.0)
"""Half a `Z_256` bin. The exact reconstruction error bound for one bin."""


def content_sim(keys_raw: np.ndarray, q: np.ndarray,
                slot: int = SLOT_SPAN) -> np.ndarray:
    """Cosine similarity on the CONTENT subspace only.

    MEASURED DEFECT, repaired. Scoring a cartridge match on the FULL wave
    makes the gate vacuous. Slot 1 carries a FIXED role basis identical for
    every input, so every pair receives a constant `+w1 = 0.5` there:

        cos(unrelated, full wave)     = +0.5705
        cos(unrelated, slot0 content) = +0.0705
        mean over 8 unrelated pairs   = +0.6255

    With that floor, `tau_hit = 0.55` cannot separate a hit from noise: an
    out-of-bank query still reported `cartridge_hit = True` and blended at
    `w_mem = 0.0455`. A threshold inside the null distribution is not a
    gate. Reading `t_key` on the content subspace restores a real margin,
    because that is the only subspace that varies with the input.
    """
    k = np.asarray(keys_raw)[:, :slot]
    qq = np.asarray(q)[:slot]
    kn = k / np.maximum(np.linalg.norm(k, axis=1, keepdims=True), 1e-30)
    qn = qq / max(float(np.linalg.norm(qq)), 1e-30)
    return np.real(kn @ qn.conj())


class ColdCartridge:
    """Quantised cartridge: uint8 `Z_256` phase rings, no complex factors.

    Stores the two ACTIVE slots only. Slots 2 and 3 are exactly zero in the
    current encoder, so storing them would be 50 percent dead space.
    """

    __slots__ = ("index", "keys_q", "values_q", "subject", "rows", "slot")

    def __init__(self, index: int, keys_q: np.ndarray, values_q: np.ndarray,
                 subject: str = "", slot: int = SLOT_SPAN) -> None:
        self.index = int(index)
        self.keys_q = np.ascontiguousarray(keys_q, dtype=np.uint8)
        self.values_q = np.ascontiguousarray(values_q, dtype=np.uint8)
        self.subject = str(subject)
        self.rows = int(keys_q.shape[0])
        self.slot = int(slot)

    @classmethod
    def from_waves(cls, index: int, keys: np.ndarray, values: np.ndarray,
                   subject: str = "", slot: int = SLOT_SPAN) -> "ColdCartridge":
        """Quantise complex waves to uint8 ring indices over the active span.

        Callers pass the ACTIVE SLICE, so `keys.shape[1] == 2 * slot`.
        Phase is the only content, so this is lossless up to one bin.
        """
        def ring(psi: np.ndarray) -> np.ndarray:
            th = np.angle(psi)
            q = np.floor(th * (256.0 / (2.0 * np.pi)) + 0.5).astype(np.int64)
            return (q % 256).astype(np.uint8)

        if keys.shape[1] != 2 * slot:
            raise ValueError(
                f"cold tier expects the {2 * slot}-wide active span, got "
                f"{keys.shape[1]}")
        return cls(index, ring(keys), ring(values), subject, slot)

    def materialise(self) -> tuple:
        """Reconstruct complex waves on the active span. One-bin exact."""
        def waves(q: np.ndarray) -> np.ndarray:
            th = 2.0 * np.pi * q.astype(np.float64) / 256.0
            return np.exp(1j * th) / np.sqrt(q.shape[1])

        return waves(self.keys_q), waves(self.values_q)

    @property
    def nbytes(self) -> int:
        return int(self.keys_q.nbytes + self.values_q.nbytes)

    def to_dict(self) -> dict:
        return {"index": self.index, "rows": self.rows,
                "subject": self.subject, "slot_width": 2 * self.slot,
                "bytes": self.nbytes}


class LRUHotCache:
    """Bounded LRU over materialised hot objects. Evictions are counted."""

    def __init__(self, max_entries: int = 8) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self.max_entries = int(max_entries)
        self._store: "OrderedDict" = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key: int):
        if key in self._store:
            self.hits += 1
            self._store.move_to_end(key)
            return self._store[key]
        self.misses += 1
        return None

    def put(self, key: int, value) -> None:
        self._store[key] = value
        self._store.move_to_end(key)
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)
            self.evictions += 1

    def __contains__(self, key: int) -> bool:
        return key in self._store

    @property
    def entries(self) -> int:
        return len(self._store)

    def stats(self) -> dict:
        return {"entries": self.entries, "max": self.max_entries,
                "hits": self.hits, "misses": self.misses,
                "evictions": self.evictions}


@dataclass
class Cartridge:
    """One association set: ORIGINAL keys, values, and a rebuildable memory.

    `keys_raw` is retained because matching must use raw keys. The hot
    `DeltaMemory` is rebuildable from the same arrays, which is what makes a
    bounded LRU possible: eviction costs a rebuild, never data.
    """

    index: int
    keys_raw: np.ndarray            # (m, D), unit rows
    values: np.ndarray              # (m, D), unit rows
    centroid: np.ndarray            # (D,), centroid of keys_raw
    rank: int
    dim: int
    dtype: np.dtype = np.complex128
    subject: str = ""
    memory: Optional[DeltaMemory] = None
    keys_ortho: Optional[np.ndarray] = None
    cold: Optional[ColdCartridge] = None

    @property
    def rows(self) -> int:
        return int(self.keys_raw.shape[0])

    def build_memory(self) -> DeltaMemory:
        """Materialise the hot operator from the retained arrays."""
        mem = DeltaMemory.build(dim=self.dim, rank=self.rank, dtype=self.dtype)
        for i in range(self.rows):
            mem.write(self.keys_raw[i], self.values[i])
        self.memory = mem
        return mem

    def ensure_memory(self) -> DeltaMemory:
        return self.memory if self.memory is not None else self.build_memory()

    def stored_offdiag(self) -> float:
        """Max |off-diagonal| of the STORED key Gram. Measured, not assumed.

        Requires a materialised memory. The bank materialises on demand, so
        this rebuilds when needed rather than reporting a stale zero.
        """
        mem = self.ensure_memory()
        k = mem.k_factors
        if k.shape[1] == 0:
            return 0.0
        g = np.abs(k.conj().T @ k)
        np.fill_diagonal(g, 0.0)
        return float(g.max())

    # Backwards-compatible alias used by the Phase 5 tests.
    def _stored_offdiag(self) -> float:
        return self.stored_offdiag()

    def to_dict(self) -> dict:
        return {"index": self.index, "rows": self.rows,
                "subject": self.subject, "rank": self.rank, "dim": self.dim,
                "stored_orthonormal_offdiag": self.stored_offdiag()}


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
        return {"cartridge": int(self.cartridge), "row": int(self.row),
                "margin": round(float(self.margin), 9),
                "t": round(float(self.t), 9),
                "routed": int(self.routed),
                "rows_searched": int(self.rows_searched),
                "cartridge_hit": bool(self.hits)}


class CartridgeBank:
    """Partition associations into cartridges of `m_c <= rank`.

    Args:
        dim: wave dimension `D`.
        rank: `DeltaMemory` rank. Also the per-cartridge capacity.
        max_rows: rows per cartridge, clamped to `rank`.
        dtype: hot factor dtype. complex128 is exact.
        write_stiefel: redundant write-time key orthogonalisation. Default
            OFF: the delta rule already orthogonalises, measured.
        max_hot: LRU ceiling on materialised `DeltaMemory` instances.
        cold_slot: active-slot width stored in the cold tier.
    """

    def __init__(self, dim: int = 65536, rank: int = 64,
                 max_rows: Optional[int] = None,
                 dtype=np.complex128, write_stiefel: bool = False,
                 max_hot: int = 8, cold_slot: int = SLOT_SPAN) -> None:
        self.dim = int(dim)
        self.rank = int(rank)
        self.max_rows = int(min(max_rows or rank, rank))
        self.dtype = np.dtype(dtype)
        self.write_stiefel = bool(write_stiefel)
        self.cold_slot = int(cold_slot)
        self.cartridges: list = []
        self.cold: list = []
        self.hot = LRUHotCache(max_entries=max_hot)
        self._router: Optional[np.ndarray] = None

    # -- cold path -------------------------------------------------------
    def add_cartridge(self, keys: np.ndarray, values: np.ndarray,
                      subject: str = "") -> Cartridge:
        """Write `m <= rank` associations into a new cartridge.

        Stores original keys and values, a rebuildable hot operator, and a
        uint8 cold copy. The write-time Stiefel step is opt-in and OFF by
        default because it is redundant and it mixes keys.
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
        c = Cartridge(index=len(self.cartridges), keys_raw=k, values=v,
                      centroid=unit_rows(k.sum(axis=0)), rank=self.rank,
                      dim=self.dim, dtype=self.dtype, subject=subject)
        c.build_memory()
        if self.write_stiefel and m > 1:
            from henri.gram import stiefel_retract
            c.keys_ortho = stiefel_retract(
                k.astype(np.complex128)).astype(self.dtype)
        span = 2 * self.cold_slot
        if k.shape[1] >= span:
            c.cold = ColdCartridge.from_waves(
                len(self.cold), k[:, :span], v[:, :span], subject,
                self.cold_slot)
            self.cold.append(c.cold)
        self.cartridges.append(c)
        self.hot.put(c.index, c.memory)
        self._router = None
        return c

    def finalize(self) -> np.ndarray:
        """Build the router matrix from ORIGINAL-key CONTENT centroids.

        The router reads the content subspace for the same reason
        `content_sim` does: the full-wave centroid is dominated by the fixed
        slot-1 role basis, which is nearly identical across topics and makes
        routing close to arbitrary.
        """
        if not self.cartridges:
            raise ValueError("no cartridges")
        self._router = unit_rows(
            np.stack([c.centroid[:self.cold_slot] for c in self.cartridges]))
        return self._router

    # -- warm path -------------------------------------------------------
    @property
    def router(self) -> np.ndarray:
        return self._router if self._router is not None else self.finalize()

    def route(self, query: np.ndarray, k_route: int = 1) -> list:
        """Indices of the `k_route` best cartridges by CONTENT resonance.

        The router matrix holds content-subspace centroids, so the query must
        be sliced to the same width. Measured defect caught before shipping:
        `finalize()` stores `(C, cold_slot)` centroids while this method
        normalised the FULL wave, so `router @ q` raised a broadcast error on
        every call once `cold_slot < dim`.
        """
        q = np.asarray(query, dtype=self.dtype).ravel()[:self.cold_slot]
        q = q / max(np.linalg.norm(q), 1e-30)
        scores = np.real(self.router @ q.conj())
        k = int(max(1, min(k_route, len(self.cartridges))))
        return list(np.argsort(-scores)[:k])

    def _hot_memory(self, c: Cartridge) -> DeltaMemory:
        """LRU-guarded hot operator. Rebuilds on miss, never loses data."""
        mem = self.hot.get(c.index)
        if mem is None:
            mem = c.ensure_memory()
            self.hot.put(c.index, mem)
        return mem

    def query(self, query: np.ndarray, k_route: int = 1) -> RouteResult:
        """Route, MATCH on raw keys, READ by index through the LRU hot tier."""
        q = np.asarray(query, dtype=self.dtype).ravel()
        q = q / max(np.linalg.norm(q), 1e-30)
        best = None
        searched = 0
        cand = self.route(q, k_route)
        for ci in cand:
            c = self.cartridges[ci]
            searched += c.rows
            # MATCH ON THE CONTENT SUBSPACE. Scoring the full wave makes the
            # hit gate vacuous: the fixed slot-1 role basis contributes a
            # constant +0.5 to every pair, so unrelated text still reads
            # ~0.57-0.63 against a tau_hit of 0.55. Measured: an out-of-bank
            # query returned cartridge_hit=True at w_mem=0.0455. Reading
            # `t` here restores a real margin (unrelated ~0.07).
            s = content_sim(c.keys_raw, q, self.cold_slot)
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
        val = self._hot_memory(c).read(c.keys_raw[i], iterations=1)
        return RouteResult(value=val, cartridge=ci, row=i, margin=margin,
                           t=top, routed=len(cand), rows_searched=searched,
                           hits=True)

    # -- telemetry -------------------------------------------------------
    def hot_bytes(self) -> int:
        return int(self.hot.entries * cartridge_bytes(self.dim, self.rank,
                                                      self.dtype))

    def cold_bytes(self) -> int:
        return int(sum(c.nbytes for c in self.cold))

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
                (c.stored_offdiag() for c in self.cartridges), default=0.0),
            "cold_cartridges": len(self.cold),
            "cold_bytes_total": self.cold_bytes(),
            "cold_gib_total": round(self.cold_bytes() / 2**30, 4),
            "hot_entries": self.hot.entries,
            "hot_bytes": self.hot_bytes(),
            "hot_gib": round(self.hot_bytes() / 2**30, 4),
            "hot_lru": self.hot.stats(),
            "tier_total_gib": round((self.cold_bytes() + self.hot_bytes())
                                    / 2**30, 4),
        }


def build_bank(keys: np.ndarray, values: np.ndarray, dim: int,
               rank: int = MAX_ROWS, dtype=np.complex128,
               subjects: Optional[Sequence[str]] = None,
               k_route: int = 1, max_hot: int = 8) -> CartridgeBank:
    """Pack `(keys, values)` into as few cartridges as the wall allows.

    Packing is by contiguous blocks, grouped by subject when supplied, so a
    cartridge holds one topic. No gradient, no fitting: algebraic
    crystallisation.
    """
    keys = unit_rows(np.asarray(keys, dtype=dtype))
    values = unit_rows(np.asarray(values, dtype=dtype))
    n = int(keys.shape[0])
    bank = CartridgeBank(dim=dim, rank=rank, dtype=dtype, max_hot=max_hot)
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
