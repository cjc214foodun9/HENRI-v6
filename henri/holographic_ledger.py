"""Zone C: holographic agentic ledger.

The name is `holographic_ledger`, not `timescaledb_holographic_ledger`. The
README lists TimescaleDB hypertables as BLOCKED, and a filename must not promise
what the code cannot deliver. `BlockedTimescaleDBStore` exists so a caller gets a
typed refusal instead of a silent fallback.

What is real here:
  - uint8 qFHRR records, 1 byte per channel (65,536 channels = 64 KiB at D=65536)
  - HHRG Stiefel charts (c_max < 1e-6) reused from `hhrg_memory`
  - a JIT pager that returns a bounded Pockels voltage map and a fixed-width
    (prefix_tokens, d_tiny) tensor regardless of N_ctx
  - transactional commit: an injected store failure restores chart state,
    boundary, and record index

Limits stated, not hidden: the B-tree DEPTH is O(log_256 N). Retrieval SNR is
NOT logarithmic; it is measured and reported as-is. No TimescaleDB service ran.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np

from henri.hhrg_memory import HHRGMemory, HHRGRetrievalResult

TWO_PI = 2.0 * math.pi


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.zeros_like(v)


@dataclass
class LedgerRecord:
    record_id: int
    bins: np.ndarray
    label: str
    chart_id: int
    local_index: int


class LedgerStore(Protocol):
    def append(self, record: LedgerRecord) -> None: ...
    def fetch_all(self) -> List[LedgerRecord]: ...
    def count(self) -> int: ...


class InMemoryLedgerStore:
    def __init__(self) -> None:
        self._records: List[LedgerRecord] = []

    def append(self, record: LedgerRecord) -> None:
        self._records.append(record)

    def fetch_all(self) -> List[LedgerRecord]:
        return list(self._records)

    def count(self) -> int:
        return len(self._records)


class BlockedTimescaleDBStore:
    """Typed refusal. No TimescaleDB service exists on this host."""

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "BLOCKED: TimescaleDB hypertable backend is not available. "
            "Use InMemoryLedgerStore. The README already lists hypertables as BLOCKED."
        )


class HolographicAgenticLedger:
    def __init__(
        self,
        d_model: int = 16384,
        chart_capacity: int = 256,
        quant_bins: int = 256,
        store: Optional[LedgerStore] = None,
        d_tiny: int = 64,
        prefix_tokens: int = 16,
        seed: int = 42,
    ) -> None:
        if quant_bins != 256:
            raise ValueError("uint8 records require quant_bins == 256")
        self.d_model = int(d_model)
        self.quant_bins = int(quant_bins)
        self.d_tiny = int(d_tiny)
        self.prefix_tokens = int(prefix_tokens)
        self.memory = HHRGMemory(d_model=self.d_model, chart_capacity=chart_capacity)
        self.store: LedgerStore = store if store is not None else InMemoryLedgerStore()
        self.boundary = np.zeros(self.d_model, dtype=np.complex128)
        self.records: Dict[int, LedgerRecord] = {}
        self._next_id = 0

        rng = np.random.default_rng(seed)
        chunk = self.d_model // self.prefix_tokens
        self.prefix_proj = rng.standard_normal((chunk, self.d_tiny)) / math.sqrt(float(chunk))

    # ------------------------------------------------------------ quantisation
    def quantize(self, wave: np.ndarray) -> np.ndarray:
        w = np.asarray(wave, dtype=np.complex128).reshape(-1)
        bins = np.rint((np.angle(w) % TWO_PI) * (self.quant_bins / TWO_PI)).astype(np.int64)
        return (bins % self.quant_bins).astype(np.uint8)

    def dequantize(self, bins: np.ndarray) -> np.ndarray:
        angles = np.asarray(bins, dtype=np.float64) * (TWO_PI / self.quant_bins)
        return np.exp(1j * angles)

    # --------------------------------------------------------------- snapshot
    def _snapshot(self):
        return {
            "charts": self.memory.charts,
            "sizes": [c.size for c in self.memory.charts],
            "values": [None if c.values is None else c.values.copy() for c in self.memory.charts],
            "labels": [list(c.labels) for c in self.memory.charts],
            "boundary": self.boundary.copy(),
            "records": dict(self.records),
            "next_id": self._next_id,
        }

    def _restore(self, snap) -> None:
        self.memory.charts = snap["charts"]
        for chart, n, vals, labels in zip(
            snap["charts"], snap["sizes"], snap["values"], snap["labels"]
        ):
            chart.raw_keys = chart.raw_keys[:n]
            chart.labels = list(labels)[:n]
            chart.values = None if vals is None else vals[:n].copy()
            if n:
                chart.stiefel_retract()
            else:
                chart.ortho_keys = None
                chart.centroid = None
                chart.c_max = 0.0
        self.boundary = snap["boundary"]
        self.records = snap["records"]
        self._next_id = snap["next_id"]

    # ----------------------------------------------------------------- commit
    def commit(self, key_wave: np.ndarray, value_wave: np.ndarray, label: str = "") -> LedgerRecord:
        snap = self._snapshot()
        try:
            chart_id, _cmax = self.memory.ingest(key_wave, value_wave, label=label)
            self.boundary = _unit(self.boundary + np.asarray(value_wave, dtype=np.complex128))
            idx = self.memory.charts[chart_id].size - 1
            rec = LedgerRecord(
                record_id=self._next_id,
                bins=self.quantize(value_wave),
                label=label,
                chart_id=int(chart_id),
                local_index=int(idx),
            )
            self.store.append(rec)
            self.records[rec.record_id] = rec
            self._next_id += 1
            return rec
        except Exception:
            self._restore(snap)
            raise

    # ------------------------------------------------------------------ query
    def query(self, wave: np.ndarray, top_k_charts: int = 2) -> HHRGRetrievalResult:
        return self.memory.query(wave, top_k_charts=top_k_charts)

    # ---------------------------------------------------------------- JIT page
    def jit_page(self, query_wave: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, int]:
        """Return (V_jit in [-1.5, 1.5]^D, prefix, latency_ms, chart_hops)."""
        t0 = time.perf_counter()
        ret = self.query(query_wave)
        value = ret.matched_value if ret.matched_value is not None else ret.value_wave
        phase_diff = np.angle(np.asarray(value, dtype=np.complex128) * np.conj(
            np.asarray(query_wave, dtype=np.complex128).reshape(-1)
        ))
        v_jit = 1.5 * np.sin(phase_diff)
        w = np.asarray(value, dtype=np.complex128).reshape(-1)
        chunks = w.reshape(self.prefix_tokens, -1)
        prefix = (np.real(chunks) @ self.prefix_proj + np.imag(chunks) @ self.prefix_proj) / math.sqrt(2.0)
        hops = len(self.memory.charts)
        return v_jit, prefix, (time.perf_counter() - t0) * 1e3, hops

    # ------------------------------------------------------------ diagnostics
    @property
    def size(self) -> int:
        return self.memory.total_engrams

    def max_coherence(self) -> float:
        return max((c.c_max for c in self.memory.charts), default=0.0)
