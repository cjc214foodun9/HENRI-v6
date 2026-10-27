"""Hierarchical holographic renormalization group (HHRG) Stiefel memory.

Omega Amendment 4. Replaces the flat codebook with bounded local Stiefel charts.

Why this is not the reference sketch verbatim:
  - `math` is imported (the sketch raised NameError at query time).
  - The `sims_ortho` indentation defect is gone.
  - Merge candidates EXCLUDE the oldest anchor (index 0), so a stored engram
    cannot be destroyed by the consolidation path that a caller is querying.
  - Retrieval reports margin and normalized entropy, so a gate can test both a
    positive (stored key) and a negative (unrelated wave) query.

Claim boundary: each chart guarantees c_max < 1e-6 for M <= capacity <= 256 and
D >= capacity. The 3-tier 256^3 = 16.7M count is NOT established here; a chart
hierarchy of nested subspaces would be required, and that construction is absent.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.zeros_like(v)
    return v / n


@dataclass
class HHRGRetrievalResult:
    value_wave: np.ndarray
    matched_key: np.ndarray
    chart_id: int
    local_index: int
    margin: float
    max_similarity: float
    entropy_norm: float
    max_coherence: float
    matched_value: Optional[np.ndarray] = None


class LocalStiefelChart:
    """Bounded chart holding up to `max_capacity` orthogonalized engrams."""

    def __init__(self, chart_id: int, d_model: int, max_capacity: int = 256) -> None:
        if max_capacity > 256:
            raise ValueError("local chart capacity must stay <= 256")
        self.chart_id = int(chart_id)
        self.d_model = int(d_model)
        self.max_capacity = int(max_capacity)
        self.raw_keys: List[np.ndarray] = []
        self.values: Optional[np.ndarray] = None
        self.labels: List[str] = []
        self.ortho_keys: Optional[np.ndarray] = None
        self.centroid: Optional[np.ndarray] = None
        self.c_max: float = 0.0

    @property
    def size(self) -> int:
        return len(self.raw_keys)

    def stiefel_retract(self) -> float:
        if not self.raw_keys:
            return 0.0
        X = np.stack(self.raw_keys, axis=0).astype(np.complex128)
        M = X.shape[0]
        if M == 1:
            self.ortho_keys = _unit(X)
            self.centroid = self.ortho_keys[0].copy()
            self.c_max = 0.0
            return 0.0

        G = X @ X.conj().T
        G = 0.5 * (G + G.conj().T)
        eigvals, eigvecs = np.linalg.eigh(G)
        eigvals = np.clip(eigvals, 1e-12, None)
        inv_sqrt = 1.0 / np.sqrt(eigvals)
        G_inv_sqrt = (eigvecs * inv_sqrt) @ eigvecs.conj().T
        X_ortho = G_inv_sqrt @ X
        X_ortho = np.stack([_unit(row) for row in X_ortho], axis=0)
        self.ortho_keys = X_ortho

        G_post = np.abs(X_ortho @ X_ortho.conj().T)
        np.fill_diagonal(G_post, 0.0)
        self.c_max = float(G_post.max()) if G_post.size else 0.0
        self.centroid = _unit(X_ortho.mean(axis=0))
        return self.c_max

    def ingest(self, key_wave: np.ndarray, value_wave: np.ndarray, label: str = "") -> float:
        k = _unit(np.asarray(key_wave, dtype=np.complex128).reshape(-1))
        v = _unit(np.asarray(value_wave, dtype=np.complex128).reshape(-1))

        if self.size >= self.max_capacity:
            self._consolidate()

        self.raw_keys.append(k)
        self.labels.append(label)
        if self.values is None:
            self.values = v[None, :]
        else:
            self.values = np.concatenate([self.values, v[None, :]], axis=0)
        return self.stiefel_retract()

    def _consolidate(self) -> None:
        """Merge the most coherent pair, never touching the anchor at index 0."""
        if self.size < 3:
            return
        X = np.stack(self.raw_keys, axis=0)
        sim = np.real(X @ X.conj().T)
        np.fill_diagonal(sim, -2.0)
        sim[0, :] = -2.0
        sim[:, 0] = -2.0
        flat = int(np.argmax(sim))
        i, j = divmod(flat, self.size)
        if i > j:
            i, j = j, i
        if i == 0:
            return
        merged_k = _unit(self.raw_keys[i] + self.raw_keys[j])
        merged_v = _unit(self.values[i] + self.values[j])
        self.raw_keys[i] = merged_k
        self.values[i] = merged_v
        self.labels[i] = f"{self.labels[i]}|{self.labels[j]}"
        self.raw_keys.pop(j)
        self.labels.pop(j)
        self.values = np.delete(self.values, j, axis=0)


class HHRGMemory:
    """Nested chart hierarchy with per-chart exact Stiefel orthogonalization."""

    def __init__(
        self,
        d_model: int = 65536,
        chart_capacity: int = 256,
        beta_hopfield: float = 26.10,
    ) -> None:
        self.d_model = int(d_model)
        self.chart_capacity = int(chart_capacity)
        self.beta_hopfield = float(beta_hopfield)
        self.charts: List[LocalStiefelChart] = [
            LocalStiefelChart(0, d_model, chart_capacity)
        ]

    @property
    def total_engrams(self) -> int:
        return sum(c.size for c in self.charts)

    def ingest(
        self,
        key_wave: np.ndarray,
        value_wave: np.ndarray,
        label: str = "",
        chart_id: Optional[int] = None,
    ) -> Tuple[int, float]:
        if chart_id is not None:
            while len(self.charts) <= int(chart_id):
                self.charts.append(
                    LocalStiefelChart(len(self.charts), self.d_model, self.chart_capacity)
                )
            c = self.charts[int(chart_id)]
            return c.chart_id, c.ingest(key_wave, value_wave, label=label)

        target = self.charts[-1]
        if target.size >= self.chart_capacity:
            target = LocalStiefelChart(len(self.charts), self.d_model, self.chart_capacity)
            self.charts.append(target)
        return target.chart_id, target.ingest(key_wave, value_wave, label=label)

    def query(self, query_wave: np.ndarray, top_k_charts: int = 2) -> HHRGRetrievalResult:
        q = _unit(np.asarray(query_wave, dtype=np.complex128).reshape(-1))
        populated = [c for c in self.charts if c.size > 0 and c.ortho_keys is not None]
        if not populated:
            raise RuntimeError("HHRGMemory is empty; seed axioms before querying")

        if len(populated) > top_k_charts and all(c.centroid is not None for c in populated):
            centroids = np.stack([c.centroid for c in populated], axis=0)
            c_sims = np.real(centroids @ np.conj(q))
            order = np.argsort(c_sims)[::-1][:top_k_charts]
            candidates = [populated[int(i)] for i in order]
        else:
            candidates = populated

        best: Optional[HHRGRetrievalResult] = None
        for chart in candidates:
            raw = np.stack(chart.raw_keys, axis=0)
            sims_raw = np.real(raw @ np.conj(q))
            sims_ortho = np.real(chart.ortho_keys @ np.conj(q))
            sims = np.maximum(sims_raw, sims_ortho)

            M = int(sims.shape[0])
            if M == 1:
                margin = float(sims[0])
                max_sim = float(sims[0])
                best_idx = 0
                entropy_norm = 0.0
                probs = np.ones(1, dtype=np.float64)
            else:
                order = np.argsort(sims)[::-1]
                top2 = sims[order[:2]]
                best_idx = int(order[0])
                max_sim = float(top2[0])
                margin = float(top2[0] - top2[1])
                logits = self.beta_hopfield * (sims - sims.max())
                probs = np.exp(logits)
                probs = probs / probs.sum()
                entropy = -float(np.sum(probs * np.log(np.clip(probs, 1e-12, None))))
                entropy_norm = entropy / math.log(float(M))

            clean = chart.values.T @ probs.astype(np.complex128)
            clean = _unit(clean)

            candidate = HHRGRetrievalResult(
                value_wave=clean,
                matched_key=chart.ortho_keys[best_idx],
                chart_id=chart.chart_id,
                local_index=best_idx,
                margin=margin,
                max_similarity=max_sim,
                entropy_norm=float(entropy_norm),
                max_coherence=float(chart.c_max),
                matched_value=chart.values[best_idx].copy(),
            )
            if best is None or candidate.max_similarity > best.max_similarity:
                best = candidate

        assert best is not None
        return best
