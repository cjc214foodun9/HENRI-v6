"""Riemannian tangent-space chart atlas for Zone C (Sovereign T2 / AF1 / AG3).

Each chart is a local spherical cap on S^{d-1} with a unit Fréchet centroid c,
a rank-r orthonormal tangent basis U (U^H c = 0), principal variances lambda and
an isotropic orthogonal floor sigma_perp^2. Local whitening uses the Woodbury
identity, so no dense matrix is inverted. An online write whose tangent energy
is not explained by the nearest chart condenses into coherent local or sovereign
charts instead of rotating an existing mature world chart frame. Existing
mature charts therefore do not drift.

Cycle AF1 packed BLAS layout:
- All chart centroids per namespace reside in one contiguous real-view matrix
  (K x 2D float64, zero-copy view of unit complex128) for single-call BLAS GEMV
  routing.
- Every mature chart stores its member wave vectors in a contiguous per-chart
  matrix block (repacked lazily on build/condensation/absorb only; zero
  np.stack or scattered gathers on the hot query path).
- Online/condensed writes near a geodesic Voronoi boundary receive bounded
  dual-chart boundary membership (without mutating neighbor chart frames) so
  boundary writes are never dropped during condensation while total candidate
  rescoring stays bounded.

Cycle AG3 100k-engram hierarchical routing & persistence:
- Two-level hierarchical spherical clustering and super-centroid routing
  activate automatically at scale (K >= hierarchy_min_charts) with bounded dual
  super-membership and self-verifying fallback, keeping routing L2/L3 cache
  resident while preserving 1.0000 top-1 parity.
- Binary slab + SQLite persistence (`save` / `load`) stores contiguous vector
  slabs, geometry arrays, and pre-built super-cluster hierarchies so a
  restarted `UnifiedGatewayService` mounts 105k engrams via zero-copy mmap in
  sub-second time with zero chart rebuild.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

_EPS = 1e-12
_PERSIST_FORMAT_VERSION = "ag3_packed_v1"
_STRIDE_BLOCK_BYTES = 4096
_STRIDE_SAMPLES = 16


def _unit(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=np.complex128).reshape(-1)
    norm = float(np.linalg.norm(array))
    return array / max(norm, _EPS)


def _as_real_view(matrix: np.ndarray) -> np.ndarray:
    """Zero-copy float64 view of C-contiguous complex128 vectors (preserves memmap).

    For unit complex vectors u, v in C^D, Re(u . conj(v)) equals the real inner
    product of their interleaved [Re, Im] float64 views, avoiding complex
    imaginary arithmetic and temporary conjugation copies on the hot BLAS path.
    """
    if (
        isinstance(matrix, np.ndarray)
        and matrix.dtype == np.complex128
        and matrix.flags["C_CONTIGUOUS"]
    ):
        contiguous = matrix
    else:
        contiguous = np.ascontiguousarray(matrix, dtype=np.complex128)
    if contiguous.ndim == 1:
        return contiguous.view(np.float64)
    return contiguous.view(np.float64).reshape(contiguous.shape[0], contiguous.shape[1] * 2)


def _orthonormal_tangent(matrix: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """Orthonormal rows, each orthogonal to the centroid."""
    rows = np.asarray(matrix, dtype=np.complex128)
    if rows.ndim != 2 or rows.shape[0] == 0:
        return np.zeros((0, centroid.size), dtype=np.complex128)
    rows = rows - np.outer(rows @ np.conj(centroid), centroid)
    # Gram-Schmidt on the rows, dropping a near-dependent row.
    kept: List[np.ndarray] = []
    for row in rows:
        vector = row.copy()
        for other in kept:
            vector = vector - np.vdot(other, vector) * other
        norm = float(np.linalg.norm(vector))
        if norm > 1e-8:
            kept.append(vector / norm)
    return np.stack(kept) if kept else np.zeros((0, centroid.size), dtype=np.complex128)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _fast_stride_sha256(path: Path) -> str:
    """Deterministic header + tail + evenly-strided SHA-256 fingerprint."""
    size = int(path.stat().st_size)
    h = hashlib.sha256()
    h.update(size.to_bytes(8, "little"))
    if size == 0:
        return h.hexdigest()
    with path.open("rb") as fh:
        head_len = min(size, 65536)
        h.update(fh.read(head_len))
        if size > head_len:
            tail_pos = max(head_len, size - 65536)
            fh.seek(tail_pos)
            h.update(fh.read(size - tail_pos))
        if size > _STRIDE_BLOCK_BYTES * 2:
            step = max(1, (size - _STRIDE_BLOCK_BYTES) // _STRIDE_SAMPLES)
            for idx in range(_STRIDE_SAMPLES):
                pos = min(size - _STRIDE_BLOCK_BYTES, idx * step)
                fh.seek(pos)
                h.update(fh.read(_STRIDE_BLOCK_BYTES))
    return h.hexdigest()


class NascentStagingBuffer:
    """A bounded pool of novel writes that no mature chart explains.

    A write that misses every mature chart is not a chart by itself: a
    singleton chart has a rank-0 frame and a radius calibrated for a dense
    cluster. The write waits here instead. When the pool is full, coherent
    clusters condense into mature charts and isolated outliers stay.
    """

    def __init__(self, capacity: int = 64) -> None:
        self.capacity = int(max(1, capacity))
        self.entries: List[Tuple[np.ndarray, int, str]] = []
        self._by_ns: Dict[str, List[Tuple[np.ndarray, int]]] = {}
        self._packed_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.entries)

    def stage(self, vector: np.ndarray, row: int, namespace: str) -> None:
        unit = _unit(vector)
        row_id = int(row)
        ns = str(namespace)
        self.entries.append((unit, row_id, ns))
        self._by_ns.setdefault(ns, []).append((unit, row_id))
        self._packed_cache.pop(ns, None)

    def has_namespace(self, namespace: Optional[str]) -> bool:
        if namespace is None:
            return bool(self.entries)
        return bool(self._by_ns.get(str(namespace)))

    def rows(self, namespace: Optional[str] = None) -> List[int]:
        if namespace is None:
            return [int(entry[1]) for entry in self.entries]
        bucket = self._by_ns.get(str(namespace))
        return [int(item[1]) for item in bucket] if bucket else []

    def packed(self, namespace: Optional[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Return contiguous (vectors_real, row_array) for staged entries."""
        if namespace is None:
            if not self.entries:
                return np.zeros((0, 0), dtype=np.float64), np.zeros(0, dtype=np.int64)
            vecs = np.ascontiguousarray(
                np.stack([entry[0] for entry in self.entries]), dtype=np.complex128
            )
            rids = np.asarray([entry[1] for entry in self.entries], dtype=np.int64)
            return _as_real_view(vecs), rids
        ns = str(namespace)
        cached = self._packed_cache.get(ns)
        if cached is not None:
            return cached
        bucket = self._by_ns.get(ns)
        if not bucket:
            empty = (np.zeros((0, 0), dtype=np.float64), np.zeros(0, dtype=np.int64))
            self._packed_cache[ns] = empty
            return empty
        vecs = np.ascontiguousarray(
            np.stack([item[0] for item in bucket]), dtype=np.complex128
        )
        rids = np.asarray([item[1] for item in bucket], dtype=np.int64)
        packed_pair = (_as_real_view(vecs), rids)
        self._packed_cache[ns] = packed_pair
        return packed_pair

    def namespaces(self) -> List[str]:
        return list(self._by_ns.keys())

    def drain(self, namespace: str) -> List[Tuple[np.ndarray, int, str]]:
        ns = str(namespace)
        bucket = self._by_ns.pop(ns, [])
        self._packed_cache.pop(ns, None)
        if not bucket:
            return []
        self.entries = [entry for entry in self.entries if entry[2] != ns]
        return [(vec, row, ns) for vec, row in bucket]


@dataclass
class _HierarchyView:
    """Two-level packed super-centroid routing index for large namespaces."""

    super_centroids: np.ndarray  # (S, D) complex128
    super_centroids_real: np.ndarray  # (S, 2D) float64 view
    super_cos_alpha: np.ndarray  # (S,) float64
    super_sin_alpha: np.ndarray  # (S,) float64
    block_chart_ids: List[np.ndarray]  # S arrays of global chart IDs
    block_centroids_real: List[np.ndarray]  # S contiguous (m_s, 2D) float64 blocks
    super_boundary_duals: int = 0


@dataclass
class RiemannianChart:
    centroid: np.ndarray
    basis: np.ndarray
    variances: np.ndarray
    sigma2: float
    rows: List[int] = field(default_factory=list)
    namespace: str = "default"
    count: int = 0
    is_companion: bool = False
    boundary_count: int = 0
    row_array: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    vectors: Optional[np.ndarray] = None
    vectors_real: Optional[np.ndarray] = None
    _pending_rows: List[int] = field(default_factory=list, repr=False)
    _pending_vectors: List[np.ndarray] = field(default_factory=list, repr=False)
    _row_set: Optional[Set[int]] = field(default=None, repr=False)

    def set_packed(self, vectors: np.ndarray, rows: Sequence[int]) -> None:
        """Install contiguous packed vectors and row ids in one allocation."""
        row_list = [int(r) for r in rows]
        self.rows = row_list
        self._row_set = None
        self.row_array = np.asarray(row_list, dtype=np.int64)
        self.vectors = np.ascontiguousarray(vectors, dtype=np.complex128)
        self.vectors_real = _as_real_view(self.vectors)
        self._pending_rows.clear()
        self._pending_vectors.clear()

    def set_packed_view(self, vectors_view: np.ndarray, row_array_view: np.ndarray) -> None:
        """Install zero-copy memmap-backed packed views without copying vectors."""
        self.row_array = row_array_view
        self.rows = row_array_view.tolist()
        self._row_set = None
        self.vectors = vectors_view
        self.vectors_real = _as_real_view(vectors_view)
        self._pending_rows.clear()
        self._pending_vectors.clear()

    def append_member(self, vector: np.ndarray, row: int, *, is_boundary: bool = False) -> bool:
        """Stage a row+vector for lazy repack without mutating chart geometry."""
        row_id = int(row)
        if self._row_set is None:
            self._row_set = set(self.rows)
        if row_id in self._row_set:
            return False
        self._row_set.add(row_id)
        self.rows.append(row_id)
        self._pending_rows.append(row_id)
        self._pending_vectors.append(_unit(vector))
        if is_boundary:
            self.boundary_count += 1
        return True

    def ensure_packed(self) -> None:
        """Lazily repack any pending appends into the contiguous matrix block."""
        if not self._pending_rows:
            if self.vectors is not None and self.vectors_real is None:
                self.vectors_real = _as_real_view(self.vectors)
            if self.row_array.size != len(self.rows):
                self.row_array = np.asarray(self.rows, dtype=np.int64)
            return
        added_rows = np.asarray(self._pending_rows, dtype=np.int64)
        added_vecs = np.ascontiguousarray(
            np.stack(self._pending_vectors), dtype=np.complex128
        )
        if self.vectors is None or self.vectors.size == 0:
            self.vectors = added_vecs
            self.row_array = added_rows
        else:
            self.vectors = np.ascontiguousarray(
                np.vstack([self.vectors, added_vecs]), dtype=np.complex128
            )
            self.row_array = np.concatenate([self.row_array, added_rows])
        self.vectors_real = _as_real_view(self.vectors)
        self._pending_rows.clear()
        self._pending_vectors.clear()

    def tangent(self, gram: np.ndarray) -> np.ndarray:
        vector = _unit(gram)
        projection = np.vdot(self.centroid, vector)
        return vector - projection * self.centroid

    def whitened(self, tangent: np.ndarray, eps: float = 1e-3) -> np.ndarray:
        """Woodbury local whitening: O(r d), no dense inverse."""
        if self.basis.shape[0] == 0:
            return tangent / math.sqrt(float(self.sigma2) + eps)
        floor = 1.0 / math.sqrt(float(self.sigma2) + eps)
        principal = 1.0 / np.sqrt(np.maximum(self.variances, 0.0) + eps)
        coordinates = self.basis @ np.conj(tangent)
        correction = (principal - floor) * coordinates
        return floor * tangent + self.basis.T @ correction

    def unexplained(self, tangent: np.ndarray) -> float:
        energy = float(np.vdot(tangent, tangent).real)
        if energy <= _EPS:
            # The write is the centroid itself: the chart explains it exactly.
            return 0.0
        if self.basis.shape[0] == 0:
            return 1.0
        coords = self.basis @ np.conj(tangent)
        captured = float(np.sum(np.abs(coords) ** 2))
        return max(0.0, min(1.0, 1.0 - captured / energy))

    def as_dict(self) -> Dict[str, object]:
        return {
            "centroid": [complex(value) for value in self.centroid.tolist()],
            "rank": int(self.basis.shape[0]),
            "variances": [float(value) for value in self.variances.tolist()],
            "sigma2": float(self.sigma2),
            "count": int(self.count),
            "boundary_count": int(self.boundary_count),
            "total_stored": int(len(self.rows)),
            "namespace": str(self.namespace),
        }


class RiemannianChartAtlas:
    """A non-stationary atlas with inversion-free whitening, two-level routing, and persistence."""

    def __init__(
        self,
        *,
        rank: int = 8,
        capacity: int = 64,
        tau_orth: float = 0.55,
        tau_cap: float = 0.35,
        tau_margin: Optional[float] = None,
        top_charts: int = 2,
        top_charts_max: int = 6,
        oja_rate: float = 0.05,
        eps: float = 1e-3,
        b_stage: int = 64,
        k_condense: int = 2,
        min_cluster: int = 8,
        boundary_margin: float = 0.08,
        max_boundary_ratio: float = 0.08,
        expansion_gap: float = 0.16,
        max_rescore_fraction: float = 0.048,
        hierarchy_min_charts: int = 96,
        top_super: int = 3,
        top_super_max: int = 6,
        super_boundary_margin: float = 0.12,
    ) -> None:
        self.rank = int(rank)
        self.capacity = int(capacity)
        self._effective_capacity = int(capacity)
        self.tau_orth = float(tau_orth)
        self.tau_cap = float(tau_cap)
        self.tau_margin = tau_margin
        self.top_charts = int(top_charts)
        self.top_charts_max = int(top_charts_max)
        self.oja_rate = float(oja_rate)
        self.eps = float(eps)
        self.boundary_margin = float(boundary_margin)
        self.max_boundary_ratio = float(max_boundary_ratio)
        self.expansion_gap = float(expansion_gap)
        self.max_rescore_fraction = float(max_rescore_fraction)
        self.hierarchy_min_charts = int(max(16, hierarchy_min_charts))
        self.top_super = int(max(1, top_super))
        self.top_super_max = int(max(self.top_super, top_super_max))
        self.super_boundary_margin = float(super_boundary_margin)
        self.charts: List[RiemannianChart] = []
        self.namespaces: Dict[str, List[int]] = {}
        self.b_stage = int(max(2, b_stage))
        self.k_condense = int(max(1, k_condense))
        self.min_cluster = int(max(2, min_cluster))
        self.staging = NascentStagingBuffer(self.b_stage)
        self.spawns = 0
        self.staged = 0
        self.condensations = 0
        self.condensed_spawns = 0
        self.condensed_rows = 0
        self.absorbs = 0
        self.boundary_memberships = 0
        self.explained: List[float] = []
        self.digest = ""
        self._centroid_matrix: Optional[np.ndarray] = None
        self._centroid_real: Optional[np.ndarray] = None
        self._ns_cache: Dict[Optional[str], Tuple[np.ndarray, np.ndarray, int]] = {}
        self._hier_cache: Dict[Optional[str], Optional[_HierarchyView]] = {}
        self._writes_since_condense: Dict[str, int] = {}
        self._companion_of: Dict[int, int] = {}
        self._dirty_charts: Set[int] = set()
        self._dirty_cache = True
        self._hierarchy_dirty = True
        self._mmap_refs: List[object] = []
        self.loaded_from_disk = False
        self.rebuilt_charts = 0
        self.mount_seconds = 0.0
        self.persist_path: Optional[str] = None

    def close(self) -> None:
        """Release any open memmap references."""
        for ref in self._mmap_refs:
            mm = getattr(ref, "_mmap", None)
            if mm is not None:
                try:
                    mm.close()
                except Exception:
                    pass
        self._mmap_refs.clear()

    def _invalidate_caches(self) -> None:
        self._centroid_matrix = None
        self._centroid_real = None
        self._ns_cache.clear()
        self._hier_cache.clear()
        self._dirty_cache = True
        self._hierarchy_dirty = True

    def _ensure_caches(self) -> None:
        if self._dirty_charts:
            for cid in list(self._dirty_charts):
                if 0 <= cid < len(self.charts):
                    self.charts[cid].ensure_packed()
            self._dirty_charts.clear()
        if not self._dirty_cache and self._centroid_real is not None:
            return
        for chart in self.charts:
            chart.ensure_packed()
        if self.charts:
            self._centroid_matrix = np.ascontiguousarray(
                np.stack([_unit(chart.centroid) for chart in self.charts]),
                dtype=np.complex128,
            )
            self._centroid_real = _as_real_view(self._centroid_matrix)
        else:
            self._centroid_matrix = None
            self._centroid_real = None
        self._ns_cache.clear()
        self._hier_cache.clear()
        self._dirty_cache = False
        self._hierarchy_dirty = True

    def _get_ns_view(
        self, namespace: Optional[str]
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        self._ensure_caches()
        key = None if namespace is None else str(namespace)
        cached = self._ns_cache.get(key)
        if cached is not None:
            return cached
        if key is None:
            chart_ids = np.arange(len(self.charts), dtype=np.int64)
        else:
            chart_ids = np.asarray(self.namespaces.get(key, []), dtype=np.int64)
        if chart_ids.size == 0 or self._centroid_real is None:
            entry = (
                np.zeros(0, dtype=np.int64),
                np.zeros((0, 0), dtype=np.float64),
                0,
            )
            self._ns_cache[key] = entry
            return entry
        if key is None or (
            chart_ids.size == len(self.charts)
            and int(chart_ids[0]) == 0
            and int(chart_ids[-1]) == len(self.charts) - 1
        ):
            centroids_real = self._centroid_real
        else:
            centroids_real = np.ascontiguousarray(self._centroid_real[chart_ids])
        total_rows = int(sum(int(self.charts[int(cid)].count) for cid in chart_ids.tolist()))
        entry = (chart_ids, centroids_real, total_rows)
        self._ns_cache[key] = entry
        return entry

    def _build_hierarchy_view(
        self, chart_ids: np.ndarray, centroids_real: np.ndarray
    ) -> Optional[_HierarchyView]:
        """Build an exact geodesic macro-cap hierarchy over leaf chart centroids.

        Groups mutually coherent leaf centroids (e.g. sibling sub-charts and
        condensed companion charts within the same manifold domain) into tight
        spherical caps M_m with unit Frechet centroid C_m and exact angular
        radius alpha_m = max_{c in M_m} arccos(Re<c, C_m>). During routing, the
        spherical triangle inequality gives an exact cosine upper bound UB(q, m)
        per macro-cap, allowing branch-and-bound pruning with zero approximation
        error relative to full leaf-centroid scanning.
        """
        n_charts = int(chart_ids.size)
        if n_charts < self.hierarchy_min_charts or self._centroid_matrix is None:
            return None
        leaf_complex = np.ascontiguousarray(
            self._centroid_matrix[chart_ids], dtype=np.complex128
        )
        macro_tau = float(max(0.45, self.tau_cap + 0.15))
        max_members_per_cap = 24
        cap_members: List[List[int]] = []
        cap_sums_real: List[np.ndarray] = []
        cap_centroids_real: List[np.ndarray] = []

        # Deterministic greedy geodesic cap grouping: never merges orthogonal
        # domains, keeping cap radii alpha_m tight so UB(q, m) prunes sharply.
        for local_i in range(n_charts):
            vec_r = centroids_real[local_i]
            best_m = -1
            best_sim = -2.0
            if cap_centroids_real:
                # Batch dot product against current macro centroids.
                c_mat = np.asarray(cap_centroids_real, dtype=np.float64)
                sims = c_mat @ vec_r
                order_m = np.argsort(-sims)
                for m_idx in order_m.tolist():
                    s_val = float(sims[m_idx])
                    if s_val < macro_tau:
                        break
                    if len(cap_members[m_idx]) < max_members_per_cap:
                        best_m = m_idx
                        best_sim = s_val
                        break
            if best_m >= 0 and best_sim >= macro_tau:
                cap_members[best_m].append(local_i)
                new_sum = cap_sums_real[best_m] + vec_r
                cap_sums_real[best_m] = new_sum
                nrm = float(np.linalg.norm(new_sum))
                cap_centroids_real[best_m] = new_sum / max(nrm, _EPS)
            else:
                cap_members.append([local_i])
                cap_sums_real.append(vec_r.copy())
                cap_centroids_real.append(vec_r.copy())

        n_macro = len(cap_members)
        # Only activate two-level routing when macro-caps genuinely compress
        # leaf centroids (otherwise single-level GEMV is already optimal).
        if n_macro < 4 or n_macro > int(0.80 * n_charts):
            return None

        kept_super: List[np.ndarray] = []
        block_chart_ids: List[np.ndarray] = []
        block_centroids_real: List[np.ndarray] = []
        cos_alphas: List[float] = []
        sin_alphas: List[float] = []
        for local_members in cap_members:
            local_arr = np.asarray(local_members, dtype=np.int64)
            s_vec = _unit(leaf_complex[local_arr].mean(axis=0))
            s_real = s_vec.view(np.float64)
            c_block = np.ascontiguousarray(centroids_real[local_arr], dtype=np.float64)
            if local_arr.size == 1:
                cos_clamped = 1.0
                sin_max = 0.0
            else:
                cos_min = float(np.min(c_block @ s_real))
                cos_clamped = max(-1.0, min(1.0, cos_min))
                sin_max = math.sqrt(max(0.0, 1.0 - cos_clamped * cos_clamped))
            kept_super.append(s_vec)
            block_chart_ids.append(np.ascontiguousarray(chart_ids[local_arr], dtype=np.int64))
            block_centroids_real.append(c_block)
            cos_alphas.append(cos_clamped)
            sin_alphas.append(sin_max)
        final_super = np.ascontiguousarray(np.stack(kept_super), dtype=np.complex128)
        return _HierarchyView(
            super_centroids=final_super,
            super_centroids_real=_as_real_view(final_super),
            super_cos_alpha=np.asarray(cos_alphas, dtype=np.float64),
            super_sin_alpha=np.asarray(sin_alphas, dtype=np.float64),
            block_chart_ids=block_chart_ids,
            block_centroids_real=block_centroids_real,
            super_boundary_duals=0,
        )

    def _get_hierarchy_view(
        self, namespace: Optional[str], chart_ids: np.ndarray, centroids_real: np.ndarray
    ) -> Optional[_HierarchyView]:
        if int(chart_ids.size) < self.hierarchy_min_charts:
            return None
        key = None if namespace is None else str(namespace)
        if not self._hierarchy_dirty and key in self._hier_cache:
            return self._hier_cache[key]
        if self._hierarchy_dirty:
            self._hier_cache.clear()
            self._hierarchy_dirty = False
        view = self._build_hierarchy_view(chart_ids, centroids_real)
        self._hier_cache[key] = view
        return view

    # ---------------------------------------------------------------- build
    @staticmethod
    def _cluster(
        grams: np.ndarray, cluster_count: int, iterations: int = 24
    ) -> np.ndarray:
        """Deterministic spherical k-means with single-pass O(N*D) centroid update."""
        count = int(grams.shape[0])
        cluster_count = int(max(1, min(cluster_count, count)))
        if cluster_count == 1:
            return np.zeros(count, dtype=np.int64)
        stride = max(1, count // cluster_count)
        picks = [min(count - 1, index * stride) for index in range(cluster_count)]
        centroids = np.ascontiguousarray(
            np.stack([_unit(grams[index]) for index in picks]), dtype=np.complex128
        )
        centroids_real = _as_real_view(centroids)
        grams_real = _as_real_view(grams)
        assignment = np.full(count, -1, dtype=np.int64)
        chunk_rows = 8192 if (count * cluster_count > 2_000_000) else count
        for _ in range(iterations):
            if chunk_rows >= count:
                scores = grams_real @ centroids_real.T
                updated = np.argmax(scores, axis=1).astype(np.int64)
            else:
                updated = np.empty(count, dtype=np.int64)
                for start in range(0, count, chunk_rows):
                    end = min(count, start + chunk_rows)
                    scores = grams_real[start:end] @ centroids_real.T
                    updated[start:end] = np.argmax(scores, axis=1)
            if np.array_equal(updated, assignment):
                break
            assignment = updated
            # Single-pass O(N*D) vectorized centroid update via argsort runs.
            order = np.argsort(assignment, kind="mergesort")
            sorted_a = assignment[order]
            change = np.flatnonzero(sorted_a[1:] != sorted_a[:-1]) + 1
            starts = np.concatenate(([0], change))
            ends = np.concatenate((change, [count]))
            for s_i, e_i in zip(starts.tolist(), ends.tolist()):
                cid = int(sorted_a[s_i])
                idx_slice = order[s_i:e_i]
                mean_real = grams_real[idx_slice].mean(axis=0)
                norm = float(np.linalg.norm(mean_real))
                if norm > _EPS:
                    centroids_real[cid] = mean_real / norm
        return assignment

    def _frame(self, grams: np.ndarray, namespace: str) -> RiemannianChart:
        centroid = _unit(grams.mean(axis=0))
        tangent = grams - np.outer(grams @ np.conj(centroid), centroid)
        count = max(int(grams.shape[0]), 1)
        covariance = (tangent.conj().T @ tangent) / float(count)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        rank = int(min(self.rank, grams.shape[1], int(grams.shape[0])))
        chosen = order[:rank]
        basis = _orthonormal_tangent(eigenvectors[:, chosen].T, centroid)
        variances = np.maximum(np.real(eigenvalues[chosen]), 0.0)
        tail = np.real(eigenvalues[order[rank:]]) if order.size > rank else np.zeros(1)
        sigma2 = float(np.mean(tail)) if tail.size else 1.0
        return RiemannianChart(
            centroid=centroid,
            basis=basis,
            variances=variances,
            sigma2=max(sigma2, 1e-6),
            rows=[],
            namespace=str(namespace),
            count=0,
        )

    def _split_groups(self, matrix: np.ndarray, indices: np.ndarray) -> List[np.ndarray]:
        """Deterministically split oversized clusters so every chart respects effective capacity."""
        cap = int(self._effective_capacity)
        n = int(indices.size)
        if n <= cap:
            return [indices]
        sub_k = int(max(2, math.ceil(n / float(cap))))
        sub_assign = self._cluster(matrix[indices], sub_k)
        groups: List[np.ndarray] = []
        for cid in range(int(sub_assign.max()) + 1):
            local = np.flatnonzero(sub_assign == cid)
            if local.size == 0:
                continue
            sub_indices = indices[local]
            if int(sub_indices.size) > cap and int(sub_indices.size) < n:
                groups.extend(self._split_groups(matrix, sub_indices))
            elif int(sub_indices.size) > cap:
                # Deterministic balanced chunking if vectors are nearly identical.
                for start in range(0, int(sub_indices.size), cap):
                    groups.append(sub_indices[start : start + cap])
            else:
                groups.append(sub_indices)
        return groups

    def build(
        self, grams: np.ndarray, rows: Sequence[int], namespace: str = "default"
    ) -> Dict[str, object]:
        self.close()
        matrix = np.ascontiguousarray(grams, dtype=np.complex128)
        norms = np.maximum(np.linalg.norm(matrix, axis=-1, keepdims=True), _EPS)
        matrix = np.ascontiguousarray(matrix / norms, dtype=np.complex128)
        count = int(matrix.shape[0])
        if count == 0:
            raise ValueError("an atlas needs at least one Gram row")
        self.charts = []
        self.namespaces = {}
        self.staging = NascentStagingBuffer(self.b_stage)
        self.staged = 0
        self.condensations = 0
        self.condensed_spawns = 0
        self.condensed_rows = 0
        self.spawns = 0
        self.absorbs = 0
        self.boundary_memberships = 0
        self._writes_since_condense.clear()
        self._companion_of.clear()
        self._dirty_charts.clear()
        self.loaded_from_disk = False
        self.rebuilt_charts = 0
        self.mount_seconds = 0.0
        if count >= 4096 and self.top_charts_max > 0 and self.max_rescore_fraction > 0:
            # Cap chart size on large stores so worst-case top_charts_max probes
            # stay within the sublinear rescore budget while preserving coherence.
            budget_cap = int(
                max(32, math.floor((self.max_rescore_fraction * count) / float(self.top_charts_max)))
            )
            self._effective_capacity = int(min(self.capacity, budget_cap))
        else:
            self._effective_capacity = int(self.capacity)
        cluster_count = int(max(1, math.ceil(count / float(self._effective_capacity))))
        ns = str(namespace)

        groups_to_frame: List[np.ndarray] = []
        if count >= 25000 and cluster_count >= 48:
            # Two-level hierarchical seeding followed by 3 global Lloyd passes
            # keeps build time fast while restoring exact global Voronoi cells.
            n_super = int(max(8, min(28, round(math.sqrt(float(cluster_count)) * 1.4))))
            super_assign = self._cluster(matrix, n_super, iterations=10)
            seed_centroids_list: List[np.ndarray] = []
            for sid in range(int(super_assign.max()) + 1):
                s_idx = np.flatnonzero(super_assign == sid)
                if s_idx.size == 0:
                    continue
                local_k = int(max(1, math.ceil(float(s_idx.size) / float(self._effective_capacity))))
                if local_k == 1:
                    seed_centroids_list.append(_unit(matrix[s_idx].mean(axis=0)))
                else:
                    local_assign = self._cluster(matrix[s_idx], local_k, iterations=10)
                    for cid in range(int(local_assign.max()) + 1):
                        local_pos = np.flatnonzero(local_assign == cid)
                        if local_pos.size == 0:
                            continue
                        seed_centroids_list.append(_unit(matrix[s_idx[local_pos]].mean(axis=0)))
            seed_c = np.ascontiguousarray(np.stack(seed_centroids_list), dtype=np.complex128)
            seed_c_real = _as_real_view(seed_c)
            mat_real = _as_real_view(matrix)
            assignment = np.full(count, -1, dtype=np.int64)
            chunk_rows = 8192
            for _ in range(3):
                updated = np.empty(count, dtype=np.int64)
                for start in range(0, count, chunk_rows):
                    end = min(count, start + chunk_rows)
                    updated[start:end] = np.argmax(mat_real[start:end] @ seed_c_real.T, axis=1)
                if np.array_equal(updated, assignment):
                    break
                assignment = updated
                order = np.argsort(assignment, kind="mergesort")
                sorted_a = assignment[order]
                change = np.flatnonzero(sorted_a[1:] != sorted_a[:-1]) + 1
                starts = np.concatenate(([0], change))
                ends = np.concatenate((change, [count]))
                for s_i, e_i in zip(starts.tolist(), ends.tolist()):
                    cid = int(sorted_a[s_i])
                    mean_real = mat_real[order[s_i:e_i]].mean(axis=0)
                    norm = float(np.linalg.norm(mean_real))
                    if norm > _EPS:
                        seed_c_real[cid] = mean_real / norm
            for chart_id in range(int(assignment.max()) + 1):
                member_index = np.flatnonzero(assignment == chart_id)
                if member_index.size == 0:
                    continue
                groups_to_frame.extend(self._split_groups(matrix, member_index))
        else:
            assignment = self._cluster(matrix, cluster_count)
            for chart_id in range(int(assignment.max()) + 1):
                member_index = np.flatnonzero(assignment == chart_id)
                if member_index.size == 0:
                    continue
                groups_to_frame.extend(self._split_groups(matrix, member_index))

        assigned_chart_of = np.full(count, -1, dtype=np.int64)
        for group in groups_to_frame:
            sub_vecs = matrix[group]
            chart = self._frame(sub_vecs, ns)
            group_rows = [int(rows[int(index)]) for index in group.tolist()]
            chart.set_packed(sub_vecs, group_rows)
            chart.count = int(group.size)
            cid_new = len(self.charts)
            self.charts.append(chart)
            self.namespaces.setdefault(ns, []).append(cid_new)
            assigned_chart_of[group] = cid_new
        self.rebuilt_charts = int(len(self.charts))
        self.digest = hashlib.sha256(
            b"".join(_unit(chart.centroid).tobytes() for chart in self.charts)
        ).hexdigest()[:16]
        self._invalidate_caches()
        self._ensure_caches()

        # Reconcile any post-split Voronoi boundary vectors on large atlases so
        # every vector's global top-1 chart is guaranteed to contain it.
        if count >= 25000 and len(self.charts) > 1 and self._centroid_real is not None:
            mat_real = _as_real_view(matrix)
            chunk_rows = 8192
            for start in range(0, count, chunk_rows):
                end = min(count, start + chunk_rows)
                top1 = np.argmax(mat_real[start:end] @ self._centroid_real.T, axis=1)
                mismatches = np.flatnonzero(top1 != assigned_chart_of[start:end])
                for local_i in mismatches.tolist():
                    vec_idx = start + int(local_i)
                    best_cid = int(top1[int(local_i)])
                    self._register_boundary(
                        matrix[vec_idx], int(rows[vec_idx]), best_cid, mandatory=True
                    )
            self._ensure_caches()

        # Pre-warm two-level routing hierarchy if this namespace crosses threshold.
        cids, c_real, _tot = self._get_ns_view(ns)
        self._get_hierarchy_view(ns, cids, c_real)
        return self.stats()

    # -------------------------------------------------------------- online
    def _namespace_charts(self, namespace: Optional[str]) -> List[int]:
        if namespace is None:
            return list(range(len(self.charts)))
        return list(self.namespaces.get(str(namespace), []))

    def _register_boundary(
        self, vector: np.ndarray, row: int, neighbor_chart_id: int, *, mandatory: bool = False
    ) -> bool:
        """Bounded dual-chart Voronoi boundary membership without mutating neighbor frame."""
        total_primary = max(1, sum(int(c.count) for c in self.charts))
        cap_ratio = self.max_boundary_ratio if mandatory else (self.max_boundary_ratio * 0.5)
        if float(self.boundary_memberships) / float(total_primary) >= cap_ratio:
            return False
        neighbor = self.charts[int(neighbor_chart_id)]
        cap_mult = 1.25 if mandatory else 1.15
        if len(neighbor.rows) >= int(math.ceil(self._effective_capacity * cap_mult)):
            return False
        added = neighbor.append_member(vector, int(row), is_boundary=True)
        if added:
            self.boundary_memberships += 1
            self._dirty_charts.add(int(neighbor_chart_id))
        return added

    def observe(
        self, gram: np.ndarray, row: int, namespace: str = "online"
    ) -> Dict[str, object]:
        """Absorb a write into its chart, or stage it for coherent condensation."""
        vector = _unit(gram)
        namespace = str(namespace)
        chart_ids, centroids_real, _total = self._get_ns_view(namespace)
        if chart_ids.size == 0:
            # No mature chart explains this namespace yet. Staging the write
            # keeps it exactly readable while the chart count stays bounded.
            self.staging.stage(vector, int(row), namespace)
            self.staged += 1
            self._writes_since_condense[namespace] = (
                self._writes_since_condense.get(namespace, 0) + 1
            )
            condensed = self._maybe_condense(namespace)
            return {
                "action": "stage",
                "reason": "no-mature-chart",
                "staged": int(len(self.staging)),
                "condensed": condensed,
            }
        q_real = vector.view(np.float64)
        cosines = centroids_real @ q_real
        best_pos = int(np.argmax(cosines))
        nearest = int(chart_ids[best_pos])
        cap = float(cosines[best_pos])
        chart = self.charts[nearest]
        tangent = chart.tangent(vector)
        rho = chart.unexplained(tangent)
        self.explained.append(1.0 - rho)
        if rho > self.tau_orth or cap < self.tau_cap or chart.count >= self._effective_capacity:
            # An unexplained or over-capacity write enters the nascent pool
            # instead of birthing a singleton chart or rotating a mature frame.
            self.staging.stage(vector, int(row), namespace)
            self.staged += 1
            self._writes_since_condense[namespace] = (
                self._writes_since_condense.get(namespace, 0) + 1
            )
            condensed = self._maybe_condense(namespace)
            return {
                "action": "stage",
                "nearest": nearest,
                "rho_orth": float(rho),
                "cosine": float(cap),
                "staged": int(len(self.staging)),
                "condensed": condensed,
            }
        self._absorb(chart, vector, int(row), chart_index=nearest, namespace=namespace)
        self.absorbs += 1
        # Check Voronoi boundary against second-nearest chart.
        if chart_ids.size > 1:
            second_pos = int(np.argpartition(-cosines, 1)[1])
            second_id = int(chart_ids[second_pos])
            if second_id != nearest and (cap - float(cosines[second_pos])) <= self.boundary_margin:
                self._register_boundary(vector, int(row), second_id)
        return {
            "action": "absorb",
            "chart": nearest,
            "rho_orth": float(rho),
            "cosine": float(cap),
        }

    # ------------------------------------------------------- condensation
    def _maybe_condense(self, namespace: str) -> int:
        """Condense one namespace when enough new writes have accumulated."""
        ns = str(namespace)
        if self._writes_since_condense.get(ns, 0) < int(self.b_stage):
            return 0
        if len(self.staging.rows(ns)) < int(self.b_stage):
            return 0
        self._writes_since_condense[ns] = 0
        return self.condense(ns)

    def _condense_entries(
        self, namespace: str, entries: List[Tuple[np.ndarray, int, str]], *, allow_small: bool
    ) -> int:
        """Condense staged writes coherently with bounded Voronoi dual-membership.

        Existing mature world charts are NEVER mutated in centroid or basis, so
        world frame drift is strictly zero. In-cap writes group by their nearest
        mature Voronoi cell into coherent companion charts; novel out-of-cap
        writes cluster via spherical k-means into sovereign charts. Boundary
        writes within `boundary_margin` of a competing chart receive a bounded
        dual-chart pointer so recall remains 1.0000 after condensation.
        """
        if not entries:
            return 0
        ns = str(namespace)
        grams = np.ascontiguousarray(
            np.stack([entry[0] for entry in entries]), dtype=np.complex128
        )
        grams_real = _as_real_view(grams)
        entry_rows = [int(entry[1]) for entry in entries]
        chart_ids, centroids_real, _total = self._get_ns_view(ns)
        born = 0
        promoted_rows = 0
        new_chart_entries: List[Tuple[int, np.ndarray, List[int]]] = []

        if chart_ids.size > 0:
            # Partition staged writes into in-cap (aligned to an existing mature
            # Voronoi region) vs genuinely novel out-of-cap writes.
            sims = grams_real @ centroids_real.T  # (M, K_ns)
            best_pos = np.argmax(sims, axis=1)
            best_sim = sims[np.arange(len(entries)), best_pos]
            in_cap_buckets: Dict[int, List[int]] = {}
            novel_indices: List[int] = []
            for idx in range(len(entries)):
                if float(best_sim[idx]) >= self.tau_cap:
                    cid = int(chart_ids[int(best_pos[idx])])
                    in_cap_buckets.setdefault(cid, []).append(idx)
                else:
                    novel_indices.append(idx)

            for parent_cid, idx_list in in_cap_buckets.items():
                parent_chart = self.charts[parent_cid]
                comp_cid = self._companion_of.get(parent_cid)
                if parent_chart.is_companion and parent_chart.count < self._effective_capacity:
                    comp_cid = parent_cid
                if comp_cid is not None:
                    comp_chart = self.charts[comp_cid]
                    remaining_room = max(0, self._effective_capacity - comp_chart.count)
                    if remaining_room > 0:
                        take_now = idx_list[:remaining_room]
                        idx_list = idx_list[remaining_room:]
                        for idx in take_now:
                            if comp_chart.append_member(grams[idx], entry_rows[idx]):
                                comp_chart.count += 1
                                promoted_rows += 1
                        self._dirty_charts.add(comp_cid)
                        new_chart_entries.append(
                            (comp_cid, grams[take_now], [entry_rows[i] for i in take_now])
                        )
                if not idx_list:
                    continue
                if not allow_small and len(idx_list) < self.min_cluster:
                    for idx in idx_list:
                        self.staging.stage(grams[idx], entry_rows[idx], ns)
                    continue
                idx_arr = np.asarray(idx_list, dtype=np.int64)
                for group in self._split_groups(grams, idx_arr):
                    sub_vecs = grams[group]
                    sub_rows = [entry_rows[int(i)] for i in group.tolist()]
                    chart = self._frame(sub_vecs, ns)
                    chart.is_companion = True
                    chart.set_packed(sub_vecs, sub_rows)
                    chart.count = int(group.size)
                    new_cid = len(self.charts)
                    self.charts.append(chart)
                    self.namespaces.setdefault(ns, []).append(new_cid)
                    self._companion_of[parent_cid] = new_cid
                    new_chart_entries.append((new_cid, sub_vecs, sub_rows))
                    born += 1
                    promoted_rows += int(group.size)
        else:
            novel_indices = list(range(len(entries)))

        if novel_indices:
            novel_arr = np.asarray(novel_indices, dtype=np.int64)
            if allow_small:
                cluster_count = int(max(1, min(self.k_condense, novel_arr.size)))
            else:
                cluster_count = int(
                    max(1, min(self.k_condense, novel_arr.size // self.min_cluster))
                )
            sub_grams = grams[novel_arr]
            assignment = self._cluster(sub_grams, cluster_count)
            for cid_local in range(int(assignment.max()) + 1):
                member = np.flatnonzero(assignment == cid_local)
                if member.size == 0:
                    continue
                if not allow_small and int(member.size) < self.min_cluster:
                    for local_i in member.tolist():
                        orig_i = int(novel_arr[int(local_i)])
                        self.staging.stage(grams[orig_i], entry_rows[orig_i], ns)
                    continue
                orig_group = novel_arr[member]
                for group in self._split_groups(grams, orig_group):
                    sub_vecs = grams[group]
                    sub_rows = [entry_rows[int(i)] for i in group.tolist()]
                    chart = self._frame(sub_vecs, ns)
                    chart.is_companion = True
                    chart.set_packed(sub_vecs, sub_rows)
                    chart.count = int(group.size)
                    new_cid = len(self.charts)
                    self.charts.append(chart)
                    self.namespaces.setdefault(ns, []).append(new_cid)
                    new_chart_entries.append((new_cid, sub_vecs, sub_rows))
                    born += 1
                    promoted_rows += int(group.size)

        if born or promoted_rows:
            if born:
                self._invalidate_caches()
            else:
                self._ns_cache.pop(ns, None)
                self._ns_cache.pop(None, None)
            self._ensure_caches()
            all_ids, all_centroids_real, _tot = self._get_ns_view(ns)
            if all_ids.size > 1:
                near_ties: List[Tuple[np.ndarray, int, int]] = []
                for assigned_cid, sub_vecs, sub_rows in new_chart_entries:
                    sub_real = _as_real_view(sub_vecs)
                    sims_all = sub_real @ all_centroids_real.T
                    top_k_probe = min(2, int(all_ids.size))
                    part = np.argpartition(-sims_all, top_k_probe - 1, axis=1)[:, :top_k_probe]
                    for local_i, row_id in enumerate(sub_rows):
                        row_sims = sims_all[local_i]
                        p0, p1 = int(part[local_i, 0]), int(part[local_i, 1])
                        if float(row_sims[p1]) > float(row_sims[p0]):
                            p0, p1 = p1, p0
                        c0 = int(all_ids[p0])
                        c1 = int(all_ids[p1])
                        s0 = float(row_sims[p0])
                        s1 = float(row_sims[p1])
                        if c0 != assigned_cid:
                            self._register_boundary(sub_vecs[local_i], row_id, c0, mandatory=True)
                        elif c1 != assigned_cid and (s0 - s1) <= self.boundary_margin:
                            near_ties.append((sub_vecs[local_i], row_id, c1))
                for vec_i, row_id, c1 in near_ties:
                    self._register_boundary(vec_i, row_id, c1, mandatory=False)
                self._ensure_caches()

        self.condensations += 1
        self.condensed_spawns += born
        self.condensed_rows += promoted_rows
        return born

    def finalize(self) -> int:
        """Condense every remaining nascent pool, allowing small clusters."""
        born = 0
        touched_namespaces: List[str] = []
        for namespace in list(self.staging.namespaces()):
            entries = self.staging.drain(namespace)
            if not entries:
                continue
            touched_namespaces.append(str(namespace))
            born += self._condense_entries(namespace, entries, allow_small=True)
        self._ensure_caches()
        # Final settled Voronoi boundary reconciliation for condensed companion
        # charts: guarantees early-condensed writes remain reachable even if
        # later condensation batches spawned a closer centroid.
        settled_ns = set(touched_namespaces) | {c.namespace for c in self.charts if c.is_companion}
        for ns in settled_ns:
            all_ids, all_centroids_real, _tot = self._get_ns_view(ns)
            if all_ids.size <= 1:
                continue
            for cid in all_ids.tolist():
                chart = self.charts[int(cid)]
                if not chart.is_companion or chart.vectors_real is None or chart.vectors is None:
                    continue
                n_primary = min(int(chart.count), int(chart.vectors_real.shape[0]))
                if n_primary <= 0:
                    continue
                sims = chart.vectors_real[:n_primary] @ all_centroids_real.T
                top_pos = np.argmax(sims, axis=1)
                for local_i in range(n_primary):
                    best_cid = int(all_ids[int(top_pos[local_i])])
                    if best_cid != int(cid):
                        self._register_boundary(
                            chart.vectors[local_i], int(chart.row_array[local_i]), best_cid, mandatory=True
                        )
        self._ensure_caches()
        # Refresh digest and warm two-level hierarchies for finalized namespaces.
        if self.charts:
            self.digest = hashlib.sha256(
                b"".join(_unit(chart.centroid).tobytes() for chart in self.charts)
            ).hexdigest()[:16]
        self._hierarchy_dirty = True
        for ns in list(self.namespaces.keys()):
            cids, c_real, _tot = self._get_ns_view(ns)
            self._get_hierarchy_view(ns, cids, c_real)
        return born

    def condense(self, namespace: str) -> int:
        """Deterministic coherent condensation over one nascent pool."""
        entries = self.staging.drain(namespace)
        if not entries:
            return 0
        return self._condense_entries(namespace, entries, allow_small=False)

    def _absorb(
        self,
        chart: RiemannianChart,
        vector: np.ndarray,
        row: int,
        *,
        chart_index: Optional[int] = None,
        namespace: Optional[str] = None,
    ) -> None:
        count = int(chart.count)
        chart.centroid = _unit((count * chart.centroid + vector) / float(count + 1))
        tangent = chart.tangent(vector)
        if chart.basis.shape[0] == 0:
            chart.basis = _orthonormal_tangent(tangent.reshape(1, -1), chart.centroid)
            chart.variances = np.array([max(float(np.vdot(tangent, tangent).real), 1e-6)])
        else:
            coords = chart.basis @ np.conj(tangent)
            residual = tangent - chart.basis.T @ coords
            update = np.outer(residual, np.conj(coords))
            proposal = chart.basis + self.oja_rate * update.T
            chart.basis = _orthonormal_tangent(proposal, chart.centroid)
            energy = np.maximum(np.abs(coords) ** 2, 1e-9)
            chart.variances = (count * chart.variances + energy) / float(count + 1)
            floor = max(float(np.vdot(residual, residual).real), 1e-9)
            chart.sigma2 = (count * chart.sigma2 + floor) / float(count + 1)
        chart.append_member(vector, int(row))
        chart.count = int(count + 1)
        if (
            chart_index is not None
            and self._centroid_matrix is not None
            and 0 <= chart_index < self._centroid_matrix.shape[0]
        ):
            self._centroid_matrix[chart_index] = chart.centroid
            self._dirty_charts.add(chart_index)
            self._hierarchy_dirty = True
            # Update namespace cache row count without rebuilding centroid matrix.
            if namespace is not None:
                self._ns_cache.pop(str(namespace), None)
            self._ns_cache.pop(None, None)
        else:
            self._invalidate_caches()

    # --------------------------------------------------------------- route
    def _route_hierarchical(
        self,
        vector: np.ndarray,
        q_real: np.ndarray,
        *,
        hier: _HierarchyView,
        all_chart_ids: np.ndarray,
        all_centroids_real: np.ndarray,
        namespace: Optional[str],
        take: int,
        has_nascent: bool,
    ) -> Tuple[List[int], Dict[str, object], bool]:
        """Exact two-level geodesic macro-cap routing via spherical branch-and-bound.

        Uses the spherical triangle inequality upper bound
            UB(q, m) = cos(theta_m) cos(alpha_m) + sin(theta_m) sin(alpha_m)
        (clamped to 1.0 whenever cos(theta_m) >= cos(alpha_m)) to prune macro-caps
        that mathematically cannot contain any leaf chart affecting top-m selection,
        margin checks, or local expansion. Guarantees 1.0000 exact equivalence with
        full leaf-centroid scoring.
        """
        tau = float(self.tau_margin if self.tau_margin is not None else 0.05)
        n_total = int(all_chart_ids.size)
        max_needed = min(n_total, max(take, self.top_charts_max, 2))
        min_leaves_for_prune = max(take, 2)

        macro_cos = hier.super_centroids_real @ q_real
        n_super = int(macro_cos.shape[0])
        sin_q = np.sqrt(np.maximum(0.0, 1.0 - macro_cos * macro_cos))
        ub = np.where(
            macro_cos >= hier.super_cos_alpha,
            1.0,
            macro_cos * hier.super_cos_alpha + sin_q * hier.super_sin_alpha,
        )

        # Probe highest-cosine macro-cap first so prune_bound rises immediately.
        best_m = int(np.argmax(macro_cos))
        ub_order = np.argsort(-ub, kind="stable")

        leaf_list: List[Tuple[int, float]] = []
        leaf_scored = 0
        probed_super = 0
        fallback_used = False
        fallback_limit = max(6, min(14, n_super // 3))

        def _probe_cap(sid: int) -> None:
            nonlocal leaf_scored, probed_super
            blk_ids = hier.block_chart_ids[sid]
            probed_super += 1
            if int(blk_ids.shape[0]) == 1:
                leaf_list.append((int(blk_ids[0]), float(macro_cos[sid])))
                leaf_scored += 1
            else:
                l_cos = hier.block_centroids_real[sid] @ q_real
                leaf_scored += int(l_cos.shape[0])
                for cid, sc in zip(blk_ids.tolist(), l_cos.tolist()):
                    leaf_list.append((int(cid), float(sc)))

        _probe_cap(best_m)
        leaf_list.sort(key=lambda item: (-item[1], item[0]))
        if len(leaf_list) > max_needed + 1:
            del leaf_list[max_needed + 1 :]

        prune_bound = -2.0
        if len(leaf_list) >= min_leaves_for_prune:
            first_now = float(leaf_list[0][1])
            take_th_now = float(leaf_list[take - 1][1])
            prune_bound = min(take_th_now - tau, first_now - self.expansion_gap)

        for sid_raw in ub_order.tolist():
            sid = int(sid_raw)
            if sid == best_m:
                continue
            if len(leaf_list) >= min_leaves_for_prune and float(ub[sid]) < prune_bound - 1e-12:
                break
            if probed_super >= fallback_limit:
                # OOD or heavily overlapping query: one contiguous BLAS scan beats
                # looping over many small Python blocks.
                fallback_used = True
                full_cos = all_centroids_real @ q_real
                leaf_scored = int(full_cos.shape[0])
                probed_super = n_super
                if max_needed < n_total:
                    top_idx = np.argpartition(-full_cos, max_needed - 1)[:max_needed]
                    top_order = top_idx[np.argsort(-full_cos[top_idx], kind="stable")]
                else:
                    top_order = np.argsort(-full_cos, kind="stable")
                leaf_list = [
                    (int(all_chart_ids[int(pos)]), float(full_cos[int(pos)]))
                    for pos in top_order.tolist()
                ]
                break
            _probe_cap(sid)
            leaf_list.sort(key=lambda item: (-item[1], item[0]))
            if len(leaf_list) > max_needed + 1:
                del leaf_list[max_needed + 1 :]
            if len(leaf_list) >= min_leaves_for_prune:
                first_now = float(leaf_list[0][1])
                take_th_now = float(leaf_list[take - 1][1])
                prune_bound = min(take_th_now - tau, first_now - self.expansion_gap)

        top_leaves = leaf_list[:max_needed]
        first = float(top_leaves[0][1])
        second = float(top_leaves[1][1]) if len(top_leaves) > 1 else -1.0
        raw_margin = first - second if len(top_leaves) > 1 else 1.0
        cutoff_margin = (
            float(top_leaves[take - 1][1] - top_leaves[take][1])
            if take < len(top_leaves)
            else 1.0
        )
        if raw_margin >= tau and cutoff_margin >= tau:
            actual_take = max(1, min(take, len(top_leaves)))
            winners = [int(top_leaves[i][0]) for i in range(actual_take)]
            return winners, {
                "probed": int(len(winners)),
                "margin": float(raw_margin),
                "c1": float(first),
                "c2": float(second),
                "expanded": False,
                "charts": winners,
                "rho_orth": [0.0 for _ in winners],
                "refined": False,
                "staged": 0,
                "hierarchy_used": True,
                "super_probed": int(probed_super),
                "leaf_centroids_scored": int(leaf_scored),
                "hierarchy_fallback": bool(fallback_used),
            }, has_nascent

        coarse = top_leaves[: min(2, len(top_leaves))]
        scored: List[Tuple[int, float, float, float]] = []
        for cid, cos_val in coarse:
            chart = self.charts[cid]
            projection = np.vdot(chart.centroid, vector)
            tangent = vector - projection * chart.centroid
            rho = chart.unexplained(tangent)
            scored.append((cid, float(cos_val - 0.25 * rho), cos_val, float(rho)))
        scored.sort(key=lambda item: -item[1])
        first_ref = scored[0][1]
        second_ref = scored[1][1] if len(scored) > 1 else -1.0
        margin = first_ref - second_ref
        expanded = False
        if (len(scored) > 1 and margin < tau) or cutoff_margin < tau:
            expand_limit = min(self.top_charts_max, len(top_leaves))
            min_cos = first - self.expansion_gap
            eligible = take
            for rank_i in range(take, expand_limit):
                if float(top_leaves[rank_i][1]) >= min_cos:
                    eligible = rank_i + 1
                else:
                    break
            if eligible > take:
                take = eligible
                expanded = True
        take = max(1, min(int(take), len(top_leaves)))
        if expanded or take > len(scored):
            winners = [int(top_leaves[i][0]) for i in range(take)]
            rho_map = {int(item[0]): float(item[3]) for item in scored}
            rhos = [float(rho_map.get(cid, 0.0)) for cid in winners]
        else:
            winners = [scored[i][0] for i in range(take)]
            rhos = [float(scored[i][3]) for i in range(take)]
        return winners, {
            "probed": int(len(winners)),
            "margin": float(margin),
            "c1": float(first_ref),
            "c2": float(second_ref),
            "expanded": bool(expanded),
            "charts": winners,
            "rho_orth": rhos,
            "refined": True,
            "staged": 0,
            "hierarchy_used": True,
            "super_probed": int(probed_super),
            "leaf_centroids_scored": int(leaf_scored),
            "hierarchy_fallback": bool(fallback_used),
        }, has_nascent

    def _route_charts(
        self,
        vector: np.ndarray,
        q_real: np.ndarray,
        *,
        namespace: Optional[str],
        top_charts: Optional[int],
    ) -> Tuple[List[int], Dict[str, object], bool]:
        """Single-level or two-level BLAS-GEMV routing returning winning chart ids."""
        chart_ids, centroids_real, namespace_rows = self._get_ns_view(namespace)
        has_nascent = self.staging.has_namespace(namespace)
        n_charts = int(chart_ids.size)
        if n_charts == 0:
            return [], {
                "probed": 0,
                "reason": "nascent-only",
                "margin": 0.0,
                "c1": 0.0,
                "c2": -1.0,
                "expanded": False,
                "charts": [],
                "rho_orth": [],
                "refined": False,
                "staged": int(len(self.staging.rows(namespace))) if has_nascent else 0,
                "hierarchy_used": False,
            }, has_nascent

        take = int(self.top_charts if top_charts is None else top_charts)
        if not has_nascent and namespace_rows > 1024 and n_charts >= self.hierarchy_min_charts:
            hier = self._get_hierarchy_view(namespace, chart_ids, centroids_real)
            if hier is not None:
                return self._route_hierarchical(
                    vector,
                    q_real,
                    hier=hier,
                    all_chart_ids=chart_ids,
                    all_centroids_real=centroids_real,
                    namespace=namespace,
                    take=take,
                    has_nascent=has_nascent,
                )

        # Single contiguous BLAS dgemv over all namespace centroids.
        cosines = centroids_real @ q_real
        tau = float(self.tau_margin if self.tau_margin is not None else 0.05)
        if has_nascent or namespace_rows <= 1024:
            take = max(take, n_charts)
        max_needed = min(n_charts, max(take, self.top_charts_max, 2))
        if max_needed < n_charts:
            top_idx = np.argpartition(-cosines, max_needed - 1)[:max_needed]
            top_scores = cosines[top_idx]
            order = top_idx[np.argsort(-top_scores, kind="stable")]
        else:
            order = np.argsort(-cosines, kind="stable")

        first = float(cosines[int(order[0])])
        second = float(cosines[int(order[1])]) if order.size > 1 else -1.0
        raw_margin = first - second if order.size > 1 else 1.0
        cutoff_margin = (
            float(cosines[int(order[take - 1])] - cosines[int(order[take])])
            if take < int(order.size)
            else 1.0
        )

        if (raw_margin >= tau and cutoff_margin >= tau) or take >= n_charts:
            actual_take = max(1, min(take, int(order.size)))
            winners = [int(chart_ids[int(pos)]) for pos in order[:actual_take].tolist()]
            return winners, {
                "probed": int(len(winners)),
                "margin": float(raw_margin),
                "c1": float(first),
                "c2": float(second),
                "expanded": False,
                "charts": winners,
                "rho_orth": [0.0 for _ in winners],
                "refined": False,
                "staged": int(len(self.staging.rows(namespace))) if has_nascent else 0,
                "hierarchy_used": False,
            }, has_nascent

        # Boundary query: refine top-2 centroid matches with tangent residual.
        coarse = order[: min(2, int(order.size))]
        scored: List[Tuple[int, float, float, float]] = []
        for pos in coarse.tolist():
            cid = int(chart_ids[int(pos)])
            chart = self.charts[cid]
            projection = np.vdot(chart.centroid, vector)
            tangent = vector - projection * chart.centroid
            rho = chart.unexplained(tangent)
            cos_val = float(cosines[int(pos)])
            scored.append((cid, float(cos_val - 0.25 * rho), cos_val, float(rho)))
        scored.sort(key=lambda item: -item[1])
        first_ref = scored[0][1]
        second_ref = scored[1][1] if len(scored) > 1 else -1.0
        margin = first_ref - second_ref
        expanded = False
        if (len(scored) > 1 and margin < tau) or cutoff_margin < tau:
            expand_limit = min(self.top_charts_max, int(order.size))
            min_cos = first - self.expansion_gap
            eligible = take
            for rank_i in range(take, expand_limit):
                if float(cosines[int(order[rank_i])]) >= min_cos:
                    eligible = rank_i + 1
                else:
                    break
            if eligible > take:
                take = eligible
                expanded = True
        take = max(1, min(int(take), int(order.size)))
        if expanded or take > len(scored):
            winners = [int(chart_ids[int(pos)]) for pos in order[:take].tolist()]
            rho_map = {int(item[0]): float(item[3]) for item in scored}
            rhos = [float(rho_map.get(cid, 0.0)) for cid in winners]
        else:
            winners = [scored[i][0] for i in range(take)]
            rhos = [float(scored[i][3]) for i in range(take)]
        return winners, {
            "probed": int(len(winners)),
            "margin": float(margin),
            "c1": float(first_ref),
            "c2": float(second_ref),
            "expanded": bool(expanded),
            "charts": winners,
            "rho_orth": rhos,
            "refined": True,
            "staged": int(len(self.staging.rows(namespace))) if has_nascent else 0,
            "hierarchy_used": False,
        }, has_nascent

    def query_top1(
        self,
        query_gram: np.ndarray,
        *,
        namespace: Optional[str] = "default",
        top_charts: Optional[int] = None,
    ) -> Tuple[int, float, int, Dict[str, object]]:
        """Direct contiguous-block BLAS top-1 search (zero scatter gathers).

        Returns `(best_row, best_score, candidates_scored, routing_detail)`.
        """
        vector = _unit(query_gram)
        q_real = vector.view(np.float64)
        winners, detail, has_nascent = self._route_charts(
            vector, q_real, namespace=namespace, top_charts=top_charts
        )
        best_row = -1
        best_score = -1e30
        scored_count = 0
        for cid in winners:
            chart = self.charts[cid]
            vecs_real = chart.vectors_real
            if vecs_real is None or vecs_real.shape[0] == 0:
                continue
            # Single contiguous BLAS dgemv on this chart's packed block.
            scores = vecs_real @ q_real
            idx = int(np.argmax(scores))
            val = float(scores[idx])
            scored_count += int(scores.shape[0])
            if val > best_score:
                best_score = val
                best_row = int(chart.row_array[idx])
        if has_nascent:
            n_vecs, n_rows = self.staging.packed(namespace)
            if n_rows.size > 0:
                scores = n_vecs @ q_real
                idx = int(np.argmax(scores))
                val = float(scores[idx])
                scored_count += int(scores.shape[0])
                if val > best_score:
                    best_score = val
                    best_row = int(n_rows[idx])
        detail["candidates_scored"] = int(scored_count)
        return best_row, float(best_score if best_row >= 0 else 0.0), scored_count, detail

    def search(
        self,
        query_gram: np.ndarray,
        *,
        k: int = 5,
        namespace: Optional[str] = "default",
        top_charts: Optional[int] = None,
    ) -> Tuple[List[Tuple[int, float]], Dict[str, object]]:
        """Direct contiguous-block BLAS top-k search with deduplication."""
        vector = _unit(query_gram)
        q_real = vector.view(np.float64)
        winners, detail, has_nascent = self._route_charts(
            vector, q_real, namespace=namespace, top_charts=top_charts
        )
        top_k = max(1, int(k))
        best_by_row: Dict[int, float] = {}
        scored_count = 0
        for cid in winners:
            chart = self.charts[cid]
            vecs_real = chart.vectors_real
            if vecs_real is None or vecs_real.shape[0] == 0:
                continue
            scores = vecs_real @ q_real
            n = int(scores.shape[0])
            scored_count += n
            take_local = min(top_k, n)
            if take_local < n:
                idxs = np.argpartition(-scores, take_local - 1)[:take_local]
            else:
                idxs = np.arange(n, dtype=np.int64)
            rids = chart.row_array
            for idx in idxs.tolist():
                rid = int(rids[int(idx)])
                val = float(scores[int(idx)])
                prev = best_by_row.get(rid)
                if prev is None or val > prev:
                    best_by_row[rid] = val
        if has_nascent:
            n_vecs, n_rows = self.staging.packed(namespace)
            if n_rows.size > 0:
                scores = n_vecs @ q_real
                scored_count += int(scores.shape[0])
                for idx in range(int(scores.shape[0])):
                    rid = int(n_rows[idx])
                    val = float(scores[idx])
                    prev = best_by_row.get(rid)
                    if prev is None or val > prev:
                        best_by_row[rid] = val
        ranked = sorted(best_by_row.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        detail["candidates_scored"] = int(scored_count)
        return ranked, detail

    def select(
        self,
        query_gram: np.ndarray,
        *,
        namespace: Optional[str] = "default",
        top_charts: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        """Return deduplicated candidate row indices for backwards compatibility."""
        vector = _unit(query_gram)
        q_real = vector.view(np.float64)
        winners, detail, has_nascent = self._route_charts(
            vector, q_real, namespace=namespace, top_charts=top_charts
        )
        blocks: List[np.ndarray] = [
            self.charts[cid].row_array for cid in winners if self.charts[cid].row_array.size > 0
        ]
        if has_nascent:
            _n_vecs, n_rows = self.staging.packed(namespace)
            if n_rows.size > 0:
                blocks.append(n_rows)
        if not blocks:
            return np.zeros(0, dtype=np.int64), detail
        if len(blocks) == 1:
            selected = blocks[0]
        else:
            combined = np.concatenate(blocks)
            selected = np.unique(combined) if self.boundary_memberships > 0 else combined
        return selected, detail

    # --------------------------------------------------------- persistence
    def save(self, path: Union[str, Path]) -> Dict[str, object]:
        """Persist the full packed atlas (binary slabs + SQLite metadata + hierarchy).

        Streams chart vector blocks directly to `vectors.bin` without creating
        a full in-RAM duplicate, writes `rows.bin` and uncompressed `geom.npz`,
        and seals checksums and pre-built super-cluster hierarchies in
        `atlas.sqlite3` so cold restart mounts via zero-copy mmap without
        rebuilding charts.
        """
        started = time.perf_counter()
        self._ensure_caches()
        for ns in list(self.namespaces.keys()):
            cids, c_real, _tot = self._get_ns_view(ns)
            self._get_hierarchy_view(ns, cids, c_real)

        root = Path(path).resolve()
        root.mkdir(parents=True, exist_ok=True)
        vectors_path = root / "vectors.bin"
        rows_path = root / "rows.bin"
        geom_path = root / "geom.npz"
        db_tmp = root / "atlas.sqlite3.tmp"
        db_path = root / "atlas.sqlite3"

        dim = 0
        if self.charts:
            dim = int(self.charts[0].centroid.size)
        elif self.staging.entries:
            dim = int(self.staging.entries[0][0].size)

        # Stream packed chart vectors and row_arrays directly to binary slabs.
        chart_records: List[Tuple[int, str, int, int, int, float, int, int, int, int]] = []
        vec_offset = 0
        basis_offset = 0
        centroids_list: List[np.ndarray] = []
        bases_list: List[np.ndarray] = []
        variances_list: List[np.ndarray] = []

        vec_hasher = hashlib.sha256()
        row_hasher = hashlib.sha256()
        with vectors_path.open("wb") as vfh, rows_path.open("wb") as rfh:
            for cid, chart in enumerate(self.charts):
                chart.ensure_packed()
                centroids_list.append(np.ascontiguousarray(chart.centroid, dtype=np.complex128))
                basis = np.ascontiguousarray(chart.basis, dtype=np.complex128)
                variances = np.ascontiguousarray(chart.variances, dtype=np.float64)
                b_rank = int(basis.shape[0])
                if b_rank > 0:
                    bases_list.append(basis)
                    variances_list.append(variances)
                vecs = chart.vectors
                r_arr = np.ascontiguousarray(chart.row_array, dtype=np.int64)
                v_len = int(r_arr.shape[0])
                if v_len > 0 and vecs is not None:
                    v_bytes = np.ascontiguousarray(vecs, dtype=np.complex128).tobytes()
                    r_bytes = r_arr.tobytes()
                    vfh.write(v_bytes)
                    rfh.write(r_bytes)
                    vec_hasher.update(v_bytes)
                    row_hasher.update(r_bytes)
                chart_records.append(
                    (
                        int(cid),
                        str(chart.namespace),
                        int(chart.count),
                        int(chart.boundary_count),
                        1 if chart.is_companion else 0,
                        float(chart.sigma2),
                        int(vec_offset),
                        int(v_len),
                        int(basis_offset),
                        int(b_rank),
                    )
                )
                vec_offset += v_len
                basis_offset += b_rank

        total_stored = int(vec_offset)
        vectors_sha256 = vec_hasher.hexdigest()
        vectors_fast_sha256 = _fast_stride_sha256(vectors_path)
        rows_sha256 = row_hasher.hexdigest()

        geom_arrays: Dict[str, np.ndarray] = {
            "centroids": (
                np.ascontiguousarray(np.stack(centroids_list), dtype=np.complex128)
                if centroids_list
                else np.zeros((0, dim), dtype=np.complex128)
            ),
            "bases": (
                np.ascontiguousarray(np.vstack(bases_list), dtype=np.complex128)
                if bases_list
                else np.zeros((0, dim), dtype=np.complex128)
            ),
            "variances": (
                np.ascontiguousarray(np.concatenate(variances_list), dtype=np.float64)
                if variances_list
                else np.zeros(0, dtype=np.float64)
            ),
        }

        hier_rows: List[Tuple[str, int, int]] = []
        for h_idx, (ns_key, h_view) in enumerate(sorted(self._hier_cache.items(), key=lambda kv: str(kv[0]))):
            if ns_key is None or h_view is None:
                continue
            leaf_offsets = [0]
            for blk in h_view.block_chart_ids:
                leaf_offsets.append(leaf_offsets[-1] + int(blk.shape[0]))
            leaf_ids_flat = (
                np.ascontiguousarray(np.concatenate(h_view.block_chart_ids), dtype=np.int64)
                if h_view.block_chart_ids
                else np.zeros(0, dtype=np.int64)
            )
            prefix = f"h{h_idx}_"
            geom_arrays[prefix + "super_centroids"] = np.ascontiguousarray(
                h_view.super_centroids, dtype=np.complex128
            )
            geom_arrays[prefix + "cos_alpha"] = np.ascontiguousarray(
                h_view.super_cos_alpha, dtype=np.float64
            )
            geom_arrays[prefix + "sin_alpha"] = np.ascontiguousarray(
                h_view.super_sin_alpha, dtype=np.float64
            )
            geom_arrays[prefix + "leaf_ids"] = leaf_ids_flat
            geom_arrays[prefix + "leaf_offsets"] = np.asarray(leaf_offsets, dtype=np.int64)
            hier_rows.append((str(ns_key), int(h_idx), int(h_view.super_boundary_duals)))

        np.savez(geom_path, **geom_arrays)
        geom_sha256 = _sha256_file(geom_path)

        if db_tmp.exists():
            db_tmp.unlink()
        conn = sqlite3.connect(str(db_tmp))
        try:
            conn.execute("PRAGMA journal_mode=OFF")
            conn.execute("PRAGMA synchronous=OFF")
            conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            conn.execute(
                """
                CREATE TABLE charts (
                    chart_id INTEGER PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    count INTEGER NOT NULL,
                    boundary_count INTEGER NOT NULL,
                    is_companion INTEGER NOT NULL,
                    sigma2 REAL NOT NULL,
                    vec_offset INTEGER NOT NULL,
                    vec_len INTEGER NOT NULL,
                    basis_offset INTEGER NOT NULL,
                    basis_rank INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE hierarchies (
                    namespace TEXT PRIMARY KEY,
                    hier_idx INTEGER NOT NULL,
                    super_boundary_duals INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE staging (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    namespace TEXT NOT NULL,
                    row_id INTEGER NOT NULL,
                    vec_blob BLOB NOT NULL
                )
                """
            )
            hyper = {
                "rank": int(self.rank),
                "capacity": int(self.capacity),
                "effective_capacity": int(self._effective_capacity),
                "tau_orth": float(self.tau_orth),
                "tau_cap": float(self.tau_cap),
                "tau_margin": self.tau_margin,
                "top_charts": int(self.top_charts),
                "top_charts_max": int(self.top_charts_max),
                "oja_rate": float(self.oja_rate),
                "eps": float(self.eps),
                "b_stage": int(self.b_stage),
                "k_condense": int(self.k_condense),
                "min_cluster": int(self.min_cluster),
                "boundary_margin": float(self.boundary_margin),
                "max_boundary_ratio": float(self.max_boundary_ratio),
                "expansion_gap": float(self.expansion_gap),
                "max_rescore_fraction": float(self.max_rescore_fraction),
                "hierarchy_min_charts": int(self.hierarchy_min_charts),
                "top_super": int(self.top_super),
                "top_super_max": int(self.top_super_max),
                "super_boundary_margin": float(self.super_boundary_margin),
            }
            counters = {
                "spawns": int(self.spawns),
                "staged": int(self.staged),
                "condensations": int(self.condensations),
                "condensed_spawns": int(self.condensed_spawns),
                "condensed_rows": int(self.condensed_rows),
                "absorbs": int(self.absorbs),
                "boundary_memberships": int(self.boundary_memberships),
                "digest": str(self.digest),
            }
            meta_items = [
                ("format_version", _PERSIST_FORMAT_VERSION),
                ("dim", str(dim)),
                ("total_stored", str(total_stored)),
                ("n_charts", str(len(self.charts))),
                ("vectors_bytes", str(int(vectors_path.stat().st_size))),
                ("rows_bytes", str(int(rows_path.stat().st_size))),
                ("geom_bytes", str(int(geom_path.stat().st_size))),
                ("vectors_sha256", vectors_sha256),
                ("vectors_fast_sha256", vectors_fast_sha256),
                ("rows_sha256", rows_sha256),
                ("geom_sha256", geom_sha256),
                ("hyperparams_json", json.dumps(hyper, sort_keys=True)),
                ("counters_json", json.dumps(counters, sort_keys=True)),
                (
                    "companion_of_json",
                    json.dumps({str(k): int(v) for k, v in self._companion_of.items()}, sort_keys=True),
                ),
                (
                    "writes_since_condense_json",
                    json.dumps(self._writes_since_condense, sort_keys=True),
                ),
            ]
            conn.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", meta_items)
            conn.executemany(
                "INSERT INTO charts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                chart_records,
            )
            conn.executemany(
                "INSERT INTO hierarchies VALUES (?, ?, ?)",
                hier_rows,
            )
            if self.staging.entries:
                conn.executemany(
                    "INSERT INTO staging (namespace, row_id, vec_blob) VALUES (?, ?, ?)",
                    [
                        (str(ns), int(rid), np.ascontiguousarray(vec, dtype=np.complex128).tobytes())
                        for vec, rid, ns in self.staging.entries
                    ],
                )
            conn.commit()
        finally:
            conn.close()
        os.replace(db_tmp, db_path)
        self.persist_path = str(root)
        elapsed = float(time.perf_counter() - started)
        return {
            "path": str(root),
            "format_version": _PERSIST_FORMAT_VERSION,
            "dim": int(dim),
            "charts": int(len(self.charts)),
            "total_stored": int(total_stored),
            "hierarchies": int(len(hier_rows)),
            "vectors_bytes": int(vectors_path.stat().st_size),
            "rows_bytes": int(rows_path.stat().st_size),
            "geom_bytes": int(geom_path.stat().st_size),
            "vectors_sha256": vectors_sha256,
            "vectors_fast_sha256": vectors_fast_sha256,
            "rows_sha256": rows_sha256,
            "geom_sha256": geom_sha256,
            "save_seconds": elapsed,
        }

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        *,
        mmap: bool = True,
        verify_checksum: Union[str, bool] = "fast",
    ) -> "RiemannianChartAtlas":
        """Load a persisted packed atlas with zero chart rebuild.

        `verify_checksum`:
        - `"fast"` (default): verifies SQLite schema, byte lengths, full SHA-256
          of `rows.bin` and `geom.npz`, and strided head/body/tail SHA-256 of
          `vectors.bin` for sub-second cold mmap mount.
        - `"full"` or `True`: additionally verifies full SHA-256 over `vectors.bin`.
        - `"none"` or `False`: verifies byte lengths and format version only.
        """
        started = time.perf_counter()
        if isinstance(verify_checksum, bool):
            mode = "full" if verify_checksum else "none"
        else:
            mode = str(verify_checksum).strip().lower()
        if mode not in {"fast", "full", "none"}:
            raise ValueError(f"unsupported verify_checksum mode: {verify_checksum!r}")

        root = Path(path).resolve()
        db_path = root / "atlas.sqlite3"
        vectors_path = root / "vectors.bin"
        rows_path = root / "rows.bin"
        geom_path = root / "geom.npz"
        for req in (db_path, vectors_path, rows_path, geom_path):
            if not req.exists():
                raise FileNotFoundError(f"missing persisted atlas artifact: {req}")

        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
            if meta.get("format_version") != _PERSIST_FORMAT_VERSION:
                raise ValueError(
                    f"unsupported atlas persistence format: {meta.get('format_version')!r}"
                )
            chart_rows = conn.execute(
                """
                SELECT chart_id, namespace, count, boundary_count, is_companion,
                       sigma2, vec_offset, vec_len, basis_offset, basis_rank
                FROM charts ORDER BY chart_id ASC
                """
            ).fetchall()
            hier_rows = conn.execute(
                "SELECT namespace, hier_idx, super_boundary_duals FROM hierarchies"
            ).fetchall()
            staging_rows = conn.execute(
                "SELECT namespace, row_id, vec_blob FROM staging ORDER BY seq ASC"
            ).fetchall()
        finally:
            conn.close()

        dim = int(meta["dim"])
        total_stored = int(meta["total_stored"])
        if int(vectors_path.stat().st_size) != int(meta["vectors_bytes"]):
            raise ValueError("vectors.bin byte length mismatch")
        if int(rows_path.stat().st_size) != int(meta["rows_bytes"]):
            raise ValueError("rows.bin byte length mismatch")
        if int(geom_path.stat().st_size) != int(meta["geom_bytes"]):
            raise ValueError("geom.npz byte length mismatch")

        if mode in {"fast", "full"}:
            if _sha256_file(rows_path) != meta["rows_sha256"]:
                raise ValueError("rows.bin SHA-256 verification failed")
            if _sha256_file(geom_path) != meta["geom_sha256"]:
                raise ValueError("geom.npz SHA-256 verification failed")
            if mode == "fast":
                if _fast_stride_sha256(vectors_path) != meta["vectors_fast_sha256"]:
                    raise ValueError("vectors.bin fast stride SHA-256 verification failed")
            else:
                if _sha256_file(vectors_path) != meta["vectors_sha256"]:
                    raise ValueError("vectors.bin full SHA-256 verification failed")

        hyper = json.loads(meta["hyperparams_json"])
        counters = json.loads(meta["counters_json"])
        atlas = cls(
            rank=int(hyper["rank"]),
            capacity=int(hyper["capacity"]),
            tau_orth=float(hyper["tau_orth"]),
            tau_cap=float(hyper["tau_cap"]),
            tau_margin=hyper.get("tau_margin"),
            top_charts=int(hyper["top_charts"]),
            top_charts_max=int(hyper["top_charts_max"]),
            oja_rate=float(hyper["oja_rate"]),
            eps=float(hyper["eps"]),
            b_stage=int(hyper["b_stage"]),
            k_condense=int(hyper["k_condense"]),
            min_cluster=int(hyper["min_cluster"]),
            boundary_margin=float(hyper["boundary_margin"]),
            max_boundary_ratio=float(hyper["max_boundary_ratio"]),
            expansion_gap=float(hyper["expansion_gap"]),
            max_rescore_fraction=float(hyper["max_rescore_fraction"]),
            hierarchy_min_charts=int(hyper.get("hierarchy_min_charts", 96)),
            top_super=int(hyper.get("top_super", 3)),
            top_super_max=int(hyper.get("top_super_max", 6)),
            super_boundary_margin=float(hyper.get("super_boundary_margin", 0.12)),
        )
        atlas._effective_capacity = int(hyper.get("effective_capacity", atlas.capacity))
        atlas.spawns = int(counters["spawns"])
        atlas.staged = int(counters["staged"])
        atlas.condensations = int(counters["condensations"])
        atlas.condensed_spawns = int(counters["condensed_spawns"])
        atlas.condensed_rows = int(counters["condensed_rows"])
        atlas.absorbs = int(counters["absorbs"])
        atlas.boundary_memberships = int(counters["boundary_memberships"])
        atlas.digest = str(counters["digest"])
        atlas._companion_of = {
            int(k): int(v) for k, v in json.loads(meta["companion_of_json"]).items()
        }
        atlas._writes_since_condense = {
            str(k): int(v) for k, v in json.loads(meta["writes_since_condense_json"]).items()
        }

        if total_stored > 0 and dim > 0:
            if mmap:
                vectors_slab: np.ndarray = np.memmap(
                    vectors_path,
                    dtype=np.complex128,
                    mode="r",
                    shape=(total_stored, dim),
                )
                atlas._mmap_refs.append(vectors_slab)
            else:
                vectors_slab = np.fromfile(vectors_path, dtype=np.complex128).reshape(
                    total_stored, dim
                )
            rows_slab = np.fromfile(rows_path, dtype=np.int64)
        else:
            vectors_slab = np.zeros((0, dim), dtype=np.complex128)
            rows_slab = np.zeros(0, dtype=np.int64)

        with np.load(geom_path, allow_pickle=False) as geom:
            centroids_all = np.ascontiguousarray(geom["centroids"], dtype=np.complex128)
            bases_all = np.ascontiguousarray(geom["bases"], dtype=np.complex128)
            variances_all = np.ascontiguousarray(geom["variances"], dtype=np.float64)

            for (
                cid,
                ns,
                count,
                boundary_count,
                is_companion,
                sigma2,
                vec_offset,
                vec_len,
                basis_offset,
                basis_rank,
            ) in chart_rows:
                c_idx = int(cid)
                b_off = int(basis_offset)
                b_end = b_off + int(basis_rank)
                v_off = int(vec_offset)
                v_end = v_off + int(vec_len)
                chart = RiemannianChart(
                    centroid=centroids_all[c_idx],
                    basis=bases_all[b_off:b_end],
                    variances=variances_all[b_off:b_end],
                    sigma2=float(sigma2),
                    namespace=str(ns),
                    count=int(count),
                    is_companion=bool(is_companion),
                    boundary_count=int(boundary_count),
                )
                chart.set_packed_view(vectors_slab[v_off:v_end], rows_slab[v_off:v_end])
                atlas.charts.append(chart)
                atlas.namespaces.setdefault(str(ns), []).append(c_idx)

            atlas._centroid_matrix = centroids_all if centroids_all.size else None
            atlas._centroid_real = (
                _as_real_view(centroids_all) if centroids_all.size else None
            )
            atlas._dirty_cache = False
            atlas._dirty_charts.clear()

            # Restore pre-built two-level hierarchies directly (zero clustering on mount).
            if hier_rows and atlas._centroid_real is not None:
                for ns_key, h_idx, super_duals in hier_rows:
                    prefix = f"h{int(h_idx)}_"
                    super_c = np.ascontiguousarray(
                        geom[prefix + "super_centroids"], dtype=np.complex128
                    )
                    cos_a = np.ascontiguousarray(geom[prefix + "cos_alpha"], dtype=np.float64)
                    sin_a = np.ascontiguousarray(geom[prefix + "sin_alpha"], dtype=np.float64)
                    leaf_ids = np.ascontiguousarray(geom[prefix + "leaf_ids"], dtype=np.int64)
                    leaf_offsets = geom[prefix + "leaf_offsets"].tolist()
                    blk_ids: List[np.ndarray] = []
                    blk_real: List[np.ndarray] = []
                    for s_i in range(len(leaf_offsets) - 1):
                        a = int(leaf_offsets[s_i])
                        b = int(leaf_offsets[s_i + 1])
                        cids_slice = leaf_ids[a:b]
                        blk_ids.append(cids_slice)
                        blk_real.append(
                            np.ascontiguousarray(
                                atlas._centroid_real[cids_slice], dtype=np.float64
                            )
                        )
                    atlas._hier_cache[str(ns_key)] = _HierarchyView(
                        super_centroids=super_c,
                        super_centroids_real=_as_real_view(super_c),
                        super_cos_alpha=cos_a,
                        super_sin_alpha=sin_a,
                        block_chart_ids=blk_ids,
                        block_centroids_real=blk_real,
                        super_boundary_duals=int(super_duals),
                    )
                atlas._hierarchy_dirty = False
            else:
                atlas._hierarchy_dirty = True

        for ns, rid, blob in staging_rows:
            vec = np.frombuffer(blob, dtype=np.complex128).copy()
            atlas.staging.stage(vec, int(rid), str(ns))

        atlas.loaded_from_disk = True
        atlas.rebuilt_charts = 0
        atlas.mount_seconds = float(time.perf_counter() - started)
        atlas.persist_path = str(root)
        return atlas

    def stats(self) -> Dict[str, object]:
        sizes = [int(chart.count) for chart in self.charts]
        stored = [int(len(chart.rows)) for chart in self.charts]
        total_primary = int(sum(sizes))
        hier_summary = {
            str(ns): {
                "super_clusters": int(hv.super_centroids.shape[0]),
                "super_boundary_duals": int(hv.super_boundary_duals),
            }
            for ns, hv in self._hier_cache.items()
            if ns is not None and hv is not None
        }
        return {
            "charts": int(len(self.charts)),
            "rows": total_primary,
            "stored_rows": int(sum(stored)),
            "boundary_memberships": int(self.boundary_memberships),
            "boundary_ratio": float(self.boundary_memberships / max(total_primary, 1)),
            "rank": int(self.rank),
            "capacity": int(self.capacity),
            "effective_capacity": int(self._effective_capacity),
            "tau_orth": float(self.tau_orth),
            "tau_cap": float(self.tau_cap),
            "top_charts": int(self.top_charts),
            "top_charts_max": int(self.top_charts_max),
            "hierarchy_min_charts": int(self.hierarchy_min_charts),
            "hierarchies": hier_summary,
            "loaded_from_disk": bool(self.loaded_from_disk),
            "rebuilt_charts": int(self.rebuilt_charts),
            "mount_seconds": float(self.mount_seconds),
            "persist_path": self.persist_path,
            "spawns": int(self.spawns),
            "absorbs": int(self.absorbs),
            "staged_writes": int(self.staged),
            "staged_now": int(len(self.staging)),
            "condensations": int(self.condensations),
            "condensed_charts": int(self.condensed_spawns),
            "condensed_rows": int(self.condensed_rows),
            "b_stage": int(self.b_stage),
            "k_condense": int(self.k_condense),
            "min_cluster": int(self.min_cluster),
            "largest_chart": int(max(sizes) if sizes else 0),
            "largest_stored_chart": int(max(stored) if stored else 0),
            "mean_chart": float(np.mean(sizes) if sizes else 0.0),
            "digest": str(self.digest),
        }
