"""Riemannian tangent-space chart atlas for Zone C (Sovereign T2 / Cycle AF1).

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
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_EPS = 1e-12


def _unit(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=np.complex128).reshape(-1)
    norm = float(np.linalg.norm(array))
    return array / max(norm, _EPS)


def _as_real_view(matrix: np.ndarray) -> np.ndarray:
    """Zero-copy float64 view of C-contiguous complex128 vectors.

    For unit complex vectors u, v in C^D, Re(u . conj(v)) equals the real inner
    product of their interleaved [Re, Im] float64 views, avoiding complex
    imaginary arithmetic and temporary conjugation copies on the hot BLAS path.
    """
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
    _row_set: set = field(default_factory=set, repr=False)

    def set_packed(self, vectors: np.ndarray, rows: Sequence[int]) -> None:
        """Install contiguous packed vectors and row ids in one allocation."""
        row_list = [int(r) for r in rows]
        self.rows = row_list
        self._row_set = set(row_list)
        self.row_array = np.asarray(row_list, dtype=np.int64)
        self.vectors = np.ascontiguousarray(vectors, dtype=np.complex128)
        self.vectors_real = _as_real_view(self.vectors)
        self._pending_rows.clear()
        self._pending_vectors.clear()

    def append_member(self, vector: np.ndarray, row: int, *, is_boundary: bool = False) -> bool:
        """Stage a row+vector for lazy repack without mutating chart geometry."""
        row_id = int(row)
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
    """A non-stationary atlas with inversion-free local whitening and packed BLAS routing."""

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
        self._writes_since_condense: Dict[str, int] = {}
        self._companion_of: Dict[int, int] = {}
        self._dirty_cache = True

    def _invalidate_caches(self) -> None:
        self._centroid_matrix = None
        self._centroid_real = None
        self._ns_cache.clear()
        self._dirty_cache = True

    def _ensure_caches(self) -> None:
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
        self._dirty_cache = False

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

    # ---------------------------------------------------------------- build
    @staticmethod
    def _cluster(
        grams: np.ndarray, cluster_count: int, iterations: int = 24
    ) -> np.ndarray:
        count = int(grams.shape[0])
        cluster_count = int(max(1, min(cluster_count, count)))
        if cluster_count == 1:
            return np.zeros(count, dtype=np.int64)
        stride = max(1, count // cluster_count)
        picks = [min(count - 1, index * stride) for index in range(cluster_count)]
        centroids = np.ascontiguousarray(
            np.stack([_unit(grams[index]) for index in picks]), dtype=np.complex128
        )
        grams_real = _as_real_view(grams)
        assignment = np.full(count, -1, dtype=np.int64)
        for _ in range(iterations):
            scores = grams_real @ _as_real_view(centroids).T
            updated = np.argmax(scores, axis=1).astype(np.int64)
            if np.array_equal(updated, assignment):
                break
            assignment = updated
            for chart_id in range(cluster_count):
                members = grams[assignment == chart_id]
                if members.size:
                    centroids[chart_id] = _unit(members.mean(axis=0))
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
        assignment = self._cluster(matrix, cluster_count)
        ns = str(namespace)
        for chart_id in range(int(assignment.max()) + 1):
            member_index = np.flatnonzero(assignment == chart_id)
            if member_index.size == 0:
                continue
            for group in self._split_groups(matrix, member_index):
                sub_vecs = matrix[group]
                chart = self._frame(sub_vecs, ns)
                group_rows = [int(rows[int(index)]) for index in group.tolist()]
                chart.set_packed(sub_vecs, group_rows)
                chart.count = int(group.size)
                self.charts.append(chart)
                self.namespaces.setdefault(ns, []).append(len(self.charts) - 1)
        self.digest = hashlib.sha256(
            b"".join(_unit(chart.centroid).tobytes() for chart in self.charts)
        ).hexdigest()[:16]
        self._invalidate_caches()
        self._ensure_caches()
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
        if len(neighbor.rows) >= int(math.ceil(self._effective_capacity * 1.15)):
            return False
        added = neighbor.append_member(vector, int(row), is_boundary=True)
        if added:
            self.boundary_memberships += 1
            self._dirty_cache = True
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
        if rho > self.tau_orth or cap < self.tau_cap or chart.count >= self.capacity:
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
        self._absorb(chart, vector, int(row))
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
                # If a companion chart already exists for this Voronoi region
                # and has room, append to that companion chart cleanly (never
                # touching the mature world chart's centroid or basis).
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
                        new_chart_entries.append(
                            (comp_cid, grams[take_now], [entry_rows[i] for i in take_now])
                        )
                        self._dirty_cache = True
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
            self._invalidate_caches()
            self._ensure_caches()
            # Bounded Voronoi boundary dual-membership: if a newly condensed
            # write's highest-cosine chart across the updated namespace is NOT
            # its assigned chart (or a neighbor chart competes within
            # boundary_margin), register a bounded boundary pointer so top-m
            # routing never drops it.
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
                            # Priority 1: write sits on the far side of a Voronoi
                            # boundary; attach bounded pointer to top-1 chart c0.
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
        for ns in set(touched_namespaces) | {c.namespace for c in self.charts if c.is_companion}:
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
        return born

    def condense(self, namespace: str) -> int:
        """Deterministic coherent condensation over one nascent pool."""
        entries = self.staging.drain(namespace)
        if not entries:
            return 0
        return self._condense_entries(namespace, entries, allow_small=False)

    def _absorb(self, chart: RiemannianChart, vector: np.ndarray, row: int) -> None:
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
        self._invalidate_caches()

    # --------------------------------------------------------------- route
    def _route_charts(
        self,
        vector: np.ndarray,
        q_real: np.ndarray,
        *,
        namespace: Optional[str],
        top_charts: Optional[int],
    ) -> Tuple[List[int], Dict[str, object], bool]:
        """Single-BLAS-GEMV routing returning winning chart ids and metadata."""
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
            }, has_nascent

        # Single contiguous BLAS dgemv over all namespace centroids.
        cosines = centroids_real @ q_real
        tau = float(self.tau_margin if self.tau_margin is not None else 0.05)
        take = int(self.top_charts if top_charts is None else top_charts)
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
            # Expand up to top_charts_max ONLY across charts within the local
            # geodesic expansion_gap of c1 (avoiding orthogonal unrelated charts).
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

    def stats(self) -> Dict[str, object]:
        sizes = [int(chart.count) for chart in self.charts]
        stored = [int(len(chart.rows)) for chart in self.charts]
        total_primary = int(sum(sizes))
        return {
            "charts": int(len(self.charts)),
            "rows": total_primary,
            "stored_rows": int(sum(stored)),
            "boundary_memberships": int(self.boundary_memberships),
            "boundary_ratio": float(self.boundary_memberships / max(total_primary, 1)),
            "rank": int(self.rank),
            "capacity": int(self.capacity),
            "tau_orth": float(self.tau_orth),
            "tau_cap": float(self.tau_cap),
            "top_charts": int(self.top_charts),
            "top_charts_max": int(self.top_charts_max),
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
