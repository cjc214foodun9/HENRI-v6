"""Riemannian tangent-space chart atlas for Zone C (Sovereign T2).

Each chart is a local spherical cap on S^{d-1} with a unit Fréchet centroid c,
a rank-r orthonormal tangent basis U (U^H c = 0), principal variances lambda and
an isotropic orthogonal floor sigma_perp^2. Local whitening uses the Woodbury
identity, so no dense matrix is inverted. An online write whose tangent energy
is not explained by the nearest chart spawns a new chart at the write instead
of rotating an existing frame. Existing charts therefore do not drift.
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

    def __len__(self) -> int:
        return len(self.entries)

    def stage(self, vector: np.ndarray, row: int, namespace: str) -> None:
        self.entries.append((_unit(vector), int(row), str(namespace)))

    def rows(self, namespace: Optional[str] = None) -> List[int]:
        if namespace is None:
            return [int(entry[1]) for entry in self.entries]
        return [int(entry[1]) for entry in self.entries if entry[2] == str(namespace)]

    def namespaces(self) -> List[str]:
        seen: List[str] = []
        for _vector, _row, namespace in self.entries:
            if namespace not in seen:
                seen.append(namespace)
        return seen

    def drain(self, namespace: str) -> List[Tuple[np.ndarray, int, str]]:
        kept: List[Tuple[np.ndarray, int, str]] = []
        taken: List[Tuple[np.ndarray, int, str]] = []
        for entry in self.entries:
            (taken if entry[2] == str(namespace) else kept).append(entry)
        self.entries = kept
        return taken


@dataclass
class RiemannianChart:
    centroid: np.ndarray
    basis: np.ndarray
    variances: np.ndarray
    sigma2: float
    rows: List[int] = field(default_factory=list)
    namespace: str = "default"
    count: int = 0

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
            "namespace": str(self.namespace),
        }


class RiemannianChartAtlas:
    """A non-stationary atlas with inversion-free local whitening."""

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
    ) -> None:
        self.rank = int(rank)
        self.capacity = int(capacity)
        self.tau_orth = float(tau_orth)
        self.tau_cap = float(tau_cap)
        self.tau_margin = tau_margin
        self.top_charts = int(top_charts)
        self.top_charts_max = int(top_charts_max)
        self.oja_rate = float(oja_rate)
        self.eps = float(eps)
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
        self.explained: List[float] = []
        self.digest = ""
        self._centroid_matrix: Optional[np.ndarray] = None

    # ---------------------------------------------------------------- build
    @staticmethod
    def _cluster(
        grams: np.ndarray, cluster_count: int, iterations: int = 24
    ) -> np.ndarray:
        count = int(grams.shape[0])
        cluster_count = int(max(1, min(cluster_count, count)))
        stride = max(1, count // cluster_count)
        picks = [min(count - 1, index * stride) for index in range(cluster_count)]
        centroids = np.stack([_unit(grams[index]) for index in picks])
        assignment = np.full(count, -1, dtype=np.int64)
        for _ in range(iterations):
            scores = np.real(grams @ np.conj(centroids.T))
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

    def build(
        self, grams: np.ndarray, rows: Sequence[int], namespace: str = "default"
    ) -> Dict[str, object]:
        matrix = np.asarray(grams, dtype=np.complex128)
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
        cluster_count = int(max(1, math.ceil(count / float(self.capacity))))
        assignment = self._cluster(matrix, cluster_count)
        for chart_id in range(int(assignment.max()) + 1):
            member_index = np.flatnonzero(assignment == chart_id)
            if member_index.size == 0:
                continue
            chart = self._frame(matrix[member_index], namespace)
            chart.rows = [int(rows[int(index)]) for index in member_index]
            chart.count = int(member_index.size)
            self.charts.append(chart)
            self.namespaces.setdefault(str(namespace), []).append(len(self.charts) - 1)
        self.digest = hashlib.sha256(
            b"".join(_unit(chart.centroid).tobytes() for chart in self.charts)
        ).hexdigest()[:16]
        self._centroid_matrix = None
        return self.stats()

    # -------------------------------------------------------------- online
    def _namespace_charts(self, namespace: Optional[str]) -> List[int]:
        if namespace is None:
            return list(range(len(self.charts)))
        return list(self.namespaces.get(str(namespace), []))

    def observe(
        self, gram: np.ndarray, row: int, namespace: str = "online"
    ) -> Dict[str, object]:
        """Absorb a write into its chart, or spawn a sovereign chart."""
        vector = _unit(gram)
        namespace = str(namespace)
        candidates = self._namespace_charts(namespace)
        if not candidates:
            # No mature chart explains this namespace yet. Staging the write
            # keeps it exactly readable while the chart count stays bounded.
            self.staging.stage(vector, int(row), namespace)
            self.staged += 1
            condensed = self._maybe_condense(namespace)
            return {
                "action": "stage",
                "reason": "no-mature-chart",
                "staged": int(len(self.staging)),
                "condensed": condensed,
            }
        cosine = np.array(
            [float(np.real(np.vdot(self.charts[index].centroid, vector))) for index in candidates]
        )
        nearest = candidates[int(np.argmax(cosine))]
        chart = self.charts[nearest]
        tangent = chart.tangent(vector)
        rho = chart.unexplained(tangent)
        cap = float(np.real(np.vdot(chart.centroid, vector)))
        self.explained.append(1.0 - rho)
        if rho > self.tau_orth or cap < self.tau_cap or chart.count >= self.capacity:
            # An unexplained write enters the nascent pool instead of birthing
            # a singleton chart. A full pool condenses coherent clusters.
            self.staging.stage(vector, int(row), namespace)
            self.staged += 1
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
        return {
            "action": "absorb",
            "chart": nearest,
            "rho_orth": float(rho),
            "cosine": float(cap),
        }

    # ------------------------------------------------------- condensation
    def _maybe_condense(self, namespace: str) -> int:
        """Condense one namespace when its nascent pool is full."""
        if len(self.staging.rows(namespace)) < int(self.b_stage):
            return 0
        return self.condense(namespace)

    def finalize(self) -> int:
        """Condense every remaining nascent pool, allowing small clusters.

        The stress test ends with a partial pool. Leaving it staged is exact
        but keeps a second scan tier alive; condensing it makes the mature
        atlas a complete partition of the online rows.
        """
        born = 0
        for namespace in list(self.staging.namespaces()):
            entries = self.staging.drain(namespace)
            if not entries:
                continue
            grams = np.stack([entry[0] for entry in entries])
            cluster_count = int(max(1, min(self.k_condense, len(entries))))
            assignment = self._cluster(grams, cluster_count)
            for chart_id in range(int(assignment.max()) + 1):
                member = np.flatnonzero(assignment == chart_id)
                if member.size == 0:
                    continue
                chart = self._frame(grams[member], namespace)
                chart.rows = [int(entries[int(index)][1]) for index in member.tolist()]
                chart.count = int(member.size)
                self.charts.append(chart)
                self.namespaces.setdefault(str(namespace), []).append(len(self.charts) - 1)
                born += 1
            self.condensed_rows += int(len(entries))
            self.condensations += 1
        if born:
            self._centroid_matrix = None
            self.condensed_spawns += born
        return born

    def condense(self, namespace: str) -> int:
        """Deterministic spherical k-means over one nascent pool.

        Coherent clusters of at least `min_cluster` writes become mature
        charts. Isolated outliers return to the pool. Existing charts are
        read, never written, so their frames cannot drift.
        """
        entries = self.staging.drain(namespace)
        if not entries:
            return 0
        grams = np.stack([entry[0] for entry in entries])
        cluster_count = int(
            max(1, min(self.k_condense, len(entries) // self.min_cluster))
        )
        assignment = self._cluster(grams, cluster_count)
        born = 0
        promoted_rows = 0
        for chart_id in range(int(assignment.max()) + 1):
            member = np.flatnonzero(assignment == chart_id)
            if int(member.size) < self.min_cluster:
                for index in member.tolist():
                    vector, row, name = entries[int(index)]
                    self.staging.stage(vector, row, name)
                continue
            chart = self._frame(grams[member], namespace)
            chart.rows = [int(entries[int(index)][1]) for index in member.tolist()]
            chart.count = int(member.size)
            self.charts.append(chart)
            self.namespaces.setdefault(str(namespace), []).append(len(self.charts) - 1)
            born += 1
            promoted_rows += int(member.size)
        if born:
            self._centroid_matrix = None
        self.condensations += 1
        self.condensed_spawns += born
        self.condensed_rows += promoted_rows
        return born

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
        chart.rows.append(int(row))
        chart.count = int(count + 1)
        self._centroid_matrix = None

    # --------------------------------------------------------------- route
    def select(
        self,
        query_gram: np.ndarray,
        *,
        namespace: Optional[str] = "default",
        top_charts: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        candidates = self._namespace_charts(namespace)
        nascent_rows = self.staging.rows(namespace)
        if not candidates:
            # Only staged writes exist for this namespace. The bounded pool is
            # scanned exactly, so recall is immediate and still correct.
            selected = np.asarray(sorted(set(nascent_rows)), dtype=np.int64)
            return selected, {
                "probed": 0,
                "reason": "nascent-only",
                "margin": 0.0,
                "c1": 0.0,
                "c2": -1.0,
                "expanded": False,
                "charts": [],
                "rho_orth": [],
                "refined": False,
                "staged": int(len(nascent_rows)),
            }
        vector = _unit(query_gram)
        if self._centroid_matrix is None or int(self._centroid_matrix.shape[0]) != len(self.charts):
            self._centroid_matrix = np.stack([_unit(chart.centroid) for chart in self.charts])
        cosine_all = np.real(self._centroid_matrix @ np.conj(vector))
        local = np.asarray(candidates, dtype=np.int64)
        order_local = local[np.argsort(-cosine_all[local], kind="stable")]
        tau = float(self.tau_margin if self.tau_margin is not None else 0.05)
        take = int(self.top_charts if top_charts is None else top_charts)
        namespace_rows = int(sum(len(self.charts[index].rows) for index in candidates))
        if nascent_rows or namespace_rows <= 1024:
            # A namespace with a live nascent pool is still consolidating, and
            # a small namespace is cheap to read whole. In both cases the
            # router reads every chart of that namespace, so recall is exact;
            # a large settled namespace still routes by centroid.
            take = max(take, int(len(candidates)))
        raw_margin = (
            float(cosine_all[int(order_local[0])] - cosine_all[int(order_local[1])])
            if order_local.size > 1
            else 1.0
        )
        expanded = False
        if raw_margin >= tau:
            # The centroid boundary is unambiguous: the tangent residual
            # cannot overturn a gap wider than its own bounded correction.
            # Skip the Woodbury refinement entirely on this query.
            first = float(cosine_all[int(order_local[0])])
            second = (
                float(cosine_all[int(order_local[1])]) if order_local.size > 1 else -1.0
            )
            margin = first - second
            winners = [int(value) for value in order_local[: max(1, min(take, order_local.size))].tolist()]
            rows: List[int] = []
            for chart_id in winners:
                rows.extend(self.charts[chart_id].rows)
            rows.extend(nascent_rows)
            selected = np.asarray(sorted(set(rows)), dtype=np.int64)
            return selected, {
                "probed": int(len(winners)),
                "margin": float(margin),
                "c1": float(first),
                "c2": float(second),
                "expanded": False,
                "charts": winners,
                "rho_orth": [0.0 for _ in winners],
                "refined": False,
                "staged": int(len(nascent_rows)),
            }
        # Boundary query: refine the two best centroid matches with the
        # tangent-space residual.
        coarse = order_local[: min(2, order_local.size)]
        scored: List[Tuple[int, float, float, float]] = []
        for index in coarse.tolist():
            chart = self.charts[int(index)]
            projection = np.vdot(chart.centroid, vector)
            tangent = vector - projection * chart.centroid
            rho = chart.unexplained(tangent)
            score = float(cosine_all[int(index)] - 0.25 * rho)
            scored.append((int(index), score, float(cosine_all[int(index)]), float(rho)))
        scored.sort(key=lambda row: -row[1])
        if len(scored) < 2 and order_local.size > len(scored):
            fallback = int(order_local[len(scored)])
            scored.append((fallback, float(cosine_all[fallback]), float(cosine_all[fallback]), 0.0))
        first = scored[0][1]
        second = scored[1][1] if len(scored) > 1 else -1.0
        margin = first - second
        if len(scored) > 1 and margin < tau:
            take = max(take, min(self.top_charts_max, order_local.size))
            expanded = True
        take = max(1, int(take))
        if expanded or take > len(scored):
            # A broad route reads every chart it named, ordered by centroid.
            take = min(take, int(order_local.size))
            winners = [int(value) for value in order_local[:take]]
            rho_map = {int(row[0]): float(row[3]) for row in scored}
            rhos = [float(rho_map.get(chart_id, 0.0)) for chart_id in winners]
        else:
            take = min(take, len(scored))
            winners = [scored[index][0] for index in range(take)]
            rhos = [float(scored[index][3]) for index in range(take)]
        rows: List[int] = []
        for chart_id in winners:
            rows.extend(self.charts[chart_id].rows)
        rows.extend(nascent_rows)
        selected = np.asarray(sorted(set(rows)), dtype=np.int64)
        return selected, {
            "probed": int(take),
            "margin": float(margin),
            "c1": float(first),
            "c2": float(second),
            "expanded": bool(expanded),
            "charts": winners,
            "rho_orth": rhos,
            "staged": int(len(nascent_rows)),
        }

    def stats(self) -> Dict[str, object]:
        sizes = [int(chart.count) for chart in self.charts]
        return {
            "charts": int(len(self.charts)),
            "rows": int(sum(sizes)),
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
            "mean_chart": float(np.mean(sizes) if sizes else 0.0),
            "digest": str(self.digest),
        }
