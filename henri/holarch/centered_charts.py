"""Anisotropic mean-centered chart routing for Zone C (Directive T2).

The S3 ChartIndex routes a query by raw centroid coherence. On long natural
language prompts the Bochner sequence waves share a non-zero mean direction,
so every engram sits inside a narrow spherical cap and the raw centroid score
is dominated by the shared projection. The discriminative residual is the
component orthogonal to that mean. This module fits that residual, whitens it,
and routes on the whitened geometry.

The index never changes a stored wave and never changes the exact-coherence
rescoring at egress. It only chooses WHICH resident rows are scored, and the
caller always rescores the selected rows with the deployed arithmetic.
"""

from __future__ import annotations

import hashlib
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_EPS = 1e-12


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return (matrix / np.maximum(norms, _EPS)).astype(np.complex128)


class CenteredChartIndex:
    """Mean-centered, ZCA-whitened chart routing with a boundary margin.

    Build once from the resident Gram rows of one namespace. A query is
    transformed by the SAME mean and whitener, scored against the chart
    centroids, and expanded to top_charts_max charts when the top-1 / top-2
    margin is below tau_boundary. Selection returns global store rows.
    """

    def __init__(
        self,
        *,
        capacity: int = 64,
        affinity: float = 0.10,
        tau_boundary: float = 0.10,
        top_charts: int = 4,
        top_charts_max: int = 6,
        eps: float = 1e-3,
        namespace: Optional[str] = None,
        replication: int = 1,
    ) -> None:
        self.capacity = int(max(int(capacity), 1))
        self.affinity = float(affinity)
        self.tau_boundary = float(tau_boundary)
        self.top_charts = int(max(int(top_charts), 1))
        self.top_charts_max = int(max(int(top_charts_max), self.top_charts))
        self.eps = float(eps)
        # Each row may be indexed in its top-r chart centroids. One assignment
        # makes the router depend on a single centroid; replication gives a
        # row r independent chances to be in the probed union at a small cost
        # in rows scanned. The centroids themselves are the k-means centroids.
        self.replication = int(max(int(replication), 1))
        self.namespace = None if namespace is None else str(namespace)
        self.mu = np.zeros(0, dtype=np.complex128)
        self.whitener = np.zeros((0, 0), dtype=np.complex128)
        self.eigenvalues = np.zeros(0, dtype=np.float64)
        self.centroids = np.zeros((0, 0), dtype=np.complex128)
        self.charts: List[Dict[str, object]] = []
        self.centroids_conj = np.zeros((0, 0), dtype=np.complex128)
        self.rows = np.zeros(0, dtype=np.int64)
        self.size = 0
        self.synchronized = False
        self.digest = ""
        self.probes = 0
        self.expanded = 0
        self.scanned = 0

    # ------------------------------------------------------------------ fit
    @staticmethod
    def _whiten(
        grams: np.ndarray, eps: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (mu, W, eigenvalues) for the full-rank Gram ZCA.

        The Gram width is 32 in the deployed profile, so the Hermitian
        eigendecomposition is exact and cheap. W = V diag(1/sqrt(lambda+eps))
        V^H isotropises the residual covariance: for C = V diag(lambda) V^H,
        W^H C W = diag(lambda/(lambda+eps)) -> I as eps -> 0.
        """
        mu = np.asarray(grams.mean(axis=0), dtype=np.complex128)
        residual = np.asarray(grams, dtype=np.complex128) - mu
        count = max(int(residual.shape[0]), 1)
        covariance = (residual.conj().T @ residual) / float(count)
        eigenvalues, vectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(np.real(eigenvalues), 0.0)
        inverse = 1.0 / np.sqrt(eigenvalues + float(eps))
        whitener = (vectors * inverse[None, :]) @ vectors.conj().T
        return mu, whitener.astype(np.complex128), eigenvalues

    def _transform(self, grams: np.ndarray) -> np.ndarray:
        residual = np.asarray(grams, dtype=np.complex128) - self.mu
        return _unit_rows(residual @ self.whitener)

    def _transform_one(self, gram: np.ndarray) -> np.ndarray:
        """Single-query transform without the batch norm call."""
        residual = np.asarray(gram, dtype=np.complex128).reshape(-1) - self.mu
        transformed = residual @ self.whitener
        norm = float(np.linalg.norm(transformed))
        return transformed / max(norm, _EPS)

    def build(
        self,
        grams: np.ndarray,
        *,
        rows: Optional[Sequence[int]] = None,
        namespace: Optional[str] = None,
    ) -> Dict[str, object]:
        """Fit the mean, the whitener and the charts over the given rows."""
        matrix = np.asarray(grams, dtype=np.complex128)
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            raise ValueError("a centered chart index needs a non-empty Gram matrix")
        self.namespace = namespace if namespace is None else str(namespace)
        self.rows = np.arange(matrix.shape[0], dtype=np.int64) if rows is None else np.asarray(rows, dtype=np.int64)
        self.size = int(matrix.shape[0])
        self.mu, self.whitener, self.eigenvalues = self._whiten(matrix, self.eps)
        centered = self._transform(matrix)
        # K-means, not a greedy cap fill. Greedy assignment can leave a row in
        # a chart whose centroid is NOT its nearest one once the preferred
        # chart is full, and a query near that row then routes elsewhere.
        # Lloyd iterations restore the invariant the router relies on: every
        # indexed row sits in its nearest centroid's chart. The capacity is
        # the target cluster size K = ceil(N / capacity), not a hard cap.
        cluster_count = int(max(1, math.ceil(self.size / float(self.capacity))))
        cluster_count = int(min(cluster_count, self.size))
        stride = max(1, self.size // cluster_count)
        picks = [min(self.size - 1, index * stride) for index in range(cluster_count)]
        centroids = np.stack([centered[int(index)] for index in picks]).astype(np.complex128)
        assignment = np.full(self.size, -1, dtype=np.int64)
        for _iteration in range(24):
            scores = np.real(centered @ centroids.conj().T)
            updated = np.argmax(scores, axis=1).astype(np.int64)
            if np.array_equal(updated, assignment):
                break
            assignment = updated
            for chart_id in range(cluster_count):
                members = centered[assignment == chart_id]
                if members.size == 0:
                    continue
                total = members.sum(axis=0)
                centroids[chart_id] = total / max(float(np.linalg.norm(total)), _EPS)
        self.charts = []
        self.centroids = centroids
        self.centroids_conj = np.conj(centroids)
        if self.replication <= 1:
            memberships = [
                np.flatnonzero(assignment == chart_id).astype(np.int64)
                for chart_id in range(cluster_count)
            ]
        else:
            take = int(min(self.replication, cluster_count))
            scores = np.real(centered @ centroids.conj().T)
            top = np.argpartition(-scores, take - 1, axis=1)[:, :take]
            memberships = [
                np.flatnonzero((top == chart_id).any(axis=1)).astype(np.int64)
                for chart_id in range(cluster_count)
            ]
        for chart_id in range(cluster_count):
            rows = memberships[chart_id]
            self.charts.append(
                {"rows": rows.tolist(), "size": int(rows.size), "row_array": rows}
            )
        self.synchronized = True
        self.digest = hashlib.sha256(
            np.ascontiguousarray(
                np.concatenate([self.mu.reshape(-1), self.whitener.reshape(-1)])
            ).tobytes()
        ).hexdigest()[:16]
        return self.stats()

    # --------------------------------------------------------------- route
    def centroids_for(self) -> np.ndarray:
        return self.centroids

    def select(
        self,
        query_gram: np.ndarray,
        *,
        top_charts: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        """Return global rows to score and the routing decision."""
        if not self.synchronized or not self.charts:
            return np.zeros(0, dtype=np.int64), {"probed": 0, "reason": "not-built"}
        transformed = self._transform_one(query_gram)
        coherence = np.real(self.centroids_conj @ transformed)
        order = np.argsort(-coherence, kind="stable")
        first = float(coherence[int(order[0])]) if order.size else 0.0
        second = float(coherence[int(order[1])]) if order.size > 1 else -1.0
        margin = first - second
        take = int(self.top_charts if top_charts is None else top_charts)
        take = max(1, min(take, self.top_charts_max, len(self.charts)))
        expanded = False
        if order.size > 1 and margin < self.tau_boundary:
            take = min(self.top_charts_max, len(self.charts))
            expanded = True
            self.expanded += 1
        winners = order[:take]
        rows: List[np.ndarray] = []
        for chart_id in winners:
            rows.append(np.asarray(self.charts[int(chart_id)]["row_array"], dtype=np.int64))
        local = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
        self.probes += 1
        self.scanned += int(local.size)
        selected = self.rows[local] if local.size else np.zeros(0, dtype=np.int64)
        return selected, {
            "probed": int(take),
            "margin": float(margin),
            "c1": float(first),
            "c2": float(second),
            "expanded": bool(expanded),
            "tau_boundary": float(self.tau_boundary),
            "charts": [int(value) for value in winners.tolist()],
        }

    def calibrate_margin(self, grams: np.ndarray, *, sample: int = 256) -> Dict[str, object]:
        """Set tau_boundary from the store's own margin distribution.

        A deterministic stride sample of the fitted rows is routed with the
        whitened geometry. The median top-1 / top-2 margin is the boundary:
        half of the store is unambiguous and half asks for a wider probe.
        Calibration uses stored rows, never evaluation queries.
        """
        matrix = np.asarray(grams, dtype=np.complex128)
        count = int(matrix.shape[0])
        if count == 0:
            return {"tau_boundary": float(self.tau_boundary), "samples": 0}
        stride = max(1, count // max(int(sample), 1))
        margins: List[float] = []
        for index in range(0, count, stride):
            transformed = self._transform(matrix[index : index + 1])[0]
            coherence = np.real(self.centroids.conj() @ transformed)
            if coherence.size < 2:
                continue
            ordered = np.sort(coherence)[::-1]
            margins.append(float(ordered[0] - ordered[1]))
        if margins:
            # The boundary is the 10th percentile, not the median: only the
            # genuinely ambiguous tenth of the store widens the probe. A
            # median boundary expands half the queries, which buys no extra
            # exact match and pays the multi-probe cost on every read.
            self.tau_boundary = float(np.percentile(margins, 10))
        return {
            "tau_boundary": float(self.tau_boundary),
            "samples": int(len(margins)),
            "margin_p10": float(np.percentile(margins, 10)) if margins else 0.0,
            "margin_p50": float(np.median(margins)) if margins else 0.0,
            "margin_p90": float(np.percentile(margins, 90)) if margins else 0.0,
        }

    def stats(self) -> Dict[str, object]:
        sizes = [int(chart["size"]) for chart in self.charts]
        return {
            "rows": int(self.size),
            "charts": int(len(self.charts)),
            "capacity": int(self.capacity),
            "affinity": float(self.affinity),
            "tau_boundary": float(self.tau_boundary),
            "top_charts": int(self.top_charts),
            "top_charts_max": int(self.top_charts_max),
            "replication": int(self.replication),
            "largest_chart": int(max(sizes) if sizes else 0),
            "mean_chart": float(np.mean(sizes) if sizes else 0.0),
            "eigen_min": float(self.eigenvalues.min()) if self.eigenvalues.size else 0.0,
            "eigen_max": float(self.eigenvalues.max()) if self.eigenvalues.size else 0.0,
            "digest": str(self.digest),
            "synchronized": bool(self.synchronized),
        }
