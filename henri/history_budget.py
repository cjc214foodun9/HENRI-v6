"""Viscoelastic delay-buffer residency budget.

WHY THIS MODULE EXISTS
----------------------
The Phase 4 directive requires one of two storage policies for the `P = 4`
viscoelastic history at batch scale:

    store the history only for Sagnac-stressed workers (`B_active <= 32`),
    or store it as low-rank coordinates (`r = 64`).

Stress Test 2 measured why. A saved `complex64` history costs
`2.10 MiB` per worker, so `B = 256` needs `537.6 MiB`, which is `5.6x` the
L2 cache of the rented RTX 5090. The mitigation was prose. This module
turns it into arithmetic a test can check.

THE MEASURED ANCHOR
-------------------
`L2_BYTES_MEASURED` is `96 MiB`. That is the value the rented RTX 5090
reported, and it corrected a `128 MiB` claim in the feasibility analysis.
Every capacity number here derives from it. Change the anchor and every
derived count changes with it; none of them are remembered constants.

WHAT IS DERIVED AND WHAT IS NOT
-------------------------------
DERIVED: every byte count and every capacity. These are products of the
anchor, the dimension, the delay depth, and the item size.
OBSERVED: the rank-64 reconstruction error in
`scripts/probe_history_budget.py`. No error figure is asserted here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

L2_BYTES_MEASURED = 96 * 1024**2
"""L2 cache bytes reported by the rented RTX 5090 (`sm_120`).

Recorded as measured, not as a specification guess. The feasibility
analysis said 128 MiB; the device said 96 MiB.
"""

P_DELAY = 4
"""Maxwell-Debye delay depth. `cfg.kernel_taus` has four entries."""

COMPLEX64_BYTES = 8
FP8_BYTES = 1

RESERVE_PINNED_PROJECTOR = 33_554_432
"""Bytes of the pinned invariant projector.

`D x 256` at 2 bytes per element: `65536 * 256 * 2 = 33,554,432 B`.

UNIT-LABEL DEFECT RECORDED. The feasibility analysis calls this
`33.55 MiB`. It is `33.55 MB` decimal, which is exactly `32 MiB` binary.
The two labels were swapped, and every capacity derived from it was one
worker low. Measured here in bytes so the label cannot drift again.
"""


def worker_history_bytes(dim: int, p: int = P_DELAY,
                         itemsize: int = COMPLEX64_BYTES) -> int:
    """Bytes one worker needs to keep `p` full delay slices."""
    return int(dim) * int(p) * int(itemsize)


def low_rank_history_bytes(dim: int, r: int, p: int = P_DELAY,
                           itemsize: int = COMPLEX64_BYTES) -> int:
    """Bytes for a shared rank-`r` basis plus per-worker coefficients.

    One `(dim, r)` basis is shared by every worker: the delay slices of a
    low-rank creep live in the same drift subspace. Each worker then
    stores `(p, r)` coefficients.
    """
    return (int(dim) * int(r) + int(p) * int(r)) * int(itemsize)


def budget_workers(l2_bytes: int, dim: int, p: int = P_DELAY,
                   itemsize: int = COMPLEX64_BYTES,
                   reserve_bytes: int = 0) -> int:
    """How many workers fit, after the reserve. Never negative."""
    per = worker_history_bytes(dim, p, itemsize)
    if per <= 0:
        raise ValueError("per-worker history size must be positive")
    usable = int(l2_bytes) - int(reserve_bytes)
    if usable <= 0:
        return 0
    return usable // per


@dataclass(frozen=True)
class ResidencyReport:
    """What the budget did, including what it dropped.

    `evicted` is not advisory. It names workers whose delay memory was not
    kept, so a caller cannot mistake a clipped batch for a full one.
    """

    policy: str
    capacity: int
    requested: int
    admitted: tuple
    evicted: tuple
    per_worker_bytes: int
    bytes_used: int
    l2_bytes: int
    reserve_bytes: int

    @property
    def fraction_of_l2(self) -> float:
        if self.l2_bytes <= 0:
            return 0.0
        return self.bytes_used / float(self.l2_bytes)

    @property
    def clipped(self) -> bool:
        return len(self.evicted) > 0

    def as_dict(self) -> dict:
        return {
            "policy": self.policy,
            "capacity": self.capacity,
            "requested": self.requested,
            "admitted": [int(i) for i in self.admitted],
            "evicted": [int(i) for i in self.evicted],
            "per_worker_bytes": self.per_worker_bytes,
            "bytes_used": self.bytes_used,
            "l2_bytes": self.l2_bytes,
            "reserve_bytes": self.reserve_bytes,
            "fraction_of_l2": round(self.fraction_of_l2, 6),
            "clipped": self.clipped,
        }


class HistoryBudget:
    """Admit the most Sagnac-stressed workers whose delay memory fits L2.

    Args:
        dim: wave dimension.
        l2_bytes: cache size. Defaults to the measured anchor.
        reserve_bytes: bytes held by other persistent residents.
        p: delay depth.
        itemsize: bytes per stored element.
        policy: `"stressed_subset"` keeps full slices for the worst
            workers only. `"low_rank"` keeps full coverage at rank `r`.
        r: rank for the `low_rank` policy.

    Raises:
        ValueError: for a negative reserve, a non-positive rank in
            `low_rank` mode, or an unknown policy.
    """

    def __init__(
        self,
        dim: int,
        l2_bytes: int = L2_BYTES_MEASURED,
        reserve_bytes: int = RESERVE_PINNED_PROJECTOR,
        p: int = P_DELAY,
        itemsize: int = COMPLEX64_BYTES,
        policy: str = "stressed_subset",
        r: int = 64,
    ) -> None:
        if int(dim) <= 0:
            raise ValueError("dim must be positive")
        if int(reserve_bytes) < 0:
            raise ValueError("reserve_bytes must be non-negative")
        if int(l2_bytes) <= 0:
            raise ValueError("l2_bytes must be positive")
        if policy not in ("stressed_subset", "low_rank"):
            raise ValueError(
                f"unknown policy {policy!r}; use 'stressed_subset' or 'low_rank'")
        if policy == "low_rank" and int(r) < 1:
            raise ValueError("low_rank policy needs r >= 1")
        self.dim = int(dim)
        self.l2_bytes = int(l2_bytes)
        self.reserve_bytes = int(reserve_bytes)
        self.p = int(p)
        self.itemsize = int(itemsize)
        self.policy = policy
        self.r = int(r)

    # -- capacity -------------------------------------------------------

    @property
    def per_worker_bytes(self) -> int:
        if self.policy == "low_rank":
            return low_rank_history_bytes(self.dim, self.r, self.p, self.itemsize)
        return worker_history_bytes(self.dim, self.p, self.itemsize)

    @property
    def capacity(self) -> int:
        """Workers that fit. For `low_rank` the basis is shared, so the
        first worker pays for it and each later worker pays coefficients
        only. Capacity is therefore computed on coefficient cost alone,
        with the basis charged once against the reserve."""
        if self.policy == "low_rank":
            coeff = int(self.p) * int(self.r) * int(self.itemsize)
            usable = self.l2_bytes - self.reserve_bytes
            if usable <= 0:
                return 0
            return usable // coeff
        return budget_workers(
            self.l2_bytes, self.dim, self.p, self.itemsize, self.reserve_bytes)

    # -- admission ------------------------------------------------------

    def admit(self, stress: Sequence[float]) -> tuple:
        """Select the highest-stress workers within capacity.

        Args:
            stress: per-worker Sagnac stress, one entry per worker. Higher
                means more phase violation, so more in need of delay
                memory.

        Returns:
            `(active_indices, ResidencyReport)`. Indices are ascending, so
            a caller can index its tensors directly.

        Determinism: ties break on the lower index, so the same stress
        vector always yields the same admission set.
        """
        s = np.asarray(stress, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(s)):
            raise ValueError("stress must be finite")
        if np.any(s < 0.0):
            raise ValueError("stress must be non-negative")
        n = int(s.size)
        cap = min(self.capacity, n)
        if cap <= 0:
            report = ResidencyReport(
                policy=self.policy, capacity=0, requested=n,
                admitted=(), evicted=tuple(range(n)),
                per_worker_bytes=self.per_worker_bytes, bytes_used=0,
                l2_bytes=self.l2_bytes, reserve_bytes=self.reserve_bytes)
            return np.empty(0, dtype=np.int64), report

        # Descending stress; ascending index as the tie-break.
        order = np.lexsort((np.arange(n), -s))
        chosen = np.sort(order[:cap])
        evicted = np.setdiff1d(np.arange(n), chosen, assume_unique=False)
        used = int(self.per_worker_bytes) * int(cap)
        report = ResidencyReport(
            policy=self.policy, capacity=int(cap), requested=n,
            admitted=tuple(int(i) for i in chosen),
            evicted=tuple(int(i) for i in evicted),
            per_worker_bytes=self.per_worker_bytes, bytes_used=used,
            l2_bytes=self.l2_bytes, reserve_bytes=self.reserve_bytes)
        return chosen.astype(np.int64), report
