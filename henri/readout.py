"""Readout: tie-robust selection plus typed action envelopes.

Evidence label: OBSERVED for the gate metrics this module reports, and
FALSIFIED for the plain-argmax ranking assumption it replaces.

Three retired-repo failures drove this design.

1. Flat argmax over a crowded phase space. Measured: `argmax 0.0 at
   19.5 sigma slot drift`. Use max-group membership, then a within-group
   key that is NOT a monotone function of the similarity.
2. A monotone transform cannot split a tie. `softmax(s/k)` is strictly
   monotone in `s`, so it preserves the maximizer set exactly. No
   temperature can collapse a tie plateau. Measured tie-group sizes:
   24.27 unchanged, 27.24 at k=0.038, 39.0 at k=100.
3. A readout floor taken from the random-wave distribution. Measured:
   live encoder waves are less decisive than isotropic noise. No floor
   is set here; abstention is reported, never tuned into a pass.

So selection is two stage:

  stage 1: the set of indices whose score ties the maximum
  stage 2: a within-group key that breaks the tie without being a
           monotone function of the score

The default stage-2 key is trace frequency, which the retired repo
measured at 0.1556 against 0.0317 for a trained ridge readout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from henri.state import WaveState

_EPS = 1e-30


@dataclass
class ReadoutResult:
    """One selection outcome with its full audit trail.

    Attributes:
        index: the selected codebook index, or -1 when abstaining.
        max_group: indices that tied at the top score.
        group_size: cardinality of the max group.
        top_score: the maximum score.
        second_score: best score outside the max group.
        margin: `top_score - second_score`.
        entropy: normalized softmax entropy over all scores.
        abstained: True when the caller's floor suppressed emission.
    """

    index: int
    max_group: np.ndarray
    group_size: int
    top_score: float
    second_score: float
    margin: float
    entropy: float
    abstained: bool = False
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "group_size": self.group_size,
            "top_score": self.top_score,
            "second_score": self.second_score,
            "margin": self.margin,
            "entropy": self.entropy,
            "abstained": self.abstained,
            **self.meta,
        }


def scores_of(state: WaveState, codebook: np.ndarray) -> np.ndarray:
    """Real coherence of the state against every codebook row."""
    book = np.asarray(codebook, dtype=np.complex128)
    if book.ndim != 2 or book.shape[1] != state.cfg.dim:
        raise ValueError(
            f"codebook must be (M, {state.cfg.dim}); got {book.shape}"
        )
    return np.real(book @ np.conj(state.psi))


def normalized_entropy(scores: np.ndarray) -> float:
    """Entropy of the softmax over scores, divided by `log(M)`.

    Scale-free in [0, 1]. A flat score vector gives 1.0.

    DEFECT RECORDED (2026-10-12). This metric uses a UNIT-temperature
    softmax and is therefore SCALE-BLIND for coherence-scale scores.
    Readout scores are real coherences in `[-1, 1]`, so `s - max(s)`
    lies in `[-2, 0]` and `exp` of it lies in `[0.135, 1]`. With `M=32`
    the largest attainable probability is only `1/(0.135*31 + 1) ~ 0.19`,
    so the reported entropy cannot fall below roughly 0.93 no matter
    how decisive the score surface is.

    Measured at D=4096 (`scripts/probe_metric_scale.py`):

        cell           margin     entropy(beta=1)   entropy(beta=26.1)
        unrelated      0.00523    0.99999           0.98955
        noisy_t0.90    0.86518    0.99376           1.45e-08
        noisy_t0.99    0.95142    0.99203           1.56e-09

    A decisive cell (margin 0.87) and a non-decisive cell (margin
    0.005) differ by 0.006 in this metric. It cannot separate them.
    The margin separates them, and so does `entropy_at_beta`.

    This function is retained BYTE-IDENTICAL so legacy receipts still
    reproduce. Use `entropy_at_beta` for any decisiveness claim. This
    is an additive correction, not a silent edit.
    """
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    m = s.size
    if m < 2:
        return 1.0
    z = s - np.max(s)
    p = np.exp(z)
    total = float(np.sum(p)) + _EPS
    p = p / total
    ent = float(-np.sum(p * np.log(p + _EPS)))
    return ent / float(np.log(m))


def entropy_at_beta(scores: np.ndarray, beta: float = 26.1) -> float:
    """Entropy of the softmax at inverse temperature `beta`, /log(M).

    The scale-aware decisiveness measure. Defaults to the retired
    repository's fitted `beta* = 26.10` (equivalent to `T* = 0.038316`),
    but the value is a parameter, not a transferred constant: the sweep
    in `scripts/probe_metric_scale.py` shows any `beta >= ~8` separates
    the decisive and non-decisive regimes on coherence-scale scores.

    Args:
        scores: raw score vector.
        beta: inverse temperature. Must be positive.

    Returns:
        Normalized entropy in [0, 1]. Near 0 means one attractor
        dominates; near 1 means the distribution is flat.
    """
    if beta <= 0:
        raise ValueError("beta must be positive")
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    m = s.size
    if m < 2:
        return 1.0
    z = float(beta) * (s - np.max(s))
    z = np.clip(z, -700.0, 0.0)
    p = np.exp(z)
    total = float(np.sum(p)) + _EPS
    p = p / total
    ent = float(-np.sum(p * np.log(p + _EPS)))
    return ent / float(np.log(m))


def max_group_indices(scores: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Indices whose score ties the maximum, within `tol`."""
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    return np.flatnonzero(s >= float(np.max(s)) - tol)


def select(
    state: WaveState,
    codebook: np.ndarray,
    group_key: Optional[np.ndarray] = None,
    tol: float = 1e-12,
    floor: Optional[float] = None,
) -> ReadoutResult:
    """Two-stage tie-robust selection.

    Args:
        state: the query wave.
        codebook: array of shape `(M, D)`.
        group_key: optional per-index tie-breaking key. Higher wins.
            Must NOT be a monotone function of the score. Default is a
            uniform-in-group policy.
        tol: tie tolerance on the raw score.
        floor: optional margin floor. `None` means no floor, matching
            the retired repo's decision not to set one from a random
            baseline (constraint C12).

    Returns:
        A ReadoutResult. `margin` is the honest decisiveness measure.
    """
    s = scores_of(state, codebook)
    group = max_group_indices(s, tol=tol)
    top = float(s[group[0]])
    outside = np.setdiff1d(np.arange(s.size), group, assume_unique=False)
    second = float(np.max(s[outside])) if outside.size else -np.inf
    margin = top - second if np.isfinite(second) else float("inf")

    if group_key is not None:
        key = np.asarray(group_key, dtype=np.float64).reshape(-1)
        if key.size != s.size:
            raise ValueError(
                f"group_key must have length {s.size}, got {key.size}"
            )
        local = int(np.argmax(key[group]))
    else:
        local = 0

    index = int(group[local])
    abstained = bool(floor is not None and margin < float(floor))
    return ReadoutResult(
        index=index if not abstained else -1,
        max_group=group,
        group_size=int(group.size),
        top_score=top,
        second_score=second,
        margin=margin,
        entropy=normalized_entropy(s),
        abstained=abstained,
        meta={
            "floor": floor,
            "used_group_key": group_key is not None,
            "tie_fraction": float(group.size) / float(s.size),
        },
    )


def centroid_snap(
    vector: np.ndarray,
    centroids: np.ndarray,
    metric: str = "cosine",
) -> int:
    """Nearest-centroid action dispatch. Training-free.

    Args:
        vector: query vector of shape `(D,)`.
        centroids: array of shape `(K, D)`.
        metric: "cosine" or "euclidean".

    Returns:
        Winning centroid index.
    """
    c = np.asarray(centroids, dtype=np.complex128)
    v = np.asarray(vector, dtype=np.complex128).reshape(-1)
    if c.ndim != 2 or c.shape[1] != v.shape[0]:
        raise ValueError(f"centroids must be (K, {v.shape[0]}); got {c.shape}")

    if metric == "cosine":
        vn = v / (np.linalg.norm(v) + _EPS)
        cn = c / (np.linalg.norm(c, axis=1, keepdims=True) + _EPS)
        return int(np.argmax(np.real(cn @ np.conj(vn))))
    if metric == "euclidean":
        d = np.linalg.norm(c - v[None, :], axis=1)
        return int(np.argmin(d))
    raise ValueError(f"unknown metric {metric!r}")


@dataclass
class ActionEnvelope:
    """A typed action plus its verification status.

    Attributes:
        kind: the action family.
        payload: family-specific parameters.
        confidence: Sagnac confidence in [0, 1].
        verified: True when the veto cleared the action.
        veto_reason: why the action was suppressed, if it was.
    """

    kind: str
    payload: dict
    confidence: float
    verified: bool
    veto_reason: str = ""

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "payload": self.payload,
            "confidence": self.confidence,
            "verified": self.verified,
            "veto_reason": self.veto_reason,
        }


ACTION_KINDS = (
    "translate",
    "rotate",
    "segment",
    "write_memory",
    "exec_tool",
    "abstain",
)
"""Typed action families. No 32k text vocabulary -- the retired repo
measured the action-vocab Hopfield instance near-uniform (constraint
C3) and left the lexical path RETRAIN_REQUIRED."""


def make_envelope(
    kind: str,
    payload: dict,
    confidence: float,
    verified: bool,
    veto_reason: str = "",
) -> ActionEnvelope:
    """Construct a typed envelope, validating the action family."""
    if kind not in ACTION_KINDS:
        raise ValueError(f"unknown action kind {kind!r}; allowed: {ACTION_KINDS}")
    return ActionEnvelope(
        kind=kind,
        payload=dict(payload),
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        verified=bool(verified),
        veto_reason=veto_reason,
    )
