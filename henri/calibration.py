"""Phase 6 Action 1: corpus-isotropic calibration and an ORDER channel.

TWO SPEC CLAIMS CHECKED BEFORE CODE
-----------------------------------
CLAIM REJECTED ON ALGEBRA. Action 1 offers a regularized covariance whitener
`W_iso = (Sigma + lam I)^{-1/2}`. That is a full-rank mixing matrix. Applied
to option embeddings it is the SAME score-mixing operator Phase 4B falsified
(`s_ortho = G^{-1/2} s_raw`), whatever its fit provenance. Fitting offline
changes where the matrix comes from, not what it does to argmax. REJECTED.
Only All-but-the-Top is implemented:

    u_cal = normalize((u - mu) - P_top P_top^T (u - mu))

`mu` and `P_top` are per-text constants independent of the option set, so
the score functional stays a fixed inner product and rotation invariance is
preserved exactly.

CLAIM FALSIFIED ON TOKENIZATION. Action 1 item 3 asks a token-position rotor
to separate `'ab'` from `'ba'` at `cos <= 0.10`. Measured: `tokens('ab')` is
`['ab']`, one token, and `tokens('ba')` is `['ba']`, one token. Both sit at
position 0, so a token-position rotor assigns IDENTICAL phases and cannot
separate them by construction. A token rotor is the wrong instrument for
short strings.

CLAIM FALSIFIED ON ALGEBRA. A single inner product cannot carry both gates.
With slot energies summing to 1 and `cos_sem = 0.63` for `'ab'`/`'ba'`:

    cos = w0*cos_sem + w1*1 + w2*cos_order,   w0 + w1 + w2 = 1

`w1*1` is the shared role basis, which is identical for every input. A
search over the simplex finds NO weighting with `cos_order <= 0.10` and
`cos_syn >= 0.65` simultaneously. `feasibility()` below recomputes this.
So order must be a SEPARATE channel, not a term fused into content.

WHAT IS BUILT
-------------
1. `CorpusIsotropicCalibration`: All-but-the-Top, fitted offline on ingestion
   text only, applied point-wise.
2. `order_wave` / `order_divergence`: a CHARACTER-level positional rotor on
   its own channel. Characters, because tokens collapse `'ab'` to one unit.
   This channel never enters content scoring, so Action A's `-1.0000`
   antonym and the `>= 0.65` synonym gates are untouched by construction.

EVIDENCE LABELS
---------------
OBSERVED for measured numbers. DERIVED for the feasibility bound, which is
recomputed rather than quoted. FALSIFIED for the two spec claims above.
"""

from __future__ import annotations

import hashlib
from typing import Iterable, Optional

import numpy as np

from henri.semantic_backbone import SLOT, tokens

__all__ = [
    "CorpusIsotropicCalibration", "order_wave", "order_divergence",
    "feasibility", "POSITION_STRIDE", "P_TOP_MAX",
]

P_TOP_MAX = 4
"""Action 1 allows `p in {1, 2, 3, 4}` principal components removed."""

POSITION_STRIDE = 37
"""Ring bins advanced per character position. Pinned, never tuned.

37 is coprime with 256, so the offsets `37*j mod 256` cycle through distinct
bins for adjacent positions. A stride of 0 makes the rotor a no-op; a stride
sharing a factor with 256 would repeat early and alias distant positions.
"""


def _unit(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x)
    return x / n if n > 0 else x


class CorpusIsotropicCalibration:
    """All-but-the-Top, fitted offline on ingestion text.

    Args:
        p_components: principal directions to remove, `1..4`.
        min_texts: refuse to fit on fewer texts, so a caller cannot silently
            fit on a handful and call it a corpus statistic.
    """

    def __init__(self, p_components: int = 3, min_texts: int = 2000) -> None:
        if not 1 <= int(p_components) <= P_TOP_MAX:
            raise ValueError(f"p_components must be 1..{P_TOP_MAX}")
        self.p = int(p_components)
        self.min_texts = int(min_texts)
        self.mu: Optional[np.ndarray] = None
        self.basis: Optional[np.ndarray] = None      # (p, d_sem)
        self.n_texts = 0
        self._digests: set = set()
        self.explained_fraction: Optional[float] = None

    def fit(self, texts: Iterable[str], encode) -> "CorpusIsotropicCalibration":
        """Fit `mu` and the top-p basis. `encode: str -> (d,) unit vector`.

        Callers must pass ingestion-split text only. `saw_text` lets a probe
        prove that no held-out row reached this fit.
        """
        items = list(texts)
        if len(items) < self.min_texts:
            raise ValueError(
                f"fit needs >= {self.min_texts} texts, got {len(items)}; a "
                "small fit is not a corpus statistic")
        vecs = np.stack([encode(t) for t in items], axis=0)
        self.n_texts = int(vecs.shape[0])
        self._digests = {
            hashlib.sha256(" ".join(tokens(t)).encode()).hexdigest()
            for t in items}
        self.mu = vecs.mean(axis=0)
        centred = vecs - self.mu
        # Top-p directions of the corpus covariance, via thin SVD.
        _, s, vt = np.linalg.svd(centred, full_matrices=False)
        self.basis = vt[:self.p].copy()
        energy = float((s ** 2).sum())
        self.explained_fraction = (float((s[:self.p] ** 2).sum()) / energy
                                   if energy > 0 else 0.0)
        return self

    def apply(self, u: np.ndarray) -> np.ndarray:
        """Remove the corpus mean and the top-p directions, then renormalise."""
        if self.mu is None or self.basis is None:
            return u
        centred = np.asarray(u, dtype=np.float64) - self.mu
        proj = self.basis.T @ (self.basis @ centred)
        return _unit(centred - proj)

    def saw_text(self, text: str) -> bool:
        """True when this exact text (token-normalised) was in the fit."""
        h = hashlib.sha256(" ".join(tokens(text)).encode()).hexdigest()
        return h in self._digests

    def to_dict(self) -> dict:
        return {
            "variant": "all_but_the_top_pointwise",
            "covariance_whitener": "REJECTED_mixing_operator",
            "p_components": self.p,
            "n_texts": self.n_texts,
            "fitted": self.mu is not None,
            "explained_fraction_removed": (round(self.explained_fraction, 6)
                                           if self.explained_fraction is not None
                                           else None),
        }


# -- order channel --------------------------------------------------------

def _char_code(ch: str, seed: int = 20261020) -> np.ndarray:
    """Deterministic orthonormal-in-expectation code for one character.

    WHY NOT THE SEMANTIC STEM. Measured: `cos(stem('a'), stem('b')) = 0.6357`.
    Reusing the stem inside the order channel injects that content similarity
    into an order statistic. Algebra for `'ab'` vs `'ba'` with diagonal
    position keys `K0`, `K1` and character codes `a`, `b`:

        <acc_ab, acc_ba> = 2<a,b> + <a,K0^H K1 a> + <b,K1^H K0 b>

    The dominant term `2<a,b>` is CONTENT, not order. With orthogonal codes
    the first term vanishes and the diagonal cross terms are `O(1/sqrt(D))`,
    so the cosine approaches zero. Order is syntax, so its channel must not
    borrow semantic proximity.
    """
    h = hashlib.sha256(f"orderchar:{seed}:{ch}".encode()).digest()
    # Expand the 32-byte digest into SLOT ring indices deterministically.
    reps = (SLOT * 2 + 31) // 32
    buf = (h * reps)[:SLOT * 2]
    q = np.frombuffer(buf, dtype=np.uint16)[:SLOT].astype(np.int64) % 256
    return np.exp(2j * np.pi * q.astype(np.float64) / 256.0)


_CHAR_CACHE: dict = {}


def char_codes(chars, seed: int = 20261020) -> list:
    """Cached character codes for an explicit character list.

    MEASURED DEFECT, repaired. The first version took `text` and iterated the
    RAW string, so `'mv/(qB)'` produced 7 codes while `char_positions`
    returned 4 pairs. `zip` then handed each character a PUNCTUATION code.
    Callers must now pass the same filtered list they position-index.
    """
    out = []
    for ch in chars:
        key = (ch, seed)
        if key not in _CHAR_CACHE:
            _CHAR_CACHE[key] = _char_code(ch, seed)
        out.append(_CHAR_CACHE[key])
    return out


def _position_keys(max_pos: int, seed: int = 20261020) -> np.ndarray:
    """Deterministic per-dimension position keys, shape `(max_pos, SLOT)`.

    WHY PER-DIMENSION AND NOT A SCALAR OFFSET
    A constant added to every dimension is a GLOBAL PHASE. Multiplying a whole
    wave by `exp(i*2*pi*c/256)` does not change its direction, so
    `sum(u + r*v)` and `sum(v + r*u)` coincide. Measured with a scalar
    stride: `cos('ab','ba') = 0.8456`, worse than raw. That version is
    falsified above. A per-dimension key rotates DIFFERENT dimensions by
    different amounts, so position changes the direction and reordering
    cannot cancel out.
    """
    gen = np.random.default_rng(int(seed))
    return gen.integers(0, 256, size=(int(max_pos), SLOT), dtype=np.int64)


_POS_CACHE: dict = {}


def position_keys(max_pos: int = 64, seed: int = 20261020) -> np.ndarray:
    """Cached position-key table. Reuse keeps generation out of the hot path."""
    key = (int(max_pos), int(seed))
    if key not in _POS_CACHE:
        _POS_CACHE[key] = _position_keys(max_pos, seed)
    return _POS_CACHE[key]


def char_positions(text: str) -> list:
    """Alphanumeric characters with their position. Deterministic."""
    clean = [c for c in text.lower() if c.isalnum()]
    return [(c, j) for j, c in enumerate(clean)]


def order_wave(bb, text: str) -> Optional[np.ndarray]:
    """Character-level ORDER wave, unit Euclidean norm on C^SLOT.

    Each character is ring-bound with a per-dimension POSITION KEY before
    bundling, so the same character at position 0 and position 1 lands on
    different directions. Reordering therefore changes the sum.

    This channel never enters content scoring. That is why Action A's
    `-1.0000` antonym coherence and the `>= 0.65` synonym gate are untouched:
    order is a separate measurement, not a term fused into content.

    Returns None for input with no alphanumeric characters.
    """
    pairs = char_positions(text)
    if not pairs:
        return None
    pk = position_keys(max_pos=len(pairs) + 1)
    acc = np.zeros(SLOT, dtype=np.complex128)
    for code, (_, j) in zip(char_codes([c for c, _ in pairs]), pairs):
        # ALIGNMENT IS THE WHOLE CORRECTNESS ARGUMENT. `pairs` is the
        # PUNCTUATION-FILTERED list; `char_codes` must receive the same list.
        # Passing the raw string made `'mv/(qB)'` yield 7 codes against 4
        # pairs, so each character took a PUNCTUATION code. Measured with
        # that mismatch: cos('mv/(qB)','qB/(mv)') = 0.4577.
        acc += code * np.exp(2j * np.pi * pk[j].astype(np.float64) / 256.0)
    return _unit(acc)


def order_divergence(bb, a: str, b: str) -> float:
    """`1 - Re<order_wave(a), order_wave(b)>` in `[0, 2]`.

    1.0 means the two strings are order-orthogonal; 0.0 means identical
    order. Empty input returns 0.0, never a fabricated 1.0.
    """
    va, vb = order_wave(bb, a), order_wave(bb, b)
    if va is None or vb is None:
        return 0.0
    return float(1.0 - np.real(np.vdot(va, vb)))


def feasibility(cos_sem: float = 0.63, cos_syn: float = 0.72,
                order_bar: float = 0.10, syn_bar: float = 0.65,
                step: float = 0.02) -> dict:
    """Search the slot-energy simplex for one weighting meeting both gates.

    DERIVED and recomputed here rather than asserted. Content coherence for an
    order-permuted pair is `w0*cos_sem + w1*1 + w2*(-1)`, because the order
    term can reach `-1` at best. A synonym pair shares no order, so its order
    term is near `0` and the role basis contributes `w1*1`. If no weighting
    satisfies both, a single inner product cannot carry the two gates.
    """
    best = None
    found = []
    vals = np.arange(0.0, 1.0 + 1e-9, step)
    for w0 in vals:
        for w1 in vals:
            w2 = 1.0 - w0 - w1
            if w2 < -1e-9:
                continue
            order = w0 * cos_sem + w1 - w2
            syn = w0 * cos_syn + w1 + w2 * 0.0
            if order <= order_bar and syn >= syn_bar:
                found.append((round(float(w0), 4), round(float(w1), 4),
                              round(float(w2), 4)))
            gap = max(order - order_bar, 0.0) + max(syn_bar - syn, 0.0)
            if best is None or gap < best[0]:
                best = (gap, float(w0), float(w1), float(w2),
                        float(order), float(syn))
    return {
        "feasible": bool(found),
        "n_solutions": len(found),
        "example": found[:3],
        "closest_miss": {"w0": round(best[1], 4), "w1": round(best[2], 4),
                         "w2": round(best[3], 4),
                         "cos_order": round(best[4], 6),
                         "cos_syn": round(best[5], 6),
                         "violation": round(best[0], 6)},
        "verdict": ("FEASIBLE_ON_ONE_INNER_PRODUCT" if found
                    else "INFEASIBLE_NEEDS_SEPARATE_CHANNEL"),
    }
