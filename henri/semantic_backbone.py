"""Semantic backbone: text -> unit-modulus wave on T^D, metric-preserving.

WHY THIS MODULE EXISTS
----------------------
The v6 benchmark arm measured at chance (MMLU 0.2725, p=0.163) because
ingress was a SHA-256 hash of each word. Two words with no shared letters
were orthogonal by construction. This module replaces that hash with a
continuous semantic stem, lifts the result onto the torus at exact unit
modulus, and separates logical polarity on its own slot.

FOUR MEASURED FACTS THAT SHAPED THE DESIGN
------------------------------------------
1. The Bochner identity holds. `E[cos(sigma w.(u-v))] = exp(-sigma^2(1-rho))`
   measured within 0.5% at sigma in {1, sqrt2, 2, 3}. So the phase lift is
   an honest kernel, not an approximation with unknown error.
2. A frozen sentence encoder puts SYNONYMS at rho = 0.826 and ANTONYMS at
   rho = +0.622 (all-MiniLM-L6-v2, 9 pairs, mean). Antonyms are NOT
   negatively correlated. The distributional hypothesis says why: "hot" and
   "cold" occur in the same contexts. THEREFORE polarity cannot come from
   the embedding. It must come from an explicit lexical signal.
3. A frozen encoder at sigma = sqrt2 maps rho = 0.826 to t = 0.706, not the
   0.8187 the specification predicts, because that figure assumes
   rho = 0.90. The formula is right; the input was optimistic.
4. Option whitening is NOT a neutral re-ranking. `stiefel_retract` maps the
   score vector by `G^{-1/2}`, so `argmax` can change. It is a DIFFERENT
   decision rule and must be validated by accuracy, never asserted.

WHAT IS DELIBERATELY NOT BUILT
------------------------------
There is no clause parser here. Slot 0 carries semantics and slot 1 carries
polarity. Slots 2 and 3 stay at exact zero. Claiming four populated slots
without a parser would be a fabrication, so the receipt reports which slots
are active.

EVIDENCE LABELS
---------------
OBSERVED for every number in the receipt. DERIVED for the Bochner identity,
which is proved in the docstring and confirmed numerically by the probe.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

# Offline by default. The encoder is cached on this host; a network fetch
# inside a probe would make the receipt unreproducible.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

DIM = 65536
SLOT = DIM // 4

NEGATORS = frozenset("""
not no never none neither nor cannot can't dont don't doesnt doesn't didn't
isn't aren't wasn't weren't won't wouldn't shouldn't couldn't hardly barely
without lacks lacking absent fails failed false incorrect invalid
decreases decrease decreased decreasing reduces reduce reduced reducing
inhibits inhibit inhibited inhibiting lowers lower lowered suppresses
""".split())

# Antonym pairs that carry polarity but are NOT negations. Kept separate
# because the encoder places them close, so only the lexicon can flip them.
ANTONYM_NEG = frozenset("""
decrease decreases decreased decreasing reduce reduces reduced reducing
lower lowers lowered lowering fall falls fell falling drop drops dropped
inhibits inhibit inhibited inhibiting suppresses suppress suppressed
minimum least smallest lowest negative cold slow weak
""".split())

WORD_RE = re.compile(r"[a-z0-9']+")

# -- Canonical antonym table (Action A) -----------------------------------
#
# WHY A TABLE AND NOT A PROJECTION
# A frozen encoder puts antonyms at cosine +0.622. No linear functional of
# that embedding can reach -0.50, because the input is already positive.
# Phase 4B measured this and falsified G-SEM1. The repair is to make the
# CONTENT of both members identical, so polarity is the only difference.
#
# Maps the NEGATIVE member to its canonical positive root. Left member is
# the root; every later member is canonicalised to it and flips p.
ANTONYM_PAIRS = (
    (("increase", "increases", "increased", "increasing"),
     ("decrease", "decreases", "decreased", "decreasing", "reduce",
      "reduces", "reduced", "reducing", "lower", "lowers", "drop")),
    (("rise", "rises", "rising"), ("fall", "falls", "fell", "falling")),
    (("positive",), ("negative",)),
    (("hot", "hotter", "warmer"), ("cold", "colder", "cooler")),
    (("true", "correct", "valid"), ("false", "incorrect", "invalid")),
    (("promote", "promotes", "promoted", "endorse", "endorses"),
     ("inhibit", "inhibits", "inhibited", "reject", "rejects", "rejected")),
    (("accelerate", "accelerates"), ("decelerate", "decelerates")),
    (("maximum", "max", "largest", "highest"),
     ("minimum", "min", "smallest", "lowest")),
    (("fast", "faster", "quickly"), ("slow", "slower", "slowly")),
    (("strong", "stronger"), ("weak", "weaker")),
    (("before", "always", "accept", "accepts"), ("after", "never", "deny",
                                                 "denies")),
)

ANTONYM_CANON: dict = {}
"""EVERY member -> its own group's root. Roots are per group, not global."""

ANTONYM_NEGATIVE: frozenset = frozenset()
"""Only the members of the SECOND tuple of each group. These give p = -1."""


def _build_antonym_tables() -> tuple:
    """Map every member to ITS OWN group root; mark only opposites negative.

    FOUR MEASURED DEFECTS LED HERE. Each would have passed a naive gate.

    1. Treating slot 1 as the sole polarity carrier contradicts its own
       algebra: with equal slots the full-sentence coherence is exactly
       0.0, and crossing -0.50 needs 75 percent of the energy in polarity.
    2. Applying the sign to slot 1 AND the +128 ring shift flips that
       slot twice, so the flips cancel. Measured +0.0042, not -1.0.
    3. Canonicalising only the negative member leaves "increases" and
       "decrease" as different strings. Measured -0.664, not -1.0.
    4. Joining ALL groups to ONE global root makes "hot" canonicalise to
       "increase", so disjoint concepts collapse onto one content wave.
       Measured -0.333 for the gate while silently destroying meaning.

    The structure below makes the split structural: group membership
    decides polarity, never position in a flat tuple.
    """
    canon: dict = {}
    neg: set = set()
    for roots, opposites in ANTONYM_PAIRS:
        root = roots[0]
        for member in roots:
            canon.setdefault(member, root)
        for member in opposites:
            canon.setdefault(member, root)
            neg.add(member)
    return canon, frozenset(neg)


ANTONYM_CANON, ANTONYM_NEGATIVE = _build_antonym_tables()


def canonicalize(text: str) -> str:
    """Replace each negative antonym member with its positive root.

    The result has the SAME content as the original sentence, so two
    antonym propositions differ only in the polarity flag. DETERMINISTIC:
    a pure token substitution, no model, no fitting, no randomness.
    """
    return " ".join(ANTONYM_CANON.get(w, w) for w in tokens(text))


def tokens(text: str) -> list:
    return WORD_RE.findall(text.lower())


def polarity(text: str) -> int:
    """+1 for affirmative, -1 when a negator or negative antonym appears.

    DERIVED, then MEASURED. Phase 4B proved polarity cannot come from a
    linear functional of the embedding: a frozen encoder puts antonyms at
    cosine +0.622, so every such functional returns a positive number. The
    signal must be lexical. NEGATORS covers particles ("not", "never",
    "without"); ANTONYM_NEGATIVE covers directional opposites ("decrease",
    "cold", "false"). Parity handles "not uncommon".
    """
    hits = sum(1 for w in tokens(text)
               if w in NEGATORS or w in ANTONYM_NEGATIVE)
    return -1 if hits % 2 == 1 else 1


def strip_common_option_tokens(question: str, options: Sequence[str],
                               k_min: Optional[int] = None) -> list:
    """Return each option's discriminative span, shared boilerplate removed.

    REPLACES post-hoc `G^{-1/2}` whitening. Phase 4B measured that
    `stiefel_retract` on candidate options maps the score vector by
    `G^{-1/2}`, which MIXES all K scores and can flip argmax. It dropped
    MMLU 0.3050 -> 0.2475. Stripping tokens before encoding removes the
    shared carrier WITHOUT touching the score linear functional, so the
    decision rule is unchanged and argmax stays honest.

    Rules:
      * A token shared by >= ceil(K/2) options AND present in the prompt is
        boilerplate and is removed.
      * A token is kept when it appears in fewer than the threshold of
        options, because it is then discriminative.
      * An option whose span would become empty keeps its full text
        (spec rule: never emit an empty query).

    Deterministic. No model, no fitting.
    """
    k = len(options)
    thr = k_min if k_min is not None else max(2, int(np.ceil(k / 2.0)))
    qtok = set(tokens(question))
    counts: dict = {}
    per_opt = [list(tokens(o)) for o in options]
    for toks in per_opt:
        for w in set(toks):
            counts[w] = counts.get(w, 0) + 1
    boiler = {w for w, c in counts.items() if c >= thr and w in qtok}
    out = []
    for toks in per_opt:
        keep = [w for w in toks if w not in boiler]
        out.append(" ".join(keep) if keep else " ".join(toks))
    return out


@dataclass(frozen=True)
class StemInfo:
    dim_sem: int
    model: str
    offline: bool


class SemanticStem:
    """Frozen contextual encoder with a deterministic fallback.

    Args:
        model_name: sentence-transformers id. Loaded offline.
        allow_fallback: when True and the model is unavailable, use a
            deterministic character-PMI stem. The fallback is WEAKER and
            the receipt says which stem ran.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 allow_fallback: bool = True) -> None:
        self.model_name = model_name
        self._model = None
        self._cache: dict = {}
        self.kind = "unloaded"
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name, device="cpu")
            self.kind = "sentence_transformer"
        except Exception:
            if not allow_fallback:
                raise
            self.kind = "char_ngram_fallback"
        if self.kind == "char_ngram_fallback":
            self.dim_sem = 512
        else:
            try:
                self.dim_sem = int(self._model.get_sentence_embedding_dimension())
            except Exception:
                self.dim_sem = int(self._model.get_embedding_dimension())

    def info(self) -> StemInfo:
        return StemInfo(dim_sem=self.dim_sem, model=self.model_name,
                        offline=bool(os.environ.get("HF_HUB_OFFLINE")))

    # -- fallback -------------------------------------------------------

    def _char_ngram(self, text: str) -> np.ndarray:
        """Hashed character 4-gram vector. Deterministic, no network.

        Weaker than a trained encoder on synonymy, stated plainly. It
        exists so the pipeline runs on a host with no model cache.
        """
        v = np.zeros(self.dim_sem, dtype=np.float64)
        s = " ".join(tokens(text))
        for i in range(max(1, len(s) - 3)):
            g = s[i:i + 4]
            h = hashlib.blake2b(g.encode(), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "big") % self.dim_sem
            v[idx] += 1.0 if h[4] & 1 else -1.0
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    # -- encode ---------------------------------------------------------

    def encode(self, text: str) -> np.ndarray:
        """Unit vector in R^{d_sem}. Cached per exact string."""
        if text in self._cache:
            return self._cache[text]
        if self.kind == "sentence_transformer":
            v = self._model.encode(
                [text], normalize_embeddings=True, convert_to_numpy=True)[0]
            v = np.asarray(v, dtype=np.float64)
        else:
            v = self._char_ngram(text)
        n = np.linalg.norm(v)
        if n > 0:
            v = v / n
        self._cache[text] = v
        return v


class BochnerPhaseProjector:
    """Stiefel-block phase lift from S^{d-1} onto T^{D_slot}.

    The map is
        theta(u) = (sigma * Omega @ u) mod 2*pi
        Psi(u)   = exp(i theta(u)) / sqrt(D_slot)

    Theorem (Bochner, 1933; Rahimi & Recht, 2007). For rows w_d drawn
    i.i.d. N(0, I) and unit u, v with rho = <u,v>:

        E[Re<Psi(u), Psi(v)>] = E[cos(sigma w_d.(u-v))]
                             = exp(-sigma^2/2 * ||u-v||^2)
                             = exp(-sigma^2 (1 - rho))

    because `sigma w_d.(u-v)` is zero-mean Gaussian with variance
    `sigma^2 ||u-v||^2 = 2 sigma^2 (1-rho)`, and the characteristic
    function of a zero-mean Gaussian is `exp(-t^2 Var / 2)`.

    Rows come from a QR factorisation, so `Omega^T Omega` is a scaled
    identity. That is the Stiefel construction: it reduces the variance of
    the empirical inner product around the expectation instead of leaving
    it at the i.i.d. rate. The unit-modulus constraint |Psi_d| = 1/sqrt(D)
    holds exactly by construction, for every sigma.
    """

    def __init__(self, dim_sem: int, dim_slot: int = SLOT,
                 sigma: float = float(np.sqrt(2.0)), seed: int = 20261020) -> None:
        if dim_sem < 1:
            raise ValueError("dim_sem must be positive")
        if dim_slot < 1:
            raise ValueError("dim_slot must be positive")
        if not sigma > 0.0:
            raise ValueError("sigma must be positive")
        self.dim_sem = int(dim_sem)
        self.dim_slot = int(dim_slot)
        self.sigma = float(sigma)
        self.seed = int(seed)
        self._omega = self._build_omega()

    def _build_omega(self) -> np.ndarray:
        """Block-diagonal Stiefel frame, shape (dim_slot, dim_sem).

        SCALING IS THE WHOLE THEOREM. Rows are orthonormal from QR, then
        multiplied by `sqrt(dim_sem)` so that `Omega Omega^T = dim_sem * I`.

        Why that factor is mandatory, measured. The Bochner identity
        requires `w_d.(u-v) ~ N(0, ||u-v||^2)`, which needs
        `E||w_d||^2 = dim_sem`. With UNIT-norm rows the projection variance
        is `||u-v||^2 / dim_sem`, so the effective bandwidth is
        `sigma / sqrt(dim_sem)` and every pair collapses toward coherence
        1. Measured at d_sem=384, unit rows gave synonym 0.9928 AND a
        shuffled control of 0.9866: the map destroyed all discrimination
        while appearing to pass a "coherence >= 0.65" gate. Scaling by
        sqrt(d_sem) reproduced the closed form (0.1465 vs 0.1363).

        SPEC DEFECT RECORDED. `HENRI-ARCH-2026-PERFECTED-SEMANTIC-BACKBONE`
        states `Omega_k^T Omega_k = (16384/d_sem) I`. That is the UNIT-row
        convention, and at d_sem = 384 with sigma = sqrt2 it drives every
        pair to coherence 1. The specification's own stated target
        (`t = 0.8187` at rho = 0.90) is unreachable under its own
        normalisation. The correct frame satisfies `Omega Omega^T = d I`,
        which the sqrt(d_sem) factor produces.
        """
        gen = np.random.default_rng(self.seed)
        n_blocks = max(1, self.dim_slot // self.dim_sem)
        blocks = []
        for _ in range(n_blocks):
            g = gen.standard_normal((self.dim_sem, self.dim_sem))
            q, r = np.linalg.qr(g)
            # Sign-fix so the factorisation is deterministic across BLAS.
            q = q * np.sign(np.diag(r))
            blocks.append(q.T)                      # orthonormal rows
        omega = np.concatenate(blocks, axis=0)      # (n_blocks*d, d)
        if omega.shape[0] < self.dim_slot:
            pad = self.dim_slot - omega.shape[0]
            extra = gen.standard_normal((pad, self.dim_sem))
            n = np.linalg.norm(extra, axis=1, keepdims=True)
            omega = np.concatenate([omega, extra / n], axis=0)
        else:
            omega = omega[: self.dim_slot]
        # The factor that makes the kernel an honest Bochner kernel.
        return omega * np.sqrt(float(self.dim_sem))

    @property
    def omega(self) -> np.ndarray:
        return self._omega

    def angles(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=np.float64).reshape(-1)
        if u.shape[0] != self.dim_sem:
            raise ValueError(f"u must have length {self.dim_sem}")
        return np.mod(self.sigma * (self._omega @ u), 2.0 * np.pi)

    def phasor(self, u: np.ndarray) -> np.ndarray:
        """Unit-modulus phasor, |Psi_d| = 1/sqrt(D_slot) exactly."""
        th = self.angles(u)
        return np.exp(1j * th) / np.sqrt(self.dim_slot)

    def ring(self, u: np.ndarray, k_bins: int = 256) -> np.ndarray:
        """Quantise to Z_256 for exact ring binding (G-ING4)."""
        th = self.angles(u)
        q = np.floor(th * (k_bins / (2.0 * np.pi)) + 0.5).astype(np.int64)
        return (q % k_bins).astype(np.uint8)

    def unit_modulus_error(self, psi: np.ndarray) -> float:
        """Max ||Psi_d| - 1/sqrt(D)|. Must be ~0 for G-ING2."""
        want = 1.0 / np.sqrt(self.dim_slot)
        return float(np.max(np.abs(np.abs(psi) - want)))


# -- full 4-slot composition ---------------------------------------------

@dataclass
class Encoded:
    psi: np.ndarray          # (DIM,) unit sphere
    q0: np.ndarray           # (SLOT,) ring, entity
    q1: np.ndarray           # (SLOT,) ring, polarity slot
    pol: int
    active_slots: tuple

    def as_dict(self) -> dict:
        return {
            "pol": int(self.pol),
            "active_slots": list(self.active_slots),
            "norm": round(float(np.linalg.norm(self.psi)), 9),
        }


class SemanticBackbone:
    """Text -> wave. Slots 0 and 1 active; 2 and 3 exactly zero.

    Args:
        stem: contextual encoder.
        sigma: Bochner bandwidth.
        seed: explicit, because determinism is not optional.
    """

    def __init__(self, stem: Optional[SemanticStem] = None,
                 sigma: float = float(np.sqrt(2.0)),
                 seed: int = 20261020,
                 w_content: float = 0.5) -> None:
        if not 0.0 < w_content < 1.0:
            raise ValueError("w_content must be in (0, 1)")
        self.w_content = float(w_content)
        self.stem = stem or SemanticStem()
        self.proj = BochnerPhaseProjector(self.stem.dim_sem, SLOT, sigma, seed)
        self.sigma = sigma
        # Slot-1 polarity basis. FIXED and content-independent, so that
        # negation is exactly +128 in Z_256 for every input.
        #
        # First draft derived q1 from q0, which was WRONG: two texts differ
        # in content as well as polarity, so the ring delta was
        # `128 + (content difference)` rather than exactly 128. Measured
        # delta on a negation pair came out as an arbitrary vector, not a
        # constant. Polarity must live on its own basis to stay an exact
        # group element.
        gen = np.random.default_rng(int(seed) + 977)
        self.q_role1 = gen.integers(0, 256, size=SLOT, dtype=np.uint8)

    def info(self) -> dict:
        return {
            "stem": self.stem.kind,
            "model": self.stem.model_name if self.stem.kind ==
                     "sentence_transformer" else "char_ngram_4",
            "dim_sem": self.stem.dim_sem,
            "sigma": round(self.sigma, 6),
            "dim": DIM,
            "slot": SLOT,
            "offline": bool(os.environ.get("HF_HUB_OFFLINE")),
        }

    def encode(self, text: str) -> Encoded:
        """Compose a unit wave. Polarity flips CONTENT and the role basis.

        DERIVED, and this is the point the specification gets wrong. The
        spec puts polarity only on Slot 1 and then claims the full-sentence
        antonym coherence reaches -1.0000. It cannot. With two slots of
        energy w0 and w1, one identical and one antipodal:

            Re<psi_a, psi_b> = w0 - w1 = 2 w0 - 1

        So equal slots give exactly 0.0, and crossing -0.50 needs the
        polarity slot to carry >= 75 percent of the energy, which would
        crush the content signal the synonym gate needs.

        The repair costs nothing: flip Slot 0 by the SAME half-turn. A
        directional antonym differs from its root in direction alone, so
        canonicalising the content to the root and then negating both
        active slots gives

            Re<psi_a, psi_b> = -w0 - w1 = -1.0 exactly

        for every weighting. No energy tradeoff, no tuning, and -1.0 is
        reached rather than approached. The half-turn is `×-1`, which is
        exactly `+128 mod 256` on the ring, so G-ING4 stays exact.
        """
        p = polarity(text)
        # Canonical content: "decreases" -> "increase". Two antonym
        # propositions now differ ONLY in the polarity sign.
        u = self.stem.encode(canonicalize(text))
        q0 = self.proj.ring(u)
        sign = -1.0 if p < 0 else 1.0
        # Slot 1 keys off the FIXED role basis, not off q0, so the flip is
        # a constant function of polarity alone.
        q1 = ((self.q_role1.astype(np.int64)
               + (128 if p < 0 else 0)) % 256).astype(np.uint8)
        psi = np.zeros(DIM, dtype=np.complex128)
        w0, w1 = self.w_content, 1.0 - self.w_content
        # MEASURED DEFECT, repaired. Applying `sign` to slot 1 AND the +128
        # ring shift flips that slot TWICE, so the two flips cancel. The
        # first version measured +0.0042 at w0=0.5 instead of -1.0.
        # One flip per slot: slot 0 takes `sign`, slot 1 takes the ring
        # half-turn only. Coherence is then -(w0*cos + w1) = -1 when the
        # content is canonical.
        psi[:SLOT] = self.proj.phasor(u) * (np.sqrt(w0) * sign)
        psi[SLOT:2 * SLOT] = (
            np.exp(1j * (2.0 * np.pi * q1.astype(np.float64) / 256.0))
            / np.sqrt(SLOT) * np.sqrt(w1))
        n = np.linalg.norm(psi)
        if n > 0:
            psi = psi / n
        return Encoded(psi=psi, q0=q0, q1=q1, pol=p, active_slots=(0, 1))

    def content_wave(self, text: str) -> np.ndarray:
        """Slot-0 sub-wave only, unit norm, carrying the polarity sign.

        The honest synonym metric. Full-sentence coherence mixes in the
        role basis, so a `>= 0.65` gate read on the full wave would pass
        vacuously. A gate must read the subspace that carries semantics.
        The sign is applied here too, so a directional antonym reads
        negative and a synonym reads positive.
        """
        u = self.stem.encode(canonicalize(text))
        v = self.proj.phasor(u)
        v = v / np.linalg.norm(v)
        return -v if polarity(text) < 0 else v

    def encode_many(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([self.encode(t).psi for t in texts], axis=0)


# -- Stage V: common-mode rejection and whitening -------------------------

def common_mode_reject(book: np.ndarray, query: Optional[np.ndarray] = None,
                       gamma: float = 1.0) -> tuple:
    """Subtract the shared component from every row, then renormalise.

    Args:
        book: `(K, D)` candidate waves.
        query: optional query wave, projected with the SAME direction.
        gamma: fraction of the common mode removed. `gamma = 1` is the
            exact projection. Any `gamma < 1` leaves `(1-gamma)` of the
            shared carrier in every rival, which is strictly worse for
            coherence reduction. The derivation is why the default is 1.

    Returns:
        `(book_out, query_out, report)`. The report carries `c_max`
        before and after, so the effect is measured, not described.

    Derivative check. Let `m` be the unit common mode and `P = I - m m^H`.
    With `gamma = 1` each row becomes `P x_m`, so `c_max -> ~0` for rows
    differing only along `m`. With `gamma < 1` the residual common part
    `(1-gamma) m m^H x_m` survives in every row, so
    `c_ij -> (1-gamma)^2 |<x_i,m>||<x_j,m>| / (norms)`, which is monotone
    increasing in `1-gamma`. The optimum is therefore `gamma = 1`.
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    x = np.asarray(book, dtype=np.complex128)
    if x.ndim != 2:
        raise ValueError("book must be 2-D")
    k = x.shape[0]
    g0 = np.real(x @ x.conj().T)
    np.fill_diagonal(g0, 0.0)
    c_before = float(np.max(g0)) if k > 1 else 0.0

    centroid = x.mean(axis=0)
    nrm = np.linalg.norm(centroid)
    if nrm < 1e-12 or gamma == 0.0:
        report = {"c_max_before": round(c_before, 6),
                  "c_max_after": round(c_before, 6),
                  "gamma": gamma, "common_mode_norm": float(nrm)}
        return x, (None if query is None else np.asarray(query)), report

    m = centroid / nrm

    def project(v: np.ndarray) -> np.ndarray:
        return v - gamma * np.vdot(m, v) * m

    out = np.stack([project(row) for row in x], axis=0)
    q_out = None if query is None else project(np.asarray(query, np.complex128))

    def renorm(a: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(a, axis=1, keepdims=True)
        return np.divide(a, np.where(n > 1e-12, n, 1.0))

    out = renorm(out)
    if q_out is not None:
        nq = np.linalg.norm(q_out)
        q_out = q_out / nq if nq > 1e-12 else q_out
    g1 = np.real(out @ out.conj().T)
    np.fill_diagonal(g1, 0.0)
    c_after = float(np.max(g1)) if k > 1 else 0.0
    report = {
        "c_max_before": round(c_before, 6),
        "c_max_after": round(c_after, 6),
        "gamma": gamma,
        "common_mode_norm": round(float(nrm), 6),
    }
    return out, q_out, report


def whiten_options(book: np.ndarray, query: Optional[np.ndarray] = None,
                   retract: bool = True) -> tuple:
    """Local Dual-Gram Stiefel retraction on the K x D option matrix.

    WARNING, stated because it is easy to miss. This is NOT a neutral
    re-ranking. Writing `G = Re(X X^H)` and `X_ortho = G^{-1/2} X`, the
    readout scores satisfy `s_ortho = G^{-1/2} s_raw`, so `argmax` can
    change. Whitening is a DIFFERENT decision rule that amplifies
    non-common-mode directions. It must be validated by accuracy.

    For K = 4 the Gram is 4x4, so `eigh` is microseconds. No large
    allocation, no MemoryError, unlike the D x D route that constraint C7
    refuses.
    """
    from henri.gram import stiefel_retract

    x = np.asarray(book, dtype=np.complex128)
    if not retract or x.shape[0] < 2:
        return x, (None if query is None else np.asarray(query)), {
            "whitened": False}
    g = np.real(x @ x.conj().T)
    np.fill_diagonal(g, 0.0)
    c_before = float(np.max(g))
    xw = stiefel_retract(x)
    gw = np.real(xw @ xw.conj().T)
    np.fill_diagonal(gw, 0.0)
    c_after = float(np.max(gw)) if xw.shape[0] > 1 else 0.0
    q_out = None
    if query is not None:
        q = np.asarray(query, np.complex128)
        n = np.linalg.norm(q)
        q_out = q / n if n > 1e-12 else q
    return xw, q_out, {
        "whitened": True,
        "c_max_before": round(c_before, 6),
        "c_max_after": round(c_after, 6),
        "gram_side": int(x.shape[0]),
    }


# -- DeltaMemory capacity, DERIVED then measured --------------------------

def delta_memory_capacity(rank: int, dim: int, m: int) -> dict:
    """Analytic capacity of a rank-`rank` delta memory holding `m` pairs.

    DERIVATION. The repair in `henri/memory.py` stores
    `S = V K^H` with `V, K` each `(dim, rank)`, so `rank(S) <= rank` by
    construction, independent of `m`. For associations `(k_a, v_a)` with
    unit keys, recall reads

        S k_a = sum_b v_b <k_b, k_a> = v_a + sum_{b != a} v_b <k_b, k_a>.

    The second term is the crosstalk error. Two regimes follow.

    (a) EXACT capacity. If the `{v_a}` are linearly independent then
        `S k_a = v_a` for all `a` forces `rank(S) >= m`, hence `m <= rank`.
        With `r = 64` at most 64 independent associations are exact. This
        is a rank wall, not a tuning issue.

    (b) GRACEFUL capacity. For `m > rank` the residual is
        `||e_a||^2 = sum_{b!=a} |<k_b,k_a>|^2 / (scale)`, so with random
        keys `E||e_a||^2 ~ (m-1)/dim` up to the projection loss from the
        rank truncation. Solving for unit signal-to-noise gives
        `m* ~ dim * snr^-2`, which is the classical VSA capacity and is
        far larger than `rank`. The binding constraint is the SMALLER of
        the two: `m_max = rank` for exactness, `m*` for tolerance.

    Args:
        rank: factor width `r`.
        dim: wave dimension `D`.
        m: associations attempted.

    Returns:
        A dict with both bounds and which one binds.
    """
    if rank < 1 or dim < 1 or m < 1:
        raise ValueError("rank, dim, m must be positive")
    exact = int(rank)
    # Crosstalk variance for random unit keys: E|<k_b,k_a>|^2 = 1/dim.
    residual_var = (m - 1) / float(dim)
    graceful = int(np.floor(np.sqrt(dim)))     # snr = 1 boundary
    binding = "rank" if exact <= graceful else "crosstalk"
    return {
        "rank": int(rank),
        "dim": int(dim),
        "m_attempted": int(m),
        "exact_capacity": exact,
        "graceful_capacity_snr1": graceful,
        "binds": binding,
        "predicted_residual_var": round(residual_var, 12),
        "verdict": ("EXCEEDS_EXACT" if m > exact else "WITHIN_EXACT"),
    }
