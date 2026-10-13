"""qFHRR phase-domain tokenizer: integer phase ring Z_256 over one slot.

Evidence label: DERIVED for the ring algebra; OBSERVED for every
fidelity number `tests/test_ingress.py` measures.

MODULE DIRECTIVE 2 of the Phase 2 ingress specification
(`project_henri_phase_2_grounded_multimodal_ingress_data_sourcing_specification.md`).

TWO BINDING OPERATORS EXIST IN THIS PROJECT. DO NOT CONFUSE THEM.
------------------------------------------------------------------
| operator | algebra | domain | inverse |
|---|---|---|---|
| `henri.binding.bind` | circular convolution (FFT) | continuous waves | approximate, crosstalk-limited |
| `ring_bind` (here) | element-wise modular addition in Z_256 | discrete tokens | EXACT modular subtraction |

Gate G-ING4 (`recovery >= 0.9999`) is provable only in the ring, because
modular subtraction is the exact group inverse. Routing discrete tokens
through the convolution operator reproduces the F2 saturation at
`1/sqrt(2) ~ 0.7071` and fails the gate. `tests/test_ingress.py` pins both
arms so the distinction cannot silently regress.

THE MAGNITUDE CONVENTION (one term per meaning)
-----------------------------------------------
The canonical form of a token is the INTEGER ring vector (uint8), not a
complex vector. Deep binding happens IN THE RING, where it is exact at any
depth and never accumulates scale error.

`to_complex_phasor` is the sphere VIEW of that ring vector:

    Psi_d = (1/sqrt(D)) * exp(i * 2*pi*q_d/256),   |Psi| = 1 exactly

At `D = 16384` the magnitude `1/sqrt(D)` is exactly `1/128`, so gate
G-ING2's `< 1e-15` bar is reachable in complex128.

Do not multiply sphere-view vectors to bind. That squares the component
scale and collapses the norm, exactly as recorded in `henri/qfhrr.py`.
Bind in the ring; lift once.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from henri.config import N_SLOTS, WaveConfig

RING = 256
"""Modulus of the integer phase ring, Z_256. 8-bit phase quantization."""


class QFHRRTokenizer:
    """Frozen-codebook tokenizer over the integer phase ring Z_256.

    Args:
        cfg: the wave configuration. The slot span fixes the vector
            length, so `cfg.slot_span` must equal the intended
            `D_slot`.
        seed: explicit RNG seed for every frozen codebook. Determinism
            is not optional; `PYTHONHASHSEED` must never be relied on.
        n_categories: rows in the category codebook.
        n_semantics: rows in the semantic codebook.
        n_instances: rows in the instance codebook.
        n_families: rows in the §3.2 family baseplate.
        n_members: rows in the §3.2 personal rotor table.
    """

    def __init__(
        self,
        cfg: WaveConfig,
        seed: int = 20261012,
        n_categories: int = 256,
        n_semantics: int = 512,
        n_instances: int = 4096,
        n_families: int = 64,
        n_members: int = 256,
    ) -> None:
        if cfg.dim % N_SLOTS != 0:
            raise ValueError(
                f"dim={cfg.dim} must divide by N_SLOTS={N_SLOTS} for slot ingress"
            )
        self.cfg = cfg
        self.dim = int(cfg.slot_span)
        self.seed = int(seed)

        self.n_categories = int(n_categories)
        self.n_semantics = int(n_semantics)
        self.n_instances = int(n_instances)
        self.n_families = int(n_families)
        self.n_members = int(n_members)

        gen = np.random.default_rng(self.seed)
        self.q_cat = gen.integers(0, RING, size=(self.n_categories, self.dim), dtype=np.uint8)
        self.q_sem = gen.integers(0, RING, size=(self.n_semantics, self.dim), dtype=np.uint8)
        self.q_id = gen.integers(0, RING, size=(self.n_instances, self.dim), dtype=np.uint8)

        # §3.2 "shared categorical baseplate + individual identity rotor".
        self.q_family = gen.integers(
            0, RING, size=(self.n_families, self.dim), dtype=np.uint8
        )
        self.q_personal = gen.integers(
            0, RING, size=(self.n_members, self.dim), dtype=np.uint8
        )
        # A per-family boolean mask marks the SHARED positions. Roughly
        # half of the dimensions stay identical across a family; the
        # rest carry individual identity.
        self.q_masks = gen.random((self.n_families, self.dim)) < 0.5

        # Frozen positional step vector for `positional_bind`.
        self.q_step = gen.integers(0, RING, size=(self.dim,), dtype=np.uint8)

    # -- ring algebra --------------------------------------------------

    @staticmethod
    def _as_ring(q: np.ndarray) -> np.ndarray:
        """Coerce to int64 in `[0, RING)` so modular math cannot wrap int64."""
        arr = np.asarray(q).reshape(-1).astype(np.int64)
        return np.mod(arr, RING)

    def ring_bind(self, q_a: np.ndarray, q_b: np.ndarray) -> np.ndarray:
        """Element-wise modular addition in Z_256. The binding operator."""
        a = self._as_ring(q_a)
        b = self._as_ring(q_b)
        if a.shape != b.shape:
            raise ValueError(f"ring shape mismatch: {a.shape} vs {b.shape}")
        return ((a + b) % RING).astype(np.uint8)

    def ring_unbind(self, q_bound: np.ndarray, q_key: np.ndarray) -> np.ndarray:
        """Element-wise modular subtraction in Z_256. The EXACT inverse."""
        c = self._as_ring(q_bound)
        k = self._as_ring(q_key)
        if c.shape != k.shape:
            raise ValueError(f"ring shape mismatch: {c.shape} vs {k.shape}")
        return ((c - k) % RING).astype(np.uint8)

    def ring_bundle(self, vectors: Sequence[np.ndarray]) -> np.ndarray:
        """Majority superposition on the ring, per dimension.

        Each dimension takes the circular mean of its phase values. This
        is the ring analogue of bundling and keeps the uint8 form.
        """
        if not vectors:
            raise ValueError("ring_bundle needs at least one vector")
        stack = np.stack([self._as_ring(v) for v in vectors], axis=0)
        if stack.shape[1] != self.dim:
            raise ValueError(
                f"ring vectors must have length {self.dim}, got {stack.shape[1]}"
            )
        # Circular mean: average the unit phasors, then take the angle.
        angles = 2.0 * np.pi * stack / RING
        mean_vec = np.mean(np.exp(1j * angles), axis=0)
        phase = np.angle(mean_vec)
        q = np.round(np.mod(phase, 2.0 * np.pi) * RING / (2.0 * np.pi))
        return (q.astype(np.int64) % RING).astype(np.uint8)

    # -- directive 2 operators ----------------------------------------

    def encode_token(
        self,
        category_id: int,
        semantic_id: int,
        instance_id: int,
    ) -> np.ndarray:
        """Compose a token id triple into one ring vector.

        Literal form of MODULE DIRECTIVE 2:

            q_comp = (q_cat[category] + q_sem[semantic] + q_id[instance]) mod 256

        NOTE ON ISOLATION: this composition makes distinct tokens
        mutually quasi-orthogonal. Section 3.2 of the specification
        rejects pure isolation for semantic families. Use
        `encode_family_token` when relatedness must be measurable.
        """
        q_c = self.q_cat[int(category_id) % self.n_categories]
        q_s = self.q_sem[int(semantic_id) % self.n_semantics]
        q_i = self.q_id[int(instance_id) % self.n_instances]
        return self.ring_bind(self.ring_bind(q_c, q_s), q_i)

    def fractional_bind(self, q_vector: np.ndarray, offset_float: float) -> np.ndarray:
        """Literal MODULE DIRECTIVE 2 fractional power bind.

            q_shifted = round(offset_float * q_vector) mod 256

        CAUTION: this is NOT a group homomorphism for fractional
        `offset_float`. Rounding breaks the group property, so
        `fractional_bind(fractional_bind(q, 0.5), 2) != q` in general.
        Verified by measurement in `tests/test_ingress.py`: exact for
        integer offsets, bounded deviation for fractional ones. The
        directive mandates this operator anyway, so it is implemented as
        specified and its deviation is measured rather than assumed.

        For a true phase rotation use `positional_bind`.
        """
        q = self._as_ring(q_vector)
        if q.shape[0] != self.dim:
            raise ValueError(f"q_vector must have length {self.dim}")
        shifted = np.round(float(offset_float) * q.astype(np.float64))
        return (shifted.astype(np.int64) % RING).astype(np.uint8)

    def positional_bind(
        self,
        q_vector: np.ndarray,
        position: float,
        q_step: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Section 3.2 position form:

            q_sequence = (q_token + round(position * q_step)) mod 256

        Uses the frozen `q_step` vector by default, so positions
        reproduce across processes.
        """
        step = self.q_step if q_step is None else np.asarray(q_step)
        q = self._as_ring(q_vector)
        if q.shape[0] != self.dim:
            raise ValueError(f"q_vector must have length {self.dim}")
        step = self._as_ring(step)
        if step.shape[0] != self.dim:
            raise ValueError(f"q_step must have length {self.dim}")
        add = np.round(float(position) * step.astype(np.float64))
        return ((q + add.astype(np.int64)) % RING).astype(np.uint8)

    def to_complex_phasor(self, q_vector: np.ndarray) -> np.ndarray:
        """Lift a ring vector to the unit-modulus sphere view.

            Psi_d = (1/sqrt(D)) * exp(i * 2*pi*q_d / 256)

        Returns complex128 of length `D_slot` with `|Psi| = 1`.
        At `D = 16384` the component magnitude is exactly `1/128`.
        """
        q = self._as_ring(q_vector)
        if q.shape[0] != self.dim:
            raise ValueError(
                f"q_vector must have length {self.dim}, got {q.shape[0]}"
            )
        angle = 2.0 * np.pi * q.astype(np.float64) / RING
        return np.exp(1j * angle) / np.sqrt(self.dim)

    # -- section 3.2 structured semantic path -------------------------

    def encode_family_token(
        self,
        family_id: int,
        member_id: int,
    ) -> np.ndarray:
        """Section 3.2 composite: shared baseplate plus identity rotor.

        The ring realization of `Psi_cat (x) Psi_feat + eps*Psi_id`:

            q = (q_family[family] + mask(family) ? 0 : q_personal[member]) mod 256

        Dimensions marked shared carry the family baseplate alone, so
        two members of one family agree exactly there. The remaining
        dimensions carry independent identity. Measured intra-family
        coherence therefore tracks the shared fraction (~0.5), while
        cross-family coherence stays near zero. That resolves the
        "Random-Ring Orthogonal Isolation Trap" of section 3.2.
        """
        f = int(family_id) % self.n_families
        m = int(member_id) % self.n_members
        base = self._as_ring(self.q_family[f])
        personal = self._as_ring(self.q_personal[m])
        mask = np.asarray(self.q_masks[f], dtype=bool)
        if mask.shape[0] != self.dim:
            raise ValueError("family mask has the wrong length")
        delta = np.where(mask, 0, personal)
        return ((base + delta) % RING).astype(np.uint8)

    def family_shared_fraction(self, family_id: int) -> float:
        """Fraction of dimensions held identical within a family."""
        mask = np.asarray(self.q_masks[int(family_id) % self.n_families], dtype=bool)
        return float(np.mean(mask))

    # -- measurement ---------------------------------------------------

    def token_codebook(
        self,
        triples: Sequence[Sequence[int]],
        lift: bool = True,
    ) -> np.ndarray:
        """Build a codebook from id triples.

        Args:
            triples: sequence of `(category, semantic, instance)`.
            lift: True returns the complex sphere view of shape
                `(M, D_slot)`. False returns uint8 ring rows.

        This is the negative-control substrate for G-U2: mean
        off-diagonal crosstalk must stay `<= 0.12`.
        """
        qs = [self.encode_token(c, s, i) for (c, s, i) in triples]
        if not qs:
            raise ValueError("triples must be non-empty")
        if not lift:
            return np.stack(qs, axis=0)
        return np.stack([self.to_complex_phasor(q) for q in qs], axis=0)

    def crosstalk(self, triples: Sequence[Sequence[int]]) -> Dict[str, float]:
        """Gate G-U2 metrics on a token codebook.

        Returns mean squared off-diagonal coherence and the worst
        single off-diagonal coherence. Both are scale-free.
        """
        book = self.token_codebook(triples, lift=True)
        k = book.shape[0]
        if k < 2:
            return {"mean_squared": 0.0, "max": 0.0, "k": float(k)}
        gram = book @ np.conj(book.T)
        mag2 = np.abs(gram) ** 2
        off_sum = float(np.sum(mag2) - np.sum(np.diag(mag2)))
        mean_sq = off_sum / (k * (k - 1))
        mag = np.abs(gram).copy()
        np.fill_diagonal(mag, 0.0)
        return {"mean_squared": mean_sq, "max": float(np.max(mag)), "k": float(k)}

    def unit_modulus_error(self, q_vector: np.ndarray) -> float:
        """Gate G-ING2 metric: worst `| |Psi_d| - 1/sqrt(D)|`."""
        psi = self.to_complex_phasor(q_vector)
        target = 1.0 / np.sqrt(self.dim)
        return float(np.max(np.abs(np.abs(psi) - target)))

    def norm_error(self, q_vector: np.ndarray) -> float:
        """Gate G-ING2 companion: `| ||Psi|| - 1 |`."""
        psi = self.to_complex_phasor(q_vector)
        return float(abs(np.linalg.norm(psi) - 1.0))

    def unbinding_coherence(
        self,
        q_a: np.ndarray,
        q_b: np.ndarray,
    ) -> float:
        """Gate G-ING4 metric, ring arm.

        Binds in the ring, unbinds with the exact modular inverse, and
        reports the coherence of the recovered phasor against the true
        factor. Exact recovery gives 1.0.
        """
        q_bound = self.ring_bind(q_a, q_b)
        q_recovered = self.ring_unbind(q_bound, q_a)
        rec = self.to_complex_phasor(q_recovered)
        true = self.to_complex_phasor(q_b)
        denom = float(np.linalg.norm(rec) * np.linalg.norm(true))
        return float(np.real(np.vdot(rec, true)) / (denom + 1e-30))

    def recovery_is_bitwise_exact(self, q_a: np.ndarray, q_b: np.ndarray) -> bool:
        """True when ring unbinding returns the original integers exactly."""
        q_bound = self.ring_bind(q_a, q_b)
        return bool(np.array_equal(self.ring_unbind(q_bound, q_a), self._as_ring(q_b).astype(np.uint8)))

    # -- introspection -------------------------------------------------

    def grade_layout(self) -> List[dict]:
        """Declared codebook layout, for docs and receipts."""
        return [
            {"codebook": "q_cat", "rows": self.n_categories, "dim": self.dim, "dtype": "uint8"},
            {"codebook": "q_sem", "rows": self.n_semantics, "dim": self.dim, "dtype": "uint8"},
            {"codebook": "q_id", "rows": self.n_instances, "dim": self.dim, "dtype": "uint8"},
            {"codebook": "q_family", "rows": self.n_families, "dim": self.dim, "dtype": "uint8"},
            {"codebook": "q_personal", "rows": self.n_members, "dim": self.dim, "dtype": "uint8"},
        ]
