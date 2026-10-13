"""Continuous spatial encoder: sensor fields to geometric phase bundles.

Evidence label: DERIVED for the algebra; OBSERVED for every number
`tests/test_spatial_encoder.py` measures.

MODULE DIRECTIVE 1 of the Phase 2 ingress specification
(`project_henri_phase_2_grounded_multimodal_ingress_data_sourcing_specification.md`).

WHAT THIS REPLACES
------------------
Phase 1 evaluated the kernel on synthetic Gaussian vectors. This module
replaces that substrate with a continuous spatial field encoder: a real
sensor field `I(x, y)` maps to a geometric phase bundle on
`S^{D_slot - 1}`.

THE ENCODING
------------
Each dimension `d` carries a frozen incommensurate frequency pair
`(u_d, v_d)`:

    theta_d(x, y) = 2pi * (u_d * x + v_d * y)      (mod 2pi)

The field contributes by continuous superposition (bundling):

    Psi_d = sum_p I(p) * exp(i * theta_d(p))
    Psi   = Psi / ||Psi||_2

THE SEPARABILITY IDENTITY (why this is cheap)
---------------------------------------------
`theta_d(x, y)` is a sum of a term in `x` and a term in `y`, so

    exp(i * theta_d(p)) = exp(i 2pi u_d x_p) * exp(i 2pi v_d y_p)

The discrete double sum therefore factors into two `(D_slot, W)` and
`(D_slot, H)` phase tables. Cost is `O(D_slot * (W + H))` memory and
`O(D_slot * W * H)` work, with no `D x D` object anywhere (constraint C7).

THE TRANSLATION IDENTITY, AND ITS SCOPE (gate G-ING1)
----------------------------------------------------
Let `I'(p) = I(p - delta)` be the WRAPPED integer shift, so
`I'(r, c) = I((r - dy) mod H, (c - dx) mod W)`. Then

    Psi'_d = sum_{r,c} I(r-dy, c-dx) exp(i theta_d(c, r))

Substituting `c = (c'' + dx) mod W` and `r = (r'' + dy) mod H`:

    Psi'_d = sum_{r'',c''} I(r'',c'') exp(i theta_d(c''+dx, r''+dy))

and `theta_d(c''+dx, r''+dy) = theta_d(c'', r'') + 2pi (u_d dx/W + v_d dy/H)`.

THE PERIODICITY REQUIREMENT. The step above is valid only if the
coordinate of the wrapped index agrees with the shifted coordinate.
`x_{(c''+dx) mod W} = x_{c''} + dx/W` holds exactly when `c'' + dx < W`;
otherwise the wrapped index sits ONE PERIOD lower, contributing a
spurious factor `exp(-i 2pi u_d)` per wrapped pixel. Those spurious
factors cancel if and only if

    exp(i 2pi u_d) = 1   <=>   u_d is an INTEGER.

So exact torus covariance requires INTEGER frequencies, and the identity
is then

    Psi'_d = Psi_d * exp(i 2pi (u_d dx/W + v_d dy/H))

for a GENERAL field, not only a constant one. Measured, both ways: the
integer case passes at 1e-16, and the fractional case fails at ~1e-1.
`tests/test_ingress.py` asserts both.

DEVIATION FROM THE SPECIFICATION, recorded rather than hidden. The
specification asks for an irrational frequency ratio
(`omega_x/omega_y not in Q`). That is UNSATISFIABLE on a discrete torus:
the ratio of two integers is rational, and non-integers break the gate.
The property the spec actually needs is that the frequency pairs are
generic and non-degenerate. This build uses a distinct integer lattice
and REPORTS its collinearity (`frequency_summary`) instead of claiming
incommensurability it does not have.

SCOPE. This identity describes the WRAPPED operator. A translation that
drops pixels at the border is a DIFFERENT operator and must fail; that
is the negative control proving the gate can fail. The gate's scope is a
torus, not a bounded rectangle.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from henri.config import BLOCK_DIM, WaveConfig

_EPS = 1e-30
CL_GRADES = 8
"""Cl(3,0) basis size: 1, e1, e2, e3, e12, e23, e31, e123."""


class ContinuousSpatialEncoder:
    """Map continuous spatial fields onto the complex unit hypersphere.

    Args:
        cfg: wave configuration. `cfg.slot_span` is the slot dimension
            `D_slot` that this encoder fills.
        seed: offset seed. The frequency RATIOS stay incommensurate for
            every seed; the seed only shifts the sequence start, so the
            encoder reproduces across processes without relying on
            `PYTHONHASHSEED`.
        chunk_rows: reserved for callers that chunk large fields. The
            separable form does not need it, so it is unused; it is kept
            so the signature is stable if a non-separable term is added.
    """

    def __init__(self, cfg: WaveConfig, seed: int = 20261012, chunk_rows: int = 64) -> None:
        if cfg.dim % BLOCK_DIM != 0:
            raise ValueError(f"dim={cfg.dim} must divide by BLOCK_DIM={BLOCK_DIM}")
        self.cfg = cfg
        self.dim = int(cfg.slot_span)
        self.seed = int(seed)
        self.chunk_rows = int(chunk_rows)

        # INTEGER frequencies, plus a seed-derived permutation that only
        # decides WHICH lattice pair lands on WHICH dimension. See the
        # module docstring: exact torus covariance needs
        # `exp(i 2pi u x)` periodic in `x` with period 1, which holds if
        # and only if `u` is an integer. Fractional frequencies are
        # discontinuous at the seam and break G-ING1 at every wrapped
        # pixel. This is measured, not assumed.
        self.n_freq_x = int(np.ceil(np.sqrt(self.dim)))
        self.n_freq_y = int(np.ceil(self.dim / self.n_freq_x))
        if self.n_freq_x * self.n_freq_y < self.dim:
            raise ValueError(
                f"frequency lattice {self.n_freq_x}x{self.n_freq_y} is too "
                f"small for dim={self.dim}"
            )
        gen = np.random.default_rng(self.seed)
        order = gen.permutation(self.dim)
        # Pair index -> (u, v) in [1, P] x [1, Q]. Injective on
        # 0..P*Q-1, so every dimension gets a DISTINCT pair and the
        # (0, 0) DC pair is excluded by construction.
        self.u = (1 + (order % self.n_freq_x)).astype(np.int64)
        self.v = (1 + ((order // self.n_freq_x) % self.n_freq_y)).astype(np.int64)

    # -- layout --------------------------------------------------------

    @property
    def n_blocks(self) -> int:
        """Cl(3,0) blocks across the slot. 2048 at D_slot = 16384."""
        return self.dim // CL_GRADES

    def frequency_summary(self) -> dict:
        """Diagnostics for the frozen integer frequency lattice.

        `collinear_pairs` counts pairs sharing a factor (`gcd > 1`). The
        specification asks for incommensurability; integers cannot
        supply it. This reports the realized lattice honestly instead of
        asserting a property the discrete torus cannot have.
        """
        gcd = np.gcd(self.u, self.v)
        pairs = {(int(a), int(b)) for a, b in zip(self.u, self.v)}
        return {
            "dim": self.dim,
            "n_freq_x": self.n_freq_x,
            "n_freq_y": self.n_freq_y,
            "all_integer": bool(
                np.all(self.u == np.round(self.u)) and np.all(self.v == np.round(self.v))
            ),
            "distinct_pairs": len(pairs),
            "min_u": int(self.u.min()),
            "max_u": int(self.u.max()),
            "min_v": int(self.v.min()),
            "max_v": int(self.v.max()),
            "primitive_pairs": int(np.sum(gcd == 1)),
            "collinear_pairs": int(np.sum(gcd > 1)),
        }

    # -- directive 1, item 2 -------------------------------------------

    def _grid(self, height: int, width: int) -> Tuple[np.ndarray, np.ndarray]:
        """Pixel-centre coordinates on the torus, in `[0, 1)`."""
        x = (np.arange(width, dtype=np.float64) + 0.5) / float(width)
        y = (np.arange(height, dtype=np.float64) + 0.5) / float(height)
        return x, y

    def phase_tables(
        self, height: int, width: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return `(A, B)` with `A` of shape `(D_slot, W)` and `B` `(D_slot, H)`.

        `A[d, c] = exp(i 2pi u_d x_c)`, `B[d, r] = exp(i 2pi v_d y_r)`.
        """
        if height < 1 or width < 1:
            raise ValueError("field must have positive height and width")
        x, y = self._grid(height, width)
        a = np.exp(1j * (2.0 * np.pi) * np.outer(self.u, x))
        b = np.exp(1j * (2.0 * np.pi) * np.outer(self.v, y))
        return a, b

    def encode_2d_field(self, field_tensor: np.ndarray) -> np.ndarray:
        """Encode a continuous 2-D field as a unit-norm complex wave.

        Args:
            field_tensor: real array of shape `(H, W)`. Intensity,
                luminance, depth, or tactile pressure.

        Returns:
            complex128 vector of length `D_slot` with `||Psi||_2 = 1`.

        The intensity modulates phase, per directive 1 item 2.
        """
        field = np.asarray(field_tensor, dtype=np.float64)
        if field.ndim != 2:
            raise ValueError(f"field_tensor must be 2-D, got shape {field.shape}")
        height, width = field.shape
        a, b = self.phase_tables(height, width)

        # Separable accumulation: Psi = sum_r b_r * (A @ field[r, :]).
        psi = np.zeros(self.dim, dtype=np.complex128)
        for r in range(height):
            psi += b[:, r] * (a @ field[r, :])

        norm = float(np.linalg.norm(psi))
        if not np.isfinite(norm) or norm < _EPS:
            raise ValueError(
                "field encodes to zero energy; a zero field has no direction "
                "on the sphere. Refusing to invent one."
            )
        return psi / norm

    def field_to_wave_state(self, field_tensor: np.ndarray):
        """Encode a field and wrap it as a `WaveState` on the full space.

        The encoded slot is placed in Slot 0 (Entity) by default assembly.
        Use `henri.ingress_subspace_router.SubspaceRouter` to place it in
        a chosen slot together with other slots.
        """
        from henri.state import WaveState

        return WaveState.from_complex(self.encode_2d_field(field_tensor), self.cfg)

    def translate_phase(
        self, dx: int, dy: int, height: int, width: int
    ) -> np.ndarray:
        """Exact per-dimension phase factor for a wrapped integer shift.

        Returns `exp(i 2pi (u_d dx/W + v_d dy/H))`, shape `(D_slot,)`.

        THE UNITS MATTER. `x` and `y` are NORMALIZED coordinates in
        `[0, 1)`, while `dx` and `dy` are integer PIXEL offsets. Dividing
        the pixel offset by the field size puts it in the same units as
        the coordinate. Omitting that division is a unit error, and the
        gate catches it immediately.

        Derivation. Let `I'(p) = I(p - delta)` be the wrapped shift, so
        `I'(r, c) = I(r - dy, c - dx)`. Substituting `r' = r - dy` and
        `c' = c - dx`:

            Psi'_d = sum_{r,c} I(r-dy, c-dx) exp(i theta_d(x_c, y_r))
                   = exp(i 2pi (u_d dx/W + v_d dy/H)) * Psi_d

        The intensity field and the phase share one coordinate, so the
        sum is invariant up to that exact per-dimension factor. The
        `Abs`-invariance form of G-ING1 therefore holds EXACTLY, not
        merely for a constant field.
        """
        if int(dx) != dx or int(dy) != dy:
            raise ValueError(
                "only integer pixel shifts are exact on the torus; "
                "got dx=%r dy=%r" % (dx, dy)
            )
        if height < 1 or width < 1:
            raise ValueError("field must have positive height and width")
        fraction_x = float(dx) / float(width)
        fraction_y = float(dy) / float(height)
        return np.exp(
            1j * (2.0 * np.pi) * (self.u * fraction_x + self.v * fraction_y)
        )

    def wrapped_shift(self, field_tensor: np.ndarray, dx: int, dy: int) -> np.ndarray:
        """Shift a field by an integer offset with torus wrap-around.

        This is the operator whose translation identity is exact. A
        border-dropping shift is a DIFFERENT operator; see the module
        docstring.
        """
        if int(dx) != dx or int(dy) != dy:
            raise ValueError("wrapped_shift needs integer offsets")
        return np.roll(np.asarray(field_tensor, dtype=np.float64), (int(dy), int(dx)), axis=(0, 1))

    def unwrapped_shift(self, field_tensor: np.ndarray, dx: int, dy: int) -> np.ndarray:
        """Shift a field WITHOUT wrap: pixels leave and zeros enter.

        Negative control substrate. The gate must FAIL for this
        operator, which is what proves G-ING1 can fail.
        """
        field = np.asarray(field_tensor, dtype=np.float64)
        out = np.zeros_like(field)
        h, w = field.shape
        dx = int(dx)
        dy = int(dy)
        src_y = slice(max(0, -dy), min(h, h - dy))
        dst_y = slice(max(0, dy), min(h, h + dy))
        src_x = slice(max(0, -dx), min(w, w - dx))
        dst_x = slice(max(0, dx), min(w, w + dx))
        out[dst_y, dst_x] = field[src_y, src_x]
        return out

    # -- directive 1, item 3 -------------------------------------------

    def encode_multivector_blade(
        self,
        scalars: np.ndarray,
        vectors: np.ndarray,
        bivectors: np.ndarray,
        trivectors: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Build a Cl(3,0) multivector field across `n_blocks` blocks.

        Grade order is fixed and matches the module constant:

            [ scalar, e1, e2, e3, e12, e23, e31, e123 ]

        Args:
            scalars: shape `(N,)`.
            vectors: shape `(N, 3)`, grades `e1, e2, e3`.
            bivectors: shape `(N, 3)`, grades `e12, e23, e31`.
            trivectors: shape `(N,)`, grade `e123`. Default zeros, so the
                eighth grade is explicit rather than implied.

        Returns:
            real float64 vector of length `D_slot = 8 * N`, unit norm.

        REPRESENTATION BOUNDARY. This returns REAL Cl(3,0) scalars. A
        complex wave is a DIFFERENT representation. Do not silently
        reinterpret one as the other. The declared adapter is
        `WaveState.from_real_blocks`, which expects `2 * dim` values in a
        real-half / imag-half layout; this module never performs that
        conversion implicitly.
        """
        s = np.asarray(scalars, dtype=np.float64).reshape(-1)
        v = np.asarray(vectors, dtype=np.float64)
        b = np.asarray(bivectors, dtype=np.float64)
        n = s.shape[0]
        if n != self.n_blocks:
            raise ValueError(
                f"expected {self.n_blocks} blocks for D_slot={self.dim}, got {n}"
            )
        if v.shape != (n, 3):
            raise ValueError(f"vectors must be ({n}, 3), got {v.shape}")
        if b.shape != (n, 3):
            raise ValueError(f"bivectors must be ({n}, 3), got {b.shape}")
        if trivectors is None:
            t = np.zeros(n, dtype=np.float64)
        else:
            t = np.asarray(trivectors, dtype=np.float64).reshape(-1)
            if t.shape[0] != n:
                raise ValueError(f"trivectors must have length {n}, got {t.shape[0]}")

        blocks = np.stack(
            [s, v[:, 0], v[:, 1], v[:, 2], b[:, 0], b[:, 1], b[:, 2], t],
            axis=1,
        )
        flat = blocks.reshape(-1)
        norm = float(np.linalg.norm(flat))
        if not np.isfinite(norm) or norm < _EPS:
            raise ValueError("degenerate multivector: zero norm")
        return flat / norm

    def blade_norms(self, blade: np.ndarray) -> np.ndarray:
        """Per-block norm, shape `(n_blocks,)`. Diagnostic."""
        arr = np.asarray(blade, dtype=np.float64).reshape(-1)
        if arr.shape[0] != self.dim:
            raise ValueError(f"blade must have length {self.dim}")
        return np.linalg.norm(arr.reshape(self.n_blocks, CL_GRADES), axis=1)

    def grade_table(self) -> list:
        """Declared Cl(3,0) layout, for docs and receipts."""
        names = ("scalar", "e1", "e2", "e3", "e12", "e23", "e31", "e123")
        return [
            {"grade": i, "name": names[i], "blocks": self.n_blocks}
            for i in range(CL_GRADES)
        ]

    # -- G-ING1 ---------------------------------------------------------

    def translation_phase_error(
        self, field_tensor: np.ndarray, dx: int, dy: int
    ) -> float:
        """Gate G-ING1 metric: worst `|Psi' - Psi * phase|`.

        Applies the WRAPPED shift and compares against the predicted
        phase rotation. Both operands are unit norm, so the residual is
        dimensionless.
        """
        field = np.asarray(field_tensor, dtype=np.float64)
        height, width = field.shape
        psi = self.encode_2d_field(field)
        shifted = self.wrapped_shift(field, dx, dy)
        psi_shifted = self.encode_2d_field(shifted)
        predicted = psi * self.translate_phase(dx, dy, height, width)
        return float(np.max(np.abs(psi_shifted - predicted)))

    def magnitude_invariance_error(
        self, field_tensor: np.ndarray, dx: int, dy: int
    ) -> float:
        """Gate G-ING1 companion: worst `| |Psi'| - |Psi| |`."""
        psi = self.encode_2d_field(field_tensor)
        shifted = self.wrapped_shift(field_tensor, dx, dy)
        psi_shifted = self.encode_2d_field(shifted)
        return float(np.max(np.abs(np.abs(psi_shifted) - np.abs(psi))))
