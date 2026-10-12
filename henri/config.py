"""Single source of truth for dimensions, bounds, and thresholds.

Evidence label: DERIVED. Every constant here is a declared design
choice or a bound carried from the verification matrices. Bounds are
guards, not measured results.

Spec ambiguity resolved here (recorded, not hidden)
---------------------------------------------------
`henri_wave_mechanical_model_scaffolding_specification.md` declares both

    Psi in S^{D-1} subset C^D, D = 65,536        (complex dimension)

and

    8,192 Cl(3,0) blocks x 8 real basis = 65,536 real scalars   (real dimension)

These cannot both hold: 8,192 blocks of 8 real components is 32,768
complex components, not 65,536. This build takes `dim` as the COMPLEX
dimension and maps Cl(3,0) blocks across complex indices:
`n_blocks = dim // BLOCK_DIM`. The choice is pinned by
`tests/test_config.py::test_block_layout_is_pinned`.
"""

from __future__ import annotations

from dataclasses import dataclass

BLOCK_DIM = 8
"""Cl(3,0) multivector basis size: 1, e1, e2, e3, e12, e23, e31, e123."""

N_SLOTS = 4
SLOT_NAMES = ("entity", "action", "target", "context")
"""The 4-slot structural subspace rule, in order."""


@dataclass(frozen=True)
class WaveConfig:
    """Immutable configuration for every wave-mechanical operator.

    Args:
        dim: complex dimension of the state vector. Default 65536
            honours the theory; tests pass a small value for speed.
        r_koopman: low-rank adapter width for the Koopman operator.
        r_memory: rank cap for the delta-memory factors.
        tau_hopfield: Modern Hopfield temperature. NOT transplanted
            from the retired functor-score fit (constraint C3).
        hopfield_iters: CCCP iteration cap.
        kernel_taus: Maxwell-Debye relaxation timescales in seconds.
        kernel_weights: per-timescale kernel weights g_p.
        veto_sigma: standard deviations above baseline for the veto.
        veto_ceiling: absolute upper bound on the calibrated epsilon.
        slot_bleed_max: allowed cross-slot energy fraction (G-U1).
        crosstalk_max: allowed mean Gram crosstalk (G-U2).
        norm_tol: unit-norm preservation tolerance.
    """

    dim: int = 65536
    n_blocks: int = 0  # 0 means "derive from dim"

    r_koopman: int = 64
    r_memory: int = 64

    tau_hopfield: float = 0.05
    hopfield_iters: int = 12

    kernel_taus: tuple = (50e-6, 2e-3, 50e-3, float("inf"))
    kernel_weights: tuple = (0.4, 0.3, 0.2, 0.1)

    veto_sigma: float = 3.0
    veto_ceiling: float = 0.35

    slot_bleed_max: float = 0.0015
    crosstalk_max: float = 0.12
    norm_tol: float = 1e-5

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        span = N_SLOTS * BLOCK_DIM
        if self.dim % span != 0:
            raise ValueError(
                f"dim={self.dim} must be divisible by N_SLOTS*BLOCK_DIM={span} "
                "so slots and Cl(3,0) blocks tile the vector exactly"
            )
        if self.r_koopman < 1 or self.r_memory < 1:
            raise ValueError("low-rank widths must be >= 1")
        if self.r_memory > self.dim:
            raise ValueError("r_memory cannot exceed dim")
        if len(self.kernel_taus) != len(self.kernel_weights):
            raise ValueError("kernel_taus and kernel_weights must align")
        if self.veto_sigma <= 0:
            raise ValueError("veto_sigma must be positive")
        if not 0.0 < self.veto_ceiling <= 1.0:
            raise ValueError("veto_ceiling must be in (0, 1]")

    # -- derived views -------------------------------------------------

    @property
    def blocks(self) -> int:
        """Number of Cl(3,0) blocks. 8192 at the default dimension."""
        return self.n_blocks or (self.dim // BLOCK_DIM)

    @property
    def slot_span(self) -> int:
        """Complex components per structural slot."""
        return self.dim // N_SLOTS

    def slot_slice(self, index: int) -> slice:
        """Return the half-open index range for a structural slot."""
        if not 0 <= index < N_SLOTS:
            raise IndexError(f"slot {index} out of range 0..{N_SLOTS - 1}")
        start = index * self.slot_span
        return slice(start, start + self.slot_span)

    def slot_ranges(self) -> tuple:
        """All four slot ranges, in structural order."""
        return tuple(self.slot_slice(i) for i in range(N_SLOTS))

    # -- constructors --------------------------------------------------

    @classmethod
    def small(cls, dim: int = 512, **overrides) -> "WaveConfig":
        """Fast configuration for CI. Same structure, smaller vector."""
        base = dict(dim=dim, r_koopman=16, r_memory=16, hopfield_iters=8)
        base.update(overrides)
        return cls(**base)
