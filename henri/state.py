"""Wave state on the complex unit hypersphere.

Evidence label: OBSERVED (norm, slot energy) and DERIVED (the
zero-padding rule, which follows from constraint C6).

Invariant: ||psi||_2 = 1. Every constructor returns a unit-norm state.
The one deliberate exception is `WaveState.raw`, which marks an
intermediate value so callers cannot mistake it for a settled state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from henri.config import N_SLOTS, SLOT_NAMES, WaveConfig

_EPS = 1e-30


@dataclass
class WaveState:
    """A complex wave on S^{D-1}, partitioned into four structural slots.

    Attributes:
        psi: complex128 array of shape (dim,).
        cfg: the configuration that fixed the dimension.
        settled: True when psi has been normalized and is safe to read.
    """

    psi: np.ndarray
    cfg: WaveConfig
    settled: bool = field(default=True)

    def __post_init__(self) -> None:
        self.psi = np.asarray(self.psi, dtype=np.complex128).reshape(-1)
        if self.psi.shape[0] != self.cfg.dim:
            raise ValueError(
                f"psi has length {self.psi.shape[0]}, config dim is {self.cfg.dim}"
            )

    # -- constructors --------------------------------------------------

    @classmethod
    def zeros(cls, cfg: WaveConfig) -> "WaveState":
        """All-zero state. NOT valid on the sphere; used for assembly."""
        return cls(np.zeros(cfg.dim, dtype=np.complex128), cfg, settled=False)

    @classmethod
    def random(
        cls,
        cfg: WaveConfig,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
    ) -> "WaveState":
        """Uniform random unit-norm complex state.

        The seed is explicit so results reproduce across processes.
        """
        gen = _generator(rng, seed)
        real = gen.standard_normal(cfg.dim)
        imag = gen.standard_normal(cfg.dim)
        return cls.from_complex(real + 1j * imag, cfg)

    @classmethod
    def real(
        cls,
        cfg: WaveConfig,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
    ) -> "WaveState":
        """Real-valued unit-norm state.

        This is the wave family on which the retired repo measured a
        chance coherence 12.3x higher than the complex family
        (constraint C2). Both families are exercised in tests.
        """
        gen = _generator(rng, seed)
        return cls.from_complex(gen.standard_normal(cfg.dim).astype(np.complex128), cfg)

    @classmethod
    def from_complex(cls, values: np.ndarray, cfg: WaveConfig) -> "WaveState":
        """Normalize arbitrary complex values onto the sphere."""
        arr = np.asarray(values, dtype=np.complex128).reshape(-1)
        n = np.linalg.norm(arr)
        if not np.isfinite(n) or n < _EPS:
            raise ValueError("cannot normalize a zero or non-finite vector")
        return cls(arr / n, cfg, settled=True)

    @classmethod
    def from_real_blocks(cls, values: np.ndarray, cfg: WaveConfig) -> "WaveState":
        """Lift real scalars onto the sphere as a complex state.

        Accepts a real array of length `2*dim` (real/imag interleaved by
        halves) so the Cl(3,0) real-scalar layout stays usable.
        """
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.shape[0] != 2 * cfg.dim:
            raise ValueError(f"expected {2 * cfg.dim} real scalars, got {arr.shape[0]}")
        return cls.from_complex(arr[: cfg.dim] + 1j * arr[cfg.dim :], cfg)

    @classmethod
    def raw(cls, values: np.ndarray, cfg: WaveConfig) -> "WaveState":
        """Wrap an intermediate value without normalizing or validating.

        `settled=False` marks the state as not yet on the sphere.
        """
        return cls(np.asarray(values, dtype=np.complex128).reshape(-1), cfg, settled=False)

    # -- measurement ---------------------------------------------------

    @property
    def norm(self) -> float:
        """Euclidean norm of the state."""
        return float(np.linalg.norm(self.psi))

    def unit_norm_error(self) -> float:
        """Absolute deviation from unit norm. Gate metric."""
        return abs(self.norm - 1.0)

    def normalize(self) -> "WaveState":
        """Project back onto the sphere in place. Returns self."""
        n = self.norm
        if not np.isfinite(n) or n < _EPS:
            raise ValueError("cannot normalize a zero or non-finite state")
        self.psi = self.psi / n
        self.settled = True
        return self

    def slot_energy(self) -> np.ndarray:
        """Energy fraction per structural slot. Sums to ~1 for a unit state."""
        total = float(np.sum(np.abs(self.psi) ** 2)) + _EPS
        return np.array(
            [float(np.sum(np.abs(self.psi[self.cfg.slot_slice(i)]) ** 2)) / total
             for i in range(N_SLOTS)]
        )

    def slot_bleed(self, active: Sequence[int]) -> float:
        """Cross-slot energy outside the `active` slots (G-U1 metric)."""
        energy = self.slot_energy()
        mask = np.zeros(N_SLOTS, dtype=bool)
        mask[list(active)] = True
        return float(np.sum(energy[~mask]))

    def slot_bleed_max(self) -> float:
        """Worst single-slot bleed against the dominant slot."""
        energy = self.slot_energy()
        order = np.argsort(energy)[::-1]
        return float(np.sum(energy[order[1:]]))

    def inner(self, other: "WaveState") -> complex:
        """Hermitian inner product <self, other>."""
        _check_same(self, other)
        return complex(np.vdot(self.psi, other.psi))

    def coherence(self, other: "WaveState") -> float:
        """Normalized real coherence Re<self, other> / (||self|| ||other||)."""
        _check_same(self, other)
        denom = self.norm * other.norm + _EPS
        return float(np.real(np.vdot(self.psi, other.psi)) / denom)

    # -- assembly ------------------------------------------------------

    def put_slot(self, index: int, values: np.ndarray) -> "WaveState":
        """Write one slot in place. Length must equal the slot span."""
        sl = self.cfg.slot_slice(index)
        vals = np.asarray(values, dtype=np.complex128).reshape(-1)
        if vals.shape[0] != (sl.stop - sl.start):
            raise ValueError(
                f"slot {index} expects {sl.stop - sl.start} values, got {vals.shape[0]}"
            )
        self.psi[sl] = vals
        return self

    def zero_slot(self, index: int) -> "WaveState":
        """Explicitly zero a slot.

        Constraint C6: unused slots are zeroed, never noise-filled.
        The retired repo documented a 99.85% noise-fill defect.
        """
        self.psi[self.cfg.slot_slice(index)] = 0.0
        return self

    def copy(self) -> "WaveState":
        return WaveState(self.psi.copy(), self.cfg, self.settled)

    # -- interop -------------------------------------------------------

    def to_real_blocks(self) -> np.ndarray:
        """Return real scalars of length 2*dim (real half, then imag half)."""
        return np.concatenate([self.psi.real, self.psi.imag])

    def phases(self) -> np.ndarray:
        """Phase angles in radians, shape (dim,)."""
        return np.angle(self.psi)

    def __len__(self) -> int:
        return self.cfg.dim


def _generator(
    rng: Optional[np.random.Generator], seed: Optional[int]
) -> np.random.Generator:
    """Return an explicit generator. Determinism is not optional."""
    if rng is not None:
        return rng
    return np.random.default_rng(seed)


def _check_same(a: WaveState, b: WaveState) -> None:
    if a.cfg.dim != b.cfg.dim:
        raise ValueError(f"dimension mismatch: {a.cfg.dim} vs {b.cfg.dim}")


def slot_table(cfg: WaveConfig) -> list:
    """Human-readable slot table for the docs and receipts."""
    return [
        {
            "slot": i,
            "role": SLOT_NAMES[i],
            "start": cfg.slot_slice(i).start,
            "stop": cfg.slot_slice(i).stop,
        }
        for i in range(N_SLOTS)
    ]
