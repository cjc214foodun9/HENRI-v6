"""4-slot subspace router: assemble a full wave from slot vectors.

Evidence label: DERIVED for the partition; OBSERVED for the bleed
measurements in `tests/test_subspace_router.py`.

MODULE DIRECTIVE 3 of the Phase 2 ingress specification
(`project_henri_phase_2_grounded_multimodal_ingress_data_sourcing_specification.md`).

THE CANONICAL PARTITION
-----------------------
At `D = 65536` with `D_slot = 16384`:

| slot | role | complex indices |
|---|---|---|
| 0 | Entity | `[0, 16384)` |
| 1 | Action | `[16384, 32768)` |
| 2 | Target | `[32768, 49152)` |
| 3 | Context | `[49152, 65536)` |

DO NOT CREATE A SECOND SLOT CONVENTION. This partition already exists
in `henri/config.py` as `N_SLOTS = 4`, `SLOT_NAMES`, and
`WaveConfig.slot_slice`. This module CONSUMES that convention. It
verifies the layout and refuses to run if the configured dimension does
not tile into four equal slots.

ZERO PADDING (constraint C6)
----------------------------
An unpopulated slot is set to NUMERICAL ZERO. It is never noise-filled.
The retired repository measured a 99.85 percent noise-fill defect, which
is why C6 is a hard rule rather than a style preference.

Why zero bleed is EXACT, not a tolerance
----------------------------------------
Normalization multiplies every component by one scalar. Zero times any
finite scalar is still exactly zero. So an unpopulated slot contributes
exactly `0.0` energy after normalization. Gates G-U1 and G-ING3 assert
equality with `0.0`, not a small bound. A tolerance here would hide a
real defect, such as a stray write into a slot that should be empty.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from henri.config import N_SLOTS, SLOT_NAMES, WaveConfig
from henri.state import WaveState

_EPS = 1e-30


class SubspaceRouter:
    """Assemble and disassemble the canonical 4-slot wave vector.

    Args:
        cfg: wave configuration. `cfg.slot_span` is `D_slot`.
    """

    def __init__(self, cfg: WaveConfig) -> None:
        if cfg.dim % N_SLOTS != 0:
            raise ValueError(
                f"dim={cfg.dim} does not tile into {N_SLOTS} equal slots"
            )
        self.cfg = cfg
        self.dim = int(cfg.dim)
        self.slot_dim = int(cfg.slot_span)
        if self.slot_dim * N_SLOTS != self.dim:
            raise ValueError(
                f"slot partition does not cover the vector: "
                f"{self.slot_dim} * {N_SLOTS} != {self.dim}"
            )

    # -- layout --------------------------------------------------------

    @property
    def slots(self) -> tuple:
        """Ordered slot names. The single source of truth is `config`."""
        return tuple(SLOT_NAMES)

    def slot_index(self, name: str) -> int:
        """Map a slot name to its index, or raise."""
        try:
            return self.slots.index(str(name))
        except ValueError:
            raise ValueError(
                f"unknown slot {name!r}; allowed: {self.slots}"
            ) from None

    def partition(self) -> list:
        """Human-readable partition table, for docs and receipts."""
        return [
            {
                "slot": i,
                "role": SLOT_NAMES[i],
                "start": self.cfg.slot_slice(i).start,
                "stop": self.cfg.slot_slice(i).stop,
                "dim": self.slot_dim,
            }
            for i in range(N_SLOTS)
        ]

    # -- directive 3, item 2 -------------------------------------------

    @staticmethod
    def _coerce_slot(values, slot_dim: int, name: str) -> np.ndarray:
        arr = np.asarray(values, dtype=np.complex128).reshape(-1)
        if arr.shape[0] != slot_dim:
            raise ValueError(
                f"slot {name!r} expects {slot_dim} complex values, got {arr.shape[0]}"
            )
        return arr

    def assemble_ingress(
        self,
        entity=None,
        action=None,
        target=None,
        context=None,
    ) -> WaveState:
        """Assemble a verified `WaveState` from any subset of slots.

        Steps, in order:
            1. Insert each provided slot vector at its canonical range.
            2. Zero every unpopulated slot explicitly (constraint C6).
            3. Apply global polar normalization `Psi / ||Psi||_2`.
            4. Return a settled `WaveState`.

        Args:
            entity: length `D_slot` slot vector, or None.
            action: length `D_slot` slot vector, or None.
            target: length `D_slot` slot vector, or None.
            context: length `D_slot` slot vector, or None.

        Returns:
            `WaveState` with `||psi||_2 = 1` and exactly zero energy in
            every unpopulated slot.

        Raises:
            ValueError: when every slot is None, because a zero vector
                has no direction on the sphere.
        """
        provided = {
            "entity": entity,
            "action": action,
            "target": target,
            "context": context,
        }
        if all(v is None for v in provided.values()):
            raise ValueError(
                "assemble_ingress needs at least one slot; refusing to "
                "normalize the zero vector"
            )

        psi = np.zeros(self.dim, dtype=np.complex128)
        active = []
        for name in self.slots:
            values = provided[name]
            idx = self.slot_index(name)
            sl = self.cfg.slot_slice(idx)
            if values is None:
                # Constraint C6, explicit. Never noise-fill.
                psi[sl] = 0.0
                continue
            psi[sl] = self._coerce_slot(values, self.slot_dim, name)
            active.append(idx)

        norm = float(np.linalg.norm(psi))
        if not np.isfinite(norm) or norm < _EPS:
            raise ValueError(
                "provided slots carry no energy; refusing to normalize"
            )
        return WaveState(psi / norm, self.cfg, settled=True)

    def disassemble_ingress(self, state: WaveState) -> dict:
        """Split a wave back into its named slot vectors.

        Returns:
            dict mapping slot name to a length-`D_slot` complex array.
        """
        if state.cfg.dim != self.dim:
            raise ValueError(
                f"state dim {state.cfg.dim} does not match router dim {self.dim}"
            )
        return {
            name: state.psi[self.cfg.slot_slice(self.slot_index(name))].copy()
            for name in self.slots
        }

    def active_slots(self, state: WaveState, tol: float = 0.0) -> list:
        """Indices whose slot energy exceeds `tol`.

        Default `tol = 0.0` reports every slot carrying ANY energy, which
        is what the exact-zero gates need.
        """
        energy = state.slot_energy()
        return [int(i) for i in np.flatnonzero(energy > tol)]

    # -- gates ---------------------------------------------------------

    def bleed(self, state: WaveState, active: Sequence[int]) -> float:
        """G-U1 metric: energy fraction outside the `active` slots."""
        return float(state.slot_bleed(list(active)))

    def bleed_outside(self, state: WaveState, active: Sequence[int]) -> float:
        """Raw (un-normalized) energy outside the active slots.

        G-ING3 states the bound as a raw sum of squared magnitudes, so
        this returns the raw quantity rather than a fraction. For a unit
        state the two agree; they are reported separately so a future
        non-normalized caller cannot silently change the meaning.
        """
        idx = set(int(i) for i in active)
        total = 0.0
        for i in range(N_SLOTS):
            if i in idx:
                continue
            sl = self.cfg.slot_slice(i)
            total += float(np.sum(np.abs(state.psi[sl]) ** 2))
        return total

    def unpopulated_energy(self, state: WaveState, provided: dict) -> float:
        """Raw energy in every slot that `provided` marked as absent.

        This is the exact-zero probe: it must return `0.0`, not a small
        number.
        """
        total = 0.0
        for name in self.slots:
            if provided.get(name) is not None:
                continue
            sl = self.cfg.slot_slice(self.slot_index(name))
            total += float(np.sum(np.abs(state.psi[sl]) ** 2))
        return total
