"""Sagnac homodyne veto gate.

Evidence label: DERIVED for the gate logic; OBSERVED for every number
`tests/test_sagnac_gate.py` measures.

PHASE 3 MODULE DIRECTIVE 1, with three measured corrections recorded.

CORRECTION 1: THE STATISTIC IS NOT NEW
`scripts/probe_phase3_gate_analysis.py` (claim A1) measured the spec's
homodyne statistic

    delta_sagnac = 1 - mean_d cos(delta_theta_d)

against the EXISTING `henri.sagnac.sagnac_delta = 1 - Re<c, a>` on 12
unit-modulus pairs:

    max |difference| = 0.000e+00

They are IDENTICAL for unit-modulus waves, because
`Re<c,a> = sum_d |c_d||a_d| cos(dtheta_d)` and `|c_d| = |a_d| = 1/sqrt(D)`.
Adding a second phase-mismatch convention would create exactly the
silent-drift hazard this repository has been burned by. This module
therefore COMPUTES the spec's phase form and ASSERTS it equals
`sagnac_delta`, so the equivalence is pinned by test rather than assumed.

The two forms DO diverge for non-unit-modulus rows (measured 5.770e-03
on a Gaussian pair), which is why the assertion is scoped to unit
modulus and the scope is stated.

CORRECTION 2: THE SPEC'S EQUATION IS INTERNALLY INCONSISTENT
Section 3.1 gives two expressions it declares equal:

    Delta = 1/(2D) * sum_d (1 - cos(dtheta_d)) = 1 - mean_d cos(dtheta_d)

The left side equals `HALF` of `(1 - mean cos)`, not `(1 - mean cos)`.
The two differ by a factor of 2 for every input. This module uses the
phase form `1 - mean(cos)` because that is the form the gate threshold
0.0431 was calibrated against, and because it is the form that matches
the existing `sagnac_delta`. The factor-of-2 discrepancy is recorded,
not silently reconciled.

CORRECTION 3: `settled` DOES NOT MEAN PRE-CLEANUP
The directive says `assert psi_cand.settled == False`. In this
repository `settled` means "normalized onto the sphere", which is True
for every valid candidate; asserting False would reject correct inputs.
`WaveState.cleaned` (additive, default False) carries the intended
meaning. The gate REQUIRES `not psi_cand.cleaned` and fails closed.

DARK-PORT ENERGY IS KEPT
A vetoed candidate records its mismatch signature into the recoil
buffer. Discarding it with a bare `continue` is a named anti-pattern:
it blinds the thermostat to the failure mode and degenerates search
into a random walk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from henri.config import WaveConfig
from henri import form_pin
from henri.sagnac import sagnac_delta
from henri.state import WaveState

_EPS = 1e-30

TAU_VETO_CEILING = 0.0431
"""Hard ceiling from the Phase 3 spec, section 2.2.

NOT the search-veto value 0.35 used elsewhere in this repository. The
two constants answer different questions and must never be swapped.
"""

CONSTRUCTIVE = "CONSTRUCTIVE"
DESTRUCTIVE = "DESTRUCTIVE_DARK_PORT"

RECOIL_CAPACITY = 64


@dataclass
class RecoilBuffer:
    """`SRAM_DARK_RECOIL`: circular store of dark-port failure energy.

    Capacity 64 entries, oldest overwritten. Each entry keeps the dark
    intensity AND the per-dimension mismatch comb, because the comb is
    what the anisotropic thermostat actually steers on.

    Attributes:
        capacity: number of slots.
        intensities: dark-port intensity per event, newest last.
        combs: per-dimension |delta theta| arrays, newest last.
        dropped: count of events overwritten by the circular rule.
    """

    capacity: int = RECOIL_CAPACITY
    intensities: List[float] = field(default_factory=list)
    combs: List[np.ndarray] = field(default_factory=list)
    dropped: int = field(default=0, init=False)

    def push(self, intensity_dark: float, mismatch: np.ndarray) -> None:
        """Record one veto event. Oldest entry is overwritten at capacity."""
        comb = np.abs(np.asarray(mismatch, dtype=np.float64).reshape(-1)).copy()
        if len(self.intensities) >= self.capacity:
            self.intensities.pop(0)
            self.combs.pop(0)
            self.dropped += 1
        self.intensities.append(float(intensity_dark))
        self.combs.append(comb)

    def __len__(self) -> int:
        return len(self.intensities)

    @property
    def total_energy(self) -> float:
        """Sum of stored dark intensities. Landauer-relevant quantity."""
        return float(np.sum(self.intensities)) if self.intensities else 0.0

    def latest_comb(self) -> Optional[np.ndarray]:
        """Most recent mismatch comb, or None when empty."""
        return self.combs[-1] if self.combs else None


@dataclass
class PassReceipt:
    """A candidate that cleared the gate."""

    status: str
    wave: WaveState
    delta_sagnac: float
    verified: bool = True
    statistic: str = field(default=form_pin.DENSE_FORM)
    dilution: float = field(default=1.0)

    def as_dict(self) -> dict:
        out = {
            "status": self.status,
            "delta_sagnac": float(self.delta_sagnac),
            "verified": bool(self.verified),
        }
        # Emitted only when it is not the identity, so dense receipts stay
        # byte-identical. A reported dilution is measurable; silence was the
        # defect (see henri/form_pin.py).
        if self.dilution != 1.0 or self.statistic != form_pin.DENSE_FORM:
            out["statistic"] = self.statistic
            out["dilution"] = float(self.dilution)
        return out


@dataclass
class VetoReceipt:
    """A candidate annihilated at the dark port."""

    status: str
    delta_sagnac: float
    intensity_dark: float
    mismatch: np.ndarray
    verified: bool = False
    statistic: str = field(default=form_pin.DENSE_FORM)
    dilution: float = field(default=1.0)

    def as_dict(self) -> dict:
        out = {
            "status": self.status,
            "delta_sagnac": float(self.delta_sagnac),
            "intensity_dark": float(self.intensity_dark),
            "mismatch_norm": float(np.linalg.norm(self.mismatch)),
            "verified": False,
        }
        if self.dilution != 1.0 or self.statistic != form_pin.DENSE_FORM:
            out["statistic"] = self.statistic
            out["dilution"] = float(self.dilution)
        return out


class SagnacHomodyneGate:
    """Common-path homodyne veto over PRE-cleanup candidate waves.

    Args:
        tau_veto: calibrated per-family threshold. Default 0.0431, the
            spec's hard ceiling.
        cfg: optional configuration. Used only for dimension checks.
    """

    def __init__(
        self, tau_veto: float = TAU_VETO_CEILING, cfg: Optional[WaveConfig] = None,
        statistic: Optional[str] = None, family: Optional[str] = None,
    ) -> None:
        if not 0.0 < tau_veto <= 1.0:
            raise ValueError("tau_veto must be in (0, 1]")
        self.tau_veto = float(tau_veto)
        self.cfg = cfg
        self.family = family
        # Pins the FORM at construction. An unmeasured family or an unknown
        # statistic name raises here, before any wave is read.
        self.statistic = form_pin.require_form(statistic, None, family)

    def statistic_value(self, psi_cand: WaveState, psi_axiom: WaveState) -> float:
        """The pinned statistic for this gate.

        `form_pin.DENSE_FORM` is the spec form and is what tau was calibrated
        against on unit-modulus rows. `form_pin.INVARIANT_FORM` is immune to
        zero-filled dimensions, which constraint C6 mandates.
        """
        if self.statistic == form_pin.INVARIANT_FORM:
            return form_pin.form_inner_product(psi_cand.psi, psi_axiom.psi)
        return form_pin.form_phase(psi_cand.psi, psi_axiom.psi)

    # -- statistics ----------------------------------------------------

    @staticmethod
    def phase_mismatch(cand: WaveState, axiom: WaveState) -> np.ndarray:
        """Element-wise `|delta theta|` in [0, pi], shape (D,).

        `delta theta_d = angle(cand_d * conj(axiom_d))`, which is the
        non-reciprocal phase divergence the spec defines. A scalar
        cosine check cannot express this.
        """
        if cand.cfg.dim != axiom.cfg.dim:
            raise ValueError("candidate and axiom dimension mismatch")
        return np.abs(np.angle(cand.psi * np.conj(axiom.psi)))

    def compute_phase_divergence(
        self, psi_cand: WaveState, psi_axiom: WaveState
    ) -> tuple:
        """Return `(delta_sagnac, |delta theta|)` for a PRE-cleanup candidate.

        Raises:
            ValueError: if the candidate carries the cleanup marker. A
                veto read after Hopfield relaxation cannot fail, because
                that operator has already folded the candidate toward the
                bank. Reading it there is the vacuous-gate defect.
        """
        if psi_cand.cleaned:
            raise ValueError(
                "candidate is marked post-cleanup; the Sagnac veto must read "
                "the PRE-cleanup wave (gate G-ACT1). Hopfield relaxation "
                "attracts the state toward the nearest codebook row and "
                "artificially deflates the error."
            )
        mismatch = self.phase_mismatch(psi_cand, psi_axiom)
        delta = 1.0 - float(np.mean(np.cos(mismatch)))
        return float(delta), mismatch

    def equivalent_scalar_delta(self, cand: WaveState, axiom: WaveState) -> float:
        """The existing scalar statistic, for the equivalence pin."""
        return sagnac_delta(cand, axiom)

    # -- decision ------------------------------------------------------

    def evaluate_trajectory(
        self,
        psi_cand: WaveState,
        psi_axiom: WaveState,
        recoil_buffer: Optional[RecoilBuffer] = None,
    ):
        """Split-gate decision. Returns a Pass or Veto receipt.

        On veto the dark intensity AND the mismatch comb are committed
        to `recoil_buffer`. Passing `None` is allowed but the caller then
        loses the failure signature the thermostat needs.
        """
        delta, mismatch = self.compute_phase_divergence(psi_cand, psi_axiom)
        if self.statistic != form_pin.DENSE_FORM:
            delta = self.statistic_value(psi_cand, psi_axiom)
        dilution = form_pin.dilution_factor(psi_cand.psi)
        if delta <= self.tau_veto:
            return PassReceipt(
                status=CONSTRUCTIVE, wave=psi_cand, delta_sagnac=delta,
                statistic=self.statistic, dilution=dilution,
            )
        intensity_dark = 0.5 * (1.0 - float(np.cos(delta)))
        if recoil_buffer is not None:
            recoil_buffer.push(intensity_dark, mismatch)
        return VetoReceipt(
            status=DESTRUCTIVE,
            delta_sagnac=delta,
            intensity_dark=intensity_dark,
            mismatch=mismatch,
            statistic=self.statistic,
            dilution=dilution,
        )

    def veto_decision(
        self,
        psi_cand: WaveState,
        psi_axiom: WaveState,
        recoil_buffer: Optional[RecoilBuffer] = None,
    ) -> bool:
        """True when the candidate is vetoed. Convenience wrapper."""
        return isinstance(
            self.evaluate_trajectory(psi_cand, psi_axiom, recoil_buffer),
            VetoReceipt,
        )
