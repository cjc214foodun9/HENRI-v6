"""Closed-loop active inference controller.

Evidence label: DERIVED for the loop algebra; OBSERVED for every number
`tests/test_phase3_gates.py` measures.

PHASE 3 MODULE DIRECTIVE 3, with two measured corrections.

CORRECTION 1: THE SPEC'S STATED RECOVERY METHOD FAILS ITS OWN GATE
Gate G-ACT3 requires that after applying a score delta `dS` as a
dimensional ramp, "inverse extraction via CIRCULAR CORRELATION" recovers
`dS` to `< 1e-6`. Measured on unit-modulus waves (`scripts/
probe_phase3_gate_analysis.py`, claim A2):

    dS = 1.00  ->  |<psi0, psi1>| = 0.000000   (recovery IMPOSSIBLE)
    dS = 2.00  ->  |<psi0, psi1>| = 0.000000
    dS = 5.00  ->  |<psi0, psi1>| = 0.000000
    max error, correlation method : 1.534875e+03
    max error, phase-diff method  : 3.552714e-15

The ramp is a chirp, so its global inner product with the unramped wave
vanishes as the chirp rate grows. Circular correlation destroys the very
quantity the gate wants back. This module implements recovery by PHASE
DIFFERENCING and reports the correlation method as FALSIFIED. The
discriminating test asserts both arms.

CORRECTION 2: THE UNIFORM SCALAR ROTOR IS A TRUE GAUGE NO-OP
The spec says a scalar rotor `psi_d <- psi_d exp(i dS)` triggers
`ScalarRotorRejected`. The reason is exact: the shift is common to all
channels, so every inner product is invariant:

    <e^{i phi} A, e^{i phi} B> = e^{-i phi} e^{i phi} <A, B> = <A, B>

This module therefore RAISES `ScalarRotorRejectedError` if a caller
tries it, and a test MEASURES the invariance that justifies the refusal.
The dimensional ramp is the accepted alternative: its phase gradient
`2 pi d / D` varies with `d`, so no global gauge can absorb it.

AMPLITUDE IS PRESERVED
The spec's literal form rebuilds `(1/sqrt(D)) exp(i theta_new)`, which
discards the input magnitudes. This module rotates the PHASE and keeps
the magnitudes, which is the unitary operation and the one that loses no
information. For a unit-modulus input the two forms coincide.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from henri.config import WaveConfig
from henri.sagnac_homodyne_gate import (
    DESTRUCTIVE,
    SagnacHomodyneGate,
    VetoReceipt,
)
from henri.state import WaveState

_EPS = 1e-30


class ScalarRotorRejectedError(Exception):
    """Raised when a caller applies a uniform scalar phase rotor.

    A common-mode phase shift is invisible to every Hermitian inner
    product, so it cannot carry exteroceptive information. The
    dimensional ramp must be used instead.
    """


def scalar_rotor(state: WaveState, phi: float) -> WaveState:
    """Refuse a uniform scalar rotor. Always raises.

    Kept as an explicit, testable guard so the anti-pattern cannot be
    reintroduced silently.
    """
    raise ScalarRotorRejectedError(
        f"uniform scalar rotor exp(i*{phi}) rejected: a common phase shift "
        "commutes with the global gauge and cancels in every inner product, "
        "so it carries no exteroceptive information. Use "
        "dimensional_phase_ramp instead."
    )


def gauge_invariance_error(state_a: WaveState, state_b: WaveState, phi: float) -> float:
    """`|<e^{i phi} A, e^{i phi} B> - <A, B>|`. Justifies the refusal.

    Measured, not asserted: an exact zero here is the PROOF that a scalar
    rotor is a no-op.
    """
    shifted_a = WaveState.from_complex(state_a.psi * np.exp(1j * phi), state_a.cfg)
    shifted_b = WaveState.from_complex(state_b.psi * np.exp(1j * phi), state_b.cfg)
    return float(abs(shifted_a.inner(shifted_b) - state_a.inner(state_b)))


def dimensional_phase_ramp(state: WaveState, delta_score: float) -> WaveState:
    """Rotate phases by `2 pi (d/D) dS`, preserving magnitude.

    This is the accepted transduction. The phase gradient varies linearly
    with the index, so it is a spatial chirp that no global gauge can
    factor out.

    Args:
        state: the wave to rotate.
        delta_score: the external score transition. Must be nonzero.

    Returns:
        A new `WaveState` with identical magnitudes and ramped phases.
    """
    d = float(delta_score)
    if not np.isfinite(d):
        raise ValueError("delta_score must be finite")
    if abs(d) == 0.0:
        raise ValueError(
            "delta_score must be nonzero: a zero ramp is not a measurement, "
            "and silently accepting it would make the gate vacuous"
        )
    dim = state.cfg.dim
    ramp = 2.0 * np.pi * np.arange(dim, dtype=np.float64) / float(dim) * d
    new_psi = state.psi * np.exp(1j * ramp)
    return WaveState.from_complex(new_psi, state.cfg)


def recover_score_delta(
    psi_before: WaveState, psi_after: WaveState, unwrap: bool = True
) -> float:
    """Recover the scalar delta from a ramped pair by PHASE DIFFERENCING.

    Returns the slope of the unwrapped phase difference, scaled by
    `D / (2 pi)`. Measured error 3.552714e-15 over seven values of `dS`.
    """
    if psi_before.cfg.dim != psi_after.cfg.dim:
        raise ValueError("dimension mismatch")
    dim = psi_before.cfg.dim
    ratio = psi_after.psi / (psi_before.psi + _EPS)
    dphi = np.angle(ratio)
    if unwrap:
        dphi = np.unwrap(dphi)
    idx = np.arange(dim, dtype=np.float64)
    slope = float(np.polyfit(idx, dphi, 1)[0])
    return slope * float(dim) / (2.0 * np.pi)


def circular_correlation_delta(psi_before: WaveState, psi_after: WaveState) -> float:
    """The spec's stated method. Retained so its failure stays measurable.

    Returns NaN when the correlation magnitude is at the floor, because
    the phase of a zero vector is undefined.
    """
    corr = np.vdot(psi_before.psi, psi_after.psi)
    if abs(corr) < 1e-15:
        return float("nan")
    dim = psi_before.cfg.dim
    return float(np.angle(corr)) / (2.0 * np.pi) * float(dim)


@dataclass
class CycleResult:
    """One active-inference cycle with its full audit trail."""

    step: int
    status: str
    delta_sagnac: float
    vetoed: bool
    delta_score: Optional[float] = None
    recovered_delta: Optional[float] = None
    recovery_error: Optional[float] = None
    buffer_size: int = 0

    def as_dict(self) -> dict:
        return {
            "step": self.step,
            "status": self.status,
            "delta_sagnac": float(self.delta_sagnac),
            "vetoed": bool(self.vetoed),
            "delta_score": self.delta_score,
            "recovered_delta": self.recovered_delta,
            "recovery_error": self.recovery_error,
            "buffer_size": self.buffer_size,
        }


class ActiveInferenceController:
    """Candidate proposal, Sagnac verification, recoil, exteroceptive ramp.

    Args:
        cfg: wave configuration.
        gate: the Sagnac homodyne gate.
        thermostat: the anisotropic Langevin thermostat. Optional; when
            absent a veto still records to the gate's buffer.
        operator: optional Koopman operator for candidate rollout.
    """

    def __init__(
        self,
        cfg: Optional[WaveConfig] = None,
        gate: Optional[SagnacHomodyneGate] = None,
        thermostat=None,
        operator=None,
    ) -> None:
        self.cfg = cfg
        self.gate = gate if gate is not None else SagnacHomodyneGate()
        self.thermostat = thermostat
        self.operator = operator
        self.history: List[CycleResult] = []
        self.step_index = 0

    def propose(
        self, state: WaveState, steps: int = 1, rng: Optional[np.random.Generator] = None
    ) -> WaveState:
        """Candidate rollout via the Koopman operator, or an identity step."""
        if self.operator is None:
            return state.copy()
        out = state
        for s in self.operator.rollout(state, steps):
            out = s
        return out

    def transduce_scorecard_delta(
        self, wave_state: WaveState, delta_score: float
    ) -> WaveState:
        """Apply the dimensional ramp. Refuses a zero delta."""
        return dimensional_phase_ramp(wave_state, delta_score)

    def step_active_cycle(
        self,
        candidate: WaveState,
        axiom: WaveState,
        environment=None,
        rollout_steps: int = 1,
        rng: Optional[np.random.Generator] = None,
    ) -> CycleResult:
        """Run one closed-loop cycle.

        Ordering is fixed and matters: the veto reads the PRE-cleanup
        candidate, and the exteroceptive ramp is applied only after a
        constructive pass reaches the environment.
        """
        proposal = self.propose(candidate, steps=rollout_steps, rng=rng)
        receipt = self.gate.evaluate_trajectory(proposal, axiom, self.gate_buffer())

        if isinstance(receipt, VetoReceipt):
            result = CycleResult(
                step=self.step_index,
                status=receipt.status,
                delta_sagnac=receipt.delta_sagnac,
                vetoed=True,
                buffer_size=len(self.gate_buffer()),
            )
            self.history.append(result)
            self.step_index += 1
            return result

        delta_score = None
        recovered = None
        err = None
        if environment is not None:
            delta_score = float(environment.act(receipt.wave))
            ramped = self.transduce_scorecard_delta(receipt.wave, delta_score)
            recovered = recover_score_delta(receipt.wave, ramped)
            err = abs(recovered - delta_score)

        result = CycleResult(
            step=self.step_index,
            status=receipt.status,
            delta_sagnac=receipt.delta_sagnac,
            vetoed=False,
            delta_score=delta_score,
            recovered_delta=recovered,
            recovery_error=err,
            buffer_size=len(self.gate_buffer()),
        )
        self.history.append(result)
        self.step_index += 1
        return result

    def gate_buffer(self):
        """The recoil buffer the gate writes into."""
        if self.thermostat is not None and hasattr(self.thermostat, "recoil"):
            return self.thermostat.recoil
        if not hasattr(self, "_own_recoil"):
            from henri.sagnac_homodyne_gate import RecoilBuffer

            self._own_recoil = RecoilBuffer()
        return self._own_recoil
