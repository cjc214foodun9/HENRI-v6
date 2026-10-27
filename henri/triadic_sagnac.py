"""Triadic contrapuntal Sagnac interferometer with baseplate annealing.

Omega Amendment 3.

Fixes applied to the reference sketch:
  - `anomaly_streak` is keyed to the observed baseplate. A stale streak can no
    longer trigger a spurious paradigm shift on an unrelated baseplate.
  - Slot errors are magnitude-weighted, so an almost-empty slot cannot dominate
    the global verdict.
  - A single contradicting observation never anneals when confirm_threshold > 1.

The gate is falsifiable by construction: removing the sensory override, or
lowering confirm_threshold to 1, changes the measured regime.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass
class TriadicSagnacResult:
    veto_fired: bool
    epistemic_regime: str
    delta_phi_global: float
    delta_phi_slots: np.ndarray
    slot_thermal_mask: np.ndarray
    updated_baseplate: np.ndarray


def _retract(wave: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(wave))
    if n < 1e-12:
        return np.zeros_like(wave)
    return wave / n


class TriadicSagnacGate:
    """Three-wave phase comparator: prediction, stored axiom, live sensation."""

    def __init__(
        self,
        d_model: int = 65536,
        num_slots: int = 4,
        tau_veto: float = 0.0431,
        anneal_rate: float = 0.35,
        confirm_threshold: int = 2,
    ) -> None:
        if d_model % num_slots != 0:
            raise ValueError("d_model must be divisible by num_slots")
        if not 0.0 < tau_veto <= 1.0:
            raise ValueError("tau_veto must be in (0, 1]")
        self.d_model = int(d_model)
        self.num_slots = int(num_slots)
        self.d_slot = self.d_model // self.num_slots
        self.tau_veto = float(tau_veto)
        self.anneal_rate = float(anneal_rate)
        self.confirm_threshold = int(confirm_threshold)
        self.anomaly_streak = 0
        self._streak_key: Optional[bytes] = None

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _fingerprint(wave: np.ndarray) -> bytes:
        return np.round(np.asarray(wave, dtype=np.complex128).real, 9).tobytes() + np.round(
            np.asarray(wave, dtype=np.complex128).imag, 9
        ).tobytes()

    def slot_homodyne_errors(self, psi_a: np.ndarray, psi_b: np.ndarray) -> np.ndarray:
        a = np.asarray(psi_a, dtype=np.complex128).reshape(self.num_slots, self.d_slot)
        b = np.asarray(psi_b, dtype=np.complex128).reshape(self.num_slots, self.d_slot)
        norm_a = np.linalg.norm(a, axis=-1)
        norm_b = np.linalg.norm(b, axis=-1)
        active = (norm_a > 1e-6) & (norm_b > 1e-6)
        a_u = a / np.clip(norm_a[:, None], 1e-12, None)
        b_u = b / np.clip(norm_b[:, None], 1e-12, None)
        cos = np.clip(np.real(np.sum(a_u * np.conj(b_u), axis=-1)), -1.0, 1.0)
        raw = 1.0 - cos
        weight = np.minimum(norm_a, norm_b) / max(1.0, float(np.max(np.minimum(norm_a, norm_b))))
        return np.where(active, raw * weight, 0.0)

    # ----------------------------------------------------------------- evaluate
    def evaluate(
        self,
        psi_pred: np.ndarray,
        psi_inv: Optional[np.ndarray] = None,
        psi_sens: Optional[np.ndarray] = None,
    ) -> TriadicSagnacResult:
        pred = np.asarray(psi_pred, dtype=np.complex128).reshape(-1)
        zero_slots = np.zeros(self.num_slots, dtype=np.float64)

        if psi_inv is None or float(np.linalg.norm(psi_inv)) < 1e-6:
            source = psi_sens if psi_sens is not None else pred
            self.anomaly_streak = 0
            self._streak_key = None
            return TriadicSagnacResult(
                veto_fired=False,
                epistemic_regime="COLD_ANCHORED",
                delta_phi_global=0.0,
                delta_phi_slots=zero_slots.copy(),
                slot_thermal_mask=zero_slots.copy(),
                updated_baseplate=_retract(np.asarray(source, dtype=np.complex128).reshape(-1)),
            )

        inv = _retract(np.asarray(psi_inv, dtype=np.complex128).reshape(-1))
        pred_u = _retract(pred)
        key = self._fingerprint(inv)
        if key != self._streak_key:
            self._streak_key = key
            self.anomaly_streak = 0

        delta_model = float(1.0 - np.real(np.vdot(pred_u, inv)))
        slot_errors = self.slot_homodyne_errors(pred_u, inv)
        slot_mask = (slot_errors > self.tau_veto).astype(np.float64)

        if psi_sens is not None and float(np.linalg.norm(psi_sens)) > 1e-6:
            sens = _retract(np.asarray(psi_sens, dtype=np.complex128).reshape(-1))
            delta_reality = float(1.0 - np.real(np.vdot(pred_u, sens)))
            delta_axiom = float(1.0 - np.real(np.vdot(inv, sens)))

            if delta_axiom > self.tau_veto and delta_reality <= self.tau_veto:
                self.anomaly_streak += 1
                if self.anomaly_streak >= self.confirm_threshold:
                    annealed = _retract(
                        (1.0 - self.anneal_rate) * inv + self.anneal_rate * sens
                    )
                    new_errors = self.slot_homodyne_errors(pred_u, annealed)
                    new_mask = (new_errors > self.tau_veto).astype(np.float64)
                    new_global = float(1.0 - np.real(np.vdot(pred_u, annealed)))
                    self.anomaly_streak = 0
                    self._streak_key = self._fingerprint(annealed)
                    return TriadicSagnacResult(
                        veto_fired=bool(new_global > self.tau_veto),
                        epistemic_regime="AXIOM_ANNEALED",
                        delta_phi_global=new_global,
                        delta_phi_slots=new_errors,
                        slot_thermal_mask=new_mask,
                        updated_baseplate=annealed,
                    )
            else:
                self.anomaly_streak = 0

        veto = bool(delta_model > self.tau_veto or np.any(slot_mask > 0.0))
        return TriadicSagnacResult(
            veto_fired=veto,
            epistemic_regime="HALLUCINATION_VETO" if veto else "RESONANT",
            delta_phi_global=delta_model,
            delta_phi_slots=slot_errors,
            slot_thermal_mask=slot_mask,
            updated_baseplate=inv,
        )
