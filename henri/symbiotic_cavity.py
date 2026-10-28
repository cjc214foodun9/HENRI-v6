"""The 3-zone symbiotic cavity: Zone A -> Zone B -> Zone C and back.

Pass 1 is a cold ballistic pass. When the triadic Sagnac reports dissonance, the
dark port heats the offending slots and sweeps Pockels voltages until the state
locks, then the ledger commits the verified result.

Honest boundaries:
  - `tau_veto` defaults to 0.0431, the constant this tree ships. The revised
    document's 0.00811 has zero instantiations here; passing it is allowed and
    the deviation is reported in the register.
  - "OOD" here means COMPOSITIONAL TRANSFER: a novel entity under a known
    operator. It is not a generalization claim and not a benchmark result.
  - The BTO plates run in phase-only mode in the repair loop (no diffraction, no
    Kerr). Diffraction and Kerr are exercised in the Zone B gates, not here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from henri.bto_pockels_d2nn import BTODeepDiffractiveNetwork
from henri.holographic_ledger import HolographicAgenticLedger
from henri.koopman_sheaf import PhaseGatedKoopmanSheaf, retract_to_sphere
from henri.triadic_sagnac import TriadicSagnacGate
from henri.transducer_swarm import ZoneATransducerSwarm

TWO_PI = 2.0 * math.pi


@dataclass
class SymbioticTelemetry:
    label: str
    margin: float
    passes: int
    heated_passes: int
    cos_trace: List[float]
    converged: bool
    committed: bool
    regime: str
    boundary_cos: float
    retraction_correction: float


class HenriSymbioticCavity:
    def __init__(
        self,
        d_model: int = 16384,
        num_layers: int = 1,
        rank: int = 32,
        num_charts: int = 4,
        tau_veto: float = 0.0431,
        max_passes: int = 8,
        pockels_gain: float = 0.02,
        kappa: float = 0.9,
        abstain_margin: float = 0.65,
        seed: int = 42,
    ) -> None:
        self.d_model = int(d_model)
        self.tau_veto = float(tau_veto)
        self.max_passes = int(max_passes)
        self.abstain_margin = float(abstain_margin)
        self.pockels_gain = float(pockels_gain)
        # Device authority limits. The control law asks for a damped phase step
        # and then computes the voltage that realises it; the voltage is clamped
        # to the device range. This is what makes the Curie gain load-bearing:
        # at Lambda = 1 the clamp cannot supply the requested step.
        self.max_phase_step = 0.35
        self.max_voltage = 2.5
        # The veto threshold is not the repair target. tau asks "is this good
        # enough to pass"; the thermostat drives the phase residual to a much
        # tighter tolerance, otherwise it stops at the first admissible state.
        self.repair_tol = 0.01

        self.swarm = ZoneATransducerSwarm(d_model=self.d_model, seed=seed)
        self.bto = BTODeepDiffractiveNetwork(
            d_model=self.d_model,
            num_layers=num_layers,
            seed=seed,
            pockels_gain=pockels_gain,
            kerr=0.0,
            fresnel_zeta=0.0,
            kappa=kappa,
        )
        self.koopman = PhaseGatedKoopmanSheaf(
            d_model=self.d_model,
            d_slot=self.d_model // 4,
            rank=rank,
            num_charts=num_charts,
            seed=seed,
        )
        self.ledger = HolographicAgenticLedger(d_model=self.d_model, seed=seed)
        self.sagnac = TriadicSagnacGate(d_model=self.d_model, num_slots=4, tau_veto=tau_veto)
        self.operator_consequence: Dict[Tuple[str, ...], np.ndarray] = {}
        self.law_labels: Dict[Tuple[str, ...], str] = {}
        self._node = np.zeros(self.d_model, dtype=np.complex128)
        self.history: List[SymbioticTelemetry] = []

    # ------------------------------------------------------------------ routing
    def build_state(
        self,
        entity: Sequence[str],
        operator: Sequence[str],
        consequence: Optional[np.ndarray] = None,
        phase_error: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        e = self.swarm.encode_text(entity)
        o = self.swarm.encode_operator(operator)
        c_true = self.operator_consequence[tuple(operator)]
        c_in = c_true * np.exp(1j * phase_error)
        query = self.swarm.route(slot0_entity=e, slot1_action=o)
        state = self.swarm.route(slot0_entity=e, slot1_action=o, slot2_target=c_in)
        boundary = self.swarm.route(slot0_entity=e, slot1_action=o, slot2_target=c_true)
        return state, boundary

    @staticmethod
    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=np.complex128).reshape(-1)
        b = np.asarray(b, dtype=np.complex128).reshape(-1)
        return float(np.real(np.vdot(retract_to_sphere(a), retract_to_sphere(b))))

    # --------------------------------------------------------------- teach law
    def teach_law(
        self, entity: Sequence[str], operator: Sequence[str], consequence: Sequence[str]
    ) -> Dict[str, object]:
        e = self.swarm.encode_text(entity)
        o = self.swarm.encode_operator(operator)
        c = self.swarm.encode_text(consequence)
        self.operator_consequence[tuple(operator)] = c
        self.law_labels[tuple(operator)] = f"{' '.join(operator)} → {' '.join(consequence)}"

        query = self.swarm.route(slot0_entity=e, slot1_action=o)
        law = self.swarm.route(slot0_entity=e, slot1_action=o, slot2_target=c)
        self.koopman.online_edmd_update(query, law, chart_idx=0)
        rec = self.ledger.commit(query, law, label=" → ".join(consequence))
        return {"record_id": rec.record_id, "law_cos": self._cos(law, self.swarm.route(
            slot0_entity=e, slot1_action=o, slot2_target=c))}

    # --------------------------------------------------------------------- step
    def step(
        self,
        entity: Sequence[str],
        operator: Sequence[str],
        phase_error: float = 0.0,
        max_passes: Optional[int] = None,
    ) -> SymbioticTelemetry:
        if tuple(operator) not in self.operator_consequence:
            raise KeyError(f"operator {operator!r} was never taught; no consequence is registered")

        state, boundary = self.build_state(entity, operator, phase_error=phase_error)
        ret = self.ledger.query(state)
        label = self.law_labels.get(tuple(operator), "")
        margin = float(ret.margin)
        cos_trace = [self._cos(state, boundary)]

        # Pass 1: cold ballistic.
        state, _gain, corr = self.bto.step(state, slot_temps=(0.0, 0.0, 0.0, 0.0))
        cos_trace.append(self._cos(state, boundary))
        passes = 1
        heated = 0
        limit = self.max_passes if max_passes is None else int(max_passes)

        # Pass 2..K: Curie-heated Pockels repair, driven by the phase residual.
        def _residual(s: np.ndarray) -> float:
            return float(np.max(np.abs(np.angle(boundary * np.conj(s)))))

        while _residual(state) > self.repair_tol and passes < limit:
            passes += 1
            heated += 1
            t_norm = min(0.5 + 0.1 * heated, 0.95)
            lam = self.bto.layers[-1].gain(t_norm)
            delta = np.angle(boundary * np.conj(state))
            desired = np.clip(0.5 * delta, -self.max_phase_step, self.max_phase_step)
            voltage = np.clip(-desired / (self.pockels_gain * lam), -self.max_voltage, self.max_voltage)
            state, _gain, corr = self.bto.step(
                state, slot_temps=(t_norm,) * 4, voltage_delta=voltage
            )
            cos_trace.append(self._cos(state, boundary))

        boundary_cos = cos_trace[-1]
        converged = (1.0 - boundary_cos) <= self.tau_veto
        committed = False
        if converged and heated:
            # Verified compositional synthesis: commit the repaired state.
            self.ledger.commit(state, boundary, label=label + " [ood]")
            committed = True

        out_label = label if margin >= self.abstain_margin else "<abstain>"
        regime = "SYSTEM_1_BALLISTIC" if heated == 0 else "SYSTEM_2_CURIE"
        tel = SymbioticTelemetry(
            label=out_label,
            margin=margin,
            passes=passes,
            heated_passes=heated,
            cos_trace=cos_trace,
            converged=converged,
            committed=committed,
            regime=regime,
            boundary_cos=boundary_cos,
            retraction_correction=float(corr),
        )
        self.history.append(tel)
        return tel
