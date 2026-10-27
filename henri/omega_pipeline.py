"""End-to-end Omega pipeline: Zone A -> Zone B -> Zone C -> egress.

Omega Amendment 7.

Differences from the reference sketch, each deliberate:
  - Retrieval reads the QUERY wave (`psi_in`), not the prediction. A memory
    readout answers "what does this input retrieve", and retrieving on the
    prediction lets a broken Zone B still score.
  - `tau_veto` defaults to 0.0431, the constant the baseline actually ships.
    0.00811 has no instantiation in HENRI-v6.
  - The thermostat heats the slot with a DEFICIT when the global veto fires but
    no slot exceeds tau (the reference loop could not terminate).
  - Text labels come from the chart's own label list, not a mismatched dict.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from henri.egress_router import EgressOutput, OmniModalEgressRouter
from henri.harmonic_lexicon import HarmonicLexicon
from henri.hhrg_memory import HHRGMemory, HHRGRetrievalResult
from henri.holographic_stack import HolographicCallStack
from henri.koopman_sheaf import PhaseGatedKoopmanSheaf, retract_to_sphere
from henri.omega_resonator import AdiabaticResonator
from henri.triadic_sagnac import TriadicSagnacGate, TriadicSagnacResult


@dataclass
class OmegaStepTelemetry:
    output: EgressOutput
    retrieval: Optional[HHRGRetrievalResult]
    sagnac: TriadicSagnacResult
    thermostat_steps: int
    active_chart_weights: np.ndarray
    psi_in: np.ndarray
    psi_pred: np.ndarray
    psi_out: np.ndarray
    retrieval_repaired: Optional[HHRGRetrievalResult] = None


class HenriOmegaPipeline:
    def __init__(
        self,
        d_model: int = 16384,
        rank: int = 32,
        num_charts: int = 4,
        tau_veto: float = 0.0431,
        max_thermostat_steps: int = 12,
        thermostat_sigma: float = 0.35,
        auto_bootstrap: bool = True,
        seed: int = 42,
    ) -> None:
        if d_model % 4 != 0:
            raise ValueError("d_model must be divisible by 4")
        self.d_model = int(d_model)
        self.d_slot = self.d_model // 4
        self.tau_veto = float(tau_veto)
        self.max_thermostat_steps = int(max_thermostat_steps)
        self.thermostat_sigma = float(thermostat_sigma)
        self.seed = int(seed)

        self.lexicon = HarmonicLexicon(
            vocab_size=512, d_slot=self.d_slot, d_sem=64, seed=seed
        )
        self.call_stack = HolographicCallStack(d_slot=self.d_slot, seed=seed)
        self.sheaf = PhaseGatedKoopmanSheaf(
            d_model=self.d_model,
            d_slot=self.d_slot,
            rank=rank,
            num_charts=num_charts,
            seed=seed,
        )
        self.sagnac = TriadicSagnacGate(
            d_model=self.d_model, num_slots=4, tau_veto=tau_veto
        )
        self.memory = HHRGMemory(d_model=self.d_model, chart_capacity=32)
        self.resonator = AdiabaticResonator(d_slot=self.d_slot, tau_lock=tau_veto, seed=seed)
        self.egress = OmniModalEgressRouter(d_model=self.d_model, seed=seed)

        self.baseplate_inv = np.zeros(self.d_model, dtype=np.complex128)
        # Readout margin below which the text egress abstains. Without this the
        # pipeline has no "I do not know" path and maps every input onto the bank.
        self.abstain_margin = 0.65

        rng = np.random.default_rng(seed)
        raw = rng.standard_normal((5, self.d_model)) + 1j * rng.standard_normal((5, self.d_model))
        self.ast_codebook = np.stack([retract_to_sphere(r) for r in raw], axis=0)

        self.axioms: List[Tuple[List[str], str]] = [
            (["photon", "light", "quantum"], "electromagnetic_wave_packet"),
            (["gravity", "mass", "curvature"], "spacetime_geodesic_metric"),
            (["unitary", "stiefel", "manifold"], "norm_preserving_wave_evolution"),
            (["python", "sort", "list"], "returns_ordered_sequence"),
        ]

        if auto_bootstrap:
            self.bootstrap_default_axioms()

    # ------------------------------------------------------------------ routing
    def route_slots(
        self,
        slot0_entity: Optional[np.ndarray] = None,
        slot1_action: Optional[np.ndarray] = None,
        slot2_target: Optional[np.ndarray] = None,
        slot3_context: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        psi = np.zeros(self.d_model, dtype=np.complex128)
        for k, wave in enumerate((slot0_entity, slot1_action, slot2_target, slot3_context)):
            if wave is None:
                continue
            flat = np.asarray(wave, dtype=np.complex128).reshape(-1)
            n = float(np.linalg.norm(flat))
            if n > 1e-9:
                psi[k * self.d_slot : (k + 1) * self.d_slot] = flat / n
        return retract_to_sphere(psi)

    # ---------------------------------------------------------------- bootstrap
    def bootstrap_default_axioms(self) -> None:
        src: List[int] = []
        dst: List[int] = []
        w: List[float] = []
        for words, _ans in self.axioms:
            ids = [self.lexicon.register_token(t) for t in words]
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    src.extend([ids[i], ids[j]])
                    dst.extend([ids[j], ids[i]])
                    w.extend([2.5, 2.5])
        if src:
            self.lexicon.diffuse_ppmi_graph(
                np.array(src), np.array(dst), np.array(w), steps=30
            )

        accum = np.zeros(self.d_model, dtype=np.complex128)
        for idx, (words, answer) in enumerate(self.axioms):
            chart = idx % self.sheaf.num_charts
            _, s0 = self.lexicon.encode_tokens(words, positional=False)
            _, s2 = self.lexicon.encode_tokens([answer], positional=False)
            ctx = self.sheaf.chart_keys[chart]
            key = self.route_slots(slot0_entity=s0, slot3_context=ctx)
            val = self.route_slots(slot0_entity=s0, slot2_target=s2, slot3_context=ctx)
            self.sheaf.online_edmd_update(key, val, chart_idx=chart)
            self.memory.ingest(val, val, label=answer)
            accum = accum + val
        self.baseplate_inv = retract_to_sphere(accum)

    # ------------------------------------------------------------------- static
    @staticmethod
    def one_shot_rule_extract(z_in_torus: np.ndarray, z_out_torus: np.ndarray) -> np.ndarray:
        return np.asarray(z_out_torus) * np.conj(np.asarray(z_in_torus))

    @staticmethod
    def apply_torus_rule(z_test_torus: np.ndarray, rule_torus: np.ndarray) -> np.ndarray:
        return np.asarray(rule_torus) * np.asarray(z_test_torus)

    # --------------------------------------------------------------------- step
    def step(
        self,
        query_tokens: Optional[Sequence[str]] = None,
        custom_wave: Optional[np.ndarray] = None,
        sensory_override: Optional[np.ndarray] = None,
        reference_baseplate: Optional[np.ndarray] = None,
        egress_mode: str = "text",
        chart_idx: int = 0,
    ) -> OmegaStepTelemetry:
        if custom_wave is not None:
            psi_in = retract_to_sphere(np.asarray(custom_wave, dtype=np.complex128).reshape(-1))
        elif query_tokens is not None:
            _, s0 = self.lexicon.encode_tokens(query_tokens, positional=False)
            psi_in = self.route_slots(
                slot0_entity=s0, slot3_context=self.sheaf.chart_keys[int(chart_idx)]
            )
        else:
            raise ValueError("provide query_tokens or custom_wave")

        psi_pred, weights = self.sheaf.step(psi_in, chart_idx=int(chart_idx))
        retrieval = self.memory.query(psi_in)

        # The invariant is the stored VALUE (the axiom). The orthogonalized key is
        # an addressing artifact and must not be used as the epistemic reference.
        baseplate = (
            np.asarray(reference_baseplate, dtype=np.complex128).reshape(-1)
            if reference_baseplate is not None
            else retrieval.matched_value
        )
        result = self.sagnac.evaluate(psi_pred, baseplate, psi_sens=sensory_override)

        thermo_steps = 0
        psi_curr = psi_pred
        ou_noise = np.zeros(self.d_model, dtype=np.complex128)

        while result.veto_fired and thermo_steps < self.max_thermostat_steps:
            thermo_steps += 1
            slots = psi_curr.reshape(4, self.d_slot).copy()
            ref = result.updated_baseplate.reshape(4, self.d_slot)

            mask = result.slot_thermal_mask.copy()
            if not np.any(mask > 0.0):
                ref_norm = np.linalg.norm(ref, axis=-1)
                cur_norm = np.linalg.norm(slots, axis=-1)
                mask = (ref_norm > cur_norm + 1e-9).astype(np.float64)
                if not np.any(mask > 0.0):
                    mask = np.ones(4, dtype=np.float64)

            for k in range(4):
                if mask[k] <= 0.0:
                    continue
                err_k = max(float(result.delta_phi_slots[k]), 1e-6)
                white = (
                    np.random.standard_normal(self.d_slot)
                    + 1j * np.random.standard_normal(self.d_slot)
                ) / math.sqrt(2.0 * float(self.d_slot))
                band = slice(k * self.d_slot, (k + 1) * self.d_slot)
                ou_noise[band] = 0.8 * ou_noise[band] + math.sqrt(0.36) * white
                kick = self.thermostat_sigma * math.sqrt(err_k) * ou_noise[band]
                slots[k] = 0.65 * ref[k] + 0.35 * (slots[k] + kick)

            psi_curr = retract_to_sphere(slots.reshape(-1))
            result = self.sagnac.evaluate(psi_curr, result.updated_baseplate, psi_sens=sensory_override)

        final_ret = self.memory.query(psi_curr)

        if egress_mode == "code_ast":
            egress_out = self.egress.decode_ast(psi_curr, self.ast_codebook)
        elif egress_mode == "motor_efe":
            egress_out = self.egress.decode_motor(psi_curr, sagnac_verified=not result.veto_fired)
        else:
            # Egress reads the PRE-thermostat retrieval. Reading the repaired
            # state would let a broken Zone B score by collapsing onto memory.
            source = retrieval
            chart = self.memory.charts[source.chart_id]
            if source.margin < self.abstain_margin:
                label = "<abstain>"
            else:
                label = chart.labels[source.local_index] if chart.labels else ""
            egress_out = EgressOutput(
                mode="text",
                content=label,
                margin=source.margin,
                entropy_norm=source.entropy_norm,
                syntax_valid=True,
                sagnac_verified=not result.veto_fired,
            )

        return OmegaStepTelemetry(
            output=egress_out,
            retrieval=retrieval,
            sagnac=result,
            thermostat_steps=thermo_steps,
            active_chart_weights=weights,
            psi_in=psi_in,
            psi_pred=psi_pred,
            psi_out=psi_curr,
            retrieval_repaired=final_ret,
        )
