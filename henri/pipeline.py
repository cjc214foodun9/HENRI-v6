"""End-to-end pipeline: ingress -> relax -> cleanup -> veto -> egress.

Evidence label: OBSERVED for whatever the smoke run reports. This
module makes no capability claim. It wires the validated primitives
into one executable path so the integration is exercised, not asserted.

Stage order follows the scaffolding specification sections 3.1-3.5,
pruned to the CPU-verifiable subset:

    ingress   deterministic slot-wise construction, unused slots zero
    relax     Koopman step, then Kuramoto lock check
    cleanup   Modern Hopfield CCCP against a codebook
    veto      calibrated Sagnac check against a boundary axiom
    egress    typed ActionEnvelope, never a flat token argmax
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from henri.config import N_SLOTS, WaveConfig
from henri.hopfield import cleanup
from henri.koopman import KoopmanOperator
from henri.kuramoto import LOCK_THRESHOLD, KuramotoState, lock, order_parameter
from henri.readout import ActionEnvelope, ReadoutResult, make_envelope, select
from henri.sagnac import VetoCalibration, confidence, veto
from henri.state import WaveState

_ACTIVE_SLOTS = (0, 1, 2)


@dataclass
class PipelineResult:
    """Full audit trail for one pipeline run.

    Attributes:
        ingress: the constructed input wave.
        relaxed: the wave after the dynamics stage.
        cleaned: the Hopfield-settled wave.
        index: the selected codebook index, or -1 on abstention.
        envelope: the typed action, if one was emitted.
        order_parameter: Kuramoto coherence after relaxation.
        locked: True when coherence met the target.
        veto_fired: True when the Sagnac gate rejected the candidate.
        stages: per-stage compact metrics.
    """

    ingress: WaveState
    relaxed: WaveState
    cleaned: WaveState
    index: int
    envelope: Optional[ActionEnvelope]
    order_parameter: float
    locked: bool
    veto_fired: bool
    readout: Optional[ReadoutResult] = None
    stages: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "order_parameter": self.order_parameter,
            "locked": self.locked,
            "veto_fired": self.veto_fired,
            "envelope": self.envelope.as_dict() if self.envelope else None,
            "readout": self.readout.as_dict() if self.readout else None,
            "stages": self.stages,
        }


def build_ingress(
    cfg: WaveConfig,
    slot_values: Sequence[np.ndarray],
    active: Sequence[int] = _ACTIVE_SLOTS,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> WaveState:
    """Assemble a wave from per-slot complex values.

    Constraint C6: any slot named in `slot_values` is written; every
    other slot is EXPLICITLY ZEROED. Noise-fill is never used.

    Args:
        cfg: wave configuration.
        slot_values: sequence of length 4. Each entry is an array of the
            slot span, or None to zero that slot.
        active: slots expected to carry signal. Used for the bleed
            report, not to gate construction.
        rng: accepted for signature symmetry with stochastic ingress
            variants; unused here because construction is deterministic.
        seed: accepted for signature symmetry; unused.

    Returns:
        A settled unit-norm WaveState.
    """
    del rng, seed  # determinism: this constructor never samples
    if len(slot_values) != N_SLOTS:
        raise ValueError(f"expected {N_SLOTS} slot entries, got {len(slot_values)}")

    state = WaveState.zeros(cfg)
    for i, values in enumerate(slot_values):
        if values is None:
            state.zero_slot(i)
        else:
            state.put_slot(i, np.asarray(values, dtype=np.complex128))
    state.normalize()
    return state


def run_pipeline(
    cfg: WaveConfig,
    ingress: WaveState,
    codebook: np.ndarray,
    axiom: WaveState,
    calibration: VetoCalibration,
    operator: Optional[KoopmanOperator] = None,
    group_key: Optional[np.ndarray] = None,
    coupling: float = 6.0,
    dt: float = 1e-3,
    relax_steps: int = 3,
    lock_steps: int = 200,
    seed: Optional[int] = None,
) -> PipelineResult:
    """Execute the full CPU pipeline once.

    Args:
        cfg: wave configuration.
        ingress: the input wave.
        codebook: array of shape `(M, D)` of memory patterns.
        axiom: boundary axiom for the Sagnac check.
        calibration: a calibrated veto threshold.
        operator: Koopman operator. Built from `seed` when omitted.
        group_key: optional tie-breaking key for stage-2 selection.
        coupling: Kuramoto coupling K.
        dt: integration step.
        relax_steps: Koopman steps.
        lock_steps: Kuramoto iterations.
        seed: seed for operator construction.

    Returns:
        A PipelineResult. Every published number here is a plain
        measurement, not a score.
    """
    rng = np.random.default_rng(seed)
    operator = operator or KoopmanOperator.build(cfg, rng=rng)

    # 1. relax: Koopman transition
    relaxed = ingress
    for _ in range(int(relax_steps)):
        relaxed = operator.step(relaxed)
    norm_drift = relaxed.unit_norm_error()

    # 2. relax: Kuramoto lock, reported not assumed
    ku = KuramotoState.from_wave(relaxed, rng=rng)
    ku_locked, r_trace = lock(ku, coupling, dt, lock_steps)
    r_final = order_parameter(ku_locked.phases)[0]
    locked = bool(r_final >= LOCK_THRESHOLD)

    # 3. cleanup: Hopfield CCCP
    cleaned, energy_trace = cleanup(relaxed, codebook)

    # 4. readout: tie-robust selection
    result = select(cleaned, codebook, group_key=group_key)

    # 5. veto: calibrated Sagnac
    fired = veto(cleaned, axiom, calibration)
    conf = confidence(cleaned, axiom)

    # 6. egress: typed envelope, never a flat token argmax
    if fired:
        envelope = make_envelope(
            "abstain",
            {"reason": "sagnac_dark_port"},
            conf,
            verified=False,
            veto_reason="delta exceeds calibrated epsilon",
        )
        index = -1
    else:
        envelope = make_envelope(
            "write_memory",
            {"index": result.index, "group_size": result.group_size},
            conf,
            verified=True,
        )
        index = result.index

    return PipelineResult(
        ingress=ingress,
        relaxed=relaxed,
        cleaned=cleaned,
        index=index,
        envelope=envelope,
        order_parameter=float(r_final),
        locked=locked,
        veto_fired=bool(fired),
        readout=result,
        stages={
            "norm_drift": float(norm_drift),
            "slot_energy_ingress": ingress.slot_energy().tolist(),
            "slot_bleed_ingress": float(ingress.slot_bleed(_ACTIVE_SLOTS)),
            "energy_initial": float(energy_trace[0]),
            "energy_final": float(energy_trace[-1]),
            "energy_steps": int(energy_trace.size),
            "r_initial": float(r_trace[0]),
            "r_final": float(r_final),
            "margin": float(result.margin),
            "group_size": int(result.group_size),
            "entropy": float(result.entropy),
            "confidence": float(conf),
            "epsilon": float(calibration.epsilon),
        },
    )


def make_codebook(
    cfg: WaveConfig,
    count: int,
    seed: int = 0,
    orthogonalize: bool = True,
) -> np.ndarray:
    """Deterministic complex codebook of `count` patterns.

    Args:
        cfg: wave configuration.
        count: number of patterns.
        seed: explicit seed. Required for reproducibility.
        orthogonalize: apply the Stiefel polar retraction so the rows
            are orthonormal. This is the honest way to build a codebook
            that satisfies the separation condition the Hopfield
            single-step snap depends on.
    """
    from henri.gram import stiefel_retract

    if count < 1:
        raise ValueError("count must be >= 1")
    rng = np.random.default_rng(seed)
    raw = (
        rng.standard_normal((count, cfg.dim))
        + 1j * rng.standard_normal((count, cfg.dim))
    )
    if orthogonalize and count <= cfg.dim:
        try:
            raw = stiefel_retract(raw)
        except ValueError:
            pass
    return raw.astype(np.complex128)
