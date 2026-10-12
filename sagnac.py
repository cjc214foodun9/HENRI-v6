"""Calibrated Sagnac homodyne veto.

Evidence label: OBSERVED for the measured calibration. The retired
repository is the source of constraint C2.

Failure that this module exists to prevent
------------------------------------------
The retired repo hardcoded `epsilon = 0.35`. That value was derived for
the entry-unimodular COMPLEX wave family, whose chance coherence is
about `0.009792`. The evaluation bank actually ran on the REAL-valued
family, whose chance coherence is about `0.120168` -- 12.3x higher.
Every candidate therefore fell below the fixed threshold, the gate
saturated, and the measured AUC collapsed to `0.4982`, which is chance.

Two rules follow, and both are enforced here:

1. `epsilon` is calibrated from the empirical chance coherence of the
   SAME wave family that will be evaluated. Never a literal.
2. The statistic is read from the PRE-mixing wave. A veto computed
   after a cleanup operator has folded the candidate into the bank
   cannot fail, because that operator has already mixed them.

Definition:

    delta = 1 - Re<cand, axiom> / (||cand|| * ||axiom||)

`delta = 0` means identical phase, `delta = 1` means orthogonal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from henri.state import WaveState

_EPS = 1e-30


@dataclass
class VetoCalibration:
    """Empirical calibration record for the veto threshold.

    Attributes:
        epsilon: the calibrated threshold actually used.
        baseline_mean: mean chance coherence of the same wave family.
        baseline_std: standard deviation of chance coherence.
        sigma: multiplier applied to the standard deviation.
        ceiling: absolute cap that `epsilon` may not exceed.
        n_pairs: number of null pairs measured.
        capped: True when the ceiling bound `epsilon`.
    """

    epsilon: float
    baseline_mean: float
    baseline_std: float
    sigma: float
    ceiling: float
    n_pairs: int
    capped: bool = False
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "epsilon": self.epsilon,
            "baseline_coherence_mean": self.baseline_mean,
            "baseline_coherence_std": self.baseline_std,
            "sigma": self.sigma,
            "ceiling": self.ceiling,
            "n_pairs": self.n_pairs,
            "capped": self.capped,
            **self.meta,
        }


def sagnac_delta(
    candidate: WaveState, axiom: WaveState, normalize: bool = True
) -> float:
    """Phase mismatch in [0, 2].

    Args:
        candidate: proposed wave.
        axiom: reference wave from memory.
        normalize: divide by both norms. Default True. Set False only
            when both states are already settled and the caller wants
            the raw form.

    Returns:
        `1 - Re<cand, axiom> / (||cand|| ||axiom||)`. Values near 0
        mean constructive phase; values above the threshold mean
        destructive phase.
    """
    if candidate.cfg.dim != axiom.cfg.dim:
        raise ValueError("candidate and axiom dimension mismatch")
    dot = np.vdot(candidate.psi, axiom.psi)
    if normalize:
        denom = candidate.norm * axiom.norm + _EPS
        dot = dot / denom
    return float(1.0 - np.real(dot))


def calibrate_epsilon(
    waves: Sequence[WaveState],
    sigma: float = 3.0,
    ceiling: float = 0.35,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
    n_pairs: int = 2000,
) -> VetoCalibration:
    """Calibrate the threshold from the empirical chance coherence.

    Chance coherence is measured by pairing DISTINCT waves drawn from
    the same family that will be evaluated. Using the wrong family is
    exactly the C2 defect, so the caller must pass its evaluation
    family here.

    Args:
        waves: at least two states from the evaluation wave family.
        sigma: standard deviations added to the baseline mean.
        ceiling: hard cap on the returned epsilon.
        rng: explicit generator, or `seed`.
        n_pairs: null pairs to sample.

    Returns:
        A VetoCalibration. `capped` is True when the ceiling bound it.
    """
    waves = list(waves)
    if len(waves) < 2:
        raise ValueError("calibration needs at least two reference waves")
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    gen = rng if rng is not None else np.random.default_rng(seed)
    n = len(waves)
    deltas = np.empty(int(n_pairs), dtype=np.float64)
    for i in range(int(n_pairs)):
        a = int(gen.integers(0, n))
        b = int(gen.integers(0, n - 1))
        if b >= a:
            b += 1
        deltas[i] = sagnac_delta(waves[a], waves[b])

    coh = 1.0 - deltas                      # chance coherence
    mean = float(np.mean(coh))
    std = float(np.std(coh))
    raw = mean + sigma * std
    capped = raw > ceiling
    epsilon = float(min(raw, ceiling))

    return VetoCalibration(
        epsilon=epsilon,
        baseline_mean=mean,
        baseline_std=std,
        sigma=float(sigma),
        ceiling=float(ceiling),
        n_pairs=int(n_pairs),
        capped=capped,
        meta={
            "delta_mean": float(np.mean(deltas)),
            "delta_p95": float(np.percentile(deltas, 95)),
            "is_complex_family": bool(np.any(np.abs(np.imag(waves[0].psi)) > 1e-9)),
        },
    )


def perturb_with_coherence(
    state: WaveState,
    target_coherence: float,
    rng: np.random.Generator,
) -> WaveState:
    """Return a unit-norm wave whose coherence with `state` is EXACTLY ~target.

    Construction
    ------------
    Let `v` be the unit wave, `n` a unit direction ORTHOGONAL to `v`
    (obtained by projecting `v` out of a random draw), and

        pert = v + eps * n

    Then `|<v, n>| = 0`, `||pert||^2 = 1 + eps^2`, and

        cos(v, pert) = 1 / sqrt(1 + eps^2)

    Inverting that relation gives the EXACT scale:

        eps = sqrt(1 - target^2) / target

    The FIRST-ORDER form `eps = sqrt(2 * (1 - target))` is a Taylor
    expansion of this, valid only as `target -> 1`. It is measured to
    break down badly away from that limit:

        target 0.999 -> achieved 0.999  (error 3e-6)
        target 0.900 -> achieved 0.911  (error 1.1e-2)
        target 0.500 -> achieved 0.708  (error 2.1e-1)  <- 42% off

    The orthogonal projection matters. Without it, `Re<v, n>` is a random
    `O(1/sqrt(D))` quantity, so the achieved coherence carries a
    dimension-dependent wobble on top of the approximation error.

    Dimension invariance
    --------------------
    `eps` is the TOTAL perturbation norm and `n` is unit, so the
    construction holds `target` fixed at every `D`. A FIXED per-component
    scale does not: a scale of 0.002 gives a relative perturbation of
    `0.002 * sqrt(2D)` -- 0.064 at D=512 but 0.72 at D=65536. The same
    literal is therefore a clean gold pair in CI and a destroyed one at
    production dimension. The retired repository hit exactly this class
    of error when a threshold calibrated at one scale was reused at
    another.
    """
    if not 0.0 < target_coherence <= 1.0:
        raise ValueError("target_coherence must be in (0, 1]")
    cfg = state.cfg
    eps = float(np.sqrt(max(1.0 - target_coherence**2, 0.0)) / target_coherence)

    direction = (
        rng.standard_normal(cfg.dim) + 1j * rng.standard_normal(cfg.dim)
    )
    # Project `state.psi` out so <psi, direction> == 0 exactly.
    direction = direction - state.psi * np.vdot(state.psi, direction)
    nrm = float(np.linalg.norm(direction))
    if nrm < _EPS:
        raise ValueError("degenerate random direction; redraw with another rng")
    direction = direction / nrm

    return WaveState.from_complex(state.psi + eps * direction, cfg)


def veto(candidate: WaveState, axiom: WaveState, calibration: VetoCalibration) -> bool:
    """True when the candidate is REJECTED (dark port fires).

    Rejection means the phase mismatch exceeds the calibrated threshold.
    """
    return sagnac_delta(candidate, axiom) > calibration.epsilon


def confidence(candidate: WaveState, axiom: WaveState) -> float:
    """Sagnac confidence `1 - delta`, clamped to [0, 1].

    Reported alongside every dispatched action. Confidence is not
    permission: the veto and this score are deterministic.
    """
    return float(np.clip(1.0 - sagnac_delta(candidate, axiom), 0.0, 1.0))


def separation(
    gold_pairs: Sequence, bad_pairs: Sequence, calibration: VetoCalibration
) -> dict:
    """Measure veto separation on labelled pair sets.

    Args:
        gold_pairs: sequence of `(candidate, axiom)` pairs that should
            PASS (constructive phase).
        bad_pairs: sequence of `(candidate, axiom)` pairs that should
            be REJECTED.
        calibration: the calibrated threshold.

    Returns:
        Dict with `true_veto` (P(veto | bad)), `false_veto`
        (P(veto | gold)), and `separation` = true_veto - false_veto.
        Separation near 0 means the gate cannot discriminate.
    """
    def rate(pairs):
        pairs = list(pairs)
        if not pairs:
            return float("nan")
        fired = sum(1 for c, a in pairs if veto(c, a, calibration))
        return fired / len(pairs)

    true_veto = rate(bad_pairs)
    false_veto = rate(gold_pairs)
    return {
        "true_veto": true_veto,
        "false_veto": false_veto,
        "separation": (true_veto - false_veto)
        if np.isfinite(true_veto) and np.isfinite(false_veto)
        else float("nan"),
        "n_gold": len(list(gold_pairs)),
        "n_bad": len(list(bad_pairs)),
        "epsilon": calibration.epsilon,
    }
