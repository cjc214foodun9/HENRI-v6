"""Pinned statistic form for the Sagnac veto, per wave family.

Evidence label: DERIVED for the algebra, OBSERVED for every number in
`MEASURED`. The pin is enforced: an unpinned request raises.

THE TWO FORMS

    form_phase(c, a)         = 1 - mean_d cos(angle(c_d * conj(a_d)))
    form_inner_product(c, a) = 1 - Re<c, a> / (|c| |a|)

On a unit-modulus pair they are identical: both reduce to
`1 - (1/D) sum_d cos(dtheta_d)`, because `|c_d| = |a_d| = 1/sqrt(D)`.
Measured agreement `3.331e-16` (receipt: tau_veto_transfer_curve, D=65536).

THE DEFECT THIS MODULE EXISTS TO PREVENT

`form_phase` averages over ALL D dimensions with equal weight. A dimension
where `c_d == 0` has `angle(0) == 0`, hence `cos == 1`, hence contributes
ZERO stress. The mean therefore divides real stress by D while summing
stress over the active dimensions only:

    form_phase = (n_active / D) * (1 - mean_{active} cos(dtheta))

Constraint C6 mandates zero-filled unused slots. A one-slot state is
therefore measured 4x too leniently, and the veto crossing moves from
`sqrt(2*tau) = 0.2936` to `sqrt(2*tau*D/n_active) = 0.5872`.

`form_inner_product` is invariant to this. Its weights are
`w_d = |c_d||a_d| / sum_j |c_j||a_j|`, which sum to 1 over the support and
give exactly-zero dimensions weight 0. The dilution factor is 1 for every
magnitude profile.

MEASURED, C6-style state (slot 0 populated, slots 1-3 exactly zero, D=65536):

    eps      form_phase   form_inner   ratio
    0.1500   0.002797     0.011187     4.00
    0.2936   0.010543     0.042171     4.00
    0.4000   0.019227     0.076906     4.00
    0.5872   0.039638     0.158550     4.00

At `eps = 0.40` the phase form reads 0.019 (PASS) while the inner-product
form reads 0.077 (VETO). That is a real decision divergence, not a value
difference.

WHY THE TRANSFER-CURVE PROBE DID NOT SEE IT

All seven families in `probe_tau_veto_transfer_curve.py` are DENSE. Every
dimension carries a defined phase, so `n_active == D` and the two forms
cross within `0.0002` in eps. The `sensor` divergence of `1.231e-01` is
a curve MAXIMUM at large eps, not a crossing difference. A probe whose
families cannot be sparse cannot detect a sparsity defect.

CONSEQUENCE, STATED AS A SPEC AMENDMENT

The register proposed pinning `form_phase` as authoritative because tau
was calibrated against it. That calibration is a unit-modulus result and
does not transfer to sparse states. The invariant statistic is
`form_inner_product`. `form_phase` may be used only where the state is
declared dense: `n_active == D`.

BACKWARD COMPATIBILITY

`DENSE_FORM` remains the gate default so existing receipts reproduce.
The gate now REPORTS the dilution factor it applied. Silence was the
defect; a reported dilution is a measurable one.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

TAU_SPEC = 0.0431
"""Hard ceiling from the Phase 3 spec. Not the search-veto 0.35."""

DENSE_FORM = "phase"
INVARIANT_FORM = "inner_product"
VALID_FORMS = (DENSE_FORM, INVARIANT_FORM)

EVIDENCE_RECEIPT = "receipts/tau_veto_transfer_curve_full_D.json"
"""Probe that produced `MEASURED`. Its digest identifies a RUN, not a
content hash: it embeds interpreter paths and wall-clock. Never cite the
digest as a stable artifact."""

MEASURED: Dict[str, float] = {
    # family -> max |form_phase - form_inner_product| over the eps sweep.
    # D = 65536, eps in [0, 1], 12 trials per point.
    "unit_modulus": 3.331e-16,
    "qfhrr_token": 5.551e-16,
    "qfhrr_family": 4.441e-16,
    "stiefel": 9.114e-03,
    "learned": 1.096e-02,
    "sensor": 1.231e-01,
    "gaussian": 9.136e-03,
}

DENSE_AT_MEASUREMENT = frozenset(MEASURED)
"""Every measured family had full support at measurement time. A family
listed here can still be sparse at call time: C6 zeroes slots per input.
The pin is therefore evaluated on the STATE, not on the family label."""


class UnpinnedFormError(ValueError):
    """Raised when a statistic is requested that the state cannot support."""


class UnknownFamilyError(ValueError):
    """Raised for a family with no measured divergence. Fails closed."""


def active_fraction(cand: np.ndarray) -> float:
    """Fraction of dimensions with non-zero candidate magnitude."""
    c = np.asarray(cand)
    if c.size == 0:
        raise ValueError("empty candidate")
    return float(np.count_nonzero(np.abs(c) > 0.0)) / float(c.size)


def dilution_factor(cand: np.ndarray) -> float:
    """How far `form_phase` under-reports stress. 1.0 means undiluted.

    Exactly `D / n_active`. A dimension with `c_d == 0` reads
    `cos(angle(0)) == 1`, so it adds zero stress to the numerator while
    adding one to the denominator.
    """
    f = active_fraction(cand)
    if f <= 0.0:
        raise ValueError("candidate is all zeros")
    return 1.0 / f


def form_phase(cand: np.ndarray, axiom: np.ndarray) -> float:
    """Spec form: `1 - mean_d cos(angle(c_d * conj(a_d)))`."""
    return float(1.0 - np.mean(np.cos(np.angle(
        np.asarray(cand) * np.conj(np.asarray(axiom))))))


def form_inner_product(cand: np.ndarray, axiom: np.ndarray) -> float:
    """Magnitude-weighted form: `1 - Re<c, a> / (|c| |a|)`."""
    c, a = np.asarray(cand), np.asarray(axiom)
    return float(1.0 - np.real(np.vdot(c, a))
                 / (np.linalg.norm(c) * np.linalg.norm(a)))


def measured_divergence(family: str) -> float:
    """Recorded form divergence for `family`. Raises when unmeasured."""
    try:
        return float(MEASURED[family])
    except KeyError:
        raise UnknownFamilyError(
            f"family {family!r} has no measured form divergence. Measure it "
            "before pinning; an unmeasured family cannot be certified."
        ) from None


def is_decision_decisive(family: str, tau: float = TAU_SPEC) -> bool:
    """True when the recorded divergence exceeds tau at the curve maximum.

    This is a VALUE criterion. It is NOT evidence that the veto DECISION
    differs: the crossing eps can still agree. `decision_flip_band` is the
    quantity that answers the decision question.
    """
    return measured_divergence(family) > tau


def require_form(
    statistic: Optional[str],
    cand: Optional[np.ndarray] = None,
    family: Optional[str] = None,
) -> str:
    """Return the statistic to use, or raise. Fails closed.

    Rules, in order:
      1. `None` selects `DENSE_FORM` (backward compatible).
      2. An unknown statistic name raises.
      3. `DENSE_FORM` on a state with `n_active < D` raises, because the
         reading would be diluted by a known factor. Pass the candidate to
         enable this check; omitting it is treated as dense.
      4. A `family` label must be measured before it can be pinned.
    """
    form = DENSE_FORM if statistic is None else str(statistic)
    if form not in VALID_FORMS:
        raise UnpinnedFormError(
            f"unknown statistic {form!r}; valid: {VALID_FORMS}")
    if family is not None:
        measured_divergence(family)  # raises for an unmeasured family
    if form == DENSE_FORM and cand is not None:
        d = dilution_factor(cand)
        if d > 1.0:
            raise UnpinnedFormError(
                f"form_phase on a state with active fraction "
                f"{1.0 / d:.6f} under-reports stress by {d:.4f}x. Use "
                f"{INVARIANT_FORM!r}, or densify the state first.")
    return form


def decision_flip_band(
    cand_dense: np.ndarray,
    n_active: int,
    tau: float = TAU_SPEC,
) -> Dict[str, float]:
    """Closed-form eps band where the two forms disagree about the veto.

    For a pure phase perturbation of scale `eps` on `n_active` dimensions
    out of `D`:

        form_phase  ~ (n_active / D) * eps^2 / 2
        form_inner  ~ eps^2 / 2

    so the crossings are `sqrt(2*tau*D/n_active)` and `sqrt(2*tau)`.
    Returned in radians of per-dimension phase corruption.
    """
    d = int(np.asarray(cand_dense).size)
    if not 0 < n_active <= d:
        raise ValueError("n_active must be in (0, D]")
    ratio = d / float(n_active)
    return {
        "D": float(d), "n_active": float(n_active), "dilution": ratio,
        "eps_cross_phase": float(np.sqrt(2.0 * tau * ratio)),
        "eps_cross_inner": float(np.sqrt(2.0 * tau)),
        "flip_band_width": float(np.sqrt(2.0 * tau * ratio)
                                 - np.sqrt(2.0 * tau)),
    }
