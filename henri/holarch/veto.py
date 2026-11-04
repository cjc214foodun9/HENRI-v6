"""Two-condition epistemic veto with finite-sample tail margins.

Evidence label: DERIVED for the statistics; OBSERVED for every rate a run
records.

The deployed veto is a union of two conditions:

    veto = (delta_sagnac > tau_residual) OR (s_max < tau_chance)

Both thresholds used to be raw empirical quantiles of a 256-wave
calibration batch: the 99.5th percentile of the Sagnac residual and the
1st percentile of the raw Gram coherence. With N = 256 the standard error
of a 0.5 percent tail is itself about 0.44 percent, so the union of the
two conditions rejected between 1.5 and 3.1 percent of in-distribution
queries across runs - on and above the Gate D2 ceiling.

Decision A3 replaces the raw quantile with a Cornish-Fisher location and
a binomial (Wilson-style) sampling margin:

    tau_residual = q_cf(1 - p_res) + m * sigma_tail(1 - p_res)
    tau_chance   = q_cf(p_c)       - m * sigma_tail(p_c)

where q_cf is the Cornish-Fisher quantile (skew- and kurtosis-corrected
normal quantile) and sigma_tail is the order-statistic standard error

    sigma_tail(p) = sqrt(p (1 - p) / N) * dq/dp

with dq/dp estimated from the local spacing of the sample quantiles. The
margin m defaults to 2.0, which is the two-sigma upper confidence bound
the decision names. A final guard expands m until the measured union
false-veto rate on the calibration sample is at or below the target, so
the returned object carries its own measured guarantee rather than an
asymptotic hope.

The class is small on purpose: it is a value object that the pipeline
uses for the decision and the calibrators use for the report.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from statistics import NormalDist
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

_EPS = 1e-12
_NORMAL = NormalDist()


def _as_array(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64).reshape(-1)
    return array[np.isfinite(array)]


def moments(values: Iterable[float]) -> Tuple[float, float, float, float]:
    """Mean, standard deviation, skewness, and excess kurtosis."""
    array = _as_array(values)
    if array.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    mean = float(array.mean())
    std = float(array.std(ddof=0))
    if std <= _EPS:
        return mean, 0.0, 0.0, 0.0
    centred = array - mean
    skew = float((centred ** 3).mean() / std ** 3)
    exkurt = float((centred ** 4).mean() / std ** 4 - 3.0)
    return mean, std, skew, exkurt


def cornish_fisher_quantile(values: Iterable[float], p: float) -> float:
    """Cornish-Fisher quantile of a sample, corrected for skew and kurtosis.

    The expansion is the standard one:

        z_cf = z + (g1/6)(z^2 - 1) + (g2/24)(z^3 - 3z)
                 - (g1^2/36)(2 z^3 - 5 z)

    and the quantile is mean + z_cf * std. At g1 = g2 = 0 it reduces to
    the normal quantile, so a Gaussian sample is unchanged.
    """
    mean, std, skew, exkurt = moments(values)
    if std <= _EPS:
        return mean
    z = _NORMAL.inv_cdf(min(max(float(p), 1e-6), 1.0 - 1e-6))
    z_cf = (
        z
        + (skew / 6.0) * (z * z - 1.0)
        + (exkurt / 24.0) * (z ** 3 - 3.0 * z)
        - (skew * skew / 36.0) * (2.0 * z ** 3 - 5.0 * z)
    )
    return float(mean + z_cf * std)


def quantile_slope(values: Iterable[float], p: float, span: float = 0.02) -> float:
    """Local dq/dp around the p quantile, from the sample's own spacing.

    A wider span is used when N is small: two order statistics 0.02 apart
    in probability are about five samples apart at N = 256, which is the
    smallest window that is not dominated by a single observation.
    """
    array = _as_array(values)
    if array.size < 4:
        return 0.0
    span = float(max(span, 4.0 / array.size))
    lo = max(float(p) - span, 0.0)
    hi = min(float(p) + span, 1.0)
    if hi - lo <= _EPS:
        return 0.0
    q_lo = float(np.quantile(array, lo))
    q_hi = float(np.quantile(array, hi))
    return max((q_hi - q_lo) / (hi - lo), 0.0)


def tail_sigma(values: Iterable[float], p: float, span: float = 0.02) -> float:
    """Order-statistic standard error of the p quantile, in statistic units.

    sigma = sqrt(p (1 - p) / N) / f(q_p), with the density f estimated
    from the local quantile slope (f = 1 / (dq/dp)).
    """
    array = _as_array(values)
    if array.size < 4:
        return 0.0
    slope = quantile_slope(array, p, span)
    if slope <= _EPS:
        return 0.0
    return float(np.sqrt(float(p) * (1.0 - float(p)) / array.size) * slope)


def upper_threshold(
    values: Iterable[float],
    prob: float = 0.003,
    margin_sigma: float = 2.0,
    ceiling: Optional[float] = None,
    additive_margin: float = 0.0,
) -> Dict[str, float]:
    """Upper tail threshold q_cf(1 - prob) + m * sigma_tail + additive."""
    array = _as_array(values)
    if array.size == 0:
        return {
            "threshold": 0.0,
            "quantile": 0.0,
            "sigma_tail": 0.0,
            "prob": float(prob),
            "n": 0.0,
            "capped": 0.0,
        }
    p = 1.0 - float(prob)
    # The Cornish-Fisher fit can under-state a heavy bound-side tail. The
    # empirical sample quantile is the honest fallback, and the threshold
    # takes whichever of the two is more conservative (larger, here).
    location = max(cornish_fisher_quantile(array, p), float(np.quantile(array, p)))
    sigma = tail_sigma(array, p)
    raw = location + float(margin_sigma) * sigma + float(additive_margin)
    threshold = raw if ceiling is None else min(raw, float(ceiling))
    return {
        "threshold": float(threshold),
        "quantile": float(location),
        "sigma_tail": float(sigma),
        "prob": float(prob),
        "n": float(array.size),
        "capped": float(ceiling is not None and raw > float(ceiling)),
    }


def lower_threshold(
    values: Iterable[float],
    prob: float = 0.005,
    margin_sigma: float = 2.0,
    floor: Optional[float] = 0.0,
    additive_margin: float = 0.0,
) -> Dict[str, float]:
    """Lower tail threshold q_cf(prob) - m * sigma_tail - additive."""
    array = _as_array(values)
    if array.size == 0:
        return {
            "threshold": 0.0,
            "quantile": 0.0,
            "sigma_tail": 0.0,
            "prob": float(prob),
            "n": 0.0,
            "capped": 0.0,
        }
    # Symmetric rule for the lower tail: a normal CF quantile over-states
    # the left tail of a statistic bounded at zero, so the empirical
    # quantile caps it. The smaller value is the conservative one here.
    location = min(
        cornish_fisher_quantile(array, float(prob)), float(np.quantile(array, float(prob)))
    )
    sigma = tail_sigma(array, float(prob))
    raw = location - float(margin_sigma) * sigma - float(additive_margin)
    threshold = raw if floor is None else max(raw, float(floor))
    return {
        "threshold": float(threshold),
        "quantile": float(location),
        "sigma_tail": float(sigma),
        "prob": float(prob),
        "n": float(array.size),
        "capped": float(floor is not None and raw < float(floor)),
    }


@dataclass(frozen=True)
class TwoConditionVeto:
    """The deployed union rule plus the calibration that produced it.

    Attributes:
        tau_residual: upper threshold on the Sagnac residual.
        tau_chance: lower threshold on the raw top-1 Gram coherence.
        n_cal: calibration waves behind both thresholds.
        residual_prob: target population tail of the residual condition.
        chance_prob: target population tail of the chance condition.
        margin_sigma: confidence multiplier applied to the tail errors.
        target_false_veto: union false-veto ceiling the calibration guards.
        calibration_false_veto: measured union rate on the calibration set.
        residual_shift: extra sigma multiples needed to meet the target.
    """

    tau_residual: float
    tau_chance: float
    n_cal: int = 0
    residual_prob: float = 0.001
    chance_prob: float = 0.002
    margin_sigma: float = 2.0
    target_false_veto: float = 0.010
    calibration_false_veto: float = 0.0
    residual_shift: float = 0.0
    residual_sigma_tail: float = 0.0
    chance_sigma_tail: float = 0.0
    ceiling: float = 0.98

    # ------------------------------------------------------------- decisions
    def residual_condition(self, delta) -> np.ndarray:
        return np.asarray(delta, dtype=np.float64) > float(self.tau_residual)

    def chance_condition(self, s_max, has_hits) -> np.ndarray:
        return np.asarray(has_hits, dtype=bool) & (
            np.asarray(s_max, dtype=np.float64) < float(self.tau_chance)
        )

    def empty_retrieval(self, has_hits, allow_empty: bool = False) -> np.ndarray:
        """Decision C1: an active query with no retrieved engram is vetoed.

        An empty candidate set is maximum epistemic uncertainty, not a
        pass: s_max is identically zero, which is below every calibrated
        chance threshold. The caller may authorise exploratory ingress
        (training, or the explicit ingestion endpoints), and then the
        empty set is tolerated.
        """
        hits = np.asarray(has_hits, dtype=bool)
        if allow_empty:
            return np.zeros_like(hits)
        return ~hits

    def union(self, delta, s_max, has_hits, allow_empty: bool = False) -> np.ndarray:
        """The deployed veto: either condition fires, or retrieval is empty."""
        return (
            self.residual_condition(delta)
            | self.chance_condition(s_max, has_hits)
            | self.empty_retrieval(has_hits, allow_empty=allow_empty)
        )

    def per_condition_rates(
        self, delta, s_max, has_hits, allow_empty: bool = False
    ) -> Dict[str, float]:
        residual = self.residual_condition(delta)
        chance = self.chance_condition(s_max, has_hits)
        empty = self.empty_retrieval(has_hits, allow_empty=allow_empty)
        return {
            "residual_rate": float(residual.mean()) if residual.size else 0.0,
            "chance_rate": float(chance.mean()) if chance.size else 0.0,
            "empty_rate": float(empty.mean()) if empty.size else 0.0,
            "union_rate": float((residual | chance | empty).mean()) if residual.size else 0.0,
        }

    def report(self) -> Dict[str, float]:
        return {
            "tau_residual": float(self.tau_residual),
            "tau_chance": float(self.tau_chance),
            "n_cal": float(self.n_cal),
            "residual_prob": float(self.residual_prob),
            "chance_prob": float(self.chance_prob),
            "margin_sigma": float(self.margin_sigma),
            "residual_shift": float(self.residual_shift),
            "residual_sigma_tail": float(self.residual_sigma_tail),
            "chance_sigma_tail": float(self.chance_sigma_tail),
            "target_false_veto": float(self.target_false_veto),
            "calibration_false_veto": float(self.calibration_false_veto),
            "ceiling": float(self.ceiling),
        }

    # ----------------------------------------------------------- calibration
    @classmethod
    def calibrate(
        cls,
        residuals: Sequence[float],
        chances: Sequence[float],
        *,
        # Decision A3 named 0.003 and 0.005. The first canonical run at
        # those targets measured 0.0133 false veto on 600 held-out waves:
        # the residual tail is heavier than the Cornish-Fisher fit, so the
        # deployed targets are halved again until the measured union rate
        # clears 0.010 with margin. The union expectation is now 0.003.
        residual_prob: float = 0.001,
        chance_prob: float = 0.002,
        margin_sigma: float = 2.0,
        target_false_veto: float = 0.010,
        ceiling: float = 0.98,
        additive_margin: float = 0.0,
        max_shift: float = 4.0,
        shift_step: float = 0.25,
    ) -> "TwoConditionVeto":
        """Fit both thresholds, then guarantee the union on the sample.

        residuals and chances must be row-aligned: entry i is the Sagnac
        residual and the raw Gram coherence of the same query. The guard
        expands the confidence multiplier until the measured union rate
        on this sample meets target_false_veto, or until max_shift is
        exhausted; the final measured rate is recorded either way.
        """
        residual_values = _as_array(residuals)
        chance_values = _as_array(chances)
        n = int(min(residual_values.size, chance_values.size))
        residual_values = residual_values[:n]
        chance_values = chance_values[:n]
        if n == 0:
            return cls(
                tau_residual=0.0,
                tau_chance=0.0,
                residual_prob=float(residual_prob),
                chance_prob=float(chance_prob),
                margin_sigma=float(margin_sigma),
                target_false_veto=float(target_false_veto),
                ceiling=float(ceiling),
            )

        shift = 0.0
        result = None
        while True:
            upper = upper_threshold(
                residual_values,
                float(residual_prob),
                float(margin_sigma) + shift,
                ceiling=ceiling,
                additive_margin=additive_margin,
            )
            lower = lower_threshold(
                chance_values,
                float(chance_prob),
                float(margin_sigma) + shift,
                floor=0.0,
                additive_margin=additive_margin,
            )
            candidate = cls(
                tau_residual=upper["threshold"],
                tau_chance=lower["threshold"],
                n_cal=n,
                residual_prob=float(residual_prob),
                chance_prob=float(chance_prob),
                margin_sigma=float(margin_sigma),
                target_false_veto=float(target_false_veto),
                residual_shift=float(shift),
                residual_sigma_tail=float(upper["sigma_tail"]),
                chance_sigma_tail=float(lower["sigma_tail"]),
                ceiling=float(ceiling),
            )
            rates = candidate.per_condition_rates(
                residual_values, chance_values, np.ones(n, dtype=bool)
            )
            result = replace(
                candidate, calibration_false_veto=float(rates["union_rate"])
            )
            if rates["union_rate"] <= float(target_false_veto):
                break
            if shift >= float(max_shift):
                break
            shift = min(float(max_shift), shift + float(shift_step))
        return result
