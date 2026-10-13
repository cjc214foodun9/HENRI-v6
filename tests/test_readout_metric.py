"""Tests for the scale-aware entropy metric.

Pins the defect recorded in `henri.readout.normalized_entropy` and the
fix in `entropy_at_beta`.

The defect: `normalized_entropy` uses a UNIT-temperature softmax. For
coherence-scale scores in [-1, 1], `s - max(s)` lies in [-2, 0], so
`exp` of it lies in [0.135, 1]. With M=32 the largest attainable
probability is ~0.19, so the reported entropy cannot fall below ~0.94
however decisive the surface is. Measured at D=4096: a decisive cell
(margin 0.87) and a flat cell (margin 0.005) differ by 0.006.

`entropy_at_beta` multiplies by beta before the softmax, so the two
regimes separate by ~0.99.
"""

from __future__ import annotations

import numpy as np
import pytest

from henri.readout import entropy_at_beta, normalized_entropy


def _decisive_scores() -> np.ndarray:
    """One dominant score, then a tail. Coherence-scale amplitude."""
    s = np.zeros(32)
    s[0] = 0.90
    s[1:] = 0.02
    return s


def _flat_scores() -> np.ndarray:
    """All scores equal: a genuine plateau."""
    return np.zeros(32)


def test_flat_scores_give_entropy_one_in_both_metrics():
    flat = _flat_scores()
    assert normalized_entropy(flat) == pytest.approx(1.0, abs=1e-9)
    assert entropy_at_beta(flat, 26.1) == pytest.approx(1.0, abs=1e-9)


def test_legacy_metric_is_scale_blind():
    """The recorded defect: the legacy metric cannot separate the two.

    Both a decisive surface and a flat one read above 0.99. That is why
    the plateau report showed entropy 1.000000 next to a decisive cell
    at 0.9938: the metric had almost no dynamic range.
    """
    decisive = normalized_entropy(_decisive_scores())
    assert decisive > 0.99, f"expected scale-blindness, got {decisive}"
    assert decisive - 1.0 > -0.02


def test_beta_metric_separates_decisive_from_flat():
    """The fix: at beta=26.1 the decisive surface collapses to ~0."""
    beta_flat = entropy_at_beta(_flat_scores(), 26.1)
    beta_decisive = entropy_at_beta(_decisive_scores(), 26.1)
    assert beta_flat == pytest.approx(1.0, abs=1e-9)
    assert beta_decisive < 0.01, f"expected collapse, got {beta_decisive}"
    assert beta_flat - beta_decisive > 0.95


def test_beta_one_matches_legacy_metric():
    """At beta=1 the two functions must agree exactly.

    This pins that `entropy_at_beta` is a strict generalization, not a
    different statistic.
    """
    for scores in (_decisive_scores(), _flat_scores(),
                   np.linspace(-1.0, 1.0, 32)):
        assert entropy_at_beta(scores, 1.0) == pytest.approx(
            normalized_entropy(scores), abs=1e-12
        )


def test_monotone_in_beta():
    """Raising beta cannot increase the entropy of a fixed surface."""
    s = _decisive_scores()
    values = [entropy_at_beta(s, b) for b in (1.0, 8.0, 26.1, 100.0)]
    for a, b in zip(values, values[1:]):
        assert b <= a + 1e-12, f"not monotone: {values}"


def test_rejects_nonpositive_beta():
    with pytest.raises(ValueError):
        entropy_at_beta(_flat_scores(), 0.0)
    with pytest.raises(ValueError):
        entropy_at_beta(_flat_scores(), -1.0)


def test_single_score_returns_one():
    """Degenerate size: entropy is defined as 1.0 by the module."""
    assert entropy_at_beta(np.array([0.5]), 26.1) == 1.0
    assert normalized_entropy(np.array([0.5])) == 1.0
