"""Determinism of the Omega pipeline's thermostat noise.

Before this test, the thermostat drew from the process-global RNG, so two runs
with the same seed could differ.  These tests pin the corrected property:

* the pipeline never touches the global RNG, and
* ``reset_rng()`` replays the same trajectory.
"""

from __future__ import annotations

import numpy as np
import pytest

from henri.omega_pipeline import HenriOmegaPipeline

D_MODEL = 1024
RANK = 16
CHARTS = 2
TOKENS = ["python", "sort", "list"]


@pytest.fixture(scope="module")
def pipeline() -> HenriOmegaPipeline:
    # A tiny tau_veto makes the veto fire, so the thermostat loop — and its
    # noise — actually runs.  A test that never enters the loop proves nothing.
    return HenriOmegaPipeline(
        d_model=D_MODEL,
        rank=RANK,
        num_charts=CHARTS,
        tau_veto=1e-9,
        max_thermostat_steps=4,
        seed=11,
    )


def test_thermostat_never_uses_the_global_rng(pipeline, monkeypatch) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("the pipeline used the process-global RNG")

    monkeypatch.setattr(np.random, "standard_normal", forbidden)
    telemetry = pipeline.step(query_tokens=TOKENS)
    assert telemetry.psi_out.shape == (D_MODEL,)
    assert telemetry.thermostat_steps > 0, "the veto did not fire, so no noise was drawn"


def test_reset_rng_replays_the_same_trajectory(pipeline) -> None:
    pipeline.reset_rng()
    first = pipeline.step(query_tokens=TOKENS)
    pipeline.reset_rng()
    second = pipeline.step(query_tokens=TOKENS)
    assert first.thermostat_steps == second.thermostat_steps
    assert np.array_equal(first.psi_out, second.psi_out)


def test_two_fresh_pipelines_agree() -> None:
    def build() -> HenriOmegaPipeline:
        return HenriOmegaPipeline(
            d_model=D_MODEL,
            rank=RANK,
            num_charts=CHARTS,
            tau_veto=1e-9,
            max_thermostat_steps=4,
            seed=29,
        )

    a = build()
    b = build()
    result_a = a.step(query_tokens=TOKENS)
    result_b = b.step(query_tokens=TOKENS)
    assert np.array_equal(result_a.psi_out, result_b.psi_out)
    assert result_a.thermostat_steps == result_b.thermostat_steps
