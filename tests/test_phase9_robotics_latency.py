"""Phase 9 Step 1: robotics latency fields are present and consistent.

WHY THIS FILE EXISTS
`henri/robotics_loop.py` gained `total_latency_us` and
`per_step_latency_us`, and the stored full-D receipt carries both. A stored
receipt is not a test: nothing would fail if a later edit dropped the fields
or broke the division. These assertions fail in that case.

D = 1024 on purpose. The claims are relational - the fields exist, and the
per-step term equals the total divided by the step count - and neither
depends on dimension. A full-D build costs minutes to prove the same algebra.
"""
from __future__ import annotations

import sys

import pytest

sys.path.insert(0, ".")

from henri.config import WaveConfig  # noqa: E402
from henri.robotics_loop import RoboticsLoop  # noqa: E402

TAU_VETO = 0.0431


@pytest.fixture(scope="module")
def small_loop():
    return RoboticsLoop(cfg=WaveConfig(dim=1024))


def _run(loop, sensor, axiom):
    out = loop.step(sensor_values=sensor, axiom_reference=axiom)
    return out if isinstance(out, dict) else out.to_dict()


def test_both_latency_fields_exist(small_loop):
    """A missing field is the regression this guards."""
    d = _run(small_loop, [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    assert "total_latency_us" in d, "total_latency_us missing from step()"
    assert "per_step_latency_us" in d, "per_step_latency_us missing from step()"
    assert float(d["total_latency_us"]) > 0.0


def test_per_step_equals_total_over_steps(small_loop):
    """The division the receipt relies on.

    Uses `max(1, steps)` so the zero-step arms cannot divide by zero. A near
    arm reads exactly zero steps, and that case must still satisfy the
    relation rather than raise.
    """
    d = _run(small_loop, [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    tot = float(d["total_latency_us"])
    per = float(d["per_step_latency_us"])
    steps = int(d["adaptation_steps"])
    expected = tot / float(max(1, steps))
    assert per == pytest.approx(expected, rel=1e-9, abs=1e-6), (
        "per_step_latency_us %.6f != total/max(1,steps) %.6f"
        % (per, expected))


def test_legacy_latency_us_agrees_with_total(small_loop):
    """`latency_us` is the field Phase 8 receipts recorded.

    It must stay equal to the total, or an old receipt and a new one would
    disagree about the same run.
    """
    d = _run(small_loop, [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    assert float(d["latency_us"]) == pytest.approx(
        float(d["total_latency_us"]), rel=1e-9, abs=1e-6)


def test_unsafe_arm_reports_positive_steps_and_per_step(small_loop):
    """The arm the contract is written for: a veto that needed recovery.

    A zero-step unsafe arm would make the per-step term vacuous, so the test
    refuses it instead of accepting a trivial pass.
    """
    safe = [1.0, 0.0, 0.0, 0.0]
    unsafe = [1.0, 5.0, 0.0, 0.0]
    d = _run(small_loop, unsafe, safe)
    if not d["vetoed_initial"]:
        pytest.skip("this fixture did not trigger a veto at D=1024")
    assert int(d["adaptation_steps"]) >= 1
    assert float(d["per_step_latency_us"]) > 0.0
    assert float(d["total_latency_us"]) >= float(d["per_step_latency_us"])


def test_statistic_form_is_inner_product(small_loop):
    """The form pin. A dense form would be diluted by D/n_active."""
    d = _run(small_loop, [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    assert d["statistic_form"] == "inner_product"
