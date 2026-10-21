"""Gate: viscoelastic history residency. The budget must bind, and fail.

Every test here can fail. The negative controls are the point: an eviction
that never fires and a rank that never loses anything would both pass
vacuously, which is the defect class this repository has caught twice.
"""

from __future__ import annotations

import numpy as np
import pytest

from henri import history_budget as hb


def test_measured_l2_anchor_is_96_mib():
    """The rented RTX 5090 reported 96 MiB. The analysis said 128."""
    assert hb.L2_BYTES_MEASURED == 96 * 1024**2
    assert hb.L2_BYTES_MEASURED < 128 * 1024**2


def test_complex64_full_history_does_not_fit_b256():
    """The headline defect: B=256 of full complex64 history overflows L2."""
    per = hb.worker_history_bytes(65536)
    assert per == 2_097_152                      # 2.00 MiB per worker
    need = per * 256
    assert need == 536_870_912                   # 512 MiB
    assert need > hb.L2_BYTES_MEASURED
    assert need / hb.L2_BYTES_MEASURED == pytest.approx(5.333, abs=1e-3)


def test_capacity_after_pinned_projector_is_exactly_32():
    """Arithmetic, not a remembered constant.

    96 MiB - 32.00 MiB = 64.00 MiB usable; 64 / 2.00 = exactly 32 workers.
    The pinned projector is `65536 * 256 * 2 = 33,554,432 B = 32 MiB`
    exactly, so the directive's `B_active <= 32` is EXACTLY saturated and
    leaves zero L2 slack for a codebook. 32 is therefore an upper bound,
    not a working setpoint.
    """
    assert hb.RESERVE_PINNED_PROJECTOR == 32 * 1024**2
    assert hb.L2_BYTES_MEASURED - hb.RESERVE_PINNED_PROJECTOR == 64 * 1024**2
    cap = hb.budget_workers(hb.L2_BYTES_MEASURED, 65536,
                            reserve_bytes=hb.RESERVE_PINNED_PROJECTOR)
    assert cap == 32

    # Zero slack: the admitted set consumes 100% of L2 by construction.
    budget = hb.HistoryBudget(dim=65536)
    _, rep = budget.admit(np.ones(64))
    assert rep.bytes_used == hb.L2_BYTES_MEASURED - hb.RESERVE_PINNED_PROJECTOR
    assert rep.fraction_of_l2 == pytest.approx(2.0 / 3.0, abs=1e-9)


def test_admission_keeps_the_most_stressed_workers():
    budget = hb.HistoryBudget(dim=65536)
    stress = np.linspace(0.0, 1.0, 256)
    chosen, rep = budget.admit(stress)
    assert chosen.size == 32
    assert rep.clipped is True
    assert rep.requested == 256
    assert len(rep.evicted) == 224
    # The admitted set must be the highest-stress end of the ramp.
    ev = np.asarray(rep.evicted, dtype=np.int64)
    assert float(np.min(stress[chosen])) > float(np.max(stress[ev]))
    assert rep.bytes_used <= hb.L2_BYTES_MEASURED - hb.RESERVE_PINNED_PROJECTOR


def test_negative_control_flat_stress_still_evicts():
    """A control that cannot fire is not a control.

    With every worker equally stressed there is no stress signal to rank
    on, so the same clip must still happen. If admission depended on the
    stress values ordering rather than the budget, this would differ from
    the ramp case in size, and it must not.
    """
    budget = hb.HistoryBudget(dim=65536)
    _, ramp = budget.admit(np.linspace(0.0, 1.0, 256))
    _, flat = budget.admit(np.ones(256))
    assert flat.capacity == ramp.capacity == 32
    assert len(flat.evicted) == len(ramp.evicted) == 224
    # Determinism: flat stress resolves to the lowest indices.
    assert flat.admitted[0] == 0


def test_low_rank_policy_covers_full_batch():
    """r=64 coefficients fit every worker, but only with a shared basis."""
    budget = hb.HistoryBudget(dim=65536, policy="low_rank", r=64)
    assert budget.capacity >= 256
    stress = np.linspace(0.0, 1.0, 256)
    chosen, rep = budget.admit(stress)
    assert chosen.size == 256
    assert rep.clipped is False
    assert rep.evicted == ()


def test_negative_control_low_rank_basis_must_be_shared():
    """The viability of r=64 rests entirely on ONE shared basis.

    A single `(65536, 64)` complex64 basis is 32.00 MiB. Charging that per
    worker instead of once would need 8 GiB for 256 workers, which is 85x
    the L2 cache. The policy is only sound because the basis is shared, so
    the arithmetic must show both sides.
    """
    one_basis = hb.low_rank_history_bytes(65536, 64)
    assert one_basis == 33_556_480                    # 32.00 MiB, once
    assert one_basis < hb.L2_BYTES_MEASURED
    if_per_worker = one_basis * 256
    assert if_per_worker > 80 * hb.L2_BYTES_MEASURED  # ~8 GiB, impossible
    # Shared: coefficients only, which is what capacity charges.
    coeff = hb.P_DELAY * 64 * hb.COMPLEX64_BYTES
    assert coeff * 256 < hb.L2_BYTES_MEASURED


def test_rejects_negative_stress_and_bad_policy():
    budget = hb.HistoryBudget(dim=1024)
    with pytest.raises(ValueError, match="non-negative"):
        budget.admit(np.array([0.1, -0.1]))
    with pytest.raises(ValueError, match="finite"):
        budget.admit(np.array([0.1, np.nan]))
    with pytest.raises(ValueError, match="unknown policy"):
        hb.HistoryBudget(dim=1024, policy="compress")


def test_zero_capacity_reports_everything_evicted():
    """A budget that cannot hold one worker must say so, not crash."""
    budget = hb.HistoryBudget(dim=65536, l2_bytes=1024, reserve_bytes=0)
    chosen, rep = budget.admit(np.ones(8))
    assert chosen.size == 0
    assert rep.capacity == 0
    assert len(rep.evicted) == 8
    assert rep.bytes_used == 0
