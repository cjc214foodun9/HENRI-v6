"""Phase 4 digital-twin contracts: blockwise coherence, lock budget, gate split.

Three owner directives, each pinned by a test that can fail.

1. `crosstalk` and `needs_retraction` must work at production cartridge
   size. The old 512-row guard made the retraction trigger unevaluable.
2. The Kuramoto step budget must scale with dimension. A fixed 200 was
   measured to reach r = 0.166 at D = 65536 where the derived 3697 reaches
   r = 0.985.
3. The 50 us limit is TELEMETRY_ONLY and the digital twin gate is 50 ms.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from henri import gram
from henri.config import WaveConfig
from henri.kuramoto import lock_steps_for_dim


def _book(m: int, d: int, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((m, d)) + 1j * rng.standard_normal((m, d))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _dense_reference(x: np.ndarray) -> tuple:
    g = np.abs(x @ x.conj().T)
    np.fill_diagonal(g, 0.0)
    k = x.shape[0]
    return float(np.sum(g ** 2) / (k * (k - 1))), float(g.max())


# ------------------------------------------------------- 1. blockwise Gram
def test_blockwise_matches_dense_reference_exactly():
    for m, d in ((16, 256), (64, 512), (97, 512)):
        x = _book(m, d)
        ref_mean, ref_max = _dense_reference(x)
        assert gram.crosstalk(x) == pytest.approx(ref_mean, rel=1e-12, abs=0.0)
        assert gram.max_crosstalk(x) == pytest.approx(ref_max, rel=1e-12, abs=0.0)


def test_coherence_runs_above_the_old_512_guard():
    """The defect: M >= 1024 raised MemoryError from crosstalk()."""
    x = _book(1024, 2048)
    assert gram.max_crosstalk(x) > 0.0
    assert gram.crosstalk(x) > 0.0
    assert isinstance(gram.needs_retraction(x), bool)

    # stiefel_retract must still work at the same size and drop c_max.
    # M <= D is required: a Gram of rank <= D cannot orthonormalize more
    # rows than the dimension, so the retraction correctly drops the rest.
    retracted = gram.stiefel_retract(x)
    assert retracted.shape == x.shape
    assert gram.max_crosstalk(retracted) < gram.max_crosstalk(x)


def test_gram_matrix_still_refuses_above_the_byte_budget():
    """The guard moved from 512 rows to a byte budget. It must still fire."""
    saved = gram.GRAM_BYTES_LIMIT
    try:
        gram.GRAM_BYTES_LIMIT = 1024
        with pytest.raises(MemoryError):
            gram.gram_matrix(_book(16, 64))
    finally:
        gram.GRAM_BYTES_LIMIT = saved
    # ...and must not fire below it.
    assert gram.gram_matrix(_book(16, 64)).shape == (16, 16)


def test_dense_side_limit_refuses_wave_dimension_square():
    """A D x D object stays refused; a dual M x M does not.

    The limit is patched down so the test does not allocate 4 GiB to prove
    that a 4 GiB allocation is refused.
    """
    saved = gram.DENSE_SIDE_LIMIT
    try:
        gram.DENSE_SIDE_LIMIT = 64
        with pytest.raises(MemoryError):
            gram.assert_no_dense_dxd(np.zeros((65, 65), dtype=np.complex128))
        gram.assert_no_dense_dxd(np.zeros((64, 64), dtype=np.complex128))
    finally:
        gram.DENSE_SIDE_LIMIT = saved
    assert gram.DENSE_SIDE_LIMIT >= 4096, (
        "the limit must admit production cartridge sizes (M=4096)")


# ----------------------------------------------------- 2. lock-step budget
def test_lock_budget_scales_with_log_dimension():
    small = lock_steps_for_dim(512, coupling=6.0, dt=1e-3)
    large = lock_steps_for_dim(65536, coupling=6.0, dt=1e-3)
    assert large > small
    assert large == 3697, large
    # The formula is 2 ln(D) / K / dt, so doubling ln not the dimension.
    expect = int(np.ceil(2.0 * np.log(65536) / 6.0 / 1e-3))
    assert large == expect


def test_pipeline_uses_the_derived_budget_when_none():
    """The pipeline must not fall back to a fixed constant."""
    from henri.pipeline import make_codebook, run_pipeline
    from henri.sagnac import VetoCalibration
    from henri.state import WaveState

    cfg = WaveConfig(dim=512)
    book = make_codebook(cfg, 16, seed=1)
    ingress = WaveState.from_complex(book[0], cfg)
    axiom = WaveState.from_complex(book[0], cfg)
    calib = VetoCalibration(epsilon=0.0431, baseline_mean=0.0, baseline_std=0.0,
                            sigma=0.0, ceiling=0.0431, n_pairs=0)
    res = run_pipeline(cfg, ingress, book, axiom, calib, relax_steps=0, seed=2)
    assert res.stages["lock_steps_used"] == lock_steps_for_dim(512, 6.0, 1e-3)

    pinned = run_pipeline(cfg, ingress, book, axiom, calib, relax_steps=0,
                          lock_steps=7, seed=2)
    assert pinned.stages["lock_steps_used"] == 7


# --------------------------------------------------------- 3. gate split
def test_optical_target_is_telemetry_and_twin_gate_is_50ms():
    import run_gact4_latency as h

    assert h.OPTICAL_CONTRACT_US == pytest.approx(50.0)
    assert h.DIGITAL_TWIN_CONTRACT_US == pytest.approx(50_000.0)
    assert h.CONTRACT_US == h.DIGITAL_TWIN_CONTRACT_US
    assert h.CONTRACT_US / h.OPTICAL_CONTRACT_US == pytest.approx(1000.0)


def test_measured_5090_number_passes_the_twin_gate_and_fails_the_optical_one():
    """Pin the recorded 215.24 us p99 against both contracts.

    This is the bifurcation in one assertion: the same measurement FAILS
    the optical target and PASSES the digital twin gate. A test that can
    only see one of those is the defect the owner directive removed.
    """
    import run_gact4_latency as h

    p99_us = 215.236097574234
    assert p99_us > h.OPTICAL_CONTRACT_US
    assert p99_us <= h.DIGITAL_TWIN_CONTRACT_US
