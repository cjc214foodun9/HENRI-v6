"""Shared test fixtures.

Every fixture is deterministic. No test draws entropy without an
explicit seed, because the retired repository's worst defect class was
metrics that could not be reproduced across processes.
"""

from __future__ import annotations

import numpy as np
import pytest

from henri.config import WaveConfig
from henri.gram import stiefel_retract
from henri.koopman import KoopmanOperator
from henri.sagnac import calibrate_epsilon
from henri.state import WaveState

DIM = 512
"""CI dimension. Small enough for seconds, same structure as production."""


def pytest_addoption(parser):
    """Register `--run-slow`.

    The slow tier holds the full D = 65536 runs. They are excluded by
    default so the gate sweep stays fast, and they are never silently
    skipped in a report: the count is printed in the summary.
    """
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run tests marked `slow` (full-dimension runs)",
    )


def pytest_collection_modifyitems(config, items):
    """Skip `slow` tests unless `--run-slow` is passed."""
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(reason="slow tier; pass --run-slow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture
def cfg() -> WaveConfig:
    return WaveConfig.small(dim=DIM)


@pytest.fixture
def cfg_full() -> WaveConfig:
    return WaveConfig()


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20261012)


@pytest.fixture
def wave(cfg, rng) -> WaveState:
    return WaveState.random(cfg, rng=rng)


@pytest.fixture
def wave_real(cfg, rng) -> WaveState:
    """Real-valued wave family.

    The retired repo measured chance coherence 0.120168 here against
    0.009792 for the complex family -- 12.3x higher. Constraint C2.
    """
    return WaveState.real(cfg, rng=rng)


@pytest.fixture
def operator(cfg) -> KoopmanOperator:
    return KoopmanOperator.build(cfg, seed=7)


@pytest.fixture
def codebook(cfg) -> np.ndarray:
    rng = np.random.default_rng(11)
    raw = (
        rng.standard_normal((24, cfg.dim))
        + 1j * rng.standard_normal((24, cfg.dim))
    )
    return stiefel_retract(raw)


@pytest.fixture
def calibration(cfg, rng) -> object:
    refs = [WaveState.random(cfg, rng=rng) for _ in range(12)]
    return calibrate_epsilon(refs, sigma=3.0, ceiling=0.35, rng=rng)
