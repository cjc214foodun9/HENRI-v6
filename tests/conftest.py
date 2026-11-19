"""Shared test fixtures.

Every fixture is deterministic. No test draws entropy without an
explicit seed, because the retired repository's worst defect class was
metrics that could not be reproduced across processes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from henri.config import WaveConfig
from henri.gram import stiefel_retract
from henri.koopman import KoopmanOperator
from henri.sagnac import calibrate_epsilon
from henri.state import WaveState

DIM = 512
"""CI dimension. Small enough for seconds, same structure as production."""


def pytest_configure(config):
    """Keep pytest's temporary root inside the workspace (Directive K4).

    On the Windows DSH sandbox the system temp directory is created with
    mode 0o700 and cannot be enumerated by the test process, so pytest's
    default basetemp can fail before a single test runs. The fallback is a
    workspace-local directory: it is a real directory with the workspace's
    own ACL, it is ignored by git, and pytest's normal basetemp handling
    still owns it. The workaround is logged once so a receipt never claims
    the default path was used.
    """
    if getattr(config.option, "basetemp", None):
        return
    candidate = Path.cwd() / ".pytest_tmp"
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        probe = candidate / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        config.option.basetemp = str(candidate.resolve())
        config._dsh_basetemp_note = str(candidate.resolve())
    except (PermissionError, OSError) as error:
        config._dsh_basetemp_note = f"workspace basetemp unavailable: {error}"


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


_MMLU_REL = Path("data/official_benchmarks/canonical/mmlu/mmlu.csv")


def _is_missing_gitignored_mmlu(exc: BaseException) -> bool:
    """True only when gitignored mmlu.csv is absent on a clean checkout."""
    return (
        isinstance(exc, FileNotFoundError)
        and "language corpus missing:" in str(exc)
        and not (Path.cwd() / _MMLU_REL).exists()
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_setup(item):
    """Record explicit provenance skip when gitignored MMLU CSV is absent."""
    outcome = yield
    excinfo = outcome.excinfo
    if excinfo is not None and _is_missing_gitignored_mmlu(excinfo[1]):
        outcome.force_exception(
            pytest.skip.Exception("BLOCKED_MISSING_GITIGNORED_MMLU_CSV")
        )


@pytest.hookimpl(hookwrapper=True)
def pytest_pyfunc_call(pyfuncitem):
    """Record explicit provenance skip when gitignored MMLU CSV is absent."""
    outcome = yield
    excinfo = outcome.excinfo
    if excinfo is not None and _is_missing_gitignored_mmlu(excinfo[1]):
        outcome.force_exception(
            pytest.skip.Exception("BLOCKED_MISSING_GITIGNORED_MMLU_CSV")
        )


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
