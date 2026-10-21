"""G-ACT4 real-time cycle latency gate, hardware-marked.

Two tests, deliberately different in kind:

1. `test_torch_veto_ops_match_the_numpy_definition` runs on CPU. It
   validates the EXACT torch op chain the CUDA path uses against the
   repository's own statistic. No GPU needed, so it runs everywhere. The
   timed G-ACT4 path cannot execute on a CPU host, so without this test
   the CUDA algebra would first be exercised on the target machine -- an
   unvalidated code path arriving at the one place it cannot be debugged.

2. `test_gact4_full_cycle_within_contract` is marked `hardware` and SKIPS
   when no CUDA device is present. It never converts a missing
   prerequisite into a PASS.

Measured on this host (2026-10-12): no working CUDA device. See
docs/EVIDENCE_REGISTER.md section 11.4.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))


def test_torch_veto_ops_match_the_numpy_definition():
    """Pin the CUDA op chain to the repository statistic, on CPU."""
    torch = pytest.importorskip("torch")
    import numpy as np

    from henri.config import WaveConfig
    from henri.sagnac import sagnac_delta
    from henri.state import WaveState
    from run_gact4_latency import veto_torch

    rng = np.random.default_rng(20261012)
    for trial in range(3):
        q = rng.integers(0, 256, size=(2, 128))
        psi = np.exp(1j * 2 * np.pi * q[0] / 256.0) / np.sqrt(128)
        ax = np.exp(1j * 2 * np.pi * q[1] / 256.0) / np.sqrt(128)

        # The definition, computed from scratch.
        z = psi * np.conj(ax)
        ref_defn = float(1.0 - np.mean(np.cos(np.angle(z))))

        # The repository statistic. It takes WaveState, NOT raw arrays:
        # `sagnac_delta` reads `candidate.cfg.dim`. Passing arrays raised
        # AttributeError: 'numpy.ndarray' object has no attribute 'cfg'.
        # Caught by running this test, not by writing it.
        cfg = WaveConfig(dim=128)
        ref_prod = float(sagnac_delta(
            WaveState.from_complex(psi, cfg),
            WaveState.from_complex(ax, cfg),
        ))

        got = float(veto_torch(torch.from_numpy(psi), torch.from_numpy(ax)))

        assert abs(ref_defn - ref_prod) < 1e-12, (
            "trial %d: sagnac_delta deviates from the definition by %.3e"
            % (trial, abs(ref_defn - ref_prod)))
        assert abs(got - ref_defn) < 1e-12, (
            "trial %d: torch op chain deviates by %.3e"
            % (trial, abs(got - ref_defn)))


def _cuda_ready() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available()
                    and torch.cuda.device_count() > 0)
    except Exception:
        return False


@pytest.mark.hardware
def test_gact4_digital_twin_cycle_within_contract(tmp_path):
    """Skip with no CUDA device; assert the DIGITAL TWIN contract when there is.

    The 50 us optical target is asserted to be PRESENT and to be labelled
    TELEMETRY_ONLY. It is not asserted to pass, because it is not a gate.
    Bifurcation: a receipt that dropped the optical number would fail here.
    """
    if not _cuda_ready():
        pytest.skip("no working CUDA device; G-ACT4 BLOCKED by prerequisite")

    out = tmp_path / "gact4.json"
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "run_gact4_latency.py"),
         "--out", str(out)],
        capture_output=True, text=True, timeout=1200,
    )
    assert r.returncode == 0, "harness failed rc=%d\n%s" % (
        r.returncode, r.stderr[-2000:])

    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["status"] == "G-ACT4_DIGITAL_TWIN_PASS", rec.get("status")
    assert rec["full_cycle"]["p99_us"] <= rec["contract_us"]
    assert rec["contract_us"] == 50_000.0, (
        "the digital twin gate must be 50 ms, not the 50 us optical target")
    assert rec["device"]["name"], "device name must be recorded"

    tele = rec["optical_telemetry"]
    assert tele["status"] == "TELEMETRY_ONLY"
    assert tele["contract_us"] == 50.0
    assert tele["measured_cycle_p99_us"] == rec["full_cycle"]["p99_us"]
    assert isinstance(tele["would_pass_optical_contract"], bool)
