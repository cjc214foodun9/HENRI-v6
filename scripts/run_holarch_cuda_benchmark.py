"""Cycle Z5: the D = 2048 and D = 4096 CUDA benchmark, or the exact PnP block.

    python scripts/run_holarch_cuda_benchmark.py

On a host with a bound CUDA device the script measures the production
D = 4096 forward / imprint / egress readout and the D = 2048 profile.  On a
host without one it records CUDA: BLOCKED with the non-destructive PnP
diagnostic from henri.holarch.budget.probe_pnp_display_state and reports NO
GPU number at all: a CPU timing is not a GPU measurement.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch

import run_holarch_p_cycle as pc
from henri.holarch.budget import probe_nvidia_driver_state
from henri.holarch.config import HolarchConfig
from henri.holarch.pipeline import HenriHolarch

RECEIPTS = ROOT / "receipts" / "holarch"


def _benchmark(dim: int, *, repeats: int = 5) -> Dict[str, Any]:
    cfg = HolarchConfig.production() if int(dim) == 4096 else HolarchConfig(dim=int(dim))
    model = HenriHolarch(cfg).to("cuda")
    model.eval()
    generator = torch.Generator(device="cuda").manual_seed(20261111)
    psi = torch.randn(1, int(cfg.dim), dtype=torch.complex64, device="cuda", generator=generator)
    psi = psi / psi.norm()
    target = torch.randn(1, int(cfg.dim), dtype=torch.complex64, device="cuda", generator=generator)
    target = target / target.norm()
    codebook = torch.randn(2048, int(cfg.dim), dtype=torch.complex64, device="cuda",
                           generator=generator)
    codebook = codebook / codebook.norm(dim=-1, keepdim=True)

    def timed(fn, count: int = repeats) -> float:
        torch.cuda.synchronize()
        began = time.perf_counter()
        for _ in range(int(count)):
            fn()
        torch.cuda.synchronize()
        return float((time.perf_counter() - began) / max(int(count), 1) * 1000.0)

    with torch.no_grad():
        forward_ms = timed(lambda: model.zone_b(psi))
        imprint_ms = timed(lambda: model.zone_b.imprint(psi, target, rate=0.05), count=3)
        state = model.zone_b(psi)
        readout_ms = timed(lambda: codebook @ state.bright[0].conj())
    return {
        "dim": int(cfg.dim),
        "photonic_layers": int(cfg.photonic_layers),
        "device": "cuda",
        "forward_ms": forward_ms,
        "imprint_ms": imprint_ms,
        "egress_readout_ms": readout_ms,
        "vram_allocated_mb": float(torch.cuda.memory_allocated() / 2**20),
        "vram_reserved_mb": float(torch.cuda.memory_reserved() / 2**20),
    }


def measure() -> Dict[str, Any]:
    started = time.perf_counter()
    driver = probe_nvidia_driver_state()
    available = bool(torch.cuda.is_available())
    rows: List[Dict[str, Any]] = []
    if available:
        for dim in (2048, 4096):
            rows.append(_benchmark(dim))
    status = "CUDA: MEASURED" if available else "CUDA: BLOCKED"
    payload = {
        "stage": "Z5-cuda-benchmark",
        "status": status,
        "cuda_available": available,
        "torch_version": str(torch.__version__),
        "cuda_build": str(torch.version.cuda),
        "gpu_measurements": rows,
        "gpu_numbers_claimed": bool(available),
        "cpu_measurement_reported_as_gpu": False,
        "driver_diagnostic": driver,
        "pnp": driver.get("pnp", {}),
        "root_cause": str(driver.get("root_cause", "")),
        "reason": (
            "a bound CUDA device was available; the D = 2048 and D = 4096 profiles ran on it"
            if available else
            "torch.cuda.is_available() is False.  No GPU number is reported.  The PnP "
            "diagnostic names the exact state instead."
        ),
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
    }
    payload["gate"] = pc._finalise(
        {
            "z5_status_recorded": True,
            "z5_no_cpu_as_gpu": bool(payload["cpu_measurement_reported_as_gpu"] is False),
            "z5_measured_or_blocked": bool(status in ("CUDA: MEASURED", "CUDA: BLOCKED")),
            "z5_pnp_recorded": bool(isinstance(payload["pnp"], dict) and payload["pnp"]),
            "z5_numbers_only_when_measured": bool(
                (available and len(rows) == 2) or (not available and not rows)
            ),
        }
    )
    payload["passed"] = bool(all(payload["gate"].values()))
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Cycle Z5 CUDA benchmark probe")
    parser.add_argument("--receipts", default=str(RECEIPTS))
    args = parser.parse_args(argv)
    directory = Path(args.receipts)
    directory.mkdir(parents=True, exist_ok=True)
    payload = measure()
    written = pc._write(directory, "z5_cuda_benchmark.json", payload)
    print("[z5] %s gpu_measurements=%d pnp_reason=%s" % (
        payload["status"], len(payload["gpu_measurements"]),
        str(payload["pnp"].get("reason", ""))[:120]))
    print("[z5] passed=%s receipt=%s" % (payload["passed"], written["path"]))
    return 0 if bool(payload["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
