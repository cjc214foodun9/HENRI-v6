#!/usr/bin/env python3
"""Torch parity for the four ported Omega hot paths.

Reports the device it actually used. Latency is printed only with that device
name attached. Exit 0 requires every agreement check to pass.

Tolerance derivation: torch runs complex64 (eps ~1.2e-7) against NumPy
complex128. A length-D reduction accumulates about sqrt(D)*eps, so at
D = 16384 the expected floor is ~1.6e-5. The vector bar is 1e-4 to leave one
order of margin above that floor. It is not tuned to a measurement.
"""
from __future__ import annotations

import math
import sys
import time

import numpy as np
import torch

from henri.koopman_sheaf import PhaseGatedKoopmanSheaf, retract_to_sphere
from henri.omega_torch import (
    KoopmanSheafT,
    hopfield_readout_t,
    sagnac_slot_errors_t,
    stiefel_retract_t,
)
from henri.triadic_sagnac import TriadicSagnacGate

FAILS = []


def check(name: str, ok: bool, detail: str) -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        FAILS.append(name)


def unit_np(d: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(d) + 1j * rng.standard_normal(d)
    return v / np.linalg.norm(v)


def rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12))


def main() -> int:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== TORCH PARITY on device={dev} torch={torch.__version__} "
          f"cuda_available={torch.cuda.is_available()} ===")
    if dev.type == "cuda":
        print(f"    gpu={torch.cuda.get_device_name(0)}")
    else:
        print("    NOTE: CPU device. No CUDA latency is claimed by this run.")

    rng = np.random.default_rng(2026)

    # 1. Hopfield readout
    d, M = 4096, 64
    cb = np.stack([unit_np(d, rng) for _ in range(M)])
    q = unit_np(d, rng)
    sims = np.real(cb @ np.conj(q))
    p = np.exp(26.10 * (sims - sims.max()))
    p /= p.sum()
    clean_np = p.astype(np.complex128) @ cb
    clean_np /= np.linalg.norm(clean_np)

    cb_t = torch.from_numpy(cb).to(dtype=torch.complex64, device=dev)
    q_t = torch.from_numpy(q).to(dtype=torch.complex64, device=dev)
    clean_t, margin_t, _ = hopfield_readout_t(q_t, cb_t)
    e = rel(clean_np, clean_t.cpu().numpy())
    check("hopfield-readout", e < 1e-4, f"rel_err={e:.2e} (bar 1e-4)")

    # 2. Stiefel retraction
    X = torch.from_numpy(np.stack([unit_np(d, rng) for _ in range(32)])).to(
        dtype=torch.complex64, device=dev
    )
    cmax_t = stiefel_retract_t(X)
    X64 = X.cpu().numpy().astype(np.complex128)
    G = 0.5 * (X64 @ X64.conj().T + (X64 @ X64.conj().T).conj().T) + 1e-10 * np.eye(32)
    w, V = np.linalg.eigh(G)
    Gio = (V * (1.0 / np.sqrt(np.clip(w, 1e-12, None)))) @ V.conj().T
    Xo = Gio @ X64
    Xo = Xo / np.linalg.norm(Xo, axis=-1, keepdims=True)
    Gp = np.abs(Xo @ Xo.conj().T)
    np.fill_diagonal(Gp, 0.0)
    cmax_np = float(Gp.max())
    check(
        "stiefel-retraction",
        abs(cmax_t - cmax_np) < 1e-6,
        f"c_max numpy={cmax_np:.2e} torch={cmax_t:.2e} |diff|={abs(cmax_t - cmax_np):.2e} (bar 1e-6)",
    )

    # 3. Koopman sheaf step + EDMD, sharing the same A/B
    d_model, rank = 4096, 32
    ref = PhaseGatedKoopmanSheaf(
        d_model=d_model, d_slot=d_model // 4, rank=rank, num_charts=2, seed=3
    )
    por = KoopmanSheafT(d_model=d_model, rank=rank, num_charts=2, seed=3, device=dev)
    por.A = torch.from_numpy(ref.stiefel_A).to(dtype=torch.complex64, device=dev)
    por.B = torch.from_numpy(ref.stiefel_B).to(dtype=torch.complex64, device=dev)

    psi = unit_np(d_model, rng)
    c = np.random.default_rng(5).standard_normal(rank) + 1j * np.random.default_rng(
        5
    ).standard_normal(rank)
    c /= np.linalg.norm(c)
    target = retract_to_sphere(psi + ref.stiefel_A @ c)

    psi_t = torch.from_numpy(psi).to(dtype=torch.complex64, device=dev)
    tgt_t = torch.from_numpy(target).to(dtype=torch.complex64, device=dev)

    err_np = ref.online_edmd_update(psi, target, chart_idx=0)
    err_t = por.online_edmd_update(psi_t, tgt_t, chart=0)
    step_np, _ = ref.step(psi, chart_idx=0)
    step_t = por.step(psi_t, chart=0).cpu().numpy()
    e = rel(step_np, step_t)
    check(
        "koopman-sheaf",
        e < 1e-4 and abs(err_np - err_t) < 1e-4,
        f"step rel_err={e:.2e} (bar 1e-4), residual numpy={err_np:.6f} torch={err_t:.6f}",
    )

    # 4. Triadic Sagnac slot errors
    gate = TriadicSagnacGate(d_model=d_model, num_slots=4, tau_veto=0.0431)
    a = unit_np(d_model, rng)
    b = retract_to_sphere(a + 0.3 * unit_np(d_model, rng))
    slots_np = gate.slot_homodyne_errors(a, b)
    slots_t = sagnac_slot_errors_t(
        torch.from_numpy(a).to(dtype=torch.complex64, device=dev),
        torch.from_numpy(b).to(dtype=torch.complex64, device=dev),
    ).cpu().numpy()
    e = float(np.abs(slots_np - slots_t).max())
    check("sagnac-slots", e < 1e-4, f"max_abs_diff={e:.2e} (bar 1e-4)")

    # Latency, device-attributed
    if dev.type == "cpu":
        for _ in range(2):
            por.step(psi_t)
        t0 = time.perf_counter()
        for _ in range(50):
            por.step(psi_t)
        dt = (time.perf_counter() - t0) / 50
        print(f"[INFO] koopman step D={d_model} r={rank} on CPU: {dt * 1e3:.2f} ms/call")
        print("[INFO] CUDA latency: NOT MEASURED (no CUDA device on this host)")

    print()
    if FAILS:
        print(f"=== {len(FAILS)} PARITY FAILURES: {FAILS} ===")
        return 1
    print("=== ALL TORCH PARITY CHECKS PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
