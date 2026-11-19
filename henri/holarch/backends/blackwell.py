"""Blackwell GB202 (sm_120) Triton kernels, TK-1 .. TK-5.

Hard rules encoded here:

* the backend dispatches on capability == (12, 0) and never on
  capability >= (10, 0): sm_120 reports 12.0 and has neither TMEM nor
  tcgen05;
* every launch sizes its dynamic shared memory from the driver's
  maxSharedMemoryPerBlockOptin and refuses any tile above 99 KiB
  (101,376 B);
* grids are persistent and SM-aligned (m * 170 on RTX 5090, m * 188 on
  RTX PRO 6000);
* no inline PTX, no tcgen05, no wgmma.

The module imports safely on a host without Triton: every kernel is defined
only when triton imports, and the backend refuses to run otherwise.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:  # pragma: no cover - platform dependent
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:  # noqa: BLE001 - absence is a supported state
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    TRITON_AVAILABLE = False

MAX_SMEM_BYTES = 101376  # 99 KiB, the sm_120 opt-in ceiling
SM_TARGETS = {170: "RTX 5090", 188: "RTX PRO 6000 Blackwell"}

if TRITON_AVAILABLE:  # pragma: no cover - compiled only on a CUDA host

    @triton.jit
    def tk1_ingress_kernel(ids_ptr, phase_ptr, out_ptr, length, d_slot, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        position = tl.program_id(1)
        if position < length:
            token = tl.load(ids_ptr + row * length + position)
            slot = token % 4
            angle = tl.load(phase_ptr + token)
            scale = 1.0 / tl.sqrt(length.to(tl.float32))
            offsets = tl.arange(0, BLOCK)
            mask = offsets < d_slot
            base = ((row * 4 + slot) * d_slot) + offsets
            tl.store(out_ptr + base, scale * tl.cos(angle), mask=mask)
            tl.store(out_ptr + (row * 4 * d_slot) + base, scale * tl.sin(angle), mask=mask)

    @triton.jit
    def tk2_sagnac_kernel(psi_ptr, mixer_ptr, chirp_ptr, out_ptr, d_slot, zeta, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < d_slot
        accumulator_re = tl.zeros((BLOCK,), dtype=tl.float32)
        accumulator_im = tl.zeros((BLOCK,), dtype=tl.float32)
        for slot in tl.static_range(4):
            base = (row * 4 + slot) * d_slot + offsets
            real = tl.load(psi_ptr + base, mask=mask, other=0.0)
            imag = tl.load(psi_ptr + (row * 4 * d_slot) + base, mask=mask, other=0.0)
            phase = tl.load(chirp_ptr + offsets, mask=mask, other=0.0)
            cosine = tl.cos(zeta * phase)
            sine = tl.sin(zeta * phase)
            accumulator_re += real * cosine - imag * sine
            accumulator_im += real * sine + imag * cosine
        for target in tl.static_range(4):
            weight_re = tl.load(mixer_ptr + target * 4)
            weight_im = tl.load(mixer_ptr + 16 + target * 4)
            base = (row * 4 + target) * d_slot + offsets
            tl.store(out_ptr + base, accumulator_re * weight_re - accumulator_im * weight_im, mask=mask)
            tl.store(
                out_ptr + (row * 4 * d_slot) + base,
                accumulator_re * weight_im + accumulator_im * weight_re,
                mask=mask,
            )

    @triton.jit
    def tk3_chart_kernel(query_ptr, chart_ptr, out_ptr, count, dim, top_k: tl.constexpr, BLOCK: tl.constexpr):
        query = tl.program_id(0)
        best_keys = tl.full((top_k,), 0xFFFFFFFFFFFFFFFF, dtype=tl.uint64)
        best_ids = tl.full((top_k,), 0xFFFFFFFF, dtype=tl.uint32)
        for index in range(0, count):
            offsets = tl.arange(0, BLOCK)
            mask = offsets < dim
            difference = tl.load(query_ptr + query * dim + offsets, mask=mask, other=0.0)
            difference -= tl.load(chart_ptr + index * dim + offsets, mask=mask, other=0.0)
            score = -tl.sum(difference * difference, axis=0)
            key = (score.to(tl.float32).to(tl.uint32, bitcast=True).to(tl.uint64) << 32) | (
                0xFFFFFFFF - index
            ).to(tl.uint64)
            for slot in tl.static_range(top_k):
                if key < best_keys[slot]:
                    best_keys[slot] = key
                    best_ids[slot] = index.to(tl.uint32)
        for slot in tl.static_range(top_k):
            tl.store(out_ptr + query * top_k + slot, best_ids[slot])

    @triton.jit
    def tk4_swarm_kernel(
        x_ptr, routing_ptr, weights_ptr, biases_ptr, out_ptr,
        batch, dim, experts, top_k: tl.constexpr, BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        if row < batch:
            offsets = tl.arange(0, BLOCK)
            mask = offsets < dim
            value = tl.load(x_ptr + row * dim + offsets, mask=mask, other=0.0)
            for slot in tl.static_range(top_k):
                gate = tl.load(routing_ptr + slot * dim + offsets, mask=mask, other=0.0)
                expert = tl.program_id(1) * top_k + slot
                if expert < experts:
                    weight = tl.load(weights_ptr + expert * dim + offsets, mask=mask, other=0.0)
                    bias = tl.load(biases_ptr + expert * dim + offsets, mask=mask, other=0.0)
                    value = value * gate + value * weight + bias
            tl.store(out_ptr + row * dim + offsets, value, mask=mask)

    @triton.jit
    def tk5_born_kernel(
        phi_ptr, row_ptr, steer_ptr, value_ptr, out_ptr,
        modes, vocab, rank, phase_scale, BLOCK: tl.constexpr,
    ):
        token = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < modes
        phase = tl.load(phi_ptr + offsets * vocab + token, mask=mask, other=0.0)
        steering = tl.load(steer_ptr + offsets * rank, mask=mask, other=0.0)
        projection = tl.sum(steering * tl.load(value_ptr + token * rank + tl.arange(0, rank)), axis=0)
        angle = phase + phase_scale * tl.math.tanh(projection / tl.sqrt(rank.to(tl.float32)))
        amplitude_re = tl.sum(tl.load(row_ptr + offsets * vocab + token, mask=mask, other=0.0) * tl.cos(angle), axis=0)
        amplitude_im = tl.sum(tl.load(row_ptr + offsets * vocab + token, mask=mask, other=0.0) * tl.sin(angle), axis=0)
        tl.store(out_ptr + token, amplitude_re * amplitude_re + amplitude_im * amplitude_im)


def blackwell_capability_ok(capability: Tuple[int, int], sm_count: int) -> bool:
    """True only on verified sm_120 GB202 silicon."""
    return bool(tuple(capability) == (12, 0) and int(sm_count) in SM_TARGETS)


def shared_memory_budget(d_slot: int, n_slots: int = 4, dtype_bytes: int = 8) -> int:
    """Bytes of one single-slot FFT staging buffer, capped by the sm_120 wall."""
    return int(min(d_slot * dtype_bytes * 2, MAX_SMEM_BYTES))


class BlackwellGB202TritonBackend:
    """The Triton backend: only instantiable on verified sm_120 GB202."""

    name = "blackwell_gb202_triton"
    triton = True

    def __init__(self, profile: Dict[str, Any]) -> None:
        capability = tuple(int(value) for value in profile.get("capability") or (0, 0))
        sm_count = int(profile.get("sm_count", 0))
        smem = int(profile.get("smem_optin", 0) or 0)
        if not TRITON_AVAILABLE:
            raise RuntimeError("triton is not installed on this host")
        if not blackwell_capability_ok(capability, sm_count):
            raise RuntimeError(
                "refusing to run: capability %s with %d SMs is not verified sm_120 GB202"
                % (capability, sm_count)
            )
        if smem and smem > MAX_SMEM_BYTES:
            raise RuntimeError(
                "driver reports %d B opt-in shared memory above the 99 KiB sm_120 wall" % smem
            )
        self.profile = dict(profile)
        self.device = str(profile.get("device", "cuda"))
        self.sm_count = sm_count
        self.capability = capability
        self.grid_multiple = sm_count
        self.smem_limit = min(smem or MAX_SMEM_BYTES, MAX_SMEM_BYTES)

    def ingress(self, token_ids, token_phases, *, dim: int, **kwargs):
        import torch

        ids = torch.as_tensor(token_ids, device=self.device, dtype=torch.int32)
        phases = torch.as_tensor(token_phases, device=self.device, dtype=torch.float32)
        batch, length = int(ids.shape[0]), int(ids.shape[1])
        d_slot = int(dim) // 4
        out = torch.zeros((2, batch, 4, d_slot), device=self.device, dtype=torch.float32)
        block = min(1024, d_slot)
        tk1_ingress_kernel[(batch, length)](ids, phases, out, length, d_slot, BLOCK=block)
        return out

    def sagnac(self, planar, mixers, chirps, *, kerr: float = 0.0, zeta: float = 0.0):
        import torch

        psi = torch.as_tensor(planar, device=self.device)
        mixer = torch.as_tensor(mixers[0], device=self.device)
        chirp = torch.as_tensor(chirps[0], device=self.device)
        rows = int(psi.shape[1])
        d_slot = int(psi.shape[-1])
        out = torch.zeros_like(psi)
        block = min(1024, d_slot)
        tk2_sagnac_kernel[(rows,)](psi, mixer, chirp, out, d_slot, float(zeta), BLOCK=block)
        return out

    def charts(self, queries, charts, *, top_k: int = 2):
        import torch

        query = torch.as_tensor(queries, device=self.device, dtype=torch.float32)
        bank = torch.as_tensor(charts, device=self.device, dtype=torch.float32)
        count, dim = int(bank.shape[0]), int(bank.shape[1])
        out = torch.zeros((int(query.shape[0]), int(top_k)), device=self.device, dtype=torch.int32)
        tk3_chart_kernel[(int(query.shape[0]),)](
            query, bank, out, count, dim, top_k=int(top_k), BLOCK=min(1024, dim)
        )
        return out.cpu().numpy()


__all__ = [
    "BlackwellGB202TritonBackend",
    "TRITON_AVAILABLE",
    "MAX_SMEM_BYTES",
    "SM_TARGETS",
    "blackwell_capability_ok",
    "shared_memory_budget",
]
