"""VRAM budget accounting for the 32 GiB Blackwell target.

Evidence label: DERIVED. The parameter bytes are counted exactly. The
activation bytes are measured by the autograd saved-tensor hook, which
records every tensor the backward pass keeps alive for one real batch.
The runtime reserve is a declared constant, not a measurement, and is
labelled as such in the report.

No GPU was available on the machine that wrote this file. The report
therefore states the device it measured on and never presents a CPU
measurement as a Blackwell result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn

from henri.holarch.config import VRAM_BUDGET_GB, HolarchConfig

GIB = 1024.0 ** 3
RUNTIME_RESERVE_GIB = 0.75
"""CUDA context, cuFFT plans, allocator slack. Declared, not measured."""


class _SavedTensorCounter:
    """Count live autograd saved tensors for one forward/backward."""

    def __init__(self) -> None:
        self.live = 0
        self.peak = 0
        self.total = 0

    def pack(self, tensor: torch.Tensor) -> torch.Tensor:
        size = tensor.numel() * tensor.element_size()
        self.live += size
        self.total += size
        self.peak = max(self.peak, self.live)
        return tensor

    def unpack(self, tensor: torch.Tensor) -> torch.Tensor:
        size = tensor.numel() * tensor.element_size()
        self.live = max(0, self.live - size)
        return tensor


def parameter_breakdown(model: nn.Module) -> Dict[str, int]:
    """Trainable parameters grouped by top-level module."""
    groups: Dict[str, int] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group = name.split(".", 1)[0]
        groups[group] = groups.get(group, 0) + int(parameter.numel())
    groups["total"] = int(sum(groups.values()))
    return groups


def _module_device(model: nn.Module) -> torch.device:
    """The device the model actually lives on, not the configured one."""
    try:
        return next(model.parameters()).device
    except StopIteration:  # pragma: no cover - a model with no parameters
        return torch.device("cpu")


def _peak_saved_bytes(model: nn.Module, batch_size: int, cfg: HolarchConfig) -> int:
    """Measure saved-tensor bytes for one real vision batch."""
    counter = _SavedTensorCounter()
    x = torch.zeros(int(batch_size), 1, 8, 8, device=_module_device(model))
    model.zero_grad(set_to_none=True)
    with torch.autograd.graph.saved_tensors_hooks(counter.pack, counter.unpack):
        output = model("vision", x, training=True, use_memory=False)
        loss = output.task_logits.sum()
        loss.backward()
    model.zero_grad(set_to_none=True)
    return int(counter.peak)


@torch.no_grad()
def _forward_peak_bytes(model: nn.Module, batch_size: int, cfg: HolarchConfig) -> int:
    """Estimate inference activation bytes: the largest single tensor set."""
    x = torch.zeros(int(batch_size), 1, 8, 8, device=_module_device(model))
    output = model("vision", x, training=False, use_memory=False)
    tensors = [
        output.psi_in,
        output.psi_fused,
        output.photonic.bright,
        output.photonic.dark,
        output.photonic.features,
        output.features,
        output.logits,
    ]
    return int(sum(t.numel() * t.element_size() for t in tensors))


def budget_report(
    model: nn.Module,
    batch_size: int,
    cfg: HolarchConfig,
    budget_gb: float = VRAM_BUDGET_GB,
    optimizer_bytes_per_param: float = 8.0,
) -> Dict[str, float]:
    """Return the estimated resident footprint for training and inference.

    Args:
        model: the holarch.
        batch_size: training batch size.
        cfg: the configuration, for memory dimensions.
        budget_gb: the device budget in GiB.
        optimizer_bytes_per_param: Adam moments (8 bytes for two fp32
            moments). Master weights and gradients are accounted
            separately at 4 bytes each.

    Returns:
        A dictionary of byte counts and the pass/fail verdict.
    """
    params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    bytes_params_fp32 = params * 4
    bytes_grads = params * 4
    bytes_optimizer = int(params * float(optimizer_bytes_per_param))
    saved = _peak_saved_bytes(model, batch_size, cfg)
    forward = _forward_peak_bytes(model, batch_size, cfg)
    memory = getattr(model, "memory", None)
    resident_index = 0
    if memory is not None:
        resident_index = int(memory.cache_capacity) * int(cfg.dim) * 8
        resident_index += 65536 * int(cfg.gram_dims) * 8
    runtime = int(RUNTIME_RESERVE_GIB * GIB)

    training = bytes_params_fp32 + bytes_grads + bytes_optimizer + saved + resident_index + runtime
    inference = bytes_params_fp32 // 2 + forward + resident_index + runtime
    total_gb = training / GIB
    cuda_peak = 0.0
    if torch.cuda.is_available():
        # Decision C3: when the model really ran on a CUDA device, record the
        # allocator's own peak instead of only the analytic accounting.
        try:
            cuda_peak = float(torch.cuda.max_memory_allocated() / GIB)
        except Exception:  # pragma: no cover - driver present but no context
            cuda_peak = 0.0
    return {
        "parameters": float(params),
        "parameter_bytes_fp32": float(bytes_params_fp32),
        "gradient_bytes": float(bytes_grads),
        "optimizer_bytes": float(bytes_optimizer),
        "activation_bytes_training": float(saved),
        "activation_bytes_inference": float(forward),
        "resident_index_bytes": float(resident_index),
        "runtime_reserve_bytes": float(runtime),
        "training_gib": total_gb,
        "inference_gib": inference / GIB,
        "budget_gib": float(budget_gb),
        "headroom_gib": float(budget_gb - total_gb),
        "fits": float(total_gb <= float(budget_gb)),
        "batch_size": float(batch_size),
        "method": "counted params + measured saved tensors + declared reserve",
        "device": cfg.resolved_device(),
        "measured_device": str(_module_device(model)),
        "cuda_peak_allocated_gib": cuda_peak,
    }


class BudgetExceeded(RuntimeError):
    """Raised when the configured model cannot fit the device budget."""


def assert_fits(report: Dict[str, float]) -> Dict[str, float]:
    """Raise unless the report fits. Returns the report unchanged."""
    if not report.get("fits", 0.0):
        raise BudgetExceeded(
            f"training footprint {report['training_gib']:.3f} GiB exceeds "
            f"the {report['budget_gib']:.1f} GiB budget"
        )
    return report
