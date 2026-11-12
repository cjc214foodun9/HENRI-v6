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

import json
from dataclasses import dataclass
from typing import Dict, List

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


# ---------------------------------------------------------------------------
# Directive F4: CUDA driver visibility diagnostic
# ---------------------------------------------------------------------------
def probe_nvidia_driver_state() -> Dict[str, object]:
    """Explain why a registry-detected NVIDIA adapter is not visible to torch.

    Non-destructive: it reads the driver API surface and the environment. It
    does not load a kernel, does not reset a device, and does not write to
    the driver. Every field is evidence, and the verdict is a reading of that
    evidence, not a guess.
    """
    import os
    import platform
    import shutil
    import subprocess
    from pathlib import Path as _Path

    report: Dict[str, object] = {
        "platform": platform.platform(),
        "torch_version": str(torch.__version__),
        "cuda_compiled_version": str(torch.version.cuda),
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "cuda_visible_devices_set": "CUDA_VISIBLE_DEVICES" in os.environ,
        "cuda_home": os.environ.get("CUDA_HOME", ""),
        "cuda_path_set": "CUDA_PATH" in os.environ,
    }

    smi = shutil.which("nvidia-smi")
    report["nvidia_smi_path"] = str(smi or "")
    report["nvidia_smi_reachable"] = bool(smi)
    devices: List[str] = []
    driver_version = ""
    smi_error = ""
    if smi:
        try:
            result = subprocess.run(
                [str(smi), "-L"], capture_output=True, text=True, timeout=8
            )
            if result.returncode == 0:
                devices = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            else:
                smi_error = (result.stderr or "").strip()[:240]
        except Exception as error:  # pragma: no cover - host dependent
            smi_error = f"{type(error).__name__}: {error}"
        try:
            result = subprocess.run(
                [str(smi), "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=8,
            )
            if result.returncode == 0:
                driver_version = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
        except Exception as error:  # pragma: no cover - host dependent
            smi_error = smi_error or f"{type(error).__name__}: {error}"
    report["nvidia_smi_devices"] = devices
    report["nvidia_smi_driver_version"] = driver_version
    report["nvidia_smi_error"] = smi_error

    candidates = [
        r"C:\Windows\System32\nvcuda.dll",
        r"C:\Windows\SysWOW64\nvcuda.dll",
        r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
    ]
    on_windows = platform.system().lower().startswith("win")
    if not on_windows:
        candidates = [
            "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
            "/usr/lib64/libcuda.so.1",
            "/usr/local/cuda/lib64/libcuda.so",
            "/usr/bin/nvidia-smi",
        ]
    present = {path: bool(_Path(path).exists()) for path in candidates}
    report["driver_libraries"] = present
    report["nvcuda_present"] = any(
        present.get(path, False) for path in candidates if "nvcuda" in path.lower()
    )
    report["libcuda_present"] = any(
        present.get(path, False) for path in candidates if "libcuda" in path.lower()
    )
    report["driver_files_found"] = int(sum(1 for value in present.values() if value))
    report["on_windows"] = bool(on_windows)

    reasons: List[str] = []
    if not report["nvidia_smi_reachable"]:
        reasons.append("nvidia-smi is not on PATH")
    elif not devices:
        reasons.append("nvidia-smi ran but listed no device")
    if on_windows and not report["nvcuda_present"]:
        reasons.append("nvcuda.dll was not found in System32 or SysWOW64")
    if (not on_windows) and not report["libcuda_present"]:
        reasons.append("libcuda.so was not found in the standard loader paths")
    if report["cuda_visible_devices_set"] and not report["cuda_visible_devices"].strip():
        reasons.append("CUDA_VISIBLE_DEVICES is set to an empty value")
    if not torch.cuda.is_available() and not reasons:
        reasons.append(
            "the driver is present but torch reports no device: the process likely "
            "cannot reach the driver API (session 0 isolation, sandbox device access, "
            "or a disabled adapter in Device Manager)"
        )
    report["root_cause"] = "; ".join(reasons) if reasons else "CUDA device present and visible"
    report["verdict"] = "VISIBLE" if torch.cuda.is_available() else "BLOCKED"
    # Directive Z5: the PnP state is the missing evidence.  A GPU that the
    # registry enumerates but nvidia-smi does not list is a driver problem.
    try:
        report["pnp"] = probe_pnp_display_state()
    except Exception as error:  # pragma: no cover - a probe must never raise
        report["pnp"] = {"queried": False, "reason": f"{type(error).__name__}: {error}"}
    report["evidence"] = "OBSERVED"
    return report

# ---------------------------------------------------------------------------
# Directive Z5: non-destructive Plug-and-Play state of the display adapters
# ---------------------------------------------------------------------------
DISPLAY_CLASS_GUID = "{4d36e968-e325-11ce-bfc1-08002be10318}"
"""The Windows display-adapter setup class.  A GPU enumerated here but absent
from nvidia-smi is a PnP/driver problem, not a torch problem."""

_PNP_SCRIPT = (
    "Get-CimInstance Win32_PnPEntity | "
    "Where-Object { $_.PNPClass -eq 'Display' -or $_.Service -eq 'nvlddmkm' } | "
    "Select-Object Name,DeviceID,Status,ConfigManagerErrorCode,ConfigManagerUserConfig,"
    "Service,PNPClass | ConvertTo-Json -Compress"
)

_REGISTRY_SCRIPT = (
    "Get-ChildItem 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Class\\"
    + DISPLAY_CLASS_GUID
    + "' -ErrorAction SilentlyContinue | ForEach-Object { "
    + "[pscustomobject]@{ Key = $_.PSChildName; "
    + "DriverDesc = (Get-ItemProperty $_.PSPath -ErrorAction SilentlyContinue).DriverDesc } } | "
    + "ConvertTo-Json -Compress"
)


def probe_pnp_display_state() -> Dict[str, object]:
    """Read the PnP status of every display adapter, without changing it.

    ConfigManagerErrorCode is the Device Manager problem code: 0 = working,
    10 = cannot start, 12 = not enough resources, 22 = disabled, 28 = no
    driver installed, 31 = cannot load driver, 43 = stopped with an error.
    The query is read-only.  A non-Windows host or a blocked PowerShell
    returns an empty report with the reason, never a fabricated state.
    """
    import platform
    import subprocess

    report: Dict[str, object] = {
        "platform": platform.platform(),
        "display_class_guid": DISPLAY_CLASS_GUID,
        "queried": False,
        "devices": [],
        "class_keys": [],
        "reason": "",
    }
    if not platform.system().lower().startswith("win"):
        report["reason"] = "PnP query is Windows-only"
        return report
    shell = "powershell.exe"
    try:
        completed = subprocess.run(
            [shell, "-NoProfile", "-NonInteractive", "-Command", _PNP_SCRIPT],
            capture_output=True,
            text=True,
            timeout=25,
        )
    except Exception as error:  # pragma: no cover - host dependent
        report["reason"] = f"{type(error).__name__}: {error}"
        return report
    report["queried"] = True
    report["exit_code"] = int(completed.returncode)
    text = (completed.stdout or "").strip()
    devices: List[Dict[str, object]] = []
    if text:
        try:
            payload = json.loads(text)
            rows = payload if isinstance(payload, list) else [payload]
            for row in rows:
                code = row.get("ConfigManagerErrorCode")
                devices.append(
                    {
                        "name": str(row.get("Name", "")),
                        "device_id": str(row.get("DeviceID", "")),
                        "status": str(row.get("Status", "")),
                        "service": str(row.get("Service", "") or ""),
                        "pnp_class": str(row.get("PNPClass", "") or ""),
                        "config_manager_error_code": int(code) if code is not None else None,
                        "problem": _PNP_CODE_MEANING.get(
                            int(code) if code is not None else -1, "unknown"
                        ),
                    }
                )
        except Exception as error:  # pragma: no cover - malformed output
            report["reason"] = f"unparsed PnP output: {type(error).__name__}"
    report["devices"] = devices
    try:
        completed = subprocess.run(
            [shell, "-NoProfile", "-NonInteractive", "-Command", _REGISTRY_SCRIPT],
            capture_output=True,
            text=True,
            timeout=25,
        )
        text = (completed.stdout or "").strip()
        if text:
            payload = json.loads(text)
            rows = payload if isinstance(payload, list) else [payload]
            report["class_keys"] = [
                {
                    "key": str(row.get("Key", "")),
                    "driver_desc": str(row.get("DriverDesc", "") or ""),
                }
                for row in rows
            ]
    except Exception:  # pragma: no cover - registry read is optional evidence
        report["class_keys"] = []
    nvidia = [row for row in devices if "NVIDIA" in str(row["name"]).upper()
              or "10DE" in str(row["device_id"]).upper()]
    report["nvidia_devices"] = nvidia
    report["nvidia_problem_codes"] = [
        row["config_manager_error_code"] for row in nvidia
    ]
    ghost = [
        row for row in report["class_keys"]
        if "NVIDIA" in str(row.get("driver_desc", "")).upper()
    ]
    report["ghost_nvidia_driver_keys"] = ghost
    if not devices:
        report["reason"] = "no display adapter enumerated"
    elif not nvidia and ghost:
        report["reason"] = (
            "the NVIDIA driver class key is registered for "
            + ", ".join(str(row.get("driver_desc", "")) for row in ghost)
            + " but no NVIDIA PnP display device is enumerated: the adapter is "
            "absent from the bus, disabled in firmware, or was removed after a "
            "driver install, so nvidia-smi lists no device and torch cannot reach CUDA"
        )
    elif not nvidia:
        report["reason"] = "no NVIDIA adapter enumerated"
    elif all(row["config_manager_error_code"] == 0 for row in nvidia):
        report["reason"] = "NVIDIA adapter enumerated with problem code 0"
    else:
        report["reason"] = "NVIDIA adapter enumerated with a non-zero problem code"
    return report


_PNP_CODE_MEANING = {
    -1: "no code",
    0: "working",
    1: "not configured",
    3: "driver corrupt",
    10: "cannot start",
    12: "not enough free resources",
    18: "reinstall the drivers",
    22: "disabled",
    24: "device not present",
    28: "no driver installed",
    31: "cannot load driver",
    43: "driver stopped with an error",
}

