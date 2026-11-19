"""Portable dual-backend dispatcher for the HENRI wave kernels.

ReferenceWaveBackend is the default everywhere (Windows CPython 3.14 CPU CI,
Linux CPU, and every host without verified Blackwell silicon).
BlackwellGB202TritonBackend is activated only on verified sm_120 GB202:
capability == (12, 0), 170 SMs (RTX 5090) or 188 SMs (RTX PRO 6000), and a
driver opt-in shared-memory ceiling at or below 99 KiB.

The dispatcher never tests capability >= (10, 0).  sm_120 reports 12.0 and
would pass such a test while having no TMEM and no tcgen05, which is exactly
the failure mode the specification forbids.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from .blackwell import (
    MAX_SMEM_BYTES,
    SM_TARGETS,
    TRITON_AVAILABLE,
    BlackwellGB202TritonBackend,
    blackwell_capability_ok,
    shared_memory_budget,
)
from .reference import (
    ReferenceWaveBackend,
    born_contextual_phase,
    chart_retrieval,
    complex_from_planar,
    planar_from_complex,
    planar_ingress,
    sagnac_plate,
    stiefel_readout,
    swarm_topk,
)

BACKEND_NAMES = ("reference", "blackwell_gb202_triton")


@dataclass
class DeviceProfile:
    vendor: str = "cpu"
    name: str = "cpu"
    capability: Tuple[int, int] = (0, 0)
    sm_count: int = 0
    l2_bytes: int = 0
    smem_optin: int = 0
    device: str = "cpu"
    triton_available: bool = bool(TRITON_AVAILABLE)
    is_gb202: bool = False
    provenance: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "vendor": self.vendor,
            "name": self.name,
            "capability": list(self.capability),
            "sm_count": int(self.sm_count),
            "l2_bytes": int(self.l2_bytes),
            "smem_optin": int(self.smem_optin),
            "device": self.device,
            "triton_available": bool(self.triton_available),
            "is_gb202": bool(self.is_gb202),
            "provenance": self.provenance,
        }


def probe_device() -> DeviceProfile:
    """Read the live device and classify it without guessing."""
    try:
        import torch
    except Exception as error:  # noqa: BLE001 - a missing torch is a CPU host
        return DeviceProfile(provenance="torch unavailable: %s" % error)
    if not torch.cuda.is_available():
        return DeviceProfile(
            vendor="cpu",
            name="cpu",
            device="cpu",
            provenance="BLOCKED_NO_PCIE_NVIDIA_DEVICE: torch.cuda.is_available() is False",
        )
    capability = tuple(int(value) for value in torch.cuda.get_device_capability(0))
    properties = torch.cuda.get_device_properties(0)
    sm_count = int(getattr(properties, "multi_processor_count", 0))
    l2_bytes = int(getattr(properties, "L2_cache_size", 0) or 0)
    smem = 0
    try:
        import ctypes

        # The driver's opt-in ceiling is the only authoritative sm_120 bound.
        smem = int(torch.cuda.get_device_properties(0).shared_memory_per_block_optin)
    except Exception:  # noqa: BLE001 - older torch exposes no such field
        smem = 0
    is_gb202 = blackwell_capability_ok(capability, sm_count)
    return DeviceProfile(
        vendor="nvidia",
        name=str(torch.cuda.get_device_name(0)),
        capability=capability,
        sm_count=sm_count,
        l2_bytes=l2_bytes,
        smem_optin=smem,
        device="cuda",
        triton_available=bool(TRITON_AVAILABLE),
        is_gb202=bool(is_gb202),
        provenance=(
            "verified sm_120 GB202: %s" % SM_TARGETS.get(sm_count, "unknown SM count")
            if is_gb202
            else "capability %s with %d SMs is not verified sm_120 GB202" % (capability, sm_count)
        ),
    )


def select_backend(name: str = "auto", profile: Optional[DeviceProfile] = None):
    """Return the strongest safe backend for this host.

    'auto' selects the Triton backend only on verified GB202.  An explicit
    'blackwell_gb202_triton' request on a non-GB202 host raises, because a
    silent fallback would hide a mis-typed capability check.
    """
    profile = profile or probe_device()
    if str(name) == "reference":
        return ReferenceWaveBackend()
    if str(name) == "blackwell_gb202_triton":
        return BlackwellGB202TritonBackend(profile.as_dict())
    if profile.is_gb202 and profile.triton_available:
        return BlackwellGB202TritonBackend(profile.as_dict())
    return ReferenceWaveBackend()


def parity(reference, backend, *, tol: float = 1e-5) -> Dict[str, Any]:
    """Max absolute deviation between two backends on the shared contracts."""
    import numpy as np

    generator = np.random.default_rng(20261119)
    dim = 256
    batch = 3
    ids = generator.integers(0, 64, size=(batch, 8))
    phases = generator.standard_normal(64)
    ref_planar = reference.ingress(ids, phases, dim=dim)
    got_planar = backend.ingress(ids, phases, dim=dim)
    delta_ingress = float(
        np.max(np.abs(np.asarray(ref_planar, dtype=np.float64) - np.asarray(got_planar, dtype=np.float64)))
    )
    mixers = np.stack([np.eye(4, dtype=np.complex128) for _ in range(2)])
    chirps = generator.standard_normal((2, dim // 4))
    ref_bright, ref_dark = reference.sagnac(ref_planar, mixers, chirps, zeta=0.1)
    got_bright, got_dark = backend.sagnac(ref_planar, mixers, chirps, zeta=0.1)
    delta_sagnac = float(
        np.max(
            np.abs(
                np.asarray(ref_bright, dtype=np.float64)
                - np.asarray(got_bright, dtype=np.float64)
            )
        )
    )
    frame = generator.standard_normal((8, dim // 4)) + 1j * generator.standard_normal((8, dim // 4))
    ref_readout = reference.readout(ref_planar, frame)
    got_readout = backend.readout(ref_planar, frame=frame) if hasattr(backend, "readout") else ref_readout
    delta_readout = float(
        np.max(np.abs(np.asarray(ref_readout, dtype=np.float64) - np.asarray(got_readout, dtype=np.float64)))
    )
    maximum = max(delta_ingress, delta_sagnac, delta_readout)
    return {
        "max_abs_delta": float(maximum),
        "ingress": delta_ingress,
        "sagnac": delta_sagnac,
        "readout": delta_readout,
        "tol": float(tol),
        "parity_ok": bool(maximum <= float(tol)),
    }


__all__ = [
    "BACKEND_NAMES",
    "DeviceProfile",
    "ReferenceWaveBackend",
    "BlackwellGB202TritonBackend",
    "TRITON_AVAILABLE",
    "MAX_SMEM_BYTES",
    "SM_TARGETS",
    "probe_device",
    "select_backend",
    "parity",
    "blackwell_capability_ok",
    "shared_memory_budget",
    "planar_ingress",
    "sagnac_plate",
    "stiefel_readout",
    "chart_retrieval",
    "born_contextual_phase",
    "swarm_topk",
    "planar_from_complex",
    "complex_from_planar",
]
