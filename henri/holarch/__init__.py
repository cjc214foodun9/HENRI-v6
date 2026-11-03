"""HENRI holarch: the three-zone wave-mechanical machine-learning pipeline.

Zone A  lean continuous-learning expert swarm (ingress and action)
Zone B  BaTiO3 diffractive photonic kernel with Sagnac homodyne readout
Zone C  holographic TimescaleDB/NVMe agentic memory

The package is additive. It does not modify the validated v6 primitives in
henri.*; it composes them where they already hold.
"""

from __future__ import annotations

from henri.holarch.config import (
    DEFAULT_DIM,
    TASK_IDS,
    TASK_NAMES,
    VRAM_BUDGET_GB,
    HolarchConfig,
)
from henri.holarch.pipeline import HenriHolarch, HolarchOutput, WaveProjector
from henri.holarch.zone_a import FastWeightExpert, LeanSwarm, ReplayBuffer, SwarmOutput
from henri.holarch.zone_b import (
    DiffractivePlate,
    MicroHeaterArray,
    PhotonicCore,
    PhotonicState,
    curie_weiss_gain,
    sagnac_delta,
)
from henri.holarch.zone_c import (
    Engram,
    MemoryHit,
    MemoryRead,
    SQLiteEngramStore,
    TimescaleEngramStore,
    ZoneCMemory,
)

__all__ = [
    "DEFAULT_DIM",
    "TASK_IDS",
    "TASK_NAMES",
    "VRAM_BUDGET_GB",
    "HolarchConfig",
    "HenriHolarch",
    "HolarchOutput",
    "WaveProjector",
    "FastWeightExpert",
    "LeanSwarm",
    "ReplayBuffer",
    "SwarmOutput",
    "DiffractivePlate",
    "MicroHeaterArray",
    "PhotonicCore",
    "PhotonicState",
    "curie_weiss_gain",
    "sagnac_delta",
    "Engram",
    "MemoryHit",
    "MemoryRead",
    "SQLiteEngramStore",
    "TimescaleEngramStore",
    "ZoneCMemory",
]
