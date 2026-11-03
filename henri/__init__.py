"""HENRI v6 - wave-mechanical kernel, built from first principles.

This package implements the *validated* physics primitives only.
Every module states its evidence label. No module claims capability.
"""

from henri.config import WaveConfig, SLOT_NAMES
from henri.state import WaveState

# Omega (v7) modules. Additive: the legacy spine above is unchanged.
from henri.harmonic_lexicon import HarmonicLexicon
from henri.holographic_stack import HolographicCallStack
from henri.koopman_sheaf import PhaseGatedKoopmanSheaf
from henri.triadic_sagnac import TriadicSagnacGate
from henri.hhrg_memory import HHRGMemory
from henri.omega_resonator import AdiabaticResonator
from henri.egress_router import OmniModalEgressRouter
from henri.omega_pipeline import HenriOmegaPipeline

__all__ = [
    "WaveConfig",
    "WaveState",
    "SLOT_NAMES",
    "HarmonicLexicon",
    "HolographicCallStack",
    "PhaseGatedKoopmanSheaf",
    "TriadicSagnacGate",
    "HHRGMemory",
    "AdiabaticResonator",
    "OmniModalEgressRouter",
    "HenriOmegaPipeline",
]
__version__ = "0.8.0"

_LAZY_HOLARCH = {
    "HolarchConfig": "config",
    "HenriHolarch": "pipeline",
    "ZoneCMemory": "zone_c",
    "LeanSwarm": "zone_a",
    "PhotonicCore": "zone_b",
}


def __getattr__(name: str):
    """Serve the holarch names lazily.

    The base package is numpy-only. Importing torch for every consumer that
    only needs WaveState would be a silent cost, so the holarch is resolved
    on first access.
    """
    if name in _LAZY_HOLARCH:
        import importlib

        module = importlib.import_module("henri.holarch." + _LAZY_HOLARCH[name])
        return getattr(module, name)
    raise AttributeError(f"module 'henri' has no attribute {name!r}")


__all__ += list(_LAZY_HOLARCH)
