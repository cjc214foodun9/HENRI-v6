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
__version__ = "0.7.0"
