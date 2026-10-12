"""HENRI v6 - wave-mechanical kernel, built from first principles.

This package implements the *validated* physics primitives only.
Every module states its evidence label. No module claims capability.
"""

from henri.config import WaveConfig, SLOT_NAMES
from henri.state import WaveState

__all__ = ["WaveConfig", "WaveState", "SLOT_NAMES"]
__version__ = "0.6.0"
