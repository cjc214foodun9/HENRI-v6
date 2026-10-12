"""Circular-convolution binding and unbinding.

Evidence label: DERIVED. This is standard VSA / Holographic Reduced
Representation algebra, applied to the complex phase vector.

Binding is circular convolution, evaluated in the Fourier domain:

    bind(a, b)   = IFFT( FFT(a) * FFT(b) )
    unbind(c, a) = IFFT( conj(FFT(a)) * FFT(c) )

Two properties matter and are both tested:

1. Binding preserves the unit norm up to float error, so a bound
   state stays on the sphere.
2. Unbinding recovers the bound factor to accuracy O(1/sqrt(D)) for
   independent random factors. The residual is the orthogonal
   crosstalk term, and it shrinks as the dimension grows.
"""

from __future__ import annotations

import numpy as np

from henri.state import WaveState


def _fft(v: np.ndarray) -> np.ndarray:
    return np.fft.fft(v)


def _ifft(v: np.ndarray) -> np.ndarray:
    return np.fft.ifft(v)


def bind(a: WaveState, b: WaveState) -> WaveState:
    """Bind two states by circular convolution. Result is unit norm."""
    _check(a, b)
    out = _ifft(_fft(a.psi) * _fft(b.psi))
    return WaveState.from_complex(out, a.cfg)


def unbind(bound: WaveState, key: WaveState) -> WaveState:
    """Recover the factor bound with `key`. Result is unit norm."""
    _check(bound, key)
    out = _ifft(np.conj(_fft(key.psi)) * _fft(bound.psi))
    return WaveState.from_complex(out, bound.cfg)


def bundle(states, cfg=None) -> WaveState:
    """Superpose states by summation, then renormalize.

    Bundling trades fidelity for capacity: the recovered factor carries
    crosstalk from the other bundle members.
    """
    states = list(states)
    if not states:
        raise ValueError("bundle needs at least one state")
    cfg = cfg or states[0].cfg
    acc = np.zeros(cfg.dim, dtype=np.complex128)
    for s in states:
        if s.cfg.dim != cfg.dim:
            raise ValueError("bundle dimension mismatch")
        acc += s.psi
    return WaveState.from_complex(acc, cfg)


def unbind_residual(
    factor: WaveState, recovered: WaveState
) -> float:
    """1 - coherence. Zero means perfect recovery, 1 means orthogonal.

    This is the direct analogue of the Sagnac mismatch and is used by
    the binding tests to state recovery accuracy.
    """
    return float(1.0 - factor.coherence(recovered))


def _check(a: WaveState, b: WaveState) -> None:
    if a.cfg.dim != b.cfg.dim:
        raise ValueError(f"dimension mismatch: {a.cfg.dim} vs {b.cfg.dim}")
