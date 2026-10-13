"""Recoil-coupled anisotropic Langevin annealing.

Evidence label: DERIVED for the GLE algebra; OBSERVED for every number
`tests/test_phase3_gates.py` measures.

PHASE 3 MODULE DIRECTIVE 2, with three measured corrections.

CORRECTION 1: NO DENSE D x D PROJECTOR (constraint C7)
The spec writes `P_null = I - V (V^H V)^{-1} V^H`, a dense `D x D`
complex matrix. At `D = 65536` that is 68.7 GiB and the repo's own C7
guard refuses it. This module applies the SAME operator in the dual
low-rank form:

    P_null X = X - V @ solve(V^H V, V^H X)

for any `X` of shape `(D, m)`. Identical action, `O(D k m)` storage.
A test asserts no `D x D` object is formed.

CORRECTION 2: THE SPEC'S OWN EQUATION CANNOT PASS ITS OWN GATE
Gate G-ACT2 demands BOTH
    ||P_coherent (W_{t+k} - W_t)||_F < 1e-5
    ||P_null     (W_{t+k} - W_t)||_F >= 0.15
The spec's update
    W_{t+1} = W_t - eta grad + sqrt(2 T_active eta) xi
does NOT project the GRADIENT. Taking `P_coherent` of that update
leaves `-eta P_coherent(grad)`, which is not small unless the gradient
already lies in the null space. Measured before repair: the literal
form FAILED the first clause. This module projects the WHOLE update
through `P_null`, which makes the coherent component vanish exactly
(`V^H P_null = 0`). Both the projected and literal arms are shipped so
the difference stays measurable.

CORRECTION 3: `P_null @ T` CAN BE NEGATIVE, AND sqrt OF IT IS NaN
`T_d` is a positive temperature vector. `P_null` is not a positive
operator on real vectors, so the projected temperatures can go negative,
and `sqrt(2 T_active dt)` then yields NaN. This module CLAMPS at zero
and REPORTS the clamped fraction in `ThermostatReport`. The clamp is a
measured defect of the construction, not a silent
convenience.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from henri.config import WaveConfig
from henri.gram import assert_no_dense_dxd, stiefel_retract
from henri.sagnac_homodyne_gate import RECOIL_CAPACITY, RecoilBuffer

_EPS = 1e-30
_LS_RIDGE = 1e-12


def null_space_apply(
    basis_coherent: np.ndarray, X: np.ndarray, ridge: float = _LS_RIDGE
) -> np.ndarray:
    """`P_null @ X` in the dual form. Never forms a `D x D` operator.

    Args:
        basis_coherent: `V`, shape `(D, k)`. Columns span the confirmed
            resonant invariants. Need not be orthonormal.
        X: shape `(D,)` or `(D, m)`.
        ridge: Tikhonov term added to `V^H V` so a rank-deficient basis
            still solves.

    Returns:
        `P_null @ X`, same shape as `X`.
    """
    v = np.asarray(basis_coherent, dtype=np.complex128)
    if v.ndim != 2:
        raise ValueError(f"basis must be 2-D, got {v.shape}")
    x = np.asarray(X, dtype=np.complex128)
    was_1d = x.ndim == 1
    if was_1d:
        x = x.reshape(-1, 1)
    if x.shape[0] != v.shape[0]:
        raise ValueError(f"X rows {x.shape[0]} != basis rows {v.shape[0]}")

    gram = v.conj().T @ v                                  # (k, k)
    k = gram.shape[0]
    coeff = np.linalg.solve(gram + ridge * np.eye(k, dtype=np.complex128),
                            v.conj().T @ x)
    out = x - v @ coeff
    assert_no_dense_dxd(out)
    return out.reshape(-1) if was_1d else out


def coherent_space_apply(basis_coherent: np.ndarray, X: np.ndarray) -> np.ndarray:
    """`P_coherent @ X = V (V^H V)^{-1} V^H X`. Dual form."""
    v = np.asarray(basis_coherent, dtype=np.complex128)
    x = np.asarray(X, dtype=np.complex128)
    was_1d = x.ndim == 1
    if was_1d:
        x = x.reshape(-1, 1)
    gram = v.conj().T @ v
    k = gram.shape[0]
    coeff = np.linalg.solve(gram + _LS_RIDGE * np.eye(k, dtype=np.complex128),
                            v.conj().T @ x)
    out = v @ coeff
    return out.reshape(-1) if was_1d else out


@dataclass
class ThermostatReport:
    """Measurements from one temperature computation.

    Attributes:
        clamped_fraction: fraction of coordinates where the projected
            temperature was negative and got clamped to zero. Nonzero
            means the null-space projection is not positivity
            preserving.
        raw_min: most negative projected temperature before clamping.
        active_max: largest active temperature after clamping.
        coherent_leak: `||P_coherent T_active||_2`, the temperature mass
            that survived projection. Should be ~0.
    """

    clamped_fraction: float
    raw_min: float
    active_max: float
    coherent_leak: float


class AnisotropicLangevinThermostat:
    """Anisotropic thermal relaxation driven by dark-port recoil.

    Args:
        cfg: wave configuration (for dimension checks).
        T_base: baseline temperature applied to every coordinate.
        kappa: coupling of the phase-mismatch comb to temperature.
        eta: learning rate in the GLE update.
        dt: integration step.
        seed: explicit RNG seed. Determinism is not optional.
        capacity: recoil buffer capacity.
    """

    def __init__(
        self,
        cfg: Optional[WaveConfig] = None,
        T_base: float = 0.01,
        kappa: float = 1.0,
        eta: float = 0.05,
        dt: float = 1e-3,
        seed: Optional[int] = 20261012,
        capacity: int = RECOIL_CAPACITY,
    ) -> None:
        if T_base < 0.0:
            raise ValueError("T_base must be non-negative")
        if kappa < 0.0:
            raise ValueError("kappa must be non-negative")
        if eta <= 0.0:
            raise ValueError("eta must be positive")
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self.cfg = cfg
        self.T_base = float(T_base)
        self.kappa = float(kappa)
        self.eta = float(eta)
        self.dt = float(dt)
        self.seed = seed
        self.recoil = RecoilBuffer(capacity=capacity)

    # -- temperatures ---------------------------------------------------

    def compute_spectral_temperatures(
        self,
        delta_phi_comb: np.ndarray,
        basis_coherent: np.ndarray,
    ) -> Tuple[np.ndarray, ThermostatReport]:
        """Element-wise temperatures, projected off the coherent subspace.

        `T_d = T_base + kappa (1 - exp(-delta_phi_d))`, then
        `T_active = P_null @ T`. Negative entries are clamped to zero and
        the clamped fraction is reported.
        """
        comb = np.asarray(delta_phi_comb, dtype=np.float64).reshape(-1)
        if basis_coherent is not None:
            v = np.asarray(basis_coherent)
            if v.ndim == 2 and v.shape[0] != comb.shape[0]:
                raise ValueError(
                    f"comb length {comb.shape[0]} != basis rows {v.shape[0]}"
                )
        raw = self.T_base + self.kappa * (1.0 - np.exp(-comb))
        if basis_coherent is None or np.asarray(basis_coherent).shape[1] == 0:
            active = raw
            leak = 0.0
        else:
            projected = null_space_apply(basis_coherent, raw)
            active = np.real(projected)
            leak = float(np.linalg.norm(coherent_space_apply(basis_coherent, active)))
        raw_min = float(np.min(active))
        n_neg = int(np.sum(active < 0.0))
        active = np.clip(active, 0.0, None)
        report = ThermostatReport(
            clamped_fraction=float(n_neg) / float(active.size),
            raw_min=raw_min,
            active_max=float(np.max(active)),
            coherent_leak=leak,
        )
        return active, report

    # -- parameter update -----------------------------------------------

    def apply_viscoelastic_creep(
        self,
        weights_tensor: np.ndarray,
        gradient_tensor: np.ndarray,
        T_active: np.ndarray,
        basis_coherent: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
        project_update: bool = True,
        retract: bool = True,
    ) -> np.ndarray:
        """One generalized-Langevin update with anisotropic noise.

        Args:
            weights_tensor: `W`, shape `(D, m)`.
            gradient_tensor: `grad`, same shape.
            T_active: per-coordinate temperature, shape `(D,)`.
            basis_coherent: `V`, shape `(D, k)`. When given and
                `project_update` is True, the WHOLE update is projected
                through `P_null`, which is what makes G-ACT2's first
                clause satisfiable.
            rng: explicit generator.
            project_update: project the update through `P_null`. Set
                False to reproduce the spec-literal form, which fails
                G-ACT2's first clause.
            retract: apply the Stiefel polar retraction.

        Returns:
            `W_new`, same shape as `W`.
        """
        w = np.asarray(weights_tensor, dtype=np.complex128)
        g = np.asarray(gradient_tensor, dtype=np.complex128)
        if w.shape != g.shape:
            raise ValueError(f"weights {w.shape} and gradient {g.shape} differ")
        if w.ndim != 2:
            raise ValueError("weights must be 2-D (D, m)")
        t = np.asarray(T_active, dtype=np.float64).reshape(-1)
        if t.shape[0] != w.shape[0]:
            raise ValueError(f"T_active length {t.shape[0]} != rows {w.shape[0]}")
        if np.any(t < 0.0):
            raise ValueError("T_active must be non-negative; clamp before calling")

        gen = rng if rng is not None else np.random.default_rng(self.seed)
        xi = (
            gen.standard_normal(w.shape) + 1j * gen.standard_normal(w.shape)
        ) / np.sqrt(2.0)

        drift = -self.eta * g * self.dt
        noise = np.sqrt(2.0 * t * self.dt)[:, None] * xi
        update = drift + noise

        if project_update and basis_coherent is not None:
            update = null_space_apply(basis_coherent, update)

        w_new = w + update

        if retract:
            # stiefel_retract works on rows; transpose in and out.
            rows = w_new.conj().T
            w_new = stiefel_retract(rows).conj().T
        return w_new

    # -- full anneal ----------------------------------------------------

    def anneal(
        self,
        weights_tensor: np.ndarray,
        basis_coherent: np.ndarray,
        gradient_fn,
        steps: int = 5,
        rng: Optional[np.random.Generator] = None,
        project_update: bool = True,
        retract: bool = True,
    ) -> Tuple[np.ndarray, dict]:
        """Run `steps` of recoil-coupled annealing.

        `gradient_fn(W)` returns the gradient of the free-energy
        functional at `W`. On each step the stored recoil comb drives the
        anisotropic temperature.
        """
        comb = self.recoil.latest_comb()
        if comb is None or comb.shape[0] != weights_tensor.shape[0]:
            comb = np.zeros(weights_tensor.shape[0], dtype=np.float64)
        w = np.asarray(weights_tensor, dtype=np.complex128)
        gen = rng if rng is not None else np.random.default_rng(self.seed)
        log = []
        for _ in range(int(steps)):
            t_active, rep = self.compute_spectral_temperatures(comb, basis_coherent)
            w = self.apply_viscoelastic_creep(
                w,
                gradient_fn(w),
                t_active,
                basis_coherent=basis_coherent,
                rng=gen,
                project_update=project_update,
                retract=retract,
            )
            log.append(
                {
                    "clamped_fraction": rep.clamped_fraction,
                    "raw_min": rep.raw_min,
                    "active_max": rep.active_max,
                    "coherent_leak": rep.coherent_leak,
                }
            )
        return w, {"steps": log}
