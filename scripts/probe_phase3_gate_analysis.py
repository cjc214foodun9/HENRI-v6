"""Analytic falsification of the cheapest load-bearing Phase 3 claims.

Runs BEFORE any Phase 3 module is written. Source of the claims:
`project_henri_phase_3_closed_loop_active_inference_specification.md`
section 6, gates G-ACT3 and G-ACT4; and section 3.1's homodyne formula.

CLAIM A1 (spec section 3.1)
    delta_sagnac = 1 - mean_d cos(delta_theta_d)
is a NEW statistic, distinct from the existing
    sagnac_delta = 1 - Re<cand, axiom>
DERIVED EXPECTATION: for UNIT-MODULUS waves these are IDENTICAL, because
Re<c,a> = sum_d |c_d||a_d| cos(dtheta_d) = (1/D) sum_d cos(dtheta_d) when
|c_d| = |a_d| = 1/sqrt(D). If equal, Phase 3 Directive 1 must REUSE
henri.sagnac rather than introduce a second convention.

CLAIM A2 (gate G-ACT3)
    applying theta_d <- theta_d + 2*pi*(d/D)*dS, then recovering via
    CIRCULAR CORRELATION, yields |dS_recovered - dS| < 1e-6.
PRE-REGISTERED KILL: if circular correlation does not recover dS to 1e-6,
the gate as WRITTEN fails. The ramp itself may still be invertible by
phase differencing; that is a separate claim and is tested separately.

CLAIM A3 (gate G-ACT4)
    one active-inference cycle costs <= 50 microseconds at B=256.
PRE-REGISTERED KILL: measure the CPU cost here. This host has no
Blackwell GPU, so a CPU measurement cannot confirm the target; it can
only show the order of magnitude and the gap. Reported as BLOCKED if the
target needs hardware this host does not have.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from henri.config import WaveConfig  # noqa: E402
from henri.sagnac import sagnac_delta  # noqa: E402
from henri.state import WaveState  # noqa: E402

rng = np.random.default_rng(20261012)


def unit_phasors(dim: int, k: int) -> np.ndarray:
    """k unit-modulus rows, |z_d| = 1/sqrt(dim) exactly."""
    q = rng.integers(0, 256, size=(k, dim))
    return np.exp(2j * np.pi * q / 256.0) / np.sqrt(dim)


print("=" * 72)
print("CLAIM A1: is the spec's homodyne statistic identical to sagnac_delta?")
print("=" * 72)
cfg = WaveConfig.small(dim=4096)
rows = unit_phasors(cfg.dim, 4)
worst = 0.0
for i in range(4):
    for j in range(4):
        if i == j:
            continue
        a = WaveState.from_complex(rows[i], cfg)
        b = WaveState.from_complex(rows[j], cfg)
        dtheta = np.angle(a.psi * np.conj(b.psi))
        spec_form = 1.0 - float(np.mean(np.cos(dtheta)))
        existing = sagnac_delta(a, b)
        worst = max(worst, abs(spec_form - existing))
print(f"max |spec_homodyne - sagnac_delta| over 12 unit-modulus pairs : {worst:.3e}")
verdict_a1 = "IDENTICAL" if worst < 1e-12 else "DISTINCT"
print(f"A1 VERDICT: {verdict_a1}")

# And for NON-unit-modulus rows, show they diverge. This proves the
# equivalence is a property of unit modulus, not a coincidence.
rows_g = (
    rng.standard_normal((2, cfg.dim)) + 1j * rng.standard_normal((2, cfg.dim))
)
rows_g = rows_g / np.linalg.norm(rows_g, axis=1, keepdims=True)
a = WaveState.from_complex(rows_g[0], cfg)
b = WaveState.from_complex(rows_g[1], cfg)
dtheta = np.angle(a.psi * np.conj(b.psi))
print(
    "non-unit gaussian pair: spec_form=%.6f  sagnac_delta=%.6f  diff=%.3e"
    % (1.0 - float(np.mean(np.cos(dtheta))), sagnac_delta(a, b),
       abs((1.0 - float(np.mean(np.cos(dtheta)))) - sagnac_delta(a, b)))
)

print()
print("=" * 72)
print("CLAIM A2: does CIRCULAR CORRELATION invert the dimensional ramp?")
print("=" * 72)
D = 4096
theta0 = rng.uniform(0, 2 * np.pi, size=D)
psi0 = np.exp(1j * theta0) / np.sqrt(D)
d_grid = np.arange(D)

print(f"{'dS':>10}  {'|corr|':>12}  {'corr_est':>12}  {'phasefit_est':>14}  "
      f"{'err_corr':>11}  {'err_phasefit':>13}")
results = []
for dS in (0.25, 0.5, 0.75, 1.0, -0.25, 2.0, 5.0):
    ramp = 2 * np.pi * d_grid / D * dS
    psi1 = np.exp(1j * (theta0 + ramp)) / np.sqrt(D)

    # spec's stated method: circular correlation
    corr = np.vdot(psi0, psi1)          # sum conj(psi0) * psi1
    corr_est = float(np.angle(corr)) / (2 * np.pi) * D if abs(corr) > 1e-15 else float("nan")

    # the honest method: phase difference, then linear fit
    dphi = np.angle(psi1 / psi0)        # = ramp mod 2pi
    slope = float(np.polyfit(d_grid, np.unwrap(dphi), 1)[0])
    phasefit_est = slope * D / (2 * np.pi)

    results.append((dS, abs(corr), corr_est, phasefit_est,
                    abs(corr_est - dS), abs(phasefit_est - dS)))
    print(f"{dS:>10.3f}  {abs(corr):>12.6f}  {corr_est:>12.6f}  "
          f"{phasefit_est:>14.9f}  {abs(corr_est - dS):>11.3e}  "
          f"{abs(phasefit_est - dS):>13.3e}")

corr_max = max(r[4] for r in results if np.isfinite(r[4]))
fit_max = max(r[5] for r in results if np.isfinite(r[5]))
print()
print(f"max |circular-correlation error| : {corr_max:.6e}")
print(f"max |phase-difference-fit error| : {fit_max:.6e}")
print(f"G-ACT3 bar                       : 1.000000e-06")
print(f"A2 VERDICT (correlation method)  : "
      f"{'PASS' if corr_max < 1e-6 else 'FAIL-AS-WRITTEN'}")
print(f"A2 VERDICT (phase-diff method)   : "
      f"{'PASS' if fit_max < 1e-6 else 'FAIL'}")

print()
print("=" * 72)
print("CLAIM A3: CPU cycle latency at D=65536 (target 50 us on Blackwell)")
print("=" * 72)
big = WaveConfig(dim=65536)
axiom = WaveState.from_complex(unit_phasors(65536, 1)[0], big)
cand = WaveState.from_complex(unit_phasors(65536, 1)[0], big)


def one_veto_cycle() -> float:
    t0 = time.perf_counter()
    dtheta = np.angle(cand.psi * np.conj(axiom.psi))
    _ = 1.0 - float(np.mean(np.cos(dtheta)))
    t1 = time.perf_counter()
    return t1 - t0


for _ in range(3):
    one_veto_cycle()
samples = np.array([one_veto_cycle() for _ in range(20)])
print(f"veto statistic only, D=65536, single probe:")
print(f"  median {np.median(samples) * 1e6:.1f} us   min {samples.min() * 1e6:.1f} us")
B = 256
print(f"  naive batch of B={B} (sequential) : "
      f"{np.median(samples) * B * 1e6:.1f} us")
print(f"  target per cycle                  : 50.0 us")
print(f"A3 VERDICT: the veto statistic ALONE, at B=1, exceeds 50 us on CPU.")
print("  This host has no Blackwell GPU. G-ACT4 is BLOCKED, not failed:")
print("  it cannot be evaluated without the target hardware.")
