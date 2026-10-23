"""Profile the full-D robotics step. Measure before optimizing anything.

WHY THIS FILE EXISTS
The directive claims the 72,979 us total cost comes from "full-65,536 complex
operations and allocations even on zeroed/inactive slots". That is a
HYPOTHESIS. It is checkable, and it may be false: the step budget may be
dominated by the ~3 ACTIVE slots, in which case slicing buys nothing.

This script attributes the cost per stage BEFORE any optimization, so a
later latency claim has a baseline to compare against. The rule the project
uses: a change that cannot be attributed cannot be defended.

WHAT IS MEASURED
  1. Slot occupancy of the assembled candidate. Slicing is only safe if the
     inactive slots are EXACTLY zero, not approximately zero. Exactness is
     asserted, not eyeballed.
  2. Per-stage wall clock for one step, repeated N times, median reported.
  3. The same with the recovery loop exercised (vetoed arm), which is the
     path the 72,979 us figure describes.

NOTHING HERE MUTATES THE LOOP. It imports and calls it.
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np

from henri.config import WaveConfig
from henri.robotics_loop import RoboticsLoop

N_REPEAT = 5
SENSOR = [0.10, -0.20, 0.30, -0.40]
AXIOM = [0.95, -0.05, 0.10, 0.02]


def timeit(fn, n=N_REPEAT):
    """Median microseconds over `n` calls, plus the spread.

    MEDIAN, NOT MEAN, and N is fixed here before the run. A single 6-step
    sample near the 50,000 us bar flickers across it; the median plus the
    observed range is the honest report.
    """
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1.0e6)
    return {"median_us": round(statistics.median(ts), 1),
            "min_us": round(min(ts), 1),
            "max_us": round(max(ts), 1),
            "n": n}


def main() -> int:
    loop = RoboticsLoop(cfg=WaveConfig())
    cfg = loop.cfg
    print("dim            %d" % cfg.dim)
    print("slot_span      %d" % cfg.slot_span)
    print("slot_slice(0)  %s" % (cfg.slot_slice(0),))
    print("slot_slice(3)  %s" % (cfg.slot_slice(3),))

    # -- 1. occupancy ----------------------------------------------------
    w = loop.encode(SENSOR)
    pred = loop.operator.step(w)
    cand = loop._assemble(w, pred)
    print("\nSLOT OCCUPANCY of the assembled candidate")
    occ = {}
    for s in range(4):
        seg = cand.psi[cfg.slot_slice(s)]
        nz = int(np.count_nonzero(seg))
        occ[s] = {"norm": float(np.linalg.norm(seg)),
                  "nonzero": nz, "exact_zero": bool(nz == 0)}
        print("  slot%d  norm %.9f  nonzero %6d/%6d  exact_zero=%s"
              % (s, occ[s]["norm"], nz, seg.shape[0], occ[s]["exact_zero"]))
    inactive_exact = all(occ[s]["exact_zero"] for s in (2, 3))
    print("  INACTIVE_SLOTS_EXACTLY_ZERO %s" % inactive_exact)

    # Is the whole second half empty, or only slots 2-3 in a ragged way?
    half = cfg.dim // 2
    print("  psi[%d:] nonzero = %d  exact_zero=%s"
          % (half, int(np.count_nonzero(cand.psi[half:])),
             bool(np.count_nonzero(cand.psi[half:]) == 0)))

    # -- 2. per-stage attribution ---------------------------------------
    print("\nPER-STAGE wall clock (median of %d, D=%d)" % (N_REPEAT, cfg.dim))
    axiom_state = loop.encode(AXIOM)
    axiom_pred = loop.operator.step(axiom_state)
    axiom = loop._assemble(axiom_state, axiom_pred)
    rng = np.random.default_rng(1)
    buf = loop.thermostat.recoil

    stages = {}
    stages["encode"] = timeit(lambda: loop.encode(SENSOR))
    stages["koopman_step"] = timeit(lambda: loop.operator.step(w))
    stages["assemble"] = timeit(lambda: loop._assemble(w, pred))
    # The veto primes the recoil buffer; the relaxation path reads it.
    loop.gate.evaluate_trajectory(cand, axiom, buf)
    stages["gate_statistic"] = timeit(
        lambda: loop.gate.statistic_value(cand, axiom))
    stages["relax_once"] = timeit(lambda: loop._relax_once(cand, axiom, rng))
    stages["step_noadapt"] = timeit(
        lambda: loop.step(SENSOR, axiom_reference=AXIOM, adapt_on_veto=False))
    stages["step_full"] = timeit(
        lambda: loop.step(SENSOR, axiom_reference=AXIOM, max_steps=6))

    denom = stages["step_full"]["median_us"] or 1.0
    for k, v in stages.items():
        print("  %-16s median %9.1f us   [%8.1f .. %8.1f]   %5.1f%% of step_full"
              % (k, v["median_us"], v["min_us"], v["max_us"],
                 100.0 * v["median_us"] / denom))

    # -- 3. where does relax_once spend its time? -----------------------
    comb = loop.thermostat.recoil.latest_comb()
    if comb is None or comb.shape[0] != cfg.dim:
        comb = np.zeros(cfg.dim, dtype=np.float64)
    basis = axiom.psi.reshape(-1, 1)
    t_active, rep = loop.thermostat.compute_spectral_temperatures(comb, basis)
    print("\nRELAX internals (median of %d)" % N_REPEAT)
    print("  spectral_temps   %9.1f us"
          % timeit(lambda: loop.thermostat.compute_spectral_temperatures(
              comb, basis))["median_us"])
    wcol = cand.psi.reshape(-1, 1)
    grad = wcol - basis
    print("  viscoelastic     %9.1f us"
          % timeit(lambda: loop.thermostat.apply_viscoelastic_creep(
              wcol, grad, t_active, basis_coherent=basis, rng=rng,
              project_update=False, retract=False))["median_us"])
    print("  from_complex     %9.1f us"
          % timeit(lambda: type(cand).from_complex(
              np.asarray(cand.psi), cfg))["median_us"])

    # -- 4. does the ACTIVE HALF hold the cost? --------------------------
    print("\nTHE DECISIVE QUESTION: cost vs active width")
    halfD = cfg.dim // 2
    print("  full   D=%d  gate statistic %9.1f us"
          % (cfg.dim, timeit(lambda: loop.gate.statistic_value(
              cand, axiom))["median_us"]))
    # A truncated copy is NOT equivalent; this is a lower bound on what any
    # slice-based implementation could reach, measured for planning only.
    try:
        from henri.wave_state import WaveState
        small = WaveState.from_complex(cand.psi[:halfD].copy(),
                                      type(cfg)(dim=halfD))
        small_ax = WaveState.from_complex(axiom.psi[:halfD].copy(),
                                         type(cfg)(dim=halfD))
        print("  half   D=%d  gate statistic %9.1f us  (NOT equivalent, "
              "planning only)"
              % (halfD, timeit(lambda: loop.gate.statistic_value(
                  small, small_ax))["median_us"]))
    except Exception as exc:                       # pragma: no cover
        print("  half-size probe skipped: %r" % (exc,))

    print("\nVERDICT_INPUTS")
    print("  inactive_exactly_zero %s" % inactive_exact)
    print("  step_full_median_us   %.1f" % stages["step_full"]["median_us"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
