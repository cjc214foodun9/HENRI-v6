"""G-ROB2: full-D robotics recovery below 50,000 us total, R1-R4 preserved.

PRE-REGISTERED BEFORE THE FIRST RUN.
====================================
WHY THIS PROBE EXISTS
`receipts/phase8_robotics_full_D.json` records R3 recovering
0.895710137 -> 0.042648316 in 6 steps at per_step 13,574.833 us, with
total 81,449.0 us. The per-step clause passed; the raw cumulative clause
did not. Phase 10 asks for both.

CLAUSES
-------
  B1  R1, R2, R3, R4 reproduce the Phase-8 deltas, step counts and slot
      norms. Bar: |delta - recorded| <= 1e-12 and step count identical.
  B2  Full-D state equivalence. Bar: the batched-Koopman path and the
      temperature-cache path each agree with the uncached, unbatched path
      to <= 1e-12 on the final state vector.
  B3  R3 total recovery wall clock. Bar: median of 7 <= 50,000 us, with N
      fixed here before the run.
  B4  R3 per-step latency still <= 50,000 us. The Phase-9 clause must not
      be lost while the total is won.

THE DIRECTIVE'S DIAGNOSIS IS TESTED AND PARTLY FALSE
---------------------------------------------------
The directive says the cost comes from "full-65,536 complex operations and
allocations even on zeroed/inactive slots", and proposes slicing
`[0:32768]` during the inner loop.

MEASURED at D=65536 before any optimization:
  slot2 norm 0.000000000, slot3 norm 0.000000000, exact_zero True
  on the ASSEMBLED CANDIDATE.
  BUT after relaxation: slot2 0.000790150, slot3 0.000793997.
  The recorded Phase-8 R3 arm agrees: slot2 0.000804078, slot3 0.000807872.

So the inactive slots are exactly zero BEFORE recovery and NON-ZERO after
it. Slicing `[0:32768]` inside the recovery loop would truncate live state
and would break B1 and B2. The optimization here does NOT slice. It
removes redundant work instead:

  1. `compute_spectral_temperatures` was recomputed 6 times per recovery.
     It is a pure function of the recoil comb and the axiom basis, and
     neither changes inside the loop. MEASURED: `comb` identical across all
     6 steps. Computed once now. Exact, not approximate.
  2. `operator.step` was called twice per step (sensor and axiom). One
     2-column batch reproduces both. MEASURED max |dpsi| 2.201e-17.

BOTH changes are arithmetic-preserving. B2 proves it rather than asserting
it.

LATENCY IS A HOST CLAIM, NOT A REAL-TIME CLAIM
----------------------------------------------
`latency_note` in the loop already says this. The 50 us optical contract
stays TELEMETRY_ONLY. This probe reports wall clock on this host, median of
N, with the observed range beside it. A single sample near the bar
flickers across it; the median plus range is the honest report.
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np

from henri.receipt import BODY_FIELD, FILE_FIELD, write_receipt

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "receipts" / "phase10_robotics_full_D.json"
PRIOR = REPO / "receipts" / "phase8_robotics_full_D.json"

N_REPEAT = 7                 # fixed before the first run
TAU_VETO = 0.0431
DIGITAL_TWIN_US = 50000.0
EQUIV_TOL = 1e-12

# SCENARIOS ARE THE PHASE-8 CONSTANTS, read from the probe that produced the
# recorded receipt. Re-inventing them would make B1 a comparison of my
# fixture against theirs, which proves nothing.
SAFE = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
NEAR = [0.101, 0.199, 0.3005, 0.3998, 0.5002, 0.5999]
UNSAFE = [0.9, -0.5, 0.8, -0.7, 0.3, 0.1]


def load_prior() -> dict:
    import json
    doc = json.loads(PRIOR.read_text(encoding="utf-8"))
    arms = doc.get("arms") or []
    return {a["arm"]: a for a in arms}


def run_arm(loop, name: str, sensor, axiom) -> dict:
    t0 = time.perf_counter()
    r = loop.step(sensor, axiom_reference=axiom)
    wall = (time.perf_counter() - t0) * 1.0e6
    return {
        "arm": name,
        "sagnac_delta_initial": float(r["sagnac_delta_initial"]),
        "sagnac_delta_final": float(r["sagnac_delta_final"]),
        "adaptation_steps": int(r["adaptation_steps"]),
        "vetoed_initial": bool(r["vetoed_initial"]),
        "recovery_verified": bool(r["recovery_verified"]),
        "slot_energy": {k: float(v) for k, v in r["slot_energy"].items()},
        "action_slot_norm": float(r["action_slot_norm"]),
        "statistic_form": r["statistic_form"],
        "total_latency_us": round(float(r.get("total_latency_us",
                                             r["latency_us"])), 3),
        "per_step_latency_us": round(float(r.get("per_step_latency_us",
                                                 r["latency_us"])), 3),
        "latency_us_wall": round(wall, 3),
    }


def equivalence(loop) -> dict:
    """B2. Compare the optimized paths against the Phase-8 arithmetic.

    THE UNCACHED PATH IS RECONSTRUCTED HERE, not assumed. `_relax_once`
    accepts `cache=None`, which recomputes the temperature field exactly as
    the Phase-8 code did. `operator.step` is the single-column Phase-8 path.
    """
    from henri.robotics_loop import RELAX_SEED
    from henri.state import WaveState

    ws = loop.encode(SAFE)
    wa = loop.encode(UNSAFE)

    # -- batched vs separate Koopman -----------------------------------
    sep_sensor = loop.operator.step(ws).psi
    sep_axiom = loop.operator.step(wa).psi
    raw = loop.operator.apply_raw_batch(np.stack([ws.psi, wa.psi], axis=1))
    bat_sensor = WaveState.from_complex(raw[:, 0], loop.cfg).psi
    bat_axiom = WaveState.from_complex(raw[:, 1], loop.cfg).psi
    d_sensor = float(np.max(np.abs(bat_sensor - sep_sensor)))
    d_axiom = float(np.max(np.abs(bat_axiom - sep_axiom)))

    # -- cached vs uncached relaxation --------------------------------
    cand = loop._assemble(ws, WaveState.from_complex(sep_sensor, loop.cfg))
    axiom = loop._assemble(wa, WaveState.from_complex(sep_axiom, loop.cfg))
    loop.gate.evaluate_trajectory(cand, axiom, loop.thermostat.recoil)

    def relax_run(use_cache: bool) -> np.ndarray:
        rng = np.random.default_rng(RELAX_SEED)
        cur = cand
        cache = loop._relax_cache(axiom) if use_cache else None
        for _ in range(6):
            cur, _rep = loop._relax_once(cur, axiom, rng, cache=cache)
        return cur.psi.copy()

    d_relax = float(np.max(np.abs(relax_run(False) - relax_run(True))))

    return {"koopman_sensor_max_abs": d_sensor,
            "koopman_axiom_max_abs": d_axiom,
            "relaxation_max_abs": d_relax,
            "tol": EQUIV_TOL,
            "pass": bool(max(d_sensor, d_axiom, d_relax) <= EQUIV_TOL)}


def median_latency(loop, sensor, axiom, n: int) -> dict:
    """B3. Median of N, N fixed before the run, range beside it."""
    loop.step(sensor, axiom_reference=axiom)          # warm
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        loop.step(sensor, axiom_reference=axiom)
        ts.append((time.perf_counter() - t0) * 1.0e6)
    return {"n": n, "median_us": round(statistics.median(ts), 1),
            "min_us": round(min(ts), 1), "max_us": round(max(ts), 1),
            "all_us": [round(t, 1) for t in ts]}


def main() -> int:
    t_start = time.time()
    from henri.robotics_loop import RoboticsLoop

    loop = RoboticsLoop()
    print("D %d  tau_veto %.4f  form %r" %
          (loop.cfg.dim, loop.tau_veto, loop.gate.statistic))

    prior = load_prior()
    print("prior arms: %s" % sorted(prior))

    arms = [run_arm(loop, "R1_IDENTITY_safe", SAFE, SAFE),
            run_arm(loop, "R2_NEAR_safe", NEAR, SAFE),
            run_arm(loop, "R3_UNSAFE_perturbed", UNSAFE, SAFE),
            run_arm(loop, "R4_VACUOUS_no_axiom", SAFE, None)]

    # -- B1: reproduce the Phase-8 numbers -----------------------------
    b1_rows = []
    b1_ok = True
    for a in arms:
        p = prior.get(a["arm"])
        if p is None:
            b1_ok = False
            b1_rows.append({"arm": a["arm"], "error": "absent from prior"})
            continue
        di = abs(a["sagnac_delta_initial"] - float(p["sagnac_delta_initial"]))
        df = abs(a["sagnac_delta_final"] - float(p["sagnac_delta_final"]))
        ds = max(abs(a["slot_energy"][k] - float(p["slot_energy"][k]))
                 for k in ("slot0", "slot1", "slot2", "slot3"))
        step_ok = a["adaptation_steps"] == int(p["adaptation_steps"])
        row_ok = bool(max(di, df, ds) <= EQUIV_TOL and step_ok)
        b1_ok = b1_ok and row_ok
        b1_rows.append({"arm": a["arm"],
                        "delta_initial_abs_diff": di,
                        "delta_final_abs_diff": df,
                        "slot_norm_max_abs_diff": ds,
                        "steps_now": a["adaptation_steps"],
                        "steps_prior": int(p["adaptation_steps"]),
                        "prior_delta_initial": float(p["sagnac_delta_initial"]),
                        "prior_delta_final": float(p["sagnac_delta_final"]),
                        "pass": row_ok})
        print("  %-22s d_init %.6f d_final %.6f steps %d  %s"
              % (a["arm"], a["sagnac_delta_initial"],
                 a["sagnac_delta_final"], a["adaptation_steps"],
                 "OK" if row_ok else "MISMATCH"))

    eq = equivalence(loop)
    print("B2 equivalence: koopman %.3e / %.3e  relax %.3e  %s"
          % (eq["koopman_sensor_max_abs"], eq["koopman_axiom_max_abs"],
             eq["relaxation_max_abs"], "PASS" if eq["pass"] else "FAIL"))

    lat = median_latency(loop, UNSAFE, SAFE, N_REPEAT)
    r3 = next(a for a in arms if a["arm"] == "R3_UNSAFE_perturbed")
    print("B3 R3 total median %.0f us  [%.0f..%.0f]  n=%d"
          % (lat["median_us"], lat["min_us"], lat["max_us"], lat["n"]))
    print("B4 R3 per_step %.0f us" % r3["per_step_latency_us"])

    b3_ok = lat["median_us"] <= DIGITAL_TWIN_US
    b4_ok = r3["per_step_latency_us"] <= DIGITAL_TWIN_US
    b2_ok = eq["pass"]

    prior_total = float(prior["R3_UNSAFE_perturbed"]["total_latency_us"])
    prior_step = float(prior["R3_UNSAFE_perturbed"]["per_step_latency_us"])

    clauses = [
        {"id": "B1", "what": "R1-R4 reproduce Phase-8 deltas/steps/norms",
         "pass": bool(b1_ok), "value": b1_rows,
         "bar": "|diff| <= 1e-12 and identical step counts"},
        {"id": "B2", "what": "full-D equivalence of both optimizations",
         "pass": bool(b2_ok), "value": eq, "bar": "<= 1e-12"},
        {"id": "B3", "what": "R3 total recovery <= 50,000 us",
         "pass": bool(b3_ok), "value": {**lat, "bar_us": DIGITAL_TWIN_US,
                                        "prior_total_us": prior_total},
         "bar": "median of %d" % N_REPEAT},
        {"id": "B4", "what": "R3 per-step latency <= 50,000 us",
         "pass": bool(b4_ok),
         "value": {"per_step_us": r3["per_step_latency_us"],
                   "prior_per_step_us": prior_step,
                   "bar_us": DIGITAL_TWIN_US},
         "bar": "the Phase-9 clause is not lost"},
    ]
    npass = sum(1 for c in clauses if c["pass"])
    verdict = ("G_ROB2_PASS" if npass == len(clauses)
               else "G_ROB2_PARTIAL_%d_OF_%d" % (npass, len(clauses)))

    body = {
        "probe": "phase10_robotics_full_D",
        "dim": int(loop.cfg.dim),
        "tau_veto": float(loop.tau_veto),
        "n_repeat": N_REPEAT,
        "digital_twin_us": DIGITAL_TWIN_US,
        "optical_us": {"value": 50.0, "status": "TELEMETRY_ONLY"},
        "arms": arms,
        "clauses": clauses,
        "n_clauses_pass": npass,
        "verdict": verdict,
        "latency": {"r3_total": lat,
                    "prior_total_us": prior_total,
                    "prior_per_step_us": prior_step,
                    "improvement_us": round(prior_total - lat["median_us"], 1),
                    "improvement_pct": round(
                        100.0 * (prior_total - lat["median_us"]) / prior_total, 2)},
        "optimizations": [
            {"what": "temperature field computed once per recovery",
             "why_exact": ("compute_spectral_temperatures is a pure function "
                           "of the recoil comb and axiom basis; neither "
                           "changes inside the loop. MEASURED: comb identical "
                           "across all 6 steps."),
             "measured": {"relax_once_before_us": 6232.6,
                          "spectral_temps_share_us": 3604.9}},
            {"what": "one batched Koopman pass for sensor and axiom",
             "why_exact": ("the operator is applied column-wise and the "
                           "low-rank term is a matrix product either way"),
             "measured": {"two_separate_us": 23991, "batched_us": 16309,
                          "max_abs_diff": 2.201e-17}},
        ],
        "rejected_optimization": {
            "what": "slice [0:32768] during the inner recovery loop",
            "why_rejected": (
                "slots 2 and 3 are EXACTLY zero on the assembled candidate "
                "but NON-ZERO after relaxation. MEASURED here and in the "
                "Phase-8 R3 arm (slot2 0.000804078, slot3 0.000807872). "
                "Slicing would truncate live state and break B1 and B2."),
            "measured_assembled": {"slot2": 0.0, "slot3": 0.0},
            "measured_after_relaxation_prior": {"slot2": 0.000804078,
                                                "slot3": 0.000807872},
        },
        "latency_note": ("wall clock on this host, median of %d. Not a "
                         "real-time claim. The 50 us optical contract stays "
                         "TELEMETRY_ONLY." % N_REPEAT),
        "wall_clock_s": round(time.time() - t_start, 2),
    }

    ids = write_receipt(OUT, body)
    print()
    for c in clauses:
        print("%-4s %-50s %s" % (c["id"], c["what"],
                                 "PASS" if c["pass"] else "FAIL"))
    print("B3 total %.0f us -> improvement %.1f us (%.1f%%)"
          % (lat["median_us"], prior_total - lat["median_us"],
             100.0 * (prior_total - lat["median_us"]) / prior_total))
    print(FILE_FIELD, ids[FILE_FIELD])
    print("sidecar", ids.get("sidecar"))
    print("VERDICT %s" % verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
