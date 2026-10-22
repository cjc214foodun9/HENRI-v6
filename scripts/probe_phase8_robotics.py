"""Phase 8 ACTION 4 probe: full-D closed-loop robotics, receipted.

Pre-registered clauses, fixed BEFORE the first run of this script:

  R1  IDENTITY  sensor == axiom. delta_initial == 0 exactly, 0 steps.
      NECESSARY BUT NEAR-VACUOUS: the reference equals the input, so exact
      zero is structural. Reported, never credited alone.
  R2  NEAR      sensor != axiom, per-component perturbation 1e-3. This is
      the DISCRIMINATING safe clause: non-identical, so delta == 0 would be
      a defect and delta > tau_veto would be a false veto.
      Bar: 0 steps, delta_final == delta_initial <= tau_veto.
  R3  UNSAFE    sign-flipped trajectory. Bar: vetoed_initial True,
      delta_final <= tau_veto, and the recovery arm within the digital-twin
      contract.
  R4  LATENCY   two contracts, never conflated:
      optical 50 us         -> TELEMETRY_ONLY, never gated (Phase 4 ruling)
      digital twin 50000 us -> GATED.
      A k-step recovery is compared against a contract written for ONE
      veto cycle. The bar is NOT moved; instead the receipt also reports
      derived per-step latency so a reader can see the difference.

DEFECT RECORDED IN THIS FILE'S FIRST DRAFT. It re-encoded the sensor
values and reported slot1 = 0.0 while `henri.robotics_loop` reported
act_norm = 0.621304 for the same step. A fresh encode parks all energy in
the entity slot; the proposal slot is filled by `_assemble`. The probe now
reads the loop's own `slot_energy` field, so it cannot measure the wrong
object. Reference models reported act_norm = 0.000786 for this run; that
number is not in any output on this host.

No hyperparameter is retuned here. RELAX_* live in henri/robotics_loop.py
and were fixed before its first run.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

from henri.robotics_loop import RoboticsLoop

OUT = Path("receipts/phase8_robotics_full_D.json")

SAFE = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
NEAR = [0.101, 0.199, 0.3005, 0.3998, 0.5002, 0.5999]
UNSAFE = [0.9, -0.5, 0.8, -0.7, 0.3, 0.1]

TAU_VETO = 0.0431
DIGITAL_TWIN_US = 50000.0
OPTICAL_US = 50.0


def arm(loop, name, sensor, axiom, bar: dict) -> dict:
    t0 = time.perf_counter()
    r = loop.step(sensor, axiom_reference=axiom)
    wall_us = (time.perf_counter() - t0) * 1.0e6
    steps = int(r["adaptation_steps"])
    rec = {
        "arm": name,
        "vetoed_initial": r["vetoed_initial"],
        "sagnac_delta_initial": r["sagnac_delta_initial"],
        "sagnac_delta_final": r["sagnac_delta_final"],
        "adaptation_steps": steps,
        "recovery_verified": r["recovery_verified"],
        "statistic_form": r["statistic_form"],
        "dilution_applied": r["dilution_applied"],
        "slot_energy": r["slot_energy"],
        "action_slot_norm": r["action_slot_norm"],
        "action_projection": r["action_projection"],
        "relaxation": r["relaxation"],
        "recoil_events": r["recoil_events"],
        "sensor_sha256": r["sensor_sha256"],
        "latency_us_internal": round(float(r["latency_us"]), 3),
        "latency_us_wall": round(float(wall_us), 3),
        "latency_per_step_us_derived": (
            round(float(r["latency_us"]) / steps, 3) if steps else None),
        "bar": bar,
    }
    checks = {}
    if "delta_initial_exact_zero" in bar:
        checks["delta_initial_exact_zero"] = (
            float(r["sagnac_delta_initial"]) == 0.0)
    if "delta_initial_nonzero" in bar:
        # A near case that reads exactly zero would mean the statistic is
        # blind to the perturbation. This control must stay able to fire.
        checks["delta_initial_nonzero"] = (
            float(r["sagnac_delta_initial"]) > 0.0)
    if "no_veto" in bar:
        checks["no_veto"] = bool(r["vetoed_initial"]) is False
    if "zero_steps" in bar:
        checks["zero_steps"] = steps == 0
    if "initial_above_tau" in bar:
        checks["initial_above_tau"] = float(r["sagnac_delta_initial"]) > TAU_VETO
    if "final_under_tau" in bar:
        checks["final_under_tau"] = float(r["sagnac_delta_final"]) <= TAU_VETO
    if "twin_latency" in bar:
        checks["twin_latency"] = float(r["latency_us"]) <= DIGITAL_TWIN_US
    rec["checks"] = checks
    rec["arm_verdict"] = "PASS" if all(checks.values()) else "FAIL"
    return rec


def main() -> int:
    t0 = time.time()
    loop = RoboticsLoop()
    build_s = time.time() - t0
    print("build_s %.2f  D %d  dim %d  tau_veto %.4f  form %r"
          % (build_s, loop.cfg.dim, loop.cfg.dim, loop.tau_veto,
             loop.gate.statistic))

    arms = []
    arms.append(arm(loop, "R1_IDENTITY_safe", SAFE, SAFE, {
        "delta_initial_exact_zero": True,
        "no_veto": True,
        "zero_steps": True,
        "final_under_tau": True,
    }))
    arms.append(arm(loop, "R2_NEAR_safe", NEAR, SAFE, {
        "delta_initial_nonzero": True,
        "no_veto": True,
        "zero_steps": True,
        "final_under_tau": True,
    }))
    arms.append(arm(loop, "R3_UNSAFE_perturbed", UNSAFE, SAFE, {
        "initial_above_tau": True,
        "final_under_tau": True,
        "twin_latency": True,
    }))
    arms.append(arm(loop, "R4_VACUOUS_no_axiom", SAFE, None, {
        "no_veto": True,
        "zero_steps": True,
    }))
    arms[-1]["vacuous_axiom"] = True

    for a in arms:
        print("%-22s delta_init %.6f vetoed %-5s steps %d final %.6f "
              "rec %-5s %s"
              % (a["arm"], a["sagnac_delta_initial"], a["vetoed_initial"],
                 a["adaptation_steps"], a["sagnac_delta_final"],
                 a["recovery_verified"], a["arm_verdict"]))
        print("    slot_energy %s act_norm %.6f dilution %.3f "
              "lat %.1f us per_step %s"
              % (a["slot_energy"], a["action_slot_norm"],
                 a["dilution_applied"], a["latency_us_internal"],
                 a["latency_per_step_us_derived"]))

    body = {
        "probe": "phase8_robotics_full_D",
        "dim": int(loop.cfg.dim),
        "tau_veto": float(loop.tau_veto),
        "build_s": round(build_s, 3),
        "arms": arms,
        "latency_policy": {
            "optical_us": OPTICAL_US,
            "optical_status": "TELEMETRY_ONLY",
            "digital_twin_us": DIGITAL_TWIN_US,
            "digital_twin_status": "GATED",
            "host": "CPU, this machine; no working NVIDIA device present",
            "note": ("The digital-twin contract was written for ONE veto "
                     "cycle. A k-step recovery is charged k cycles. The bar "
                     "is not moved; per-step latency is reported instead."),
        },
        "clause_note": (
            "R1 identity is near-vacuous. R2 is the discriminating safe "
            "clause and carries a control that must be able to fire. "
            "R3 gates veto, recovery and twin latency. R4 records the "
            "no-axiom path, which is fail-open by design and marked."),
        "defect_note": (
            "This probe's first draft re-encoded sensor values and "
            "reported slot1 = 0.0 against the loop's act_norm = 0.621304. "
            "The probe was wrong, not the loop."),
    }
    ser = json.dumps(body, indent=2, sort_keys=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(ser.encode("utf-8"))          # write_bytes: no CRLF
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    body["receipt_sha256_of_body_before_self_field"] = digest
    OUT.write_bytes(json.dumps(body, indent=2, sort_keys=True).encode("utf-8"))

    # READBACK: a digest that cannot authenticate the bytes on disk proves
    # nothing. This is the defect the v6 report records for its own harness.
    back = json.loads(OUT.read_text(encoding="utf-8"))
    ok = (back["arms"][2]["sagnac_delta_final"]
          == arms[2]["sagnac_delta_final"])
    print("receipt %s sha256 %s readback %s"
          % (OUT, digest[:16], "OK" if ok else "MISMATCH"))

    gated = [a for a in arms if a["arm"] != "R4_VACUOUS_no_axiom"]
    all_pass = all(a["arm_verdict"] == "PASS" for a in gated)
    print("VERDICT %s" % ("PHASE8_ROBOTICS_PASS" if all_pass
                          else "PHASE8_ROBOTICS_PARTIAL"))
    for a in gated:
        if a["arm_verdict"] == "FAIL":
            failing = [k for k, v in a["checks"].items() if not v]
            print("  FAIL %s on %s" % (a["arm"], failing))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
