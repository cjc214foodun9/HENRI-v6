"""Phase 6 Action 2 probe: uint8 cold cartridges and bounded LRU hot cache.

PRE-REGISTERED GATES, FIXED BEFORE THE RUN
------------------------------------------
G-P6B1  QUANTISATION BOUND. Reconstructing a unit wave from uint8 Z_256
        ring indices must stay within half a bin: `pi/256 = 0.012272` rad.
G-P6B2  EXACT RECALL THROUGH THE BANK. Paired recall error must be 0.0,
        because `DeltaMemory` is built from RAW complex keys and the cold
        tier is a separate copy, not an input to recall.
G-P6B3  FOOTPRINT. 13,632 rows / 213 cartridges must fit the 3.0 GiB gate.
        Complex128 alone was 26.62 GiB.
G-P6B4  LRU EVICTS. Pushing past capacity must produce evictions > 0 and
        entries <= max. A cache that never evicts is unbounded.
G-P6B5  REDUNDANT STIEFEL RETIRED. The default path must leave `keys_ortho`
        unset. Enabling it must still work.

NEGATIVE CONTROLS
  N1  A cold cartridge built from a NON-unit wave must reconstruct at
      `1/sqrt(D)` per component, NOT at the input scale. This proves the
      quantiser stores phase and does not preserve magnitude.
  N2  A zero-entry LRU ceiling is rejected at construction, so the bound
      cannot be silently disabled.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from henri.cartridge_bank import (                      # noqa: E402
    CartridgeBank, ColdCartridge, SLOT_SPAN, build_bank,
    cartridge_bytes, cold_cartridge_bytes,
)


def _unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-30)


def main() -> int:
    rec: dict = {"probe": "phase6_action2_cold_lru"}
    rng = np.random.default_rng(4242)
    half_bin = float(np.pi / 256.0)

    # G-P6B1 quantisation bound on the ACTIVE span.
    span = 2 * SLOT_SPAN
    k = _unit(rng.standard_normal((64, span)) +
              1j * rng.standard_normal((64, span)))
    v = _unit(rng.standard_normal((64, span)) +
              1j * rng.standard_normal((64, span)))
    cc = ColdCartridge.from_waves(0, k, v)
    kr, vr = cc.materialise()
    # Compare PHASE, which is what the ring stores.
    e_k = float(np.max(np.abs(np.angle(kr * np.conj(k)))))
    e_v = float(np.max(np.abs(np.angle(vr * np.conj(v)))))
    rec["G_P6B1"] = {
        "bytes_per_cartridge": cold_cartridge_bytes(64),
        "max_phase_error_keys": round(e_k, 9),
        "max_phase_error_values": round(e_v, 9),
        "half_bin_bound": round(half_bin, 9),
        "verdict": "PASS" if max(e_k, e_v) <= half_bin + 1e-12 else "FAIL",
    }

    # N1: non-unit input must not preserve magnitude.
    big = 7.5 * (rng.standard_normal((1, span)) +
                 1j * rng.standard_normal((1, span)))
    ccn = ColdCartridge.from_waves(0, big, big)
    kn, _ = ccn.materialise()
    mag = float(np.abs(kn).mean())
    rec["N1_nonunit"] = {
        "input_mean_magnitude": round(float(np.abs(big).mean()), 6),
        "reconstructed_mean_magnitude": round(mag, 9),
        "expected": round(1.0 / np.sqrt(span), 9),
        "note": "Stores phase only. Magnitude is NOT preserved by design.",
        "verdict": "PASS" if abs(mag - 1.0 / np.sqrt(span)) < 1e-9 else "FAIL",
    }

    # G-P6B2 exact recall through the bank at a small eval dimension.
    D = 1024
    kk = _unit(rng.standard_normal((64, D)) + 1j * rng.standard_normal((64, D)))
    vv = _unit(rng.standard_normal((64, D)) + 1j * rng.standard_normal((64, D)))
    bank = CartridgeBank(dim=D, rank=64)
    bank.add_cartridge(kk, vv, subject="probe")
    bank.finalize()
    i = 41
    res = bank.query(kk[i], k_route=1)
    perr = (float(np.linalg.norm(res.value - vv[i]))
            if res.value is not None else None)
    rec["G_P6B2"] = {"paired_recall_error": perr,
                     "router_cartridge": int(res.cartridge),
                     "verdict": "PASS" if perr is not None and perr < 1e-9
                     else "FAIL"}

    # G-P6B3 footprint, derived not allocated.
    rows = 13632
    n_carts = int(np.ceil(rows / 64))
    cold_one = cold_cartridge_bytes(64)
    cold_total = n_carts * cold_one
    hot_one = cartridge_bytes(65536, 64, np.complex128)
    hot_ceiling = 8 * hot_one
    rec["G_P6B3"] = {
        "rows": rows, "cartridges": n_carts,
        "cold_bytes_per_cartridge": cold_one,
        "cold_mib_per_cartridge": round(cold_one / 2**20, 3),
        "cold_total_gib": round(cold_total / 2**30, 4),
        "complex128_gib": round(n_carts * hot_one / 2**30, 4),
        "reduction_factor": round((n_carts * hot_one) / cold_total, 2),
        "hot_ceiling_gib_max8": round(hot_ceiling / 2**30, 4),
        "tier_total_gib": round((cold_total + hot_ceiling) / 2**30, 4),
        "gate_gib": 3.0,
        "verdict": "PASS" if (cold_total + hot_ceiling) <= 3.0 * 2**30
        else "FAIL",
    }

    # G-P6B4 LRU evicts under pressure.
    b2 = CartridgeBank(dim=256, rank=8, max_hot=3)
    for ci in range(6):
        a = _unit(rng.standard_normal((8, 256)) +
                  1j * rng.standard_normal((8, 256)))
        b = _unit(rng.standard_normal((8, 256)) +
                  1j * rng.standard_normal((8, 256)))
        b2.add_cartridge(a, b, subject=f"c{ci}")
    b2.finalize()
    for ci in range(6):
        b2.query(b2.cartridges[ci].keys_raw[0], k_route=1)
    st = b2.hot.stats()
    rec["G_P6B4"] = {**st,
                     "verdict": "PASS" if st["evictions"] > 0
                     and st["entries"] <= st["max"] else "FAIL"}

    # N2: zero ceiling must be rejected.
    try:
        CartridgeBank(dim=64, rank=4, max_hot=0)
        rejected = False
    except ValueError:
        rejected = True
    rec["N2_zero_ceiling"] = {"rejected": rejected,
                              "verdict": "PASS" if rejected else "FAIL"}

    # G-P6B5 write-time Stiefel retired from the default path.
    b3 = CartridgeBank(dim=256, rank=8)
    a3 = _unit(rng.standard_normal((4, 256)) + 1j * rng.standard_normal((4, 256)))
    c3 = b3.add_cartridge(a3, a3, subject="s")
    b4 = CartridgeBank(dim=256, rank=8, write_stiefel=True)
    c4 = b4.add_cartridge(a3, a3, subject="s")
    rec["G_P6B5"] = {
        "default_write_stiefel": b3.write_stiefel,
        "default_keys_ortho_set": c3.keys_ortho is not None,
        "optin_keys_ortho_set": c4.keys_ortho is not None,
        "verdict": "PASS" if (not b3.write_stiefel
                              and c3.keys_ortho is None
                              and c4.keys_ortho is not None) else "FAIL",
    }

    rec["verdicts"] = {k: rec[k]["verdict"] for k in
                       ("G_P6B1", "G_P6B2", "G_P6B3", "G_P6B4", "G_P6B5",
                        "N1_nonunit", "N2_zero_ceiling")}
    rec["all_pass"] = all(v == "PASS" for v in rec["verdicts"].values())

    out = REPO / "receipts" / "phase6_action2.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
    print(json.dumps(rec["verdicts"], indent=2))
    print(json.dumps(rec["G_P6B3"], indent=2))
    print(f"receipt: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
