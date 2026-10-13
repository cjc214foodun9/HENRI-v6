"""Phase 2 ingress smoke: measure every gate and write a receipt.

Run at production dimension:

    python scripts/run_ingress_smoke.py --dim 65536 --out receipts/ingress_smoke.json

This exercises the REAL Phase 2 modules on a REAL synthetic field. It is
not a narration of the gates; it measures each one and records the value.

The receipt is written with a sidecar digest, and the digest is computed
over the BYTES written to disk (not over a re-serialized copy). See
`scripts/run_kernel_smoke.py` for why that distinction cost a bug.

Evidence labels:
  OBSERVED  every number below is measured in this run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from henri.binding import bind, unbind  # noqa: E402
from henri.config import N_SLOTS, SLOT_NAMES, WaveConfig  # noqa: E402
from henri.ingress_subspace_router import SubspaceRouter  # noqa: E402
from henri.qfhrr_tokenizer import RING, QFHRRTokenizer  # noqa: E402
from henri.spatial_encoder import CL_GRADES, ContinuousSpatialEncoder  # noqa: E402
from henri.state import WaveState  # noqa: E402


def _field(h: int, w: int, seed: int = 20261012) -> np.ndarray:
    """Deterministic intensity field with nonzero borders."""
    gen = np.random.default_rng(seed)
    base = gen.random((h, w))
    # Guarantee nonzero borders so the unwrapped control can differ.
    base[0, :] = 0.5 + base[0, :]
    base[-1, :] = 0.5 + base[-1, :]
    base[:, 0] = 0.5 + base[:, 0]
    base[:, -1] = 0.5 + base[:, -1]
    return base


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=65536)
    ap.add_argument("--field", type=int, default=32)
    ap.add_argument("--codebook", type=int, default=24)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    t0 = time.perf_counter()
    cfg = WaveConfig(dim=args.dim)
    receipt = {
        "schema": "henri.ingress_smoke.v1",
        "params": {
            "dim": cfg.dim,
            "slot_dim": cfg.slot_span,
            "n_slots": N_SLOTS,
            "field_side": args.field,
            "codebook": args.codebook,
            "ring": RING,
            "cl_grades": CL_GRADES,
            "seed": 20261012,
        },
        "stages": {},
    }
    stages = receipt["stages"]

    # -- layout -------------------------------------------------------
    print(f"[1/7] layout  D={cfg.dim}  D_slot={cfg.slot_span}", file=sys.stderr)
    router = SubspaceRouter(cfg)
    stages["layout"] = {
        "partition": router.partition(),
        "slot_names": list(router.slots),
        "matches_config": list(router.slots) == list(SLOT_NAMES),
    }

    # -- directive 1: spatial encoder ---------------------------------
    print("[2/7] spatial encoder", file=sys.stderr)
    enc = ContinuousSpatialEncoder(cfg, seed=20261012)
    field = _field(args.field, args.field)
    psi_spatial = enc.encode_2d_field(field)

    dx, dy = 3, -2
    psi_wrapped = enc.encode_2d_field(enc.wrapped_shift(field, dx, dy))
    psi_predicted = psi_spatial * enc.translate_phase(dx, dy, args.field, args.field)
    g_ing1_phase = float(np.max(np.abs(psi_wrapped - psi_predicted)))
    g_ing1_mag = float(np.linalg.norm(np.abs(psi_wrapped) - np.abs(psi_spatial)))

    psi_unwrapped = enc.encode_2d_field(enc.unwrapped_shift(field, dx, dy))
    g_ing1_control = float(np.max(np.abs(psi_unwrapped - psi_predicted)))

    stages["spatial_encoder"] = {
        "gate_ing1_phase_residual": g_ing1_phase,
        "gate_ing1_magnitude_residual": g_ing1_mag,
        "gate_ing1_bar": 1e-12,
        "gate_ing1_pass": bool(g_ing1_phase < 1e-12 and g_ing1_mag < 1e-12),
        "control_unwrapped_residual": g_ing1_control,
        "control_must_fail": bool(g_ing1_control > 1e-3),
        "spatial_norm_error": float(abs(np.linalg.norm(psi_spatial) - 1.0)),
        "frequency_summary": enc.frequency_summary(),
    }

    # -- directive 1: blade -------------------------------------------
    print("[3/7] Cl(3,0) blade", file=sys.stderr)
    gen = np.random.default_rng(5)
    n_blocks = enc.n_blocks
    blade = enc.encode_multivector_blade(
        gen.standard_normal(n_blocks),
        gen.standard_normal((n_blocks, 3)),
        gen.standard_normal((n_blocks, 3)),
    )
    stages["multivector_blade"] = {
        "n_blocks": n_blocks,
        "grades": CL_GRADES,
        "length": int(blade.shape[0]),
        "norm_error": float(abs(np.linalg.norm(blade) - 1.0)),
        "grade_order": [row["name"] for row in enc.grade_table()],
    }

    # -- directive 2: qFHRR tokenizer ---------------------------------
    print("[4/7] qFHRR tokenizer", file=sys.stderr)
    tok = QFHRRTokenizer(cfg, seed=20261012, n_categories=256, n_semantics=512, n_instances=4096)
    q_a = tok.encode_token(3, 11, 5)
    q_b = tok.encode_token(20, 2, 90)

    triples = [(i % 64, (i * 7) % 128, i) for i in range(args.codebook)]
    u2 = tok.crosstalk(triples)

    stages["qfhrr_tokenizer"] = {
        "ring": RING,
        "dtype": str(q_a.dtype),
        "gate_ing2_max_modulus_error": tok.unit_modulus_error(q_a),
        "gate_ing2_bar": 1e-15,
        "gate_ing2_pass": bool(tok.unit_modulus_error(q_a) < 1e-15),
        "gate_ing2_norm_error": tok.norm_error(q_a),
        "expected_component_magnitude": float(1.0 / np.sqrt(tok.dim)),
        "gate_ing4_ring_coherence": tok.unbinding_coherence(q_a, q_b),
        "gate_ing4_bitwise_exact": tok.recovery_is_bitwise_exact(q_a, q_b),
        "gate_ing4_bar": 0.9999,
        "gate_u2_mean_squared": u2["mean_squared"],
        "gate_u2_max": u2["max"],
        "gate_u2_bar": 0.12,
        "gate_u2_pass": bool(u2["mean_squared"] <= 0.12),
    }

    # -- directive 2 control: convolution arm must fail ----------------
    print("[5/7] convolution control arm", file=sys.stderr)
    wa = WaveState.random(cfg, seed=101)
    wb = WaveState.random(cfg, seed=202)
    conv_coh = wb.coherence(unbind(bind(wa, wb), wa))
    stages["control_convolution_arm"] = {
        "recovery_coherence": conv_coh,
        "gate_ing4_bar": 0.9999,
        "must_fail": bool(conv_coh < 0.9999),
        "note": "F2 preserved as a test: convolution saturates, ring is exact",
    }

    # -- section 3.2 family path --------------------------------------
    print("[6/7] section 3.2 family tokens", file=sys.stderr)
    pa = tok.to_complex_phasor(tok.encode_family_token(0, 1))
    pb = tok.to_complex_phasor(tok.encode_family_token(0, 2))
    po = tok.to_complex_phasor(tok.encode_family_token(1, 1))

    def _coh(p, q):
        return float(np.real(np.vdot(p, q)) / (np.linalg.norm(p) * np.linalg.norm(q)))

    stages["family_tokens"] = {
        "intra_family": _coh(pa, pb),
        "cross_family": _coh(pa, po),
        "shared_fraction": tok.family_shared_fraction(0),
        "intra_band": [0.35, 0.65],
        "intra_in_band": bool(0.35 <= _coh(pa, pb) <= 0.65),
        "cross_isolated": bool(abs(_coh(pa, po)) < 0.10),
    }

    # -- directive 3: router ------------------------------------------
    print("[7/7] subspace router", file=sys.stderr)
    state = router.assemble_ingress(entity=psi_spatial)
    provided = {"entity": psi_spatial, "action": None, "target": None, "context": None}
    u1_energy = router.unpopulated_energy(state, provided)
    u1_bleed = router.bleed_outside(state, active=[0])
    tail_energy = float(np.sum(np.abs(state.psi[cfg.slot_span :]) ** 2))

    stages["subspace_router"] = {
        "gate_u1_unpopulated_energy": u1_energy,
        "gate_u1_bleed": u1_bleed,
        "gate_ing3_tail_sum": tail_energy,
        "gate_ing3_bar": 0.0,
        "gate_ing3_pass": bool(u1_energy == 0.0 and u1_bleed == 0.0 and tail_energy == 0.0),
        "assembled_norm_error": float(abs(state.norm - 1.0)),
        "slot_energy": [float(x) for x in state.slot_energy()],
        "active_slots": router.active_slots(state),
    }

    # -- end-to-end two-slot assembly ---------------------------------
    gen2 = np.random.default_rng(31)
    context = gen2.standard_normal(cfg.slot_span) + 1j * gen2.standard_normal(cfg.slot_span)
    state2 = router.assemble_ingress(entity=psi_spatial, context=context)
    provided2 = {"entity": psi_spatial, "action": None, "target": None, "context": context}
    stages["two_slot_assembly"] = {
        "unpopulated_energy": router.unpopulated_energy(state2, provided2),
        "norm_error": float(abs(state2.norm - 1.0)),
        "slot_energy": [float(x) for x in state2.slot_energy()],
        "parts_present": sorted(router.disassemble_ingress(state2).keys()),
    }

    receipt["mutual_load"] = {
        "state_norm": float(np.real(np.vdot(state.psi, state.psi))),
        "field_energy": float(np.sum(field**2)),
    }

    receipt["wall_clock_s"] = time.perf_counter() - t0
    receipt["status"] = "COMPLETE"

    # -- write BYTES, hash those bytes --------------------------------
    payload = json.dumps(receipt, indent=2, sort_keys=True)
    body = payload.encode("utf-8")
    digest = hashlib.sha256(body).hexdigest()

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(body)
        back = hashlib.sha256(out_path.read_bytes()).hexdigest()
        out_path.with_suffix(out_path.suffix + ".sha256").write_bytes(
            f"{digest}  {out_path.name}\n".encode("utf-8")
        )
        print(f"wrote {out_path}", file=sys.stderr)
        print(f"sha256 {digest}  readback={'match' if back == digest else 'MISMATCH'}",
              file=sys.stderr)
        receipt["receipt_path"] = str(out_path)
        receipt["sha256"] = digest
        receipt["readback_match"] = bool(back == digest)
        print(json.dumps(receipt, indent=2, sort_keys=True))
    else:
        print(json.dumps(receipt, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    sys.exit(main())
