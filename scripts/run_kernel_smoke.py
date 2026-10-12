"""Run the HENRI v6 kernel end-to-end and write a receipt.

This is the executable evidence for the build. It runs the REAL
pipeline at the REAL dimension and writes every measured number to
`receipts/kernel_smoke.json`.

Usage:
    python scripts/run_kernel_smoke.py [--dim 65536] [--out PATH]

Machine output goes to stdout as JSON. Human progress goes to stderr,
so a consuming `json.load` is never corrupted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from henri.binding import bind, unbind  # noqa: E402
from henri.config import N_SLOTS, WaveConfig  # noqa: E402
from henri.gram import crosstalk, stiefel_retract  # noqa: E402
from henri.koopman import KoopmanOperator  # noqa: E402
from henri.kuramoto import (  # noqa: E402
    KuramotoState,
    lock,
    lock_steps_for_dim,
    order_parameter,
)
from henri.pipeline import build_ingress, run_pipeline  # noqa: E402
from henri.readout import centroid_snap, select  # noqa: E402
from henri.sagnac import (  # noqa: E402
    calibrate_epsilon,
    perturb_with_coherence,
    separation,
)
from henri.state import WaveState  # noqa: E402

SEED = 20261012


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def git_rev() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else "uncommitted"
    except Exception:
        return "unavailable"


def main() -> int:
    ap = argparse.ArgumentParser(description="HENRI v6 kernel smoke run")
    ap.add_argument("--dim", type=int, default=65536,
                    help="complex dimension (theory default 65536)")
    ap.add_argument("--codebook", type=int, default=32,
                    help="codebook size")
    ap.add_argument("--out", type=str,
                    default=str(ROOT / "receipts" / "kernel_smoke.json"))
    ap.add_argument("--quick", action="store_true",
                    help="use dim=1024 for a fast end-to-end check")
    args = ap.parse_args()

    dim = 1024 if args.quick else args.dim
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    receipt: dict = {
        "schema": "henri.v6.kernel_smoke/1",
        "evidence_label": "OBSERVED",
        "scope": (
            "CPU kernel only. No GPU, no ARC task solved, no capability claim."
        ),
        "non_claims": [
            "No benchmark task was attempted or solved.",
            "No photonic substrate figure is measured.",
            "Code health is not intelligence.",
        ],
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "git_rev": git_rev(),
        },
        "params": {"dim": dim, "codebook": args.codebook, "seed": SEED},
        "stages": {},
    }

    t0 = time.perf_counter()

    # -- stage 1: configuration ---------------------------------------
    log(f"[1/8] config: D={dim}")
    cfg = WaveConfig(dim=dim)
    receipt["stages"]["config"] = {
        "dim": cfg.dim,
        "blocks": cfg.blocks,
        "slot_span": cfg.slot_span,
        "n_slots": N_SLOTS,
        "r_koopman": cfg.r_koopman,
        "r_memory": cfg.r_memory,
    }

    # -- stage 2: ingress with the zero-pad rule ----------------------
    log("[2/8] ingress")
    rng = np.random.default_rng(SEED)
    span = cfg.slot_span
    slots = [
        (rng.standard_normal(span) + 1j * rng.standard_normal(span))
        for _ in range(3)
    ] + [None]  # slot 3 explicitly zeroed
    ingress = build_ingress(cfg, slots)
    receipt["stages"]["ingress"] = {
        "norm": ingress.norm,
        "unit_norm_error": ingress.unit_norm_error(),
        "slot_energy": [float(x) for x in ingress.slot_energy()],
        "slot_bleed_active": float(ingress.slot_bleed([0, 1, 2])),
        "slot3_is_exactly_zero": bool(
            np.all(ingress.psi[cfg.slot_slice(3)] == 0.0)
        ),
    }

    # -- stage 3: binding round trip ----------------------------------
    log("[3/8] binding")
    a = WaveState.random(cfg, seed=SEED + 1)
    b = WaveState.random(cfg, seed=SEED + 2)
    recovered = unbind(bind(a, b), a)
    receipt["stages"]["binding"] = {
        "bound_norm_error": bind(a, b).unit_norm_error(),
        "recovery_coherence": b.coherence(recovered),
        "recovery_residual": 1.0 - b.coherence(recovered),
        "note": (
            "coherence saturates near 0.707 = 1/sqrt(2); it does NOT "
            "improve with dimension"
        ),
    }

    # -- stage 4: Koopman dynamics + norm-drift measurement -----------
    log("[4/8] koopman")
    operator = KoopmanOperator.build(cfg, seed=SEED)
    raw_drift = operator.norm_drift(ingress)
    stepped = operator.step(ingress)
    receipt["stages"]["koopman"] = {
        "skew_hermitian": operator.is_skew_hermitian(),
        "is_unitary": operator.is_unitary(),
        "raw_norm_drift": raw_drift,
        "projected_norm_error": stepped.unit_norm_error(),
        "spec_claim_falsified": bool(raw_drift > 1e-9),
    }

    # -- stage 5: Kuramoto locking ------------------------------------
    log("[5/8] kuramoto")
    ku = KuramotoState.from_wave(stepped, rng=rng)
    r_before = order_parameter(ku.phases)[0]
    # The random initial order parameter is r0 ~ 1/sqrt(D), so the time
    # to lock scales as ln(D)/(K/2). A fixed step count calibrated at
    # small D is too short at production D -- this is a real scaling
    # property of the mean-field dynamics, not a harness defect.
    k_coupling, k_dt = 8.0, 5e-3
    k_steps = lock_steps_for_dim(cfg.dim, k_coupling, dt=k_dt)
    ku_locked, trace = lock(ku, coupling=k_coupling, dt=k_dt, steps=k_steps)
    r_after = order_parameter(ku_locked.phases)[0]
    receipt["stages"]["kuramoto"] = {
        "r_before": float(r_before),
        "r_after": float(r_after),
        "r_init_predicted_1_over_sqrt_D": float(1.0 / np.sqrt(cfg.dim)),
        "coupling": k_coupling,
        "dt": k_dt,
        "steps": int(k_steps),
        "elapsed_time": float(k_steps * k_dt),
        "increased": bool(r_after > r_before),
        "lock_threshold": 0.880,
        "locked": bool(r_after >= 0.880),
        "note": (
            "r0 ~ 1/sqrt(D) makes lock time scale as ln(D)/(K/2); "
            "a fixed step count is D-fragile"
        ),
    }

    # -- stage 6: codebook + orthogonalization ------------------------
    log("[6/8] gram / codebook")
    raw_book = (
        rng.standard_normal((args.codebook, cfg.dim))
        + 1j * rng.standard_normal((args.codebook, cfg.dim))
    )
    book = stiefel_retract(raw_book)
    receipt["stages"]["codebook"] = {
        "n_patterns": int(book.shape[0]),
        "crosstalk_before": crosstalk(raw_book),
        "crosstalk_after": crosstalk(book),
        "gate_u2_bound": 0.12,
        "gate_u2_pass": bool(crosstalk(book) <= 0.12),
    }

    # -- stage 7: calibrated Sagnac veto ------------------------------
    log("[7/8] sagnac calibration")
    refs = [WaveState.random(cfg, seed=SEED + 100 + i) for i in range(24)]
    calibration = calibrate_epsilon(
        refs, sigma=3.0, ceiling=0.35, seed=SEED, n_pairs=6000
    )
    axioms = refs[:12]
    gold, bad = [], []
    for ax in axioms:
        # Dimension-invariant construction: a fixed per-component noise
        # scale would give relative perturbations of 0.064 at D=512 but
        # 0.72 at D=65536, destroying the gold pair at production D.
        gold.append((perturb_with_coherence(ax, 0.999, rng), ax))
        bad.append((WaveState.random(cfg, rng=rng), ax))
    stats = separation(gold, bad, calibration)
    receipt["stages"]["sagnac"] = {
        "calibration": calibration.as_dict(),
        "epsilon_is_not_the_retired_literal": bool(
            abs(calibration.epsilon - 0.35) > 1e-12 or calibration.capped
        ),
        "separation": stats,
        "directive3_bar": 0.70,
        "directive3_pass": bool(
            np.isfinite(stats["separation"]) and stats["separation"] >= 0.70
        ),
    }

    # -- stage 8: full pipeline + readout ----------------------------
    log("[8/8] pipeline")
    axiom = WaveState.from_complex(book[0], cfg)
    # Stage 5 scales lock steps with ln(D); stage 8 must use the SAME
    # regime, or the receipt silently mixes a locked and an unlocked
    # phase field at the same measured dimension.
    result = run_pipeline(
        cfg=cfg,
        ingress=ingress,
        codebook=book,
        axiom=axiom,
        calibration=calibration,
        seed=SEED,
        relax_steps=3,
        lock_steps=lock_steps_for_dim(cfg.dim, k_coupling, dt=k_dt),
    )
    selection = select(result.cleaned, book)
    centroids = book[:6]
    receipt["stages"]["pipeline"] = result.as_dict()
    receipt["stages"]["readout"] = {
        "selected_index": selection.index,
        "group_size": selection.group_size,
        "margin": selection.margin,
        "entropy": selection.entropy,
        "floor": selection.meta["floor"],
        "no_floor_smuggled": selection.meta["floor"] is None,
    }
    receipt["stages"]["centroid_snap"] = {
        "index": centroid_snap(result.cleaned.psi, centroids),
        "n_centroids": int(centroids.shape[0]),
    }

    elapsed = time.perf_counter() - t0
    receipt["wall_clock_s"] = elapsed
    receipt["status"] = "COMPLETE"

    # -- write and hash ------------------------------------------------
    # PROVENANCE. The digest must authenticate the bytes on disk. An
    # earlier version hashed the payload and then wrote `receipt_sha256`
    # INTO that payload, so the stored digest could never match the file
    # -- a self-refuting receipt, and the exact defect class the retired
    # register keeps catching.
    #
    # Fix: serialise the payload ONCE, hash those exact bytes, write them
    # verbatim, and emit the digest to stdout plus a sidecar file. The
    # in-file field is deliberately absent, because any digest inside the
    # bytes it covers is wrong by construction.
    payload = json.dumps(receipt, indent=2, sort_keys=True)
    body = payload.encode("utf-8")
    digest = hashlib.sha256(body).hexdigest()

    # Write BYTES, not text. `Path.write_text` applies platform newline
    # translation, so on Windows every `\n` becomes `\r\n` and the file
    # on disk is NOT the byte string that was hashed. Measured: the first
    # run of this check failed with `readback match: False` for exactly
    # that reason. `write_bytes` performs no translation.
    out_path.write_bytes(body)
    digest_path = out_path.with_suffix(out_path.suffix + ".sha256")
    digest_path.write_bytes((digest + "\n").encode("utf-8"))

    # Read back and confirm the digest covers the bytes actually stored.
    on_disk = sha256_file(out_path)
    verified = on_disk == digest

    log(f"[done] {elapsed:.2f}s -> {out_path}")
    log(f"[done] sha256 {digest} (readback match: {verified})")
    if not verified:
        log("[FAIL] receipt digest does not match the bytes on disk")
        return 2

    print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
