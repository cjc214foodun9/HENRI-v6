"""Additive resolution probe for the production-D readout plateau.

Supersedes the verdict in `receipts/readout_geometry_full_D.json`
(v1 = STILL_UNRESOLVED). The v1 receipt is NOT edited. v1 returned
STILL_UNRESOLVED because its plateau predicate required
`entropy >= 0.99` on EVERY unrelated cell, and one cell
(clustered / cleaned, sigma=0.35) read 0.969301. That single cell is
itself mechanistic evidence, and this probe tests it directly.

MECHANISM UNDER TEST
--------------------
M1 mean-attractor. For a query equidistant from every row of an
   ISOTROPIC codebook, the CCCP fixed point is the codebook mean:
       cos(cleaned, mean(rows)) ~ 1
   and the selection softmax is (near) uniform. The plateau is then
   CORRECT dynamics, not a readout defect.
M2 decisiveness. The same readout is decisive the moment the query
   carries information, on every family.
M3 metric scale. The reported `entropy ~ 1.000000` is partly a
   UNIT-TEMPERATURE artifact. `s - max(s)` for coherence-scale scores
   lies in [-2, 0], so with M=32 the softmax cannot fall below ~0.94.
   `entropy_at_beta(26.1)` separates decisive (~0) from flat (~0.99).
M4 structure lifts the query off the mean. For a STRUCTURED codebook
   an unrelated query is not equidistant; the fixed point is pulled
   toward the local cluster, so cos(cleaned, mean) is well below 1 and
   the legacy entropy dips below 0.99. This is why v1's predicate
   failed on exactly one cell.

PRE-REGISTERED DECISION RULE (fixed before the first run)
---------------------------------------------------------
RESOLVED_GEOMETRY iff ALL of:
  A. mean-attractor: cos(cleaned, normalized mean) >= 0.99 for BOTH
     isotropic families on an unrelated query.
  B. decisiveness: for t >= 0.90, raw arm, argmax == target row AND
     margin >= 0.10 on ALL THREE families.
  C. metric: legacy entropy >= 0.99 on the unrelated cells while
     entropy_at_beta(26.1) <= 0.05 on the t >= 0.90 cells.
RESOLVED_DEFECT iff B fails on all three families.
STILL_UNRESOLVED otherwise. Every cell is recorded verbatim.

KILL TESTS (must be able to fail)
---------------------------------
K1 cos(cleaned, mean) < 0.99 on any isotropic family -> M1 refuted.
K2 margin at t = 0.99 < 0.50 on any family -> M2 falsified.
K3 the clustered sigma sweep must produce at least one sigma whose
   margin at t = 0.99 falls below 0.10, or the crowding frontier is
   not exercised and the sweep is reported vacuous.

SCOPE. Synthetic geometry. No learned engram. No task attempted. No
capability claim. The end-to-end pipeline is not re-run here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from henri.config import WaveConfig  # noqa: E402
from henri.hopfield import cleanup  # noqa: E402
from henri.qfhrr import phasor_vector  # noqa: E402
from henri.readout import entropy_at_beta, normalized_entropy, select  # noqa: E402
from henri.state import WaveState  # noqa: E402

SEED = 20261012
BETA = 26.1
SIGMAS = (0.05, 0.15, 0.35, 0.60)


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n < 1e-30:
        raise ValueError("zero or non-finite vector")
    return v / n


def gauss_rows(d: int, k: int, rng) -> np.ndarray:
    return np.stack(
        [unit(rng.standard_normal(d) + 1j * rng.standard_normal(d)) for _ in range(k)]
    )


def phasor_rows(d: int, k: int, rng) -> np.ndarray:
    return np.stack([unit(phasor_vector(d, rng)) for _ in range(k)])


def clustered_rows(d: int, k: int, sigma: float, rng) -> np.ndarray:
    """4 clusters of k//4. Within-cluster coherence grows with sigma."""
    per = k // 4
    rows = []
    for _ in range(4):
        center = unit(rng.standard_normal(d) + 1j * rng.standard_normal(d))
        for _ in range(per):
            noise = unit(rng.standard_normal(d) + 1j * rng.standard_normal(d))
            rows.append(unit(center + sigma * noise))
    return np.stack(rows)


def informative(rows: np.ndarray, j: int, t: float, cfg, rng) -> WaveState:
    """Exact construction: cos(base, result) = t.

    Noise is projected orthogonal to base first, so
    cos = 1/sqrt(1+eps^2) exactly and eps = sqrt(1-t^2)/t.
    """
    base = rows[j]
    noise = rng.standard_normal(cfg.dim) + 1j * rng.standard_normal(cfg.dim)
    noise = unit(noise - base * np.vdot(base, noise))
    eps = float(np.sqrt(max(1.0 - t * t, 0.0)) / t)
    return WaveState.from_complex(unit(base + eps * noise), cfg)


def cell(rows: np.ndarray, wave: WaveState, arm: str, tau: float | None) -> dict:
    """One readout measurement, legacy and beta-scaled entropy both."""
    work = wave
    if arm == "cleaned":
        work, _ = cleanup(wave, rows, tau=tau, iters=8)
    s = np.real(rows @ np.conj(work.psi))
    r = select(work, rows)
    cos_mean = abs(complex(np.vdot(work.psi, unit(rows.mean(axis=0)))))
    return {
        "argmax": int(r.index),
        "margin": float(r.margin),
        "legacy_entropy": float(r.entropy),
        "entropy_beta": float(entropy_at_beta(s, BETA)),
        "tie_group_size": int(r.group_size),
        "cos_with_codebook_mean": float(cos_mean),
        "score_std": float(s.std()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=65536)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--out", type=str, default="receipts/readout_plateau_resolution.json")
    a = ap.parse_args()

    t0 = time.perf_counter()
    cfg = WaveConfig(dim=a.dim)
    rng = np.random.default_rng(SEED)
    rec: dict = {
        "probe": "readout_plateau_resolution",
        "supersedes_verdict_in": "receipts/readout_geometry_full_D.json",
        "seed": SEED,
        "params": {"dim": a.dim, "k": a.k, "beta": BETA, "sigmas": list(SIGMAS)},
        "decision_rule": "pre-registered in the module docstring before the first run",
        "cells": [],
        "sweep": [],
    }

    def emit(tag: str, fam: str, rows, wave, arm, tau, **extra) -> dict:
        c = cell(rows, wave, arm, tau)
        c.update({"test": tag, "family": fam, "arm": arm, "tau": tau, **extra})
        rec["cells"].append(c)
        return c

    # ---- codebooks ---------------------------------------------------
    books = {
        "isotropic_gauss": gauss_rows(a.dim, a.k, rng),
        "isotropic_phasor": phasor_rows(a.dim, a.k, rng),
        "clustered": clustered_rows(a.dim, a.k, 0.35, rng),
    }
    rec["codebooks"] = {}
    for name, rows in books.items():
        g = rows @ rows.conj().T
        off = np.abs(g[np.triu_indices(len(rows), k=1)])
        rec["codebooks"][name] = {
            "coherence_mean": float(off.mean()),
            "coherence_std": float(off.std()),
            "coherence_max": float(off.max()),
        }

    # ---- A / M1: mean-attractor on unrelated queries -----------------
    for fam in ("isotropic_gauss", "isotropic_phasor", "clustered"):
        rows = books[fam]
        q = WaveState.random(cfg, rng=rng)
        emit("A_mean_attractor", fam, rows, q, "raw", None)
        emit("A_mean_attractor", fam, rows, q, "cleaned", 1.0 / BETA)

    # ---- B / M2: decisiveness on informative queries -----------------
    for fam in ("isotropic_gauss", "isotropic_phasor", "clustered"):
        rows = books[fam]
        for t in (0.50, 0.90, 0.99):
            q = informative(rows, 0, t, cfg, rng)
            emit("B_decisive", fam, rows, q, "raw", None, target_row=0, target_coherence=t)

    # ---- D / M4: crowding frontier (clustered sigma sweep) -----------
    for sigma in SIGMAS:
        rows = clustered_rows(a.dim, a.k, sigma, rng)
        q = informative(rows, 0, 0.99, cfg, rng)
        c = cell(rows, q, "raw", None)
        g = rows @ rows.conj().T
        off = np.abs(g[np.triu_indices(len(rows), k=1)])
        rec["sweep"].append(
            {"sigma": sigma, "coherence_max": float(off.max()), **c}
        )

    # ---- verdict ------------------------------------------------------
    cells = rec["cells"]
    iso = [c for c in cells if c["test"] == "A_mean_attractor"
           and c["family"].startswith("isotropic") and c["arm"] == "cleaned"]
    cond_a = bool(iso) and all(c["cos_with_codebook_mean"] >= 0.99 for c in iso)

    dec = [c for c in cells if c["test"] == "B_decisive" and c["target_coherence"] >= 0.90]
    fams_dec = sorted({
        c["family"] for c in dec
        if c["argmax"] == c["target_row"] and c["margin"] >= 0.10
    })
    cond_b = len(fams_dec) == 3

    unrel = [c for c in cells if c["test"] == "A_mean_attractor"]
    cond_c = (
        all(c["legacy_entropy"] >= 0.99 for c in unrel)
        and all(c["entropy_beta"] <= 0.05 for c in dec)
    )

    if cond_a and cond_b and cond_c:
        verdict = "RESOLVED_GEOMETRY"
    elif not fams_dec:
        verdict = "RESOLVED_DEFECT"
    else:
        verdict = "STILL_UNRESOLVED"

    k1 = all(c["cos_with_codebook_mean"] >= 0.99 for c in iso) if iso else False
    k2 = all(c["margin"] >= 0.50 for c in dec)
    k3 = any(s["margin"] < 0.10 for s in rec["sweep"])

    rec["verdict"] = verdict
    rec["conditions"] = {"A_mean_attractor": cond_a, "B_decisive": cond_b, "C_metric": cond_c}
    rec["families_decisive"] = fams_dec
    rec["kill_tests"] = {
        "K1_mean_attractor_holds": bool(k1),
        "K2_margin_ge_0.50_at_t0.99": bool(k2),
        "K3_crowding_frontier_exercised": bool(k3),
    }
    rec["scope_limits"] = [
        "Synthetic geometry; no learned engram was used.",
        "The end-to-end pipeline was not re-run with a structured codebook.",
        "No task was attempted or solved. No capability claim follows.",
        "The v1 receipt is preserved unedited as an additive record.",
    ]
    rec["wall_clock_s"] = time.perf_counter() - t0
    rec["status"] = "COMPLETE"

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(rec, indent=2, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(body).hexdigest()
    out.write_bytes(body)
    out.with_suffix(out.suffix + ".sha256").write_bytes((digest + "\n").encode())
    rb = hashlib.sha256(out.read_bytes()).hexdigest()

    for c in cells:
        print(
            "%-16s %-15s %-8s arg=%-3s margin=%+.6f legacy=%.6f beta=%.6f cosmean=%.6f"
            % (c["family"], c["test"], c["arm"], c["argmax"], c["margin"],
               c["legacy_entropy"], c["entropy_beta"], c["cos_with_codebook_mean"]),
            file=sys.stderr,
        )
    print("sweep: " + " | ".join(
        "sigma=%.2f maxcoh=%.3f margin=%.4f" % (s["sigma"], s["coherence_max"], s["margin"])
        for s in rec["sweep"]), file=sys.stderr)
    print("conditions: " + json.dumps(rec["conditions"], sort_keys=True), file=sys.stderr)
    print("kill_tests: " + json.dumps(rec["kill_tests"], sort_keys=True), file=sys.stderr)
    print(f"verdict={verdict} sha256={digest} readback={rb == digest}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
