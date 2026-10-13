"""Closed-form margin law for the readout plateau.

Owner directive 2026-10-12: formally resolve the production-D readout
plateau recorded UNRESOLVED in docs/report.html.

DERIVATION (stated before the run)
----------------------------------
Codebook rows M_k are unit norm, dim D. Construct the query

    Psi = unit( t * M_j + eps * n ),   n perpendicular to M_j,
                                       t^2 + eps^2 = 1

so Re<M_j, Psi> = t exactly. For any k,

    score_k = Re<M_k, Psi> = t * c_kj + eps * g_k
    c_kj = Re<M_k, M_j>,   g_k = Re<M_k, n> = O(1/sqrt(D))

The target row scores t. The runner-up maximises t*c_kj + eps*g_k, so
with c_max = max_{k != j} c_kj:

    margin = t - t*c_max - eps*max_k g_k
           ~ t * (1 - c_max)   -  O(eps / sqrt(D))          (LAW)

CONSEQUENCES, both testable
---------------------------
L1 plateau. As t -> 0 (a query unrelated to every row) the margin -> 0
   for ANY codebook. The recorded plateau IS the t -> 0 limit. This is
   a statement about the fixture, not about a defect.
L2 crowding boundary. Decisiveness needs t*(1 - c_max) >= bar. A
   structured codebook with large c_max shrinks the margin even as
   t -> 1. At c_max = 0.892, t = 0.99 the law gives 0.107, barely
   above a 0.10 bar.

PRE-REGISTERED DECISION RULE (fixed before the first run)
---------------------------------------------------------
Informative cells are t >= 0.50 with argmax == target row.
LAW_CONFIRMED : max |margin - t*(1 - c_max)| <= 0.05 on EVERY family.
LAW_FALSIFIED : max |residual| > 0.05 on any family.
A cell whose argmax is not the target row is reported NOT_APPLICABLE and
excluded, because the law is conditioned on the target winning.

Unrelated cells (t ~ 0) are reported separately: only the O(eps/sqrt(D))
noise term remains, so the deterministic law is not testable there.

Scope: synthetic geometry; no learned engram; no task attempted; no
capability claim.
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
from henri.gram import stiefel_retract  # noqa: E402
from henri.qfhrr import phasor_vector  # noqa: E402
from henri.readout import entropy_at_beta, select  # noqa: E402
from henri.state import WaveState  # noqa: E402

SEED = 20261012
SIGMAS = (0.05, 0.15, 0.35, 0.60)
TARGETS = (0.50, 0.75, 0.90, 0.99)
RESIDUAL_BAR = 0.05


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
    per = k // 4
    rows = []
    for _ in range(4):
        center = unit(rng.standard_normal(d) + 1j * rng.standard_normal(d))
        for _ in range(per):
            noise = unit(rng.standard_normal(d) + 1j * rng.standard_normal(d))
            rows.append(unit(center + sigma * noise))
    return np.stack(rows)


def stiefel_rows(d: int, k: int, rng) -> np.ndarray:
    """Orthonormal rows: c_max ~ 0 by construction. The clean limit."""
    raw = rng.standard_normal((k, d)) + 1j * rng.standard_normal((k, d))
    return stiefel_retract(raw)


def make_query(rows: np.ndarray, j: int, t: float, cfg, rng) -> WaveState:
    """Exact construction: Re<M_j, Psi> = t."""
    base = rows[j]
    if t <= 0.0:
        return WaveState.random(cfg, rng=rng)
    noise = rng.standard_normal(cfg.dim) + 1j * rng.standard_normal(cfg.dim)
    noise = unit(noise - base * np.vdot(base, noise))
    eps = float(np.sqrt(max(1.0 - t * t, 0.0)) / t)
    return WaveState.from_complex(unit(base + eps * noise), cfg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=65536)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--out", type=str, default="receipts/margin_law.json")
    a = ap.parse_args()

    t0 = time.perf_counter()
    cfg = WaveConfig(dim=a.dim)
    rng = np.random.default_rng(SEED)
    rec: dict = {
        "probe": "margin_law",
        "law": "margin ~= t * (1 - c_max)",
        "seed": SEED,
        "params": {"dim": a.dim, "k": a.k, "residual_bar": RESIDUAL_BAR},
        "decision_rule": "pre-registered in the module docstring before the first run",
        "families": {},
        "cells": [],
    }

    books = {
        "stiefel_orthonormal": stiefel_rows(a.dim, a.k, rng),
        "isotropic_gauss": gauss_rows(a.dim, a.k, rng),
        "isotropic_phasor": phasor_rows(a.dim, a.k, rng),
    }
    for s in SIGMAS:
        books[f"clustered_s{s:.2f}"] = clustered_rows(a.dim, a.k, s, rng)

    for fam, rows in books.items():
        g = np.abs(rows @ rows.conj().T)
        np.fill_diagonal(g, 0.0)
        # c_max for the target row 0.
        c_max = float(g[0].max())
        # Whole-codebook worst pair, for context.
        c_worst = float(g.max())
        rec["families"][fam] = {
            "c_max_target_row0": c_max,
            "c_worst_pair": c_worst,
        }

        # -- informative cells -----------------------------------------
        for t in TARGETS:
            q = make_query(rows, 0, t, cfg, rng)
            r = select(q, rows)
            t_actual = float(np.real(np.vdot(rows[0], q.psi)))
            pred = t_actual * (1.0 - c_max)
            applies = int(r.index) == 0
            cells = rec["cells"]
            cells.append(
                {
                    "family": fam,
                    "kind": "informative",
                    "t_requested": t,
                    "t_actual": t_actual,
                    "c_max": c_max,
                    "argmax": int(r.index),
                    "margin": float(r.margin),
                    "predicted_margin": float(pred),
                    "residual": float(r.margin - pred),
                    "applicable": bool(applies),
                    "entropy": float(r.entropy),
                    "entropy_beta": float(
                        entropy_at_beta(np.real(rows @ np.conj(q.psi)), 26.1)
                    ),
                    "tie_group_size": int(r.group_size),
                }
            )

        # -- unrelated cell --------------------------------------------
        q = make_query(rows, 0, 0.0, cfg, rng)
        r = select(q, rows)
        rec["cells"].append(
            {
                "family": fam,
                "kind": "unrelated",
                "t_requested": 0.0,
                "t_actual": float(np.real(np.vdot(rows[0], q.psi))),
                "c_max": c_max,
                "argmax": int(r.index),
                "margin": float(r.margin),
                "predicted_margin": None,
                "residual": None,
                "applicable": False,
                "entropy": float(r.entropy),
                "tie_group_size": int(r.group_size),
            }
        )

    # -- verdict --------------------------------------------------------
    inf = [c for c in rec["cells"] if c["kind"] == "informative" and c["applicable"]]
    cells_all = rec["cells"]
    nonapp = [
        c for c in cells_all if c["kind"] == "informative" and not c["applicable"]
    ]
    resid = np.array([c["residual"] for c in inf], dtype=np.float64)
    max_abs = float(np.max(np.abs(resid))) if resid.size else float("nan")

    y = np.array([c["margin"] for c in inf], dtype=np.float64)
    x = np.array([c["predicted_margin"] for c in inf], dtype=np.float64)
    if x.size > 1 and x.std() > 0:
        r2 = float(np.corrcoef(x, y)[0, 1] ** 2)
    else:
        r2 = float("nan")

    per_family_max = {}
    for fam in books:
        vals = [abs(c["residual"]) for c in inf if c["family"] == fam]
        per_family_max[fam] = float(max(vals)) if vals else None

    if not inf:
        verdict = "NOT_APPLICABLE"
    elif max_abs <= RESIDUAL_BAR:
        verdict = "LAW_CONFIRMED"
    else:
        verdict = "LAW_FALSIFIED"

    unrel = [c for c in cells_all if c["kind"] == "unrelated"]
    rec["unrelated_summary"] = {
        "max_margin": float(max(c["margin"] for c in unrel)) if unrel else None,
        "all_plateau": bool(all(c["entropy"] >= 0.99 for c in unrel)) if unrel else None,
    }
    rec["law_fit"] = {
        "n_applicable": len(inf),
        "n_not_applicable": len(nonapp),
        "max_abs_residual": max_abs,
        "r2_margin_vs_prediction": r2,
        "per_family_max_abs_residual": per_family_max,
        "residual_bar": RESIDUAL_BAR,
    }
    rec["crowding_boundary"] = {
        "note": "margin falls below 0.10 at t=0.90 when 0.90*(1-c_max) < 0.10, i.e. c_max > 0.889",
        "families_below_bar_at_t0.90": sorted(
            {
                c["family"]
                for c in inf
                if c["t_requested"] == 0.90 and c["margin"] < 0.10
            }
        ),
    }
    rec["verdict"] = verdict
    rec["scope_limits"] = [
        "Synthetic geometry; no learned engram was used.",
        "No task was attempted or solved. No capability claim follows.",
        "The law is conditioned on the target row winning; else NOT_APPLICABLE.",
        "The unrelated cells are excluded from the residual test by construction.",
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

    for c in rec["cells"]:
        if c["kind"] == "unrelated":
            print(
                "%-22s UNRELATED   t=%+.5f margin=%+.6f ent=%.6f"
                % (c["family"], c["t_actual"], c["margin"], c["entropy"]),
                file=sys.stderr,
            )
        else:
            print(
                "%-22s t=%.2f c_max=%.4f margin=%+.6f pred=%+.6f resid=%+.6f %s"
                % (
                    c["family"], c["t_requested"], c["c_max"], c["margin"],
                    c["predicted_margin"], c["residual"],
                    "OK" if c["applicable"] else "N/A",
                ),
                file=sys.stderr,
            )
    print("law_fit: " + json.dumps(rec["law_fit"], sort_keys=True), file=sys.stderr)
    print(
        "crowding: " + json.dumps(rec["crowding_boundary"], sort_keys=True),
        file=sys.stderr,
    )
    print(f"verdict={verdict} sha256={digest} readback={rb == digest}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
