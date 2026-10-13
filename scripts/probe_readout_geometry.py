"""Discriminating probe for the production-D readout plateau.

Objective (owner directive 2026-10-12): formally resolve the UNRESOLVED
readout plateau recorded in docs/report.html -- margin 6.8e-05 and
normalized entropy 1.000000 at D=65536 against an isotropic random
codebook.

Hypotheses under test
---------------------
H_plateau: an equidistant query drives the selection softmax uniform,
so the Hopfield CCCP fixed point is the MEAN PATTERN and every score is
nearly equal. Under H_plateau the plateau is CORRECT dynamics for a
degenerate query, not a readout defect.
H_defect: the selection mechanism cannot discriminate even when the
query carries information about a codebook row.

Discriminating design
---------------------
Cross three factors.

Codebook families (K rows, all unit norm):
  isotropic_gauss   unit-normalized complex Gaussian rows
  isotropic_phasor  unit-modulus phase rows (directive-2 encoding)
  clustered         4 clusters; within-cluster rows are
                    unit(center + sigma * unit(noise)), sigma = 0.35,
                    giving hard negatives at controlled coherence

Query types:
  unrelated      a fresh random wave with no codebook overlap. This is
                 the plateau-reproduction cell.
  noisy_t{T}     unit(base + eps * orthogonal_noise) with the EXACT
                 relation cos = 1/sqrt(1 + eps^2), so
                 eps = sqrt(1 - T^2) / T. T in {0.50, 0.90, 0.99}.
  bound_pair     target unit(pbind(a, b)); foils share exactly one
                 factor: unit(pbind(a, c)) and unit(pbind(d, b)).

Readout arms:
  raw       select() on the query directly
  cleaned   hopfield cleanup() then select()

Tau sweep: the cleaned arm runs tau in {0.05, 0.5, 1.0, 4.0} on
T in {0.90, 0.99}. The retired repository's T* = 0.038316 was fitted on
a DIFFERENT readout (a functor-score fit over 60 tasks), so the transfer
is MEASURED here rather than inherited.

PRE-REGISTERED DECISION RULE (fixed before the first run)
---------------------------------------------------------
A cell is a PLATEAU cell when entropy >= 0.99 AND margin < 0.01.
A cell is DECISIVE when argmax == target_row AND margin >= 0.10.

RESOLVED_GEOMETRY : every unrelated cell is a plateau cell, AND the
                    noisy_t0.90 and noisy_t0.99 raw cells are decisive
                    on at least 2 of the 3 families.
RESOLVED_DEFECT   : no noisy_t0.90 or noisy_t0.99 raw cell is decisive
                    on any family.
STILL_UNRESOLVED  : anything else. Every cell is recorded verbatim.

Kill test: the clustered family's maximum within-cluster coherence must
exceed 0.05. If it does not, the crowding leg of the design is vacuous
and the probe records that rather than reporting a pass.

No capability claim follows from any verdict. This probe measures
selection geometry and nothing else.
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
from henri.qfhrr import pbind, phasor_vector  # noqa: E402
from henri.readout import select  # noqa: E402
from henri.state import WaveState  # noqa: E402

SEED = 20261012
CLUSTER_SIGMA = 0.35
TAUS = (0.05, 0.5, 1.0, 4.0)


def unit(v: np.ndarray) -> np.ndarray:
    """Normalize to unit norm. Rejects the zero vector."""
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n < 1e-30:
        raise ValueError("cannot normalize a zero or non-finite vector")
    return v / n


def make_family(name: str, cfg: WaveConfig, k: int, rng: np.random.Generator):
    """Build a unit-norm codebook. Returns (rows, meta)."""
    d = cfg.dim
    if name == "isotropic_gauss":
        rows = np.stack(
            [unit(rng.standard_normal(d) + 1j * rng.standard_normal(d)) for _ in range(k)]
        )
        return rows, {}
    if name == "isotropic_phasor":
        rows = np.stack([unit(phasor_vector(d, rng)) for _ in range(k)])
        return rows, {}
    if name == "clustered":
        n_clusters = 4
        per = k // n_clusters
        rows, cluster_of = [], []
        for c in range(n_clusters):
            center = unit(rng.standard_normal(d) + 1j * rng.standard_normal(d))
            for _ in range(per):
                noise = unit(rng.standard_normal(d) + 1j * rng.standard_normal(d))
                rows.append(unit(center + CLUSTER_SIGMA * noise))
                cluster_of.append(c)
        return np.stack(rows), {
            "sigma": CLUSTER_SIGMA,
            "n_clusters": n_clusters,
            "cluster_of": cluster_of,
        }
    raise ValueError(f"unknown family {name!r}")


def offdiag_stats(rows: np.ndarray) -> dict:
    """Magnitude summary of the off-diagonal coherences."""
    gram = rows @ rows.conj().T
    iu = np.triu_indices(len(rows), k=1)
    off = np.abs(gram[iu])
    return {
        "mean": float(off.mean()),
        "std": float(off.std()),
        "max": float(off.max()),
    }


def noisy_wave(
    rows: np.ndarray, j: int, target: float, cfg: WaveConfig, rng: np.random.Generator
) -> WaveState:
    """Unit wave with coherence EXACTLY ~target against rows[j].

    The noise is projected orthogonal to the base row first, so
    `cos(base, result) = 1 / sqrt(1 + eps^2)` holds with no first-order
    approximation. Solving for the target gives `eps = sqrt(1-T^2)/T`.
    """
    d = cfg.dim
    base = rows[j]
    noise = rng.standard_normal(d) + 1j * rng.standard_normal(d)
    noise = unit(noise - base * np.vdot(base, noise))
    eps = float(np.sqrt(max(1.0 - target * target, 0.0)) / target)
    return WaveState.from_complex(unit(base + eps * noise), cfg)


def measure(
    rows: np.ndarray, wave: WaveState, arm: str, tau: float | None
) -> dict:
    """One cell. Returns the compact selection statistics."""
    if arm == "cleaned":
        wave, _ = cleanup(wave, rows, tau=tau, iters=8)
    elif arm != "raw":
        raise ValueError(f"unknown arm {arm!r}")
    r = select(wave, rows)
    return {
        "argmax": int(r.index),
        "margin": float(r.margin),
        "entropy": float(r.entropy),
        "group_size": int(r.group_size),
        "top_score": float(r.top_score),
        "second_score": float(r.second_score),
    }


def is_plateau(cell: dict) -> bool:
    """Pre-registered plateau predicate."""
    return cell["entropy"] >= 0.99 and cell["margin"] < 0.01


def is_decisive(cell: dict) -> bool:
    """Pre-registered decisive predicate, against the labelled target."""
    return cell["argmax"] == cell["target_row"] and cell["margin"] >= 0.10


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=65536)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--out", type=str, default="receipts/readout_geometry.json")
    args = ap.parse_args()
    if args.k % 4 != 0:
        raise SystemExit("--k must be divisible by 4 for the clustered family")

    t0 = time.perf_counter()
    cfg = WaveConfig(dim=args.dim)
    rng = np.random.default_rng(SEED)

    receipt = {
        "probe": "readout_geometry",
        "seed": SEED,
        "params": {"dim": args.dim, "k": args.k},
        "decision_rule": {
            "plateau_cell": "entropy >= 0.99 and margin < 0.01",
            "decisive_cell": "argmax == target_row and margin >= 0.10",
            "pre_registered": "fixed in the module docstring before the first run",
        },
        "codebooks": {},
        "cells": [],
    }

    families = ["isotropic_gauss", "isotropic_phasor", "clustered"]
    books = {}
    for fam in families:
        rows, meta = make_family(fam, cfg, args.k, rng)
        books[fam] = rows
        stats = offdiag_stats(rows)
        receipt["codebooks"][fam] = {**meta, "coherence": stats}
        print(
            f"[codebook] {fam:<17} mean={stats['mean']:.5f} "
            f"std={stats['std']:.5f} max={stats['max']:.5f}",
            file=sys.stderr,
        )

    kill_inputs = {}

    for fam in families:
        rows = books[fam]
        target_row = 0
        clustered = fam == "clustered"
        if clustered:
            # Hard-negative witness: max coherence of row 0 to its siblings.
            sib = np.abs(rows[1:] @ np.conj(rows[0]))
            kill_inputs["max_sibling_coherence"] = float(sib.max())

        # -- unrelated: plateau reproduction ---------------------------
        q = WaveState.random(cfg, rng=rng)
        for arm in ("raw", "cleaned"):
            cell = measure(rows, q, arm, tau=None)
            receipt["cells"].append(
                {
                    "family": fam,
                    "query": "unrelated",
                    "arm": arm,
                    "tau": None,
                    "target_row": None,
                    **cell,
                }
            )

        # -- noisy target: raw arm, exact construction -----------------
        for t in (0.50, 0.90, 0.99):
            q = noisy_wave(rows, target_row, t, cfg, rng)
            cell = measure(rows, q, "raw", tau=None)
            receipt["cells"].append(
                {
                    "family": fam,
                    "query": f"noisy_t{t:.2f}",
                    "arm": "raw",
                    "tau": None,
                    "target_row": target_row,
                    "requested_coherence": t,
                    "achieved_coherence": float(
                        np.real(np.vdot(rows[target_row], q.psi))
                    ),
                    **cell,
                }
            )

        # -- noisy target: cleaned arm, tau sweep ----------------------
        for t in (0.90, 0.99):
            q = noisy_wave(rows, target_row, t, cfg, rng)
            for tau in TAUS:
                cell = measure(rows, q, "cleaned", tau=tau)
                receipt["cells"].append(
                    {
                        "family": fam,
                        "query": f"noisy_t{t:.2f}",
                        "arm": "cleaned",
                        "tau": tau,
                        "target_row": target_row,
                        **cell,
                    }
                )

        # -- bound pair: exact phasor unbinding ------------------------
        a = phasor_vector(args.dim, rng)
        b = phasor_vector(args.dim, rng)
        c = phasor_vector(args.dim, rng)
        e = phasor_vector(args.dim, rng)
        target = unit(pbind(a, b))
        book_bp = np.stack([target, unit(pbind(a, c)), unit(pbind(e, b))])
        q = WaveState.from_complex(target, cfg)
        for arm in ("raw", "cleaned"):
            cell = measure(book_bp, q, arm, tau=None)
            receipt["cells"].append(
                {
                    "family": fam,
                    "query": "bound_pair",
                    "arm": arm,
                    "tau": None,
                    "target_row": 0,
                    **cell,
                }
            )

    # -- verdict ---------------------------------------------------------
    cells = receipt["cells"]
    unrelated = [c for c in cells if c["query"] == "unrelated"]
    decisive_cells = [
        c
        for c in cells
        if c["query"] in ("noisy_t0.90", "noisy_t0.99")
        and c["arm"] == "raw"
        and is_decisive(c)
    ]
    plateau_ok = all(is_plateau(c) for c in unrelated)
    families_decisive = sorted({c["family"] for c in decisive_cells})

    if plateau_ok and len(families_decisive) >= 2:
        verdict = "RESOLVED_GEOMETRY"
    elif not families_decisive:
        verdict = "RESOLVED_DEFECT"
    else:
        verdict = "STILL_UNRESOLVED"

    receipt["kill_test"] = {
        "requirement": "clustered max within-cluster coherence > 0.05",
        "measured": kill_inputs.get("max_sibling_coherence"),
        "passed": bool(kill_inputs.get("max_sibling_coherence", 0.0) > 0.05),
    }
    receipt["decision_inputs"] = {
        "all_unrelated_are_plateau_cells": bool(plateau_ok),
        "families_decisive_raw": families_decisive,
        "n_cells": len(cells),
    }
    receipt["verdict"] = verdict
    receipt["scope_limits"] = [
        "The end-to-end pipeline was not re-run with a structured codebook.",
        "The geometry is synthetic; no learned engram was used.",
        "No task was attempted or solved. No capability claim follows.",
    ]

    receipt["wall_clock_s"] = time.perf_counter() - t0
    receipt["status"] = "COMPLETE"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(receipt, indent=2, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(body).hexdigest()
    out.write_bytes(body)
    out.with_suffix(out.suffix + ".sha256").write_bytes(
        (digest + "\n").encode("utf-8")
    )
    readback = hashlib.sha256(out.read_bytes()).hexdigest()

    print(f"[done] {receipt['wall_clock_s']:.2f}s -> {out}", file=sys.stderr)
    print(f"[done] verdict={verdict}", file=sys.stderr)
    print(f"[done] sha256={digest} readback_match={readback == digest}", file=sys.stderr)
    print(f"[done] families_decisive={families_decisive}", file=sys.stderr)
    print(f"[done] kill_test={receipt['kill_test']['passed']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
