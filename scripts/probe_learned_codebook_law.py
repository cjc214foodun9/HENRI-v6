"""Learned-codebook test of the readout margin law.

Additive to `scripts/probe_margin_law.py`, which tested SYNTHETIC draws
and returned LAW_CONFIRMED: R^2 0.9999810, max absolute residual
0.004947, 28 of 28 informative cells at D = 65536.

QUESTION (owner directive 2026-10-12)
Does `margin ~= t (1 - c_max)` survive on LEARNED engram codebooks, or
does it hold only for the random draws that produced it?

THE LAW AND ITS STATED ASSUMPTIONS
The derivation in probe_margin_law.py assumes each codebook row `M_k`
has UNIT NORM. It then writes

    score_k = Re<M_k, Psi> = t * c_kj + eps * g_k
    c_kj = Re<M_k, M_j>,   g_k = Re<M_k, n> = O(1/sqrt(D))

and concludes `margin ~= t (1 - c_max)` with `c_max = max_{k != j} c_kj`.

Nothing in that chain mentions HOW a row was produced. If the derivation
is correct, the law is architecture-independent and must hold for learned
rows too. If it fails for learned rows, either the derivation is wrong or
an unstated assumption (row independence) is doing real work. Both
outcomes are informative; neither is assumed here.

PRE-REGISTERED DECISION RULE (fixed before the first run)
Informative cell: `t >= 0.50` with `argmax == target row`.
LAW_CONFIRMED : `max |residual| <= 0.05` on EVERY unit-norm learned family.
LAW_FALSIFIED : `max |residual| > 0.05` on any unit-norm learned family.
CONTROL       : the deliberately NON-unit-norm family must exceed 0.05.
                If it does not, the unit-norm assumption is not load
                bearing and the derivation needs revision.

THE LEARNER
The production learner of this repository: `henri.memory.DeltaMemory`,
the rank-r gated delta rule `S = V K^H`. Codebook rows come from
`mem.read(key)`, so they are LEARNED rather than drawn.

DISCLOSURE, stated plainly
The learning TASK is generated pairs. These are learned engrams over
generated data, NOT engrams over a real-world corpus. The contrast the
directive asks about is exactly this: random draw versus learner output.
No real corpus is claimed, and no task is attempted.

Scope: no task solved; no capability claim.
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
from henri.memory import DeltaMemory  # noqa: E402
from henri.readout import select  # noqa: E402
from henri.state import WaveState  # noqa: E402

SEED = 20261012
TARGETS = (0.50, 0.75, 0.90, 0.99)
RESIDUAL_BAR = 0.05


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n < 1e-30:
        raise ValueError("zero or non-finite vector")
    return v / n


def crand(dim: int, k: int, rng) -> np.ndarray:
    """K complex unit rows, isotropic."""
    return np.stack(
        [unit(rng.standard_normal(dim) + 1j * rng.standard_normal(dim))
         for _ in range(k)]
    )


def clustered_values(dim: int, k: int, sigma: float, rng) -> np.ndarray:
    """Values drawn around 4 centers. Produces real c_max > 0."""
    per = max(k // 4, 1)
    rows = []
    for _ in range(4):
        center = unit(rng.standard_normal(dim) + 1j * rng.standard_normal(dim))
        for _ in range(per):
            noise = unit(rng.standard_normal(dim) + 1j * rng.standard_normal(dim))
            rows.append(unit(center + sigma * noise))
    while len(rows) < k:
        rows.append(rows[-1])
    return np.stack(rows[:k])


def learn_codebook(
    dim: int,
    k: int,
    rank: int,
    lam: float,
    rng,
    targets: np.ndarray,
    epochs: int = 2,
) -> tuple:
    """Train DeltaMemory on (key, value) pairs; return LEARNED rows.

    Returns `(keys, recalled_raw, memory_updates)`. `recalled_raw` has the
    memory's native magnitude -- the caller decides whether to normalize,
    which is how the un-normalized control is built.
    """
    mem = DeltaMemory.build(dim, rank, lam=lam)
    keys = crand(dim, k, rng)
    for _ in range(int(epochs)):
        for kk, vv in zip(keys, targets):
            mem.write(kk, vv, beta=1.0)
    recalled = np.stack([mem.read(kk) for kk in keys])
    return keys, recalled, mem.updates


def make_query(rows: np.ndarray, j: int, t: float, cfg, rng) -> WaveState:
    """Exact construction: Re<M_j, Psi> = t, with n perpendicular to M_j."""
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
    ap.add_argument("--out", type=str, default="receipts/learned_codebook_law.json")
    ap.add_argument("--epochs", type=int, default=2)
    a = ap.parse_args()

    t0 = time.perf_counter()
    cfg = WaveConfig(dim=a.dim)
    rng = np.random.default_rng(SEED)

    rec: dict = {
        "probe": "learned_codebook_law",
        "law": "margin ~= t * (1 - c_max)",
        "seed": SEED,
        "learner": "henri.memory.DeltaMemory (rank-r gated delta rule S = V K^H)",
        "params": {
            "dim": a.dim, "epochs": a.epochs, "residual_bar": RESIDUAL_BAR,
        },
        "decision_rule": "pre-registered in the module docstring before the first run",
        "disclosure": (
            "Learned engrams over GENERATED pairs. Not a real-world corpus. "
            "No task attempted or solved."
        ),
        "families": {},
        "cells": [],
    }

    # -- build learned codebooks ----------------------------------------
    # well-fit:   K < rank  -> learner can store the pairs
    # stressed:   K > rank  -> genuine approximation, rows differ from targets
    # clustered:  structured targets -> real c_max > 0
    spec = [
        ("learned_wellfit",   32, 64, 1.0, "random"),
        ("learned_stressed", 128, 32, 1.0, "random"),
        ("learned_clustered", 32, 64, 1.0, "clustered"),
    ]
    books: dict = {}
    meta: dict = {}
    for name, k, rank, lam, style in spec:
        if style == "clustered":
            targets = clustered_values(a.dim, k, 0.30, rng)
        else:
            targets = crand(a.dim, k, rng)
        keys, recalled, upd = learn_codebook(
            a.dim, k, rank, lam, rng, targets, epochs=a.epochs
        )
        rows = np.stack([unit(r) for r in recalled])
        books[name] = rows
        # How far did learning move from the target values? A real measure
        # of whether the codebook is LEARNED rather than copied.
        align = np.array([abs(np.vdot(rows[i], targets[i])) for i in range(k)])
        meta[name] = {
            "k": k, "rank": rank, "epochs": a.epochs,
            "memory_updates": int(upd),
            "target_alignment_mean": float(align.mean()),
            "target_alignment_min": float(align.min()),
            "recall_norm_min": float(np.linalg.norm(recalled, axis=1).min()),
            "recall_norm_max": float(np.linalg.norm(recalled, axis=1).max()),
        }

    # CONTROL: same learner, magnitudes NOT corrected. The law's
    # derivation assumes unit rows, so this family must violate the bar.
    k_c, rank_c = 128, 32
    tg = crand(a.dim, k_c, rng)
    _, recalled_c, _ = learn_codebook(
        a.dim, k_c, rank_c, 1.0, rng, tg, epochs=a.epochs
    )
    books["control_nonunit"] = recalled_c
    meta["control_nonunit"] = {
        "k": k_c, "rank": rank_c, "epochs": a.epochs,
        "norm_min": float(np.linalg.norm(recalled_c, axis=1).min()),
        "norm_max": float(np.linalg.norm(recalled_c, axis=1).max()),
        "note": "learned rows with NATIVE magnitude; the law's unit-norm assumption is violated by construction",
    }

    # -- measure --------------------------------------------------------
    for fam, rows in books.items():
        norms = np.linalg.norm(rows, axis=1)
        g_real = np.real(rows @ rows.conj().T)
        g_abs = np.abs(rows @ rows.conj().T)
        np.fill_diagonal(g_real, 0.0)
        np.fill_diagonal(g_abs, 0.0)
        c_max = float(g_real[0].max())
        c_max_abs = float(g_abs[0].max())
        unit_ok = bool(np.allclose(norms, 1.0, atol=1e-9))
        rec["families"][fam] = {
            **meta[fam],
            "row_norm_min": float(norms.min()),
            "row_norm_max": float(norms.max()),
            "unit_norm_atol_1e9": unit_ok,
            "c_max_target_row0_realpart": c_max,
            "c_max_target_row0_abs": c_max_abs,
            "c_worst_pair": float(g_abs.max()),
        }

        for t in TARGETS:
            q = make_query(rows, 0, t, cfg, rng)
            r = select(q, rows)
            t_actual = float(np.real(np.vdot(rows[0], q.psi)))
            pred = t_actual * (1.0 - c_max)
            applies = int(r.index) == 0
            rec["cells"].append({
                "family": fam,
                "kind": "informative",
                "unit_norm_family": unit_ok,
                "t_requested": t,
                "t_actual": t_actual,
                "c_max": c_max,
                "argmax": int(r.index),
                "margin": float(r.margin),
                "predicted_margin": float(pred),
                "residual": float(r.margin - pred),
                "applicable": bool(applies),
                "tie_group_size": int(r.group_size),
            })

        q = make_query(rows, 0, 0.0, cfg, rng)
        r = select(q, rows)
        rec["cells"].append({
            "family": fam,
            "kind": "unrelated",
            "unit_norm_family": unit_ok,
            "t_requested": 0.0,
            "t_actual": float(np.real(np.vdot(rows[0], q.psi))),
            "c_max": c_max,
            "argmax": int(r.index),
            "margin": float(r.margin),
            "predicted_margin": None,
            "residual": None,
            "applicable": False,
            "tie_group_size": int(r.group_size),
        })

    # -- verdict --------------------------------------------------------
    inf = [c for c in rec["cells"]
           if c["kind"] == "informative" and c["applicable"]]
    unit_inf = [c for c in inf if c["unit_norm_family"]]
    ctrl_inf = [c for c in inf if not c["unit_norm_family"]]

    def max_abs(cells):
        if not cells:
            return None
        return float(max(abs(c["residual"]) for c in cells))

    per_family = {}
    for fam in books:
        vals = [abs(c["residual"]) for c in inf if c["family"] == fam]
        per_family[fam] = float(max(vals)) if vals else None

    unit_max = max_abs(unit_inf)
    ctrl_max = max_abs(ctrl_inf)

    if not unit_inf:
        verdict = "NOT_APPLICABLE"
    elif unit_max is not None and unit_max <= RESIDUAL_BAR:
        verdict = "LAW_CONFIRMED_ON_LEARNED"
    else:
        verdict = "LAW_FALSIFIED_ON_LEARNED"

    control_exceeded = ctrl_max is not None and ctrl_max > RESIDUAL_BAR

    y = np.array([c["margin"] for c in unit_inf], dtype=np.float64)
    x = np.array([c["predicted_margin"] for c in unit_inf], dtype=np.float64)
    r2 = float(np.corrcoef(x, y)[0, 1] ** 2) if x.size > 1 and x.std() > 0 else float("nan")

    rec["law_fit"] = {
        "n_informative_applicable": len(inf),
        "n_unit_norm_cells": len(unit_inf),
        "n_control_cells": len(ctrl_inf),
        "max_abs_residual_unit_norm": unit_max,
        "max_abs_residual_control": ctrl_max,
        "r2_margin_vs_prediction_unit_norm": r2,
        "per_family_max_abs_residual": per_family,
        "residual_bar": RESIDUAL_BAR,
        "control_exceeded_bar": control_exceeded,
    }
    rec["verdict"] = verdict
    rec["scope_limits"] = [
        "Learned engrams over GENERATED pairs; no real-world corpus.",
        "No task attempted or solved. No capability claim follows.",
        "The law is conditioned on the target row winning; else NOT_APPLICABLE.",
        "The control family violates the unit-norm assumption by construction.",
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
        if c["kind"] == "informative":
            print(
                "%-18s unit=%-5s t=%.2f c_max=%+.4f margin=%+.6f pred=%+.6f resid=%+.6f %s"
                % (c["family"], c["unit_norm_family"], c["t_requested"], c["c_max"],
                   c["margin"], c["predicted_margin"], c["residual"],
                   "OK" if c["applicable"] else "N/A"),
                file=sys.stderr,
            )
    for fam, m in rec["families"].items():
        print(
            "  %-18s unit_norm=%-5s align_mean=%.4f norm=[%.4f,%.4f]"
            % (fam, m["unit_norm_atol_1e9"], m.get("target_alignment_mean", float("nan")),
               m.get("row_norm_min", m.get("norm_min")), m.get("row_norm_max", m.get("norm_max"))),
            file=sys.stderr,
        )
    print("law_fit: " + json.dumps(rec["law_fit"], sort_keys=True), file=sys.stderr)
    print(f"verdict={verdict} sha256={digest} readback={rb == digest}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
