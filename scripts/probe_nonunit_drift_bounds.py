"""Non-unit-norm DRIFT BOUNDS for the readout margin law.

Additive to `scripts/probe_learned_codebook_law.py`, which answered the
owner question (does the law survive on LEARNED codebooks) with
LAW_CONFIRMED_ON_LEARNED, R^2 0.9999840, max abs residual 0.002334 at
D = 65536.

WHAT THE PARENT PROBE LEFT OPEN, stated plainly
Its only non-unit control used row norms `[0.8224, 46.2575]` with
`c_max = 47.16`. That is maximally extreme and OUT OF DOMAIN: the law
predicted -2085 while measurement stayed near +44. It proves the
unit-norm premise is load bearing. It says NOTHING about the regime a
real learned codebook enters, where rows drift slightly off the sphere.
A bank with norms in [0.99, 1.01] is the realistic failure mode, and no
measurement in this repository covers it.

THIS PROBE MEASURES THAT REGIME. It sweeps uniform scale from 1.000
(exact) to 2.000 (far out of domain) and row-wise drift width from 0.000
to 0.250, and reports where the law stops predicting the margin.

THE CONTROL, and why it is NOT a closed form
An earlier draft of this probe asserted a closed form
`residual(s) = t(s-1) + c_max(t - s^2)`. That was WRONG and was caught
before the first run: the marginal (rival) score carries the product of
the target-row and rival-row norms, so it scales as `s^2`, while
`c_max_of` is computed on the ALREADY-SCALED rows. Conflating those two
introduced a spurious `s^2` term. Rather than ship a derivation that was
not verified, this probe asserts only an identity that is EXACT and
independently checkable:

    c_max_of(apply_uniform(rows, s), 0) == s^2 * c_max_of(rows, 0)

That control is falsifiable and it verifies the scaling plumbing, which
is the part most likely to be silently wrong. It is NOT used to grade
the law. It cannot certify a passing arm.

CONVENTION, stated so a reader cannot misread the numbers
`c_max` is the largest ABSOLUTE off-diagonal coherence against row 0,
taken on the rows AS THEY ARE (scaled), matching the parent probe's
convention. `t` is the MEASURED coherence `Re<rows[0], Psi>`, never the
requested value.

PRE-REGISTERED DECISION RULE, fixed before the first run
- Informative cell: `t_requested >= 0.50` AND `argmax == 0`.
- SCALING_CONTROL_HOLDS : max identity gap <= 1e-12 on every s. A failure
  INVALIDATES THE PROBE and no drift bound is reported.
- DRIFT_BOUNDS_REPORTED : report the smallest uniform s, and the smallest
  row-wise width, at which max |residual| > 0.05.
- The bounds are REPORTED, not asserted. Either outcome is informative.
  A bound of `None` means the law held across the whole swept range.

DISCLOSURE. Same learner and same generated targets as the parent probe.
Learned engrams over generated data, not a real corpus. No task is
attempted or solved. No capability claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))

# Reuse the parent probe's helpers verbatim. Importing beats copying: a
# second definition of `unit` or `make_query` could silently drift from
# the one that produced the sealed receipt.
import probe_learned_codebook_law as base  # noqa: E402
from henri.config import WaveConfig  # noqa: E402
from henri.readout import select  # noqa: E402

RESIDUAL_BAR = base.RESIDUAL_BAR
TARGETS = base.TARGETS
SEED = base.SEED
unit = base.unit
crand = base.crand
clustered_values = base.clustered_values
learn_codebook = base.learn_codebook
make_query = base.make_query

UNIFORM_SCALES = (1.000, 1.010, 1.025, 1.050, 1.100, 1.250, 2.000)
ROWWISE_WIDTHS = (0.000, 0.010, 0.025, 0.050, 0.100, 0.250)
FAMILIES = (("wellfit", 32, 64, "random"), ("clustered", 32, 64, "clustered"))


def build_rows(dim, k, rank, lam, rng, style, epochs=2):
    """Learned rows, unit-normalized. Same path as the parent probe."""
    targets = (
        clustered_values(dim, k, 0.30, rng)
        if style == "clustered"
        else crand(dim, k, rng)
    )
    _, recalled, upd = learn_codebook(dim, k, rank, lam, rng, targets,
                                      epochs=epochs)
    return np.stack([unit(r) for r in recalled]), upd


def c_max_of(rows, j=0):
    """Largest ABSOLUTE off-diagonal coherence against row j.

    `rows.conj() @ rows[j]`, NOT `np.vdot(rows, rows[j])`. vdot flattens
    BOTH arguments, so it silently requires len(rows) == rows.shape[1]
    and raises otherwise. Matmul keeps the k-index.
    """
    g = np.abs(rows.conj() @ rows[j])
    g = np.delete(g, j)
    return float(g.max()) if g.size else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=65536)
    ap.add_argument("--out", type=str,
                    default="receipts/nonunit_drift_bounds.json")
    ap.add_argument("--epochs", type=int, default=2)
    a = ap.parse_args()

    t0 = time.perf_counter()
    cfg = WaveConfig(dim=a.dim)
    rng = np.random.default_rng(SEED)

    rec: dict = {
        "probe": "nonunit_drift_bounds",
        "law": "margin ~= t * (1 - c_max)",
        "parent_probe": "scripts/probe_learned_codebook_law.py",
        "parent_verdict": "LAW_CONFIRMED_ON_LEARNED",
        "parent_sealed_digest": (
            "62624b9d6eb393964fbfc16196849a450f3fada2ac264393b58eb43218e47c48"
        ),
        "seed": SEED,
        "learner": "henri.memory.DeltaMemory",
        "convention": (
            "c_max = max absolute off-diagonal coherence against row 0, "
            "on the rows AS SCALED; t = MEASURED coherence Re<row0, Psi>"
        ),
        "control": "c_max(s*rows) == s^2 * c_max(rows), exact to 1e-12",
        "params": {
            "dim": a.dim, "epochs": a.epochs, "residual_bar": RESIDUAL_BAR,
            "uniform_scales": list(UNIFORM_SCALES),
            "rowwise_widths": list(ROWWISE_WIDTHS),
        },
        "decision_rule": "pre-registered in the module docstring",
        "disclosure": (
            "Learned engrams over GENERATED pairs. Not a real-world corpus. "
            "No task attempted or solved."
        ),
        "families": {},
        "cells": [],
    }

    control_gap = 0.0
    for name, k, rank, style in FAMILIES:
        rows0, upd = build_rows(a.dim, k, rank, 1.0, rng, style,
                                epochs=a.epochs)
        cm0 = c_max_of(rows0)
        rec["families"][name] = {
            "k": k, "rank": rank, "style": style,
            "memory_updates": int(upd), "c_max_unit": cm0,
        }

        for s in UNIFORM_SCALES:
            rows = rows0 * float(s)
            cm = c_max_of(rows)
            # EXACT scaling identity. Verifies the plumbing; cannot grade
            # the law.
            control_gap = max(control_gap, abs(cm - s * s * cm0))
            for t in TARGETS:
                q = make_query(rows, 0, t, cfg, rng)
                r = select(q, rows)
                t_act = float(np.real(np.vdot(rows[0], q.psi)))
                law = t_act * (1.0 - cm)
                rec["cells"].append({
                    "family": name, "arm": "uniform", "drift": float(s),
                    "t_requested": t, "t_actual": t_act, "c_max": cm,
                    "argmax": int(r.index), "margin": float(r.margin),
                    "predicted_margin": float(law),
                    "residual": float(r.margin - law),
                    "abs_residual": float(abs(r.margin - law)),
                    "within_bar": bool(abs(r.margin - law) <= RESIDUAL_BAR),
                    "applicable": bool(int(r.index) == 0),
                })

        for wd in ROWWISE_WIDTHS:
            wrng = np.random.default_rng(SEED + 7919)
            if wd <= 0.0:
                rows = rows0.copy()
            else:
                sc = wrng.uniform(1.0 - float(wd), 1.0 + float(wd),
                                  size=rows0.shape[0])
                rows = rows0 * sc[:, None]
            cm = c_max_of(rows)
            for t in TARGETS:
                q = make_query(rows, 0, t, cfg, rng)
                r = select(q, rows)
                t_act = float(np.real(np.vdot(rows[0], q.psi)))
                law = t_act * (1.0 - cm)
                rec["cells"].append({
                    "family": name, "arm": "rowwise", "drift": float(wd),
                    "t_requested": t, "t_actual": t_act, "c_max": cm,
                    "argmax": int(r.index), "margin": float(r.margin),
                    "predicted_margin": float(law),
                    "residual": float(r.margin - law),
                    "abs_residual": float(abs(r.margin - law)),
                    "within_bar": bool(abs(r.margin - law) <= RESIDUAL_BAR),
                    "applicable": bool(int(r.index) == 0),
                })

    inf = [c for c in rec["cells"] if c["applicable"]]

    def first_break(arm):
        by_drift: dict = {}
        for c in inf:
            if c["arm"] != arm:
                continue
            d = c["drift"]
            by_drift[d] = max(by_drift.get(d, 0.0), c["abs_residual"])
        for d in sorted(by_drift):
            if by_drift[d] > RESIDUAL_BAR:
                return d, by_drift[d]
        return None, None

    ub, ur = first_break("uniform")
    rb, rr = first_break("rowwise")
    control_ok = control_gap <= 1e-12

    rec["drift_bounds"] = {
        "scaling_control_max_gap": control_gap,
        "scaling_control_holds": bool(control_ok),
        "uniform_first_break_scale": ub,
        "uniform_first_break_residual": ur,
        "rowwise_first_break_width": rb,
        "rowwise_first_break_residual": rr,
        "residual_bar": RESIDUAL_BAR,
        "n_informative_cells": len(inf),
        "n_cells_total": len(rec["cells"]),
    }
    rec["wall_clock_s"] = time.perf_counter() - t0
    rec["status"] = "COMPLETE"

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(rec, indent=2, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(body).hexdigest()
    out.write_bytes(body)
    out.with_suffix(out.suffix + ".sha256").write_bytes(
        (digest + "\n").encode())
    rb2 = hashlib.sha256(out.read_bytes()).hexdigest()

    print("=== NON-UNIT DRIFT BOUNDS ===", file=sys.stderr)
    print(f"scaling control max gap  : {control_gap:.3e}  "
          f"holds={control_ok}", file=sys.stderr)
    for fam, m in rec["families"].items():
        print(f"  {fam:<12} c_max_unit={m['c_max_unit']:+.4f}  "
              f"k={m['k']} rank={m['rank']}", file=sys.stderr)
    for fam in rec["families"]:
        for arm in ("uniform", "rowwise"):
            rows_ = [c for c in inf
                     if c["family"] == fam and c["arm"] == arm]
            for d in sorted({c["drift"] for c in rows_}):
                sel = [c for c in rows_ if c["drift"] == d]
                mx = max(c["abs_residual"] for c in sel)
                print("%-10s %-8s drift=%.3f  max|resid|=%.6f  %s"
                      % (fam, arm, d, mx,
                         "OUT-OF-BAR" if mx > RESIDUAL_BAR else "IN-BAR"),
                      file=sys.stderr)
    print(f"uniform first break s    : {ub}", file=sys.stderr)
    print(f"rowwise first break width: {rb}", file=sys.stderr)
    verdict = ("DRIFT_BOUNDS_REPORTED" if control_ok
               else "PROBE_INVALID_SCALING_CONTROL_FAILED")
    print(f"verdict={verdict} sha256={digest} readback={rb2 == digest}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
