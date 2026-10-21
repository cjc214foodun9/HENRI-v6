"""Stage-8 readout probe: does Margin == t (1 - c_max) on NOISY targets?

Why this probe exists
---------------------
`docs/report.html` recorded the production-D readout as UNRESOLVED. The
probe that was supposed to settle it failed vacuously: its target-present
arm inserted the query as codebook row 0, so `margin = top - second` was
1.0 BY CONSTRUCTION. The report asked for the correct fixture:

    a target that is present but NOT identical to the query.

This probe builds exactly that:

    Psi_q = to_sphere( t * X_i + sqrt(1 - t^2) * Psi_perp )
    Re< Psi_perp, X_i > = 0

so the signal projection onto the target row is `t` by construction, and
`t = 1` is excluded. Three arms:

  A  readout-only on the constructed query. Isolates the law.
  B  the same query through `run_pipeline`. Exercises stages 1-8.
  C  learned `DeltaMemory` trajectories with a corrupted key.

Pre-registered decision rule
----------------------------
- PASS when |margin - t(1 - c_max)| <= 0.05 for every t in the grid AND
  top-1 accuracy is 1.0 for t >= 0.5.
- FAIL otherwise, reported per cell. No bar moves after the first run.
- The law's noise term is reported, not subtracted: the residual must be
  comparable to `sqrt(1 - t^2) * E[max|noise|]`, whose closed form is
  `sqrt(2 ln M / (2D))`. A residual far ABOVE that is a defect, not noise.

Evidence label: OBSERVED for every number.
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

from henri.config import WaveConfig
from henri.gram import max_crosstalk, stiefel_retract
from henri.memory import DeltaMemory
from henri.pipeline import run_pipeline
from henri.readout import entropy_at_beta, scores_of, select
from henri.sagnac import VetoCalibration
from henri.state import WaveState

BETA_STAR = 26.10
T_GRID = (0.3, 0.5, 0.7, 0.8, 0.9)
MARGIN_TOL = 0.05


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def orthogonal_to(v: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Remove the complex projection of `v` onto `x`, then normalize.

    The subtraction removes the FULL complex inner product, so
    `Re<v, x> == 0` and not merely `|Re<v, x>|` small.
    """
    coeff = np.vdot(x, v) / np.vdot(x, x)
    out = v - coeff * x
    return unit(out)


def build_query(x_i: np.ndarray, t: float, rng, max_tries: int = 8):
    """Psi_q with signal projection exactly `t` onto `x_i`."""
    d = x_i.shape[0]
    best = None
    for _ in range(max_tries):
        perp = orthogonal_to(
            rng.standard_normal(d) + 1j * rng.standard_normal(d), x_i)
        q = t * x_i + np.sqrt(max(0.0, 1.0 - t * t)) * perp
        q = unit(q)
        got = float(np.real(np.vdot(x_i, q)))
        if best is None or abs(got - t) < abs(best[0] - t):
            best = (got, q)
        if abs(got - t) < 1e-9:
            break
    return best[1], best[0]


def make_codebook(cfg, m: int, seed: int, orthogonalize: bool) -> np.ndarray:
    rng = np.random.default_rng(seed)
    d = cfg.dim
    raw = (rng.standard_normal((m, d)) + 1j * rng.standard_normal((m, d)))
    rows = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    return stiefel_retract(rows) if orthogonalize else rows


def arm_a(cfg, book, trials: int, rng) -> list:
    out = []
    m = book.shape[0]
    for t in T_GRID:
        margins, preds, hits, ents, got_t, groups = [], [], [], [], [], []
        for _ in range(trials):
            i = int(rng.integers(0, m))
            q, real_t = build_query(book[i], t, rng)
            state = WaveState.from_complex(q, cfg)
            res = select(state, book)
            s = scores_of(state, book)
            margins.append(res.margin)
            preds.append(real_t * (1.0 - 0.0))
            hits.append(int(res.index == i))
            ents.append(entropy_at_beta(s, BETA_STAR))
            got_t.append(real_t)
            groups.append(int(res.group_size))
        out.append({
            "t_requested": t, "t_realized": float(np.mean(got_t)),
            "margin_mean": float(np.mean(margins)),
            "margin_min": float(np.min(margins)),
            "top1_accuracy": float(np.mean(hits)),
            "entropy_at_beta": float(np.mean(ents)),
            "group_size_mean": float(np.mean(groups)),
        })
    return out


def arm_b(cfg, book, trials: int, rng, lock_steps: int) -> list:
    """Same queries through run_pipeline (stages 1-8).

    Two sub-arms, because the first version of this function measured the
    wrong thing. It held ONE random axiom for every query, so the Sagnac
    veto fired on 100% of rows (delta ~ 1.0 against an unrelated wave) and
    every result abstained. That is fail-closed behaviour working
    correctly, not a readout failure. The fixture was the defect.

    matched   : axiom = the query's own target row. The pipeline must
                recover i*.
    unrelated : axiom = a fixed random wave. The pipeline must abstain.
    """
    out = []
    m = book.shape[0]
    calib = VetoCalibration(epsilon=0.0431, baseline_mean=0.0,
                            baseline_std=0.0, sigma=0.0, ceiling=0.0431,
                            n_pairs=0)
    unrelated = WaveState.from_complex(unit(
        rng.standard_normal(cfg.dim) + 1j * rng.standard_normal(cfg.dim)), cfg)
    for t in T_GRID:
        hits, margins, idxs, locked, rs = [], [], [], [], []
        ctrl_abstain = []
        for _ in range(trials):
            i = int(rng.integers(0, m))
            q, _ = build_query(book[i], t, rng)
            ingress = WaveState.from_complex(q, cfg)
            matched_axiom = WaveState.from_complex(book[i], cfg)
            res = run_pipeline(cfg, ingress, book, matched_axiom, calib,
                               relax_steps=0, lock_steps=lock_steps, seed=7)
            hits.append(int(res.index == i))
            margins.append(float(res.readout.margin) if res.readout else float("nan"))
            idxs.append(int(res.index))
            locked.append(bool(res.locked))
            rs.append(float(res.order_parameter))
            ctrl = run_pipeline(cfg, ingress, book, unrelated, calib,
                                relax_steps=0, lock_steps=lock_steps, seed=7)
            ctrl_abstain.append(int(ctrl.index == -1))
        out.append({
            "t_requested": t,
            "matched_top1_accuracy": float(np.mean(hits)),
            "matched_margin_mean": float(np.mean(margins)),
            "matched_abstain_rate": float(np.mean([x == -1 for x in idxs])),
            "unrelated_axiom_abstain_rate": float(np.mean(ctrl_abstain)),
            "kuramoto_locked_rate": float(np.mean(locked)),
            "order_parameter_mean": float(np.mean(rs)),
        })
    return out


def arm_c(cfg, trials: int, rng) -> list:
    """Learned DeltaMemory recall from a corrupted key."""
    d = cfg.dim
    mem = DeltaMemory.build(d, rank=64)
    keys, vals = [], []
    for _ in range(64):
        k = unit(rng.standard_normal(d) + 1j * rng.standard_normal(d))
        v = unit(rng.standard_normal(d) + 1j * rng.standard_normal(d))
        mem.write(k, v)
        keys.append(k); vals.append(v)
    book = np.stack(vals)
    out = []
    for t in T_GRID:
        hits, margins, ents = [], [], []
        for i in range(len(keys)):
            q = t * keys[i] + np.sqrt(1.0 - t * t) * orthogonal_to(
                rng.standard_normal(d) + 1j * rng.standard_normal(d), keys[i])
            stored = mem.read(q)
            state = WaveState.from_complex(stored, cfg)
            res = select(state, book)
            hits.append(int(res.index == i))
            margins.append(res.margin)
            ents.append(entropy_at_beta(scores_of(state, book), BETA_STAR))
        out.append({
            "t_requested": t,
            "recall_accuracy": float(np.mean(hits)),
            "margin_mean": float(np.mean(margins)),
            "entropy_at_beta": float(np.mean(ents)),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=65536)
    ap.add_argument("--rows", type=int, default=256)
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--lock-steps", type=int, default=200,
                    help="pinned for arm B to bound wall clock; the Kuramoto "
                         "stage does not feed the readout")
    ap.add_argument("--orthogonalize", action="store_true",
                    help="Stiefel-retract the codebook (c_max ~ 0). Default "
                         "off so the law is tested with a non-zero c_max.")
    ap.add_argument("--out", default="receipts/stage8_readout_full_D.json")
    a = ap.parse_args()

    t0 = time.perf_counter()
    cfg = WaveConfig(dim=a.dim)
    rng = np.random.default_rng(20261020)
    book = make_codebook(cfg, a.rows, seed=11, orthogonalize=a.orthogonalize)
    c_max = max_crosstalk(book)

    rec = {
        "probe": "stage8_readout", "dim": a.dim, "rows": a.rows,
        "beta_star": BETA_STAR, "t_grid": list(T_GRID),
        "orthogonalized": bool(a.orthogonalize),
        "c_max_measured": c_max,
        "noise_floor_predicted": float(np.sqrt(2 * np.log(a.rows) / (2 * a.dim))),
        "margin_tolerance": MARGIN_TOL,
    }

    rec["arm_a_readout_only"] = arm_a(cfg, book, a.trials, rng)
    rec["arm_b_pipeline"] = arm_b(cfg, book, min(a.trials, 6), rng, a.lock_steps)
    rec["arm_c_learned_delta_memory"] = arm_c(cfg, min(a.trials, 4), rng)

    # decision rule, fixed before the first run
    fails = []
    for row in rec["arm_a_readout_only"]:
        t = row["t_realized"]
        pred = t * (1.0 - c_max)
        row["margin_predicted"] = pred
        row["residual"] = row["margin_mean"] - pred
        row["residual_vs_noise"] = (abs(row["residual"])
                                    / max(rec["noise_floor_predicted"], 1e-12))
        if abs(row["residual"]) > MARGIN_TOL:
            fails.append({"t": t, "residual": row["residual"]})
        if t >= 0.5 and row["top1_accuracy"] < 1.0:
            fails.append({"t": t, "top1": row["top1_accuracy"]})
    rec["fails"] = fails
    rec["verdict"] = "STAGE8_LAW_CONFIRMED" if not fails else "STAGE8_LAW_FAILED"
    rec["wall_clock_s"] = time.perf_counter() - t0

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(rec, indent=2, sort_keys=True).encode("utf-8")
    out.write_bytes(body)
    digest = hashlib.sha256(body).hexdigest()
    out.with_suffix(out.suffix + ".sha256").write_bytes(
        (digest + "\n").encode())

    print("=== STAGE-8 READOUT PROBE ===", file=sys.stderr)
    print(f"dim={a.dim} rows={a.rows} c_max={c_max:.6f} "
          f"noise_floor={rec['noise_floor_predicted']:.5f}", file=sys.stderr)
    print(f"{'t':>7} {'margin':>9} {'t(1-cmax)':>10} {'resid':>8} "
          f"{'resid/noise':>11} {'top1':>6} {'H(26.1)':>9}", file=sys.stderr)
    for row in rec["arm_a_readout_only"]:
        print(f"{row['t_realized']:7.3f} {row['margin_mean']:9.5f} "
              f"{row['margin_predicted']:10.5f} {row['residual']:8.5f} "
              f"{row['residual_vs_noise']:11.2f} {row['top1_accuracy']:6.2f} "
              f"{row['entropy_at_beta']:9.2e}", file=sys.stderr)
    print(f"verdict={rec['verdict']} sha256={digest}", file=sys.stderr)
    return 0 if rec["verdict"] == "STAGE8_LAW_CONFIRMED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
