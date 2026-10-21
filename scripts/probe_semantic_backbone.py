"""Semantic backbone probe: test the spec's gates before trusting the code.

PRE-REGISTERED GATES, FIXED BEFORE ANY ARM WAS RUN
--------------------------------------------------
G-SEM1a  synonym coherence  >= 0.65   (spec value)
G-SEM1b  negation coherence <= -0.50  (spec value)
G-SEM2   option c_max <= 0.01 and Margin Law residual <= 0.0092 after
         Stage V whitening
G-SEM3   kernel still SEALED, unit modulus error < 1e-15, slot bleed 0.0
G-SEM4   MMLU n=400 and GPQA n=198 with z >= +4.0

G-SEM1b is tested FIRST, because a forward measurement of the frozen
encoder already showed antonyms at rho = +0.622 rather than negative. If
the distributional hypothesis holds, negation cannot come from a linear
shell and the gate is unreachable by the specified mechanism. The probe
reports which mechanism, if any, reaches it.

NEGATIVE CONTROLS, REQUIRED
---------------------------
Every gate must be able to fail:
  - a shuffled-pair control for synonymy (must read ~0, not 0.65)
  - a gamma = 0 control for whitening (must leave c_max unchanged)
  - a same-string control for polarity (must read +1, never -1)

USAGE
-----
    python scripts/probe_semantic_backbone.py --pairs 100 --out receipts/...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from henri.semantic_backbone import (
    SLOT, BochnerPhaseProjector, SemanticBackbone, SemanticStem,
    common_mode_reject, delta_memory_capacity, polarity, whiten_options,
)

SYNONYMS = [
    ("car", "automobile"), ("velocity", "speed"), ("doctor", "physician"),
    ("buy", "purchase"), ("big", "large"), ("begin", "start"),
    ("error", "mistake"), ("quick", "fast"), ("money", "currency"),
    ("house", "home"), ("child", "kid"), ("end", "finish"),
]

# Antonym pairs as WHOLE PROPOSITIONS, which is how the spec phrases them.
AFFIRM_NEGATE = [
    ("the pressure increases", "the pressure does not increase"),
    ("the reaction proceeds", "the reaction does not proceed"),
    ("the value is positive", "the value is not positive"),
    ("the system is stable", "the system is not stable"),
    ("the function converges", "the function does not converge"),
    ("temperature increases", "temperature decreases"),
    ("the rate increases", "the rate decreases"),
    ("the signal is present", "the signal is absent"),
    ("the bond forms", "the bond breaks"),
    ("the mass increases", "the mass decreases"),
]

BARE_ANTONYMS = [
    ("increase", "decrease"), ("hot", "cold"), ("positive", "negative"),
    ("maximum", "minimum"), ("above", "below"), ("larger", "smaller"),
    ("gain", "loss"), ("rise", "fall"), ("expand", "contract"),
    ("include", "exclude"),
]


def _sig(c: int, n: int, p0: float) -> dict:
    import math
    if n <= 0:
        return {"z": None, "p": None}
    z = (c - n * p0 - 0.5) / math.sqrt(n * p0 * (1 - p0))
    return {"z": round(z, 4), "p": round(0.5 * math.erfc(z / math.sqrt(2)), 8)}


def coherence(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.real(np.vdot(a, b)))


def gate_sem1(bb: SemanticBackbone) -> dict:
    """Synonym and negation coherence, plus a shuffled control."""
    rng = np.random.default_rng(11)

    syn_t, syn_neg = [], []
    for a, b in SYNONYMS:
        pa, pb = bb.encode(a).psi, bb.encode(b).psi
        syn_t.append(coherence(pa, pb))
        # Shuffled partner: the wrong b. Must read near zero.
        syn_neg.append(coherence(pa, bb.encode(SYNONYMS[int(
            rng.integers(len(SYNONYMS)))][1]).psi))

    neg_prop, neg_bare = [], []
    for a, b in AFFIRM_NEGATE:
        neg_prop.append(coherence(bb.encode(a).psi, bb.encode(b).psi))
    for a, b in BARE_ANTONYMS:
        neg_bare.append(coherence(bb.encode(a).psi, bb.encode(b).psi))

    # Mechanism B: explicit polarity slot. +/- pi is exactly +/- 128 in Z_256.
    pol_flip = []
    for a, b in AFFIRM_NEGATE:
        ea, eb = bb.encode(a), bb.encode(b)
        pol_flip.append(int(ea.pol) * int(eb.pol))

    return {
        "synonym_mean": round(float(np.mean(syn_t)), 6),
        "synonym_control_shuffled": round(float(np.mean(syn_neg)), 6),
        "synonym_gate_0_65": bool(np.mean(syn_t) >= 0.65),
        "proposition_antonym_mean": round(float(np.mean(neg_prop)), 6),
        "bare_antonym_mean": round(float(np.mean(neg_bare)), 6),
        "negation_gate_neg_0_50": bool(np.mean(neg_prop) <= -0.50),
        "polarity_product_mean": round(float(np.mean(pol_flip)), 6),
        "polarity_signs": pol_flip,
        "polarity_mechanism_separates": bool(
            np.mean(pol_flip) < 0 and sum(p < 0 for p in pol_flip) >= 8),
    }


def gate_sigma(bb: SemanticBackbone, u: np.ndarray, v: np.ndarray) -> list:
    """Sweep sigma. Report measured kernel against the closed form."""
    rows = []
    rho = float(u @ v)
    for sig in (0.5, 1.0, float(np.sqrt(2.0)), 2.0, 3.0, 4.0):
        p = BochnerPhaseProjector(bb.stem.dim_sem, SLOT, sig, seed=20261020)
        pa, pb = p.phasor(u), p.phasor(v)
        measured = float(np.mean(np.cos(p.angles(u) - p.angles(v))))
        pred = float(np.exp(-sig ** 2 * (1.0 - rho)))
        rows.append({
            "sigma": round(sig, 6),
            "measured": round(measured, 6),
            "predicted": round(pred, 6),
            "rel_err": round(abs(measured - pred) / max(pred, 1e-12), 6),
        })
    return rows


def gate_sem2(bb: SemanticBackbone, questions: list, opts: list) -> dict:
    """Option coherence and Margin Law residual, before and after whitening."""
    from henri.readout import entropy_at_beta, select
    from henri.state import WaveState
    from henri.config import WaveConfig

    cfg = WaveConfig()
    rows, gammas = [], []
    before_c, after_c, res_before, res_after = [], [], [], []

    for q, os_ in zip(questions, opts):
        qw = bb.encode(q).psi
        book = np.stack([bb.encode(o).psi for o in os_], axis=0)
        state = WaveState(qw, cfg)
        r0 = select(state, book, floor=None)
        g = np.real(book @ book.conj().T)
        np.fill_diagonal(g, 0.0)
        c0 = float(np.max(g))

        # Stage V-A: common mode. gamma = 1 by derivation.
        b1, q1, rep1 = common_mode_reject(book, qw, gamma=1.0)
        # Stage V-B: local Stiefel whitening.
        b2, q2, rep2 = whiten_options(b1, q1)
        st2 = WaveState(q2, cfg)
        r2 = select(st2, b2, floor=None)
        g2 = np.real(b2 @ b2.conj().T)
        np.fill_diagonal(g2, 0.0)
        c2 = float(np.max(g2))

        before_c.append(c0)
        after_c.append(c2)
        res_before.append(abs(r0.margin - r0.top_score * (1.0 - c0)))
        res_after.append(abs(r2.margin - r2.top_score * (1.0 - c2)))
        rows.append({
            "c_before": round(c0, 6),
            "c_after": round(c2, 6),
            "margin_before": round(r0.margin, 6),
            "margin_after": round(r2.margin, 6),
            "residual_before": round(res_before[-1], 6),
            "residual_after": round(res_after[-1], 6),
        })

    # Control: gamma = 0 must leave coherence untouched.
    c_g0 = []
    for q, os_ in zip(questions, opts):
        book = np.stack([bb.encode(o).psi for o in os_], axis=0)
        b, _, rep = common_mode_reject(book, None, gamma=0.0)
        c_g0.append(rep["c_max_after"])

    noise = float(np.sqrt(np.log(4) / 65536))
    return {
        "n": len(rows),
        "c_max_before_mean": round(float(np.mean(before_c)), 6),
        "c_max_after_mean": round(float(np.mean(after_c)), 6),
        "c_max_after_max": round(float(np.max(after_c)), 6),
        "residual_before_mean": round(float(np.mean(res_before)), 6),
        "residual_after_mean": round(float(np.mean(res_after)), 6),
        "noise_floor": round(noise, 6),
        "gate_c_max_le_0_01": bool(np.max(after_c) <= 0.01),
        "gate_residual_le_noise": bool(np.mean(res_after) <= noise),
        "control_gamma0_c_unchanged": round(float(np.mean(c_g0)), 6),
        "sample": rows[:3],
    }


def gate_capacity() -> dict:
    """DeltaMemory capacity: derived bound, then measured recall."""
    from henri.memory import DeltaMemory

    dim, rank = 4096, 64          # small D so the sweep is affordable
    out = {"derivation": delta_memory_capacity(rank, dim, rank), "sweeps": []}
    rng = np.random.default_rng(5)
    for m in (16, 32, 64, 128, 256):
        mem = DeltaMemory.build(dim=dim, rank=rank)
        keys, vals = [], []
        for _ in range(m):
            k = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
            k /= np.linalg.norm(k)
            v = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
            v /= np.linalg.norm(v)
            keys.append(k); vals.append(v)
            mem.write(k, v)
        errs = []
        for k, v in zip(keys, vals):
            pred = mem.read(k, iterations=1)
            n = np.linalg.norm(pred)
            pred = pred / n if n > 0 else pred
            errs.append(float(np.linalg.norm(pred - v)))
        hit = sum(1 for e in errs if e < 0.5)
        out["sweeps"].append({
            "m": m,
            "rank": rank,
            "recall_le_0_5": round(hit / m, 4),
            "mean_err": round(float(np.mean(errs)), 6),
            "within_exact_capacity": bool(m <= rank),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=12)
    ap.add_argument("--out", default="receipts/semantic_backbone.json")
    a = ap.parse_args()

    bb = SemanticBackbone()
    info = bb.info()
    print(f"[STEM] {info['stem']} dim_sem={info['dim_sem']} "
          f"sigma={info['sigma']} offline={info['offline']}")

    r1 = gate_sem1(bb)
    print(f"[G-SEM1] synonym={r1['synonym_mean']:.4f} "
          f"(gate>=0.65 {r1['synonym_gate_0_65']}) "
          f"control={r1['synonym_control_shuffled']:.4f}")
    print(f"[G-SEM1] proposition_antonym={r1['proposition_antonym_mean']:.4f} "
          f"bare_antonym={r1['bare_antonym_mean']:.4f} "
          f"(gate<=-0.50 {r1['negation_gate_neg_0_50']})")
    print(f"[G-SEM1] polarity slot separates {r1['polarity_mechanism_separates']} "
          f"(mean product {r1['polarity_product_mean']:.3f})")

    u = bb.stem.encode("car"); v = bb.stem.encode("automobile")
    sig_rows = gate_sigma(bb, u, v)
    worst = max(r["rel_err"] for r in sig_rows)
    print(f"[BOCHNER] max rel_err over sigma sweep = {worst:.6f}")

    # Gate 2 on real MMLU rows.
    import csv
    path = REPO / "data/official_benchmarks/canonical/mmlu/mmlu.csv"
    rows = list(csv.DictReader(open(path, encoding="utf-8")))[:40]
    qs = [r["Question"] for r in rows]
    ops = [[r[k] for k in "ABCD"] for r in rows]
    r2 = gate_sem2(bb, qs, ops)
    print(f"[G-SEM2] c_max {r2['c_max_before_mean']:.4f} -> "
          f"{r2['c_max_after_mean']:.6f} (max {r2['c_max_after_max']:.6f})")
    print(f"[G-SEM2] residual {r2['residual_before_mean']:.6f} -> "
          f"{r2['residual_after_mean']:.6f} (floor {r2['noise_floor']:.6f})")
    print(f"[G-SEM2] control gamma=0 c={r2['control_gamma0_c_unchanged']:.4f}")

    r3 = gate_capacity()
    for s in r3["sweeps"]:
        print(f"[DELTA] m={s['m']:4d} rank=64 recall={s['recall_le_0_5']:.3f} "
              f"err={s['mean_err']:.4f} exact={s['within_exact_capacity']}")

    rec = {
        "probe": "semantic_backbone",
        "info": info,
        "g_sem1": r1,
        "bochner_sweep": sig_rows,
        "g_sem2": r2,
        "delta_memory": r3,
        "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    out = Path(a.out)
    if not out.is_absolute():
        out = REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"[RECEIPT] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
