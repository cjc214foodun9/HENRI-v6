"""Discriminating test: is the uniform readout a WALL or CORRECT behaviour?

The full-D smoke measured readout margin 6.8e-05 with entropy 1.000000.
Two competing explanations:

  H_wall     The readout cannot discriminate. Real defect.
  H_correct  The smoke codebook is an unrelated random orthonormal
             basis, so a uniform softmax is the RIGHT answer for an
             unrelated input.

This decides between them by placing the TARGET in the codebook. If the
readout is decisive when the target is present, H_correct holds and no
wall is established. If it stays uniform even with the target present,
H_wall holds and the negative is real.

Evidence label: this is the experiment that assigns the label.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from henri.config import WaveConfig
from henri.gram import stiefel_retract
from henri.pipeline import build_ingress, run_pipeline
from henri.readout import select
from henri.sagnac import calibrate_epsilon, perturb_with_coherence
from henri.state import WaveState

SEED = 20261012


def make_codebook(cfg, n, rng):
    raw = (
        rng.standard_normal((n, cfg.dim)) + 1j * rng.standard_normal((n, cfg.dim))
    )
    return stiefel_retract(raw)


def probe(dim: int) -> dict:
    cfg = WaveConfig(dim=dim)
    rng = np.random.default_rng(SEED)
    span = cfg.slot_span
    slots = [
        (rng.standard_normal(span) + 1j * rng.standard_normal(span))
        for _ in range(3)
    ] + [None]
    ingress = build_ingress(cfg, slots)

    refs = [WaveState.random(cfg, seed=SEED + 100 + i) for i in range(24)]
    cal = calibrate_epsilon(refs, sigma=3.0, ceiling=0.35, seed=SEED, n_pairs=6000)

    out = {}
    for label, target_in_book in (("unrelated", False), ("target_present", True)):
        book = make_codebook(cfg, 32, np.random.default_rng(SEED + 7))
        axiom = WaveState.from_complex(book[0], cfg)
        res = run_pipeline(
            cfg=cfg, ingress=ingress, codebook=book, axiom=axiom,
            calibration=cal, seed=SEED, relax_steps=3,
            lock_steps=555 if dim >= 4096 else 200,
        )
        if target_in_book:
            # Replace row 0 with the wave the pipeline actually settled on,
            # then re-orthogonalize the OTHER rows against it so the book
            # stays a valid orthonormal codebook WITH the target inside.
            others = book[1:]
            others = others - np.outer(others @ res.cleaned.psi.conj(), res.cleaned.psi)
            keep = np.linalg.norm(others, axis=1) > 1e-6
            others = others[keep]
            others = stiefel_retract(others)
            book = np.vstack([res.cleaned.psi, others])
        sel = select(res.cleaned, book)
        out[label] = {
            "margin": float(sel.margin),
            "entropy": float(sel.entropy),
            "group_size": int(sel.group_size),
            "index": int(sel.index),
            "n_patterns": int(book.shape[0]),
        }
    return {"dim": dim, "unrelated": out["unrelated"], "target_present": out["target_present"]}


if __name__ == "__main__":
    results = [probe(d) for d in (512, 4096)]
    print(json.dumps(results, indent=2))
    print()
    verdict = []
    for r in results:
        u, t = r["unrelated"], r["target_present"]
        decisive = t["margin"] > 10 * max(u["margin"], 1e-12)
        verdict.append(
            f"D={r['dim']}: margin {u['margin']:.3e} -> {t['margin']:.3e} "
            f"(x{t['margin']/max(u['margin'],1e-12):.1f}) "
            f"{'DECISIVE when target present' if decisive else 'STILL UNIFORM'}"
        )
    print("\n".join(verdict))
    print()
    print(
        "VERDICT: H_correct (uniform is right for unrelated input, no wall)"
        if all(float(r["target_present"]["margin"]) > 0.1 for r in results)
        else "VERDICT: H_wall (readout cannot discriminate even with target present)"
    )
