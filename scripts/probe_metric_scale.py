"""Scale-sensitivity probe for the readout entropy metric.

Defect under test
-----------------
`henri.readout.normalized_entropy` takes Shannon entropy of
`softmax(scores)` with NO inverse-temperature factor: it computes
`exp(s - max(s))` at unit scale. Readout scores are real coherences
`Re<M_k, Psi>`. At large D those have scale O(1/sqrt(D)) ~ 0.004 for
unrelated rows and O(1) for a matched row. At unit temperature both
regimes give a near-uniform softmax, so the metric reads ~1.0 in BOTH.

Evidence that this is real, from my own probe receipt
(`receipts/readout_geometry_full_D.json`, D=65536):

    unrelated       margin 6.8e-05   entropy 0.999999
    noisy_t0.90     margin 0.894174  entropy 0.993768
    noisy_t0.99     margin 0.984241  entropy 0.992032

A DECISIVE cell (margin 0.89) and a NON-decisive cell (margin 6.8e-05)
differ in entropy by 0.006. The metric cannot separate them.

Prediction, stated before running
---------------------------------
For the SAME score vector:
  unrelated    : entropy(beta=1) ~ 1.0 AND entropy(beta=26.1) ~ 1.0
  noisy_t0.90  : entropy(beta=1) ~ 0.99 AND entropy(beta=26.1) ~ 0.0

If entropy at beta=26.1 separates the two cases while entropy at
beta=1 does not, the legacy metric is scale-blind. Then the honest
decisiveness measures are the margin and a beta-calibrated entropy,
not the legacy unit-temperature entropy.

No threshold is moved here. This probe explains an existing reading.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from henri.config import WaveConfig  # noqa: E402
from henri.readout import normalized_entropy, scores_of, select  # noqa: E402
from henri.state import WaveState  # noqa: E402

BETAS = (1.0, 26.1, 100.0)


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n < 1e-30:
        raise ValueError("cannot normalize a zero or non-finite vector")
    return v / n


def entropy_at(scores: np.ndarray, beta: float) -> float:
    """Shannon entropy of softmax(beta * scores), normalized by log(M).

    Identical to `normalized_entropy` when beta = 1.0.
    """
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    m = s.size
    if m < 2:
        return 1.0
    z = beta * (s - float(np.max(s)))
    p = np.exp(z)
    p = p / (float(np.sum(p)) + 1e-30)
    return float(-np.sum(p * np.log(p + 1e-30)) / float(np.log(m)))


def noisy(rows, j, target, cfg, rng):
    """Same exact construction as the geometry probe."""
    base = rows[j]
    noise = rng.standard_normal(cfg.dim) + 1j * rng.standard_normal(cfg.dim)
    noise = unit(noise - base * np.vdot(base, noise))
    eps = float(np.sqrt(max(1.0 - target * target, 0.0)) / target)
    return WaveState.from_complex(unit(base + eps * noise), cfg)


def main() -> int:
    cfg = WaveConfig(dim=4096)
    rng = np.random.default_rng(20261012)
    m = 32
    rows = np.stack(
        [unit(rng.standard_normal(cfg.dim) + 1j * rng.standard_normal(cfg.dim)) for _ in range(m)]
    )

    out = {"dim": cfg.dim, "k": m, "rows": []}
    for label, q in (
        ("unrelated", WaveState.random(cfg, rng=rng)),
        ("noisy_t0.90", noisy(rows, 0, 0.90, cfg, rng)),
        ("noisy_t0.99", noisy(rows, 0, 0.99, cfg, rng)),
    ):
        s = scores_of(q, rows)
        r = select(q, rows)
        rec = {
            "query": label,
            "argmax": int(r.index),
            "margin": float(r.margin),
            "legacy_entropy": float(r.entropy),
            "legacy_entropy_matches_pure": abs(
                float(r.entropy) - normalized_entropy(s)
            )
            < 1e-15,
            "score_std": float(s.std()),
            "score_max": float(s.max()),
        }
        for b in BETAS:
            rec[f"entropy_beta_{b}"] = entropy_at(s, b)
        out["rows"].append(rec)
        print(json.dumps(rec, sort_keys=True), file=sys.stderr)

    # -- decision ------------------------------------------------------
    unrelated = out["rows"][0]
    decisive = [r for r in out["rows"] if r["query"] != "unrelated"]
    legacy_gap = min(abs(d["legacy_entropy"] - unrelated["legacy_entropy"]) for d in decisive)
    beta_gap = min(
        abs(d["entropy_beta_26.1"] - unrelated["entropy_beta_26.1"]) for d in decisive
    )
    out["decision"] = {
        "legacy_entropy_gap_abs": legacy_gap,
        "beta26_entropy_gap_abs": beta_gap,
        "legacy_metric_is_scale_blind": bool(legacy_gap < 0.05),
        "beta26_metric_separates": bool(beta_gap > 0.5),
    }
    print(json.dumps(out["decision"], sort_keys=True))

    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
