"""Case (a) mean-attractor discriminator.

Additive analysis. It tests the EXPLANATORY claim behind the plateau --
not a threshold -- so it cannot move a bar.

Claim under test
----------------
The received explanation for the plateau is that an equidistant query
drives the selection softmax uniform, so the Hopfield CCCP fixed point
is the MEAN PATTERN ("Case (a) global-mean attractor"). If that is
correct, the cleaned wave must have near-unit coherence with the
normalized codebook mean:

    |<cleaned, mean(rows)>| ~ 1.0

This is independent evidence. It does not read the pre-registered
entropy/margin thresholds from probe_readout_geometry.py, and it does
not modify that receipt.

Prediction, stated before running
---------------------------------
isotropic_gauss, isotropic_phasor : coherence with the mean ~ 1.0
clustered                         : LOWER, because the codebook's own
                                    structure pulls the fixed point off
                                    the global mean toward a cluster.

If isotropic families do NOT show ~1.0 coherence with the mean, the
Case (a) explanation is falsified and must be reported as such.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from henri.config import WaveConfig  # noqa: E402
from henri.hopfield import cleanup  # noqa: E402
from henri.qfhrr import phasor_vector  # noqa: E402
from henri.readout import normalized_entropy, select  # noqa: E402
from henri.state import WaveState  # noqa: E402

SEED = 20261012
DIM = 65536
K = 32


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n < 1e-30:
        raise ValueError("cannot normalize a zero or non-finite vector")
    return v / n


def build(name: str, cfg: WaveConfig, rng: np.random.Generator) -> np.ndarray:
    d = cfg.dim
    if name == "isotropic_gauss":
        return np.stack(
            [unit(rng.standard_normal(d) + 1j * rng.standard_normal(d)) for _ in range(K)]
        )
    if name == "isotropic_phasor":
        return np.stack([unit(phasor_vector(d, rng)) for _ in range(K)])
    if name == "clustered":
        sigma, per = 0.35, K // 4
        rows = []
        for _ in range(4):
            center = unit(rng.standard_normal(d) + 1j * rng.standard_normal(d))
            for _ in range(per):
                noise = unit(rng.standard_normal(d) + 1j * rng.standard_normal(d))
                rows.append(unit(center + sigma * noise))
        return np.stack(rows)
    raise ValueError(f"unknown family {name!r}")


def main() -> int:
    cfg = WaveConfig(dim=DIM)
    rng = np.random.default_rng(SEED)
    out = {"probe": "case_a_mean_attractor", "seed": SEED, "dim": DIM, "rows": []}

    for fam in ("isotropic_gauss", "isotropic_phasor", "clustered"):
        rows = build(fam, cfg, rng)
        mean = unit(rows.mean(axis=0))
        q = WaveState.random(cfg, rng=rng)

        for arm in ("raw", "cleaned"):
            wave = q
            if arm == "cleaned":
                wave, _ = cleanup(q, rows)
            cos_mean = abs(complex(np.vdot(wave.psi, mean)))
            r = select(wave, rows)
            rec = {
                "family": fam,
                "arm": arm,
                "coherence_with_codebook_mean": float(cos_mean),
                "margin": float(r.margin),
                "entropy": float(r.entropy),
                "tie_group_size": int(r.group_size),
            }
            out["rows"].append(rec)
            print(
                "%-17s %-8s cos_mean=%.6f margin=%.6f entropy=%.6f ties=%d"
                % (
                    fam,
                    arm,
                    cos_mean,
                    r.margin,
                    r.entropy,
                    r.group_size,
                ),
                file=sys.stderr,
            )

    iso = [
        r["coherence_with_codebook_mean"]
        for r in out["rows"]
        if r["family"] != "clustered" and r["arm"] == "cleaned"
    ]
    clus = [
        r["coherence_with_codebook_mean"]
        for r in out["rows"]
        if r["family"] == "clustered" and r["arm"] == "cleaned"
    ]
    out["prediction"] = {
        "isotropic_cleaned_cos_mean": iso,
        "clustered_cleaned_cos_mean": clus,
        "case_a_supported": bool(iso and min(iso) > 0.99),
        "clustered_lower_than_isotropic": bool(clus and iso and clus[0] < min(iso)),
    }
    print(json.dumps(out["prediction"], sort_keys=True), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
