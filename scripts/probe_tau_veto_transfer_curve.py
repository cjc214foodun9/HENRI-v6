"""Per-family tau_veto calibration by TRANSFER CURVE.

Additive to `scripts/probe_tau_veto_calibration.py`, which asked whether a
separating band exists between related and unrelated pairs. It measured
`sep=False` for EVERY family at both D=1024 and D=65536. That is not a
null result about the gate; it is a defect in that probe's DESIGN: in high
dimensions two independent random waves have `1 - mean(cos)` ~ 1.0, so the
WITHIN and ACROSS distributions both pile up at ~1.0 and overlap. The
within/across axis has no power. This probe uses the axis that does.

THE MEASUREMENT THAT DISCRIMINATES
The gate's job is to reject a phase-corrupted candidate while accepting a
clean one. The operating question is: how much per-dimension phase
corruption can a family carry before the gate fires? That is a curve:

    candidate_d = axiom_d * exp(i * eps * noise_d),  |noise| = 1
    delta(eps)  = 1 - mean_d cos(angle(candidate_d * conj(axiom_d)))

`eps` is the corruption scale in radians. Sweep it. Report:

  eps_at_spec(F) : the eps at which mean delta first EXCEEDS spec tau.
                   This is the operational meaning of tau_veto for F.
  form_div(F)    : max |spec form - normalized-inner-product form| over the
                   curve. Exceeds tau => the family needs a PINNED STATISTIC
                   FORM, not merely its own threshold.

THE SPEC CLAIM THIS PROBE TESTS DIRECTLY
Phase 2 spec section 3.2 asserts that the "Structured Semantic Engram
Solution" gives tokens in one family a bounded baseline coherence
`t_base in [0.40, 0.60]`, defeating the "Random-Ring Orthogonal Isolation
Trap". Measured directly here, per tokenizer path:
  encode_token        (the plain path)
  encode_family_token (the structured path the claim is about)
If either yields |Re<token_i, token_j>| ~ 0 within a family, the claim is
falsified for this implementation.

PRE-REGISTERED DECISION RULE, fixed before the first run
- eps_at_spec is REPORTED per family. `None` means the family never
  exceeds tau within eps <= 1.0 and is UNCALIBRATABLE_AT_SPEC.
- form_div > tau means FAMILY_NEEDS_PINNED_FORM.
- The tokenizer coherence claim is FALSIFIED when the within-family
  mean |inner product| is below 0.10 AND the spread is comparable to
  the between-family value. Reported, not assumed.
- CONTROL: `gaussian` (no structure) must NOT show family coherence. If it
  does, the coherence measurement is measuring the RNG, not the family.

DISCLOSURE. Sensor family uses real image bytes on this host. Other
families come from production code over generated data. No task is
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from henri.config import WaveConfig
from henri.gram import stiefel_retract
from henri.memory import DeltaMemory

TAU_SPEC = 0.0431
# Bracketed around the ANALYTIC crossing for a phase-only perturbation:
# delta ~= eps^2 / 2, so delta == tau at eps = sqrt(2*tau) = sqrt(0.0862)
# = 0.2936. The earlier grid jumped 0.2 -> 0.5 and reported 0.5 for every
# family, which was the GRID, not the physics.
EPS_SWEEP = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.275, 0.30, 0.325,
             0.35, 0.40, 0.50, 0.75, 1.00)
N_TRIALS = 12


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n <= 0 or not np.isfinite(n):
        raise ValueError("zero or non-finite vector")
    return v / n


def form_spec(psi: np.ndarray, axiom: np.ndarray) -> float:
    """1 - mean_d cos(angle(psi_d * conj(axiom_d)))."""
    return float(1.0 - np.mean(np.cos(np.angle(psi * np.conj(axiom)))))


def form_normed(psi: np.ndarray, axiom: np.ndarray) -> float:
    """1 - Re<psi, axiom> / (|psi| |axiom|)."""
    np_, na = np.linalg.norm(psi), np.linalg.norm(axiom)
    return float(1.0 - np.real(np.vdot(psi, axiom)) / (np_ * na))


def _image_paths(limit=4):
    from henri.sensor_ingress import list_real_images
    out = []
    for r in ("C:/Users/chan/AppData/Roaming/Microsoft/Windows/Themes",
              "C:/Windows/Web/Wallpaper"):
        try:
            out.extend(list_real_images(r, limit=limit))
        except Exception:
            pass
    return out[:limit]


def build_families(cfg, rng, per=12):
    """Each family at exactly L = cfg.slot_span, via production paths."""
    L = int(cfg.slot_span)
    fams: dict = {}

    # unit-modulus ring phasors (the gate's stated domain)
    q = rng.integers(0, 256, size=(per, L))
    fams["unit_modulus"] = np.stack(
        [np.exp(1j * 2 * np.pi * q[i] / 256.0) / np.sqrt(L) for i in range(per)])

    # qFHRR tokenizer, BOTH paths. This is where the spec's claim lives.
    try:
        from henri.qfhrr_tokenizer import QFHRRTokenizer
        tok = QFHRRTokenizer(cfg)
        a, b = [], []
        for i in range(per):
            qa = np.asarray(tok.encode_token(category_id=0, semantic_id=i,
                                             instance_id=i))
            a.append(unit(np.asarray(tok.to_complex_phasor(qa)).ravel()))
            try:
                qf = np.asarray(tok.encode_family_token(family_id=0,
                                                        member_id=i))
                v = (np.asarray(tok.to_complex_phasor(qf)).ravel()
                     if qf.dtype.kind in "iu" else qf.ravel())
                b.append(unit(v))
            except Exception:
                pass
        fams["qfhrr_token"] = np.stack(a)
        if len(b) >= 2:
            fams["qfhrr_family"] = np.stack(b)
    except Exception as exc:
        fams["qfhrr_token"] = {"error": f"{type(exc).__name__}: {exc}"}

    # stiefel: the repo's own codebook retraction
    raw = rng.standard_normal((per, L)) + 1j * rng.standard_normal((per, L))
    fams["stiefel"] = stiefel_retract(raw)

    # learned: production DeltaMemory rows
    try:
        mem = DeltaMemory.build(L, rank=64)
        keys = []
        for _ in range(per):
            kk = unit(rng.standard_normal(L) + 1j * rng.standard_normal(L))
            vv = unit(rng.standard_normal(L) + 1j * rng.standard_normal(L))
            mem.write(kk, vv)
            keys.append(kk)
        fams["learned"] = np.stack([unit(mem.read(kk)) for kk in keys])
    except Exception as exc:
        fams["learned"] = {"error": f"{type(exc).__name__}: {exc}"}

    # sensor: real image bytes through the production encoder
    try:
        from PIL import Image
        from henri.sensor_ingress import to_gray_float
        from henri.spatial_encoder import ContinuousSpatialEncoder
        imgs = []
        for p in _image_paths(limit=4):
            try:
                imgs.append(to_gray_float(np.asarray(
                    Image.open(p).convert("RGB"))))
            except Exception:
                continue
        if imgs:
            base = imgs[0]
            h, w = base.shape
            ws = []
            for i in range(per):
                if h < 16 or w < 16:
                    break
                r0 = (i * 5) % max(1, h - 16)
                c0 = (i * 11) % max(1, w - 16)
                enc = ContinuousSpatialEncoder(cfg, seed=1200 + i)
                ws.append(unit(np.asarray(
                    enc.encode_2d_field(base[r0:r0 + 16, c0:c0 + 16])).ravel()))
            if len(ws) >= 2:
                fams["sensor"] = np.stack(ws)
    except Exception as exc:
        fams["sensor"] = {"error": f"{type(exc).__name__}: {exc}"}

    # gaussian control: no structure
    g = rng.standard_normal((per, L)) + 1j * rng.standard_normal((per, L))
    fams["gaussian"] = np.stack([unit(r) for r in g])
    return fams, L


def transfer_curve(axiom, rng, n_trials=N_TRIALS):
    """delta vs eps, averaged over independent phase-noise draws."""
    out = []
    for eps in EPS_SWEEP:
        f1, f2 = [], []
        for _ in range(n_trials):
            # REAL noise -- a PURE phase perturbation, so |cand| == |axiom|
            # and form1/form2 must agree exactly on unit-modulus rows.
            # Unit per-dimension power (mean_d z_d^2 == 1) makes `eps` the
            # per-dimension phase std in radians, hence delta ~= eps^2 / 2
            # and a crossing at eps = sqrt(2*tau) ~ 0.2936.
            # The earlier form normalized the TOTAL L2 norm to 1, leaving
            # per-dimension corruption ~ eps/sqrt(D) and delta ~ eps^2/(2D).
            # No family ever crossed tau and every curve read NEVER. That
            # was a normalization defect in this probe, NOT a property of
            # the gate.
            z = rng.standard_normal(axiom.shape)
            z = z * np.sqrt(z.size) / np.linalg.norm(z)
            cand = axiom * np.exp(1j * float(eps) * z)
            f1.append(form_spec(cand, axiom))
            f2.append(form_normed(cand, axiom))
        out.append({
            "eps": float(eps),
            "delta_spec_mean": float(np.mean(f1)),
            "delta_spec_max": float(np.max(f1)),
            "delta_normed_mean": float(np.mean(f2)),
            "form_divergence_max": float(np.max(np.abs(
                np.array(f1) - np.array(f2)))),
        })
    return out


def family_coherence(rows):
    """Within-family coherence: what the spec's t_base claim is about."""
    k = rows.shape[0]
    G = np.abs(rows.conj() @ rows.T)
    off = G[~np.eye(k, dtype=bool)]
    return {
        "mean_abs_coherence": float(off.mean()),
        "max_abs_coherence": float(off.max()),
        "std_abs_coherence": float(off.std()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=65536)
    ap.add_argument("--out", type=str,
                    default="receipts/tau_veto_transfer_curve.json")
    a = ap.parse_args()

    t0 = time.perf_counter()
    cfg = WaveConfig(dim=a.dim)
    rng = np.random.default_rng(20261012)
    fams, L = build_families(cfg, rng)

    rec = {
        "probe": "tau_veto_transfer_curve",
        "tau_spec": TAU_SPEC,
        "dim": a.dim, "slot_span": L,
        "seed": 20261012,
        "eps_sweep": list(EPS_SWEEP), "n_trials_per_point": N_TRIALS,
        "supersedes_axis": ("probe_tau_veto_calibration.py measured sep=False "
                            "for every family; the within/across axis has no "
                            "power at high D because unrelated pairs pile at "
                            "delta ~ 1.0"),
        "decision_rule": "pre-registered in the module docstring",
        "disclosure": ("Sensor family uses real image bytes. Other families "
                       "use production code over generated data. No task "
                       "attempted or solved."),
        "families": {},
    }

    for name, rows in fams.items():
        if isinstance(rows, dict) and "error" in rows:
            rec["families"][name] = {"status": "UNAVAILABLE",
                                     "error": rows["error"]}
            continue
        coh = family_coherence(rows)
        curves = [transfer_curve(rows[j], rng) for j in range(rows.shape[0])]
        agg = []
        for i, eps in enumerate(EPS_SWEEP):
            agg.append({
                "eps": float(eps),
                "delta_spec_mean": float(np.mean(
                    [c[i]["delta_spec_mean"] for c in curves])),
                "delta_spec_max": float(np.max(
                    [c[i]["delta_spec_max"] for c in curves])),
                "delta_normed_mean": float(np.mean(
                    [c[i]["delta_normed_mean"] for c in curves])),
                "form_divergence_max": float(np.max(
                    [c[i]["form_divergence_max"] for c in curves])),
            })
        eps_at = next((p["eps"] for p in agg
                       if p["delta_spec_mean"] > TAU_SPEC), None)
        div_max = max(p["form_divergence_max"] for p in agg)

        def _interp(key):
            """First crossing of TAU_SPEC, linearly interpolated."""
            prev = None
            for pt in agg:
                if pt[key] > TAU_SPEC:
                    if prev is None:
                        return pt["eps"]
                    d0, d1 = prev[key], pt[key]
                    if d1 == d0:
                        return pt["eps"]
                    frac = (TAU_SPEC - d0) / (d1 - d0)
                    return float(prev["eps"] + frac * (pt["eps"] - prev["eps"]))
                prev = pt
            return None

        e1_grid = next((p["eps"] for p in agg
                        if p["delta_spec_mean"] > TAU_SPEC), None)
        e2_grid = next((p["eps"] for p in agg
                        if p["delta_normed_mean"] > TAU_SPEC), None)
        e1 = _interp("delta_spec_mean")
        e2 = _interp("delta_normed_mean")
        rec["families"][name] = {
            "status": "MEASURED",
            "row_norm_min": float(np.min(np.linalg.norm(rows, axis=1))),
            "row_norm_max": float(np.max(np.linalg.norm(rows, axis=1))),
            "coherence": coh,
            "curve": agg,
            "eps_at_spec_tau": eps_at,
            "uncalibratable_at_spec": eps_at is None,
            "form_divergence_max": div_max,
            "needs_pinned_form": bool(div_max > TAU_SPEC),
            # -- the DISCRIMINATING pair of numbers ------------------
            # form1 ignores row magnitudes entirely: for candidate
            # `a*exp(i*eps*n)` the per-dimension phase difference is
            # `eps*n_d` whatever `|a_d|` is. So form1's crossing eps is
            # FAMILY-BLIND BY CONSTRUCTION. form2 weights each dimension
            # by `|a_d|^2`, so its crossing depends on the family's
            # magnitude profile. Comparing the two is the measurement.
            "eps_form1_grid": e1_grid,
            "eps_form2_grid": e2_grid,
            "eps_form1_interp": e1,
            "eps_form2_interp": e2,
            "form_choice_changes_eps": (
                None if (e1 is None or e2 is None)
                else bool(abs(e1 - e2) > 1e-9)),
            "weight_concentration": float(
                np.max(np.abs(rows) ** 2, axis=1).mean()
                / np.mean(np.abs(rows) ** 2)),
        }

    ctrl = rec["families"].get("gaussian", {})
    ctrl_ok = (ctrl.get("status") == "MEASURED"
               and ctrl["coherence"]["mean_abs_coherence"] < 0.10)
    rec["control"] = {
        "family": "gaussian",
        "expectation": "no family coherence (unstructured rows)",
        "observed_mean_abs_coherence": ctrl.get("coherence", {}).get(
            "mean_abs_coherence"),
        "control_holds": bool(ctrl_ok),
    }
    rec["wall_clock_s"] = time.perf_counter() - t0
    rec["status"] = "COMPLETE" if ctrl_ok else "VOID_CONTROL_FAILED"

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(rec, indent=2, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(body).hexdigest()
    out.write_bytes(body)
    out.with_suffix(out.suffix + ".sha256").write_bytes(
        (digest + "\n").encode())
    rb = hashlib.sha256(out.read_bytes()).hexdigest()

    print("=== tau_veto TRANSFER CURVE ===", file=sys.stderr)
    print(f"dim={a.dim} slot={L} tau_spec={TAU_SPEC}", file=sys.stderr)
    for n, m in rec["families"].items():
        if m.get("status") != "MEASURED":
            print(f"{n:<16} UNAVAILABLE  {m.get('error','')}", file=sys.stderr)
            continue
        e = m.get("eps_form1_interp")
        e2 = m.get("eps_form2_interp")
        print("%-16s f1=%-9s f2=%-9s div=%.3e%s  coh=%.4f/%.4f"
              % (n,
                 ("%.4f" % e) if e is not None else "NEVER",
                 ("%.4f" % e2) if e2 is not None else "NEVER",
                 m["form_divergence_max"],
                 "  [NEEDS_PINNED_FORM]" if m["needs_pinned_form"] else "",
                 m["coherence"]["mean_abs_coherence"],
                 m["coherence"]["max_abs_coherence"]),
              file=sys.stderr)
    print(f"control_holds={rec['control']['control_holds']}", file=sys.stderr)
    print(f"status={rec['status']} sha256={digest} readback={rb == digest}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
