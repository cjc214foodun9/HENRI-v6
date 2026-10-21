"""Tripartite resonator: discriminating ablation and convergence law.

Q3 in `probe_blackwell_scaling.py` returned accuracy ~0 at EVERY noise
level, including eps = 0. A resonator that fails without noise has a
broken iteration, not a noise problem. The suspect is the missing
projection back onto the wave manifold between iterations: the softmax
readout returns a CONVEX COMBINATION of codebook rows, whose entries are
not unit-modulus, so it is not a legal key for the next unbinding.

This probe runs both arms. The unprojected arm is the negative control:
if both arms fail, the defect is elsewhere and the projection claim is
falsified.

Also measures per-iteration latency, so the "< 90 us for 15 steps" claim
gets a number instead of an assertion.

Evidence label: OBSERVED for every accuracy and timing.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch


def build(dev, dim, m, seed=777):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randint(0, 256, (m, dim), generator=g)
    ang = 2 * torch.pi * q.float() / 256.0
    X = torch.polar(torch.full((m, dim), dim ** -0.5), ang).to(torch.complex64).to(dev)
    return X


def run(dim=65536, m=1024, iters=25, trials=6, betas=(8.0, 26.10, 64.0),
        eps_grid=(0.0, 0.10, 0.2936, 0.40), project=True, time_it=False):
    dev = torch.device("cuda:0")
    X = build(dev, dim, m)
    g = torch.Generator(device="cpu").manual_seed(4242)

    def unit_modulus(v):
        """Project onto |z_d| = 1/sqrt(D) keeping the phase."""
        return torch.polar(torch.full_like(v.abs(), dim ** -0.5), torch.angle(v))

    def bind3(a, b, c):
        return torch.fft.ifft(torch.fft.fft(a) * torch.fft.fft(b) * torch.fft.fft(c))

    def unbind2(t, b, c):
        return torch.fft.ifft(torch.fft.fft(t)
                              * torch.conj(torch.fft.fft(b))
                              * torch.conj(torch.fft.fft(c)))

    def readout(x, beta):
        s = (X @ x.conj()).real
        p = torch.softmax(beta * s, dim=0)
        return int(s.argmax()), p

    def step(target, e1, e2, e3, beta):
        e1 = unbind2(target, e2, e3)
        h1, p1 = readout(e1, beta)
        e1 = p1.to(torch.complex64) @ X
        e2 = unbind2(target, e1, e3)
        h2, p2 = readout(e2, beta)
        e2 = p2.to(torch.complex64) @ X
        e3 = unbind2(target, e1, e2)
        h3, p3 = readout(e3, beta)
        e3 = p3.to(torch.complex64) @ X
        if project:
            e1, e2, e3 = unit_modulus(e1), unit_modulus(e2), unit_modulus(e3)
        return e1, e2, e3, (h1, h2, h3)

    out = {"dim": dim, "M": m, "project": project, "iters": iters,
           "trials": trials, "curves": []}

    for beta in betas:
        for eps in eps_grid:
            accs, ts, conv = [], [], []
            for _ in range(trials):
                idx = torch.randperm(m, generator=g)[:3]
                f = [X[i].to(dev) for i in idx]
                target = bind3(f[0], f[1], f[2])
                if eps > 0:
                    z = torch.randn(dim, generator=g).to(dev)
                    z = z * (dim ** 0.5 / z.norm())
                    target = target * torch.exp(1j * eps * z)
                e1, e2, e3 = (X[torch.randint(0, m, (1,), generator=g).item()].to(dev)
                              for _ in range(3))
                hit_at = None
                for it in range(iters):
                    e1, e2, e3, hits = step(target, e1, e2, e3, beta)
                    k = sum(hits[i] == idx[i].item() for i in range(3))
                    if k == 3 and hit_at is None:
                        hit_at = it + 1
                accs.append(k / 3.0)
                ts.append(float(((e1.conj() * f[0]).sum()).real / (e1.norm() * f[0].norm())))
                conv.append(hit_at)
            ok = [c for c in conv if c is not None]
            out["curves"].append({
                "beta": beta, "eps": eps, "acc_final": float(statistics.mean(accs)),
                "t_final": float(statistics.mean(ts)),
                "frac_all3": len(ok) / trials,
                "median_iters": (statistics.median(ok) if ok else None),
            })

    if time_it:
        x = X[0].clone(); y = X[1].clone()
        for _ in range(50):
            step(bind3(x, y, X[2]), x, y, X[2], 26.10)
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(100):
            step(bind3(x, y, X[2]), x, y, X[2], 26.10)
        e.record()
        torch.cuda.synchronize()
        out["step_latency_us"] = e.elapsed_time(s) * 1000.0 / 100
        out["fifteen_step_us"] = out["step_latency_us"] * 15
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="resonator_ablation.json")
    a = ap.parse_args()
    if not torch.cuda.is_available():
        print(json.dumps({"status": "BLOCKED_NO_CUDA"})); return 2
    rec = {"probe": "resonator_ablation",
           "device": torch.cuda.get_device_name(0),
           "torch": torch.__version__,
           "projected": run(project=True),
           "unprojected_control": run(project=False),
           "status": "COMPLETE"}
    Path(a.out).write_text(json.dumps(rec, indent=2, sort_keys=True))
    for arm in ("unprojected_control", "projected"):
        print("==", arm)
        for c in rec[arm]["curves"]:
            print("  beta=%-6.2f eps=%-6.4f acc=%-6.3f t=%-7.4f all3=%-5.2f med=%s"
                  % (c["beta"], c["eps"], c["acc_final"], c["t_final"],
                     c["frac_all3"], c["median_iters"]))
        if "step_latency_us" in rec[arm]:
            print("  step=%.1f us  15 steps=%.1f us"
                  % (rec[arm]["step_latency_us"], rec[arm]["fifteen_step_us"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
