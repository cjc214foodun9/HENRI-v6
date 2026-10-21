"""Resonator: separate the operator ceiling from the coupled dynamics.

Established before this probe:
  * A single unbind with TRUE keys recovers a factor at coherence 0.48
    with a readout gap of 0.905 over a 256-row codebook. The operator has
    signal.
  * The coupled 3-factor loop returns chance-level accuracy at every
    beta in {8, 26.1, 64} and every noise level including eps = 0.

So the failure is in the coupled dynamics or in the initialization, not
in the algebra. This probe measures the CEILING first, then ablates:

  k = 2 vs k = 3          is coupling depth the problem?
  annealed vs fixed beta  does saturation on iteration 1 lock a wrong row?
  random vs one-known     is there a convergence basin at all?

Evidence label: OBSERVED.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import torch


def build(dev, dim, m, seed=777):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randint(0, 256, (m, dim), generator=g)
    ang = 2 * torch.pi * q.float() / 256.0
    return torch.polar(torch.full((m, dim), dim ** -0.5), ang).to(torch.complex64).to(dev)


def main(dim=65536, m=1024, iters=30, trials=6, tau_beta=26.10):
    dev = torch.device("cuda:0")
    X = build(dev, dim, m)
    g = torch.Generator(device="cpu").manual_seed(4242)
    sk = dim ** -0.5

    def um(v):
        return torch.polar(torch.full_like(v.abs(), sk), torch.angle(v))

    def bnd(*xs):
        acc = torch.fft.fft(xs[0])
        for x in xs[1:]:
            acc = acc * torch.fft.fft(x)
        return torch.fft.ifft(acc)

    def unb(t, *keys):
        acc = torch.fft.fft(t)
        for kk in keys:
            acc = acc * torch.conj(torch.fft.fft(kk))
        return torch.fft.ifft(acc)

    def read(x, beta):
        s = (X @ x.conj()).real
        p = torch.softmax(beta * s, dim=0)
        return int(s.argmax()), p

    out = {"dim": dim, "M": m, "ceiling": {}, "arms": [], "latency": {}}

    # ---------------- ceiling: unbind with TRUE keys, no coupling
    coh, gaps, hits = [], [], []
    for _ in range(12):
        idx = torch.randperm(m, generator=g)[:3]
        f1, f2, f3 = (X[i].to(dev) for i in idx)
        t = bnd(f1, f2, f3)
        r = unb(t, f2, f3)
        coh.append(float(((r.conj() * f1).sum()).real / (r.norm() * f1.norm())))
        s = (X @ r.conj()).real
        top2 = torch.topk(s, 2).values
        gaps.append(float(top2[0] - top2[1]))
        hits.append(int(s.argmax()) == idx[0].item())
    out["ceiling"] = {"coherence_mean": float(statistics.mean(coh)),
                      "gap_mean": float(statistics.mean(gaps)),
                      "top1_hits": int(sum(hits)), "n": len(hits),
                      "M": m}

    # ---------------- arms
    def schedule(kind, it):
        if kind == "fixed":
            return tau_beta
        return float(min(tau_beta, 1.0 * (1.6 ** it)))

    for k in (2, 3):
        for kind in ("fixed", "annealed"):
            for init in ("random", "one_known"):
                accs, conv = [], []
                for _ in range(trials):
                    idx = torch.randperm(m, generator=g)[:k]
                    f = [X[i].to(dev) for i in idx]
                    t = bnd(*f)
                    if init == "one_known":
                        est = [f[0].clone()] + [X[torch.randint(0, m, (1,), generator=g).item()].to(dev) for _ in range(k - 1)]
                    else:
                        est = [X[torch.randint(0, m, (1,), generator=g).item()].to(dev) for _ in range(k)]
                    hit_at = None
                    for it in range(iters):
                        beta = schedule(kind, it)
                        for j in range(k):
                            others = [est[i2] for i2 in range(k) if i2 != j]
                            est[j] = unb(t, *others)
                            _, p = read(est[j], beta)
                            est[j] = um(p.to(torch.complex64) @ X)
                        ok = sum(int(read(est[j], tau_beta)[0]) == idx[j].item() for j in range(k))
                        if ok == k and hit_at is None:
                            hit_at = it + 1
                    accs.append(ok / k)
                    conv.append(hit_at)
                good = [c for c in conv if c is not None]
                out["arms"].append({
                    "k": k, "beta": kind, "init": init,
                    "acc_final": float(statistics.mean(accs)),
                    "frac_solved": len(good) / trials,
                    "median_iters": (statistics.median(good) if good else None),
                })

    # ---------------- latency, wall clock, one sync
    f = [X[i] for i in (0, 1, 2)]
    t = bnd(*f)
    for _ in range(20):
        e = [unb(t, f[1], f[2]), f[1], f[2]]
        _ = read(e[0], tau_beta)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    N = 200
    for _ in range(N):
        e = [unb(t, f[1], f[2]), f[1], f[2]]
        _ = read(e[0], tau_beta)
    torch.cuda.synchronize()
    one_unbind_one_read = (time.perf_counter() - t0) / N
    out["latency"] = {"one_unbind_plus_readout_us": one_unbind_one_read * 1e6,
                      "three_factor_iteration_us_est": one_unbind_one_read * 3 * 1e6,
                      "fifteen_iterations_us_est": one_unbind_one_read * 3 * 15 * 1e6}
    return out


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print(json.dumps({"status": "BLOCKED_NO_CUDA"}))
        sys.exit(2)
    rec = main()
    rec["status"] = "COMPLETE"
    rec["device"] = torch.cuda.get_device_name(0)
    Path("/workspace/res_v2.json").write_text(json.dumps(rec, indent=2, sort_keys=True))
    print(json.dumps(rec, indent=1, sort_keys=True))
