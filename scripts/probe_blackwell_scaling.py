"""Blackwell scaling probe: latency decomposition, Dual Gram, Resonator.

Three questions, one device, no fabricated numbers.

Q1 G-ACT4 decomposition. The shipped harness times `veto().item()`, so its
   event pair brackets a device-to-host copy AND a synchronization. That
   number answers "what does a host-synchronous tick cost", not "what does
   the kernel cost". This probe separates the two, and adds a fused
   variant that reuses buffers. A contract that cannot be met by one
   implementation and can be met by another is an ARCHITECTURE finding,
   not a kernel verdict.

Q2 Dual Gram Stiefel retraction scaling (M x M versus D x D).
   Forms G = X X^H at M x M, retracts by eigh, and independently by
   Newton-Schulz on the inverse square root. Reports wall time, c_max and
   orthogonality error per M. Tests the repo's claim that the closed form
   is cheap and the architecture claim that Newton-Schulz is.

Q3 Tripartite Resonator convergence under Sagnac phase noise.
   Builds a 3-way circular-convolution binding over a unit-modulus
   codebook, factorizes by iterative unbinding, and sweeps the injected
   phase-noise scale eps. Reports per-iteration accuracy. `eps` maps to
   the Sagnac statistic by delta ~= eps^2 / 2, so the tau crossing
   eps = sqrt(2*tau) = 0.2936 is on the grid.

Evidence labels: OBSERVED for every timing and accuracy below.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

TAU_SPEC = 0.0431
CONTRACT_US = 50.0
EPS_AT_TAU = float((2.0 * TAU_SPEC) ** 0.5)


def pct(v, q):
    s = sorted(v)
    return float(s[min(len(s) - 1, int(q * len(s)))])


def stats_us(v):
    return {
        "p50_us": pct(v, 0.50), "p95_us": pct(v, 0.95), "p99_us": pct(v, 0.99),
        "mean_us": float(statistics.mean(v)), "max_us": float(max(v)),
        "n": len(v),
    }


# ----------------------------------------------------------------- Q1
def q1_latency(dev, dim=65536):
    rng = torch.Generator(device="cpu").manual_seed(20261012)
    q = torch.randint(0, 256, (2, dim), generator=rng)
    psi = (torch.exp(2j * torch.pi * q[0].double() / 256.0) / dim ** 0.5)
    axi = (torch.exp(2j * torch.pi * q[1].double() / 256.0) / dim ** 0.5)
    psi = psi.to(torch.complex64).to(dev)
    axi = axi.to(torch.complex64).to(dev)
    conj_axi = torch.conj(axi)

    def veto_shipped():
        z = psi * torch.conj(axi)
        return 1.0 - (z.real / torch.sqrt(z.real ** 2 + z.imag ** 2 + 1e-300)).mean()

    def veto_fused():
        # one temporary for the product, one for the magnitude, no conj copy
        z = psi * conj_axi
        return 1.0 - (z.real / z.abs().clamp_min(1e-30)).mean()

    out = {"dim": dim}
    for name, fn in (("shipped", veto_shipped), ("fused", veto_fused)):
        for _ in range(30):
            fn()
        torch.cuda.synchronize()

        # (a) host-synchronous: what run_gact4_latency.py measures
        sync_us = []
        for _ in range(200):
            s, e = (torch.cuda.Event(True), torch.cuda.Event(True))
            s.record(); v = float(fn().item()); e.record()
            torch.cuda.synchronize()
            sync_us.append(s.elapsed_time(e) * 1000.0)
        # (b) device-resident: K launches, one sync, amortized
        K = 2000
        s, e = (torch.cuda.Event(True), torch.cuda.Event(True))
        for _ in range(20):
            fn()
        torch.cuda.synchronize()
        s.record()
        for _ in range(K):
            fn()
        e.record()
        torch.cuda.synchronize()
        amort_us = s.elapsed_time(e) * 1000.0 / K
        # (c) empty launch cost, to expose the fixed floor
        s, e = (torch.cuda.Event(True), torch.cuda.Event(True))
        s.record()
        for _ in range(K):
            (psi * conj_axi)
        e.record()
        torch.cuda.synchronize()
        base_us = s.elapsed_time(e) * 1000.0 / K
        # (d) D2H alone
        sc = torch.zeros(1, device=dev)
        torch.cuda.synchronize()
        s, e = (torch.cuda.Event(True), torch.cuda.Event(True))
        s.record()
        for _ in range(200):
            float(sc.item())
        e.record()
        torch.cuda.synchronize()
        d2h_us = s.elapsed_time(e) * 1000.0 / 200

        out[name] = {
            "host_synchronous": stats_us(sync_us),
            "device_resident_amortized_us": amort_us,
            "bare_multiply_floor_us": base_us,
            "d2h_per_call_us": d2h_us,
        }
    return out


# ----------------------------------------------------------------- Q2
def q2_dual_gram(dev, dim=65536, ms=(512, 1024, 2048, 4096)):
    out = {"dim": dim, "rows": []}
    for m in ms:
        g = torch.Generator(device="cpu").manual_seed(1234 + m)
        q = torch.randint(0, 256, (m, dim), generator=g)
        ang = (2 * torch.pi * q.float() / 256.0)
        x = torch.polar(torch.full((m, dim), dim ** -0.5), ang).to(torch.complex64).to(dev)

        torch.cuda.synchronize(); t0 = time.perf_counter()
        gram = x @ x.conj().transpose(0, 1)
        torch.cuda.synchronize(); t_form = time.perf_counter() - t0

        torch.cuda.synchronize(); t0 = time.perf_counter()
        vals, vecs = torch.linalg.eigh(gram)
        torch.cuda.synchronize(); t_eigh = time.perf_counter() - t0

        # retraction through eigh: (X^H)^-1/2 X
        keep = vals > 1e-10 * vals.max()
        nk = int(keep.sum())
        inv_sqrt = (vecs[:, keep] * vals[keep].rsqrt().unsqueeze(0))
        torch.cuda.synchronize(); t0 = time.perf_counter()
        xo = (inv_sqrt.conj().transpose(0, 1) @ x)
        torch.cuda.synchronize(); t_apply = time.perf_counter() - t0

        # independent route: Newton-Schulz inverse square root, matmul only
        lam = float(gram.diagonal().sum().real)  # trace upper-bounds lambda_max
        s = gram / lam
        eye = torch.eye(m, device=dev, dtype=torch.complex64)
        y = eye.clone()
        ns_iters, ns_hist = 0, []
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for it in range(20):
            y2 = y @ y
            resid = float((eye - s @ y2).abs().max())
            ns_hist.append(resid)
            ns_iters = it + 1
            if resid < 1e-4:
                break
            y = 0.5 * (y @ (3.0 * eye - s @ y2))
        torch.cuda.synchronize(); t_ns = time.perf_counter() - t0
        # y -> s^-1/2 ; gram^-1/2 = y / sqrt(lam)
        xo_ns = ((y / lam ** 0.5).conj().transpose(0, 1) @ x)

        def cmax(mat):
            gg = mat @ mat.conj().transpose(0, 1)
            off = gg.abs().clone()
            off.fill_diagonal_(0.0)
            return float(off.max())

        def orth_err(mat):
            gg = mat @ mat.conj().transpose(0, 1)
            return float((gg - torch.eye(mat.shape[0], device=dev)).abs().max())

        kappa = float(vals.max() / vals[keep].min())
        out["rows"].append({
            "M": m, "bytes_X_MiB": round(m * dim * 8 / 2 ** 20, 1),
            "bytes_G_MiB": round(m * m * 8 / 2 ** 20, 1),
            "t_form_ms": t_form * 1e3, "t_eigh_ms": t_eigh * 1e3,
            "t_apply_ms": t_apply * 1e3,
            "t_retract_eigh_ms": (t_form + t_eigh + t_apply) * 1e3,
            "t_newton_schulz_ms": t_ns * 1e3, "ns_iters": ns_iters,
            "ns_residual_first": ns_hist[0], "ns_residual_last": ns_hist[-1],
            "cond_G": kappa, "cond_X_implied": kappa ** 0.5,
            "c_max_eigh": cmax(xo), "c_max_newton_schulz": cmax(xo_ns),
            "orth_err_eigh": orth_err(xo), "orth_err_newton_schulz": orth_err(xo_ns),
            "rows_kept": nk,
        })
        del x, gram, vecs, xo, xo_ns, y, eye, s
        torch.cuda.empty_cache()
    return out


# ----------------------------------------------------------------- Q3
def q3_resonator(dev, dim=65536, m=1024, iters=25, beta=26.10,
                 trials=8, eps_grid=(0.0, 0.05, 0.10, 0.15, 0.20, 0.2936, 0.40)):
    g = torch.Generator(device="cpu").manual_seed(777)
    q = torch.randint(0, 256, (m, dim), generator=g)
    ang = 2 * torch.pi * q.float() / 256.0
    X = torch.polar(torch.full((m, dim), dim ** -0.5), ang).to(torch.complex64)
    X = X.to(dev)
    fX = torch.fft.fft(X, dim=1)                       # codebook in spectral domain

    def bind3(a, b, c):
        return torch.fft.ifft(torch.fft.fft(a) * torch.fft.fft(b) * torch.fft.fft(c))

    def unbind2(target, b, c):
        return torch.fft.ifft(torch.fft.fft(target)
                              * torch.conj(torch.fft.fft(b))
                              * torch.conj(torch.fft.fft(c)))

    def readout(x):
        s = (X @ x.conj()).real
        p = torch.softmax(beta * s, dim=0)
        return int(s.argmax()), p

    out = {"dim": dim, "M": m, "beta": beta, "eps_at_tau": EPS_AT_TAU, "curves": []}
    for eps in eps_grid:
        acc_hist = [[] for _ in range(4)]
        t_hist = [[] for _ in range(4)]
        converged = []
        deltas = []
        for _ in range(trials):
            idx = torch.randperm(m, generator=g)[:3].to(dev)
            f1, f2, f3 = X[idx[0]], X[idx[1]], X[idx[2]]
            target = bind3(f1, f2, f3)
            clean = target
            if eps > 0:
                z = torch.randn(dim, generator=g).to(dev)
                z = z * (dim ** 0.5 / z.norm())
                target = target * torch.exp(1j * eps * z)
            zz = target * clean.conj()
            delta_meas = float(1.0 - (zz.real / zz.abs().clamp_min(1e-30)).mean())
            e2, e3 = X[torch.randint(0, m, (1,), generator=g).item()].to(dev), \
                     X[torch.randint(0, m, (1,), generator=g).item()].to(dev)
            e1 = e2
            hit_at = None
            for it in range(iters):
                e1 = unbind2(target, e2, e3)
                h1, p1 = readout(e1); e1 = (p1.to(torch.complex64) @ X)
                e2 = unbind2(target, e1, e3)
                h2, p2 = readout(e2); e2 = (p2.to(torch.complex64) @ X)
                e3 = unbind2(target, e1, e2)
                h3, p3 = readout(e3); e3 = (p3.to(torch.complex64) @ X)
                hits = (h1 == idx[0].item()) + (h2 == idx[1].item()) + (h3 == idx[2].item())
                ts = float(((e1.conj() * f1).sum()).real / e1.norm())
                if it < 3:
                    acc_hist[it].append(hits / 3.0)
                    t_hist[it].append(ts)
                if hits == 3 and hit_at is None:
                    hit_at = it + 1
                if it == 3:
                    pass
            # final-iteration accuracy at convergence horizon
            acc_hist[3].append(hits / 3.0)
            t_hist[3].append(ts)
            converged.append(hit_at)
        deltas.append(delta_meas)
        ok = [c for c in converged if c is not None]
        out["curves"].append({
            "eps": eps, "sagnac_delta_measured": float(statistics.mean(deltas)),
            "acc_iter1": float(statistics.mean(acc_hist[0])),
            "acc_iter2": float(statistics.mean(acc_hist[1])),
            "acc_iter3": float(statistics.mean(acc_hist[2])),
            "acc_final": float(statistics.mean(acc_hist[3])),
            "t_iter1": float(statistics.mean(t_hist[0])),
            "t_final": float(statistics.mean(t_hist[3])),
            "frac_converged": len(ok) / trials,
            "median_iters_to_all3": (statistics.median(ok) if ok else None),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="blackwell_scaling.json")
    ap.add_argument("--skip-q2", action="store_true")
    a = ap.parse_args()

    avail = torch.cuda.is_available()
    props = torch.cuda.get_device_properties(0) if avail else None
    rec = {
        "probe": "blackwell_scaling",
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "device": None if not avail else {
            "name": props.name, "cc": f"{props.major}.{props.minor}",
            "sm": props.multi_processor_count,
            "l2_bytes": getattr(props, "L2_cache_size", None),
            "mem_gb": round(props.total_memory / 2 ** 30, 2)},
        "tau_spec": TAU_SPEC, "contract_us": CONTRACT_US,
    }
    if not avail:
        rec["status"] = "BLOCKED_NO_CUDA"
        Path(a.out).write_text(json.dumps(rec, indent=2))
        print(json.dumps(rec, indent=2)); return 2

    dev = torch.device("cuda:0")
    rec["q1_latency"] = q1_latency(dev)
    Path(a.out).write_text(json.dumps(rec, indent=2, sort_keys=True))
    if not a.skip_q2:
        rec["q2_dual_gram"] = q2_dual_gram(dev)
        Path(a.out).write_text(json.dumps(rec, indent=2, sort_keys=True))
    rec["q3_resonator"] = q3_resonator(dev)
    rec["status"] = "COMPLETE"
    Path(a.out).write_text(json.dumps(rec, indent=2, sort_keys=True))
    print(json.dumps(rec, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
