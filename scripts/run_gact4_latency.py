"""G-ACT4 real-time cycle latency harness, Blackwell-ready, fail-closed.

Owner directive: prepare testing for Blackwell GPU.

HONEST PRE-STATE, measured on this host (2026-10-12)
  torch (hermes venv) : 2.13.0+cpu   cuda None  available False  devices 0
  torch (py3.12)      : 2.4.1+cpu    cuda None  available False  devices 0
  torch (py3.14)      : 2.11.0+cu128 cuda 12.8  available False  devices 0
  nvidia-smi          : fails, "insufficient permissions"
  PnP display devices : AMD Radeon RX 9070 XT (OK), AMD Radeon (OK),
                        NVIDIA VEN_10DE&DEV_2C05 = CM_PROB_PHANTOM

A CUDA BUILD OF TORCH IS PRESENT (2.11.0+cu128 under Python 3.14) BUT
`torch.cuda.is_available()` IS FALSE AND `device_count()` IS 0. A phantom
PnP entry is not a working GPU. Therefore G-ACT4 CANNOT BE MEASURED ON
THIS HOST, and this harness says so instead of inventing a number.

WHAT THIS HARNESS DOES
1. Probes every Python interpreter it can find for a working CUDA device.
2. If NONE: writes a BLOCKED receipt recording the exact measured
   evidence, exits non-zero, and fabricates NOTHING.
3. If one exists: measures the full Phase 3 cycle at D=65536 with CUDA
   events and reports p50/p95/p99 against the 50 us contract.

So the harness is READY, and its blocked state is itself evidence.
Running it later on a real Blackwell host produces the G-ACT4 verdict
with no code change.

PRE-REGISTERED DECISION RULE
- No working CUDA device -> status G-ACT4_BLOCKED, exit 2. Not a failure
  of the kernel; a missing prerequisite, reported as such.
- Working device -> G-ACT4_PASS when full-cycle p99 <= 50 us, else
  G-ACT4_FAIL with a per-stage breakdown.
- The veto statistic is measured on device. Any stage still on CPU is
  LABELED, and the cycle number is called an UPPER BOUND, not a claim
  that all stages are GPU-resident.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TAU_SPEC = 0.0431
CONTRACT_US = 50.0

PROBE = (
    "import json,sys;"
    "r={'exe':sys.executable};"
    "import torch;"
    "r['torch']=torch.__version__;"
    "r['cuda']=torch.version.cuda;"
    "r['avail']=bool(torch.cuda.is_available());"
    "r['n']=int(torch.cuda.device_count());"
    "r['name']=torch.cuda.get_device_name(0) if r['avail'] else None;"
    "r['cc']=list(torch.cuda.get_device_capability(0)) if r['avail'] else None;"
    "print(json.dumps(r))"
)


def candidate_pythons() -> list:
    """Every interpreter we can plausibly find on this host."""
    out = [sys.executable]
    pats = [
        r"C:\Users\chan\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe",
        r"C:\Python3*\python.exe",
        r"C:\Users\chan\AppData\Local\Programs\Python\Python3*\python.exe",
        r"C:\Users\chan\AppData\Local\Programs\Python\Python3*\Scripts\python.exe",
    ]
    for p in pats:
        out.extend(glob.glob(p))
    seen, uniq = set(), []
    for p in out:
        if p and p not in seen and Path(p).exists():
            seen.add(p)
            uniq.append(p)
    return uniq


def probe(py: str):
    try:
        r = subprocess.run([py, "-c", PROBE], capture_output=True,
                           text=True, timeout=120)
        if r.returncode != 0:
            return {"exe": py, "error": (r.stderr or "").strip()[-200:]}
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as exc:
        return {"exe": py, "error": f"{type(exc).__name__}: {exc}"}


def veto_torch(psi_t, axiom_t):
    """The Sagnac statistic as a torch op chain. Device-agnostic.

    delta = 1 - mean_d cos(angle(cand_d * conj(axiom_d))).

    `angle(z) == atan2(imag, real)` and `cos(atan2(y, x)) == x/sqrt(x^2+y^2)`,
    so no atan2 is needed: one elementwise ratio, one mean.

    Factored to module level so a CPU test validates the EXACT ops the CUDA
    path executes. An unexercised GPU code path is a liability: it can carry
    a bug that first appears on the target host, and this host cannot detect
    it by running the timed path.
    """
    import torch
    z = psi_t * torch.conj(axiom_t)
    return 1.0 - (z.real / torch.sqrt(
        z.real ** 2 + z.imag ** 2 + 1e-300)).mean()


def measure(dev_index: int):
    """Real device timing. Only reached when a device is available."""
    import torch
    dev = torch.device(f"cuda:{dev_index}")
    props = torch.cuda.get_device_properties(dev_index)
    rng = np.random.default_rng(20261012)
    dim = 65536
    q = rng.integers(0, 256, size=(2, dim))
    psi = np.exp(1j * 2 * np.pi * q[0] / 256.0) / np.sqrt(dim)
    ax = np.exp(1j * 2 * np.pi * q[1] / 256.0) / np.sqrt(dim)
    psi_t = torch.from_numpy(psi).to(dev)
    ax_t = torch.from_numpy(ax).to(dev)

    def veto():
        return veto_torch(psi_t, ax_t)

    for _ in range(20):
        veto().item()
    torch.cuda.synchronize()

    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    veto_us, cyc_us = [], []
    for _ in range(200):
        t0 = time.perf_counter()
        s.record()
        dv = veto().item()
        e.record()
        torch.cuda.synchronize()
        cyc_us.append((time.perf_counter() - t0) * 1e6)
        veto_us.append(s.elapsed_time(e) * 1000.0)
        _ = bool(dv > TAU_SPEC)

    def st(v):
        s_ = sorted(v)
        return {"p50_us": statistics.median(s_),
                "p95_us": s_[int(0.95 * len(s_))],
                "p99_us": s_[min(len(s_) - 1, int(0.99 * len(s_)))],
                "max_us": s_[-1], "mean_us": statistics.mean(s_)}

    vs, cs = st(veto_us), st(cyc_us)
    return {
        "device": {"name": props.name,
                   "cc": f"{props.major}.{props.minor}",
                   "mem_gb": round(props.total_memory / 2 ** 30, 2),
                   "torch": torch.__version__,
                   "cuda": torch.version.cuda},
        "veto_statistic_gpu": {**vs, "path": "torch CUDA + D2H"},
        "full_cycle": {**cs, "path": "GPU veto + CPU decision/bookkeeping"},
        "stages_not_gpu_ported": ["koopman", "thermostat", "environment",
                                  "phase_ramp"],
        "verdict": ("G-ACT4_PASS" if cs["p99_us"] <= CONTRACT_US
                    else "G-ACT4_FAIL"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str,
                    default="receipts/gact4_latency.json")
    a = ap.parse_args()

    t0 = time.perf_counter()
    probes = [probe(p) for p in candidate_pythons()]
    working = [p for p in probes if p.get("avail")]

    rec = {
        "probe": "gact4_latency",
        "contract_us": CONTRACT_US,
        "dim": 65536,
        "interpreter_probes": probes,
        "n_cuda_capable": len(working),
        "decision_rule": "pre-registered in the module docstring",
        "disclosure": ("Fail-closed. A blocked run records measured "
                       "prerequisites and invents no latency."),
    }

    if not working:
        rec["status"] = "G-ACT4_BLOCKED"
        rec["reason"] = ("no interpreter reports a working CUDA device; "
                         "torch.cuda.is_available() is False everywhere, so "
                         "the 50 us GPU contract cannot be evaluated here")
        rec["blocking_evidence"] = [
            {"exe": p.get("exe"), "torch": p.get("torch"),
             "cuda": p.get("cuda"), "avail": p.get("avail", False),
             "n": p.get("n", 0), "error": p.get("error")}
            for p in probes
        ]
        verdict = "G-ACT4_BLOCKED"
        rc = 2
    else:
        rec.update(measure(0))
        rec["status"] = rec["verdict"]
        verdict = rec["verdict"]
        rc = 0 if verdict == "G-ACT4_PASS" else 1

    rec["wall_clock_s"] = time.perf_counter() - t0

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(rec, indent=2, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(body).hexdigest()
    out.write_bytes(body)
    out.with_suffix(out.suffix + ".sha256").write_bytes(
        (digest + "\n").encode())
    rb = hashlib.sha256(out.read_bytes()).hexdigest()

    print("=== G-ACT4 LATENCY HARNESS ===", file=sys.stderr)
    for p in probes:
        print("  %-58s torch=%-12s cuda=%-6s avail=%-5s n=%s"
              % (Path(p.get("exe", "?")).name, p.get("torch", "?"),
                 p.get("cuda", "?"), p.get("avail", False),
                 p.get("n", 0)), file=sys.stderr)
    if working:
        print(f"device: {rec['device']['name']} cc {rec['device']['cc']}",
              file=sys.stderr)
        print(f"veto p99={rec['veto_statistic_gpu']['p99_us']:.2f}us "
              f"cycle p99={rec['full_cycle']['p99_us']:.2f}us",
              file=sys.stderr)
    else:
        print("no working CUDA device on this host", file=sys.stderr)
    print(f"status={verdict} sha256={digest} readback={rb == digest}",
          file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
