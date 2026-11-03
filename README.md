# HENRI v6

Project HENRI v6 is a **fresh, CPU-runnable wave-mechanical kernel**.
It rebuilds the validated physics core from first principles.

It does **not** port the retired repository. That repository is
read-only reference material. Its audit register supplies the
"what NOT to build" list in `docs/HENRI_V6_CONSTRAINTS.md`.

## Status

`OBSERVED` at the level of code health only. Test results are
software evidence. They are not model-intelligence scores.

## Holarch: the three-zone pipeline

`henri/holarch/` is the wired system: Zone A lean continuous-learning
swarm, Zone B BaTiO3 diffractive kernel with Sagnac homodyne readout,
and Zone C holographic TimescaleDB/NVMe memory. It composes the v6
primitives without changing them.

```powershell
python -m henri.holarch selftest     # 7 integration checks
python -m henri.holarch physics      # Sagnac, Curie-Weiss, heater authority
python -m henri.holarch memory --n 2000
python -m henri.holarch budget       # 32 GiB accounting
python -m henri.holarch train --steps 300 --name holarch_small
```

Measured on CPU: 93.0 percent validation and 94.3 percent test on 8x8
digits, 25.3 percent next-character prediction over 96 classes, 73.8
percent one-step grid-world accuracy and 54.2 percent closed-loop success,
56.3 percent on held-out relational triples, 0.965 memory recall@1.
Footprint 0.91 GiB at production width on a 32 GiB budget. The optical
stage contributes +22.7 points over identity optics. Directive gates D1 to
D3 pass; D4 is a measured failure. Full numbers, ablations,
continual-learning tables, and non-claims are in
[docs/HOLARCH.md](docs/HOLARCH.md).

No GPU was present in this checkout. Every receipt says `device: cpu`;
no Blackwell timing claim is made.

## Scope

Built: state space, Clifford binding, Hopfield cleanup, Koopman
transition, Kuramoto locking, gram/Stiefel orthogonalization,
calibrated Sagnac veto, viscoelastic kernel, low-rank delta memory,
Hopfield/centroid readout, end-to-end pipeline.

Deferred with `BLOCKED` labels: BaTiO3 photonics, TimescaleDB
hypertables, the 448M decoder, FP8/NVFP4 quantization, L2 pinning,
Triton kernels, GPU latency and VRAM gates.

## Compute scope: CPU only, by decision

`run_pipeline` is a `numpy`/`complex128` implementation at `D = 65536`. It is
**CPU-only**. There is no CUDA path for the pipeline, and a GPU benchmark of the
kernel is therefore not possible today. `omega_torch.py` carries a small torch
subset (`KoopmanSheafT`, `hopfield_readout_t`, `stiefel_retract_t`); it is not the
pipeline.

This is a recorded decision, not an oversight. Porting the pipeline is a project
with its own budget, and the parity test it needs does not exist yet. Measure the
kernel on CPU, label the machine, and never present a serving-model GPU run as a
kernel GPU claim. The decision record is
`../deepseek-harness/default-workspace/HENRI/docs/KERNEL_SCOPE.md`.

Evidence label: `DERIVED` from the module inventory, `BLOCKED` for any GPU claim.

## Evidence policy

Every material result carries one label:

- `OBSERVED` - returned by a recorded execution.
- `DERIVED` - calculated from observed data by a stated rule.
- `INFERRED` - reasoned from observed data, not directly measured.
- `HYPOTHESIS` - proposed mechanism, not yet tested.
- `FALSIFIED` - contradicted by a valid test.
- `BLOCKED` - required evidence or execution is unavailable.

Pytest results are code-health evidence. They are not capability claims.

## Run

```bash
python -m pytest
```

Full-dimension slow run:

```bash
python -m pytest -m slow --run-slow
```