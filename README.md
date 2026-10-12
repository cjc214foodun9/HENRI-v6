# HENRI v6

Project HENRI v6 is a **fresh, CPU-runnable wave-mechanical kernel**.
It rebuilds the validated physics core from first principles.

It does **not** port the retired repository. That repository is
read-only reference material. Its audit register supplies the
"what NOT to build" list in `docs/HENRI_V6_CONSTRAINTS.md`.

## Status

`OBSERVED` at the level of code health only. Test results are
software evidence. They are not model-intelligence scores.

## Scope

Built: state space, Clifford binding, Hopfield cleanup, Koopman
transition, Kuramoto locking, gram/Stiefel orthogonalization,
calibrated Sagnac veto, viscoelastic kernel, low-rank delta memory,
Hopfield/centroid readout, end-to-end pipeline.

Deferred with `BLOCKED` labels: BaTiO3 photonics, TimescaleDB
hypertables, the 448M decoder, FP8/NVFP4 quantization, L2 pinning,
Triton kernels, GPU latency and VRAM gates.

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
