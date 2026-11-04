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

Measured on CPU: 95.0 percent validation and 95.6 percent test on 8x8
digits, 23.4 percent next-character prediction over 96 classes, 77.5
percent one-step grid-world accuracy and 58.3 percent closed-loop success,
0.955 memory recall@1. Footprint 0.77 GiB at CI width on a 32 GiB budget.
Removing the optical feature path costs 59 points. With the Stiefel rows
anchored to a physical Fourier detector frame and an intensity-dependent
plate phase, the plates beat the identity bypass by 0.069 at 8x8 and 0.091
at 16x16 (mean 0.080 against the strict 0.10 bar). Gate D4 now passes on
both relational splits (0.765 against 0.704 standard, 0.481 against 0.406
low data) once the bilinear model gets 600 optimisation steps; the previous
0.710 ceiling was under-training, not feature rank. A new symbolic solver
navigates unseen rule graphs at 1.000 exact match for 1-3 hops and refuses
every unreachable query, and the agentic sidecar answers structural code
questions from Zone C with zero file reads and compiles ACE playbooks whose
steps it recovers exactly. Full numbers, ablations, continual-learning
tables, and non-claims are in [docs/HOLARCH.md](docs/HOLARCH.md).

No GPU was present in this checkout. Every receipt says `device: cpu`;
no Blackwell timing claim is made.

## Harness CLI: the eight tools

`henri/cli.py` and `henri/__main__.py` expose the interface the DeepSeek
Harness plugin calls. Every command prints one JSON document
`{ok, exitCode, summary, result}`, and `exitCode: 1` means a gate failed.

```powershell
python -m henri doctor --json
python -m henri ste100 --file docs/HOLARCH.md --mode descriptive --json
python -m henri jev --json
python -m henri holon --problem "close gate D4" --profile quick --json
python -m henri gates --profile quick --json
python -m henri ledger leaderboard --json
python -m henri zone-check --path henri/cli.py --zone 2 --json
python -m henri vast --json
```

`holon` runs the six-holon scientific graph (FRAME, MEASURE, FALSIFY,
ATTRIBUTE, RECORD, GATE). It stops at the first failure, proves the integrity
checker can reject a corrupted receipt, and appends to the immutable
hash-chained ontology at `data/holarch/ontology.jsonl`. See
[docs/HARNESS_CLI.md](docs/HARNESS_CLI.md).

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