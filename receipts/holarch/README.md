# Holarch receipts

Evidence for `docs/HOLARCH.md`. Each JSON is produced by one command and
carries the git HEAD, the configuration, the seed, and a SHA-256 of its own
bytes in the sidecar `.sha256`.

| File | Command |
|---|---|
| `holarch_small.json` | `python -m henri.holarch train --steps 300 --name holarch_small --fresh --dagger 20` |
| `ablation_vision.json` | `python scripts/run_holarch_ablation.py --steps 300` |
| `gate_d1.json` | `python scripts/run_holarch_gate_d1.py --steps 300 --seeds 3` |
| `relational.json` | `python scripts/run_holarch_relational.py --steps 600 --seeds 3` |
| `noise_vision.json` | `python scripts/run_holarch_noise.py --steps 300 --sigma 0.25 --seeds 5` |
| `continual_vision.json` | `python scripts/run_holarch_continual.py --steps 150` |
| `symbolic.json` | `python scripts/run_holarch_symbolic.py --seeds 3` |
| `sidecar.json` | `python scripts/run_holarch_sidecar_demo.py --steps 120` |

Labels follow the v6 policy. Every number here is `OBSERVED` on CPU. No
GPU timing exists, and no row may be presented as a Blackwell result.

The decision gates are recorded in `docs/HOLARCH.md` sections 7, 12 and 14.
A gate that did not pass is written as a failure with the measured number.
In this revision Gate D4 passes on both relational splits at the declared
600-step budget, and the symbolic solver passes its abstention gate. The
strict Gate D1 bar is still open at 0.080 of the required 0.10.
