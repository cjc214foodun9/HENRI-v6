# Holarch receipts

Evidence for `docs/HOLARCH.md`. Each JSON is produced by one command and
carries the git HEAD, the configuration, the seed, and a SHA-256 of its own
bytes in the sidecar `.sha256`.

| File | Command |
|---|---|
| `holarch_small.json` | `python -m henri.holarch train --steps 300 --name holarch_small --fresh --dagger 8` |
| `ablation_vision.json` | `python scripts/run_holarch_ablation.py --steps 150` |
| `relational.json` | `python scripts/run_holarch_relational.py --steps 400` |
| `continual_vision.json` | `python scripts/run_holarch_continual.py --steps 150` |

Labels follow the v6 policy. Every number here is `OBSERVED` on CPU. No
GPU timing exists, and no row may be presented as a Blackwell result.

The directive gates are recorded in `docs/HOLARCH.md` section 7. A gate
that did not pass is written as a failure with the measured number. The
relational gate is such a row.
