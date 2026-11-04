# Holarch receipts

Evidence for `docs/HOLARCH.md`. Each JSON is produced by one command and
carries the git HEAD, the configuration, the seed, and a SHA-256 of its own
bytes in the sidecar `.sha256`.

| File | Command |
|---|---|
| `holarch_small.json` | `python -m henri.holarch train --steps 300 --name holarch_small --fresh --dagger 20` |
| `ablation_vision.json` | `python scripts/run_holarch_ablation.py --steps 300` |
| `gate_d1.json` | `python scripts/run_holarch_gate_d1.py --steps 450 --seeds 3` |
| `relational.json` | `python scripts/run_holarch_relational.py --steps 600 --seeds 3` |
| `noise_vision.json` | `python scripts/run_holarch_noise.py --steps 300 --sigma 0.25 --seeds 5` |
| `continual_vision.json` | `python scripts/run_holarch_continual.py --steps 150` |
| `symbolic.json` | `python scripts/run_holarch_symbolic.py --seeds 3` |
| `sidecar.json` | `python scripts/run_holarch_sidecar_demo.py --steps 120` |
| `holarch_production.json` | `python -m henri.holarch train --steps 600 --name holarch_production --fresh --dagger 20` |
| `agent_eval.json` | `python scripts/run_holarch_agent_eval.py` |
| `c_cycle.json` | `python scripts/run_holarch_c_cycle.py` (cycle C verification ledger) |
| `budget.json` | `python -m henri.holarch budget` |
| `gate_d1_k*.json` | the Decision A1 detector-width sweep; kept as evidence of the arm that failed |
| `diagnostics/holarch_small_rel1x.json` | the Decision B1 one-times coupling control |

Labels follow the v6 policy. Every number here is `OBSERVED` on CPU. No
GPU timing exists, and no row may be presented as a Blackwell result. The
budget receipt records the CUDA probe as `BLOCKED` with the probe that was
run.

The decision gates are recorded in `docs/HOLARCH.md` sections 7, 12, 14 and
17. A gate that did not pass is written as a failure with the measured
number. This revision closes Gate D1 strict (+0.111 mean plate margin),
Gate D2 (OOD 1.000, false veto 0.0067), Gate D3 on the Decision A2 sign rule
(positive at every sigma from 0.25 to 1.00) and Gate D4 (0.779 vs 0.704 and
0.488 vs 0.406). Gate D4b is unchanged from the previous receipt. The
hardware column stays BLOCKED because no CUDA device is attached.
