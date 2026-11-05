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
| `holarch_production.json` | `python -m henri.holarch train --steps 600 --profile medium --name holarch_production --fresh --dagger 40 --coupling 0.5` |
| `holarch_production_k24_c0*.json` | the Decision D2 coupling and DAgger sweep |
| `holarch_medium.json` | `python -m henri.holarch train --steps 600 --profile medium --dagger 40 --coupling 0.5 --name holarch_medium --fresh --backend sqlite` (Cycle E1) |
| `holarch_medium.pt` | the converged E1 checkpoint, written beside the receipt and mounted by the sidecar |
| `e_cycle.json` | `python scripts/run_holarch_e_cycle.py` (cycle E verification ledger) |
| `d_cycle.json` | `python scripts/run_holarch_d_cycle.py` (cycle D verification ledger) |
| `f_cycle.json` | `python scripts/run_holarch_f_cycle.py` (cycle F ledger) |
| `code_lang_eval.json`, `code_lang_l1/l2/l3.json` | `python scripts/run_holarch_code_lang.py` (L1-L3) |
| `g_cycle.json` | `python scripts/run_holarch_g_cycle.py` (cycle G ledger) |
| `g1_wave_lm.json`, `g2_parametric_synth.json`, `g3_depth2_lookahead.json`, `g4_persistence.json` | the four Cycle G directive receipts |
| `agent_eval.json` | `python scripts/run_holarch_agent_eval.py` (64 queries in Cycle H) |
| `h_cycle.json` | `python scripts/run_holarch_h_cycle.py --stages h1,h2,h3,h4,eval` (cycle H ledger) |
| `h1_entropy_gated_lm.json` | H1: entropy-gated wave blend on the pinned 67e9b25 corpus |
| `h2_multiplan_synth.json` | H2: 24 base + 12 compositional + 8 two-plan pipelines + cold start |
| `h3_api_surface.json` | H3: checkpoint rotor, plan/verify over REST and MCP |
| `h4_delta_index.json` | H4: single-file delta index, exactness and latency |
| `c_cycle.json` | `python scripts/run_holarch_c_cycle.py` (cycle C verification ledger) |
| `budget.json` | `python -m henri.holarch budget` |
| `gate_d1_k*.json` | the Decision A1 detector-width sweep; kept as evidence of the arm that failed |
| `diagnostics/holarch_small_rel1x.json` | the Decision B1 one-times coupling control |

Labels follow the v6 policy. Every number here is `OBSERVED` on CPU. No
GPU timing exists, and no row may be presented as a Blackwell result. The
budget receipt records the CUDA probe as `BLOCKED` with the probe that was
run.

The decision gates are recorded in `docs/HOLARCH.md` sections 7, 12, 14,
17 and 21. A gate that did not pass is written as a failure with the measured
number. Cycle E adds `e_cycle.json`: E1 passes at closed-loop 0.7083,
E2 passes with the mounted medium checkpoint and 44 valid receipts, and E3
records Path B with CUDA BLOCKED. This revision closes Gate D1 strict (+0.111 mean plate margin),
Gate D2 (OOD 1.000, false veto 0.0067), Gate D3 on the Decision A2 sign rule
(positive at every sigma from 0.25 to 1.00) and Gate D4 (0.779 vs 0.704 and
0.488 vs 0.406). Gate D4b is unchanged from the previous receipt. The
hardware column stays BLOCKED because no CUDA device is attached.
