# HENRI v6 build constraints — what NOT to build

Status: `OBSERVED` (each item is a measured refutation from the retired
repository's audit register and the forensic probe specification).
Source of record: `HENRI 7B SWARM/.worktrees/basal-syncytium/HENRI V2/
references/henri_natural_intelligence_audit_register.md`.

This file is the PR checklist. A change that reintroduces an item below
is rejected.

| # | Do NOT build / repeat | Measured evidence | Label |
|---|---|---|---|
| C1 | Tripartite Resonator as a replacement for linear least-squares | resonator **0/16** real ARC tasks; treatment **+0.144086** vs incumbent **+0.395123**; shuffled control wins 7/16. "It must not ship." | `FALSIFIED` |
| C2 | Hardcoded Sagnac `epsilon = 0.35` | chance coherence for real wave family is **0.120168** vs **0.009792** complex (12.3x). Gate saturated; AUC **0.4982** = chance | `FALSIFIED` |
| C3 | Transferring `T* = 0.038316` / `beta* = 26.10` across readouts | fitted on the **functor-score** readout; the Hopfield snap calibration is **unmeasured** | `FALSIFIED` |
| C4 | `max / min` extrema-ratio metrics | `min` is an order statistic; shrinks to 0 with N. Ratio degraded 1.4444 -> 2.200 with sample size | `FALSIFIED` |
| C5 | Static spec-string hashing as ingress | `AUC_shipped = 0.5012`, `AUC_LOO = 0.4982` = chance. `I(psi(X); Y) = 0` | `FALSIFIED` |
| C6 | Noise-fill of unused Clifford slots | documented 99.85% noise-fill defect. Unused slots must be **explicitly zero** | `FALSIFIED` |
| C7 | Dense `D x D` allocations | at `D = 65536` complex64 is ~34 GB. Zero instances allowed | `DERIVED` |
| C8 | Photonics / membrane / gap-junction / apoptosis claims | in-tree hits: `membrane_potential` 6, `gap_junction` 11, `apoptosis` 6, `batio3` 11, `pockels` 6 — **named only**, never simulated | `BLOCKED` |
| C9 | Sparse-delta closed loop as specified | `VOID_DOES_NOT_REPRODUCE_DENSE_BASELINE`; dense baseline 0.0000/0.6833 vs receipt 0.1944/0.8125; sign retrieval degenerate `positive_frac = 1.000` | `FALSIFIED` |
| C10 | Isotropic weight decay dominating the preconditioned drive | `-gamma W` ~0.51 vs drive ~0.08 => **6:1**; gradient cosine positive on only **74/150** steps = chance | `FALSIFIED` |
| C11 | `top-k` ranking on a tie plateau | `argmax` ties always; `argmax`/`argsort` agree on only ~2.6%. Use max-group membership + non-monotone within-group key | `FALSIFIED` |
| C12 | Setting a readout floor from the random-wave distribution | live encoder waves are **less decisive than noise** (margin 0.023 vs 0.050; higher entropy) | `FALSIFIED` |
| C13 | Vendoring retired-repo files | the retired tree contains every defect above. Reference only | `DERIVED` |
| C14 | Claiming NotebookLM grounding | data-layer auth returns `ClientAuthenticationError` (see `docs/GROUNDING.md`) | `BLOCKED` |
| C15 | Treating "norm conservation" as a result | follows from Hermiticity by construction; establishes no learned property | `DERIVED` |
| C16 | Treating `T_eff -> 0 <=> delta_Sagnac <= 0.0431` as a theorem | it is a **definition** written into the document, not a derivation | `FALSIFIED` |

## Two conflations to refuse

- `+0.6306` and `T* = 0.038316` belong to the functor-score readout.
  Never attach them to the Hopfield snap without a new measurement.
- "The low score was not an ontological failure of the wave core" is
  half-right. The encoder carries **0.7333** of **0.7833**. The core is
  real; the *prescription* (C1) is a measured regression.

## Positive scope

The audit confirms these primitives were genuinely implemented:
`sagnac_veto` (160 hits), `stiefel_retraction` (272), `hopfield` (273),
`kuramoto` (227), `sgld` (39). These define the honest v6 surface.

## Spec claims corrected in this build

- `koopman.py`: "`S` skew-Hermitian enforces `||K psi|| = 1`" is false.
  `K = diag(m) + A S A^H` has eigenvalues `1 + i*lambda`, not unit
  modulus. The build measures raw drift and projects explicitly.
- `memory.py`: the spec's dense `S in C^{D x D}` delta rule is forbidden
  by C7. Replaced by a rank-`r` factorized form, recorded as a
  substitution.
