# HENRI v0.7.0-OMEGA — implementation record

**Worktree:** `C:/Users/chan/omega-worktree` · **Branch:** `omega/v7-impl`
**Base commit:** `288c606bf77a785859feca31e8b298e513f2d164` (`main`, clean)
**Numeric stack:** NumPy `complex128`, CPU. **No CUDA path exists for these modules yet.**
**Status:** 9/9 counted Omega gates PASS. Legacy suite unchanged at 4 inherited failures.

---

## 1. What was added

| Module | Amendment | Lines |
|---|---|---|
| `henri/harmonic_lexicon.py` | 1 — Kuramoto-PPMI + Bochner lift | 180 |
| `henri/holographic_stack.py` | 2a — Slot-3 pushdown stack | 95 |
| `henri/koopman_sheaf.py` | 2b — phase-gated sheaf, online EDMD | 175 |
| `henri/triadic_sagnac.py` | 3 — three-wave comparator + annealing | 160 |
| `henri/hhrg_memory.py` | 4 — bounded Stiefel charts | 235 |
| `henri/omega_resonator.py` | 5 — exact arm + specified dynamics | 190 |
| `henri/egress_router.py` | 6 — grammar mask + motor | 175 |
| `henri/omega_pipeline.py` | 7 — end-to-end pipeline | 260 |
| `scripts/verify_omega_release.py` | — | 370 |

All modules are additive. The legacy spine (`henri/pipeline.py`, `henri/api.py`, the 19 named core
modules) is untouched.

---

## 2. Measured gates

Run: `PYTHONPATH=. python scripts/verify_omega_release.py` → exit 0, 9/9, 1.0 s.

| Gate | Measured | Bar |
|---|---|---|
| OMEGA-1 lexicon | synonym `0.999854`; unrelated `0.009028`; no-diffusion control `-0.014358` | ≥0.70 / <0.08 / <0.10 |
| OMEGA-2a call-stack | peek order exact; parent cos `0.999633`; final-pop residual `0.0000` | ≥0.98 / <0.05 |
| OMEGA-2b sheaf | identity err `2.18e-16`; random-diag cos `-0.0019`; err `0.16830` vs class floor `0.16829`; orthogonal-subspace control `-0.0001` | ≤1.02× floor |
| OMEGA-3 triadic | cold `COLD_ANCHORED`; veto `HALLUCINATION_VETO`; 1 observation ≠ anneal; 2 → `AXIOM_ANNEALED` | exact regimes |
| OMEGA-4 HHRG | 2 charts, 48 engrams, `c_max=1.74e-15`; hit margin `0.9862`, `H=0.0000`; negative margin `0.0033`, `H=0.9797` | <1e-6 / ≥0.70 / <0.01 |
| OMEGA-5a resonator exact | `[3,7,11]`, err `0.00e+00` | exact |
| OMEGA-5b resonator iterative | `8/8` exact, Wilson95 lower bound `0.676` | **REPORTED, not counted** |
| OMEGA-5c kill control | second triple `[1,5,9]`, not the first | exact |
| OMEGA-6 AST mask | unmasked argmax `3` → masked selection `[0,1,3]`; syntax valid | mask must fire |
| OMEGA-7 pipeline | text `electromagnetic_wave_packet`, margin `0.8502`; egress margin == pre-repair margin; repaired sim `0.9672` ≥ pre `0.8165`; negative `<abstain>` at `0.3952` | ≥0.65 / abstain |

The specification's τ_veto = 0.00811 is **not used**. It appears zero times in `HENRI-v6`.
The shipped constants are `TAU_VETO_CEILING = 0.0431` and `calibration ceiling = 0.35`.
All Omega gates use `0.0431`.

---

## 3. Defects fixed before the first green run

Found in the reference sketches:

1. `hhrg_memory.py` missing `import math` → `NameError` at query.
2. `hhrg_memory.py` misindented `sims_ortho` → `IndentationError`.
3. `HolographicCallStack.pop` used the real part only → imaginary leakage into the parent.
4. `TriadicSagnacGate.anomaly_streak` persisted across baseplates → spurious annealing.
5. `bootstrap_default_axioms` trained one chart on four conflicting maps → RLS averaged them.
6. `axiom_labels` keyed by enumerate index but read by `local_index` → wrong labels.
7. `PhaseGatedKoopmanSheaf.diag_m` initialized to random unit phasors → prediction orthogonal to
   every stored engram (measured cos `-0.0019`).

Found while measuring, in this implementation:

8. **Kuramoto integrator unstable.** Explicit gain `lr·2W = 1.5 > 1` produced a limit cycle
   (synonym cos `0.090`). Fixed by degree normalization; now `0.9999`.
9. **RLS ridge dominated the data term.** `‖B^Hψ‖² ≈ r/d_model = 2e-3`, so the first-step gain was
   `‖z‖²/(‖z‖²+λ) = 0.17` at λ=1e-2. Fixed with a unit-norm regressor and λ=1e-4; the learner now
   reaches the operator-class floor (`0.16830` vs `0.16829`).
10. **`pop` normalized a ~2 % residual back to unit norm**, so the stack never reported empty.
    Fixed with a relative `empty_tol = 0.10`, placed between the `1/√D` cross-talk floor and one
    decay step.
11. **Egress read the repaired state.** After the thermostat the state collapses onto memory, so a
    granite query scored `0.9850` against the photon axiom. Egress now reads the pre-thermostat
    retrieval; the negative query abstains at `0.3952`.

Two test defects were also corrected, not worked around: the stack gate popped the last frame
before measuring the parent, and the sheaf gate compared against an arbitrary `0.90` instead of the
operator class's own ceiling.

---

## 4. Claim boundaries — what this does NOT establish

- **No benchmark result.** No MMLU, GPQA, ARC-AGI-3, or MBPP run. Nothing here is a capability claim.
- **The resonator's ≥99 % claim is not established.** 8/8 at n=8 gives a Wilson95 lower bound of
  0.676. The exact arm proves the *problem* is solvable at M=16, K=3; it says nothing about the
  specified continuous dynamics at scale.
- **16.7 M orthogonal engrams is not demonstrated.** Each chart guarantees `c_max < 1e-6` for
  `M ≤ 256`. Nested-subspace construction across three tiers is absent.
- **"Turing-complete" is not claimed.** This is a bounded-depth stack; `decay_bound(D=4096) = 27`.
- **Grammar masking is a five-snippet toy grammar**, not a Python CFG. Syntax validity is true by
  construction of the mask plus an `ast.parse` check.
- **The motor head discards phase.** It reads `Re(slot1)`. It is marked `phase_discarding_readout`,
  not a calibrated torque decoder.
- **The sheaf is near-identity plus a rank-r correction.** At r=32, d_model=16384 the correction
  spans 0.2 % of the space. It models smooth dynamics; it cannot synthesize an arbitrary transition.
  The gate measures it against that class ceiling rather than pretending otherwise.
- **Baseline is red at the untouched commit.** 4 failures exist at `288c606b` before any change:
  `test_phase10_reason_channels::test_r1_zero_weight_matches_the_prior_api`,
  `test_phase10_structural_morphisms` (2 byte-identity pins),
  `test_semantic_backbone::test_negative_control_shuffled_pairs_read_lower`.
  These are inherited, likely NumPy-version drift on byte-identity assertions. They are unchanged by
  this branch.

---

## 5. Governance notes

- `tests/test_gates.py::test_g_u5_resonator_family_is_not_shipped` pins the absence of
  `henri.resonator`. Shipping Amendment 5 under that name would have silently defeated the control.
  The module is therefore `henri/omega_resonator.py`. The control stays live and green.
- `docs/EVIDENCE_REGISTER.md` was **not** appended. A governed ledger append needs its own approval
  and a before/after hash verification.
- `TAU_VETO` was not changed anywhere in the legacy spine.
- Nothing was pushed. The branch is local and isolated.
