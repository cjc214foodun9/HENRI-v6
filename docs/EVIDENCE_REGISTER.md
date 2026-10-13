# Evidence register

Every material result carries one label:

`OBSERVED` `DERIVED` `INFERRED` `HYPOTHESIS` `FALSIFIED` `BLOCKED`

Pytest results are code-health evidence. They are not model-intelligence
scores and they are not capability claims.

## 1. What this build demonstrates

All rows below are measured at PRODUCTION dimension `D = 65536` unless
the row says otherwise. The full-D receipt is
`receipts/kernel_smoke_full_D.json`, sha256
`3586a0b4e22ca04244aeef50ea4b65c873ac7fdd65fbd5a0f9da49be2a142e3a`,
with the digest in the sidecar `.sha256` file and a readback check that
passes.

| Item | Evidence | Label |
|---|---|---|
| CPU test suite passes | measured return code | `OBSERVED` |
| Unit norm preserved by every operator | tests assert `abs(norm-1) < 1e-5` | `OBSERVED` |
| Zero dense `D x D` allocation | guard raises; low-rank paths only | `OBSERVED` |
| Slot bleed <= 0.0015 with zeroed slots | measured 0.0 exactly | `OBSERVED` |
| Gram crosstalk <= 0.12 after retraction | measured -7.16e-18 at D=65536 | `OBSERVED` |
| Hopfield energy non-increasing | measured `diff <= 1e-9` | `OBSERVED` |
| Veto separation >= 0.70 (directive 3) | 1.000 at D=65536, false_veto 0.000 | `OBSERVED` |
| Kuramoto reaches lock threshold 0.880 | r 0.0039 -> 0.9919 at D=65536 | `OBSERVED` |
| Receipt digest authenticates the bytes on disk | readback match `True` | `OBSERVED` |
| No resonator module exists | import raises `ModuleNotFoundError` | `OBSERVED` |
| Koopman norm drift at raw transition | 0.3514 at D=65536 | `OBSERVED` |

The full-D run takes 9.48 s wall clock. It attempts no task and makes no
capability claim.

## 2. Claims this build FALSIFIES

These come from the attached specifications. Each is measured, not
asserted.

### F1 - The Koopman unitarity claim is false

Spec section 6.2 states `K = diag(m) + A S B^dagger` with `S` skew-Hermitian
implies `||K psi|| = 1`. It does not. The composite has eigenvalues near
`1 + i*lambda`. Measured: `is_unitary()` is False, raw norm drift is
nonzero. The build applies an explicit projection and reports the drift.

### F2 - Binding recovery does not improve with dimension

Measured coherence by dimension (24 trials each):

| D | correct key | wrong key |
|---|---|---|
| 64 | 0.709 | -0.015 |
| 512 | 0.711 | +0.005 |
| 2048 | 0.708 | -0.007 |
| 4096 | 0.706 | +0.001 |

Correct-key coherence SATURATES near `1/sqrt(2) = 0.707`. What improves
is the SEPARATION, not the recovery. The residual floors near 0.29 and
must not be asserted lower.

### F3 - The autocorrelation of a random wave is not an impulse

Measured at D=512: peak mass 0.0030, argmax at index 293. A test that
expects `conv(a, conj(a))` to be a delta fails for a correct
implementation. Replaced by the true identity `conv(a, delta) = a`,
measured at 7.4e-15.

### F4 - Transposed-Gram retraction silently keeps degenerate rows

The symmetric form `vecs @ diag(1/sqrt) @ vecs^H @ x` is `(k,k) @ (k,D)`
and returns ALL k rows. Measured: 3 collinear rows in, 3 rows out, so
rank deficiency was never handled. The rectangular form returns 1 row.
Fixed.

### F5 - The projector convention matters by a factor of `||row||`

For orthonormal rows, the projector onto the row space is
`P = R^T conj(R)`. The pairing `R^H R` is a transpose-convention error
and leaves a residual of 31.7 against ~4e-14. Both are now pinned by
test, with the wrong one as a negative control.

### F6 - The family gap is in the SPREAD, not the signed mean

The retired repo quotes chance coherence 0.009792 (complex) vs 0.120168
(real), a 12.3x gap. Measured here as SIGNED cosine, both means are near
zero:

| family | signed mean | signed std | 3-sigma epsilon |
|---|---|---|---|
| complex | -0.0030 | 0.0299 | 0.0870 |
| real | +0.0012 | 0.0426 | 0.1252 |

So a threshold must be calibrated from the spread of the family that
will be evaluated. A fixed 0.35 saturates both.

### F7 - Fixed per-component noise scales are dimension-fragile

The pre-fix smoke built "gold" Sagnac pairs with `0.002 * randn(D)`. The
relative perturbation is `0.002 * sqrt(2D)`:

| D | relative perturbation |
|---|---|
| 512 | 0.064 |
| 65536 | 0.72 |

The same literal is a clean gold pair in CI and a DESTROYED one at
production dimension. Measured effect at D=65536: `false_veto = 1.000`,
`separation = 0.000`. The pre-fix receipt is preserved as
`receipts/kernel_smoke_full_D_prefix.json` (sha `2da9af43...`).

Fix: `perturb_with_coherence` builds `v + eps*n` with `n` projected
orthogonal to `v`, so `cos(v, pert) = 1/sqrt(1+eps^2)` holds exactly and
`eps = sqrt(1 - t^2) / t`. Post-fix at D=65536: `separation = 1.000`,
`false_veto = 0.000`.

A second-order error was found inside the fix itself. The first-order
form `eps = sqrt(2(1-t))` is a Taylor expansion valid only as `t -> 1`:

| target | achieved | error |
|---|---|---|
| 0.999 | 0.999 | 3e-06 |
| 0.900 | 0.911 | 1.1e-02 |
| 0.500 | 0.708 | 2.1e-01 |

### F8 - Kuramoto lock time scales as `ln(D)`

The order parameter of a random phase field is `r0 ~ 1/sqrt(D)`, which is
0.0039 at D=65536. Near `r = 0` the mean field gives `dr/dt ~ (K/2) r`,
so the time to lock scales as `2 ln(D) / K`, not as a fixed step count.

| quantity | predicted | measured |
|---|---|---|
| `t_lock` at D=65536, K=8 | 2.84 | lock reached at t=2.77 |
| r before | ~0.0039 | 0.003906 |
| r after | ~1.0 | 0.991938 |

A fixed 300-step budget at `dt=1e-3` therefore cannot lock at production
dimension. Fix: `lock_steps_for_dim(dim, coupling, dt, factor)` returns
`ceil(2 ln(D) / (K dt))`.

An earlier form hardcoded a `* 1000` factor, silently assuming
`dt = 1e-3` while the caller used `dt = 5e-3`. It returned 11091 steps
where 554 suffice -- a 20x overkill that costs real wall-clock.

### F9 - CI dimension and production dimension disagree QUALITATIVELY

Both F7 and F8 passed at D=512 and failed at D=65536. The suite was green
throughout. A green suite at CI dimension is not evidence about
production dimension for any dimension-scaled quantity.

## 3. Deliberate substitution

`memory.py` implements a rank-`r` factorization `S = V K^H` instead of
the specified dense `S in C^{D x D}`. At `D = 65536` the dense form is
about 68 GB; constraint C7 forbids it. Measured at `D=512, r=16`:
262144 bytes versus 4194304 for dense complex128. Recorded as a
substitution, not a port.

## 4. BLOCKED items

| Item | Reason |
|---|---|
| NotebookLM grounding | data-layer auth returns `ClientAuthenticationError`; see `docs/GROUNDING.md` |
| G-U3 swarm latency <= 50 us | needs RTX 5090 CUDA; host reports `cuda_available = False` |
| G-U7 VRAM <= 2.5 GiB | needs a GPU |
| G-U4 `I(H;Psi)/H(Psi) >= 0.95` | KSG estimator not implemented |
| G-M3 prefetch hit rate | no trajectory bank |
| G-M6 30-day aging | longitudinal simulation not run |
| BaTiO3 / Pockels / microcombs | hardware substrate not present |
| TimescaleDB hypertables | no live database in v6 |
| 448M decoder, 65M memory model | out of scope for the first kernel |
| License grant | RESOLVED 2026-10-12 - "Diverse Agency" License sealed; see `LICENSE-NOTICE.md` |

## 5. Non-claims

- No ARC-AGI task was attempted or solved by anything in this build.
- The retired repository's live benchmark score remains 0.0%.
- The sealed action readout measured LESS decisive than noise on live
  encoder waves (margin 0.023 vs 0.050). No floor is set from a random
  baseline.
- **The v6 full-D readout plateau is UNRESOLVED, not a confirmed wall.**
  [SUPERSEDED 2026-10-12 by section 7. Retained verbatim per the
  additive-retraction rule. Do not edit this bullet.]
  Measured at D=65536: margin 6.8e-05, entropy 1.000000. That value is
  consistent with BOTH (a) correct behaviour when the codebook is
  unrelated to the wave, and (b) a genuine inability to discriminate.
  `scripts/probe_readout_decisiveness.py` FAILED to decide between them:
  its `target_present` arm inserts the query as codebook row 0, so
  `margin = top - second` is 1.0 BY CONSTRUCTION and carries no
  information. The `unrelated` arm is itself D-dependent (margin 0.9999
  at D=512 versus 0.0019 at D=4096, unexplained). No claim in either
  direction is supported by this probe. A discriminating test needs a
  target that is present but NOT identical to the query.
- No biology-to-physics translation is claimed. The isomorphism in the
  retired repo is a design metaphor.
- No photonic energy figure is promoted to measurement.
- A green test suite means the code works. It does not mean the model
  is intelligent.

## 6. Reproduction

```bash
cd "HENRI v6"
python -m pytest                      # gate sweep, small D
python -m pytest -m slow --run-slow   # full D = 65536
```

## 7. Superseding record: the readout plateau is RESOLVED by a law

Appended 2026-10-12. This record **supersedes** the `UNRESOLVED` bullet in
section 5 without editing it, per the additive-retraction rule.

### The law

    margin ~= t * (1 - c_max)

- `t` = the query's real coherence with its intended target row.
- `c_max` = the target row's worst off-diagonal coherence against any other
  row in the codebook.

Derived before measurement, then tested against 28 pre-registered cells at
production dimension `D = 65536`.

| Metric | Measured | Bar | Label |
|---|---|---|---|
| max abs residual | `0.004947` | `0.05` | `OBSERVED` |
| R^2 (predicted vs measured) | `0.9999810` | - | `OBSERVED` |
| cells applicable | `28 / 28` | 0 excluded | `OBSERVED` |

Receipt: `receipts/margin_law.json`, raw sha256
`300e959eddbf7248ff5ade92c399ea893d021286137cfb8fe3c4f8c9986f168f`,
readback verified. The receipt is gitignored by repo policy; the digest is
recorded here instead, which is the established practice.

### What it explains

Section 5 recorded margin `6.8e-05` and entropy `1.000000` at `D=65536`,
and could not decide between a defect and correct behaviour. The law
decides: as `t -> 0`, the margin goes to zero for ANY codebook. The
recorded plateau **is** that limit. It is a property of the fixture.

### The engineering consequence

A decisive margin of `0.10` at `t = 0.90` requires

    1 - c_max >= 0.10 / 0.90 = 0.1111  =>  c_max <= 0.8889

Codebook geometry, not temperature, sets the operating margin. A monotone
transform cannot split a tie. This matches the retired repo's tie-plateau
finding.

### Correction to my own prior work

`scripts/decompose_plateau_verdict.py` returned `READOUT_DEFECT` at
`D=65536`. **That verdict is retracted.** Its conditions C1 and C2 were
written by me and are wrong: C1 demanded a tie-group size from a different
probe's tolerance, and C2 applied a mean-attractor test to raw-arm cells
where it cannot hold. The script is retained, not deleted. The law above
supersedes its verdict.

### Retracted label

`UNRESOLVED` is retired for this item. The residual claim is narrower than
either prior claim: the plateau is a measured identity on synthetic
codebooks. It is **not** evidence about learned engrams, because no learned
codebook has been tested yet. That test is Phase 2 work.

