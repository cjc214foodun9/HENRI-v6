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

## 8. Phase 2 ingress: measured gate results

Appended 2026-10-12. Production dimension `D = 65536`, `D_slot = 16384`.
Receipt `receipts/ingress_smoke_full_D.json`, sha256
`daf36e403f08a8d8be8d4dee2836839ddadd17908880ca19d3e563bab51cddb0`,
readback verified against the sidecar. Wall clock 0.19 s.

### 8.1 Gates that PASS

| Gate | Claim | Measured | Bar | Label |
|---|---|---|---|---|
| G-ING1 | wrapped translation is a pure phase factor | `2.589e-15` | `< 1e-12` | `OBSERVED` |
| G-ING1b | magnitude invariance under that shift | `1.986e-14` | `< 1e-12` | `OBSERVED` |
| G-ING2 | `\|\|Psi_d\| - 1/sqrt(D)\|` | `1.735e-18` | `< 1e-15` | `OBSERVED` |
| G-ING4 | ring unbinding recovery coherence | `1.0000000000` | `>= 0.9999` | `OBSERVED` |
| G-U2 | mean squared off-diagonal crosstalk | `6.112605e-05` | `<= 0.12` | `OBSERVED` |
| G-U1 | energy outside active slots | `0.0` | `== 0.0` | `OBSERVED` |
| G-ING3 | tail sum `\|Psi_d\|^2` over `d >= 16384` | `0.0` | `== 0.0` | `OBSERVED` |

`|Psi_d|` target is exactly `0.0078125 = 1/128`, so the G-ING2 bar is
reachable in complex128. Bleed is asserted as **equality with `0.0`**, not
a tolerance: normalization multiplies by one scalar, and zero times any
finite scalar is exactly zero. A tolerance would hide a stray write.

### 8.2 Negative controls that FAIL as required

| Control | Measured | Must be |
|---|---|---|
| G-ING1 on a border-dropping shift | `2.467e-02` | `> 1e-3` |
| G-ING4 via circular convolution | `0.713168` | `< 0.9999` |

The convolution control reproduces F2's `~1/sqrt(2)` saturation. This is
F2 preserved as a live test. The two binding algebras are therefore
separable by measurement, not by prose.

### 8.3 F10 - the spec's incommensurate-frequency requirement is unsatisfiable

The Phase 2 spec asks for a frequency ratio
`omega_x / omega_y not in Q`. **That is impossible on a discrete torus.**
Exact G-ING1 covariance requires `exp(i 2pi u x)` to be periodic in `x`
with period 1, which holds if and only if `u` is an integer; and the ratio
of two integers is rational.

Measured, both ways: fractional frequencies give a wrapped-shift residual
near `1e-1`, integer frequencies give `2.589e-15`. The first
implementation of this module used fractional frequencies and failed.

Resolution: HENRI v6 uses a **distinct integer frequency lattice**
(`128 x 128` at `D_slot = 16384`, `(0,0)` excluded by construction) and
**reports** its realized collinearity rather than claiming a property the
discrete torus cannot have. Measured lattice:
`all_integer = True`, `distinct_pairs = 16384`, `primitive_pairs = 10043`,
`collinear_pairs = 6341`. The collinear count is disclosed, not hidden.

### 8.4 Defects found in my own Phase 2 work

All three were found by running, not by reading. Each is recorded in the
module or test docstring at the site of the fix.

1. **Unit error in the phase factor.** `x` is normalized to `[0, 1)` while
   the shift `dx` is in pixels. The phase factor needs `dx / W`. Detected
   by a hand-check that disagreed with the derivation, before the gate ran.
2. **Fractional frequencies break torus covariance.** Cause of F10 above.
3. **Non-deterministic Kuramoto fixture.** `KuramotoState.from_wave` fell
   through to `np.random.default_rng(None)`, which seeds from OS entropy.
   Measured on one wave: identical phases, `omega` differing by `4.840156`
   between two calls. Separately, the locking test used 400 steps where the
   module's own `lock_steps_for_dim(512, 8.0, 1e-3)` predicts 1560, i.e.
   `0.26x` budget. The combination made
   `test_locking_raises_order_parameter` a coin flip: it failed inside one
   full-suite run and passed in isolation. Both causes are fixed.
   `from_wave` now fails closed without an explicit `rng` or `seed`,
   mirroring the existing rule in `kuramoto_step`. **The assertion itself is
   unchanged**, so no bar moved.

### 8.5 Section 3.2 semantic path, measured

| Quantity | Measured | Band |
|---|---|---|
| intra-family coherence | `0.500306` | `[0.35, 0.65]` |
| cross-family coherence | `-0.011584` | `abs < 0.10` |
| shared baseplate fraction | `0.4998` | - |

Directive 2's literal `encode_token` yields mutually quasi-orthogonal
tokens. Section 3.2 rejects that for semantic families. **Both paths are
implemented and both are tested against their own claim**, because the two
specification sections demand different behavior.

### 8.6 Adapter between the two qFHRR surfaces

Two modules carry the term `qFHRR`. They are distinct and the boundary is
declared and tested rather than assumed:

| Module | Surface | Domain | Operator |
|---|---|---|---|
| `henri/qfhrr.py` | continuous unit-modulus phasors | angular, float | pointwise product |
| `henri/qfhrr_tokenizer.py` | integer ring `Z_256` | discrete, uint8 | modular addition |

`to_complex_phasor` is the declared adapter. Three tests pin it: the lift
is a legal operand for the continuous module; lifting commutes with
binding up to a single global scale that `to_sphere` removes; and ring
unbinding is bitwise exact on the lift.

### 8.7 Non-claims for Phase 2

- The spatial fields are **synthetic arrays**, not camera or tactile data.
  No live sensor is connected. This is the "engine with no transmission"
  gap, and it is not closed here.
- No learned codebook was tested. The margin law remains unverified on
  learned engrams.
- No token vocabulary is defined. `encode_token` takes integer ids; no
  text has been tokenized.
- No benchmark task was attempted or solved.

## 9. Learned-codebook test of the margin law (supersedes the Phase 1 scope limit)

**The open question is answered.** The Phase 1 law `margin ~= t (1 - c_max)`
was tested on LEARNED codebooks rather than synthetic draws.

Receipt `receipts/learned_codebook_law_full_D.json`, sha256
`62624b9d6eb393964fbfc16196849a450f3fada2ac264393b58eb43218e47c48`,
sidecar readback verified, D = 65536.

| Family | learner | unit norm | max abs residual |
|---|---|---|---|
| learned_wellfit | k=32, rank=64 | yes | 0.002334 |
| learned_stressed | k=128, rank=32 | yes | 0.002317 |
| learned_clustered | k=32, rank=64 | yes | 0.002025 |
| control_nonunit | k=128, rank=32 | **no** | **2130.2** |

Verdict `LAW_CONFIRMED_ON_LEARNED`: R² 0.9999840, max absolute residual
**0.002334** against a 0.05 bar, 12 of 12 unit-norm cells applicable.
Rows come from `henri.memory.DeltaMemory`, the repository's own production
learner. Target alignment 1.0000 for well-fit and clustered families, so
the recalled rows are the stored values, not copies of the query.

**The unit-norm premise is load bearing, not incidental.** The
non-unit-norm control exceeded the bar by a factor of roughly 10⁶. The
derivation assumes `|M_k| = 1`; that assumption does real work.

**Scope limit that remains.** The learning targets are GENERATED pairs.
These are learned engrams over generated data, not a real-world corpus.
No task was attempted or solved.

**Measurement-channel correction (same day, additive).** The per-row
label in the probe's stderr originally read `OK` whenever
`applicable` was true, and `applicable` means only *the argmax hit the
intended row*. It says nothing about the residual bar. A reader of the
transcript alone therefore saw `control_nonunit resid=+2129.081451 OK`
and could mistake the control for a pass. The aggregate verdict was
never wrong (`control_exceeded_bar: true`), but a per-row label that
cannot fail is a defect in its own right.

The label now reports the BAR in three states — `IN-BAR`,
`OUT-OF-BAR`, `CONTROL-UNGRADED` — and the receipt stores
`abs_residual`, `within_bar`, and `bar_applies_to_family` as fields
rather than deriving the verdict at print time. Verified at `D = 256`
only. The sealed production-D receipt was deliberately NOT re-run, so
its digest stays valid and unchanged:
`62624b9d6eb393964fbfc16196849a450f3fada2ac264393b58eb43218e47c48`.

**Note on `learned_stressed` alignment.** Its `align_mean` is 0.2462,
not a defect: `rank 32 < k 128` makes exact storage impossible. The law
still holds there at residual 0.002317 — the smallest of the three
families. The law survives poor recall. That is a stronger result than
the exactly-stored wellfit case and belongs in the record.

### 9.1 Two defects found in `henri/memory.py`, both measured before repair

**D1 — `write` was a total no-op that reported success.**
The update scaled by `1 / ||K^H k||^2`. On a fresh memory `K` is zero, so
`coeff = K^H k = 0`, so `outer(x, 0) * gain` is zero for EVERY finite
gain. Both factors stayed exactly zero forever while `updates`
incremented. Measured on the pre-fix code:

```
write returned  : 0.0
updates counter : 1
||V||_F         : 0.000000000000e+00
||K||_F         : 0.000000000000e+00
predict(k1)     : 0.000000000000e+00
```

Repair keeps the key columns orthonormal and inserts a new column when
the key has a component outside the stored span. Verified after repair:
`write` returns 1.0, `||V||_F = 1.0`, `||K||_F = 1.0`, residual 0.0.

**D2 — `read(iterations=3)` destroyed exact recall.**
`S = V K^H` is not idempotent, so the extra passes moved a correct answer
away. Measured at D=1024, k=32, rank=64:

| iterations | max error |
|---|---|
| 1 | **9.937e-16** |
| 2 | 1.057e+00 |
| 3 | 1.011e+00 |
| 5 | 1.000e+00 |

Default changed to `iterations=1`. Higher values remain available for
explicit use. Direct measurement of the non-idempotence:
`||V K^H v0 - v0|| = 1.030` for a stored value `v0`.

**Why the existing tests missed both.** They asserted `after <= before`,
`delta < 1e-6`, and `updates >= start`. A total no-op satisfies all
three. Those three assertions could not fail on broken code, so they
carried no information. Four discriminating tests now fail on the
pre-fix implementation. No bar was moved.

## 10. Real sensor ingress (closes the SENSOR half of the transmission gap)

`henri/sensor_ingress.py` accepts REAL bytes and drives
`ContinuousSpatialEncoder` → `SubspaceRouter` → `WaveState`. It reads
Windows wallpaper JPEGs and binary PNM/PGM through a stdlib reader, and
uses PIL when a suffixed format needs it.

**It fails closed.** An undecodable source raises; it never substitutes
synthetic bytes. A silent substitution would reintroduce exactly the
synthetic-input problem this module removes.

**The real-data claim is tested, not asserted.** A test separates a real
photograph from white noise by spatial autocorrelation: the mean absolute
adjacent-pixel difference of the photograph is under 0.6× that of
uniform noise. Gaussian draws do not have that property.

**The ACTUATOR half stays open.** There is no action egress, no
environment, and no control task in this module. `henri/sensor_ingress.py`
is ingress only. No live camera is connected; `cv2` is absent on this
host and no camera path is implemented.


