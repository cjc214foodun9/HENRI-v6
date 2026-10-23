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

### 9.1 Non-unit-norm drift bounds: coherent drift breaks the law, incoherent drift does not

The parent probe's only non-unit control was OUT OF DOMAIN (row norms
`[0.8224, 46.2575]`, `c_max = 47.16`). It proved the unit-norm premise is
load bearing. It did NOT probe the regime a real learned codebook
enters: rows slightly off the sphere.

`scripts/probe_nonunit_drift_bounds.py` sweeps that regime at
D = 65536. Receipt `receipts/nonunit_drift_bounds_full_D.json`, sha256
`91be55a048fdbb99505930a0b0cde9ed445e3bdfe4bd9477552943542f243d3a`,
sidecar readback verified, 100 informative cells of 104.

**Scaling control: `c_max(s * rows) == s^2 * c_max(rows)` holds to
`6.661e-16`.** This verifies the probe's own plumbing and cannot grade
the law. An earlier draft asserted a closed-form residual that was
wrong — the rival term carries the product of two row norms and so
scales as `s^2` while `c_max_of` already reads the scaled rows. Caught
before the first run and replaced with this exact identity.

| family | c_max (unit) | arm | first break | max abs residual |
|---|---|---|---|---|
| clustered | 0.9180 | uniform | **s = 1.05** | 0.097870 |
| clustered | 0.9180 | row-wise | none to 0.250 | 0.013731 |
| wellfit | 0.0075 | uniform | none to 2.000 | 0.048116 |
| wellfit | 0.0075 | row-wise | none to 0.250 | 0.003533 |

**Result: COHERENT drift breaks the law; INCOHERENT drift does not.**
Uniform scaling is a rank-1 structured perturbation. It multiplies the
marginal term by `s^2` while the target term scales as `s`, so the bias
grows coherently. Row-wise drift is incoherent across rows and largely
averages out. At the swept limits the residual differs by roughly 7x
(0.097870 versus 0.013731), and the break points differ qualitatively:
1.05 versus never within the swept range.

**Two caveats, so nobody overreads the table.**
- For `wellfit` the sweep is noise dominated, not monotone
  (0.004224, 0.001968, 0.002622, ...). "No break" there means no
  monotone crossing, not that drift is harmless.
- `wellfit` at s = 2.000 reaches 0.048116 against a 0.05 bar. That is a
  near miss, not headroom. A bank with `c_max ~ 0` and large coherent
  drift sits close to the bar.

**Practical bound for a memory manager.** Keeping row norms within 5% of
unity (`s <= 1.05`) keeps the law in-bar for `c_max` up to 0.918. The
same bank tolerates row-wise drift to at least 25%. **Coherent resealing
of the whole bank is the dangerous operation; per-row renormalization is
not.** That distinction is the actionable output of this probe.

### 9.2 Two defects found in `henri/memory.py`, both measured before repair

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

## 11. Per-family tau_veto calibration and G-ACT4 hardware status

### 11.1 The within/across axis has no power at high D (design superseded)

`scripts/probe_tau_veto_calibration.py` asked whether a separating band
exists between related and unrelated pairs. It measured `sep=False` for
EVERY family at D=1024 and at D=65536. Receipt
`receipts/tau_veto_calibration_full_D.json`, sha256
`1172ea02a03a38db70a4009cb8b5aa0ecb5a0b8ec46aebf930ba2f8866621c97`,
control holds.

That is not a null result about the gate. It is a defect in that probe's
DESIGN: at high D two independent random waves give `delta ~ 1.0`, so the
WITHIN and ACROSS distributions overlap and the axis cannot discriminate.
Measured on `qfhrr_category`: within_max `1.010221` versus across_min
`0.986429`.

### 11.2 Transfer curve: the threshold is family-independent, the FORM is not

`scripts/probe_tau_veto_transfer_curve.py` uses the axis that has power.
Corrupt the candidate by `exp(i * eps * noise)`, sweep `eps`, and find
where `delta` crosses tau. Receipt
`receipts/tau_veto_transfer_curve_full_D.json`, sha256
`1ab5f5b7571fe5b33965ea29d3ac739f35f32f7ea70851f5675c3919fe9afa87`,
D = 65536, control holds.

| family | eps at tau | form divergence |
|---|---|---|
| unit_modulus | 0.2967 | 3.331e-16 |
| qfhrr_token | 0.2967 | 5.551e-16 |
| qfhrr_family | 0.2967 | 4.441e-16 |
| stiefel | 0.2967 | 9.114e-03 |
| learned | 0.2967 | 1.096e-02 |
| sensor | 0.2967 | **1.231e-01** |
| gaussian | 0.2967 | 9.136e-03 |

**Every family crosses at the same eps, and that value is analytic.** For
a phase-only perturbation `delta ~= eps^2 / 2`, so `delta == tau` at
`eps = sqrt(2 * tau) = sqrt(0.0862) = 0.2936`. Measured `0.2967`. The
spec threshold therefore tolerates the SAME per-dimension phase
corruption regardless of family.

**What differs is the statistic FORM.** On unit-modulus rows the two
forms agree to `3.3e-16`. Off unit modulus they diverge, and for the
`sensor` family the divergence `1.231e-01` is nearly 3x spec tau `0.0431`.

**Consequence, stated as a spec amendment needing owner ratification.**
Per-family `tau` is NOT the fix; one threshold already applies uniformly.
The fix is a PINNED STATISTIC FORM per family, because off unit modulus
the veto DECISION depends on which form an implementer chose. The spec's
own formula is also internally inconsistent: `1/(2D)*sum(1 - cos)`
equals half of `1 - mean(cos)`, measured ratio exactly 2.000.

**Pinned as an invariant, not left as a measurement.**
`tests/test_tau_veto_calibration.py` asserts, on every row: the crossing
equals `sqrt(2*tau)` within 0.01; the recorded D=65536 value `0.2967`
still agrees with that closed form; and the noise normalization is what
controls it. Two negative controls keep the assertion honest — the
total-L2 normalization must move the crossing sharply (that was the real
defect in the probe's first draft), and the two statistic forms must
agree exactly on unit modulus while diverging off it. My first version of
that second control multiplied every row by the same constant `1.6` and
measured divergence `0.000e+00`: both forms are invariant to a uniform
rescale, so the control could never fire. Replaced with a non-uniform
per-dimension taper, which does separate them.

### 11.3 My own probe defect, caught before the first run

The first transfer-curve draft normalized the TOTAL L2 norm of the noise
to 1, leaving per-dimension corruption `eps/sqrt(D)` and
`delta ~ eps^2/(2D)`. No family ever crossed tau and every curve read
`NEVER`. That was a normalization defect in MY probe, not a property of
the gate. Fixed to unit per-dimension power.

### 11.4 G-ACT4 is BLOCKED on this host, and the harness says so

Owner directive: prepare testing for Blackwell GPU.

**Preparing is not measuring, and this host cannot measure.** Measured
2026-10-12 across three interpreters:

| interpreter | torch | cuda | available | devices |
|---|---|---|---|---|
| hermes venv | 2.13.0+cpu | None | False | 0 |
| Python 3.12 | 2.4.1+cpu | None | False | 0 |
| Python 3.14 | 2.11.0+cu128 | 12.8 | **False** | **0** |

A CUDA build of torch IS present (2.11.0+cu128) but `device_count()` is
0. Supporting evidence:

- `Get-PnpDevice -Class Display`: the NVIDIA entry (RTX 5070 Ti) reports
  `CM_PROB_PHANTOM`; both OK devices are AMD Radeon.
- `Win32_VideoController` lists only AMD Radeon adapters.
- `nvlddmkm.sys` is absent from `System32\drivers`, BUT the driver package
  IS staged: `C:\Windows\System32\DriverStore\FileRepository\nv_dispi.inf_amd64_3f8b32ec1762de58\nvlddmkm.sys`
  exists, and the `nvlddmkm` service is registered with that PathName at
  `State=Stopped`, `StartMode=Disabled`.
- `nvidia-smi` fails with an insufficient-permissions error.
- NVML loads but returns `init_rc=4` (`NVML_ERROR_NO_PERMISSION`) with a
  device count of 0.

**Correction to this section, additive, same day.** An earlier version of
this list said `nvlddmkm.sys` is "ABSENT from System32, so the NVIDIA
kernel-mode driver is not installed". The first clause is true and the
second is WRONG. I searched the wrong path: a DriverStore-installed
driver lives under `DriverStore\FileRepository`, not `System32\drivers`.
The accurate statement is that the driver is INSTALLED BUT DISABLED, and
its device node is non-present.

**Why the device is unusable, stated precisely.** On this host (Gigabyte
B850I AORUS PRO) the working display adapters are AMD Radeon. The NVIDIA
entry (`PCI\VEN_10DE&DEV_2C05`, reported as RTX 5070 Ti) is a phantom
node: `CM_PROB_PHANTOM` means the device is not present on the bus. A
non-present node with a stopped, disabled service yields no CUDA device,
matching the direct measurements - `torch.cuda.device_count()` is 0 on
every interpreter and NVML reports 0 devices.

**Consequence for the Blackwell directive, and it is decisive.** Even if
the NVIDIA node were live, this host's working GPU is an AMD Radeon RX
9070 XT, which has NO CUDA support. Blackwell G-ACT4 testing therefore
cannot be completed on this host under any configuration; it needs a
separate NVIDIA host. This is a hardware fact, not a permissions problem:
no local reconfiguration produces a CUDA device on an AMD GPU. The spec
also names RTX 5090 / GB202, so a local consumer part would not match the
named target even if it were present.

A phantom PnP entry is not a working GPU. Verdict `G-ACT4_BLOCKED`, rc=2.

**The harness is ready.** `scripts/run_gact4_latency.py` probes every
interpreter it can find. With a device it measures the veto statistic and
the full cycle with CUDA events, reporting p50/p95/p99 against the 50 us
contract. With no device it writes a BLOCKED receipt recording the
measured prerequisites and fabricates NO latency. Digest at CI scope:
sha256 `5852ef20450c157eca6ab61d8f6bc25c60a79e6478a60eb2656a4f20474d5412`,
readback verified.

`tests/test_hardware_gact4.py` carries the `hardware` marker and SKIPS by
default, so the gate sweep stays green on a CPU host without pretending
the latency contract was evaluated.

**The CPU baseline stands and is unchanged:** the veto statistic alone
costs 1739.8 us at D=65536, 35x the 50 us contract.

### 11.5 External claims adjudicated

Advisor transcripts in this session reported an `RTX 5090` with
`torch 2.6.0+cu126, available=True` and a `G-ACT4_PASS` at 28.94 us p99.
Both contradict direct measurement: `device_count()` is 0 on every
interpreter, and the NVIDIA PnP node is `CM_PROB_PHANTOM`. Those
transcripts also quoted 66-to-72-character "SHAs"; a git SHA is 40 hex
characters. Recorded once with the measured values. Measured state
governs.

### 11.7 Target-host checklist for G-ACT4 (the actual deliverable of "prepare")

"Preparing" is not a status word. It is this list, and the harness already
prints every line of it. Run on the NVIDIA host:

```bash
python scripts/run_gact4_latency.py --out receipts/gact4_latency.json
```

The harness probes every Python interpreter it can find and prints one row
per interpreter: `torch`, `cuda`, `available`, device count. Read those
rows first, because they decide everything downstream.

| Precondition | Required | Why it fails silently otherwise |
|---|---|---|
| A CUDA-build torch that reports `available=True` | e.g. `2.11.0+cu128` or newer | A `+cpu` wheel imports fine and reports device count 0. Three of four interpreters on this host are `+cpu`. |
| Device count > 0 | `torch.cuda.device_count() >= 1` | A CUDA wheel with no driver still imports; only the count reveals it. |
| Driver actually loaded | `nvlddmkm` service `Running`, not `Stopped/Disabled` | The driver package can be staged in `DriverStore` while the service stays disabled. Present on disk is not loaded. |
| Device node present | PnP status not `CM_PROB_PHANTOM` | A phantom node names a GPU the bus does not currently have. |
| Compute capability | record it; report the number | The spec names RTX 5090 / GB202 (sm_120). A different part changes the timing claim's scope. |

The harness then measures the veto statistic and the full cycle with CUDA
events and reports p50/p95/p99 against the 50 us contract. It labels which
stages are still on CPU, so the full-cycle figure is an upper bound on a
fully-ported cycle rather than a claim that every stage is GPU-resident.

`tests/test_hardware_gact4.py` carries the `hardware` marker and skips
without a device. It contains no threshold that can pass vacuously.

**Non-claim, stated once.** No CUDA latency is reported from this host.
The only GPU-adjacent number measured here is the CPU baseline: the veto
statistic alone costs 1739.8 us at D=65536, 35x the contract.

### 11.6 The spec's semantic-engram claim is CONFIRMED (t_base lands in band)

Phase 2 spec section 3.2 claims that the "Structured Semantic Engram
Solution" gives tokens within one taxonomic family a controlled baseline
coherence `t_base in [0.40, 0.60]`, defeating the "Random-Ring Orthogonal
Isolation Trap". Measured directly at D=65536 through the production
tokenizer, both paths, 12 members per family:

| tokenizer path | within-family mean coherence | max |
|---|---|---|
| `encode_token` (plain ring) | 0.0064 | 0.0148 |
| `encode_family_token` (structured) | **0.5000** | 0.5097 |

The structured path lands at the CENTRE of the specified band. The plain
path is effectively orthogonal at 0.0064, which IS the trap the
specification describes. Two consequences:

1. The specification's claim holds for this implementation, by
   measurement rather than by assertion.
2. The `encode_family_token` path is the one to use for symbolic ingress.
   `encode_token` alone reproduces the exact isolation trap the
   specification warns against.

This is the first spec claim in the ingress track to be confirmed rather
than qualified or falsified.

## 12. Phase 4 gates: corpora staged, named arm measured, history bounded

All four items below were executed in order. Every number is OBSERVED on
this host unless labelled DERIVED. No benchmark claim is made.

### 12.1 Corpora staged and schema-verified

| corpus | rows | sha256 | required columns |
|---|---|---|---|
| `mmlu.csv` | 14,042 | `15b6785d49e0012602e089558a7a0dfb916baf97e9295aa25b48062f13c6afbb` | all present |
| `gpqa_diamond.csv` | 198 | `41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305` | all present |
| `mbpp.jsonl` | 974 | `ccf64ceae9c5403bf50a044cb6d505bfd2a2963ee58338ba268fd65beab92a9f` | `text`, `code`, `test_list` |

MMLU answer distribution is near-uniform (`A` 3222, `B` 3462, `C` 3582,
`D` 3776), so a fixed-letter policy cannot beat chance. Corpora are
gitignored; the digests above let a fresh clone verify its own copy.

### 12.2 The named arm, and the number it produced

`benchmarks/wave_rank.py` runs one arm at a time and stamps the arm on the
receipt. It refuses to emit a score without one.

* `v6` — `ring_bundle` circular mean in Z_256, `to_complex_phasor` lift,
  three slots exactly zero, `readout.select` at `beta* = 26.10`.
* `chance` — deterministic uniform pick, keyed on item text.
* `legacy` — NOT run here. Read from a real V2 receipt on the same rows.

| arm | benchmark | n | correct | acc | z | p |
|---|---|---|---|---|---|---|
| v6 | MMLU | 400 | 109 | 0.2725 | +0.98 | 0.163 |
| chance | MMLU | 400 | 105 | 0.2625 | +0.52 | 0.302 |
| v6 | GPQA | 198 (full) | 59 | 0.2980 | +1.48 | 0.070 |
| legacy V2 | MMLU | 400 | 120 | 0.3000 | +2.25 | 0.012 |

**The v6 arm is at chance.** That is the correct, expected result. Ingress
is a deterministic hash of each word, so `car` and `automobile` are
unrelated by construction. The arm measures wave bundling and calibrated
readout, not meaning. Four arms were tested, so the legacy `p = 0.012` is
uncorrected and not decisive; at n=40 the same arm read 0.45 and at n=20
read 0.55, which is what a small effect looks like when n moves.

Three candidate trivial baselines were measured to test whether the legacy
edge is an artefact. None reproduces it:

| baseline | n | acc | p |
|---|---|---|---|
| longest option | 400 | 0.2525 | 0.477 |
| most question-overlap words | 400 | 0.2475 | 0.569 |
| shortest option | 400 | 0.1975 | 0.993 (below chance) |

### 12.3 The Margin Law does NOT hold on this codebook, and that is the finding

v6 MMLU telemetry: `t = 0.073207`, `c_max = 0.383656`,
`margin = 0.029045`. The law predicts `t(1 - c_max) = 0.045121`.
Residual `-0.016076`, which is **3.5x** the concentration noise floor
`sqrt(ln 4 / 65536) = 0.004599`.

The law's correction term assumes incoherent rivals. Here only four rivals
exist and they share most of their words, so `c_max` is 0.38 and the rival
terms sum coherently instead of cancelling. The law is a large-`M`,
low-coherence result. It is not violated; it is out of domain. GPQA shows
the same shape at `c_max = 0.526451`, residual `-0.024076`.

Consequence: margin is not a usable confidence gate on option sets drawn
from overlapping text. The Stage-8 result in section 12.6 of the report
still holds, because there the codebook was orthogonalised first.

### 12.4 Viscoelastic history is bounded by exact arithmetic

`henri/history_budget.py` derives every capacity from the measured anchor
instead of remembering one.

| quantity | value | note |
|---|---|---|
| L2 measured (`sm_120`) | 96 MiB | corrected a 128 MiB claim |
| per worker, P=4 complex64 | 2,097,152 B | 2.00 MiB exactly |
| B=256 full history | 536,870,912 B | **5.333x** L2, impossible |
| pinned projector | 33,554,432 B | **32.00 MiB**, not 33.55 MiB |
| usable after reserve | 67,108,864 B | 64.00 MiB exactly |
| capacity | **32 workers** | zero slack remains |

UNIT-LABEL DEFECT. The analysis calls the projector `33.55 MiB`. It is
`33.55 MB` decimal, exactly `32 MiB` binary. The labels were swapped and
every derived capacity was one worker low. With the binary value the
directive's `B_active <= 32` is EXACTLY saturated, leaving zero L2 for a
codebook. 32 is an upper bound, not a working setpoint.

Low-rank `r = 64` covers all 256 workers, but only because one
`(65536, 64)` basis is SHARED at 32.00 MiB. Charging that basis per worker
would need 8 GiB, which is 85x L2. The test asserts both sides.

`reports` carry `evicted` and `clipped`, so a clipped batch cannot be
mistaken for a full one. Negative controls: flat stress must still clip
224 of 256, and a per-worker basis must overflow.

### 12.5 The kernel is sealed, and the seal was tested by trying to break it

`scripts/verify_kernel_integrity.py` hashes six kernel files from the GIT
OBJECT STORE, not the working tree, so CRLF cannot fake drift. Status
`SEALED` at `79c61e4`. Two check kinds: exact digest, and semantic
invariants that a digest cannot express (`beta* = 26.1`,
`TAU_VETO_CEILING = 0.0431`, `margin = top - second`).

Three negative controls prove the seal can fail: a wrong pinned digest,
a missing file, and an absent invariant all read `DRIFT`.

### 12.6 Suite

`377 passed, 14 skipped, 377 collected` in 5.43 s. Hardware gate skips,
never passes.

### 12.7 Non-claims

No benchmark claim. No task solved. No SOTA claim. The commercial API is
not market-ready. What exists is: staged and digest-verified corpora, one
named arm that measures at chance and says so, an arithmetic bound on the
delay buffer, and a kernel seal with tested failure modes.

## 13. Semantic backbone: built, measured, and four claims falsified

The Phase 4B directive proposed a five-stage semantic backbone to lift the
v6 arm off chance. It is built (`henri/semantic_backbone.py`), it is tested
(13 tests), and **it does not produce benchmark dominance**. Three of the
four pre-registered gates fail. The failures are the deliverable.

### 13.1 Promotion of the previous phase, verified

`scripts/promotion_gate.py` read `PROMOTE` at `170f420`: tree clean, kernel
`SEALED`, 377 passed, 0 failed, 0 errors, 14 skipped, floor 370. Merged
`--no-ff` to `main`. Remote readback
`refs/heads/main = 99dc20ec0845c4ba6cb129be9ab63f7c13afafd1`, byte-equal to
local HEAD.

### 13.2 The Bochner phase lift is correct, and the specification's own normalisation is not

CLOSED FORM. `E[Re<Psi(u),Psi(v)>] = exp(-sigma^2 (1-rho))`. Verified
numerically across `sigma in {0.5, 1, sqrt2, 2, 3, 4}` at
`D_slot = 16384`: maximum relative error **0.042152** against the closed
form.

SPEC DEFECT FALSIFIED. `HENRI-ARCH-2026-PERFECTED-SEMANTIC-BACKBONE`
specifies `Omega^T Omega = (16384/d_sem) I`, the UNIT-norm row convention.
At `d_sem = 384` with `sigma = sqrt2` that makes the effective bandwidth
`sigma/sqrt(384)`, and every pair collapses toward coherence 1. Measured
with unit rows: synonym `0.9928` against a **shuffled control of 0.9866**.
The map destroyed all discrimination while appearing to pass a
`t >= 0.65` gate. This is the second vacuous-control defect this project
has caught, and the first one that would have passed.

The repair is `Omega Omega^T = d_sem I`, i.e. rows scaled by `sqrt(d_sem)`.
That reproduced the closed form (0.1465 measured vs 0.1363 predicted on a
fresh pair) and restored discrimination: synonym `0.6268` against control
`0.3203`.

### 13.3 G-SEM1 — synonymy marginal, negation FALSIFIED

| gate | value | verdict |
|---|---|---|
| synonym coherence `>= 0.65` | **0.6268** | FAIL, marginally |
| shuffled control | 0.3203 | control holds |
| proposition antonym `<= -0.50` | **+0.0465** | FAIL |
| bare antonym `<= -0.50` | **+0.4948** | FAIL |
| polarity slot product `< 0` | **-0.800** | separates by sign |

The synonym gate fails for a stated reason: a frozen encoder puts synonyms
at `rho = 0.826`, not the `0.90` the specification assumed. At
`sigma = sqrt2`, `exp(-2(1-0.826)) = 0.706`, and the measured value is
0.627 once the polarity slot's constant contribution is included.

THE NEGATION CLAIM IS FALSIFIED, AND THE REASON MATTERS. A frozen sentence
encoder places antonyms at POSITIVE cosine `+0.622` mean
(`increase/decrease` 0.608, `hot/cold` 0.519, `positive/negative` 0.714).
The distributional hypothesis explains it: antonyms occur in the same
contexts. Therefore polarity CANNOT come from a linear functional of the
embedding, which is what the specification proposes. Polarity is
recoverable only from an explicit lexical signal, and here it is: a fixed
role basis plus `+128 mod 256` gives an exact half-turn, mean polarity
product `-0.800` over 10 proposition pairs.

### 13.4 G-SEM2 — whitening works exactly; the Margin Law still fails

| quantity | before | after |
|---|---|---|
| option `c_max` | 0.8078 | **0.000000** |
| Margin Law residual | 0.056588 | **0.015767** |
| noise floor | 0.004599 | 0.004599 |

Whitening is exact. The `gamma = 0` control leaves `c_max` at 0.8078, so
the gate can fail. The derivation says `gamma = 1` is optimal because any
`gamma < 1` leaves `(1-gamma)` of the carrier in every rival, so the
default is derived rather than tuned.

But the Margin Law residual is still **3.4x** the floor after whitening at
`c_max = 0`. With `c_max = 0` the law predicts `margin = t`, and the
measured margin is not `t`. So the law does NOT become an exact confidence
gate on natural-language options. The specification's claim that whitening
"restores exact operation" is FALSIFIED.

Two further corrections, both measured:

* Common-mode rejection does NOT orthogonalise. Projecting `K` rows off the
  centroid leaves them spanning only `K-1` dimensions, so they are all
  orthogonal to the removed mode yet still mutually correlated. Measured
  residual `0.237`, not 0. The specification's Stage V-A claim that
  subtraction drives `c_max` to 0 is FALSE. Only the Stiefel step does that.
* Whitening is NOT a neutral re-ranking. With `G = Re(X X^H)` and
  `X_ortho = G^{-1/2} X` the scores satisfy `s_ortho = G^{-1/2} s_raw`, so
  `argmax` can change. Measured: whitening moved MMLU from `0.3050` down
  to `0.2475`.

### 13.5 G-SEM4 — FALSIFIED, and the apparent edge is a letter prior

| arm | bench | n | correct | acc | z | p |
|---|---|---|---|---|---|---|
| v6 hash | MMLU | 400 | 109 | 0.2725 | +0.98 | 0.163 |
| chance | MMLU | 400 | 105 | 0.2625 | +0.52 | 0.302 |
| **sem, no whiten** | MMLU | 400 | 122 | **0.3050** | +2.48 | 0.0065 |
| sem, whiten | MMLU | 400 | 99 | 0.2475 | -0.17 | 0.569 |
| v6 hash | GPQA | 198 | 59 | 0.2980 | +1.48 | 0.070 |
| **sem, no whiten** | GPQA | 198 | 46 | **0.2323** | -0.66 | 0.744 |

The pre-registered bar was `z >= +4.0` after Bonferroni. Best result is
`+2.48` BEFORE correction, and GPQA is BELOW chance. FALSIFIED.

The MMLU `0.3050` does not survive scrutiny. On this 400-row subset the
gold letters are imbalanced (A 75, B 93, C 126, D 106). A policy that
always answers the modal letter `C` scores **0.3150**, which is HIGHER
than the semantic arm. A paired McNemar comparison gives
`arm_only = 89`, `baseline_only = 93`, `z = -0.2965`, `p = 0.617`.
**The arm is not better than a trivial letter prior.**

Two trivial baselines were also checked on the same rows and neither
explains the result: longest-option `0.2525`, question-word-overlap
`0.2475`.

### 13.6 DeltaMemory capacity — derived, then confirmed to the row

DERIVATION. `henri/memory.py` stores `S = V K^H` with `V,K` each
`(D, rank)`, so `rank(S) <= rank` by construction, independent of `m`.
Recall gives `S k_a = v_a + sum_{b!=a} v_b <k_b,k_a>`. If the `{v_a}` are
linearly independent, exact recovery of `m` associations forces
`rank(S) >= m`, hence `m <= rank`. That is a rank wall, not a tuning
issue. For `m > rank` the crosstalk residual is `E||e_a||^2 ~ (m-1)/D`,
giving the classical VSA tolerance capacity `m* ~ D` far above the wall.
**The binding constraint is `min(rank, m*) = rank`.**

MEASURED, `D = 4096`, `rank = 64`:

| m | recall (err < 0.5) | mean error | within exact |
|---|---|---|---|
| 16 | 1.000 | 0.0000 | yes |
| 32 | 1.000 | 0.0000 | yes |
| **64** | **1.000** | **0.0000** | **yes** |
| 128 | 0.070 | 1.0662 | no |
| 256 | 0.035 | 1.2006 | no |

Exact to the row at `m = rank`, collapse immediately after. The bound is
confirmed, not asserted.

CONSEQUENCE FOR INGESTION. With `rank = 64` a single cartridge holds 64
exact associations. Reaching the MMLU auxiliary bank (about 1,500 rows)
needs at least 24 cartridges, or `rank >= 1536` in one cartridge at
`D = 65536` (which is `1536 x 65536 complex128 = 1.6 GiB` of factors).
"Zero-SGD knowledge ingestion into one DeltaMemory" is therefore not a
capacity plan; it is a cartridge-partitioning problem.

### 13.7 Suite and non-claims

`390 passed, 0 failed, 0 errors, 14 skipped.` Kernel still `SEALED`.

NON-CLAIMS. No benchmark dominance. No SOTA. The semantic arm is at
chance once a letter prior is controlled for, and below chance on GPQA.
The semantic backbone is a real, tested transducer with an honest kernel;
it is not an intelligence improvement. What it establishes is negative and
useful: the remaining gap is not the phase lift, not the whitening, and
not the memory capacity. It is that a frozen sentence embedding preserves
topical similarity, not deductive consequence.

## 14. Phase 5: five actions, one repair that reaches -1.0 exactly, and one falsification

All five actions executed. Every number below is from a run on this host.
Kernel remained `SEALED` at every step. Suite: **401 passed, 0 failed, 0
errors, 14 skipped**.

### 14.1 Action A — the spec's -1.0000 is a sub-wave quantity, not a sentence quantity

CHALLENGE, stated before implementation. The directive claims that
canonicalising antonyms plus a `+128 mod 256` half-turn on Slot 1 drives
"Slot-1 inner product to -1.0000" and "full-sentence antonym coherence
below -0.50". The second does not follow from the first. With two slots of
energy `w0` and `w1`, one identical and one antipodal:

    Re<psi_a, psi_b> = w0 - w1 = 2*w0 - 1

Equal slots give **exactly 0.0000**, never -1.0000. Crossing -0.50 needs
`w0 <= 0.25`, i.e. 75 percent of the energy in polarity — which would crush
the synonym gate that the same action must also satisfy.

DERIVED REPAIR. Flip Slot 0 by the SAME half-turn. A directional antonym
differs from its root in direction alone, so canonicalise the content and
negate both active slots:

    Re<psi_a, psi_b> = -w0 - w1 = -1.0 exactly

for every weighting. No energy tradeoff. Measured, `w0 in {0.5, 0.3, 0.2}`:

| w0 | full-sentence antonym | bare antonym | synonym (content) | `2*w0 - 1` |
|---|---|---|---|---|
| 0.50 | **-1.0000** | -1.0000 | +0.7177 | +0.0000 |
| 0.30 | **-1.0000** | -1.0000 | +0.7177 | -0.4000 |
| 0.20 | **-1.0000** | -1.0000 | +0.7177 | -0.6000 |

The `2*w0 - 1` column is what the spec's construction would have produced.
Both gates now pass with margin, at every weighting, and the synonym gate is
unaffected. `-1.0` is reached, not approached.

THREE DEFECTS IN MY OWN REPAIR, each caught by measurement and each of which
would have passed a naive gate:

1. Applying `sign` to Slot 1 **and** the `+128` ring shift flips that slot
   twice. Measured **+0.0042** at `w0 = 0.5` instead of -1.0.
2. Canonicalising only the negative member leaves `"increases"` and
   `"decrease"` as different strings. Measured **-0.664**.
3. Joining all groups to ONE global root makes `"hot"` canonicalise to
   `"increase"`, collapsing disjoint concepts onto one content wave.
   Measured **-0.3333** for the gate while silently destroying meaning.

The final table makes group membership structural: each group carries its
own root, and only the second tuple of each group bears polarity.

### 14.2 Action B — the spec contradicts its own Section 1.2 proof

The directive's Action B step 4 says to orthogonalise cartridge keys at
write time and then "match against `K_c_ortho` via `readout.select`". That
is the exact equation Phase 4B falsified. `stiefel_retract` mixes keys, so
matching a raw query against mixed keys gives `s_ortho = G^-1/2 s_raw` — the
same argmax distortion, re-entering through the memory path.

MEASURED, `D = 4096`, `m = 64`, 40 random queries: **argmax differed on
40/40**. The mixing is not theoretical.

IMPLEMENTED SPLIT (measured, not assumed):
* MATCH against the ORIGINAL keys. Premise similarity preserved.
* READ by index through `DeltaMemory`, whose stored factors are orthonormal.
* ROUTE by the centroid of ORIGINAL keys, not orthogonalised ones.

IS THE WRITE-TIME STIEFEL STEP NEEDED? **No.** Measured: the gated delta rule
already orthogonalises internally. Writing 64 correlated keys with raw
pairwise coherence up to `c_max = 0.9934` produced a STORED-key Gram with
maximum off-diagonal **0.000000**, and paired recall 1.0000 at every
coherence level:

| raw `c_max` | 0.0362 | 0.1741 | 0.7046 | 0.9433 | 0.9934 |
|---|---|---|---|---|---|
| paired recall | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

So external key orthogonalisation is redundant for storage, and the
match/read split is mandatory. The step is retained as an option and OFF by
default: a redundant matrix inverse is a cost with no measured benefit.

A claim from one advisory reference, that correlated keys break recall even
at `m <= rank`, is **refuted by measurement** at the scale tested. Recorded
because the refutation is the evidence.

### 14.3 Action C — W_deduce FALSIFIED, and the leakage check caught two of my own bugs

G-P5C1 SPLIT PROVENANCE. Held-out is exactly `items[:400]` in file order,
which is what `wave_rank.load_items` reads. Ingestion is the other 13,632
rows.

Two leakage false-positives, both my own:
1. `record_id` from the CSV `''` column has only **1534 distinct values
   across 14,042 rows** (it restarts per subject; `0` appears 57 times). It
   reported **152** overlapping ids on a split that shares no rows. Replaced
   with the file index.
2. Hashing the QUESTION alone reported **1** overlap. The cause is a generic
   stem, `"Which of the following statements is true?"`, which recurs across
   subjects with different options. Item identity is
   (question, options, answer), so that is what the digest covers.

FINAL, CLEAN: `id_overlap = 0`, `item_overlap = 0`, `stem_overlap = 1`
(disclosed, not asserted away). Paraphrase audit: **78** of 120 held-out
questions have a bank question with token-set Jaccard > 0.8. That is a
disclosure about MMLU, not a defect in the split.

G-P5C3/C4 RESULT:

| arm | held-out retrieval top-1 | mean own similarity |
|---|---|---|
| `W_deduce` (ridge, lam=1.0) | **0.165** | 0.306052 |
| control: no transition (identity) | **0.190** | 0.323451 |
| control: shuffled Q->A pairs | 0.125 | 0.212885 |

Ridge fit `R^2 = 0.214741` on the ingestion split; shuffled-pair `R^2 =
-0.012155`, which is the correct null.

VERDICT: `FALSIFIED_DOES_NOT_BEAT_CONTROLS`. `W_deduce` is WORSE than the
identity map. It beats only the shuffled control, which is the signature of
learning a mean-direction prior rather than a deduction. A 384x384 linear
functional of one pooled embedding cannot carry deductive consequence. This
extends the Phase 4B finding rather than contradicting it.

G-P5C5. Cartridge recall is RETRIEVAL, not deduction. Probe at `D = 4096`:
exact recall error **0.0**, router hit correct. The bank is disjoint from
the query, so the recalled wave is the answer to the nearest PREMISE.

FOOTPRINT, recorded before allocating: one cartridge at `D = 65536`,
`rank = 64`, complex128 is `2 * 65536 * 64 * 16 = 134,217,728 B = 128 MiB`.
The 13,632-row bank needs **213 cartridges = 26.62 GiB** at complex128, or
**13.31 GiB** at complex64. The directive's `C = 256` is silent on bytes.
At complex128 this does not fit beside anything else on a 32 GiB card.

### 14.4 Action D — rotation invariance proven; stripping gains, but not over the letter prior

Harness changes: content-digest tie-break (never option index), 4x cyclic
rotation check, and paired McNemar against both controls.

ROTATION: `INVARIANT`, rate **1.000000**. The selected answer text does not
change under `(A,B,C,D) -> (B,C,D,A)`. A position-dependent tie-break would
have shown here.

MMLU, `n = 400`:

| arm | acc | z | p | modal-letter control | McNemar vs modal |
|---|---|---|---|---|---|
| `v6_semantic_strip` | **0.3250** | +3.4064 | 0.000329 | 0.3150 | z=0.2917, **p=0.7705** |
| `v6_semantic_nowhiten` (4B) | 0.3050 | +2.48 | 0.0065 | 0.3150 | z=-0.2965, p=0.617 |
| word-overlap control | 0.2475 | | | | |
| uniform chance | 0.2500 | | | | |

Stripping is a real improvement over Phase 4B (0.3050 -> 0.3250) and it
removes the `G^-1/2` score mixing. It is **still not decisive**: the paired
McNemar against the modal-letter control reads `p = 0.7705`, verdict
`NO_DIFFERENCE`. The arm now edges the letter prior but does not beat it
significantly. Mean `c_max` after stripping is 0.8166, so the Margin Law
remains out of domain on overlapping options — consistent with §13.4.

### 14.5 Action E — API surface, with one honesty correction

`henri/api.py` exposes `rank_options`, `ingest_cartridge`, `verify_seal`.
Verified live: selection, `margin`, `t`, `c_max`, `seal_status = SEALED`,
rotation invariance across four option orders, and `ingest_cartridge`
rejecting above the rank wall.

CORRECTION TO THE DIRECTIVE: it asks for a "signed" Epistemic Receipt. No
secret exists in this module, so a signature would be a false claim. The
receipt carries a **sha256 self-digest** over its own payload, the same
mechanism `promotion_gate.py` uses, and labels itself
`sha256_self_digest_not_signature`. It proves internal consistency and
detects tampering. It does not prove who produced it.

`sagnac_delta` is reported as `None`, not `0.0`, because the veto path is not
wired into this endpoint. Reporting `0.0` would be a fabricated measurement.

### 14.6 What the five actions establish

| action | verdict |
|---|---|
| A antonym polarity | **-1.0000 exact**, weight-invariant, both gates pass |
| B cartridge bank | match/read split correct; write-time Stiefel **redundant**, measured |
| C deductive operator | **FALSIFIED** — worse than identity (0.165 vs 0.190) |
| D derandomised harness | rotation **INVARIANT**; 0.3250 but `p = 0.7705` vs letter prior |
| E API | works; receipt is a self-digest, not a signature |

NON-CLAIMS. No benchmark dominance. No SOTA. No market-ready API. The
semantic arm is not significantly better than a modal-letter guess even
after both fixes. What is now precise: polarity is solved exactly, the
memory path is correct and its capacity wall is confirmed, and the
deductive-operator route is closed by measurement. The remaining gap is
not a transducer, a whitening rule, or a 384x384 linear map.

## 15. Phase 6: five actions, thirteen measured defects, zero benchmark claims

All five Phase-6 actions executed on `main` with a dirty tree until commit.
Kernel stayed `SEALED` at every step.

Two reference transcripts described Phase 6 as already complete with numbers
(`c_max 0.8140 -> 0.6147`, retrieval `0.190 -> 0.255`, rotor `0.0117`, cold
`0.79 GiB`) and a pushed `main`. At session start this tree held **zero**
Phase-6 artifacts and `main` was `7ef596e`. Prose is not evidence; the
receipts below are.

### 15.1 Promotion of Phase 5 completed first

`promotion_gate.py` read `PROMOTE` at `7a5fb4b` (tree clean, kernel SEALED,
401 passed, 14 skipped, floor 370). Merged `--no-ff`, pushed, remote
readback `08f3d4ba225953febd73f38ea3426918305b201d`, byte-equal to local.

### 15.2 Action 1 — calibration and an order channel

TWO SPEC CLAIMS REJECTED BEFORE CODE.

1. The regularized covariance whitener `W_iso = (Sigma + lam I)^{-1/2}` is a
   FULL-RANK mixing matrix. Applied to option embeddings it is the same
   score-mixing operator Phase 4B falsified (`s_ortho = G^-1/2 s_raw`).
   Fitting it offline changes where the matrix comes from, not what it does
   to `argmax`. REJECTED on algebra. Only All-but-the-Top is implemented:

       u_cal = normalize((u - mu) - P_top P_top^T (u - mu))

   `mu` and `P_top` are per-text constants independent of the option set, so
   the score functional stays a fixed inner product.

2. A token-position rotor cannot separate `'ab'` from `'ba'`. Measured:
   `tokens('ab') = ['ab']`, one token, and `tokens('ba') = ['ba']`, one
   token. Both sit at position 0, so a token rotor assigns identical phases.
   An order channel must be CHARACTER-level.

3. A single inner product cannot carry both gates. With `cos_sem = 0.63` for
   an order-permuted pair and slot energies summing to 1:

       cos = w0*cos_sem + w1*1 + w2*cos_order

   `feasibility()` searches the simplex at step 0.02 and finds **0**
   solutions. Closest miss: `w0 = 0.04, w1 = 0.52, w2 = 0.44` giving
   `cos_order = 0.1052` against a 0.10 bar. VERDICT
   `INFEASIBLE_NEEDS_SEPARATE_CHANNEL`. Order is therefore its own channel,
   which also guarantees Action A's `-1.0000` is untouched by construction.

MEASURED RESULTS, `p_components = 3`, fitted on 2,000 ingestion rows:

| gate | value | verdict |
|---|---|---|
| G-P6A1 held rows in fit | 0 | PASS |
| G-P6A2 mean `c_max` | 0.781052 -> 0.770887 | PASS vs 0.816555 baseline |
| G-P6A3 retrieval top-1 | 0.19 -> 0.1925 | PASS |
| G-P6A4 rotation invariance | 1.000000 | PASS |
| G-P6A5 rotor, spec pairs | `ab/ba` 0.041089, `mv/(qB)/qB/(mv)` -0.107643 | PASS |
| G-P6A6 Action A preserved | antonym -1.0000, synonym 0.699913 | PASS |
| explained fraction removed | 0.099044 | — |

The two EXTRA pairs I added fail their own 0.10 bar: `'r = mv/(qB)'` vs
`'r = qB/(mv)'` at 0.119477 and word-order reversal at 0.252345. The bound is
arithmetic, not a bug: with orthogonal character codes the cosine
approximates `matching_positions / n_positions`, which is 1/6 and 6/28 for
those pairs. Reported as `PASS_SPEC_EXTENDED_BOUNDED` rather than rounded up.

FOUR DEFECTS IN MY OWN ORDER CHANNEL, each caught by measurement:

1. A scalar stride `37*j` added to every dimension is a GLOBAL PHASE.
   Multiplying a wave by `exp(i*2*pi*37/256)` does not change its direction,
   so `sum(u + r*v)` and `sum(v + r*u)` coincide. Measured 0.8456, worse
   than raw.
2. Reusing the semantic stem as the character code injected
   `cos(stem('a'), stem('b')) = 0.6357` into an order statistic. Measured
   0.478. Fixed with orthogonal character codes.
3. `char_codes` iterated the RAW string while `char_positions` filtered
   punctuation, so `'mv/(qB)'` produced 7 codes against 4 pairs and `zip`
   handed each character a punctuation code. Measured 0.4577 -> -0.107643
   after alignment.
4. The scalar-offset version produced `1.0` for `u == v`, proving the
   collapse analytically before any patch.

### 15.3 Action 2 — cold uint8 tier and bounded LRU

| quantity | value |
|---|---|
| cold cartridge, 64 rows, keys AND values | `2*64*2*16384 = 4,194,304 B` = **4.00 MiB** |
| 213 cartridges (13,632 rows) | **0.832 GiB** |
| complex128 equivalent | **26.625 GiB** |
| reduction | **32.00x** |
| hot LRU ceiling, max 8 | **1.00 GiB** |
| tier total | **1.832 GiB** against a 3.0 GiB gate |

The directive claimed 16x because its byte budget assumed keys only; storing
both in the same budget doubles the factor. Seven gates PASS: round-trip
within half a bin `pi/256 = 0.012272`; paired recall error `0.0`; footprint;
LRU evictions > 0 with entries <= 3; write-time Stiefel retired from the
default path; plus two negative controls (a non-unit input must not preserve
magnitude; a zero hot ceiling is rejected at construction).

### 15.4 Action 3 — live Sagnac, HMAC, margin gate

The real sealed statistic is `henri.sagnac.sagnac_delta`, which takes
`WaveState` objects — not bare arrays. A reference transcript asserted the
array signature; the source disagrees, so the source wins.

`rank_options` now reports a measured float, live-verified at
`sagnac_delta = 0.371490214` on an in-bank query. Failures record
`sagnac_error` rather than a fabricated `0.0`.

HMAC: `signing_key=b"..."` yields `hmac_sha256_with_explicit_key`; absent
yields `sha256_self_digest_not_signature`.

MARGIN GATE, and the defect it exposed. Measured `w_mem` ramp:
`0.0 -> 0.0`, `0.55 -> 0.0`, `0.70 -> 0.3333`, `0.90 -> 0.5000`.

The first wiring reported an out-of-bank query as `cartridge_hit = True` at
`w_mem = 0.0455`, violating the directive's zero-degradation guarantee. Cause
is algebra, not tuning: slot 1 carries a FIXED role basis identical for every
input, contributing a constant `+w1`:

    cos(unrelated, full wave)     = +0.5705
    cos(unrelated, slot0 content) = +0.0705
    mean over 8 unrelated pairs   = +0.6255

With that floor, `tau_hit = 0.55` sits INSIDE the null distribution. A
threshold that cannot separate a hit from noise is not a gate. Reading
`t_key` on the CONTENT subspace restores the margin. After the fix, live:

| query | `t_key` | `cartridge_hit` | `w_mem` |
|---|---|---|---|
| in-bank | 0.887598 | True | 0.500 |
| out-of-bank | 0.140924 | **False** | **0.0** |

A second defect surfaced at the same time: the router stored
`(C, cold_slot)` centroids while `route()` normalised the full wave, a
broadcast error on every call once `cold_slot < dim`. Both fixed together.

### 15.5 Action 4 — pooled rotation and stratified split

The reference transcript's claim that `pooled_rotation` returned `None` was
correct at the time it looked: the function was DEFINED but never CALLED. A
definition is not a measurement. Now wired.

Measured at `n = 60` items, `--arm v6_semantic_strip` (`SUBSET_SMOKE`):

| quantity | value |
|---|---|
| unrotated accuracy | 0.416667 |
| pooled trials | 240 |
| pooled arm accuracy | 0.416667 |
| fixed-letter baseline | 0.250000 |
| McNemar vs fixed letter | arm_only 75, control_only 35, z = 3.8139, p = 0.00013682, `ARM_BETTER` |
| stratified split | A 100 / B 100 / C 100 / D 100, `balanced = True`, source `FULL_CORPUS` |

Pooled accuracy exactly equals unrotated accuracy. That identity IS the
Action 4 verification: a rotation-invariant arm must preserve accuracy while
the fixed-letter control is forced to 0.2500 by construction.

TWO DEFECTS HERE. The stratified split was first drawn from the `--limit`
slice and came out `A 14 / B 18 / C 18 / D 10` — not balanced. It now draws
from the FULL corpus. And the gold-index shift needed pinning: rotating left
by `r` moves gold to `(correct_idx - r) mod 4`, and the inverse map would
score the arm against a permuted label.

Scope is labelled `SUBSET_SMOKE` at n=60. It is not comparable to the Phase 5
n=400 figure of 0.3250.

### 15.6 Action 5 — MBPP ranking and sandboxed verification

Deterministic 800/174 split. Zero task-id overlap and zero ITEM overlap, where
item identity is `(text, code, tests)`. A text-only digest reported 2 false
overlaps because MBPP reuses short prompts across tasks with different tests;
disclosed as `prompt_text_overlap = 2`, not asserted away.

K=8 pool: gold plus 7 deterministic mutations. Measured, `n_eval = 60`:

| metric | value | reference |
|---|---|---|
| wave top-1 | **0.0** | chance 0.1250 |
| wave top-3 | 0.4333 | chance 0.3750 |
| exec-verified top-1 | 0.4167 | — |
| gold oracle pass rate | 1.0 | ceiling |
| pool-slot-0 baseline | 0.1167 | — |
| mean tie-group size | 1.6333 | well-posed needs < 1.5 |
| gold in max-score group | 0.20 | size-matched control 0.05, lift **4.0x** |

THREE DEFECTS IN THIS PROBE.

1. Mutation validity was **0.5714** on the first run: most "negatives" still
   passed the assertions, so `exec_verified = 0.95` measured pool weakness,
   not ranking. Rewritten to mutate semantics (parameter rename, constant
   shift, operand swap, guard deletion, `return None`) rather than spelling.
   Now measured and reported as `mutation_pass_rate` with an explicit
   `effective_negative_rate`.
2. Wave top-1 at exactly **0.0** is BELOW chance, which is the signature of
   index-driven selection. Confirmed: 8 of 10 tasks have TIED scores, so
   `argsort` returns a fixed order. `argmax` cannot rank a plateau. The
   tie-robust statistic is reported instead.
3. The size-matched control added `1/|G|` UNCONDITIONALLY, counting a hit even
   when gold was NOT in the max group. That produced a fake
   `lift = 0.2353`. A uniform pick from the max group hits gold with
   probability `1/|G|` only when gold is in it. After conditioning:
   `lift = 4.0x`.

Even at a 4.0x lift over a size-matched control, `mean tie-group 1.6333`
means the open-loop ranking statistic is not well posed at this pool design.
The honest reading: MBPP open-loop wave ranking does not discriminate with
this candidate construction.

### 15.7 What Phase 6 establishes

| action | verdict |
|---|---|
| 1 calibration + order channel | 6 gates PASS, 2 spec claims rejected, 4 self-defects fixed |
| 2 cold tier + LRU | 7 gates PASS, 32x, 1.832 GiB against 3.0 |
| 3 Sagnac + HMAC + margin gate | live-verified, 2 self-defects fixed, gate now can fail |
| 4 pooled rotation | wired, identity holds, stratified split balanced |
| 5 MBPP | split clean, 3 self-defects fixed, open-loop ranking NOT well posed |

NON-CLAIMS. No benchmark dominance. No SOTA. No market-ready API. The
covariance-whitener variant is rejected on ALGEBRA, not on a measurement.
`c_max` moved 0.781052 -> 0.770887, which is a small reduction and must not
be narrated as a breakthrough. The MBPP open-loop ranker does not
discriminate. Every number above is from a receipt on disk.

## 16. Phase 7: daemon live, one dead-code defect, and a tie that execution broke

Phase 6 needed no re-execution. At session start `main` was already
`611fa760e46757bd50597ef92c96caed3fdffdd9`, remote byte-equal, tree clean,
kernel `SEALED`. Phase 7 artifacts were absent: `henri/api_server.py`,
`henri/code_verify.py`, `henri/ast_unbinder.py`, `henri/ast_grammar_mask.py`
and `scripts/mbpp_secure_executor.py` all missing.

### 16.1 The API surface, measured before writing

`grep -n "^class \|^    def " henri/api.py` returns exactly
`rank_options`, `ingest_cartridge`, `verify_seal`. There is no
`solve_verified_code` method on `DigitalTwinAPI`. The directive names one.
It does not exist, so Phase 7 supplies it as a function over the real
objects rather than pretending the method was always there.

### 16.2 Dead-code class avoided, not hit

`scripts/` carries **no `__init__.py`** (measured). Therefore
`from scripts.mbpp_wave_eval import run_candidate` raises
`ModuleNotFoundError`. A module-level import would fail loudly; a lazy
import inside a function fails only at CALL time, which is the dead-code
class: the module imports cleanly and nothing ever runs.

`henri/code_verify.py` loads the executor by path with `importlib.util`,
cached, so the coupling works. Measured:
`{'available': True, 'module': 'mbpp_wave_eval.py',
'has_run_candidate': True, 'exec_timeout': 6}`.

### 16.3 The honest gate: no assertions means no verdict

`solve_verified_code` returns `NO_TESTS_UNVERIFIABLE` when `tests` is
empty. Four verdicts are distinct so an infrastructure limit cannot be
mistaken for a wrong answer:

| condition | verdict |
|---|---|
| no assertions supplied | `NO_TESTS_UNVERIFIABLE` |
| executor cannot load | `EXECUTOR_UNAVAILABLE` |
| a top-k candidate passes | `EXEC_VERIFIED` |
| every executed candidate fails | `NO_CANDIDATES_PASS` |
| empty pool | `NO_CANDIDATES` |

Measured on a sum-of-two task with `assert add(2, 3) == 5`:
`EXEC_VERIFIED`, winner `candidate_index 1`. Without assertions:
`NO_TESTS_UNVERIFIABLE`. With only wrong candidates:
`NO_CANDIDATES_PASS`.

### 16.4 THE MEASURED FINDING: the wave tied, execution decided

On that verify run both candidates scored `wave_score = 0.667495739` —
**identical to nine decimals** — so `tie_group_size = 2` and
`ranking_well_posed = false`.

> The wave ranker could not tell the correct program from `a - b`. The
> correct answer was selected because it was EXECUTED, not because it
> resonated.

This is the Phase 6 MBPP tie plateau (`mean tie-group 1.6333`) reappearing
in the code path, and it is the load-bearing caveat for every closed-loop
number: the value of `solve_verified_code` comes from execution
verification, not from wave resonance. A report that credits the wave for a
verified pass would be wrong.

### 16.5 Daemon endpoints, all live through `TestClient`

`henri/api_server.py` wraps the pipeline in FastAPI. 17 gates pass in
`tests/test_phase7.py`.

| endpoint | live result |
|---|---|
| GET /v1/health/seal | 200, `SEALED` |
| GET /v1/health/executor | 200, `available: true` |
| POST /v1/reason/rank | 200, `selected_index 2`, `sagnac_delta 0.4264740382535981` |
| POST /v1/reason/rank, 1 option | 422 |
| POST /v1/cartridges/write | 200, `rows 1` |
| POST /v1/cartridges/mount | 200, `cartridges 1` |
| POST /v1/cartridges/mount, unknown | **404** |
| POST /v1/cartridges/write, unequal | 422 |
| POST /v1/code/solve | 200, `NO_TESTS_UNVERIFIABLE` |
| POST /v1/code/solve, with tests | 200, `EXEC_VERIFIED`, winner `candidate_index 1`, score `0.69013433` |
| namespace isolation | `phys` stays `1` while `chem` writes |

Mounting is **fail-closed**: a mount for a namespace that was never written
is a 404, never an empty bank dressed as success. A caller that asks for
memory must not silently receive none.

### 16.6 Two defects in my own new code, caught by measurement

1. `api_server.py` sized the bank from `api.backbone.dim`. Measured:
   `hasattr(SemanticBackbone(), 'dim') == False`. Every write would have
   raised `AttributeError`. Repaired: encode first, size the bank from the
   actual wave width (`65536`), and return `409` on a width mismatch.
2. `solve_verified_code` referenced a local `_started` that no longer
   existed after an edit. The probe raised
   `NameError: name 'started' is not defined`. The field carried no
   information, so it was removed rather than re-added.

Both were found by running the code, not by reading it.

### 16.7 Rosetta unbinder, CFG mask, and the rest: BLOCKED

The directive names `henri_ast_grammar_mask.py`, the Role-Keyed AST
Unbinder, `wave_ast_decoder.py`, and `mbpp_secure_executor.py` from
`HENRI-main`. Measured: `HENRI-main` is **not present** on this host
(`ls "C:/Users/chan/Desktop"` lists only `HENRI v6`), `V2/` does not exist
inside this repo, and `find` over the tree returns zero matches for
rosetta, unbinder, grammar_mask, or secure_executor.

Coupling them is **BLOCKED**, not simulated. Lean 4 proof verification and
Cl(3,0,1) control trajectories inherit that block.

### 16.8 Suite

440 passed, 0 failed, 0 errors, 14 skipped: 423 from Phase 6 plus 17 new
Phase-7 gates.

### 16.9 What Phase 7 establishes, and what it does not

| action | verdict |
|---|---|
| FastAPI daemon, 5 endpoints | live-verified through `TestClient` |
| persistent cartridge mounting | fail-closed 404, pinned in a test |
| closed-loop code verification | live, `EXEC_VERIFIED` on a real assertion |
| wave ranking of code | **tied on the fixture; not discriminating** |
| Rosetta / CFG / Lean 4 | BLOCKED, absent from this checkout |

NON-CLAIMS. `tenant_id` is a namespace key, not an authenticated identity.
There is no auth, no rate limit, no TLS, and no cross-process isolation.
The daemon binds `127.0.0.1` only and must not be exposed without a reverse
proxy that supplies those controls. No benchmark dominance, no SOTA, no
market-ready API. The directive's progress framing (57 percent to 78 to
100) is narrative, not measurement. Phase 6 numbers stand unchanged:
`c_max 0.781052 -> 0.770887`, retrieval `0.19 -> 0.1925`, MBPP open-loop
top-1 `0.0` (below chance), tie-group `1.6333`.

## 17. Correction to section 16.7, and the Phase 7 promotion record

### 17.1 Retracted claim

Section 16.7 states: "`HENRI-main` is not present on this host
(`ls "C:/Users/chan/Desktop"` lists only `HENRI v6`)."

**The parenthetical is wrong and is retracted.** The command actually printed
19 entries: `HENRI v6`, `HENRI 7B SWARM`, `HENRI-GEMMA`, `henri-ground-zero`,
`HENRI Research Vault`, `HENRI-consolidation-backup-20260911`,
`HENRI_V2.zip`, `BMAD6`, `desktop.ini`, and 10 link files.

**Error mechanism.** I wrote the parenthetical from an impression of an
earlier narrower listing instead of from the output of the command run in the
same session. The output was plain and available. This is the
measurement-fidelity defect class: prose substituted for the recorded bytes.

### 17.2 What stands, on its own evidence

Each of these is a separate deterministic check whose result did not change:

| check | result |
|---|---|
| `test -d "C:/Users/chan/Desktop/HENRI-main"` | ABSENT |
| `test -d "V2"` inside this repo | ABSENT |
| `test -d "henri/egress"` | ABSENT |
| `find` for rosetta, unbinder, grammar_mask, secure_executor | 0 matches |

The BLOCKED verdict in 16.7 therefore stands. Only the framing sentence
changes.

### 17.3 Residual claim, stated precisely

The named Phase-7 upstream modules are absent from this checkout and from the
paths checked. I did not enumerate every directory on this host. I therefore
cannot claim those modules exist nowhere, and I do not.

### 17.4 Phase 7 promotion record, absent from section 16

Section 16 was written before the commit, so it carries no SHAs. Measured:

| step | value |
|---|---|
| review branch | `review/phase7-daemon` |
| commit | `80bb8bbf2430b3eb034a5c0215352a58bd414e29` |
| commit contents | 7 files, 980 insertions, 0 deletions |
| promotion gate | `verdict=PROMOTE rev=80bb8bb`, digest `e4c93251471ec90d445992c1ccb32b02` |
| gate inputs | tree clean, kernel SEALED, 440 passed, 14 skipped, floor 370 |
| merge | `534d8d25bac890b26308feb7ad199d6f47ae20c2`, `--no-ff`, ancestor rc=0 |
| push | `08f3d4b..`/`611fa76..534d8d2`, `PUSH_RC=0` |
| local SHA | `534d8d25bac890b26308feb7ad199d6f47ae20c2` |
| remote SHA | `534d8d25bac890b26308feb7ad199d6f47ae20c2` |
| merged tree | `3d1ac60571c64aa3e55128839ac71b893cfe809e` |
| gated tree | `3d1ac60571c64aa3e55128839ac71b893cfe809e` |

Identical tree hashes certify that promoted content equals gated content
without re-running the suite.

### 17.5 Non-claims carried forward

No benchmark dominance, no SOTA, no market-ready API. `tenant_id` is a
namespace key, not an authenticated identity; there is no auth, rate limit,
TLS, or cross-process isolation, and the daemon binds `127.0.0.1` only.
gRPC is installed but no gRPC service is wired; only FastAPI is exercised.
Phase 6 numbers stand unchanged: c_max 0.781052 -> 0.770887, retrieval
0.19 -> 0.1925, MBPP open-loop top-1 0.0 (below chance), tie-group 1.6333.

## 18. Phase 8: operator ingress, robotics active inference, and a broken ruler

Phase 8 is additive. No sealed kernel file changed; `verify_kernel_integrity.py`
reports `SEALED` on 6 pinned files throughout.

### 18.1 Operator ingress, and what it did not fix

`henri/operator_ingress.py` maps Python AST operators to antipodal slots. On the
Phase 7 fixture the operator-slot cosine between `a + b` and `a - b` is exactly
`-1.000000000000000`, and `tie_group_size` drops `2 -> 1` with gold ranked first
before execution. That clause passes.

The corpus clause `G-AST1` FAILS as pre-registered: mean tie group `1.85` against
a `< 1.6333` bar, open-loop top-1 z `-2.147` against `>= +2.5`, and mutation
validity `0.5119` against `>= 0.90`. The gate was not relaxed.

### 18.2 Three measured defects in the evaluation harness

1. `scripts/mbpp_wave_eval.py` scores `bb.stem.encode`, a 384-dimensional
   sentence cosine, and never builds a wave. No wave improvement could move it.
2. 47.74 percent of MBPP docstrings (465 of 974) produce a zero-width operator
   slot. The Phase 8 note of 45 percent is close; the measured value is 0.4774.
3. 49 percent of mutated "negatives" still pass `test_list`. The random base rate
   of any candidate passing is `0.5521`, so every arm's `exec_verified_top1` sits
   BELOW a blindfolded pick. Open-loop ranking on that pool is ill-posed.

`mbpp_wave_eval.py` stays unmodified as the registered Phase 6 control. Its
receipts were the harness-fidelity check that reproduced every stored Phase 6
number exactly.

### 18.3 Robotics active inference at D = 65536

`henri/robotics_loop.py` is wired read-only over six sealed modules and served at
`POST /v1/robotics/step`.

| arm | delta_init | vetoed | steps | delta_final |
|---|---|---|---|---|
| R1 identity | 0.000000 | False | 0 | 0.000000 |
| R2 near, non-identical | 0.000001 | False | 0 | 0.000001 |
| R3 unsafe | 0.895710 | True | 6 | 0.042648 |
| R4 no axiom | 0.000000 | False | 0 | 0.000000 |

R3 recovers under `tau_veto = 0.0431`. The raw total is `72979.1 us`, which
EXCEEDS the 50000 us digital-twin contract. The contract is written for one veto
cycle and a 6-step recovery is charged six cycles, so per-step latency
`12163.2 us` is also recorded. Both numbers stand; neither replaces the other.

Per-slot energy at R3 is `slot0 = 0.783569`, `slot1 = 0.621304`. The action slot
is not near-empty. `statistic_form` reads `inner_product`, which IS
`INVARIANT_FORM` (`form_pin.py:78`).

### 18.4 Retracted external claims

Two external reference notes asserted the remote was stale at `c40ccbe6`.
Measured false: `git ls-remote origin` returns `main = 534d8d2`, byte-equal to
local HEAD.

The same notes reported a robotics table with `delta 0.4975`, `5 steps`,
`213.8 us`, `dilution 4.0`, `act_norm 0.000786`. None of those values appears in
any receipt on this host. Falsified.

## 19. Phase 9 Step 1: G-KOOP1 verdict, router tests, and a receipts trap

### 19.1 G-KOOP1 FAILS on both arms, and parsimony holds

`receipts/phase8_g_koop1.json`, sha256 `d19179d04bc08fe0...`. Split is the
registered 800/174 with zero task overlap; fit uses 80 rows and held-out 40.

| arm | fit choice | held arm | held control | delta | McNemar | z vs 1/8 | verdict |
|---|---|---|---|---|---|---|---|
| strip | w 0.0 / 0.0 | 0.3750 | 0.3750 | +0.0000 | NO_DIFFERENCE | 4.78 | FAIL |
| opast | w 0.0 / 0.0 | 0.3500 | 0.3500 | +0.0000 | NO_DIFFERENCE | 4.30 | FAIL |

No weight in the grid improved held-out accuracy, so the installed weights stay
`0.0` for both arms. The transition coupling is NOT claimed.

The arm scores `0.3750` against a `0.1250` chance bar with `z = 4.78`. That is
not evidence the ranker is strong. It is the degenerate pool again: the measured
base rate was `0.5521`, so a blindfolded pick should score near `0.5521`. Read
`z` here as a pool property, not a skill property.

### 19.2 The import block was two defects, not one

`dir(henri.operator_ingress)` exposes `OperatorAwareBackbone` and
`OperatorEncoded`. There is no `OperatorAwareEncoder` and no `OperatorIngress`.
The measured constructor is `(base=None, weights=(0.4, 0.1, 0.3, 0.2), dim=65536)`
and `encode` returns a dataclass with fields `psi (65536,) complex128`,
`slot_operator (16384,)`, `slot_structure (16384,)`, `pol`, `parsed`,
`operator_stream`, `active_slots`.

Fixing the name alone would have failed again. `np.asarray` on the dataclass
yields a 0-d object array, so `_psi` would raise `ValueError: encoder emits width
0`. The probe now extracts `.psi`.

### 19.3 Cluster test coverage closes the Phase 8 gap

`tests/test_phase8_koopman_router.py` adds 9 tests against a stub backbone, so no
sentence weights load. Covered: `config_for` sizes from the wave not from a
guessed attribute; `_psi` accepts an object with no `cfg` (the Phase 8
`AttributeError`); a width mismatch raises instead of mixing dimensions; the
pluggable encoder breaks the `a + b` versus `a - b` tie with the unencoded
control held at `tie_group_size = 2`; the parsimony rule installs `w = 0.0` on an
all-zero grid; `domain_of` matches a literal sha256 bucket; McNemar ignores
concordant pairs; `CartridgeRecall` round-trips.

Two defects in the first draft of that file, both repaired: `v = base - base`
produced a zero vector and raised in `from_complex`, and `recall` was called with
a bare array where it is typed `WaveState`.

### 19.4 The receipts trap

`git check-ignore -v receipts/phase8_robotics_full_D.json` matched
`.gitignore:8:receipts/*`. Every evidence receipt was ignored, so `git add -A`
would have carried ZERO receipts while a report could still claim "receipts
committed". Repaired with explicit negation for `receipts/phase*.json` and
`receipts/kernel_*.json`. Scratch console dumps (`receipts/_*.txt`) stay ignored
and are deleted, never force-added.

### 19.5 Step 2 fixture, measured before any edit

Measured on the pre-existing module, so Step 2 starts from fact:

| clause | measured | bar | verdict |
|---|---|---|---|
| cosine(sum_doc, difference_doc) | -1.000000000000 | == -1.0 | PASS |
| cosine(sum_doc, `return a + b`) | +0.707106781187 | > +0.70 | PASS |
| cosine(sum_doc, `return a - b`) | -0.707106781187 | < -0.70 | PASS |
| zero operator slot, 974 docstrings | 0.4774 (465) | <= 0.10 | FAIL |

The lexical bridge already exists for the three fixture clauses. Both references
predicted `cos = 0.0` because `"and"` would map to the Boolean axis. Measured
false: the shared operator feature set is `['ARITH_ADDITIVE']` alone, with weight
`+1` against `-1`. `"and"` never enters the operator slot.

The cross-modal margins are `0.0071` above the bar. A single stray stopword in
the operator channel would flip those two clauses to FAIL. That fragility is the
real Step 2 finding.

## 20. Phase 9 Step 2-4 and generative synthesis

Measured 2026-10-21 on `main` at `661a353`. Kernel `SEALED`. No sealed file
changed; every edit is additive.

### 20.1 Step 2 cross-modal lexical-to-operator homomorphism

Lexicon 81 -> 121 keys. Gate S2_PARTIAL_6_OF_7, six clauses of seven.

| clause | measured | bar | verdict |
|---|---|---|---|
| S2-1 cos(sum_doc, difference_doc) | -1.000000000000 | == -1.0 | PASS |
| S2-2 cos(sum_doc, `a + b`) | +0.707106781187 | > +0.70 | PASS |
| S2-3 cos(sum_doc, `a - b`) | -0.707106781187 | < -0.70 | PASS |
| S2-4 zero operator slot | 0.3747 | <= 0.10 | FAIL |
| S2-5 bleed control, cos to `*` | 0.0 | < +0.70 | PASS |
| S2-6 mean pairwise collision | 0.07268 | < 0.10 | PASS |
| S2-7 code antipody held | -1.0 | == -1.0 | PASS |

S2-4 STAYS FAIL. The bar is unreachable on this corpus by arithmetic, not by
tuning. Coverage moved 0.4774 -> 0.3747, which is 465 -> 365 zero-slot
docstrings. The residue names no operation at all: find, check, convert,
sort, regex, extract. Driving the rate under 0.10 would require mapping
verbs that denote no operation into the operator basis, which destroys the
S2-6 collision control. Recorded as a bounded mechanism result.

FALSIFIED REFERENCE CLAIM. Two advisory references predicted
`cos(sum_doc, difference_doc) = 0.0`, on the argument that `"and"` maps to
the Boolean axis. Measured -1.000000000000. The shared operator feature set
is `['ARITH_ADDITIVE']` alone at +1 against -1. `"and"` never enters the
operator slot, and the primary-root rule they proposed already exists in the
module.

### 20.2 Step 3 G-AST2 on verified-failing negatives

Receipt `phase9_g_ast2.json`, verdict G_AST2_PASS, five clauses of five.

| clause | measured | bar | verdict |
|---|---|---|---|
| A2-1 pool validity after rejection | 1.0000 constructed | == 1.0 | PASS |
| A2-2 raw generator validity | 0.778571 | reported | PASS |
| A2-3 wave top-1 | 0.4000, z = +5.259 | > 0.1250 | PASS |
| A2-4 wave vs stem_384d | 0.4000 vs 0.2500 | wave > stem | PASS |
| A2-5 closed-loop vs pool base | raw base 0.318750 | reported | PASS |

40 held-out tasks, train/held overlap 0, 0 dropped, 36.7 s.

THIS IS A DISCRIMINATION RESULT, NOT A GENERATION RESULT. A negative is
admitted only when the sandbox reports failure, so validity 1.0 holds BY
CONSTRUCTION and the 0.1250 chance level is constructed rather than
discovered. Filtering by execution carries answer knowledge. The number
supports no generation claim and must not be cited as one.

Two recorded inflators. Ties break toward index 0, where the gold source
sits, so the two tied items (mean tie group 1.05) can inflate 0.40 by up to
0.05. `mbpp_wave_eval.py` is untouched as the registered Phase 6 control;
`probe_phase9_mbpp_wave.py` is the additive replacement.

### 20.3 Step 4 generative synthesis, G-SYNTH1

Receipt `phase9_g_synth1.json`, verdict G_SYNTH1_PASS, executed rate 0.225.

Caller supplies only a docstring and assertions. `candidates=[]`,
`synthesize_from_bank=True`, 800-row bank, held 174 tasks, 40 evaluated,
overlap 0 by task id.

| clause | measured | bar | verdict |
|---|---|---|---|
| SY1 held-out overlap | 0 | == 0 | PASS |
| SY2 exec verified rate | 0.2250, Wilson [0.1232, 0.3750] | reported | REPORTED |
| SY3 full vs no-synthesis control | 0.2250 vs 0.0000 | strictly greater | PASS |
| SY4 NameError share of failures | 0.0 of 652 | reported | REPORTED |
| SY5 mean executions per task | 16.65 | <= 24 cap | PASS |

Nine of forty held-out tasks solved with all assertions passing, from a cold
docstring. Winners split `bank_adapted` 5 and `mutated` 4, so signature
adaptation and operator-guided mutation each carry part of the result. Peak
RSS 5679 MiB during the real 800-row ingest, measured rather than
extrapolated.

LEAKAGE CONTROL. Task-id overlap is not code overlap. A separate check
compared each of the nine winning gold sources against all 800 bank rows,
by normalised text and by AST with the function name stripped. Result 9/9
`none`: `leak_exact_share 0.0000`, `leak_ast_share 0.0000`. Receipt
`phase9_g_synth1_leakcheck.json`. The 0.225 is not retrieval of a stored
answer.

### 20.4 Defects found in my own work, each by running it

`adapt_arity` substituted synthetic parameter names while the body still
referenced the originals, producing `def add(a, b): return x + y + z`. Every
adapted candidate raised `NameError`, and the generative gate would have
recorded a false zero for a reason unrelated to the operator channel. Fixed
before commit `661a353` and verified by execution: adapted source returns 7,
truncation returns 42.

`TOP_K` was declared but never passed to the call. `solve_verified_code`
defaults to `top_k=3` and truncates the ranked pool BEFORE execution.
Measured on task 803 of the real bank: `top_k=3` gives
`NO_CANDIDATES_PASS` with `n_exec 3`, and `top_k=24` gives `EXEC_VERIFIED`
with `n_exec 23`. The default would have measured the truncation and
reported it as a generation failure.

`wave_guided_order` defaulted to `dim=4096` while the operator slot basis is
16384 wide, introducing a second slot-width convention. Aligned to 16384.

The probe's print loop indexed `clauses["SY3"]["measured"]`, which does not
exist, and raised `KeyError` AFTER the receipt was already written. A
completed measurement looked like a crash. Display fix only; no clause
changed.

### 20.5 Retraction: the 14643 MiB bank figure

A measurement script printed `rss delta 14643 MiB` for 800 templates. That
number is WRONG. The per-template term added base RSS divided by the sample
count, inflating it. The measured delta was 232 MiB over 50 templates. An
advisory reference then repeated 14643 MiB back as a blocker. The real
figure, measured during the actual G-SYNTH1 ingest, is
`rss_delta_mib 4999.6`, `peak_rss_mib 5679.2` on a host with 31.1 GiB total
and 17.0 GiB available. Anyone resurrecting 14643 MiB is citing a bug.

### 20.6 Phase 9 Step 1, G-KOOP1

Receipt `phase8_g_koop1.json`, sha256 `d19179d04bc08fe0`. Verdict FAIL on
both arms, installed weights 0.0. Parsimony held: held-out accuracy did not
strictly improve over the static control, so the coupling is NOT claimed.
The measured `z = 4.78` against 1/8 must be read as a pool property, not a
capability: the Phase 8 pool base rate was 0.5521, so near-chance play
scores high there.

The single-line import was two defects. The real exports are
`OperatorAwareBackbone(base=None, weights=(0.4,0.1,0.3,0.2), dim=65536)`
returning `OperatorEncoded`, a dataclass with `psi (65536,) complex128`.
No `OperatorAwareEncoder` and no `OperatorIngress` exist. Fixing only the
name would have failed again, because `np.asarray(dataclass)` yields a 0-d
object array. Router tests `tests/test_phase8_koopman_router.py` pass 9.

### 20.7 Robotics per-step latency

`total_latency_us` and `per_step_latency_us` are now emitted and recorded
beside the raw total. At D = 65536, arm R3 measures raw total 76457.8 us
against the 50000 us single-cycle contract, which FAILS, and per-step
12742.967 us, which passes the same contract. Both numbers are recorded and
neither replaces the other. The arm stays FAIL on `twin_latency`: a 6-step
recovery is charged 6 contract windows, and the contract is written per
cycle. The bar is not moved; the applicable term is reported beside the
failure.

### 20.8 Standing constraints learned this phase

A gate that cannot fail will pass. S2-4 was kept FAIL although it blocks a
clean report, because a relaxed bar is not evidence.

`receipts/*` was in `.gitignore`, so `git add -A` carried ZERO receipts
while a report could still claim they were committed. Negated for
`receipts/phase*.json`; scratch `_*.txt` stays ignored and is never
force-added.

Commit before measuring. Two consecutive phases lost measured work to a
tool-iteration ceiling while it sat uncommitted.
CORRECTION, APPENDED AFTER THE SECTION 20 WRITE. Section 20.7 quotes the first
full-D re-run: raw total 76457.8 us, per-step 12742.967 us. The receipt that
ships, `phase8_robotics_full_D.json` sha256 `51c302f13853e343`, records a
SECOND run of the same probe: raw total 81449.0 us, per-step 13574.833 us.
Both runs are real.

Latency is run-variable on this host. The RATIO is the stable quantity:
12742.967 / 76457.8 = 0.16667 and 13574.833 / 81449.0 = 0.16667, one sixth
across six steps in both runs. The shipped receipt is authoritative for the
numbers. This note exists so the two runs cannot be confused, and neither
number is adjusted.
## 20.9 Receipt identity: body self-digest vs file sha256

MEASURED CORRECTION. Sections 20.2, 20.3 and 20.6 quote a "sha256" for
receipt files. Those values are BODY SELF-DIGESTS, not file hashes. The label
was wrong and I wrote it into this register and into the commit message for
`08b9307`. No receipt is corrupt; only the label is.

Two identities exist per receipt that carries the self-field:

| identity | how to reproduce | scope |
|---|---|---|
| body self-digest | strip `receipt_sha256_of_body_before_self_field`, re-serialise `json.dumps(indent=2, sort_keys=True)`, sha256 | proves internal consistency |
| file sha256 | `sha256sum receipts/<name>.json` | identifies the committed bytes |

Measured, all four self-fields recompute exactly:

| receipt | body self-digest | file sha256 |
|---|---|---|
| `phase8_robotics_full_D.json` | `51c302f13853e343` | `5db6db8aa0517232` |
| `phase8_g_koop1.json` | `d19179d04bc08fe0` | `f4013a56d85828fb` |
| `phase9_opaxis.json` | `5e2e7949c19c00fc` | `25f9293f823387b9` |
| `phase9_g_ast2.json` | `b00d04c9f7bc61d4` | `08b32288c6381d60` |

`phase9_g_synth1.json` (`c3cd5ed68f49b29f`) and
`phase9_g_synth1_leakcheck.json` (`d4dbd944649a3651`) carry no self-field,
so only the file hash exists for them.

A self-digest proves linkage from payload to field. It does not prove the
field is unaltered, because an editor can rewrite both. It is not a
signature and not a MAC. Section 17 already records this limit; this section
adds the reproduction recipe so a reader can check either number instead of
trusting the citation.

SOURCE FIXED. `scripts/probe_phase8_robotics.py` printed the body digest as
`receipt sha256`. It now prints `body_sha256_self_field` and
`file_sha256_on_disk` on separate lines.

## 21. Phase 10: structural morphisms, CEGIS repair, reason channels, robotics latency

Phase 10 executed four pre-registered gates on top of `854605d`. Three mechanisms
were added; one existing loop was optimized. Two gates pass whole, two pass
partial. Every FAIL is recorded as measured, and two of the four gates produced a
null result for the added mechanism.

### 21.1 Step 1, G-MORPH1: PARTIAL 3 of 4, M4 FAIL

An 8-axis structural morphism channel was added to `henri/operator_ingress.py` as
an APPEND-ONLY block. The arithmetic channel is byte-identical: pins
`0d55b5ee` (slot 1) and `59ed073d` (slot 2) were taken BEFORE the edit and
reproduce after it.

| Clause | Measured | Bar | Verdict |
| :--- | :--- | :--- | :--- |
| M1 | axis projection `-1.000000000000` (full channel `0.0`) | `== -1.0` | PASS |
| M2 | `+1.0` / `-0.707106781187` | `> +0.70` / `< -0.70` | PASS |
| M3 | `0.901437` combined, both-empty `0.098563` | `>= 0.88` | PASS |
| M4 | `0.3250` | `>= 0.4000` | **FAIL** |

M4 is the load-bearing clause and it fails. Attribution was measured, not guessed:
of 40 `G-AST2` prompts, the 31 that fire a morphism axis fall `12 -> 9` correct,
while the 9 that fire none stay `4 -> 4`. Populating a fifth slot is zero-sum,
because `psi` is unit-normalised across four slots: slots 0-2 rescale
`0.707107 -> 0.632456` when slot 3 gains energy.

M3 and M4 are one phenomenon. Coverage rose because `MORPH_COUNT_IDX` fires on
`558` of 974 docstrings and `MORPH_MAP_DICT` on `384`, driven by the words
`count`, `values`, `find`, `first`, `last`, `list`, `set`. Those words dominate
ARITHMETIC prose. Coverage is a corpus statistic and is not evidence of ranking
power.

A pre-registered weight sweep, selecting on train and evaluating once on held-out,
chose `t = 0.00`, which DISABLES the channel. The verdict string was corrected
from `M4_RECOVERED` to `M4_PARSIMONY_HELD_CHANNEL_OFF` before commit. Reporting a
baseline as a channel win is the Phase 9 label defect in new clothing.

M1 also required a correction. The directive's own fixture cannot reach `-1.0`:
`"reverse a list"` and `"sort a list"` BOTH fire Axis 8 on `list`, so the full
channel reads `0.0`. The `-1.0` is measured on the axis projection. Both numbers
ship.

### 21.2 Step 2, G-CEGIS1: PASS 4 of 4

`henri/code_verify.py` gained an optional second round, `max_cegis_rounds`. It
defaults to `1`, so the registered `G-SYNTH1` path is unchanged and the gate
measures `1` against `2` rather than comparing a new path to itself.

| Clause | Measured |
| :--- | :--- |
| C1 no regression at rounds=1 | byte-identical baseline |
| C2 held-out `exec_verified_rate` | `0.2250 -> 0.2750`, 2 tasks gained of 40 |
| C3 provenance | winners tagged `origin="cegis_repair"` |
| C4 refusal | `unclassified` returned when the residual matches no case |

The margin is `+2` tasks on `n = 40`. It is recorded at that width and is not
stated as a capability. The classifier refuses rather than guesses: a fixture with
delta `3` correctly returned `unclassified`.

### 21.3 Step 3, G-REASON1: PARTIAL 2 of 3, parsimony holds `w = 0.0`

Operator and morphism channels were wired into `/v1/reason/rank` and
`benchmarks/wave_rank.py` as an ADDITIVE score term with `channel_weight = 0.0`
by default, so the prior path is byte-identical. A fifth slot was rejected here
for the reason measured in 21.1.

| Clause | Measured | Verdict |
| :--- | :--- | :--- |
| R1 4x rotation invariance | `mmlu_w0.0 1.000000`, `mmlu_w0.2 1.000000`, `gpqa_w0.2 0.950000` | **FAIL** |
| R2 STEM operator fixture | `3/8` wave-only vs `3/8` channel, no gain | reported, NOT counted |
| R3 MMLU/GPQA held-out | both corpora `delta +0.0000`, parity | PASS |
| R4 selection on train only | `w = 0.0` chosen on both | PASS |

Parsimony selected `w = 0.0` on BOTH corpora, which disables the channel. The bar
is then met by the wave core, not by the added term. The verdict string names
this: `G_REASON1_PARTIAL_2_OF_3`.

The R2 clause originally carried `"pass": True` while being described as
"reported, not a bar", and the tally counted it. That is a gate that cannot fail.
A clause I cannot fail must not consume a pass slot, so R2 now carries
`counted: false` and the tally excludes it.

Scope is `SUBSET_SMOKE_STRATIFIED`: 200 MMLU rows and 198 GPQA rows, fixed before
the run. A subset result is not a benchmark score and is not quoted as one.

### 21.4 Step 4, G-ROB2: PASS 4 of 4, total recovery below the 50 ms twin

| Clause | Measured | Bar | Verdict |
| :--- | :--- | :--- | :--- |
| B1 R1-R4 reproduce Phase 8 | deltas, steps and slot norms identical | `<= 1e-12` | PASS |
| B2 full-D equivalence | koopman `1.390e-17` / `1.304e-17`, relaxation `0.000e+00` | `<= 1e-12` | PASS |
| B3 R3 total recovery | `42,932.8 us` median of 7 | `<= 50,000` | PASS |
| B4 R3 per-step latency | `6,632 us` | `<= 50,000` | PASS |

Recorded prior total was `81,449.0 us`. Measured now `42,932.8 us`, a reduction of
`38,516.2 us` or `47.29` percent. R3 recovery is unchanged at
`0.895710 -> 0.042648` in 6 steps, and the slot norms match Phase 8 digit for
digit: `slot2 0.000804078`, `slot3 0.000807872`.

THE DIRECTIVE'S SLICING PRESCRIPTION WAS TESTED AND REJECTED. It proposed
operating on `[0:32768]` during the inner loop because the inactive slots are
zero. Measured: slots 2 and 3 are EXACTLY zero on the assembled candidate, but
NON-ZERO after relaxation, in this run and in the Phase 8 R3 arm. Slicing would
truncate live state and break B1 and B2. The savings came from removing
redundant work instead, with arithmetic preserved:

| Optimization | Before | After | Why exact |
| :--- | :--- | :--- | :--- |
| temperature field per recovery | 6 recomputations, `3,604.9 us` each | computed once | pure function of comb and basis; comb byte-identical across all 6 steps |
| Koopman pass | two passes `23,991 us` | one 2-column batch `16,309 us` | column-wise application; `max abs diff 2.201e-17` |
| right adjoint | `right.conj().T` per call `9,156 us` | cached contiguous `845 us` | identical arithmetic; non-contiguous transpose blocked the fast path; diff `0.000e+00` |

An intermediate state measured `50,233 us` and FAILED B3 while 8 to 9 competing
Python processes ran on this host. That run was recorded as FAIL. Further
optimization, not a re-run, brought the measurement to `42,932.8 us`.

### 21.5 Defect found in my own receipt writer

`henri/receipt.py` was written to prevent the Phase 9 label defect. It reproduced
it. The writer inserted `file_sha256_on_disk`, then hashed the file and returned
that hash. The returned value was correct; the EMBEDDED value was the hash of the
serialization before the field was inserted. Measured on all four Phase 10
receipts: embedded value did not equal `sha256sum` of the file.

The arithmetic is not repairable by iteration: a file cannot contain its own
SHA256. The reproducible identity therefore moved to a sibling
`<name>.json.sha256` sidecar, verifiable with `sha256sum -c`, and the embedded
field now carries a note saying what it is. `verify_receipt` checks the sidecar.
Measured after the fix: `body_match True` and `file_match True` on all four
receipts.

### 21.6 Standing constraints learned this phase

- A file cannot contain its own hash. Put the reproducible identity in a sibling.
- A clause that cannot fail must not consume a pass slot. Count only clauses that can fail.
- Coverage is not ranking power. A word-frequency win can bind the mass that ranking needs.
- Do not slice an inactive slot unless it is EXACTLY zero AT THE POINT OF USE, not merely at assembly.
- `conj().T` on a `(65536, 64)` complex array is 10 times slower than a precomputed contiguous adjoint, for identical arithmetic.
- Report latency as a median of N fixed before the run, with the observed range, and state competing host load.
- Three added mechanisms held parsimony at zero: `G-KOOP1` `w = 0.0`, the M4 sweep `t = 0.0`, `G-REASON1` `w = 0.0`. On these benchmarks the wave core carries the signal and the added channels do not beat it.

## 22. Phase 11 Step 1-2: tie-break repair, falsified premises, null-space gate

Baseline for this section: `origin/main = ae1882b` (Step 1). Step 2 is recorded
here and committed in the same change set.

### 22.1 Step 1 (`ae1882b`, pushed): GPQA rotation was a tie-break defect

The directive attributed the stalled `G-REASON1` rotation arm to re-encoding cost.
Measurement falsifies that attribution.

`strip_common_option_tokens` collapses two distinct options to identical stripped
text on 25 of 198 GPQA items. The old sort key was
`(-round(score, 12), sha256(stripped_text))`. Colliding options share both keys,
so Python's stable sort preserves caller order, and a cyclic permutation of
`[A, B, C, D]` changes the winner.

| Measurement | Value |
|---|---|
| GPQA items with colliding stripped text | 25 of 198 |
| Previously non-equivariant items | 14 of 198 |
| Non-equivariant items with colliding stripped text | 14 of 14 |
| Non-equivariant items with 4 distinct texts | 0 of 184 |

The repair adds the ORIGINAL option digest as a permutation-equivariant secondary
key. Scores are not rescaled. A cached encoder alone leaves `0.950000` unchanged.

Receipt `receipts/phase11_g_reason1_repair.json`, verdict
`G_REASON1_REPAIR_PARTIAL_3_OF_4`, sidecar OK:

| Clause | Measurement | Verdict |
|---|---|---|
| P1 | `gpqa_w0.2 = 1.000000` (frozen prior `0.950000`), all 4 arms `1.000000` | PASS |
| P2 | 25 collide, 14 previously broke | FAIL, recorded as FAIL |
| P3 | `w = 0.0` held, `0.280000` / `0.282828`, delta `+0.000000000` | PASS |
| P4 | 20111 calls to 1762 unique encodes, `max abs diff 0.000e+00`, mutation-safe | PASS |

P2 fails by design: collision is necessary, not sufficient, for the defect. The
clause exists to record that distinction rather than hide it.

The frozen `receipts/phase10_g_reason1.json` keeps its measured `0.950000` and is
cited as `prior_measurement`. A receipt describes the revision that produced it.

### 22.2 Two directive premises falsified before implementation

**Active-axis normalisation is a no-op.** The directive required dividing the
structural channel score by the number of ACTIVE structural axes. Measured:
`ring_slot` already unit-normalises, so `||slot|| = 1.000000` for 1, 2, and 3
active axes. Inactive axes contribute zero energy and dilute nothing. The
divisor would always be 1. Not implemented; recorded as a falsified premise.

**The Step 1 stall was tie-breaking, not encoding cost.** Measured above.

### 22.3 Refused verdict: a PASS earned by never firing

Probe run 3 printed `G_MORPH1B_PASS` with M1-M4 all passing. The measured table
alongside it:

| `w_morph` | train | held-out | gate shut | exact |
|---|---|---|---|---|
| 0.00 | 0.3750 | 0.4000 | 40/40 | 40 |
| 0.05 | 0.4000 | 0.4000 | 40/40 | 40 |
| 0.10 | 0.4000 | 0.4000 | 40/40 | 40 |
| 0.20 | 0.4000 | 0.4000 | 40/40 | 40 |

The gate shut on every held-out item at every weight. Each clause passed because
each return was `base` untouched. This verdict was not committed. Two clauses
were added so the mechanism itself is tested: M5 (channel must act on held-out)
and M6 (dedicated structural split must engage without regression).

### 22.4 Step 2 gate of record: `G_MORPH1B_STRUCTURAL_ONLY_4_OF_5`

Receipt `receipts/phase11_g_morph1b_run7.json`, sidecar OK, provenance
`held_out_plus_train_topup` (held 16 / train 9, rejected_slot1_active 36).

| Clause | Measurement | Verdict |
|---|---|---|
| M1 | `w = 0.0` reproduces `0.4000` and returns base exactly | PASS |
| M2 | shut gate bit-exact; held-out `0.4000` | PASS |
| M3 | slot1-silent items not worse | PASS |
| M4 | parsimony, selected `w = 0.05` | PASS, uncounted |
| M5 | channel acts on G-AST2 held-out | FAIL |
| M6 | structural split engages, no regression | PASS, `0.4400 -> 0.5200`, `+2/25` |

**M5 is unsatisfiable by specification, not a channel defect.** Every G-AST2
prompt carries an active, untied Slot 1. The null-space gate shuts on exactly
that population. M5's bar targets a population where firing is forbidden by the
design under test. Engagement evidence is M6.

**M6 scope limit.** The split is 16 held-out rows plus 9 topped up from train.
The receipt records `rejected_slot1_active: 36`, the count of lexicon-firing rows
discarded because Slot 1 was active. `+2/25` is narrow and partly in-sample. This
records a real in-scope win on the split built to test the channel. It does NOT
establish a general structural-ranking improvement.

### 22.5 Three defects in the probe, caught before the producing run

Each was measured, repaired, and the run that produced the committed number was
launched after the repair.

1. **Split selection violated the pre-registration.** Run 4 selected on lexicon
   alone and produced 15 of 25 items with an ACTIVE Slot 1. The directive defines
   the structural split as `slot1_active == False`. Measured populations: 266
   structural docstrings, 100 slot1-silent. Fixed with `slot1_is_silent`.
2. **Receipt field contradicted its own log.** Run 6 recorded
   `provenance: none` while its log printed `held_out_plus_train_topup, held 16 /
   train 9`. A section re-initialised the provenance dict after it was set. Removed.
3. **Verdict ladder masked a scoped partial.** The INERT label was consulted
   before M6, so a structural-only engagement would have been reported as inert.
   Reordered. New label `G_MORPH1B_STRUCTURAL_ONLY_4_OF_5` names which split fired.

Falsification trail retained: runs 3, 4, 5, 6 receipts are committed alongside
run 7. Run 3 is the refused vacuous PASS. Run 4 is the split-selection defect.
Run 6 is the provenance defect.

### 22.6 Invariants

| Invariant | Measured |
|---|---|
| Kernel | `SEALED`, 6 pinned files |
| Register | pure CRLF, 0 bare LF |
| Phase 10 Slot-1/Slot-2 byte pins | intact, 39/39 pins and channels suite |
| Step 2 source diff | additions only, `172/0` and `183/1` |
| Full suite | 607 collected, 593 passed, 14 skipped, 0 failed |

### 22.7 Extracted epiplexity

The recurring failure mode this phase was a gate satisfied by the baseline rather
than by the mechanism, the same shape as the Phase 9 defect. Two premises were
falsified by single measurements, and one PASS was refused at the cost of a run.
The reusable rule: a clause must be able to fail BECAUSE the mechanism did not
fire, and a verdict label must name which split engaged.
