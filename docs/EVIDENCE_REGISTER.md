# Evidence register

Every material result carries one label:

`OBSERVED` `DERIVED` `INFERRED` `HYPOTHESIS` `FALSIFIED` `BLOCKED`

Pytest results are code-health evidence. They are not model-intelligence
scores and they are not capability claims.

## 1. What this build demonstrates

| Item | Evidence | Label |
|---|---|---|
| CPU test suite passes | measured return code | `OBSERVED` |
| Unit norm preserved by every operator | tests assert `|‖psi‖-1| < 1e-5` | `OBSERVED` |
| Zero dense `D x D` allocation | guard raises; low-rank paths only | `OBSERVED` |
| Slot bleed <= 0.0015 with zeroed slots | measured 0.0 exactly | `OBSERVED` |
| Gram crosstalk <= 0.12 after retraction | measured <= 1e-17 orthonormal | `OBSERVED` |
| Hopfield energy non-increasing | measured `diff <= 1e-9` | `OBSERVED` |
| Veto separation >= 0.70 (directive 3 bar) | calibration + labelled pairs | `OBSERVED` |
| No resonator module exists | import raises `ModuleNotFoundError` | `OBSERVED` |

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
| License grant | owner decision outstanding |

## 5. Non-claims

- No ARC-AGI task was attempted or solved by anything in this build.
- The retired repository's live benchmark score remains 0.0%.
- The sealed action readout measured LESS decisive than noise on live
  encoder waves (margin 0.023 vs 0.050). No floor is set from a random
  baseline.
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
