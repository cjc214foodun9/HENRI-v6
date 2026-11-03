# HENRI Holarch: the three-zone wave-mechanical pipeline

Status: OBSERVED on CPU, 2026-11-02. Every number here comes from a file in
receipts/holarch/ or from a test named next to it. A test result is code-health
evidence, not a capability claim.

This revision implements Directives 1 to 4. Section 7 records which gates
passed and which did not, with the measured number in both cases.

## 1. Data path

~~~text
                    HENRI HOLARCH
    Zone A  lean swarm          Zone B  photonic kernel      Zone C  memory
    ---------------             ----------------------      ---------------
    fast weights                BaTiO3 Pockels plates       holographic engrams
    top-k routing               microheater thermal grid    TimescaleDB hypertable
    novelty search              Sagnac homodyne loop        NVMe SQLite fallback
    DAgger + replay             Stiefel slot unbinding      Gram index + pager
~~~

~~~mermaid
flowchart LR
  X[pixels / tokens / triples] --> E[modality encoder]
  E --> P[wave projector / slot binding]
  P --> R[Zone C raw top-1 + context]
  R --> F[wave fusion]
  F --> B[Zone B plates + slot mixer]
  B --> V[two-condition veto]
  V --> L[thermodynamic relax loop]
  L --> G[Stiefel slot unbinding]
  G --> A[Zone A swarm]
  A --> Y[task logits]
~~~

The sensor skip path is severed in phase 1: skip_alpha = 0.0 by default, so
the task gradient reaches the BaTiO3 phase masks and the heater controller
through the optical path only. The path can be restored by setting
skip_alpha > 0 for a later phase; the ablation in section 6 measures the
severed configuration.

## 2. Zone B: the photonic kernel

### 2.1 Plates, heaters, and the slot mixer

Each plate is a BaTiO3 film with a phase mask theta[s,d]. A microheater grid
sets a normalised temperature T. The Pockels response follows the
Curie-Weiss susceptibility:

~~~text
Lambda(T) = (1 + kappa / ((1 - T)^gamma + eps)) / (1 + kappa / (1 + eps))
phi[s,d]  = theta[s,d] + direction * g * Lambda(T[s]) * V[s]
~~~

Lambda(0) = 1 and Lambda(0.85) = 3.665. A bounded drive of 0.1 rad moves the
field 0.19 rad at T = 0.1 and 0.52 rad at T = 0.85, a factor of 2.74.

A learned 4x4 unitary SlotMixer sits between plates. It is the Cayley
transform of a skew-Hermitian matrix, so it is exactly unitary for every
parameter value. This is the cross-slot coupler: the FFT chirp mixes
frequencies inside a slot, and the mixer mixes slots.

### 2.2 Slot-specific Stiefel unbinding

The readout is not a magnitude average. For each slot s the field is
projected onto a learned Stiefel basis U_s in C^{K x d_slot}:

~~~text
g_s = Re(U_s psi_s)          U_s U_s^H = I       (QR retraction every pass)
features = [g_0 || g_1 || g_2 || g_3 || dark energy || 8 scalars]
~~~

K = 32 in the CI and production profiles, 64 in the Blackwell profile. The
QR retraction runs in the forward pass, so the basis stays on the manifold
during training rather than being regularised toward it. A test measures
max |U U^H - I| < 1e-4.

### 2.3 Two-condition veto

Directive 2 replaces the single softmax-context check with a disjunction:

~~~text
s_max        = max_k Re<gram_q, gram_k> / (|gram_q| |gram_k|)   (raw, no softmax)
delta_sagnac = 1 - Re<psi_out, psi_axiom>                       (post-photonic)
Veto(x)      = 1[s_max < tau_chance] OR 1[delta_sagnac > tau_residual]
~~~

The axiom is the top-1 engram, not the softmax bundle, and it is used at the
coupler without being fused into the query at weight above 0.5 (the learned
fusion gate is 0.27 at initialization).

tau_chance is the 1st percentile of in-distribution s_max minus a margin;
tau_residual is the 99.5th percentile of the raw residual plus a margin. The
residual is calibrated with relaxation disabled, because relaxation is
monotone and can only lower the deployed residual.

The raw statistic needed a representation that separates uniform noise from
digits. Outlier exposure during training does that: a small auxiliary loss
pushes the wave of a uniform input away from the wave of a real input.

### 2.4 Thermodynamic relaxation loop

When the residual exceeds its threshold, Zone B runs up to four optical
passes. The Sagnac error drives the heaters:

~~~text
T[k+1][s] = clip(T[k][s] + alpha_thermal * delta[k][s], 0, 0.999)
psi[k+1]  = normalise(psi[k] * exp(i * fraction * angle(axiom * conj(psi[k]))))
~~~

The loop keeps the best pass and stops when the residual is below tolerance
or two passes fail to improve. The measured trace on a random pair falls
from 0.993 to 0.381 in three passes.

## 3. Zone C: holographic memory

Unchanged in structure, extended with the raw statistic. One engram is one
row: four complex64 slot blobs, a 64-value complex Gram projection, the
label, the viscoelastic weight, and the tier. TimescaleDB serves when the DSN
is reachable; WAL-mode SQLite on NVMe serves otherwise, and the backend is
recorded.

The context tensor is (context_tokens, context_dim) at every store size. The
canonical run stored 1,200 engrams with 2.45 ms mean retrieval and 0.965
recall@1 over 200 held-out windows.

## 4. Zone A: the lean swarm and four tasks

Each expert is a two-layer MLP plus a rank-r delta-rule fast memory. The fast
write is restricted to the task's own label block, and the four blocks are
disjoint (vision 0-9, language 10-105, action 106-109, relational 110-117).

Tasks:

| Task | Input | Target | Split |
|---|---|---|---|
| vision | 8x8 digits | 10 classes | 1200 / 300 / 297 |
| language | 24-character window | next character, 96 classes | 90 / 10 percent |
| action | 5x5 view + goal vector, 79 values | BFS optimal action | 400 / 80 layouts |
| relational | (entity, action, target) triple | trilinear outcome, 8 classes | 216 / 72 triples |

The relational task holds out 25 percent of the (e, a, t) combinations,
stratified by class. Every primitive and every class appears in training.
The model binds the three role waves by pointwise phasor product and writes
the product into slot 3, which is VSA binding and is parameter free.

## 5. Measured results

Receipt: receipts/holarch/holarch_small.json
SHA-256: 4e91866f163467bfe7fbb848f7dcba66a21e2c8d5a055a3a502e17a0cc31166f

Configuration: HolarchConfig.small(), 300 steps per task, batch 32, 20 DAgger
rounds, outlier exposure weight 3.0, 189,711 parameters, TimescaleDB backend.

| Task | Metric | Validation | Test | Chance |
|---|---|---|---|---|
| Vision | accuracy | 0.967 | 0.966 | 0.100 |
| Language | next-char accuracy | 0.222 | 0.222 | 0.010 |
| Language | perplexity | 16.60 | 17.70 | 96 |
| Action | one-step accuracy | 0.763 | 0.763 | 0.250 |
| Action | closed-loop success | 0.417 | - | 0.031 random |
| Relational | held-out triple accuracy | 0.479 | 0.479 | 0.125 |
| Memory | recall@1 over 200 windows | 0.965 | - | - |

Per-task selective prediction:

| Task | Veto rate | Coverage | Selective accuracy |
|---|---|---|---|
| Vision | 0.017 | 0.983 | 0.973 |
| Language | 0.014 | 0.986 | 0.225 |
| Action | 0.000 | 1.000 | 0.763 |
| Relational | 0.000 | 1.000 | 0.479 |

Outlier exposure still trades accuracy for separation: with weight 0.2 the
same pipeline reaches 0.95 vision accuracy but only 0.79 OOD veto; with
weight 3.0 it reaches 0.966 and 1.000. The DC-subtracted projector recovered
the accuracy at the hardened setting; it did not make weight 0.2 sufficient.
The canonical receipt records the hardened setting.

## 6. Ablation

receipts/holarch/ablation_vision.json, 150 steps, same seed, vision only.

| Variant | Accuracy | Delta vs full |
|---|---|---|
| full | 0.890 | 0.000 |
| identity optics (plates bypassed) | 0.847 | -0.043 |
| no Zone B features | 0.313 | -0.577 |
| no routing (one expert) | 0.893 | +0.003 |
| no Zone C | 0.883 | -0.007 |
| no fast weights | 0.903 | +0.013 |

With the skip severed, the optical feature path is load-bearing: removing it
costs 57.7 points. The stricter identity-optics control now costs only 4.3
points, down from 22.7 in the previous revision. The richer Stiefel readout
(K = 32 plus bilinear products) gives the bypass enough linear capacity to
recover most of the task. The gate passes on the removal definition and
fails the stricter control, and both numbers are reported.

## 7. Directive gates

| Gate | Requirement | Measured | Verdict |
|---|---|---|---|
| D1 | removing Zone B drops accuracy by at least 0.10 | features removed: -0.577; identity optics: -0.043 | PASS on removal / FAIL strict |
| D2 | OOD veto >= 0.95 and false veto <= 0.02 | 1.000 and 0.017 | PASS |
| D3 | cold-start action success >= 0.40 | 0.417 (no relax: 0.458) | PASS on threshold |
| D4 | holarch factorises unseen triples, flat MLP < 0.30 | standard 0.648 vs 0.634; low-data 0.355 vs 0.410 | FAIL |

D3 needs a note. The threshold passes by 0.017, but the thermodynamic loop is
negative: the same checkpoint without the loop scores 0.458, so the loop costs
0.042. The gain over the earlier cold start comes from 20 DAgger rounds. The
noisy-observation probe confirms the direction: 0.930 with the loop against
0.940 without at sigma = 0.25. The loop lowers the wave residual, but a lower
residual does not translate into a better action.

D4 also needs a note. With the factored trilinear Stiefel readout the
holarch now beats the flat MLP on the standard split (0.648 against 0.634)
and overfits less (train 0.889 against 1.000). At the low-data split the
holarch reaches 0.355 against the MLP's 0.410, so the MLP is not below 0.30
and the gate fails. The readout narrowed the gap; it did not invert it at 25
percent training density. The canonical 300-step run reaches 0.479 on the
standard task.

## 8. Continuous learning

receipts/holarch/continual_vision.json. A trained checkpoint is adapted to a
fixed pixel permutation for 24 batches. Retention is accuracy on the original
domain after adaptation.

| Method | Shifted accuracy | Gain | Retention |
|---|---|---|---|
| no adaptation | 0.093 | 0.000 | 1.000 |
| fast weights only | 0.110 | +0.017 | 0.968 |
| optical imprint only | 0.087 | -0.007 | 0.993 |
| imprint + fast | 0.083 | -0.010 | 0.950 |
| fine-tune only | 0.643 | +0.550 | 0.862 |
| replay (mixed) | 0.677 | +0.583 | 1.000 |
| fast + replay (mixed) | 0.640 | +0.547 | 0.979 |

Replay carries the adaptation. The fast path adds nothing on this probe and
costs 0.003 retention. Retention above 1.0 for replay is measured, not a
typo: the original domain also improves.

## 9. VRAM budget

Method: counted parameters, measured autograd saved tensors for one real
batch, declared 0.75 GiB runtime reserve.

| Profile | Dim | Parameters | Training GiB | Headroom GiB | Fits |
|---|---|---|---|---|---|
| small (CI) | 256 | 189,711 | 0.766 | 31.234 | yes |
| production | 4096 | 5,757,301 | 0.914 | 31.086 | yes |
| blackwell | 8192 | 26,302,327 | 1.368 | 30.632 | yes |

No CUDA device was present in this checkout. The device column of every
receipt says cpu. The budget is an accounting result labelled DERIVED; GPU
measurement is BLOCKED on hardware.

## 10. Command line

~~~powershell
python -m henri.holarch selftest
python -m henri.holarch physics
python -m henri.holarch memory --n 2000
python -m henri.holarch budget
python -m henri.holarch train --steps 300 --name holarch_small --fresh --dagger 20
python scripts/run_holarch_ablation.py --steps 150
python scripts/run_holarch_relational.py --steps 400
python scripts/run_holarch_continual.py --steps 150
~~~

## 11. Claims and non-claims

Observed:

- Skip severed, Stiefel slot unbinding, slot mixing, and a two-condition veto
  all run end to end on CPU.
- The optical feature path is load-bearing on vision: removing it costs
  0.577. The stricter identity-optics control costs 0.043.
- The two-condition veto reaches 1.000 on uniform noise at 0.017 false veto.
- The thermodynamic loop lowers the residual monotonically but costs 0.042
  on grid success (0.417 with, 0.458 without).
- The relational task is learnable and the holarch generalises to held-out
  triples, but a flat MLP generalises better at this scale.

Not claimed:

- No GPU run, no Blackwell timing, no 32 GiB device measurement.
- No state-of-the-art result. The tasks remain small.
- The OOD result depends on outlier exposure during training. Without it the
  same veto reaches 0.668 on the same probe. That is a training-time
  dependency, not a property of the statistic alone.
- The relational gate did not pass.
- The thermodynamic loop is not shown to improve action success; it is
  measured as negative on the canonical checkpoint and on the noisy probe.

## 12. Decisions 1-4 outcomes

Decision 1, factored trilinear readout. The slot readouts are projected
onto four Stiefel bases, and the element-wise products g0*g1*g2 (K) and
[g0*g1 || g1*g2 || g0*g2] (3K) are added to the feature vector. A linear
trilinear head reads the product directly into the relational classes.
Per-slot chirp is now a learnable parameter, so a symbolic slot can drive
its dispersion toward zero.

| Split | Holarch train | Holarch test | MLP train | MLP test |
|---|---|---|---|---|
| standard, 25 percent held out | 0.889 | 0.648 | 1.000 | 0.634 |
| low data, 75 percent held out | 0.986 | 0.355 | 1.000 | 0.410 |

The trilinear readout inverts the standard comparison and halves the
overfit gap. It does not put the MLP below 0.30 on the low-data split, so
Gate D4 remains FAIL.

Decision 2, physical re-propagation under noise. Relaxation already calls
the full photonic forward pass on every step with the updated heater
feedback, so the plates re-propagate; the earlier implementation was not
open-loop in that sense. The missing piece was a task-level benefit, and it
is still missing. At sigma = 0.25 on the vision split:

| Condition | Accuracy |
|---|---|
| clean | 0.973 |
| noisy, relaxation on | 0.930 |
| noisy, relaxation off | 0.940 |

The loop costs 1.0 point on noisy input. Target-slot associative unbinding
was not implemented; adding it would need a label codebook in slot 3 and a
memory readout, which is a new mechanism rather than a repair of this one.

Decision 3, removing the exposure tax. The projector now subtracts the
latent DC and renormalises before the phase map. That recovered the
hardened-setting accuracy to 0.966 test (above the 0.957 target) while
keeping OOD at 1.000. It did not make weight 0.2 sufficient: at 0.2 the
same pipeline gives 0.95 accuracy and 0.79 OOD, because the s_max
distributions still overlap in the tail. The phase-dispersion statistic was
measured and discarded: in-distribution 0.945 against noise 0.948.

Decision 4, coordinated plasticity. Zone B optical imprint and Zone A fast
weights were run together on every adaptation batch, with no autograd.

| Arm | Shifted | Gain | Retention |
|---|---|---|---|
| no adaptation | 0.093 | 0.000 | 1.000 |
| optical imprint only | 0.087 | -0.007 | 0.993 |
| fast weights only | 0.110 | +0.017 | 0.968 |
| imprint + fast | 0.083 | -0.010 | 0.950 |
| replay (mixed) | 0.677 | +0.583 | 1.000 |

The single-plate imprint cannot undo a pixel permutation, and combining it
with the fast path is worse than either alone. Replay remains the only arm
that adapts without forgetting. The coordination hypothesis is not
supported by this measurement.

## 13. Reproduce

~~~powershell
$env:PYTHONPATH = "$PWD"
python -m pytest tests/test_holarch_*.py -q
python -m henri.holarch train --steps 300 --name holarch_small --fresh --dagger 20
python scripts/run_holarch_ablation.py --steps 150
python scripts/run_holarch_relational.py --steps 400
python scripts/run_holarch_continual.py --steps 150
~~~
