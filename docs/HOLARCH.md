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

K = 16 in the CI profile, 32 in production, 64 in the Blackwell profile. The
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
SHA-256: 78a45fd665233f196460cdf07be5b9c86514956f6ed2d98ea6ce0029048e5e54

Configuration: HolarchConfig.small(), 300 steps per task, batch 32, 20 DAgger
rounds, outlier exposure weight 3.0, 172,031 parameters, TimescaleDB backend.

| Task | Metric | Validation | Test | Chance |
|---|---|---|---|---|
| Vision | accuracy | 0.930 | 0.943 | 0.100 |
| Language | next-char accuracy | 0.253 | 0.241 | 0.010 |
| Language | perplexity | 16.23 | 17.52 | 96 |
| Action | one-step accuracy | 0.738 | 0.738 | 0.250 |
| Action | closed-loop success | 0.542 | - | 0.031 random |
| Relational | held-out triple accuracy | 0.563 | 0.563 | 0.125 |
| Memory | recall@1 over 200 windows | 0.965 | - | - |

Per-task selective prediction:

| Task | Veto rate | Coverage | Selective accuracy |
|---|---|---|---|
| Vision | 0.013 | 0.987 | 0.936 |
| Language | 0.003 | 0.997 | 0.254 |
| Action | 0.013 | 0.988 | 0.734 |
| Relational | 0.085 | 0.916 | 0.523 |

Outlier exposure costs accuracy: with weight 0.2 the same pipeline reaches
0.957 vision accuracy but only 0.668 OOD veto; with weight 3.0 it reaches
0.930 and 1.000 OOD veto. The canonical receipt records the hardened setting
because Directive 2 gates on the OOD rate.

## 6. Ablation

receipts/holarch/ablation_vision.json, 150 steps, same seed, vision only.

| Variant | Accuracy | Delta vs full |
|---|---|---|
| full | 0.903 | 0.000 |
| identity optics (plates bypassed) | 0.677 | -0.227 |
| no Zone B features | 0.230 | -0.673 |
| no routing (one expert) | 0.770 | -0.133 |
| no Zone C | 0.880 | -0.023 |
| no fast weights | 0.850 | -0.053 |

With the skip severed, the optical transform is load-bearing. Removing the
plates and keeping the same Stiefel readout costs 22.7 points; removing the
optical features entirely costs 67.3 points. Routing and the fast path now
carry measurable signal as well.

## 7. Directive gates

| Gate | Requirement | Measured | Verdict |
|---|---|---|---|
| D1 | removing Zone B drops accuracy by at least 0.10 | features removed: -0.673; identity optics: -0.227 | PASS |
| D2 | OOD veto >= 0.95 and false veto <= 0.02 | 1.000 and 0.013 | PASS |
| D3 | cold-start action success >= 0.40 | 0.542 (no relax: 0.542) | PASS |
| D4 | holarch factorises unseen triples, flat MLP < 0.25 | holarch 0.549, MLP 0.704 | FAIL |

D3 needs a note. The threshold passes, but the thermodynamic loop is neutral:
the same checkpoint without the loop scores 0.542 as well. The gain over the
earlier cold start comes from 20 DAgger rounds. The loop lowers the wave
residual, but a lower residual does not translate into a better action on this
grid, and the receipt records both numbers.

D4 also needs a note. The holarch generalises to unseen triples (0.549
against 0.125 chance) and its train-test gap is smaller than the MLP's
(0.811 to 0.549, versus 1.000 to 0.704). It does not beat the MLP, and the
MLP is far above the 0.25 ceiling. With 217 training triples a 159k-parameter
MLP memorises the training set and still generalises; the optical readout is
not constrained to the trilinear form, so binding does not confer an
advantage at this scale. The gate is a measured failure, not a wording
problem. The canonical 300-step run reaches 0.563 on the same task; the
400-step comparison script above is the number in this row.

## 8. Continuous learning

receipts/holarch/continual_vision.json. A trained checkpoint is adapted to a
fixed pixel permutation for 24 batches. Retention is accuracy on the original
domain after adaptation.

| Method | Shifted accuracy | Gain | Retention |
|---|---|---|---|
| no adaptation | 0.107 | 0.000 | 1.000 |
| fast weights only | 0.113 | +0.007 | 0.981 |
| fine-tune only | 0.633 | +0.527 | 0.739 |
| replay (mixed) | 0.717 | +0.610 | 1.057 |
| fast + replay (mixed) | 0.707 | +0.600 | 1.053 |

Replay carries the adaptation. The fast path adds nothing on this probe and
costs 0.003 retention. Retention above 1.0 for replay is measured, not a
typo: the original domain also improves.

## 9. VRAM budget

Method: counted parameters, measured autograd saved tensors for one real
batch, declared 0.75 GiB runtime reserve.

| Profile | Dim | Parameters | Training GiB | Headroom GiB | Fits |
|---|---|---|---|---|---|
| small (CI) | 256 | 172,031 | 0.765 | 31.24 | yes |
| production | 4096 | 5,724,253 | 0.911 | 31.089 | yes |
| blackwell | 8192 | 26,170,711 | 1.359 | 30.641 | yes |

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
- The optical stage is load-bearing on vision: identity optics costs 0.227.
- The two-condition veto reaches 1.000 on uniform noise at 0.010 false veto.
- The thermodynamic loop lowers the residual monotonically but is neutral on
  grid success in this configuration (0.542 with and without).
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
  measured as neutral on the canonical checkpoint.

## 12. Reproduce

~~~powershell
$env:PYTHONPATH = "$PWD"
python -m pytest tests/test_holarch_*.py -q
python -m henri.holarch train --steps 300 --name holarch_small --fresh --dagger 20
python scripts/run_holarch_ablation.py --steps 150
python scripts/run_holarch_relational.py --steps 400
python scripts/run_holarch_continual.py --steps 150
~~~
