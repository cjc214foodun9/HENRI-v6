# HENRI Holarch: the three-zone wave-mechanical pipeline

Status: OBSERVED on CPU, 2026-11-02. Every number here comes from a file in
receipts/holarch/ or from a test named next to it. A test result is code-health
evidence, not a capability claim.

This revision implements Decisions A to C and the associative rule-shift
probe. Section 12 records the measured outcome of each decision, including
the ones that did not reach their target.

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
  B --> A0[post-photonic axiom]
  A0 --> V[two-condition veto]
  V --> L[output-space relax]
  L --> G[Stiefel slot unbinding]
  G --> A[Zone A swarm]
  A --> Y[task logits]
~~~

The sensor skip path is severed in phase 1: skip_alpha = 0.0 by default.
The optical feature path is the only task path for vision, language and
action.

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
parameter value. The FFT chirp mixes frequencies inside a slot; the mixer
mixes slots. The chirp dispersion is a learnable parameter per slot, so a
symbolic slot can drive its chirp toward zero.

### 2.2 Slot-specific Stiefel unbinding

For each slot s the field is projected onto a learned Stiefel basis U_s in
C^{K x d_slot}:

~~~text
g_s = Re(U_s psi_s)          U_s U_s^H = I       (QR retraction every pass)
raw features = [g_0 || g_1 || g_2 || g_3 || dark energy || 8 scalars]
~~~

K = 16 in the CI profile, 32 in production, 64 in the Blackwell profile.
The CI profile therefore keeps a 4:1 per-slot compression ratio: 4K = 64
readout values against D = 256 wave components.

The trilinear product g0*g1*g2 and the pairwise products g0*g1, g1*g2,
g0*g2 are computed in the same pass and stored after the raw block. Only the
relational path reads them; vision, language and action never see the
polynomial expansion.

### 2.3 Two-condition veto

~~~text
s_max        = max_k Re<gram_q, gram_k> / (|gram_q| |gram_k|)   (raw, no softmax)
delta_sagnac = 1 - Re<psi_out, psi_axiom_out>                   (post-photonic)
Veto(x)      = 1[s_max < tau_chance] OR 1[delta_sagnac > tau_residual]
~~~

The axiom is the post-photonic image of the softmax context bundle,
B_theta(psi_context_in), computed once with no gradient. Comparing output to
output is the Decision B fix; the previous loop pulled a post-photonic wave
toward a pre-photonic memory wave and partly erased the optical transform.

tau_chance is the 1st percentile of in-distribution s_max minus a margin;
tau_residual is the 99.5th percentile of the raw output-space residual plus a
margin, calibrated with relaxation disabled.

Outlier exposure during training separates uniform noise from digits in the
raw Gram statistic.

### 2.4 Thermodynamic relaxation in output space

When the output-space residual exceeds its threshold, Zone B runs up to four
passes. Each pass re-propagates the plates with heater feedback and then
applies a gentle egress phase correction toward the post-photonic axiom:

~~~text
T[k+1][s] = clip(T[k][s] + alpha_thermal * delta[k][s], 0, 0.999)
psi[k+1]  = normalise(psi_out[k] * exp(i * eta * angle(axiom_out * conj(psi_out[k]))))
~~~

The loop keeps the pass with the lowest residual and stops when the residual
is below tolerance or two passes fail to improve.

## 3. Zone C: holographic memory

One engram is one row: four complex64 slot blobs, a 64-value complex Gram
projection, the label, the viscoelastic weight, and the tier. TimescaleDB
serves when the DSN is reachable; WAL-mode SQLite on NVMe serves otherwise,
and the backend is recorded.

The context tensor is (context_tokens, context_dim) at every store size. The
canonical run stored 1,200 engrams with 0.93 recall@1 over 200 held-out
windows.

## 4. Zone A: the lean swarm and four tasks

Each expert is a two-layer MLP plus a rank-r delta-rule fast memory. The four
label blocks are disjoint (vision 0-9, language 10-105, action 106-109,
relational 110-117).

| Task | Input | Target | Split |
|---|---|---|---|
| vision | 8x8 digits | 10 classes | 1200 / 300 / 297 |
| language | 24-character window | next character, 96 classes | 90 / 10 percent |
| action | 5x5 view + goal vector, 79 values | BFS optimal action | 400 / 80 layouts |
| relational | (entity, action, target) triple | trilinear outcome, 8 classes | 217 / 71 triples |

The relational task holds out 25 percent of triples in the standard split and
75 percent in the low-data split, stratified by class with full primitive
coverage. The three role waves are bound by pointwise phasor product.

## 5. Measured results

Receipt: receipts/holarch/holarch_small.json
SHA-256: 054cc72f823ed583be7583289d1e669620eebf87d39051517d1de1e561e2c037

Configuration: HolarchConfig.small(), 300 steps per task, batch 32, 20 DAgger
rounds, outlier exposure weight 3.0, 175,311 parameters, TimescaleDB backend.

| Task | Metric | Validation | Test | Chance |
|---|---|---|---|---|
| Vision | accuracy | 0.933 | 0.956 | 0.100 |
| Language | next-char accuracy | 0.231 | 0.231 | 0.010 |
| Language | perplexity | 16.55 | 17.78 | 96 |
| Action | one-step accuracy | 0.763 | 0.763 | 0.250 |
| Action | closed-loop success | 0.479 | - | 0.031 random |
| Relational | held-out triple accuracy | 0.493 | 0.493 | 0.125 |
| Memory | recall@1 over 200 windows | 0.930 | - | - |

Per-task selective prediction:

| Task | Veto rate | Coverage | Selective accuracy |
|---|---|---|---|
| Vision | 0.007 | 0.993 | 0.936 |
| Language | 0.008 | 0.992 | 0.233 |
| Action | 0.013 | 0.988 | 0.760 |
| Relational | 0.014 | 0.986 | 0.486 |

At the hardened exposure weight the veto reaches 1.000 on uniform noise with
0.007 false veto.

## 6. Ablation

receipts/holarch/ablation_vision.json, 300 steps, same seed, vision only.

| Variant | Accuracy | Delta vs full |
|---|---|---|
| full | 0.953 | 0.000 |
| identity optics (plates bypassed) | 0.967 | +0.013 |
| no Zone B features | 0.283 | -0.670 |
| no routing (one expert) | 0.963 | +0.010 |
| no Zone C | 0.943 | -0.010 |
| no fast weights | 0.973 | +0.020 |

Removing the optical feature path costs 67 points. Bypassing the plates costs
nothing measurable; the identity control is 1.3 points ahead. At K = 16 the
readout is small enough that the plates are not the capacity bottleneck, and
the input-wave baseline is fully sufficient for 8x8 digits. The strict Gate
D1 control therefore fails while the removal definition passes.

## 7. Directive gates

| Gate | Requirement | Measured | Verdict |
|---|---|---|---|
| D1 | removing Zone B drops accuracy by at least 0.10 | features removed: -0.670; identity optics: +0.013 | PASS on removal / FAIL strict |
| D2 | OOD veto >= 0.95 and false veto <= 0.02 | 1.000 and 0.007 | PASS |
| D3 | cold-start action success >= 0.40 | 0.479 (no relax: 0.479) | PASS on threshold, neutral attribution |
| D4 | holarch beats MLP on both splits | standard 0.507 vs 0.634; low-data 0.419 vs 0.410 | FAIL |

D3 passes its number and is neutral: the canonical checkpoint scores 0.479
with and without the loop. The noisy-observation probe is slightly negative
(0.827 with, 0.830 without). The gain over the original cold start comes from
20 DAgger rounds.

D4 changed shape under Decision C. The regularised trilinear head now wins
the low-data split by 0.009 (0.419 against 0.410) but loses the standard
split by 0.127 (0.507 against 0.634). The new gate requires both, so it
fails.

## 8. Continuous learning

receipts/holarch/continual_vision.json. Two probes share one checkpoint.

Pixel permutation (sensory shift), 24 adaptation batches:

| Arm | Shifted | Gain | Retention |
|---|---|---|---|
| no adaptation | 0.083 | 0.000 | 1.000 |
| optical imprint only | 0.070 | -0.013 | 0.964 |
| fast weights only | 0.117 | +0.033 | 0.905 |
| imprint + fast | 0.107 | +0.023 | 0.933 |
| fine-tune only | 0.500 | +0.417 | 0.571 |
| replay (mixed) | 0.583 | +0.500 | 0.988 |
| fast + replay (mixed) | 0.597 | +0.513 | 0.992 |

Associative rule shift (labels rotate by three, pixels unchanged):

| Arm | Shifted | Gain | Retention |
|---|---|---|---|
| no adaptation | 0.010 | 0.000 | 1.000 |
| fast weights only | 0.030 | +0.020 | 0.984 |
| replay (mixed) | 0.253 | +0.243 | 0.385 |

The rule-shift probe is a contradiction, not a transfer: the same pixels map
to two different labels, so retention and adaptation trade directly. Fast
weights cannot separate them without context. Replay adapts and forgets, as
it must. The probe is reported because it was requested, with the caveat
that it does not test continual learning of compatible rules.

## 9. VRAM budget

Method: counted parameters, measured autograd saved tensors for one real
batch, declared 0.75 GiB runtime reserve.

| Profile | Dim | Parameters | Training GiB | Headroom GiB | Fits |
|---|---|---|---|---|---|
| small (CI) | 256 | 175,311 | 0.765 | 31.235 | yes |
| production | 4096 | 5,757,429 | 0.915 | 31.085 | yes |
| blackwell | 8192 | 26,302,583 | 1.370 | 30.631 | yes |

No CUDA device was present. The device column of every receipt says cpu. The
budget is an accounting result labelled DERIVED; GPU measurement is BLOCKED.

## 10. Command line

~~~powershell
python -m henri.holarch selftest
python -m henri.holarch physics
python -m henri.holarch memory --n 2000
python -m henri.holarch budget
python -m henri.holarch train --steps 300 --name holarch_small --fresh --dagger 20
python scripts/run_holarch_ablation.py --steps 300
python scripts/run_holarch_relational.py --steps 300
python scripts/run_holarch_noise.py --steps 200 --sigma 0.25
python scripts/run_holarch_continual.py --steps 150
~~~

## 11. Claims and non-claims

Observed:

- The three zones run end to end on CPU with measured accuracy, latency and
  memory behaviour.
- Removing the optical feature path costs 67 points on vision.
- The two-condition veto reaches 1.000 on uniform noise at 0.007 false veto.
- The output-space relaxation lowers the residual but is neutral on the
  canonical checkpoint and slightly negative on the noisy probe.
- The regularised trilinear head wins the low-data relational split and loses
  the standard split.

Not claimed:

- No GPU run, no Blackwell timing, no 32 GiB device measurement.
- No state-of-the-art result. The tasks remain small.
- The plates are not shown to be necessary at K = 16 on 8x8 digits; the
  identity control matches them.
- Gate D4 did not pass under the both-splits rule.
- The rule-shift probe does not demonstrate compatible continual learning.

## 12. Decisions A-C outcomes

Decision A, readout capacity. stiefel_k is 16 in the CI profile, and the
trilinear and pairwise products are stored after the raw block and read only
by the relational path. The strict identity-optics drop did not return: at
K = 16 and 300 steps the full model is 0.953 and the bypass is 0.967, a
+1.3 point advantage for the bypass. The 4:1 compression bottleneck reduced
the readout from 268 to 140 features and did lower the polynomial noise, but
the input wave remains a sufficient statistic for this task. Gate D1 passes
only on the removal definition (-0.670).

Decision B, attractor space. relax_output computes the post-photonic axiom
B_theta(context) with no gradient, measures the residual in output space,
re-propagates the plates with heater feedback on every pass, and applies an
egress phase correction with eta = 0.15. The residual is now measured
output-to-output. On the noisy probe the loop is still slightly negative:

| Condition | Accuracy |
|---|---|
| clean | 0.847 |
| noisy sigma = 0.25, relax on | 0.827 |
| noisy sigma = 0.25, relax off | 0.830 |

Decision C, relational regularisation. The head reads the gated trilinear
product plus pairwise products initialised at weight 0.1, with dropout 0.2
and L2 0.05 on the head and gate.

| Split | Holarch train | Holarch test | MLP test |
|---|---|---|---|
| standard, 25 percent held out | 0.696 | 0.507 | 0.634 |
| low data, 75 percent held out | 0.915 | 0.419 | 0.410 |

The low-data split now favours the holarch by 0.009. The standard split
favours the MLP by 0.127, so the both-splits gate fails.

Checklist item 4, associative rule shift, is the second table in section 8.
Fast weights gain 0.020 with 0.984 retention; mixed replay gains 0.243 with
0.385 retention because the two rules are contradictory on identical inputs.

## 13. Reproduce

~~~powershell
$env:PYTHONPATH = "$PWD"
python -m pytest tests/test_holarch_*.py -q
python -m henri.holarch train --steps 300 --name holarch_small --fresh --dagger 20
python scripts/run_holarch_ablation.py --steps 300
python scripts/run_holarch_relational.py --steps 300
python scripts/run_holarch_noise.py --steps 200 --sigma 0.25
python scripts/run_holarch_continual.py --steps 150
~~~
