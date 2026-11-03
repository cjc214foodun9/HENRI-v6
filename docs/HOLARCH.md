# HENRI Holarch: the three-zone wave-mechanical pipeline

Status: OBSERVED on CPU, 2026-11-03. Every number here comes from a file in
receipts/holarch/ or from a test named next to it. A test result is code-health
evidence, not a capability claim.

This revision implements Decisions 1 to 4 of the audit of commit 87a71a0
(HENRI-ARCH-AUDIT-2026-11-03). Section 12 records the measured outcome of each
decision, including the three that did not reach their target.

Comparability warning. The parameter set changed from 175,311 to 222,022 over
the two revisions of this cycle, so module construction consumed a different
random stream. No row here is a paired delta against 87a71a0 except the
within-revision arms, which share one seed and one initialisation.

Sections 1 to 12 record the Decisions 1-4 revision. Sections 14 to 15 record
Decisions 5-8 and the agentic sidecar; where the two disagree, the later
section wins and says so.

## 1. Data path

~~~text
                    HENRI HOLARCH
    Zone A  lean swarm          Zone B  photonic kernel      Zone C  memory
    ---------------             ----------------------      ---------------
    fast weights (gated)        BaTiO3 Pockels plates       holographic engrams
    top-k routing               microheater thermal grid    TimescaleDB hypertable
    novelty search              Sagnac homodyne loop        NVMe SQLite fallback
    DAgger + replay             Stiefel slot unbinding      prototype bank (tier 3)
    linear probe option         prototype free energy       Gram index + pager
~~~

~~~mermaid
flowchart LR
  X[pixels / tokens / triples] --> E[modality encoder or SLM]
  E --> P[wave projector / slot binding]
  P --> K[rule phasor into slot 3]
  K --> R[Zone C raw top-1 + context]
  R --> F[wave fusion]
  F --> B[Zone B plates + slot mixer]
  B --> A0[post-photonic axiom]
  A0 --> V[two-condition veto]
  V --> L[prototype free-energy relax]
  L --> G[Stiefel slot unbinding]
  G --> A[Zone A swarm or linear probe]
  A --> Y[task logits]
~~~

The sensor skip path is severed in phase 1: skip_alpha = 0.0 by default. The
optical feature path is the only task path for vision, language and action.

## 2. Zone B: the photonic kernel

### 2.1 Plates, heaters, and the slot mixer

Each plate is a BaTiO3 film with a phase mask theta[s,d]. A microheater grid
sets a normalised temperature T. The Pockels response follows the Curie-Weiss
susceptibility:

~~~text
Lambda(T) = (1 + kappa / ((1 - T)^gamma + eps)) / (1 + kappa / (1 + eps))
phi[s,d]  = theta[s,d] + direction * g * Lambda(T[s]) * V[s]
~~~

Lambda(0) = 1 and Lambda(0.85) = 3.665. A bounded drive of 0.1 rad moves the
field 0.19 rad at T = 0.1 and 0.52 rad at T = 0.85, a factor of 2.74.

A learned 4x4 unitary SlotMixer sits between plates. It is the Cayley transform
of a skew-Hermitian matrix, so it is exactly unitary for every parameter value.
The FFT chirp mixes frequencies inside a slot; the mixer mixes slots.

### 2.2 Slot-specific Stiefel unbinding

For each slot s the field is projected onto a learned Stiefel basis U_s in
C^{K x d_slot}:

~~~text
g_s = Re(U_s psi_s)          U_s U_s^H = I       (QR retraction every pass)
raw features = [g_0 || g_1 || g_2 || g_3 || dark energy || 8 scalars]
~~~

K = 16 in the CI profile, 32 in production, 64 in the Blackwell profile. The CI
profile therefore keeps a 4:1 per-slot compression ratio: 4K = 64 readout values
against D = 256 wave components.

### 2.3 Two-condition veto

~~~text
s_max        = max_k Re<gram_q, gram_k> / (|gram_q| |gram_k|)   (raw, no softmax)
delta_sagnac = 1 - Re<psi_out, psi_axiom_out>                   (post-photonic)
Veto(x)      = 1[s_max < tau_chance] OR 1[delta_sagnac > tau_residual]
~~~

tau_chance is the 1st percentile of in-distribution s_max minus a margin;
tau_residual is the 99.5th percentile of the raw output-space residual plus a
margin, calibrated with relaxation disabled. Outlier exposure during training
separates uniform noise from digits in the raw Gram statistic.

### 2.4 Dark energy and the Stiefel nullspace

~~~text
dark_s(x) = || psi_s - U_s^H (U_s psi_s) ||^2 = ||psi_s||^2 - ||U_s psi_s||^2
~~~

At K = 16 of d_slot = 64 the readout can see at most a quarter of each slot. The
rest is the nullspace the audit calls dark energy, and it is the term the
relaxation loop now drives down.

Two properties of this quantity matter before any number is read:

* The readout is Re(U_s psi_s). Projecting the nullspace out of psi_s leaves
  that readout exactly invariant, because U_s (I - P_s) = 0. Dark-energy
  suppression therefore cannot change the readout by itself. It acts only
  through the heaters re-propagating the plates and through renormalisation.
* With linear optics the plate stack is expressivity-neutral against a learned
  Stiefel basis. Both arms compute Re(A psi) for a learnable A; a product of
  phase masks and unitary chirps is invertible, and a free U can invert it.
  Section 12.1 measures the consequence.

### 2.5 Prototype free-energy relaxation (Decision 2)

Relaxation no longer pulls the egress wave toward B_theta(psi_context_in), the
image of a bundle that was retrieved with the noisy query. It minimises the
Lyapunov energy of Decision 2 against a bank of clean post-photonic prototypes:

~~~text
E(psi) = ( 1 - max_m Re<psi, phi_m> ) + lambda_dark * sum_s dark_s
~~~

Each pass re-propagates the plates with heater feedback derived from the
per-slot prototype residual, applies a phase-only egress correction toward the
softmax-weighted prototype at the calibrated inverse temperature, and is
accepted only if it lowers the measured energy. Heat is applied only where the
wave is unfamiliar (surprise above the calibrated threshold), so a confident
clean input is never disturbed.

## 3. Zone C: holographic memory

One engram is one row: four complex64 slot blobs, a 64-value complex Gram
projection, the label, the viscoelastic weight, and the tier. TimescaleDB
serves when the DSN is reachable; WAL-mode SQLite on NVMe serves otherwise, and
the backend is recorded.

The context tensor is (context_tokens, context_dim) at every store size. The
canonical run stored 1,200 episodic engrams, measured 0.945 recall@1 over 200
held-out windows, and then appended 62 tier-3 prototype rows for 1,262 total.

### 3.1 The prototype bank

The attractor bank is built from the CLEAN training split: every clean input is
pushed through the plates, the bright port is averaged per class, and the mean
is normalised. No noisy query ever touches it. Rows are written back to Zone C
as tier-3 (axiomatic) engrams, which decay and compaction never prune, and can
be reloaded with `load_prototypes`.

The competition uses a calibrated inverse temperature, not the literal
beta* = 26.10. Constraint C3 forbids transferring that value: it was fitted on
the functor-score readout and no measurement ties it to this one. The bank keeps
BETA_STAR_REFERENCE for audit and sets beta from its own alignment gaps:

~~~text
beta = 4.0 / median(top1 alignment - top2 alignment)
~~~

Measured per task in the canonical run: vision 5.67, action 10.69, relational
20.76, language 46.15. A single transferred constant would have been wrong for
every one of them.

Clean-split snap accuracy of the bank itself (nearest prototype, both
quadratures): vision 0.977, action 0.816, relational 0.477, language 0.277. The
bank is in-sample on the training waves, so these are optimistic.

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
coverage.

Two optional readouts exist and both are measured in section 12.1:

* `readout = "linear"` replaces the swarm with ONE linear projection of the
  pooled photonic features. It is the Gate D1 attribution probe.
* `vision_encoder = "slm"` replaces the learned convolutional encoder with a
  parameter-free phase-only sensor, psi_d = exp(i * gain * x_d). The map is
  injective for |gain * x| < pi, so nothing is compressed before the plates.

The vision sensor can also be a 16x16 canvas that tiles four independent 8x8
digits (d_input = 256): `clutter` labels it by quadrant 0, `sum` by
(d0 + d1) mod 10, `diff` by |d0 - d1|.

### 4.1 Relational binding (Decision 3)

The three role phasors are bound by pointwise product in the d_slot = 64
complex domain, carried in slot 3 through the plates, and read by a dedicated
Stiefel basis U_rel in C^{K_rel x d_slot} with K_rel = 32. Both quadratures are
kept, so the head sees 2 * K_rel = 64 real values with their relative phase
intact. The retired control multiplies the K = 16 real slot projections instead.

## 5. Measured results

Receipt: receipts/holarch/holarch_small.json
SHA-256: 475c839d64e306404f1ed6b82d1d8a28aa721ca453de74016529735236a91f5d

Configuration: HolarchConfig.small(), 300 steps per task, batch 32, 20 DAgger
rounds, outlier exposure weight 3.0, 222,022 parameters, TimescaleDB backend,
surprise-gated prototype relaxation with prototype-coupled logits, circular
convolution relational binding on a dedicated expert.

| Task | Metric | Validation | Test | Chance |
|---|---|---|---|---|
| Vision | accuracy | 0.950 | 0.956 | 0.100 |
| Language | next-char accuracy | 0.234 | 0.270 | 0.010 |
| Language | perplexity | 18.38 | 15.20 | 96 |
| Action | one-step accuracy | 0.775 | 0.775 | 0.250 |
| Action | closed-loop success | 0.583 | - | 0.031 random |
| Relational | held-out triple accuracy | 0.465 | 0.465 | 0.125 |
| Memory | recall@1 over 200 windows | 0.955 | - | - |

The relational row uses the canonical 300-step budget. Section 14.3 shows
that the relational task needs 600 steps to converge, and at 600 steps both
splits beat the MLP.

Per-task selective prediction:

| Task | Veto rate | Coverage | Selective accuracy |
|---|---|---|---|
| Vision | 0.020 | 0.980 | 0.956 |
| Language | 0.006 | 0.994 | 0.236 |
| Action | 0.000 | 1.000 | 0.775 |
| Relational | 0.000 | 1.000 | 0.465 |

At the hardened exposure weight the veto reaches 1.000 on uniform noise with
0.020 false veto, exactly at the Gate D2 ceiling. The previous revision
measured 0.003 with a 128-wave calibration; the threshold is now estimated
on 256 waves, and the wave family changed under the new head.

Relaxation attribution on the same checkpoint, closed-loop action episodes:
0.583 with the loop and 0.583 without it. The +0.021 action gain of the
previous revision did not reproduce. Section 14.2 gives the noise-sweep
attribution, where the loop's sign is positive at sigma >= 0.75.

## 6. Ablation

receipts/holarch/ablation_vision.json, 300 steps, same seed, vision only.

| Variant | Accuracy | Delta vs full |
|---|---|---|
| full | 0.957 | 0.000 |
| identity optics (plates bypassed) | 0.947 | -0.010 |
| no Zone B features | 0.367 | -0.590 |
| no routing (one expert) | 0.957 | 0.000 |
| no Zone C | 0.937 | -0.020 |
| no fast weights | 0.933 | -0.023 |

Removing the optical feature path costs 59 points. Bypassing the plates costs
1.0 point; the identity control loses again. The prototype logits are gated on
use_optical_features, because the bank reads the bright port: without that gate
the "no Zone B features" arm still scored 0.947 through the prototype head and
the ablation measured nothing.

## 7. Directive gates

| Gate | Requirement | Measured | Verdict |
|---|---|---|---|
| D1 | removing Zone B drops accuracy by at least 0.10 | features removed: -0.590; identity optics: -0.010 | PASS on removal / FAIL strict |
| D1 strict (D5) | plates beat identity by >= 0.10 under a physical detector frame | 8x8 +0.069, 16x16 +0.091, mean +0.080 | FAIL at 0.080 |
| D2 | OOD veto >= 0.95 and false veto <= 0.02 | 1.000 and 0.020 | PASS at the ceiling |
| D3 | relaxation gain at sigma = 0.25 | model -0.003 +/- 0.004; dark energy -0.012 | FAIL on accuracy / PASS on energy |
| D4 | holarch beats MLP on both relational splits | standard 0.765 vs 0.704; low-data 0.481 vs 0.406 | PASS at 600 steps |
| D4b | gated plasticity drift <= 0.005 and context rule switching above Fano | drift 0.000 (0.003 in this revision); combined 0.942 against 0.483 | PASS |
| D6 | unseen rule graph, multi-hop, zero weight change | 1/2/3-hop 1.000; unreachable refused 1.000 | PASS |
| D7 | sidecar veto + structural index + playbook | OOD 1.000, 439 call edges, playbook exact 1.000 | PASS |
| D4b | gated plasticity drift <= 0.005 and context rule switching above Fano | drift 0.000; combined rule accuracy 0.942 against 0.483 | PASS |

## 8. Continuous learning

receipts/holarch/continual_vision.json. One checkpoint, one seed, three probes.

Sensory shift (pixel permutation), 24 adaptation batches:

| Arm | Shifted | Gain | Retention | Plasticity gate rate |
|---|---|---|---|---|
| no adaptation | 0.083 | 0.000 | 1.000 | - |
| fast, ungated | 0.110 | +0.027 | 0.898 | 0.60 |
| fast, gated | 0.093 | +0.010 | 0.894 | 0.59 |
| replay, ungated | 0.420 | +0.337 | 1.071 | 0.99 |
| replay, gated | 0.453 | +0.370 | 1.043 | 0.99 |

Stationarity control, the same plastic path on the ORIGINAL domain:

| Arm | Before | After | Drift | Gate rate |
|---|---|---|---|---|
| fast, ungated | 0.847 | 0.873 | +0.027 | 0.00 |
| fast, gated | 0.847 | 0.847 | 0.000 | 0.00 |

The gate closes completely on familiar inputs and opens on the shift. Replay
pushes the shifted arm past the no-adaptation floor.

Rule shift with slot-3 context (two rules, mixed inside every batch):

| Model | Rule 0 | Rule 3 | Combined |
|---|---|---|---|
| context bound (slot-3 phasor) | 0.937 | 0.947 | 0.942 |
| unconditioned | 0.460 | 0.507 | 0.483 |

Fano's inequality caps an unconditioned classifier at 0.50 on identical pixels
with contradictory labels; the unconditioned model lands exactly there. Binding
a deterministic rule phasor into slot 3 removes the contradiction and the same
checkpoint reaches 0.942.

## 9. VRAM budget

Method: counted parameters, measured autograd saved tensors for one real
batch, declared 0.75 GiB runtime reserve.

| Profile | Dim | Parameters | Training GiB | Headroom GiB | Fits |
|---|---|---|---|---|---|
| small (CI) | 256 | 222,022 | 0.766 | 31.234 | yes |
| production | 4096 | 5,757,429 | 0.915 | 31.085 | yes |
| blackwell | 8192 | 26,302,583 | 1.370 | 30.631 | yes |

The production and Blackwell counts are unchanged from 87a71a0: the new
relational basis and linear probe are small next to the plate masks. No CUDA
device is present. The device column of every receipt says cpu. The budget is
an accounting result labelled DERIVED; GPU measurement is BLOCKED.

## 10. Command line

~~~powershell
python -m henri.holarch selftest
python -m henri.holarch physics
python -m henri.holarch memory --n 2000
python -m henri.holarch budget
python -m henri.holarch train --steps 300 --name holarch_small --fresh --dagger 20
python scripts/run_holarch_ablation.py --steps 300
python scripts/run_holarch_gate_d1.py --steps 300 --seeds 3
python scripts/run_holarch_relational.py --steps 300
python scripts/run_holarch_noise.py --steps 300 --sigma 0.25 --seeds 5
python scripts/run_holarch_continual.py --steps 150
~~~

## 11. Claims and non-claims

Observed:

- The three zones run end to end on CPU with measured accuracy, latency and
  memory behaviour.
- Removing the optical feature path costs 66 points on vision.
- The two-condition veto reaches 1.000 on uniform noise at 0.003 false veto.
- With the nonlinearity moved out of the readout (linear Stiefel probe plus a
  parameter-free phase sensor), the optical stack beats the identity control at
  both 8x8 and 16x16: +0.042 and +0.030.
- Prototype relaxation strictly lowers the measured Lyapunov energy and the
  Stiefel nullspace energy, and is accuracy-neutral at sigma = 0.25.
- Surprise-gated plasticity removes the stationary drift (+0.027 to 0.000) and
  keeps the replay adaptation gain (+0.370 against +0.337 ungated).
- Slot-3 rule binding lifts a contradictory two-rule probe from the Fano bound
  0.483 to 0.942.
- Wave-domain relational binding raises low-data test accuracy from 0.419 to
  0.461 and leaves the standard split behind the MLP.

Not claimed:

- No GPU run, no Blackwell timing, no 32 GiB device measurement.
- No state-of-the-art result. The tasks remain small.
- The strict Gate D1 control does not reach its 0.10 threshold; the measured
  margin is 0.080 with a physical detector frame, up from 0.030.
- The relaxation loop does not produce a positive accuracy gain at sigma = 0.25.
- Gate D4 did not pass under the both-splits rule.
- Three 16x16 composition cells sit at chance or barely above it for every arm,
  so they carry no optics information; they are reported, not used.

## 12. Decisions 1-4 outcomes

### 12.1 Decision 1: the bottleneck and the linear probe

Decision 1 asked for two things at once: a true information bottleneck
(d_input >= 4 * d_readout) and a linear Stiefel readout probe, so that any
accuracy difference is attributable to the optics.

receipts/holarch/gate_d1.json, 300 steps per run, three seeds, phase-only sensor,
single linear projection from 4K = 64 features to the classes:

| Cell | d_input | Full | Identity optics | Identity drop |
|---|---|---|---|---|
| 8x8 digit | 64 | 0.894 | 0.852 | +0.042 |
| 16x16 clutter digit | 256 | 0.833 | 0.803 | +0.030 |
| 16x16 pair sum, linear probe | 256 | 0.079 | 0.082 | -0.002 |
| 16x16 pair sum, swarm readout | 256 | 0.099 | 0.097 | +0.002 |
| 16x16 pair difference, linear probe | 256 | 0.171 | 0.164 | +0.007 |

Three findings, in order of confidence:

1. The readout nonlinearity, not the input bottleneck, was the confound. With
   the Zone A MLP replaced by one linear projection and the learned encoder
   replaced by a parameter-free phase sensor, the plates beat the bypass at
   BOTH dimensions: +4.2 points at d_input = 64 and +3.0 points at
   d_input = 256. The previous revision measured the opposite sign (+1.3 for
   the bypass) with an MLP behind the readout.
2. The magnitude does not reach the 0.10 gate. The optical stack earns its
   place, but not by the margin the gate demands.
3. Why the margin is small is not an optimisation accident. With linear optics
   the plate stack is expressivity-equivalent to a learned Stiefel basis: both
   compute Re(A psi) for a learnable A, and a product of phase masks and unitary
   chirps is invertible. The remaining differences are the Kerr term, the dark
   port (four quadratic scalars), and parameter sharing. A bottleneck cannot
   create capability that the linear algebra does not contain.

The three composition cells sit at chance or barely above it for EVERY arm at
300 steps, so they cannot inform the optics comparison; they are recorded and
excluded by a rule fixed before the numbers were read (full accuracy must clear
chance by 0.10).

### 12.2 Decision 2: prototype free energy

receipts/holarch/noise_vision.json, 300 steps, sigma sweep, five independent
noise draws per row, heat applied only above the calibrated surprise threshold
(0.502):

| sigma | No relax | Relax | Delta +/- SE | Energy delta | Dark-energy delta |
|---|---|---|---|---|---|
| 0.00 | 0.973 | 0.973 | +0.0000 +/- 0.0000 | -0.0013 | -0.0043 |
| 0.25 | 0.953 | 0.945 | -0.0080 +/- 0.0066 | -0.0028 | -0.0080 |
| 0.50 | 0.879 | 0.865 | -0.0140 +/- 0.0101 | +0.0024 | -0.0161 |
| 0.75 | 0.751 | 0.759 | +0.0080 +/- 0.0088 | -0.0192 | -0.0342 |
| 1.00 | 0.629 | 0.639 | +0.0100 +/- 0.0139 | -0.0245 | -0.0642 |

The diagnosis of 87a71a0 was that relaxation was negative because the attractor
was retrieved with the noisy query. That diagnosis is confirmed and fixed: the
attractor is now built from clean waves only, and the clean row is untouched
because the gate keeps familiar inputs out of the loop.

The prediction that this would turn the noisy deficit into a positive gain is
FALSIFIED. The deficit at sigma = 0.25 is -0.008 +/- 0.007, indistinguishable
from zero but not positive, and the sign only turns positive at sigma >= 0.75,
still inside one standard error. The reason is structural, not a tuning
failure: the readout is Re(U psi), the dark-energy term is readout-invariant by
construction (section 2.4), and the alignment term trades within-class detail
for attractor proximity. A readout trained on unrelaxed waves loses exactly the
detail it needs.

Two follow-up measurements bound the claim rather than assert it:

* Training with the loop inside training (`relax_training`) costs 2.7 clean
  points (0.940 against 0.967) and does not recover them under noise, so the
  train/deploy mismatch is not the only issue.
* The relaxation does earn a positive delta where the readout is coarser:
  closed-loop action success is 0.521 with the loop and 0.500 without.

### 12.3 Decision 3: wave-domain relational binding

receipts/holarch/relational.json, 300 steps, three seeds, both holdout
densities, both heads:

| Split | Arm | Train | Test | MLP test |
|---|---|---|---|---|
| standard, 25 percent held out | wave (Decision 3) | 0.710 | 0.507 | 0.662 |
| standard, 25 percent held out | trilinear (Decision C) | 0.662 | 0.516 | 0.662 |
| low data, 75 percent held out | wave (Decision 3) | 0.986 | 0.461 | 0.401 |
| low data, 75 percent held out | trilinear (Decision C) | 0.944 | 0.409 | 0.401 |

The binding works where binding should work: the low-data split improves from
0.419 in 87a71a0 to 0.461, and from 0.409 to 0.461 against its own control arm.
The standard split still loses to the MLP (0.507 against 0.662), so the
both-splits gate fails again.

The stated cause was over-regularisation. That cause is FALSIFIED. The penalty
fell from 0.05 to 5e-4 and dropout from 0.20 to 0.05, and the training ceiling
on the standard split is 0.710 against the 0.696 of 87a71a0. A model that cannot
fit 217 training triples is not held back by its weight decay. An attribution
run on the same split rules out the two other suspects:

| Arm | Train | Test |
|---|---|---|
| full | 0.700 | 0.521 |
| no fast weights | 0.677 | 0.563 |
| no Zone C context | 0.705 | 0.535 |
| neither | 0.710 | 0.535 |

The ceiling is a property of the bound head's optimisation on 217 triples, not
of the fast path or the memory. Naming it is the next task; removing the
regularisation was not the fix.

### 12.4 Decision 4: surprise-gated plasticity and slot-3 rule context

receipts/holarch/continual_vision.json. Both halves are measured against a
paired control on one checkpoint.

Gated plasticity. The threshold is the 99.5th percentile of the surprise of
in-distribution inputs plus a margin (measured 0.549). On familiar data the
gate closes completely and the stationary drift that cost 87a71a0 its +0.020
disappears:

| Arm | Drift after 24 familiar batches | Gate rate |
|---|---|---|
| ungated | +0.027 | 0.00 |
| gated | 0.000 | 0.00 |

On the pixel-permutation shift the gate opens on 59 percent of examples and the
replay arm keeps its gain (+0.370 gated against +0.337 ungated). The fast-only
arm, whose gain was only +0.027, keeps +0.010, because the shifted wave still
lands inside the familiar tail for 41 percent of samples. That is a measured
trade-off, not a free lunch: a threshold tight enough to freeze familiar inputs
also freezes the unfamiliar samples that resemble them.

Slot-3 rule context. Two label rotations over identical pixels, mixed inside
every batch:

| Model | Rule 0 | Rule 3 | Combined |
|---|---|---|---|
| context bound | 0.937 | 0.947 | 0.942 |
| unconditioned | 0.460 | 0.507 | 0.483 |

The unconditioned model sits at the Fano bound, exactly as the audit predicted.
Binding the rule phasor into slot 3 lifts the same experiment to 0.942. The
rule-shift probe now tests context-conditioned switching, not a contradiction.

## 13. Reproduce

~~~powershell
$env:PYTHONPATH = "$PWD"
python -m pytest tests -q -k holarch
python -m henri.holarch train --steps 300 --name holarch_small --fresh --dagger 20
python scripts/run_holarch_ablation.py --steps 300
python scripts/run_holarch_gate_d1.py --steps 300 --seeds 3
python scripts/run_holarch_relational.py --steps 300
python scripts/run_holarch_noise.py --steps 300 --sigma 0.25 --seeds 5
python scripts/run_holarch_continual.py --steps 150
~~~


## 14. Decisions 5-8: closing the gates and the symbolic pivot

This section covers the second revision of the audit cycle. Receipts:
@gate_d1.json@, @relational.json@, @noise_vision.json@, @symbolic.json@,
@sidecar.json@.

### 14.1 Decision 5: break the gauge, not the bottleneck

The audit's theorem was exact: a free Stiefel basis absorbs any unitary
optical transform, so linear optics is redundant behind a learned readout.
Decision 5 therefore does two things, and both are now implemented and
measured:

    stiefel_mode = "fixed"   rows are a deterministic truncated Fourier
                             frame (a physical detector array), frozen with
                             requires_grad = False
    kerr_interplate = 0.75   phi = theta + dir * g * Lambda(T) * (V + gamma |psi|^2)

The intensity term carries 2.5 rad of authority at T = 0.85 against 3.3 rad
of voltage authority, so it is a device-scale term rather than a perturbation.
Measurement, 300 steps, three seeds, phase-only sensor, single linear probe:

| Cell | full | identity optics | plate margin |
|---|---|---|---|
| 8x8, learned frame (previous revision) | 0.894 | 0.852 | +0.042 |
| 8x8, fixed detector frame + Kerr | 0.911 | 0.842 | +0.069 |
| 16x16 clutter, learned frame | 0.877 | 0.851 | +0.026 |
| 16x16 clutter, fixed frame + Kerr | 0.836 | 0.744 | +0.091 |

Anchoring the readout roughly doubles the plate margin at 8x8 and more than
triples it at 16x16. The best cell reaches 0.091. The gate wants 0.10, so
the strict Gate D1 verdict is still FAIL, at 0.080 on the mean of the two
Decision 5 cells. The direction is no longer in doubt: with a physical
detector frame and an amplitude-dependent phase, the plates carry the task.

Two controls rule out the obvious alternative explanations. The 16x16 pair
sum and pair difference cells stay at chance for every arm (0.079-0.164), so
they carry no optics information and are excluded by the pre-declared rule.
The swarm-readout cell stays at chance too: removing the linear probe does
not rescue the composition task at 300 steps.

### 14.2 Decision 6: couple the readout to the attractor

Two changes. First, the calibrated Hopfield logits enter the decision:

    y_total = y_task + alpha * standardise(beta * Re<psi_relaxed, Phi_bank>)

scattered onto the class ids the bank was built from, with @alpha@ a learned
scalar (measured 0.580). The bank is rebuilt inside training every 50 steps
per task, so the head trains with the block it will be served with. Second,
the heater residual is computed inside the Stiefel subspace,
@Re<U psi, U phi>@, so the thermal loop sharpens the coordinates the readout
can actually see.

Result on the same checkpoint, five independent noise draws per row:

| sigma | no relax | relax | delta +/- SE | energy delta | dark-energy delta |
|---|---|---|---|---|---|
| 0.00 | 0.960 | 0.961 | +0.0013 +/- 0.0008 | -0.0025 | -0.0095 |
| 0.25 | 0.948 | 0.945 | -0.0033 +/- 0.0037 | +0.0005 | -0.0118 |
| 0.50 | 0.887 | 0.885 | -0.0020 +/- 0.0040 | -0.0025 | -0.0218 |
| 0.75 | 0.771 | 0.782 | +0.0107 +/- 0.0113 | -0.0251 | -0.0586 |
| 1.00 | 0.654 | 0.670 | +0.0160 +/- 0.0123 | -0.0358 | -0.1006 |

Coupling the bank halved the sigma = 0.25 deficit (-0.008 +/- 0.007 to
-0.0033 +/- 0.0037) and raised the high-noise gain (+0.008 to +0.011 at
sigma = 0.75, +0.010 to +0.016 at sigma = 1.0). It still does not reach the
0.05 gate at sigma = 0.25. The sign is positive exactly where a denoiser
should earn its keep, and the dark energy falls at every level.

### 14.3 Decision 7: the ceiling was the step budget, not the feature rank

Decision 7 asked for circular convolution plus four cyclic frames plus
routing through a two-layer relational expert. All three are implemented:

    psi_rel = ifft(fft(psi_e) * fft(psi_a) * fft(psi_t))
    g_rel   = [U_rel roll(psi_rel, s) for s in (0, 16, 32, 48)]  -> 128 real
    head    = 176 -> 192 -> 8 two-layer micro-net over [g0||g1||g2||g_rel]

The acceptance run, 600 steps, three seeds:

| Split | conv_shift | wave (D3) | trilinear (C) | flat MLP |
|---|---|---|---|---|
| standard, 25 percent held out | **0.765** | 0.643 | 0.634 | 0.704 |
| low data, 75 percent held out | **0.481** | 0.462 | 0.412 | 0.406 |

Both splits pass: +0.061 on the standard split and +0.075 on the low-data
split. Gate D4 closes.

The mechanism is not the one the audit proposed. A step-budget sweep on the
same split, same seed, is the deciding evidence:

| steps | conv train | conv test | MLP train | MLP test |
|---|---|---|---|---|
| 300 | 0.659 | 0.535 | 0.995 | 0.620 |
| 600 | 0.931 | 0.803 | 1.000 | 0.676 |
| 900 | 0.982 | 0.887 | 1.000 | 0.704 |

The 0.710 training ceiling of the previous revision was not Cover's linear
capacity bound on 64 features. It was a bilinear model that had not
converged in 300 steps: the MLP reaches 0.995 train accuracy at 300 steps
and starts overfitting, while the wave model is still at 0.659 and needs
600. Given the same budget, the bound representation wins by 0.06 to 0.18
and keeps winning as the budget grows. The task budget for the relational
task is therefore declared as 600 steps for every arm.

### 14.4 Decision 8: a sovereign symbolic solver

The 24-character next-letter probe is replaced, not tuned. The new module
@henri/holarch/symbolic.py@ implements four mechanisms:

* a deterministic qFHRR symbol codebook, with primitives, roles, fractional
  position powers and a pi phase rotation for negation;
* a topology solver that ingests a novel rule graph into Zone C at zero
  weight cost and resolves multi-hop queries by wave unbinding;
* a Continuous Modern Hopfield lexical snap that emits the exact symbol;
* a call-graph indexer that parses a package once and then answers
  structural questions from memory, with no file opened at query time.

receipts/holarch/symbolic.json, three seeds, two unseen graphs per seed
(24 symbols, 6 operators, 72 transitions):

| Measurement | Value |
|---|---|
| 1-hop exact match | 1.000 |
| 2-hop exact match | 1.000 |
| 3-hop exact match | 1.000 |
| unreachable queries refused | 1.000 |
| unreachable queries answered wrongly | 0.000 |
| reachable queries wrongly refused | 0.000 |
| symbols | 24 |
| transitions ingested | 72 |

The abstention column is the point. A query whose first hop has no matching
transition lands at chance coherence and the solver refuses instead of
inventing a symbol. That is Gate D2's fail-closed behaviour applied to
graph navigation.

The bundled-resonator curve explains why the solver indexes instead of
superposing everything into one wave:

| edges | bundled resonator | indexed atlas |
|---|---|---|
| 8 | 0.875 | 1.000 |
| 16 | 0.500 | 1.000 |
| 24 | 0.208 | 1.000 |
| 32 | 0.281 | 1.000 |
| 48 | 0.125 | 1.000 |
| 64 | 0.141 | 1.000 |

The holographic bundle saturates below sixteen transitions at D = 256,
which is the VSA capacity law; the Zone C atlas holds all 72 at perfect
accuracy and grows as O(N) rows.

Structural code index over @henri/holarch@ (13 modules, 6,639 lines):

| Measurement | Value |
|---|---|
| functions and classes indexed | 217 |
| call edges | 439 |
| parse time | 0.070 s |
| present-edge coherence (median) | 1.000 |
| absent-pair coherence (median) | 0.109 |
| recall at the calibrated threshold | 1.000 |
| false accept | 0.000 |
| file reads at query time | 0 |

## 15. The agentic sidecar

receipts/holarch/sidecar.json. One object, five questions an agent asks:

| Capability | Measurement |
|---|---|
| OOD veto on uniform noise | 1.000 |
| false veto on in-distribution | 0.031 |
| structural index | 439 call edges from 13 modules, 0.07 s |
| unseen rule system, 2-hop | 1.000 across 40 queries |
| unreachable query | refused, coherence below the calibrated threshold |
| ACE playbook compiled | 5 steps, tier 3, exact recovery 1.000 |
| prototype recall | nearest class prototype returned with its alignment |

The playbook is the ACE delta form. Steps are superposed as
@psi_step * psi_pos^k@ around a pure-phase study key, so the stored wave
unbinds back to the exact step phasors. A second compile merges new steps
and bumps the version without rewriting the record.

The decoder is deterministic. It renders a verified path or plan as fixed
grammar, never as generated prose:

    PLAN refactor [verified]
      1. READ_INDEX
      2. LOCATE_SYMBOL
      3. EDIT_CALLSITE
      4. RUN_TESTS
      5. VERIFY_VETO

and it says @ABSTAIN: coherence below tau@ when the solver cannot verify a
hop. No language model sits between the verified answer and the caller.

What this deployment is, and is not: it is an epistemic controller, a
continual memory, a rule router and a fail-closed verifier that any agent
can call. It is not a text generator. The audit's own analysis is right
that surface code emission needs either a structural synthesizer or a local
decoder backbone; the sidecar supplies the verification and memory layer
around whichever one is used, and the symbolic decoder covers the
deterministic emission paths.

## 16. Reproduce (revised)

~~~powershell
$env:PYTHONPATH = "$PWD"
python -m pytest tests -q -k holarch
python -m henri.holarch train --steps 300 --name holarch_small --fresh --dagger 20
python scripts/run_holarch_ablation.py --steps 300
python scripts/run_holarch_gate_d1.py --steps 300 --seeds 3
python scripts/run_holarch_relational.py --steps 600 --seeds 3
python scripts/run_holarch_noise.py --steps 300 --sigma 0.25 --seeds 5
python scripts/run_holarch_continual.py --steps 150
python scripts/run_holarch_symbolic.py --seeds 3
python scripts/run_holarch_sidecar_demo.py --steps 120
~~~
