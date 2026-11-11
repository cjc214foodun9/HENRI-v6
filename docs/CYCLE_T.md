# Cycle T engineering detail

This file carries the long-form sections of docs/HOLARCH.md Section 35.
It is not part of the language corpus; the corpus reads docs/HOLARCH.md.

## 35.1 Decision T1: one document, a stratified batch, and a clock

The Cycle S mid-session truncation left two copies of Sections 26-28. The merge
keeps the full Cycles J, K and M text and deletes the reconstructed summaries.
Every receipt filename that only the deleted block named is now in the receipt
index at 35.6, so the documentation-consistency check still resolves all of
them. The header sequence is 1..35 with no duplicate and no gap.

q2_causal_vla drew the identity regime twelve times because it sampled four
validation layouts plus the eight traps and never stratified. Cycle T probes the
deterministic layout stream with the planner itself: the validation layouts,
then the declared traps, then draws from a freshly seeded grid generator. A
layout is accepted only when the planner reaches the goal, and the batch is
filled round-robin across the three measured regimes. The twelve episodes are
0,1,2,0,1,2,0,1,2,0,1,2. Every counterfactual donor is from a different class,
so all twelve swaps change the operator and execute. No weight and no
ActionRegimeReadout parameter was changed; the episode now consults the fitted
head only when the caller asks for it, which is the flag R2 already sets.

The s3 stopwatch is replaced by two numbers that a clock cannot move. The raw
charted and flat p50 values stay in the receipt (1.115 ms and 1.855 ms in this
run). t_ref_ms is the median of 25 warm 256x1024 complex64 matrix-vector
products: 0.094 ms here. The gate is flat/charted at least 1.40x, evaluated
candidates at most 0.40 of N, and (charted at most 0.45 ms OR charted/t_ref at
most 20). Twenty is the archived cool-CPU ratio: 0.4317 ms charted against the
0.0241 ms pulse implied by the same receipt's 3.56x flat slowdown, i.e. 17.9,
plus a 12 percent scheduler margin. A throttled host scales both terms and the
ratio stays inside the envelope.

## 35.2 Decision T2: a centered, whitened atlas on real MMLU

ChartIndex routed on raw centroid coherence. On 256-token MMLU questions the
Bochner sequence waves share a mean direction, so every engram sits in a narrow
spherical cap and the centroid score is dominated by the shared projection. The
CenteredChartIndex removes that direction and whitens the residual:

    mu_gram = mean(G)
    W_zca   = V diag(1/sqrt(lambda + eps)) V^H,  C = (G - mu)^H (G - mu) / N
    z_i     = normalize(W_zca (g_i - mu))

Charts are deterministic k-means clusters in z-space with K = ceil(N / capacity),
not a greedy capacity fill: a greedy fill can leave a row in a chart whose
centroid is not its nearest one, and a query near that row then routes to the
wrong chart. The chart count is 40 for the 2,500-row namespace. A query is
transformed by the same mu and W, scores the K centroids, and probes the best
chart. When the top-1 / top-2 centroid margin falls below tau_boundary (the
10th percentile of the stored margin distribution) the probe widens to at most
six charts. The union is then rescored by the exact full-wave Hermitian inner
product; no wave is altered.

The store is the real MMLU file: ten subjects at 250 questions each, 256 tokens
per question, namespace t2_mmlu, plus the 1,349-row seed cartridge in the
default namespace. The exact-match arm is 100 in-domain questions read back; the
charted top-1 is the stored engram in 100/100. Pillar A's digest-bound read rule
returns grounded 1.0000 with charts, OOD abstention 1.0000 and hallucination
0.0000. The charted read scans 0.049 of the namespace on average (maximum 0.181
on an expanded query) and is 1.48x faster at p50 (0.395 ms vs 0.586 ms) with a
resident 4,096-wave cache. Charted routing becomes the automatic path when a
synchronized centered index exists and the namespace holds at least 1,024 rows.

## 35.3 Decision T3: induce the table, not execute a table

WaveProgramInducer receives only (tape_in, tape_out) pairs. It has no rule
argument. On an unmapped (q, s) key it scores candidate (q', s', dir) bindings by
the phase-coherence contribution of the written symbol to psi_target in Slot 2,
with the target-cell match weighted as an attractor, and breaks ties by
description length: prefer an existing state, prefer the no-write symbol when
the cell already matches, prefer halting when the tape already equals the
target. A branch that halts with the wrong tape, exceeds the step horizon, or
repeats a configuration is written to Zone C as a REPULSOR axiom against that
binding and the search backtracks. A table that reproduces every training pair
is crystallized as a Tier-3 engram with its rule JSON in the metadata.

Eight of the ten declared tasks induce and score 1.0000 exact on held-out tapes
of length 6..12: bit flip, unary increment, unary decrement, strip leading
zeros, find first 1, append marker, two-symbol replace, unary add-one-and-halt.
Binary increment and parity-accept-reject exhaust the 40,000-node budget without
a table that both fits the four short examples and generalizes; they are
reported as ABSTAIN, not as passes. The four contradictory specifications (one
tape_in, two different tape_out values) return exactly
ABSTAIN: unsatisfiable specification with zero search nodes. The path is exposed
as POST /v1/holarch/turing with mode="induce", the MCP tool holarch_turing, and
the service method; all three return INDUCED on the bit-flip examples.

## 35.4 Decision T4: a loop with a sandbox, a gate and a repulsor

HolarchCoProcessor wraps the sidecar in the Stirrup contract. Each turn is one
of browse, code_exec, verify, finish, or abandon_task_finish. Browse resolves a
path through SandboxedCorpusBrowser before opening it. Code execution parses
with ASTGate, refuses an unknown callee or a forbidden name, then runs in a
namespace with a fixed module set and a fixed builtin set. A command that fails
once is registered as a REPULSOR key; the identical command is refused before
the interpreter runs, so a loop cannot form. finish is refused while a required
evidence path has not been read. abandon_task_finish is refused when the cited
path WAS read, except for a file the run proved corrupted. Every turn appends
one SHA-256 link whose parent is the previous link.

Sixteen scenarios run: twelve solvable repository and data-analysis tasks of
three to six turns (CSV sums, maxima and means, JSON config, regex extraction,
function counting, a two-file join, and one fault-recovery task), and four
impossible or corrupted-source tasks. All twelve deliverables match the
independently computed expected object; all four impossible tasks abandon with
the exact path (audit.csv, ledger.md, forecast.json, broken.csv). One identical
failing command is blocked by the REPULSOR registry; no failing command is
executed twice; no impossible task emits a deliverable. The chain is 53/53
contiguous links.

The hardware probe still reads BLOCKED. nvidia-smi is on PATH and both nvcuda
DLLs exist, but nvidia-smi lists zero devices and torch.cuda.is_available() is
False, so no budget run and no GPU timing is claimed. The receipt records the
CPU verdict, the driver evidence and WSL2 absence.
