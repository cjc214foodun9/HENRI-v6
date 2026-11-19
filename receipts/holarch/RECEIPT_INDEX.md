# Holarch receipt index

Every JSON receipt in this directory is named here.
The documentation-consistency check reads this file together with docs/HOLARCH.md,
so a receipt is documented exactly once.

~~~text
aaii_v5_eval.json  ablation_vision.json  agent_eval.json  budget.json  c_cycle.json
code_lang_eval.json  code_lang_l1.json  code_lang_l2.json  code_lang_l3.json  continual_vision.json
d_cycle.json  e_cycle.json  f1_namespaces.json  f2_rotor.json  f3_transport.json
f4_impact.json  f_cycle.json  g1_wave_lm.json  g2_parametric_synth.json  g3_depth2_lookahead.json
g4_persistence.json  g_cycle.json  gate_d1.json  gate_d1_k12.json  gate_d1_k16.json
gate_d1_k32.json  gate_d1_k8_s600.json  gate_d1_l3.json  gate_d1_l4.json  gate_d1_probe.json
h1_entropy_gated_lm.json  h2_multiplan_synth.json  h3_api_surface.json  h4_delta_index.json  h_cycle.json
holarch_medium.json  holarch_production.json  holarch_production_c050.json  holarch_production_c065.json  holarch_production_k24_c050.json
holarch_production_k24_c050_d40.json  holarch_production_k24_c065.json  holarch_small.json  i1_vocab_lock.json  i2_dag_synthesis.json
i3_horizon_traps.json  i4_agent_scale.json  i_cycle.json  j1_cartridge.json  j1_cleanroom_planner.json
j2_elastic_vocab.json  j3_diverse_dag.json  j4_api_v1.json  j_cycle.json  j_eval.json
k1_full_cartridge.json  k2_boundary_discovery.json  k3_unified_vla_and_dag.json  k4_rebase_and_concurrency.json  k_cycle.json
k_eval.json  m1_online_repulsor_vla.json  m2_low_latency_concurrency.json  m3_full_reference_cartridge.json  m4_guard_ast_and_eval.json
m_cycle.json  m_eval.json  noise_vision.json  o1_swat_telepathy.json  o2_api_repulsor_and_linearizable_lanes.json
o3_live_tree_language.json  o4_holonic_and_gauntlet_128.json  o_cycle.json  o_eval.json  p1_live_corpus_language_closure.json
p2_beyond_context_telepathy_api.json  p3_end_to_end_telepathic_vla.json  p4_henri_holon_and_gauntlet_140.json  p4_henri_holon_plugin.json  p_cycle.json
p_eval.json  q1_query_agnostic_telepathy_and_fact_veto.json  q2_causal_trimodal_telepathic_vla.json  q3_disjoint_and_full_decile_language.json  q4_warm_holon_and_aaii_v5.json
q_cycle.json  q_eval.json  r1_unstructured_telepathy_and_wave_fact_veto.json  r2_balanced_causal_vla_matrix.json  r3_static_and_streaming_full_decile_language.json
r4_real_corpus_and_coprocessor_aaii_v5.json  r_cycle.json  r_eval.json  relational.json  s1_language_closure.json
s2_latent_wave_turing_machine.json  s3_hierarchical_charts_and_sandboxed_browser.json  s4_four_pillar_and_176_query_gauntlet.json  s_cycle.json  s_eval.json
sidecar.json  symbolic.json  t1_hygiene_and_determinism.json  t2_centered_charts_mmlu.json  t3_inductive_wave_turing.json
t4_coprocessor_and_substrate.json  t_cycle.json  vocab_lock.json
~~~
t2_riemannian_chart_atlas.json  t3_foveated_bidirectional_turing.json  t4_sovereign_ccg_and_repo_sheaf.json

## Cycle U receipts

u1_fast_closure_and_hygiene.json  u2_condensed_riemannian_atlas.json  u3_inductive_turing_audit.json
u5_live_coprocessor_aaii_harness.json  u_cycle.json  closure_manifest.json


## Cycle V and W receipts

v1_factored_turing_closure.json  v2_fast_gauntlet.json  v3_live_aaii_v5_arms.json
v4_live_holon_plugin.json  v5_full_closure_ledger.json  v_cycle.json
w1_pass0_gauntlet_240.json  w2_pytest_and_split_hygiene.json
w3_complete_one_cell_turing.json  w4_live_four_arm_aaii_v5.json
w5_wave_egress_and_full_closure.json  w_cycle.json

## Cycle X receipts

x1_live_four_arm_aaii_v5.json  x1_stage_four_arm.json
x2_sovereign_wave_egress.json  x2_stage_wave_egress.json
x3_twelve_of_twelve_turing.json  x3_stage_turing_closure.json
x4_gauntlet_256_queries.json  x5_full_ledger_reconciliation.json  x_cycle.json

## Cycle Y receipts

y1_photonic_wave_egress.json  y2_cold_start_synthesis.json
y3_unified_wave_induction.json
y4_scaled_aaii_v5.json  y5_hardware_and_ledger_closure.json
y1_stage_photonic_wave_egress.json  y2_stage_cold_synthesis.json
y3_stage_unified_induction.json  y4_stage_scaled_aaii_v5.json
y_jev_mathematical_challenges.json  y_cycle.json  agent_eval.json

## Cycle Z receipts

z1_d2048_pointer_wave_egress.json  z2_external_cold_synthesis.json
z3_spatial_arc_induction.json  z4_live_aaii_and_api_stress.json
z5_cuda_benchmark.json  z_jev_mathematical_challenges.json
z1_stage_pointer_egress.json  z2_stage_external_synthesis.json
z3_stage_spatial_induction.json  z4_stage_live_aaii_and_stress.json
z5_hardware_ledger_closure.json  z_cycle.json
pinned_lang_split.json

## Cycle AA (per-token phase egress, prose grounding, external ARC, cascade safety)

| receipt | stage | content |
| :--- | :--- | :--- |
| pinned_corpus_snapshot.json | AA1 | frozen bytes of the 38 package files and the 758,656 / 84,086 token split |
| aa1_per_token_phase_egress.json | AA1 | token-indexed phase table, gated long-range channels, held-out metrics and ablations |
| aa2_external_prose_and_loop_synthesis.json | AA2 | prose/doctest extractor, stateful schemas, MBPP and HumanEval slices, trap and transfer results |
| aa3_external_arc_agi_audit.json | AA3 | 48-task official ARC-AGI slice, codec audit, pass@1, precision, Z3 regression |
| aa4_complete_cascade_and_aaii80.json | AA4 | five-arm live battery, cascade escalation, spend guard, HTTP/SSE stress, gauntlet |
| aa5_cuda_benchmark.json | AA5 | CUDA probe and D = 2048 / D = 4096 benchmark, or the honest blocked state |
| aa5_hardware_and_ledger_closure.json | AA5 | PnP provenance, JEV pointer, sidecar parity, pytest, credential scan |
| aa_jev_mathematical_challenges.json | AA5 | the two JEV Tier-1 challenges and their verdicts |
| aa_cycle.json | AA | aggregate ledger that supersedes y_cycle.json and z_cycle.json |

## Cycle AB (object-centric ARC, context phase, sheaf retrieval, portable Blackwell backend)

| receipt | stage | content |
| :--- | :--- | :--- |
| ab1_contextual_phase_egress.json | AB1 | frozen-snapshot held-out bpc and top-1, static phase marginal, rank-8 steering with the validation veto, dual 24/256-token copy recall |
| ab2_sheaf_retrieval_and_loop_synthesis.json | AB2 | in-retrieval domain/codomain sheaf filter, boundary consensus gate, extended loop schemas, MBPP and HumanEval slices, Arm B transfer |
| ab3_object_centric_external_arc.json | AB3 | object scene graph, canvas dimension induction, strict Leave-One-Out inducer, 48-task official ARC audit |
| ab5_cuda_benchmark.json | AB5 | dual-backend probe, PnP provenance and the honest BLOCKED_NO_PCIE_NVIDIA_DEVICE state |
| ab_jev_mathematical_challenges.json | AB5 | the two JEV Tier-1 challenges and their verdicts |
| ab_cycle.json | AB | aggregate ledger that supersedes aa_cycle.json |

## Cycle AC (orthogonal exact-ID induction, MDL-sheaf selection, multi-scale ARC, 80-task suite)

| receipt | stage | content |
| :--- | :--- | :--- |
| ac1_kronecker_induction_egress.json | AC1 | exact-identity phasor codebook audit, exact-ID causal induction, IDF dual copy caches, mode-phase torque, held-out bpc/top-1, phase and channel marginals, copy recall |
| ac2_mdl_sheaf_synthesis.json | AC2 | IDF intent coverage, no premature single-plan exit, joint MDL-sheaf posterior over verified candidates, MBPP and HumanEval slices, Arm B transfer |
| ac3_multiscale_external_arc.json | AC3 | four universal morphism families, wave-invariant router, residual depth-2 Leave-One-Out composition, 48-task official ARC audit |
| ac4_cascade_and_aaii80.json | AC4 | 80-task five-arm battery, arm0 standalone AC2/AC3 engines, arm4 cascade, spend guard, HTTP/SSE stress, 304-query gauntlet |
| ac5_hardware_and_ledger_closure.json | AC5 | authorization flag, PnP provenance, Vast.ai dry-run catalog, credential scan |
| ac_jev_mathematical_challenges.json | AC5 | the two Cycle AC JEV Tier-1 challenges and their verdicts |
| ac_cycle.json | AC | aggregate ledger that supersedes ab_cycle.json |
