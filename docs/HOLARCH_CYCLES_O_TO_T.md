# Project HENRI: Cycles O to T reconstructed from sealed receipts

This archive restores the long-form record of Cycles O, P, Q, R, S and T.
It is deliberately **outside** `docs/HOLARCH.md`: the language corpus reads
that file, so the R3 streaming bits-per-character stay at their sealed value.
Every table cell is read from one sealed JSON receipt, and every receipt is
identified by the first 16 hex digits of its SHA-256.

## Cycle O: Sovereign telepathy, SWAT prefill transfer, and the holonic API

| Stage | Receipt | SHA-256 (16) | Passed | Wall (s) |
| :--- | :--- | :--- | :---: | ---: |
| o1 | `o1_swat_telepathy.json` | `7081c701077ff43d` | true | 82.56 |
| o2 | `o2_api_repulsor_and_linearizable_lanes.json` | `2167709be900acf6` | true | 76.02 |
| o3 | `o3_live_tree_language.json` | `4fd12abe75b863e2` | true | 337.97 |
| o4 | `o4_holonic_and_gauntlet_128.json` | `0dc394595f77b19d` | true | 648.74 |
| eval | `o_eval.json` | `484f3ca8bfd5dddf` | true | 614.78 |

Cycle aggregate `passed`: **true**.

Measured gates: `o1a_24_queries`=true, `o1a_swat_at_least_0_55`=true, `o1a_delta_at_least_15_points`=true, `o1a_ttft_at_least_4x`=true, `o1a_zero_receiver_prefill_tokens`=true, `o1b_vla_12_completed`=true, `o1b_vla_goals_12`=true, `o1b_cross_modal_digit_12`=true, `o1c_corrupted_16`=true, `o1c_veto_16_of_16`=true, `o1c_zero_hallucination`=true, `o2_traps_8`=true, `o2_online_escape_8`=true, `o2_reactive_control_0`=true, `o2_episodes_online_at_least_44`=true, `o2_episodes_baseline_38`=true, `o2_lanes_8_fresh`=true, `o2_bench_p50_at_most_120`=true, `o2_bench_ratio_at_least_2_5`=true, `o3_tier1_shifts_zero`=true, `o3_pinned_locked_top1_0_55`=true, `o3_pinned_locked_bpc_1_30`=true, `o3_elastic_top1_0_55`=true, `o3_elastic_bpc_1_31`=true, `o3_live_l3_recorded`=true, `o3_char96_recorded`=true, `o4_prior_84`=true, `o4_discovery_4`=true, `o4_traps_refused_4`=true, `o4_synthesis_92`=true, `o4_queries_128`=true, `o4_gauntlet_passed`=true, `o4_exit_zero`=true, `o4_zero_reads`=true, `o4_reachable_1`=true, `o4_hallucination_0`=true, `queries_128`=true, `gates_passed`=true, `exit_zero`=true.

### `o1_swat_telepathy.json`

Stage `O1`, evidence `OBSERVED`, aggregate passed **true**, SHA-256 `7081c701077ff43d`.

| Measured field | Value |
| :--- | :--- |
| `budgets.8.swat_correct` | 24 |
| `budgets.8.t2t_correct` | 0 |
| `budgets.8.queries` | 24 |
| `budgets.8.swat_accuracy` | 1 |
| `budgets.8.t2t_accuracy` | 0 |
| `budgets.8.delta_points` | 100 |
| `budgets.8.attention_mass` | 1 |
| `budgets.8.receiver_prefill_tokens` | 0 |
| `budgets.16.swat_correct` | 24 |
| `budgets.16.t2t_correct` | 0 |
| `budgets.16.queries` | 24 |
| `budgets.16.swat_accuracy` | 1 |
| `budgets.16.t2t_accuracy` | 0 |
| `budgets.16.delta_points` | 100 |

### `o2_api_repulsor_and_linearizable_lanes.json`

Stage `O2`, evidence `OBSERVED`, aggregate passed **true**, SHA-256 `2167709be900acf6`.

| Measured field | Value |
| :--- | :--- |
| `api_traps.selected` | 8 |
| `api_traps.online_escape` | 8 |
| `api_traps.reactive_control` | 0 |
| `api_traps.deployed_depth2` | 8 |
| `episodes.online_wins` | 45 |
| `episodes.baseline_wins` | 38 |
| `episodes.delta` | 7 |
| `episodes.online_rate` | 0.9375 |
| `lanes.count` | 8 |
| `lanes.fresh` | 8 |
| `benchmark_128.cache_enabled` | false |
| `benchmark_128.threads` | 16 |
| `benchmark_128.lanes` | 8 |
| `benchmark_128.requests` | 128 |

### `o3_live_tree_language.json`

Stage `O3`, evidence `OBSERVED`, aggregate passed **true**, SHA-256 `4fd12abe75b863e2`.

| Measured field | Value |
| :--- | :--- |
| `tier1.locked_identifiers` | 965 |
| `tier1.shifts` | 0 |
| `tier1.elastic_vocabulary` | 4096 |
| `tier1.elastic_cap` | 4096 |
| `tier1.tier2_ids` | 2048 |
| `pinned_locked.top1` | 0.5763 |
| `pinned_locked.bits_per_char` | 1.10503 |
| `pinned_locked.beta` | 0.02 |
| `live_tree_elastic.top1` | 0.567059 |
| `live_tree_elastic.bits_per_char` | 1.27356 |
| `live_tree_elastic.beta` | 0.05 |
| `live_tree_elastic.corpus.files` | 21 |
| `live_corpus.top1` | 0.540504 |
| `live_corpus.bits_per_char` | 1.57548 |

### `o4_holonic_and_gauntlet_128.json`

Stage `O4`, evidence `OBSERVED`, aggregate passed **true**, SHA-256 `0dc394595f77b19d`.

| Measured field | Value |
| :--- | :--- |
| `synthesis.prior_total` | 84 |
| `synthesis.prior_passed` | 84 |
| `synthesis.holonic_accepted` | 4 |
| `synthesis.holonic_traps_refused` | 4 |
| `synthesis.total` | 92 |
| `synthesis.passed` | 92 |
| `gauntlet.total_queries` | 176 |
| `gauntlet.returncode` | 0 |
| `gauntlet.gate.mcp_zero_query_reads` | true |
| `gauntlet.gate.mcp_reachable_at_least_0_98` | true |
| `gauntlet.gate.mcp_hallucination_zero` | true |
| `gauntlet.gate.mcp_playbook_exact` | true |
| `gauntlet.gate.composite_mcp_at_least_0_90` | true |
| `gauntlet.gate.composite_traps_refused` | true |

### Explicit non-claims

* The SWAT arm is measured on a CPU digital twin, not on optical hardware.
* The 128-query gauntlet is a symbolic self-audit, not an external benchmark.

## Cycle P: Live-corpus language closure, beyond-context telepathy, and the plugin

| Stage | Receipt | SHA-256 (16) | Passed | Wall (s) |
| :--- | :--- | :--- | :---: | ---: |
| p1 | `p1_live_corpus_language_closure.json` | `e42f59636ce961cb` | true | 286.97 |
| p2 | `p2_beyond_context_telepathy_api.json` | `8e983d2ef6f16d6c` | true | 28.82 |
| p3 | `p3_end_to_end_telepathic_vla.json` | `504d9b5d3d5b0ce3` | true | 92.85 |
| p4 | `p4_henri_holon_and_gauntlet_140.json` | `2b81f985ab81ffd0` | true | 648.84 |
| eval | `p_eval.json` | `7e35f51ab1e6b9cd` | true | 637.45 |

Cycle aggregate `passed`: **true**.

Measured gates: `p1_tier1_shifts_zero`=true, `p1_pinned_locked_top1_ge_0_5600`=true, `p1_pinned_locked_bpc_le_1_3000`=true, `p1_pinned_elastic_top1_ge_0_5650`=true, `p1_pinned_elastic_bpc_le_1_2700`=true, `p1_live_elastic_top1_ge_0_5550`=true, `p1_live_elastic_bpc_le_1_3100`=true, `p1_live_corpus_target_met`=true, `p1_full_split_fit`=true, `p1_no_fit_cap`=true, `p1_tail_anchored_window`=true, `p1_historical_arms_recorded`=true, `p2_24_queries`=true, `p2_swat_24_of_24`=true, `p2_truncation_le_0_25`=true, `p2_bm25_le_0_55`=true, `p2_ttft_at_least_5x`=true, `p2_zero_receiver_prefill`=true, `p2_veto_4_of_4`=true, `p2_api_rest`=true, `p3_episodes_12`=true, `p3_goals_12`=true, `p3_code_executed_12`=true, `p3_zero_text_glue`=true, `p3_zero_receiver_prefill`=true, `p3_chain_valid_12`=true, `p3_corrupted_4_vetoed`=true, `p3_zero_hallucination`=true, `p4_battery_84`=true, `p4_prior_8_no_drift`=true, `p4_new_8`=true, `p4_synthesis_100`=true, `p4_stationary_drift_zero`=true, `p4_residual_state_zero`=true, `p4_plugin_seeds_4`=true, `p4_plugin_accepted_4`=true, `p4_plugin_refused_4`=true, `p4_plugin_residual_zero`=true, `p4_plugin_exit_zero`=true, `p4_queries_140`=true, `p4_probe_categories_20`=true, `p4_gauntlet_passed`=true, `p4_exit_zero`=true, `p4_zero_reads`=true, `p4_reachable_1`=true, `p4_hallucination_0`=true, `queries_140`=true, `gates_passed`=true, `exit_zero`=true, `reachable_1`=true, `hallucination_0`=true, `zero_reads`=true, `chain_valid`=true.

### `p1_live_corpus_language_closure.json`

Stage `P1`, evidence `OBSERVED`, aggregate passed **true**, SHA-256 `e42f59636ce961cb`.

| Measured field | Value |
| :--- | :--- |
| `tier1.locked_identifiers` | 965 |
| `tier1.shifts` | 0 |
| `tier1.elastic_vocabulary` | 4096 |
| `tier1.elastic_cap` | 4096 |
| `tier1.tier2_ids` | 2048 |
| `arms.pinned_locked.split_byte_offset` | 307149 |
| `arms.pinned_locked.top1` | 0.5763 |
| `arms.pinned_locked.bits_per_char` | 1.10503 |
| `arms.pinned_locked.perplexity` | 7.10388 |
| `arms.pinned_locked.base_top1` | 0.5763 |
| `arms.pinned_locked.base_bits_per_char` | 1.10503 |
| `arms.pinned_locked.base_perplexity` | 7.10388 |
| `arms.pinned_locked.beta` | 0.02 |
| `arms.pinned_locked.beta_gate.beta_low` | 0.01 |

### `p2_beyond_context_telepathy_api.json`

Stage `P2`, evidence `OBSERVED`, aggregate passed **true**, SHA-256 `8e983d2ef6f16d6c`.

| Measured field | Value |
| :--- | :--- |
| `dataset.content_chunks` | 4 |
| `dataset.corrupted_chunks` | 4 |
| `dataset.facts_per_content_chunk` | 256 |
| `dataset.total_facts` | 1004 |
| `dataset.receiver_window_facts` | 256 |
| `dataset.context_over_window` | 4 |
| `dataset.queries` | 24 |
| `dataset.hops` | 3 |
| `dataset.budget_u` | 32 |
| `dataset.per_chunk_u` | 8 |
| `accuracy.swat` | 1 |
| `accuracy.truncation` | 0 |
| `accuracy.bm25_t2t` | 0 |
| `accuracy.swat_correct` | 24 |

### `p3_end_to_end_telepathic_vla.json`

Stage `P3`, evidence `OBSERVED`, aggregate passed **true**, SHA-256 `504d9b5d3d5b0ce3`.

| Measured field | Value |
| :--- | :--- |
| `episodes.count` | 12 |
| `episodes.goals_reached` | 12 |
| `episodes.code_executed` | 12 |
| `episodes.text_tokens_exchanged` | 0 |
| `episodes.receiver_prefill_tokens` | 0 |
| `episodes.receipt_chain_valid` | 12 |
| `adversarial.episodes` | 4 |
| `adversarial.vetoed_and_refused` | 4 |
| `adversarial.hallucinated_sources` | 0 |
| `adversarial.gate_calibration.rows` | 15 |
| `adversarial.gate_calibration.tau_chance` | 0.268298 |
| `adversarial.gate_calibration.tau_residual` | 1.04742 |
| `adversarial.gate_calibration.s_max_min` | 0.315644 |
| `adversarial.gate_calibration.s_max_median` | 0.817998 |

### `p4_henri_holon_plugin.json`

Stage ``, evidence `OBSERVED`, aggregate passed **false**, SHA-256 `e56edc1dfa5eff49`.

| Measured field | Value |
| :--- | :--- |
| `harness.seeds` | 4 |
| `harness.accepted` | 4 |
| `harness.traps` | 4 |
| `harness.refused` | 4 |
| `harness.residual_state` | 0 |
| `harness.candidates` | 60 |
| `harness.plan_store_size` | 4 |
| `harness.engine.vocabulary_size` | 2048 |
| `harness.engine.tokens` | 688596 |
| `harness.engine.files` | 25 |
| `harness.engine.vocab_locked` | true |
| `harness.passed` | true |

### Explicit non-claims

* Telepathy is measured on layouts drawn from the same task family.
* The plugin result is a dry-run of the discovery engine, not a deployment.

## Cycle Q: Query-agnostic telepathy, causal Action-KV operator, disjoint splits

| Stage | Receipt | SHA-256 (16) | Passed | Wall (s) |
| :--- | :--- | :--- | :---: | ---: |
| q2 | `q2_causal_trimodal_telepathic_vla.json` | `9c83dee6652fe8b1` | true | 157.02 |

Cycle aggregate `passed`: **true**.

Measured gates: `q2_goals_12`=true, `q2_executed_12`=true, `q2_zero_text_glue`=true, `q2_zero_receiver_prefill`=true, `q2_regimes_3`=true, `q2_counterfactual_12`=true, `q2_vision_veto_4`=true, `q2_action_veto_4`=true, `q2_zero_hallucination`=true.

### `q1_query_agnostic_telepathy_and_fact_veto.json`

Stage `Q1`, evidence `OBSERVED`, aggregate passed **true**, SHA-256 `4f9a71fa3fa51d66`.

| Measured field | Value |
| :--- | :--- |
| `dataset.content_chunks` | 4 |
| `dataset.corrupted_chunks` | 4 |
| `dataset.fabricated_chunks` | 4 |
| `dataset.facts_per_content_chunk` | 256 |
| `dataset.queries` | 24 |
| `dataset.hops` | 3 |
| `dataset.budget_u` | 32 |
| `ingest.admitted_chunks` | 4 |
| `ingest.compacted_slots_u` | 32 |
| `ingest.bridge_slots` | 1000 |
| `ingest.reference_facts` | 304 |
| `ingest.query_agnostic` | true |
| `accuracy.unseen_post_ingest` | 1 |
| `accuracy.correct` | 24 |

### `q2_causal_trimodal_telepathic_vla.json`

Stage `Q2`, evidence `OBSERVED`, aggregate passed **true**, SHA-256 `9c83dee6652fe8b1`.

| Measured field | Value |
| :--- | :--- |
| `episodes.count` | 12 |
| `episodes.goals_reached` | 12 |
| `episodes.executed` | 12 |
| `episodes.text_tokens_exchanged` | 0 |
| `episodes.receiver_prefill_tokens` | 0 |
| `episodes.counterfactual_pairs` | 12 |
| `episodes.counterfactual_changes` | 12 |
| `episodes.stratification.candidates` | 48 |
| `episodes.stratification.probed` | 16 |
| `episodes.stratification.classes.0` | 4 |
| `episodes.stratification.classes.1` | 4 |
| `episodes.stratification.classes.2` | 4 |
| `episodes.stratification.batch` | 12 |
| `adversarial.vision_vetoes` | 4 |

### `q3_disjoint_and_full_decile_language.json`

Stage `Q3`, evidence `OBSERVED`, aggregate passed **true**, SHA-256 `c65ecfb884e297d2`.

| Measured field | Value |
| :--- | :--- |
| `tier1.identifiers` | 965 |
| `tier1.shifts` | 0 |
| `arms.pinned_locked.split_byte_offset` | 307149 |
| `arms.pinned_locked.top1` | 0.5763 |
| `arms.pinned_locked.bits_per_char` | 1.10503 |
| `arms.pinned_locked.perplexity` | 7.10388 |
| `arms.pinned_locked.base_top1` | 0.5763 |
| `arms.pinned_locked.base_bits_per_char` | 1.10503 |
| `arms.pinned_locked.base_perplexity` | 7.10388 |
| `arms.pinned_locked.beta` | 0.02 |
| `arms.pinned_locked.beta_gate.beta_low` | 0.01 |
| `arms.pinned_locked.beta_gate.beta_high` | 0.02 |
| `arms.pinned_locked.beta_gate.tau_h` | 3 |
| `arms.pinned_locked.beta_gate.tau_order` | 4 |

### `q4_warm_holon_and_aaii_v5.json`

Stage `Q4`, evidence `OBSERVED`, aggregate passed **true**, SHA-256 `084afb691589760e`.

| Measured field | Value |
| :--- | :--- |
| `synthesis.battery_total` | 84 |
| `synthesis.battery_passed` | 84 |
| `synthesis.o_accepted` | 4 |
| `synthesis.o_refused` | 4 |
| `synthesis.p_accepted` | 4 |
| `synthesis.p_refused` | 4 |
| `synthesis.q_accepted` | 4 |
| `synthesis.q_refused` | 4 |
| `synthesis.total` | 108 |
| `synthesis.passed` | 108 |
| `synthesis.drift` | 0 |
| `synthesis.residual_state` | 0 |
| `synthesis.engine.vocabulary_size` | 2048 |
| `synthesis.engine.tokens` | 688596 |

### Explicit non-claims

* The causal operator is measured on a 24-step episode budget.
* AAII v5.0 numbers are proxies on ingested slices, not the official report.

## Cycle R: Unstructured telepathy, balanced causal matrix, static fact veto

| Stage | Receipt | SHA-256 (16) | Passed | Wall (s) |
| :--- | :--- | :--- | :---: | ---: |
| r3 | `r3_static_and_streaming_full_decile_language.json` | `d074511d012f2d9d` | true | 659.68 |

Cycle aggregate `passed`: **true**.

Measured gates: `r3_tier1_id_shifts_zero`=true, `r3_locked_overlap_zero`=true, `r3_elastic_overlap_zero`=true, `r3_live_overlap_zero`=true, `r3_full_overlap_zero`=true, `r3_locked_static_top1_ge_0_5600`=true, `r3_locked_static_bpc_le_1_2800`=true, `r3_elastic_static_top1_ge_0_5750`=true, `r3_elastic_static_bpc_le_1_2200`=true, `r3_live_static_top1_ge_0_5750`=true, `r3_live_static_bpc_le_1_2000`=true, `r3_full_static_top1_ge_0_5250`=true, `r3_full_static_bpc_le_1_4200`=true, `r3_full_stream_top1_ge_0_5600`=true, `r3_full_stream_bpc_le_1_2200`=true, `r3_full_scored_ge_50000`=true, `r3_recency_selected`=true, `r3_recency_static_inductive`=true, `r3_recency_zero_state`=true, `r3_suffix_selected`=true, `r3_suffix_orders_six`=true, `r3_suffix_static_inductive`=true, `r3_suffix_zero_state`=true, `r3_suffix_reads_no_target`=true, `r3_role_discounts_selected`=true, `r3_kn_mix_selected`=true.

### `r1_unstructured_telepathy_and_wave_fact_veto.json`

Stage `R1`, evidence `OBSERVED`, aggregate passed **true**, SHA-256 `3d0f83ed671bcfde`.

| Measured field | Value |
| :--- | :--- |
| `dataset.content_documents` | 4 |
| `dataset.ood_documents` | 4 |
| `dataset.fabricated_documents` | 6 |
| `dataset.queries` | 24 |
| `dataset.hops` | 3 |
| `dataset.context_tokens` | 18818 |
| `dataset.receiver_window_tokens` | 32 |
| `dataset.context_over_receiver_window` | 588.062 |
| `extraction.triples` | 724 |
| `extraction.registry_facts` | 562 |
| `extraction.documents_total` | 14 |
| `extraction.connected_documents` | 10 |
| `extraction.auto_registry` | true |
| `extraction.caller_reference_facts` | 0 |

### `r2_balanced_causal_vla_matrix.json`

Stage `R2`, evidence `OBSERVED`, aggregate passed **true**, SHA-256 `237b615cc0b180ad`.

| Measured field | Value |
| :--- | :--- |
| `readout.fit.rows` | 19 |
| `readout.fit.features` | 1921 |
| `readout.fit.classes` | 3 |
| `readout.fit.ridge` | 0.01 |
| `readout.fit.lambda` | 0.000508035 |
| `readout.fit.stiefel_rank` | 3 |
| `readout.fit.train_accuracy` | 1 |
| `readout.fit.feature_slots` | 4 |
| `readout.fit.feature_width` | 128 |
| `readout.train_accuracy` | 1 |
| `readout.heldout_accuracy` | 1 |
| `readout.heldout_rows` | 8 |
| `readout.persisted.bytes` | 7910227 |
| `episodes.count` | 12 |

### `r3_static_and_streaming_full_decile_language.json`

Stage `R3`, evidence `OBSERVED`, aggregate passed **true**, SHA-256 `d074511d012f2d9d`.

| Measured field | Value |
| :--- | :--- |
| `tier1.identifiers` | 965 |
| `tier1.shifts` | 0 |
| `arms.pinned_locked.split_byte_offset` | 307149 |
| `arms.pinned_locked.top1` | 0.5763 |
| `arms.pinned_locked.bits_per_char` | 1.10503 |
| `arms.pinned_locked.perplexity` | 7.10388 |
| `arms.pinned_locked.base_top1` | 0.5763 |
| `arms.pinned_locked.base_bits_per_char` | 1.10503 |
| `arms.pinned_locked.base_perplexity` | 7.10388 |
| `arms.pinned_locked.beta` | 0.02 |
| `arms.pinned_locked.beta_gate.beta_low` | 0.01 |
| `arms.pinned_locked.beta_gate.beta_high` | 0.02 |
| `arms.pinned_locked.beta_gate.tau_h` | 3 |
| `arms.pinned_locked.beta_gate.tau_order` | 4 |

### `r4_real_corpus_and_coprocessor_aaii_v5.json`

Stage `R4`, evidence `OBSERVED`, aggregate passed **true**, SHA-256 `8ace861d070284d1`.

| Measured field | Value |
| :--- | :--- |
| `aaii.returncode` | 0 |
| `aaii.passed` | true |
| `aaii.metrics.a_grounded_accuracy` | 1 |
| `aaii.metrics.a_non_hallucination` | 1 |
| `aaii.metrics.a_hallucination_rate` | 0 |
| `aaii.metrics.a_ood_abstain_rate` | 1 |
| `aaii.metrics.a_omni_index` | 100 |
| `aaii.metrics.a_omni_index_all_queries` | 50 |
| `aaii.metrics.a_omni_delta` | 132 |
| `aaii.metrics.b_rubric_pass_rate` | 1 |
| `aaii.metrics.b_abstain_rate` | 1 |
| `aaii.metrics.c_pass_at_1_rate` | 1 |
| `aaii.metrics.c_hallucinated_rejected` | 8 |
| `aaii.metrics.c_buggy_rejected` | 8 |

### Explicit non-claims

* The static fact veto is measured on the ingested corpus only.
* The balanced matrix holds three regimes, not the full layout space.

## Cycle S: Suffix purity, latent wave Turing machine, hierarchical charts

| Stage | Receipt | SHA-256 (16) | Passed | Wall (s) |
| :--- | :--- | :--- | :---: | ---: |
| s1 | `s1_language_closure.json` | `69f0ae65c969b647` | true | 3.76 |
| s2 | `s2_latent_wave_turing_machine.json` | `32de578ba037241c` | true | 1.02 |
| s3 | `s3_hierarchical_charts_and_sandboxed_browser.json` | `baf4eb4d28748ec8` | true | 2.24 |

Cycle aggregate `passed`: **true**.

Measured gates: `s1_r3_receipt_fresh`=true, `s1_r3_rerun_exit_zero`=true, `s1_static_top1_ge_0_5250`=true, `s1_static_bpc_le_1_4200`=true, `s1_stream_top1_ge_0_5600`=true, `s1_stream_bpc_le_1_2200`=true, `s1_full_scored_ge_50000`=true, `s1_calibration_overlap_zero`=true, `s1_is_static_inductive`=true, `s1_suffix_selected`=true, `s1_suffix_orders_six`=true, `s1_suffix_zero_state`=true, `s1_call_order_invariant`=true, `s1_recency_static`=true, `s1_r3_gate_passed`=true, `s1_r3_stage_passed`=true, `s2_programs_16`=true, `s2_exact_16`=true, `s2_zero_text_tokens`=true, `s2_step_ms_le_1`=true, `s2_traps_4`=true, `s2_traps_vetoed_4`=true, `s2_false_accepts_zero`=true, `s2_slots_4`=true, `s2_dim_8192`=true, `s3_engrams_ge_2500`=true, `s3_charts_ge_12`=true, `s3_chart_capacity_256`=true, `s3_latency_clock_invariant`=true, `s3_speedup_ge_1_40`=true, `s3_evaluated_candidates_le_0_40`=true, `s3_charted_faster_than_flat`=true, `s3_recall_parity_ge_0_99`=true, `s3_exact_top1_ge_0_99`=true, `s3_crosstalk_reduced`=true, `s3_browser_probes_12`=true, `s3_browser_grounded_12`=true, `s3_browser_envelope_4`=true, `s3_browser_traps_blocked_4`=true, `s3_browser_zero_refusals_in_probes`=true.

### `s1_language_closure.json`

Stage `S1`, evidence `OBSERVED`, aggregate passed **true**, SHA-256 `69f0ae65c969b647`.

| Measured field | Value |
| :--- | :--- |
| `r3.static_top1` | 0.577869 |
| `r3.static_bits_per_char` | 1.22368 |
| `r3.stream_top1` | 0.588599 |
| `r3.stream_bits_per_char` | 1.16604 |
| `r3.scored_tokens` | 61417 |
| `r3.suffix_floor.weight` | 1.25 |
| `r3.suffix_floor.base_nll` | 2.86218 |
| `r3.suffix_floor.base_top1` | 0.507368 |
| `r3.suffix_floor.selected_nll` | 2.59272 |
| `r3.suffix_floor.selected_top1` | 0.553684 |
| `r3.suffix_floor.window` | 2048 |
| `r3.suffix_floor.orders` | 6 |
| `r3.suffix_floor.alpha.1` | 0.05 |
| `r3.suffix_floor.alpha.2` | 0.15 |

### `s2_latent_wave_turing_machine.json`

Stage `S2`, evidence `OBSERVED`, aggregate passed **true**, SHA-256 `32de578ba037241c`.

| Measured field | Value |
| :--- | :--- |
| `machine.slots` | 4 |
| `machine.dim` | 8192 |
| `machine.tau_read` | 0.35 |
| `machine.tau_transition` | 0.1 |
| `machine.tau_cycle` | 0.995 |
| `metrics.exact_rate` | 1 |
| `metrics.veto_rate` | 1 |
| `metrics.false_accepts` | 0 |
| `metrics.step_ms_mean` | 0.805899 |
| `metrics.step_ms_max` | 1.60673 |
| `metrics.text_tokens` | 0 |
| `metrics.total_steps` | 1035 |
| `gate.s2_programs_16` | true |
| `gate.s2_exact_16` | true |

### `s3_hierarchical_charts_and_sandboxed_browser.json`

Stage `S3`, evidence `OBSERVED`, aggregate passed **true**, SHA-256 `baf4eb4d28748ec8`.

| Measured field | Value |
| :--- | :--- |
| `vault.charts` | 12 |
| `vault.chart_capacity` | 256 |
| `vault.engrams` | 2880 |
| `vault.largest_chart` | 256 |
| `vault.smallest_chart` | 208 |
| `vault.mean_chart` | 240 |
| `vault.engrams_written` | 2880 |
| `vault.write_seconds` | 1.29369 |
| `latency.charted_p50_ms` | 0.9467 |
| `latency.charted_p90_ms` | 1.30313 |
| `latency.flat_p50_ms` | 1.46825 |
| `latency.flat_p90_ms` | 1.80568 |
| `latency.speedup` | 1.55091 |
| `latency.t_ref_ms` | 0.1314 |

### `s4_four_pillar_and_176_query_gauntlet.json`

Stage `S4`, evidence `OBSERVED`, aggregate passed **true**, SHA-256 `a4c9a80ab499777f`.

| Measured field | Value |
| :--- | :--- |
| `aaii.returncode` | 0 |
| `aaii.passed` | true |
| `gauntlet.returncode` | 0 |
| `gauntlet.total_queries` | 176 |
| `gauntlet.categories` | 32 |
| `gauntlet.reachable_accuracy` | 1 |
| `gauntlet.hallucination_rate` | 0 |
| `gauntlet.query_file_reads` | 0 |
| `pillars.A.metrics.grounded_accuracy` | 1 |
| `pillars.A.metrics.ood_abstain_rate` | 1 |
| `pillars.A.metrics.non_hallucination_rate` | 1 |
| `pillars.A.metrics.hallucination_rate` | 0 |
| `pillars.A.metrics.henri_correct` | 100 |
| `pillars.A.metrics.henri_incorrect` | 0 |

### Explicit non-claims

* Suffix purity is a language property of the pinned corpus.
* The latent wave Turing machine is a 16-cell bench, not a general computer.

## Cycle T: Hygiene and determinism, centered charts, foveated Turing, CCG and sheaf

| Stage | Receipt | SHA-256 (16) | Passed | Wall (s) |
| :--- | :--- | :--- | :---: | ---: |
| t1 | `t1_hygiene_and_determinism.json` | `0881cc48e291ff91` | true | 0.01 |
| t2 | `t2_riemannian_chart_atlas.json` | `2f03d17887c31fa9` | true | 47.62 |
| t3 | `t3_foveated_bidirectional_turing.json` | `5173534b01e90cfb` | false | 98.69 |
| t4 | `t4_sovereign_ccg_and_repo_sheaf.json` | `5e60fa748c8230c0` | true | 6.98 |

Cycle aggregate `passed`: **false**.

Measured gates: `t1_docs_monotonic`=true, `t1_docs_no_duplicates`=true, `t1_docs_sections_1_to_36`=true, `t1_docs_receipts_referenced`=true, `t1_q2_passed`=true, `t1_q2_goals_12`=true, `t1_q2_text_tokens_zero`=true, `t1_q2_pairs_ge_12`=true, `t1_q2_accuracy_1`=true, `t1_q2_regimes_3`=true, `t1_s3_passed`=true, `t1_s3_parity_1`=true, `t1_s3_speedup_ge_1_40`=true, `t1_s3_evaluated_ratio_le_0_40`=true, `t1_s3_latency_clock_invariant`=true, `t2_mmlu_parity_ge_0_995`=true, `t2_grounded_ge_0_99`=true, `t2_drift_zero`=true, `t2_ood_abstain_1`=true, `t2_hallucination_zero`=true, `t2_candidates_le_0_35`=true, `t2_speedup_ge_1_40`=true, `t2_online_recall_ge_0_99`=true, `t2_atlas_synchronized`=true, `t3_capacity_exact_long`=true, `t3_margin_ge_0_40`=true, `t3_margin_invariant`=true, `t3_crosstalk_zero`=true, `t3_pages_paged`=true, `t3b_solved_ge_10`=false, `t3b_generalized_ge_10`=false, `t3b_zero_text_tokens`=true, `t3b_contradictions_4`=true, `u4_parse_ge_30`=true, `u4_pairs_8`=true, `u4_reversals_3`=true, `u4_contradictions_6`=true, `u4_packages_6`=true, `u4_unannotated_caught_24`=true, `u4_polymorphic_accepted_12`=true, `u4_kappa_le_50`=true.

### `t1_hygiene_and_determinism.json`

Stage `T1`, evidence `OBSERVED`, aggregate passed **true**, SHA-256 `0881cc48e291ff91`.

| Measured field | Value |
| :--- | :--- |
| `documents.count` | 36 |
| `documents.first` | 1 |
| `documents.last` | 36 |
| `documents.monotonic` | true |
| `documents.sections_1_to_36` | true |
| `documents.receipts` | 116 |
| `documents.referenced` | 116 |
| `q2.passed` | true |
| `q2.goals_reached` | 12 |
| `q2.episodes` | 12 |
| `q2.text_tokens_exchanged` | 0 |
| `q2.counterfactual_pairs` | 12 |
| `q2.counterfactual_changes` | 12 |
| `q2.counterfactual_accuracy` | 1 |

### `t2_centered_charts_mmlu.json`

Stage `T2`, evidence `OBSERVED`, aggregate passed **true**, SHA-256 `ee87fb81c12657ce`.

| Measured field | Value |
| :--- | :--- |
| `dataset.ingested` | 2500 |
| `dataset.total_engrams_with_seed_cartridge` | 3849 |
| `dataset.per_subject` | 250 |
| `dataset.grounded_queries` | 100 |
| `dataset.ood_queries` | 100 |
| `index.legacy.charts` | 62 |
| `index.legacy.chart_capacity` | 64 |
| `index.legacy.engrams` | 3849 |
| `index.legacy.largest_chart` | 64 |
| `index.legacy.smallest_chart` | 23 |
| `index.legacy.mean_chart` | 62.0806 |
| `index.centered.rows` | 2500 |
| `index.centered.charts` | 40 |
| `index.centered.capacity` | 64 |

### `t3_foveated_bidirectional_turing.json`

Stage `T3`, evidence `OBSERVED`, aggregate passed **false**, SHA-256 `5173534b01e90cfb`.

| Measured field | Value |
| :--- | :--- |
| `capacity.margin_16` | 0.741923 |
| `capacity.margin_256` | 0.741923 |
| `capacity.crosstalk_max` | 0 |
| `induction.solved` | 8 |
| `induction.generalized` | 8 |
| `induction.text_tokens` | 0 |
| `induction.contradictions_failed_closed` | 4 |
| `induction.options_crystallized` | 8 |
| `gate.t3_capacity_exact_long` | true |
| `gate.t3_margin_ge_0_40` | true |
| `gate.t3_margin_invariant` | true |
| `gate.t3_crosstalk_zero` | true |
| `gate.t3_pages_paged` | true |
| `gate.t3b_solved_ge_10` | false |

### `t4_sovereign_ccg_and_repo_sheaf.json`

Stage `T4`, evidence `OBSERVED`, aggregate passed **true**, SHA-256 `5e60fa748c8230c0`.

| Measured field | Value |
| :--- | :--- |
| `ccg.corpus_total` | 32 |
| `ccg.parsed_exact` | 32 |
| `sheaf.mismatch_caught` | 24 |
| `sheaf.polymorphic_accepted` | 12 |
| `sheaf.kappa_plus` | 1 |
| `sheaf.naive_wildcard_control_false_accept_rate` | 1 |
| `gate.u4_parse_ge_30` | true |
| `gate.u4_pairs_8` | true |
| `gate.u4_reversals_3` | true |
| `gate.u4_contradictions_6` | true |
| `gate.u4_packages_6` | true |
| `gate.u4_unannotated_caught_24` | true |
| `gate.u4_polymorphic_accepted_12` | true |
| `gate.u4_kappa_le_50` | true |

### Explicit non-claims

* The foveated two-cell lookahead is diagnostic and is never counted as one-cell induction.
* Centered charts are measured on the MMLU slice, not on open-domain traffic.

## Provenance

Each stage table above is a projection of a sealed receipt. The receipt
and its sidecar, not this prose, are the durable evidence.
