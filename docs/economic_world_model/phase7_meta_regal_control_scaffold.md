# Phase 7 Meta-Regal-Node / Control WM Scaffold

## Scope

Phase 7 starts as Stage A typed non-neural governance scaffolding. It composes
domain-governance nodes under shadow-only Pareto, lexicographic, veto,
advisory, and confidence-weighted relations. It does not grant control-WM
authority.

The scaffold consumes:

- Phase 6.5 local meta-node neuralization report
- Phase 3.5 / 4 / 6.5 local closure audit

It emits:

- `Phase7GovernanceNodeSurface`
- `Phase7CompositionModeSpec`
- `Phase7ConflictOverrideReceipt`
- `Phase7AdmissibleRegionSpec`
- `Phase7ControlFieldSlot`
- `Phase7TrainingRowSlot`
- `Phase7PromotionGate`
- `Phase7MetaRegalControlScaffoldReport`

## Local Artifact Result

Current CLI:

```bash
python3 scripts/economic_world_model/prepare_phase7_meta_regal_control_scaffold.py --no-run-dependencies
```

Current artifact report:

```text
artifacts/economic_world_model/phase7_meta_regal_control_scaffold/phase7_meta_regal_control_scaffold_report_v1.json
```

Current local result:

- `status=ok`
- `governance_node_surface_count=8`
- `composition_mode_count=5`
- `conflict_override_receipt_count=6`
- `admissible_region_count=6`
- `control_field_slot_count=7`
- `training_row_slot_count=6`
- `promotion_gate_count=8`
- `local_phase7_scaffold_complete=true`
- `ready_for_runtime_wiring=true`
- `runtime_wiring_executed=false`
- `phase7_authority_granted=false`
- `live_control_authority=false`
- `training_executed=false`
- `weights_written=false`
- `provider_executed=false`
- `hardware_executed=false`
- `unitree_sim_runtime_executed=false`
- `live_policy_control=false`
- `reward_math_mutation=false`
- `promotion_eligible=false`

## Shadow Runtime / Event-Spine Wiring

Current CLI:

```bash
python3 scripts/economic_world_model/wire_phase7_meta_regal_runtime_shadow.py --no-run-dependencies --episodes 2 --run-id phase7_shadow_runtime_local
```

Current artifact directory:

```text
artifacts/economic_world_model/phase7_meta_regal_shadow_runtime
```

Current local result:

- `phase7_meta_regal_shadow.enabled=true`
- `phase7_meta_regal_shadow.episode_report_count=2`
- `phase7_meta_regal_shadow.control_field_runtime_receipt_count=14`
- `phase7_meta_regal_shadow.conflict_runtime_join_receipt_count=12`
- `phase7_meta_regal_shadow.shadow_event_spine_wiring_executed=true`
- `phase7_meta_regal_shadow.decision_ledger_wiring_executed=true`
- `phase7_meta_regal_shadow.local_shadow_runtime_wiring_complete=true`
- event spine contains `phase7_control_field_shadow_emitted`
- event spine contains `phase7_conflict_override_shadow_joined`
- decision ledger contains `phase7_control_field_shadow_recorded`
- decision ledger contains `phase7_conflict_override_shadow_recorded`
- `phase7_authority_granted=false`
- `live_dispatch_allowed=false`
- `hard_veto_dispatch=false`
- `training_executed=false`
- `weights_written=false`
- `provider_executed=false`
- `hardware_executed=false`
- `unitree_sim_runtime_executed=false`
- `live_policy_control=false`
- `reward_math_mutation=false`
- `promotion_eligible=false`

This wiring connects the Phase 7 control-field slots and conflict receipts to
the existing shadow runtime `event_spine.json` and `decision_ledger.json`
sidecars. It is still shadow-only. It records what the Meta-Regal layer would
shape or join; it does not dispatch actions, hard vetoes, policy changes,
reward mutations, weight writes, or promotions.

## Meta-Governance Evaluation / Outcome Joins

Current CLI:

```bash
python3 scripts/economic_world_model/evaluate_phase7_meta_governance_shadow.py --no-run-dependencies
```

Current artifact directory:

```text
artifacts/economic_world_model/phase7_meta_governance_eval
```

Current local result:

- `control_field_eval_count=14`
- `conflict_join_eval_count=12`
- `pareto_regime_eval_count=2`
- `outcome_join_row_count=28`
- `phase7_event_count=26`
- `phase7_decision_count=26`
- `control_field_only_eval_complete=true`
- `conflict_join_eval_complete=true`
- `pareto_regime_eval_complete=true`
- `outcome_join_slots_complete=true`
- `local_meta_governance_eval_complete=true`
- `replay_export_ready=true`
- `phase7_authority_granted=false`
- `live_dispatch_allowed=false`
- `hard_veto_dispatch=false`
- `training_executed=false`
- `weights_written=false`
- `provider_executed=false`
- `hardware_executed=false`
- `unitree_sim_runtime_executed=false`
- `live_policy_control=false`
- `reward_math_mutation=false`
- `promotion_eligible=false`

The evaluation harness decomposes Phase 7 event-spine rows into:

- control-field-only eval reports
- conflict-join eval reports
- Pareto/regime eval reports
- replay-ready outcome join rows

The outcome rows carry false-veto, false-allow, counterfactual composition,
regime-label, policy-regret, and downstream-effect slots for later labeled
benchmarks. These are training targets only; no training or promotion is
performed.

## Governance Signal Adapters

Current CLI:

```bash
python3 scripts/economic_world_model/adapt_phase7_governance_node_signals.py --no-run-dependencies
python3 scripts/economic_world_model/wire_phase7_meta_regal_runtime_shadow.py --no-run-dependencies --run-id phase7_shadow_runtime_local --phase7-signal-adapter-dir artifacts/economic_world_model/phase7_governance_signal_adapters
```

Current artifact directory:

```text
artifacts/economic_world_model/phase7_governance_signal_adapters
```

Current local result:

- `adapter_count=8`
- `signal_receipt_count=8`
- `source_artifact_count=35`
- `missing_source_artifact_count=0`
- `lower_wm_receipt_backed_node_count=8`
- `all_eight_nodes_signal_backed=true`
- `shadow_runtime_feed_ready=true`
- `local_signal_adapter_complete=true`
- `phase7_meta_regal_shadow.node_signal_receipt_count=8`
- `phase7_meta_regal_shadow.lower_wm_signal_backed=true`
- `phase7_authority_granted=false`
- `live_dispatch_allowed=false`
- `hard_veto_dispatch=false`
- `training_executed=false`
- `weights_written=false`
- `provider_executed=false`
- `hardware_executed=false`
- `unitree_sim_runtime_executed=false`
- `live_policy_control=false`
- `reward_math_mutation=false`
- `promotion_eligible=false`

The adapter layer maps actual local lower-WM receipts into the eight governance
nodes: Phase 3.5 bipedal/refit receipts, Phase 4 controller/safety/Unitree
receipts, Phase 6 transport advisory/eval receipts, Phase 6.5 meta-node
receipts, and Phase 7 shadow runtime/eval receipts. Runtime events and
receipts now carry `node_signal_receipt_ids` and `lower_wm_signal_backed`
metadata when the adapter directory is provided.

This is still a shadow feed. The signals are typed inputs for governance-node
evaluation and later training labels; they do not grant authority or execute
hard vetoes.

## Governance Node Surfaces

The scaffold defines these domain-governance surfaces:

- `economic_allocation_governance`
- `reward_integrity_governance`
- `plausibility_geometry_governance`
- `deployment_truth_governance`
- `safety_constraint_governance`
- `data_value_governance`
- `embodiment_limit_governance`
- `coordination_operator_governance`

Each surface is advisory-only, training-aware, and denied live authority. The
Economic WM is a first-class allocative contributor, not the sovereign
governor.

## Composition Modes

The scaffold defines:

- `pareto_relation`
- `lexicographic_priority`
- `veto_constraint`
- `advisory_evidence`
- `confidence_weighted`

These are shadow composition modes only. Even `veto_constraint` is a typed
candidate field, not a live hard-dispatch authority.

## Shadow Control Fields

The scaffold creates slots for:

- `cross_wm_shaping_field`
- `economic_budget_constraint_field`
- `safety_veto_field`
- `deployment_truth_veto_field`
- `operator_recovery_handoff_field`
- `data_collection_priority_field`
- `embodiment_mode_demote_field`

The embodiment field preserves the current doctrine:
`bipedal_whole_body` is primary, `stable_base_mobile_manipulator` is the
safety fallback/degraded mode, and `fixed_base_tabletop` is curriculum and
regression only.

## Training Awareness

`Phase7TrainingRowSlot` records replay/export-ready row families for future
training:

- governance-node snapshots
- conflict/override rows
- admissible-region rows
- control-field shadow outcome rows
- counterfactual composition targets
- governance failure and recovery rows

All rows are training targets only. They do not train, write weights, dispatch
policy, or promote authority.

## Denied Authorities

The scaffold explicitly denies:

- training execution
- weight writes
- provider execution
- hardware execution
- Unitree sim runtime authority claims
- live policy control
- reward math mutation
- promotion
- live cross-WM control
- hard veto dispatch
- lower-WM replacement
- scalar governance collapse
- Phase 7 runtime authority

## Remaining Blockers

Phase 7 can proceed to runtime wiring only after the typed surfaces above
exist. Runtime authority, training, and promotion remain blocked by:

- lower-WM bounded runtime authority
- governance-node training and benchmark evidence
- cross-WM governance corpus density
- meta-composition learning
- real governance benchmark evidence
- live runtime wiring execution
- provider/hardware deployment evidence
- ground-truth outcome labels
- false-veto / false-allow labels
- counterfactual composition benchmarks
- trained meta-composition policy
- live lower-WM runtime streams
- labeled governance signal outcomes
- trained governance signal weights

## Boundary

This scaffold does not mutate frozen Phase B baseline behavior, trust net,
`w_econ` lattice, lambda controller, reward math, lower-WM contracts, or live
control surfaces. It is additive and local.
