# Phase 6.5 Local Meta-Node Neuralization and Robustness

Date: 2026-05-23

## Purpose

This pass defines the local Phase 6.5 scaffold after:

1. Phase 6 transport local closure;
2. Phase 3.5 humanoid capacity and environment refit;
3. Phase 4 local non-hardware deployment-enabler sweep.

The goal is to make local meta-node surfaces more stateful, trainable,
replayable, and robust without creating the Phase 7 Meta-Regal-Node / control
WM prematurely.

This is local scaffold doctrine only. It does not train meta-node weights,
promote meta-node authority, run providers, execute hardware, mutate reward
math, or grant live policy control.

## Local Meta-Node Surfaces

Minimum local surfaces:

| Surface | Purpose | Authority posture |
| --- | --- | --- |
| `MetaNodeState` | Canonical state for one local governance/control node, including activation, confidence, target refs, and posture scope | state only |
| `MetaNodeTrajectoryReceipt` | Time-indexed activation/deactivation/intervention history with event and replay refs | observational |
| `MetaNodeInterventionReceipt` | Veto, defer, shape, fallback, or operator-handoff proposal with rationale and target refs | advisory/shadow |
| `MetaNodeCounterfactualTarget` | Target rows for when a node should activate, how strongly, and with which downstream effect | training target only |
| `MetaNodeRobustnessReport` | Stability under replay shift, calibration, governance satisfaction, and neighbor-interaction consistency | evaluation only |
| `MetaNodePromotionGate` | Explicit denial or bounded authority request with missing evidence listed | denied by default |

## Training Targets

Local Phase 6.5 should prepare training rows for:

- activation timing;
- activation strength;
- target selection;
- veto/defer/fallback choice;
- operator-handoff request;
- governance satisfaction;
- stability under replay perturbation;
- interaction consistency with neighboring nodes;
- counterfactual downstream improvement;
- demotion / rollback sensitivity.

These rows may consume:

- Economic WM allocation envelopes and shadow outcomes;
- Phase 6 transport eval reports and shadow join slots;
- Phase 3.5 posture tags and humanoid schema deltas;
- Phase 4 timing, placement, degraded-mode, and recovery contracts;
- event-spine, governance-trace, and replay refs.

## Robustness Metrics

| Metric | Meaning | Must not claim |
| --- | --- | --- |
| Activation calibration | Whether confidence matches observed usefulness | promotion without heldout evidence |
| Replay-shift stability | Whether node behavior is stable under replay perturbations | deployment robustness |
| Neighbor consistency | Whether adjacent local nodes interact without oscillation or hidden veto loops | global governance optimality |
| Governance satisfaction | Whether constraints and rationale receipts remain coherent | reward-math authority |
| Degraded-mode integrity | Whether fallback/defer/operator handoff is explicit and replayable | hardware recovery closure |
| Transport sensitivity | Whether Phase 6 transport quality changes node behavior legibly | trained transport authority |

## Denied Gates

Every local Phase 6.5 artifact should deny:

- meta-node training execution unless a real trainer ran;
- weight writes unless real weights were written;
- provider or hardware execution unless actually run;
- live policy control;
- reward math mutation;
- promotion;
- replacement of lower-WM contracts;
- direct Phase 7 control-WM authority.

## Exit Criteria

Local Phase 6.5 is structurally complete only when:

- local meta-node state is canonical and replayable;
- intervention and trajectory receipts exist;
- counterfactual target row shapes exist;
- robustness report surfaces exist;
- promotion gates are explicit and denied by default;
- docs/tests keep Phase 7 separate from local meta-node maturity.

Remaining blockers after local Phase 6.5 should be counterfactual corpus density,
trained node weights, robustness benchmarks, provider/hardware/deployment
evidence, and real governance benchmark evidence.

## Phase 7 and Phase 8 Handoff

Phase 7 may begin only after local meta-nodes have their own state, receipts,
training rows, robustness reports, and denied/conditional promotion gates.

Phase 8 remains production-loop runtime and weekly GPU operations. It should not
be claimed until training/provider capacity actually exists and recurring GPU or
RunPod operations are real rather than planned.

## Local Scaffold Implementation

As of 2026-05-24 this phase is backed by typed local artifacts, not only this
planning note.

Code and CLI surfaces:

- `src/world_model/humanoid_readiness/phase65.py`
- `src/world_model/humanoid_readiness/closure.py`
- `scripts/economic_world_model/prepare_phase65_meta_node_neuralization.py`
- `scripts/economic_world_model/audit_phase35_4_65_local_closure.py`
- `tests/test_humanoid_phase35_4_65_scaffolds.py`

Current artifact output:

- `artifacts/economic_world_model/phase65_meta_node_neuralization/phase65_meta_node_neuralization_report_v1.json`
- `node_state_count=5`
- `trajectory_receipt_count=5`
- `intervention_receipt_count=5`
- `counterfactual_target_count=5`
- `robustness_report_count=5`
- `promotion_gate_count=5`
- `local_meta_node_scaffold_complete=true`
- `ready_for_phase7_scaffold=true`
- `phase7_authority_granted=false`

The integrated local closure audit is:

- `artifacts/economic_world_model/phase35_4_65_local_closure/phase35_4_65_local_closure_audit_v1.json`
- `all_local_structures_complete=true`

Denied gates remain explicit:

- `training_executed=false`
- `weights_written=false`
- `provider_executed=false`
- `hardware_executed=false`
- `unitree_sim_runtime_executed=false`
- `live_policy_control=false`
- `reward_math_mutation=false`
- `promotion_eligible=false`
