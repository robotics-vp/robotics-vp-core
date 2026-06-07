# Neural Trainability Audit

Date: 2026-06-07

## Purpose

This audit makes the local trainability surface explicit before GPU/provider or
hardware work. It inventories neural, seam, encoder, policy, head, bridge,
receiver, trainer, training-script, and backlog surfaces and emits executable
follow-up rows for the missing code, row, data, provider, GPU, hardware, and
benchmark pieces.

It is a planning and receipt surface only. It does not train, write weights,
run providers, launch RunPod, execute hardware, mutate reward/controller math,
grant Phase 7 authority, or claim promotion.

## Artifact Paths

```bash
python3 scripts/economic_world_model/compile_neural_trainability_audit.py
```

Current artifact outputs:

- `artifacts/economic_world_model/neural_trainability_audit/neural_trainability_audit_report_v1.json`
- `artifacts/economic_world_model/neural_trainability_audit/neural_trainability_components_v1.jsonl`
- `artifacts/economic_world_model/neural_trainability_audit/neural_trainability_followups_v1.jsonl`
- `artifacts/economic_world_model/neural_trainability_audit/neural_trainability_audit_v1.md`
- `artifacts/economic_world_model/neural_trainability_audit/neural_trainability_audit_validation_v1.json`

## Current Local Result

- `status=ok_neural_trainability_audit_non_training`
- `component_count=21`
- `followup_count=27`
- `ready_for_training_count=0`
- `promotion_eligible_count=0`
- `local_static_ready_count=20`
- `validation.status=ok`
- `validation.error_count=0`
- `validation.safe_for_training=false`
- `validation.safe_for_promotion=false`

Follow-up plane counts:

- `local=0`
- `codex=8`
- `runpod_provider=5`
- `runpod_train=11`
- `hardware_runtime=3`

Blocker counts:

- `code=6`
- `row=2`
- `data=1`
- `provider=5`
- `gpu=7`
- `hardware=3`
- `benchmark_missing=3`

## Surface Roles

The audit rows classify touched and adjacent trainability surfaces as:

- `lower-WM producer`
- `trainer/runtime lane`
- `curriculum/regression source`
- `provider/hardware adapter`
- `receipt substrate`
- `legacy/dev-only tool`

Old training scripts are not treated as obsolete by default. They are either
mapped to active WM trainer/runtime lanes or kept as legacy/dev-only tools with
explicit migration follow-ups.

## Covered Component Families

The curated trainability rows cover:

- Perception evidence-fusion and V-JEPA temporal seams
- Vision backbone projection heads
- Sim/Synth predictive V-JEPA component
- Embodiment Phase 3.4/4 whole-body neural scaffolds
- Economic WM neural components
- WM transport bridge/receiver trainer
- Phase 6.5 meta-node trainer
- Phase 7 signal adapter consumer
- Bio/neuro trainer family
- Orchestrator semantic runtime trainers
- VLA/OpenVLA/recap heads
- RL/HRL curriculum policy family

Remaining `scripts/TRAINING_MIGRATION_BACKLOG.json` rows are folded into
auto-generated training-backlog components and follow-up rows.

## Boundary

All rows remain non-promotional:

- `training_executed=false`
- `weights_written=false`
- `provider_executed=false`
- `gpu_executed=false`
- `runpod_launched=false`
- `hardware_executed=false`
- `live_policy_control=false`
- `reward_math_mutation=false`
- `phase7_authority_granted=false`
- `promotion_eligible=false`

RunPod and hardware rows are blocked follow-ups, not execution claims.
