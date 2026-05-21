# Economic WM Trainer Scaffold v0

Date: 2026-05-21

## Purpose

`train_economic_world_model_v0.py` is a trainer scaffold, not a trainer run. It shape-checks the Phase-5 local prep corpus against the neural architecture manifest, emits dataset/model/loss contracts, and runs deterministic CPU-only smoke forwards.

It emits:

- `economic_wm_trainer_dataset_contract_v1.json`
- `economic_wm_trainer_model_component_config_v1.json`
- `economic_wm_trainer_loss_definitions_v1.json`
- `economic_wm_trainer_cpu_smoke_forward_v1.json`
- `economic_wm_trainer_scaffold_manifest_v1.json`
- `economic_wm_trainer_scaffold_v1.md`

## Executable path

```bash
python3 scripts/train_economic_world_model_v0.py \
  --output-dir artifacts/economic_world_model/economic_wm_trainer_scaffold \
  --phase5-prep artifacts/economic_world_model/economic_wm_phase5_local_prep/economic_wm_phase5_local_prep_manifest_v1.json \
  --neural-manifest artifacts/economic_world_model/economic_wm_neural_architecture_manifest/economic_wm_neural_architecture_manifest_v1.json
```

## Current authority

The scaffold must report:

- `training_executed=false`
- `weights_written=false`
- `ready_for_gpu_training=false`
- `promotion_eligible=false`
- `reward_math_mutation=false`
- `authority_class=trainer_scaffold_only`

## Current local result

The current local artifact run reports:

- `authority_class=trainer_scaffold_only`
- `dataset_contract_ready=true`
- `cpu_smoke_forward_passed=true`
- `training_executed=false`
- `weights_written=false`
- `promotion_eligible=false`

## Boundary

The CPU smoke forward proves only shape compatibility and finite deterministic outputs. Real estimator, dynamics, allocator, datapack-composition, and governance learning remains blocked until GPU/provider capacity, non-stub teacher/runtime evidence, and promotion-grade shadow benchmarks exist.
