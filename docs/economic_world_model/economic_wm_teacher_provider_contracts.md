# Economic WM Teacher/Provider Evidence Contracts

Date: 2026-05-21

## Purpose

This contract pack is the local prep step recommended by the Economic WM shadow allocation eval. It names the exact evidence surfaces that must exist before non-stub teacher runtime, provider bring-up, GPU training, or model promotion can be claimed.

It emits:

- `economic_wm_teacher_provider_contract_v1.json`
- `economic_wm_teacher_provider_contract_v1.md`

## Executable path

```bash
python3 scripts/economic_world_model/prepare_economic_wm_teacher_provider_contracts.py \
  --output-dir artifacts/economic_world_model/economic_wm_teacher_provider_contracts \
  --scaffold-report artifacts/economic_world_model/economic_wm_scaffold/economic_wm_scaffold_report_v1.json \
  --allocation-eval artifacts/economic_world_model/economic_wm_shadow_allocation_eval/economic_wm_shadow_allocation_eval_v1.json \
  --corpus-manifest artifacts/economic_world_model/economic_wm_training_rows/economic_wm_training_corpus_manifest_v1.json \
  --rows artifacts/economic_world_model/economic_wm_training_rows/economic_wm_replay_feature_rows_v1.jsonl
```

If inputs are missing, the script can run the shadow allocation evaluator first, which can materialize rows and scaffold artifacts as needed.

## Current local result

Current status remains contract-prep only:

- `authority_class=evidence_contract_only`
- `provider_bringup_ready=false`
- `gpu_training_ready=false`
- `promotion_eligible=false`
- `reward_math_mutation=false`

Aggregate scores from the current local corpus:

- `teacher_contract_fraction=1.0`
- `teacher_real_fraction=0.0`
- `provider_gap_weight_mean=1.0`
- `benchmark_ready_fraction=0.4`
- `replay_export_flow=1.0`

## Required evidence families

The contract currently requires:

- non-stub teacher runtime invocation receipts
- external provider runtime truth receipts
- promotion-grade benchmark evidence
- GPU training runtime receipts
- replay-row linkage integrity

Only replay-row linkage is locally satisfied. Teacher/provider/GPU/promotion requirements remain blocked until they literally run and produce receipts.

## Boundary

This does not run OpenVLA, V-JEPA, diffusion providers, SceneTracks, GPU training, or promotion benchmarks. It only defines the evidence shape those future runs must satisfy.
