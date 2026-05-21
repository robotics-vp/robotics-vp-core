# Economic WM Shadow Execution

Date: 2026-05-21

## Purpose

This pass lets Economic WM outputs produce advisory shadow work orders and future outcome-comparison slots. It is the first local loop where Economic WM recommendations can be compared against lower-WM outcomes later without controlling reward math or live policy.

It emits:

- `economic_wm_shadow_execution_report_v1.json`
- `economic_wm_shadow_work_orders_v1.jsonl`
- `economic_wm_shadow_outcome_comparisons_v1.jsonl`
- `economic_wm_shadow_execution_v1.md`

## Executable path

```bash
python3 scripts/economic_world_model/run_economic_wm_shadow_execution.py \
  --output-dir artifacts/economic_world_model/economic_wm_shadow_execution \
  --phase5-prep artifacts/economic_world_model/economic_wm_phase5_local_prep/economic_wm_phase5_local_prep_manifest_v1.json \
  --allocation-eval artifacts/economic_world_model/economic_wm_shadow_allocation_eval/economic_wm_shadow_allocation_eval_v1.json \
  --trainer-scaffold artifacts/economic_world_model/economic_wm_trainer_scaffold/economic_wm_trainer_scaffold_manifest_v1.json
```

## What this proves locally

- Allocation candidates can become typed shadow work orders.
- Work orders point at Phase-5 composition rows, temporal windows, and outcome receipt slots.
- Outcome-comparison rows are ready to receive later receipts for counterfactual accuracy and Pareto-quality evaluation.
- Denied authority is explicit: no live policy control, reward-math mutation, provider truth substitution, GPU training execution, or promotion decision.

## Current local result

The current local artifact run reports:

- `work_order_count=3`
- `outcome_comparison_count=3`
- `ready_for_shadow_comparison=true`
- `live_policy_control=false`
- `reward_math_mutation=false`
- `promotion_eligible=false`

## Boundary

This is `shadow_execution_only`. It emits advisory work orders and comparison slots only; it does not execute live allocation, hardware control, provider bring-up, GPU training, promotion, or frozen math mutation.
