# Economic WM Entry Preflight

Date: 2026-05-21

## Purpose

This preflight separates two states that must not be collapsed:

1. **Ready to start the Economic WM scaffold**: lower-WM outputs are stable enough to define the Economic WM input/evaluation contracts.
2. **Ready to train or promote an Economic WM**: GPU/provider-backed runs, non-stub teacher/runtime evidence, and promotion-grade benchmarks exist.

The current local state is **scaffold-ready, training-blocked**.

## Executable gate

Run:

```bash
python3 scripts/economic_world_model/economic_wm_entry_preflight.py \
  --output-dir artifacts/economic_world_model/economic_wm_entry_preflight
```

The preflight runs the Stage-1 bridge-readiness sweep when no existing sweep report is supplied, then emits:

- `economic_wm_entry_preflight_report.json`
- `economic_wm_entry_preflight_report.md`

## Local scaffold entry criteria

The scaffold gate requires:

- Stage-1 bridge sweep status is `ok`
- at least five manifest shapes are exercised
- Stage-1 admissions, RLDS episodes, and LeRobot rows match in count
- at least one benchmark-ready example exists
- at least one shadow-only example exists
- every scenario report passes
- RLDS and LeRobot preserve benchmark-gate truth
- RLDS and LeRobot preserve future-training truth
- no promotion claim is made

## Current result

Latest local run:

- readiness class: `scaffold_ready_training_blocked`
- ready for scaffold: `true`
- ready for training: `false`
- scenario count: `5`
- benchmark-ready rows: `2`
- shadow-only rows: `3`

Training remains blocked by:

- `gpu_training_not_run`
- `provider_bringup_not_run`
- `non_stub_teacher_runtime_not_verified`
- `promotion_grade_benchmark_evidence_missing`

## What this allows next

It is now reasonable to start the **Economic WM scaffold**: input contracts, replay feature extraction, target rows, evaluation reports, and admission gates.

It is still not reasonable to claim an Economic WM has been trained or promoted.
