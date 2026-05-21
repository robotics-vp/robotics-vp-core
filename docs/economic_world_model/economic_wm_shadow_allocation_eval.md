# Economic WM Shadow Allocation Eval

Date: 2026-05-21

## Purpose

This evaluator is the first advisory Economic WM allocation surface. It reads the scaffold, local row-corpus manifest, and replay feature rows, then scores candidate next allocations without executing them.

It emits:

- `economic_wm_shadow_allocation_eval_v1.json`
- `economic_wm_shadow_allocation_eval_v1.md`

## Executable path

```bash
python3 scripts/economic_world_model/evaluate_economic_wm_shadow_allocations.py \
  --output-dir artifacts/economic_world_model/economic_wm_shadow_allocation_eval \
  --scaffold-report artifacts/economic_world_model/economic_wm_scaffold/economic_wm_scaffold_report_v1.json \
  --corpus-manifest artifacts/economic_world_model/economic_wm_training_rows/economic_wm_training_corpus_manifest_v1.json \
  --rows artifacts/economic_world_model/economic_wm_training_rows/economic_wm_replay_feature_rows_v1.jsonl
```

If row inputs are missing, the script can run the row materializer first.

## Current local result

The current shadow eval recommends:

- `prepare_teacher_provider_evidence_contracts`

The recommendation is driven by row-level teacher/runtime and provider gaps. It is not a provider bring-up claim; it only says the next local allocation of development effort should prepare the evidence contracts and blocker surfaces needed when GPU/provider capacity exists.

Other candidates remain visible:

- `curate_benchmark_ready_replay`
- `close_shadow_gap_replay`
- `run_gpu_training` — denied by the allocation envelope and training blockers

## Boundary

This is `authority_class=shadow_eval_only`:

- no allocation is executed
- no reward math is mutated
- no model is trained or promoted
- GPU/provider work remains blocked until it literally runs
