# Economic WM Phase-5 Local Prep

Date: 2026-05-21

## Purpose

This pass deepens native Economic WM ingestion beyond Stage-1 rows. It joins canonical lower-WM state, resource receipts, counterfactual/value-target refs, and temporal replay windows into local row families that a future trainer can consume without summary-only shortcuts.

It emits:

- `economic_wm_phase5_local_prep_manifest_v1.json`
- `economic_wm_datapack_composition_rows_v1.jsonl`
- `economic_wm_counterfactual_value_joins_v1.jsonl`
- `economic_wm_temporal_windows_v1.jsonl`
- `economic_wm_phase5_local_prep_v1.md`

## Executable path

```bash
python3 scripts/economic_world_model/prepare_economic_wm_phase5_local_prep.py \
  --output-dir artifacts/economic_world_model/economic_wm_phase5_local_prep \
  --corpus-manifest artifacts/economic_world_model/economic_wm_training_rows/economic_wm_training_corpus_manifest_v1.json \
  --rows artifacts/economic_world_model/economic_wm_training_rows/economic_wm_replay_feature_rows_v1.jsonl \
  --lower-wm-preflight artifacts/economic_world_model/economic_wm_lower_wm_consumption_preflight/economic_wm_lower_wm_consumption_preflight_v1.json \
  --lower-wm-consumption-rows artifacts/economic_world_model/economic_wm_lower_wm_consumption_preflight/economic_wm_canonical_consumption_rows_v1.jsonl \
  --resource-manifest artifacts/economic_world_model/economic_wm_resource_surfaces/economic_wm_resource_ingestion_manifest_v1.json \
  --resource-receipts artifacts/economic_world_model/economic_wm_resource_surfaces/economic_wm_resource_receipts_v1.jsonl \
  --queue-telemetry-surfaces artifacts/economic_world_model/economic_wm_resource_surfaces/economic_wm_queue_telemetry_surfaces_v1.jsonl
```

## Row families

- `economic_wm_datapack_composition_row_v1`: material provenance, functional contribution, canonical lower-WM refs, resource receipt refs, queue refs, and feature/target vectors.
- `economic_wm_counterfactual_value_join_row_v1`: structural joins between counterfactual eval refs, value target packs, and value ledgers.
- `economic_wm_temporal_window_row_v1`: local replay windows over composition rows so trainer scaffolds can check temporal shapes before GPU training exists.

## Current local result

The current local artifact run reports:

- `composition_row_count=5`
- `counterfactual_value_join_count=5`
- `temporal_window_count=3`
- `ready_for_trainer_scaffold=true`
- `ready_for_gpu_training=false`
- `promotion_eligible=false`

## Boundary

This is `phase5_local_prep_only`. It prepares native Economic WM training/evaluation structure, but it does not train, promote, invoke providers, grant live allocation authority, or mutate reward math.
