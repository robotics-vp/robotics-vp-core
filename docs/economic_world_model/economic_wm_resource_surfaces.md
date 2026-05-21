# Economic WM Resource Surfaces

Date: 2026-05-21

## Purpose

This Phase-5 local pass defines the resource and companion-compute surfaces the Economic WM needs before it can reason over compute, battery, thermal, latency, queues, and degraded modes as economic budget objects.

It emits:

- `economic_wm_resource_ingestion_manifest_v1.json`
- `economic_wm_resource_receipts_v1.jsonl`
- `economic_wm_companion_compute_contracts_v1.jsonl`
- `economic_wm_degraded_mode_runbooks_v1.jsonl`
- `economic_wm_queue_telemetry_surfaces_v1.jsonl`
- `economic_wm_resource_surfaces_v1.md`

## Executable path

```bash
python3 scripts/economic_world_model/prepare_economic_wm_resource_surfaces.py \
  --output-dir artifacts/economic_world_model/economic_wm_resource_surfaces \
  --corpus-manifest artifacts/economic_world_model/economic_wm_training_rows/economic_wm_training_corpus_manifest_v1.json \
  --rows artifacts/economic_world_model/economic_wm_training_rows/economic_wm_replay_feature_rows_v1.jsonl
```

## What this adds

The manifest defines Economic WM ingestion slots for:

- capacity receipts
- latency receipts
- thermal receipts
- battery receipts
- companion-compute contracts
- degraded-mode runbooks
- queue telemetry surfaces

The receipts turn compute and battery into allocatable budget objects for inference, routing, simulation, diffusion, data collection, conservation, and inferential work orders. The companion-compute contracts preserve the control split: Economic WM can emit advisory shadow work orders, lower WMs remain canonical state owners, live policy control is denied, and frozen reward/trust/`w_econ`/lambda math is not touched.

## Current local result

The current local artifact run reports:

- `row_count=5`
- `receipt_count=5`
- `contract_count=5`
- `runbook_count=5`
- `telemetry_surface_count=5`
- `ready_for_phase5_local_prep=true`
- `ready_for_training=false`
- `promotion_eligible=false`

## Boundary

This is `resource_receipt_schema_only` / local-prep infrastructure. It does not run GPU training, provider bring-up, live control, hardware execution, promotion, or reward-math mutation.
