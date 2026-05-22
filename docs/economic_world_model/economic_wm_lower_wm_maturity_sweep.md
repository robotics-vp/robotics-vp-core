# Economic WM Lower-WM Maturity Sweep

Date: 2026-05-21

## Purpose

This Phase-5.1 pass distinguishes structural canonical-ref readiness from production maturity. The Economic WM can now see lower-WM canonical state directly; this sweep checks whether those refs, reconstruction reports, benchmark gates, teacher traces, and resource receipts are mature enough for the next contract layer.

It emits:

- `economic_wm_lower_wm_maturity_sweep_v1.json`
- `economic_wm_lower_wm_maturity_rows_v1.jsonl`
- `economic_wm_lower_wm_maturity_sweep_v1.md`

## Executable path

```bash
python3 scripts/economic_world_model/sweep_economic_wm_lower_wm_maturity.py \
  --output-dir artifacts/economic_world_model/economic_wm_lower_wm_maturity_sweep \
  --phase5-prep artifacts/economic_world_model/economic_wm_phase5_local_prep/economic_wm_phase5_local_prep_manifest_v1.json \
  --lower-wm-preflight artifacts/economic_world_model/economic_wm_lower_wm_consumption_preflight/economic_wm_lower_wm_consumption_preflight_v1.json \
  --lower-wm-consumption-rows artifacts/economic_world_model/economic_wm_lower_wm_consumption_preflight/economic_wm_canonical_consumption_rows_v1.jsonl \
  --resource-manifest artifacts/economic_world_model/economic_wm_resource_surfaces/economic_wm_resource_ingestion_manifest_v1.json
```

## Current local result

The current local artifact run reports:

- `maturity_row_count=15`
- `structural_ready_count=15`
- `production_ready_count=0`
- `ready_for_phase6_contracts=true`
- `ready_for_production=false`
- `promotion_eligible=false`

The important distinction is intentional: canonical lower-WM refs are structurally ready for Phase-6 contract scaffolding, but production maturity is still blocked by real deployment/provider/hardware evidence and non-stub runtime gaps.

## Boundary

This sweep does not promote lower WMs, run providers, run hardware, train models, or grant Economic WM authority. It is a maturity/readiness diagnostic only.
