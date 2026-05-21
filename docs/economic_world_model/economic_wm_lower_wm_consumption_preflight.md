# Economic WM Lower-WM Consumption Preflight

Date: 2026-05-21

## Purpose

This preflight proves that Economic WM replay rows can consume canonical lower-WM state artifacts directly instead of relying only on summary sidecars. The required lower WMs are:

- `perception_grounding` -> `perception_grounding_world_state_v1`
- `sim_synth_physics` -> `sim_synth_physics_world_state_v1`
- `embodiment_actuation` -> `embodiment_actuation_world_state_v1`

## Landed surfaces

- `src/world_model/economic_world_model/lower_wm_consumption.py`
- `scripts/economic_world_model/prepare_economic_wm_lower_wm_consumption_preflight.py`
- `tests/test_economic_wm_lower_wm_consumption.py`

The output row type is `economic_wm_canonical_consumption_row_v1`. It preserves the original Economic WM replay row and adds direct canonical refs under:

```json
source_refs.canonical_lower_wm_refs
```

Fresh Stage-1/Economic row production now emits direct canonical lower-WM refs natively. When stale source rows do not yet include those refs, the preflight can still compile a local reference pack under `lower_wm_reference_pack/<row_id>/`. That fallback is structural preparation only: it is not a claim that the original lower-WM producers emitted those refs natively.

## Current artifact result

The local artifact run wrote:

- `artifacts/economic_world_model/economic_wm_lower_wm_consumption_preflight/economic_wm_lower_wm_consumption_preflight_v1.json`
- `artifacts/economic_world_model/economic_wm_lower_wm_consumption_preflight/economic_wm_canonical_consumption_rows_v1.jsonl`
- `artifacts/economic_world_model/economic_wm_lower_wm_consumption_preflight/economic_wm_lower_wm_consumption_preflight_v1.md`

Observed summary:

- `status=ok`
- `row_count=5`
- `compiled_reference_count=0`
- `direct_reference_count=15`
- `missing_reference_count=0`
- `ready_for_neural_manifest=true`
- `ready_for_training=false`
- `promotion_eligible=false`

## Boundary

This is a consumption and reference-integrity preflight. It does not run providers, train models, promote models, create benchmark evidence, or mutate frozen reward/trust/`w_econ`/lambda-controller math.

## Verify

```bash
python3 scripts/economic_world_model/prepare_economic_wm_lower_wm_consumption_preflight.py \
  --output-dir artifacts/economic_world_model/economic_wm_lower_wm_consumption_preflight \
  --corpus-manifest artifacts/economic_world_model/economic_wm_training_rows/economic_wm_training_corpus_manifest_v1.json \
  --rows artifacts/economic_world_model/economic_wm_training_rows/economic_wm_replay_feature_rows_v1.jsonl \
  --no-compile-missing-refs
```
