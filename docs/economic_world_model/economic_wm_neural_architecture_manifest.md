# Economic WM Neural Architecture Manifest

Date: 2026-05-21

## Purpose

The Economic WM neural architecture manifest names the learned components, inputs, outputs, losses, gates, and blockers that can be built before any GPU-backed training run exists. It is a topology and training-contract artifact, not a weights artifact.

## Landed surfaces

- `src/world_model/economic_world_model/neural_architecture_manifest.py`
- `scripts/economic_world_model/build_economic_wm_neural_architecture_manifest.py`
- `tests/test_economic_wm_neural_architecture_manifest.py`

## Planned neural components

| Component | Role | Current plane |
| --- | --- | --- |
| `datapack_composition_network` | Encodes material provenance, functional contribution, lineage topology, and marginal utility seeds. | `gpu_train_required` |
| `economic_state_estimator` | Estimates regimes, bottlenecks, slow-manifold projections, and shadow-price seeds from canonical lower-WM receipts. | `gpu_train_required` |
| `economic_dynamics_model` | Forecasts economic transitions and counterfactual outcomes under candidate allocations. | `gpu_train_required` |
| `distributional_pareto_allocator` | Produces Pareto frontier slices, risk fields, shadow-price fields, and training-slice priorities. | `gpu_train_required` |
| `discrete_receding_horizon_allocator` | Solves bounded finite-set work-order and resource-routing choices, with optional learned warm starts later. | `local_solver_scaffold_gpu_training_optional` |
| `governance_reciprocity_compiler` | Converts economic allocation outputs into downward shaping, budget envelopes, persistence annotations, and admissible regions. | `gpu_train_required` |

## Current artifact result

The local artifact run wrote:

- `artifacts/economic_world_model/economic_wm_neural_architecture_manifest/economic_wm_neural_architecture_manifest_v1.json`
- `artifacts/economic_world_model/economic_wm_neural_architecture_manifest/economic_wm_neural_architecture_manifest_v1.md`

Observed summary:

- `component_count=6`
- `gpu_train_required_count=5`
- `ready_for_training_scaffold=true`
- `ready_for_gpu_training=false`
- `gpu_training_ready=false`
- `provider_bringup_ready=false`
- `promotion_eligible=false`
- `reward_math_mutation=false`

## Boundary

This manifest does not instantiate model weights, run training, invoke providers, promote outputs, or grant Economic WM authority over lower WMs. Component specs stay `training_ready=false`, `promotion_eligible=false`, and `authority_class=neural_scaffold_only` until real GPU/provider/runtime/benchmark evidence exists.

## Verify

```bash
python3 scripts/economic_world_model/build_economic_wm_neural_architecture_manifest.py \
  --output-dir artifacts/economic_world_model/economic_wm_neural_architecture_manifest \
  --lower-wm-preflight artifacts/economic_world_model/economic_wm_lower_wm_consumption_preflight/economic_wm_lower_wm_consumption_preflight_v1.json
```
