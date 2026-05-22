# Phase 6.3 Transport Neural Scaffold

Date: 2026-05-22

## Purpose

This pass closes the local Phase 6.3 scaffold: it adds the neural architecture
manifest, explicit loss ledger, and non-training trainer scaffold for the
Cross-WM Transport layer.

It remains scaffold-only. No bridge or receiver weights are trained, no provider
or hardware path is invoked, no live authority is granted, no promotion is made,
and frozen reward/trust/`w_econ`/lambda-controller math is untouched.

## Landed surfaces

```text
src/world_model/transport/
  neural_manifest.py
  losses.py
  training.py

scripts/economic_world_model/build_phase6_transport_neural_manifest.py
scripts/train_wm_transport_bridge_v0.py
tests/test_wm_transport_phase63_neural_scaffold.py
```

`src/world_model/transport/__init__.py` now exports the Phase-6.3 manifest,
loss, and trainer-scaffold helpers.

## Artifact paths

Build the neural manifest and loss ledger:

```bash
python3 scripts/economic_world_model/build_phase6_transport_neural_manifest.py \
  --output-dir artifacts/economic_world_model/phase6_transport_neural_manifest \
  --no-run-dependencies
```

Build the non-training trainer scaffold:

```bash
python3 scripts/train_wm_transport_bridge_v0.py \
  --output-dir artifacts/economic_world_model/phase6_transport_trainer_scaffold \
  --no-run-dependencies
```

Current local artifact result:

- `component_count=8`
- `loss_count=14`
- `training_row_count=160`
- `ready_for_trainer_scaffold=true`
- `cpu_smoke_forward_passed=true`
- `ready_for_training=false`
- `ready_for_gpu_training=false`
- `training_executed=false`
- `weights_written=false`
- `promotion_eligible=false`

## Neural components

The manifest defines eight learned-component slots:

| Component | Role |
| --- | --- |
| `typed_source_exporter_bank` | Per-WM exporters from source-native canonical state into the WM-transport ontology. |
| `isomorphic_transport_bridge` | Relation-aware bridge preserving topology, causality, uncertainty, provenance, and semantic compatibility across adjacent WMs. |
| `target_receiver_transformer_bank` | Per-WM receiver transformers that decode transported objects into target-native state/actionability surfaces. |
| `roundtrip_cycle_decoder` | Cycle-consistency / round-trip reconstruction scaffold. |
| `topology_causal_preservation_heads` | Separate topology and causal/dependency preservation heads. |
| `transport_uncertainty_calibrator` | Confidence, calibration, and abstention head. |
| `governance_actionability_classifier` | Governance, provenance, authority, and receiver-boundary classifier. |
| `downstream_shadow_transport_critic` | Offline shadow critic/ranker for downstream usefulness labels and sample weights. |

The training shape is still:

```text
source per-WM exporter E_s
  -> isomorphic transport bridge B_s_t
  -> target per-WM receiver R_t
```

The bridge is not a policy and not a universal decoder. It is middleware.

## Loss ledger

The loss ledger defines 14 losses:

- `source_export_reconstruction_loss`
- `translation_reconstruction_loss`
- `topological_contrastive_alignment_loss`
- `topology_preservation_loss`
- `causal_edge_preservation_loss`
- `target_native_reconstruction_loss`
- `receiver_actionability_loss`
- `roundtrip_consistency_loss`
- `uncertainty_nll_brier_ece_loss`
- `provenance_consistency_loss`
- `governance_constraint_satisfaction_loss`
- `downstream_yield_proxy_loss`
- `postmortem_counterfactual_improvement_loss`
- `contextual_bandit_shadow_ranking_loss`

The last three may use RL-style or bandit-style signals as offline labels,
constraints, rankings, or sample weights. They are explicitly not direct policy
RL for the bridge.

## Trainer scaffold

`train_wm_transport_bridge_v0.py` emits:

- trainer dataset contract;
- model component config;
- CPU smoke-forward report;
- trainer scaffold manifest;
- markdown summary.

The CPU smoke pass checks finite deterministic forwards over the current
transport training rows and finite proxy loss values. It does not initialize or
write real model weights.

## Boundary

Still open after Phase 6.3:

- GPU-backed bridge/receiver training;
- cross-WM corpus-density evidence;
- provider or hardware transport evidence;
- topology/latency benchmarks;
- promotion-grade downstream benchmarks;
- Phase 6.4 advisory runtime proposals over learned or calibrated bridge outputs.
