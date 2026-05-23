# Phase 6.0-6.2 Transport Scaffold

Date: 2026-05-21

## Purpose

This pass materializes the local Cross-WM Transport scaffold described in the
multi-WM roadmap. It covers Phase 6.0 through 6.2:

- **6.0 contracts and per-WM receiver/exporter posture**
- **6.1 transport and receiver training rows**
- **6.2 topology, round-trip, and uncertainty receipts**

It is structural local work only. It does not train bridge weights, invoke
providers, run hardware, promote outputs, grant live policy authority, or mutate
frozen reward/trust/`w_econ`/lambda-controller math.

## Landed surfaces

```text
src/world_model/transport/
  __init__.py
  bridge_contracts.py
  wm_transformers.py
  topology_metrics.py
  uncertainty.py
  roundtrip.py
  training_rows.py
  runtime.py

scripts/economic_world_model/prepare_phase6_transport_scaffold.py
tests/test_wm_transport_phase6_scaffold.py
```

## Artifact path

```bash
python3 scripts/economic_world_model/prepare_phase6_transport_scaffold.py \
  --output-dir artifacts/economic_world_model/phase6_transport_scaffold \
  --no-run-dependencies
```

Current local artifact result:

- `contract_count=20`
- `transformer_count=7`
- `roundtrip_receipt_count=20`
- `training_row_count=160`
- `status=ok`
- `ready_for_phase6_3_neural_scaffold=true`
- `ready_for_training=false`
- `promotion_eligible=false`

## 6.0 contract scaffold

`bridge_contracts.py` builds typed adjacent-WM contracts from the Phase-5.1
lower-WM maturity sweep and Phase-5 local prep manifest.

Current adjacent bridge families:

- `perception_grounding_to_sim_synth_physics`
- `sim_synth_physics_to_embodiment_actuation`
- `embodiment_actuation_to_economic`
- `lower_wm_bundle_to_economic`

Each contract requires:

- source endpoint;
- target endpoint;
- source exporter transformer id;
- target receiver transformer id;
- WM-transport ontology mapping;
- topology map;
- causal/dependency map;
- uncertainty profile;
- provenance payload;
- advisory-only authority;
- explicit blockers for GPU/provider/hardware/promotion evidence.

The contracts forbid raw hidden-state transport and preserve the rule that the
isomorphic bridge is middleware, not a mother-WM.

## Per-WM transformer scaffold

`wm_transformers.py` materializes exporter and receiver posture per WM. The
current registry includes WM-local transformers for:

- Perception / Grounding exporter
- Sim / Synth / Physics exporter and receiver
- Embodiment / Actuation exporter and receiver
- lower-WM bundle exporter
- Economic WM receiver

The receiver is evaluated separately from the bridge so a good bridge cannot
hide a target-WM actionability failure.

## 6.1 row materialization

`training_rows.py` emits row families for later transport and receiver training:

- `wm_transport_pair_row_v1`
- `wm_transport_roundtrip_row_v1`
- `wm_transport_topology_alignment_row_v1`
- `wm_transport_causal_dependency_row_v1`
- `wm_transport_uncertainty_calibration_row_v1`
- `wm_transport_downstream_yield_row_v1`
- `wm_transport_postmortem_counterfactual_row_v1`
- `wm_receiver_transformer_row_v1`

Rows are ready for trainer scaffolds, not GPU training. They carry explicit
blockers for corpus density, provider/hardware evidence, and promotion-grade
benchmarks.

## 6.2 evaluation receipts

`roundtrip.py`, `topology_metrics.py`, and `uncertainty.py` emit deterministic
local receipts for:

- source reconstruction shape;
- target receiver actionability shape;
- round-trip consistency;
- topology field coverage;
- causal/dependency edge coverage;
- semantic/actionability/governance coverage;
- uncertainty calibration proxy.

These are local structural receipts. They are not learned bridge evals or
promotion-grade downstream benchmarks.

## Training posture

The intended training decomposition remains:

```text
source per-WM exporter E_s
  -> isomorphic transport bridge B_s_t
  -> target per-WM receiver R_t
```

Training should be staged:

1. WM-local exporter/receiver pretraining;
2. bridge-only topology and causal alignment;
3. round-trip and receiver-actionability training;
4. downstream shadow shaping from counterfactual, postmortem, governance, and
   economic-yield receipts.

RL-style structures should live in bounded Economic WM allocator/governance and
later meta-governance lanes. They may shape transport through receipts, labels,
constraints, and sample weights. They must not turn the transport bridge into a
direct policy or bypass the target-WM receiver.

## Boundary

This pass does not close:

- Phase 6.3 neural manifest / trainer scaffold;
- GPU-backed transport or receiver training;
- cross-WM corpus-density evidence;
- topology/latency benchmark evidence;
- provider or hardware transport evidence;
- promotion-grade downstream benchmarks.

## Follow-up status

Later local scaffold passes closed Phase 6.3 and Phase 6.4 structurally:

- `phase6_transport_neural_scaffold.md` covers the neural manifest, loss ledger,
  and non-training trainer scaffold.
- `phase6_transport_advisory_runtime.md` covers advisory runtime proposals,
  invocations, receipts, decomposed eval reports, and shadow outcome join slots.

The remaining blockers are still training/evidence blockers: GPU-backed bridge
and receiver training, cross-WM corpus density, provider or hardware transport
evidence, topology/latency benchmarks, promotion-grade downstream benchmarks,
and any bounded live authority grant.
