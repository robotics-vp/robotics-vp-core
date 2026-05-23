# Phase 6.4 Transport Advisory Runtime

Date: 2026-05-23

## Purpose

This pass closes the local Phase 6.4 scaffold for the Cross-WM Transport layer:
advisory runtime proposals, invocations, receipts, decomposed evaluation reports,
and shadow outcome join slots.

It is local runtime scaffolding only. It does not train bridge or receiver
weights, write weights, invoke providers, run hardware, grant live policy
authority, bypass target-WM receivers, mutate reward math, or promote transport
outputs.

## Landed surfaces

```text
src/world_model/transport/advisory_runtime.py
scripts/economic_world_model/run_phase6_transport_advisory_runtime.py
tests/test_wm_transport_phase64_runtime_eval.py
```

`src/world_model/transport/__init__.py` now exports the Phase-6.4 proposal,
invocation, receipt, decomposed-eval, shadow-join, and report helpers.

## Artifact path

```bash
python3 scripts/economic_world_model/run_phase6_transport_advisory_runtime.py \
  --output-dir artifacts/economic_world_model/phase6_transport_advisory_runtime \
  --no-run-dependencies
```

Current local artifact result:

- `proposal_count=20`
- `invocation_count=20`
- `receipt_count=20`
- `eval_report_count=20`
- `shadow_join_slot_count=20`
- `joined_shadow_outcome_count=10`
- `ready_for_decomposed_eval=true`
- `ready_for_training=false`
- `ready_for_gpu_training=false`
- `training_executed=false`
- `weights_written=false`
- `provider_executed=false`
- `hardware_executed=false`
- `live_policy_control=false`
- `reward_math_mutation=false`
- `promotion_eligible=false`

## Runtime surfaces

The advisory runtime emits:

- `TransportProposal`: source object ref, bridge contract, source exporter,
  target receiver, topology/causal fields, governance constraints, uncertainty
  profile, provenance, and denied authority.
- `TransportInvocation`: local advisory invocation record for the exporter ->
  bridge contract -> target receiver -> receipt sequence.
- `TransportReceipt`: survived topology, uncertainty, provenance, governance,
  receiver actionability, and shadow outcome join status.
- `WMTransportShadowOutcomeJoinSlot`: local structural shadow outcome link where
  an Economic-WM target can be associated with an existing shadow outcome
  receipt.
- `WMTransportDecomposedEvalReport`: bridge-only, receiver-only,
  downstream-only, joint, and interaction terms for each transport path.

## Evaluation decomposition

Every report separates:

- bridge-only quality from topology, uncertainty, and source export shape;
- receiver-only quality from target receiver actionability;
- downstream-only quality from available local structural shadow outcomes;
- joint quality from the existing Phase-6.2 round-trip receipt;
- interaction effect as the residual between joint behavior and independent
  bridge/receiver/downstream terms.

This is not a learned benchmark. It is a deterministic local evaluation surface
so later training and promotion work cannot hide receiver weakness behind bridge
quality or hide bridge weakness behind downstream heuristics.

## Shadow outcome posture

Economic-target transports can join to existing local structural shadow outcome
receipts when available. These joins are explicitly non-promotional:

- joined outcomes are local structural receipts, not hardware or provider
  outcomes;
- promotion-grade downstream benchmarks remain missing;
- shadow joins are evaluation labels/slots only, not authority grants.

When shadow outcomes are unavailable, the runtime still emits an explicit
`awaiting_shadow_outcome_receipt` slot for Economic-WM targets.

## Denied gates

Every proposal, invocation, receipt, eval report, and top-level manifest denies:

- training execution;
- weight writes;
- provider execution;
- hardware execution;
- live policy control;
- reward math mutation;
- target receiver bypass;
- promotion.

## Boundary

Still open after Phase 6.4:

- GPU-backed bridge and receiver training;
- cross-WM corpus-density evidence;
- provider or hardware transport evidence;
- topology/latency benchmarks;
- promotion-grade downstream benchmarks;
- bounded authority grants with rollback and demotion evidence.

## Closure Audit

`phase6_transport_closure_audit.md` is the owning local closure audit after
Phase 6.4. It confirms that no local transport contract/runtime surface remains
missing. The open work is evidence and training depth: corpus density,
GPU-backed bridge/receiver training, topology/latency benchmarks,
provider/hardware transport evidence, and promotion-grade downstream benchmarks.
