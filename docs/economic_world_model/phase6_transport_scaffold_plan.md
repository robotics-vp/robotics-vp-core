# Phase 6 Transport Scaffold Plan — 2026-05-21

## Status

Phase 5 is locally good enough to start Phase 6 contract work. In this context,
"good enough" means **structurally ready for Phase-6 contracts**, not fully
closed in the production or training sense.

The current local evidence says:

- Economic WM local prep rows exist.
- A non-training Economic WM trainer scaffold exists.
- Shadow execution and local structural outcome joins exist.
- Lower-WM maturity sweep reports structural Phase-6 readiness.
- No GPU training, provider bring-up, hardware execution, promotion, or live
  policy authority has been claimed.

## Follow-up Phase-5 closure carried into Phase 6.0

The Phase-6 scaffold should carry an explicit Phase-5 follow-up ledger so the
transport layer does not accidentally treat local structural readiness as
production maturity.

| Missing Phase-5 item | Current local state | Why it remains open | Future closure evidence |
| --- | --- | --- | --- |
| GPU-trained Economic WM components | Shape-checked trainer scaffold only | Estimator, dynamics, allocator, and governance helpers have not trained | training manifests, checkpoints, loss curves, heldout evals |
| Non-stub provider / teacher receipts | Provider contracts and runbook templates only | No provider runtime has produced promotion-grade evidence | provider manifests, teacher traces, replay artifacts |
| Production lower-WM maturity | Structural lower-WM refs exist; production-ready count is zero | Local refs are not benchmarked runtime/hardware truth | benchmark receipts, native runtime evidence, demotion gates |
| Real shadow outcome corpus | Local structural outcome receipts only | Shadow work orders have not been compared to real outcomes at scale | outcome ledger, counterfactual accuracy, Pareto-quality metrics |
| Real resource telemetry | Resource receipt schemas and synthetic/local examples | Capacity, latency, thermal, battery, queueing, and companion-compute receipts are not measured deployment streams | telemetry manifests, degraded-mode receipts, capacity traces |
| Promotion-grade economic authority | Advisory work orders only | Economic outputs do not control reward math, live policy, or deployment allocation | shadow benchmark pass, bounded authority grant, rollback tests |
| Phase-4 hardware/deployment enabler evidence | Contracts and surfaces exist for compute/resource consequences | Real-time control, sensor fusion, safety, SLAM, companion compute, and operator recovery are not hardware-proven | runtime/hardware logs, safety evidence, recovery drills |

Phase 6.0 can begin because these are **evidence and training blockers**, not
missing local contract-shape blockers. Phase 6 must preserve the distinction in
all receipts and docs.

## Phase 6 objective

Phase 6 builds a cross-WM isomorphic transport layer: typed middleware between
adjacent world models.

It is not:

- a mother-WM,
- a shared latent state that bypasses WM boundaries,
- a direct reward/control authority layer,
- or a reason to collapse each WM's native functionality into one generic
  representation.

The transport layer has two separable parts:

1. **WM-transport ontology** — the typed semantic, uncertainty, provenance,
   topology, causal, and governance contract for adjacent-WM mappings.
2. **Isomorphic transport bridge** — the compiled differentiable realization of
   that contract.

## Per-WM transformer requirement

The isomorphic transporter must not write directly into every WM. Each WM has a
specific function, state vocabulary, actionability boundary, and loss surface.
Therefore each adjacent transport path needs a WM-specific receiver transformer
and, where needed, a WM-specific exporter transformer.

```text
source canonical WM state
  -> source per-WM export transformer
  -> WM-transport ontology object
  -> isomorphic transport bridge
  -> target per-WM receiver transformer
  -> target-native canonical intake / advisory proposal
```

This gives the architecture three distinct layers:

| Layer | Role | What it must not do |
| --- | --- | --- |
| WM-local canonical state | Owns the native state and functionality of one WM | Pretend to be globally interchangeable |
| WM-transport ontology / bridge | Preserves typed topology, causality, uncertainty, provenance, and semantic compatibility across adjacent WMs | Replace the native WM state or decode straight into control authority |
| Per-WM receiver transformer | Converts transported objects into target-WM-native vocabulary, actionability constraints, rows, and receipts | Share one universal decoder across WMs with different jobs |

Initial receiver-transformer families:

| Target WM | Receiver transformer responsibility | Example local losses / checks |
| --- | --- | --- |
| Perception / Grounding | Convert belief or upstream evidence transport into observation, scene-track, calibration, and grounding surfaces | spatial/temporal alignment, calibration consistency, concept-presence confidence |
| Semantic WM | Convert perception transport into entity, relation, affordance, and semantic-density state | relation reconstruction, affordance coverage, semantic stability |
| Sim / Synth / Physics WM | Convert semantic transport into branch-ready physical state, constraints, and candidate-world parameters | topology preservation, physical plausibility, branch-yield prediction |
| Embodiment / Actuation WM | Convert simulated/physical transport into morphology, affordance, kinematic, control-rate, resource, and actionability state | kinematic feasibility, morphology consistency, actionability calibration |
| Economic WM | Convert lower-WM transport into resource, value, bottleneck, uncertainty, and allocation state | counterfactual value fit, Pareto-quality proxy, shadow-outcome correlation |
| Local meta-node surfaces | Convert economic/governance transport into node activation, conflict, veto, persistence, and intervention state | governance satisfaction, conflict calibration, hysteresis stability |

Design rule: bridge weights and per-WM receiver transformers may become learned,
but typed contracts and receiver boundaries stay explicit. The bridge preserves
compatibility; the receiver restores target-WM meaning.

## Training shape between transformers and transport bridge

Training should be decomposed across three coupled but separately measurable
modules:

```text
source exporter E_s
  -> transport bridge B_s_t
  -> target receiver R_t
```

The exporter and receiver are WM-local functionality layers. The bridge is the
adjacent-WM compatibility layer. They should share gradients only through typed
transport objects and receipt-backed objectives, not by erasing WM boundaries.

Recommended staged training shape:

1. **WM-local exporter / receiver pretraining**
   - Train `E_s` to encode source canonical state into the WM-transport ontology.
   - Train `R_t` to decode ontology/transport objects into target-native state,
     rows, actionability constraints, and receipts.
   - Keep adjacent WMs frozen while their exporter/receiver contracts are being
     learned.
2. **Bridge-only topology alignment**
   - Train `B_s_t` on ontology-to-ontology transport with topology, causal,
     uncertainty, provenance, and semantic-compatibility losses.
   - Use topological contrastive positives from valid adjacent-WM pairs and
     negatives from topology-breaking, causally invalid, provenance-invalid, or
     receiver-incompatible pairs.
3. **Round-trip and receiver-actionability training**
   - Train `E_s -> B_s_t -> R_t` and the reverse path where available.
   - Score both bridge fidelity and whether the target WM can actually use the
     received object.
4. **Downstream shadow shaping**
   - Use shadow outcomes, counterfactual improvement, governance satisfaction,
     and downstream economic yield as delayed labels or sample weights.
   - Do not let downstream yield become direct uncontrolled task-reward RL for
     the bridge.

A typical local loss ledger should separate:

```text
L_total =
  lambda_export   * L_source_export
+ lambda_bridge   * (L_transport + L_topology + L_causal + L_uncertainty)
+ lambda_receiver * (L_target_native_reconstruction + L_actionability)
+ lambda_cycle    * L_round_trip
+ lambda_gov      * L_governance_satisfaction
+ lambda_down     * L_downstream_shadow_proxy
```

Evaluation must report bridge-only, receiver-only, downstream-only, joint, and
interaction terms. This prevents a good bridge from hiding a weak receiver, and
prevents a strong target-WM heuristic from hiding a weak transport bridge.

## RL and control-learning placement

Transport and receiver layers are middleware. They should primarily train with
supervised, contrastive, predictive, calibration, topology-preservation,
round-trip, and counterfactual/postmortem losses. Direct RL on task reward is
not the default training regime for transport bridges.

RL-style structures belong in bounded Economic WM and later meta-governance
lanes:

- **Distributional Pareto multi-objective RL** for allocator/frontier learning,
  with return distributions over throughput, error, energy, compute, wear,
  safety, and value.
- **Constrained/offline RL** over shadow allocation outcomes, using explicit
  promotion gates and off-policy evaluation before any authority grant.
- **Augmented-Lagrangian / shadow-price updates** for scarce resource
  constraints such as compute, battery, thermal headroom, queue pressure, and
  provider capacity.
- **CVaR / coherent-risk critics** for tail-risk-aware allocation and
  deployment-cost sensitivity.
- **Finite-set receding-horizon allocation** for bounded routing problems such
  as replay-slice selection, sim-budget dispatch, work-order queues, and
  companion-compute placement.
- **Contextual-bandit / off-policy ranking loops** for cheap local shadow
  comparison before full allocator training exists.

For Phase 6 specifically, RL outputs should shape transport through receipts,
weights, constraints, and downstream labels. They should not give the transport
bridge direct policy authority, mutate frozen reward math, or bypass the
target-WM receiver transformer.

## Economic WM Phase-5 subsystem coverage check

The Economic WM subsystems articulated in the roadmap are wired into Phase 5 at
the local scaffold/contract level. None are trained or promoted yet.

| Roadmap subsystem | Phase-5 local wiring | Current status | Still missing |
| --- | --- | --- | --- |
| Datapack composition / mereotopological source encoder | `phase5_local_prep.py` emits datapack-composition rows; neural manifest includes `datapack_composition_network`; trainer scaffold consumes row shapes | Wired locally | GPU training and downstream improvement evidence |
| Economic State Estimator | `scaffold.py` emits `EconomicState`; neural manifest includes `economic_state_estimator` with DS3M / RED-SDS-style posture | Wired as state + neural scaffold | trained regime estimator, real provider/runtime receipts, slow-manifold validation |
| Economic Dynamics Model | counterfactual/value joins and temporal windows exist; neural manifest includes `economic_dynamics_model`; trainer scaffold has loss/config surfaces | Wired as training contract | trained counterfactual dynamics, real outcome corpus, forecast calibration |
| Economic Allocator / Compiler | `AllocationEnvelope`, shadow allocation eval, shadow work orders, resource surfaces, and `distributional_pareto_allocator` manifest exist | Wired shadow-only | trained distributional Pareto allocator, promotion-grade shadow benchmarks, live authority gate |
| Optional discrete receding-horizon allocator | neural manifest includes `discrete_receding_horizon_allocator`; shadow work orders expose finite routing targets | Partially wired as optional local solver scaffold | actual receding-horizon solve loop, regret metrics, queue/resource outcome corpus |
| Resource / compute / battery budgeting | `resource_surfaces.py` defines capacity, latency, thermal, battery, companion-compute, degraded-mode, and queue receipts | Wired locally as receipt schema + examples | measured deployment telemetry and provider/hardware receipts |
| Governance / Reciprocity compiler | allocation envelopes carry shaping fields, budget envelopes, and persistence annotations; neural manifest includes `governance_reciprocity_compiler`; shadow outcomes preserve denied authority | Wired advisory-only | trained downward compiler, lower-WM response corpus, meta-regal override trace, bounded authority promotion |

So the answer is yes at the scaffold/contract level: each Economic WM subsystem
from the roadmap has a Phase-5 local anchor. The honest gap is depth, not
coverage: training, real outcome density, provider/hardware evidence, and
promotion-grade authority remain future work.

## Phase 6.0 — contract and receiver scaffold

Local Phase 6 should start with additive transport package scaffolding:

```text
src/world_model/transport/
  __init__.py
  bridge_contracts.py
  wm_transformers.py
  topology_metrics.py
  uncertainty.py
  roundtrip.py
  training_rows.py
  neural_manifest.py
  losses.py
  runtime.py
```

Minimum local objects:

- `WMTransportBridgeContract`
- `TransportEndpoint`
- `TransportAuthority`
- `TransportOntologyMapping`
- `TransportTopologyMap`
- `TransportCausalMap`
- `TransportUncertaintyProfile`
- `TransportProvenance`
- `PerWMTransportExporter`
- `PerWMTransportReceiver`
- `TransportProposal`
- `TransportReceipt`
- `RoundTripReceipt`

Minimum invariants:

- adjacent-WM only;
- typed object refs only, no raw hidden-state transport;
- target receiver required for every bridge;
- uncertainty and provenance required;
- source and target WM identities preserved;
- advisory authority only until promotion evidence exists;
- no mutation of frozen Phase-B reward/trust/`w_econ`/lambda-controller math.

## Phase 6.1 — training rows

Training rows should be materialized before any GPU training claim:

- `wm_transport_pair_row_v1`
- `wm_transport_roundtrip_row_v1`
- `wm_transport_topology_alignment_row_v1`
- `wm_transport_causal_dependency_row_v1`
- `wm_transport_uncertainty_calibration_row_v1`
- `wm_transport_downstream_yield_row_v1`
- `wm_transport_postmortem_counterfactual_row_v1`
- `wm_receiver_transformer_row_v1`

Rows should consume Phase-5 and Phase-5.1 artifacts where possible:

- lower-WM maturity rows,
- supervision substrate records,
- shadow execution work orders,
- shadow outcome comparisons,
- datapack composition rows,
- counterfactual/value-target joins,
- temporal windows,
- resource and compute surfaces.

## Phase 6.2 — topology, round-trip, and uncertainty evaluation

Local evaluation should produce receipts even before training:

- bridge-only quality,
- receiver-only quality,
- downstream-WM-only quality,
- joint bridge + receiver quality,
- interaction term,
- round-trip reconstruction,
- topology preservation,
- causal/dependency preservation,
- uncertainty calibration,
- governance satisfaction,
- downstream economic-yield proxy.

The receiver-specific breakdown matters: a bridge may preserve topology while a
receiver fails to make it actionable for the target WM.

## Phase 6.3 — neural scaffold and losses

Create a manifest-producing trainer scaffold only. It should include CPU smoke
forwards and denied-promotion gates, not weight writes.

Current local Phase-6.3 artifacts (2026-05-22):

- `neural_manifest.py` emits an eight-component neural architecture manifest for source exporters, the isomorphic bridge, target receivers, round-trip decoder, topology/causal heads, uncertainty calibration, governance/actionability classification, and downstream shadow criticism.
- `losses.py` emits a 14-loss ledger covering export/translation reconstruction, topological contrastive alignment, topology/causal preservation, receiver actionability, round-trip consistency, uncertainty calibration, provenance, governance satisfaction, downstream-yield proxy, postmortem counterfactual improvement, and contextual-bandit shadow ranking.
- `training.py` and `train_wm_transport_bridge_v0.py` emit a non-training trainer scaffold with dataset contract, model component config, finite CPU smoke-forward report, denied-promotion gates, and no weight writes.
- Current local artifacts report `component_count=8`, `loss_count=14`, `training_row_count=160`, `cpu_smoke_forward_passed=true`, `ready_for_training=false`, and `promotion_eligible=false`.

Model components:

- source typed-object exporter bank,
- isomorphic transport bridge,
- target per-WM receiver transformer bank,
- round-trip cycle decoder,
- topology / causal preservation heads,
- uncertainty calibrator,
- governance/actionability classifier,
- downstream shadow transport critic.

Losses:

- source export / translation reconstruction loss,
- topological contrastive alignment loss,
- round-trip consistency loss,
- topology preservation loss,
- causal edge preservation loss,
- uncertainty NLL / Brier / ECE-style calibration loss,
- provenance consistency loss,
- receiver actionability loss,
- governance constraint loss,
- downstream yield proxy loss,
- postmortem counterfactual improvement loss,
- contextual-bandit shadow ranking loss.

Topological contrastive training should use valid adjacent-WM transport pairs as
positives and topology-breaking, causally invalid, provenance-invalid, or
receiver-incompatible pairs as negatives. RL-style rows may shape transport only
as offline labels, rankings, constraints, or sample weights; direct task-reward
RL must not turn the bridge into a policy.

## Phase 6.4 — advisory runtime

Runtime should emit transport proposals and receipts only:

- no live policy control,
- no reward mutation,
- no direct hardware authority,
- no production promotion,
- no bypass around target-WM receiver transformers.

The runtime story is useful when it can answer:

1. What source WM object was transported?
2. Which bridge contract applied?
3. Which target receiver transformed it?
4. What topology, uncertainty, provenance, and governance properties survived?
5. Did the adjacent WM find the object actionable?
6. What downstream result or shadow outcome should later validate the transport?

## Local Phase-6 closure criteria

Local Phase 6 should be considered structurally closed only when:

1. bridge contracts exist for the initial adjacent WM pairs;
2. every bridge has explicit source exporter and target receiver posture;
3. training rows exist for transport and receiver-transformer learning;
4. topology, round-trip, uncertainty, governance, and receiver-actionability
   receipts exist;
5. a non-training neural scaffold emits model component configs, losses, CPU
   smoke-forward evidence, and denied-promotion gates;
6. advisory runtime can produce transport proposals without authority escalation;
7. docs and tests preserve the Phase-5 follow-up closure ledger.

Remaining after local Phase 6 will be corpus density, GPU-backed training,
provider/hardware evidence, latency evaluation, and promotion-grade downstream
benchmarks.
