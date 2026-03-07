# Economic World Model Architecture Gap Analysis

## Scope

This repo should be prepared for a future multimodal economic world model that sits above a federation of specialist models. The target is not a monolithic model now. VLA, OpenVLA, and similar foundation-model paths remain external, pluggable, and advisory until the native contract, evidence, economics, and governance layers are mature enough to supervise a learned top layer honestly.

## Executive Read

The repo already has unusually strong typed scaffolding for objective tensors, econ tensors, pricing, ledgers, replay schemas, semantic fusion, and advisory regality. The biggest gap is not "missing ML." The biggest gap is the lack of one canonical runtime packet and one dense append-only event/evidence/governance spine that make the stack legible to itself across training, inference, replay, simulation, and deployment.

## Repo Crosswalk

| Precondition | Current status | Existing ground truth | Additive scaffolding to build now | Defer until later maturity |
| --- | --- | --- | --- | --- |
| 1. Canonical runtime contract layer | Partial | `src/objectives/runtime_builder.py`, `src/objectives/tensor.py`, `src/economics/econ_tensor.py`, `src/constraints/constraint_set.py`, `src/replay/schema.py`, `src/training/training_manifest.py` | `src/runtime/packets.py`, packet sidecars in `src/shadow_runtime/control_plane.py` and `src/replay/ingest.py`, then `ActionAdapterV2` and `ObservationAdapterV2` contracts | Making packet emission mandatory in default SAC/PPO or deployment paths |
| 2. Embodiment and action normalization middleware | Partial | `src/observation/adapter.py`, `src/embodiment/core.py`, `src/orchestrator/workcell_adapter.py`, `src/env/isaac_adapter.py`, `src/ingestion/x_humanoid_adapter.py`, `src/motor_backend/*` | `src/embodiment/registry.py`, `CapabilityProfile`, schema refs plus translator refs, `ActionAdapterV2` and `ObservationAdapterV2` specs | Unified humanoid/mobile/warehouse action semantics backed by production telemetry |
| 3. Temporal event spine | Weak partial | `src/ontology/models.py` `EpisodeEvent`, `src/ontology/store.py`, `src/logging/episode_logger.py`, `src/replay/schema.py`, `src/orchestrator/fast_slow_econ_bridge.py`, trajectory audits | `EventSpine`, `DecisionLedger`, `GovernanceTrace` sidecars, append-only temporal traces for replans/vetoes/adaptation admits/denies | Hard real-time deployment ingestion and fleet-scale event settlement |
| 4. Cross-model evidence fusion middleware | Partial-strong | `src/orchestrator/semantic_fusion.py`, `src/orchestrator/semantic_fusion_runner.py`, `src/vla/semantic_evidence.py`, `src/vision/map_first_supervision/*`, `src/vision/scene_ir_tracker/*`, `src/embodiment/core.py` | `EvidenceBus`, `BeliefState`, first-class evidence validity windows and disagreement summaries, `TeacherTrace` sidecars | Learned evidence calibration and router training from real deployment disagreement |
| 5. Dense economic supervision | Partial-strong | `src/economics/pricing_sentinel.py`, `src/economics/value_ledger.py`, `src/economics/functor.py`, `src/economics/reward_engine.py`, `src/valuation/datapack_schema.py`, `src/regality/shadow_nodes.py` | `CounterfactualEval`, dense local targets for adapt vs no-op / collect-data vs no-op / route A vs B, per-decision `ValueLedger` hooks | Customer-specific rebates/insurance curves backed by live deployments |
| 6. Learned control-plane before learned sovereign | Partial | `src/shadow_runtime/control_plane.py`, `src/orchestrator/*`, `src/phase_h/controller.py`, `src/phase_h/economic_learner.py`, `src/hrl/high_level_controller.py`, `src/orchestrator/queue_selection.py` | Router/planner/critic scaffolds that consume packets, event spine, and evidence bus without replacing specialists | A learned sovereign world model or end-to-end policy stack |
| 7. Safety / governance / constraint middleware | Partial-strong | `src/constraints/constraint_set.py`, `src/tfd/safety_rules.py`, `src/deployment/deploy_gate.py`, `src/regal/objective_integrity.py`, `src/regal/reward_safety.py`, `src/regality/*`, `src/valuation/valuation_verifier.py` | `GovernanceTrace`, veto rationale receipts, pricing-truth and semantic-policy sidecars, export to replay/trainable traces | Hard runtime blocking in default deployment loops without promotion/calibration proof |

## Detailed Gap Map

### 1. Canonical runtime contract layer

Already-implemented scaffolding:

- `src/objectives/runtime_builder.py` already defines `ObjectiveRuntimeRecord`, `ObjectiveRuntimeWindow`, and the explicit contract compile boundary.
- `src/objectives/tensor.py` provides a portable `ObjectiveTensor`.
- `src/economics/econ_tensor.py` provides a runtime `EconTensor`.
- `src/constraints/constraint_set.py` carries geometry, safety, semantic evidence, uncertainty, and trust metadata.
- `src/replay/schema.py` and `src/training/training_manifest.py` prove the repo already uses typed sidecars rather than free-form blobs.

Partial or mismatched pieces:

- `src/inference/demo_policy.py` builds observations, but it does not emit a canonical packet carrying objective/econ/constraint/evidence identity.
- `src/contracts/schemas.py` contains strict validation for closed-loop artifacts, but it is not the canonical runtime packet surface the rest of the stack shares.

Additive scaffolding to build now:

- `src/runtime/packets.py` for `ContractPacket` and `RuntimePacket`.
- `src/runtime/action_adapter_v2.py` and `src/runtime/observation_adapter_v2.py` for schema identity and timing semantics.
- Packet sidecar emission in `src/shadow_runtime/control_plane.py`, `src/replay/ingest.py`, and later `src/inference/demo_policy.py`.

Deferred later:

- Making the packet mandatory in default SAC/PPO hot paths before the additive sidecar path proves useful.
- Any learned packet compiler or implicit scalarization upstream of `ObjectiveCompiler`.

### 2. Embodiment and action normalization middleware

Already-implemented scaffolding:

- `src/observation/adapter.py` already normalizes heterogeneous signals into a typed `Observation`.
- `src/embodiment/core.py` computes advisory embodiment structure from scene tracks and semantic fusion.
- `src/orchestrator/workcell_adapter.py`, `src/env/isaac_adapter.py`, `src/ingestion/x_humanoid_adapter.py`, and `src/motor_backend/*` show multiple embodiment seams already exist.

What is missing:

- No shared `EmbodimentRegistry` or `CapabilityProfile` maps robot identity to normalized action/observation schemas, workspace bounds, and translator refs.
- No `ActionAdapterV2` / `ObservationAdapterV2` contract records latency, provenance, and embodiment-specific translators.

Build now:

- `src/embodiment/registry.py` for normalized capability metadata.
- `src/runtime/action_adapter_v2.py` and `src/runtime/observation_adapter_v2.py`.
- Sidecar wiring from `src/motor_backend/*`, `src/envs/*`, and `src/ingestion/*` into schema refs instead of ad hoc metadata.

Defer later:

- Full humanoid/mobile/warehouse normalization semantics until there is real hardware diversity and telemetry to validate them.

### 3. Temporal event spine

Already-implemented scaffolding:

- `src/ontology/models.py` includes `EpisodeEvent`.
- `src/ontology/store.py` persists append-only-ish JSONL events.
- `src/logging/episode_logger.py` captures per-step episode data.
- `src/replay/schema.py` and `src/orchestrator/fast_slow_econ_bridge.py` already think in windows and settlement records.
- Trajectory audits in `src/envs/workcell_env/trajectory_audit.py` and `src/envs/dishwashing_regal/trajectory_audit.py` provide event-count style summaries.

What is missing:

- No first-class append-only `EventSpine` or `DecisionLedger` for replans, vetoes, constraint-tightening, adaptation admits/denies, price ticks, and realized outcomes.
- Governance reasons, economic implications, and evidence disagreement still get summarized after the fact rather than logged as trainable temporal traces.

Build now:

- `src/runtime/event_spine.py` or `src/events/spine.py`.
- `src/governance/trace.py`.
- Replay/export glue so event-spine rows can sit beside `ReplayStepRecord` and `ReplayWindowRecord`.

Defer later:

- Live streaming ingestion from real robot telemetry or fleet buses.

### 4. Cross-model evidence fusion middleware

Already-implemented scaffolding:

- `src/orchestrator/semantic_fusion.py` and `src/orchestrator/semantic_fusion_runner.py` already fuse VLA and map-first evidence.
- `src/vla/semantic_evidence.py` provides a versioned sidecar format.
- `src/vision/map_first_supervision/*` and `src/vision/scene_ir_tracker/*` already serialize evidence-like artifacts.
- `src/embodiment/core.py` consumes semantic fusion as advisory context.

What is missing:

- There is no first-class `EvidenceBus` or `BeliefState` spanning VLA, map-first, SceneIR, embodiment, motion hierarchy, process reward, and future learned world-model outputs.
- Confidence, disagreement, and validity windows are still component-local.

Build now:

- `src/evidence/bus.py` and `src/evidence/belief_state.py`.
- `src/evidence/teacher_trace.py` for external VLA or foundation-model traces as sidecars.
- Additive evidence publication hooks in `src/orchestrator/semantic_fusion_runner.py`, `src/vla/rollout_labeler.py`, and map-first runners.

Defer later:

- Learned calibration of evidence reliability under deployment shift.

### 5. Dense economic supervision

Already-implemented scaffolding:

- `src/economics/pricing_sentinel.py` and `src/economics/value_ledger.py` already create auditable per-decision economic artifacts.
- `src/economics/functor.py` maps objective outcomes to econ tensors.
- `src/valuation/datapack_schema.py` has room for econ/objective sidecars and counterfactual metadata.
- `src/regality/shadow_nodes.py` already reasons about pricing truth and data credit evidence.

What is missing:

- No dedicated `CounterfactualEval` substrate for adapt vs no-op / collect-data vs no-op / route A vs B traces.
- No dense supervision target pack for constraint-violation cost, frontier gain from more data, uptime delta, uncertainty-adjusted price tick delta, or rebate deltas.

Build now:

- `src/economics/counterfactual_eval.py`.
- `src/economics/supervision_targets.py` or `src/economics/value_targets.py`.
- Tighter linkage from `PricingSentinel` and `ValueLedger` receipts to packet/event identities instead of summary-only joins.

Defer later:

- Customer-specific rebates, insurance adjustments, and full data-credit economics from live contracts.

### 6. Learned control-plane before learned sovereign

Already-implemented scaffolding:

- `src/shadow_runtime/control_plane.py` already expresses an additive control-plane proof.
- `src/orchestrator/*` contains queue selection, semantic orchestration, adaptation budgeting, and homeostatic planning.
- `src/phase_h/*` and `src/hrl/high_level_controller.py` provide controller-level scaffolding.

What is missing:

- No packet-native router/planner/critic interface deciding which specialist to invoke, whether to adapt, whether to collect more data, and whether to recompile a contract regime.
- No clear training/eval trace for these decisions.

Build now:

- `src/control_plane/router.py`, `src/control_plane/critic.py`, `src/control_plane/planner.py`.
- Training/eval harnesses that consume `RuntimePacket`, `EventSpine`, `BeliefState`, and `CounterfactualEval`.

Defer later:

- Any sovereign learned world model or end-to-end planner replacing the specialist federation.

### 7. Safety / governance / constraint middleware

Already-implemented scaffolding:

- `src/constraints/constraint_set.py` is already richer than motor-only safety checks.
- `src/tfd/safety_rules.py`, `src/deployment/deploy_gate.py`, `src/regal/objective_integrity.py`, `src/regal/reward_safety.py`, and `src/regality/*` already provide governance logic.
- `src/valuation/valuation_verifier.py` gives strong court-record style provenance verification.

What is missing:

- No explicit `GovernanceTrace` linking vetoes to rules, semantic evidence, price/trust implications, and future trainable labels.
- Governance outcomes are still distributed across regality artifacts, ledgers, and verifier outputs rather than one temporal trace.

Build now:

- `src/governance/trace.py`.
- Additive governance sidecars emitted from shadow control plane, replay ingest, and deployment gates.

Defer later:

- Hard online blocking in non-shadow execution without repeated promotion/calibration evidence.

## Dataset Bridge Layer

Current reusable pieces:

- `src/valuation/portable_datapacks.py`
- `src/videoio/dataset_spec.py`
- `src/vla/recap_dataset_builder.py`
- `src/process_reward/dataset_pairs.py`

Recommendation:

- Keep the internal schema richer than public bridges.
- Add export/import bridges under `src/dataset_bridges/rlds_bridge.py` and `src/dataset_bridges/lerobot_bridge.py`.
- Preserve econ tensors, objective tensors, governance traces, and teacher traces as richer internal sidecars even when the public bridge drops them.

## TeacherTrace Sidecars

Current sidecar-style evidence already exists:

- `src/vla/semantic_evidence.py`
- `src/valuation/vla_ingest.py`
- `src/encoders/teacher_adapter.py`

Recommendation:

- Standardize external foundation-model traces as `TeacherTrace` sidecars under an evidence namespace.
- Do not merge them into native truth schemas or overwrite internal objective/econ/governance records.

## Better Naming Choices Than Extending Existing Files

- Put `RuntimePacket` under `src/runtime/packets.py`, not `src/contracts/schemas.py`. The packet is a runtime binding artifact, not just static validation.
- Put `EmbodimentRegistry` under `src/embodiment/registry.py`, not `src/ontology/store.py`. It is normalization middleware, not ontology persistence.
- Put `EventSpine` and `GovernanceTrace` beside runtime/governance code, not inside `src/ontology/store.py`. The store is persistence; the spine is the canonical event model.
- Put `TeacherTrace` under an evidence namespace so external FM/VLA traces remain advisory.

## Training Backlog Crosswalk

The training migration backlog is useful, but it is not the same thing as middleware readiness. The most relevant entries in `scripts/TRAINING_MIGRATION_BACKLOG.json` map like this:

| Backlog script | Why it matters to this roadmap | Preconditions touched |
| --- | --- | --- |
| `train_high_level_controller.py` | Higher-level routing and control-plane training | 1, 6 |
| `train_orchestration_transformer_v1_curriculum.py` | Packet-native orchestration once runtime contracts are stable | 1, 4, 6 |
| `train_meta_transformer_synthetic.py` | Future top-layer planning/critic pretraining | 1, 5, 6 |
| `train_motion_hierarchy_node.py` | Better specialist evidence for motion hierarchy | 4 |
| `train_vla_recap_offline.py` | External teacher traces and advisory evidence | 4, TeacherTrace sidecars |
| `train_offline_with_local_synth.py` | Counterfactual evaluation and dense local econ supervision | 3, 5 |
| `train_trust_weighted_offline.py` | Governance-aware admission and weighting | 5, 7 |
| `train_w_econ_lattice.py` / `train_w_econ_lattice_from_J.py` | Important to economics, but Phase B frozen and not a first-pass roadmap entry point | Frozen zone; do not touch now |

## Immediate Conclusion

The repo is already good enough to support a disciplined "middleware republic" phase. The next step is not a learned world model. The next step is to finish the contract, embodiment, event, evidence, econ-supervision, and governance seams so that a future world model has truthful inputs and auditable targets.
