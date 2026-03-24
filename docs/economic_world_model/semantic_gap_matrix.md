# Semantic Gap Matrix

Date: 2026-03-24

## Plain-English Translation

Before the first pass, the repo had semantic artifacts but not a shared semantic operating packet. Stage 1 emitted keyword tags, the fusion path emitted confidence-weighted class probabilities, and Stage 2 emitted richer symbolic proposals, but those lanes did not converge into one object-centric state that downstream code could pass around.

The first pass added a shared additive packet:

- `SemanticWorldModelState` in `src/world_model/semantic_world_model.py`
- runtime bridging in `src/semantic/runtime_backbone.py`
- Stage 1 materialization in `scripts/run_stage1_pipeline.py`
- rollout-fusion materialization in `src/orchestrator/semantic_fusion_runner.py`
- first-class carriage in `src/semantic/models.py`
- observation / condition / sampler consumption in `src/observation/adapter.py`, `src/observation/condition_vector_builder.py`, and `src/rl/episode_sampling.py`
- meta-node-oriented orchestration in `src/orchestrator/semantic_orchestrator_v2.py`

The second pass attacks the upstream quality gap inside the producers:

- `src/vision/scene_ir_tracker/io/datapack_frame_reader.py` now derives per-frame class labels and semantic object context from datapack metadata / `scene_spec`
- `src/vision/scene_ir_tracker/io/scene_tracks_runner.py` now emits semantically enriched `SceneTracks_v1` artifacts instead of only geometry-quality summaries
- `src/evidence/teacher_trace.py`, `src/vla/teacher_runtime.py`, and `src/vla/semantic_evidence.py` now carry structured teacher-side object / affordance / risk hints
- `src/world_model/semantic_world_model.py` now consumes those producer-side semantic fields instead of ignoring them

The third pass removes more of the remaining heuristics:

- `src/motor_backend/sensor_bundle.py` and `src/motor_backend/workcell_env_backend.py` now emit explicit `segmentation_label_map` and `scene_object_catalog` bundle metadata
- `src/vision/scene_ir_tracker/types.py`, `src/vision/scene_ir_tracker/tracker.py`, and `src/vision/scene_ir_tracker/kalman_track_manager.py` now preserve `source_instance_id` / `source_object_id` through stable tracking
- the SAM3D adapters now expose real-vs-stub/fallback backend modes and fail explicitly when real execution is requested without allowed fallbacks
- `src/vla/rollout_labeler.py` now preserves structured teacher semantics instead of flattening them back into plain tags

The semantic stack still does not govern the frozen Phase B baseline. It now behaves more like an orchestrator for meta-nodes, routing risk, recovery, affordance, fusion, ontology, and task-graph work into a shared advisory backbone.

The transformer boundary is no longer a pure gap. Both transformer callouts now consume semantic-world-model state directly and emit bounded execution packets with readiness/work-order surfaces. The remaining gap is therefore less about missing plumbing and more about replacing the current bounded heuristics with learned routing once execution evidence is dense enough.

## Runtime Topology

```text
Stage 1 video ref / manifest
  -> semantic seed tags
  -> BeliefState
  -> GovernedVideoWorldModel snapshot + hypotheses
  -> SemanticWorldModelState
  -> SemanticSnapshot
  -> OrchestratorAdvisory (meta-node weighted)
  -> datapack tags / signal bundle / sidecars

Rollout / semantic fusion
  -> fused semantic evidence
  -> EvidenceBus / BeliefState
  -> SemanticWorldModelState
  -> SemanticSnapshot
  -> OrchestratorAdvisory
  -> episode metadata sidecars

Stage 2 symbolic proposals
  -> SemanticSnapshot semantic_world_model field
  -> meta-node routing context
  -> observation / condition / sampler surfaces

Transformer callouts
  -> shared semantic-WM feature bridge
  -> MetaTransformer bounded routing packet
  -> OrchestrationTransformer bounded tool/activation packet
  -> pipeline / executor-facing work orders

Semantic runtime learning loop
  -> replay-backed semantic runtime rows
  -> shadow counterfactuals and regret labels
  -> meta-transformer runtime dataset
  -> orchestration runtime dataset
```

## Gap Matrix

| Semantic lane | Prior capability | Prior wiring status | Landed integration | Remaining gap |
| --- | --- | --- | --- | --- |
| Stage 1 semantics | Flat keyword tags plus governed-video modes | Wired locally inside Stage 1 only | Stage 1 now builds `SemanticWorldModelState`, `SemanticSnapshot`, and `OrchestratorAdvisory` sidecars; upstream datapack/SceneTracks production now also emits semantic context, explicit segmentation label maps where available, preserved source object IDs, per-track label provenance, and semantic density summaries so the world model becomes more often track-grounded instead of tag-only | Real sensor paths outside workcell-style bundles still need explicit label exporters and non-stub tracker execution |
| Runtime semantic fusion | Fused VLA + map-first class probabilities, evidence bus, belief state | Wired to embodiment and diagnostics, not to a shared semantic packet | Fusion runner now materializes the same semantic world-model/snapshot/advisory trio and writes their sidecars into episode metadata; producer-side teacher/runtime semantics plus explicit track source refs now give the fusion packet richer object/risk/affordance provenance instead of only class-prob tensors | Still depends on upstream track quality and real teacher availability |
| Stage 2 symbolic semantics | Ontology proposals, task refinements, semantic enrichments | Rich capabilities, mostly advisory/offline | `SemanticSnapshot` now has a first-class `semantic_world_model` field so Stage 2 outputs can join the same packet instead of living beside it | Ontology/task-graph proposals are still not auto-applied, by design |
| Semantic snapshot spine | Carrier for Stage 2 slices and econ/meta summaries | Thin, not central to runtime | Snapshot now carries `semantic_world_model` and runtime summary metadata | More producers still need to emit it natively |
| Orchestrator | Sampler/objective advice from enrichments and recap | Partially wired | V2 now reads world-model topology, capability scores, and meta-node scores; emits `meta_node_weights` instead of only shallow priority tags | Meta-node consumers beyond sampler/conditioning are still sparse |
| Transformer callouts | Packet stubs and context scaffolds | Not materially wired | `MetaTransformer` and `OrchestrationTransformer` now consume semantic-WM state directly, emit bounded routing/activation packets, and surface execution preconditions/work orders | Decision heuristics are still deterministic; learned routing is the next layer |
| Runtime learning / inferential loop | No canonical corpus tying semantic runtime state to later training | Missing | `src/orchestrator/semantic_runtime_learning.py` now builds canonical rows joining semantic WM, OpenVLA/teacher evidence, DINO/SceneTracks proxy evidence, outcomes, and counterfactuals; export script emits runtime datasets for both transformer lanes; `src/orchestrator/semantic_runtime_scorers.py` and `src/orchestrator/semantic_runtime_scorer_training.py` now add lightweight live-shadow scorers plus a heavier scorer-training/checkpoint path over the same row schema | Still needs denser real executed transformer work-order joins and later regal-style training integration |
| Observation / condition | Flattened semantic tag bag | Wired, but lossy | Adapter now exposes capability, topology, object, and meta-node signals; condition builder can derive skill mode from active meta nodes | Structured graph information still gets flattened for policy tensors |
| Sampler / curriculum | Novelty / recap / advisory weights | Wired | Sampler now boosts candidates using meta-node weights for risk, recovery, semantic refresh, and efficiency | No full graph-aware prioritizer yet |

## Capability Matrix

| Capability | Current source of truth | Runtime consumer | Status today |
| --- | --- | --- | --- |
| Object memory | `SemanticWorldModelState.objects` | snapshot, observation adapter, sidecars | Present, track-grounded when SceneTracks exist |
| Relation graph | `SemanticWorldModelState.relations` | snapshot metadata, orchestrator | Present, track-grounded when SceneTracks exist |
| Producer semantic density | `SceneTracks_v1` semantic summary + teacher semantic hints | world-model builder, datapack metadata, execution preconditions | Present, newly upstream-wired |
| Explicit object identity joins | sensor-bundle label maps + `track_source_object_id` | scene-tracks runner, world-model builder | Present for workcell/sensor-bundle paths |
| Stub/fallback visibility | adapter backend modes + execution preconditions | scene-tracks admission/training readiness | Present, newly explicit |
| Affordance grounding | object affordances + governed hypotheses + teacher/VLA cues | orchestrator, datapack tags | Present, partially grounded |
| Risk reasoning | object risk tags + constraints + meta nodes | orchestrator, sampler | Present, useful |
| Recovery reasoning | belief disagreement + recovery tags + hypotheses | orchestrator, condition builder | Present, useful |
| Fusion bridge | semantic fusion summary into world model | fusion runner, episode metadata | Present, newly wired |
| Stage 2 bridge | snapshot field + advisory metadata | snapshot/orchestrator | Partial |
| Meta-node orchestration | `SemanticMetaNode` scores | orchestrator, condition builder, sampler | Present, newly wired |

## Sources Used For Design Direction

- [V-JEPA 2](https://github.com/facebookresearch/vjepa2): video-predictive representations for action-conditioned temporal state.
- [OpenVLA](https://openvla.github.io/): open vision-language-action baseline for instruction-conditioned robot behavior.
- [PerAct](https://peract.github.io/): 3D action-centric transformer grounding for manipulation.
- [Open X-Embodiment](https://robotics-transformer-x.github.io/paper.pdf): broad cross-embodiment skill/data coverage.
- [CoTracker](https://github.com/facebookresearch/co-tracker): stable point/track grounding that informs track-centric semantic state.

## Remaining Recommendation

The next serious upgrade is now mostly about evidence density and promotion quality rather than missing substrate: keep densifying real executed transformer work-order traces in replay, then migrate `train_semantic_runtime_scorers.py` into the full regal/promotion training envelope so learned semantic reranking can be promoted on real execution evidence rather than only shadow labels.
