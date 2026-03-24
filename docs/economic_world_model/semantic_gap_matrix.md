# Semantic Gap Matrix

Date: 2026-03-22

## Plain-English Translation

Before this pass, the repo had semantic artifacts but not a shared semantic operating packet. Stage 1 emitted keyword tags, the fusion path emitted confidence-weighted class probabilities, and Stage 2 emitted richer symbolic proposals, but those lanes did not converge into one object-centric state that downstream code could pass around.

This pass adds a shared additive packet:

- `SemanticWorldModelState` in `src/world_model/semantic_world_model.py`
- runtime bridging in `src/semantic/runtime_backbone.py`
- Stage 1 materialization in `scripts/run_stage1_pipeline.py`
- rollout-fusion materialization in `src/orchestrator/semantic_fusion_runner.py`
- first-class carriage in `src/semantic/models.py`
- observation / condition / sampler consumption in `src/observation/adapter.py`, `src/observation/condition_vector_builder.py`, and `src/rl/episode_sampling.py`
- meta-node-oriented orchestration in `src/orchestrator/semantic_orchestrator_v2.py`

The semantic stack still does not govern the frozen Phase B baseline. It now behaves more like an orchestrator for meta-nodes, routing risk, recovery, affordance, fusion, ontology, and task-graph work into a shared advisory backbone.

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
```

## Gap Matrix

| Semantic lane | Prior capability | Prior wiring status | Landed integration | Remaining gap |
| --- | --- | --- | --- | --- |
| Stage 1 semantics | Flat keyword tags plus governed-video modes | Wired locally inside Stage 1 only | Stage 1 now builds `SemanticWorldModelState`, `SemanticSnapshot`, and `OrchestratorAdvisory` sidecars; when manifests provide `SceneTracks_v1` / teacher / VLA evidence, the world model becomes track-grounded instead of tag-only | Still depends on manifest-provided grounding and still falls back to heuristics when those artifacts are absent |
| Runtime semantic fusion | Fused VLA + map-first class probabilities, evidence bus, belief state | Wired to embodiment and diagnostics, not to a shared semantic packet | Fusion runner now materializes the same semantic world-model/snapshot/advisory trio and writes their sidecars into episode metadata; it also feeds SceneTracks plus teacher/VLA evidence into world-model construction | Still depends on upstream track quality and teacher availability |
| Stage 2 symbolic semantics | Ontology proposals, task refinements, semantic enrichments | Rich capabilities, mostly advisory/offline | `SemanticSnapshot` now has a first-class `semantic_world_model` field so Stage 2 outputs can join the same packet instead of living beside it | Ontology/task-graph proposals are still not auto-applied, by design |
| Semantic snapshot spine | Carrier for Stage 2 slices and econ/meta summaries | Thin, not central to runtime | Snapshot now carries `semantic_world_model` and runtime summary metadata | More producers still need to emit it natively |
| Orchestrator | Sampler/objective advice from enrichments and recap | Partially wired | V2 now reads world-model topology, capability scores, and meta-node scores; emits `meta_node_weights` instead of only shallow priority tags | Meta-node consumers beyond sampler/conditioning are still sparse |
| Observation / condition | Flattened semantic tag bag | Wired, but lossy | Adapter now exposes capability, topology, object, and meta-node signals; condition builder can derive skill mode from active meta nodes | Structured graph information still gets flattened for policy tensors |
| Sampler / curriculum | Novelty / recap / advisory weights | Wired | Sampler now boosts candidates using meta-node weights for risk, recovery, semantic refresh, and efficiency | No full graph-aware prioritizer yet |

## Capability Matrix

| Capability | Current source of truth | Runtime consumer | Status today |
| --- | --- | --- | --- |
| Object memory | `SemanticWorldModelState.objects` | snapshot, observation adapter, sidecars | Present, track-grounded when SceneTracks exist |
| Relation graph | `SemanticWorldModelState.relations` | snapshot metadata, orchestrator | Present, track-grounded when SceneTracks exist |
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

The next serious upgrade is not another advisory module. It is improving upstream SceneTracks class labeling, calibration quality, and teacher/VLA semantic evidence coverage so the now-grounded semantic packet is populated more often and with less stub/fallback behavior.
