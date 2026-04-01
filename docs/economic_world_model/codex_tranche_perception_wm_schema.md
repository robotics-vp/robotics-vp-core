# Codex Tranche: Perception / Grounding WM Canonical State Schema

## Classification

- **Type**: contract/schema update + scaffolding implementation
- **Branch**: `codex/multi-wm-architecture-plan`
- **Priority**: HELD — prepared as adjacent spec work, not the active implementation target
- **Sequencing**: this tranche activates AFTER `codex_tranche_sim_synth_closure.md` is structurally complete
- **Timeline**: target completion before end of June 2026 (after Sim/Synth/Physics closure)

## Motivation

The Perception / Grounding WM canonical state surface is the single highest-risk structural gap. Every downstream WM (Sim/Synth/Physics, Embodiment, Economic) will eventually consume perception canonical state. If this schema is wrong or absent when September 2026 training begins, all lower-WM training inherits a flawed representational substrate.

The current branch has strong Sim/Synth/Physics WM state objects (`SimSynthPhysicsWorldState`, etc.) and those already reference perception inputs (belief state, scene tracks, semantic WM state). But there is no canonical `PerceptionGroundingWorldState` that owns the typed surface those consumers actually need.

## What Codex Should Implement

### 1. Canonical State Schema

Create `src/world_model/perception_grounding/state.py` with frozen dataclass state objects:

```
PerceptionGroundingWorldState
  version: str
  timestamp: str
  scene_object_state: SceneObjectState
  scene_relation_state: SceneRelationState
  temporal_continuity_state: TemporalContinuityState
  concept_segmentation_state: ConceptSegmentationState
  grounding_confidence_state: GroundingConfidenceState
  provider_truth_state: ProviderTruthState
  perception_receipts: list[PerceptionReceipt]
```

Key sub-objects:

**SceneObjectState**: Per-object canonical state
- `objects: list[CanonicalObjectToken]`
- Each `CanonicalObjectToken` carries:
  - `object_id: str` (stable identity)
  - `class_label: str` (concept label from SAM/provider or SceneTracks)
  - `position_3d: tuple[float, float, float] | None` (metric if available)
  - `bbox_2d: tuple[float, float, float, float] | None`
  - `mask_confidence: float`
  - `grounding_source: str` ("sam3", "scene_tracks", "stub", "fused")
  - `semantic_embedding: list[float] | None` (from DINOv2/SigLIP if available)
  - `affordance_hints: list[str]`
  - `risk_hints: list[str]`
  - `temporal_track_id: str | None`
  - `last_observed_timestamp: str`
  - `observation_count: int`
  - `confidence: float`

**SceneRelationState**: Pairwise relations
- `relations: list[ObjectRelation]`
- Each `ObjectRelation` carries:
  - `source_object_id: str`
  - `target_object_id: str`
  - `relation_type: str` (spatial_adjacency, contact, containment, occlusion, support, affordance)
  - `confidence: float`
  - `source: str` (provider, inferred, fused)

**TemporalContinuityState**: Scene persistence
- `active_track_count: int`
- `lost_track_count: int`
- `reidentified_count: int`
- `temporal_stability_score: float`
- `scene_change_rate: float`

**ConceptSegmentationState**: SAM/provider concept segmentation summary
- `provider: str` ("sam3", "sam3.1", "stub", "unavailable")
- `provider_available: bool`
- `prompt_mode: str` ("text", "exemplar", "box", "point", "mask", "none")
- `concept_count: int`
- `mask_count: int`
- `average_mask_confidence: float`
- `multiplex_active: bool` (SAM 3's object-multiplex for high-count scenes)
- `memory_mode: str` ("none", "per_frame", "video_tracking")

**GroundingConfidenceState**: Overall grounding quality
- `grounding_quality_score: float` (0-1)
- `semantic_density: float` (fraction of objects with semantic embeddings)
- `spatial_coverage: float` (fraction of scene with depth/3D grounding)
- `provider_coverage: dict[str, float]` (per-provider coverage fractions)

**ProviderTruthState**: Honest provider availability
- `providers: list[ProviderStatus]`
- Each `ProviderStatus`:
  - `provider_id: str`
  - `available: bool`
  - `mode: str` ("real", "unavailable", "stub", "planning_only")
  - `last_invocation_timestamp: str | None`
  - `calibration_quality: float | None`

**PerceptionReceipt**: Typed receipts
- `GroundingReceipt` (per-frame or per-window grounding quality + provider truth)
- `SegmentationReceipt` (SAM invocation + mask quality + concept hit rate)
- `TemporalTrackingReceipt` (track lifecycle + persistence + re-identification)
- `CalibrationReceipt` (camera/depth calibration status and quality)

All state objects: frozen dataclass, versioned, `to_dict()` serialization, stable ID via SHA256.

### 2. Compiler

Create `src/world_model/perception_grounding/compiler.py`:

- `compile_perception_grounding_world_state(...)` that takes:
  - `scene_tracks` (from SceneTracks runner)
  - `belief_state` (from evidence bus)
  - `semantic_wm_state` (from existing semantic world model)
  - `teacher_traces` (from VLA teacher runtime, optional)
  - `provider_statuses` (SAM, DINOv2, depth, V-JEPA 2 availability)
  - `embodiment_context` (optional, for body-aware perception)
- Produces `PerceptionGroundingWorldState`
- Uses existing semantic world model objects/relations as initial input
- Enriches with provider-truthed segmentation and grounding when available
- Falls back honestly to reduced-quality state when providers are unavailable
- Emits typed receipts

### 3. Adapter Interfaces

Create `src/world_model/perception_grounding/adapters/`:

- `sam_adapter.py`: typed adapter for SAM 3/3.1 provider
  - `describe_sam_provider_status(...)` → `ProviderStatus`
  - `compile_concept_segmentation_state(...)` → `ConceptSegmentationState`
  - Mode: `real | unavailable | stub` with explicit posture
  - When unavailable: emit honest `ProviderStatus` receipt, do not silently degrade

- `scene_tracks_adapter.py`: adapter from existing SceneTracks into canonical object state
  - Reads `SceneTracks_v1` artifacts
  - Produces `list[CanonicalObjectToken]` + `list[ObjectRelation]`
  - Preserves `source_instance_id` / `source_object_id` through tracking

- `depth_adapter.py`: adapter for depth provider (DepthAnythingV2/UniDepth)
  - Enriches object tokens with metric 3D positions when depth is available

- `semantic_evidence_adapter.py`: adapter from existing semantic evidence payloads
  - Reads `build_vla_semantic_evidence_payload()` outputs
  - Enriches object tokens with teacher-sourced semantic hints

### 4. Receipt and Replay Integration

- Perception receipts should be consumable by:
  - `SimSynthPhysicsWorldState` compiler (already references belief state; should now reference perception canonical state)
  - Replay ingest
  - Training manifests
  - Economic WM ingestion (later)

- Add a typed adapter in `src/world_model/sim_synth_physics/adapters/perception_inputs.py`:
  - Reads `PerceptionGroundingWorldState`
  - Produces the perception-side inputs that the sim/synth/physics compiler currently derives from `BeliefState` and `SemanticWorldModelState`
  - This is the first cross-WM typed contract boundary

### 5. Semantic Bridge Contract (Schema Only)

Create `src/world_model/perception_grounding/bridge_contracts.py`:

- Define the **typed interface** for the Semantic→SimSynthPhysics bridge:
  - Input: `PerceptionGroundingWorldState` (specifically the object tokens + relations)
  - Output: `SimSynthPhysicsSemanticView` (physics-weighted object/relation subset)
  - This is a schema-only contract; the learned bridge implementation comes later

- Define the **typed interface** for the Semantic→Embodiment bridge:
  - Input: `PerceptionGroundingWorldState` + `EmbodimentContext`
  - Output: `EmbodimentSemanticView` (body-centric affordance/spatial projection)
  - Schema-only contract

### 6. Tests

- `tests/world_model/perception_grounding/test_state.py`: state construction, serialization, versioning
- `tests/world_model/perception_grounding/test_compiler.py`: compiler smoke with mock inputs
- `tests/world_model/perception_grounding/test_adapters.py`: adapter smoke tests
- `tests/world_model/perception_grounding/test_bridge_contracts.py`: schema validation

### 7. Promotion Posture

- All helper/learned seams in this tranche should use the existing `disabled|auto|required` promotion pattern from `src/world_model/sim_synth_physics/promotion.py`
- Initially all learned seams are `disabled` (heuristic prior only)
- Provider adapters use `real | unavailable | stub` posture
- Receipts trace promotion stage and provider truth

## What Codex Should NOT Do

- Do not implement the learned Graph Transformer semantic abstraction yet (that comes when providers are actually available for training data)
- Do not implement the semantic-to-WM bridge neural layers yet (schema contracts only)
- Do not modify existing `SimSynthPhysicsWorldState` compiler internals; add a new adapter that sits between perception canonical state and the existing compiler inputs
- Do not touch frozen Phase B baseline
- Do not create new stub providers that silently masquerade as capability

## What Should Be Held For Later

- Neural semantic abstraction implementation (Graph Transformer over object tokens)
- Learned semantic-to-WM bridge layers
- SAM 3/3.1 real provider bring-up (requires GPU host)
- V-JEPA 2 real provider bring-up (requires GPU host + upstream repo)
- Multi-rate observation infrastructure (separate tranche)
- Depth provider integration (separate provider bring-up item)

## Verification

After implementation:
```bash
python3 -m compileall src/ && pytest tests/ -v
pytest tests/world_model/perception_grounding/ -v
```

## Doctrinal Compliance

- [ ] All state objects are frozen dataclasses with versioning and to_dict()
- [ ] Provider adapters use real-or-unavailable posture
- [ ] Learned seams have disabled|auto|required promotion pattern
- [ ] Receipts are emitted for every compilation
- [ ] No new stubs introduced without explicit stub posture
- [ ] Cross-WM bridge contracts are schema-only (no premature implementation)
- [ ] Tests pass

## Required Handoff Artifact

After completing this tranche, Codex must emit:

**`docs/economic_world_model/claude_to_comment_on.md`**

containing:
- what was implemented
- what changed topologically (new WM boundary, new cross-WM contract)
- which modules were added or altered
- what tests were added
- what remains missing from the Perception/Grounding WM
- what doctrinal questions are open (e.g., should the perception compiler consume embodiment context? how should temporal continuity interact with SIMA-2 action segmentation?)
- whether the existing Sim/Synth/Physics WM compiler needs updates to consume the new perception adapter
- whether docs/roadmap should change
