# Map-First Pseudo-Supervision (MapFirstSupervision_v1)

## Overview
Map-First Pseudo-Supervision is a deterministic, CPU-only node that consumes `SceneTracks_v1` and produces numpy-only artifacts for downstream training. It builds a world-frame static map from tracked entities, computes inconsistency residuals for dynamic evidence, optionally densifies into depth targets, and stabilizes semantics without running any model inference.

**Placement:**
`SceneTracks_v1` -> `MapFirstPseudoSupervisionNode` -> (semantic pseudo-GT, depth targets, 3D boxes, dynamic/static evidence) + summary metadata

## Inputs
- **SceneTracks_v1** (numpy-only): `poses_R`, `poses_t`, `scales`, `visibility`, `occlusion`, etc.
- **Optional geometry sidecar**: external point clouds/meshes if available; otherwise primitives (box/sphere) are sampled.
- **Optional semantics**:
  - Per-frame per-entity class probabilities `(T, K, C)`
  - Or per-frame per-point labels (sparse) for map stabilization
  - Or `VLA_SemanticEvidence_v1` sidecar (see below)
- **Optional camera metadata**:
  - `camera_params` (preferred, `CameraParams`)
  - or `camera_intrinsics` + `camera_extrinsics` dicts

### VLA_SemanticEvidence_v1 (sidecar)
Map-First consumes VLA evidence but does not fuse it. The sidecar is versioned and numpy-only.

Required keys (prefix `vla_semantic_evidence_v1/`):
- `version` `"v1"`
- `class_probs` `(T, K, C)` float32

Optional keys:
- `confidence` `(T, K)` float32
- `provenance` (JSON-serializable dict)
- `embed` `(T, K, D)` float32
- `track_ids` `(K,)` strings (for alignment)

Alternate form:
- `instances`: per-frame list keyed by `track_id` with `class_probs` entries

## Outputs (MapFirstSupervision_v1 NPZ)
All outputs are numpy-only, versioned, and namespaced with `map_first_supervision_v1/`.

Required keys:
- `dynamic_evidence` `(T, K)` float32
- `dynamic_mask` `(T, K)` bool
- `residual_mean` `(T, K)` float32
- `boxes3d` `(T, K, 7)` float32 `[x,y,z,dx,dy,dz,yaw]`
- `confidence` `(T, K)` float32

Optional keys:
- `densify_depth` `(T, H, W)` float16 (if camera params and depth mode)
- `densify_mask` `(T, H, W)` uint8
- `densify_world_points` `(T, M, 3)` float16 (if no camera params or world mode)
- `densify_world_mask` `(T, M)` uint8
- `semantics_stable` `(T, K, C)` float16
- `meta_semantics_stability` `(T, K)` float32
- `static_map_centroids`, `static_map_counts`, `static_map_semantics` (debug/analysis)
- `evidence_dynamics_score`, `evidence_geom_residual`, `evidence_occlusion`
- `evidence_map_semantics`, `evidence_map_stability`
- `vla_class_probs`, `vla_confidence`, `vla_embed`, `vla_provenance_json` (optional passthrough)

## Static Map Construction
1. Sample geometry points per entity from `SceneTracks_v1` poses/scales.
2. Accumulate into a voxel hash map (`VoxelHashMap`) in world frame.
3. Update policy:
   - `visible_only`: add points only for visible entities
   - `visible_and_low_residual`: rebuild using low-residual entity frames

## Inconsistency Residuals
Residuals are computed as mean distance between per-frame entity samples and nearest voxel centroids. Visibility and occlusion are used to downweight or mask residuals to avoid false positives.

- `dynamic_evidence`: raw residual or z-scored residual
- `dynamic_mask`: thresholded evidence, disabled when occluded

## Densification Targets
If camera metadata is available:
- Reproject static map voxels into each frame
- Apply spherical-bin culling (nearest depth per bin)
- Export sparse depth targets + mask

If camera metadata is not available:
- Export world-frame densified points `(T, M, 3)` + mask

## Semantic Stabilization (No Inference)
The node accepts upstream semantic distributions or labels and performs EMA updates per voxel. Per-entity stabilized semantics are aggregated by sampling entity geometry and averaging voxel posteriors.

Map-First emits evidence channels for the Semantic Orchestrator:
- `E_map_semantics` -> `evidence_map_semantics`
- `E_map_stability` -> `evidence_map_stability`
- `E_geom_residuals` -> `evidence_geom_residual`
- `E_occlusion` -> `evidence_occlusion`
- `E_dynamic_evidence` -> `evidence_dynamics_score`

## Confidence & Gating
Each entity-frame receives a confidence score derived from:
- Map coverage (valid voxel hits)
- Visibility / occlusion
- Residual stability

`MapFirstSummary` is produced alongside the artifact:
- `dynamic_pct`, `static_map_coverage`, `depth_coverage_pct`
- `residual_p50`, `residual_p90`
- `semantic_stability_score`
- `usable_pct_after_gating`
- `map_first_quality_score`

The summary is intended to be logged/stored in datapack metadata (`map_first_summary`, `map_first_quality_score`) following the same pattern as `SceneIRSummary`, `MHN summary`, and `process_reward_profile`.

## Contract (Schema Lock)
Map-First output keys are namespaced under `map_first_supervision_v1/`. Required keys are listed above; optional keys may be omitted if data is unavailable.

Key list (with shapes):
- `dynamic_evidence`: `(T, K)`
- `dynamic_mask`: `(T, K)`
- `residual_mean`: `(T, K)`
- `boxes3d`: `(T, K, 7)`
- `confidence`: `(T, K)`
- `evidence_dynamics_score`: `(T, K)`
- `evidence_geom_residual`: `(T, K)`
- `evidence_occlusion`: `(T, K)`
- `evidence_map_semantics`: `(T, K, C)` (optional)
- `evidence_map_stability`: `(T, K)` (optional)
- `meta_semantics_stability`: `(T, K)` (optional)
- `vla_class_probs`: `(T, K, C)` (optional passthrough)
- `vla_confidence`: `(T, K)` (optional passthrough)

Track alignment:
- `track_ids` must align with `SceneTracks_v1/track_ids`. If the VLA evidence uses a different ordering, it is re-ordered to match.

Missing VLA evidence:
- If `VLA_SemanticEvidence_v1` is absent, `vla_*` keys are omitted and semantic stabilization is disabled unless point labels are provided.

Confidence calibration:
- `confidence` and `vla_confidence` are in `[0, 1]`, where `0` means unreliable evidence and `1` means high trust.

## CLI
Use python3 for CLI invocations (python is not guaranteed on PATH):

```bash
python3 -m src.vision.map_first_supervision.cli_run_map_first_supervision --input-npz path/to/episode.npz --output-dir map_first_outputs
```

## Testing
Local timing visibility:

```bash
python3 -m pytest tests/ -m "not slow" --durations=20 -q
```

## Future Extensions (LiDAR/IMU)
This node is designed to accept richer geometry and motion priors without breaking the contract:
- Swap `GeometryProvider` to consume LiDAR point clouds or fused meshes
- Add IMU-based motion priors to refine dynamic evidence without changing the output schema
- Extend `VoxelHashMap` to store per-voxel motion statistics

The output artifacts and summary fields remain stable as the internal geometry sources become more accurate.
