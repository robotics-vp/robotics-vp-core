# Phase 2 Closure Standard: Perception / Grounding WM

## Framework

Same Category A / B / C classification as Phase 1.

- **Category A**: internal, implementable gap — keeps the phase open
- **Category B**: externally blocked gap — phase can close
- **Category C**: judgment call — must be classified before closure

## Category A Examples (Internal — Keeps Phase Open)

1. Canonical `PerceptionGroundingWorldState` schema missing or incomplete
2. Provider contract family for SAM 3/3.1, DINOv2/SigLIP, V-JEPA 2, Depth not typed
3. Evidence routing / fusion ownership not WM-owned (still in scattered orchestrator code)
4. Temporal grounding state schema missing
5. Calibration receipt types missing for provider truth
6. Promotion/demotion machinery (`disabled|auto|required`) not wired for provider adapters
7. Replay/training export surfaces missing for perception state
8. Downstream consumption hooks missing — SimSynth WM cannot consume perception state
9. Downstream consumption hooks missing — annotation/evidence bridge not wired
10. Object track state not persistent across episodes (missing temporal persistence schema)
11. `SemanticVLA` not explicitly demoted to scaffolding-only with clear successor posture
12. `backbone_stub.py` not behind typed provider contract with `real-or-unavailable` posture
13. Graph Transformer / scene graph state schema not defined
14. Provider evidence token format not defined (heterogeneous provider outputs cannot be fused)
15. Heuristic-only fusion (existing `semantic_fusion.py`) not behind `disabled|auto|required` promotion posture with learned fusion as successor
16. `SemanticBridgeRegistry` or downstream bridge receipt family missing from top-level Perception WM state
17. Habitat-style lower-WM provider/dataset/task/deployment-resource surfaces not typed
18. Provider availability, inference headroom, or deployment-resource receipts missing

## Category B Examples (External — Phase Can Close)

1. Real SAM 3/3.1 execution on GPU host (requires weights + GPU)
2. Real DINOv2/SigLIP feature extraction on GPU host (requires weights + GPU)
3. Real V-JEPA 2 inference (requires weights + GPU + `facebookresearch/vjepa2`)
4. Real depth model execution (DepthAnythingV2/UniDepth; requires weights + GPU)
5. Real 3D grounding at scale (SAM3D + GPU + point cloud data)
6. Real Unitree egocentric camera feeds and sensor corpora
7. Real camera calibration data for Unitree G1 cameras
8. Production-scale cluttered-scene datasets at humanoid scale
9. Multi-provider calibration with real concurrent SAM + DINOv2 + depth execution
10. V-JEPA 2 at real-time inference rates on companion compute
11. Long-horizon tracking data with real humanoid self-occlusion patterns

## Category C Examples (Judgment Call — Must Classify)

1. Temporal grounding capacity sizing for humanoid regime — likely B (needs data)
   but only if the schema and promotion machinery are already structural (otherwise A)
2. Provider fusion architecture beyond MVP — likely A if heuristic fusion is the
   only path with no learned successor posture; likely B if promotion machinery
   exists and the blocker is training data
3. Graph Transformer architecture finalization — likely C→A (architecture choices
   can be made from existing public research and repo episodes)
4. SAM 3/3.1 video tracking mode evaluation — likely C→B (requires SAM runtime
   to evaluate, but the typed contract surface is implementable now)
5. Evidence routing optimization — likely A if routing is still heuristic-only;
   likely B if learned routing has promotion machinery and the blocker is data

## Decision Rule

Phase 2 is closed when:

- **Zero Category A items** across the audited surfaces
- **All Category C items** classified as either A (fixed) or B (externalized)
- Remaining gaps are all Category B (GPU, provider execution, real data, calibration)
- The lower-WM provider/dataset/task/deployment-resource family is present as
  typed, replayable Perception WM state and receipts rather than deferred to a
  later economic layer
- The semantic successor stack is explicit: canonical substrate +
  WM-native bridge family + transitional `SemanticVLA` scaffolding posture

## Transition Triggers

- **≤2 isolated Category A** → may begin parallel Phase 3 (Embodiment) spec/doctrine work
- **Zero Category A** → shift implementation center of gravity to Phase 3
- **Phase 2 reopens** if Phase 3 or later work discovers a genuine missing Perception contract

## Required Evidence for Closure

Each closure assessment must include:

| Finding | Category | Rationale | File(s) |
|---------|----------|-----------|---------|
| (one row per finding) | A / B / C | why this classification | affected source |

Plus: explicit list of remaining Category B items with expected timing.

## Maturity Ladder Position

Phase 2 targets at minimum `shadow_runtime` before closure:

1. `schema_only` — canonical state types exist
2. `logging_only` — state is compiled and logged but not consumed
3. `shadow_runtime` — **target** — state is consumed by downstream WMs in shadow mode
4. `bounded_runtime_authority` — learned seams affect runtime decisions
5. `benchmark_gated_primary` — promotion gated by evidence
6. `production_recurrent` — full production loop

Phase 2 should not claim closure below `shadow_runtime` unless the gap is
genuinely external (all remaining items are GPU/provider/data blocked).

Current branch read:

- the branch has now reached early `shadow_runtime` on audited surfaces:
  - Perception state compiles from real local inputs
  - one Sim / Synth shadow consumer exists
  - one annotation/evidence shadow consumer exists
- Phase 2 is still open because additional Category A work remains around:
  - live receipt emission
  - provider/runtime truth compilation
  - replay/export surfaces
  - additional downstream consumers

## Required Surface Families

The audited Phase 2 closure sheet must explicitly check for:

- `ProviderSurfaceState`
- `DatasetSurfaceState`
- `TaskMeasurementSurface`
- `DeploymentResourceSurface`
- `ComputeEnvelopeState`
- `InferenceCapacityState`
- `BatteryState`
- `ThermalState`
- `ProviderAvailabilityReceipt`
- `InferenceHeadroomReceipt`
- `DeploymentResourceReceipt`

These are lower-WM surfaces. They should not be skipped just because later
economic allocation work will consume them too.

## Anti-Patterns

Do not:

- Call something Category B just because it has not been scoped carefully
- Treat "we need data" as Category B when the real gap is "we have not defined what data"
- Mark provider bring-up as external when public weights/repos are available and acquirable now
- Leave heuristic-only subsystems without learned successor posture and call Phase 2 closed
- Claim `shadow_runtime` maturity when downstream WMs are not actually consuming the state
