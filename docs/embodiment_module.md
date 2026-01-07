# Embodiment Module (v1)

The Embodiment module is a deterministic, CPU-first node that converts SceneTracks + MHN + SemanticFusion into
entropy-reducing, semantically rich datapack artifacts. It does **not** change reward weights; it emits
non-blocking diagnostics and economics-facing attribution hooks.

## Inputs
- SceneTracks_v1 (required)
- Motion Hierarchy summary (optional)
- SemanticFusionResult_v1 (optional)
- ProcessReward outputs (optional)
- Action stream + joint state (optional)
- Task constraints from orchestrator (optional)
- Backend tags + physics profile hash (optional)
- Failure taxonomy / resets / safety clamps (optional)

Missing inputs lower confidence but do not block emission.

## Outputs
Artifacts are versioned and numpy-only where applicable:
- `EmbodimentProfile_v1.npz`: contacts, confidence, impossible-contact flags
- `AffordanceGraph_v1.npz`: contact/affordance edges with confidence
- `SkillSegments_v1.npz`: interaction primitive segmentation
- `EmbodimentCostBreakdown_v1.json`: Wh/time/risk per segment
- `EmbodimentValueAttribution_v1.json`: ΔMPL/Δerror/ΔEP attribution by segment
- `EmbodimentDriftReport_v1.json`: contact drift + constraint drift + sim/backend mismatch
- `CalibrationTargets_v1.json`: advisory physics knob deltas

Key scalars:
- `w_embodiment` in [0,1]
- `trust_override_candidate` boolean

## Integration Points
- `src/policies/unified_quality.py`: logs embodiment weight + drift + impossible contacts (optional weight)
- `src/rl/episode_sampling.py`: optional sampling strategies using embodiment metrics
- `src/valuation/datapack_schema.py`: EmbodimentProfileSummary stores artifact pointers + key scalars

## Economics Wiring (Non-Coupling)
Embodiment does not alter reward math. It provides advisory signals that can be
combined with the existing trust × w_econ × λ_budget gating loop:
- `w_embodiment` can be multiplied into admission scoring as a plausibility factor.
- Default behavior is non-blocking; explicit flags are required to enforce gating.

## Non-blocking by Default
Embodiment outputs are advisory. Hard gating is only enabled when explicitly configured (e.g., ingest adapters).
