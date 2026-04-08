# Doctrine: Semantic Bridge Successor Stack

## Problem

`SemanticVLA` is an insufficient placeholder that returns empty structure
for all semantic surfaces except passthrough tags.  It must be replaced.

The replacement is NOT one monolithic semantic-analysis model.  It is a
**distributed semantic bridge family** that makes semantics both:

- **topologically organized**: each WM consumes semantics in its native form
- **functionally robust**: semantics is load-bearing for actual loop runs

## Architecture

### 1. Canonical Semantic Substrate

**Owner**: Perception / Grounding WM

**Surface**: `SceneGraphState` — the graph-transformer-produced
object-relational-temporal representation.

- Per-object tokens (d=128) from provider fusion
- Typed spatial/temporal edges (d=64) with explicit edge vocabulary
- Scene-level summary token (d=256)
- Temporal persistence from causal transformer

This is the single canonical semantic surface.  All downstream consumers
read from it through typed bridges.  No WM consumes it raw if doing so
makes semantics non-native to their regime.

### 2. Per-WM Semantic Bridge Layers

Each consuming WM has a dedicated bridge that transforms canonical
semantics into WM-native form.  These are the **semantic superposition
mechanism** — the same content in multiple WM-specific structural forms.

These bridges should be shaped against explicit lower-WM provider/resource
surfaces from Phase 2 onward, not only against abstract scene tokens. In
practice that means the substrate and bridge contracts should expect:

- `ProviderSurfaceState`
- `DatasetSurfaceState`
- `TaskMeasurementSurface`
- `DeploymentResourceSurface`

and, where relevant:

- `ComputeEnvelopeState`
- `InferenceCapacityState`
- `BatteryState`
- `ThermalState`

#### Bridge 1: Semantic → Sim / Synth / Physics

**Purpose**: Physics-relevant semantics for branch comparison, object
preservation, synthetic-vs-real alignment, branch outcome labeling,
diffusion conditioning.

**Architecture**: Topology-preserving cross-attention from sim planning
context (coverage gaps, economic urgency, branch specs) to semantic
object tokens + spatial graph convolution re-weighting edges by physics
relevance (contact > occlusion > containment).

**Governing WM**: Sim/Synth/Physics WM governs output vocabulary,
capacity, training objectives.

**Capacity**: 2-5M params.

**Training**: Branch evaluation prediction + diffusion conditioning quality.
Supervised on branch outcome receipts.  No direct RL.  Later shaped by
transport fidelity (Phase 6+) and economic yield.

**What it enables for G1**: When the robot evaluates "should I simulate
reaching for this cup?", this bridge tells the sim WM which objects and
spatial relations are physics-relevant for that branch, what contact
topology matters, and whether the synthetic branch preserves the real
scene's object structure.

#### Bridge 2: Semantic → Embodiment / Actuation

**Purpose**: Affordance and action-relevance semantics for bodily
feasibility, grasp planning, manipulation/locomotion relevance.

**Architecture**: Cross-attention from embodiment state (body config,
end-effector state, capability profile, resource constraints) to semantic
object tokens + bipartite body-object attention producing pairwise
body-part-to-object affordance scores.

**Governing WM**: Embodiment/Actuation WM governs output vocabulary,
capacity, training.

**Capacity**: 1-3M (gripper) → 5-10M (bimanual humanoid G1).  Scales
with embodiment DoF, not scene complexity.

**Training**: Affordance classification + grasp/contact prediction +
action success correlation.  Supervised, not RL.  Shaping from
downstream motor policy success.

**What it enables for G1**: When the robot asks "can I reach that handle
with my left hand while holding this with my right?", this bridge provides
per-object affordance scores conditioned on current body state and resource
constraints (battery, thermal, compute).

#### Bridge 3: Semantic → Annotation / Evidence

**Purpose**: Object-linked primitive/event labeling, failure/recovery
interpretation, teacher alignment evaluation, training dataset formation.

**Architecture**: Projection heads (MLPs / shallow attention) from semantic
object tokens → class labels, confidence, affordance hints, risk hints,
primitive-segment alignment, event labels, failure/recovery tags.

**Governing WM**: Perception/Grounding WM (this bridge feeds the
annotation/evidence subsystem, not a specific consuming WM).

**Capacity**: 500K-2M params.  Architecturally lightweight but
**functionally load-bearing** — this is the primary mechanism by which
semantic state becomes training-usable evidence.

**Training**: Annotation labeling accuracy + primitive-segment alignment +
downstream training dataset quality.

**What it enables for G1**: When the stack labels a rollout for training,
this bridge connects visual object evidence to behavioral segments,
explains failures in terms of specific objects/relations, and evaluates
teacher/runtime alignment against canonical object state.

#### Bridge 4: Semantic → Economic (later)

**Purpose**: Fixed-dimensional summaries for allocation, pricing,
governance decisions.

**Architecture**: Perceiver-style cross-attention from learned economic
query tokens (16-64) to semantic object tokens → fixed-dim summaries.

**Governing WM**: Economic WM governs output vocabulary, training.

**Capacity**: 1-3M params.

**Training**: Economic value prediction + allocation quality.  Supervised
on economic receipts.

**What it enables for G1**: The Economic WM gets a constant-dimensional
semantic summary regardless of scene cardinality, carrying: semantic
density, object diversity, affordance richness, grounding confidence,
temporal stability, concept coverage.

### 3. Training / Shaping Story

The bridges have a staged training posture:

1. **Pre-Phase 6 (now → early 2027)**: bridges shaped by supervised and
   predictive losses tied to downstream WM task performance.
   - SimSynth bridge: branch evaluation accuracy, branch yield quality
   - Embodiment bridge: affordance prediction, grasp success correlation
   - Annotation bridge: labeling accuracy, primitive alignment
   - Economic bridge: allocation quality, value correlation

2. **Phase 6 (transport)**: transport fidelity provides explicit gradient
   signals that shape bridge parameters to produce representations
   transportable across WM boundaries.

3. **Phase 7 (meta-node)**: meta-node can adjust transport shaping weights,
   creating hierarchy: meta-node → transport → bridge → WM representation.

**RL placement rule**: bridges should NOT be trained with direct RL on task
reward.  They are middleware — supervised/contrastive/predictive training
is correct for structural fidelity.  Indirect RL shaping comes through
Economic WM allocation performance.

### 4. `SemanticVLA` Succession

`SemanticVLA` is explicitly non-terminal.  Its replacement is staged:

| Stage | What happens |
|-------|-------------|
| Now (Tranche 2.0) | Canonical substrate schema + bridge state types exist |
| Tranche 2.1 | Compiler builds substrate from SceneTracks + evidence. SemanticVLA explicitly demoted with successor metadata in outputs |
| Tranche 2.2 | Annotation bridge skeleton replaces SemanticVLA tag extraction. Heuristic bridges behind `disabled\|auto\|required` |
| Tranche 2.3+ | SimSynth bridge consumes substrate for branch evaluation. Embodiment bridge placeholder for affordance |
| GPU bring-up | Learned bridges trained on downstream task signals |
| Phase 6+ | Transport fidelity shapes bridge parameters |

The `SemanticVLA.analyze_episode()` return dict now carries
`_semantic_vla_status: "scaffolding_only"` and
`_semantic_vla_successor: "distributed_semantic_bridge_family"` so
downstream consumers can detect the posture programmatically.

### 5. How Semantics Becomes Load-Bearing

Semantics is load-bearing when downstream WMs change behavior based on
semantic bridge outputs.  The maturity stages:

1. **Schema only** (current): bridge state types exist, no runtime
2. **Heuristic bridge**: static projection or passthrough, no learned params
3. **Shadow bridge**: learned bridge runs but does not affect decisions
4. **Bounded authority**: learned bridge affects non-critical decisions
   (branch selection ranking, annotation confidence weighting)
5. **Benchmark-gated primary**: learned bridge is the primary semantic
   path, gated by downstream performance benchmarks
6. **Production recurrent**: bridge is in the live G1 loop

### 6. Pre-GPU Structural Requirements

What must be structurally real before GPU/provider bring-up:

- [ ] Canonical substrate schema (`SceneGraphState`) — done
- [ ] Bridge state types for all 4 bridges — done
- [ ] Bridge receipt types — done
- [ ] Bridge promotion/demotion machinery — done
- [ ] SemanticVLA explicit successor posture — done
- [ ] Provider/dataset/task/deployment-resource surfaces typed under the Perception WM
- [ ] Compiler that builds substrate from SceneTracks + evidence
- [ ] Heuristic bridge implementations behind promotion posture
- [ ] Downstream consumption hooks in SimSynth and annotation code
- [ ] Replay/training export for bridge outputs

### 7. First GPU/Provider Bring-Up Items

1. DINOv2 features → object token initialization → substrate quality
2. SAM 3/3.1 masks → object identity → substrate cardinality
3. Annotation bridge training on labeling accuracy with existing episodes
4. SimSynth bridge training on branch evaluation accuracy
5. Depth model → metric 3D → spatial edge quality
6. V-JEPA 2 → temporal prediction → temporal grounding quality

### 8. What Is Truly External

Only after public-repo / OSS / public-config investigation is exhausted:

- Real G1 egocentric multi-camera feeds
- Real onboard/companion compute latency for bridge inference
- Real cluttered-scene datasets at humanoid scale
- Multi-provider calibration with concurrent SAM + DINOv2 + depth + V-JEPA
- Long-horizon tracking data with humanoid self-occlusion patterns
