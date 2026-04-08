# Neuralization Bridge Doctrine

## Purpose

This document specifies the neural architecture, reward shaping, RL placement, and semantic-bridge topology for the multi-WM stack. It is the operating doctrine for neuralization decisions from Phase 1 through Phase 7.

It is intentionally prescriptive. Every neural implementation target in this document has:

- a layer classification (WM-native, semantic-to-WM bridge, WM-to-WM transport-shaped, or provider-backed interpretation)
- an explicit governing WM
- a hyperparameter governance source
- a reward/shaping placement
- a capacity scaling band
- interaction constraints with the transport layer

## Core Principle: Semantic Superposition

Semantic subsystems in this stack are not local to one WM.

The repo has semantics in multiple forms and layers: object tokens, relational state, affordance annotations, primitive/action segments, teacher proposals, scene continuity, concept-conditioned tracking. These operate across WM boundaries. A single semantic representation cannot serve all downstream WMs equally, because each WM's regime imposes different structural requirements on the same underlying semantic content.

The correct architecture is therefore not:

- one monolithic semantic model that all WMs consume identically
- isolated per-WM semantic encoders with no shared abstraction

It is a **structured semantic-to-WM bridge family** — a superpositioning layer whose explicit role is to mediate between semantic subsystems that serve as operating WM-contribution subsystems and the specific WM-local subsystems they need to feed.

This superpositioning layer has three structural tiers:

1. A **canonical semantic abstraction** produces a shared object-relational-temporal representation. It is owned canonically by Perception/Grounding WM but is **not shaped only by perception needs**. Downstream WMs exert shaping pressure on what semantic distinctions survive: bridge performance, transport fidelity, embodiment usefulness, simulation usefulness, and later economic usefulness all influence what the canonical abstraction learns to represent. The abstraction must serve all downstream consumers, not just the WM that owns it.

2. **Per-WM semantic bridge layers** transform that shared representation into WM-specific formats shaped by each WM's regime. These bridges are not local convenience modules or thin adapters. They are part of the topological mediation of the stack — the mechanism by which the same underlying semantic content takes on different structural forms for different representational regimes. A generalist semantic abstraction (e.g., a graph-transformer-style object/relation state) may need to become something more topology-explicit (e.g., a spatial-graph-weighted regime) when spatial topology matters more directly for a downstream WM like Sim/Synth/Physics. The stack should be able to learn from the full chain — semantics → bridge → WM-specific representation → downstream performance — what architecture and shaping are actually best for each bridge.

3. **WM-to-WM transport learning** shapes the bridge hyperparameters and reward structure over time. The transport layer is the best judge of whether a semantic bridge is producing representations that survive cross-WM translation. Over time, the stack learns: what semantic distinctions must be preserved through each bridge, how spatial and semantic content should be balanced for each downstream WM, and how bridge parameters should evolve as downstream WMs mature.

This is the **semantic superposition principle**: the same semantic content exists simultaneously in multiple WM-specific forms, connected by learnable bridges whose quality is measured by transport fidelity and downstream WM performance. The bridges are not mere projections — they are the mechanism by which the stack resolves the tension between shared abstraction and WM-specific structural need.

## Layer Classification Taxonomy

Every neural module in the stack falls into one of four categories:

### 1. WM-Native Layer

A module that lives entirely inside one WM, is governed by that WM's objectives, and produces canonical state consumed only within that WM's boundary or emitted as typed receipts.

Examples: contractive latent dynamics, backend selector, branch yield predictor, whole-body state encoder.

### 2. Semantic-to-WM Bridge Layer

A module that transforms the canonical semantic abstraction into a WM-specific representation. It reads from the shared semantic surface and writes into one WM's native vocabulary. Its hyperparameters are governed jointly by the semantic source WM and the consuming WM, with transport quality as a shaping signal.

Examples: spatial-topology bridge for Sim/Synth/Physics WM, affordance projection for Embodiment/Actuation WM, concept-conditioned branch evaluator for Sim/Synth/Physics WM.

### 3. WM-to-WM Transport-Shaped Layer

A module whose parameters are explicitly shaped by cross-WM transport learning. It may live inside one WM, but its capacity, regularization, and training signals include transport fidelity as a first-class objective.

Examples: cross-WM receipt critic in Economic WM, transport bridge modules in Phase 6, meta-node state encoders that must be interpretable across WM boundaries.

### 4. Provider-Backed Interpretation Layer

A module that wraps an external OSS provider (SAM 3/3.1, V-JEPA 2, OpenVLA, DINOv2, etc.) and produces typed, calibrated outputs for downstream consumption. The provider is not native truth. The interpretation layer owns calibration, uncertainty estimation, and fallback semantics.

Examples: SAM 3/3.1 concept segmentation adapter, V-JEPA 2 temporal prediction wrapper, DINOv2 feature extraction adapter.

## Autoencoder / Codebook Posture Across the Stack

Autoencoder-family models (contractive / denoising AEs, VAEs, VQ-VAEs,
codebook learners, bounded bottleneck encoders) are **allowed** where they
act as **compressors, manifold learners, or auxiliary regularizers** inside
a WM or at a **bridge-local bottleneck**. They are **not** a default
replacement for typed state, transport contracts, governance composition,
or the primary model families already named in this doctrine (Graph
Transformer semantic abstraction, Perceiver-style economic pooling,
switching-SSM estimators in the Economic WM, inverse-dynamics and
diffusion lanes in Embodiment, etc.).

**Layer taxonomy (where they may appear):**

| Typical role | Layer classification | Notes |
|--------------|------------------------|--------|
| Compression / denoising **inside** a WM’s native encoder or receipt path | `WM-native` | Must emit or preserve **typed** surfaces; bottleneck is a seam, not a new ontology. |
| Optional pre-bottleneck shaping **in front of** a semantic-to-WM bridge | `semantic-to-WM bridge` (strictly subordinate) | Does **not** replace the bridge’s stated architecture (e.g. Perceiver queries for Semantic→Economic). |
| Rare: auxiliary alignment on transport-shaped features | `WM-to-WM transport-shaped` | Only if it improves **typed fidelity** metrics—not “reconstruct everything.” |
| Provider feature cleanup before projection | `provider-backed interpretation` | Small denoising / calibration head on top of frozen provider outputs. |

**What they are not:** the primary answer for cross-WM **transport**
(ontology mediation, affine maps, contrastive alignment—not reconstruction
loss as the doctrine). **Not** the meta-regal story (governance composition,
Pareto adjudication). **Not** an excuse to collapse to “one latent is truth.”

### Perception / Grounding

**Good fit (WM-native or bridge-local auxiliaries):** graph / relational /
object–temporal compression; compact semantic bottlenecks; denoising or
archetype stabilization that **supports** the canonical object-token substrate.

**Constraint:** canonical typed scene / grounding state and receipts remain
authoritative. An AE must not become the hidden master representation that
replaces `BeliefState`-class contracts or the Graph Transformer’s role as
the shared semantic abstraction.

### Semantic→Economic bridge (Economic-facing)

**Primary posture (unchanged):** **Perceiver-style** learned query tokens
pooling variable-cardinality semantic tokens into **fixed-dimensional**
economic summaries. That remains the architecturally correct Semantic→Economic
bridge family.

**Optional only:** a **bounded** denoising or pre-compression auxiliary on
semantic tokens **before** the Perceiver readout, or a small regularizer that
stabilizes summaries—**not** a second competing bridge, **not** a reason to
make the Economic input path scene-dimensional again, **not** a replacement
for Economic WM ownership of how summaries are trained against allocation
and receipts.

### Embodiment / Actuation

**Good fit (WM-native):** posture/contact or action-chunk manifold
compression; **skill / synergy codebooks** (VQ-style); interoceptive
bottlenecks where telemetry is high-rate but control needs a typed low-D
state.

**Constraint:** does not replace the six-subsystem Embodiment ontology,
inverse lane, action proposal head, or drift/cost surfaces as **owners** of
their contracts.

### Sim / Synth / Physics (light touch)

Optional: branch- or rollout-level **archetype** compression, synthetic-episode
motifs—only where it improves branch evaluation or dataset formation without
replacing typed branch receipts or physics-facing bridges.

### Transport / meta-governance

**Explicit:** autoencoder-first design is **wrong** here. Transport is typed
fidelity and structural alignment; meta-regal is inter-domain composition.
Neither is “train a big AE across WMs and call it integrated.”

## Semantic Abstraction Architecture

### Canonical Semantic Abstraction (Perception/Grounding WM owns this)

**What it represents**: The shared object-relational-temporal representation that downstream WMs consume through their respective bridge layers.

**Architecture**: **Graph Transformer** over object tokens with explicit spatial and temporal edges.

Why Graph Transformer rather than pure Set Transformer: spatial topology matters structurally in this stack. Sim/Synth/Physics WM needs spatial relations for branch evaluation. Embodiment WM needs spatial relations for action feasibility. Economic WM needs spatial relations for deployment realism assessment. A Set Transformer with learned pairwise embeddings can approximate this, but a Graph Transformer makes spatial topology explicit in the attention structure, which produces representations that survive cross-WM translation better because the topology is preserved in the architecture rather than learned implicitly.

Specifically:

- **Node features**: per-object tokens derived from SAM 3/3.1 masks + DINOv2/SigLIP features + depth + SceneTracks geometry + temporal persistence state
- **Edge types**: spatial adjacency, contact, containment, occlusion, temporal co-occurrence, affordance relation
- **Temporal structure**: causal attention over time steps within each object token's trajectory; cross-object attention within each time step
- **Output**: per-object token embeddings (d=128), per-edge relation embeddings (d=64), scene-level summary token (d=256)

**Layer classification**: WM-native to Perception/Grounding WM, but its output is the shared semantic surface that all downstream bridges consume.

**Governing WM**: Perception/Grounding WM governs capacity and training canonically.

**Downstream shaping rule**: Although Perception/Grounding WM owns this canonically, the abstraction is not shaped only by perception needs. Downstream WMs should exert shaping pressure through:
- **Bridge performance feedback**: if a semantic-to-WM bridge layer struggles to produce useful WM-specific representations, that failure signal should propagate back to shape what the abstraction encodes. Before Phase 6, this is approximated by downstream WM task performance metrics used as auxiliary supervised targets. After Phase 6, transport fidelity provides an explicit gradient.
- **Embodiment usefulness**: the abstraction should preserve spatial/affordance structure that the Embodiment WM needs, not just visual reconstruction quality.
- **Simulation usefulness**: the abstraction should preserve contact/topology structure that the Sim/Synth/Physics WM needs for branch evaluation.
- **Economic usefulness**: later, the abstraction should preserve whatever object/scene features most predict economic value.

The concrete training consequence is that the canonical abstraction's loss function should eventually include weighted terms from multiple downstream consumers, not only local perception objectives. This prevents the abstraction from overfitting to perception metrics that are irrelevant to downstream WM performance.

**Hyperparameter governance**:
- **Local**: object reconstruction, temporal prediction, grounding accuracy
- **Downstream-shaped**: auxiliary supervised terms from downstream WM task performance (bridge output quality, branch evaluation accuracy, affordance prediction accuracy, economic value correlation)
- **Transport-shaped** (Phase 6+): transport round-trip reconstruction quality across all consuming WMs provides an explicit gradient signal

**Scale**: Capacity bands are provisional and should be treated as tied to representational burden (object count, cardinality, temporal horizon), latency budget, provider maturity, replay density, and Unitree G1-scale constraints rather than as settled truth.
- **Tabletop regime** (10-30 objects, 3-5 edge types): order of 5-10M params is a reasonable starting point
- **Humanoid-facing regime** (50-100 objects, richer edge vocabulary, egocentric depth, body-aware occlusion): order of 20-50M params is a reasonable scaling target
- **Underpowered signal**: cannot capture relational structure with temporal persistence (likely below ~3M params)
- **Overbuilt signal**: model is trying to internalize raw visual features that should live in the vision backbone provider (likely above ~100M params)
- These bands should be revisited during the Phase 3.5 humanoid capacity audit and whenever provider maturity or replay density changes the representational substrate

**Training**: Graph reconstruction + temporal object prediction + concept grounding accuracy + downstream bridge performance terms + (later) transport fidelity shaping term.

### Per-WM Semantic Bridge Layers

Each downstream WM consumes the canonical semantic abstraction through a dedicated bridge layer that transforms the shared representation into the WM's native vocabulary.

These bridges are the **semantic superposition mechanism**. The same object-relational-temporal content is projected into different structural forms depending on what each WM needs.

---

#### Bridge 1: Semantic-to-SimSynthPhysics Bridge

**What it represents**: Transformation of semantic object-relational state into the structural vocabulary needed for synthetic branch evaluation, sim agenda compilation, and diffusion conditioning.

**Architecture**: **Topology-Preserving Cross-Attention** from sim planning context (coverage gaps, economic urgency, branch specs) to semantic object tokens, followed by a **spatial graph convolution** layer that re-weights edges by physics relevance (contact > occlusion > containment for physics planning).

Why this structure: Sim/Synth/Physics WM needs semantic content re-weighted toward spatial topology, contact structure, and physics-relevant relations. The cross-attention selects which objects/relations matter for a given sim job. The graph convolution re-encodes spatial relations with physics-appropriate edge weighting.

**Layer classification**: Semantic-to-WM bridge layer.

**Governing WMs**: Perception/Grounding WM governs the input representation. Sim/Synth/Physics WM governs the output vocabulary, capacity, and training objectives.

**Hyperparameter governance**:
- **Primary**: Sim/Synth/Physics WM branch evaluation accuracy, branch yield prediction quality, diffusion conditioning quality
- **Transport-shaped**: when the Phase 6 Perception↔Sim/Synth transport bridge exists, that bridge's translation quality directly shapes this module's regularization (the bridge should produce representations that the transport layer can translate cleanly)
- **Economic yield**: later, Economic WM allocation performance over sim receipts provides an indirect shaping signal

**Scale**: ~2-5M params. This is a bridge, not a major encoder. **Underpowered**: <500K loses spatial topology preservation. **Overbuilt**: >10M means the bridge is doing work that should live in either the semantic abstraction or the sim WM's native layers.

**Training**: Branch evaluation prediction loss + diffusion conditioning quality + (later) transport round-trip fidelity term.

**Reward/shaping**: No direct RL. Supervised on branch outcome receipts. Shaping comes from downstream Sim/Synth/Physics WM performance and transport fidelity.

---

#### Bridge 2: Semantic-to-Embodiment Bridge (Affordance Projection)

**What it represents**: Transformation of semantic object-relational state into body-relevant affordance structure for action feasibility, grasp planning, and manipulation/locomotion relevance.

**Architecture**: **Cross-attention from embodiment state to semantic object tokens**, where embodiment state includes body configuration, end-effector state, capability profile, and resource constraints (compute/battery/thermal). Followed by per-object affordance heads (classification + continuous parameters) and a body-object **bipartite attention** layer that produces pairwise body-part-to-object affordance scores.

Why bipartite attention: affordance is fundamentally a relation between body parts and objects. For a gripper, this is simple (one end-effector, many objects). For a humanoid with two hands, this becomes bipartite attention over hand-state × object-state. The structure generalizes correctly as embodiment complexity grows.

**Layer classification**: Semantic-to-WM bridge layer.

**Governing WMs**: Perception/Grounding WM governs the input. Embodiment/Actuation WM governs output vocabulary, capacity, and training.

**Hyperparameter governance**:
- **Primary**: Embodiment WM action feasibility prediction, grasp success prediction, contact quality
- **Transport-shaped**: Perception↔Embodiment transport bridge quality shapes how the affordance projection preserves relational structure
- **Resource-aware**: affordance scores should be conditioned on available compute/battery/thermal state, because a resource-constrained robot has different effective affordances than an unconstrained one

**Scale**: Start ~1-3M params for gripper. Scale to ~5-10M for dexterous bimanual. The capacity jump is driven by embodiment complexity, not scene complexity. **Underpowered**: affordance without body-conditioning (flat classifier on object features) will not generalize across embodiments. **Overbuilt**: large generative models here are wrong; affordances are discriminative.

**Training**: Affordance classification + grasp/contact prediction + action success correlation. Supervised, not RL. Shaping from downstream motor policy success.

---

#### Bridge 3: Semantic-to-Economic Bridge (Semantic Receipt Encoder)

**What it represents**: Transformation of semantic object-relational state into the compact summary format the Economic WM needs for allocation, pricing, and governance decisions.

**Architecture**: **Perceiver-style cross-attention** from a small set of learned economic query tokens (16-64) to the full semantic object-token set, producing fixed-dimensional economic-semantic summaries. These summaries carry: semantic density, object count/diversity, affordance richness, grounding confidence, temporal stability, and concept-coverage metrics.

Why Perceiver: The Economic WM needs fixed-dimensional summaries from variable-cardinality semantic state. Perceiver's learned query tokens are the architecturally correct pooling mechanism. Mean/max pooling loses too much relational information. Full attention over all object tokens would make the Economic WM's input dimensionality scene-dependent.

**Layer classification**: Semantic-to-WM bridge layer.

**Governing WMs**: Perception/Grounding WM governs input. Economic WM governs output vocabulary and training.

**Hyperparameter governance**:
- **Primary**: Economic WM allocation accuracy, value prediction quality
- **Transport-shaped**: transport bridge quality from lower WMs to Economic WM
- **Economic yield**: directly shaped by economic allocation performance

**Scale**: ~1-3M params. Economic WM bridges should stay compact. **Overbuilt**: >5M params means the Economic WM is trying to re-encode raw scene state.

**Training**: Economic value prediction + allocation quality. Supervised on economic receipts.

---

#### Bridge 4: Semantic-to-Annotation/Evidence Bridge

**What it represents**: Transformation of semantic state into structured evidence payloads for rollout labeling, semantic evidence, annotation crosswalk, and training dataset formation.

**Functional importance**: Although the projection architecture is lighter than the other bridges, the function is load-bearing. This is where:
- **Object-linked primitive annotations** cash out: connecting visual object evidence to behavioral/action segmentation
- **Failure and recovery interpretation** get grounded: explaining what went wrong in terms of specific objects and relations
- **Teacher/runtime alignment** is evaluated: comparing teacher proposals to canonical object state
- **Semantic evidence fusion** produces training-consumable artifacts: the annotation bridge is what makes semantic state usable for training dataset formation, replay labeling, and downstream valuation

Do not treat this bridge as trivial just because it is architecturally lightweight. It is the primary mechanism by which semantic state becomes training-usable evidence.

**Architecture**: **Projection heads** (MLPs or shallow attention) from semantic object tokens into annotation-format outputs: class labels, confidence scores, bounding regions, affordance hints, risk hints, primitive-segment alignment scores, object-linked event labels, failure/recovery interpretation tags.

**Layer classification**: Semantic-to-WM bridge layer (feeds annotation/evidence subsystem rather than a specific WM, but governed by Perception/Grounding WM).

**Scale**: Order of 500K-2M params. The architecture should be lightweight relative to the other bridges, but the function must be taken seriously — particularly primitive-segment alignment and failure interpretation, which may need slightly richer attention structure than pure MLP projection.

---

### How Transport Learning Shapes Bridge Hyperparameters

This is the critical architectural innovation beyond standard multi-model stacks.

#### Principle

The WM-to-WM transport layer (Phase 6) is the best judge of whether a semantic bridge is producing representations that survive cross-WM translation. Therefore, transport quality should shape bridge learning through a structured feedback mechanism:

#### Mechanism

1. **Forward path**: Semantic abstraction → Bridge → WM-native representation → Transport bridge → Adjacent WM
2. **Transport quality signal**: The transport bridge measures translation fidelity (round-trip reconstruction, topology preservation, downstream task delta)
3. **Gradient path**: Transport quality gradient flows back through the transport bridge into the semantic-to-WM bridge, shaping the bridge to produce representations that are more transportable
4. **Regularization**: Transport fidelity acts as a regularization term on bridge parameters, preventing bridges from producing representations that are locally optimal for one WM but untranslatable to others

#### Staged Realization

- **Before Phase 6** (now through early 2027): Bridge hyperparameters are shaped only by local WM objectives and downstream consumer performance. Transport shaping is approximated by: (a) ensuring bridge outputs conform to typed canonical schemas, (b) penalizing bridge representations that diverge excessively from the shared semantic abstraction's native structure.
- **Phase 6 onward**: Transport bridges provide explicit gradient signals. Each transport bridge emits a `transport_fidelity_score` that becomes a weighted term in the corresponding semantic-to-WM bridge's loss function.
- **Phase 7 onward**: The Meta-Node Superposition WM can learn to adjust the transport shaping weights themselves, creating a hierarchy: meta-node governs transport emphasis → transport shapes bridge parameters → bridges shape WM-native representations.

#### RL Placement for Transport-Shaped Layers

Transport-shaped layers should **not** be trained with direct RL on task reward. They should be trained with:

- **Supervised losses** on transport quality (round-trip reconstruction, topology preservation)
- **Contrastive losses** on cross-WM alignment (representations from the same entity/event in different WMs should be close)
- **Auxiliary predictive losses** on downstream WM performance (not RL reward, but supervised prediction of downstream performance metrics)
- **Later**: indirect RL shaping through the Economic WM's allocation performance, where better transport quality leads to better allocation decisions leads to better economic outcomes

The reason to avoid direct RL here: transport bridges are middleware. Making them directly RL-trained on task reward creates an optimization shortcut where the transport layer learns to game the reward rather than preserve representational structure. Supervised and contrastive objectives are the correct training regime for structural fidelity.

---

## Revised Neural Prescriptions by Layer Classification

### WM-Native Layers

#### Sim/Synth/Physics WM

| Module | Architecture | Params | Training | Reward/Shaping |
|--------|-------------|--------|----------|----------------|
| Branch evaluation / future estimation | V-JEPA 2-style latent predictor over structured state tokens (not raw video) | 20-100M | Predictive: latent state prediction conditioned on branch spec + action plan | Self-supervised predictive loss. No direct RL. Shaping from branch outcome receipts. |
| Backend/fidelity selector | MLP classifier over job features + resource constraints | 100K-500K | Supervised on execution outcome receipts | Classification loss. Reward signal: execution success + resource efficiency. |
| Branch yield predictor | MLP/small transformer over branch plan features + inferential contract | 200K-1M | Supervised on actual yield (coverage delta, economic value, training improvement) | Regression loss. Reward signal: downstream training improvement from branches. |
| Agenda ranker | Learned-to-rank module over simulation job features | 500K-2M | Supervised on realized job value rankings from receipts | Ranking loss (ListNet/LambdaRank). Shaping from economic yield. |
| Diffusion conditioning compiler | Small transformer over top-K job features → diffusion prompt structure | 1-3M | Supervised on diffusion output quality + branch evaluation quality | Reconstruction + downstream quality. |

#### Embodiment/Actuation WM

| Module | Architecture | Params | Training | Reward/Shaping |
|--------|-------------|--------|----------|----------------|
| Whole-body state encoder | GNN (message-passing) over kinematic tree + contact topology | 3-20M | Self-supervised: next-state prediction, contact prediction, forward dynamics | Predictive loss + contact accuracy. No direct RL on this encoder. |
| Action feasibility checker | MLP + contact-conditioned classifier on body-state encoder output | 500K-2M | Supervised on action execution outcomes | Binary/graded classification loss. |
| Motor policy (whole-body control) | Diffusion Policy over action sequences, conditioned on body state + skill embedding + perception tokens + resource state | 10-100M | BC pretraining → RL fine-tuning in sim | **RL here**: stability reward + energy efficiency + joint compliance + contact quality. NOT economic reward. Economic signals shape skill selection (upstream π_H), not joint torques. |
| Compute/battery/thermal forecaster | Temporal MLP or small LSTM over resource telemetry sequences | 500K-2M | Supervised on actual resource trajectories | Prediction loss. Feeds feasibility checker and resource-aware control. |
| Latency-aware control support | MLP head on body-state encoder + resource forecaster outputs | 200K-1M | Supervised on control success under latency/resource pressure | Classification/regression loss. |

#### Perception/Grounding WM

| Module | Architecture | Params | Training | Reward/Shaping |
|--------|-------------|--------|----------|----------------|
| Canonical semantic abstraction (Graph Transformer) | Graph Transformer over object tokens with spatial/temporal edges | 5-50M | Self-supervised: graph reconstruction, temporal prediction, concept grounding + (later) transport fidelity term | No direct RL. Supervised + predictive + (later) transport shaping. |
| Vision backbone adapter (DINOv2/SigLIP wrapper) | Provider-backed: frozen backbone + learned projection head | 1-5M (projection only) | Supervised: grounding accuracy, downstream task correlation | Provider-backed interpretation layer. |
| SAM 3/3.1 concept segmentation adapter | Provider-backed: SAM inference + learned calibration/uncertainty head | 500K-2M (calibration only) | Supervised: segmentation quality, uncertainty calibration | Provider-backed interpretation layer. |
| V-JEPA 2 temporal state adapter | Provider-backed: V-JEPA 2 inference + learned projection into perception WM token space | 2-5M (projection only) | Predictive: temporal state prediction quality | Provider-backed interpretation layer. |
| Temporal grounding / scene persistence | Causal transformer over object token trajectories | 3-10M | Predictive: next-frame object state, occlusion prediction, re-identification | Self-supervised predictive. No RL. |

#### Economic WM

| Module | Architecture | Params | Training | Reward/Shaping |
|--------|-------------|--------|----------|----------------|
| Multi-WM receipt critic | Cross-attention Transformer over WM receipt tokens with WM-identity embeddings | 5-15M | Supervised on economic outcomes + counterfactual evaluation | Supervised + (later) RL shaping from allocation performance. |
| Resource allocator (compute/battery/sim budget) | Lagrangian dual actor-critic over resource constraint receipts | 1-3M | **RL here**: Lagrangian dual ascent with constraint satisfaction. Economic reward shapes allocation. | Direct RL with Lagrangian structure. Reward: economic yield under resource constraints. |
| Source-mixture allocator | Set-input network (DeepSets or Perceiver) over datapack composition features | 2-5M | **RL here**: allocation decisions with economic reward signal | RL with economic reward. Shaping from lower-WM contribution encoders. |
| Counterfactual composition critic | MLP/small transformer over composition features + objective tensor | 2-5M | Supervised on realized counterfactual outcomes | Supervised. No direct RL on the critic itself. |
| Datapack utility critic | MLP over datapack embedding + active objective tensor | 1-3M | Supervised on marginal training improvement | Supervised on replay outcomes. |

### Semantic-to-WM Bridge Layers

(Detailed above in the Per-WM Semantic Bridge Layers section)

| Bridge | Target WM | Architecture | Capacity Band (provisional) | Hyperparameter Governance |
|--------|-----------|-------------|----------------------------|---------------------------|
| Semantic→SimSynthPhysics | Sim/Synth/Physics | Topology-preserving cross-attention + spatial graph conv | Order of 2-5M (scales with scene complexity and physics edge vocabulary) | SimSynth WM objectives + transport fidelity + downstream economic yield |
| Semantic→Embodiment | Embodiment/Actuation | Cross-attention from body state + bipartite body-object attention | Order of 1-10M (scales with embodiment DoF and action space complexity) | Embodiment WM objectives + transport fidelity + motor policy success |
| Semantic→Economic | Economic | Perceiver-style cross-attention with learned economic queries | Order of 1-3M (should stay compact relative to lower WMs) | Economic WM objectives + transport fidelity |
| Semantic→Annotation | Annotation/Evidence | Projection heads + shallow attention for primitive-segment alignment | Order of 500K-2M (lightweight architecture but load-bearing function) | Perception WM grounding quality + annotation/labeling accuracy + downstream training dataset quality |

### WM-to-WM Transport-Shaped Layers (Phase 6+)

| Bridge | WMs Connected | Architecture | Params | Training Regime |
|--------|---------------|-------------|--------|-----------------|
| Perception↔SimSynthPhysics | Perception, Sim/Synth | Learned affine maps with orthogonality reg + lightweight cross-attention | 500K-2M | Round-trip reconstruction + topology preservation + contrastive alignment. NOT direct RL. |
| Perception↔Embodiment | Perception, Embodiment | Same family, with body-topology-aware alignment | 500K-2M | Same regime + body-state preservation metrics |
| Embodiment↔SimSynthPhysics | Embodiment, Sim/Synth | Same family, with physics-relevance weighting | 500K-2M | Same regime + physics fidelity metrics |
| Lower-WMs→Economic | All lower, Economic | Cross-attention aggregation over heterogeneous WM receipt tokens | 1-3M | Same regime + economic allocation quality |
| Economic→MetaNode | Economic, Meta-Node | Learned governance-vocabulary projection | 500K-1M | Same regime + governance satisfaction |

### Provider-Backed Interpretation Layers

| Provider | Architecture | Params (adapter only) | Calibration Training | Fallback Posture |
|----------|--------------|-----------------------|---------------------|------------------|
| SAM 3/3.1 | Frozen SAM + learned uncertainty/calibration head | 500K-2M | Supervised: calibration quality, uncertainty estimation | real-or-unavailable; explicit stub only for smoke |
| V-JEPA 2 | Frozen V-JEPA 2 + learned projection into WM token space | 2-5M | Predictive: temporal state quality | real-or-unavailable; planning-only fallback if GPU blocked |
| DINOv2/SigLIP | Frozen backbone + learned projection | 1-5M | Supervised: grounding/matching accuracy | real-or-unavailable |
| OpenVLA | Frozen VLA + learned action calibration + confidence head | 1-3M | Supervised: action prediction quality | explicit unavailable with teacher trace |
| Depth (DepthAnythingV2/UniDepth) | Frozen depth model + learned metric calibration | 500K-1M | Supervised: metric depth accuracy | real-or-unavailable |

---

## Hierarchical Reward Shaping Doctrine

### Core Rule

Reward shaping and RL placement must be topologically governed. Reward signals propagate downward through the WM hierarchy, but each layer receives the reward in the form appropriate to its representational burden.

### Reward by Layer

#### Level 0: Motor / Reflex (Embodiment WM, innermost loop)

**What reward belongs here**: physical execution quality
- Joint compliance and limit satisfaction
- Contact quality and force regulation
- Energy efficiency per action
- Stability and balance (for humanoid: support polygon, CoM tracking)
- Latency compliance (action executed within control-rate deadline)

**What reward does NOT belong here**: economic signals, throughput, coverage improvement, governance satisfaction. These are too abstract for joint-level control and would create degenerate reward landscapes.

**RL structure**: PPO or SAC over motor policy. Reward is dense, per-step, physical. Critic observes body state + action + resource state.

**Auxiliary losses**:
- Forward dynamics prediction (body state encoder)
- Contact prediction
- Energy consumption prediction

#### Level 1: Skill / Primitive (HRL π_L, Embodiment WM)

**What reward belongs here**: skill execution success
- Skill completion (did the grasp succeed? did the drawer open?)
- Skill parameter satisfaction (was the target clearance achieved?)
- Skill timing (completed within timeout?)
- Skill safety (no collisions during execution?)

**Economic shaping at this level**: mild. Skill-level economic reward (e.g., value of completing a grasp) can shape skill policy learning, but should be a minority term (10-20% weight) compared to skill-completion reward. The economic signal's role here is to break ties between equally successful skill executions, not to drive skill policy optimization.

**RL structure**: PPO over skill-conditioned policy. Reward is per-skill-episode. Critic observes body state + skill embedding + object state.

**Auxiliary losses**:
- Skill termination prediction
- Object state prediction at skill completion

#### Level 2: Task / Planning (HRL π_H, straddles Embodiment and Sim/Synth WMs)

**What reward belongs here**: task-level economic reward
- MPL, EP, error rate, wage parity (current `compute_econ_reward`)
- Task completion and quality
- Safety compliance at the task level

**This is where economic reward is primary.** The high-level controller's job is to select skills and parameters that maximize economic value. Economic reward should be 60-80% of the signal at this level.

**RL structure**: PPO over skill selection. Reward is per-episode. Critic observes task state + semantic state + economic context.

**Auxiliary losses**:
- Task outcome prediction
- Economic value prediction (contract-aware critic)
- Counterfactual skill selection evaluation

#### Level 3: Simulation / Synthesis (Sim/Synth/Physics WM)

**What reward belongs here**: NOT direct RL in most cases.

Most Sim/Synth/Physics WM modules should be trained with supervised or predictive losses on receipts, not with RL on economic reward directly. The reason: simulation planning is an information-gathering activity, not a direct control task. RL on "did this simulation improve training?" has extremely delayed, noisy reward and would require impractically long credit assignment horizons.

**Exceptions where RL applies**:
- Agenda ranking (learned-to-rank with receipt-based reward)
- Resource-constrained simulation budgeting (Lagrangian RL over compute/time constraints)

**Primary training regime**: supervised on receipts, predictive for future estimation, contrastive for branch comparison.

**Shaping from Economic WM**: The Economic WM's valuation of sim-generated data provides a shaping signal for the branch yield predictor and agenda ranker. This is not direct RL; it is a supervised target derived from economic receipts.

#### Level 4: Economic Allocation (Economic WM)

**What reward belongs here**: economic yield under constraints
- Allocation decisions that maximize throughput, minimize error, reduce cost
- Resource allocation that satisfies compute/battery/safety constraints
- Governance satisfaction

**RL structure**: Lagrangian dual actor-critic. The dual variables (λ) for constraints are learned via dual ascent, consistent with the existing wage-parity λ controller pattern.

**What should remain supervised at this level**:
- Value prediction (critic training is supervised on realized outcomes)
- Counterfactual evaluation (supervised on counterfactual outcome data)

#### Level 5: Meta-Regal-Node Governance (Meta-Regal-Node WM, Phase 7)

**What reward belongs here**: inter-domain governance quality
- Inter-domain Pareto improvement across governance nodes (economics,
  anti-reward-hacking, plausibility, safety, deployment truth)
- Governance satisfaction across domain nodes, not just WMs
- Long-horizon stability of governance composition
- Conflict resolution quality (how well competing domain signals were composed)
- Governance pluralism preservation (no single node silently dominates)

**RL structure**: Multi-objective actor-critic with Pareto front tracking
(hypernetwork-conditioned policy over the governance composition space).
Regime-conditioned composition weights. Confidence-aware node weighting.

**Key architectural distinction**: the intra-domain Pareto optimization
(within the Economic WM) is fundamentally different from the inter-domain
Pareto composition (within the meta-regal-node WM). The meta-layer governs
whether the economic allocator's recommendations can be trusted in the
current regime.

Full doctrine:
`docs/economic_world_model/doctrine_meta_regal_node_wm.md`

### Reward Propagation Rules

1. **Downward propagation**: Higher-level reward signals shape lower-level behavior only through the appropriate interface:
   - Economic reward → task-level skill selection (Level 2)
   - Task success → skill policy learning (Level 1) via skill-completion reward
   - Skill success → motor policy (Level 0) via motor execution quality
   - Never: economic reward → joint torques

2. **Upward propagation**: Lower-level receipts inform higher-level reward computation:
   - Motor execution quality receipts → skill success evaluation
   - Skill execution receipts → task-level economic computation
   - Sim execution receipts → economic allocation valuation
   - All receipts → meta-node governance evaluation

3. **Lateral propagation through transport**: Transport fidelity shapes semantic bridge parameters, not reward signals. Transport does not carry reward; it carries representation quality signals.

### Auxiliary Loss Placement

| Subsystem | Auxiliary Losses | Purpose |
|-----------|-----------------|---------|
| Vision backbone adapter | Contrastive (SimCLR/MoCo-style), reconstruction | Learn useful visual features |
| Semantic abstraction (Graph Transformer) | Graph reconstruction, temporal prediction, concept grounding | Learn object-relational-temporal structure |
| Temporal grounding | Next-frame object state prediction, occlusion prediction | Learn scene persistence and continuity |
| Whole-body state encoder | Forward dynamics, contact prediction | Learn physically grounded body representation |
| Branch evaluation (V-JEPA 2-style) | Latent state prediction | Learn predictive representations for branch comparison |
| Semantic-to-WM bridges | Downstream WM task performance (supervised), transport fidelity (supervised) | Learn transportable WM-specific representations |
| Transport bridges | Round-trip reconstruction, topology preservation, contrastive alignment | Learn structure-preserving translation |

### Training Regime Classification

The following classification distinguishes between **not primarily RL-owned** and **never influenced by downstream return/regret**. This nuance matters because some modules that should not be directly RL-trained may still receive indirect downstream-shaped pressure through supervised auxiliary terms derived from downstream performance or economic outcomes.

#### Not primarily RL-owned (primary training is supervised/self-supervised/contrastive/predictive)

These modules should NOT have their primary loss function be an RL return signal. But they MAY receive downstream-shaped pressure through:
- supervised auxiliary terms derived from downstream WM task performance
- transport fidelity signals
- economic yield correlation as a supervised shaping target (not as an RL reward)

| Module | Primary Training | Downstream Shaping Allowed? | Reason |
|--------|-----------------|---------------------------|--------|
| Semantic abstraction (Graph Transformer) | Self-supervised: graph reconstruction, temporal prediction, grounding | Yes: auxiliary supervised terms from bridge/WM performance | Direct RL would collapse representation toward one WM. Supervised downstream terms shape without collapsing. |
| Transport bridges | Supervised: round-trip reconstruction, topology preservation, contrastive alignment | Yes: transport fidelity influenced by downstream WM success metrics | Direct RL creates gaming shortcuts. Supervised quality metrics are safe. |
| Semantic-to-WM bridges | Supervised: downstream WM task prediction, transport fidelity | Yes: primary training includes downstream WM performance as supervised targets | Bridges exist to serve downstream WMs; their supervised targets should reflect downstream needs. |
| Provider interpretation layers | Supervised: calibration, uncertainty estimation | Limited: provider-truth calibration may improve from downstream correlation, but not from task reward | These are calibration modules. Downstream influence should only inform calibration quality, not task optimization. |
| Contribution encoders | Supervised: marginal value prediction | Yes: supervised targets come from realized downstream value, which is inherently downstream-shaped | These predict value; their targets are inherently derived from downstream outcomes. |
| Critics and value predictors | Supervised on realized outcomes | Yes: critics learn from realized returns, which is downstream-shaped by definition | The critic learns to predict returns; the actor is RL-trained using the critic. The critic itself is supervised. |

#### Directly RL-owned (primary training is RL with task/economic reward)

| Module | RL Structure | Reward Source |
|--------|-------------|---------------|
| Motor policy (whole-body control) | PPO/SAC | Physical execution quality (stability, compliance, energy, contact) |
| Skill policy (π_L) | PPO | Skill completion + mild economic shaping (10-20%) |
| Task policy (π_H) | PPO | Economic reward (60-80%) + task completion |
| Resource allocator (compute/battery) | Lagrangian dual actor-critic | Economic yield under resource constraints |
| Source-mixture allocator | RL over allocation decisions | Economic reward shaped by contribution encoders |
| Meta-node governance policy | Multi-objective actor-critic | Pareto governance quality |

#### Key principle

The distinction is not "RL vs. no RL" but rather:
- **RL-owned**: the module's primary optimization objective is an RL return signal
- **Downstream-shaped but not RL-owned**: the module receives information about downstream performance through supervised/predictive/contrastive targets derived from downstream outcomes, but its primary loss is not an RL return
- **Locally-shaped only**: the module's training is determined entirely by local objectives with no downstream influence (rare in this stack; mostly limited to early-stage provider calibration)

This matters because the stack should learn end-to-end in a topologically governed way — but "end-to-end" does not mean "RL everywhere." It means supervised and predictive targets should carry downstream information without creating the optimization pathologies that direct RL would cause in structural/representational modules.

---

## Semantic-to-Spatial Superposition

### The Core Tension

Semantics and space must become superpositioned. The canonical semantic abstraction is a graph transformer that encodes both semantic and spatial structure. But different WMs need different weightings of semantic vs. spatial information:

- **Sim/Synth/Physics WM** needs heavy spatial topology: contact structure, collision geometry, physical adjacency, object placement accuracy. Semantic content is important but secondary to spatial fidelity.
- **Embodiment/Actuation WM** needs body-relative spatial structure: reachability, grasp geometry, approach vectors, contact surfaces. Spatial structure is body-centric rather than scene-centric.
- **Economic WM** needs abstract spatial summaries: scene complexity, object density, workspace coverage. Detailed spatial topology is noise at this level.

### How Bridges Handle This

Each semantic-to-WM bridge layer re-weights the semantic/spatial balance for its target WM:

1. **Semantic→SimSynthPhysics bridge**: The spatial graph convolution layer in this bridge explicitly up-weights spatial edges (contact, adjacency, containment) and down-weights purely semantic edges (affordance, role). This produces a more spatially explicit representation.

2. **Semantic→Embodiment bridge**: The bipartite body-object attention computes spatial relations in body-centric coordinates (distance from each end-effector, angle of approach, contact normal). This transforms scene-centric spatial structure into body-centric spatial structure.

3. **Semantic→Economic bridge**: The Perceiver queries learn to pool spatial information into abstract density/complexity/coverage features, discarding detailed topology.

### Transport Layer Governance of the Semantic/Spatial Balance

The transport layer provides the correct long-term governance of how much spatial vs. semantic detail each bridge should preserve:

- If the Perception↔SimSynthPhysics transport bridge struggles to reconstruct spatial topology, that signal increases the spatial weight in the Semantic→SimSynthPhysics bridge.
- If the Perception↔Embodiment transport bridge struggles with body-relative spatial accuracy, that signal shapes the Semantic→Embodiment bridge's coordinate transformation.
- If transport fidelity is already high, the bridges can allocate more capacity to semantic structure.

This is the mechanism by which the transport layer governs semantic-bridge hyperparameters through RL-informed feedback, without making the bridges themselves RL-trained.

---

## Multi-Rate Architecture Doctrine

### The Problem

The Unitree G1 produces observations at multiple rates:
- IMU / proprioception: 200-1000 Hz
- Vision: 10-30 Hz
- Economic / governance signals: ~1 Hz
- Sim evaluation results: episodic / batch

Current RL infrastructure assumes single-rate episode steps.

### The Solution

The stack must support multi-rate observation processing with explicit rate-aware architecture:

1. **Motor policy (Level 0)**: operates at servo rate (200-1000 Hz). Consumes only proprioceptive state + latest vision embedding + latest skill command. Must produce actions within 1-5ms.

2. **Skill policy (Level 1)**: operates at control rate (10-50 Hz). Consumes vision embeddings + proprioceptive summaries + skill parameters. Produces skill-conditioned setpoints for the motor policy.

3. **Task policy (Level 2)**: operates at planning rate (1-5 Hz). Consumes full semantic state + economic context + skill execution status. Produces skill selections and parameter updates.

4. **Economic/governance (Level 3+)**: operates at governance rate (~0.1-1 Hz). Consumes receipt summaries + resource state + governance context. Produces allocation decisions and governance actions.

### Multi-Rate Replay and Training

The replay, event spine, and training manifest infrastructure must support multi-rate artifact streams:

- **Replay records** must carry timestamps, not just step indices
- **Event spine** must support events at different temporal resolutions
- **Training manifests** must specify which rate each training target uses
- **Datapack schema** must carry multi-rate observation bundles (fast proprio stream + slower vision stream + slowest governance stream)

This is a structural requirement that must be in place before September 2026 training begins.

---

## Capacity Band Honesty Rule

All parameter counts and capacity bands in this document are **provisional**. They should be treated as:

- order-of-magnitude guidance tied to representational burden
- dependent on object count/cardinality, temporal horizon, embodiment DoF, and scene complexity
- dependent on latency budget (onboard vs. companion vs. offline)
- dependent on provider maturity and replay density (denser replay → more data → larger models become trainable)
- dependent on Unitree G1-scale compute and deployment constraints
- subject to revision during the Phase 3.5 humanoid capacity audit
- subject to revision whenever provider availability, replay density, or training evidence changes the substrate

The doctrine does not treat exact parameter counts as settled truth. The bands indicate structural expectations about where scale is needed (embodiment/motor > perception/semantic > economic/governance) and where scale would signal a topology mistake (e.g., an economic critic that needs >30M params is probably trying to re-encode raw embodiment state).

## Interaction with Existing Doctrines

This document extends but does not replace:

- **Cross-Phase Neuralization Rule**: every module here launches with `disabled|auto|required` promotion posture
- **No-Stub / Real-or-Unavailable Rule**: provider-backed layers use real-or-unavailable, not stubs
- **Complete Subsystem Rule**: each WM's neural layers are part of its subsystem completeness criteria
- **Mereotopological Datapack Rule**: the datapack composition network and contribution encoders are the neural realization of this doctrine
- **Ontology Layering Rule**: the semantic abstraction is operational ontology; the transport bridges are WM-transport ontology

---

## Staged Realization Timeline

| Phase | Neural Work | Semantic Bridge Work | Transport Work |
|-------|------------|---------------------|----------------|
| Phase 1 (Sim/Synth WM) | Branch evaluator, backend selector, yield predictor, agenda ranker | Semantic→SimSynthPhysics bridge reserved as typed contract; heuristic prior until perception canonical state exists | Not yet |
| Phase 2 (Perception WM) | Semantic abstraction (Graph Transformer), provider adapters, temporal grounding | Semantic→SimSynthPhysics bridge becomes learnable; Semantic→Embodiment bridge reserved | Not yet |
| Phase 3 (Embodiment WM) | Body-state encoder (GNN), motor policy (diffusion), feasibility checker, resource forecaster | Semantic→Embodiment bridge becomes learnable; Semantic→Economic bridge reserved | Not yet |
| Phase 3.5 (Humanoid Refit) | Capacity audit, multi-rate architecture, scale review | All bridges reviewed for humanoid capacity requirements | Not yet |
| Phase 5 (Economic WM) | Multi-WM receipt critic, resource allocator, source-mixture allocator | Semantic→Economic bridge becomes learnable | Not yet |
| Phase 6 (Transport) | Transport bridges between all adjacent WMs | All semantic bridges gain transport fidelity shaping term | Transport learning begins |
| Phase 6.5 (Meta-Node) | Local meta-node neuralization | Bridges gain meta-node state awareness | Transport quality shapes meta-node embeddings |
| Phase 7 (Meta-Node WM) | Multi-objective Pareto policy over WM governance | Full semantic superposition with transport governance | Transport shaping weights become learnable by meta-node WM |
